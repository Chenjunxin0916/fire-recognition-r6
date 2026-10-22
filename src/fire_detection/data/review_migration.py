"""Strict, write-once migration of scene-review decisions between candidate versions."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from fire_detection.data.candidate_metadata import load_candidate_metadata
from fire_detection.data.grouping import (
    CandidatePair,
    ReviewDecision,
    read_candidates,
    read_decisions,
    read_signatures,
    write_decisions,
)
from fire_detection.data.reviewer import (
    ReviewHistoryEntry,
    ReviewStore,
    read_review_history,
    validate_review_audit_for_lock,
)
from fire_detection.data.similarity import ImageSignature, sha256_file

_VERSION_PATTERN = re.compile(r"v([1-9][0-9]*)")
_SIGNATURE_FIELDS = ("filename", "label", "sha256", "phash", "width", "height")
_CANDIDATE_FIELDS = tuple(field.name for field in fields(CandidatePair))
_DECISION_FIELDS = tuple(field.name for field in fields(ReviewDecision))
_HISTORY_FIELDS = tuple(field.name for field in fields(ReviewHistoryEntry))
_DIFF_FIELDS = (
    "artifact",
    "change",
    "identifier",
    "changed_fields",
    "source",
    "target",
)


@dataclass(frozen=True)
class ReviewMigrationResult:
    """Published review migration and lock-readiness summary."""

    decisions_path: Path
    history_path: Path
    differences_csv_path: Path
    report_path: Path
    migrated_decisions: int
    migrated_history_entries: int
    difference_count: int
    lock_ready: bool
    lock_audit_passed: bool
    lock_audit_error: str | None


@dataclass(frozen=True)
class _Difference:
    artifact: str
    change: str
    identifier: str
    changed_fields: tuple[str, ...]
    source: dict[str, Any] | None
    target: dict[str, Any] | None

    def json_record(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "change": self.change,
            "identifier": self.identifier,
            "changed_fields": list(self.changed_fields),
            "source": self.source,
            "target": self.target,
        }

    def csv_record(self) -> dict[str, str]:
        return {
            "artifact": self.artifact,
            "change": self.change,
            "identifier": self.identifier,
            "changed_fields": "|".join(self.changed_fields),
            "source": (
                ""
                if self.source is None
                else json.dumps(
                    self.source, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            ),
            "target": (
                ""
                if self.target is None
                else json.dumps(
                    self.target, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            ),
        }


def _require_version_directory(path: Path, *, minimum: int = 2) -> tuple[str, int]:
    match = _VERSION_PATTERN.fullmatch(path.name)
    if match is None or int(match.group(1)) < minimum:
        raise ValueError(f"candidate version directory must be v{minimum} or newer: {path}")
    return path.name, int(match.group(1))


def _require_header(path: Path, expected: tuple[str, ...]) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            actual = tuple(csv.reader(stream).__next__())
    except StopIteration as exc:
        raise ValueError(f"CSV artifact is empty: {path}") from exc
    if actual != expected:
        raise ValueError(f"CSV artifact has an invalid schema: {path}; expected {expected}")


def _verify_optional_metadata(version_dir: Path, expected_version: str, *, active: bool) -> None:
    metadata_path = version_dir / "candidates.meta.json"
    if not metadata_path.exists():
        return
    metadata = load_candidate_metadata(metadata_path)
    if metadata.get("candidate_version") != expected_version:
        raise ValueError(
            f"candidate metadata version does not match directory {expected_version}: "
            f"{metadata_path}"
        )
    if active and metadata.get("status") != "active":
        raise ValueError(f"target candidate metadata must be active: {metadata_path}")
    artifact_hashes = metadata["artifact_sha256"]
    assert isinstance(artifact_hashes, dict)
    for name in ("image_hashes.csv", "candidates.csv"):
        expected_hash = artifact_hashes.get(name)
        artifact = version_dir / name
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"candidate metadata has an invalid hash for {name}: {metadata_path}")
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ValueError(f"candidate artifact differs from metadata: {artifact}")


def _signature_record(signature: ImageSignature, label: int) -> dict[str, Any]:
    return {
        "filename": signature.filename,
        "label": label,
        "sha256": signature.sha256,
        "phash": signature.phash,
        "width": signature.width,
        "height": signature.height,
    }


def _signature_maps(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, ImageSignature]]:
    _require_header(path, _SIGNATURE_FIELDS)
    signatures, labels = read_signatures(path)
    objects = {item.filename: item for item in signatures}
    return (
        {item.filename: _signature_record(item, labels[item.filename]) for item in signatures},
        objects,
    )


def _candidate_map(path: Path) -> dict[str, CandidatePair]:
    _require_header(path, _CANDIDATE_FIELDS)
    candidates = read_candidates(path)
    return {item.pair_id: item for item in candidates}


def _validate_candidate_signatures(
    candidates: dict[str, CandidatePair],
    signature_records: dict[str, dict[str, Any]],
    *,
    version: str,
) -> None:
    for candidate in candidates.values():
        for suffix in ("a", "b"):
            filename = getattr(candidate, f"filename_{suffix}")
            signature = signature_records.get(filename)
            if signature is None:
                raise ValueError(f"{version} candidate references an unknown signature: {filename}")
            for field_name in ("label", "sha256", "phash"):
                candidate_value = getattr(candidate, f"{field_name}_{suffix}")
                if signature[field_name] != candidate_value:
                    raise ValueError(
                        f"{version} candidate/signature {field_name} mismatch for "
                        f"{candidate.pair_id}"
                    )


def _changed_fields(source: dict[str, Any], target: dict[str, Any]) -> tuple[str, ...]:
    if source.keys() != target.keys():
        raise ValueError("comparison records do not use the same schema")
    return tuple(key for key in source if source[key] != target[key])


def _compare_signatures(
    source: dict[str, dict[str, Any]], target: dict[str, dict[str, Any]]
) -> tuple[list[_Difference], set[str]]:
    differences: list[_Difference] = []
    identical: set[str] = set()
    for filename in sorted(set(source) | set(target)):
        source_record = source.get(filename)
        target_record = target.get(filename)
        if source_record is None:
            differences.append(
                _Difference("image_hash", "added", filename, (), None, target_record)
            )
        elif target_record is None:
            differences.append(
                _Difference("image_hash", "deleted", filename, (), source_record, None)
            )
        elif source_record == target_record:
            identical.add(filename)
        else:
            differences.append(
                _Difference(
                    "image_hash",
                    "changed",
                    filename,
                    _changed_fields(source_record, target_record),
                    source_record,
                    target_record,
                )
            )
    return differences, identical


def _candidate_record(
    candidate: CandidatePair,
    signatures: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    record = asdict(candidate)
    record["sources"] = list(candidate.sources)
    record["signature_a"] = signatures[candidate.filename_a]
    record["signature_b"] = signatures[candidate.filename_b]
    return record


def _candidate_changed_fields(
    source: CandidatePair,
    target: CandidatePair,
    source_signatures: dict[str, dict[str, Any]],
    target_signatures: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    changed = [
        field_name
        for field_name in _CANDIDATE_FIELDS
        if getattr(source, field_name) != getattr(target, field_name)
    ]
    for suffix in ("a", "b"):
        source_signature = source_signatures[getattr(source, f"filename_{suffix}")]
        target_signature = target_signatures[getattr(target, f"filename_{suffix}")]
        changed.extend(
            f"signature_{suffix}.{field_name}"
            for field_name in _changed_fields(source_signature, target_signature)
        )
    return tuple(changed)


def _compare_candidates(
    source: dict[str, CandidatePair],
    target: dict[str, CandidatePair],
    source_signatures: dict[str, dict[str, Any]],
    target_signatures: dict[str, dict[str, Any]],
    identical_signatures: set[str],
) -> tuple[list[_Difference], set[str]]:
    differences: list[_Difference] = []
    migratable: set[str] = set()
    for pair_id in sorted(set(source) | set(target)):
        source_candidate = source.get(pair_id)
        target_candidate = target.get(pair_id)
        if source_candidate is None:
            assert target_candidate is not None
            differences.append(
                _Difference(
                    "candidate",
                    "added",
                    pair_id,
                    (),
                    None,
                    _candidate_record(target_candidate, target_signatures),
                )
            )
            continue
        if target_candidate is None:
            differences.append(
                _Difference(
                    "candidate",
                    "deleted",
                    pair_id,
                    (),
                    _candidate_record(source_candidate, source_signatures),
                    None,
                )
            )
            continue
        endpoint_signatures_match = {
            source_candidate.filename_a,
            source_candidate.filename_b,
        } <= identical_signatures
        if source_candidate == target_candidate and endpoint_signatures_match:
            migratable.add(pair_id)
            continue
        differences.append(
            _Difference(
                "candidate",
                "changed",
                pair_id,
                _candidate_changed_fields(
                    source_candidate,
                    target_candidate,
                    source_signatures,
                    target_signatures,
                ),
                _candidate_record(source_candidate, source_signatures),
                _candidate_record(target_candidate, target_signatures),
            )
        )
    return differences, migratable


def _history_matches_decision(entry: ReviewHistoryEntry, decision: ReviewDecision) -> bool:
    return (
        entry.pair_id,
        entry.filename_a,
        entry.filename_b,
        entry.decision,
        entry.reviewer,
        entry.reviewed_at,
        entry.notes,
    ) == (
        decision.pair_id,
        decision.filename_a,
        decision.filename_b,
        decision.decision,
        decision.reviewer,
        decision.reviewed_at,
        decision.notes,
    )


def _validate_review_snapshot(
    decisions: dict[str, ReviewDecision], history: tuple[ReviewHistoryEntry, ...]
) -> None:
    history_by_pair: dict[str, list[ReviewHistoryEntry]] = {}
    for entry in history:
        history_by_pair.setdefault(entry.pair_id, []).append(entry)
    orphan_history = sorted(set(history_by_pair) - set(decisions))
    if orphan_history:
        raise ValueError(f"review history has no current decision: {orphan_history[:10]}")
    for pair_id, decision in decisions.items():
        if not decision.decision:
            if pair_id in history_by_pair:
                raise ValueError(f"empty decision has review history: {pair_id}")
            continue
        pair_history = history_by_pair.get(pair_id, [])
        if not pair_history or not _history_matches_decision(pair_history[-1], decision):
            raise ValueError(f"current decision does not match final review history: {pair_id}")


def _write_history(path: Path, entries: tuple[ReviewHistoryEntry, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_HISTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(entry) for entry in entries)
        stream.flush()
        os.fsync(stream.fileno())


def _write_differences(path: Path, differences: tuple[_Difference, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_DIFF_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(item.csv_record() for item in differences)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _capture_lock_audit(
    candidates_path: Path,
    decisions_path: Path,
    history_path: Path,
    *,
    large_component_size: int,
) -> tuple[bool, str | None]:
    ReviewStore(
        candidates_path,
        decisions_path,
        history_path,
        large_component_size=large_component_size,
    )
    try:
        validate_review_audit_for_lock(
            candidates_path,
            decisions_path,
            history_path,
            large_component_size=large_component_size,
        )
    except ValueError as exc:
        return False, str(exc)
    return True, None


def _publish_staged_file(source: Path, destination: Path) -> None:
    """Atomically create a write-once destination on the same filesystem."""

    os.link(source, destination)


def _count_changes(differences: tuple[_Difference, ...], artifact: str) -> dict[str, int]:
    return {
        change: sum(item.artifact == artifact and item.change == change for item in differences)
        for change in ("added", "changed", "deleted")
    }


def _report_directory(path: Path, project_root: Path | None) -> str:
    """Return a portable directory identifier for the migration report."""

    if project_root is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"migration directory must be inside project_root: {path}") from exc


def migrate_review_version(
    source_dir: Path,
    target_dir: Path,
    *,
    source_decisions_path: Path | None = None,
    source_history_path: Path | None = None,
    target_decisions_path: Path | None = None,
    target_history_path: Path | None = None,
    differences_csv_path: Path | None = None,
    report_path: Path | None = None,
    project_root: Path | None = None,
    large_component_size: int = 10,
    expected_signatures: int | None = None,
    expected_candidates: int | None = None,
    expected_decisions: int | None = None,
    expected_history_entries: int | None = None,
) -> ReviewMigrationResult:
    """Migrate only byte-semantically identical candidate reviews into a newer version.

    Candidate equality covers every persisted candidate field and both endpoint rows in
    ``image_hashes.csv``. Changed, added, and deleted records are reported but never inherit
    decisions. Outputs are staged, validated with :class:`ReviewStore`, and published without
    overwriting any existing target review artifact.
    """

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    source_version, source_number = _require_version_directory(source_dir)
    target_version, target_number = _require_version_directory(target_dir)
    if target_number <= source_number:
        raise ValueError("target candidate version must be newer than source candidate version")
    if source_dir.resolve() == target_dir.resolve():
        raise ValueError("source and target candidate directories must differ")
    if large_component_size < 2:
        raise ValueError("large_component_size must be at least 2")

    source_decisions = source_decisions_path or source_dir / "review_decisions.csv"
    source_history = source_history_path or source_dir / "review_history.csv"
    target_decisions = target_decisions_path or target_dir / "review_decisions.csv"
    target_history = target_history_path or target_dir / "review_history.csv"
    differences_csv = differences_csv_path or target_dir / "review_migration_diff.csv"
    report = report_path or target_dir / "review_migration_report.json"
    outputs = (differences_csv, report, target_history, target_decisions)
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("target migration output paths must be distinct")
    if any(path.parent.resolve() != target_dir.resolve() for path in outputs):
        raise ValueError("all migration outputs must be stored directly in target_dir")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite target review artifacts: {existing}")
    if not source_decisions.is_file() or not source_history.is_file():
        raise FileNotFoundError("source review decisions and history must both exist")

    source_hashes_path = source_dir / "image_hashes.csv"
    source_candidates_path = source_dir / "candidates.csv"
    target_hashes_path = target_dir / "image_hashes.csv"
    target_candidates_path = target_dir / "candidates.csv"
    _verify_optional_metadata(source_dir, source_version, active=False)
    _verify_optional_metadata(target_dir, target_version, active=True)
    source_signature_records, _ = _signature_maps(source_hashes_path)
    target_signature_records, _ = _signature_maps(target_hashes_path)
    source_candidate_map = _candidate_map(source_candidates_path)
    target_candidate_map = _candidate_map(target_candidates_path)
    _validate_candidate_signatures(
        source_candidate_map, source_signature_records, version=source_version
    )
    _validate_candidate_signatures(
        target_candidate_map, target_signature_records, version=target_version
    )

    _require_header(source_decisions, _DECISION_FIELDS)
    _require_header(source_history, _HISTORY_FIELDS)
    source_review_hashes = {
        "decisions": sha256_file(source_decisions),
        "history": sha256_file(source_history),
    }
    ReviewStore(
        source_candidates_path,
        source_decisions,
        source_history,
        large_component_size=large_component_size,
    )
    validate_review_audit_for_lock(
        source_candidates_path,
        source_decisions,
        source_history,
        large_component_size=large_component_size,
    )
    decisions = read_decisions(source_decisions)
    history = read_review_history(source_history)
    _validate_review_snapshot(decisions, history)

    signature_differences, identical_signatures = _compare_signatures(
        source_signature_records, target_signature_records
    )
    candidate_differences, migratable_pair_ids = _compare_candidates(
        source_candidate_map,
        target_candidate_map,
        source_signature_records,
        target_signature_records,
        identical_signatures,
    )
    differences = tuple(
        sorted(
            (*signature_differences, *candidate_differences),
            key=lambda item: (item.artifact, item.change, item.identifier),
        )
    )
    migrated_decisions = tuple(
        decision
        for pair_id, decision in sorted(decisions.items())
        if pair_id in migratable_pair_ids
    )
    migrated_history = tuple(entry for entry in history if entry.pair_id in migratable_pair_ids)

    expectation_values = {
        "signatures": (expected_signatures, len(source_signature_records)),
        "candidates": (expected_candidates, len(source_candidate_map)),
        "decisions": (expected_decisions, len(migrated_decisions)),
        "history_entries": (expected_history_entries, len(migrated_history)),
    }
    expectations = {
        name: {"expected": expected, "actual": actual, "matched": expected in {None, actual}}
        for name, (expected, actual) in expectation_values.items()
    }
    expectations_met = all(item["matched"] for item in expectations.values())

    transaction_lock = target_dir / ".review_migration.lock"
    try:
        lock_fd = os.open(transaction_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"another review migration is active: {transaction_lock}") from exc
    os.close(lock_fd)
    staging = target_dir / f".review-migration.building-{uuid.uuid4().hex}"
    published: list[Path] = []
    target_lock = target_decisions.with_name(f".{target_decisions.name}.lock")
    target_lock_existed = target_lock.exists()
    try:
        staging.mkdir()
        staged_decisions = staging / target_decisions.name
        staged_history = staging / target_history.name
        staged_differences = staging / differences_csv.name
        staged_report = staging / report.name
        write_decisions(staged_decisions, migrated_decisions)
        _write_history(staged_history, migrated_history)
        _write_differences(staged_differences, differences)
        staged_audit_passed, staged_audit_error = _capture_lock_audit(
            target_candidates_path,
            staged_decisions,
            staged_history,
            large_component_size=large_component_size,
        )
        lock_ready = (
            not differences
            and len(migrated_decisions) == len(decisions)
            and len(migrated_history) == len(history)
            and expectations_met
            and staged_audit_passed
        )
        report_payload: dict[str, Any] = {
            "schema_version": 1,
            "source_version": source_version,
            "target_version": target_version,
            "source": {
                "directory": _report_directory(source_dir, project_root),
                "signatures": len(source_signature_records),
                "candidates": len(source_candidate_map),
                "decisions": len(decisions),
                "history_entries": len(history),
                "artifact_sha256": {
                    "image_hashes.csv": sha256_file(source_hashes_path),
                    "candidates.csv": sha256_file(source_candidates_path),
                    "review_decisions.csv": source_review_hashes["decisions"],
                    "review_history.csv": source_review_hashes["history"],
                },
            },
            "target": {
                "directory": _report_directory(target_dir, project_root),
                "signatures": len(target_signature_records),
                "candidates": len(target_candidate_map),
                "migrated_decisions": len(migrated_decisions),
                "migrated_history_entries": len(migrated_history),
                "eligible_pairs": len(migratable_pair_ids),
                "artifact_sha256": {
                    target_decisions.name: sha256_file(staged_decisions),
                    target_history.name: sha256_file(staged_history),
                    differences_csv.name: sha256_file(staged_differences),
                },
            },
            "comparison": {
                "image_hashes": {
                    "unchanged": len(identical_signatures),
                    **_count_changes(differences, "image_hash"),
                },
                "candidates": {
                    "unchanged": len(migratable_pair_ids),
                    **_count_changes(differences, "candidate"),
                },
                "differences": [item.json_record() for item in differences],
            },
            "expectations": expectations,
            "validation": {
                "review_store_passed": True,
                "lock_audit_passed": staged_audit_passed,
                "lock_audit_error": staged_audit_error,
                "lock_ready": lock_ready,
            },
        }
        _write_json(staged_report, report_payload)

        if source_review_hashes != {
            "decisions": sha256_file(source_decisions),
            "history": sha256_file(source_history),
        }:
            raise RuntimeError("source review artifacts changed during migration")
        if any(path.exists() for path in outputs):
            raise FileExistsError("target review artifacts appeared during migration")
        staged_by_output = {
            differences_csv: staged_differences,
            report: staged_report,
            target_history: staged_history,
            target_decisions: staged_decisions,
        }
        for destination in outputs:
            _publish_staged_file(staged_by_output[destination], destination)
            published.append(destination)

        final_audit_passed, final_audit_error = _capture_lock_audit(
            target_candidates_path,
            target_decisions,
            target_history,
            large_component_size=large_component_size,
        )
        if (final_audit_passed, final_audit_error) != (
            staged_audit_passed,
            staged_audit_error,
        ):
            raise RuntimeError("published review audit differs from staged review audit")
        return ReviewMigrationResult(
            target_decisions,
            target_history,
            differences_csv,
            report,
            len(migrated_decisions),
            len(migrated_history),
            len(differences),
            lock_ready,
            final_audit_passed,
            final_audit_error,
        )
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        if not target_lock_existed:
            target_lock.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        transaction_lock.unlink(missing_ok=True)
