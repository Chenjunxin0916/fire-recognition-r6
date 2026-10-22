"""Read-only integrity verification for an already locked Fold bundle."""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps

from fire_detection.data.annotation_review import (
    AnnotationReviewError,
    AnnotationReviewEvidence,
    validate_approved_annotation_review,
)
from fire_detection.data.candidate_metadata import REQUIRED_CANDIDATE_CODE_PATHS
from fire_detection.data.corrections import (
    CorrectionManifest,
    apply_bbox_correction,
    bbox_overflow,
    load_correction_manifest,
)
from fire_detection.data.folds import (
    FoldGateError,
    FoldRecord,
    FoldStats,
    small_fire_filenames,
    validate_candidates_metadata,
    validate_source_provenance,
)
from fire_detection.data.grouping import (
    CandidatePair,
    GroupingConflictError,
    GroupRecord,
    ReviewDecision,
    ReviewIncompleteError,
    member_group_identifier,
    read_candidates,
    read_decisions,
    read_signatures,
    resolve_scene_groups,
)
from fire_detection.data.reviewer import validate_review_audit_for_lock
from fire_detection.data.schema import DatasetIndex, DataValidationError, load_dataset_index
from fire_detection.data.similarity import ImageSignature, sha256_file

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GROUP_FIELDS = (
    "filename",
    "label",
    "sha256",
    "phash",
    "width",
    "height",
    "group_id",
    "group_size",
)
_FOLD_FIELDS = ("filename", "label", "group_id", "fold")
_PROCESSED_SOURCE_FIELDS = {
    "train_coco.json",
    "train_image.json",
    "audit_report",
    "data_corrections.yaml",
    "folds.csv",
    "folds.meta.json",
}
_QUALITY_GATES = {
    "minimum_negative_per_fold": 30,
    "maximum_positive_ratio_delta": 0.05,
    "maximum_small_fire_ratio_delta": 0.05,
}


class LockedFoldVerificationError(RuntimeError):
    """Raised when immutable Fold or processed-data evidence fails verification."""


@dataclass(frozen=True)
class VerificationWarning:
    """One non-fatal provenance difference from the historical candidate run."""

    code: str
    message: str
    path: str
    expected_sha256: str
    actual_sha256: str | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable warning record."""

        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
        }


@dataclass(frozen=True)
class LockedFoldVerificationResult:
    """Successful locked-Fold integrity report."""

    fold_version: str
    image_count: int
    group_count: int
    processed_image_count: int
    artifact_sha256: Mapping[str, str]
    warnings: tuple[VerificationWarning, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the stable machine-readable verifier result."""

        return {
            "schema_version": 1,
            "status": "valid",
            "fold_version": self.fold_version,
            "image_count": self.image_count,
            "group_count": self.group_count,
            "processed_image_count": self.processed_image_count,
            "artifact_sha256": dict(sorted(self.artifact_sha256.items())),
            "warning_count": len(self.warnings),
            "warnings": [warning.as_dict() for warning in self.warnings],
        }


@dataclass(frozen=True)
class LockedArtifacts:
    """Validated in-memory representation of the three locked Fold artifacts."""

    metadata: dict[str, Any]
    groups: tuple[GroupRecord, ...]
    folds: tuple[FoldRecord, ...]
    stats: tuple[FoldStats, ...]


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    class DuplicateKeyError(ValueError):
        pass

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(key)
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as exc:
        raise LockedFoldVerificationError(f"cannot strictly read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LockedFoldVerificationError(f"{name} must be a JSON object: {path}")
    return value


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LockedFoldVerificationError(f"{field} must be an object")
    return value


def _integer(value: object, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise LockedFoldVerificationError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise LockedFoldVerificationError(f"{field} must be >= {minimum}")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LockedFoldVerificationError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LockedFoldVerificationError(f"{field} must be a finite number")
    return result


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise LockedFoldVerificationError(f"{field} must be a lowercase SHA-256")
    return value


def _resolve_recorded_path(value: object, project_root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LockedFoldVerificationError(f"{field} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _verify_file_hash(path: Path, expected: object, field: str) -> str:
    expected_hash = _digest(expected, field)
    if not path.is_file():
        raise LockedFoldVerificationError(f"required file does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected_hash:
        raise LockedFoldVerificationError(
            f"{field} hash mismatch for {path}: expected {expected_hash}, got {actual}"
        )
    return actual


def historical_code_warnings(
    candidate_metadata: Mapping[str, Any], *, project_root: Path
) -> tuple[VerificationWarning, ...]:
    """Report current-code drift without invalidating immutable historical artifacts."""

    code_hashes = _mapping(candidate_metadata.get("code_sha256"), "code_sha256")
    missing_required = sorted(set(REQUIRED_CANDIDATE_CODE_PATHS) - set(code_hashes))
    if missing_required:
        raise LockedFoldVerificationError(
            f"candidate code manifest omits required paths: {missing_required}"
        )
    warnings: list[VerificationWarning] = []
    for relative, raw_expected in sorted(code_hashes.items()):
        if not isinstance(relative, str) or not relative:
            raise LockedFoldVerificationError("candidate code manifest contains an invalid path")
        expected = _digest(raw_expected, f"code_sha256.{relative}")
        path = _resolve_recorded_path(relative, project_root, f"code_sha256.{relative}")
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            warnings.append(
                VerificationWarning(
                    code="historical_code_sha256_drift",
                    message=(
                        "current code differs from the candidate-generation snapshot; "
                        "locked artifact integrity remains independently verified"
                    ),
                    path=relative,
                    expected_sha256=expected,
                    actual_sha256=actual,
                )
            )
    return tuple(warnings)


def _read_groups(path: Path) -> tuple[GroupRecord, ...]:
    rows: list[GroupRecord] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _GROUP_FIELDS:
                raise LockedFoldVerificationError("groups.csv has invalid columns")
            for offset, row in enumerate(reader, start=2):
                filename = row.get("filename", "")
                if not filename or filename in seen:
                    raise LockedFoldVerificationError(
                        f"groups.csv has an empty/duplicate filename at row {offset}: {filename!r}"
                    )
                seen.add(filename)
                record = GroupRecord(
                    filename=filename,
                    label=int(row.get("label", "")),
                    sha256=row.get("sha256", ""),
                    phash=row.get("phash", ""),
                    width=int(row.get("width", "")),
                    height=int(row.get("height", "")),
                    group_id=row.get("group_id", ""),
                    group_size=int(row.get("group_size", "")),
                )
                if record.label not in {0, 1}:
                    raise LockedFoldVerificationError(
                        f"groups.csv has an invalid label for {filename}"
                    )
                _digest(record.sha256, f"groups.csv[{filename}].sha256")
                _digest(record.group_id, f"groups.csv[{filename}].group_id")
                if not re.fullmatch(r"[0-9a-f]{16}", record.phash):
                    raise LockedFoldVerificationError(
                        f"groups.csv has an invalid pHash for {filename}"
                    )
                if record.width <= 0 or record.height <= 0 or record.group_size <= 0:
                    raise LockedFoldVerificationError(
                        f"groups.csv has invalid dimensions/group size for {filename}"
                    )
                rows.append(record)
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, LockedFoldVerificationError):
            raise
        raise LockedFoldVerificationError(f"cannot read groups.csv {path}: {exc}") from exc
    if not rows:
        raise LockedFoldVerificationError("groups.csv is empty")
    return tuple(sorted(rows, key=lambda row: row.filename))


def _read_folds(path: Path) -> tuple[FoldRecord, ...]:
    rows: list[FoldRecord] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _FOLD_FIELDS:
                raise LockedFoldVerificationError("folds.csv has invalid columns")
            for offset, row in enumerate(reader, start=2):
                filename = row.get("filename", "")
                if not filename or filename in seen:
                    raise LockedFoldVerificationError(
                        f"folds.csv has an empty/duplicate filename at row {offset}: {filename!r}"
                    )
                seen.add(filename)
                record = FoldRecord(
                    filename=filename,
                    label=int(row.get("label", "")),
                    group_id=row.get("group_id", ""),
                    fold=int(row.get("fold", "")),
                )
                if record.label not in {0, 1} or record.fold not in range(5):
                    raise LockedFoldVerificationError(
                        f"folds.csv has an invalid label/fold for {filename}"
                    )
                _digest(record.group_id, f"folds.csv[{filename}].group_id")
                rows.append(record)
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, LockedFoldVerificationError):
            raise
        raise LockedFoldVerificationError(f"cannot read folds.csv {path}: {exc}") from exc
    if not rows:
        raise LockedFoldVerificationError("folds.csv is empty")
    return tuple(sorted(rows, key=lambda row: row.filename))


def _compute_stats(
    folds: Sequence[FoldRecord], small_fire_images: Collection[str]
) -> tuple[FoldStats, ...]:
    small = frozenset(small_fire_images)
    result: list[FoldStats] = []
    for fold in range(5):
        rows = [row for row in folds if row.fold == fold]
        negative = sum(row.label == 0 for row in rows)
        positive = sum(row.label == 1 for row in rows)
        small_count = sum(row.filename in small for row in rows)
        result.append(
            FoldStats(
                fold=fold,
                count=len(rows),
                negative=negative,
                positive=positive,
                positive_ratio=positive / len(rows) if rows else 0.0,
                small_fire=small_count,
                small_fire_ratio=small_count / len(rows) if rows else 0.0,
                small_fire_positive_ratio=small_count / positive if positive else 0.0,
            )
        )
    return tuple(result)


def _validate_metadata_stats(metadata: Mapping[str, Any], stats: Sequence[FoldStats]) -> None:
    stored = metadata.get("folds")
    if not isinstance(stored, list) or len(stored) != 5:
        raise LockedFoldVerificationError("folds.meta.json folds must contain exactly five rows")
    integer_fields = ("fold", "count", "negative", "positive", "small_fire")
    number_fields = (
        "positive_ratio",
        "small_fire_ratio",
        "small_fire_positive_ratio",
    )
    for expected, raw in zip(stats, stored, strict=True):
        row = _mapping(raw, f"folds[{expected.fold}]")
        for field in integer_fields:
            if _integer(row.get(field), f"folds[{expected.fold}].{field}") != getattr(
                expected, field
            ):
                raise LockedFoldVerificationError(
                    f"fold statistics drift for fold {expected.fold}: {field}"
                )
        for field in number_fields:
            if not math.isclose(
                _number(row.get(field), f"folds[{expected.fold}].{field}"),
                getattr(expected, field),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise LockedFoldVerificationError(
                    f"fold statistics drift for fold {expected.fold}: {field}"
                )


def verify_locked_artifacts(
    *,
    locked_dir: Path,
    signatures: Sequence[ImageSignature],
    labels: Mapping[str, int],
    small_fire_images: Collection[str],
    input_paths: Mapping[str, Path],
    expected_version: str,
    expected_count: int,
) -> LockedArtifacts:
    """Verify locked CSV hashes, row constraints, grouping, Fold gates, and statistics."""

    groups_path = locked_dir / "groups.csv"
    folds_path = locked_dir / "folds.csv"
    metadata_path = locked_dir / "folds.meta.json"
    metadata = _strict_json(metadata_path, name="locked Fold metadata")
    if metadata.get("schema_version") != 2 or metadata.get("status") != "locked":
        raise LockedFoldVerificationError(
            "Fold metadata must declare schema_version=2/status=locked"
        )
    if metadata.get("fold_version") != expected_version:
        raise LockedFoldVerificationError(
            "Fold metadata version does not match the candidate version"
        )
    if (
        metadata.get("strategy") != "StratifiedGroupKFold"
        or metadata.get("stratification") != ["negative", "positive", "small_fire_positive"]
        or metadata.get("small_fire_area_ratio_threshold") != 0.001
        or metadata.get("n_splits") != 5
        or metadata.get("shuffle") is not True
        or metadata.get("seed") != 42
    ):
        raise LockedFoldVerificationError("Fold metadata strategy/seed/stratification is invalid")

    artifacts = _mapping(metadata.get("artifact_sha256"), "artifact_sha256")
    if set(artifacts) != {"groups.csv", "folds.csv"}:
        raise LockedFoldVerificationError("Fold artifact hash manifest has invalid keys")
    _verify_file_hash(groups_path, artifacts["groups.csv"], "artifact_sha256.groups.csv")
    _verify_file_hash(folds_path, artifacts["folds.csv"], "artifact_sha256.folds.csv")

    recorded_inputs = _mapping(metadata.get("input_sha256"), "input_sha256")
    if set(recorded_inputs) != set(input_paths):
        raise LockedFoldVerificationError(
            "Fold input hash manifest differs from required inputs; "
            f"missing={sorted(set(input_paths) - set(recorded_inputs))}, "
            f"extra={sorted(set(recorded_inputs) - set(input_paths))}"
        )
    for name, path in sorted(input_paths.items()):
        _verify_file_hash(path, recorded_inputs[name], f"input_sha256.{name}")

    groups = _read_groups(groups_path)
    folds = _read_folds(folds_path)
    if len(groups) != expected_count or len(folds) != expected_count:
        raise LockedFoldVerificationError(
            f"locked Fold count mismatch: groups={len(groups)}, folds={len(folds)}, "
            f"expected={expected_count}"
        )
    signature_by_name = {signature.filename: signature for signature in signatures}
    group_by_name = {row.filename: row for row in groups}
    fold_by_name = {row.filename: row for row in folds}
    expected_names = set(labels)
    if set(signature_by_name) != expected_names:
        raise LockedFoldVerificationError("signature and source-label filename sets differ")
    if set(group_by_name) != expected_names or set(fold_by_name) != expected_names:
        raise LockedFoldVerificationError("locked artifacts do not cover the source filename set")

    members_by_group: dict[str, list[str]] = {}
    folds_by_group: dict[str, set[int]] = {}
    for filename in sorted(expected_names):
        signature = signature_by_name[filename]
        group = group_by_name[filename]
        fold = fold_by_name[filename]
        if (
            group.label != labels[filename]
            or group.sha256 != signature.sha256
            or group.phash != signature.phash
            or (group.width, group.height) != (signature.width, signature.height)
        ):
            raise LockedFoldVerificationError(
                f"groups.csv differs from image signatures/source label for {filename}"
            )
        if fold.label != group.label or fold.group_id != group.group_id:
            raise LockedFoldVerificationError(f"folds.csv differs from groups.csv for {filename}")
        members_by_group.setdefault(group.group_id, []).append(filename)
        folds_by_group.setdefault(group.group_id, set()).add(fold.fold)
    for group_id, members in members_by_group.items():
        if member_group_identifier(members) != group_id:
            raise LockedFoldVerificationError(f"group id does not match members: {group_id}")
        expected_size = len(members)
        if any(group_by_name[name].group_size != expected_size for name in members):
            raise LockedFoldVerificationError(f"group_size drift for group {group_id}")
    split_groups = sorted(group for group, fold_ids in folds_by_group.items() if len(fold_ids) != 1)
    if split_groups:
        raise LockedFoldVerificationError(
            f"scene groups cross validation folds: {split_groups[:10]}"
        )

    stats = _compute_stats(folds, small_fire_images)
    _validate_metadata_stats(metadata, stats)
    quality = _mapping(metadata.get("quality_gates"), "quality_gates")
    if set(quality) != {
        "minimum_negative_per_fold",
        "maximum_positive_ratio_delta",
        "maximum_small_fire_ratio_delta",
    }:
        raise LockedFoldVerificationError("Fold quality_gates has invalid keys")
    minimum_negative = _integer(
        quality["minimum_negative_per_fold"], "quality_gates.minimum_negative_per_fold", minimum=1
    )
    max_positive_delta = _number(
        quality["maximum_positive_ratio_delta"],
        "quality_gates.maximum_positive_ratio_delta",
    )
    max_small_delta = _number(
        quality["maximum_small_fire_ratio_delta"],
        "quality_gates.maximum_small_fire_ratio_delta",
    )
    if (
        minimum_negative != _QUALITY_GATES["minimum_negative_per_fold"]
        or max_positive_delta != _QUALITY_GATES["maximum_positive_ratio_delta"]
        or max_small_delta != _QUALITY_GATES["maximum_small_fire_ratio_delta"]
    ):
        raise LockedFoldVerificationError(
            "Fold quality_gates differ from the fixed 30/0.05/0.05 contract"
        )
    overall_positive = sum(row.label == 1 for row in folds) / len(folds)
    overall_small = len(small_fire_images) / len(folds)
    for row in stats:
        if not row.count or not row.positive or row.negative < minimum_negative:
            raise LockedFoldVerificationError(f"Fold quality gate failed for fold {row.fold}")
        if abs(row.positive_ratio - overall_positive) > max_positive_delta:
            raise LockedFoldVerificationError(
                f"positive-ratio quality gate failed for fold {row.fold}"
            )
        if abs(row.small_fire_ratio - overall_small) > max_small_delta:
            raise LockedFoldVerificationError(
                f"small-fire-ratio quality gate failed for fold {row.fold}"
            )

    expected_metadata_counts = {
        "image_count": len(folds),
        "negative_count": sum(row.label == 0 for row in folds),
        "positive_count": sum(row.label == 1 for row in folds),
        "small_fire_count": len(small_fire_images),
        "group_count": len(members_by_group),
    }
    for field, expected in expected_metadata_counts.items():
        if _integer(metadata.get(field), field, minimum=0) != expected:
            raise LockedFoldVerificationError(f"Fold metadata count drift: {field}")
    return LockedArtifacts(metadata, groups, folds, stats)


def _verify_candidate_inputs(
    candidate_metadata: Mapping[str, Any], *, project_root: Path, config_path: Path
) -> None:
    inputs = _mapping(candidate_metadata.get("inputs"), "candidate inputs")
    config = _mapping(inputs.get("config"), "candidate inputs.config")
    config_recorded_path = _resolve_recorded_path(
        config.get("path"), project_root, "candidate inputs.config.path"
    )
    if config_recorded_path.resolve() != config_path.resolve():
        raise LockedFoldVerificationError("candidate metadata identifies a different data config")
    _verify_file_hash(config_path, config.get("sha256"), "candidate inputs.config.sha256")
    offline = _mapping(inputs.get("offline_manifest"), "candidate inputs.offline_manifest")
    offline_path = _resolve_recorded_path(
        offline.get("path"), project_root, "candidate inputs.offline_manifest.path"
    )
    _verify_file_hash(
        offline_path,
        offline.get("sha256"),
        "candidate inputs.offline_manifest.sha256",
    )
    embedding = _mapping(candidate_metadata.get("scene_embedding"), "scene_embedding")
    embedding_path = _resolve_recorded_path(
        embedding.get("path"), project_root, "scene_embedding.path"
    )
    _verify_file_hash(embedding_path, embedding.get("sha256"), "scene_embedding.sha256")
    if embedding_path.stat().st_size != _integer(
        embedding.get("size_bytes"), "scene_embedding.size_bytes", minimum=1
    ):
        raise LockedFoldVerificationError("scene embedding weight size differs from metadata")


def _verify_candidate_rows(
    candidate_metadata: Mapping[str, Any], *, image_hashes: Path, candidates: Path
) -> tuple[CandidatePair, ...]:
    signatures, labels = read_signatures(image_hashes)
    signature_by_name = {row.filename: row for row in signatures}
    thresholds = _mapping(candidate_metadata.get("thresholds"), "candidate thresholds")
    large_component_size = _integer(
        thresholds.get("large_component_size"),
        "candidate thresholds.large_component_size",
        minimum=2,
    )
    rows = read_candidates(candidates, large_component_size=large_component_size)
    for row in rows:
        left = signature_by_name.get(row.filename_a)
        right = signature_by_name.get(row.filename_b)
        if left is None or right is None:
            raise LockedFoldVerificationError(
                f"candidate references an unknown signature: {row.pair_id}"
            )
        if (row.label_a, row.label_b) != (labels[row.filename_a], labels[row.filename_b]):
            raise LockedFoldVerificationError(f"candidate labels are stale: {row.pair_id}")
        if (row.sha256_a, row.sha256_b) != (left.sha256, right.sha256) or (
            row.phash_a,
            row.phash_b,
        ) != (left.phash, right.phash):
            raise LockedFoldVerificationError(f"candidate fingerprints are stale: {row.pair_id}")
    counts = _mapping(candidate_metadata.get("counts"), "candidate counts")
    expected_counts = {
        "images": len(signatures),
        "candidates": len(rows),
        "mandatory_reviews": sum(row.requires_review for row in rows),
        "embedding_candidates": sum("scene_embedding" in row.sources for row in rows),
    }
    for field, expected in expected_counts.items():
        if counts.get(field) != expected:
            raise LockedFoldVerificationError(f"candidate metadata count drift: {field}")
    source_counts: dict[str, int] = {}
    for row in rows:
        for source in row.sources:
            source_counts[source] = source_counts.get(source, 0) + 1
    if counts.get("source_counts") != source_counts:
        raise LockedFoldVerificationError("candidate metadata source_counts drift")
    return rows


def _verify_annotation_provenance(
    metadata: Mapping[str, Any], evidence: AnnotationReviewEvidence
) -> None:
    provenance = _mapping(metadata.get("provenance"), "provenance")
    review = _mapping(provenance.get("annotation_review"), "provenance.annotation_review")
    expected: dict[str, object] = {
        "review_sha256": evidence.review_sha256,
        "manifest_sha256": evidence.manifest_sha256,
        "page_bundle_sha256": evidence.page_bundle_sha256,
        "page_count": evidence.page_count,
        "reviewer": evidence.reviewer,
        "reviewed_at": evidence.reviewed_at,
        "disposition": evidence.disposition,
        "issue_count": evidence.issue_count,
    }
    if review != expected:
        raise LockedFoldVerificationError(
            "annotation review evidence differs from locked provenance"
        )


def verify_source_signature_binding(
    *,
    index: DatasetIndex,
    signatures: Sequence[ImageSignature],
    signature_labels: Mapping[str, int],
    images_dir: Path,
) -> None:
    """Bind source labels and EXIF-normalized dimensions to the signature artifact."""

    source_names = set(index.labels)
    signature_by_name = {row.filename: row for row in signatures}
    if set(signature_by_name) != source_names or set(signature_labels) != source_names:
        raise LockedFoldVerificationError("source JSON and image_hashes.csv filename sets differ")
    mismatched_labels = sorted(
        name for name in source_names if signature_labels[name] != index.labels[name]
    )
    if mismatched_labels:
        raise LockedFoldVerificationError(
            f"source labels differ from image_hashes.csv: {mismatched_labels[:10]}"
        )

    for image_record in index.images:
        signature = signature_by_name[image_record.file_name]
        if (signature.width, signature.height) != (image_record.width, image_record.height):
            raise LockedFoldVerificationError(
                f"COCO dimensions differ from image_hashes.csv for {image_record.file_name}"
            )
        image_path = images_dir / image_record.file_name
        _verify_file_hash(
            image_path,
            signature.sha256,
            f"image_hashes.csv[{image_record.file_name}].sha256",
        )
        try:
            with Image.open(image_path) as image:
                image.load()
                visual_size = ImageOps.exif_transpose(image).size
        except OSError as exc:
            raise LockedFoldVerificationError(
                f"raw image is unreadable: {image_path}: {exc}"
            ) from exc
        if visual_size != (signature.width, signature.height):
            raise LockedFoldVerificationError(
                "EXIF-normalized raw dimensions differ from image_hashes.csv for "
                f"{image_record.file_name}"
            )


def verify_locked_group_semantics(
    *,
    signatures: Sequence[ImageSignature],
    labels: Mapping[str, int],
    candidates: Sequence[CandidatePair],
    decisions: Mapping[str, ReviewDecision],
    locked_groups: Sequence[GroupRecord],
) -> None:
    """Require locked groups to equal reviewed and automatic same-scene closure."""

    expected = resolve_scene_groups(
        tuple(signatures),
        dict(labels),
        tuple(candidates),
        dict(decisions),
    )
    actual = tuple(sorted(locked_groups, key=lambda row: row.filename))
    if expected == actual:
        return
    expected_by_name = {row.filename: row for row in expected}
    actual_by_name = {row.filename: row for row in actual}
    differing = sorted(
        name
        for name in set(expected_by_name) | set(actual_by_name)
        if expected_by_name.get(name) != actual_by_name.get(name)
    )
    raise LockedFoldVerificationError(
        f"locked groups differ from candidate/review semantic closure: {differing[:10]}"
    )


def _expected_processed_annotations(
    index: DatasetIndex,
    corrections: CorrectionManifest,
) -> tuple[
    dict[int, tuple[float, float, float, float]],
    list[dict[str, object]],
]:
    images = index.images_by_id
    annotations_by_image = index.annotations_by_image
    if {(category.id, category.name) for category in index.categories} != {(0, "fire")}:
        raise LockedFoldVerificationError("processed data requires category mapping 0=fire")
    for image in index.images:
        count = len(annotations_by_image[image.id])
        if index.labels[image.file_name] == 0 and count:
            raise LockedFoldVerificationError(
                f"negative source image has fire boxes: {image.file_name}"
            )
        if index.labels[image.file_name] == 1 and not count:
            raise LockedFoldVerificationError(
                f"positive source image has no fire boxes: {image.file_name}"
            )

    boxes: dict[int, tuple[float, float, float, float]] = {}
    applications: list[dict[str, object]] = []
    applied: set[int] = set()
    for annotation in index.annotations:
        image = images[annotation.image_id]
        correction = corrections.bbox.get(annotation.id)
        if correction is not None:
            record = apply_bbox_correction(annotation, image, correction)
            boxes[annotation.id] = record.corrected_bbox
            applications.append(
                {
                    "annotation_id": record.annotation_id,
                    "filename": record.filename,
                    "action": record.action,
                    "reason": record.reason,
                    "original_bbox": list(record.original_bbox),
                    "corrected_bbox": list(record.corrected_bbox),
                }
            )
            applied.add(annotation.id)
        elif bbox_overflow(annotation.bbox, image.width, image.height) > 1e-9:
            raise LockedFoldVerificationError(
                f"annotation {annotation.id} has an unreviewed boundary overflow"
            )
        else:
            boxes[annotation.id] = annotation.bbox
    missing = sorted(set(corrections.bbox) - applied)
    if missing:
        raise LockedFoldVerificationError(f"bbox correction entries were not applied: {missing}")
    return boxes, sorted(
        applications,
        key=lambda row: _integer(row.get("annotation_id"), "expected bbox annotation_id"),
    )


def _verify_yolo_labels(
    *,
    processed_root: Path,
    index: DatasetIndex,
    boxes: Mapping[int, tuple[float, float, float, float]],
) -> None:
    label_dir = processed_root / "labels"
    if not label_dir.is_dir():
        raise LockedFoldVerificationError(f"processed label directory is missing: {label_dir}")
    stems = [Path(image.file_name).stem for image in index.images]
    if len(stems) != len(set(stems)):
        raise LockedFoldVerificationError("source image stems are not unique")
    expected_names = {f"{stem}.txt" for stem in stems}
    actual_names = {path.name for path in label_dir.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise LockedFoldVerificationError("processed label directory has missing/extra files")

    annotations_by_image = index.annotations_by_image
    for image in index.images:
        lines: list[str] = []
        for annotation in annotations_by_image[image.id]:
            x, y, width, height = boxes[annotation.id]
            values = (
                (x + width / 2.0) / image.width,
                (y + height / 2.0) / image.height,
                width / image.width,
                height / image.height,
            )
            lines.append("0 " + " ".join(format(value, ".12g") for value in values))
        expected = "\n".join(lines) + ("\n" if lines else "")
        label_path = label_dir / f"{Path(image.file_name).stem}.txt"
        if label_path.read_text(encoding="utf-8") != expected:
            raise LockedFoldVerificationError(
                f"processed YOLO label differs from source/corrections: {label_path}"
            )


def _verify_detection_views(*, processed_root: Path, folds: Sequence[FoldRecord]) -> None:
    detection_root = processed_root / "detection"
    if not detection_root.is_dir():
        raise LockedFoldVerificationError(
            f"processed detection directory is missing: {detection_root}"
        )
    expected_directories = {f"fold_{fold}" for fold in range(5)}
    actual_directories = {path.name for path in detection_root.iterdir() if path.is_dir()}
    if actual_directories != expected_directories:
        raise LockedFoldVerificationError("processed detection Fold directories are invalid")
    expected_yaml = {"train": "train.txt", "val": "val.txt", "names": {0: "fire"}}
    for held_out in range(5):
        fold_dir = detection_root / f"fold_{held_out}"
        actual_files = {path.name for path in fold_dir.iterdir() if path.is_file()}
        if actual_files != {"train.txt", "val.txt", "data.yaml"}:
            raise LockedFoldVerificationError(
                f"processed detection files are invalid for fold {held_out}"
            )
        for split, selected in (
            ("train", (row for row in folds if row.fold != held_out)),
            ("val", (row for row in folds if row.fold == held_out)),
        ):
            expected = "".join(f"./../../images/{row.filename}\n" for row in selected)
            path = fold_dir / f"{split}.txt"
            if path.read_text(encoding="utf-8") != expected:
                raise LockedFoldVerificationError(
                    f"processed detection {split} list differs from locked Fold {held_out}"
                )
        try:
            dataset_yaml = yaml.safe_load((fold_dir / "data.yaml").read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise LockedFoldVerificationError(
                f"cannot parse processed detector YAML for fold {held_out}: {exc}"
            ) from exc
        if dataset_yaml != expected_yaml:
            raise LockedFoldVerificationError(
                f"processed detector YAML differs from the fixed contract for fold {held_out}"
            )


def _verify_processed_manifest(
    *,
    manifest_path: Path,
    corrections_yaml: Path,
    raw_images_dir: Path,
    index: DatasetIndex,
    signatures: Sequence[ImageSignature],
    folds: Sequence[FoldRecord],
    source_paths: Mapping[str, Path],
    fold_version: str,
) -> int:
    """Verify processed manifest, annotations, canonical images, and all Fold views."""

    manifest = _strict_json(manifest_path, name="processed-data manifest")
    if manifest.get("schema_version") != 1 or manifest.get("processed_version") != fold_version:
        raise LockedFoldVerificationError("processed manifest schema/version does not match Fold")
    if manifest.get("fold_version") != fold_version:
        raise LockedFoldVerificationError("processed manifest fold_version does not match Fold")
    corrections = load_correction_manifest(corrections_yaml)
    if manifest.get("correction_version") != corrections.version:
        raise LockedFoldVerificationError(
            "processed correction_version differs from corrections YAML"
        )
    boxes, expected_corrections = _expected_processed_annotations(index, corrections)
    if manifest.get("bbox_corrections") != expected_corrections:
        raise LockedFoldVerificationError(
            "processed bbox_corrections differ from source/correction manifest"
        )
    expected_counts = {
        "image_count": len(index.images),
        "annotation_count": len(index.annotations),
        "negative_count": sum(label == 0 for label in index.labels.values()),
        "positive_count": sum(label == 1 for label in index.labels.values()),
    }
    for field, expected in expected_counts.items():
        if _integer(manifest.get(field), f"processed.{field}", minimum=0) != expected:
            raise LockedFoldVerificationError(f"processed manifest count drift: {field}")

    source_hashes = _mapping(manifest.get("source_sha256"), "processed.source_sha256")
    if set(source_hashes) != _PROCESSED_SOURCE_FIELDS:
        raise LockedFoldVerificationError("processed source_sha256 has invalid keys")
    all_source_paths = dict(source_paths)
    all_source_paths["data_corrections.yaml"] = corrections_yaml
    if set(all_source_paths) != _PROCESSED_SOURCE_FIELDS:
        raise LockedFoldVerificationError("processed verifier source path map is incomplete")
    for field, path in sorted(all_source_paths.items()):
        _verify_file_hash(path, source_hashes[field], f"processed.source_sha256.{field}")

    signature_by_name = {row.filename: row for row in signatures}
    fold_by_name = {row.filename: row for row in folds}
    raw_images = manifest.get("images")
    if not isinstance(raw_images, list) or len(raw_images) != len(fold_by_name):
        raise LockedFoldVerificationError("processed images must cover every locked Fold row")
    seen: set[str] = set()
    processed_dir = manifest_path.parent / "images"
    if not processed_dir.is_dir():
        raise LockedFoldVerificationError(
            f"processed canonical image directory is missing: {processed_dir}"
        )
    applied_orientations: set[str] = set()
    for offset, raw in enumerate(raw_images):
        row = _mapping(raw, f"processed.images[{offset}]")
        filename = row.get("filename")
        if not isinstance(filename, str) or not filename or filename in seen:
            raise LockedFoldVerificationError(
                f"processed manifest has an empty/duplicate image filename: {filename!r}"
            )
        seen.add(filename)
        signature = signature_by_name.get(filename)
        fold = fold_by_name.get(filename)
        if signature is None or fold is None:
            raise LockedFoldVerificationError(
                f"processed manifest references an unknown image: {filename}"
            )
        source_hash = _digest(row.get("source_sha256"), f"processed.images[{filename}].source")
        processed_hash = _digest(
            row.get("processed_sha256"), f"processed.images[{filename}].processed"
        )
        if (
            source_hash != signature.sha256
            or row.get("label") != fold.label
            or row.get("fold") != fold.fold
        ):
            raise LockedFoldVerificationError(
                f"processed image provenance differs from locked Fold for {filename}"
            )
        raw_image_path = raw_images_dir / filename
        _verify_file_hash(
            raw_image_path,
            source_hash,
            f"processed.images[{filename}].source_sha256",
        )
        try:
            with Image.open(raw_image_path) as raw_image:
                raw_image.load()
                orientation_value = raw_image.getexif().get(274)
                source_orientation = orientation_value if type(orientation_value) is int else None
        except OSError as exc:
            raise LockedFoldVerificationError(
                f"raw image is unreadable while verifying processed data: {raw_image_path}: {exc}"
            ) from exc
        if row.get("source_orientation") != source_orientation:
            raise LockedFoldVerificationError(
                f"processed source_orientation differs from raw EXIF for {filename}"
            )
        orientation_fix = corrections.orientation.get(filename)
        if orientation_fix is not None:
            if source_orientation != orientation_fix.expected_orientation:
                raise LockedFoldVerificationError(
                    f"raw EXIF orientation differs from correction manifest for {filename}"
                )
            expected_operation = "exif_transpose_reencode"
            applied_orientations.add(filename)
        else:
            if source_orientation not in {None, 1}:
                raise LockedFoldVerificationError(
                    f"processed data contains an unreviewed EXIF orientation for {filename}"
                )
            expected_operation = "byte_copy"
        image_path = processed_dir / filename
        _verify_file_hash(image_path, processed_hash, f"processed image {filename}")
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                actual_size = image.size
                processed_orientation = image.getexif().get(274)
        except OSError as exc:
            raise LockedFoldVerificationError(
                f"processed image is unreadable: {image_path}: {exc}"
            ) from exc
        if actual_size != (
            _integer(row.get("width"), f"processed.images[{filename}].width", minimum=1),
            _integer(row.get("height"), f"processed.images[{filename}].height", minimum=1),
        ):
            raise LockedFoldVerificationError(
                f"processed image dimensions differ from manifest for {filename}"
            )
        operation = row.get("operation")
        if operation != expected_operation:
            raise LockedFoldVerificationError(
                f"processed image operation differs from correction contract for {filename}"
            )
        if operation == "byte_copy" and processed_hash != source_hash:
            raise LockedFoldVerificationError(f"byte-copy hash differs from source for {filename}")
        if operation == "exif_transpose_reencode" and processed_orientation is not None:
            raise LockedFoldVerificationError(
                f"processed EXIF orientation was not removed for {filename}"
            )
    if applied_orientations != set(corrections.orientation):
        raise LockedFoldVerificationError("orientation correction entries were not all applied")
    if seen != set(fold_by_name):
        raise LockedFoldVerificationError("processed image filename set differs from locked Fold")
    actual_image_names = {path.name for path in processed_dir.iterdir() if path.is_file()}
    if actual_image_names != seen:
        raise LockedFoldVerificationError("processed image directory has missing/extra files")

    classification = manifest.get("classification_files")
    if not isinstance(classification, list) or len(classification) != len(folds) * 5:
        raise LockedFoldVerificationError("processed classification manifest has invalid row count")
    seen_classification: set[tuple[int, str]] = set()
    for offset, raw in enumerate(classification):
        row = _mapping(raw, f"processed.classification_files[{offset}]")
        held_out = _integer(row.get("fold"), f"classification[{offset}].fold")
        filename = row.get("filename")
        if held_out not in range(5) or not isinstance(filename, str):
            raise LockedFoldVerificationError(f"invalid classification record at row {offset}")
        key = (held_out, filename)
        if key in seen_classification or filename not in fold_by_name:
            raise LockedFoldVerificationError(f"duplicate/unknown classification record: {key}")
        seen_classification.add(key)
        source = fold_by_name[filename]
        expected_split = "val" if source.fold == held_out else "train"
        if row.get("split") != expected_split or row.get("label") != source.label:
            raise LockedFoldVerificationError(
                f"classification split/label differs from Fold for {key}"
            )
        operation = row.get("operation")
        if operation not in {"hardlink", "byte_copy"}:
            raise LockedFoldVerificationError(f"invalid classification operation for {key}")
        destination = (
            manifest_path.parent
            / "classification"
            / f"fold_{held_out}"
            / expected_split
            / str(source.label)
            / filename
        )
        if not destination.is_file():
            raise LockedFoldVerificationError(f"classification image is missing: {destination}")
        canonical = processed_dir / filename
        if operation == "hardlink":
            try:
                if not destination.samefile(canonical):
                    raise LockedFoldVerificationError(
                        f"classification hardlink differs from canonical image: {destination}"
                    )
            except OSError as exc:
                raise LockedFoldVerificationError(
                    f"cannot inspect classification hardlink {destination}: {exc}"
                ) from exc
        else:
            try:
                if destination.samefile(canonical):
                    raise LockedFoldVerificationError(
                        f"classification byte_copy aliases canonical image: {destination}"
                    )
            except OSError as exc:
                raise LockedFoldVerificationError(
                    f"cannot inspect classification byte_copy {destination}: {exc}"
                ) from exc
            if sha256_file(destination) != sha256_file(canonical):
                raise LockedFoldVerificationError(
                    f"classification byte_copy differs from canonical image: {destination}"
                )
    expected_classification = {(held_out, row.filename) for held_out in range(5) for row in folds}
    if seen_classification != expected_classification:
        raise LockedFoldVerificationError("classification manifest does not cover all Fold views")
    classification_root = manifest_path.parent / "classification"
    for held_out in range(5):
        for split in ("train", "val"):
            for label in (0, 1):
                directory = classification_root / f"fold_{held_out}" / split / str(label)
                if not directory.is_dir():
                    raise LockedFoldVerificationError(
                        f"classification directory is missing: {directory}"
                    )
                expected_names = {
                    row.filename
                    for row in folds
                    if row.label == label and ("val" if row.fold == held_out else "train") == split
                }
                actual_names = {path.name for path in directory.iterdir() if path.is_file()}
                if actual_names != expected_names:
                    raise LockedFoldVerificationError(
                        f"classification directory has missing/extra files: {directory}"
                    )
    _verify_yolo_labels(processed_root=manifest_path.parent, index=index, boxes=boxes)
    _verify_detection_views(processed_root=manifest_path.parent, folds=folds)
    return len(seen)


def verify_processed_manifest(
    *,
    manifest_path: Path,
    corrections_yaml: Path,
    raw_images_dir: Path,
    index: DatasetIndex,
    signatures: Sequence[ImageSignature],
    folds: Sequence[FoldRecord],
    source_paths: Mapping[str, Path],
    fold_version: str,
) -> int:
    """Verify a processed dataset and normalize failures to one public error type."""

    try:
        return _verify_processed_manifest(
            manifest_path=manifest_path,
            corrections_yaml=corrections_yaml,
            raw_images_dir=raw_images_dir,
            index=index,
            signatures=signatures,
            folds=folds,
            source_paths=source_paths,
            fold_version=fold_version,
        )
    except LockedFoldVerificationError:
        raise
    except (DataValidationError, OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise LockedFoldVerificationError(
            f"processed-data verification failed for {manifest_path}: {exc}"
        ) from exc


def _verify_locked_fold_bundle(
    *,
    project_root: Path,
    config_path: Path,
    images_dir: Path,
    coco_json: Path,
    image_labels_json: Path,
    audit_report: Path,
    annotation_review: Path,
    image_hashes: Path,
    candidates: Path,
    candidates_metadata: Path,
    decisions: Path,
    locked_dir: Path,
    processed_manifest: Path,
    corrections_yaml: Path,
    expected_count: int,
) -> LockedFoldVerificationResult:
    """Verify the complete current-byte integrity chain without replaying Fold generation."""

    fold_version = locked_dir.parent.name if locked_dir.name == "locked" else locked_dir.name
    try:
        source_hashes = validate_source_provenance(
            audit_report=audit_report,
            images_dir=images_dir,
            coco_json=coco_json,
            image_labels_json=image_labels_json,
        )
        candidate_metadata = validate_candidates_metadata(
            candidates_metadata,
            image_hashes=image_hashes,
            candidates=candidates,
            source_hashes=source_hashes,
            config_path=None,
            expected_version=fold_version,
        )
        _verify_candidate_inputs(
            candidate_metadata,
            project_root=project_root,
            config_path=config_path,
        )
        candidate_rows = _verify_candidate_rows(
            candidate_metadata,
            image_hashes=image_hashes,
            candidates=candidates,
        )
        warnings = historical_code_warnings(candidate_metadata, project_root=project_root)
        evidence = validate_approved_annotation_review(
            annotation_review,
            project_root=project_root,
        )
        threshold = _integer(
            _mapping(candidate_metadata.get("thresholds"), "candidate thresholds").get(
                "large_component_size"
            ),
            "candidate thresholds.large_component_size",
            minimum=2,
        )
        history = decisions.with_name("review_history.csv")
        validate_review_audit_for_lock(
            candidates,
            decisions,
            history,
            large_component_size=threshold,
        )
        decision_rows = read_decisions(decisions)
        index = load_dataset_index(coco_json, image_labels_json)
        signatures, signature_labels = read_signatures(image_hashes)
        verify_source_signature_binding(
            index=index,
            signatures=signatures,
            signature_labels=signature_labels,
            images_dir=images_dir,
        )
    except (
        AnnotationReviewError,
        DataValidationError,
        FoldGateError,
        OSError,
        TimeoutError,
        UnicodeError,
        ValueError,
    ) as exc:
        if isinstance(exc, LockedFoldVerificationError):
            raise
        raise LockedFoldVerificationError(f"locked Fold provenance gate failed: {exc}") from exc

    metadata_path = locked_dir / "folds.meta.json"
    review_manifest = evidence.manifest_path
    input_paths = {
        "image_hashes.csv": image_hashes,
        "candidates.csv": candidates,
        "review_decisions.csv": decisions,
        "train_coco.json": coco_json,
        "train_image.json": image_labels_json,
        "review_history.csv": history,
        "audit_report": audit_report,
        "candidates.meta.json": candidates_metadata,
        "data_config": config_path,
        "annotation_review.json": evidence.review_path,
        "annotation_manifest.csv": review_manifest,
    }
    locked = verify_locked_artifacts(
        locked_dir=locked_dir,
        signatures=signatures,
        labels=index.labels,
        small_fire_images=small_fire_filenames(index),
        input_paths=input_paths,
        expected_version=fold_version,
        expected_count=expected_count,
    )
    verify_locked_group_semantics(
        signatures=signatures,
        labels=index.labels,
        candidates=candidate_rows,
        decisions=decision_rows,
        locked_groups=locked.groups,
    )
    provenance = _mapping(locked.metadata.get("provenance"), "provenance")
    for field, value in source_hashes.items():
        if provenance.get(field) != value:
            raise LockedFoldVerificationError(f"locked source provenance drift: {field}")
    _verify_annotation_provenance(locked.metadata, evidence)

    processed_count = verify_processed_manifest(
        manifest_path=processed_manifest,
        corrections_yaml=corrections_yaml,
        raw_images_dir=images_dir,
        index=index,
        signatures=signatures,
        folds=locked.folds,
        source_paths={
            "train_coco.json": coco_json,
            "train_image.json": image_labels_json,
            "audit_report": audit_report,
            "folds.csv": locked_dir / "folds.csv",
            "folds.meta.json": metadata_path,
        },
        fold_version=fold_version,
    )
    return LockedFoldVerificationResult(
        fold_version=fold_version,
        image_count=len(locked.folds),
        group_count=len({row.group_id for row in locked.groups}),
        processed_image_count=processed_count,
        artifact_sha256={
            "candidates.meta.json": sha256_file(candidates_metadata),
            "groups.csv": sha256_file(locked_dir / "groups.csv"),
            "folds.csv": sha256_file(locked_dir / "folds.csv"),
            "folds.meta.json": sha256_file(metadata_path),
            "processed_manifest.json": sha256_file(processed_manifest),
        },
        warnings=warnings,
    )


def verify_locked_fold_bundle(
    *,
    project_root: Path,
    config_path: Path,
    images_dir: Path,
    coco_json: Path,
    image_labels_json: Path,
    audit_report: Path,
    annotation_review: Path,
    image_hashes: Path,
    candidates: Path,
    candidates_metadata: Path,
    decisions: Path,
    locked_dir: Path,
    processed_manifest: Path,
    corrections_yaml: Path,
    expected_count: int,
) -> LockedFoldVerificationResult:
    """Verify one locked bundle while exposing a single stable failure type."""

    try:
        return _verify_locked_fold_bundle(
            project_root=project_root,
            config_path=config_path,
            images_dir=images_dir,
            coco_json=coco_json,
            image_labels_json=image_labels_json,
            audit_report=audit_report,
            annotation_review=annotation_review,
            image_hashes=image_hashes,
            candidates=candidates,
            candidates_metadata=candidates_metadata,
            decisions=decisions,
            locked_dir=locked_dir,
            processed_manifest=processed_manifest,
            corrections_yaml=corrections_yaml,
            expected_count=expected_count,
        )
    except LockedFoldVerificationError:
        raise
    except (
        AnnotationReviewError,
        DataValidationError,
        FoldGateError,
        GroupingConflictError,
        ReviewIncompleteError,
        csv.Error,
        OSError,
        TimeoutError,
        UnicodeError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        raise LockedFoldVerificationError(f"locked Fold verification failed: {exc}") from exc
