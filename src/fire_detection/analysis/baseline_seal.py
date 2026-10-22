"""Evidence-only gate for the pre-official frozen baseline.

The seal is deliberately conservative.  It hashes and references existing artifacts, but
never copies or repairs them.  Missing evidence and incomplete human/GPU work produce a
valid ``blocked`` manifest; they can never be silently promoted to ``complete``.
"""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from fire_detection.config import ConfigError, load_yaml_mapping
from fire_detection.models.offline import sha256_file
from fire_detection.utils import atomic_write_json

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_GATES = {
    "selection": "selection",
    "reproduction": "reproduction",
    "smoke_review": "smoke_review",
    "winner_errors": "winner_errors",
    "oof_analysis": "oof_analysis",
    "mock_runtime": "mock_runtime",
    "submission_validation": "submission_validation",
}
_PRIMARY_ARTIFACTS = {
    "selection": "manifest",
    "reproduction": "report",
    "smoke_review": "review",
    "winner_errors": "summary",
    "oof_analysis": "manifest",
    "mock_runtime": "runtime",
    "submission_validation": "report",
}
_ANALYSIS_OUTPUTS = {
    "ablation_report.csv",
    "ablation_fold_metrics.csv",
    "ablation_predictions.csv",
    "ablation_transitions.csv",
    "threshold_analysis.csv",
    "class_prior_simulation.csv",
    "threshold_summary.json",
    "bootstrap_confidence_interval.json",
    "class_prior_assumptions.json",
    "ablation_resource_report.json",
}


class BaselineSealError(ValueError):
    """Raised when a seal configuration or manifest is structurally invalid."""


class BaselineSealValidationError(BaselineSealError):
    """Raised when an existing seal no longer matches its referenced evidence."""


@dataclass(frozen=True)
class SealArtifactSpec:
    """One repository-relative artifact reference."""

    name: str
    path: Path
    expected_sha256: str | None


@dataclass(frozen=True)
class SealGateSpec:
    """One semantic evidence gate and the files it binds."""

    name: str
    checker: str
    artifacts: tuple[SealArtifactSpec, ...]


@dataclass(frozen=True)
class BaselineSealSpec:
    """Strict, immutable seal input contract."""

    project_root: Path
    config_path: Path
    require_clean_git: bool
    expected: dict[str, Any]
    gates: tuple[SealGateSpec, ...]


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ConfigError(
            f"{name} keys mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineSealError(f"{name} must be an object")
    return value


def _repo_path(raw: object, *, root: Path, name: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{name} must be non-empty repository-relative text")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or "\\" in raw:
        raise ConfigError(f"{name} must be a safe repository-relative POSIX path")
    resolved = (root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigError(f"{name} escapes the project root") from exc
    return resolved


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite")
    return result


def load_baseline_seal_spec(path: Path, *, project_root: Path) -> BaselineSealSpec:
    """Load the strict seal YAML without inspecting or changing any evidence artifact."""

    root = project_root.resolve()
    config = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        config.relative_to(root)
    except ValueError as exc:
        raise ConfigError("baseline seal config must stay inside the project root") from exc
    raw = load_yaml_mapping(config)
    _exact_keys(raw, {"schema_version", "policy", "expected", "evidence"}, "seal")
    if raw.get("schema_version") != 1:
        raise ConfigError("seal.schema_version must be 1")

    policy = _mapping(raw.get("policy"), "seal.policy")
    _exact_keys(policy, {"require_clean_git"}, "seal.policy")
    require_clean_git = policy.get("require_clean_git")
    if type(require_clean_git) is not bool:
        raise ConfigError("seal.policy.require_clean_git must be boolean")

    expected = dict(_mapping(raw.get("expected"), "seal.expected"))
    expected_keys = {
        "selection_sha256",
        "oof_rows",
        "smoke_review_rows",
        "winner_fp",
        "winner_fn",
        "bootstrap_iterations",
        "official_test_count",
        "detector_weight",
        "classifier_weight",
        "stable_threshold",
        "recall_threshold",
        "precision_threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    }
    _exact_keys(expected, expected_keys, "seal.expected")
    digest = expected["selection_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ConfigError("seal.expected.selection_sha256 must be lowercase SHA-256")
    for key in (
        "oof_rows",
        "smoke_review_rows",
        "winner_fp",
        "winner_fn",
        "bootstrap_iterations",
        "official_test_count",
        "tp",
        "fp",
        "fn",
        "tn",
    ):
        expected[key] = _positive_int(expected[key], f"seal.expected.{key}")
    for key in (
        "detector_weight",
        "classifier_weight",
        "stable_threshold",
        "recall_threshold",
        "precision_threshold",
        "precision",
        "recall",
        "f1",
    ):
        expected[key] = _finite_number(expected[key], f"seal.expected.{key}")

    evidence = _mapping(raw.get("evidence"), "seal.evidence")
    if set(evidence) != set(_REQUIRED_GATES):
        raise ConfigError(f"seal.evidence must contain exactly {sorted(_REQUIRED_GATES)}")
    gates: list[SealGateSpec] = []
    for gate_name in sorted(_REQUIRED_GATES):
        value = _mapping(evidence[gate_name], f"seal.evidence.{gate_name}")
        _exact_keys(value, {"checker", "artifacts"}, f"seal.evidence.{gate_name}")
        checker = value.get("checker")
        if checker != _REQUIRED_GATES[gate_name]:
            raise ConfigError(
                f"seal.evidence.{gate_name}.checker must be {_REQUIRED_GATES[gate_name]!r}"
            )
        raw_artifacts = value.get("artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise ConfigError(f"seal.evidence.{gate_name}.artifacts must be a non-empty list")
        artifacts: list[SealArtifactSpec] = []
        seen: set[str] = set()
        for index, raw_artifact in enumerate(raw_artifacts):
            item = _mapping(raw_artifact, f"seal.evidence.{gate_name}.artifacts[{index}]")
            allowed = {"name", "path", "expected_sha256"}
            if not set(item).issubset(allowed) or not {"name", "path"}.issubset(item):
                raise ConfigError(f"seal.evidence.{gate_name}.artifacts[{index}] has invalid keys")
            name = item.get("name")
            if not isinstance(name, str) or not name or name in seen:
                raise ConfigError(f"invalid or duplicate artifact name in gate {gate_name}")
            seen.add(name)
            expected_sha = item.get("expected_sha256")
            if expected_sha is not None and (
                not isinstance(expected_sha, str) or _SHA256.fullmatch(expected_sha) is None
            ):
                raise ConfigError(f"{gate_name}.{name}.expected_sha256 is invalid")
            artifacts.append(
                SealArtifactSpec(
                    name=name,
                    path=_repo_path(item.get("path"), root=root, name=f"{gate_name}.{name}.path"),
                    expected_sha256=expected_sha,
                )
            )
        if _PRIMARY_ARTIFACTS[gate_name] not in seen:
            raise ConfigError(
                f"gate {gate_name} must include artifact {_PRIMARY_ARTIFACTS[gate_name]!r}"
            )
        if gate_name == "selection":
            primary = next(item for item in artifacts if item.name == "manifest")
            if primary.expected_sha256 != expected["selection_sha256"]:
                raise ConfigError("selection manifest must bind expected selection_sha256")
        gates.append(SealGateSpec(gate_name, str(checker), tuple(artifacts)))
    return BaselineSealSpec(root, config, bool(require_clean_git), expected, tuple(gates))


def _portable(path: Path, *, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_record(spec: SealArtifactSpec, *, root: Path) -> tuple[dict[str, object], list[str]]:
    record: dict[str, object] = {
        "name": spec.name,
        "path": _portable(spec.path, root=root),
        "exists": spec.path.is_file(),
        "size_bytes": None,
        "sha256": None,
        "expected_sha256": spec.expected_sha256,
    }
    reasons: list[str] = []
    if not spec.path.is_file():
        reasons.append(f"required artifact is missing: {record['path']}")
        return record, reasons
    digest = sha256_file(spec.path)
    record["size_bytes"] = spec.path.stat().st_size
    record["sha256"] = digest
    if spec.expected_sha256 is not None and digest != spec.expected_sha256:
        reasons.append(
            f"artifact hash mismatch: {record['path']} expected "
            f"{spec.expected_sha256}, got {digest}"
        )
    return record, reasons


def _read_json(path: Path, name: str, reasons: list[str]) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        reasons.append(f"cannot read {name} JSON: {exc}")
        return None
    if not isinstance(value, dict):
        reasons.append(f"{name} JSON must be an object")
        return None
    return value


def _expect_equal(actual: object, expected: object, name: str, reasons: list[str]) -> None:
    if actual != expected:
        reasons.append(f"{name} must equal {expected!r}, got {actual!r}")


def _expect_float(actual: object, expected: float, name: str, reasons: list[str]) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        reasons.append(f"{name} must be numeric")
    elif not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        reasons.append(f"{name} differs from frozen value {expected}: {actual}")


def _check_selection(
    paths: Mapping[str, Path], expected: Mapping[str, Any], reasons: list[str]
) -> None:
    payload = _read_json(paths["manifest"], "selection manifest", reasons)
    if payload is None:
        return
    _expect_equal(payload.get("kind"), "final_selection", "selection.kind", reasons)
    _expect_equal(payload.get("status"), "complete", "selection.status", reasons)
    fusion = _mapping(payload.get("fusion"), "selection.fusion")
    _expect_float(
        fusion.get("detector_weight"),
        float(expected["detector_weight"]),
        "detector weight",
        reasons,
    )
    _expect_float(
        fusion.get("classifier_weight"),
        float(expected["classifier_weight"]),
        "classifier weight",
        reasons,
    )
    profiles = _mapping(payload.get("threshold_profiles"), "selection.threshold_profiles")
    for profile, key in (
        ("stable", "stable_threshold"),
        ("recall", "recall_threshold"),
        ("precision", "precision_threshold"),
    ):
        value = _mapping(profiles.get(profile), f"threshold_profiles.{profile}").get("threshold")
        _expect_float(value, float(expected[key]), f"{profile} threshold", reasons)
    metrics = _mapping(
        _mapping(payload.get("metrics"), "selection.metrics").get("cross_fitted"),
        "cross_fitted metrics",
    )
    for key in ("precision", "recall", "f1"):
        _expect_float(metrics.get(key), float(expected[key]), f"selection metric {key}", reasons)
    for key in ("tp", "fp", "fn", "tn"):
        _expect_equal(metrics.get(key), expected[key], f"selection metric {key}", reasons)


def _check_reproduction(
    paths: Mapping[str, Path], expected: Mapping[str, Any], reasons: list[str]
) -> None:
    payload = _read_json(paths["report"], "reproduction report", reasons)
    if payload is None:
        return
    _expect_equal(payload.get("kind"), "clean_winner_reproduction", "reproduction.kind", reasons)
    _expect_equal(payload.get("status"), "passed", "reproduction.status", reasons)
    _expect_equal(payload.get("comparison_only"), True, "comparison_only", reasons)
    _expect_equal(payload.get("training_invoked"), False, "training_invoked", reasons)
    comparison = _mapping(payload.get("comparison"), "reproduction.comparison")
    _expect_equal(comparison.get("passed"), True, "reproduction comparison passed", reasons)
    _expect_equal(comparison.get("rows"), expected["oof_rows"], "reproduction rows", reasons)
    _expect_equal(comparison.get("decision_changes"), 0, "reproduction decision changes", reasons)
    _expect_equal(
        comparison.get("tolerance_exceeded_rows"), 0, "reproduction tolerance rows", reasons
    )
    metrics = _mapping(comparison.get("candidate_metrics"), "candidate metrics")
    for key in ("precision", "recall", "f1"):
        _expect_float(metrics.get(key), float(expected[key]), f"reproduction metric {key}", reasons)
    for key in ("tp", "fp", "fn", "tn"):
        _expect_equal(metrics.get(key), expected[key], f"reproduction metric {key}", reasons)
    current = _mapping(
        _mapping(payload.get("git"), "reproduction.git").get("current"), "reproduction current git"
    )
    _expect_equal(current.get("dirty"), False, "reproduction worktree dirty", reasons)


def _read_csv(path: Path, name: str, reasons: list[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        reasons.append(f"cannot read {name} CSV: {exc}")
        return []


def _check_smoke_review(
    paths: Mapping[str, Path], expected: Mapping[str, Any], reasons: list[str]
) -> None:
    payload = _read_json(paths["review"], "Smoke2 v2 review", reasons)
    if payload is None:
        return
    _expect_equal(payload.get("status"), "approved", "Smoke2 bundle status", reasons)
    records = payload.get("records")
    if not isinstance(records, list):
        reasons.append("Smoke2 records must be a list")
        return
    count = int(expected["smoke_review_rows"])
    _expect_equal(len(records), count, "Smoke2 row count", reasons)
    filenames: set[str] = set()
    pending_files: list[str] = []
    secondary_files: list[str] = []
    reviewer_conflicts: list[str] = []
    eligibility_files: list[str] = []
    for index, value in enumerate(records):
        if not isinstance(value, dict):
            reasons.append(f"Smoke2 record {index} is not an object")
            continue
        filename = value.get("filename")
        if not isinstance(filename, str) or not filename or filename in filenames:
            reasons.append(f"Smoke2 record {index} has invalid/duplicate filename")
            continue
        filenames.add(filename)
        disposition = value.get("disposition")
        if disposition not in {"confirmed_no_fire", "uncertain", "suspected_label_error"}:
            pending_files.append(filename)
            continue
        final = disposition
        if disposition in {"uncertain", "suspected_label_error"}:
            secondary = value.get("secondary_disposition")
            if secondary not in {"confirmed_no_fire", "uncertain", "suspected_label_error"}:
                secondary_files.append(filename)
                continue
            if value.get("secondary_reviewer") == value.get("primary_reviewer"):
                reviewer_conflicts.append(filename)
            final = secondary
        eligible = value.get("training_eligible")
        if eligible is not (final == "confirmed_no_fire"):
            eligibility_files.append(filename)
    for label, values in (
        ("pending/invalid primary decisions", pending_files),
        ("required second reviews", secondary_files),
        ("same-person second reviews", reviewer_conflicts),
        ("invalid derived training eligibility", eligibility_files),
    ):
        if values:
            reasons.append(f"Smoke2 has {len(values)} {label}; examples={values[:5]}")
    if not payload.get("finalized_by") or not payload.get("finalized_at"):
        reasons.append("Smoke2 approved bundle lacks finalizer provenance")
    summary_path = paths.get("summary")
    if summary_path is not None and summary_path.is_file():
        summary = _read_json(summary_path, "Smoke2 summary", reasons)
        if summary is not None:
            for key, smoke_expected in (
                ("status", "approved"),
                ("total", count),
                ("complete", count),
                ("pending", 0),
                ("secondary_required", 0),
            ):
                _expect_equal(summary.get(key), smoke_expected, f"Smoke2 summary {key}", reasons)
    csv_path = paths.get("csv")
    if csv_path is not None and csv_path.is_file():
        rows = _read_csv(csv_path, "Smoke2 review", reasons)
        _expect_equal(len(rows), count, "Smoke2 CSV rows", reasons)


def _check_winner_errors(
    paths: Mapping[str, Path], expected: Mapping[str, Any], reasons: list[str], *, root: Path
) -> None:
    payload = _read_json(paths["summary"], "winner error summary", reasons)
    if payload is None:
        return
    _expect_equal(payload.get("kind"), "round3_winner_error_report", "error report kind", reasons)
    _expect_equal(
        payload.get("status"),
        "mechanical_review_pending",
        "mechanical error report status",
        reasons,
    )
    counts = _mapping(payload.get("counts"), "error counts")
    _expect_equal(counts.get("fp"), expected["winner_fp"], "winner FP count", reasons)
    _expect_equal(counts.get("fn"), expected["winner_fn"], "winner FN count", reasons)
    human = _mapping(payload.get("human_review"), "error human review")
    total = int(expected["winner_fp"]) + int(expected["winner_fn"])
    _expect_equal(human.get("required_rows"), total, "required human error rows", reasons)
    _expect_equal(human.get("completed_rows"), 0, "mechanical completed human rows", reasons)

    mechanical_rows: list[dict[str, str]] = []
    for name, count, confusion in (
        ("mechanical_fp", int(expected["winner_fp"]), "FP"),
        ("mechanical_fn", int(expected["winner_fn"]), "FN"),
    ):
        path = paths.get(name)
        if path is None or not path.is_file():
            continue
        current = _read_csv(path, f"winner mechanical {confusion}", reasons)
        _expect_equal(len(current), count, f"winner mechanical {confusion} rows", reasons)
        for row in current:
            if row.get("confusion") != confusion:
                reasons.append(f"winner mechanical {confusion} contains another class")
                break
            if row.get("review_status") != "pending" or any(
                row.get(field, "").strip()
                for field in (
                    "error_category",
                    "suspected_label_issue",
                    "suggested_action",
                    "reviewer",
                )
            ):
                reasons.append(f"mechanical {confusion} rows must remain semantic-free and pending")
                break
        mechanical_rows.extend(current)
    mechanical_names = [row.get("filename", "") for row in mechanical_rows]
    if mechanical_rows and (
        not all(mechanical_names) or len(set(mechanical_names)) != len(mechanical_names)
    ):
        reasons.append("mechanical FP/FN filenames are empty or duplicated")

    draft_path = paths.get("draft")
    if draft_path is not None and draft_path.is_file():
        draft = _read_json(draft_path, "winner error review draft", reasons)
        if draft is not None:
            _expect_equal(
                draft.get("review_kind"),
                "round3_winner_error_human_review",
                "winner review kind",
                reasons,
            )
            records = draft.get("records")
            if not isinstance(records, list) or len(records) != total:
                reasons.append("winner error review draft must contain all 107 rows")
            if draft.get("status") != "ready_for_finalize":
                statuses: dict[str, int] = {}
                if isinstance(records, list):
                    for record in records:
                        if isinstance(record, dict):
                            status = str(record.get("review_status"))
                            statuses[status] = statuses.get(status, 0) + 1
                reasons.append(
                    "winner error human review is not ready_for_finalize; "
                    f"status={draft.get('status')!r}, rows={statuses}"
                )

    finalized_path = paths.get("finalized_review")
    finalized_records: list[dict[str, Any]] = []
    if finalized_path is not None and finalized_path.is_file():
        finalized = _read_json(finalized_path, "finalized winner error review", reasons)
        if finalized is not None:
            _expect_equal(finalized.get("status"), "finalized", "finalized review status", reasons)
            _expect_equal(
                finalized.get("review_kind"),
                "round3_winner_error_human_review",
                "finalized review kind",
                reasons,
            )
            raw_records = finalized.get("records")
            if not isinstance(raw_records, list) or len(raw_records) != total:
                reasons.append("finalized winner error review must contain all 107 rows")
            else:
                finalized_records = [record for record in raw_records if isinstance(record, dict)]
                if len(finalized_records) != total:
                    reasons.append("finalized winner error review contains a non-object row")
            if not finalized.get("finalized_by") or not finalized.get("finalized_at"):
                reasons.append("finalized winner error review lacks finalizer provenance")

    missing_fields: dict[str, list[str]] = {
        "error_category": [],
        "suggested_action": [],
        "reviewer": [],
    }
    not_reviewed: list[str] = []
    invalid_label_issue: list[str] = []
    invalid_second_review: list[str] = []
    for row in finalized_records:
        filename = str(row.get("filename", "<unknown>"))
        for field in missing_fields:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                missing_fields[field].append(filename)
        if row.get("review_status") != "reviewed":
            not_reviewed.append(filename)
        label_issue = row.get("suspected_label_issue")
        if label_issue not in {"no", "yes", "uncertain"}:
            invalid_label_issue.append(filename)
        if label_issue in {"yes", "uncertain"} and (
            not row.get("second_reviewer")
            or str(row.get("second_reviewer", "")).casefold()
            == str(row.get("reviewer", "")).casefold()
            or row.get("label_issue_conclusion")
            not in {"no_label_issue", "label_issue", "unresolved"}
        ):
            invalid_second_review.append(filename)
    missing_fields = {field: values for field, values in missing_fields.items() if values}
    for field, values in missing_fields.items():
        reasons.append(
            f"winner errors have {len(values)} finalized rows missing {field}; "
            f"examples={values[:5]}"
        )
    if not_reviewed:
        reasons.append(
            f"winner errors have {len(not_reviewed)} non-reviewed rows; examples={not_reviewed[:5]}"
        )
    if invalid_label_issue:
        reasons.append(
            "winner errors have "
            f"{len(invalid_label_issue)} invalid suspected_label_issue values; "
            f"examples={invalid_label_issue[:5]}"
        )
    if invalid_second_review:
        reasons.append(
            "winner errors have "
            f"{len(invalid_second_review)} invalid independent second reviews; "
            f"examples={invalid_second_review[:5]}"
        )

    final_rows: list[dict[str, str]] = []
    for name, count, confusion in (
        ("final_fp", int(expected["winner_fp"]), "FP"),
        ("final_fn", int(expected["winner_fn"]), "FN"),
    ):
        path = paths.get(name)
        if path is None or not path.is_file():
            continue
        current = _read_csv(path, f"final reviewed {confusion}", reasons)
        _expect_equal(len(current), count, f"final reviewed {confusion} rows", reasons)
        if any(row.get("confusion") != confusion for row in current):
            reasons.append(f"final reviewed {confusion} CSV contains another class")
        final_rows.extend(current)
    final_names = [row.get("filename", "") for row in final_rows]
    reviewed_names = [str(row.get("filename", "")) for row in finalized_records]
    if final_rows and (not all(final_names) or len(set(final_names)) != len(final_names)):
        reasons.append("final reviewed FP/FN filenames are empty or duplicated")
    if final_rows and set(final_names) != set(reviewed_names):
        reasons.append("final reviewed FP/FN CSV rows differ from finalized review rows")

    summary_path = paths.get("review_summary")
    if summary_path is not None and summary_path.is_file():
        review_summary = _read_json(summary_path, "winner error review summary", reasons)
        if review_summary is not None:
            _expect_equal(
                review_summary.get("kind"),
                "round3_winner_error_human_review_summary",
                "winner review summary kind",
                reasons,
            )
            _expect_equal(
                review_summary.get("status"), "finalized", "winner review summary status", reasons
            )
            draft_record = _mapping(
                review_summary.get("draft_review"), "winner review summary draft_review"
            )
            draft_path = paths.get("draft")
            if draft_path is not None and draft_path.is_file():
                _expect_equal(
                    draft_record.get("path"),
                    _portable(draft_path, root=root),
                    "winner review summary draft path",
                    reasons,
                )
                _expect_equal(
                    draft_record.get("sha256"),
                    sha256_file(draft_path),
                    "winner review summary draft hash",
                    reasons,
                )
            summary_counts = _mapping(review_summary.get("counts"), "winner review counts")
            for key, summary_expected in (
                ("total", total),
                ("fp", int(expected["winner_fp"])),
                ("fn", int(expected["winner_fn"])),
                ("reviewed", total),
                ("pending", 0),
                ("needs_second_review", 0),
            ):
                _expect_equal(
                    summary_counts.get(key), summary_expected, f"winner review {key}", reasons
                )
            top = review_summary.get("top_three_error_categories")
            if not isinstance(top, list) or len(top) != 3:
                reasons.append("winner review summary must contain the top three error categories")
            actions = review_summary.get("suggested_action_counts")
            if not isinstance(actions, dict) or not actions:
                reasons.append("winner review summary must contain human suggested-action counts")

    bundle_path = paths.get("bundle")
    if bundle_path is not None and bundle_path.is_file():
        bundle = _read_json(bundle_path, "winner error review bundle", reasons)
        if bundle is not None:
            _expect_equal(
                bundle.get("kind"),
                "round3_winner_error_human_review_bundle",
                "winner review bundle kind",
                reasons,
            )
            _expect_equal(
                bundle.get("semantic_source"),
                "human_review_only",
                "winner review semantic source",
                reasons,
            )
            files = _mapping(bundle.get("files"), "winner review bundle files")
            expected_files = {
                "finalized_review.json",
                "final_fp_report.csv",
                "final_fn_report.csv",
                "review_summary.json",
            }
            if set(files) != expected_files:
                reasons.append("winner review bundle file set is incomplete or unexpected")
            for filename in sorted(expected_files.intersection(files)):
                record = _mapping(files[filename], f"winner review bundle {filename}")
                relative = record.get("path")
                if not isinstance(relative, str) or Path(relative).name != relative:
                    reasons.append(f"winner review bundle path is invalid: {filename}")
                    continue
                artifact = (bundle_path.parent / relative).resolve()
                try:
                    artifact.relative_to(bundle_path.parent.resolve())
                except ValueError:
                    reasons.append(f"winner review bundle path escapes: {filename}")
                    continue
                if not artifact.is_file() or sha256_file(artifact) != record.get("sha256"):
                    reasons.append(f"winner review bundle artifact hash drifted: {filename}")


def _resolve_manifest_output(path: object, *, root: Path) -> Path | None:
    if not isinstance(path, str) or not path:
        return None
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    resolved = (root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _check_external_runtime_benchmark(
    paths: Mapping[str, Path], reasons: list[str], *, root: Path
) -> bool:
    """Validate the separately published 100-image GPU benchmark closure."""

    required = {"resources", "benchmark_manifest", "resource_csv"}
    missing = sorted(required.difference(paths))
    if missing:
        reasons.append(f"external runtime benchmark artifacts are missing: {missing}")
        return False

    report = _read_json(paths["resources"], "ablation runtime report", reasons)
    manifest = _read_json(paths["benchmark_manifest"], "ablation runtime manifest", reasons)
    if report is None or manifest is None:
        return False

    local_reasons: list[str] = []
    _expect_equal(
        report.get("kind"),
        "pre_official_ablation_runtime_benchmark",
        "runtime report kind",
        local_reasons,
    )
    _expect_equal(report.get("status"), "complete", "runtime report status", local_reasons)
    _expect_equal(report.get("engineering_only"), True, "runtime engineering_only", local_reasons)
    _expect_equal(
        report.get("training_performed"), False, "runtime training_performed", local_reasons
    )

    protocol = report.get("protocol")
    sample = report.get("sample")
    measurements = report.get("measurements")
    if not isinstance(protocol, Mapping):
        local_reasons.append("runtime report protocol must be an object")
    else:
        _expect_equal(protocol.get("sample_count"), 100, "runtime sample_count", local_reasons)
        _expect_equal(protocol.get("repeats"), 3, "runtime repeats", local_reasons)
        _expect_equal(
            protocol.get("warmup_count_per_model_component"),
            10,
            "runtime warmup count",
            local_reasons,
        )
        _expect_equal(
            protocol.get("model_load_time_included"),
            False,
            "runtime model-load inclusion",
            local_reasons,
        )
    if not isinstance(sample, Mapping):
        local_reasons.append("runtime report sample must be an object")
    else:
        _expect_equal(sample.get("selected_count"), 100, "runtime selected_count", local_reasons)
        _expect_equal(
            sample.get("labels_consumed"), False, "runtime labels_consumed", local_reasons
        )

    expected_candidates = {
        "detector_full_original",
        "classifier_original",
        "fusion_full_original",
        "fusion_original_with_slicing",
        "fusion_full_tta",
        "fusion_slicing_tta_lofo",
        "frozen_winner",
    }
    candidate_names: set[str] = set()
    if not isinstance(measurements, list) or len(measurements) != len(expected_candidates):
        local_reasons.append("runtime report must contain exactly seven candidate measurements")
    else:
        for index, row in enumerate(measurements):
            if not isinstance(row, Mapping):
                local_reasons.append(f"runtime measurement {index} must be an object")
                continue
            candidate = row.get("candidate")
            if isinstance(candidate, str):
                candidate_names.add(candidate)
            _expect_equal(
                row.get("status"), "complete", f"runtime measurement {index} status", local_reasons
            )
            _expect_equal(
                row.get("sample_count"),
                100,
                f"runtime measurement {index} sample_count",
                local_reasons,
            )
            _expect_equal(
                row.get("repeats"), 3, f"runtime measurement {index} repeats", local_reasons
            )
            for field in (
                "mean_seconds_per_image",
                "p50_seconds_per_image",
                "p95_seconds_per_image",
                "peak_gpu_memory_bytes",
                "peak_process_tree_ram_bytes",
            ):
                value = row.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    local_reasons.append(
                        f"runtime measurement {index} {field} must be finite and positive"
                    )
    if candidate_names != expected_candidates:
        local_reasons.append(
            "runtime candidate set differs from the seven frozen ablation candidates"
        )

    _expect_equal(
        manifest.get("kind"),
        "pre_official_ablation_runtime_benchmark_manifest",
        "runtime manifest kind",
        local_reasons,
    )
    _expect_equal(manifest.get("status"), "complete", "runtime manifest status", local_reasons)
    _expect_equal(
        manifest.get("engineering_only"), True, "runtime manifest engineering_only", local_reasons
    )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        local_reasons.append("runtime manifest outputs must be an object")
    else:
        for filename, record in outputs.items():
            if not isinstance(filename, str) or not isinstance(record, Mapping):
                local_reasons.append("runtime manifest output records are malformed")
                continue
            output = _resolve_manifest_output(record.get("path"), root=root)
            digest = record.get("sha256")
            if output is None or not output.is_file() or not isinstance(digest, str):
                local_reasons.append(f"runtime manifest output is unavailable: {filename}")
            elif sha256_file(output) != digest:
                local_reasons.append(f"runtime manifest output hash drifted: {filename}")
        for filename, artifact_name in (
            ("ablation_resource_report.json", "resources"),
            ("ablation_resource_report.csv", "resource_csv"),
        ):
            record = outputs.get(filename)
            if not isinstance(record, Mapping):
                local_reasons.append(f"runtime manifest misses required output: {filename}")
                continue
            output = _resolve_manifest_output(record.get("path"), root=root)
            if output != paths[artifact_name].resolve():
                local_reasons.append(f"runtime manifest path does not bind {filename}")

    reasons.extend(local_reasons)
    return not local_reasons


def _check_oof_analysis(
    paths: Mapping[str, Path], expected: Mapping[str, Any], reasons: list[str], *, root: Path
) -> None:
    payload = _read_json(paths["manifest"], "OOF analysis manifest", reasons)
    if payload is None:
        return
    _expect_equal(payload.get("kind"), "pre_official_oof_diagnostics", "analysis kind", reasons)
    _expect_equal(payload.get("status"), "complete", "analysis status", reasons)
    runtime = _mapping(payload.get("runtime_benchmark"), "analysis runtime_benchmark")
    external_requested = any(
        name in paths for name in ("resources", "benchmark_manifest", "resource_csv")
    )
    external_complete = False
    if external_requested:
        external_complete = _check_external_runtime_benchmark(paths, reasons, root=root)
    if runtime.get("status") not in {"complete", "passed"} and not external_complete:
        reasons.append("100-image ablation runtime benchmark has not completed")
    outputs = _mapping(payload.get("outputs"), "analysis outputs")
    missing = sorted(_ANALYSIS_OUTPUTS.difference(outputs))
    if missing:
        reasons.append(f"analysis manifest misses required outputs: {missing}")
    for name in sorted(_ANALYSIS_OUTPUTS.intersection(outputs)):
        record = _mapping(outputs[name], f"analysis output {name}")
        output = _resolve_manifest_output(record.get("path"), root=root)
        digest = record.get("sha256")
        if output is None or not output.is_file() or not isinstance(digest, str):
            reasons.append(f"analysis output is unavailable: {name}")
        elif sha256_file(output) != digest:
            reasons.append(f"analysis output hash drifted: {name}")
    bootstrap_path = paths.get("bootstrap")
    if bootstrap_path is not None and bootstrap_path.is_file():
        bootstrap = _read_json(bootstrap_path, "bootstrap report", reasons)
        if bootstrap is not None:
            _expect_equal(
                bootstrap.get("iterations"),
                expected["bootstrap_iterations"],
                "bootstrap iterations",
                reasons,
            )


def _check_mock_runtime(
    paths: Mapping[str, Path], expected: Mapping[str, Any], reasons: list[str]
) -> None:
    payload = _read_json(paths["runtime"], "mock runtime report", reasons)
    if payload is None:
        return
    count = int(expected["official_test_count"])
    for key, value in (
        ("kind", "mock_test_1198_runtime_report"),
        ("engineering_only", True),
        ("status", "complete"),
        ("count", count),
        ("success_count", count),
        ("failure_count", 0),
        ("exit_code", 0),
        ("resume_verified", True),
    ):
        _expect_equal(payload.get(key), value, f"mock runtime {key}", reasons)
    mock_manifest_path = paths.get("mock_manifest")
    if mock_manifest_path is not None and mock_manifest_path.is_file():
        manifest = _read_json(mock_manifest_path, "mock input manifest", reasons)
        if manifest is not None:
            _expect_equal(manifest.get("engineering_only"), True, "mock engineering_only", reasons)
            output = _mapping(manifest.get("output"), "mock output")
            _expect_equal(output.get("count"), count, "mock input count", reasons)
    inference_path = paths.get("inference_manifest")
    if inference_path is not None and inference_path.is_file():
        inference = _read_json(inference_path, "mock inference manifest", reasons)
        if inference is not None:
            _expect_equal(inference.get("status"), "complete", "mock inference status", reasons)
    input_path = paths.get("input_manifest")
    if input_path is not None and input_path.is_file():
        input_manifest = _read_json(input_path, "mock input audit manifest", reasons)
        if input_manifest is not None:
            _expect_equal(input_manifest.get("status"), "valid", "mock input audit status", reasons)
            images = input_manifest.get("images")
            if not isinstance(images, list) or len(images) != count:
                reasons.append("mock input audit does not contain exactly 1198 images")


def _check_submission_validation(
    paths: Mapping[str, Path], expected: Mapping[str, Any], reasons: list[str], *, root: Path
) -> None:
    payload = _read_json(paths["report"], "submission validation report", reasons)
    if payload is None:
        return
    count = int(expected["official_test_count"])
    for key, value in (("status", "valid"), ("primary", "stable"), ("expected_count", count)):
        _expect_equal(payload.get(key), value, f"submission validation {key}", reasons)
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"stable", "recall", "precision"}:
        reasons.append("submission validation must contain stable/recall/precision outputs")
        return
    for profile in ("stable", "recall", "precision"):
        record = _mapping(outputs[profile], f"submission {profile}")
        _expect_equal(record.get("rows"), count, f"submission {profile} rows", reasons)
        _expect_equal(
            record.get("integer_binary_values"),
            True,
            f"submission {profile} integer values",
            reasons,
        )
        _expect_equal(record.get("utf8_bom"), False, f"submission {profile} BOM", reasons)
        output = _resolve_manifest_output(record.get("path"), root=root)
        digest = record.get("sha256")
        if output is None or not output.is_file() or not isinstance(digest, str):
            reasons.append(f"submission {profile} output is unavailable")
        elif sha256_file(output) != digest:
            reasons.append(f"submission {profile} hash drifted")


def _semantic_reasons(gate: SealGateSpec, *, expected: Mapping[str, Any], root: Path) -> list[str]:
    reasons: list[str] = []
    paths = {artifact.name: artifact.path for artifact in gate.artifacts}
    primary = paths[_PRIMARY_ARTIFACTS[gate.name]]
    if not primary.is_file():
        return reasons
    if gate.checker == "selection":
        _check_selection(paths, expected, reasons)
    elif gate.checker == "reproduction":
        _check_reproduction(paths, expected, reasons)
    elif gate.checker == "smoke_review":
        _check_smoke_review(paths, expected, reasons)
    elif gate.checker == "winner_errors":
        _check_winner_errors(paths, expected, reasons, root=root)
    elif gate.checker == "oof_analysis":
        _check_oof_analysis(paths, expected, reasons, root=root)
    elif gate.checker == "mock_runtime":
        _check_mock_runtime(paths, expected, reasons)
    elif gate.checker == "submission_validation":
        _check_submission_validation(paths, expected, reasons, root=root)
    else:  # pragma: no cover - rejected by the config loader
        reasons.append(f"unsupported checker: {gate.checker}")
    return reasons


def _run_git(root: Path, arguments: Sequence[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return 127, ""
    return result.returncode, result.stdout.strip()


def git_snapshot(root: Path) -> dict[str, object]:
    """Capture HEAD, branch/detached state, and exact porcelain entries."""

    head_code, head = _run_git(root, ["rev-parse", "HEAD"])
    status_code, status = _run_git(root, ["status", "--porcelain=v1", "--untracked-files=all"])
    branch_code, branch = _run_git(root, ["symbolic-ref", "--short", "-q", "HEAD"])
    available = head_code == 0 and status_code == 0 and bool(head)
    entries = status.splitlines() if status else []
    return {
        "available": available,
        "commit": head if available else None,
        "branch": branch if branch_code == 0 and branch else None,
        "detached": available and branch_code != 0,
        "dirty": bool(entries) if available else None,
        "porcelain": entries,
    }


def evaluate_baseline_seal(spec: BaselineSealSpec) -> dict[str, Any]:
    """Hash current evidence and return either a complete or an explicitly blocked seal."""

    git = git_snapshot(spec.project_root)
    blockers: list[dict[str, str]] = []
    if spec.require_clean_git:
        if git["available"] is not True:
            blockers.append({"gate": "git", "reason": "Git state is unavailable"})
        elif git["dirty"] is not False:
            blockers.append({"gate": "git", "reason": "authoritative worktree is not clean"})

    evidence: dict[str, Any] = {}
    for gate in spec.gates:
        reasons: list[str] = []
        records: list[dict[str, object]] = []
        for artifact in gate.artifacts:
            record, artifact_reasons = _artifact_record(artifact, root=spec.project_root)
            records.append(record)
            reasons.extend(artifact_reasons)
        try:
            reasons.extend(_semantic_reasons(gate, expected=spec.expected, root=spec.project_root))
        except BaselineSealError as exc:
            reasons.append(str(exc))
        unique_reasons = list(dict.fromkeys(reasons))
        status = "passed" if not unique_reasons else "blocked"
        evidence[gate.name] = {
            "checker": gate.checker,
            "status": status,
            "artifacts": records,
            "reasons": unique_reasons,
        }
        blockers.extend({"gate": gate.name, "reason": reason} for reason in unique_reasons)

    status = "complete" if not blockers else "blocked"
    config_record = {
        "path": _portable(spec.config_path, root=spec.project_root),
        "sha256": sha256_file(spec.config_path),
    }
    return {
        "schema_version": 1,
        "kind": "pre_official_baseline_seal",
        "status": status,
        "completion_claim_allowed": status == "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "artifact_policy": "references_only_no_copy_no_repair",
        "config": config_record,
        "policy": {"require_clean_git": spec.require_clean_git},
        "expected": dict(spec.expected),
        "git": git,
        "evidence": evidence,
        "blockers": blockers,
        "claims": {
            "official_leaderboard_result": False,
            "mock_results_are_model_metrics": False,
            "winner_content_copied": False,
        },
    }


def write_baseline_manifest(spec: BaselineSealSpec, output: Path) -> dict[str, Any]:
    """Evaluate and atomically write one immutable baseline manifest."""

    destination = (
        output.resolve() if output.is_absolute() else (spec.project_root / output).resolve()
    )
    try:
        destination.relative_to(spec.project_root)
    except ValueError as exc:
        raise BaselineSealError(
            "baseline manifest output must stay inside the project root"
        ) from exc
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite baseline seal manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = evaluate_baseline_seal(spec)
    atomic_write_json(destination, payload)
    return payload


def _artifact_specs_from_manifest(
    evidence: Mapping[str, Any], *, root: Path
) -> tuple[SealGateSpec, ...]:
    gates: list[SealGateSpec] = []
    if set(evidence) != set(_REQUIRED_GATES):
        raise BaselineSealValidationError("seal evidence gate set is incomplete or unexpected")
    for gate_name in sorted(_REQUIRED_GATES):
        record = _mapping(evidence[gate_name], f"evidence.{gate_name}")
        checker = record.get("checker")
        if checker != _REQUIRED_GATES[gate_name]:
            raise BaselineSealValidationError(f"seal checker drifted for {gate_name}")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, list):
            raise BaselineSealValidationError(f"seal artifacts are invalid for {gate_name}")
        specs: list[SealArtifactSpec] = []
        for item in artifacts:
            value = _mapping(item, f"evidence.{gate_name}.artifact")
            name = value.get("name")
            if not isinstance(name, str):
                raise BaselineSealValidationError("seal artifact name is invalid")
            path = _repo_path(value.get("path"), root=root, name=f"{gate_name}.{name}.path")
            expected_sha = value.get("expected_sha256")
            if expected_sha is not None and not isinstance(expected_sha, str):
                raise BaselineSealValidationError("seal expected artifact hash is invalid")
            specs.append(SealArtifactSpec(name, path, expected_sha))
        gates.append(SealGateSpec(gate_name, str(checker), tuple(specs)))
    return tuple(gates)


def verify_baseline_manifest(path: Path, *, project_root: Path) -> dict[str, Any]:
    """Re-evaluate every reference and reject drift or a forged completion status."""

    root = project_root.resolve()
    manifest = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineSealValidationError(f"cannot read baseline manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise BaselineSealValidationError("baseline manifest must be an object")
    if payload.get("schema_version") != 1 or payload.get("kind") != "pre_official_baseline_seal":
        raise BaselineSealValidationError("baseline manifest schema/kind is invalid")
    config_record = _mapping(payload.get("config"), "seal.config")
    config_path = _repo_path(config_record.get("path"), root=root, name="seal.config.path")
    if not config_path.is_file() or sha256_file(config_path) != config_record.get("sha256"):
        raise BaselineSealValidationError("baseline seal config is missing or hash-drifted")
    spec = load_baseline_seal_spec(config_path, project_root=root)
    stored_gates = _artifact_specs_from_manifest(
        _mapping(payload.get("evidence"), "seal.evidence"), root=root
    )
    if stored_gates != spec.gates:
        raise BaselineSealValidationError(
            "baseline manifest artifact contract differs from its config"
        )
    fresh = evaluate_baseline_seal(spec)
    for key in (
        "status",
        "completion_claim_allowed",
        "artifact_policy",
        "policy",
        "expected",
        "git",
        "evidence",
        "blockers",
        "claims",
    ):
        if payload.get(key) != fresh.get(key):
            raise BaselineSealValidationError(f"baseline manifest drifted or was forged: {key}")
    if payload["status"] == "complete" and payload.get("completion_claim_allowed") is not True:
        raise BaselineSealValidationError("complete seal does not allow its completion claim")
    return payload


__all__ = [
    "BaselineSealError",
    "BaselineSealSpec",
    "BaselineSealValidationError",
    "evaluate_baseline_seal",
    "git_snapshot",
    "load_baseline_seal_spec",
    "verify_baseline_manifest",
    "write_baseline_manifest",
]
