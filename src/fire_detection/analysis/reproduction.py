"""Strict comparison of a newly exported Round 3 OOF pair with the frozen winner.

This module never invokes training or OOF export.  It accepts already completed OOF
artifacts, validates their manifests and provenance, and publishes a new immutable
comparison directory.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd

from fire_detection.config import (
    ConfigError,
    load_yaml_mapping,
    require_bool,
    require_float,
    require_int,
    require_mapping,
    require_string,
    resolve_project_path,
)
from fire_detection.data.locked_folds import historical_code_warnings
from fire_detection.ensemble.selection import validate_final_selection_manifest
from fire_detection.inference.oof import load_oof_csv, validate_oof_artifact_manifest
from fire_detection.metrics import compute_binary_metrics
from fire_detection.models.offline import sha256_file
from fire_detection.utils import atomic_write_csv, atomic_write_json, atomic_write_text

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_EXPECTED_FOLDS = frozenset(range(5))
_IDENTITY_COLUMNS = ("filename", "label", "fold", "group_id")
_PACKAGES = ("numpy", "pandas", "torch", "torchvision", "ultralytics", "onnxruntime-gpu")


@dataclass(frozen=True)
class ReproductionSpec:
    """Decision-complete paths and tolerances for one reproduction comparison."""

    project_root: Path
    config_path: Path
    selection_manifest: Path
    reference_detector_oof: Path
    reference_detector_manifest: Path
    reference_classifier_oof: Path
    reference_classifier_manifest: Path
    folds: Path
    processed_manifest: Path
    candidate_metadata: Path
    environment_files: tuple[Path, ...]
    artifact_root: Path
    output_root: Path
    expected_count: int
    detector_score_columns: tuple[str, ...]
    classifier_score_columns: tuple[str, ...]
    detector_exact_columns: tuple[str, ...]
    classifier_exact_columns: tuple[str, ...]
    score_abs_tolerance: float
    formula_abs_tolerance: float
    metric_abs_tolerance: float
    require_clean_worktree: bool
    require_candidate_clean_git: bool
    require_candidate_commit_matches_head: bool


@dataclass(frozen=True)
class ReproductionArtifacts:
    """Paths supplied for a completed candidate OOF export."""

    detector_oof: Path
    detector_manifest: Path
    classifier_oof: Path
    classifier_manifest: Path


@dataclass(frozen=True)
class GitEvidence:
    """Current repository state recorded without mutating Git."""

    commit: str
    dirty: bool
    dirty_paths: tuple[str, ...]


@dataclass(frozen=True)
class ReproductionComparison:
    """Pure frame-comparison result ready for immutable publication."""

    differences: pd.DataFrame
    summary: dict[str, Any]


def _string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{name} must be a non-empty list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{name}[{index}] must be a non-empty string")
        normalized = item.strip()
        if normalized in result:
            raise ConfigError(f"{name} contains duplicate column {normalized!r}")
        result.append(normalized)
    return tuple(result)


def _path_sequence(value: object, *, root: Path, name: str) -> tuple[Path, ...]:
    values = _string_sequence(value, name=name)
    return tuple(resolve_project_path(root, item).resolve() for item in values)


def _probability_tolerance(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ConfigError(f"{name} must be finite and in [0, 1]")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_reproduction_spec(path: Path, *, project_root: Path) -> ReproductionSpec:
    """Load a strict comparison-only configuration."""

    root = project_root.resolve()
    config_path = (path if path.is_absolute() else root / path).resolve()
    raw = load_yaml_mapping(config_path)
    if raw.get("schema_version") != 1:
        raise ConfigError("winner reproduction schema_version must be 1")
    reference = require_mapping(raw, "reference", context="winner_reproduction")
    provenance = require_mapping(raw, "provenance", context="winner_reproduction")
    comparison = require_mapping(raw, "comparison", context="winner_reproduction")
    git = require_mapping(raw, "git", context="winner_reproduction")
    output = require_mapping(raw, "output", context="winner_reproduction")
    artifact_root = resolve_project_path(
        root,
        require_string(reference, "artifact_root", context="winner_reproduction.reference"),
    ).resolve()
    if artifact_root != root:
        raise ConfigError(
            "winner_reproduction.reference.artifact_root must resolve to project root"
        )

    score_tolerance = _probability_tolerance(
        require_float(
            comparison,
            "score_abs_tolerance",
            context="winner_reproduction.comparison",
            minimum=0.0,
        ),
        name="winner_reproduction.comparison.score_abs_tolerance",
    )
    formula_tolerance = _probability_tolerance(
        require_float(
            comparison,
            "formula_abs_tolerance",
            context="winner_reproduction.comparison",
            minimum=0.0,
        ),
        name="winner_reproduction.comparison.formula_abs_tolerance",
    )
    metric_tolerance = _probability_tolerance(
        require_float(
            comparison,
            "metric_abs_tolerance",
            context="winner_reproduction.comparison",
            minimum=0.0,
        ),
        name="winner_reproduction.comparison.metric_abs_tolerance",
    )

    def reference_path(key: str) -> Path:
        return resolve_project_path(
            root, require_string(reference, key, context="winner_reproduction.reference")
        ).resolve()

    output_root = resolve_project_path(
        root, require_string(output, "root", context="winner_reproduction.output")
    ).resolve()
    try:
        output_root.relative_to(root)
    except ValueError as exc:
        raise ConfigError("winner_reproduction.output.root must be inside project root") from exc

    return ReproductionSpec(
        project_root=root,
        config_path=config_path,
        selection_manifest=reference_path("selection_manifest"),
        reference_detector_oof=reference_path("detector_oof"),
        reference_detector_manifest=reference_path("detector_manifest"),
        reference_classifier_oof=reference_path("classifier_oof"),
        reference_classifier_manifest=reference_path("classifier_manifest"),
        folds=resolve_project_path(
            root, require_string(provenance, "folds", context="winner_reproduction.provenance")
        ).resolve(),
        processed_manifest=resolve_project_path(
            root,
            require_string(
                provenance,
                "processed_manifest",
                context="winner_reproduction.provenance",
            ),
        ).resolve(),
        candidate_metadata=resolve_project_path(
            root,
            require_string(
                provenance,
                "candidate_metadata",
                context="winner_reproduction.provenance",
            ),
        ).resolve(),
        environment_files=_path_sequence(
            provenance.get("environment_files"),
            root=root,
            name="winner_reproduction.provenance.environment_files",
        ),
        artifact_root=artifact_root,
        output_root=output_root,
        expected_count=require_int(
            comparison,
            "expected_count",
            context="winner_reproduction.comparison",
            minimum=1,
        ),
        detector_score_columns=_string_sequence(
            comparison.get("detector_score_columns"),
            name="winner_reproduction.comparison.detector_score_columns",
        ),
        classifier_score_columns=_string_sequence(
            comparison.get("classifier_score_columns"),
            name="winner_reproduction.comparison.classifier_score_columns",
        ),
        detector_exact_columns=_string_sequence(
            comparison.get("detector_exact_columns"),
            name="winner_reproduction.comparison.detector_exact_columns",
        ),
        classifier_exact_columns=_string_sequence(
            comparison.get("classifier_exact_columns"),
            name="winner_reproduction.comparison.classifier_exact_columns",
        ),
        score_abs_tolerance=score_tolerance,
        formula_abs_tolerance=formula_tolerance,
        metric_abs_tolerance=metric_tolerance,
        require_clean_worktree=require_bool(
            git, "require_clean_worktree", context="winner_reproduction.git"
        ),
        require_candidate_clean_git=require_bool(
            git, "require_candidate_clean_git", context="winner_reproduction.git"
        ),
        require_candidate_commit_matches_head=require_bool(
            git,
            "require_candidate_commit_matches_head",
            context="winner_reproduction.git",
        ),
    )


def _inside_root(path: Path, *, root: Path, name: str) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} must be inside project root: {resolved}") from exc
    return resolved


def _portable(path: Path, *, root: Path) -> str:
    return _inside_root(path, root=root, name="artifact").relative_to(root.resolve()).as_posix()


def _resolve_recorded_path(value: object, *, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}.path must be a non-empty string")
    raw = value.strip()
    portable = PurePosixPath(raw.replace("\\", "/"))
    if portable.is_absolute() or PureWindowsPath(raw).is_absolute() or ".." in portable.parts:
        raise ValueError(f"{name}.path must be repository-relative")
    return _inside_root(root / Path(*portable.parts), root=root, name=name)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _file_record(path: Path, *, root: Path, name: str) -> dict[str, object]:
    resolved = _inside_root(path, root=root, name=name)
    if not resolved.is_file():
        raise ValueError(f"{name} does not exist: {resolved}")
    return {"path": _portable(resolved, root=root), "sha256": sha256_file(resolved)}


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {name} {path}: {exc}") from exc
    return dict(_mapping(value, name=name))


def git_evidence(project_root: Path) -> GitEvidence:
    """Capture HEAD and all dirty paths without changing repository state."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot capture Git state: {exc}") from exc
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("git rev-parse returned an invalid commit")
    return GitEvidence(commit=commit, dirty=bool(status), dirty_paths=tuple(status))


def _manifest_git(manifest: Mapping[str, Any], *, name: str) -> dict[str, object]:
    raw = _mapping(manifest.get("git"), name=f"{name}.git")
    if set(raw) != {"commit", "dirty"}:
        raise ValueError(f"{name}.git must contain exactly commit and dirty")
    commit = raw.get("commit")
    dirty = raw.get("dirty")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError(f"{name}.git.commit is invalid")
    if type(dirty) is not bool:
        raise ValueError(f"{name}.git.dirty must be boolean")
    return {"commit": commit, "dirty": dirty}


def validate_git_contract(
    spec: ReproductionSpec,
    *,
    current: GitEvidence,
    detector_manifest: Mapping[str, Any],
    classifier_manifest: Mapping[str, Any],
) -> dict[str, object]:
    """Enforce clean, same-commit candidate manifests and optional HEAD binding."""

    detector_git = _manifest_git(detector_manifest, name="candidate detector manifest")
    classifier_git = _manifest_git(classifier_manifest, name="candidate classifier manifest")
    if detector_git["commit"] != classifier_git["commit"]:
        raise ValueError("candidate detector/classifier manifests use different commits")
    if spec.require_clean_worktree and current.dirty:
        raise ValueError(f"current worktree is dirty: {list(current.dirty_paths)[:10]}")
    if spec.require_candidate_clean_git and (
        detector_git["dirty"] is not False or classifier_git["dirty"] is not False
    ):
        raise ValueError("candidate detector/classifier OOF manifests must be clean")
    if spec.require_candidate_commit_matches_head and detector_git["commit"] != current.commit:
        raise ValueError(
            "candidate OOF commit does not match current HEAD: "
            f"{detector_git['commit']} != {current.commit}"
        )
    return {
        "current": asdict(current),
        "candidate_detector": detector_git,
        "candidate_classifier": classifier_git,
    }


def _verify_selection_binding(
    selection: Mapping[str, Any],
    *,
    spec: ReproductionSpec,
) -> None:
    components = _mapping(selection.get("components"), name="selection.components")
    expected = {
        "detector": (spec.reference_detector_oof, spec.reference_detector_manifest),
        "classifier": (spec.reference_classifier_oof, spec.reference_classifier_manifest),
    }
    for kind, (oof_path, manifest_path) in expected.items():
        component = _mapping(components.get(kind), name=f"selection.components.{kind}")
        for field, required_path in (
            ("oof", oof_path),
            ("oof_manifest", manifest_path),
        ):
            record = _mapping(component.get(field), name=f"selection {kind}.{field}")
            recorded_path = _resolve_recorded_path(
                record.get("path"), root=spec.project_root, name=f"selection {kind}.{field}"
            )
            if recorded_path != required_path.resolve():
                raise ValueError(f"selection {kind}.{field} path differs from reproduction config")
            expected_hash = _sha256(record.get("sha256"), name=f"selection {kind}.{field}.sha256")
            actual_hash = sha256_file(required_path)
            if actual_hash != expected_hash:
                raise ValueError(f"selection {kind}.{field} hash drifted")


def _validate_frame_identity(frame: pd.DataFrame, *, name: str, expected_count: int) -> None:
    missing = [column for column in _IDENTITY_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{name} omits identity columns: {missing}")
    if len(frame) != expected_count:
        raise ValueError(f"{name} has {len(frame)} rows, expected {expected_count}")
    if frame["filename"].duplicated().any():
        raise ValueError(f"{name} contains duplicate filenames")
    if set(int(value) for value in frame["fold"].unique()) != _EXPECTED_FOLDS:
        raise ValueError(f"{name} must contain exactly Fold 0..4")


def _aligned_pair(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    name: str,
    expected_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_frame_identity(reference, name=f"reference {name}", expected_count=expected_count)
    _validate_frame_identity(candidate, name=f"candidate {name}", expected_count=expected_count)
    reference_ordered = reference.sort_values("filename", kind="stable").reset_index(drop=True)
    candidate_ordered = candidate.sort_values("filename", kind="stable").reset_index(drop=True)
    reference_names = reference_ordered["filename"].astype(str).tolist()
    candidate_names = candidate_ordered["filename"].astype(str).tolist()
    if reference_names != candidate_names:
        missing = sorted(set(reference_names) - set(candidate_names))
        extra = sorted(set(candidate_names) - set(reference_names))
        raise ValueError(
            f"candidate {name} filename set drifted; missing={missing[:5]}, extra={extra[:5]}"
        )
    for column in _IDENTITY_COLUMNS[1:]:
        if reference_ordered[column].tolist() != candidate_ordered[column].tolist():
            raise ValueError(f"candidate {name} {column} drifted")
    return reference_ordered, candidate_ordered


def _finite_scores(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{name} omits configured score columns: {missing}")
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{column} must be numeric") from exc
        if (~np.isfinite(values)).any() or ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{name}.{column} must be finite and in [0, 1]")


def _metric_record(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    metrics = compute_binary_metrics(
        labels.astype(int).tolist(), predictions.astype(float).tolist(), 0.5
    )
    return asdict(metrics)


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> list[str]:
    values = {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}
    return [
        values[(int(label), int(pred))] for label, pred in zip(labels, predictions, strict=True)
    ]


def _score_summary(diff: np.ndarray, *, tolerance: float) -> dict[str, object]:
    return {
        "max_abs_diff": float(diff.max(initial=0.0)),
        "nonzero_rows": int(np.count_nonzero(diff > 0.0)),
        "over_tolerance_rows": int(np.count_nonzero(diff > tolerance)),
        "abs_tolerance": tolerance,
    }


def _metric_differences(
    reference: Mapping[str, float | int],
    candidate: Mapping[str, float | int],
    *,
    tolerance: float,
) -> tuple[dict[str, float | int], bool]:
    differences: dict[str, float | int] = {}
    passed = True
    for key in ("precision", "recall", "f1"):
        difference = abs(float(reference[key]) - float(candidate[key]))
        differences[key] = difference
        passed = passed and difference <= tolerance
    for key in ("tp", "fp", "fn", "tn"):
        difference = int(candidate[key]) - int(reference[key])
        differences[key] = difference
        passed = passed and difference == 0
    return differences, passed


def compare_reproduction_frames(
    *,
    reference_detector: pd.DataFrame,
    candidate_detector: pd.DataFrame,
    reference_classifier: pd.DataFrame,
    candidate_classifier: pd.DataFrame,
    selected: pd.DataFrame,
    expected_metrics: Mapping[str, float | int],
    detector_score_columns: Sequence[str],
    classifier_score_columns: Sequence[str],
    detector_exact_columns: Sequence[str],
    classifier_exact_columns: Sequence[str],
    detector_selected_column: str,
    classifier_selected_column: str,
    classifier_weight: float,
    expected_count: int,
    score_abs_tolerance: float,
    formula_abs_tolerance: float,
    metric_abs_tolerance: float,
) -> ReproductionComparison:
    """Compare normalized OOF frames without reading, writing, or fitting anything."""

    if not math.isfinite(classifier_weight) or not 0.0 <= classifier_weight <= 1.0:
        raise ValueError("classifier_weight must be finite and in [0, 1]")
    reference_detector, candidate_detector = _aligned_pair(
        reference_detector, candidate_detector, name="detector OOF", expected_count=expected_count
    )
    reference_classifier, candidate_classifier = _aligned_pair(
        reference_classifier,
        candidate_classifier,
        name="classifier OOF",
        expected_count=expected_count,
    )
    if (
        reference_detector[list(_IDENTITY_COLUMNS)].values.tolist()
        != reference_classifier[list(_IDENTITY_COLUMNS)].values.tolist()
    ):
        raise ValueError("reference detector/classifier OOF identities differ")
    if (
        candidate_detector[list(_IDENTITY_COLUMNS)].values.tolist()
        != candidate_classifier[list(_IDENTITY_COLUMNS)].values.tolist()
    ):
        raise ValueError("candidate detector/classifier OOF identities differ")

    _finite_scores(reference_detector, detector_score_columns, name="reference detector OOF")
    _finite_scores(candidate_detector, detector_score_columns, name="candidate detector OOF")
    _finite_scores(reference_classifier, classifier_score_columns, name="reference classifier OOF")
    _finite_scores(candidate_classifier, classifier_score_columns, name="candidate classifier OOF")
    if detector_selected_column not in detector_score_columns:
        raise ValueError("selected detector score column is not configured for comparison")
    if classifier_selected_column not in classifier_score_columns:
        raise ValueError("selected classifier score column is not configured for comparison")

    selected_required = {
        "filename",
        "label",
        "fold",
        "score",
        "threshold",
        "y_pred",
        "confusion",
    }
    missing_selected = sorted(selected_required.difference(selected.columns))
    if missing_selected:
        raise ValueError(f"selected cross-fitted CSV omits columns: {missing_selected}")
    if len(selected) != expected_count:
        raise ValueError(
            f"selected cross-fitted CSV has {len(selected)} rows, expected {expected_count}"
        )
    if selected["filename"].duplicated().any():
        raise ValueError("selected cross-fitted CSV contains duplicate filenames")
    if set(int(value) for value in selected["fold"].unique()) != _EXPECTED_FOLDS:
        raise ValueError("selected cross-fitted CSV must contain exactly Fold 0..4")
    selected = selected.sort_values("filename", kind="stable").reset_index(drop=True)
    if (
        selected[list(_IDENTITY_COLUMNS[:3])].values.tolist()
        != reference_detector[list(_IDENTITY_COLUMNS[:3])].values.tolist()
    ):
        raise ValueError("selected cross-fitted identities differ from component OOF")
    for column in ("score", "threshold"):
        values = pd.to_numeric(selected[column], errors="raise").to_numpy(dtype=float)
        if (~np.isfinite(values)).any() or ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"selected {column} must be finite and in [0, 1]")

    differences = reference_detector[list(_IDENTITY_COLUMNS)].copy()
    summaries: dict[str, dict[str, object]] = {}
    passed = True
    for prefix, reference, candidate, columns in (
        ("detector", reference_detector, candidate_detector, detector_score_columns),
        ("classifier", reference_classifier, candidate_classifier, classifier_score_columns),
    ):
        for column in columns:
            reference_values = reference[column].to_numpy(dtype=float)
            candidate_values = candidate[column].to_numpy(dtype=float)
            diff = np.abs(candidate_values - reference_values)
            differences[f"reference_{prefix}_{column}"] = reference_values
            differences[f"candidate_{prefix}_{column}"] = candidate_values
            differences[f"abs_diff_{prefix}_{column}"] = diff
            column_summary = _score_summary(diff, tolerance=score_abs_tolerance)
            summaries[f"{prefix}.{column}"] = column_summary
            passed = passed and column_summary["over_tolerance_rows"] == 0

    exact_mismatches: dict[str, int] = {}
    for prefix, reference, candidate, columns in (
        ("detector", reference_detector, candidate_detector, detector_exact_columns),
        ("classifier", reference_classifier, candidate_classifier, classifier_exact_columns),
    ):
        missing = [
            column for column in columns if column not in reference or column not in candidate
        ]
        if missing:
            raise ValueError(f"{prefix} exact comparison columns are missing: {missing}")
        for column in columns:
            mismatch = reference[column].tolist() != candidate[column].tolist()
            count = int((reference[column].astype(str) != candidate[column].astype(str)).sum())
            if mismatch and count == 0:
                count = expected_count
            exact_mismatches[f"{prefix}.{column}"] = count
            passed = passed and count == 0

    detector_weight = 1.0 - classifier_weight
    reference_fused = detector_weight * reference_detector[detector_selected_column].to_numpy(
        dtype=float
    ) + classifier_weight * reference_classifier[classifier_selected_column].to_numpy(dtype=float)
    candidate_fused = detector_weight * candidate_detector[detector_selected_column].to_numpy(
        dtype=float
    ) + classifier_weight * candidate_classifier[classifier_selected_column].to_numpy(dtype=float)
    selected_score = selected["score"].to_numpy(dtype=float)
    fused_diff = np.abs(candidate_fused - reference_fused)
    formula_diff = np.abs(reference_fused - selected_score)
    summaries["fused_score"] = _score_summary(fused_diff, tolerance=score_abs_tolerance)
    summaries["reference_formula_to_selection"] = _score_summary(
        formula_diff, tolerance=formula_abs_tolerance
    )
    passed = passed and summaries["fused_score"]["over_tolerance_rows"] == 0
    passed = passed and (summaries["reference_formula_to_selection"]["over_tolerance_rows"] == 0)

    labels = selected["label"].to_numpy(dtype=int)
    thresholds = selected["threshold"].to_numpy(dtype=float)
    selected_predictions = pd.to_numeric(selected["y_pred"], errors="raise").to_numpy(dtype=int)
    if not np.isin(selected_predictions, [0, 1]).all():
        raise ValueError("selected y_pred must contain integer 0/1")
    formula_reference_predictions = (reference_fused >= thresholds).astype(int)
    if not np.array_equal(selected_predictions, (selected_score >= thresholds).astype(int)):
        raise ValueError("selected y_pred does not equal score >= threshold")
    if selected["confusion"].astype(str).tolist() != _confusion(labels, selected_predictions):
        raise ValueError("selected confusion labels are invalid")
    candidate_predictions = (candidate_fused >= thresholds).astype(int)
    decision_changes = candidate_predictions != selected_predictions
    formula_decision_changes = formula_reference_predictions != selected_predictions
    passed = (
        passed and not bool(decision_changes.any()) and not bool(formula_decision_changes.any())
    )

    reference_metrics = _metric_record(labels, selected_predictions)
    formula_reference_metrics = _metric_record(labels, formula_reference_predictions)
    candidate_metrics = _metric_record(labels, candidate_predictions)
    reported_metric_differences, reported_pass = _metric_differences(
        expected_metrics, reference_metrics, tolerance=metric_abs_tolerance
    )
    candidate_metric_differences, candidate_pass = _metric_differences(
        reference_metrics, candidate_metrics, tolerance=metric_abs_tolerance
    )
    passed = passed and reported_pass and candidate_pass

    differences["reference_fused_score"] = reference_fused
    differences["selection_score"] = selected_score
    differences["abs_diff_reference_formula_to_selection"] = formula_diff
    differences["candidate_fused_score"] = candidate_fused
    differences["abs_diff_fused_score"] = fused_diff
    differences["threshold"] = thresholds
    differences["reference_y_pred"] = selected_predictions
    differences["candidate_y_pred"] = candidate_predictions
    differences["decision_changed"] = decision_changes
    differences["formula_decision_changed"] = formula_decision_changes
    differences["reference_confusion"] = _confusion(labels, selected_predictions)
    differences["candidate_confusion"] = _confusion(labels, candidate_predictions)
    abs_columns = [column for column in differences if column.startswith("abs_diff_")]
    differences["any_nonzero_difference"] = (differences[abs_columns] > 0.0).any(axis=1)
    tolerance_by_column = {
        column: formula_abs_tolerance
        if column == "abs_diff_reference_formula_to_selection"
        else score_abs_tolerance
        for column in abs_columns
    }
    differences["any_tolerance_exceeded"] = np.logical_or.reduce(
        [
            differences[column].to_numpy(dtype=float) > tolerance
            for column, tolerance in tolerance_by_column.items()
        ]
    )

    per_fold = []
    for fold in sorted(_EXPECTED_FOLDS):
        mask = selected["fold"].to_numpy(dtype=int) == fold
        per_fold.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "reference": _metric_record(labels[mask], selected_predictions[mask]),
                "candidate": _metric_record(labels[mask], candidate_predictions[mask]),
            }
        )
    summary: dict[str, Any] = {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "rows": expected_count,
        "fold_counts": {
            str(fold): int((selected["fold"].to_numpy(dtype=int) == fold).sum())
            for fold in sorted(_EXPECTED_FOLDS)
        },
        "comparison_contract": {
            "identity": list(_IDENTITY_COLUMNS),
            "detector_score_columns": list(detector_score_columns),
            "classifier_score_columns": list(classifier_score_columns),
            "detector_exact_columns": list(detector_exact_columns),
            "classifier_exact_columns": list(classifier_exact_columns),
            "detector_selected_column": detector_selected_column,
            "classifier_selected_column": classifier_selected_column,
            "detector_weight": detector_weight,
            "classifier_weight": classifier_weight,
            "score_abs_tolerance": score_abs_tolerance,
            "formula_abs_tolerance": formula_abs_tolerance,
            "metric_abs_tolerance": metric_abs_tolerance,
        },
        "score_differences": summaries,
        "exact_mismatch_counts": exact_mismatches,
        "decision_changes": int(decision_changes.sum()),
        "reference_formula_decision_changes": int(formula_decision_changes.sum()),
        "reference_metrics": reference_metrics,
        "reference_formula_metrics": formula_reference_metrics,
        "candidate_metrics": candidate_metrics,
        "reported_metric_differences": reported_metric_differences,
        "candidate_metric_differences": candidate_metric_differences,
        "per_fold": per_fold,
        "nonzero_difference_rows": int(differences["any_nonzero_difference"].sum()),
        "tolerance_exceeded_rows": int(differences["any_tolerance_exceeded"].sum()),
    }
    return ReproductionComparison(differences=differences, summary=summary)


def collect_environment_evidence(
    *, project_root: Path, environment_files: Sequence[Path]
) -> dict[str, object]:
    """Collect package, CUDA, GPU, and lock-file evidence without running inference."""

    packages: dict[str, str | None] = {}
    for package in _PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    torch_runtime: dict[str, object] = {"imported": False}
    try:
        import torch

        available = bool(torch.cuda.is_available())
        torch_runtime = {
            "imported": True,
            "version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cuda_available": available,
            "gpu_name": torch.cuda.get_device_name(0) if available else None,
        }
    except (ImportError, OSError, RuntimeError) as exc:
        torch_runtime = {"imported": False, "error": str(exc)}

    nvidia_smi: dict[str, object]
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        nvidia_smi = {"available": True, "rows": process.stdout.strip().splitlines()}
    except (OSError, subprocess.CalledProcessError) as exc:
        nvidia_smi = {"available": False, "error": str(exc)}
    return {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_runtime": torch_runtime,
        "nvidia_smi": nvidia_smi,
        "environment_files": [
            _file_record(path, root=project_root, name="environment lock")
            for path in environment_files
        ],
    }


def _historical_warnings(spec: ReproductionSpec) -> list[dict[str, object]]:
    metadata = _load_json(spec.candidate_metadata, name="Fold candidate metadata")
    return [
        warning.as_dict()
        for warning in historical_code_warnings(metadata, project_root=spec.project_root)
    ]


def _markdown(report: Mapping[str, Any]) -> str:
    summary = _mapping(report.get("comparison"), name="report.comparison")
    metrics = _mapping(summary.get("candidate_metrics"), name="candidate metrics")
    git = _mapping(report.get("git"), name="report.git")
    current = _mapping(git.get("current"), name="report.git.current")
    warnings = report.get("provenance_warnings")
    warning_rows = warnings if isinstance(warnings, list) else []
    lines = [
        "# Clean OOF reproduction report",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Rows: `{summary.get('rows')}`",
        f"- Current commit: `{current.get('commit')}`",
        f"- Worktree dirty: `{current.get('dirty')}`",
        "- Operation: comparison only; no training or OOF export was invoked.",
        "",
        "## Candidate image-level metrics",
        "",
        "| Precision | Recall | F1 | TP | FP | FN | TN |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {float(metrics.get('precision', 0.0)):.12f} "
            f"| {float(metrics.get('recall', 0.0)):.12f} "
            f"| {float(metrics.get('f1', 0.0)):.12f} "
            f"| {metrics.get('tp')} | {metrics.get('fp')} | {metrics.get('fn')} "
            f"| {metrics.get('tn')} |"
        ),
        "",
        "## Comparison gates",
        "",
        f"- Decision changes: `{summary.get('decision_changes')}`",
        f"- Rows with any non-zero score difference: `{summary.get('nonzero_difference_rows')}`",
        f"- Rows exceeding tolerance: `{summary.get('tolerance_exceeded_rows')}`",
        f"- Historical provenance warnings: `{len(warning_rows)}`",
    ]
    for warning in warning_rows:
        if isinstance(warning, Mapping):
            lines.append(f"  - `{warning.get('code')}`: `{warning.get('path')}`")
    return "\n".join(lines) + "\n"


def _publish(
    *,
    output_dir: Path,
    report: dict[str, Any],
    differences: pd.DataFrame,
    environment: dict[str, object],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite reproduction output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        diff_path = staging / "probability_diff.csv"
        environment_path = staging / "environment_report.json"
        report_path = staging / "clean_reproduction_report.json"
        markdown_path = staging / "clean_reproduction_report.md"
        atomic_write_csv(diff_path, differences)
        atomic_write_json(environment_path, environment)
        report["outputs"] = {
            "probability_diff": {
                "path": diff_path.name,
                "sha256": sha256_file(diff_path),
                "rows": len(differences),
            },
            "environment_report": {
                "path": environment_path.name,
                "sha256": sha256_file(environment_path),
            },
        }
        atomic_write_json(report_path, report)
        atomic_write_text(markdown_path, _markdown(report))
        atomic_write_json(
            staging / "report_manifest.json",
            {
                "schema_version": 1,
                "kind": "winner_reproduction_report",
                "status": report["comparison"]["status"],
                "files": {
                    path.name: {"sha256": sha256_file(path)}
                    for path in (diff_path, environment_path, report_path, markdown_path)
                },
            },
        )
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_winner_reproduction(
    spec: ReproductionSpec,
    *,
    candidate: ReproductionArtifacts,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate, compare, and atomically publish one completed OOF reproduction."""

    output_dir = _inside_root(output_dir, root=spec.project_root, name="output_dir")
    try:
        output_dir.relative_to(spec.output_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"output_dir must be below configured output root {spec.output_root}"
        ) from exc
    if output_dir == spec.output_root.resolve():
        raise ValueError("output_dir must be a new run directory below the configured output root")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite reproduction output: {output_dir}")
    candidate = ReproductionArtifacts(
        detector_oof=_inside_root(
            candidate.detector_oof,
            root=spec.project_root,
            name="candidate detector OOF",
        ),
        detector_manifest=_inside_root(
            candidate.detector_manifest,
            root=spec.project_root,
            name="candidate detector manifest",
        ),
        classifier_oof=_inside_root(
            candidate.classifier_oof,
            root=spec.project_root,
            name="candidate classifier OOF",
        ),
        classifier_manifest=_inside_root(
            candidate.classifier_manifest,
            root=spec.project_root,
            name="candidate classifier manifest",
        ),
    )

    current_git = git_evidence(spec.project_root)
    if spec.require_clean_worktree and current_git.dirty:
        raise ValueError(f"current worktree is dirty: {list(current_git.dirty_paths)[:10]}")
    selection, selection_sha = validate_final_selection_manifest(
        spec.selection_manifest, project_root=spec.project_root
    )
    _verify_selection_binding(selection, spec=spec)
    folds_sha = sha256_file(spec.folds)
    processed_sha = sha256_file(spec.processed_manifest)

    reference_detector_manifest, reference_detector_manifest_sha = validate_oof_artifact_manifest(
        spec.reference_detector_manifest,
        artifact_root=spec.artifact_root,
        kind="detector",
        oof_path=spec.reference_detector_oof,
        reference_folds_path=spec.folds,
        processed_manifest_path=spec.processed_manifest,
        expected_rows=spec.expected_count,
    )
    reference_classifier_manifest, reference_classifier_manifest_sha = (
        validate_oof_artifact_manifest(
            spec.reference_classifier_manifest,
            artifact_root=spec.artifact_root,
            kind="classifier",
            oof_path=spec.reference_classifier_oof,
            reference_folds_path=spec.folds,
            processed_manifest_path=spec.processed_manifest,
            expected_rows=spec.expected_count,
        )
    )
    candidate_detector_manifest, candidate_detector_manifest_sha = validate_oof_artifact_manifest(
        candidate.detector_manifest,
        artifact_root=spec.artifact_root,
        kind="detector",
        oof_path=candidate.detector_oof,
        reference_folds_path=spec.folds,
        processed_manifest_path=spec.processed_manifest,
        expected_rows=spec.expected_count,
    )
    candidate_classifier_manifest, candidate_classifier_manifest_sha = (
        validate_oof_artifact_manifest(
            candidate.classifier_manifest,
            artifact_root=spec.artifact_root,
            kind="classifier",
            oof_path=candidate.classifier_oof,
            reference_folds_path=spec.folds,
            processed_manifest_path=spec.processed_manifest,
            expected_rows=spec.expected_count,
        )
    )
    git_record = validate_git_contract(
        spec,
        current=current_git,
        detector_manifest=candidate_detector_manifest,
        classifier_manifest=candidate_classifier_manifest,
    )

    def load(path: Path, kind: str) -> pd.DataFrame:
        return load_oof_csv(
            path,
            kind=kind,  # type: ignore[arg-type]
            expected_rows=spec.expected_count,
            reference_folds_path=spec.folds,
            expected_folds_sha256=folds_sha,
            expected_processed_manifest_sha256=processed_sha,
            artifact_root=spec.artifact_root,
        )

    reference_detector = load(spec.reference_detector_oof, "detector")
    reference_classifier = load(spec.reference_classifier_oof, "classifier")
    candidate_detector = load(candidate.detector_oof, "detector")
    candidate_classifier = load(candidate.classifier_oof, "classifier")

    artifacts = _mapping(selection.get("artifacts"), name="selection.artifacts")
    outputs = _mapping(artifacts.get("outputs"), name="selection.artifacts.outputs")
    selected_record = _mapping(
        outputs.get("selected_cross_fitted"), name="selection selected_cross_fitted"
    )
    selected_path = _resolve_recorded_path(
        selected_record.get("path"), root=spec.project_root, name="selected_cross_fitted"
    )
    selected = pd.read_csv(selected_path)
    fusion = _mapping(selection.get("fusion"), name="selection.fusion")
    selection_components = _mapping(selection.get("components"), name="selection.components")
    model_inventory: dict[str, list[dict[str, Any]]] = {}
    for task in ("detector", "classifier"):
        component = _mapping(selection_components.get(task), name=f"selection.components.{task}")
        raw_models = component.get("models")
        if not isinstance(raw_models, list) or len(raw_models) != 5:
            raise ValueError(f"selection {task} must bind exactly five models")
        models: list[dict[str, Any]] = []
        for index, raw_model in enumerate(raw_models):
            model = _mapping(raw_model, name=f"selection {task} model {index}")
            models.append(dict(model))
        model_inventory[task] = models
    expected_metrics = _mapping(
        _mapping(selection.get("metrics"), name="selection.metrics").get("cross_fitted"),
        name="selection.metrics.cross_fitted",
    )
    comparison = compare_reproduction_frames(
        reference_detector=reference_detector,
        candidate_detector=candidate_detector,
        reference_classifier=reference_classifier,
        candidate_classifier=candidate_classifier,
        selected=selected,
        expected_metrics=expected_metrics,
        detector_score_columns=spec.detector_score_columns,
        classifier_score_columns=spec.classifier_score_columns,
        detector_exact_columns=spec.detector_exact_columns,
        classifier_exact_columns=spec.classifier_exact_columns,
        detector_selected_column=str(fusion.get("detector_score_column")),
        classifier_selected_column=str(fusion.get("classifier_score_column")),
        classifier_weight=_finite_float(
            fusion.get("classifier_weight"), name="selection.fusion.classifier_weight"
        ),
        expected_count=spec.expected_count,
        score_abs_tolerance=spec.score_abs_tolerance,
        formula_abs_tolerance=spec.formula_abs_tolerance,
        metric_abs_tolerance=spec.metric_abs_tolerance,
    )
    environment = collect_environment_evidence(
        project_root=spec.project_root, environment_files=spec.environment_files
    )
    warnings = _historical_warnings(spec)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "clean_winner_reproduction",
        "status": comparison.summary["status"],
        "comparison_only": True,
        "training_invoked": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": _file_record(spec.config_path, root=spec.project_root, name="config"),
        "git": git_record,
        "reference_git": {
            "detector": _manifest_git(
                reference_detector_manifest, name="reference detector manifest"
            ),
            "classifier": _manifest_git(
                reference_classifier_manifest, name="reference classifier manifest"
            ),
        },
        "models": model_inventory,
        "inputs": {
            "selection_manifest": {
                **_file_record(
                    spec.selection_manifest,
                    root=spec.project_root,
                    name="selection manifest",
                ),
                "validated_sha256": selection_sha,
            },
            "folds": _file_record(spec.folds, root=spec.project_root, name="folds"),
            "processed_manifest": _file_record(
                spec.processed_manifest,
                root=spec.project_root,
                name="processed manifest",
            ),
            "fold_candidate_metadata": _file_record(
                spec.candidate_metadata,
                root=spec.project_root,
                name="Fold candidate metadata",
            ),
            "reference_detector_oof": _file_record(
                spec.reference_detector_oof,
                root=spec.project_root,
                name="reference detector OOF",
            ),
            "reference_detector_manifest": {
                **_file_record(
                    spec.reference_detector_manifest,
                    root=spec.project_root,
                    name="reference detector manifest",
                ),
                "validated_sha256": reference_detector_manifest_sha,
            },
            "reference_classifier_oof": _file_record(
                spec.reference_classifier_oof,
                root=spec.project_root,
                name="reference classifier OOF",
            ),
            "reference_classifier_manifest": {
                **_file_record(
                    spec.reference_classifier_manifest,
                    root=spec.project_root,
                    name="reference classifier manifest",
                ),
                "validated_sha256": reference_classifier_manifest_sha,
            },
            "candidate_detector_oof": _file_record(
                candidate.detector_oof,
                root=spec.project_root,
                name="candidate detector OOF",
            ),
            "candidate_detector_manifest": {
                **_file_record(
                    candidate.detector_manifest,
                    root=spec.project_root,
                    name="candidate detector manifest",
                ),
                "validated_sha256": candidate_detector_manifest_sha,
            },
            "candidate_classifier_oof": _file_record(
                candidate.classifier_oof,
                root=spec.project_root,
                name="candidate classifier OOF",
            ),
            "candidate_classifier_manifest": {
                **_file_record(
                    candidate.classifier_manifest,
                    root=spec.project_root,
                    name="candidate classifier manifest",
                ),
                "validated_sha256": candidate_classifier_manifest_sha,
            },
        },
        "provenance_warnings": warnings,
        "comparison": comparison.summary,
    }
    _publish(
        output_dir=output_dir,
        report=report,
        differences=comparison.differences,
        environment=environment,
    )
    return report
