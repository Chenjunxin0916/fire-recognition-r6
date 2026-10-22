"""Pre-official, leakage-aware diagnostics for the frozen Round-3 OOF winner."""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

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
    require_value,
    resolve_project_path,
)
from fire_detection.ensemble.evaluation import (
    FoldEvaluation,
    FusionEvaluation,
    ModelEvaluation,
    evaluate_single_model,
    evaluate_weighted_fusion,
)
from fire_detection.ensemble.selection import validate_final_selection_manifest
from fire_detection.inference.oof import (
    combine_oof_frames,
    load_oof_csv,
    validate_oof_artifact_manifest,
)
from fire_detection.metrics import BinaryMetrics, build_threshold_grid, compute_binary_metrics
from fire_detection.metrics.thresholds import search_thresholds
from fire_detection.utils.io import atomic_write_csv, atomic_write_json

CandidateMethod = Literal["single_detector", "single_classifier", "weighted_average"]

DETECTOR_FULL_ORIGINAL = "detector_full_original"
CLASSIFIER_ORIGINAL = "classifier_original"
FUSION_FULL_ORIGINAL = "fusion_full_original"
FUSION_ORIGINAL_SLICING = "fusion_original_with_slicing"
FUSION_FULL_TTA = "fusion_full_tta"
FUSION_SLICING_TTA_LOFO = "fusion_slicing_tta_lofo"
FROZEN_WINNER = "frozen_winner"

CANDIDATE_ORDER = (
    DETECTOR_FULL_ORIGINAL,
    CLASSIFIER_ORIGINAL,
    FUSION_FULL_ORIGINAL,
    FUSION_ORIGINAL_SLICING,
    FUSION_FULL_TTA,
    FUSION_SLICING_TTA_LOFO,
    FROZEN_WINNER,
)

PRIMARY_REFERENCES: dict[str, str] = {
    CLASSIFIER_ORIGINAL: DETECTOR_FULL_ORIGINAL,
    FUSION_FULL_ORIGINAL: DETECTOR_FULL_ORIGINAL,
    FUSION_ORIGINAL_SLICING: FUSION_FULL_ORIGINAL,
    FUSION_FULL_TTA: FUSION_FULL_ORIGINAL,
    FUSION_SLICING_TTA_LOFO: FUSION_ORIGINAL_SLICING,
    FROZEN_WINNER: FUSION_SLICING_TTA_LOFO,
}

TRANSITION_PAIRS = (
    (DETECTOR_FULL_ORIGINAL, CLASSIFIER_ORIGINAL, "branch_alternative"),
    (DETECTOR_FULL_ORIGINAL, FUSION_FULL_ORIGINAL, "add_classifier"),
    (FUSION_FULL_ORIGINAL, FUSION_ORIGINAL_SLICING, "add_slicing"),
    (FUSION_FULL_ORIGINAL, FUSION_FULL_TTA, "add_tta_without_slicing"),
    (FUSION_ORIGINAL_SLICING, FUSION_SLICING_TTA_LOFO, "add_tta_to_slicing"),
    (FUSION_FULL_TTA, FUSION_SLICING_TTA_LOFO, "add_slicing_to_tta"),
    (FUSION_SLICING_TTA_LOFO, FROZEN_WINNER, "freeze_operating_point"),
)


@dataclass(frozen=True)
class SearchSpec:
    """LOFO weight and threshold grids used by ablation candidates 1--6."""

    weight_start: Decimal
    weight_end: Decimal
    weight_step: Decimal
    threshold_start: Decimal
    threshold_end: Decimal
    threshold_step: Decimal


@dataclass(frozen=True)
class ThresholdDiagnosticSpec:
    """Frozen-winner threshold sensitivity contract."""

    start: Decimal
    end: Decimal
    step: Decimal
    checkpoints: tuple[float, ...]
    plateau_f1_tolerance: float


@dataclass(frozen=True)
class BootstrapSpec:
    """Cluster-bootstrap contract."""

    iterations: int
    seed: int
    confidence_level: float
    diagnostic_refit_threshold: bool


@dataclass(frozen=True)
class FrozenWinnerSpec:
    """Parameters that must agree with the validated final-selection manifest."""

    classifier_weight: float
    stable_threshold: float
    recall_threshold: float
    precision_threshold: float
    expected_metrics: BinaryMetrics


@dataclass(frozen=True)
class ScoreColumns:
    """Bound OOF score columns for the seven fixed ablation candidates."""

    detector_full_original: str
    detector_original_full_slice: str
    detector_hflip_full: str
    detector_hflip_full_slice: str
    detector_tta_full_slice: str
    classifier_original: str
    classifier_tta: str


@dataclass(frozen=True)
class RuntimeBenchmarkSpec:
    """Declared benchmark protocol; measurements remain nullable until actually run."""

    status: str
    sample_count: int
    warmup_count: int
    repeats: int


@dataclass(frozen=True)
class OOFDiagnosticsSpec:
    """Strict configuration and provenance for pre-official OOF diagnostics."""

    config_path: Path
    config_sha256: str
    experiment_name: str
    detector_oof: Path
    detector_manifest: Path
    classifier_oof: Path
    classifier_manifest: Path
    selection_manifest: Path
    folds: Path
    processed_manifest: Path
    artifact_root: Path
    expected_count: int
    expected_fold_count: int
    official_test_count: int
    scores: ScoreColumns
    search: SearchSpec
    threshold: ThresholdDiagnosticSpec
    class_priors: tuple[float, ...]
    bootstrap: BootstrapSpec
    winner: FrozenWinnerSpec
    runtime_benchmark: RuntimeBenchmarkSpec
    output_dir: Path


@dataclass(frozen=True)
class CandidateResult:
    """One standardized fixed-matrix ablation result."""

    name: str
    order: int
    method: CandidateMethod
    evaluation_contract: str
    detector_feature: str | None
    classifier_feature: str | None
    pooled_threshold: float
    pooled_classifier_weight: float | None
    metrics: BinaryMetrics
    per_fold: tuple[FoldEvaluation, ...]
    predictions: pd.DataFrame


@dataclass(frozen=True)
class ValidatedInputs:
    """Manifest-validated and one-to-one aligned OOF inputs."""

    frame: pd.DataFrame
    selection: dict[str, Any]
    input_records: dict[str, dict[str, object]]


def sha256_file(path: Path) -> str:
    """Hash one file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _decimal(mapping: Mapping[str, Any], key: str, *, context: str) -> Decimal:
    raw = require_value(mapping, key, context=context)
    if isinstance(raw, bool):
        raise ConfigError(f"{context}.{key} must be a finite decimal")
    try:
        result = Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise ConfigError(f"{context}.{key} must be a finite decimal") from exc
    if not result.is_finite():
        raise ConfigError(f"{context}.{key} must be finite")
    return result


def _float_sequence(mapping: Mapping[str, Any], key: str, *, context: str) -> tuple[float, ...]:
    raw = require_value(mapping, key, context=context)
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{context}.{key} must be a non-empty list")
    values: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{context}.{key}[{index}] must be numeric")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ConfigError(f"{context}.{key}[{index}] must be finite")
        values.append(normalized)
    if len(values) != len(set(values)):
        raise ConfigError(f"{context}.{key} must not contain duplicates")
    return tuple(values)


def _expected_metrics(mapping: Mapping[str, Any]) -> BinaryMetrics:
    context = "oof_diagnostics.frozen_winner.expected_metrics"
    precision = require_float(mapping, "precision", context=context, minimum=0.0)
    recall = require_float(mapping, "recall", context=context, minimum=0.0)
    f1 = require_float(mapping, "f1", context=context, minimum=0.0)
    for name, value in (("precision", precision), ("recall", recall), ("f1", f1)):
        if value > 1.0:
            raise ConfigError(f"{context}.{name} must be <= 1")
    return BinaryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=require_int(mapping, "tp", context=context, minimum=0),
        fp=require_int(mapping, "fp", context=context, minimum=0),
        fn=require_int(mapping, "fn", context=context, minimum=0),
        tn=require_int(mapping, "tn", context=context, minimum=0),
    )


def load_oof_diagnostics_spec(path: Path, *, project_root: Path) -> OOFDiagnosticsSpec:
    """Load the complete OOF-diagnostics configuration without algorithm defaults."""

    config_path = path if path.is_absolute() else project_root / path
    raw = load_yaml_mapping(config_path)
    experiment = require_mapping(raw, "experiment", context="oof_diagnostics")
    inputs = require_mapping(raw, "inputs", context="oof_diagnostics")
    score_values = require_mapping(raw, "scores", context="oof_diagnostics")
    cross_validation = require_mapping(raw, "cross_validation", context="oof_diagnostics")
    search_values = require_mapping(raw, "search", context="oof_diagnostics")
    threshold_values = require_mapping(raw, "threshold_analysis", context="oof_diagnostics")
    prior_values = require_mapping(raw, "class_prior", context="oof_diagnostics")
    bootstrap_values = require_mapping(raw, "bootstrap", context="oof_diagnostics")
    winner_values = require_mapping(raw, "frozen_winner", context="oof_diagnostics")
    expected_values = require_mapping(
        winner_values, "expected_metrics", context="oof_diagnostics.frozen_winner"
    )
    runtime_values = require_mapping(raw, "runtime_benchmark", context="oof_diagnostics")
    output_values = require_mapping(raw, "output", context="oof_diagnostics")

    if not require_bool(inputs, "require_oof", context="oof_diagnostics.inputs"):
        raise ConfigError("oof_diagnostics.inputs.require_oof must remain true")
    if (
        require_string(cross_validation, "strategy", context="oof_diagnostics.cross_validation")
        != "leave_one_fold_out"
    ):
        raise ConfigError("oof_diagnostics cross-validation strategy must be leave_one_fold_out")
    if not require_bool(
        cross_validation,
        "fit_candidates_1_to_6_on_training_folds",
        context="oof_diagnostics.cross_validation",
    ):
        raise ConfigError("ablation candidates 1--6 must be fitted leave-one-fold-out")

    search = SearchSpec(
        weight_start=_decimal(search_values, "classifier_weight_start", context="search"),
        weight_end=_decimal(search_values, "classifier_weight_end", context="search"),
        weight_step=_decimal(search_values, "weight_step", context="search"),
        threshold_start=_decimal(search_values, "threshold_start", context="search"),
        threshold_end=_decimal(search_values, "threshold_end", context="search"),
        threshold_step=_decimal(search_values, "threshold_step", context="search"),
    )
    if not Decimal(0) <= search.weight_start <= search.weight_end <= Decimal(1):
        raise ConfigError("classifier weight bounds must satisfy 0 <= start <= end <= 1")
    if search.weight_step <= 0:
        raise ConfigError("search.weight_step must be greater than zero")
    if not Decimal(0) <= search.threshold_start <= search.threshold_end <= Decimal(1):
        raise ConfigError("search threshold bounds must satisfy 0 <= start <= end <= 1")
    if search.threshold_step <= 0:
        raise ConfigError("search.threshold_step must be greater than zero")

    threshold = ThresholdDiagnosticSpec(
        start=_decimal(threshold_values, "start", context="threshold_analysis"),
        end=_decimal(threshold_values, "end", context="threshold_analysis"),
        step=_decimal(threshold_values, "step", context="threshold_analysis"),
        checkpoints=_float_sequence(threshold_values, "checkpoints", context="threshold_analysis"),
        plateau_f1_tolerance=require_float(
            threshold_values,
            "plateau_f1_tolerance",
            context="threshold_analysis",
            minimum=0.0,
        ),
    )
    if not Decimal(0) <= threshold.start <= threshold.end <= Decimal(1):
        raise ConfigError("threshold-analysis bounds must satisfy 0 <= start <= end <= 1")
    if threshold.step <= 0:
        raise ConfigError("threshold_analysis.step must be greater than zero")
    if any(
        not float(threshold.start) <= value <= float(threshold.end)
        for value in threshold.checkpoints
    ):
        raise ConfigError("threshold-analysis checkpoints must lie inside its grid")

    confidence = require_float(
        bootstrap_values, "confidence_level", context="bootstrap", minimum=0.0
    )
    if not 0.0 < confidence < 1.0:
        raise ConfigError("bootstrap.confidence_level must be strictly between 0 and 1")
    bootstrap = BootstrapSpec(
        iterations=require_int(bootstrap_values, "iterations", context="bootstrap", minimum=1),
        seed=require_int(bootstrap_values, "seed", context="bootstrap", minimum=0),
        confidence_level=confidence,
        diagnostic_refit_threshold=require_bool(
            bootstrap_values, "diagnostic_refit_threshold", context="bootstrap"
        ),
    )
    priors = _float_sequence(prior_values, "priors", context="class_prior")
    if any(not 0.0 < value < 1.0 for value in priors):
        raise ConfigError("class_prior.priors values must be strictly between 0 and 1")

    winner = FrozenWinnerSpec(
        classifier_weight=require_float(
            winner_values, "classifier_weight", context="frozen_winner", minimum=0.0
        ),
        stable_threshold=require_float(
            winner_values, "stable_threshold", context="frozen_winner", minimum=0.0
        ),
        recall_threshold=require_float(
            winner_values, "recall_threshold", context="frozen_winner", minimum=0.0
        ),
        precision_threshold=require_float(
            winner_values, "precision_threshold", context="frozen_winner", minimum=0.0
        ),
        expected_metrics=_expected_metrics(expected_values),
    )
    if any(
        not 0.0 <= value <= 1.0
        for value in (
            winner.classifier_weight,
            winner.stable_threshold,
            winner.recall_threshold,
            winner.precision_threshold,
        )
    ):
        raise ConfigError("frozen winner weights and thresholds must be in [0, 1]")
    expected_count = require_int(inputs, "expected_count", context="inputs", minimum=1)
    if (
        sum(
            (
                winner.expected_metrics.tp,
                winner.expected_metrics.fp,
                winner.expected_metrics.fn,
                winner.expected_metrics.tn,
            )
        )
        != expected_count
    ):
        raise ConfigError("frozen winner confusion counts must sum to inputs.expected_count")

    runtime_status = require_string(runtime_values, "status", context="runtime_benchmark")
    if runtime_status != "not_run":
        raise ConfigError("runtime_benchmark.status must remain 'not_run' until measured")

    return OOFDiagnosticsSpec(
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        experiment_name=require_string(experiment, "name", context="experiment"),
        detector_oof=resolve_project_path(
            project_root, require_string(inputs, "detector_oof", context="inputs")
        ),
        detector_manifest=resolve_project_path(
            project_root, require_string(inputs, "detector_manifest", context="inputs")
        ),
        classifier_oof=resolve_project_path(
            project_root, require_string(inputs, "classifier_oof", context="inputs")
        ),
        classifier_manifest=resolve_project_path(
            project_root, require_string(inputs, "classifier_manifest", context="inputs")
        ),
        selection_manifest=resolve_project_path(
            project_root, require_string(inputs, "selection_manifest", context="inputs")
        ),
        folds=resolve_project_path(project_root, require_string(inputs, "folds", context="inputs")),
        processed_manifest=resolve_project_path(
            project_root, require_string(inputs, "processed_manifest", context="inputs")
        ),
        artifact_root=resolve_project_path(
            project_root, require_string(inputs, "artifact_root", context="inputs")
        ),
        expected_count=expected_count,
        expected_fold_count=require_int(inputs, "expected_fold_count", context="inputs", minimum=2),
        official_test_count=require_int(
            prior_values, "official_test_count", context="class_prior", minimum=1
        ),
        scores=ScoreColumns(
            detector_full_original=require_string(
                score_values, "detector_full_original", context="scores"
            ),
            detector_original_full_slice=require_string(
                score_values, "detector_original_full_slice", context="scores"
            ),
            detector_hflip_full=require_string(
                score_values, "detector_hflip_full", context="scores"
            ),
            detector_hflip_full_slice=require_string(
                score_values, "detector_hflip_full_slice", context="scores"
            ),
            detector_tta_full_slice=require_string(
                score_values, "detector_tta_full_slice", context="scores"
            ),
            classifier_original=require_string(
                score_values, "classifier_original", context="scores"
            ),
            classifier_tta=require_string(score_values, "classifier_tta", context="scores"),
        ),
        search=search,
        threshold=threshold,
        class_priors=priors,
        bootstrap=bootstrap,
        winner=winner,
        runtime_benchmark=RuntimeBenchmarkSpec(
            status=runtime_status,
            sample_count=require_int(
                runtime_values, "sample_count", context="runtime_benchmark", minimum=1
            ),
            warmup_count=require_int(
                runtime_values, "warmup_count", context="runtime_benchmark", minimum=0
            ),
            repeats=require_int(runtime_values, "repeats", context="runtime_benchmark", minimum=1),
        ),
        output_dir=resolve_project_path(
            project_root, require_string(output_values, "directory", context="output")
        ),
    )


def _manifest_profile_threshold(selection: Mapping[str, Any], profile: str) -> float:
    profiles = selection.get("threshold_profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("selection manifest threshold_profiles must be a mapping")
    record = profiles.get(profile)
    if not isinstance(record, Mapping):
        raise ValueError(f"selection manifest threshold profile {profile!r} is missing")
    value = record.get("threshold")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"selection manifest {profile} threshold must be numeric")
    return float(value)


def _assert_close(actual: float, expected: float, *, name: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name} drifted: expected {expected}, found {actual}")


def _portable_input_path(path: Path, *, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"diagnostic input must remain below project root: {path}") from exc


def _file_record(path: Path, *, project_root: Path) -> dict[str, object]:
    return {
        "path": _portable_input_path(path, project_root=project_root),
        "sha256": sha256_file(path),
    }


def _git_snapshot(project_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot capture diagnostic Git snapshot: {exc}") from exc
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("diagnostic Git commit is not a lowercase 40-character SHA-1")
    return {"commit": commit, "dirty": bool(status.strip())}


def load_validated_inputs(spec: OOFDiagnosticsSpec, *, project_root: Path) -> ValidatedInputs:
    """Validate manifests, hashes, Fold identity, score formulas, and winner parameters."""

    root = project_root.resolve()
    selection, selection_hash = validate_final_selection_manifest(
        spec.selection_manifest, project_root=root
    )
    detector_manifest, detector_manifest_hash = validate_oof_artifact_manifest(
        spec.detector_manifest,
        artifact_root=spec.artifact_root,
        kind="detector",
        oof_path=spec.detector_oof,
        reference_folds_path=spec.folds,
        processed_manifest_path=spec.processed_manifest,
        expected_rows=spec.expected_count,
    )
    classifier_manifest, classifier_manifest_hash = validate_oof_artifact_manifest(
        spec.classifier_manifest,
        artifact_root=spec.artifact_root,
        kind="classifier",
        oof_path=spec.classifier_oof,
        reference_folds_path=spec.folds,
        processed_manifest_path=spec.processed_manifest,
        expected_rows=spec.expected_count,
    )
    for kind, manifest in (("detector", detector_manifest), ("classifier", classifier_manifest)):
        git = manifest.get("git")
        if not isinstance(git, Mapping) or git.get("dirty") is not False:
            raise ValueError(f"{kind} OOF manifest must bind a clean Git snapshot")

    folds_hash = sha256_file(spec.folds)
    processed_hash = sha256_file(spec.processed_manifest)
    detector = load_oof_csv(
        spec.detector_oof,
        kind="detector",
        expected_rows=spec.expected_count,
        reference_folds_path=spec.folds,
        expected_folds_sha256=folds_hash,
        expected_processed_manifest_sha256=processed_hash,
        artifact_root=spec.artifact_root,
    )
    classifier = load_oof_csv(
        spec.classifier_oof,
        kind="classifier",
        expected_rows=spec.expected_count,
        reference_folds_path=spec.folds,
        expected_folds_sha256=folds_hash,
        expected_processed_manifest_sha256=processed_hash,
        artifact_root=spec.artifact_root,
    )
    frame = combine_oof_frames(
        detector,
        classifier,
        expected_rows=spec.expected_count,
        reference_folds_path=spec.folds,
        expected_folds_sha256=folds_hash,
        expected_processed_manifest_sha256=processed_hash,
    )
    folds = sorted(int(value) for value in frame["fold"].unique())
    if folds != list(range(spec.expected_fold_count)):
        raise ValueError(
            f"OOF Fold set drifted: expected {list(range(spec.expected_fold_count))}, found {folds}"
        )
    if frame.groupby("group_id", sort=False)["fold"].nunique().gt(1).any():
        raise ValueError("OOF group_id values cross Fold boundaries")

    fusion = selection.get("fusion")
    if not isinstance(fusion, Mapping):
        raise ValueError("selection manifest fusion must be a mapping")
    if fusion.get("method") != "weighted_average":
        raise ValueError("selection manifest fusion method must be weighted_average")
    if fusion.get("detector_score_column") != spec.scores.detector_tta_full_slice:
        raise ValueError("selection detector score column differs from diagnostic contract")
    if fusion.get("classifier_score_column") != spec.scores.classifier_tta:
        raise ValueError("selection classifier score column differs from diagnostic contract")
    weight = fusion.get("classifier_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError("selection classifier weight must be numeric")
    _assert_close(float(weight), spec.winner.classifier_weight, name="classifier weight")
    for profile, expected in (
        ("stable", spec.winner.stable_threshold),
        ("recall", spec.winner.recall_threshold),
        ("precision", spec.winner.precision_threshold),
    ):
        _assert_close(
            _manifest_profile_threshold(selection, profile),
            expected,
            name=f"{profile} threshold",
        )

    return ValidatedInputs(
        frame=frame,
        selection=selection,
        input_records={
            "config": _file_record(spec.config_path, project_root=root),
            "detector_oof": _file_record(spec.detector_oof, project_root=root),
            "detector_manifest": {
                "path": _portable_input_path(spec.detector_manifest, project_root=root),
                "sha256": detector_manifest_hash,
            },
            "classifier_oof": _file_record(spec.classifier_oof, project_root=root),
            "classifier_manifest": {
                "path": _portable_input_path(spec.classifier_manifest, project_root=root),
                "sha256": classifier_manifest_hash,
            },
            "selection_manifest": {
                "path": _portable_input_path(spec.selection_manifest, project_root=root),
                "sha256": selection_hash,
            },
            "folds": _file_record(spec.folds, project_root=root),
            "processed_manifest": _file_record(spec.processed_manifest, project_root=root),
        },
    )


def _validate_score_formulas(frame: pd.DataFrame, scores: ScoreColumns) -> pd.DataFrame:
    required = {
        "filename",
        "label",
        "fold",
        "group_id",
        "det_max_conf",
        "slice_max_conf",
        "det_original_full_slice_conf",
        "det_hflip_max_conf",
        "slice_hflip_max_conf",
        "det_hflip_full_slice_conf",
        "det_tta_mean_conf",
        "cls_probability",
        "cls_original_probability",
        "cls_hflip_probability",
        "cls_tta_mean_probability",
        *asdict(scores).values(),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"OOF diagnostics missing required columns: {missing}")
    data = frame.copy().sort_values("filename", kind="stable").reset_index(drop=True)
    score_columns = sorted(required.difference({"filename", "label", "fold", "group_id"}))
    for column in score_columns:
        values = pd.to_numeric(data[column], errors="raise").astype(float)
        if not np.isfinite(values.to_numpy()).all() or not values.between(0.0, 1.0).all():
            raise ValueError(f"OOF score column {column!r} must be finite and in [0, 1]")
        data[column] = values

    invariants = {
        "det_original_full_slice_conf": np.maximum(data["det_max_conf"], data["slice_max_conf"]),
        "det_hflip_full_slice_conf": np.maximum(
            data["det_hflip_max_conf"], data["slice_hflip_max_conf"]
        ),
        "det_tta_mean_conf": 0.5
        * (data["det_original_full_slice_conf"] + data["det_hflip_full_slice_conf"]),
        "cls_original_probability": data["cls_probability"],
        "cls_tta_mean_probability": 0.5
        * (data["cls_original_probability"] + data["cls_hflip_probability"]),
    }
    for column, expected in invariants.items():
        if not np.allclose(
            data[column].to_numpy(dtype=float),
            np.asarray(expected, dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"OOF deterministic score formula drifted for {column}")
    if not np.allclose(
        data[scores.detector_original_full_slice].to_numpy(dtype=float),
        data["det_original_full_slice_conf"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("configured detector original full+slice score formula drifted")
    if not np.allclose(
        data[scores.detector_hflip_full_slice].to_numpy(dtype=float),
        data["det_hflip_full_slice_conf"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("configured detector hflip full+slice score formula drifted")
    return data


def _standardized_predictions(
    frame: pd.DataFrame,
    *,
    candidate: str,
    order: int,
    contract: str,
) -> pd.DataFrame:
    required = {
        "filename",
        "label",
        "fold",
        "group_id",
        "score",
        "threshold",
        "y_pred",
        "margin",
        "confusion",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate {candidate} predictions missing columns: {missing}")
    result = frame[
        [
            "filename",
            "label",
            "fold",
            "group_id",
            "score",
            "threshold",
            "y_pred",
            "margin",
            "confusion",
        ]
    ].copy()
    result.insert(0, "candidate", candidate)
    result.insert(1, "candidate_order", order)
    result.insert(2, "evaluation_contract", contract)
    if "classifier_weight" in frame:
        result["classifier_weight"] = frame["classifier_weight"].astype(float)
    else:
        result["classifier_weight"] = None
    return result.sort_values("filename", kind="stable").reset_index(drop=True)


def _candidate_from_evaluation(
    *,
    name: str,
    order: int,
    method: CandidateMethod,
    detector_feature: str | None,
    classifier_feature: str | None,
    evaluation: ModelEvaluation | FusionEvaluation,
) -> CandidateResult:
    contract = "strict_leave_one_fold_out_fit"
    return CandidateResult(
        name=name,
        order=order,
        method=method,
        evaluation_contract=contract,
        detector_feature=detector_feature,
        classifier_feature=classifier_feature,
        pooled_threshold=evaluation.pooled.best.threshold,
        pooled_classifier_weight=(
            evaluation.pooled_weight if isinstance(evaluation, FusionEvaluation) else None
        ),
        metrics=evaluation.cross_fitted.metrics,
        per_fold=evaluation.cross_fitted.per_fold,
        predictions=_standardized_predictions(
            evaluation.cross_fitted.predictions,
            candidate=name,
            order=order,
            contract=contract,
        ),
    )


def _single_candidate(
    frame: pd.DataFrame,
    *,
    name: str,
    order: int,
    method: CandidateMethod,
    feature: str,
    search: SearchSpec,
) -> CandidateResult:
    source = frame[["filename", "label", "fold", "group_id", feature]].rename(
        columns={feature: "candidate_score"}
    )
    evaluation = evaluate_single_model(
        source,
        score_column="candidate_score",
        threshold_start=search.threshold_start,
        threshold_end=search.threshold_end,
        threshold_step=search.threshold_step,
    )
    return _candidate_from_evaluation(
        name=name,
        order=order,
        method=method,
        detector_feature=feature if method == "single_detector" else None,
        classifier_feature=feature if method == "single_classifier" else None,
        evaluation=evaluation,
    )


def _fusion_candidate(
    frame: pd.DataFrame,
    *,
    name: str,
    order: int,
    detector_feature: str,
    classifier_feature: str,
    search: SearchSpec,
) -> CandidateResult:
    source = frame[
        ["filename", "label", "fold", "group_id", detector_feature, classifier_feature]
    ].rename(columns={detector_feature: "detector_score", classifier_feature: "classifier_score"})
    evaluation = evaluate_weighted_fusion(
        source,
        detector_column="detector_score",
        classifier_column="classifier_score",
        weight_start=search.weight_start,
        weight_end=search.weight_end,
        weight_step=search.weight_step,
        threshold_start=search.threshold_start,
        threshold_end=search.threshold_end,
        threshold_step=search.threshold_step,
    )
    return _candidate_from_evaluation(
        name=name,
        order=order,
        method="weighted_average",
        detector_feature=detector_feature,
        classifier_feature=classifier_feature,
        evaluation=evaluation,
    )


def _confusion(label: int, prediction: int) -> str:
    return {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}[(label, prediction)]


def _frozen_candidate(
    frame: pd.DataFrame,
    *,
    detector_feature: str,
    classifier_feature: str,
    winner: FrozenWinnerSpec,
) -> CandidateResult:
    data = frame[["filename", "label", "fold", "group_id"]].copy()
    data["score"] = (1.0 - winner.classifier_weight) * frame[
        detector_feature
    ] + winner.classifier_weight * frame[classifier_feature]
    data["threshold"] = winner.stable_threshold
    data["classifier_weight"] = winner.classifier_weight
    data["y_pred"] = (data["score"] >= winner.stable_threshold).astype(int)
    data["margin"] = data["score"] - winner.stable_threshold
    data["confusion"] = [
        _confusion(int(label), int(prediction))
        for label, prediction in zip(data["label"], data["y_pred"], strict=True)
    ]
    metrics = compute_binary_metrics(data["label"].tolist(), data["y_pred"].tolist(), 0.5)
    per_fold: list[FoldEvaluation] = []
    for fold in sorted(int(value) for value in data["fold"].unique()):
        held = data[data["fold"] == fold]
        fold_metrics = compute_binary_metrics(held["label"].tolist(), held["y_pred"].tolist(), 0.5)
        per_fold.append(
            FoldEvaluation(
                fold=fold,
                threshold=winner.stable_threshold,
                classifier_weight=winner.classifier_weight,
                metrics=fold_metrics,
            )
        )
    contract = "frozen_full_oof_weight_and_threshold_no_refit"
    return CandidateResult(
        name=FROZEN_WINNER,
        order=7,
        method="weighted_average",
        evaluation_contract=contract,
        detector_feature=detector_feature,
        classifier_feature=classifier_feature,
        pooled_threshold=winner.stable_threshold,
        pooled_classifier_weight=winner.classifier_weight,
        metrics=metrics,
        per_fold=tuple(per_fold),
        predictions=_standardized_predictions(
            data,
            candidate=FROZEN_WINNER,
            order=7,
            contract=contract,
        ),
    )


def _metric_counts(metrics: BinaryMetrics) -> tuple[int, int, int, int]:
    return metrics.tp, metrics.fp, metrics.fn, metrics.tn


def _verify_frozen_metrics(
    result: CandidateResult,
    *,
    winner: FrozenWinnerSpec,
    selection: Mapping[str, Any] | None,
) -> None:
    expected = winner.expected_metrics
    if _metric_counts(result.metrics) != _metric_counts(expected):
        raise ValueError(
            "frozen winner confusion matrix drifted: "
            f"expected {_metric_counts(expected)}, found {_metric_counts(result.metrics)}"
        )
    for name in ("precision", "recall", "f1"):
        _assert_close(
            float(getattr(result.metrics, name)),
            float(getattr(expected, name)),
            name=f"frozen winner {name}",
        )
    if selection is None:
        return
    metrics = selection.get("metrics")
    cross_fitted = metrics.get("cross_fitted") if isinstance(metrics, Mapping) else None
    if not isinstance(cross_fitted, Mapping):
        raise ValueError("selection manifest cross-fitted metrics are missing")
    for name, actual in asdict(result.metrics).items():
        recorded = cross_fitted.get(name)
        if name in {"tp", "fp", "fn", "tn"}:
            if type(recorded) is not int or recorded != actual:
                raise ValueError(f"selection manifest frozen winner {name} drifted")
        elif isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
            raise ValueError(f"selection manifest frozen winner {name} is not numeric")
        else:
            _assert_close(float(recorded), float(actual), name=f"selection frozen winner {name}")


def evaluate_fixed_ablation(
    frame: pd.DataFrame,
    *,
    scores: ScoreColumns,
    search: SearchSpec,
    winner: FrozenWinnerSpec,
    selection: Mapping[str, Any] | None = None,
) -> tuple[CandidateResult, ...]:
    """Evaluate the six strict-LOFO candidates and the frozen operation point."""

    data = _validate_score_formulas(frame, scores)
    data["det_full_tta_mean_conf"] = 0.5 * (
        data[scores.detector_full_original] + data[scores.detector_hflip_full]
    )
    results = (
        _single_candidate(
            data,
            name=DETECTOR_FULL_ORIGINAL,
            order=1,
            method="single_detector",
            feature=scores.detector_full_original,
            search=search,
        ),
        _single_candidate(
            data,
            name=CLASSIFIER_ORIGINAL,
            order=2,
            method="single_classifier",
            feature=scores.classifier_original,
            search=search,
        ),
        _fusion_candidate(
            data,
            name=FUSION_FULL_ORIGINAL,
            order=3,
            detector_feature=scores.detector_full_original,
            classifier_feature=scores.classifier_original,
            search=search,
        ),
        _fusion_candidate(
            data,
            name=FUSION_ORIGINAL_SLICING,
            order=4,
            detector_feature=scores.detector_original_full_slice,
            classifier_feature=scores.classifier_original,
            search=search,
        ),
        _fusion_candidate(
            data,
            name=FUSION_FULL_TTA,
            order=5,
            detector_feature="det_full_tta_mean_conf",
            classifier_feature=scores.classifier_tta,
            search=search,
        ),
        _fusion_candidate(
            data,
            name=FUSION_SLICING_TTA_LOFO,
            order=6,
            detector_feature=scores.detector_tta_full_slice,
            classifier_feature=scores.classifier_tta,
            search=search,
        ),
        _frozen_candidate(
            data,
            detector_feature=scores.detector_tta_full_slice,
            classifier_feature=scores.classifier_tta,
            winner=winner,
        ),
    )
    if tuple(result.name for result in results) != CANDIDATE_ORDER:
        raise AssertionError("fixed candidate order changed")
    _verify_frozen_metrics(results[-1], winner=winner, selection=selection)
    return results


def _fold_metric_map(candidate: CandidateResult) -> dict[int, BinaryMetrics]:
    return {point.fold: point.metrics for point in candidate.per_fold}


def _aligned_predictions(reference: CandidateResult, candidate: CandidateResult) -> pd.DataFrame:
    left = reference.predictions[["filename", "label", "fold", "y_pred"]].rename(
        columns={"label": "reference_label", "fold": "reference_fold", "y_pred": "reference_pred"}
    )
    right = candidate.predictions[["filename", "label", "fold", "y_pred"]].rename(
        columns={"label": "candidate_label", "fold": "candidate_fold", "y_pred": "candidate_pred"}
    )
    aligned = left.merge(right, on="filename", how="outer", indicator=True, validate="one_to_one")
    if (aligned["_merge"] != "both").any():
        raise ValueError("ablation candidate filename sets differ")
    if not (aligned["reference_label"] == aligned["candidate_label"]).all():
        raise ValueError("ablation candidate labels differ")
    if not (aligned["reference_fold"] == aligned["candidate_fold"]).all():
        raise ValueError("ablation candidate Folds differ")
    return aligned


def _transition_counts(reference: CandidateResult, candidate: CandidateResult) -> dict[str, int]:
    aligned = _aligned_predictions(reference, candidate)
    label = aligned["reference_label"].to_numpy(dtype=int)
    before = aligned["reference_pred"].to_numpy(dtype=int)
    after = aligned["candidate_pred"].to_numpy(dtype=int)
    return {
        "new_fp": int(((label == 0) & (before == 0) & (after == 1)).sum()),
        "recovered_fp": int(((label == 0) & (before == 1) & (after == 0)).sum()),
        "new_fn": int(((label == 1) & (before == 1) & (after == 0)).sum()),
        "recovered_fn": int(((label == 1) & (before == 0) & (after == 1)).sum()),
    }


def build_ablation_outputs(
    candidates: Sequence[CandidateResult],
    *,
    benchmark: RuntimeBenchmarkSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build aggregate, Fold, prediction, transition, and nullable resource outputs."""

    by_name = {candidate.name: candidate for candidate in candidates}
    if tuple(by_name) != CANDIDATE_ORDER:
        raise ValueError("ablation output requires the fixed seven-candidate order")
    report_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for candidate in candidates:
        fold_f1 = np.asarray([point.metrics.f1 for point in candidate.per_fold], dtype=float)
        reference_name = PRIMARY_REFERENCES.get(candidate.name)
        reference = by_name.get(reference_name) if reference_name is not None else None
        gain: float | None = None
        folds_not_worse: int | None = None
        worst_fold_delta: float | None = None
        transitions: dict[str, int | None] = {
            "new_fp": None,
            "recovered_fp": None,
            "new_fn": None,
            "recovered_fn": None,
        }
        if reference is not None:
            gain = candidate.metrics.f1 - reference.metrics.f1
            candidate_folds = _fold_metric_map(candidate)
            reference_folds = _fold_metric_map(reference)
            deltas = [
                candidate_folds[fold].f1 - reference_folds[fold].f1
                for fold in sorted(candidate_folds)
            ]
            folds_not_worse = sum(delta >= -1e-15 for delta in deltas)
            worst_fold_delta = min(deltas)
            transitions.update(_transition_counts(reference, candidate))
        report_rows.append(
            {
                "candidate_order": candidate.order,
                "candidate": candidate.name,
                "method": candidate.method,
                "evaluation_contract": candidate.evaluation_contract,
                "detector_feature": candidate.detector_feature,
                "classifier_feature": candidate.classifier_feature,
                "pooled_threshold": candidate.pooled_threshold,
                "pooled_classifier_weight": candidate.pooled_classifier_weight,
                **asdict(candidate.metrics),
                "fold_f1_mean": float(fold_f1.mean()),
                "fold_f1_sample_std": float(fold_f1.std(ddof=1)),
                "fold_f1_min": float(fold_f1.min()),
                "primary_reference": reference_name,
                "f1_gain_vs_primary_reference": gain,
                "folds_not_worse_vs_primary_reference": folds_not_worse,
                "worst_fold_delta_vs_primary_reference": worst_fold_delta,
                "gain_at_least_0_001": gain is not None and gain >= 0.001,
                "at_least_3_folds_not_worse": (
                    folds_not_worse is not None and folds_not_worse >= 3
                ),
                **transitions,
                "runtime_benchmark_status": benchmark.status,
                "benchmark_total_seconds": None,
                "benchmark_mean_seconds_per_image": None,
                "benchmark_p50_seconds_per_image": None,
                "benchmark_p95_seconds_per_image": None,
                "benchmark_peak_gpu_memory_bytes": None,
                "benchmark_peak_process_tree_ram_bytes": None,
                "cost_gate_evaluable": False,
            }
        )
        prediction_frames.append(candidate.predictions)
        for point in candidate.per_fold:
            fold_rows.append(
                {
                    "candidate_order": candidate.order,
                    "candidate": candidate.name,
                    "method": candidate.method,
                    "evaluation_contract": candidate.evaluation_contract,
                    "fold": point.fold,
                    "threshold": point.threshold,
                    "classifier_weight": point.classifier_weight,
                    **asdict(point.metrics),
                }
            )
    for reference_name, candidate_name, change in TRANSITION_PAIRS:
        reference = by_name[reference_name]
        candidate = by_name[candidate_name]
        reference_folds = _fold_metric_map(reference)
        candidate_folds = _fold_metric_map(candidate)
        deltas = [
            candidate_folds[fold].f1 - reference_folds[fold].f1 for fold in sorted(reference_folds)
        ]
        transition_rows.append(
            {
                "change": change,
                "reference": reference_name,
                "candidate": candidate_name,
                "f1_gain": candidate.metrics.f1 - reference.metrics.f1,
                "recall_gain": candidate.metrics.recall - reference.metrics.recall,
                "precision_gain": candidate.metrics.precision - reference.metrics.precision,
                "folds_not_worse": sum(delta >= -1e-15 for delta in deltas),
                "worst_fold_delta": min(deltas),
                **_transition_counts(reference, candidate),
            }
        )
    resource_report: dict[str, object] = {
        "status": benchmark.status,
        "reason": "GPU runtime benchmark was not executed by the OOF-only analysis command",
        "declared_protocol": {
            "sample_count": benchmark.sample_count,
            "warmup_count": benchmark.warmup_count,
            "repeats": benchmark.repeats,
            "sampling": "fixed_engineering_set_covering_tile_count_quantiles",
        },
        "measurements": [
            {
                "candidate": candidate.name,
                "total_seconds": None,
                "mean_seconds_per_image": None,
                "p50_seconds_per_image": None,
                "p95_seconds_per_image": None,
                "peak_gpu_memory_bytes": None,
                "peak_process_tree_ram_bytes": None,
            }
            for candidate in candidates
        ],
    }
    return (
        pd.DataFrame(report_rows),
        pd.DataFrame(fold_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(transition_rows),
        resource_report,
    )


def _checkpoint_profile(threshold: float, winner: FrozenWinnerSpec) -> str:
    names = [
        name
        for name, value in (
            ("recall", winner.recall_threshold),
            ("stable", winner.stable_threshold),
            ("precision", winner.precision_threshold),
        )
        if math.isclose(threshold, value, rel_tol=0.0, abs_tol=1e-12)
    ]
    return "+".join(names)


def _threshold_row(
    *,
    analysis_kind: str,
    scope: str,
    fold: int | None,
    threshold: float,
    semantics: str,
    labels: Sequence[int],
    scores: Sequence[float],
    checkpoints: Sequence[float],
    winner: FrozenWinnerSpec,
) -> dict[str, object]:
    metrics = compute_binary_metrics(labels, scores, threshold)
    return {
        "analysis_kind": analysis_kind,
        "scope": scope,
        "fold": fold,
        "threshold": threshold,
        "threshold_semantics": semantics,
        "is_prefrozen_checkpoint": any(
            math.isclose(threshold, value, rel_tol=0.0, abs_tol=1e-12) for value in checkpoints
        ),
        "profile": _checkpoint_profile(threshold, winner),
        **asdict(metrics),
    }


def analyze_threshold_stability(
    winner_predictions: pd.DataFrame,
    *,
    diagnostic: ThresholdDiagnosticSpec,
    winner: FrozenWinnerSpec,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Analyze the frozen score on a dense grid and at exact score breakpoints."""

    required = {"filename", "label", "fold", "score"}
    missing = sorted(required.difference(winner_predictions.columns))
    if missing:
        raise ValueError(f"winner predictions missing threshold-analysis columns: {missing}")
    data = winner_predictions.copy().sort_values("filename", kind="stable").reset_index(drop=True)
    thresholds = tuple(
        float(value)
        for value in build_threshold_grid(diagnostic.start, diagnostic.end, diagnostic.step)
    )
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, int | None, pd.DataFrame]] = [("overall", None, data)]
    scopes.extend(
        (f"fold_{fold}", fold, data[data["fold"] == fold])
        for fold in sorted(int(value) for value in data["fold"].unique())
    )
    for scope, fold, subset in scopes:
        for threshold in thresholds:
            rows.append(
                _threshold_row(
                    analysis_kind="grid",
                    scope=scope,
                    fold=fold,
                    threshold=threshold,
                    semantics="score_greater_than_or_equal_to_threshold",
                    labels=subset["label"].tolist(),
                    scores=subset["score"].tolist(),
                    checkpoints=diagnostic.checkpoints,
                    winner=winner,
                )
            )
    exact_scores = sorted(
        {
            float(score)
            for score in data["score"]
            if float(diagnostic.start) <= float(score) <= float(diagnostic.end)
        }
    )
    for score in exact_scores:
        rows.append(
            _threshold_row(
                analysis_kind="exact_breakpoint",
                scope="overall",
                fold=None,
                threshold=score,
                semantics="inclusive_at_observed_score",
                labels=data["label"].tolist(),
                scores=data["score"].tolist(),
                checkpoints=diagnostic.checkpoints,
                winner=winner,
            )
        )
        rows.append(
            _threshold_row(
                analysis_kind="exact_breakpoint",
                scope="overall",
                fold=None,
                threshold=float(np.nextafter(score, math.inf)),
                semantics="immediately_above_observed_score",
                labels=data["label"].tolist(),
                scores=data["score"].tolist(),
                checkpoints=diagnostic.checkpoints,
                winner=winner,
            )
        )
    analysis = pd.DataFrame(rows)
    overall_grid = analysis[
        (analysis["analysis_kind"] == "grid") & (analysis["scope"] == "overall")
    ].reset_index(drop=True)
    search = search_thresholds(
        data["label"].tolist(),
        data["score"].tolist(),
        start=diagnostic.start,
        end=diagnostic.end,
        step=diagnostic.step,
    )
    best_threshold = search.best.threshold
    best_index = int(
        np.flatnonzero(
            np.isclose(
                overall_grid["threshold"].to_numpy(dtype=float),
                best_threshold,
                rtol=0.0,
                atol=1e-12,
            )
        )[0]
    )
    acceptable = overall_grid["f1"].to_numpy(dtype=float) >= (
        search.best.metrics.f1 - diagnostic.plateau_f1_tolerance - 1e-15
    )
    left = best_index
    right = best_index
    while left > 0 and acceptable[left - 1]:
        left -= 1
    while right + 1 < len(acceptable) and acceptable[right + 1]:
        right += 1
    per_fold: list[dict[str, object]] = []
    for fold in sorted(int(value) for value in data["fold"].unique()):
        held = data[data["fold"] == fold]
        oracle = search_thresholds(
            held["label"].tolist(),
            held["score"].tolist(),
            start=diagnostic.start,
            end=diagnostic.end,
            step=diagnostic.step,
        )
        fixed = compute_binary_metrics(
            held["label"].tolist(), held["score"].tolist(), winner.stable_threshold
        )
        per_fold.append(
            {
                "fold": fold,
                "oracle_contract": "descriptive_same_fold_threshold_fit_not_for_selection",
                "oracle_threshold": oracle.best.threshold,
                "oracle_metrics": asdict(oracle.best.metrics),
                "fixed_threshold": winner.stable_threshold,
                "fixed_metrics": asdict(fixed),
                "oracle_minus_fixed_f1": oracle.best.metrics.f1 - fixed.f1,
            }
        )
    checkpoint_rows = overall_grid[overall_grid["is_prefrozen_checkpoint"]]
    summary: dict[str, object] = {
        "grid": {
            "start": str(diagnostic.start),
            "end": str(diagnostic.end),
            "step": str(diagnostic.step),
            "point_count": len(thresholds),
        },
        "best": {"threshold": best_threshold, "metrics": asdict(search.best.metrics)},
        "stable_plateau": {
            "definition": "contiguous region containing grid best with f1 >= best - tolerance",
            "f1_tolerance": diagnostic.plateau_f1_tolerance,
            "lower": float(overall_grid.iloc[left]["threshold"]),
            "upper": float(overall_grid.iloc[right]["threshold"]),
            "width": float(
                overall_grid.iloc[right]["threshold"] - overall_grid.iloc[left]["threshold"]
            ),
            "point_count": right - left + 1,
            "truncated_at_lower_grid_bound": left == 0,
            "truncated_at_upper_grid_bound": right == len(overall_grid) - 1,
        },
        "prefrozen_checkpoints": checkpoint_rows[
            ["threshold", "profile", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
        ].to_dict("records"),
        "per_fold": per_fold,
        "exact_breakpoints": {
            "unique_scores_in_range": len(exact_scores),
            "rows": 2 * len(exact_scores),
            "semantics": [
                "inclusive_at_observed_score",
                "immediately_above_observed_score",
            ],
        },
    }
    return analysis, summary


def _simulated_best_index(frame: pd.DataFrame) -> int:
    f1 = frame["expected_f1"].to_numpy(dtype=float)
    recall = frame["expected_recall"].to_numpy(dtype=float)
    best_f1 = f1.max()
    mask = np.isclose(f1, best_f1, rtol=0.0, atol=1e-15)
    best_recall = recall[mask].max()
    tied = np.flatnonzero(mask & np.isclose(recall, best_recall, rtol=0.0, atol=1e-15))
    return int(tied[(len(tied) - 1) // 2])


def simulate_class_priors(
    winner_predictions: pd.DataFrame,
    *,
    diagnostic: ThresholdDiagnosticSpec,
    winner: FrozenWinnerSpec,
    priors: Sequence[float],
    official_test_count: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Simulate precision/F1 under fixed conditional TPR/FPR and varying class prior."""

    labels = winner_predictions["label"].astype(int).to_numpy()
    scores = winner_predictions["score"].astype(float).to_numpy()
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("class-prior simulation requires both positive and negative OOF rows")
    thresholds = tuple(
        float(value)
        for value in build_threshold_grid(diagnostic.start, diagnostic.end, diagnostic.step)
    )
    rows: list[dict[str, object]] = []
    for prior in priors:
        for threshold in thresholds:
            metrics = compute_binary_metrics(labels.tolist(), scores.tolist(), threshold)
            tpr = metrics.recall
            fpr = metrics.fp / negative_count
            denominator = prior * tpr + (1.0 - prior) * fpr
            precision = prior * tpr / denominator if denominator else 0.0
            recall = tpr
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(
                {
                    "prior": prior,
                    "threshold": threshold,
                    "tpr": tpr,
                    "fpr": fpr,
                    "expected_precision": precision,
                    "expected_recall": recall,
                    "expected_f1": f1,
                    "expected_fp": official_test_count * (1.0 - prior) * fpr,
                    "expected_fn": official_test_count * prior * (1.0 - tpr),
                    "is_prefrozen_checkpoint": any(
                        math.isclose(threshold, point, rel_tol=0.0, abs_tol=1e-12)
                        for point in diagnostic.checkpoints
                    ),
                    "profile": _checkpoint_profile(threshold, winner),
                    "is_simulated_optimum": False,
                }
            )
    result = pd.DataFrame(rows)
    optima: list[dict[str, object]] = []
    for prior in priors:
        subset_indices = result.index[result["prior"] == prior].to_numpy()
        local = result.loc[subset_indices].reset_index(drop=True)
        local_index = _simulated_best_index(local)
        global_index = int(subset_indices[local_index])
        result.loc[global_index, "is_simulated_optimum"] = True
        optimum = result.loc[global_index]
        optima.append(
            {
                "prior": prior,
                "threshold": float(optimum["threshold"]),
                "expected_precision": float(optimum["expected_precision"]),
                "expected_recall": float(optimum["expected_recall"]),
                "expected_f1": float(optimum["expected_f1"]),
                "interpretation": (
                    "sensitivity_only_not_authorized_for_official_threshold_selection"
                ),
            }
        )
    assumptions: dict[str, object] = {
        "official_test_count": official_test_count,
        "priors": list(priors),
        "simulated_optima": optima,
        "assumptions": [
            "OOF conditional true-positive and false-positive rates remain stable",
            "only the positive-class prior changes",
            "scene correlation and other covariate shift are not modeled",
            "official thresholds remain frozen; simulated optima are sensitivity diagnostics only",
            "test images and inferred test prevalence are never used",
        ],
    }
    return result, assumptions


def _metric_arrays(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = counts[..., 0]
    fp = counts[..., 1]
    fn = counts[..., 2]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    denominator = 2.0 * tp + fp + fn
    f1 = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator != 0,
    )
    return precision, recall, f1


def _percentile_interval(values: np.ndarray, confidence_level: float) -> dict[str, float]:
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return {"lower": float(lower), "upper": float(upper)}


def cluster_bootstrap(
    candidates: Sequence[CandidateResult],
    *,
    diagnostic: ThresholdDiagnosticSpec,
    bootstrap: BootstrapSpec,
) -> dict[str, object]:
    """Bootstrap complete ``group_id`` clusters and keep diagnostic refits separate."""

    by_name = {candidate.name: candidate for candidate in candidates}
    if tuple(by_name) != CANDIDATE_ORDER:
        raise ValueError("bootstrap requires the fixed seven-candidate order")
    reference = by_name[FROZEN_WINNER].predictions.sort_values("filename", kind="stable")
    group_codes, group_values = pd.factorize(reference["group_id"], sort=True)
    group_count = len(group_values)
    if group_count < 2:
        raise ValueError("cluster bootstrap requires at least two group_id clusters")
    if reference.groupby("group_id", sort=False)["fold"].nunique().gt(1).any():
        raise ValueError("cluster bootstrap refuses group_id values crossing Folds")

    grouped_counts = np.zeros((group_count, len(candidates), 4), dtype=np.int64)
    reference_names = reference["filename"].tolist()
    reference_labels = reference["label"].to_numpy(dtype=int)
    for candidate_index, candidate in enumerate(candidates):
        aligned = candidate.predictions.set_index("filename").loc[reference_names]
        if not np.array_equal(aligned["label"].to_numpy(dtype=int), reference_labels):
            raise ValueError(f"bootstrap candidate {candidate.name} labels differ")
        predictions = aligned["y_pred"].to_numpy(dtype=int)
        indicators = np.column_stack(
            (
                (reference_labels == 1) & (predictions == 1),
                (reference_labels == 0) & (predictions == 1),
                (reference_labels == 1) & (predictions == 0),
                (reference_labels == 0) & (predictions == 0),
            )
        ).astype(np.int64)
        np.add.at(grouped_counts[:, candidate_index, :], group_codes, indicators)

    rng = np.random.default_rng(bootstrap.seed)
    multiplicities = rng.multinomial(
        group_count,
        np.full(group_count, 1.0 / group_count, dtype=float),
        size=bootstrap.iterations,
    ).astype(float)
    sampled_counts = (multiplicities @ grouped_counts.reshape(group_count, -1)).reshape(
        bootstrap.iterations, len(candidates), 4
    )
    precision, recall, f1 = _metric_arrays(sampled_counts)
    reference_index = CANDIDATE_ORDER.index(FROZEN_WINNER)
    reference_result = by_name[FROZEN_WINNER]
    fixed_threshold: dict[str, object] = {
        "candidate": FROZEN_WINNER,
        "contract": "frozen_weight_and_threshold_applied_before_cluster_resampling",
        "point": asdict(reference_result.metrics),
        "confidence_intervals": {
            "precision": _percentile_interval(
                precision[:, reference_index], bootstrap.confidence_level
            ),
            "recall": _percentile_interval(recall[:, reference_index], bootstrap.confidence_level),
            "f1": _percentile_interval(f1[:, reference_index], bootstrap.confidence_level),
        },
    }
    paired: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        if candidate.name == FROZEN_WINNER:
            continue
        delta_f1 = f1[:, index] - f1[:, reference_index]
        delta_recall = recall[:, index] - recall[:, reference_index]
        delta_fp = sampled_counts[:, index, 1] - sampled_counts[:, reference_index, 1]
        paired.append(
            {
                "candidate": candidate.name,
                "reference": FROZEN_WINNER,
                "point_delta_f1": candidate.metrics.f1 - reference_result.metrics.f1,
                "delta_f1_ci": _percentile_interval(delta_f1, bootstrap.confidence_level),
                "probability_delta_f1_gt_zero": float(np.mean(delta_f1 > 0.0)),
                "point_delta_recall": candidate.metrics.recall - reference_result.metrics.recall,
                "delta_recall_ci": _percentile_interval(delta_recall, bootstrap.confidence_level),
                "point_delta_fp": candidate.metrics.fp - reference_result.metrics.fp,
                "delta_fp_ci": _percentile_interval(delta_fp, bootstrap.confidence_level),
                "probability_candidate_fp_less_than_reference": float(np.mean(delta_fp < 0.0)),
            }
        )

    diagnostic_refit: dict[str, object] = {
        "enabled": bootstrap.diagnostic_refit_threshold,
        "contract": "diagnostic_same_cluster_resample_threshold_refit_not_for_model_selection",
    }
    if bootstrap.diagnostic_refit_threshold:
        thresholds = np.asarray(
            [
                float(value)
                for value in build_threshold_grid(diagnostic.start, diagnostic.end, diagnostic.step)
            ],
            dtype=float,
        )
        raw_scores = reference["score"].to_numpy(dtype=float)
        positive = reference_labels == 1
        predicted = raw_scores[:, None] >= thresholds[None, :]
        grouped_tp = np.zeros((group_count, len(thresholds)), dtype=np.int64)
        grouped_fp = np.zeros((group_count, len(thresholds)), dtype=np.int64)
        np.add.at(grouped_tp, group_codes, predicted & positive[:, None])
        np.add.at(grouped_fp, group_codes, predicted & ~positive[:, None])
        positive_by_group = np.bincount(
            group_codes, weights=positive.astype(float), minlength=group_count
        )
        sampled_tp = multiplicities @ grouped_tp
        sampled_fp = multiplicities @ grouped_fp
        sampled_positive = multiplicities @ positive_by_group
        sampled_fn = sampled_positive[:, None] - sampled_tp
        refit_counts = np.stack(
            (sampled_tp, sampled_fp, sampled_fn, np.zeros_like(sampled_tp)), axis=-1
        )
        _refit_precision, refit_recall, refit_f1 = _metric_arrays(refit_counts)
        best_f1 = refit_f1.max(axis=1)
        f1_mask = np.isclose(refit_f1, best_f1[:, None], rtol=0.0, atol=1e-15)
        masked_recall = np.where(f1_mask, refit_recall, -1.0)
        best_recall = masked_recall.max(axis=1)
        tied = f1_mask & np.isclose(refit_recall, best_recall[:, None], rtol=0.0, atol=1e-15)
        first = np.argmax(tied, axis=1)
        last = len(thresholds) - 1 - np.argmax(tied[:, ::-1], axis=1)
        selected_index = (first + last) // 2
        selected_threshold = thresholds[selected_index]
        selected_f1 = refit_f1[np.arange(bootstrap.iterations), selected_index]
        selected_recall = refit_recall[np.arange(bootstrap.iterations), selected_index]
        unique_thresholds, threshold_counts = np.unique(selected_threshold, return_counts=True)
        diagnostic_refit.update(
            {
                "grid": {
                    "start": str(diagnostic.start),
                    "end": str(diagnostic.end),
                    "step": str(diagnostic.step),
                },
                "threshold_ci": _percentile_interval(
                    selected_threshold, bootstrap.confidence_level
                ),
                "threshold_median": float(np.median(selected_threshold)),
                "threshold_distribution": {
                    f"{threshold:.12g}": int(count)
                    for threshold, count in zip(unique_thresholds, threshold_counts, strict=True)
                },
                "refit_f1_ci": _percentile_interval(selected_f1, bootstrap.confidence_level),
                "refit_recall_ci": _percentile_interval(
                    selected_recall, bootstrap.confidence_level
                ),
                "warning": "same-resample refit is optimistic and remains diagnostic only",
            }
        )

    return {
        "schema_version": 1,
        "contract": "nonparametric_cluster_bootstrap_group_id_with_replacement",
        "iterations": bootstrap.iterations,
        "seed": bootstrap.seed,
        "confidence_level": bootstrap.confidence_level,
        "group_count": group_count,
        "row_count": len(reference),
        "fixed_threshold": fixed_threshold,
        "paired_candidate_comparisons": paired,
        "diagnostic_refit_threshold": diagnostic_refit,
        "limitations": [
            "complete group_id clusters are sampled; rows within a cluster are never split",
            "the interval quantifies sampling uncertainty for these fixed OOF predictions",
            "bootstrap cannot remove bias from repeated model selection on the same OOF data",
        ],
    }


def _published_path(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def publish_oof_diagnostics(
    output_dir: Path,
    *,
    project_root: Path,
    frames: Mapping[str, pd.DataFrame],
    json_outputs: Mapping[str, Mapping[str, object]],
    manifest: dict[str, object],
) -> Path:
    """Atomically publish diagnostics and refuse every existing destination."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite OOF diagnostics output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        outputs: dict[str, dict[str, object]] = {}
        for filename, frame in frames.items():
            path = staging / filename
            atomic_write_csv(path, frame)
            outputs[filename] = {
                "path": _published_path(output_dir / filename, root=project_root),
                "sha256": sha256_file(path),
                "rows": len(frame),
            }
        for filename, payload in json_outputs.items():
            path = staging / filename
            atomic_write_json(path, payload)
            outputs[filename] = {
                "path": _published_path(output_dir / filename, root=project_root),
                "sha256": sha256_file(path),
            }
        atomic_write_json(staging / "analysis_manifest.json", {**manifest, "outputs": outputs})
        staging.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def run_oof_diagnostics(spec: OOFDiagnosticsSpec, *, project_root: Path) -> Path:
    """Run all CPU-only diagnostics and atomically publish a manifest-closed report."""

    validated = load_validated_inputs(spec, project_root=project_root)
    candidates = evaluate_fixed_ablation(
        validated.frame,
        scores=spec.scores,
        search=spec.search,
        winner=spec.winner,
        selection=validated.selection,
    )
    ablation, folds, predictions, transitions, resources = build_ablation_outputs(
        candidates, benchmark=spec.runtime_benchmark
    )
    winner_predictions = candidates[-1].predictions
    threshold_analysis, threshold_summary = analyze_threshold_stability(
        winner_predictions, diagnostic=spec.threshold, winner=spec.winner
    )
    prior_simulation, prior_assumptions = simulate_class_priors(
        winner_predictions,
        diagnostic=spec.threshold,
        winner=spec.winner,
        priors=spec.class_priors,
        official_test_count=spec.official_test_count,
    )
    bootstrap = cluster_bootstrap(candidates, diagnostic=spec.threshold, bootstrap=spec.bootstrap)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "pre_official_oof_diagnostics",
        "status": "complete",
        "experiment": spec.experiment_name,
        "git": _git_snapshot(project_root),
        "implementation": _file_record(Path(__file__), project_root=project_root),
        "inputs": validated.input_records,
        "fixed_candidate_order": list(CANDIDATE_ORDER),
        "evaluation_contracts": {
            "candidates_1_to_6": "strict_leave_one_fold_out_weight_and_threshold_fit",
            "candidate_7": "frozen_full_oof_weight_and_threshold_no_refit",
            "threshold_oracles": "descriptive_only_not_for_selection",
            "class_prior_optima": "sensitivity_only_not_for_selection",
            "bootstrap_refit": "diagnostic_only_not_for_selection",
        },
        "frozen_winner": {
            "classifier_weight": spec.winner.classifier_weight,
            "thresholds": {
                "stable": spec.winner.stable_threshold,
                "recall": spec.winner.recall_threshold,
                "precision": spec.winner.precision_threshold,
            },
            "metrics": asdict(candidates[-1].metrics),
        },
        "runtime_benchmark": asdict(spec.runtime_benchmark),
        "leakage_controls": [
            "all candidate inputs are manifest-validated OOF scores",
            "candidates 1--6 fit parameters without each held-out Fold",
            "candidate 7 applies already frozen parameters without refitting",
            "group bootstrap samples complete group_id clusters",
            "no test images, training, pseudo-labels, or learned meta-models are used",
        ],
    }
    return publish_oof_diagnostics(
        spec.output_dir,
        project_root=project_root,
        frames={
            "ablation_report.csv": ablation,
            "ablation_fold_metrics.csv": folds,
            "ablation_predictions.csv": predictions,
            "ablation_transitions.csv": transitions,
            "threshold_analysis.csv": threshold_analysis,
            "class_prior_simulation.csv": prior_simulation,
        },
        json_outputs={
            "threshold_summary.json": threshold_summary,
            "bootstrap_confidence_interval.json": bootstrap,
            "class_prior_assumptions.json": prior_assumptions,
            "ablation_resource_report.json": resources,
        },
        manifest=manifest,
    )


__all__ = [
    "BootstrapSpec",
    "CANDIDATE_ORDER",
    "CandidateResult",
    "FrozenWinnerSpec",
    "OOFDiagnosticsSpec",
    "RuntimeBenchmarkSpec",
    "ScoreColumns",
    "SearchSpec",
    "ThresholdDiagnosticSpec",
    "analyze_threshold_stability",
    "build_ablation_outputs",
    "cluster_bootstrap",
    "evaluate_fixed_ablation",
    "load_oof_diagnostics_spec",
    "load_validated_inputs",
    "publish_oof_diagnostics",
    "run_oof_diagnostics",
    "simulate_class_priors",
]
