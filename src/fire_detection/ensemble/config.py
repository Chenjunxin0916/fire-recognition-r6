"""Strict runtime contract for OOF evaluation and weighted fusion."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from fire_detection.config import (
    ConfigError,
    load_yaml_mapping,
    require_bool,
    require_int,
    require_mapping,
    require_string,
    require_value,
    resolve_project_path,
)


@dataclass(frozen=True)
class EnsembleRuntimeConfig:
    """Validated parameters controlling OOF evaluation and fusion."""

    config_path: Path
    config_sha256: str
    experiment_name: str
    seed: int
    detector_oof: Path
    classifier_oof: Path
    folds: Path
    processed_manifest: Path
    oof_manifest: Path
    artifact_root: Path
    expected_count: int
    detector_score_column: str
    classifier_score_column: str
    weight_start: Decimal
    weight_end: Decimal
    weight_step: Decimal
    threshold_start: Decimal
    threshold_end: Decimal
    threshold_step: Decimal
    minimum_gain: float
    minimum_folds_not_worse: int
    maximum_fold_drop: float
    output_dir: Path


def _decimal(mapping: dict[str, object], key: str, *, context: str) -> Decimal:
    raw = require_value(mapping, key, context=context)
    if isinstance(raw, bool):
        raise ConfigError(f"{context}.{key} must be a finite decimal")
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise ConfigError(f"{context}.{key} must be a finite decimal") from exc
    if not value.is_finite():
        raise ConfigError(f"{context}.{key} must be finite")
    return value


def _finite_number(mapping: dict[str, object], key: str, *, context: str) -> float:
    raw = require_value(mapping, key, context=context)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"{context}.{key} must be numeric")
    value = float(raw)
    if not math.isfinite(value):
        raise ConfigError(f"{context}.{key} must be finite")
    return value


def load_ensemble_runtime_config(path: Path, *, project_root: Path) -> EnsembleRuntimeConfig:
    """Load every required ensemble setting without silent algorithm defaults."""

    config_path = path if path.is_absolute() else project_root / path
    raw = load_yaml_mapping(config_path)
    experiment = require_mapping(raw, "experiment", context="ensemble")
    inputs = require_mapping(raw, "inputs", context="ensemble")
    method = require_mapping(raw, "method", context="ensemble")
    cross_validation = require_mapping(raw, "cross_validation", context="ensemble")
    promotion = require_mapping(raw, "promotion", context="ensemble")
    evaluation = require_mapping(raw, "evaluation", context="ensemble")

    if not require_bool(inputs, "require_oof", context="ensemble.inputs"):
        raise ConfigError("ensemble.inputs.require_oof must remain true")
    if require_string(method, "name", context="ensemble.method") != "weighted_average":
        raise ConfigError("ensemble.method.name must be weighted_average")
    if not require_bool(method, "search_weights", context="ensemble.method"):
        raise ConfigError("ensemble.method.search_weights must remain true")
    if require_string(cross_validation, "strategy", context="ensemble.cross_validation") != (
        "leave_one_fold_out"
    ):
        raise ConfigError("ensemble.cross_validation.strategy must be leave_one_fold_out")
    if not require_bool(
        cross_validation,
        "fit_weight_and_threshold_on_training_folds",
        context="ensemble.cross_validation",
    ):
        raise ConfigError("ensemble cross-validation must fit parameters on training folds")
    if require_string(evaluation, "level", context="ensemble.evaluation") != "image":
        raise ConfigError("ensemble.evaluation.level must be image")
    if require_string(evaluation, "primary_metric", context="ensemble.evaluation") != "f1":
        raise ConfigError("ensemble.evaluation.primary_metric must be f1")
    if require_string(evaluation, "tie_breaker", context="ensemble.evaluation") != "recall":
        raise ConfigError("ensemble.evaluation.tie_breaker must be recall")

    weight_start = _decimal(method, "classifier_weight_start", context="ensemble.method")
    weight_end = _decimal(method, "classifier_weight_end", context="ensemble.method")
    weight_step = _decimal(method, "weight_step", context="ensemble.method")
    threshold_start = _decimal(evaluation, "threshold_start", context="ensemble.evaluation")
    threshold_end = _decimal(evaluation, "threshold_end", context="ensemble.evaluation")
    threshold_step = _decimal(evaluation, "threshold_step", context="ensemble.evaluation")
    if not Decimal(0) <= weight_start <= weight_end <= Decimal(1):
        raise ConfigError("ensemble classifier-weight bounds must satisfy 0 <= start <= end <= 1")
    if weight_step <= 0:
        raise ConfigError("ensemble.method.weight_step must be greater than zero")
    if not Decimal(0) <= threshold_start <= threshold_end <= Decimal(1):
        raise ConfigError("ensemble threshold bounds must satisfy 0 <= start <= end <= 1")
    if threshold_step <= 0:
        raise ConfigError("ensemble.evaluation.threshold_step must be greater than zero")

    minimum_gain = _finite_number(promotion, "minimum_f1_gain", context="ensemble.promotion")
    minimum_folds = require_int(
        promotion,
        "minimum_folds_not_worse",
        context="ensemble.promotion",
        minimum=0,
    )
    maximum_drop = _finite_number(
        promotion,
        "maximum_single_fold_drop",
        context="ensemble.promotion",
    )
    if minimum_gain < 0:
        raise ConfigError("ensemble.promotion.minimum_f1_gain must be non-negative")
    if minimum_folds > 5:
        raise ConfigError("ensemble.promotion.minimum_folds_not_worse must be <= 5")
    if maximum_drop < 0:
        raise ConfigError("ensemble.promotion.maximum_single_fold_drop must be non-negative")

    output_raw = require_string(evaluation, "output_dir", context="ensemble.evaluation")
    return EnsembleRuntimeConfig(
        config_path=config_path,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        experiment_name=require_string(experiment, "name", context="ensemble.experiment"),
        seed=require_int(experiment, "seed", context="ensemble.experiment", minimum=0),
        detector_oof=resolve_project_path(
            project_root,
            require_string(inputs, "detector_oof", context="ensemble.inputs"),
        ),
        classifier_oof=resolve_project_path(
            project_root,
            require_string(inputs, "classifier_oof", context="ensemble.inputs"),
        ),
        folds=resolve_project_path(
            project_root,
            require_string(inputs, "folds", context="ensemble.inputs"),
        ),
        processed_manifest=resolve_project_path(
            project_root,
            require_string(inputs, "processed_manifest", context="ensemble.inputs"),
        ),
        oof_manifest=resolve_project_path(
            project_root,
            require_string(inputs, "oof_manifest", context="ensemble.inputs"),
        ),
        artifact_root=resolve_project_path(
            project_root,
            require_string(inputs, "artifact_root", context="ensemble.inputs"),
        ),
        expected_count=require_int(
            inputs,
            "expected_count",
            context="ensemble.inputs",
            minimum=1,
        ),
        detector_score_column=require_string(
            method, "detector_score_column", context="ensemble.method"
        ),
        classifier_score_column=require_string(
            method, "classifier_score_column", context="ensemble.method"
        ),
        weight_start=weight_start,
        weight_end=weight_end,
        weight_step=weight_step,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        threshold_step=threshold_step,
        minimum_gain=minimum_gain,
        minimum_folds_not_worse=minimum_folds,
        maximum_fold_drop=maximum_drop,
        output_dir=resolve_project_path(project_root, output_raw),
    )
