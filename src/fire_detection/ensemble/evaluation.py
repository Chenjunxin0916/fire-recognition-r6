"""Leakage-aware single-model and weighted-fusion OOF evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

import pandas as pd

from fire_detection.metrics import (
    BinaryMetrics,
    ThresholdSearchResult,
    compute_binary_metrics,
    search_thresholds,
)
from fire_detection.metrics.thresholds import build_threshold_grid


@dataclass(frozen=True)
class FoldEvaluation:
    """Held-out Fold metrics with parameters selected on the other Folds."""

    fold: int
    threshold: float
    metrics: BinaryMetrics
    classifier_weight: float | None = None


@dataclass(frozen=True)
class CrossFittedEvaluation:
    """Cross-fitted aggregate metrics, Fold parameters, and row predictions."""

    metrics: BinaryMetrics
    per_fold: tuple[FoldEvaluation, ...]
    predictions: pd.DataFrame


@dataclass(frozen=True)
class ModelEvaluation:
    """Descriptive pooled fit plus trustworthy cross-fitted evaluation."""

    pooled: ThresholdSearchResult
    cross_fitted: CrossFittedEvaluation


@dataclass(frozen=True)
class FusionEvaluation:
    """Full-OOF fusion parameters and held-out cross-fitted evaluation."""

    pooled_weight: float
    pooled: ThresholdSearchResult
    cross_fitted: CrossFittedEvaluation


@dataclass(frozen=True)
class PromotionDecision:
    """Whether fusion satisfies the fixed promotion gates."""

    promoted: bool
    f1_gain: float
    folds_not_worse: int
    worst_fold_drop: float


CandidateEvaluation = ModelEvaluation | FusionEvaluation


def _metric_fractions(metrics: BinaryMetrics) -> tuple[Fraction, Fraction]:
    f1_denominator = 2 * metrics.tp + metrics.fp + metrics.fn
    recall_denominator = metrics.tp + metrics.fn
    f1 = Fraction(2 * metrics.tp, f1_denominator) if f1_denominator else Fraction(0, 1)
    recall = Fraction(metrics.tp, recall_denominator) if recall_denominator else Fraction(0, 1)
    return f1, recall


def _validate_evaluation_frame(frame: pd.DataFrame, score_columns: Sequence[str]) -> pd.DataFrame:
    required = {"filename", "label", "fold", *score_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"evaluation frame missing columns: {missing}")
    if frame.empty:
        raise ValueError("evaluation frame must not be empty")
    if frame["filename"].duplicated().any():
        raise ValueError("evaluation frame contains duplicate filenames")
    normalized = frame.copy().reset_index(drop=True)
    folds = sorted(int(value) for value in normalized["fold"].unique())
    if len(folds) < 2:
        raise ValueError("cross-fitted evaluation requires at least two Folds")
    for column in score_columns:
        values = pd.to_numeric(normalized[column], errors="raise").astype(float)
        if not values.between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"{column} must contain probabilities in [0, 1]")
        normalized[column] = values
    return normalized


def _confusion_name(label: int, prediction: int) -> str:
    return {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}[(label, prediction)]


def evaluate_single_model(
    frame: pd.DataFrame,
    *,
    score_column: str,
    threshold_start: Decimal = Decimal("0.01"),
    threshold_end: Decimal = Decimal("0.99"),
    threshold_step: Decimal = Decimal("0.005"),
) -> ModelEvaluation:
    """Evaluate one score using pooled and leave-one-fold-out threshold fitting."""

    data = _validate_evaluation_frame(frame, [score_column])
    pooled = search_thresholds(
        data["label"].tolist(),
        data[score_column].tolist(),
        start=threshold_start,
        end=threshold_end,
        step=threshold_step,
    )
    rows: list[pd.DataFrame] = []
    per_fold: list[FoldEvaluation] = []
    for fold in sorted(int(value) for value in data["fold"].unique()):
        fit = data[data["fold"] != fold]
        held = data[data["fold"] == fold].copy()
        if fit.empty or held.empty:
            raise ValueError(f"Fold {fold} has an empty fit or held-out partition")
        result = search_thresholds(
            fit["label"].tolist(),
            fit[score_column].tolist(),
            start=threshold_start,
            end=threshold_end,
            step=threshold_step,
        )
        threshold = result.best.threshold
        held["score"] = held[score_column]
        held["threshold"] = threshold
        held["y_pred"] = (held["score"] >= threshold).astype(int)
        held["margin"] = held["score"] - threshold
        held["confusion"] = [
            _confusion_name(int(label), int(prediction))
            for label, prediction in zip(held["label"], held["y_pred"], strict=True)
        ]
        metrics = compute_binary_metrics(held["label"].tolist(), held["y_pred"].tolist(), 0.5)
        per_fold.append(FoldEvaluation(fold=fold, threshold=threshold, metrics=metrics))
        rows.append(held)

    predictions = pd.concat(rows, ignore_index=True).sort_values("filename", kind="stable")
    aggregate = compute_binary_metrics(
        predictions["label"].tolist(), predictions["y_pred"].tolist(), 0.5
    )
    return ModelEvaluation(
        pooled=pooled,
        cross_fitted=CrossFittedEvaluation(
            metrics=aggregate,
            per_fold=tuple(per_fold),
            predictions=predictions.reset_index(drop=True),
        ),
    )


def _weight_grid(
    start: Decimal = Decimal("0"),
    end: Decimal = Decimal("1"),
    step: Decimal = Decimal("0.05"),
) -> tuple[Decimal, ...]:
    return build_threshold_grid(start, end, step)


def _select_weight_and_threshold(
    labels: Sequence[int],
    detector_scores: Sequence[float],
    classifier_scores: Sequence[float],
    *,
    weight_start: Decimal,
    weight_end: Decimal,
    weight_step: Decimal,
    threshold_start: Decimal,
    threshold_end: Decimal,
    threshold_step: Decimal,
) -> tuple[float, ThresholdSearchResult]:
    candidates: list[tuple[float, ThresholdSearchResult, tuple[int, ...]]] = []
    for decimal_weight in _weight_grid(weight_start, weight_end, weight_step):
        weight = float(decimal_weight)
        scores = [
            (1.0 - weight) * detector + weight * classifier
            for detector, classifier in zip(detector_scores, classifier_scores, strict=True)
        ]
        result = search_thresholds(
            labels,
            scores,
            start=threshold_start,
            end=threshold_end,
            step=threshold_step,
        )
        predictions = tuple(int(score >= result.best.threshold) for score in scores)
        candidates.append((weight, result, predictions))

    best_key = max(_metric_fractions(candidate[1].best.metrics) for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if _metric_fractions(candidate[1].best.metrics) == best_key
    ]
    plateaus: list[list[tuple[float, ThresholdSearchResult, tuple[int, ...]]]] = []
    for candidate in sorted(tied, key=lambda item: item[0]):
        if (
            not plateaus
            or plateaus[-1][-1][2] != candidate[2]
            or candidate[0] - plateaus[-1][-1][0] > float(weight_step) + 1e-12
        ):
            plateaus.append([candidate])
        else:
            plateaus[-1].append(candidate)
    centers = [
        sorted(plateau, key=lambda item: item[0])[(len(plateau) - 1) // 2] for plateau in plateaus
    ]
    selected = min(centers, key=lambda item: (abs(item[0] - 0.5), item[0]))
    return selected[0], selected[1]


def evaluate_weighted_fusion(
    frame: pd.DataFrame,
    *,
    detector_column: str = "det_max_conf",
    classifier_column: str = "cls_probability",
    weight_start: Decimal = Decimal("0"),
    weight_end: Decimal = Decimal("1"),
    weight_step: Decimal = Decimal("0.05"),
    threshold_start: Decimal = Decimal("0.01"),
    threshold_end: Decimal = Decimal("0.99"),
    threshold_step: Decimal = Decimal("0.005"),
) -> FusionEvaluation:
    """Fit detector/classifier weight and threshold without using each held-out Fold."""

    data = _validate_evaluation_frame(frame, [detector_column, classifier_column])
    pooled_weight, pooled = _select_weight_and_threshold(
        data["label"].tolist(),
        data[detector_column].tolist(),
        data[classifier_column].tolist(),
        weight_start=weight_start,
        weight_end=weight_end,
        weight_step=weight_step,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        threshold_step=threshold_step,
    )
    rows: list[pd.DataFrame] = []
    per_fold: list[FoldEvaluation] = []
    for fold in sorted(int(value) for value in data["fold"].unique()):
        fit = data[data["fold"] != fold]
        held = data[data["fold"] == fold].copy()
        weight, result = _select_weight_and_threshold(
            fit["label"].tolist(),
            fit[detector_column].tolist(),
            fit[classifier_column].tolist(),
            weight_start=weight_start,
            weight_end=weight_end,
            weight_step=weight_step,
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            threshold_step=threshold_step,
        )
        threshold = result.best.threshold
        held["score"] = (1.0 - weight) * held[detector_column] + weight * held[classifier_column]
        held["threshold"] = threshold
        held["classifier_weight"] = weight
        held["y_pred"] = (held["score"] >= threshold).astype(int)
        held["margin"] = held["score"] - threshold
        held["confusion"] = [
            _confusion_name(int(label), int(prediction))
            for label, prediction in zip(held["label"], held["y_pred"], strict=True)
        ]
        metrics = compute_binary_metrics(held["label"].tolist(), held["y_pred"].tolist(), 0.5)
        per_fold.append(
            FoldEvaluation(
                fold=fold,
                threshold=threshold,
                classifier_weight=weight,
                metrics=metrics,
            )
        )
        rows.append(held)

    predictions = pd.concat(rows, ignore_index=True).sort_values("filename", kind="stable")
    aggregate = compute_binary_metrics(
        predictions["label"].tolist(), predictions["y_pred"].tolist(), 0.5
    )
    return FusionEvaluation(
        pooled_weight=pooled_weight,
        pooled=pooled,
        cross_fitted=CrossFittedEvaluation(
            metrics=aggregate,
            per_fold=tuple(per_fold),
            predictions=predictions.reset_index(drop=True),
        ),
    )


def decide_candidate_promotion(
    candidate: CandidateEvaluation,
    baseline: CandidateEvaluation,
    *,
    minimum_gain: float = 0.005,
    minimum_folds_not_worse: int = 3,
    maximum_fold_drop: float = 0.02,
) -> PromotionDecision:
    """Compare any cross-fitted candidate with a baseline using fixed gates."""

    baseline_folds = {point.fold: point.metrics.f1 for point in baseline.cross_fitted.per_fold}
    candidate_folds = {point.fold: point.metrics.f1 for point in candidate.cross_fitted.per_fold}
    if baseline_folds.keys() != candidate_folds.keys():
        raise ValueError("candidate and baseline Fold sets differ")
    deltas = [candidate_folds[fold] - baseline_folds[fold] for fold in sorted(baseline_folds)]
    f1_gain = candidate.cross_fitted.metrics.f1 - baseline.cross_fitted.metrics.f1
    folds_not_worse = sum(delta >= 0.0 for delta in deltas)
    worst_fold_drop = min(deltas)
    promoted = (
        f1_gain >= minimum_gain
        and folds_not_worse >= minimum_folds_not_worse
        and worst_fold_drop >= -maximum_fold_drop
    )
    return PromotionDecision(
        promoted=promoted,
        f1_gain=f1_gain,
        folds_not_worse=folds_not_worse,
        worst_fold_drop=worst_fold_drop,
    )


def decide_fusion_promotion(
    fusion: FusionEvaluation,
    baseline: ModelEvaluation,
    *,
    minimum_gain: float = 0.005,
    minimum_folds_not_worse: int = 3,
    maximum_fold_drop: float = 0.02,
) -> PromotionDecision:
    """Backward-compatible fusion-specific wrapper around candidate promotion."""

    return decide_candidate_promotion(
        fusion,
        baseline,
        minimum_gain=minimum_gain,
        minimum_folds_not_worse=minimum_folds_not_worse,
        maximum_fold_drop=maximum_fold_drop,
    )
