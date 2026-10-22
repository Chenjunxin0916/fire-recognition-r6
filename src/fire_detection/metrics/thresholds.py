"""Deterministic threshold grids and competition-aligned threshold search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from fire_detection.metrics.binary import (
    BinaryMetrics,
    compute_binary_metrics,
    validate_binary_inputs,
)


@dataclass(frozen=True)
class ThresholdPoint:
    """Metrics at one threshold-grid point."""

    threshold: float
    metrics: BinaryMetrics


@dataclass(frozen=True)
class ThresholdSearchResult:
    """Best threshold and the complete threshold curve."""

    best: ThresholdPoint
    curve: tuple[ThresholdPoint, ...]


def _decimal(value: float | str | Decimal, *, name: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal value") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def build_threshold_grid(
    start: float | str | Decimal = Decimal("0.01"),
    end: float | str | Decimal = Decimal("0.99"),
    step: float | str | Decimal = Decimal("0.005"),
) -> tuple[Decimal, ...]:
    """Build an endpoint-inclusive grid without binary floating-point drift."""

    start_value = _decimal(start, name="start")
    end_value = _decimal(end, name="end")
    step_value = _decimal(step, name="step")
    if not Decimal(0) <= start_value <= Decimal(1):
        raise ValueError("start must be in [0, 1]")
    if not Decimal(0) <= end_value <= Decimal(1):
        raise ValueError("end must be in [0, 1]")
    if end_value < start_value:
        raise ValueError("end must be greater than or equal to start")
    if step_value <= 0:
        raise ValueError("step must be greater than zero")

    values: list[Decimal] = []
    value = start_value
    while value <= end_value:
        values.append(value)
        value += step_value
    if values[-1] != end_value:
        values.append(end_value)
    return tuple(values)


def _f1_fraction(metrics: BinaryMetrics) -> Fraction:
    denominator = 2 * metrics.tp + metrics.fp + metrics.fn
    return Fraction(2 * metrics.tp, denominator) if denominator else Fraction(0, 1)


def _recall_fraction(metrics: BinaryMetrics) -> Fraction:
    denominator = metrics.tp + metrics.fn
    return Fraction(metrics.tp, denominator) if denominator else Fraction(0, 1)


def _lower_middle(values: Sequence[ThresholdPoint]) -> ThresholdPoint:
    ordered = sorted(values, key=lambda point: point.threshold)
    return ordered[(len(ordered) - 1) // 2]


def search_thresholds(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    start: float | str | Decimal = Decimal("0.01"),
    end: float | str | Decimal = Decimal("0.99"),
    step: float | str | Decimal = Decimal("0.005"),
) -> ThresholdSearchResult:
    """Search thresholds by F1, Recall, then the lower center of a prediction plateau."""

    labels, scores = validate_binary_inputs(y_true, y_score)
    grid = build_threshold_grid(start=start, end=end, step=step)
    points: list[ThresholdPoint] = []
    predictions: dict[float, tuple[int, ...]] = {}
    for threshold in grid:
        threshold_float = float(threshold)
        metrics = compute_binary_metrics(labels, scores, threshold_float)
        point = ThresholdPoint(threshold=threshold_float, metrics=metrics)
        points.append(point)
        predictions[threshold_float] = tuple(int(score >= threshold_float) for score in scores)

    best_f1 = max(_f1_fraction(point.metrics) for point in points)
    f1_candidates = [point for point in points if _f1_fraction(point.metrics) == best_f1]
    best_recall = max(_recall_fraction(point.metrics) for point in f1_candidates)
    candidates = [
        point for point in f1_candidates if _recall_fraction(point.metrics) == best_recall
    ]

    plateaus: dict[tuple[int, ...], list[ThresholdPoint]] = {}
    for point in candidates:
        plateaus.setdefault(predictions[point.threshold], []).append(point)
    centers = [_lower_middle(plateau) for plateau in plateaus.values()]
    best = min(centers, key=lambda point: point.threshold)
    return ThresholdSearchResult(best=best, curve=tuple(points))
