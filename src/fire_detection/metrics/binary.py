"""Strict image-level binary classification metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real


@dataclass(frozen=True)
class BinaryMetrics:
    """Image-level binary classification metrics and confusion counts."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


def validate_binary_inputs(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Validate labels and probabilities and return immutable normalized values."""

    if len(y_true) == 0:
        raise ValueError("y_true and y_score must not be empty")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")

    labels: list[int] = []
    for index, label_value in enumerate(y_true):
        if isinstance(label_value, bool) or not isinstance(label_value, Integral):
            raise TypeError(f"y_true[{index}] must be a non-boolean integer 0 or 1")
        label = int(label_value)
        if label not in {0, 1}:
            raise ValueError(f"y_true[{index}] must be 0 or 1, found {label}")
        labels.append(label)

    scores: list[float] = []
    for index, score_value in enumerate(y_score):
        if isinstance(score_value, bool) or not isinstance(score_value, Real):
            raise TypeError(f"y_score[{index}] must be a non-boolean real number")
        score = float(score_value)
        if not math.isfinite(score):
            raise ValueError(f"y_score[{index}] must be finite")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"y_score[{index}] must be in [0, 1], found {score}")
        scores.append(score)

    return tuple(labels), tuple(scores)


def validate_threshold(threshold: float) -> float:
    """Validate and normalize one probability threshold."""

    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a non-boolean real number")
    normalized = float(threshold)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    return normalized


def compute_binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
) -> BinaryMetrics:
    """Compute image-level metrics using the rule ``score >= threshold``."""

    labels, scores = validate_binary_inputs(y_true, y_score)
    normalized_threshold = validate_threshold(threshold)

    tp = fp = fn = tn = 0
    for label, score in zip(labels, scores, strict=True):
        prediction = int(score >= normalized_threshold)
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )
