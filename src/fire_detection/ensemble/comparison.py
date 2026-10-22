"""Fixed OOF candidate matrices and leakage-free comparison reports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd

from fire_detection.ensemble.evaluation import (
    FusionEvaluation,
    ModelEvaluation,
    PromotionDecision,
    decide_candidate_promotion,
    evaluate_single_model,
    evaluate_weighted_fusion,
)
from fire_detection.inference.oof import canonical_filename
from fire_detection.metrics import BinaryMetrics, compute_binary_metrics

Evaluation = ModelEvaluation | FusionEvaluation
CandidateMethod = Literal["single_detector", "single_classifier", "weighted_average"]

SMALL_FIRE_COLUMNS = ("contains_small_fire", "all_boxes_small")
VALID_TRANSITIONS = (
    ("TP", "TP"),
    ("TP", "FN"),
    ("FN", "TP"),
    ("FN", "FN"),
    ("TN", "TN"),
    ("TN", "FP"),
    ("FP", "TN"),
    ("FP", "FP"),
)


@dataclass(frozen=True)
class ScoreSource:
    """One named, already provenance-validated OOF score source."""

    name: str
    frame: pd.DataFrame
    score_column: str


@dataclass(frozen=True)
class MatrixCandidate:
    """One single-model or detector/classifier candidate in a fixed matrix."""

    name: str
    method: CandidateMethod
    evaluation: Evaluation
    detector_name: str | None = None
    classifier_name: str | None = None

    @property
    def component_count(self) -> int:
        """Return the number of model components needed for inference."""

        return int(self.detector_name is not None) + int(self.classifier_name is not None)


@dataclass(frozen=True)
class CandidateMatrix:
    """Immutable set of candidates expanded before any result is inspected."""

    candidates: tuple[MatrixCandidate, ...]

    def get(self, name: str) -> MatrixCandidate:
        """Return one uniquely named matrix candidate."""

        matches = [candidate for candidate in self.candidates if candidate.name == name]
        if len(matches) != 1:
            raise KeyError(f"candidate {name!r} does not exist uniquely in the fixed matrix")
        return matches[0]


@dataclass(frozen=True)
class CandidateComparison:
    """Candidate-versus-baseline metrics, diagnostics, and aligned predictions."""

    decision: PromotionDecision
    overall_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    transitions: pd.DataFrame
    small_fire_metrics: pd.DataFrame
    aligned_predictions: pd.DataFrame


def _validate_source_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("OOF source name must be a non-empty string")
    normalized = name.strip()
    if any(separator in normalized for separator in (":", "+")):
        raise ValueError("OOF source names must not contain ':' or '+'")
    return normalized


def _integer_values(series: pd.Series, *, name: str, binary: bool = False) -> pd.Series:
    normalized: list[int] = []
    for index, value in series.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name}[{index}] must be a non-boolean integer")
        integer = int(value)
        if integer < 0 or (binary and integer not in {0, 1}):
            expected = "0 or 1" if binary else "non-negative"
            raise ValueError(f"{name}[{index}] must be {expected}")
        normalized.append(integer)
    return pd.Series(normalized, index=series.index, dtype="int64")


def _normalize_source(source: ScoreSource) -> ScoreSource:
    name = _validate_source_name(source.name)
    if not isinstance(source.score_column, str) or not source.score_column.strip():
        raise ValueError(f"OOF source {name!r} score column must be a non-empty string")
    if source.score_column in {"filename", "label", "fold"}:
        raise ValueError(f"OOF source {name!r} score column conflicts with identity columns")
    required = {"filename", "label", "fold", source.score_column}
    missing = sorted(required.difference(source.frame.columns))
    if missing:
        raise ValueError(f"OOF source {name!r} missing columns: {missing}")
    if source.frame.empty:
        raise ValueError(f"OOF source {name!r} must not be empty")
    frame = source.frame.copy().reset_index(drop=True)
    frame["filename"] = [canonical_filename(value) for value in frame["filename"]]
    if frame["filename"].duplicated().any():
        raise ValueError(f"OOF source {name!r} contains duplicate filenames")
    frame["label"] = _integer_values(frame["label"], name=f"{name}.label", binary=True)
    frame["fold"] = _integer_values(frame["fold"], name=f"{name}.fold")
    return ScoreSource(name=name, frame=frame, score_column=source.score_column)


def _normalize_sources(sources: Sequence[ScoreSource], *, kind: str) -> tuple[ScoreSource, ...]:
    if not sources:
        raise ValueError(f"fixed matrix requires at least one {kind} source")
    normalized = tuple(_normalize_source(source) for source in sources)
    names = [source.name for source in normalized]
    if len(names) != len(set(names)):
        raise ValueError(f"fixed matrix contains duplicate {kind} source names")
    return tuple(sorted(normalized, key=lambda source: source.name))


def _align_sources(detector: ScoreSource, classifier: ScoreSource) -> pd.DataFrame:
    detector_columns = ["filename", "label", "fold", detector.score_column]
    classifier_columns = ["filename", "label", "fold", classifier.score_column]
    det = detector.frame[detector_columns].rename(
        columns={
            "label": "detector_label",
            "fold": "detector_fold",
            detector.score_column: "detector_score",
        }
    )
    cls = classifier.frame[classifier_columns].rename(
        columns={
            "label": "classifier_label",
            "fold": "classifier_fold",
            classifier.score_column: "classifier_score",
        }
    )
    aligned = det.merge(cls, on="filename", how="outer", indicator=True, validate="one_to_one")
    if (aligned["_merge"] != "both").any():
        raise ValueError(
            f"detector {detector.name!r} and classifier {classifier.name!r} filename sets differ"
        )
    if not (aligned["detector_label"] == aligned["classifier_label"]).all():
        raise ValueError(
            f"detector {detector.name!r} and classifier {classifier.name!r} labels differ"
        )
    if not (aligned["detector_fold"] == aligned["classifier_fold"]).all():
        raise ValueError(
            f"detector {detector.name!r} and classifier {classifier.name!r} Folds differ"
        )
    return pd.DataFrame(
        {
            "filename": aligned["filename"],
            "label": aligned["detector_label"].astype(int),
            "fold": aligned["detector_fold"].astype(int),
            "detector_score": aligned["detector_score"].astype(float),
            "classifier_score": aligned["classifier_score"].astype(float),
        }
    ).sort_values("filename", kind="stable")


def evaluate_fixed_candidate_matrix(
    detectors: Sequence[ScoreSource],
    classifiers: Sequence[ScoreSource],
    *,
    weight_start: Decimal = Decimal("0"),
    weight_end: Decimal = Decimal("1"),
    weight_step: Decimal = Decimal("0.05"),
    threshold_start: Decimal = Decimal("0.01"),
    threshold_end: Decimal = Decimal("0.99"),
    threshold_step: Decimal = Decimal("0.005"),
) -> CandidateMatrix:
    """Evaluate all declared singles and the full detector-by-classifier matrix.

    The complete candidate list is constructed before evaluation begins.  Every
    threshold and weighted-average parameter is fitted with leave-one-fold-out data;
    pooled parameters remain descriptive/final-test fitting only.
    """

    detector_sources = _normalize_sources(detectors, kind="detector")
    classifier_sources = _normalize_sources(classifiers, kind="classifier")
    candidate_specs: list[tuple[CandidateMethod, ScoreSource, ScoreSource | None]] = []
    candidate_specs.extend(("single_detector", source, None) for source in detector_sources)
    candidate_specs.extend(("single_classifier", source, None) for source in classifier_sources)
    candidate_specs.extend(
        ("weighted_average", detector, classifier)
        for detector in detector_sources
        for classifier in classifier_sources
    )

    candidates: list[MatrixCandidate] = []
    for method, first, second in candidate_specs:
        if method == "single_detector":
            evaluation: Evaluation = evaluate_single_model(
                first.frame,
                score_column=first.score_column,
                threshold_start=threshold_start,
                threshold_end=threshold_end,
                threshold_step=threshold_step,
            )
            candidates.append(
                MatrixCandidate(
                    name=f"detector:{first.name}",
                    method=method,
                    detector_name=first.name,
                    evaluation=evaluation,
                )
            )
        elif method == "single_classifier":
            evaluation = evaluate_single_model(
                first.frame,
                score_column=first.score_column,
                threshold_start=threshold_start,
                threshold_end=threshold_end,
                threshold_step=threshold_step,
            )
            candidates.append(
                MatrixCandidate(
                    name=f"classifier:{first.name}",
                    method=method,
                    classifier_name=first.name,
                    evaluation=evaluation,
                )
            )
        else:
            assert second is not None
            combined = _align_sources(first, second)
            evaluation = evaluate_weighted_fusion(
                combined,
                detector_column="detector_score",
                classifier_column="classifier_score",
                weight_start=weight_start,
                weight_end=weight_end,
                weight_step=weight_step,
                threshold_start=threshold_start,
                threshold_end=threshold_end,
                threshold_step=threshold_step,
            )
            candidates.append(
                MatrixCandidate(
                    name=f"fusion:{first.name}+{second.name}",
                    method=method,
                    detector_name=first.name,
                    classifier_name=second.name,
                    evaluation=evaluation,
                )
            )
    return CandidateMatrix(candidates=tuple(candidates))


def _confusion(label: int, prediction: int) -> str:
    return {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}[(label, prediction)]


def _normalize_predictions(candidate: MatrixCandidate) -> pd.DataFrame:
    frame = candidate.evaluation.cross_fitted.predictions.copy().reset_index(drop=True)
    required = {"filename", "label", "fold", "score", "threshold", "y_pred"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate {candidate.name!r} predictions missing columns: {missing}")
    frame["filename"] = [canonical_filename(value) for value in frame["filename"]]
    if frame["filename"].duplicated().any():
        raise ValueError(f"candidate {candidate.name!r} predictions contain duplicate filenames")
    frame["label"] = _integer_values(frame["label"], name=f"{candidate.name}.label", binary=True)
    frame["fold"] = _integer_values(frame["fold"], name=f"{candidate.name}.fold")
    frame["y_pred"] = _integer_values(frame["y_pred"], name=f"{candidate.name}.y_pred", binary=True)
    frame["confusion"] = [
        _confusion(int(label), int(prediction))
        for label, prediction in zip(frame["label"], frame["y_pred"], strict=True)
    ]
    return frame.sort_values("filename", kind="stable").reset_index(drop=True)


def _metrics_from_predictions(frame: pd.DataFrame) -> BinaryMetrics:
    return compute_binary_metrics(frame["label"].tolist(), frame["y_pred"].tolist(), 0.5)


def _verify_recorded_metrics(candidate: MatrixCandidate, frame: pd.DataFrame) -> None:
    recorded = candidate.evaluation.cross_fitted.metrics
    recomputed = _metrics_from_predictions(frame)
    if (recorded.tp, recorded.fp, recorded.fn, recorded.tn) != (
        recomputed.tp,
        recomputed.fp,
        recomputed.fn,
        recomputed.tn,
    ):
        raise ValueError(f"candidate {candidate.name!r} recorded metrics do not match predictions")
    recorded_folds = {
        point.fold: point.metrics for point in candidate.evaluation.cross_fitted.per_fold
    }
    actual_folds = sorted(int(value) for value in frame["fold"].unique())
    if set(recorded_folds) != set(actual_folds):
        raise ValueError(
            f"candidate {candidate.name!r} recorded Fold set does not match predictions"
        )
    for fold in actual_folds:
        actual = _metrics_from_predictions(frame[frame["fold"] == fold])
        expected = recorded_folds[fold]
        if (expected.tp, expected.fp, expected.fn, expected.tn) != (
            actual.tp,
            actual.fp,
            actual.fn,
            actual.tn,
        ):
            raise ValueError(
                f"candidate {candidate.name!r} Fold {fold} metrics do not match predictions"
            )


def _align_predictions(
    baseline: MatrixCandidate,
    candidate: MatrixCandidate,
) -> pd.DataFrame:
    baseline_frame = _normalize_predictions(baseline)
    candidate_frame = _normalize_predictions(candidate)
    _verify_recorded_metrics(baseline, baseline_frame)
    _verify_recorded_metrics(candidate, candidate_frame)
    prediction_columns = [
        "filename",
        "label",
        "fold",
        "score",
        "threshold",
        "y_pred",
        "confusion",
    ]
    baseline_columns = baseline_frame[prediction_columns].rename(
        columns={
            column: f"baseline_{column}" for column in prediction_columns if column != "filename"
        }
    )
    candidate_columns = candidate_frame[prediction_columns].rename(
        columns={
            column: f"candidate_{column}" for column in prediction_columns if column != "filename"
        }
    )
    aligned = baseline_columns.merge(
        candidate_columns,
        on="filename",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if (aligned["_merge"] != "both").any():
        raise ValueError("candidate and baseline prediction filename sets differ")
    for column in ("label", "fold"):
        if not (aligned[f"baseline_{column}"] == aligned[f"candidate_{column}"]).all():
            raise ValueError(f"candidate and baseline prediction {column} values differ")
    aligned = aligned.drop(columns="_merge")
    aligned["label"] = aligned.pop("baseline_label").astype(int)
    aligned["fold"] = aligned.pop("baseline_fold").astype(int)
    aligned = aligned.drop(columns=["candidate_label", "candidate_fold"])
    aligned["transition"] = aligned["baseline_confusion"] + "->" + aligned["candidate_confusion"]
    return aligned.sort_values("filename", kind="stable").reset_index(drop=True)


def _metric_row(model: str, metrics: BinaryMetrics) -> dict[str, object]:
    return {"model": model, **asdict(metrics)}


def _fold_metric_rows(
    baseline_name: str,
    candidate_name: str,
    aligned: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in sorted(int(value) for value in aligned["fold"].unique()):
        held = aligned[aligned["fold"] == fold]
        baseline_frame = pd.DataFrame({"label": held["label"], "y_pred": held["baseline_y_pred"]})
        candidate_frame = pd.DataFrame({"label": held["label"], "y_pred": held["candidate_y_pred"]})
        baseline_metrics = _metrics_from_predictions(baseline_frame)
        candidate_metrics = _metrics_from_predictions(candidate_frame)
        row: dict[str, object] = {
            "fold": fold,
            "baseline": baseline_name,
            "candidate": candidate_name,
        }
        for key, value in asdict(baseline_metrics).items():
            row[f"baseline_{key}"] = value
        for key, value in asdict(candidate_metrics).items():
            row[f"candidate_{key}"] = value
        row["f1_delta"] = candidate_metrics.f1 - baseline_metrics.f1
        row["recall_delta"] = candidate_metrics.recall - baseline_metrics.recall
        rows.append(row)
    return pd.DataFrame(rows)


def _transition_rows(aligned: pd.DataFrame) -> pd.DataFrame:
    counts = aligned["transition"].value_counts().to_dict()
    rows = [
        {
            "baseline_confusion": baseline,
            "candidate_confusion": candidate,
            "transition": f"{baseline}->{candidate}",
            "count": int(counts.get(f"{baseline}->{candidate}", 0)),
        }
        for baseline, candidate in VALID_TRANSITIONS
    ]
    return pd.DataFrame(rows)


def _normalize_gt_summary(summary: pd.DataFrame) -> pd.DataFrame:
    normalized = summary.copy()
    if "all_boxes_small" not in normalized and "all_gt_boxes_small" in normalized:
        normalized["all_boxes_small"] = normalized["all_gt_boxes_small"]
    required = {"filename", *SMALL_FIRE_COLUMNS}
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ValueError(f"GT box summary missing columns: {missing}")
    normalized["filename"] = [canonical_filename(value) for value in normalized["filename"]]
    if normalized["filename"].duplicated().any():
        raise ValueError("GT box summary contains duplicate filenames")
    for column in SMALL_FIRE_COLUMNS:
        if (
            normalized[column].isna().any()
            or not normalized[column].map(lambda value: isinstance(value, (bool, np.bool_))).all()
        ):
            raise ValueError(f"GT box summary {column} must contain only booleans")
        normalized[column] = normalized[column].astype(bool)
    return normalized[["filename", *SMALL_FIRE_COLUMNS]]


def _optional_metric_values(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
    return asdict(_metrics_from_predictions(frame))


def _small_fire_metric_rows(
    aligned: pd.DataFrame,
    *,
    baseline_name: str,
    candidate_name: str,
) -> pd.DataFrame:
    folds = sorted(int(value) for value in aligned["fold"].unique())
    rows: list[dict[str, object]] = []
    for criterion in SMALL_FIRE_COLUMNS:
        eligible = aligned[(aligned["label"] == 1) & aligned[criterion]]
        for model_name, prediction_column in (
            (baseline_name, "baseline_y_pred"),
            (candidate_name, "candidate_y_pred"),
        ):
            for fold in ("all", *folds):
                subset = eligible if fold == "all" else eligible[eligible["fold"] == fold]
                metric_input = pd.DataFrame(
                    {"label": subset["label"], "y_pred": subset[prediction_column]}
                )
                rows.append(
                    {
                        "criterion": criterion,
                        "model": model_name,
                        "fold": fold,
                        "image_count": len(subset),
                        **_optional_metric_values(metric_input),
                    }
                )
    return pd.DataFrame(rows)


def compare_candidate_to_baseline(
    candidate: MatrixCandidate,
    baseline: MatrixCandidate,
    *,
    gt_box_summary: pd.DataFrame | None = None,
    minimum_gain: float = 0.005,
    minimum_folds_not_worse: int = 3,
    maximum_fold_drop: float = 0.02,
) -> CandidateComparison:
    """Compare two cross-fitted candidates without re-tuning on diagnostic subsets."""

    aligned = _align_predictions(baseline, candidate)
    baseline_metrics = _metrics_from_predictions(
        pd.DataFrame({"label": aligned["label"], "y_pred": aligned["baseline_y_pred"]})
    )
    candidate_metrics = _metrics_from_predictions(
        pd.DataFrame({"label": aligned["label"], "y_pred": aligned["candidate_y_pred"]})
    )
    overall = pd.DataFrame(
        [
            _metric_row(baseline.name, baseline_metrics),
            _metric_row(candidate.name, candidate_metrics),
        ]
    )
    decision = decide_candidate_promotion(
        candidate.evaluation,
        baseline.evaluation,
        minimum_gain=minimum_gain,
        minimum_folds_not_worse=minimum_folds_not_worse,
        maximum_fold_drop=maximum_fold_drop,
    )
    if gt_box_summary is None:
        small_fire_metrics = pd.DataFrame(
            columns=[
                "criterion",
                "model",
                "fold",
                "image_count",
                "precision",
                "recall",
                "f1",
                "tp",
                "fp",
                "fn",
                "tn",
            ]
        )
    else:
        summary = _normalize_gt_summary(gt_box_summary)
        aligned = aligned.merge(summary, on="filename", how="left", validate="one_to_one")
        if aligned[list(SMALL_FIRE_COLUMNS)].isna().any(axis=None):
            missing = aligned.loc[
                aligned[list(SMALL_FIRE_COLUMNS)].isna().any(axis=1), "filename"
            ].tolist()
            raise ValueError(f"GT box summary does not cover prediction files: {missing[:5]}")
        for column in SMALL_FIRE_COLUMNS:
            aligned[column] = aligned[column].astype(bool)
        small_fire_metrics = _small_fire_metric_rows(
            aligned,
            baseline_name=baseline.name,
            candidate_name=candidate.name,
        )
    return CandidateComparison(
        decision=decision,
        overall_metrics=overall,
        fold_metrics=_fold_metric_rows(baseline.name, candidate.name, aligned),
        transitions=_transition_rows(aligned),
        small_fire_metrics=small_fire_metrics,
        aligned_predictions=aligned,
    )
