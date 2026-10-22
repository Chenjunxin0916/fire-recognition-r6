"""Direction A analysis of frozen predictions; never fit on evaluation labels.

Folds are pooled within a training seed before seed summaries. Paired bootstrap
resamples complete scene groups within folds, sharing draws across methods and
seeds. Its intervals condition on the supplied trained models and thresholds.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REQUIRED_COLUMNS = (
    "experiment_id",
    "method",
    "outer_fold",
    "seed",
    "filename",
    "label",
    "group_id",
    "score",
    "threshold",
    "y_pred",
    "split",
    "condition",
    "evidence_status",
)
EVIDENCE_STATUSES = {"historical_exploratory", "internal_revalidation", "engineering_only"}
METRIC_NAMES = (
    "f1",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "mcc",
    "accuracy",
    "precision",
    "recall",
    "fpr",
    "specificity",
    "auroc",
    "ap_fire",
    "ap_negative",
)
BOOTSTRAP_METRICS = ("f1", "recall", "fpr", "balanced_accuracy")
MATCH_COLUMNS = ("seed", "filename", "outer_fold", "group_id", "label")
FORMAL_INTEGRITY_COLUMNS = (
    "budget_valid",
    "pairing_valid",
    "steps_valid",
    "exposure_valid",
    "completion_valid",
)


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject incomplete identities, inconsistent labels and calibration/test overlap."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    data = frame.copy()
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required prediction fields contain missing values")
    for column in (
        "experiment_id",
        "method",
        "filename",
        "group_id",
        "split",
        "condition",
        "evidence_status",
    ):
        data[column] = data[column].astype(str)
        if data[column].str.strip().eq("").any():
            raise ValueError(f"Empty identity in {column}")
    for column in ("seed", "outer_fold", "label", "y_pred"):
        values = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError(f"{column} must contain finite integers")
        data[column] = values.astype(int)
    for column in ("label", "y_pred"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be binary")
    for column in ("score", "threshold"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all() or not data[column].between(0, 1).all():
            raise ValueError(f"{column} must be finite and within [0, 1]")
    if not data.evidence_status.isin(EVIDENCE_STATUSES).all():
        raise ValueError("Unknown evidence_status; evidence must be explicitly classified")
    if not data.split.isin(["cal", "test"]).all():
        raise ValueError("Only cal and test prediction rows are supported")
    if not np.array_equal((data.score >= data.threshold).astype(int), data.y_pred):
        raise ValueError("y_pred must equal score >= the frozen threshold")
    key = ["evidence_status", "condition", "method", "seed", "outer_fold", "split", "filename"]
    if data.duplicated(key).any():
        raise ValueError(
            "Duplicate predictions; give distinct settings distinct method/condition IDs"
        )
    if data.groupby("filename").label.nunique().gt(1).any():
        raise ValueError("A filename has inconsistent labels")
    # Within an outer-fold experiment, complete scene groups must remain disjoint.
    grouping = ["evidence_status", "condition", "method", "seed", "outer_fold", "group_id"]
    if data.groupby(grouping).split.nunique().gt(1).any():
        raise ValueError("Scene group appears in both calibration and test within an outer fold")
    test = data.loc[data.split.eq("test")]
    sample_key = ["evidence_status", "condition", "method", "seed", "filename"]
    if test.duplicated(sample_key).any():
        raise ValueError("Test filename repeated across outer folds within a seed")
    split_key = ["evidence_status", "condition", "group_id"]
    if test.groupby(split_key).outer_fold.nunique().gt(1).any():
        raise ValueError("Test scene group crosses outer folds")
    identities = ["evidence_status", "condition", "filename"]
    if test.groupby(identities).group_id.nunique().gt(1).any():
        raise ValueError("Filename has inconsistent scene groups across methods/seeds")
    for _, part in test.groupby(["evidence_status", "condition", "method"], sort=True):
        reference: pd.DataFrame | None = None
        for _, seed_rows in part.groupby("seed", sort=True):
            identity = seed_rows.sort_values("filename")[
                ["filename", "outer_fold", "group_id", "label"]
            ]
            identity = identity.reset_index(drop=True)
            if reference is not None and not reference.equals(identity):
                raise ValueError(
                    "Training seeds must use the same complete test sample/fold identities"
                )
            reference = identity
    for _, part in test.groupby(["evidence_status", "condition"], sort=True):
        aligned_methods(part, sorted(part.method.unique()))
    return data


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    """Compute explicit binary metrics, returning None for undefined quantities."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    predictions = np.asarray(predictions)
    if (
        labels.ndim != 1
        or labels.size == 0
        or scores.shape != labels.shape
        or predictions.shape != labels.shape
    ):
        raise ValueError("Metrics require nonempty, equally sized 1D arrays")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Labels and predictions must be binary")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite and within [0, 1]")
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))

    def divide(a: float, b: float) -> float | None:
        return a / b if b else None

    f1 = divide(2 * tp, 2 * tp + fp + fn)
    f1_negative = divide(2 * tn, 2 * tn + fp + fn)
    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    both_classes = bool((tp + fn) and (tn + fp))
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    weighted_f1 = ((f1 or 0.0) * (tp + fn) + (f1_negative or 0.0) * (tn + fp)) / labels.size
    macro_f1, balanced_accuracy = None, None
    if both_classes:
        assert f1 is not None and f1_negative is not None
        assert recall is not None and specificity is not None
        macro_f1 = (f1 + f1_negative) / 2
        balanced_accuracy = (recall + specificity) / 2
    return {
        "n_images": int(labels.size),
        "n_positive": tp + fn,
        "n_negative": tn + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "f1": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": divide(tp * tn - fp * fn, denominator),
        "accuracy": (tp + tn) / labels.size,
        "precision": divide(tp, tp + fp),
        "recall": recall,
        "fpr": divide(fp, fp + tn),
        "specificity": specificity,
        "auroc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "ap_fire": float(average_precision_score(labels, scores)) if both_classes else None,
        "ap_negative": float(average_precision_score(1 - labels, 1 - scores))
        if both_classes
        else None,
    }


def summarize_predictions(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-seed pooled, per-fold descriptive, and mean/sample-SD tables."""
    tables: list[pd.DataFrame] = []
    for keys in (["method", "condition", "seed"], ["method", "condition", "seed", "outer_fold"]):
        rows = []
        for values, part in test.groupby(keys, sort=True):
            record = dict(zip(keys, values, strict=True))
            record.update(
                binary_metrics(part.label.to_numpy(), part.score.to_numpy(), part.y_pred.to_numpy())
            )
            record["n_groups"] = int(part.group_id.nunique())
            record["n_folds"] = int(part.outer_fold.nunique())
            record["experiment_ids"] = "|".join(sorted(part.experiment_id.unique()))
            rows.append(record)
        tables.append(pd.DataFrame(rows))
    seed_metrics, fold_metrics = tables
    summary = []
    if not seed_metrics.empty:
        for (method, condition), part in seed_metrics.groupby(["method", "condition"], sort=True):
            record = {
                "method": method,
                "condition": condition,
                "n_seeds": len(part),
                "seeds": "|".join(map(str, sorted(part.seed))),
                "n_images_per_seed": int(part.n_images.iloc[0]),
                "n_groups_per_seed": int(part.n_groups.iloc[0]),
                "n_folds_per_seed": int(part.n_folds.iloc[0]),
            }
            for metric in METRIC_NAMES:
                values = pd.to_numeric(part[metric], errors="coerce")
                valid = values.notna()
                record[f"{metric}_n_defined"] = int(valid.sum())
                # A mean over a selected subset of seeds would hide undefined metrics.
                record[f"{metric}_mean"] = float(values.mean()) if valid.all() else None
                record[f"{metric}_std"] = (
                    float(values.std(ddof=1)) if valid.all() and len(values) > 1 else None
                )
            summary.append(record)
    return seed_metrics, fold_metrics, pd.DataFrame(summary)


def aligned_methods(test: pd.DataFrame, methods: list[str]) -> dict[str, pd.DataFrame]:
    """Strict paired join; never silently intersect unequal methods or seed sets."""
    aligned: dict[str, pd.DataFrame] = {}
    reference: pd.DataFrame | None = None
    for method in methods:
        part = (
            test.loc[test.method.eq(method)]
            .sort_values(["seed", "filename"])
            .reset_index(drop=True)
        )
        if part.empty:
            raise ValueError(f"No test rows for method {method}")
        identities = part[list(MATCH_COLUMNS)]
        if reference is not None and not reference.equals(identities):
            raise ValueError(
                f"Paired comparison identities differ for {method}; no intersection is allowed"
            )
        reference = identities
        aligned[method] = part
    return aligned


def _count_metrics(counts: np.ndarray) -> dict[str, np.ndarray]:
    tp, fp, fn, tn = np.moveaxis(counts, -1, 0)

    def divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float), where=b != 0))

    recall, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    return {
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "recall": recall,
        "fpr": divide(fp, fp + tn),
        "balanced_accuracy": (recall + specificity) / 2,
    }


def paired_cluster_bootstrap(
    test: pd.DataFrame,
    reference_method: str,
    *,
    iterations: int = 10000,
    analysis_seed: int = 20260914,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Estimate paired raw metric differences; no bootstrap tail is called a p-value."""
    if iterations < 1 or not 0 < confidence < 1:
        raise ValueError("iterations must be positive and confidence strictly between 0 and 1")
    if (
        test.condition.nunique() != 1
        or test.evidence_status.nunique() != 1
        or not test.split.eq("test").all()
    ):
        raise ValueError("Bootstrap requires one condition, one evidence type and test rows only")
    methods = sorted(test.method.unique())
    if reference_method not in methods:
        raise ValueError(f"Reference method {reference_method} is absent")
    aligned = aligned_methods(test, methods)
    base = aligned[reference_method]
    seeds = sorted(base.seed.unique())
    sample = base.loc[base.seed.eq(seeds[0])].sort_values("filename")
    groups = sorted(sample.group_id.unique())
    if len(groups) < 2:
        raise ValueError("Cluster bootstrap needs at least two distinct scene groups")
    if sample.groupby("group_id").outer_fold.nunique().gt(1).any():
        raise ValueError("Scene groups cross test folds")
    group_index = {group: index for index, group in enumerate(groups)}
    group_codes = sample.group_id.map(group_index).to_numpy(dtype=int)
    group_counts = np.zeros((len(groups), len(seeds), len(methods), 4), dtype=float)
    for method_index, method in enumerate(methods):
        for seed_index, seed in enumerate(seeds):
            part = aligned[method].loc[aligned[method].seed.eq(seed)].sort_values("filename")
            if (
                not part[["filename", "group_id", "outer_fold", "label"]]
                .reset_index(drop=True)
                .equals(
                    sample[["filename", "group_id", "outer_fold", "label"]].reset_index(drop=True)
                )
            ):
                raise ValueError("Bootstrap seed identities differ")
            y, pred = part.label.to_numpy(), part.y_pred.to_numpy()
            flags = np.column_stack(
                (
                    (y == 1) & (pred == 1),
                    (y == 0) & (pred == 1),
                    (y == 1) & (pred == 0),
                    (y == 0) & (pred == 0),
                )
            )
            np.add.at(group_counts[:, seed_index, method_index, :], group_codes, flags)
    point = {
        metric: values.mean(axis=0)
        for metric, values in _count_metrics(group_counts.sum(axis=0)).items()
    }
    rng = np.random.default_rng(analysis_seed)
    group_folds = sample.drop_duplicates("group_id").set_index("group_id").loc[groups, "outer_fold"]
    draws: dict[str, list[np.ndarray]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    for start in range(0, iterations, 512):
        size = min(512, iterations - start)
        multiplicities = np.zeros((size, len(groups)), dtype=float)
        for fold in sorted(group_folds.unique()):
            indexes = np.flatnonzero(group_folds.to_numpy() == fold)
            n_groups = len(indexes)
            multiplicities[:, indexes] = rng.multinomial(
                n_groups, np.full(n_groups, 1 / n_groups), size=size
            )
        counts = (multiplicities @ group_counts.reshape(len(groups), -1)).reshape(
            size, len(seeds), len(methods), 4
        )
        for metric, values in _count_metrics(counts).items():
            draws[metric].append(values.mean(axis=1))
    reference_index = methods.index(reference_method)
    alpha = (1 - confidence) / 2
    rows = []
    for metric in BOOTSTRAP_METRICS:
        samples = np.concatenate(draws[metric])
        for method_index, method in enumerate(methods):
            if method == reference_method:
                continue
            delta = samples[:, method_index] - samples[:, reference_index]
            valid = np.isfinite(delta)
            lower, upper = (
                np.quantile(delta[valid], [alpha, 1 - alpha]) if valid.any() else (np.nan, np.nan)
            )
            rows.append(
                {
                    "method": method,
                    "reference": reference_method,
                    "condition": str(test.condition.iloc[0]),
                    "evidence_status": str(test.evidence_status.iloc[0]),
                    "metric": metric,
                    "effect_difference": float(
                        point[metric][method_index] - point[metric][reference_index]
                    ),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "confidence": confidence,
                    "iterations": iterations,
                    "valid_iterations": int(valid.sum()),
                    "invalid_iterations": int((~valid).sum()),
                    "analysis_seed": analysis_seed,
                    "n_groups": len(groups),
                    "n_images_per_seed": len(sample),
                    "n_seeds": len(seeds),
                    "inference_scope": "conditional_on_frozen_models_and_thresholds",
                    "comparison_role": "prespecified_primary"
                    if method == "cls_hn" and reference_method == "cls_random_negative"
                    else "diagnostic",
                    "multiplicity": "unadjusted_pointwise_CI_no_significance_claim",
                }
            )
    return pd.DataFrame(rows)


def _subgroup_table(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = [column for column in test if column == "subgroup" or column.startswith("subgroup_")]
    for column in columns:
        annotated = test.copy()
        annotated[column] = annotated[column].fillna("unknown").astype(str)
        for (method, condition, seed, value), part in annotated.groupby(
            ["method", "condition", "seed", column], sort=True
        ):
            record = {
                "method": method,
                "condition": condition,
                "seed": seed,
                "subgroup_field": column,
                "subgroup_value": value,
                "n_groups": part.group_id.nunique(),
            }
            record.update(
                binary_metrics(part.label.to_numpy(), part.score.to_numpy(), part.y_pred.to_numpy())
            )
            rows.append(record)
    return pd.DataFrame(rows)


def _save_table(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, encoding="utf-8", na_rep="")


_H1_F1_MINIMUM = 0.005
_H1_RECALL_BOUNDARY = -0.01
_H3_CONDITIONS = ("brightness_0.7", "blur_1", "noise_0.02")
_H2_PAIRS = (
    ("fusion_hn_full", "cls_hn_tta"),
    ("fusion_hn_full", "fusion_hn_no_slice"),
    ("fusion_hn_full", "fusion_hn_no_tta"),
)
_H1_PAIR = ("cls_hn", "cls_random_negative")


def _summary_complete(summary: pd.DataFrame, methods: set[str], condition: str) -> bool:
    """Return whether all requested method summaries are complete and eligible.

    ``eligible_internal_main_table`` is emitted by :func:`analyze_predictions`.
    Requiring it when present prevents a partial/engineering table from being
    promoted by the hypothesis classifier.  The fallback for small unit-test
    fixtures intentionally accepts a missing column, while still requiring the
    explicit three-seed count.
    """
    if summary.empty:
        return False
    part = summary.loc[summary.condition.eq(condition) & summary.method.isin(methods)]
    if set(part.method) != methods or len(part) != len(methods):
        return False
    image_counts = pd.to_numeric(part.n_images_per_seed, errors="coerce")
    if (
        not part.n_seeds.eq(3).all()
        or image_counts.isna().any()
        or image_counts.le(0).any()
        or image_counts.nunique() != 1
    ):
        return False
    if "n_folds_per_seed" not in part or not part.n_folds_per_seed.eq(5).all():
        return False
    for metric in ("f1", "recall"):
        defined = f"{metric}_n_defined"
        mean = f"{metric}_mean"
        if defined not in part or mean not in part:
            return False
        if not part[defined].eq(3).all():
            return False
        numeric_mean = pd.to_numeric(part[mean], errors="coerce")
        if numeric_mean.isna().any() or not np.isfinite(numeric_mean).all():
            return False
    # These optional columns are emitted by stricter collectors when available.
    # A false value is hard evidence of an invalid budget/pairing contract and
    # must never be turned into a paper-facing negative result.
    for column in (
        "protocol_complete",
        "budget_valid",
        "pairing_valid",
        "steps_valid",
        "exposure_valid",
        "completion_valid",
    ):
        if column in part:
            values = part[column].tolist()
            # Do not let strings such as "False"/"0" become truthy through
            # pandas' generic astype(bool) coercion.  Contract flags are
            # booleans, not user-facing status labels.
            if not all(type(value) is bool and value for value in values):
                return False
    if "eligible_internal_main_table" not in part:
        return False
    values = part.eligible_internal_main_table.tolist()
    if not all(type(value) is bool and value for value in values):
        return False
    return True


def _finite_ci_row(row: pd.Series | None) -> bool:
    """Validate one bootstrap row before it can affect a paper-facing status."""
    if row is None:
        return False
    required = {
        "effect_difference",
        "ci_lower",
        "ci_upper",
        "iterations",
        "valid_iterations",
        "invalid_iterations",
    }
    if not required <= set(row.index):
        return False
    if any(
        isinstance(row[column], bool)
        for column in ("iterations", "valid_iterations", "invalid_iterations")
    ):
        return False
    try:
        iterations = int(row["iterations"])
        valid_iterations = int(row["valid_iterations"])
        invalid_iterations = int(row["invalid_iterations"])
    except (TypeError, ValueError, OverflowError):
        return False
    if iterations < 1 or valid_iterations != iterations or invalid_iterations != 0:
        return False
    for column in ("protocol_valid", "budget_valid", "pairing_valid", "completion_valid"):
        if column in row.index and row[column] is not True and row[column] != 1:
            return False
    if any(
        isinstance(row[column], bool)
        for column in ("effect_difference", "ci_lower", "ci_upper")
    ):
        return False
    values = pd.to_numeric(
        pd.Series([row["effect_difference"], row["ci_lower"], row["ci_upper"]]),
        errors="coerce",
    ).to_numpy(dtype=float)
    return bool(np.isfinite(values).all() and float(row["ci_lower"]) <= float(row["ci_upper"]))


def _ci_relation(row: pd.Series, boundary: float) -> str:
    """Classify a CI relative to a strict support boundary.

    ``contains`` includes equality at either endpoint.  This is deliberate:
    touching zero (or the -0.01 recall protection boundary) is not evidence
    that the corresponding strict condition has been met.
    """
    lower, upper = float(row.ci_lower), float(row.ci_upper)
    if lower > boundary:
        return "above"
    if upper < boundary:
        return "below"
    return "contains"


def _status_from_ci(
    f1: pd.Series,
    recall: pd.Series,
    *,
    minimum_effect: float | None,
) -> tuple[str, dict[str, Any]]:
    """Apply one frozen CI truth table used by H1 and H2.

    Any CI containing its uncertainty boundary is ``inconclusive``.  Once both
    intervals exclude their boundaries, a clearly reversed/protective failure
    is ``not_supported``; otherwise the minimum-effect rule (H1 only) is
    applied.  Invalid rows are handled by the caller as ``not_run``.
    """
    f1_relation = _ci_relation(f1, 0.0)
    recall_relation = _ci_relation(recall, _H1_RECALL_BOUNDARY)
    observed = {
        "delta_f1": float(f1.effect_difference),
        "delta_f1_ci_lower": float(f1.ci_lower),
        "delta_f1_ci_upper": float(f1.ci_upper),
        "delta_recall": float(recall.effect_difference),
        "delta_recall_ci_lower": float(recall.ci_lower),
        "delta_recall_ci_upper": float(recall.ci_upper),
        "f1_ci_relation": f1_relation,
        "recall_ci_relation": recall_relation,
    }
    if f1_relation == "contains" or recall_relation == "contains":
        return "inconclusive", observed
    if f1_relation == "below" or recall_relation == "below":
        return "not_supported", observed
    if minimum_effect is not None and float(f1.effect_difference) < minimum_effect:
        return "not_supported", observed
    return "supported", observed


def _comparison_rows(
    paired: pd.DataFrame,
    method: str,
    reference: str,
    condition: str,
) -> dict[str, pd.Series] | None:
    """Get exactly one valid-looking f1/recall row for a directed contrast."""
    if paired.empty:
        return None
    part = paired.loc[
        paired.method.eq(method)
        & paired.reference.eq(reference)
        & paired.condition.eq(condition)
    ]
    # A caller may choose an H2 comparator as the global reference as well;
    # prefer the explicitly tagged prespecified-H2 rows over that diagnostic
    # duplicate, while still rejecting genuinely conflicting duplicates.
    if len(part) > 2 and "comparison_role" in part:
        expected_role = (
            "prespecified_h1"
            if (method, reference) == _H1_PAIR
            else "prespecified_h2"
            if (method, reference) in _H2_PAIRS
            else None
        )
        tagged = (
            part.loc[part.comparison_role.eq(expected_role)]
            if expected_role is not None
            else pd.DataFrame()
        )
        if not tagged.empty:
            part = tagged
    if part.metric.duplicated().any():
        return None
    rows = {
        metric: part.loc[part.metric.eq(metric)].iloc[0]
        for metric in ("f1", "recall")
        if not part.loc[part.metric.eq(metric)].empty
    }
    return rows if set(rows) == {"f1", "recall"} else None


def _h3_seed_evidence(
    seed_metrics: pd.DataFrame | None,
    summary: pd.DataFrame,
    paired: pd.DataFrame | None = None,
    expected_seeds: tuple[int, ...] = (42, 2026, 3407),
) -> tuple[dict[str, Any] | None, str | None]:
    """Evaluate the predeclared H3 seed/perturbation numbers.

    H3 is not inferred from mere condition presence: every required method,
    condition and seed must have a finite pooled F1 and an eligible summary.
    """
    methods = {"cls_hn", "cls_random_negative"}
    conditions = ("clean", *_H3_CONDITIONS)
    declared_seeds = set(expected_seeds)
    if len(declared_seeds) != 3 or any(
        type(seed) is not int or seed < 0 for seed in declared_seeds
    ):
        return None, "invalid declared seed set"
    if seed_metrics is None or seed_metrics.empty:
        return None, "missing seed-level F1 table"
    required_columns = {"method", "condition", "seed", "f1"}
    if not required_columns <= set(seed_metrics.columns):
        return None, "seed-level F1 table is missing required columns"
    for condition in conditions:
        if not _summary_complete(summary, methods, condition):
            return None, f"incomplete summary for {condition}"
        # A seed table without the predeclared paired bootstrap is descriptive
        # only.  H3 is a paper-facing hypothesis and therefore requires the
        # same complete, frozen pairing evidence as H1.
        if paired is None or paired.empty:
            return None, "missing paired H3 evidence"
        paired_rows = _comparison_rows(paired, "cls_hn", "cls_random_negative", condition)
        if paired_rows is None or any(not _finite_ci_row(row) for row in paired_rows.values()):
            return None, f"invalid paired evidence for {condition}"
    parts: dict[tuple[str, str], pd.DataFrame] = {}
    paired_relations: dict[str, dict[str, str]] = {}
    condition_metrics: dict[str, dict[str, float]] = {}
    clean_seeds: set[int] | None = None
    for condition in conditions:
        for method in methods:
            part = seed_metrics.loc[
                seed_metrics.method.eq(method) & seed_metrics.condition.eq(condition)
            ].copy()
            if len(part) != len(declared_seeds) or part.seed.nunique() != len(declared_seeds):
                return None, f"incomplete seed rows for {method}/{condition}"
            try:
                values = pd.to_numeric(part.f1, errors="coerce").to_numpy(dtype=float)
            except (TypeError, ValueError):
                return None, f"invalid F1 rows for {method}/{condition}"
            if not np.isfinite(values).all():
                return None, f"non-finite F1 rows for {method}/{condition}"
            numeric_seeds = pd.to_numeric(part.seed, errors="coerce")
            if numeric_seeds.isna().any() or not np.isfinite(numeric_seeds).all():
                return None, f"invalid seed identities for {method}/{condition}"
            if (numeric_seeds != np.floor(numeric_seeds)).any():
                return None, f"non-integer seed identities for {method}/{condition}"
            part["seed"] = numeric_seeds.astype(int)
            part["f1"] = values
            seeds = set(part.seed)
            if seeds != declared_seeds:
                return None, f"unexpected seed identities for {method}/{condition}"
            if clean_seeds is None:
                clean_seeds = seeds
            elif seeds != clean_seeds:
                return None, "seed identities differ across H3 conditions"
            parts[(method, condition)] = part.set_index("seed").sort_index()
        paired_rows = _comparison_rows(paired, "cls_hn", "cls_random_negative", condition)
        assert paired_rows is not None
        paired_relations[condition] = {
            "f1_ci_relation": _ci_relation(paired_rows["f1"], 0.0),
            "recall_ci_relation": _ci_relation(paired_rows["recall"], _H1_RECALL_BOUNDARY),
        }
        condition_part = paired.loc[
            paired.method.eq("cls_hn")
            & paired.reference.eq("cls_random_negative")
            & paired.condition.eq(condition)
        ].copy()
        if "comparison_role" in condition_part and condition_part.comparison_role.eq(
            "prespecified_h1"
        ).any():
            condition_part = condition_part.loc[
                condition_part.comparison_role.eq("prespecified_h1")
            ]
        metrics: dict[str, float] = {}
        for metric in ("f1", "recall", "fpr"):
            metric_rows = condition_part.loc[condition_part.metric.eq(metric)]
            if len(metric_rows) != 1:
                return None, f"missing paired {metric} result for {condition}"
            value = pd.to_numeric(metric_rows.effect_difference, errors="coerce").iloc[0]
            if not np.isfinite(value):
                return None, f"invalid paired {metric} result for {condition}"
            metrics[f"delta_{metric}"] = float(value)
        condition_metrics[condition] = metrics
    assert clean_seeds is not None
    clean_delta = (
        parts[("cls_hn", "clean")].f1 - parts[("cls_random_negative", "clean")].f1
    )
    perturb_delta = {
        condition: float(
            (parts[("cls_hn", condition)].f1 - parts[("cls_random_negative", condition)].f1).mean()
        )
        for condition in _H3_CONDITIONS
    }
    clean_hn_mean = float(parts[("cls_hn", "clean")].f1.mean())
    degradation = {
        condition: clean_hn_mean - float(parts[("cls_hn", condition)].f1.mean())
        for condition in _H3_CONDITIONS
    }
    observed = {
        "clean_delta_f1_by_seed": {str(seed): float(value) for seed, value in clean_delta.items()},
        "clean_positive_seed_count": int((clean_delta > 0).sum()),
        "perturbation_mean_delta_f1": perturb_delta,
        "hn_clean_f1_mean": clean_hn_mean,
        "hn_f1_degradation_from_clean": degradation,
        "paired_ci_relations": paired_relations,
        "condition_metrics": condition_metrics,
        "worst_perturbation_rule": (
            "minimum_delta_f1_then_minimum_delta_recall_then_maximum_delta_fpr"
        ),
        "worst_perturbation": min(
            _H3_CONDITIONS,
            key=lambda condition: (
                condition_metrics[condition]["delta_f1"],
                condition_metrics[condition]["delta_recall"],
                -condition_metrics[condition]["delta_fpr"],
                condition,
            ),
        ),
        "paired_ci_uncertain_conditions": [
            condition
            for condition, relation in paired_relations.items()
            if "contains" in relation.values()
        ],
        "paired_ci_reversed_conditions": [
            condition
            for condition, relation in paired_relations.items()
            if "below" in relation.values()
        ],
    }
    return observed, None


def _hypothesis_status(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    status: str,
    seed_metrics: pd.DataFrame | None = None,
    expected_seeds: tuple[int, ...] = (42, 2026, 3407),
    scheme_b_only: bool = False,
) -> dict[str, Any]:
    """Apply the frozen numeric truth tables for H1--H3.

    Missing, incomplete, or invalid evidence is always ``not_run``.  A CI that
    contains a strict support boundary is ``inconclusive``; it is never treated
    as equivalence or support.
    """
    h1: dict[str, Any] = {
        "status": "not_run",
        "comparison": "cls_hn - cls_random_negative; clean; three complete training seeds",
        "criteria": {
            "minimum_delta_f1": _H1_F1_MINIMUM,
            "delta_f1_ci_lower_strictly_above": 0,
            "delta_recall_ci_lower_strictly_above": _H1_RECALL_BOUNDARY,
            "invalid_or_missing_evidence": "not_run",
            "ci_touching_or_crossing_boundary": "inconclusive",
        },
        "inference_scope": "conditional_on_frozen_models_and_thresholds_internal_data_only",
    }
    h2: dict[str, Any] = {
        "status": "not_applicable" if scheme_b_only else "not_run",
        "comparisons": list(
            {"method": method, "reference": reference} for method, reference in _H2_PAIRS
        ),
        "criteria": {
            "primary": "fusion_hn_full - cls_hn_tta",
            "aggregate_rule": "all_three_declared_comparisons_supported",
            "f1_ci_lower_strictly_above": 0,
            "delta_recall_ci_lower_strictly_above": _H1_RECALL_BOUNDARY,
            "multiplicity": "exploratory_unadjusted_pointwise_CI_no_familywise_claim",
            "invalid_or_missing_evidence": "not_run",
            "ci_touching_or_crossing_boundary": "inconclusive",
        },
    }
    h3: dict[str, Any] = {
        "status": "not_run",
        "criteria": (
            {
                "clean_positive_seed_count_at_least": 2,
                "perturbations": "descriptive_complete_reporting_only",
                "paired_intervals": "descriptive_not_support_thresholds",
                "invalid_or_missing_evidence": "not_run",
            }
            if scheme_b_only
            else {
                "clean_positive_seed_count_at_least": 2,
                "perturbation_mean_delta_f1_nonnegative": True,
                "hn_clean_to_perturbed_f1_degradation_at_most": 0.03,
                "paired_f1_ci_lower_strictly_above": 0,
                "paired_recall_ci_lower_strictly_above": _H1_RECALL_BOUNDARY,
                "invalid_or_missing_evidence": "not_run",
                "ci_touching_or_crossing_boundary": "inconclusive",
            }
        ),
    }
    if status == "internal_revalidation":
        # H1: complete summaries + one valid paired bootstrap row per metric.
        h1_complete = _summary_complete(
            summary, {"cls_hn", "cls_random_negative"}, "clean"
        )
        h1_rows = _comparison_rows(paired, "cls_hn", "cls_random_negative", "clean")
        if h1_complete and h1_rows is not None:
            f1, recall = h1_rows["f1"], h1_rows["recall"]
            if _finite_ci_row(f1) and _finite_ci_row(recall):
                h1["status"], h1["observed"] = _status_from_ci(
                    f1, recall, minimum_effect=_H1_F1_MINIMUM
                )
            else:
                h1["reason"] = "missing, invalid, or degenerate bootstrap evidence"
        else:
            h1["reason"] = "incomplete H1 summaries or paired evidence"

        # H2 requires all three predeclared directed comparisons to be present.
        # The SOP says each comparison must be reported and prohibits promoting
        # a single effective component into a claim about the whole ablation;
        # therefore the aggregate support state requires all three pair states.
        if scheme_b_only:
            h2 = {
                "status": "not_applicable",
                "reason": "Scheme B removed H2, detector, fusion, and component ablations",
            }
        else:
            h2_complete = _summary_complete(
                summary,
                {method for pair in _H2_PAIRS for method in pair},
                "clean",
            )
            h2_rows = {
                pair: _comparison_rows(paired, pair[0], pair[1], "clean") for pair in _H2_PAIRS
            }
            if h2_complete and all(value is not None for value in h2_rows.values()):
                invalid = any(
                    not (_finite_ci_row(value["f1"]) and _finite_ci_row(value["recall"]))
                    for value in h2_rows.values()
                    if value is not None
                )
                if not invalid:
                    h2["comparisons"] = [
                        {
                            "method": method,
                            "reference": reference,
                            "status": _status_from_ci(
                                h2_rows[(method, reference)]["f1"],
                                h2_rows[(method, reference)]["recall"],
                                minimum_effect=None,
                            )[0],
                            "observed": _status_from_ci(
                                h2_rows[(method, reference)]["f1"],
                                h2_rows[(method, reference)]["recall"],
                                minimum_effect=None,
                            )[1],
                        }
                        for method, reference in _H2_PAIRS
                    ]
                    pair_states = [item["status"] for item in h2["comparisons"]]
                    if "inconclusive" in pair_states:
                        h2["status"] = "inconclusive"
                    elif "not_supported" in pair_states:
                        h2["status"] = "not_supported"
                    else:
                        h2["status"] = "supported"
                    h2["observed"] = {
                        "aggregate_rule": "all_three_declared_comparisons_supported",
                        "primary_comparison": h2["comparisons"][0]["observed"],
                    }
                else:
                    h2["reason"] = "missing, invalid, or degenerate H2 bootstrap evidence"
            else:
                h2["reason"] = "incomplete H2 method matrix or paired evidence"

        h3_observed, h3_reason = _h3_seed_evidence(
            seed_metrics, summary, paired, expected_seeds
        )
        if h3_observed is not None:
            h3["observed"] = h3_observed
            if scheme_b_only:
                h3["status"] = (
                    "supported"
                    if h3_observed["clean_positive_seed_count"] >= 2
                    else "not_supported"
                )
            elif h3_observed["paired_ci_uncertain_conditions"]:
                h3["status"] = "inconclusive"
            elif h3_observed["paired_ci_reversed_conditions"]:
                h3["status"] = "not_supported"
            else:
                h3["status"] = (
                    "supported"
                    if (
                        h3_observed["clean_positive_seed_count"] >= 2
                        and all(
                            value >= 0
                            for value in h3_observed["perturbation_mean_delta_f1"].values()
                        )
                        and all(
                            value <= 0.03
                            for value in h3_observed["hn_f1_degradation_from_clean"].values()
                        )
                    )
                    else "not_supported"
                )
        else:
            h3["reason"] = h3_reason or "incomplete H3 evidence"
    return {
        "evidence_status": status,
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "note": (
            "These numeric criteria do not establish external generalization or a causal mechanism."
        ),
    }


def analyze_predictions(
    input_path: Path,
    output_dir: Path,
    *,
    reference_method: str = "cls_random_negative",
    bootstrap_iterations: int = 10000,
    analysis_seed: int = 20260914,
    expected_seeds: tuple[int, ...] = (42, 2026, 3407),
    expected_folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    expected_images: int = 1100,
    expected_methods: tuple[str, ...] | None = None,
    formal_scheme_b: bool = False,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Write evidence-separated tables, provenance, and real-data-only figure bundles."""
    if formal_scheme_b:
        if bootstrap_iterations != 10000:
            raise ValueError(
                "formal Scheme B analysis requires exactly 10,000 bootstrap iterations"
            )
        if expected_methods != _H1_PAIR:
            raise ValueError(
                "formal Scheme B analysis requires exactly cls_hn and cls_random_negative"
            )
        if tuple(expected_seeds) != (42, 2026, 3407) or tuple(expected_folds) != (
            0,
            1,
            2,
            3,
            4,
        ):
            raise ValueError(
                "formal Scheme B analysis requires the frozen three-seed/five-fold matrix"
            )
        if expected_images != 1100:
            raise ValueError("formal Scheme B analysis requires 1,100 test images per seed")
    started = time.perf_counter()
    input_path, output_dir = Path(input_path), Path(output_dir)
    frame = validate_predictions(
        pd.read_csv(
            input_path, dtype={"filename": str, "group_id": str}, float_precision="round_trip"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "created_utc": datetime.now(UTC).isoformat(),
        "input_rows": len(frame),
        "calibration_rows_excluded": int(frame.split.eq("cal").sum()),
        "analysis_seed": analysis_seed,
        "bootstrap_iterations": bootstrap_iterations,
        "reference_method": reference_method,
        "expected_seeds": list(expected_seeds),
        "expected_folds": list(expected_folds),
        "expected_images_per_seed": expected_images,
        "expected_methods": list(expected_methods) if expected_methods is not None else None,
        "formal_scheme_b": formal_scheme_b,
        "code_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("plots.py"))
        },
        "statuses": {},
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib")},
        },
        "interpretation": [
            (
                "Pool folds once within each seed, then mean and sample SD across seeds; "
                "seeds are algorithmic repeats, not independent datasets."
            ),
            (
                "Resample entire scene groups within folds, sharing multiplicities across "
                "all methods and seeds."
            ),
            (
                "Percentile intervals are conditional on frozen trained models, weights and "
                "thresholds; training/selection uncertainty is not included."
            ),
            (
                "Historical exploration does not become confirmatory because "
                "bootstrap intervals are narrow."
            ),
            (
                "No p-values or significance claims; diagnostic intervals are pointwise, "
                "without multiplicity adjustment."
            ),
            "Null/blank denotes an undefined metric or sample SD unavailable from a single seed.",
        ],
    }
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
        report["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
        report["git_tracked_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        report["git_commit"] = report["git_tracked_dirty"] = None
    for status in sorted(frame.evidence_status.unique()):
        test = frame.loc[frame.evidence_status.eq(status) & frame.split.eq("test")].copy()
        destination = output_dir / status
        tables_dir = destination / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        _save_table(test, destination / "source_predictions.csv")
        seed_table, fold_table, summary = summarize_predictions(test)
        matrix_complete = expected_methods is None or set(test.method.unique()) == set(
            expected_methods
        )
        integrity_columns_complete = not formal_scheme_b or set(
            FORMAL_INTEGRITY_COLUMNS
        ) <= set(test.columns)
        matrix_complete = matrix_complete and integrity_columns_complete
        complete = []
        for _, row in summary.iterrows():
            part = test.loc[test.method.eq(row.method) & test.condition.eq(row.condition)]
            complete.append(
                set(part.seed.unique()) == set(expected_seeds)
                and matrix_complete
                and all(
                    set(group.outer_fold.unique()) == set(expected_folds)
                    and len(group) == expected_images
                    for _, group in part.groupby("seed")
                )
            )
        if not summary.empty:
            summary["protocol_complete"] = complete
            summary["method_matrix_complete"] = matrix_complete
            for flag in FORMAL_INTEGRITY_COLUMNS:
                if flag in test.columns:
                    summary[flag] = [
                        bool(
                            test.loc[
                                test.method.eq(row.method)
                                & test.condition.eq(row.condition),
                                flag,
                            ].map(lambda value: type(value) is bool and value).all()
                        )
                        for _, row in summary.iterrows()
                    ]
            summary["eligible_internal_main_table"] = summary.protocol_complete & (
                status == "internal_revalidation"
            )
            summary["evidence_status"] = status
        _save_table(seed_table, tables_dir / "per_seed.csv")
        _save_table(fold_table, tables_dir / "per_fold.csv")
        _save_table(summary, tables_dir / "performance.csv")
        if status == "internal_revalidation" and not summary.empty:
            _save_table(
                summary.loc[summary.eligible_internal_main_table],
                tables_dir / "main_complete_only.csv",
            )
        paired_parts, warnings = [], []
        for condition, part in test.groupby("condition", sort=True):
            condition_methods = set(part.method)
            if reference_method not in condition_methods:
                warnings.append(
                    f"{condition}: paired CI unavailable; reference {reference_method} absent"
                )
            elif part.method.nunique() < 2:
                warnings.append(f"{condition}: paired CI unavailable; only one method")
            else:
                paired_parts.append(
                    paired_cluster_bootstrap(
                        part,
                        reference_method,
                        iterations=bootstrap_iterations,
                        analysis_seed=analysis_seed,
                    )
                )
            # Keep the prespecified H1 pair available even when a caller asks
            # for a different diagnostic reference method.
            if (
                set(_H1_PAIR) <= condition_methods
                and reference_method != _H1_PAIR[1]
            ):
                h1_pair_rows = part.loc[part.method.isin(_H1_PAIR)].copy()
                h1_pair_result = paired_cluster_bootstrap(
                    h1_pair_rows,
                    _H1_PAIR[1],
                    iterations=bootstrap_iterations,
                    analysis_seed=analysis_seed,
                )
                h1_pair_result = h1_pair_result.loc[
                    h1_pair_result.method.eq(_H1_PAIR[0])
                ].copy()
                h1_pair_result["comparison_role"] = "prespecified_h1"
                paired_parts.append(h1_pair_result)
            # H2 has three predeclared directed contrasts that are not all
            # relative to the ordinary replay reference.  Keep these rows in
            # the same auditable paired-CI table so hypothesis classification
            # never infers a component effect from method presence alone.
            for method, reference in _H2_PAIRS:
                if {method, reference} <= condition_methods:
                    pair_rows = part.loc[part.method.isin([method, reference])].copy()
                    pair_result = paired_cluster_bootstrap(
                        pair_rows,
                        reference,
                        iterations=bootstrap_iterations,
                        analysis_seed=analysis_seed,
                    )
                    pair_result = pair_result.loc[pair_result.method.eq(method)].copy()
                    pair_result["comparison_role"] = "prespecified_h2"
                    paired_parts.append(pair_result)
        paired = pd.concat(paired_parts, ignore_index=True) if paired_parts else pd.DataFrame()
        _save_table(paired, tables_dir / "paired_ci.csv")
        hypothesis_status = _hypothesis_status(
            summary,
            paired,
            status,
            seed_table,
            tuple(expected_seeds),
            scheme_b_only=formal_scheme_b,
        )
        (destination / "hypotheses.json").write_text(
            json.dumps(hypothesis_status, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        subgroup = _subgroup_table(test)
        _save_table(subgroup, tables_dir / "subgroups.csv")
        figure_status: dict[str, Any] = {"status": "disabled"}
        if make_plots and not test.empty:
            from fire_detection.research.plots import create_figure_bundle

            figure_status = create_figure_bundle(
                test, summary, paired, destination / "figures", status
            )
        report["statuses"][status] = {
            "test_rows": len(test),
            "methods": sorted(test.method.unique()),
            "integrity_columns_complete": integrity_columns_complete,
            "warnings": warnings,
            "subgroup_status": "available"
            if not subgroup.empty
            else "not_run_no_subgroup_annotations",
            "figure_status": figure_status,
            "H1_status": hypothesis_status["H1"]["status"],
            "H2_status": hypothesis_status["H2"]["status"],
            "H3_status": hypothesis_status["H3"]["status"],
            "paper_main_table_allowed": status == "internal_revalidation"
            and bool(complete)
            and all(complete),
        }
    report["runtime_seconds"] = time.perf_counter() - started
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return report
