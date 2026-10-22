"""Paired HN/random summary for the frozen internal Route B extension."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fire_detection.models.offline import sha256_file

from .protocol import digest_object, publish_contract
from .statistics import paired_cluster_bootstrap, summarize_predictions, validate_predictions
from .three_arm_evaluation import _destination, _save_csv


HN_METHODS = {"cls_hn", "cls_hn_tta"}
RANDOM_METHODS = {"cls_random_negative", "cls_random_negative_tta"}


def summarize_route_b(hn_predictions, random_predictions, output_dir, spec, *, engineering_only=False):
    """Validate and summarize two separately executed arms without merging campaigns."""
    if engineering_only:
        raise ValueError("Formal HN outputs cannot be combined with engineering-only random fixtures")
    hn = hn_predictions.loc[hn_predictions.method.isin(HN_METHODS)].copy()
    random = random_predictions.copy()
    if set(hn.method) != HN_METHODS or set(random.method) != RANDOM_METHODS:
        raise ValueError("Exactly two HN views and two random-negative views are required")
    if hn.experiment_id.nunique() != 1 or random.experiment_id.nunique() != 1:
        raise ValueError("Each arm must have one unambiguous campaign identity")
    if hn.experiment_id.iloc[0] == random.experiment_id.iloc[0]:
        raise ValueError("Old HN and new random arm must retain distinct campaign identities")
    data = validate_predictions(pd.concat([hn, random], ignore_index=True))
    if set(data.method) != HN_METHODS | RANDOM_METHODS or set(data.evidence_status) != {"internal_revalidation"}:
        raise ValueError("Mixed methods or evidence classes")
    if not data.split.eq("test").all() or set(data.seed) != {42, 2026, 3407}:
        raise ValueError("Complete formal TEST/seed evidence required")
    if data.group_id.nunique() != 427 or data.filename.nunique() != 1100:
        raise ValueError("Frozen image/component identity differs")
    for _, part in data.groupby(["method", "seed"]):
        if len(part) != 1100 or set(part.outer_fold) != set(range(5)):
            raise ValueError("Incomplete five-fold TEST set")
    if spec["bootstrap"] != {"iterations": 10000, "seed": 20260914}:
        raise ValueError("Exploratory bootstrap settings differ")
    seed, fold, summary = summarize_predictions(data)
    paired = []
    for suffix in ("", "_tta"):
        part = data.loc[data.method.isin({f"cls_hn{suffix}", f"cls_random_negative{suffix}"})]
        result = paired_cluster_bootstrap(
            part, f"cls_random_negative{suffix}", iterations=10000, analysis_seed=20260914
        )
        result["comparison_role"] = "post_outcome_internal_extension"
        result["inference_scope"] = "conditional_on_frozen_models_thresholds_and_viewed_HN_TEST"
        paired.append(result)
    destination = _destination(output_dir, False)
    paths = []
    for name, frame in (
        ("predictions", data),
        ("per_seed", seed),
        ("per_fold", fold),
        ("summary", summary),
        ("paired_ci", pd.concat(paired, ignore_index=True)),
    ):
        path = destination / f"{name}.csv"
        _save_csv(path, frame)
        paths.append(path)
    manifest = destination / "summary_manifest.json"
    publish_contract(
        manifest,
        {
            "study_kind": "HN_RANDOM_CONTROL",
            "spec_sha256": digest_object(spec),
            "imported_hn_campaign": hn.experiment_id.iloc[0],
            "new_random_campaign": random.experiment_id.iloc[0],
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 20260914,
            "unit": "427 visual components within five outer folds",
            "inference": "exploratory post-outcome internal extension; HN TEST previously viewed",
            "outputs": {path.name: sha256_file(path) for path in paths},
        },
    )
    return [*paths, manifest]
