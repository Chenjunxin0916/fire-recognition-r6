"""Real-prediction quantitative figures for Direction A (Python/matplotlib only).

Figure contract: expose observed performance and uncertainty without presuming HN
benefit. Performance/difference panels are the comparison evidence; confusion and
ROC/PR panels diagnose the tradeoff. No mechanism or external-generalization claim.
Archetype: quantitative grid; 89/183 mm width; editable SVG/PDF plus 600 dpi PNG.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

PALETTE = ("#386A8B", "#BC793E", "#725B8B", "#517B6A", "#7A7A7A", "#8C5965")
SINGLE_COLUMN_IN = 89 / 25.4
DOUBLE_COLUMN_IN = 183 / 25.4


def _safe_name(value: str) -> str:
    # Path.with_suffix() treats a dot in a stem as an existing extension and
    # would discard the condition/method tail (for example ``noise_0.02``),
    # causing distinct result figures to overwrite each other.
    simple = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")[:85] or "figure"
    return f"{simple}_{hashlib.sha256(value.encode()).hexdigest()[:6]}"


def _save(fig: Any, stem: Path) -> list[str]:
    outputs = []
    for suffix in (".svg", ".pdf", ".png"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def _performance(summary: pd.DataFrame, output: Path, title: str) -> list[str]:
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, max(1.9, len(summary) * 0.32)))
    for index, (_, row) in enumerate(summary.iterrows()):
        sd = row.f1_std if pd.notna(row.f1_std) else None
        ax.errorbar(
            row.f1_mean, index, xerr=sd, fmt="o", color=PALETTE[index % len(PALETTE)], capsize=2
        )
    ax.set(
        yticks=range(len(summary)),
        yticklabels=summary.method,
        xlabel=(
            "Positive-class F1 (one seed; sample SD unavailable)"
            if summary.n_seeds.eq(1).all()
            else "Positive-class F1 (mean; error bars: seed sample SD)"
        ),
        xlim=(0, 1),
        title=title,
    )
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, output)


def _curves(test: pd.DataFrame, output: Path, title: str) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.7))
    for method_index, (method, part) in enumerate(test.groupby("method", sort=True)):
        for seed_index, (seed, sample) in enumerate(part.groupby("seed", sort=True)):
            if sample.label.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(sample.label, sample.score)
            precision, recall, _ = precision_recall_curve(sample.label, sample.score)
            label = f"{method}; seed {seed}"
            kwargs = {
                "color": PALETTE[method_index % len(PALETTE)],
                "linestyle": ("-", "--", ":", "-.")[seed_index % 4],
                "label": label,
                "linewidth": 1,
            }
            axes[0].plot(fpr, tpr, **kwargs)
            axes[1].step(recall, precision, where="post", **kwargs)
    axes[0].plot([0, 1], [0, 1], color="0.7", linestyle="--", linewidth=0.6)
    axes[0].set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(0, 1),
        ylim=(0, 1),
        title="a  ROC",
    )
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
        title="b  Precision-recall (fire)",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=6)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    return _save(fig, output)


def _confusion(test: pd.DataFrame, output: Path, title: str) -> list[str]:
    matrices = [
        confusion_matrix(part.label, part.y_pred, labels=[0, 1])
        for _, part in test.groupby("seed", sort=True)
    ]
    matrix = np.mean(matrices, axis=0)
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.7))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            label = str(int(value)) if len(matrices) == 1 else f"{value:.1f}"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if value > matrix.max() * 0.55 else "black",
            )
    ax.set(
        xticks=[0, 1],
        xticklabels=["No fire", "Fire"],
        yticks=[0, 1],
        yticklabels=["No fire", "Fire"],
        xlabel="Predicted label",
        ylabel="Observed label",
        title=title,
    )
    fig.text(
        0.5,
        -0.01,
        f"Mean counts over {len(matrices)} training seed(s); pooled test folds per seed",
        ha="center",
        fontsize=6,
    )
    fig.tight_layout()
    return _save(fig, output)


def _differences(paired: pd.DataFrame, output: Path, title: str) -> list[str]:
    selected = paired.loc[paired.metric.eq("f1")]
    if selected.empty:
        return []
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, max(1.9, len(selected) * 0.35)))
    for index, (_, row) in enumerate(selected.iterrows()):
        # Plot CI as a segment: percentile CI need not contain the point estimate.
        ax.plot([row.ci_lower, row.ci_upper], [index, index], color=PALETTE[0], linewidth=1.3)
        ax.scatter([row.effect_difference], [index], color=PALETTE[0], s=18, zorder=3)
    labels = [f"{row.method} - {row.reference}" for _, row in selected.iterrows()]
    ax.axvline(0, color="0.6", linestyle="--", linewidth=0.8)
    ax.set(
        yticks=range(len(selected)),
        yticklabels=labels,
        xlabel="Difference in seed-mean F1; pointwise 95% group bootstrap CI",
        title=title,
    )
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, output)


def create_figure_bundle(
    test: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
    evidence_status: str,
) -> dict[str, Any]:
    """Export observed panels only; explicitly record absent robustness/sensitivity inputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    outputs = []
    for condition, part in test.groupby("condition", sort=True):
        subset = summary.loc[summary.condition.eq(condition)]
        title = f"{evidence_status} | {condition}"
        outputs.extend(
            _performance(subset, output_dir / _safe_name(f"performance_{condition}"), title)
        )
        outputs.extend(_curves(part, output_dir / _safe_name(f"roc_pr_{condition}"), title))
        for method, method_rows in part.groupby("method", sort=True):
            outputs.extend(
                _confusion(
                    method_rows,
                    output_dir / _safe_name(f"confusion_{condition}_{method}"),
                    f"{method}\n{title}",
                )
            )
        if not paired.empty:
            outputs.extend(
                _differences(
                    paired.loc[paired.condition.eq(condition)],
                    output_dir / _safe_name(f"paired_f1_{condition}"),
                    title,
                )
            )
    perturbations = [
        value
        for value in test.condition.unique()
        if value not in ("clean",) and not value.startswith("sensitivity:")
    ]
    sensitivity = [value for value in test.condition.unique() if value.startswith("sensitivity:")]
    if perturbations and "clean" in set(test.condition):
        fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_IN, 2.7))
        conditions = ["clean", *sorted(perturbations)]
        for index, (method, part) in enumerate(summary.groupby("method", sort=True)):
            values = part.set_index("condition").reindex(conditions).f1_mean
            ax.plot(
                range(len(conditions)),
                values,
                "o-",
                color=PALETTE[index % len(PALETTE)],
                label=method,
            )
        ax.set(
            xticks=range(len(conditions)),
            xticklabels=conditions,
            ylabel="Positive-class F1 (seed mean)",
            ylim=(0, 1),
            title=evidence_status,
        )
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=6)
        fig.tight_layout()
        outputs.extend(_save(fig, output_dir / "robustness"))
    report = {
        "status": "generated_from_observed_predictions",
        "files": outputs,
        "backend": "python_matplotlib",
        "archetype": "quantitative_grid",
        "robustness": "generated"
        if perturbations and "clean" in set(test.condition)
        else "not_run_no_paired_clean_and_perturbed_predictions",
        "sensitivity": "per_condition_performance_panels"
        if sensitivity
        else "not_run_no_sensitivity_predictions",
        "data_rows_used": len(test),
        "rows_dropped": 0,
        "n_definition": (
            "Images within scene clusters; identical test images are repeated "
            "across training seeds."
        ),
        "error_bars": (
            "Performance: sample SD of pooled-fold metrics across seeds; absent for one "
            "seed. Paired panel: 95% percentile CI."
        ),
        "confusion": "Mean counts across seeds, never sum counts and inflate sample size.",
        "curves": (
            "One observed pooled-fold curve per method and seed. Single-class curves unavailable."
        ),
        "inference": (
            "CI conditions on trained models and frozen thresholds; no p-values, "
            "significance stars or multiplicity-adjusted inference."
        ),
        "source_data": "../source_predictions.csv; ../tables/per_seed.csv; ../tables/paired_ci.csv",
        "export": (
            "SVG/PDF editable text, 600 dpi PNG preview. "
            "Verify target journal dimensions before submission."
        ),
        "qa_notes": (
            "Source validator cannot evaluate width constants: source widths are "
            "89/25.4 and 183/25.4 inches; tight bounding boxes include labels. "
            "Verify final placement dimensions for the chosen journal."
        ),
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
