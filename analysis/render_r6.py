"""Render R6 working figures from frozen real outputs and approved project style."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


FIG = Path(__file__).resolve().parent
ROOT = FIG.parents[2]
FORMAL = ROOT / "split_repair_v2/formal_eval_20261022_v1"
RANDOM = ROOT / "split_repair_v2/hn_random_release_20261022_go_v2/workspace/runs/direction_a/hn_random_control_split_v2_20261022_v2/analysis/execution/attempt_001/products"
ROUTE = ROOT / "Paper/hn_random_route_b_20261022"
SOURCES = {
    "formal_paired": FORMAL / "paired_effects.csv",
    "formal_seed_fold": FORMAL / "seed_fold_results.csv",
    "random_paired": RANDOM / "paired_ci.csv",
    "random_seed": RANDOM / "per_seed.csv",
    "sensitivity": ROUTE / "large_component_sensitivity_effects.csv",
    "plan": ROOT / "split_repair_v2/hn_random_release_20261022_go_v2/workspace/plans/hn_random_control_split_v2_20261022_go_v2/plan.json",
}
assert all(path.is_file() for path in SOURCES.values())
plan = json.loads(SOURCES["plan"].read_text(encoding="utf-8"))
assert plan["paired_K_by_outer"] == [41, 112, 64, 70, 59]
formal_ci = pd.read_csv(SOURCES["formal_paired"])
formal_seed = pd.read_csv(SOURCES["formal_seed_fold"])
random_ci = pd.read_csv(SOURCES["random_paired"])
random_seed = pd.read_csv(SOURCES["random_seed"])
sensitivity = pd.read_csv(SOURCES["sensitivity"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 10,
    "axes.labelsize": 8.4, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": .7,
    "savefig.facecolor": "white",
})
BLUE, ORANGE, PURPLE, INK, LIGHT = "#0072B2", "#D55E00", "#7A5195", "#263238", "#EAF0F4"
QA = FIG / "qa"
QA.mkdir(exist_ok=True)
trace: list[dict] = []


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(fig, name: str) -> None:
    for ext in ("pdf", "svg", "png"):
        meta = {"Date": None} if ext == "svg" else {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, metadata=meta)
    fig.savefig(FIG / f"{name}_preview.png", dpi=170)
    plt.close(fig)
    with Image.open(FIG / f"{name}_preview.png") as im:
        ImageOps.grayscale(im).save(QA / f"{name}_gray.png")
        im.resize((im.width // 2, im.height // 2)).save(QA / f"{name}_half.png")


def box(ax, x, y, w, h, title, body, *, fill="white", edge="#CBD5DB", title_color=INK, fontsize=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.014", linewidth=.9, edgecolor=edge, facecolor=fill))
    ax.text(x + .017, y + h - .04, title, va="top", ha="left", color=title_color, fontweight="bold", fontsize=9.2)
    ax.text(x + .017, y + h - .10, body, va="top", ha="left", color=INK, fontsize=fontsize, linespacing=1.3)


# Figure 1: explanatory study map. Geometry is new; visual grammar reuses the approved project style.
fig, ax = plt.subplots(figsize=(7.09, 3.8))
fig.subplots_adjust(left=.025, right=.975, top=.97, bottom=.075)
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
ax.text(.01, .985, "Revised-split internal revalidation", fontsize=12, fontweight="bold", color=INK, va="top")
box(ax, .02, .45, .25, .43, "A  Frozen cohort", "1,100 D-Fire images\n427 visual components\n5 folds × 220 images\nFIT 660 · CAL 220 · TEST 220", fill="#F3F7F9")
box(ax, .32, .45, .33, .43, "B  Formal comparison", "Standard · HN · Focal Sampling\n5 outer folds × 3 seeds\nHN: FIT-only OOF mining + replay\nClassifier budgets differ by arm", fill="#EFF5F9", edge="#B6CFDD", title_color=BLUE)
box(ax, .70, .45, .28, .43, "C  Later control", "Fixed random-negative replay\nSame FIT and K as HN\nMatched updates and exposures\nAdded after HN TEST was viewed", fill="#FFF4ED", edge="#E4B8A3", title_color=ORANGE, fontsize=7.8)
for x1, x2 in ((.275, .305), (.655, .685)):
    ax.add_patch(FancyArrowPatch((x1, .67), (x2, .67), arrowstyle="-|>", mutation_scale=11, color="#647784", lw=1))
box(ax, .02, .10, .96, .23, "Locked evaluation roles", "CAL monitors training; after the fixed checkpoint it selects view-specific thresholds\nEach frozen model and threshold has one final TEST evaluation; pool folds within seed\nPair by 427 visual components; intervals remain exploratory", fill="#F6F7F8", fontsize=7.3)
ax.text(.02, .025, "Conservative visual components are not authenticated cameras or events; no external test set is included.", fontsize=7.4, color="#44525D")
export(fig, "Figure1_study_design")


# Figure 2: retained exploratory paired estimates, with formal and later control visibly distinct.
comparisons = [
    ("HN − Standard\nformal", formal_ci, "cls_hn", "cls_standard", BLUE),
    ("HN − Focal\nformal", formal_ci, "cls_hn", "cls_focal", PURPLE),
    ("HN − Random\nlater control", random_ci, "cls_hn", "cls_random_negative", ORANGE),
]
fig, axes = plt.subplots(1, 3, figsize=(7.09, 3.35), sharey=True)
fig.subplots_adjust(left=.19, right=.99, top=.88, bottom=.31, wspace=.34)
for ax, metric, title, direction in zip(axes, ("f1", "recall", "fpr"), ("a  F1", "b  Fire recall", "c  False-positive rate"), ("Higher is better →", "Higher is better →", "← Lower is better")):
    for j, (label, table, method, reference, color) in enumerate(comparisons):
        row = table.loc[(table.method == method) & (table.reference == reference) & (table.metric == metric)]
        assert len(row) == 1
        value = row.iloc[0]
        estimate, lower, upper = (float(value[key]) * 100 for key in ("effect_difference", "ci_lower", "ci_upper"))
        y = 2 - j
        ax.plot([lower, upper], [y, y], color=color, lw=1.65)
        ax.plot([lower, lower], [y - .065, y + .065], color=color, lw=.85)
        ax.plot([upper, upper], [y - .065, y + .065], color=color, lw=.85)
        ax.scatter(estimate, y, s=29, color=color, edgecolor="white", linewidth=.4, zorder=3)
        trace.append({"figure": "Figure2_paired_effects", "panel": metric, "comparison": label.replace("\n", " "), "estimate_pp": estimate, "lower_pp": lower, "upper_pp": upper})
    ax.axvline(0, color="#59656E", linestyle="--", lw=.8)
    ax.axhline(.5, color="#D0D9DE", lw=.65)
    ax.set(yticks=[2, 1, 0], yticklabels=[item[0] for item in comparisons], ylim=(-.48, 2.48), title=title)
    ax.grid(axis="x", color="#E8ECEF", lw=.6)
    ax.set_xlabel("HN minus comparator (pp)\n" + direction, labelpad=7)
    ax.locator_params(axis="x", nbins=4)
    ax.margins(x=.15)
fig.text(.19, .045, "Original view · points: paired mean differences · lines: exploratory 95% component intervals.\nThe random control was added after HN TEST had been viewed; intervals do not establish external validity.", fontsize=6.9, color="#495760")
export(fig, "Figure2_paired_effects")


# Figure 3: seed heterogeneity and a point-estimate-only leverage diagnostic.
seeds = (42, 2026, 3407)
formal_seed = formal_seed.loc[formal_seed.analysis_level.eq("seed_pooled")]
fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.05))
fig.subplots_adjust(left=.09, right=.985, top=.86, bottom=.26, wspace=.34)
ax = axes[0]
for label, table, reference, color, offset, marker in (
    ("HN − Standard · formal", formal_seed, "cls_standard", BLUE, -.09, "o"),
    ("HN − Random · later", random_seed, "cls_random_negative", ORANGE, .09, "s"),
):
    hn = table.loc[table.method.eq("cls_hn")].set_index("seed").loc[list(seeds)]
    ref = table.loc[table.method.eq(reference)].set_index("seed").loc[list(seeds)]
    values = (hn.f1 - ref.f1).to_numpy() * 100
    for j, (seed, value) in enumerate(zip(seeds, values)):
        ax.scatter(j + offset, value, marker=marker, color=color, s=42, zorder=3)
        trace.append({"figure": "Figure3_heterogeneity_leverage", "panel": "seed", "comparison": label, "seed": seed, "difference_pp": float(value)})
    ax.plot([], [], marker=marker, color=color, linestyle="none", label=label)
ax.axhline(0, color="#59656E", linestyle="--", lw=.8)
ax.set(xticks=range(3), xticklabels=seeds, xlim=(-.45, 2.45), xlabel="Training seed", ylabel="Pooled F1 difference (pp)", title="a  Seed-level contrasts")
ax.grid(axis="y", color="#E8ECEF", lw=.6)
ax.legend(loc="upper left", fontsize=7.1, frameon=False)
ax = axes[1]
subset = sensitivity.loc[(sensitivity.view == "original") & (sensitivity.metric == "f1")]
order = ("VR2-0003", "VR2-0001", "VR2-0008")
assert set(subset.group_id) == set(order)
full = float(subset.full_hn_minus_random.iloc[0]) * 100
ax.axhline(full, color="#59656E", linestyle=":", lw=1, label="Full 1,100 images")
for j, group in enumerate(order):
    row = subset.loc[subset.group_id.eq(group)].iloc[0]
    value = float(row.excluded_hn_minus_random) * 100
    ax.scatter(j, value, color=ORANGE, s=43, zorder=3)
    trace.append({"figure": "Figure3_heterogeneity_leverage", "panel": "leave_one_component", "group_id": group, "removed_images": int(row.removed_images), "difference_pp": value, "full_pp": full})
ax.axhline(0, color="#59656E", linestyle="--", lw=.8)
ax.set(xticks=range(3), xticklabels=("VR2-0003\n189 images", "VR2-0001\n49 images", "VR2-0008\n49 images"), xlim=(-.45, 2.45), xlabel="Component removed from both arms", ylabel="HN − random F1 (pp)", title="b  Largest-component leverage")
ax.grid(axis="y", color="#E8ECEF", lw=.6)
ax.legend(loc="lower right", fontsize=7.1, frameon=False)
fig.text(.09, .035, "Each seed reuses the same test images. Leave-one-component-out values are diagnostic point estimates, not new CIs.", fontsize=7.0, color="#495760")
export(fig, "Figure3_heterogeneity_leverage")


pd.DataFrame(trace).to_csv(QA / "figure_values.csv", index=False)
provenance = {
    "status": "RENDERED_WORKING_FIGURES_NOT_PUBLICATION_APPROVED",
    "project_id": "paper_fire_detection_20260915",
    "task_id": "R6-INTERNAL-TRADEOFFS-FULL-WORKING-DRAFT-20261022",
    "reference_mode": "PROJECT_REUSE",
    "figure_classes": {"Figure1_study_design": "EXPLANATORY", "Figure2_paired_effects": "RESULT", "Figure3_heterogeneity_leverage": "RESULT"},
    "input_sha256": {name: sha(path) for name, path in SOURCES.items()},
    "transformations": ["Rate and difference axes multiplied by 100 to percentage points", "Formal and later-control CIs read unchanged from their separate frozen tables", "Seed F1 differences computed within matching seed", "Leave-one-component-out point values read from authorized diagnostic table"],
    "source_code_sha256": sha(Path(__file__)),
    "outputs_sha256": {path.name: sha(path) for path in FIG.glob("Figure*.*")},
    "value_trace_sha256": sha(QA / "figure_values.csv"),
    "model_or_bootstrap_execution": False,
    "final_publication_approval": False,
}
(QA / "figure_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Rendered three R6 working figures from frozen sources")
