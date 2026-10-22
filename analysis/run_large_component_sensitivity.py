"""Authorized leave-one-component-out diagnostic on frozen paired predictions.

This file is a prepared source artifact. Do not run it before the separate
large-component sensitivity authorization has been recorded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = (HERE / "../../split_repair_v2/hn_random_release_20261022_go_v2/workspace").resolve()
PRODUCTS = WORKSPACE / "runs/direction_a/hn_random_control_split_v2_20261022_v2/analysis/execution/attempt_001/products"
APPROVAL = HERE / "large_component_sensitivity_authorization_20261022.json"
SELECTION = HERE / "large_component_selection.csv"
PREDICTIONS = PRODUCTS / "predictions.csv"
PAIRED = PRODUCTS / "paired_ci.csv"
EFFECTS = HERE / "large_component_sensitivity_effects.csv"
SEED_EFFECTS = HERE / "large_component_sensitivity_per_seed.csv"
MANIFEST = HERE / "large_component_sensitivity_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_values(part: list[dict]) -> dict[str, float]:
    tp = sum(int(row["label"]) == 1 and int(row["y_pred"]) == 1 for row in part)
    fp = sum(int(row["label"]) == 0 and int(row["y_pred"]) == 1 for row in part)
    fn = sum(int(row["label"]) == 1 and int(row["y_pred"]) == 0 for row in part)
    tn = sum(int(row["label"]) == 0 and int(row["y_pred"]) == 0 for row in part)
    if not part or tp + fn == 0 or fp + tn == 0 or 2 * tp + fp + fn == 0:
        raise ValueError("Metric denominator undefined after component exclusion")
    return {"f1": 2 * tp / (2 * tp + fp + fn), "recall": tp / (tp + fn), "fpr": fp / (fp + tn)}


def write_once(path: Path, data: list[dict]) -> None:
    if path.exists():
        raise FileExistsError("Refusing to overwrite " + str(path))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    selected = rows(SELECTION)
    expected_groups = ["VR2-0003", "VR2-0001", "VR2-0008"]
    if not (
        approval.get("authorized") is True
        and approval.get("status") == "AUTHORIZED_SENSITIVITY_ONLY"
        and approval.get("project_id") == "paper_fire_detection_20260915"
        and approval.get("release_id") == "hn_random_control_split_v2_20261022_go_v2"
        and approval.get("scope") == ["leave_one_top_three_component"]
        and approval.get("human_confirmation_reference")
        and approval.get("summary_predictions_sha256") == sha(PREDICTIONS)
        and approval.get("paired_ci_sha256") == sha(PAIRED)
        and approval.get("selection_sha256") == sha(SELECTION)
        and approval.get("analysis_source_sha256") == sha(Path(__file__))
        and [row["group_id"] for row in selected] == expected_groups
    ):
        raise PermissionError("Separate sensitivity authorization or frozen identity missing")
    if any(path.exists() for path in (EFFECTS, SEED_EFFECTS, MANIFEST)):
        raise FileExistsError("Sensitivity outputs already exist; no automatic rerun")
    data = rows(PREDICTIONS)
    paired = rows(PAIRED)
    by_method_seed = defaultdict(list)
    for row in data:
        by_method_seed[(row["method"], int(row["seed"]))].append(row)
    views = {"original": ("cls_hn", "cls_random_negative"), "tta": ("cls_hn_tta", "cls_random_negative_tta")}
    seeds = (42, 2026, 3407)
    full = {}
    for view, (hn, random) in views.items():
        for metric in ("f1", "recall", "fpr"):
            deltas = [metric_values(by_method_seed[(hn, seed)])[metric] - metric_values(by_method_seed[(random, seed)])[metric] for seed in seeds]
            full[(view, metric)] = sum(deltas) / 3
            frozen = next(row for row in paired if row["method"] == hn and row["reference"] == random and row["metric"] == metric)
            if not math.isclose(full[(view, metric)], float(frozen["effect_difference"]), rel_tol=0, abs_tol=1e-12):
                raise ValueError("Full-data effect differs from frozen summary")
    detailed = []
    effects = []
    for item in selected:
        group = item["group_id"]
        retained = int(item["retained_images_per_seed"])
        for view, (hn, random) in views.items():
            for metric in ("f1", "recall", "fpr"):
                deltas = []
                for seed in seeds:
                    hn_part = [row for row in by_method_seed[(hn, seed)] if row["group_id"] != group]
                    random_part = [row for row in by_method_seed[(random, seed)] if row["group_id"] != group]
                    if len(hn_part) != retained or len(random_part) != retained:
                        raise ValueError("Paired exclusion cardinality differs")
                    delta = metric_values(hn_part)[metric] - metric_values(random_part)[metric]
                    deltas.append(delta)
                    detailed.append({"group_id": group, "view": view, "metric": metric, "seed": seed, "retained_images": retained, "hn_minus_random": delta})
                excluded = sum(deltas) / 3
                effects.append({
                    "group_id": group, "removed_images": int(item["n_images"]),
                    "removed_positive": int(item["n_positive"]), "removed_negative": int(item["n_negative"]),
                    "retained_images_per_seed": retained, "view": view, "metric": metric,
                    "full_hn_minus_random": full[(view, metric)],
                    "excluded_hn_minus_random": excluded, "shift_from_full": excluded - full[(view, metric)],
                    "interpretation": "exploratory_leave_one_component_out_no_CI",
                })
    write_once(SEED_EFFECTS, detailed)
    write_once(EFFECTS, effects)
    MANIFEST.write_text(json.dumps({
        "status": "EXECUTED_LOCAL_EXPLORATORY",
        "project_id": approval["project_id"], "release_id": approval["release_id"],
        "summary_predictions_sha256": sha(PREDICTIONS), "paired_ci_sha256": sha(PAIRED),
        "selection_sha256": sha(SELECTION), "authorization_sha256": sha(APPROVAL),
        "source_sha256": sha(Path(__file__)), "effects_sha256": sha(EFFECTS),
        "per_seed_sha256": sha(SEED_EFFECTS), "effect_rows": len(effects),
        "per_seed_rows": len(detailed), "new_CI_or_p_value_computed": False,
        "training_cal_test_or_summary_rerun": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SENSITIVITY_EFFECTS_WRITTEN", len(effects), len(detailed))


if __name__ == "__main__":
    main()
