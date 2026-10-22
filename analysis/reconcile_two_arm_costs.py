"""Reconcile frozen HN and random-control costs from accepted node records."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = (HERE / "../..").resolve()
OLD = ROOT / "split_repair_v2/classifier_preflight_B_20261021_v2/workspace"
PARENT = ROOT / "split_repair_v2/convergence_B_20261020_v1/workspace"
NEW = ROOT / "split_repair_v2/hn_random_release_20261022_go_v2/workspace"
OLD_PLAN = OLD / "plans/split_v2_classifier_preflight_B_20261021_v2/plan.json"
PARENT_PLAN = PARENT / "plans/split_v2_converged_B_20261020_v1/plan.json"
NEW_PLAN = NEW / "plans/hn_random_control_split_v2_20261022_go_v2/plan.json"
OLD_COST = ROOT / "split_repair_v2/formal_eval_20261022_v1/compute_tradeoff.csv"
NEW_COST = NEW / "runs/direction_a/hn_random_control_split_v2_20261022_v2/analysis/execution/attempt_001/products/cost_attempts.csv"
LEDGER = HERE / "two_arm_cost_ledger.csv"
PAIRS = HERE / "two_arm_training_budget_pairs.csv"
AUDIT = HERE / "two_arm_cost_reconciliation_audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def near(left, right, tolerance=0.01) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def accepted_sources(workspace: Path, task: dict, imported_tasks: dict | None = None):
    base = workspace / task["output_dir"]
    accepted_path = base / "accepted.json"
    if not accepted_path.is_file() and imported_tasks and task["id"] in imported_tasks:
        workspace = PARENT
        task = imported_tasks[task["id"]]
        base = workspace / task["output_dir"]
        accepted_path = base / "accepted.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    record_path = base / "execution/attempt_001/record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    manifests = [workspace / name for name in accepted["artifacts"] if name.endswith("/research_manifest.json")]
    budgets = [workspace / name for name in accepted["artifacts"] if name.endswith("/measured_budget.json")]
    for source in [record_path, *manifests, *budgets]:
        key = source.relative_to(workspace).as_posix()
        if not source.is_file() or accepted["artifacts"].get(key) != sha(source):
            raise ValueError("Accepted cost-source artifact changed: " + key)
    manifest = json.loads(manifests[0].read_text(encoding="utf-8")) if len(manifests) == 1 else None
    budget = json.loads(budgets[0].read_text(encoding="utf-8")) if len(budgets) == 1 else None
    return accepted_path, accepted, record_path, record, manifest, budget


def main() -> None:
    old_plan = json.loads(OLD_PLAN.read_text(encoding="utf-8"))
    parent_plan = json.loads(PARENT_PLAN.read_text(encoding="utf-8"))
    new_plan = json.loads(NEW_PLAN.read_text(encoding="utf-8"))
    old_tasks = {task["id"]: task for task in old_plan["tasks"]}
    imported_ids = set(old_plan["imported_completed"]["task_ids"])
    imported_tasks = {task["id"]: task for task in parent_plan["tasks"] if task["id"] in imported_ids}
    new_tasks = {task["id"]: task for task in new_plan["tasks"]}
    old_rows = [row for row in csv_rows(OLD_COST) if row["method"] in ("hn", "HN_shared")]
    new_rows = csv_rows(NEW_COST)
    issues: list[str] = []
    ledger: list[dict] = []
    old_phase_kind = {
        "classifier_train": "train", "oof_miner": "miner", "oof_selection": "select",
        "cal_inference": "calibrate", "test_inference": "evaluate",
    }
    for row in old_rows:
        task_id = row["task"]
        task = old_tasks.get(task_id)
        if task is None or task["kind"] != old_phase_kind[row["phase"]]:
            issues.append("HN task identity/kind " + task_id)
            continue
        accepted_path, accepted, record_path, record, manifest, budget = accepted_sources(OLD, task, imported_tasks)
        if sha(accepted_path) != row["source_accepted_sha256"] or accepted["task"] != task_id or record["task"] != task_id or record["status"] != "SUCCEEDED":
            issues.append("HN accepted receipt/status " + task_id)
        if task_id in imported_ids and sha(accepted_path) != old_plan["imported_completed"]["accepted_sha256"][task_id]:
            issues.append("HN imported parent receipt mismatch " + task_id)
        if not near(record["elapsed_seconds"], row["node_wall_seconds"]):
            issues.append("HN node-wall mismatch " + task_id)
        trainer = float(row["trainer_seconds"]) if row["trainer_seconds"] else None
        if trainer is not None and (manifest is None or not near(manifest["elapsed_seconds"], trainer)):
            issues.append("HN trainer-time mismatch " + task_id)
        updates = int(float(row["updates"])) if row["updates"] else None
        exposures = int(float(row["exposures"])) if row["exposures"] else None
        if task["kind"] == "train":
            actual = budget["actual_counters"] if budget else {}
            if updates != actual.get("optimizer_steps") or exposures != actual.get("sample_exposures"):
                issues.append("HN measured budget mismatch " + task_id)
        ledger.append({
            "arm": "HN", "phase": row["phase"], "task_id": task_id,
            "outer": row["outer"], "seed": row["seed"],
            "node_wall_seconds": float(row["node_wall_seconds"]),
            "trainer_seconds_nested_not_additive": trainer,
            "classifier_or_miner_updates": updates,
            "classifier_or_miner_exposures": exposures,
            "sharing": row["sharing"], "accepted_sha256": sha(accepted_path),
            "record_sha256": sha(record_path), "source_table": "formal_eval_20261022_v1/compute_tradeoff.csv",
        })
    kind_phase = {"train": "classifier_train", "calibrate": "cal_inference", "evaluate": "test_inference"}
    for row in new_rows:
        task_id = row["task"]
        task = new_tasks.get(task_id)
        if task is None or task["kind"] != row["kind"] or task["kind"] not in kind_phase or row["method"] != "random_negative":
            issues.append("Random task identity/kind " + task_id)
            continue
        accepted_path, accepted, record_path, record, manifest, budget = accepted_sources(NEW, task)
        if accepted["task"] != task_id or record["task"] != task_id or record["status"] != "SUCCEEDED":
            issues.append("Random accepted receipt/status " + task_id)
        if not near(record["elapsed_seconds"], row["elapsed_seconds"], 1e-6):
            issues.append("Random node-wall mismatch " + task_id)
        updates = int(float(row["actual_optimizer_updates"])) if row["actual_optimizer_updates"] else None
        exposures = int(float(row["actual_sample_exposures"])) if row["actual_sample_exposures"] else None
        if task["kind"] == "train":
            actual = budget["actual_counters"] if budget else {}
            if manifest is None or updates != actual.get("optimizer_steps") or exposures != actual.get("sample_exposures"):
                issues.append("Random measured budget/trainer manifest mismatch " + task_id)
        ledger.append({
            "arm": "Random", "phase": kind_phase[task["kind"]], "task_id": task_id,
            "outer": task["outer_fold"], "seed": task["seed"],
            "node_wall_seconds": float(row["elapsed_seconds"]),
            "trainer_seconds_nested_not_additive": manifest["elapsed_seconds"] if task["kind"] == "train" and manifest else None,
            "classifier_or_miner_updates": updates,
            "classifier_or_miner_exposures": exposures,
            "sharing": "per_model", "accepted_sha256": sha(accepted_path),
            "record_sha256": sha(record_path), "source_table": "GO_v2/analysis/cost_attempts.csv",
        })
    if len(old_rows) != 65 or len(new_rows) != 45 or len(ledger) != 110:
        issues.append("Expected 65 HN plus 45 random accepted cost nodes")
    write_csv(LEDGER, ledger)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in ledger:
        grouped[(row["arm"], row["phase"])].append(row)
    totals = {}
    for arm in ("HN", "Random"):
        totals[arm] = {}
        for phase in ("oof_miner", "oof_selection", "classifier_train", "cal_inference", "test_inference"):
            part = grouped[(arm, phase)]
            totals[arm][phase] = {
                "nodes": len(part),
                "node_wall_seconds": sum(row["node_wall_seconds"] for row in part),
                "trainer_seconds_nested_not_additive": sum(row["trainer_seconds_nested_not_additive"] or 0 for row in part),
                "updates": sum(row["classifier_or_miner_updates"] or 0 for row in part),
                "exposures": sum(row["classifier_or_miner_exposures"] or 0 for row in part),
            }
        totals[arm]["full_recorded_pipeline_node_wall_seconds"] = sum(row["node_wall_seconds"] for row in ledger if row["arm"] == arm)
    hn_train = {(int(row["outer"]), int(row["seed"])): row for row in grouped[("HN", "classifier_train")]}
    random_train = {(int(row["outer"]), int(row["seed"])): row for row in grouped[("Random", "classifier_train")]}
    pairs = []
    if set(hn_train) != set(random_train) or len(hn_train) != 15:
        issues.append("Training pair identity mismatch")
    for key in sorted(set(hn_train) & set(random_train)):
        h, r = hn_train[key], random_train[key]
        if (h["classifier_or_miner_updates"], h["classifier_or_miner_exposures"]) != (r["classifier_or_miner_updates"], r["classifier_or_miner_exposures"]):
            issues.append("Training exposure/update mismatch " + str(key))
        pairs.append({
            "outer": key[0], "seed": key[1],
            "hn_task": h["task_id"], "random_task": r["task_id"],
            "hn_updates": h["classifier_or_miner_updates"], "random_updates": r["classifier_or_miner_updates"],
            "hn_exposures": h["classifier_or_miner_exposures"], "random_exposures": r["classifier_or_miner_exposures"],
            "hn_node_wall_seconds": h["node_wall_seconds"], "random_node_wall_seconds": r["node_wall_seconds"],
            "hn_minus_random_node_seconds": h["node_wall_seconds"] - r["node_wall_seconds"],
        })
    write_csv(PAIRS, pairs)
    result = {
        "status": "TWO_ARM_COST_RECONCILED_WITH_SCOPE_LIMITS" if not issues else "TWO_ARM_COST_AUDIT_FAILED",
        "project_id": "paper_fire_detection_20260915",
        "hn_release_id": old_plan["release_id"], "random_release_id": new_plan["release_id"],
        "hn_cost_source_sha256": sha(OLD_COST), "random_cost_source_sha256": sha(NEW_COST),
        "hn_plan_sha256": sha(OLD_PLAN), "hn_imported_parent_plan_sha256": sha(PARENT_PLAN), "random_plan_sha256": sha(NEW_PLAN),
        "ledger_sha256": sha(LEDGER), "training_pairs_sha256": sha(PAIRS),
        "hn_accepted_nodes": len(old_rows), "random_accepted_nodes": len(new_rows),
        "matched_training_pairs": len(pairs), "totals": totals,
        "full_pipeline_hn_minus_random_node_seconds": totals["HN"]["full_recorded_pipeline_node_wall_seconds"] - totals["Random"]["full_recorded_pipeline_node_wall_seconds"],
        "full_pipeline_hn_to_random_node_ratio": totals["HN"]["full_recorded_pipeline_node_wall_seconds"] / totals["Random"]["full_recorded_pipeline_node_wall_seconds"],
        "scope_limits": [
            "Node wall time includes setup/loading and is not GPU kernel time or energy",
            "Trainer seconds are nested within classifier/miner node time and must not be added",
            "HN shared mining/selection is once per outer fold, shared across three classifier seeds",
            "Random identity-design/preparation and operator idle time are outside the 45-node random release cost table",
            "The two releases ran at different times; observed duration differences are not causal treatment effects",
        ],
        "model_execution_performed": False,
        "issues": issues,
    }
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
