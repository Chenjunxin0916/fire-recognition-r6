"""Prepare an isolated Route B CPU candidate; never launch a model node."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCI = ROOT.parents[2]
OLD = SCI / "split_repair_v2/classifier_preflight_B_20261021_v2/workspace"
FORMAL = SCI / "split_repair_v2/formal_eval_20261022_v1"
TRAINING = SCI / "split_repair_v2/remaining_training_20261021"
DESIGN = SCI / "Paper/hn_random_route_b_20261022"
RELEASE_ID = "hn_random_control_split_v2_20261022_v1"
CAMPAIGN = "hn_random_control_split_v2_20261022_v1"
RELEASE = ROOT / "plans" / RELEASE_ID
CONFIG = ROOT / "configs" / RELEASE_ID
EXPECTED_FOLD_SHA = "3446cbc7f604828ca5120dfe6d8317d6ca96a89a58e1ecd8581f2bfc6845a88c"
EXPECTED_HN_PRED_SHA = "d21c4f2a76136b0dd30147b26efaf970223d895ee90c6db575fa99f8e3190897"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_once(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(source, target)


def task(kind: str, outer: int | None, seed: int | None, *, depends_on: list[str],
         expected_epochs=None, expected_steps=None, expected_exposures=None) -> dict:
    method = "random_negative" if kind != "summarize" else None
    if kind == "train":
        tid = f"train_random_negative_o{outer}_s{seed}"
        suffix = f"outer_{outer}/seed_{seed}/models/random_negative"
    elif kind == "calibrate":
        tid = f"cal_random_negative_o{outer}_s{seed}"
        suffix = f"outer_{outer}/seed_{seed}/calibration/random_negative"
    elif kind == "evaluate":
        tid = f"eval_random_negative_o{outer}_s{seed}"
        suffix = f"outer_{outer}/seed_{seed}/evaluation/random_negative"
    else:
        tid, suffix = "summarize_all", "analysis"
    output = f"runs/direction_a/{CAMPAIGN}/{suffix}"
    return {
        "id": tid,
        "kind": kind,
        "outer_fold": outer,
        "seed": seed,
        "method": method,
        "inner_fold": None,
        "depends_on": depends_on,
        "config": f"configs/{RELEASE_ID}/base.yaml",
        "runtime_config": f"configs/{RELEASE_ID}/runtime.yaml",
        "output_dir": output,
        "log_pattern": output + "/execution/attempt_NNN/stdout.log",
        "status": "PENDING",
        "expected_epochs": expected_epochs,
        "expected_optimizer_steps": expected_steps,
        "expected_sample_exposures": expected_exposures,
        "exposure_expression": None,
        "budget_status": "DEFINED" if kind == "train" else "NOT_APPLICABLE",
        "implementation_status": "CPU_CANDIDATE_NOT_EXECUTED",
    }


def source_snapshot() -> str:
    manifest_path = ROOT / "provenance/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["release_id"] != "split_v2_classifier_preflight_B_20261021_v2":
        raise ValueError("Parent source provenance differs")
    execution = {}
    for name in ("src", "scripts", "configs", "requirements", "environment"):
        for path in (ROOT / name).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(ROOT).as_posix()
            execution[relative] = sha(path)
            snapshot = ROOT / "provenance/execution_snapshot" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, snapshot)
    manifest["release_id"] = RELEASE_ID
    manifest["parent_release_id"] = "split_v2_classifier_preflight_B_20261021_v2"
    manifest["execution_files"] = dict(sorted(execution.items()))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sha(manifest_path)


def seal_files() -> dict[str, str]:
    files = {}
    for name in ("src", "scripts", "configs", "data", "weights", "requirements", "environment", "provenance", "imports"):
        for path in (ROOT / name).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            files[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (RELEASE / "partitions").rglob("*.json"):
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (ROOT / "runs/direction_a" / CAMPAIGN).rglob("*"):
        if path.is_file() and path.name in {"hard_negative_selection.json", "paired_initialization.json"}:
            files[path.relative_to(ROOT).as_posix()] = sha(path)
    return dict(sorted(files.items()))


def main() -> None:
    if RELEASE.exists() or CONFIG.exists():
        raise FileExistsError("Route B release/config already exists; do not overwrite")
    old_plan = json.loads((OLD / "plans/split_v2_classifier_preflight_B_20261021_v2/plan.json").read_text(encoding="utf-8"))
    old_audit = json.loads((TRAINING / "final_independent_audit.json").read_text(encoding="utf-8"))
    design = json.loads((DESIGN / "design_receipt.json").read_text(encoding="utf-8"))
    if old_audit["status"] != "PASS" or old_plan["source_sha256"] != old_audit["source_sha256"]:
        raise ValueError("Parent training identity is not accepted")
    if sha(OLD / "data/processed/dfire_eoi_snapshot_20261020_v1/folds.csv") != EXPECTED_FOLD_SHA:
        raise ValueError("Parent frozen fold hash differs")
    if sha(FORMAL / "test_predictions.csv") != EXPECTED_HN_PRED_SHA:
        raise ValueError("Formal HN prediction import hash differs")
    matrix = list(csv.DictReader((DESIGN / "paired_budget_matrix.csv").open(newline="", encoding="utf-8")))
    random_rows = list(csv.DictReader((DESIGN / "random_identities_INTERNAL_CANDIDATE.csv").open(newline="", encoding="utf-8")))
    if len(matrix) != 5 or len(random_rows) != 346 or design["random_identities_internal_sha256"] != sha(DESIGN / "random_identities_INTERNAL_CANDIDATE.csv"):
        raise ValueError("Candidate random design changed")
    CONFIG.mkdir(parents=True)
    RELEASE.mkdir(parents=True)
    base = yaml.safe_load((OLD / "configs/split_v2_classifier_preflight_B_20261021_v2/base.yaml").read_text(encoding="utf-8"))
    base["campaign"] = CAMPAIGN
    base["arms"] = ["hn", "random_negative"]
    (CONFIG / "base.yaml").write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    copy_once(OLD / "configs/split_v2_classifier_preflight_B_20261021_v2/runtime.yaml", CONFIG / "runtime.yaml")
    evaluation = {
        "schema_version": 1,
        "study_kind": "HN_RANDOM_CONTROL",
        "primary": "original_no_TTA",
        "secondary": "horizontal_flip_average_TTA",
        "bootstrap": {"iterations": 10000, "seed": 20260914},
        "inference": "exploratory_post_outcome_internal_extension",
    }
    (CONFIG / "evaluation.yaml").write_text(yaml.safe_dump(evaluation, allow_unicode=True, sort_keys=False), encoding="utf-8")
    IMPORTS = ROOT / "imports"
    IMPORTS.mkdir()
    for source, name in (
        (FORMAL / "test_predictions.csv", "hn_formal_test_predictions.csv"),
        (FORMAL / "cal_threshold_manifest.csv", "hn_formal_cal_threshold_manifest.csv"),
        (FORMAL / "final_acceptance_check.json", "hn_formal_acceptance.json"),
        (TRAINING / "final_independent_audit.json", "hn_training_audit.json"),
        (TRAINING / "model_identity_matrix.csv", "hn_model_identity_matrix.csv"),
        (OLD / "plans/split_v2_classifier_preflight_B_20261021_v2/plan.json", "parent_plan.json"),
    ):
        copy_once(source, IMPORTS / name)
    hn_selection_hashes, random_hashes, random_files, partition_digests = {}, {}, {}, {}
    for outer in range(5):
        campaign = "focal_split_v2_converged_b_20261020_v1" if outer == 0 else "focal_split_v2_classifier_preflight_b_20261021_v2"
        source_h = OLD / "runs/direction_a" / campaign / f"outer_{outer}/seed_42/hard_negative_selection.json"
        target_h = ROOT / "runs/direction_a" / CAMPAIGN / f"outer_{outer}/seed_42/hard_negative_selection.json"
        copy_once(source_h, target_h)
        hn_selection_hashes[str(outer)] = sha(target_h)
        candidate = [r["filename"] for r in random_rows if int(r["outer"]) == outer]
        if len(candidate) != int(matrix[outer]["K"]):
            raise ValueError("Candidate K differs")
        random_path = IMPORTS / f"random_identities_outer_{outer}.json"
        write_once(random_path, {"outer": outer, "K": len(candidate), "identity_seed": 20260914, "filenames": candidate, "status": "CANDIDATE_NOT_TRAINED"})
        random_hashes[str(outer)] = sha(random_path)
        random_files[str(outer)] = random_path.relative_to(ROOT).as_posix()
        for seed in (42, 2026, 3407):
            source_partition = OLD / "plans/split_v2_classifier_preflight_B_20261021_v2" / f"partitions/outer_{outer}/seed_{seed}.json"
            target_partition = RELEASE / f"partitions/outer_{outer}/seed_{seed}.json"
            copy_once(source_partition, target_partition)
            name = target_partition.relative_to(RELEASE).as_posix()
            partition_digests[name] = old_plan["partition_digests"][name]
            source_init = OLD / "runs/direction_a/focal_split_v2_classifier_preflight_b_20261021_v2" / f"outer_{outer}/seed_{seed}/paired_initialization.json"
            target_init = ROOT / "runs/direction_a" / CAMPAIGN / f"outer_{outer}/seed_{seed}/paired_initialization.json"
            copy_once(source_init, target_init)
    tasks = []
    for row in matrix:
        outer = int(row["outer"])
        for seed in (42, 2026, 3407):
            tasks.append(task("train", outer, seed, depends_on=[], expected_epochs=40,
                              expected_steps=int(row["expected_updates_per_classifier"]),
                              expected_exposures=int(row["expected_exposures_per_classifier"])))
    for outer in range(5):
        for seed in (42, 2026, 3407):
            tasks.append(task("calibrate", outer, seed, depends_on=[f"train_random_negative_o{outer}_s{seed}"]))
    all_cal = [t["id"] for t in tasks if t["kind"] == "calibrate"]
    for outer in range(5):
        for seed in (42, 2026, 3407):
            tasks.append(task("evaluate", outer, seed, depends_on=all_cal))
    tasks.append(task("summarize", None, None,
                      depends_on=[t["id"] for t in tasks if t["kind"] == "evaluate"]))
    plan = {
        "schema_version": 1,
        "study_kind": "HN_RANDOM_CONTROL",
        "release_id": RELEASE_ID,
        "project_id": "paper_fire_detection_20260915",
        "campaign": CAMPAIGN,
        "status": "CPU_CANDIDATE_NO_EXECUTION_AUTHORIZATION",
        "authorized": False,
        "author_protocol_approved": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "methods": ["random_negative"],
        "outer_folds": list(range(5)),
        "seeds": [42, 2026, 3407],
        "paired_K_by_outer": [int(r["K"]) for r in matrix],
        "training_count": 15,
        "calibration_count": 15,
        "evaluation_count": 15,
        "summary_count": 1,
        "tasks": tasks,
        "base_config": f"configs/{RELEASE_ID}/base.yaml",
        "evaluation_config": f"configs/{RELEASE_ID}/evaluation.yaml",
        "source_sha256": old_plan["source_sha256"],
        "input_snapshot_sha256": old_plan["input_snapshot_sha256"],
        "budget_policy": "B_FULL_40_EPOCHS",
        "budget_policy_version": "FULL-STRATEGY-B-40E-20261020-V1",
        "training_protocol_version": "FULL-STRATEGY-B-40E-20261020-V1",
        "partition_digests": partition_digests,
        "hn_selection_file_sha_by_outer": hn_selection_hashes,
        "random_identity_files": random_files,
        "random_identity_file_sha_by_outer": random_hashes,
        "imported_hn_predictions": "imports/hn_formal_test_predictions.csv",
        "imported_hn_predictions_sha256": EXPECTED_HN_PRED_SHA,
        "parent_release_id": old_plan["release_id"],
        "parent_plan_sha256": sha(IMPORTS / "parent_plan.json"),
        "hardware_observation_sha256": None,
        "execution_authorization_present": False,
        "blocking_issues": ["New release requires full CPU preflight, author protocol review, hardware confirmation, and staged execution authorization"],
        "imported_completed": {},
    }
    write_once(RELEASE / "plan.json", plan)
    write_once(RELEASE / "task_state.json", {
        "schema_version": 1, "release_id": RELEASE_ID, "revision": 0,
        "tasks": {t["id"]: {"status": "PENDING", "accepted_attempt": None, "attempts": []} for t in tasks},
    })
    manifest_sha = source_snapshot()
    seal = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "seal_type": "CPU_CANDIDATE_NO_EXECUTION_AUTHORIZATION",
        "files": seal_files(),
        "input_snapshot_sha256": old_plan["input_snapshot_sha256"],
    }
    write_once(RELEASE / "seal.json", seal)
    print(json.dumps({"status": "CPU_CANDIDATE_NOT_EXECUTED", "release_id": RELEASE_ID,
                      "tasks": len(tasks), "sealed_files": len(seal["files"]),
                      "source_provenance_sha256": manifest_sha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
