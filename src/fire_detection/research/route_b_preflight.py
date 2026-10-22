"""CPU-only Route B production-path checks; no model execution."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from .three_arm_budget import BudgetBlocked, require_random_budget
from .three_arm_recovery import atomic, read, sha, validate_hn_random_plan


def run_preflight(engine, selected, preflight_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", preflight_id):
        raise ValueError("Preflight identity must be a short alphanumeric token")
    if selected and not set(selected) <= set(engine.tasks):
        raise ValueError("Preflight selected unknown task")
    if engine.plan.get("study_kind") != "HN_RANDOM_CONTROL":
        raise ValueError("Wrong study kind for Route B preflight")
    from .protocol import digest_object, load_config, verify_source

    config = load_config(engine.root / engine.plan["base_config"], root=engine.root)
    _, source = verify_source(config)
    if digest_object(source) != engine.plan["source_sha256"]:
        raise ValueError("CPU preflight source identity differs from execution plan")
    root = engine.root
    rows = []
    negatives = []
    for outer in range(5):
        partition = read(engine.release / f"partitions/outer_{outer}/seed_42.json")
        fit = partition["splits"]["fit"]
        selection_path = root / "runs/direction_a" / engine.plan["campaign"] / f"outer_{outer}/seed_42/hard_negative_selection.json"
        if sha(selection_path) != engine.plan["hn_selection_file_sha_by_outer"][str(outer)]:
            raise ValueError("Imported HN selection drift")
        selection = read(selection_path)
        hard = selection["hard_names"]
        candidate_path = root / engine.plan["random_identity_files"][str(outer)]
        if sha(candidate_path) != engine.plan["random_identity_file_sha_by_outer"][str(outer)]:
            raise ValueError("Random candidate identity drift")
        candidate = read(candidate_path)
        budget = require_random_budget(
            fit, hard,
            random_seed=config["scheme_b"]["random_negative_identity_seed"],
            expected_extra_names=candidate["filenames"],
            policy=config["budget_policy"],
        )
        if (budget["K"], budget["sample_exposures"], budget["optimizer_updates"]) != (
            engine.plan["paired_K_by_outer"][outer],
            40 * (660 + engine.plan["paired_K_by_outer"][outer]),
            40 * ((660 + engine.plan["paired_K_by_outer"][outer] + 63) // 64),
        ):
            raise ValueError("Random control differs from frozen HN budget")
        rows.append({"outer": outer, "fit": len(fit), "K": budget["K"],
                     "exposures": budget["sample_exposures"], "updates": budget["optimizer_updates"],
                     "identity_check": "PASS"})
        altered = candidate["filenames"][:-1]
        try:
            require_random_budget(
                fit, hard,
                random_seed=config["scheme_b"]["random_negative_identity_seed"],
                expected_extra_names=altered, policy=config["budget_policy"],
            )
        except BudgetBlocked:
            negatives.append({"outer": outer, "mutation": "truncate_random_identity", "rejected": True})
        else:
            raise ValueError("Random identity mutation was not rejected")
    changed = copy.deepcopy(engine.plan)
    changed["methods"] = ["hn", "random_negative"]
    try:
        validate_hn_random_plan(changed)
    except ValueError:
        negatives.append({"mutation": "add_unplanned_method", "rejected": True})
    else:
        raise ValueError("Unplanned method mutation was not rejected")
    import importlib.util

    runner_path = root / "scripts/run_sealed_three_arm_v2.py"
    spec = importlib.util.spec_from_file_location("route_b_runner_preflight", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    try:
        runner.require_authorization(engine, ["train_random_negative_o0_s42"], None)
    except PermissionError:
        negatives.append({"mutation": "no_execution_approval", "rejected": True})
    else:
        raise ValueError("Missing execution approval was not rejected")
    if list(root.glob("runs/direction_a/*/outer_*/seed_*/models/*/attempts/*/weights/last.pt")):
        raise ValueError("New release unexpectedly contains model training output")
    result = {
        "status": "CPU_PREFLIGHT_PASS_NO_MODEL_EXECUTION",
        "kind": "TEST_ONLY_ENGINEERING",
        "release_id": engine.plan["release_id"],
        "release_fingerprint": engine.identity,
        "plan_sha256": sha(engine.release / "plan.json"),
        "seal_sha256": sha(engine.release / "seal.json"),
        "fold_checks": rows,
        "negative_checks": negatives,
        "training_executed": False,
        "cal_test_executed": False,
        "result_claim_allowed": False,
    }
    path = engine.release / "preflight" / f"{preflight_id}.json"
    if path.exists():
        raise FileExistsError(path)
    atomic(path, result)
    print(json.dumps({"status": result["status"], "folds": len(rows), "negative_checks": len(negatives)}, ensure_ascii=False))
    return 0
