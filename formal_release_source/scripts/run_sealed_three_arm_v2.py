"""Inspect sealed v2 by default. Execution requires explicit, release-bound scope."""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_detection.research.three_arm_recovery import NodeEngine, read, sha  # noqa: E402


def actual_k_readiness(engine, outer=None):
    """Derive readiness from verified receipts without editing the frozen plan/seal."""
    selections = [t for t in engine.tasks.values() if t["kind"] == "select" and (outer is None or t["outer_fold"] == outer)]
    if len(selections) != (5 if outer is None else 1):
        return "NO_GO"
    for task in selections:
        receipt = engine.completed(task["id"])
        if receipt is None:
            return "NO_GO"
        budget_files = [name for name in receipt["artifacts"] if name.endswith("/budget.json")]
        if len(budget_files) != 1:
            raise ValueError("Selection lacks an unambiguous budget receipt")
        budget = read(engine.root / budget_files[0])
        if not (
            budget.get("budget_gate") == "PASS"
            and type(budget.get("K")) is int
            and 0 <= budget["K"] <= sum(r["label"] == 0 for r in read(engine.release / f"partitions/outer_{task['outer_fold']}/seed_42.json")["splits"]["fit"])
            and budget.get("fit_size") == 660
            and budget.get("budget_policy") == engine.plan["budget_policy"]
            and budget.get("budget_policy_version") == engine.plan["budget_policy_version"]
            and budget.get("optimizer_updates") == 40 * ((660 + budget["K"] + 63) // 64)
            and budget.get("exposures_per_epoch") == 660 + budget["K"]
        ):
            raise ValueError("Actual K/budget is not eligible")
    return "GO"


def require_authorization(engine, selected, approval_path):
    if engine.plan.get("fold_freeze_approved") is not True or engine.plan.get("budget_policy_approved") is not True:
        raise PermissionError("Author candidate review/fold freeze and budget policy approval pending")
    if approval_path is None:
        raise PermissionError(
            "No execution authorized; explicit release-bound authorization required"
        )
    try:
        approval = read(approval_path)
    except (OSError, ValueError) as exc:
        raise PermissionError('Execution authorization missing or unreadable') from exc
    if not (
        approval.get("authorized") is True
        and approval.get("release_id") == engine.plan["release_id"]
        and approval.get("seal_sha256") == sha(engine.release / "seal.json")
        and approval.get("human_confirmation_reference")
    ):
        raise PermissionError("Authorization identity absent or mismatched")
    scope = set(approval.get("scope", []))
    if not set(selected) <= scope:
        raise PermissionError("Task outside explicitly authorized scope")
    if not selected or any(tid not in engine.tasks for tid in selected):
        raise PermissionError('Empty or unknown task scope')
    for key in ('budget_policy_version','training_protocol_version','source_sha256'):
        if approval.get(key)!=engine.plan[key]:raise PermissionError('Approval protocol/source identity differs')
    if approval.get('plan_sha256')!=sha(engine.release/'plan.json'):
        raise PermissionError('Approval plan identity differs')
    if approval.get('target_machine_confirmed') is not True or approval.get('hardware_observation_sha256')!=engine.plan['hardware_observation_sha256']:
        raise PermissionError('Intended Windows GPU machine not confirmed')
    stage=approval.get('execution_stage')
    tasks=[engine.tasks[tid] for tid in selected]
    kinds={t['kind'] for t in tasks}
    if stage=='single_outer_oof':
        expected={t['id'] for t in engine.tasks.values() if t['kind'] in ('miner','select') and t['outer_fold']==0}
        if set(selected)!=expected or scope!=expected:
            raise PermissionError('Pilot mining approval must contain exactly outer0 three miners plus selection')
    elif stage=='single_outer_classifiers':
        expected={f'train_{m}_o0_s42' for m in engine.plan['methods']}
        if set(selected)!=expected or scope!=expected or actual_k_readiness(engine,0)!='GO':
            raise PermissionError('Pilot classifiers require actual outer0 H and exact three seed42 methods')
    elif stage=='full_training':
        if not kinds <= {'miner','select','train'} or not all(engine.completed(f'train_{m}_o0_s42') for m in engine.plan['methods']):
            raise PermissionError('Full training requires successful outer0 pilot and explicit training-only scope')
    elif stage=='evaluation':
        if not kinds <= {'calibrate','evaluate','summarize'}:
            raise PermissionError('Evaluation authorization cannot start training')
    else:
        raise PermissionError('Separate explicit staged authorization required')
    if any(t['kind']=='train' and t['method']=='focal' for t in tasks) and engine.plan.get('focal_semantics_confirmed') is not True:
        raise PermissionError('Historical 11-uniform/12th-weighted semantics not confirmed')
    return approval


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, required=True
    )
    parser.add_argument("--task")
    parser.add_argument("--outer", type=int, choices=range(5))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--phase", choices=["oof", "classifiers"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--retry-reason")
    parser.add_argument("--preflight", action="store_true", help="CPU-only production-path preflight; never trains")
    parser.add_argument("--preflight-id", help="Fresh alphanumeric evidence identity for --preflight")
    args = parser.parse_args(argv)
    engine = NodeEngine(ROOT, args.plan.parent)
    selected = [
        t["id"]
        for t in engine.plan["tasks"]
        if args.all
        or t["id"] == args.task
        or (args.phase == "classifiers" and t["kind"] == "train" and t["outer_fold"] == args.outer and t["seed"] == 42)
        or (args.phase == "oof" and t["kind"] in ("miner", "select") and (args.outer is None or t["outer_fold"] == args.outer))
    ]
    if args.task and args.task not in engine.tasks:
        parser.error("Unknown task")
    if args.preflight:
        if args.execute or args.approval or not args.preflight_id:
            parser.error('CPU preflight requires a fresh --preflight-id and forbids execution approval')
        if args.all:
            selected = [t['id'] for t in engine.plan['tasks'] if t['kind'] == 'train']
        from fire_detection.research.classifier_preflight import run_preflight
        return run_preflight(engine, selected, args.preflight_id)
    if not args.execute:
        import json

        print(
            json.dumps(
                dict(
                    release=engine.plan["release_id"],
                    status=engine.plan["status"],
                    actual_k_readiness=actual_k_readiness(engine, args.outer) if args.outer is not None else actual_k_readiness(engine),
                    total_nodes=len(engine.tasks),
                    selected=selected,
                    oof_engineering_status=engine.plan.get("oof_engineering_status"),
                    training_executed=False,
                ),
                indent=2,
            )
        )
        return 0
    if not selected:
        parser.error("Choose --task, --phase oof or --all")
    require_authorization(engine, selected, args.approval)

    def handler(task, attempt):
        from fire_detection.research.node_logging import run_logged_process
        log = attempt / "stdout.log"
        run_logged_process([sys.executable, "-B", "-u", str(ROOT / "scripts/run_node_worker.py"),
            "--plan", str(args.plan.resolve()), "--task", task["id"],
            "--attempt", str(attempt.resolve()), "--approval", str(args.approval.resolve())], log)
        products = read(attempt / "child_products.json")
        if products["task"] != task["id"]:
            raise ValueError("Child output task mismatch")
        return [Path(p) for p in products["paths"]] + [attempt / "child_products.json", log]

    with engine.lock():
        for tid in selected:
            print(tid, engine.run(tid, handler, retry_reason=args.retry_reason), flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
