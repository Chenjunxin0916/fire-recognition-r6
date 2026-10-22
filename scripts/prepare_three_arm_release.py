"""Read-only experiment audit and immutable planning. Contains no training call."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RELEASE = ROOT / "plans/three_arm_strict_fp32_v1"
CONFIG = ROOT / "configs/three_arm_strict_fp32_v1"
REPORT = ROOT.parent / "Paper/formal_prelaunch_20261014"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def yaml_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        yaml.safe_dump(value, f, sort_keys=False, allow_unicode=True)


def main():
    from fire_detection.research.focal_audit import audit_attempt
    from fire_detection.research.focal_training import load_focal_config
    from fire_detection.research.protocol import build_partition, digest_object, verify_source
    from fire_detection.research.training import resolve_complete_training_attempt

    assert not RELEASE.exists(), "Existing release must not be overwritten"
    REPORT.mkdir(parents=True, exist_ok=True)
    for name in (
        "project_state.yaml",
        "resume-card.md",
        "project-events.yaml",
        "artifact-index.yaml",
        "expert-dispatch-ledger.yaml",
    ):
        shutil.copy2(ROOT.parent / "Paper" / name, REPORT / ("before_" + name))
    save(
        REPORT / "expert-task.json",
        dict(
            contract_version="2.0",
            project_id="paper_fire_detection_20260915",
            task_id="FORMAL-PRELAUNCH-20261014",
            expert_skill="sci-skills-experiment-engineering",
            expert_role="PRIMARY",
            execution_mode="INLINE",
            runtime_status="RUNNING",
            authorization="Audit, freeze and plan only; no training or GPU smoke",
        ),
    )
    base, focal = load_focal_config(ROOT / "configs/focal_sampling/dfire.yaml", ROOT)
    records, source = verify_source(base)
    gpu = read(ROOT / "runs/migration_gpu_20261014/summary.json")
    assert gpu["status"] == "GPU_VALIDATION_PASSED"
    old_source = read(ROOT / "runs/migration_gpu_20261014/source_verified.json")
    for key in (
        "image_hashes",
        "label_hashes",
        "folds_sha256",
        "source_contracts_sha256",
        "initial_weights_sha256",
    ):
        assert source[key] == old_source[key], key
    gpu_audits = []
    for case in ("switch_w2", "capacity_w4", "capacity_strict_w4"):
        attempt = (
            ROOT
            / f"runs/direction_a/focal_windows_gpu_20261014_{case}/outer_0/seed_42/focal/attempts/attempt_001"
        )
        result = audit_attempt(attempt)
        gpu_audits.append(dict(case=case, status=result["status"], audit=result["audit"]))
    env_diffs = []
    for line in (ROOT / "requirements/lock-win-py311-cu128.txt").read_text().splitlines():
        if "==" in line:
            name, expected = line.split("==")
            actual = metadata.version(name)
            if actual != expected:
                env_diffs.append(dict(name=name, expected=expected, actual=actual))
    assert platform.python_version() == "3.11.9" and not env_diffs
    protected = {}
    for dirname in ("src", "configs", "scripts", "tests", "requirements"):
        for p in (ROOT / dirname).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                protected[p.relative_to(ROOT).as_posix()] = sha(p)
    save(REPORT / "protected_code_before.json", protected)
    historical = []
    historical_k = {}
    old = ROOT / "runs/direction_a/dfire_scheme_b_formal_v3"
    for outer in range(5):
        selection = read(old / f"outer_{outer}/seed_42/hard_negative_selection.json")
        historical_k[str(outer)] = selection["K"]
        for seed in (42, 2026, 3407):
            part = build_partition(records, outer, seed)
            previous = read(old / f"outer_{outer}/seed_{seed}/partition.json")
            for key in ("splits", "miners", "outer_fold", "seed", "calibration_fold", "fit_folds"):
                assert part[key] == previous[key]
            for task in ("cls_hn", "cls_random_negative"):
                run, manifest = resolve_complete_training_attempt(
                    old / f"outer_{outer}/seed_{seed}/models/{task}"
                )
                historical.append(
                    dict(
                        outer_fold=outer,
                        seed=seed,
                        task=task,
                        run=run.relative_to(ROOT).as_posix(),
                        verified_complete=True,
                        environment=manifest["environment"],
                        budget=manifest["contract"]["budget"],
                        elapsed_seconds=manifest["elapsed_seconds"],
                        tf32_evidence="NOT_RECORDED",
                        direct_primary_comparison_allowed=False,
                        reason="Python 3.11.15 rather than 3.11.9; strict TF32-off not evidenced",
                    )
                )
            if seed == 42:
                for miner in part["miners"]:
                    name = f"miner_inner_{miner['validation_fold']}"
                    run, manifest = resolve_complete_training_attempt(
                        old / f"outer_{outer}/seed_42/models/{name}"
                    )
                    historical.append(
                        dict(
                            outer_fold=outer,
                            seed=42,
                            task=name,
                            run=run.relative_to(ROOT).as_posix(),
                            verified_complete=True,
                            environment=manifest["environment"],
                            budget=manifest["contract"]["budget"],
                            elapsed_seconds=manifest["elapsed_seconds"],
                            tf32_evidence="NOT_RECORDED",
                            reuse_for_strict_mining_allowed=False,
                        )
                    )
    save(
        REPORT / "history_audit.json",
        dict(
            formal_v3=historical,
            historical_K=historical_k,
            old_standard=dict(
                config="configs/train_classifier.yaml",
                epochs=80,
                batch=32,
                amp=True,
                dataset="v3, not frozen D-Fire pool",
                direct_primary_comparison_allowed=False,
            ),
            old_focal=dict(
                formal_complete=0, smoke_only=True, direct_primary_comparison_allowed=False
            ),
        ),
    )
    campaign = "focal_three_arm_strict_fp32_v1"
    base_raw = yaml.safe_load(
        (ROOT / "configs/direction_a/dfire_scheme_b_formal_v3.yaml").read_text(encoding="utf-8")
    )
    base_raw["campaign"] = campaign
    yaml_new(CONFIG / "base.yaml", base_raw)
    for method in ("standard", "hn", "focal"):
        yaml_new(
            CONFIG / f"{method}.yaml",
            dict(
                schema_version=1,
                method=method,
                base="configs/three_arm_strict_fp32_v1/base.yaml",
                runtime="configs/three_arm_strict_fp32_v1/runtime.yaml",
                focal_sampling=dict(enabled=method == "focal", at_epoch=10, gamma=0.5)
                if method == "focal"
                else None,
                train_entry="existing train_focal" if method == "focal" else "existing train_model",
                replay="existing HN, no selection or replay changes" if method == "hn" else "none",
                note="These wrapper configs are for run_sealed_three_arm.py, not legacy CLI parsers",
            ),
        )
    policy = dict(
        schema_version=1,
        environment=dict(
            os="Windows",
            gpu="NVIDIA GeForce RTX 5080",
            python="3.11.9",
            torch="2.11.0+cu128",
            torchvision="0.26.0+cu128",
            ultralytics="8.4.92",
            lock="requirements/lock-win-py311-cu128.txt",
        ),
        precision=dict(
            amp=False,
            matmul_allow_tf32=False,
            cudnn_allow_tf32=False,
            float32_matmul_precision="highest",
            training_dtype="torch.float32",
            inference_half=False,
        ),
        fixed=dict(
            epochs=40,
            batch=64,
            nbs=64,
            imgsz=384,
            workers=4,
            optimizer="AdamW",
            warmup_epochs=0,
            accumulate=1,
            device="cuda:0",
            resume=False,
            compile=False,
        ),
        failure=dict(
            oom="fail_current_task_and_stop_queue",
            automatic_batch_reduction=False,
            cpu_fallback=False,
            automatic_retry=False,
            precision_fallback=False,
            partial_success=False,
            resume="completed-task skip after hash validation; failed task restarts from pretrained weights in new attempt",
        ),
        checkpoint_policy="fixed_epochs_last",
        initialization="same pretrained SHA256, seed-paired 2-class head; record and compare initialized model state SHA256",
        mining=dict(
            seed=42,
            epochs=30,
            models_per_outer_fold=3,
            reuse_across_training_seeds=True,
            reuse_historical=False,
        ),
        budget_gate=dict(
            classifier_updates=440,
            standard_focal_exposures=26400,
            hn_max_K_for_440_updates=44,
            hn_K_min=0,
            if_K_zero="keep method definition and flag HN comparison uninformative",
            if_K_above_44="NO_GO; do not truncate H or alter batch/epochs",
        ),
    )
    yaml_new(CONFIG / "runtime.yaml", policy)
    yaml_new(
        CONFIG / "evaluation.yaml",
        dict(
            schema_version=1,
            status="SPECIFICATION_FROZEN_ADAPTER_NOT_IMPLEMENTED",
            checkpoint="last.pt",
            methods=["standard", "hn", "focal"],
            primary="original_no_TTA",
            secondary="horizontal_flip_average_TTA_for_all_methods",
            threshold=dict(
                start=0.01,
                end=0.99,
                step=0.005,
                selection="existing search_thresholds: F1, recall, lower center of each prediction plateau, then lowest center",
            ),
            order="complete and freeze all cal thresholds for a cell before any test inference",
            cal_test_rows=220,
            pairing=["seed", "filename", "outer_fold", "group_id", "label"],
            seed_summary="pool 5 held-out folds to 1100 test images per seed; mean and sample SD(ddof=1) across 3 seeds",
            comparisons=["hn-standard", "focal-standard", "hn-focal"],
            bootstrap=dict(
                iterations=10000,
                seed=20260914,
                unit="scene groups within outer folds; same draws across methods and seeds",
                interval="paired 95% percentile CI",
                scope="conditional on frozen models and thresholds, not training uncertainty",
            ),
            interpretation="descriptive comparisons with effect sizes and CIs; no unadjusted confirmatory multi-pair significance claim",
            prohibited="Do not call old formal_scheme_b=True three-method analysis; do not relabel random_negative as Standard",
        ),
    )
    tasks = []

    def add(task_id, kind, outer=None, seed=None, method=None, depends=None, inner=None):
        relative = Path("runs/direction_a") / campaign
        if outer is not None:
            relative /= f"outer_{outer}"
        if seed is not None:
            relative /= f"seed_{seed}"
        if kind == "miner":
            relative /= f"models/miner_inner_{inner}"
        elif kind == "train":
            relative /= f"models/{method}"
        elif kind == "select":
            relative /= "selection"
        elif kind == "calibrate":
            relative /= f"calibration/{method}"
        elif kind == "evaluate":
            relative /= f"evaluation/{method}"
        else:
            relative /= "analysis"
        tasks.append(
            dict(
                id=task_id,
                kind=kind,
                outer_fold=outer,
                seed=seed,
                method=method,
                inner_fold=inner,
                depends_on=depends or [],
                config=f"configs/three_arm_strict_fp32_v1/{method or 'base'}.yaml"
                if method in ("standard", "hn", "focal")
                else "configs/three_arm_strict_fp32_v1/base.yaml",
                runtime_config="configs/three_arm_strict_fp32_v1/runtime.yaml",
                output_dir=relative.as_posix(),
                log_pattern=f"{relative.as_posix()}/execution/attempt_NNN/stdout.log",
                status="PENDING",
                expected_epochs=30 if kind == "miner" else (40 if kind == "train" else None),
                expected_optimizer_steps=210
                if kind == "miner"
                else (440 if kind == "train" else None),
                expected_sample_exposures=13200
                if kind == "miner"
                else (26400 if kind == "train" and method != "hn" else None),
                exposure_expression="40*(660+K_outer)"
                if kind == "train" and method == "hn"
                else None,
                budget_status="PENDING_STRICT_MINING_K" if method == "hn" else "DEFINED",
                implementation_status="NOT_IMPLEMENTED"
                if kind in ("calibrate", "evaluate", "summarize")
                else "PREPARED_NOT_EXECUTED",
            )
        )

    # Complete all mining and K gates before any full comparison training.
    gate_ids = []
    for outer in range(5):
        part = build_partition(records, outer, 42)
        ids = []
        for item in part["miners"]:
            inner = item["validation_fold"]
            tid = f"mine_o{outer}_i{inner}_s42"
            ids.append(tid)
            add(tid, "miner", outer, 42, inner=inner)
        gate = f"select_o{outer}"
        gate_ids.append(gate)
        add(gate, "select", outer, 42, depends=ids)
    for outer in range(5):
        for seed in (42, 2026, 3407):
            part = build_partition(records, outer, seed)
            save(RELEASE / f"partitions/outer_{outer}/seed_{seed}.json", part)
            for method in ("standard", "hn", "focal"):
                add(
                    f"train_{method}_o{outer}_s{seed}",
                    "train",
                    outer,
                    seed,
                    method,
                    depends=gate_ids,
                )
    # Evaluation waits until every training task has been audited; no early test peeking.
    train_ids = [t["id"] for t in tasks if t["kind"] == "train"]
    for outer in range(5):
        for seed in (42, 2026, 3407):
            for method in ("standard", "hn", "focal"):
                add(
                    f"cal_{method}_o{outer}_s{seed}",
                    "calibrate",
                    outer,
                    seed,
                    method,
                    depends=train_ids,
                )
    eval_ids = []
    for outer in range(5):
        for seed in (42, 2026, 3407):
            for method in ("standard", "hn", "focal"):
                tid = f"eval_{method}_o{outer}_s{seed}"
                eval_ids.append(tid)
                add(
                    tid,
                    "evaluate",
                    outer,
                    seed,
                    method,
                    depends=[f"cal_{m}_o{outer}_s{seed}" for m in ("standard", "hn", "focal")],
                )
    add("summarize_all", "summarize", depends=eval_ids)
    blockers = [
        dict(
            id="B1",
            description="Three-method evaluation and paired aggregation adapters are not implemented/validated; legacy formal pipeline only accepts HN/random_negative",
        ),
        dict(
            id="B2",
            description="Strict-FP32 miners have not run; actual HN K and 440-update eligibility must be verified after all 5 selections; historical K is not a substitute",
        ),
        dict(
            id="B3",
            description="New strict policy wrapper has only non-training tests; release requires adapter/recovery contract review before permission to execute",
        ),
    ]
    plan = dict(
        schema_version=1,
        release_id="three_arm_strict_fp32_v1",
        project_id="paper_fire_detection_20260915",
        campaign=campaign,
        status="NO_GO",
        authorized=False,
        created_at=datetime.now(UTC).isoformat(),
        methods=["standard", "hn", "focal"],
        outer_folds=list(range(5)),
        seeds=[42, 2026, 3407],
        training_count=60,
        formal_classifier_count=45,
        miner_count=15,
        selection_count=5,
        calibration_count=45,
        evaluation_count=45,
        summary_count=1,
        tasks=tasks,
        blocking_issues=blockers,
        source_sha256=digest_object(source),
        history_reuse="No historical trained models or OOF H sets enter the new strict primary matrix",
    )
    save(RELEASE / "plan.json", plan)
    save(RELEASE / "source.json", source)
    save(
        RELEASE / "authorization.template.json",
        dict(
            authorized=False,
            release_id=plan["release_id"],
            seal_sha256=None,
            scope=[],
            human_confirmation_reference=None,
        ),
    )
    with (RELEASE / "tasks.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tasks[0]))
        writer.writeheader()
        for task in tasks:
            writer.writerow({**task, "depends_on": ";".join(task["depends_on"])})
    save(
        RELEASE / "task_state.json",
        dict(
            revision=0,
            release_id=plan["release_id"],
            tasks={
                t["id"]: {"status": "PENDING", "accepted_attempt": None, "attempts": []}
                for t in tasks
            },
        ),
    )
    audit = dict(
        status="AUDIT_COMPLETED_RELEASE_NO_GO",
        no_training_executed=True,
        no_gpu_smoke_repeated=True,
        gpu_evidence_audits=gpu_audits,
        environment_lock_differences=env_diffs,
        historical_K=historical_k,
        actual_new_K=None,
        history_records_verified=len(historical),
        partition_identity="15/15 unchanged",
        task_counts=dict(Counter(t["kind"] for t in tasks)),
        precision_explicit=True,
        algorithm_definitions_changed=False,
        historical_training_hours={
            kind: sum(
                x["elapsed_seconds"]
                for x in historical
                if ("miner" if x["task"].startswith("miner") else x["task"]) == kind
            )
            / 3600
            for kind in ("miner", "cls_hn", "cls_random_negative")
        },
        time_plan=dict(
            measured_focal_minutes=717.8463118 / 60,
            main_training_point_hours=45 * 717.8463118 / 3600,
            main_training_range_hours=[45 * 10 / 60, 45 * 16 / 60],
            miner_point_hours=15 * (717.8463118 / 60) * (30 / 40) * (7 / 11) / 60,
            miner_range_hours=[15 * 5 / 60, 15 * 10 / 60],
            evaluation_hours="NOT_MEASURED_UNDER_NEW_THREE_METHOD_ADAPTER",
            total_training_point_hours=(45 + 15 * (30 / 40) * (7 / 11)) * 717.8463118 / 3600,
            total_training_range_hours=[8.75, 14.5],
            provisional_evaluation_reserve_hours=[0.25, 1.5],
            provisional_full_campaign_hours=[9, 16],
            retry_reserve_excluded=True,
            definition="single reserved GPU wall-clock hours, not utilization-integrated kernel hours",
            basis="strict full-size 3-epoch Focal GPU measurement; Standard/HN/miner extrapolation uncertain, historical miner 1.452h corroborative only",
        ),
    )
    save(REPORT / "review_evidence.json", audit)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "tasks": len(tasks),
                "counts": audit["task_counts"],
                "history_verified": len(historical),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
