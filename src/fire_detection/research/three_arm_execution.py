"""Domain handlers for release v2; retains existing HN/Focal scientific algorithms."""

from __future__ import annotations

import importlib.metadata as metadata
import platform
from pathlib import Path

import pandas as pd
import yaml

from .three_arm_recovery import atomic, read, sha
from .immutable_inputs import snapshot_identity


def runtime_environment(root, policy, *, gpu=True):
    if (
        platform.system() != "Windows"
        or platform.python_version() != policy["environment"]["python"]
    ):
        raise RuntimeError("Frozen OS/Python mismatch")
    for line in (root / policy["environment"]["lock"]).read_text().splitlines():
        if "==" in line:
            name, version = line.split("==")
            if metadata.version(name) != version:
                raise RuntimeError(f"Dependency drift: {name}")
    if gpu:
        import torch

        from .strict_release_guard import set_precision

        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_name(0) != policy["environment"]["gpu"]
        ):
            raise RuntimeError("Pinned CUDA GPU required")
        set_precision(torch)


def execute_node(root, release, plan, task, attempt):
    """Called only after authorization, seal, dependency and retry checks."""
    from .protocol import (
        digest_object,
        load_config,
        publish_contract,
        publish_hash_sidecar,
        verify_source,
    )
    from .three_arm_evaluation import calibrate_method, evaluate_method, summarize_three_arm

    root, release = Path(root), Path(release)
    from fire_detection.source_provenance import verify_execution_source
    provenance = verify_execution_source(root)
    atomic(attempt / "source_provenance.json", provenance)
    policy = yaml.safe_load((root / task["runtime_config"]).read_text(encoding="utf-8"))
    runtime_environment(root, policy, gpu=task["kind"] != "summarize")
    config = load_config(root / plan["base_config"], root=root)
    if any(config[k]!=plan[k] for k in ("budget_policy","budget_policy_version","training_protocol_version")):
        raise ValueError("Execution config differs from sealed budget/protocol")
    _, source = verify_source(config)
    if digest_object(source) != plan["source_sha256"]:
        raise ValueError("Dataset/weights/runtime source identity drift")
    config["_focal_base_sha256"] = sha(root / plan["base_config"])
    spec = yaml.safe_load((root / plan["evaluation_config"]).read_text(encoding="utf-8"))
    targets = {t["id"]: root / t["output_dir"] for t in plan["tasks"]}
    if task["kind"] == "summarize":
        frames = []
        for parent in task["depends_on"]:
            receipt = read(targets[parent] / "accepted.json")
            files = [
                root / name for name in receipt["artifacts"] if name.endswith("/predictions.csv")
            ]
            if len(files) != 1:
                raise ValueError("Expected one accepted test prediction table per method node")
            frames.append(pd.read_csv(files[0], float_precision="round_trip"))
        if plan.get("study_kind") == "HN_RANDOM_CONTROL":
            from .route_b_evaluation import summarize_route_b

            hn_path = root / plan["imported_hn_predictions"]
            if sha(hn_path) != plan["imported_hn_predictions_sha256"]:
                raise ValueError("Imported HN prediction table changed")
            paths = summarize_route_b(
                pd.read_csv(hn_path, float_precision="round_trip"),
                pd.concat(frames, ignore_index=True),
                attempt / "products", spec,
            )
        else:
            paths = summarize_three_arm(
                pd.concat(frames, ignore_index=True), attempt / "products", spec
            )
        costs = []
        for upstream in plan["tasks"]:
            if upstream["kind"] == "summarize":
                continue
            for record_path in sorted(
                targets[upstream["id"]].glob("execution/attempt_*/record.json")
            ):
                record = read(record_path)
                measured_path = record_path.parent / "measured_budget.json"
                measured = read(measured_path) if measured_path.exists() else None
                observed_path = record_path.parent / "hn_selection_observed.json"
                observed = read(observed_path) if observed_path.exists() else None
                if record.get("elapsed_seconds") is None:
                    raise ValueError(
                        "Unreconciled attempt runtime; cost summary cannot silently omit it"
                    )
                costs.append(
                    dict(
                        task=upstream["id"],
                        kind=upstream["kind"],
                        method=upstream["method"],
                        outer_fold=upstream["outer_fold"],
                        seed=upstream["seed"],
                        attempt=record_path.parent.name,
                        status=record["status"],
                        budget_policy=plan["budget_policy"],
                        K=measured["K"] if measured else observed["K"] if observed else None,
                        actual_optimizer_updates=measured["actual_counters"]["optimizer_steps"] if measured else None,
                        actual_sample_exposures=measured["actual_counters"]["sample_exposures"] if measured else None,
                        counter_evidence="measured_trainer_audit" if measured else "not_available_do_not_impute_zero",
                        elapsed_seconds=record["elapsed_seconds"],
                        reserved_gpu_wall_hours=record["elapsed_seconds"] / 3600,
                        hn_shared_mining=upstream["kind"] in ("miner", "select"),
                        amortized_hn_hours_per_training_seed=record["elapsed_seconds"] / 10800
                        if upstream["kind"] in ("miner", "select")
                        else None,
                    )
                )
                paths.append(record_path)
        cost_path = attempt / "products/cost_attempts.csv"
        pd.DataFrame(costs).to_csv(cost_path, index=False)
        cost_note = attempt / "products/cost_definition.json"
        atomic(
            cost_note,
            dict(
                unit="single reserved GPU task wall time, not kernel activity",
                includes="setup, loading, inference/training, failed retries",
                excluded="CPU-only summary task and operator idle gaps",
                no_double_count="use outer node elapsed only; never add inner trainer elapsed",
                hn_sharing=(
                    "miner+selection executed once per outer; "
                    "per-seed amortization is /3, not additive cost"
                ),
            ),
        )
        return paths + [cost_path, cost_note]
    fold, seed, method = task["outer_fold"], task["seed"], task["method"]
    partition_name=f"partitions/outer_{fold}/seed_{seed}.json"
    partition = read(release / partition_name)
    if digest_object(partition)!=plan['partition_digests'][partition_name]:
        raise ValueError('Execution partition digest differs')
    cell = root / "runs/direction_a" / plan["campaign"] / f"outer_{fold}/seed_{seed}"
    publish_contract(cell / "partition.json", partition)
    publish_hash_sidecar(cell / "partition.json")
    if task["kind"] == "calibrate":
        return calibrate_method(
            config,
            partition,
            targets[f"train_{method}_o{fold}_s{seed}"],
            attempt / "products",
            method,
        )
    if task["kind"] == "evaluate":
        calibrations = {}
        for name in plan["methods"]:
            parent = targets[f"cal_{name}_o{fold}_s{seed}"]
            accepted = read(parent / "accepted.json")
            calibrations[name] = root / accepted["accepted_attempt"] / "products/calibration.json"
        return evaluate_method(
            config,
            partition,
            targets[f"train_{method}_o{fold}_s{seed}"],
            attempt / "products",
            method,
            calibration_paths=calibrations,
            required_methods=plan["methods"],
        )
    if task["kind"] == "select":
        return execute_selection(config, plan, task, cell, partition, attempt) + [attempt / "source_provenance.json"]
    return execute_training(config, plan, task, targets[task["id"]], cell, partition, attempt) + [attempt / "source_provenance.json"]


def execute_selection(config, plan, task, cell, partition, attempt):
    from . import pipeline
    from .protocol import digest_object
    from .three_arm_budget import require_hn_budget
    from .three_arm_evaluation import _raw, strict_predict_classifier
    from .training import resolve_complete_training_attempt

    old_predict = pipeline.predict_model
    paths = []
    root = Path(config["_root"]).resolve()
    lineage_path = cell / "strict_oof_lineage.json"
    selection_path = cell / "hard_negative_selection.json"
    expected = {}
    for miner in partition["miners"]:
        inner = miner["validation_fold"]
        name = f"miner_inner_{inner}"
        run, _ = resolve_complete_training_attempt(cell / "models" / name)
        rows = [r for r in partition["splits"]["fit"] if r["fold"] == inner]
        expected[name] = dict(
            checkpoint_sha256=sha(run / "weights/last.pt"),
            metadata_sha256=sha(run / "research_manifest.json"),
            rows_sha256=digest_object(rows),
            split="fit_oof",
            precision="strict_fp32",
            input_snapshot=snapshot_identity(config),
        )
    if selection_path.exists():
        if not lineage_path.is_file():
            raise RuntimeError(
                "OOF selection without strict lineage: manual reconciliation required"
            )
        lineage = read(lineage_path)
        if lineage["miners"] != expected:
            raise ValueError("OOF miner lineage drift")
        for name, expected_hash in lineage["artifacts"].items():
            p = (root / name).resolve()
            if not p.is_relative_to(root) or sha(p) != expected_hash:
                raise ValueError("OOF selection/prediction lineage drift")
            paths.append(p)
    elif lineage_path.exists():
        raise RuntimeError("OOF lineage without selection: manual reconciliation required")

    def predict(config, _cell, *, checkpoint, rows, split, model_name, **kwargs):
        if split != "fit_oof":
            raise ValueError("Mining may only predict held-out fit rows")
        from .split_role_guard import require_oof_rows
        require_oof_rows(partition, rows, model_name)
        if sha(checkpoint) != expected[model_name]["checkpoint_sha256"]:
            raise ValueError("OOF checkpoint identity drift")
        frame = _raw(strict_predict_classifier(config, checkpoint, rows, split, tta=False), rows)
        # Selection's existing coverage/threshold policy remains unchanged.
        path = attempt / f"{model_name}_oof.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
        atomic(
            path.with_suffix(".manifest.json"),
            dict(
                checkpoint_sha256=sha(checkpoint),
                rows_sha256=digest_object(rows),
                split=split,
                csv_sha256=sha(path),
                precision="strict_fp32",
                input_snapshot=snapshot_identity(config),
            ),
        )
        paths.append(path.with_suffix(".manifest.json"))
        return frame

    pipeline.predict_model = predict
    try:
        names = pipeline._hard_negative_selection(
            config,
            cell,
            partition,
            partition["splits"]["fit"],
            task["seed"],
            digest_object(partition),
        )
    finally:
        pipeline.predict_model = old_predict
    if not lineage_path.exists():
        paths += [selection_path, cell / "fit_mining_oof.csv"]
        atomic(
            lineage_path,
            dict(
                miners=expected, artifacts={p.relative_to(root).as_posix(): sha(p) for p in paths}
            ),
        )
    # Persist full observed H before the budget gate; A must stop without losing H.
    observed_path = attempt / "hn_selection_observed.json"
    atomic(observed_path, dict(K=len(names), hard_names=names,
        budget_policy=config["budget_policy"], source_sha256=plan["source_sha256"],
        partition_sha256=digest_object(partition), selection_sha256=sha(selection_path),
        classifier_training_executed=False))
    budget = require_hn_budget(partition["splits"]["fit"], names, seed=task["seed"], policy=config["budget_policy"])
    atomic(attempt / "budget.json", budget)
    paths += [observed_path, attempt / "budget.json", lineage_path, selection_path, cell / "fit_mining_oof.csv"]
    return list(dict.fromkeys(paths))


def execute_training(config, plan, task, target, cell, partition, attempt):
    import torch

    from . import focal_training as ft
    from . import training as rt
    from .focal_audit import audit_attempt
    from .strict_release_guard import guarded_factory
    from .three_arm_budget import (
        check_runtime_budget, install_budget_callbacks, require_hn_budget,
        require_random_budget,
    )

    epochs, method = task["expected_epochs"], task["method"]
    from .split_role_guard import training_roles
    fit, cal = training_roles(partition, task)
    hard, budget = [], None
    if method in ("hn", "random_negative"):
        imported = plan.get('imported_completed', {})
        if task['outer_fold'] == 0 and 'select_o0' in imported.get('task_ids', []):
            selection_path = (Path(config['_root']) / 'runs/direction_a'
                              / imported['parent_campaign'] / 'outer_0/seed_42/hard_negative_selection.json')
            if sha(selection_path) != plan['derived_from']['formal_h_sha256']:
                raise ValueError('Imported formal H source identity drift')
        else:
            selection_path = cell.parent / "seed_42/hard_negative_selection.json"
        if method == "random_negative" and sha(selection_path) != plan["hn_selection_file_sha_by_outer"][str(task["outer_fold"])]:
            raise ValueError("Imported HN selection file differs from the sealed source")
        selection = read(selection_path)
        hard = selection["hard_names"]
        if selection["K"] != len(hard):
            raise ValueError("K identity mismatch")
        if method == "random_negative":
            identity_path = Path(config["_root"]) / plan["random_identity_files"][str(task["outer_fold"])]
            if sha(identity_path) != plan["random_identity_file_sha_by_outer"][str(task["outer_fold"])]:
                raise ValueError("Random identity candidate file changed")
            candidate = read(identity_path)
            budget = require_random_budget(
                fit, hard,
                random_seed=config["scheme_b"]["random_negative_identity_seed"],
                expected_extra_names=candidate["filenames"],
                policy=config["budget_policy"],
            )
            if (budget["optimizer_updates"], budget["sample_exposures"]) != (
                task["expected_optimizer_steps"], task["expected_sample_exposures"]
            ):
                raise ValueError("Random arm does not match the frozen HN model budget")
        else:
            budget = require_hn_budget(fit, hard, seed=task["seed"], policy=config["budget_policy"])
        atomic(attempt / "budget.json", budget)
    expected = dict(
        epochs=epochs,
        batch=64,
        nbs=64,
        imgsz=384,
        amp=False,
        warmup_epochs=0.0,
        resume=False,
        compile=False,
    )
    original_rt, original_ft = rt._trainer_class, ft._trainer_class
    trained = []

    def factory(*args, **kwargs):
        base = guarded_factory(
            original_rt(*args, **kwargs),
            expected=expected,
            audit_path=attempt / "policy_epochs.jsonl",
            identity_path=cell / "paired_initialization.json"
            if task["kind"] == "train"
            else target / "initialization.json",
            torch=torch,
        )

        class BudgetTrainer(base):
            def _setup_train(self):
                super()._setup_train()
                if budget is not None:
                    self._budget_handle = install_budget_callbacks(self, budget)
                trained.append(self)

        return BudgetTrainer

    rt._trainer_class = ft._trainer_class = factory
    try:
        if method == "focal":
            ft.train_focal(
                config,
                {"enabled": True, "at_epoch": 10, "gamma": 0.5},
                target,
                splits=partition["splits"],
                seed=task["seed"],
                fold=task["outer_fold"],
                source_sha256=plan["source_sha256"],
            )
            pointers = list(target.glob("completed_*.json"))
            if len(pointers) != 1:
                raise ValueError("Ambiguous Focal completion")
            run = target / read(pointers[0])["attempt"]
            audit_attempt(run)
        else:
            rt.train_model(
                config,
                target,
                task="classifier",
                train_rows=fit,
                val_rows=cal,
                seed=task["seed"],
                source_contract_sha256=plan["source_sha256"],
                arm=method if method == "random_negative" else "hn",
                hard_names=hard,
                epochs=epochs,
            )
            run, manifest = rt.resolve_complete_training_attempt(target)
        if len(trained) != 1:
            raise ValueError("Expected exactly one newly executed trainer")
        if budget is not None:
            check_runtime_budget(trained[0], budget, completed=True)
        results = pd.read_csv(run / "results.csv")
        results.columns = results.columns.str.strip()
        import numpy as np

        if (
            list(results.epoch) != list(range(1, epochs + 1))
            or not np.isfinite(results.to_numpy(dtype=float)).all()
        ):
            raise ValueError("Partial or nonfinite epoch metrics")
        audit = trained[0].research_audit
        expected_steps = budget["optimizer_updates"] if budget is not None else task["expected_optimizer_steps"]
        expected_exposures = budget['sample_exposures'] if budget is not None else task['expected_sample_exposures']
        if audit['sample_exposures'] != expected_exposures:
            raise ValueError('Measured sample exposure budget differs')
        if audit["optimizer_steps"] != expected_steps:
            raise ValueError("Measured optimizer budget differs")
        policy = [
            read_line
            for read_line in (attempt / "policy_epochs.jsonl").read_text().splitlines()
            if read_line.strip()
        ]
        if len(policy) != epochs:
            raise ValueError("Incomplete strict precision epoch ledger")
        paths = [p for p in run.rglob("*") if p.is_file()]
        paths += [p for p in target.iterdir() if p.is_file() and p.name != "accepted.json"]
        paths += [attempt / "policy_epochs.jsonl", attempt / "measured_budget.json"]
        if budget is not None:
            paths.append(attempt / "budget.json")
        return paths
    finally:
        if trained:
            atomic(attempt / "measured_budget.json", dict(
                budget_policy=config["budget_policy"], budget_policy_version=config["budget_policy_version"],
                training_protocol_version=config["training_protocol_version"],task=task["id"],
                actual_counters=trained[0].research_audit,
                per_filename_exposures=dict(trained[0].exposure_counts),
                K=len(hard) if method in ("hn", "random_negative") else 0,
                expected_updates=budget["optimizer_updates"] if budget else task["expected_optimizer_steps"],
                shared_mining=task["kind"] == "miner"))
        rt._trainer_class, ft._trainer_class = original_rt, original_ft
