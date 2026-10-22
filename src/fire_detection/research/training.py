"""Offline YOLO26 training with split isolation and measured exposure/step budgets."""

from __future__ import annotations

import gc
import json
import math
import shutil
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .immutable_inputs import guarded_inputs, prepare_frozen_view, snapshot_identity

from fire_detection.models.offline import (
    configure_imported_ultralytics,
    sha256_file,
    validate_offline_runtime,
)
from fire_detection.models.training import (
    git_state,
    normalize_ultralytics_device,
    write_json_atomic,
)
from fire_detection.research.protocol import (
    digest_object,
    expected_budget,
    expected_exposure_counts,
    project_path,
    publish_contract,
    replay_indices,
    replay_seed,
)


def configure_runtime(config: Mapping[str, Any]) -> Any:
    """Validate existing local weights before importing any YOLO code."""
    root = Path(config["_root"])
    runtime = validate_offline_runtime(
        project_root=root,
        manifest_path=project_path(root, config["offline"]["manifest"]),
        yolo_config_dir=project_path(root, config["offline"]["config_dir"]),
        required_assets=("yolo26s.pt", "yolo26s-cls.pt", "yolo26n.pt", "Arial.ttf"),
        expected_ultralytics_version=config["offline"]["ultralytics_version"],
    )
    configure_imported_ultralytics(runtime)
    return runtime


class IncompleteTrainingTask(ValueError):
    """Raised when a logical task has no validated completion pointer."""


def reserved_attempt_train_io(attempt_dir: Path) -> dict[str, Any]:
    """Bind Ultralytics I/O to a uniquely reserved empty attempt directory.

    `exist_ok` is True only because the attempt directory is created first to
    reserve its identity. This is output isolation, not a scientific
    hyperparameter. `resume` stays False so retries train from the frozen
    start checkpoint rather than a partial last.pt.
    """
    return {
        "project": str(attempt_dir.parent),
        "name": attempt_dir.name,
        "exist_ok": True,
        "resume": False,
    }


def validate_complete_training_manifest(
    run_dir: Path,
    expected_contract: Mapping[str, Any],
    *,
    context: str = "training run",
) -> dict[str, Any]:
    """Validate every immutable field needed to reuse a completed training run.

    A status flag alone is not completion evidence.  Missing arguments,
    checkpoint hashes, exposure ledgers, or budget equality all fail closed.
    The function never repairs a partial directory.
    """
    if not isinstance(expected_contract, Mapping):
        raise ValueError(f"{context} contract is invalid: {run_dir}")
    manifest_path = run_dir / "research_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{context} has no completion manifest: {run_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} completion manifest is unreadable: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError(f"{context} is incomplete: {run_dir}")
    if manifest.get("contract") != dict(expected_contract):
        raise ValueError(f"{context} contract differs: {run_dir}")
    expected_hash = digest_object(expected_contract)
    if manifest.get("contract_sha256") != expected_hash:
        raise ValueError(f"{context} contract hash differs: {run_dir}")
    required = (
        "checkpoint",
        "checkpoint_sha256",
        "resolved_args_sha256",
        "audit",
        "git",
        "environment",
        "elapsed_seconds",
    )
    if any(key not in manifest for key in required):
        raise ValueError(f"{context} completion manifest is incomplete: {manifest_path}")
    checkpoint = run_dir / "weights" / "last.pt"
    args_path = run_dir / "args.yaml"
    if not checkpoint.is_file() or not args_path.is_file():
        raise ValueError(f"{context} output is incomplete: {run_dir}")
    if manifest["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError(f"{context} checkpoint drift: {run_dir}")
    if manifest["resolved_args_sha256"] != sha256_file(args_path):
        raise ValueError(f"{context} resolved arguments drift: {run_dir}")
    if Path(str(manifest.get("checkpoint", ""))).as_posix() != "weights/last.pt":
        raise ValueError(f"{context} checkpoint policy/path differs: {run_dir}")
    elapsed = manifest.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(float(elapsed)) or float(elapsed) < 0:
        raise ValueError(f"{context} elapsed runtime is invalid: {run_dir}")
    audit = manifest["audit"]
    budget = expected_contract.get("budget")
    if not isinstance(budget, Mapping) or any(
        key not in budget for key in ("sample_exposures", "optimizer_steps")
    ):
        raise ValueError(f"{context} budget contract is incomplete: {run_dir}")
    for key in ("sample_exposures", "optimizer_steps", "epochs", "batch"):
        if key in budget and (type(budget[key]) is not int or budget[key] <= 0):
            raise ValueError(f"{context} budget contract is invalid: {run_dir}")
    measured_budget_keys = ("sample_exposures", "optimizer_steps")
    if not isinstance(audit, dict) or any(key not in audit for key in measured_budget_keys):
        raise ValueError(f"{context} audit is incomplete: {run_dir}")
    for key in measured_budget_keys:
        expected = budget[key]
        if type(audit[key]) is not int or audit[key] != expected:
            raise ValueError(f"{context} measured budget differs for {key}: {run_dir}")
    if type(audit.get("amp_skipped_steps", 0)) is not int or audit.get("amp_skipped_steps", 0) != 0:
        raise ValueError(f"{context} contains AMP-skipped optimizer steps: {run_dir}")
    if expected_contract["task"] == "classifier" and audit.get(
        "per_filename_contract_verified", False
    ) is not True:
        raise ValueError(f"{context} lacks a verified exposure ledger: {run_dir}")
    if expected_contract["task"] == "classifier":
        train_rows = expected_contract.get("train")
        exposures = audit.get("per_filename_exposures")
        if not isinstance(train_rows, list) or not isinstance(exposures, dict):
            raise ValueError(f"{context} exposure ledger is incomplete: {run_dir}")
        try:
            expected_exposures = expected_exposure_counts(
                [str(row["filename"]) for row in train_rows],
                [int(row["label"]) for row in train_rows],
                expected_contract.get("hard_names", []),
                arm=str(expected_contract.get("arm", "hn")),
                seed=int(expected_contract.get("replay_seed", expected_contract["seed"])),
                epochs=int(expected_contract["budget"]["epochs"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{context} exposure contract is malformed: {run_dir}") from exc
        if exposures != expected_exposures:
            raise ValueError(f"{context} exposure ledger differs: {run_dir}")
    return manifest


def training_contract(
    config: Mapping[str, Any],
    *,
    task: str,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    seed: int,
    source_contract_sha256: str,
    arm: str = "hn",
    hard_names: Sequence[str] = (),
    epochs: int | None = None,
) -> dict[str, Any]:
    """Build the frozen scientific contract shared by every retry attempt."""
    if task not in ("classifier", "detector"):
        raise ValueError("invalid task")
    effective_epochs = config[task]["epochs"] if epochs is None else epochs
    if type(effective_epochs) is not int or effective_epochs <= 0:
        raise ValueError("training epochs must be a positive integer")
    effective_replay_seed = replay_seed(config, arm=arm, training_seed=seed)
    budget = expected_budget(
        len(train_rows),
        len(hard_names) if task == "classifier" else 0,
        config[task]["batch"],
        effective_epochs,
    )
    return {
        "input_snapshot": snapshot_identity(config),
        "source_contract_sha256": source_contract_sha256,
        "task": task,
        "arm": arm,
        "seed": seed,
        "replay_seed": effective_replay_seed,
        "train": [dict(r) for r in train_rows],
        "validation": [dict(r) for r in val_rows],
        "hard_names": sorted(hard_names),
        "budget_policy": config["budget_policy"],
        "budget_policy_version": config["budget_policy_version"],
        "training_protocol_version": config["training_protocol_version"],
        "budget": budget,
        "hyperparameters": {**config["training"], **config[task]},
        "checkpoint_policy": "fixed_epochs_last",
        "engineering_only": config["engineering_only"],
    }


def resolve_complete_training_attempt(
    task_dir: Path,
    expected_contract: Mapping[str, Any] | None = None,
    *,
    context: str = "training task",
) -> tuple[Path, dict[str, Any]]:
    """Resolve and strictly validate an immutable completed attempt.

    Legacy completed task directories remain readable. New tasks publish a
    completion pointer only after one isolated attempt validates completely.
    """
    legacy_manifest = task_dir / "research_manifest.json"
    if legacy_manifest.is_file():
        raw = json.loads(legacy_manifest.read_text(encoding="utf-8"))
        contract = expected_contract or raw.get("contract")
        if not isinstance(contract, Mapping):
            raise ValueError(f"{context} has an invalid legacy contract: {task_dir}")
        return task_dir, validate_complete_training_manifest(task_dir, contract, context=context)

    completion_path = task_dir / "completion.json"
    if not completion_path.is_file():
        raise IncompleteTrainingTask(
            f"{context} has no validated completion pointer: {task_dir}"
        )
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} completion pointer is unreadable: {completion_path}") from exc
    required = {"schema_version", "attempt", "contract_sha256", "manifest_sha256"}
    if not isinstance(completion, dict) or set(completion) != required:
        raise ValueError(f"{context} completion pointer is invalid: {completion_path}")
    attempt_value = completion["attempt"]
    if not isinstance(attempt_value, str):
        raise ValueError(f"{context} attempt identity is invalid: {completion_path}")
    attempt_dir = (task_dir / attempt_value).resolve()
    attempts_root = (task_dir / "attempts").resolve()
    if not attempt_dir.is_relative_to(attempts_root) or attempt_dir.parent != attempts_root:
        raise ValueError(f"{context} attempt path escapes its task: {completion_path}")
    manifest_path = attempt_dir / "research_manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != completion["manifest_sha256"]:
        raise ValueError(f"{context} completed attempt manifest drift: {attempt_dir}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = expected_contract or raw.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{context} completed attempt contract is invalid: {attempt_dir}")
    if completion["contract_sha256"] != digest_object(contract):
        raise ValueError(f"{context} completion contract hash differs: {attempt_dir}")
    return attempt_dir, validate_complete_training_manifest(attempt_dir, contract, context=context)


def _new_attempt_directory(task_dir: Path) -> tuple[str, Path]:
    """Allocate a never-before-used attempt directory without touching old evidence."""
    attempts = task_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1_000_000):
        identity = f"attempt_{number:03d}"
        candidate = attempts / identity
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return identity, candidate
    raise RuntimeError(f"attempt identity space exhausted: {task_dir}")


class ReplayDataset:
    """A filename-preserving wrapper with exactly N+K observations in every epoch."""

    def __init__(self, dataset: Any, *, arm: str, hard_names: Sequence[str], seed: int) -> None:
        self.dataset = dataset
        self.arm = arm
        self.hard_names = tuple(hard_names)
        self.seed = seed
        self.epoch = 0
        self.indices: tuple[int, ...] = ()
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Advance rotating uniform controls; fixed HN/random selections stay fixed."""
        self.epoch = epoch
        filenames = [Path(str(s[0])).name for s in self.dataset.samples]
        labels = [int(s[1]) for s in self.dataset.samples]
        self.indices = replay_indices(
            filenames, labels, self.hard_names, arm=self.arm, seed=self.seed, epoch=epoch
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.indices[index]
        result = dict(self.dataset[source])
        result["im_file"] = str(self.dataset.samples[source][0])
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "dataset"), name)


def _copy_isolated(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise ValueError(f"derived view conflicts with source: {destination}")
    else:
        shutil.copy2(source, destination)


def prepare_view(
    config: Mapping[str, Any],
    destination: Path,
    *,
    task: str,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Materialize isolated train/cal copies so loaders cannot mutate source data."""
    root = Path(config["_root"])
    allowed = root / "runs" / "direction_a" / config["campaign"]
    if not destination.resolve().is_relative_to(allowed.resolve()):
        raise ValueError("research dataset destination escapes its isolated campaign")
    if {r["group_id"] for r in train_rows} & {r["group_id"] for r in val_rows}:
        raise ValueError("training groups overlap validation groups")
    if task == "classifier":
        return prepare_frozen_view(config, destination, train_rows, val_rows)
    image_root = project_path(root, config["data"]["images"])
    labels_root = project_path(root, config["data"]["labels"])
    view_contract = {
        "task": task,
        "train": [dict(r) for r in train_rows],
        "val": [dict(r) for r in val_rows],
    }
    publish_contract(destination / "view_contract.json", view_contract)
    for split, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            name = row["filename"]
            if task == "classifier":
                image_target = destination / split / str(row["label"]) / name
            else:
                image_target = destination / "images" / split / name
                _copy_isolated(
                    labels_root / f"{Path(name).stem}.txt",
                    destination / "labels" / split / f"{Path(name).stem}.txt",
                )
            _copy_isolated(image_root / name, image_target)
    if task == "classifier":
        return destination
    data_path = destination / "data.yaml"
    data = {
        "path": str(destination.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "fire"},
    }
    if not data_path.exists():
        data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    elif yaml.safe_load(data_path.read_text(encoding="utf-8")) != data:
        raise ValueError("detector dataset YAML drift")
    return data_path


def _trainer_class(task: str, *, arm: str, hard_names: Sequence[str], seed: int) -> type[Any]:
    from ultralytics.models.yolo.classify import ClassificationTrainer
    from ultralytics.models.yolo.detect import DetectionTrainer

    base = ClassificationTrainer if task == "classifier" else DetectionTrainer

    class BudgetTrainer(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.research_audit: dict[str, Any] = {
                "optimizer_attempts": 0,
                "optimizer_steps": 0,
                "sample_exposures": 0,
                "batches": 0,
            }
            self.exposure_counts: Counter[str] = Counter()
            super().__init__(*args, **kwargs)
            self.add_callback("on_train_epoch_start", self._research_epoch_start)
            self.add_callback("on_train_batch_end", self._research_batch_end)

        def build_dataset(self, img_path: str, mode: str = "train", batch: Any = None) -> Any:
            if task == "classifier":
                from .frozen_dataset import build_classification_dataset
                dataset = build_classification_dataset(self, img_path, mode, batch)
            else:
                dataset = super().build_dataset(img_path, mode, batch)
            if task == "classifier" and mode == "train":
                dataset = ReplayDataset(dataset, arm=arm, hard_names=hard_names, seed=seed)
            return dataset

        def _setup_train(self) -> None:
            super()._setup_train()
            if self.accumulate != 1 or self.args.nbs != self.args.batch:
                raise RuntimeError("actual gradient accumulation violates matched budget")
            self.optimizer.register_step_post_hook(self._research_optimizer_post)

        def _research_optimizer_post(self, optimizer: Any, args: Any, kwargs: Any) -> None:
            self.research_audit["optimizer_steps"] += 1

        def optimizer_step(self) -> None:
            self.research_audit["optimizer_attempts"] += 1
            super().optimizer_step()

        def _research_epoch_start(self, trainer: Any) -> None:
            dataset = self.train_loader.dataset
            if isinstance(dataset, ReplayDataset):
                dataset.set_epoch(int(self.epoch))
                self.train_loader.reset()

        def _research_batch_end(self, trainer: Any) -> None:
            import torch

            if not bool(torch.isfinite(self.loss).all()):
                raise RuntimeError("non-finite training loss; run cannot be research evidence")

        def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
            self.research_audit["sample_exposures"] += int(batch["img"].shape[0])
            self.research_audit["batches"] += 1
            for name in batch.get("im_file", []):
                self.exposure_counts[Path(str(name)).name] += 1
            return dict(super().preprocess_batch(batch))

    return BudgetTrainer


@guarded_inputs
def train_model(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    task: str,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    seed: int,
    source_contract_sha256: str,
    arm: str = "hn",
    hard_names: Sequence[str] = (),
    epochs: int | None = None,
) -> dict[str, Any]:
    """Train one logical task using immutable, independently auditable attempts."""
    source_git_state = git_state(Path(config["_root"]))
    contract = training_contract(
        config,
        task=task,
        train_rows=train_rows,
        val_rows=val_rows,
        seed=seed,
        source_contract_sha256=source_contract_sha256,
        arm=arm,
        hard_names=hard_names,
        epochs=epochs,
    )
    budget = contract["budget"]
    contract_hash = digest_object(contract)
    try:
        _, complete = resolve_complete_training_attempt(run_dir, contract)
        return complete
    except IncompleteTrainingTask as exc:
        claimed = (run_dir / "completion.json").is_file() or (
            run_dir / "research_manifest.json"
        ).is_file()
        if claimed:
            raise ValueError(
                f"completion claim exists but was not validated: {run_dir}"
            ) from exc
    except ValueError:
        raise
    attempt_id, attempt_dir = _new_attempt_directory(run_dir)
    manifest_path = attempt_dir / "research_manifest.json"
    configure_runtime(config)
    import torch
    from ultralytics import YOLO

    data = prepare_view(
        config,
        run_dir.parent.parent / "datasets" / run_dir.name,
        task=task,
        train_rows=train_rows,
        val_rows=val_rows,
    )
    train = dict(config["training"])
    execution_profile = train.pop("execution_profile", "frozen_reference_v1")
    cache_mode = train.pop("cache", False)
    compile_mode = train.pop("compile", False)
    device = normalize_ultralytics_device(train.pop("device"))
    args = {
        **train,
        "device": device,
        "data": str(data),
        "epochs": budget["epochs"],
        "batch": config[task]["batch"],
        "nbs": config[task]["batch"],
        "imgsz": config[task]["imgsz"],
        "seed": seed,
        **reserved_attempt_train_io(attempt_dir),
        "pretrained": True,
        "val": True,
        "save": True,
        "save_period": -1,
        "plots": False,
        "cache": cache_mode,
        "compile": compile_mode,
        "verbose": False,
    }
    if task == "classifier":
        args.update(config["classifier"]["augmentation"])
    else:
        args.update(config["detector"]["augmentation"])
    model = YOLO(str(project_path(Path(config["_root"]), config[task]["weights"])), verbose=False)
    started = time.time()
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "contract": contract,
        "contract_sha256": contract_hash,
        "status": "running",
        "started_at_unix": started,
        "git": source_git_state,
        "environment": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device != 'cpu' and torch.cuda.is_available() else None,
            "execution_profile": execution_profile,
        },
    }
    try:
        model.train(
            trainer=_trainer_class(
                task,
                arm=arm,
                hard_names=hard_names,
                seed=int(contract["replay_seed"]),
            ),
            **args,
        )
        trainer: Any = model.trainer
        saved = Path(str(trainer.save_dir)).resolve()
        if saved != attempt_dir.resolve():
            raise RuntimeError(f"training output escaped reserved attempt directory: {saved}")
        audit = dict(trainer.research_audit)
        audit["amp_skipped_steps"] = audit["optimizer_attempts"] - audit["optimizer_steps"]
        audit["per_filename_exposures"] = dict(sorted(trainer.exposure_counts.items()))
        if task == "classifier":
            samples = trainer.train_loader.dataset.samples
            expected_counts = expected_exposure_counts(
                [Path(str(sample[0])).name for sample in samples],
                [int(sample[1]) for sample in samples],
                hard_names,
                arm=arm,
                seed=int(contract["replay_seed"]),
                epochs=budget["epochs"],
            )
            if audit["per_filename_exposures"] != expected_counts:
                raise RuntimeError("per-filename exposure counts violate the replay contract")
            audit["per_filename_contract_verified"] = True
        if (
            audit["sample_exposures"] != budget["sample_exposures"]
            or audit["optimizer_steps"] != budget["optimizer_steps"]
            or audit["amp_skipped_steps"] != 0
        ):
            raise RuntimeError(f"measured budget differs from protocol: {audit} versus {budget}")
        last = attempt_dir / "weights" / "last.pt"
        if not last.is_file():
            raise RuntimeError("training produced no fixed last.pt")
        manifest.update(
            status="complete",
            audit=audit,
            checkpoint=str(last.relative_to(attempt_dir)).replace("\\", "/"),
            checkpoint_sha256=sha256_file(last),
            resolved_args_sha256=sha256_file(attempt_dir / "args.yaml"),
        )
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["elapsed_seconds"] = time.time() - started
        manifest["peak_gpu_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated()) if device != 'cpu' and torch.cuda.is_available() else 0
        )
        write_json_atomic(manifest_path, manifest)
        del model
        gc.collect()
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if manifest.get("status") == "complete":
        validate_complete_training_manifest(attempt_dir, contract)
        completion_path = run_dir / "completion.json"
        if completion_path.exists():
            raise ValueError(f"completed manifest cannot be rewritten: {completion_path}")
        publish_contract(
            completion_path,
            {
                "schema_version": 1,
                "attempt": f"attempts/{attempt_id}",
                "contract_sha256": contract_hash,
                "manifest_sha256": sha256_file(manifest_path),
            },
        )
    return manifest
