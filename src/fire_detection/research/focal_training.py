# Public candidate adaptation notice (2026-10-18): this local trainer adapter
# modifies the upstream Focal Sampling workflow for Ultralytics classification
# while retaining the published method's fixed-probability sampling definition.
# Upstream: https://github.com/AlessioQuercia/sgd_biased at
# 438dbe399ca1f7f623879ece45f361d3ba906e8c (Apache-2.0).
# Relevant upstream callbacks.py credits Copyright 2024 Forschungszentrum
# Juelich GmbH and file author Alessio Quercia. See FOCAL_ADAPTATION_NOTICE.md.
"""Independent Focal Sampling adapter. Existing HN implementation is untouched."""

from __future__ import annotations

import copy
import importlib.metadata
import math
import platform
import time
from pathlib import Path
from typing import Any

import yaml

from fire_detection.models.offline import sha256_file
from fire_detection.models.training import normalize_ultralytics_device, write_json_atomic
from fire_detection.research.focal_sampling import (
    EpochPlanSampler,
    FocalState,
    IndexedClassificationDataset,
    ScoringCrossEntropy,
    digest,
    epoch_seed,
    seed_worker,
    validate_split_rows,
    write_new_json,
)
from fire_detection.research.protocol import load_config, project_path, publish_contract
from .immutable_inputs import guarded_inputs, snapshot_identity
from .frozen_dataset import build_classification_dataset
from fire_detection.research.training import (
    _new_attempt_directory,
    _trainer_class,
    configure_runtime,
    prepare_view,
    reserved_attempt_train_io,
    train_model,
)


def load_focal_config(path: Path, root: Path) -> tuple[dict, dict]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(spec) != {"base", "campaign", "focal_sampling"}:
        raise ValueError("external config must contain base, campaign, focal_sampling only")
    settings = spec["focal_sampling"]
    if set(settings) != {"enabled", "at_epoch", "gamma"} or type(settings["enabled"]) is not bool:
        raise ValueError("invalid Focal settings")
    base = project_path(root, spec["base"])
    config = load_config(base, root=root)
    if not str(spec["campaign"]).startswith("focal_") or any(c in spec["campaign"] for c in "/\\"):
        raise ValueError("use a separate focal_* campaign name")
    config["campaign"] = spec["campaign"]
    config["_focal_base_sha256"] = sha256_file(base)
    if settings["enabled"]:
        FocalState(
            ["0", "1"],
            [0, 1],
            seed=42,
            fold=0,
            epochs=config["classifier"]["epochs"],
            batch=config["classifier"]["batch"],
            at_epoch=settings["at_epoch"],
            gamma=settings["gamma"],
        )
    return config, settings


def focal_trainer_class(settings: dict, rows: list[dict], seed: int, fold: int, log_dir: Path):
    import torch
    from ultralytics.models.yolo.classify import ClassificationTrainer

    from fire_detection.research.focal_loader import FocalDataLoader

    base = _trainer_class("classifier", arm="hn", hard_names=(), seed=seed)

    class FocalTrainer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback("on_train_epoch_start", self._focal_start)
            self.add_callback("on_train_epoch_end", self._focal_end)

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            if rank not in (-1, 0) or self.world_size > 1:
                raise ValueError("only single-device Focal Sampling is audited")
            source = build_classification_dataset(self, dataset_path, mode, batch_size)
            dataset = IndexedClassificationDataset(source, rows)
            self.focal_state = FocalState(
                dataset.names,
                dataset.labels,
                seed=seed,
                fold=fold,
                at_epoch=settings["at_epoch"],
                gamma=settings["gamma"],
                epochs=self.args.epochs,
                batch=batch_size,
                log_dir=log_dir,
            )
            self.focal_sampler = EpochPlanSampler(len(dataset))
            self.focal_loader_generator = torch.Generator(device="cpu")
            workers = self.args.workers
            return FocalDataLoader(
                dataset,
                batch_size=batch_size,
                sampler=self.focal_sampler,
                drop_last=False,
                num_workers=workers,
                pin_memory=self.device.type == "cuda",
                worker_init_fn=seed_worker,
                generator=self.focal_loader_generator,
                persistent_workers=False,
                **({"multiprocessing_context": "spawn"} if workers else {}),
            )

        def _setup_train(self):
            super()._setup_train()
            if (
                self.args.amp
                or self.args.resume
                or self.args.compile
                or self.args.warmup_epochs != 0
            ):
                raise ValueError("audited adapter requires FP32, no resume/compile/warmup")
            # EMA already exists; validation/checkpoints retain the original CE criterion.
            self.model.criterion = ScoringCrossEntropy(self.focal_state)

        def _focal_start(self, trainer):
            self.focal_sampler.indices = self.focal_state.begin_epoch(int(self.epoch))
            self.focal_loader_generator.manual_seed(epoch_seed(seed, fold, int(self.epoch)) + 1)
            self.focal_step_start = dict(self.research_audit)
            self.focal_epoch_started = time.perf_counter()

        def _focal_end(self, trainer):
            before = self.focal_step_start
            self.focal_state.finish_epoch(
                self.research_audit["optimizer_steps"] - before["optimizer_steps"],
                self.research_audit["optimizer_attempts"] - before["optimizer_attempts"],
            )
            write_new_json(
                log_dir / f"epoch_{self.epoch:03d}_runtime.json",
                {
                    "epoch_zero_based": self.epoch,
                    "training_seconds": time.perf_counter() - self.focal_epoch_started,
                    "learning_rates": [group["lr"] for group in self.optimizer.param_groups],
                    "validation_included": False,
                },
            )

    return FocalTrainer


@guarded_inputs
def train_focal(
    config: dict,
    settings: dict,
    run_dir: Path,
    *,
    splits: dict,
    seed: int,
    fold: int,
    source_sha256: str,
) -> dict[str, Any]:
    validate_split_rows(splits)
    if not settings["enabled"]:
        return train_model(
            config,
            run_dir,
            task="classifier",
            train_rows=splits["fit"],
            val_rows=splits["cal"],
            seed=seed,
            source_contract_sha256=source_sha256,
            arm="hn",
            hard_names=(),
        )
    root = Path(config["_root"])
    campaign = root / "runs" / "direction_a" / config["campaign"]
    if not config["campaign"].startswith("focal_") or not run_dir.resolve().is_relative_to(
        campaign.resolve()
    ):
        raise ValueError("external baseline output must stay in its own campaign")
    n, batch, epochs = (
        len(splits["fit"]),
        config["classifier"]["batch"],
        config["classifier"]["epochs"],
    )
    contract = {
        "input_snapshot": snapshot_identity(config),
        "method": "focal_sampling",
        "budget_policy": config["budget_policy"],
        "budget_policy_version": config["budget_policy_version"],
        "training_protocol_version": config["training_protocol_version"],
        "settings": settings,
        "seed": seed,
        "fold": fold,
        "source_sha256": source_sha256,
        "splits": splits,
        "base_config_sha256": config.get("_focal_base_sha256"),
        "training": config["training"],
        "classifier": config["classifier"],
        "engineering_only": config["engineering_only"],
        "expected_budget": {
            "sample_exposures": n * epochs,
            "optimizer_steps": math.ceil(n / batch) * epochs,
        },
        "initial_weights_sha256": sha256_file(project_path(root, config["classifier"]["weights"])),
        "implementation_sha256": {
            p.name: sha256_file(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("focal_sampling.py"),
                Path(__file__).with_name("focal_loader.py"),
            ]
        },
    }
    publish_contract(run_dir / "focal_contract.json", contract)
    attempt_id, attempt = _new_attempt_directory(run_dir)
    configure_runtime(config)
    import torch
    from ultralytics import YOLO

    data = prepare_view(
        config,
        run_dir / "dataset",
        task="classifier",
        train_rows=splits["fit"],
        val_rows=splits["cal"],
    )
    args = copy.deepcopy(config["training"])
    args.pop("execution_profile", None)
    args["device"] = normalize_ultralytics_device(args["device"])
    args.update(
        data=str(data),
        epochs=epochs,
        batch=batch,
        nbs=batch,
        imgsz=config["classifier"]["imgsz"],
        seed=seed,
        pretrained=True,
        val=True,
        save=True,
        save_period=-1,
        plots=False,
        verbose=False,
        **reserved_attempt_train_io(attempt),
    )
    args.update(config["classifier"]["augmentation"])
    manifest = {
        "status": "running",
        "attempt": attempt_id,
        "contract_sha256": digest(contract),
        "engineering_only": config["engineering_only"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args["device"],
            "packages": {
                p: importlib.metadata.version(p)
                for p in ("ultralytics", "torchvision", "numpy", "Pillow")
            },
        },
    }
    started = time.perf_counter()
    if args["device"] != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        model = YOLO(str(project_path(root, config["classifier"]["weights"])), verbose=False)
        model.train(
            trainer=focal_trainer_class(settings, splits["fit"], seed, fold, attempt / "sampling"),
            **args,
        )
        trainer = model.trainer
        if Path(trainer.save_dir).resolve() != attempt.resolve():
            raise RuntimeError("output escaped reserved directory")
        audit = trainer.focal_state.summary()
        for key in ("sample_exposures", "batches", "optimizer_steps"):
            if audit[key] != trainer.research_audit[key]:
                raise RuntimeError(f"independent audit mismatch: {key}")
        if audit["per_filename_exposures"] != dict(sorted(trainer.exposure_counts.items())):
            raise RuntimeError("filename exposure audit mismatch")
        for key, expected in contract["expected_budget"].items():
            if audit[key] != expected:
                raise RuntimeError(f"budget mismatch: {key}")
        last = attempt / "weights" / "last.pt"
        manifest.update(
            status="complete",
            audit=audit,
            checkpoint="weights/last.pt",
            checkpoint_sha256=sha256_file(last),
            args_sha256=sha256_file(attempt / "args.yaml"),
            sampling_files_sha256={
                p.name: sha256_file(p)
                for p in sorted((attempt / "sampling").iterdir())
                if p.is_file()
            },
        )
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["peak_gpu_memory_bytes"] = (
            int(torch.cuda.max_memory_allocated()) if args['device'] != 'cpu' and torch.cuda.is_available() else None
        )
        write_json_atomic(attempt / "focal_manifest.json", manifest)
    write_new_json(
        run_dir / f"completed_{attempt_id}.json",
        {
            "attempt": f"attempts/{attempt_id}",
            "manifest_sha256": sha256_file(attempt / "focal_manifest.json"),
        },
    )
    return manifest
