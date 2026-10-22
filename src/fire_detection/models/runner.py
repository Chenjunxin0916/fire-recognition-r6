"""Execution boundary for one offline detector or classifier training Fold."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, cast

from fire_detection.models.adapters import (
    build_classification_trainer,
    build_detection_trainer,
    copy_config_snapshot,
)
from fire_detection.models.hard_negative import (
    ReplayContract,
    load_hard_negative_replay,
    validate_nested_split_view,
    validate_outer_classification_view,
)
from fire_detection.models.offline import (
    configure_imported_ultralytics,
    sha256_file,
    validate_offline_runtime,
)
from fire_detection.models.round3_replay import (
    HardSampleReplayContract,
    load_hard_sample_replay,
)
from fire_detection.models.training import (
    TrainingConfigurationError,
    TrainingSpec,
    build_run_manifest,
    load_fold_records,
    validation_labels,
    write_json_atomic,
)

LOGGER = logging.getLogger(__name__)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _validate_training_inputs(spec: TrainingSpec) -> None:
    if spec.run_dir.exists():
        raise TrainingConfigurationError(
            f"run directory already exists and will not be overwritten: {spec.run_dir}"
        )
    if not spec.dataset_path.exists():
        raise TrainingConfigurationError(
            f"derived Fold dataset does not exist: {spec.dataset_path}"
        )
    if spec.task == "detect" and not spec.dataset_path.is_file():
        raise TrainingConfigurationError("detector data.dataset must resolve to a Fold YAML file")
    if spec.task == "classify" and not spec.dataset_path.is_dir():
        raise TrainingConfigurationError("classifier data.dataset must resolve to a Fold directory")
    if not spec.weights_path.is_file():
        raise TrainingConfigurationError(f"initial model weights do not exist: {spec.weights_path}")


def _training_arguments(spec: TrainingSpec) -> dict[str, Any]:
    return {
        "data": str(spec.dataset_path),
        "epochs": spec.epochs,
        "batch": spec.batch,
        "imgsz": spec.imgsz,
        "workers": spec.workers,
        "device": spec.ultralytics_device,
        "amp": spec.amp,
        "deterministic": spec.deterministic,
        "seed": spec.seed,
        "patience": spec.patience,
        "cache": spec.cache,
        "plots": spec.plots,
        "project": str(spec.output_dir),
        "name": spec.run_name,
        "exist_ok": False,
        "pretrained": True,
        "val": True,
        "save": True,
        "save_period": -1,
        "conf": spec.inference_conf,
        "max_det": spec.inference_max_det,
        "augment": spec.inference_augment,
        "verbose": True,
    }


def run_training(spec: TrainingSpec) -> dict[str, Any]:
    """Train one fixed Fold and save ``last.pt`` plus ``best_image_f1.pt``."""

    _validate_training_inputs(spec)
    records = load_fold_records(spec.folds_path, expected_count=1100)
    labels = validation_labels(records, spec.fold)
    replay: ReplayContract | None = None
    hard_sample_replay: HardSampleReplayContract | None = None
    if spec.outer_fold is not None:
        validate_nested_split_view(spec=spec, records=records)
    elif spec.task == "classify":
        validate_outer_classification_view(spec, records)
    if spec.hard_negative_manifest is not None:
        replay = load_hard_negative_replay(
            path=spec.hard_negative_manifest,
            repeat_factor=spec.hard_negative_repeat_factor,
            validation_fold=spec.fold,
            records=records,
            folds_path=spec.folds_path,
            processed_manifest=spec.processed_manifest,
            project_root=spec.project_root,
            expected_seed=spec.seed,
        )
    if spec.hard_sample_replay_manifest is not None:
        if spec.hard_sample_replay_config is None or spec.hard_sample_replay_variant is None:
            raise TrainingConfigurationError("incomplete Round-3 hard-sample replay configuration")
        hard_sample_replay = load_hard_sample_replay(
            config_path=spec.hard_sample_replay_config,
            manifest_path=spec.hard_sample_replay_manifest,
            variant=spec.hard_sample_replay_variant,
            repeat_factor=spec.hard_sample_repeat_factor,
            validation_fold=spec.fold,
            records=records,
        )
    runtime = validate_offline_runtime(
        project_root=spec.project_root,
        manifest_path=spec.offline_manifest,
        yolo_config_dir=spec.yolo_config_dir,
        required_assets=(spec.weights_path.name, "yolo26n.pt", "Arial.ttf"),
        expected_ultralytics_version=spec.expected_ultralytics_version,
    )
    configured_weight = runtime.require(spec.weights_path.name)
    if configured_weight.path != spec.weights_path.resolve():
        raise TrainingConfigurationError(
            f"config weights {spec.weights_path} do not match offline manifest "
            f"{configured_weight.path}"
        )

    # Imports below this point are intentionally after configure_offline_environment().
    configure_imported_ultralytics(runtime)
    import torch

    from fire_detection.environment import collect_environment_report
    from ultralytics import YOLO

    if spec.ultralytics_device != "cpu" and not torch.cuda.is_available():
        raise TrainingConfigurationError(
            f"CUDA device {spec.requested_device!r} was requested but "
            "torch.cuda.is_available() is false"
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = (
        build_detection_trainer(spec, labels)
        if spec.task == "detect"
        else build_classification_trainer(
            spec,
            labels,
            hard_negative_filenames=(replay.hard_negative_filenames if replay is not None else ()),
            hard_sample_replay_samples=(
                hard_sample_replay.replay_samples if hard_sample_replay is not None else ()
            ),
            repeat_factor=(
                replay.repeat_factor
                if replay is not None
                else (hard_sample_replay.repeat_factor if hard_sample_replay is not None else 1)
            ),
        )
    )
    started = time.time()
    LOGGER.info(
        "Starting %s Fold %d: data=%s weights=%s imgsz=%d batch=%d epochs=%d device=%s",
        spec.task,
        spec.fold,
        spec.dataset_path,
        spec.weights_path,
        spec.imgsz,
        spec.batch,
        spec.epochs,
        spec.requested_device,
    )
    model = YOLO(str(spec.weights_path), verbose=False)
    status = "failed"
    failure: str | None = None
    last = spec.run_dir / "weights" / "last.pt"
    best_image = spec.run_dir / "weights" / "best_image_f1.pt"
    best_metrics = best_image.with_suffix(".metrics.json")
    try:
        model.train(trainer=trainer, **_training_arguments(spec))
        for required in (last, best_image, best_metrics, spec.run_dir / "args.yaml"):
            if not required.is_file():
                raise TrainingConfigurationError(
                    f"training completed without required artifact: {required}"
                )
        status = "trained"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if spec.run_dir.exists():
            additional: dict[str, Any] | None = None
            if replay is not None:
                additional = {"hard_negative_replay": replay.manifest_section(spec.project_root)}
            elif hard_sample_replay is not None:
                additional = {
                    "hard_sample_replay": hard_sample_replay.manifest_section(spec.project_root)
                }
            manifest = build_run_manifest(
                spec,
                status=status,
                started_at=started,
                additional=additional,
            )
            environment = collect_environment_report().to_dict()
            environment["amp"] = spec.amp
            manifest["environment"] = environment
            manifest["peak_gpu_memory_bytes"] = (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            )
            if failure is not None:
                manifest["failure"] = failure
            write_json_atomic(spec.run_dir / "run_manifest.json", manifest)

    copy_config_snapshot(spec)
    manifest_path = spec.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete",
            "last_weights": _relative(last, spec.project_root),
            "last_weights_sha256": sha256_file(last),
            "best_image_f1_weights": _relative(best_image, spec.project_root),
            "best_image_f1_weights_sha256": sha256_file(best_image),
            "best_image_f1_metrics": json.loads(best_metrics.read_text(encoding="utf-8")),
        }
    )
    write_json_atomic(manifest_path, manifest)
    return cast(dict[str, Any], manifest)
