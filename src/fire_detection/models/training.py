"""Validated training configuration, Fold provenance, and image-level checkpoint state."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from fire_detection.models.offline import OfflinePreparationError, sha256_file

TaskName = Literal["detect", "classify"]


class TrainingConfigurationError(ValueError):
    """Raised when a training configuration omits or contradicts required state."""


@dataclass(frozen=True)
class FoldRecord:
    """One immutable row from the fixed Fold CSV."""

    filename: str
    label: int
    group_id: str
    fold: int
    detector_eligible: bool = True


@dataclass(frozen=True)
class ThresholdSummary:
    """Image-level metrics selected for one epoch or OOF prediction table."""

    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int

    def to_dict(self) -> dict[str, float | int]:
        """Return stable metric keys."""

        return asdict(self)


@dataclass(frozen=True)
class TrainingSpec:
    """Decision-complete runtime values for one YOLO Fold."""

    project_root: Path
    config_path: Path
    task: TaskName
    experiment_name: str
    run_name: str
    fold: int
    seed: int
    folds_path: Path
    images_path: Path
    dataset_path: Path
    processed_manifest: Path
    model_name: str
    weights_path: Path
    offline_manifest: Path
    yolo_config_dir: Path
    expected_ultralytics_version: str
    epochs: int
    batch: int
    imgsz: int
    workers: int
    requested_device: str
    ultralytics_device: str
    amp: bool
    deterministic: bool
    patience: int
    cache: bool | str
    plots: bool
    output_dir: Path
    inference_conf: float
    inference_max_det: int
    inference_augment: bool
    threshold_start: float
    threshold_end: float
    threshold_step: float
    outer_fold: int | None
    split_manifest: Path | None
    hard_negative_manifest: Path | None
    hard_negative_repeat_factor: int
    hard_sample_replay_config: Path | None
    hard_sample_replay_manifest: Path | None
    hard_sample_replay_variant: str | None
    hard_sample_repeat_factor: int

    @property
    def run_dir(self) -> Path:
        """Return the exact run directory expected from Ultralytics."""

        return self.output_dir / self.run_name

    @property
    def training_folds(self) -> tuple[int, ...]:
        """Return Fold-v3 partitions that are allowed to contribute training samples."""

        excluded = {self.fold}
        if self.outer_fold is not None:
            excluded.add(self.outer_fold)
        return tuple(fold for fold in range(5) if fold not in excluded)

    @property
    def excluded_folds(self) -> tuple[int, ...]:
        """Return Fold-v3 partitions excluded in addition to validation."""

        return () if self.outer_fold is None else (self.outer_fold,)


def _sha256_training_dataset(spec: TrainingSpec) -> str:
    """Hash the complete Fold-specific dataset tree used by one training run."""

    if spec.task == "detect":
        if not spec.dataset_path.is_file():
            raise TrainingConfigurationError(
                f"detector dataset YAML does not exist: {spec.dataset_path}"
            )
        root = spec.dataset_path.parent
    else:
        if not spec.dataset_path.is_dir():
            raise TrainingConfigurationError(
                f"classifier dataset directory does not exist: {spec.dataset_path}"
            )
        root = spec.dataset_path
    if root.is_symlink():
        raise TrainingConfigurationError(f"training dataset root must not be a symlink: {root}")

    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    if not entries:
        raise TrainingConfigurationError(f"training dataset is empty: {root}")
    digest = hashlib.sha256(b"fire-training-dataset-tree-v1\0")
    for entry in entries:
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if entry.is_symlink():
            raise TrainingConfigurationError(
                f"training dataset entries must not be symlinks: {entry}"
            )
        if entry.is_dir():
            digest.update(b"D")
            continue
        if not entry.is_file():
            raise TrainingConfigurationError(f"unsupported training dataset entry: {entry}")
        digest.update(b"F")
        try:
            digest.update(bytes.fromhex(sha256_file(entry)))
        except OfflinePreparationError as exc:
            raise TrainingConfigurationError(
                f"cannot hash training dataset entry {entry}: {exc}"
            ) from exc
    return digest.hexdigest()


def normalize_ultralytics_device(value: str) -> str:
    """Normalize documented device forms to Ultralytics' single-device notation."""

    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigurationError("device must be a non-empty string")
    normalized = value.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return "0"
    if normalized.startswith("cuda:"):
        index = normalized.partition(":")[2]
        if not index.isdigit():
            raise TrainingConfigurationError(f"invalid CUDA device: {value!r}")
        return index
    if normalized.isdigit():
        return normalized
    raise TrainingConfigurationError("device must be cpu, cuda, cuda:N, or a numeric GPU index")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingConfigurationError(f"{field} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, field: str) -> Any:
    if key not in mapping:
        raise TrainingConfigurationError(f"missing required configuration field {field}.{key}")
    return mapping[key]


def _strict_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TrainingConfigurationError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise TrainingConfigurationError(f"{field} must be a boolean")
    return value


def _finite_float(
    value: Any, field: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigurationError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingConfigurationError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise TrainingConfigurationError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise TrainingConfigurationError(f"{field} must be <= {maximum}")
    return result


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigurationError(f"{field} must be a non-empty string")
    return value.strip()


def _resolve_path(
    root: Path,
    value: Any,
    field: str,
    *,
    fold: int | None = None,
    outer_fold: int | None = None,
) -> Path:
    raw = _string(value, field)
    format_values: dict[str, int] = {}
    if fold is not None:
        format_values["fold"] = fold
    if outer_fold is not None:
        format_values["outer_fold"] = outer_fold
    if "{" in raw:
        try:
            raw = raw.format(**format_values)
        except (KeyError, ValueError) as exc:
            raise TrainingConfigurationError(
                f"invalid or unresolved Fold template in {field}: {raw!r}"
            ) from exc
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_training_spec(
    path: Path,
    *,
    expected_task: TaskName,
    fold_override: int | None = None,
    epochs_override: int | None = None,
    device_override: str | None = None,
    run_name_override: str | None = None,
    outer_fold_override: int | None = None,
) -> TrainingSpec:
    """Load and strictly validate one detector or classifier YAML configuration."""

    config_path = path.resolve()
    project_root = config_path.parent.parent
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TrainingConfigurationError(
            f"cannot read training config {config_path}: {exc}"
        ) from exc
    root = _mapping(raw, "training config")
    experiment = _mapping(_required(root, "experiment", "config"), "experiment")
    data = _mapping(_required(root, "data", "config"), "data")
    model = _mapping(_required(root, "model", "config"), "model")
    train = _mapping(_required(root, "train", "config"), "train")
    evaluation = _mapping(_required(root, "evaluation", "config"), "evaluation")
    offline = _mapping(_required(root, "offline", "config"), "offline")

    task = _string(_required(model, "task", "model"), "model.task")
    if task != expected_task:
        raise TrainingConfigurationError(f"config task is {task!r}, expected {expected_task!r}")
    raw_fold = (
        fold_override if fold_override is not None else _required(experiment, "fold", "experiment")
    )
    fold = _strict_int(raw_fold, "experiment.fold")
    if fold not in range(5):
        raise TrainingConfigurationError("experiment.fold must be in 0..4")
    outer_fold: int | None = None
    if outer_fold_override is not None:
        outer_fold = _strict_int(outer_fold_override, "outer_fold")
        if outer_fold not in range(5):
            raise TrainingConfigurationError("outer_fold must be in 0..4")
        if expected_task != "classify":
            raise TrainingConfigurationError("outer_fold is supported only for classifier miners")
        if outer_fold == fold:
            raise TrainingConfigurationError(
                "outer_fold must differ from the inner validation Fold"
            )
    experiment_name = _string(_required(experiment, "name", "experiment"), "experiment.name")
    default_run_name = (
        f"{experiment_name}_fold{fold}"
        if outer_fold is None
        else f"{experiment_name}_outer{outer_fold}_inner{fold}"
    )
    run_name = run_name_override or default_run_name
    run_name = _string(run_name, "run_name")
    seed = _strict_int(_required(experiment, "seed", "experiment"), "experiment.seed")
    requested_device = device_override or _string(
        _required(train, "device", "train"), "train.device"
    )
    epochs_value = (
        epochs_override if epochs_override is not None else _required(train, "epochs", "train")
    )

    cache = _required(train, "cache", "train")
    if not isinstance(cache, (bool, str)):
        raise TrainingConfigurationError("train.cache must be a boolean or cache mode string")
    threshold_start = _finite_float(
        _required(evaluation, "threshold_start", "evaluation"),
        "evaluation.threshold_start",
        minimum=0.0,
        maximum=1.0,
    )
    threshold_end = _finite_float(
        _required(evaluation, "threshold_end", "evaluation"),
        "evaluation.threshold_end",
        minimum=0.0,
        maximum=1.0,
    )
    threshold_step = _finite_float(
        _required(evaluation, "threshold_step", "evaluation"),
        "evaluation.threshold_step",
        minimum=1e-12,
        maximum=1.0,
    )
    if threshold_start > threshold_end:
        raise TrainingConfigurationError("threshold_start must not exceed threshold_end")

    split_manifest: Path | None = None
    split_manifest_value = data.get("split_manifest")
    if split_manifest_value is not None:
        if outer_fold is None:
            raise TrainingConfigurationError(
                "data.split_manifest requires an explicit outer_fold for nested mining"
            )
        split_manifest = _resolve_path(
            project_root,
            split_manifest_value,
            "data.split_manifest",
            fold=fold,
            outer_fold=outer_fold,
        )
    elif outer_fold is not None:
        raise TrainingConfigurationError("nested classifier mining requires data.split_manifest")

    hard_negative_manifest: Path | None = None
    hard_negative_repeat_factor = 1
    replay_value = root.get("hard_negative_replay")
    if replay_value is not None:
        if expected_task != "classify":
            raise TrainingConfigurationError(
                "hard_negative_replay is supported only for classifier training"
            )
        if outer_fold is not None:
            raise TrainingConfigurationError(
                "nested miner training cannot also enable hard-negative replay"
            )
        replay = _mapping(replay_value, "hard_negative_replay")
        hard_negative_manifest = _resolve_path(
            project_root,
            _required(replay, "manifest", "hard_negative_replay"),
            "hard_negative_replay.manifest",
            fold=fold,
        )
        hard_negative_repeat_factor = _strict_int(
            _required(replay, "repeat_factor", "hard_negative_replay"),
            "hard_negative_replay.repeat_factor",
            minimum=2,
        )
        if hard_negative_repeat_factor not in {2, 3}:
            raise TrainingConfigurationError("hard_negative_replay.repeat_factor must be 2 or 3")

    hard_sample_replay_config: Path | None = None
    hard_sample_replay_manifest: Path | None = None
    hard_sample_replay_variant: str | None = None
    hard_sample_repeat_factor = 1
    hard_sample_value = root.get("hard_sample_replay")
    if hard_sample_value is not None:
        if expected_task != "classify":
            raise TrainingConfigurationError(
                "hard_sample_replay is supported only for classifier training"
            )
        if outer_fold is not None:
            raise TrainingConfigurationError(
                "nested miner training cannot also enable hard-sample replay"
            )
        if hard_negative_manifest is not None:
            raise TrainingConfigurationError(
                "hard_negative_replay and hard_sample_replay are mutually exclusive"
            )
        hard_sample = _mapping(hard_sample_value, "hard_sample_replay")
        hard_sample_replay_config = _resolve_path(
            project_root,
            _required(hard_sample, "campaign_config", "hard_sample_replay"),
            "hard_sample_replay.campaign_config",
            fold=fold,
        )
        hard_sample_replay_manifest = _resolve_path(
            project_root,
            _required(hard_sample, "manifest", "hard_sample_replay"),
            "hard_sample_replay.manifest",
            fold=fold,
        )
        hard_sample_replay_variant = _string(
            _required(hard_sample, "variant", "hard_sample_replay"),
            "hard_sample_replay.variant",
        )
        hard_sample_repeat_factor = _strict_int(
            _required(hard_sample, "repeat_factor", "hard_sample_replay"),
            "hard_sample_replay.repeat_factor",
            minimum=2,
        )
        if hard_sample_repeat_factor != 2:
            raise TrainingConfigurationError("hard_sample_replay.repeat_factor must be exactly 2")

    spec = TrainingSpec(
        project_root=project_root,
        config_path=config_path,
        task=expected_task,
        experiment_name=experiment_name,
        run_name=run_name,
        fold=fold,
        seed=seed,
        folds_path=_resolve_path(project_root, _required(data, "folds", "data"), "data.folds"),
        images_path=_resolve_path(project_root, _required(data, "images", "data"), "data.images"),
        dataset_path=_resolve_path(
            project_root,
            _required(data, "dataset", "data"),
            "data.dataset",
            fold=fold,
            outer_fold=outer_fold,
        ),
        processed_manifest=_resolve_path(
            project_root,
            _required(data, "processed_manifest", "data"),
            "data.processed_manifest",
        ),
        model_name=_string(_required(model, "name", "model"), "model.name"),
        weights_path=_resolve_path(
            project_root, _required(model, "weights", "model"), "model.weights"
        ),
        offline_manifest=_resolve_path(
            project_root, _required(offline, "manifest", "offline"), "offline.manifest"
        ),
        yolo_config_dir=_resolve_path(
            project_root,
            _required(offline, "yolo_config_dir", "offline"),
            "offline.yolo_config_dir",
        ),
        expected_ultralytics_version=_string(
            _required(offline, "ultralytics_version", "offline"),
            "offline.ultralytics_version",
        ),
        epochs=_strict_int(epochs_value, "train.epochs", minimum=1),
        batch=_strict_int(_required(train, "batch", "train"), "train.batch", minimum=1),
        imgsz=_strict_int(_required(train, "imgsz", "train"), "train.imgsz", minimum=32),
        workers=_strict_int(_required(train, "workers", "train"), "train.workers"),
        requested_device=requested_device,
        ultralytics_device=normalize_ultralytics_device(requested_device),
        amp=_strict_bool(_required(train, "amp", "train"), "train.amp"),
        deterministic=_strict_bool(
            _required(train, "deterministic", "train"), "train.deterministic"
        ),
        patience=_strict_int(_required(train, "patience", "train"), "train.patience"),
        cache=cache,
        plots=_strict_bool(_required(train, "plots", "train"), "train.plots"),
        output_dir=_resolve_path(
            project_root, _required(train, "output_dir", "train"), "train.output_dir"
        ),
        inference_conf=_finite_float(
            _required(evaluation, "inference_conf", "evaluation"),
            "evaluation.inference_conf",
            minimum=0.0,
            maximum=1.0,
        ),
        inference_max_det=_strict_int(
            _required(evaluation, "inference_max_det", "evaluation"),
            "evaluation.inference_max_det",
            minimum=1,
        ),
        inference_augment=_strict_bool(
            _required(evaluation, "inference_augment", "evaluation"),
            "evaluation.inference_augment",
        ),
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        threshold_step=threshold_step,
        outer_fold=outer_fold,
        split_manifest=split_manifest,
        hard_negative_manifest=hard_negative_manifest,
        hard_negative_repeat_factor=hard_negative_repeat_factor,
        hard_sample_replay_config=hard_sample_replay_config,
        hard_sample_replay_manifest=hard_sample_replay_manifest,
        hard_sample_replay_variant=hard_sample_replay_variant,
        hard_sample_repeat_factor=hard_sample_repeat_factor,
    )
    if spec.patience != 0:
        raise TrainingConfigurationError("train.patience must be 0 for fixed-epoch OOF training")
    if spec.outer_fold is not None:
        frozen_v3 = (spec.project_root / "data/processed/v3").resolve()
        try:
            spec.dataset_path.resolve().relative_to(frozen_v3)
        except ValueError:
            pass
        else:
            raise TrainingConfigurationError(
                "nested miner datasets must not be placed inside frozen data/processed/v3"
            )
    return spec


def load_fold_records(path: Path, *, expected_count: int | None = None) -> tuple[FoldRecord, ...]:
    """Load the fixed Fold CSV and reject duplicate or malformed provenance."""

    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"filename", "label", "group_id", "fold"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise TrainingConfigurationError(
                    f"Fold CSV must contain {sorted(required)}, found {reader.fieldnames}"
                )
            rows = list(reader)
    except OSError as exc:
        raise TrainingConfigurationError(f"cannot read Fold CSV {path}: {exc}") from exc
    records: list[FoldRecord] = []
    names: set[str] = set()
    for offset, row in enumerate(rows, start=2):
        filename = _string(row.get("filename"), f"folds.csv:{offset}.filename")
        if Path(filename).name != filename:
            raise TrainingConfigurationError(f"Fold filename must be a basename: {filename!r}")
        if filename in names:
            raise TrainingConfigurationError(f"duplicate Fold filename: {filename}")
        try:
            label = int(row["label"])
            fold = int(row["fold"])
        except (TypeError, ValueError) as exc:
            raise TrainingConfigurationError(
                f"invalid label/fold at Fold CSV line {offset}"
            ) from exc
        if str(label) != str(row["label"]).strip() or label not in {0, 1}:
            raise TrainingConfigurationError(f"invalid binary label at Fold CSV line {offset}")
        if str(fold) != str(row["fold"]).strip() or fold not in range(5):
            raise TrainingConfigurationError(f"invalid Fold number at Fold CSV line {offset}")
        group_id = _string(row.get("group_id"), f"folds.csv:{offset}.group_id")
        names.add(filename)
        detector_value = str(row.get("detector_eligible", "true")).strip().lower()
        if detector_value not in {"true", "false", "1", "0"}:
            raise TrainingConfigurationError(
                f"invalid detector_eligible at Fold CSV line {offset}"
            )
        detector_eligible = detector_value in {"true", "1"}
        records.append(FoldRecord(filename, label, group_id, fold, detector_eligible))
    records.sort(key=lambda item: item.filename)
    if expected_count is not None and len(records) != expected_count:
        raise TrainingConfigurationError(
            f"Fold CSV contains {len(records)} rows, expected {expected_count}"
        )
    return tuple(records)


def validation_labels(records: Sequence[FoldRecord], fold: int) -> dict[str, int]:
    """Return filename-to-label mapping for exactly one validation Fold."""

    result = {record.filename: record.label for record in records if record.fold == fold}
    if not result:
        raise TrainingConfigurationError(f"Fold {fold} has no validation rows")
    if set(result.values()) != {0, 1}:
        raise TrainingConfigurationError(f"Fold {fold} must contain both binary classes")
    return result


def _metric_value(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def coerce_threshold_summary(result: Any) -> ThresholdSummary:
    """Adapt the public metrics search result without binding to its storage layout."""

    point = _metric_value(result, "best") or _metric_value(result, "best_point") or result
    metrics = _metric_value(point, "metrics") or point
    threshold = _metric_value(point, "threshold")
    if threshold is None:
        threshold = _metric_value(result, "threshold")
    values = {
        name: _metric_value(metrics, name)
        for name in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")
    }
    if threshold is None or any(value is None for value in values.values()):
        raise TrainingConfigurationError(
            "threshold search result does not expose the required metrics"
        )
    return ThresholdSummary(
        threshold=float(threshold),
        precision=float(values["precision"]),
        recall=float(values["recall"]),
        f1=float(values["f1"]),
        tp=int(values["tp"]),
        fp=int(values["fp"]),
        fn=int(values["fn"]),
        tn=int(values["tn"]),
    )


SearchFunction = Callable[..., Any]


def search_image_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold_start: float,
    threshold_end: float,
    threshold_step: float,
    search_function: SearchFunction | None = None,
) -> ThresholdSummary:
    """Run the shared threshold search and normalize its selected point."""

    if search_function is None:
        try:
            from fire_detection.metrics import search_thresholds
        except ImportError as exc:
            raise TrainingConfigurationError(
                "fire_detection.metrics.search_thresholds is required for "
                "image-level model selection"
            ) from exc
        function = cast(SearchFunction, search_thresholds)
    else:
        function = search_function
    result = function(
        labels,
        scores,
        start=threshold_start,
        end=threshold_end,
        step=threshold_step,
    )
    return coerce_threshold_summary(result)


class ImageScoreAccumulator:
    """Collect one score per expected validation image and reject incomplete epochs."""

    def __init__(self, expected_labels: Mapping[str, int]) -> None:
        self._expected = dict(expected_labels)
        self._scores: dict[str, float] = {}

    def add(self, filename: str, score: float) -> None:
        """Add one validated image probability."""

        name = Path(filename).name
        if name not in self._expected:
            raise TrainingConfigurationError(f"validation produced unexpected image {name!r}")
        if name in self._scores:
            raise TrainingConfigurationError(f"validation produced duplicate image {name!r}")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TrainingConfigurationError(f"score for {name!r} must be numeric")
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise TrainingConfigurationError(f"score for {name!r} must be finite and in [0, 1]")
        self._scores[name] = value

    def ordered(self) -> tuple[list[int], list[float]]:
        """Return labels and scores in deterministic filename order."""

        missing = sorted(set(self._expected) - set(self._scores))
        if missing:
            raise TrainingConfigurationError(
                f"validation is missing {len(missing)} expected images, first={missing[:5]}"
            )
        names = sorted(self._expected)
        return [self._expected[name] for name in names], [self._scores[name] for name in names]


class ImageCheckpointSelector:
    """Keep the earliest epoch with the best lexicographic ``(F1, Recall)``."""

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best: ThresholdSummary | None = None

    def update(self, epoch: int, summary: ThresholdSummary) -> bool:
        """Return true only when the candidate strictly improves F1 then Recall."""

        if type(epoch) is not int or epoch < 0:
            raise TrainingConfigurationError("epoch must be a non-negative integer")
        if self.best is None or (summary.f1, summary.recall) > (self.best.f1, self.best.recall):
            self.best_epoch = epoch
            self.best = summary
            return True
        return False


def append_epoch_metrics(path: Path, epoch: int, summary: ThresholdSummary) -> None:
    """Append an auditable image-level epoch summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", *summary.to_dict().keys()]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({"epoch": epoch, **summary.to_dict()})


def save_best_image_checkpoint(
    *, source_last: Path, destination: Path, epoch: int, summary: ThresholdSummary
) -> None:
    """Copy the current resumable checkpoint and write image-level selection metadata."""

    if not source_last.is_file():
        raise TrainingConfigurationError(f"last checkpoint does not exist: {source_last}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_last, destination)
    metadata = destination.with_suffix(".metrics.json")
    payload = {"epoch": epoch, **summary.to_dict()}
    metadata.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def git_state(project_root: Path) -> dict[str, object]:
    """Return commit and dirty state without mutating Git."""

    provenance = Path(project_root) / "provenance/manifest.json"
    if provenance.is_file():
        from fire_detection.source_provenance import verify_execution_source
        return verify_execution_source(project_root)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TrainingConfigurationError(f"cannot determine Git state: {exc}") from exc
    return {"commit": commit, "dirty": dirty}


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON object atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def build_run_manifest(
    spec: TrainingSpec,
    *,
    status: str,
    started_at: float,
    additional: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the reproducibility manifest common to detector and classifier runs."""

    if not spec.folds_path.is_file():
        raise TrainingConfigurationError(f"Fold CSV does not exist: {spec.folds_path}")
    if not spec.processed_manifest.is_file():
        raise TrainingConfigurationError(
            f"processed-data manifest does not exist: {spec.processed_manifest}"
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "task": spec.task,
        "experiment_name": spec.experiment_name,
        "run_name": spec.run_name,
        "validation_fold": spec.fold,
        "training_folds": list(spec.training_folds),
        "seed": spec.seed,
        "model_name": spec.model_name,
        "initial_weights": spec.weights_path.relative_to(spec.project_root).as_posix(),
        "initial_weights_sha256": sha256_file(spec.weights_path),
        "config_path": spec.config_path.relative_to(spec.project_root).as_posix(),
        "config_sha256": sha256_file(spec.config_path),
        "folds_path": spec.folds_path.relative_to(spec.project_root).as_posix(),
        "folds_sha256": sha256_file(spec.folds_path),
        "processed_manifest": spec.processed_manifest.relative_to(spec.project_root).as_posix(),
        "processed_manifest_sha256": sha256_file(spec.processed_manifest),
        "dataset_path": spec.dataset_path.relative_to(spec.project_root).as_posix(),
        "dataset_sha256": _sha256_training_dataset(spec),
        "requested_device": spec.requested_device,
        "ultralytics_device": spec.ultralytics_device,
        "imgsz": spec.imgsz,
        "batch": spec.batch,
        "epochs": spec.epochs,
        "started_at_unix": started_at,
        "elapsed_seconds": max(0.0, time.time() - started_at),
        "git": git_state(spec.project_root),
    }
    if spec.outer_fold is not None:
        manifest["outer_fold"] = spec.outer_fold
        manifest["excluded_folds"] = list(spec.excluded_folds)
    if spec.split_manifest is not None:
        manifest["split_manifest"] = spec.split_manifest.relative_to(spec.project_root).as_posix()
        manifest["split_manifest_sha256"] = sha256_file(spec.split_manifest)
    if additional is not None:
        overlap = sorted(set(manifest) & set(additional))
        if overlap:
            raise TrainingConfigurationError(
                f"additional run-manifest fields overlap base fields: {overlap}"
            )
        manifest.update(dict(additional))
    return manifest
