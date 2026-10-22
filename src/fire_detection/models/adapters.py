"""Ultralytics 8.4.92 adapters for image-level F1 checkpoint selection."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fire_detection.models.features import resolve_positive_class_index
from fire_detection.models.hard_negative import build_replay_indices
from fire_detection.models.round3_replay import build_hard_sample_replay_indices
from fire_detection.models.training import (
    ImageCheckpointSelector,
    ImageScoreAccumulator,
    TrainingConfigurationError,
    TrainingSpec,
    append_epoch_metrics,
    save_best_image_checkpoint,
    search_image_metrics,
)


def _score_from_tensor(value: Any) -> float:
    if hasattr(value, "numel") and int(value.numel()) == 0:
        return 0.0
    maximum = value.max() if hasattr(value, "max") else max(value)
    return float(maximum.item() if hasattr(maximum, "item") else maximum)


if TYPE_CHECKING:

    class _TrainerShape:
        wdir: Path
        validator: Any
        epoch: int
        save_dir: Path
        last: Path

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        def save_model(self) -> bool: ...

else:
    _TrainerShape = object


class _ImageCheckpointMixin(_TrainerShape):
    image_selector: ImageCheckpointSelector
    best_image_f1: Path

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.image_selector = ImageCheckpointSelector()
        self.best_image_f1 = self.wdir / "best_image_f1.pt"

    def save_model(self) -> bool:
        saved = bool(super().save_model())
        summary = getattr(self.validator, "image_summary", None)
        if summary is None:
            raise TrainingConfigurationError(
                "custom validator did not produce image-level metrics before checkpoint save"
            )
        epoch = int(self.epoch)
        append_epoch_metrics(self.save_dir / "image_metrics_by_epoch.csv", epoch, summary)
        if self.image_selector.update(epoch, summary):
            save_best_image_checkpoint(
                source_last=Path(self.last),
                destination=Path(self.best_image_f1),
                epoch=epoch,
                summary=summary,
            )
        return saved


class _NamedClassificationDataset:
    """Pickle-safe wrapper that retains source filenames for image-level validation."""

    def __init__(
        self,
        *,
        root: str,
        args: Any,
        augment: bool,
        prefix: str,
        hard_negative_filenames: tuple[str, ...] = (),
        hard_sample_replay_samples: tuple[tuple[str, int], ...] = (),
        repeat_factor: int = 1,
    ) -> None:
        from ultralytics.data.dataset import ClassificationDataset

        self._dataset = ClassificationDataset(
            root=root,
            args=args,
            augment=augment,
            prefix=prefix,
        )
        if hard_negative_filenames and hard_sample_replay_samples:
            raise TrainingConfigurationError(
                "legacy hard-negative and Round-3 hard-sample replay are mutually exclusive"
            )
        if augment and hard_sample_replay_samples:
            self._indices = build_hard_sample_replay_indices(
                self._dataset.samples,
                hard_sample_replay_samples,
                repeat_factor=repeat_factor,
            )
        elif augment and hard_negative_filenames:
            self._indices = build_replay_indices(
                self._dataset.samples,
                hard_negative_filenames,
                repeat_factor=repeat_factor,
            )
        else:
            self._indices = tuple(range(len(self._dataset)))

    def __len__(self) -> int:
        indices = getattr(self, "_indices", None)
        return len(self._dataset) if indices is None else len(indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        indices = getattr(self, "_indices", None)
        source_index = index if indices is None else indices[index]
        sample = dict(self._dataset[source_index])
        sample["im_file"] = str(self._dataset.samples[source_index][0])
        return sample

    def __getattr__(self, name: str) -> Any:
        dataset = object.__getattribute__(self, "_dataset")
        return getattr(dataset, name)


def build_detection_trainer(spec: TrainingSpec, expected_labels: Mapping[str, int]) -> type[Any]:
    """Build a DetectionTrainer subclass after offline environment configuration."""

    from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

    class ImageF1DetectionValidator(DetectionValidator):
        image_summary: Any
        image_scores: ImageScoreAccumulator

        def init_metrics(self, model: Any) -> None:
            super().init_metrics(model)
            if not self.end2end:
                raise TrainingConfigurationError(
                    "YOLO26 baseline must retain official end-to-end, NMS-free inference"
                )
            self.image_scores = ImageScoreAccumulator(expected_labels)
            self.image_summary = None

        def update_metrics(self, preds: list[dict[str, Any]], batch: dict[str, Any]) -> None:
            paths = batch.get("im_file")
            if not isinstance(paths, (list, tuple)) or len(paths) != len(preds):
                raise TrainingConfigurationError(
                    "detection validation batch does not expose one im_file per prediction"
                )
            for path, pred in zip(paths, preds, strict=True):
                self.image_scores.add(Path(str(path)).name, _score_from_tensor(pred["conf"]))
            super().update_metrics(preds, batch)

        def get_stats(self) -> dict[str, Any]:
            stats = dict(super().get_stats())
            labels, scores = self.image_scores.ordered()
            self.image_summary = search_image_metrics(
                labels,
                scores,
                threshold_start=spec.threshold_start,
                threshold_end=spec.threshold_end,
                threshold_step=spec.threshold_step,
            )
            stats.update(
                {f"image/{key}": value for key, value in self.image_summary.to_dict().items()}
            )
            return stats

    class ImageF1DetectionTrainer(_ImageCheckpointMixin, DetectionTrainer):
        def get_validator(self) -> Any:
            self.loss_names = ["box_loss", "cls_loss", "dfl_loss"]
            return ImageF1DetectionValidator(
                self.test_loader,
                save_dir=self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
            )

    ImageF1DetectionTrainer.__name__ = "ImageF1DetectionTrainer"
    return ImageF1DetectionTrainer


def build_classification_trainer(
    spec: TrainingSpec,
    expected_labels: Mapping[str, int],
    *,
    hard_negative_filenames: tuple[str, ...] = (),
    hard_sample_replay_samples: tuple[tuple[str, int], ...] = (),
    repeat_factor: int = 1,
) -> type[Any]:
    """Build a ClassificationTrainer that retains filenames and positive probabilities."""

    from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator

    class ImageF1ClassificationValidator(ClassificationValidator):
        image_summary: Any
        image_scores: ImageScoreAccumulator
        positive_index: int

        def init_metrics(self, model: Any) -> None:
            super().init_metrics(model)
            names = self.names
            if not isinstance(names, dict):
                raise TrainingConfigurationError("classification validator has no class mapping")
            values = {str(value) for value in names.values()}
            if values != {"0", "1"}:
                raise TrainingConfigurationError(
                    f"classification class names must be exactly '0' and '1', found {self.names}"
                )
            self.positive_index = resolve_positive_class_index(names, "1")
            self.image_scores = ImageScoreAccumulator(expected_labels)
            self.image_summary = None

        def update_metrics(self, preds: Any, batch: dict[str, Any]) -> None:
            paths = batch.get("im_file")
            if not isinstance(paths, (list, tuple)) or len(paths) != len(preds):
                raise TrainingConfigurationError(
                    "classification validation batch does not expose one im_file per prediction"
                )
            positive = preds[:, self.positive_index]
            scores = positive.detach().cpu().tolist()
            for path, score in zip(paths, scores, strict=True):
                self.image_scores.add(Path(str(path)).name, float(score))
            super().update_metrics(preds, batch)

        def get_stats(self) -> dict[str, Any]:
            stats = dict(super().get_stats())
            labels, scores = self.image_scores.ordered()
            self.image_summary = search_image_metrics(
                labels,
                scores,
                threshold_start=spec.threshold_start,
                threshold_end=spec.threshold_end,
                threshold_step=spec.threshold_step,
            )
            stats.update(
                {f"image/{key}": value for key, value in self.image_summary.to_dict().items()}
            )
            return stats

    class ImageF1ClassificationTrainer(_ImageCheckpointMixin, ClassificationTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch: Any = None) -> Any:
            return _NamedClassificationDataset(
                root=img_path,
                args=self.args,
                augment=mode == "train",
                prefix=mode,
                hard_negative_filenames=hard_negative_filenames,
                hard_sample_replay_samples=hard_sample_replay_samples,
                repeat_factor=repeat_factor,
            )

        def get_validator(self) -> Any:
            self.loss_names = ["loss"]
            return ImageF1ClassificationValidator(
                self.test_loader,
                self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
            )

    ImageF1ClassificationTrainer.__name__ = "ImageF1ClassificationTrainer"
    return ImageF1ClassificationTrainer


def copy_config_snapshot(spec: TrainingSpec) -> Path:
    """Copy the user-facing config into the exact experiment directory."""

    destination = spec.run_dir / "source_config.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec.config_path, destination)
    return destination
