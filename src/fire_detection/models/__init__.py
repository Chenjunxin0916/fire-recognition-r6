"""Offline YOLO26 training, feature extraction, and OOF contracts."""

from fire_detection.models.features import (
    DetectionFeatures,
    PredictionValidationError,
    RawBox,
    classifier_positive_probability,
    extract_detection_features,
    resolve_positive_class_index,
)
from fire_detection.models.offline import (
    OfflinePreparationError,
    configure_offline_environment,
    sha256_file,
    validate_offline_runtime,
)
from fire_detection.models.training import (
    FoldRecord,
    ImageCheckpointSelector,
    ImageScoreAccumulator,
    ThresholdSummary,
    TrainingConfigurationError,
    TrainingSpec,
    load_fold_records,
    load_training_spec,
    normalize_ultralytics_device,
)

__all__ = [
    "DetectionFeatures",
    "FoldRecord",
    "ImageCheckpointSelector",
    "ImageScoreAccumulator",
    "OfflinePreparationError",
    "PredictionValidationError",
    "RawBox",
    "ThresholdSummary",
    "TrainingConfigurationError",
    "TrainingSpec",
    "classifier_positive_probability",
    "configure_offline_environment",
    "extract_detection_features",
    "load_fold_records",
    "load_training_spec",
    "normalize_ultralytics_device",
    "resolve_positive_class_index",
    "sha256_file",
    "validate_offline_runtime",
]
