"""Validated deployment interfaces for the frozen YOLO26 fire ensemble."""

from fire_detection.deployment.predictor import (
    DeploymentConfigError,
    DeploymentPredictionError,
    DeploymentSpec,
    FireBox,
    FirePrediction,
    FirePredictionBatch,
    FirePredictor,
    PredictorNotReadyError,
    load_deployment_spec,
)

__all__ = [
    "DeploymentConfigError",
    "DeploymentPredictionError",
    "DeploymentSpec",
    "FireBox",
    "FirePrediction",
    "FirePredictionBatch",
    "FirePredictor",
    "PredictorNotReadyError",
    "load_deployment_spec",
]
