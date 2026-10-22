"""Atomic ONNX export and CUDA numeric alignment for the frozen YOLO26 ensemble."""

from __future__ import annotations

import ast
import gc
import hashlib
import json
import math
import random
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageOps

from fire_detection.config import (
    ConfigError,
    load_yaml_mapping,
    require_bool,
    require_float,
    require_int,
    require_mapping,
    require_string,
    resolve_project_path,
)
from fire_detection.inference.detector_config import (
    DetectorInferenceSpec,
    load_detector_inference_spec,
)
from fire_detection.inference.slicing import LoadedTile, load_image_tiles
from fire_detection.inference.test_prediction import (
    FinalSelectionContract,
    ModelArtifact,
    SelectionValidator,
    TestImage,
    TestPredictionSpec,
    load_final_selection_contract,
    load_test_prediction_spec,
    validate_test_image_paths,
)
from fire_detection.inference.tta import (
    HorizontalFlipTTASpec,
    arithmetic_mean_score,
    image_tiles,
    load_horizontal_flipped_image,
)
from fire_detection.models.features import resolve_positive_class_index
from fire_detection.models.offline import (
    configure_imported_ultralytics,
    sha256_file,
    validate_offline_runtime,
)
from fire_detection.models.training import normalize_ultralytics_device
from fire_detection.utils import atomic_write_json

EXPECTED_FOLDS = frozenset(range(5))
SHA256_LENGTH = 64


class OnnxBundleError(RuntimeError):
    """Raised when an ONNX bundle cannot satisfy its frozen export contract."""


@dataclass(frozen=True)
class OnnxExportParameters:
    """The exact Ultralytics ONNX arguments validated by the repository smoke test."""

    opset: int
    batch: int
    dynamic: bool
    simplify: bool
    nms: bool
    detector_end2end: bool

    def to_dict(self) -> dict[str, object]:
        """Return the stable public export arguments."""

        return {
            "format": "onnx",
            "opset": self.opset,
            "batch": self.batch,
            "dynamic": self.dynamic,
            "simplify": self.simplify,
            "nms": self.nms,
            "detector_end2end": self.detector_end2end,
            "precision": "fp32",
        }


@dataclass(frozen=True)
class AlignmentParameters:
    """CUDA provider and numeric-tolerance requirements for publication."""

    provider: str
    provider_device_id: int
    rtol: float
    atol: float
    score_atol: float
    threshold_profile: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe alignment settings."""

        return {
            "provider": self.provider,
            "provider_device_id": self.provider_device_id,
            "pytorch_tf32": False,
            "ort_tf32": False,
            "rtol": self.rtol,
            "atol": self.atol,
            "score_atol": self.score_atol,
            "threshold_profile": self.threshold_profile,
        }


@dataclass(frozen=True)
class OnnxBundleSpec:
    """Strict export configuration resolved against one project root."""

    project_root: Path
    config_path: Path
    config_sha256: str
    prediction_config: Path
    selection_manifest: Path
    representative_images: tuple[Path, Path]
    output_dir: Path
    manifest_filename: str
    requested_device: str
    ultralytics_device: str
    export: OnnxExportParameters
    alignment: AlignmentParameters


@dataclass(frozen=True)
class TensorContract:
    """One ONNX graph input or output descriptor."""

    name: str
    dtype: str
    shape: tuple[int | str | None, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe tensor descriptor."""

        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class GraphContract:
    """Validated single-input, single-output ONNX graph interface."""

    metadata: Mapping[str, str]
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]

    def to_dict(self) -> dict[str, object]:
        """Return graph metadata and I/O descriptors for the bundle manifest."""

        return {
            "metadata": dict(sorted(self.metadata.items())),
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
        }


@dataclass(frozen=True)
class ComponentImageAlignment:
    """PyTorch/ORT component scores for one representative image."""

    filename: str
    pytorch_score: float
    ort_score: float
    score_abs_error: float
    compared_tensor_max_abs_error: float
    coordinate_max_abs_error: float = 0.0
    pytorch_original_score: float | None = None
    ort_original_score: float | None = None
    pytorch_hflip_score: float | None = None
    ort_hflip_score: float | None = None
    pytorch_full_score: float | None = None
    ort_full_score: float | None = None
    pytorch_slice_score: float | None = None
    ort_slice_score: float | None = None
    pytorch_hflip_full_score: float | None = None
    ort_hflip_full_score: float | None = None
    pytorch_hflip_slice_score: float | None = None
    ort_hflip_slice_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return only fields meaningful for this component task."""

        payload: dict[str, object] = {
            "filename": self.filename,
            "pytorch_score": self.pytorch_score,
            "ort_score": self.ort_score,
            "score_abs_error": self.score_abs_error,
            "compared_tensor_max_abs_error": self.compared_tensor_max_abs_error,
            "coordinate_max_abs_error": self.coordinate_max_abs_error,
        }
        if self.pytorch_original_score is not None:
            payload.update(
                {
                    "pytorch_original_score": self.pytorch_original_score,
                    "ort_original_score": self.ort_original_score,
                }
            )
        if self.pytorch_hflip_score is not None:
            payload.update(
                {
                    "pytorch_hflip_score": self.pytorch_hflip_score,
                    "ort_hflip_score": self.ort_hflip_score,
                    "view_aggregation": "arithmetic_mean(original,horizontal_flip)",
                }
            )
        else:
            payload["view_aggregation"] = "original_only"
        if self.pytorch_full_score is not None:
            payload.update(
                {
                    "pytorch_full_score": self.pytorch_full_score,
                    "ort_full_score": self.ort_full_score,
                    "pytorch_slice_score": self.pytorch_slice_score,
                    "ort_slice_score": self.ort_slice_score,
                    "detector_reduction": "max(full_score, slice_score)",
                }
            )
        if self.pytorch_hflip_full_score is not None:
            payload.update(
                {
                    "pytorch_hflip_full_score": self.pytorch_hflip_full_score,
                    "ort_hflip_full_score": self.ort_hflip_full_score,
                    "pytorch_hflip_slice_score": self.pytorch_hflip_slice_score,
                    "ort_hflip_slice_score": self.ort_hflip_slice_score,
                    "hflip_detector_reduction": "max(full_score, slice_score)",
                    "coordinates_restored_to_original_view": True,
                }
            )
        return payload


@dataclass(frozen=True)
class ComponentExportResult:
    """Graph and numeric validation returned by one component export."""

    graph: GraphContract
    session_providers: tuple[str, ...]
    node_providers: tuple[str, ...]
    images: tuple[ComponentImageAlignment, ...]
    max_score_abs_error: float
    max_compared_tensor_abs_error: float
    max_coordinate_abs_error: float


@dataclass(frozen=True)
class OnnxBundlePlan:
    """All validated source inputs required before an export may write staging files."""

    spec: OnnxBundleSpec
    prediction: TestPredictionSpec
    selection: FinalSelectionContract
    detector_inference: DetectorInferenceSpec
    representative_images: tuple[TestImage, TestImage]
    threshold: float

    @property
    def tta(self) -> HorizontalFlipTTASpec | None:
        """Return the prediction-bound TTA contract, if schema v2 is selected."""

        return self.prediction.tta


class ComponentRuntime(Protocol):
    """Injectable export/alignment boundary used by the atomic publisher."""

    def export_component(
        self,
        *,
        artifact: ModelArtifact,
        destination: Path,
        images: Sequence[TestImage],
        prediction: TestPredictionSpec,
        detector_inference: DetectorInferenceSpec,
        export: OnnxExportParameters,
        alignment: AlignmentParameters,
        requested_device: str,
        ultralytics_device: str,
    ) -> ComponentExportResult: ...

    def inspect_graph(
        self,
        path: Path,
        *,
        task: str,
        imgsz: int,
        ultralytics_version: str,
    ) -> GraphContract: ...


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ConfigError(
            f"{field} keys mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _project_path(root: Path, raw: str, field: str) -> Path:
    path = Path(resolve_project_path(root, raw)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConfigError(f"{field} must remain inside the project root") from exc
    return path


def _plain_filename(value: str, field: str) -> str:
    if (
        PurePosixPath(value).name != value
        or PureWindowsPath(value).name != value
        or value in {"", ".", ".."}
    ):
        raise ConfigError(f"{field} must be a plain filename")
    return value


def _device_index(requested: str) -> int | None:
    normalized = requested.strip().lower()
    if normalized == "cpu":
        return None
    if normalized == "cuda":
        return 0
    if normalized.startswith("cuda:"):
        return int(normalized.partition(":")[2])
    if normalized.isdigit():
        return int(normalized)
    return None


def _torch_device(requested: str) -> str:
    index = _device_index(requested)
    if index is None:
        raise OnnxBundleError("ONNX publication requires a CUDA device")
    return f"cuda:{index}"


def load_onnx_bundle_spec(
    path: Path = Path("configs/export_onnx.yaml"),
    *,
    project_root: Path | None = None,
) -> OnnxBundleSpec:
    """Load a strict, static-FP32 CUDA ONNX export configuration."""

    root = (project_root or Path.cwd()).resolve()
    config_path = path.resolve() if path.is_absolute() else (root / path).resolve()
    raw = load_yaml_mapping(config_path)
    _exact_keys(
        raw,
        {
            "schema_version",
            "source",
            "representative_images",
            "output",
            "export",
            "alignment",
        },
        "onnx_bundle",
    )
    if require_int(raw, "schema_version", context="onnx_bundle", minimum=1) != 1:
        raise ConfigError("onnx_bundle.schema_version must be 1")
    source = require_mapping(raw, "source", context="onnx_bundle")
    representatives = require_mapping(raw, "representative_images", context="onnx_bundle")
    output = require_mapping(raw, "output", context="onnx_bundle")
    export_raw = require_mapping(raw, "export", context="onnx_bundle")
    alignment_raw = require_mapping(raw, "alignment", context="onnx_bundle")
    _exact_keys(source, {"prediction_config", "selection_manifest"}, "onnx_bundle.source")
    _exact_keys(representatives, {"paths"}, "onnx_bundle.representative_images")
    _exact_keys(output, {"directory", "manifest_filename"}, "onnx_bundle.output")
    _exact_keys(
        export_raw,
        {
            "format",
            "device",
            "opset",
            "batch",
            "dynamic",
            "simplify",
            "nms",
            "detector_end2end",
        },
        "onnx_bundle.export",
    )
    _exact_keys(
        alignment_raw,
        {"provider", "provider_device_id", "rtol", "atol", "score_atol", "threshold_profile"},
        "onnx_bundle.alignment",
    )

    paths = representatives.get("paths")
    if (
        not isinstance(paths, list)
        or len(paths) != 2
        or any(not isinstance(item, str) or not item.strip() for item in paths)
    ):
        raise ConfigError("onnx_bundle.representative_images.paths must contain exactly two paths")
    representative_paths: tuple[Path, Path] = (
        _project_path(root, paths[0], "onnx_bundle.representative_images.paths[0]"),
        _project_path(root, paths[1], "onnx_bundle.representative_images.paths[1]"),
    )
    if len(set(representative_paths)) != 2:
        raise ConfigError("onnx_bundle representative images must be distinct")

    if require_string(export_raw, "format", context="onnx_bundle.export") != "onnx":
        raise ConfigError("onnx_bundle.export.format must be onnx")
    requested_device = require_string(export_raw, "device", context="onnx_bundle.export")
    try:
        ultralytics_device = normalize_ultralytics_device(requested_device)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    provider = require_string(alignment_raw, "provider", context="onnx_bundle.alignment")
    provider_device_id = require_int(
        alignment_raw,
        "provider_device_id",
        context="onnx_bundle.alignment",
        minimum=0,
    )
    if provider != "CUDAExecutionProvider":
        raise ConfigError("onnx_bundle alignment requires CUDAExecutionProvider")
    if _device_index(requested_device) != provider_device_id:
        raise ConfigError("export CUDA device and ORT provider_device_id must match")

    export = OnnxExportParameters(
        opset=require_int(export_raw, "opset", context="onnx_bundle.export", minimum=1),
        batch=require_int(export_raw, "batch", context="onnx_bundle.export", minimum=1),
        dynamic=require_bool(export_raw, "dynamic", context="onnx_bundle.export"),
        simplify=require_bool(export_raw, "simplify", context="onnx_bundle.export"),
        nms=require_bool(export_raw, "nms", context="onnx_bundle.export"),
        detector_end2end=require_bool(export_raw, "detector_end2end", context="onnx_bundle.export"),
    )
    if export != OnnxExportParameters(18, 1, False, False, False, True):
        raise ConfigError(
            "ONNX export must use the validated static FP32 contract: opset=18, batch=1, "
            "dynamic=false, simplify=false, nms=false, detector_end2end=true"
        )
    alignment = AlignmentParameters(
        provider=provider,
        provider_device_id=provider_device_id,
        rtol=require_float(alignment_raw, "rtol", context="onnx_bundle.alignment", minimum=0.0),
        atol=require_float(alignment_raw, "atol", context="onnx_bundle.alignment", minimum=0.0),
        score_atol=require_float(
            alignment_raw, "score_atol", context="onnx_bundle.alignment", minimum=0.0
        ),
        threshold_profile=require_string(
            alignment_raw, "threshold_profile", context="onnx_bundle.alignment"
        ),
    )
    if alignment.threshold_profile != "stable":
        raise ConfigError("onnx_bundle alignment threshold_profile must be stable")
    manifest_filename = _plain_filename(
        require_string(output, "manifest_filename", context="onnx_bundle.output"),
        "onnx_bundle.output.manifest_filename",
    )
    if Path(manifest_filename).suffix.lower() != ".json":
        raise ConfigError("onnx_bundle.output.manifest_filename must end in .json")

    return OnnxBundleSpec(
        project_root=root,
        config_path=config_path,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        prediction_config=_project_path(
            root,
            require_string(source, "prediction_config", context="onnx_bundle.source"),
            "onnx_bundle.source.prediction_config",
        ),
        selection_manifest=_project_path(
            root,
            require_string(source, "selection_manifest", context="onnx_bundle.source"),
            "onnx_bundle.source.selection_manifest",
        ),
        representative_images=representative_paths,
        output_dir=_project_path(
            root,
            require_string(output, "directory", context="onnx_bundle.output"),
            "onnx_bundle.output.directory",
        ),
        manifest_filename=manifest_filename,
        requested_device=requested_device,
        ultralytics_device=ultralytics_device,
        export=export,
        alignment=alignment,
    )


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise OnnxBundleError(f"{name} contains non-standard JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OnnxBundleError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OnnxBundleError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OnnxBundleError(f"{name} root must be an object")
    return payload


def _threshold(path: Path, profile: str) -> float:
    payload = _read_json_object(path, "final-selection manifest")
    profiles = payload.get("threshold_profiles")
    selected = profiles.get(profile) if isinstance(profiles, Mapping) else None
    value = selected.get("threshold") if isinstance(selected, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OnnxBundleError(f"threshold_profiles.{profile}.threshold must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise OnnxBundleError(f"threshold_profiles.{profile}.threshold must be in [0, 1]")
    return result


def _validate_detector_inference(spec: DetectorInferenceSpec) -> None:
    valid = (
        spec.inference_imgsz == 640
        and spec.slicing.enabled
        and spec.slicing.size == 640
        and spec.slicing.overlap == 0.2
        and spec.slicing.stride == 512
        and spec.augment is False
    )
    if not valid:
        raise OnnxBundleError(
            "ONNX bundle requires frozen detector full 640 + slice 640/O20 without built-in TTA"
        )


def resolve_onnx_bundle_plan(
    config: Path = Path("configs/export_onnx.yaml"),
    *,
    project_root: Path | None = None,
    selection_validator: SelectionValidator | None = None,
) -> OnnxBundlePlan:
    """Validate config, final selection, ten source hashes, slicing, and exactly two images."""

    spec = load_onnx_bundle_spec(config, project_root=project_root)
    prediction = load_test_prediction_spec(
        spec.prediction_config,
        project_root=spec.project_root,
        device_override=spec.requested_device,
    )
    if prediction.schema_version == 1 and prediction.tta is not None:
        raise OnnxBundleError("schema-v1 ONNX prediction cannot carry a TTA contract")
    if prediction.schema_version == 2 and prediction.tta is None:
        raise OnnxBundleError("schema-v2 ONNX prediction must carry the frozen TTA contract")
    if prediction.selection_manifest != spec.selection_manifest:
        raise OnnxBundleError(
            "export config selection_manifest differs from the frozen prediction config"
        )
    selection = load_final_selection_contract(prediction, validator=selection_validator)
    detector_inference = load_detector_inference_spec(prediction.detector_inference_config)
    _validate_detector_inference(detector_inference)
    images = validate_test_image_paths(spec.representative_images, expected_count=2)
    profile_threshold = _threshold(spec.selection_manifest, spec.alignment.threshold_profile)
    if prediction.tta is None:
        if selection.stable_threshold is not None:
            raise OnnxBundleError(
                "schema-v1 selection unexpectedly contains a TTA stable threshold"
            )
        threshold = profile_threshold
    else:
        if selection.stable_threshold is None:
            raise OnnxBundleError("schema-v2 selection omits its frozen stable threshold")
        if selection.stable_threshold != profile_threshold:
            raise OnnxBundleError("selection stable threshold differs from its threshold profile")
        threshold = selection.stable_threshold
    return OnnxBundlePlan(
        spec=spec,
        prediction=prediction,
        selection=selection,
        detector_inference=detector_inference,
        representative_images=(images[0], images[1]),
        threshold=threshold,
    )


def dry_run_onnx_bundle(plan: OnnxBundlePlan) -> dict[str, object]:
    """Return the verified write-free export plan."""

    return {
        "status": "ready",
        "mode": "dry-run",
        "selection_manifest_sha256": plan.selection.manifest_sha256,
        "candidate": plan.selection.candidate,
        "models": len(plan.selection.detector_models) + len(plan.selection.classifier_models),
        "representative_images": [image.filename for image in plan.representative_images],
        "output_dir": str(plan.spec.output_dir),
        "provider": plan.spec.alignment.provider,
        "prediction_schema_version": plan.prediction.schema_version,
        "classifier_weight": plan.selection.classifier_weight,
        "tta_config_sha256": None if plan.tta is None else plan.tta.config_sha256,
        "threshold_profile": plan.spec.alignment.threshold_profile,
        "threshold": plan.threshold,
    }


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise OnnxBundleError(f"artifact must remain inside the project root: {path}") from exc


def _probability(value: float, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise OnnxBundleError(f"{field} must be finite and in [0, 1]")
    return result


def _shape(value: Sequence[Any]) -> tuple[int | str | None, ...]:
    result: list[int | str | None] = []
    for item in value:
        if item is None or isinstance(item, (int, str)):
            result.append(item)
        else:
            result.append(str(item))
    return tuple(result)


def _metadata_names(raw: str, field: str) -> dict[int, str]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise OnnxBundleError(f"{field} ONNX metadata names are invalid") from exc
    if not isinstance(value, dict):
        raise OnnxBundleError(f"{field} ONNX metadata names must be a mapping")
    return {int(index): str(name) for index, name in value.items()}


def inspect_onnx_graph(
    path: Path,
    *,
    task: str,
    imgsz: int,
    ultralytics_version: str,
) -> GraphContract:
    """Check one standalone ONNX graph and return its exact static interface."""

    try:
        import onnx

        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise OnnxBundleError(f"cannot validate ONNX graph {path}: {exc}") from exc
    if any(
        getattr(initializer, "data_location", 0) != 0 for initializer in graph.graph.initializer
    ):
        raise OnnxBundleError(f"ONNX graph uses external tensor data: {path}")
    metadata = {entry.key: entry.value for entry in graph.metadata_props}
    expected_task = "detect" if task == "detector" else "classify"
    if metadata.get("task") != expected_task:
        raise OnnxBundleError(
            f"{task} ONNX metadata task must be {expected_task!r}, found {metadata.get('task')!r}"
        )
    if metadata.get("version") != ultralytics_version:
        raise OnnxBundleError(
            f"{task} ONNX metadata version must be {ultralytics_version}, "
            f"found {metadata.get('version')!r}"
        )
    if metadata.get("batch") != "1":
        raise OnnxBundleError(f"{task} ONNX metadata batch must be 1")
    try:
        metadata_imgsz = ast.literal_eval(metadata.get("imgsz", ""))
    except (SyntaxError, ValueError) as exc:
        raise OnnxBundleError(f"{task} ONNX metadata imgsz is invalid") from exc
    if metadata_imgsz != [imgsz, imgsz]:
        raise OnnxBundleError(
            f"{task} ONNX metadata imgsz must be {[imgsz, imgsz]}, found {metadata_imgsz!r}"
        )
    if task == "detector":
        if metadata.get("end2end", "").lower() != "true":
            raise OnnxBundleError("detector ONNX graph must retain end2end=true")
        if _metadata_names(metadata.get("names", ""), task) != {0: "fire"}:
            raise OnnxBundleError("detector ONNX class mapping must be {0: 'fire'}")
    elif _metadata_names(metadata.get("names", ""), task) != {0: "0", 1: "1"}:
        raise OnnxBundleError("classifier ONNX class mapping must be {0: '0', 1: '1'}")

    inputs = tuple(_onnx_tensor_contract(item) for item in graph.graph.input)
    outputs = tuple(_onnx_tensor_contract(item) for item in graph.graph.output)
    if len(inputs) != 1 or inputs[0].name != "images":
        raise OnnxBundleError(f"{task} ONNX graph must expose one input named 'images'")
    if inputs[0].dtype != "tensor(float)":
        raise OnnxBundleError(f"{task} ONNX input must be tensor(float)")
    if inputs[0].shape != (1, 3, imgsz, imgsz):
        raise OnnxBundleError(
            f"{task} ONNX input shape must be {(1, 3, imgsz, imgsz)}, found {inputs[0].shape}"
        )
    if len(outputs) != 1 or outputs[0].name != "output0":
        raise OnnxBundleError(f"{task} ONNX graph must expose one output named 'output0'")
    if outputs[0].dtype != "tensor(float)":
        raise OnnxBundleError(f"{task} ONNX output must be tensor(float)")
    output_shape = outputs[0].shape
    expected_rank = 3 if task == "detector" else 2
    expected_last = 6 if task == "detector" else 2
    if (
        len(output_shape) != expected_rank
        or output_shape[0] != 1
        or output_shape[-1] != expected_last
    ):
        raise OnnxBundleError(f"unexpected {task} ONNX output shape: {output_shape}")
    selected_metadata = {
        key: metadata[key]
        for key in ("task", "version", "batch", "imgsz", "names", "head", "end2end")
        if key in metadata
    }
    return GraphContract(selected_metadata, inputs, outputs)


def _onnx_value_shape(value: Any) -> tuple[int | str | None, ...]:
    tensor_type = value.type.tensor_type
    dimensions: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(str(dimension.dim_param))
        else:
            dimensions.append(None)
    return tuple(dimensions)


def _onnx_tensor_contract(value: Any) -> TensorContract:
    try:
        import onnx
    except ImportError as exc:
        raise OnnxBundleError(f"cannot inspect ONNX tensor type: {exc}") from exc
    element_type = int(value.type.tensor_type.elem_type)
    element_name = onnx.TensorProto.DataType.Name(element_type).lower()
    dtype = "tensor(float)" if element_name == "float" else f"tensor({element_name})"
    return TensorContract(str(value.name), dtype, _onnx_value_shape(value))


class UltralyticsOnnxRuntime:
    """Validated local Ultralytics exporter plus direct ORT CUDA alignment runner."""

    def __init__(self, plan: OnnxBundlePlan) -> None:
        runtime = validate_offline_runtime(
            project_root=plan.prediction.project_root,
            manifest_path=plan.prediction.offline_manifest,
            yolo_config_dir=plan.prediction.yolo_config_dir,
            required_assets=("yolo26s.pt", "yolo26s-cls.pt", "yolo26n.pt", "Arial.ttf"),
            expected_ultralytics_version=plan.prediction.ultralytics_version,
        )
        configure_imported_ultralytics(runtime)
        try:
            import torch
            from ultralytics.engine.exporter import Exporter

            from ultralytics import YOLO
        except ImportError as exc:
            raise OnnxBundleError(f"cannot import ONNX export runtime: {exc}") from exc
        if plan.spec.ultralytics_device == "cpu" or not torch.cuda.is_available():
            raise OnnxBundleError("ONNX publication requires an available CUDA device")
        random.seed(plan.prediction.seed)
        np.random.seed(plan.prediction.seed)
        torch.manual_seed(plan.prediction.seed)
        torch.cuda.manual_seed_all(plan.prediction.seed)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self._yolo = YOLO
        self._exporter = Exporter

    def inspect_graph(
        self,
        path: Path,
        *,
        task: str,
        imgsz: int,
        ultralytics_version: str,
    ) -> GraphContract:
        """Delegate graph validation to the standalone checker."""

        return inspect_onnx_graph(
            path, task=task, imgsz=imgsz, ultralytics_version=ultralytics_version
        )

    def export_component(
        self,
        *,
        artifact: ModelArtifact,
        destination: Path,
        images: Sequence[TestImage],
        prediction: TestPredictionSpec,
        detector_inference: DetectorInferenceSpec,
        export: OnnxExportParameters,
        alignment: AlignmentParameters,
        requested_device: str,
        ultralytics_device: str,
    ) -> ComponentExportResult:
        """Export one source-bound checkpoint and compare its component scores."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        staged_weight = destination.with_suffix(".pt")
        if destination.exists() or staged_weight.exists():
            raise OnnxBundleError(f"staging destination already exists: {destination}")
        if sha256_file(artifact.weights_path) != artifact.weights_sha256:
            raise OnnxBundleError(
                f"{artifact.task} Fold {artifact.fold} source checkpoint hash drifted"
            )
        shutil.copy2(artifact.weights_path, staged_weight)
        try:
            if sha256_file(staged_weight) != artifact.weights_sha256:
                raise OnnxBundleError(
                    f"{artifact.task} Fold {artifact.fold} staging checkpoint copy drifted"
                )
            model = self._yolo(str(staged_weight), verbose=False)
            self._validate_loaded_model(model, artifact.task)
            kwargs: dict[str, object] = {
                "format": "onnx",
                "imgsz": artifact.imgsz,
                "batch": export.batch,
                "device": ultralytics_device,
                "dynamic": export.dynamic,
                "simplify": export.simplify,
                "opset": export.opset,
                "nms": export.nms,
                "verbose": False,
            }
            if artifact.task == "detector":
                kwargs.update(
                    {
                        "end2end": export.detector_end2end,
                        "max_det": detector_inference.max_det,
                    }
                )
            source_model = model.model
            model_args = getattr(source_model, "args", None)
            if (
                source_model is None
                or isinstance(source_model, str)
                or not isinstance(model_args, Mapping)
            ):
                raise OnnxBundleError(
                    f"loaded {artifact.task} checkpoint has no executable model arguments"
                )
            if "imgsz" not in model_args:
                raise OnnxBundleError(
                    f"loaded {artifact.task} checkpoint model arguments omit imgsz"
                )
            custom: dict[str, object] = {
                "imgsz": model_args["imgsz"],
                "batch": 1,
                "data": None,
                "device": None,
                "verbose": False,
            }
            exporter_args = {**model.overrides, **custom, **kwargs, "mode": "export"}
            exporter = self._exporter(overrides=exporter_args, _callbacks=model.callbacks)
            exported = Path(str(exporter(model=source_model))).resolve()
            if exported != destination.resolve() or not exported.is_file():
                raise OnnxBundleError(
                    f"Ultralytics export escaped its bound staging path: {exported}"
                )
            model.model = exporter.model
            self._validate_prepared_model(model, artifact.task, detector_inference.max_det)
            del exporter
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            destination.unlink(missing_ok=True)
            raise OnnxBundleError(
                f"cannot export {artifact.task} Fold {artifact.fold}: {exc}"
            ) from exc
        finally:
            staged_weight.unlink(missing_ok=True)

        graph = self.inspect_graph(
            destination,
            task=artifact.task,
            imgsz=artifact.imgsz,
            ultralytics_version=prediction.ultralytics_version,
        )
        return self._align_component(
            artifact=artifact,
            onnx_path=destination,
            graph=graph,
            images=images,
            prediction=prediction,
            detector_inference=detector_inference,
            alignment=alignment,
            requested_device=requested_device,
            pt_model=model,
        )

    @staticmethod
    def _validate_loaded_model(model: Any, task: str) -> None:
        names_raw = model.names
        items = names_raw.items() if isinstance(names_raw, Mapping) else enumerate(names_raw)
        names = {int(index): str(name) for index, name in items}
        if task == "detector":
            head = model.model.model[-1]
            if names != {0: "fire"} or not bool(getattr(head, "end2end", False)):
                raise OnnxBundleError("detector checkpoint is not the selected YOLO26 fire model")
        elif names != {0: "0", 1: "1"}:
            raise OnnxBundleError("classifier checkpoint class mapping must be {0: '0', 1: '1'}")

    @staticmethod
    def _validate_prepared_model(model: Any, task: str, max_det: int) -> None:
        """Verify that alignment uses the exact model state prepared by the ONNX exporter."""

        UltralyticsOnnxRuntime._validate_loaded_model(model, task)
        if not bool(model.model.is_fused()):
            raise OnnxBundleError(f"prepared {task} PyTorch reference is not fused")
        head = model.model.model[-1]
        if not bool(getattr(head, "export", False)) or getattr(head, "format", None) != "onnx":
            raise OnnxBundleError(f"prepared {task} PyTorch reference lacks ONNX export state")
        if task == "detector" and (
            bool(getattr(head, "dynamic", True))
            or int(getattr(head, "max_det", -1)) != max_det
            or bool(getattr(head, "agnostic_nms", True))
        ):
            raise OnnxBundleError("prepared detector PyTorch reference export settings drifted")

    def _align_component(
        self,
        *,
        artifact: ModelArtifact,
        onnx_path: Path,
        graph: GraphContract,
        images: Sequence[TestImage],
        prediction: TestPredictionSpec,
        detector_inference: DetectorInferenceSpec,
        alignment: AlignmentParameters,
        requested_device: str,
        pt_model: Any,
    ) -> ComponentExportResult:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OnnxBundleError(f"onnxruntime-gpu is unavailable: {exc}") from exc
        ort.preload_dlls()
        available = set(ort.get_available_providers())
        if alignment.provider not in available:
            raise OnnxBundleError(
                f"ORT provider {alignment.provider} is unavailable; found {sorted(available)}"
            )
        profile_prefix = onnx_path.parent / f".{onnx_path.stem}.ort_profile"
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=[
                (
                    alignment.provider,
                    {"device_id": alignment.provider_device_id, "use_tf32": 0},
                )
            ],
        )
        session.disable_fallback()
        session_providers = tuple(str(item) for item in session.get_providers())
        if not session_providers or session_providers[0] != alignment.provider:
            raise OnnxBundleError(
                f"ORT did not prioritize {alignment.provider}: {session_providers}"
            )
        provider_options = session.get_provider_options().get(alignment.provider, {})
        if provider_options.get("use_tf32") != "0":
            raise OnnxBundleError("ORT CUDA provider did not disable TF32")
        image_results: list[ComponentImageAlignment] = []
        comparison_errors: list[float] = []
        coordinate_errors: list[float] = []
        score_errors: list[float] = []
        profile_path: Path | None = None
        profiling_ended = False
        try:
            for image in images:
                if artifact.task == "detector":
                    result, invocation_errors = self._align_detector_image(
                        pt_model=pt_model,
                        session=session,
                        image=image,
                        inference=detector_inference,
                        tta=prediction.tta,
                        alignment=alignment,
                        requested_device=requested_device,
                    )
                else:
                    result, invocation_errors = self._align_classifier_image(
                        pt_model=pt_model,
                        session=session,
                        image=image,
                        imgsz=prediction.classifier_imgsz,
                        positive_name=prediction.positive_class_name,
                        tta=prediction.tta,
                        alignment=alignment,
                        requested_device=requested_device,
                    )
                image_results.append(result)
                comparison_errors.extend(invocation_errors)
                coordinate_errors.append(result.coordinate_max_abs_error)
                score_errors.append(result.score_abs_error)
            profile_path = Path(session.end_profiling())
            profiling_ended = True
            node_providers = _profile_node_providers(profile_path)
        except Exception:
            if not profiling_ended:
                try:
                    profile_path = Path(session.end_profiling())
                except RuntimeError:
                    profile_path = None
            raise
        finally:
            del pt_model
            _release_cuda()
            if profile_path is not None and profile_path.is_file():
                profile_path.unlink(missing_ok=True)
        if node_providers != (alignment.provider,):
            raise OnnxBundleError(
                f"{artifact.task} Fold {artifact.fold} ORT nodes used {node_providers}, "
                f"required only {alignment.provider}"
            )
        return ComponentExportResult(
            graph=graph,
            session_providers=session_providers,
            node_providers=node_providers,
            images=tuple(image_results),
            max_score_abs_error=max(score_errors, default=0.0),
            max_compared_tensor_abs_error=max(comparison_errors, default=0.0),
            max_coordinate_abs_error=max(coordinate_errors, default=0.0),
        )

    def _align_detector_image(
        self,
        *,
        pt_model: Any,
        session: Any,
        image: TestImage,
        inference: DetectorInferenceSpec,
        tta: HorizontalFlipTTASpec | None,
        alignment: AlignmentParameters,
        requested_device: str,
    ) -> tuple[ComponentImageAlignment, list[float]]:
        with Image.open(image.path) as opened:
            full = ImageOps.exif_transpose(opened).convert("RGB")
        _shape_value, original_tiles = load_image_tiles(
            image.path, size=inference.slicing.size, stride=inference.slicing.stride
        )
        try:
            original = self._align_detector_view(
                pt_model=pt_model,
                session=session,
                full=full,
                tiles=original_tiles,
                inference=inference,
                alignment=alignment,
                requested_device=requested_device,
                horizontal_flip=False,
            )
        finally:
            full.close()
            for tile in original_tiles:
                tile.image.close()
        pt_full, ort_full, pt_slice, ort_slice, errors, coordinate_errors = original
        pt_original = max(pt_full, pt_slice)
        ort_original = max(ort_full, ort_slice)
        pt_hflip: float | None = None
        ort_hflip: float | None = None
        hflip_scores: tuple[float, float, float, float] | None = None
        if tta is None:
            pt_component = pt_original
            ort_component = ort_original
        else:
            _image_shape, flipped = load_horizontal_flipped_image(image.path)
            flipped_tiles = image_tiles(
                flipped, size=inference.slicing.size, stride=inference.slicing.stride
            )
            try:
                flipped_result = self._align_detector_view(
                    pt_model=pt_model,
                    session=session,
                    full=flipped,
                    tiles=flipped_tiles,
                    inference=inference,
                    alignment=alignment,
                    requested_device=requested_device,
                    horizontal_flip=True,
                )
            finally:
                for tile in flipped_tiles:
                    tile.image.close()
                flipped.close()
            (
                pt_hflip_full,
                ort_hflip_full,
                pt_hflip_slice,
                ort_hflip_slice,
                hflip_errors,
                hflip_coordinate_errors,
            ) = flipped_result
            errors.extend(hflip_errors)
            coordinate_errors.extend(hflip_coordinate_errors)
            pt_hflip = max(pt_hflip_full, pt_hflip_slice)
            ort_hflip = max(ort_hflip_full, ort_hflip_slice)
            pt_component = arithmetic_mean_score(pt_original, pt_hflip)
            ort_component = arithmetic_mean_score(ort_original, ort_hflip)
            hflip_scores = (
                pt_hflip_full,
                ort_hflip_full,
                pt_hflip_slice,
                ort_hflip_slice,
            )
        score_error = abs(pt_component - ort_component)
        if score_error > alignment.score_atol:
            raise OnnxBundleError(
                f"detector score mismatch for {image.filename}: {score_error} > "
                f"{alignment.score_atol}"
            )
        return (
            ComponentImageAlignment(
                filename=image.filename,
                pytorch_score=pt_component,
                ort_score=ort_component,
                score_abs_error=score_error,
                compared_tensor_max_abs_error=max(errors),
                coordinate_max_abs_error=max(coordinate_errors, default=0.0),
                pytorch_original_score=pt_original,
                ort_original_score=ort_original,
                pytorch_hflip_score=pt_hflip,
                ort_hflip_score=ort_hflip,
                pytorch_full_score=pt_full,
                ort_full_score=ort_full,
                pytorch_slice_score=pt_slice,
                ort_slice_score=ort_slice,
                pytorch_hflip_full_score=None if hflip_scores is None else hflip_scores[0],
                ort_hflip_full_score=None if hflip_scores is None else hflip_scores[1],
                pytorch_hflip_slice_score=None if hflip_scores is None else hflip_scores[2],
                ort_hflip_slice_score=None if hflip_scores is None else hflip_scores[3],
            ),
            errors,
        )

    @staticmethod
    def _align_detector_view(
        *,
        pt_model: Any,
        session: Any,
        full: Image.Image,
        tiles: Sequence[LoadedTile],
        inference: DetectorInferenceSpec,
        alignment: AlignmentParameters,
        requested_device: str,
        horizontal_flip: bool,
    ) -> tuple[float, float, float, float, list[float], list[float]]:
        full_input = _detector_input(full, inference.inference_imgsz)
        pt_full, ort_full, full_error, full_coordinate_error = _compare_detector_input(
            pt_model,
            session,
            full_input,
            confidence=inference.conf,
            alignment=alignment,
            requested_device=requested_device,
            horizontal_flip=horizontal_flip,
        )
        pt_slice = 0.0
        ort_slice = 0.0
        errors = [full_error]
        coordinate_errors = [full_coordinate_error]
        for tile in tiles:
            tile_input = _detector_input(tile.image, inference.slicing.size)
            pt_score, ort_score, tensor_error, coordinate_error = _compare_detector_input(
                pt_model,
                session,
                tile_input,
                confidence=inference.conf,
                alignment=alignment,
                requested_device=requested_device,
                horizontal_flip=horizontal_flip,
            )
            pt_slice = max(pt_slice, pt_score)
            ort_slice = max(ort_slice, ort_score)
            errors.append(tensor_error)
            coordinate_errors.append(coordinate_error)
        return pt_full, ort_full, pt_slice, ort_slice, errors, coordinate_errors

    @staticmethod
    def _align_classifier_image(
        *,
        pt_model: Any,
        session: Any,
        image: TestImage,
        imgsz: int,
        positive_name: str,
        tta: HorizontalFlipTTASpec | None,
        alignment: AlignmentParameters,
        requested_device: str,
    ) -> tuple[ComponentImageAlignment, list[float]]:
        names_raw = pt_model.names
        names = names_raw if isinstance(names_raw, Mapping) else list(names_raw)
        positive_index = resolve_positive_class_index(names, positive_name)
        model_input = _classifier_input(image.path, imgsz)
        pt_original, ort_original, original_error = _compare_classifier_input(
            pt_model,
            session,
            model_input,
            positive_index=positive_index,
            alignment=alignment,
            requested_device=requested_device,
        )
        errors = [original_error]
        pt_hflip: float | None = None
        ort_hflip: float | None = None
        if tta is None:
            pt_score = pt_original
            ort_score = ort_original
        else:
            _image_shape, flipped = load_horizontal_flipped_image(image.path)
            try:
                flipped_input = _classifier_image_input(flipped, imgsz)
            finally:
                flipped.close()
            pt_hflip, ort_hflip, flipped_error = _compare_classifier_input(
                pt_model,
                session,
                flipped_input,
                positive_index=positive_index,
                alignment=alignment,
                requested_device=requested_device,
            )
            errors.append(flipped_error)
            pt_score = arithmetic_mean_score(pt_original, pt_hflip)
            ort_score = arithmetic_mean_score(ort_original, ort_hflip)
        score_error = abs(pt_score - ort_score)
        if score_error > alignment.score_atol:
            raise OnnxBundleError(
                f"classifier score mismatch for {image.filename}: {score_error} > "
                f"{alignment.score_atol}"
            )
        return (
            ComponentImageAlignment(
                filename=image.filename,
                pytorch_score=pt_score,
                ort_score=ort_score,
                score_abs_error=score_error,
                compared_tensor_max_abs_error=max(errors),
                pytorch_original_score=pt_original,
                ort_original_score=ort_original,
                pytorch_hflip_score=pt_hflip,
                ort_hflip_score=ort_hflip,
            ),
            errors,
        )


def _profile_node_providers(path: Path) -> tuple[str, ...]:
    payload = _read_json_value(path, "ORT profile")
    if not isinstance(payload, list):
        raise OnnxBundleError("ORT profile root must be an array")
    providers = {
        str(provider)
        for event in payload
        if isinstance(event, Mapping)
        for args in (event.get("args"),)
        if isinstance(args, Mapping)
        for provider in (args.get("provider"),)
        if provider
    }
    if not providers:
        raise OnnxBundleError("ORT profile did not record any node provider")
    return tuple(sorted(providers))


def _read_json_value(path: Path, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OnnxBundleError(f"cannot read {name} {path}: {exc}") from exc


def _detector_input(image: Image.Image, imgsz: int) -> np.ndarray:
    from ultralytics.data.augment import LetterBox

    rgb = np.asarray(image.convert("RGB"))
    letterboxed = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=rgb)
    array = np.asarray(letterboxed, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    result: np.ndarray = np.ascontiguousarray(array)
    return result


def _classifier_input(path: Path, imgsz: int) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        try:
            return _classifier_image_input(image, imgsz)
        finally:
            image.close()


def _classifier_image_input(image: Image.Image, imgsz: int) -> np.ndarray:
    """Apply the exported classifier's preprocessing to an already decoded view."""

    from ultralytics.data.augment import classify_transforms

    tensor = classify_transforms(imgsz)(image).unsqueeze(0)
    result: np.ndarray = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    return result


def _torch_output(model: Any, model_input: np.ndarray, *, requested_device: str) -> np.ndarray:
    import torch

    device = _torch_device(requested_device)
    module = model.model.to(device).eval()
    with torch.inference_mode():
        value = module(torch.from_numpy(model_input).to(device))
    output = value[0] if isinstance(value, tuple) else value
    if not isinstance(output, torch.Tensor):
        raise OnnxBundleError(f"PyTorch model returned unsupported output type {type(output)!r}")
    result: np.ndarray = np.asarray(output.detach().float().cpu().numpy(), dtype=np.float32)
    return result


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def _active_detector_rows(output: np.ndarray, confidence: float, field: str) -> np.ndarray:
    """Validate end-to-end postprocessed rows and return candidates above frozen confidence."""

    if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 6:
        raise OnnxBundleError(f"unexpected {field} detector output shape: {output.shape}")
    if not np.isfinite(output).all():
        raise OnnxBundleError(f"{field} detector output contains non-finite values")
    confidences = output[0, :, 4]
    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise OnnxBundleError(f"{field} detector confidences are outside [0, 1]")
    class_ids = output[0, :, 5]
    if not np.allclose(class_ids, np.rint(class_ids), rtol=0.0, atol=1e-6):
        raise OnnxBundleError(f"{field} detector class IDs are not integral")
    if np.any(np.rint(class_ids).astype(np.int64) != 0):
        raise OnnxBundleError(f"{field} detector candidate has a non-fire class")
    if np.any(output[0, :, 2] < output[0, :, 0]) or np.any(output[0, :, 3] < output[0, :, 1]):
        raise OnnxBundleError(f"{field} detector candidate has reversed xyxy coordinates")
    rows: np.ndarray = np.asarray(output[0, confidences >= confidence], dtype=np.float32)
    if rows.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    return rows


def _detector_candidate_cost(
    pytorch: np.ndarray,
    ort: np.ndarray,
    alignment: AlignmentParameters,
) -> float | None:
    """Return a normalized semantic-match cost, or ``None`` when rows cannot match."""

    if int(round(float(pytorch[5]))) != int(round(float(ort[5]))):
        return None
    pt_values = np.asarray(pytorch[:5], dtype=np.float32)
    ort_values = np.asarray(ort[:5], dtype=np.float32)
    if not np.allclose(pt_values, ort_values, rtol=alignment.rtol, atol=alignment.atol):
        return None
    scale = alignment.atol + alignment.rtol * np.maximum(np.abs(pt_values), np.abs(ort_values))
    safe_scale = np.maximum(scale, np.finfo(np.float32).eps)
    return float(np.max(np.abs(pt_values - ort_values) / safe_scale, initial=0.0))


def _match_detector_candidates(
    pytorch: np.ndarray,
    ort: np.ndarray,
    alignment: AlignmentParameters,
) -> tuple[float, float]:
    """Match active postprocessed rows by semantics independent of row order."""

    if len(pytorch) != len(ort):
        raise OnnxBundleError(
            "detector active candidate count differs at the frozen confidence threshold: "
            f"PyTorch={len(pytorch)}, ORT={len(ort)}"
        )
    if len(pytorch) == 0:
        return 0.0, 0.0
    adjacency: list[list[int]] = []
    for pt_row in pytorch:
        eligible: list[tuple[float, int]] = []
        for ort_index, ort_row in enumerate(ort):
            cost = _detector_candidate_cost(pt_row, ort_row, alignment)
            if cost is not None:
                eligible.append((cost, ort_index))
        adjacency.append([index for _cost, index in sorted(eligible)])

    ort_to_pt = [-1] * len(ort)

    def assign(pt_index: int, seen_ort: set[int]) -> bool:
        for ort_index in adjacency[pt_index]:
            if ort_index in seen_ort:
                continue
            seen_ort.add(ort_index)
            previous = ort_to_pt[ort_index]
            if previous == -1 or assign(previous, seen_ort):
                ort_to_pt[ort_index] = pt_index
                return True
        return False

    pt_order = sorted(range(len(pytorch)), key=lambda index: (-float(pytorch[index, 4]), index))
    for pt_index in pt_order:
        if not assign(pt_index, set()):
            row = pytorch[pt_index]
            raise OnnxBundleError(
                "detector active PyTorch candidate has no one-to-one ORT semantic match: "
                f"confidence={float(row[4])}, class_id={int(round(float(row[5])))}, "
                f"xyxy={[float(value) for value in row[:4]]}"
            )

    pt_to_ort = [-1] * len(pytorch)
    for ort_index, pt_index in enumerate(ort_to_pt):
        if pt_index >= 0:
            pt_to_ort[pt_index] = ort_index
    if any(index < 0 for index in pt_to_ort):
        raise OnnxBundleError("detector active candidate matching is incomplete")
    confidence_error = max(
        abs(float(pytorch[index, 4]) - float(ort[ort_index, 4]))
        for index, ort_index in enumerate(pt_to_ort)
    )
    coordinate_error = max(
        float(np.max(np.abs(pytorch[index, :4] - ort[ort_index, :4]), initial=0.0))
        for index, ort_index in enumerate(pt_to_ort)
    )
    return confidence_error, coordinate_error


def _compare_detector_input(
    pt_model: Any,
    session: Any,
    model_input: np.ndarray,
    *,
    confidence: float,
    alignment: AlignmentParameters,
    requested_device: str,
    horizontal_flip: bool,
) -> tuple[float, float, float, float]:
    pt_output = _torch_output(pt_model, model_input, requested_device=requested_device)
    ort_output: np.ndarray = np.asarray(
        session.run(["output0"], {"images": model_input})[0], dtype=np.float32
    )
    if pt_output.shape != ort_output.shape:
        raise OnnxBundleError(
            f"detector PyTorch/ORT shapes differ: {pt_output.shape} vs {ort_output.shape}"
        )
    if pt_output.ndim != 3 or pt_output.shape[0] != 1 or pt_output.shape[2] != 6:
        raise OnnxBundleError(f"unexpected detector output shape: {pt_output.shape}")
    pt_rows = _active_detector_rows(pt_output, confidence, "PyTorch")
    ort_rows = _active_detector_rows(ort_output, confidence, "ORT")
    if horizontal_flip:
        input_width = int(model_input.shape[-1])
        pt_rows = pt_rows.copy()
        ort_rows = ort_rows.copy()
        pt_rows[:, :4] = _restore_horizontal_flip_coordinate_tensor(
            pt_rows[:, :4], width=input_width, field="PyTorch active detector coordinates"
        )
        ort_rows[:, :4] = _restore_horizontal_flip_coordinate_tensor(
            ort_rows[:, :4], width=input_width, field="ORT active detector coordinates"
        )
    confidence_error, coordinate_error = _match_detector_candidates(pt_rows, ort_rows, alignment)
    pt_score = (
        0.0
        if len(pt_rows) == 0
        else _probability(float(pt_rows[:, 4].max()), "PyTorch detector score")
    )
    ort_score = (
        0.0
        if len(ort_rows) == 0
        else _probability(float(ort_rows[:, 4].max()), "ORT detector score")
    )
    if abs(pt_score - ort_score) > alignment.score_atol:
        raise OnnxBundleError(
            f"detector invocation score mismatch: {abs(pt_score - ort_score)} > "
            f"{alignment.score_atol}"
        )
    return pt_score, ort_score, max(confidence_error, coordinate_error), coordinate_error


def _restore_horizontal_flip_coordinate_tensor(
    coordinates: np.ndarray, *, width: int, field: str
) -> np.ndarray:
    """Restore an ``xyxy`` tensor and verify that restoration is an involution."""

    if coordinates.ndim < 1 or coordinates.shape[-1] != 4:
        raise OnnxBundleError(f"{field} must end with four xyxy coordinates")
    if type(width) is not int or width <= 0:
        raise OnnxBundleError("horizontal-flip restoration width must be positive")
    if not np.isfinite(coordinates).all():
        raise OnnxBundleError(f"{field} contains non-finite values")
    restored: np.ndarray = np.empty_like(coordinates)
    restored[..., 0] = float(width) - coordinates[..., 2]
    restored[..., 1] = coordinates[..., 1]
    restored[..., 2] = float(width) - coordinates[..., 0]
    restored[..., 3] = coordinates[..., 3]
    round_trip: np.ndarray = np.empty_like(restored)
    round_trip[..., 0] = float(width) - restored[..., 2]
    round_trip[..., 1] = restored[..., 1]
    round_trip[..., 2] = float(width) - restored[..., 0]
    round_trip[..., 3] = restored[..., 3]
    round_trip_atol = max(1e-6, float(np.finfo(np.float32).eps) * width * 2.0)
    if not np.allclose(round_trip, coordinates, rtol=0.0, atol=round_trip_atol):
        raise OnnxBundleError(f"{field} failed single-restore/double-restore validation")
    return restored


def _classifier_probability(output: np.ndarray, positive_index: int, field: str) -> float:
    if output.ndim != 2 or output.shape[0] != 1:
        raise OnnxBundleError(f"unexpected {field} classifier output shape: {output.shape}")
    if not 0 <= positive_index < output.shape[1]:
        raise OnnxBundleError(f"{field} positive class index is outside the output tensor")
    if not np.isfinite(output).all():
        raise OnnxBundleError(f"{field} classifier output contains non-finite values")
    if np.any(output < 0.0) or np.any(output > 1.0):
        raise OnnxBundleError(f"{field} classifier output is not probabilistic")
    if not np.isclose(float(output[0].sum()), 1.0, rtol=1e-5, atol=1e-6):
        raise OnnxBundleError(f"{field} classifier probabilities do not sum to one")
    return _probability(float(output[0, positive_index]), f"{field} classifier score")


def _compare_classifier_input(
    pt_model: Any,
    session: Any,
    model_input: np.ndarray,
    *,
    positive_index: int,
    alignment: AlignmentParameters,
    requested_device: str,
) -> tuple[float, float, float]:
    pt_output = _torch_output(pt_model, model_input, requested_device=requested_device)
    ort_output: np.ndarray = np.asarray(
        session.run(["output0"], {"images": model_input})[0], dtype=np.float32
    )
    tensor_error = _compare_arrays(
        pt_output, ort_output, alignment, "classifier probability tensor"
    )
    pt_score = _classifier_probability(pt_output, positive_index, "PyTorch")
    ort_score = _classifier_probability(ort_output, positive_index, "ORT")
    score_error = abs(pt_score - ort_score)
    if score_error > alignment.score_atol:
        raise OnnxBundleError(
            f"classifier invocation score mismatch: {score_error} > {alignment.score_atol}"
        )
    return pt_score, ort_score, tensor_error


def _compare_arrays(
    pytorch: np.ndarray,
    ort: np.ndarray,
    alignment: AlignmentParameters,
    field: str,
) -> float:
    if pytorch.shape != ort.shape:
        raise OnnxBundleError(f"{field} shapes differ: {pytorch.shape} vs {ort.shape}")
    if not np.isfinite(pytorch).all() or not np.isfinite(ort).all():
        raise OnnxBundleError(f"{field} contains non-finite values")
    if not np.allclose(pytorch, ort, rtol=alignment.rtol, atol=alignment.atol):
        difference = np.abs(pytorch - ort)
        raise OnnxBundleError(
            f"{field} exceeds rtol={alignment.rtol}, atol={alignment.atol}; "
            f"max_abs_error={float(difference.max(initial=0.0))}"
        )
    return float(np.abs(pytorch - ort).max(initial=0.0))


def _artifact_record(
    artifact: ModelArtifact,
    result: ComponentExportResult,
    onnx_path: Path,
    *,
    plan: OnnxBundlePlan,
) -> dict[str, object]:
    names = {item.filename for item in result.images}
    expected_names = {item.filename for item in plan.representative_images}
    if names != expected_names or len(result.images) != 2:
        raise OnnxBundleError(
            f"{artifact.task} Fold {artifact.fold} alignment image set is incomplete"
        )
    if result.node_providers != (plan.spec.alignment.provider,):
        raise OnnxBundleError(
            f"{artifact.task} Fold {artifact.fold} did not execute exclusively on CUDA"
        )
    if result.max_score_abs_error > plan.spec.alignment.score_atol:
        raise OnnxBundleError(f"{artifact.task} Fold {artifact.fold} score error exceeds tolerance")
    view_aggregation = (
        "original_only" if plan.tta is None else "arithmetic_mean(original,horizontal_flip)"
    )
    return {
        "task": artifact.task,
        "fold": artifact.fold,
        "model_name": artifact.model_name,
        "imgsz": artifact.imgsz,
        "seed": artifact.seed,
        "source_weights": {
            "path": _portable_path(artifact.weights_path, plan.spec.project_root),
            "sha256": artifact.weights_sha256,
        },
        "source_run_manifest": {
            "path": _portable_path(artifact.run_manifest_path, plan.spec.project_root),
            "sha256": artifact.run_manifest_sha256,
        },
        "onnx": {
            "path": onnx_path.relative_to(onnx_path.parents[1]).as_posix(),
            "sha256": sha256_file(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
        },
        "graph": result.graph.to_dict(),
        "provider": {
            "requested": plan.spec.alignment.provider,
            "device_id": plan.spec.alignment.provider_device_id,
            "use_tf32": False,
            "session_providers": list(result.session_providers),
            "node_providers": list(result.node_providers),
        },
        "alignment": {
            "status": "passed",
            "view_aggregation": view_aggregation,
            "max_score_abs_error": result.max_score_abs_error,
            "max_compared_tensor_abs_error": result.max_compared_tensor_abs_error,
            "max_coordinate_abs_error": result.max_coordinate_abs_error,
            "images": [item.to_dict() for item in result.images],
        },
    }


def _score_by_filename(result: ComponentExportResult) -> dict[str, ComponentImageAlignment]:
    return {item.filename: item for item in result.images}


def _fusion_alignment(
    results: Mapping[tuple[str, int], ComponentExportResult], plan: OnnxBundlePlan
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    maximum_error = 0.0
    for image in plan.representative_images:
        detector = [
            _score_by_filename(results[("detector", fold)])[image.filename]
            for fold in sorted(EXPECTED_FOLDS)
        ]
        classifier = [
            _score_by_filename(results[("classifier", fold)])[image.filename]
            for fold in sorted(EXPECTED_FOLDS)
        ]
        pt_detector = math.fsum(item.pytorch_score for item in detector) / 5.0
        ort_detector = math.fsum(item.ort_score for item in detector) / 5.0
        pt_classifier = math.fsum(item.pytorch_score for item in classifier) / 5.0
        ort_classifier = math.fsum(item.ort_score for item in classifier) / 5.0
        classifier_weight = plan.selection.classifier_weight
        detector_weight = 1.0 - classifier_weight
        pt_fused = detector_weight * pt_detector + classifier_weight * pt_classifier
        ort_fused = detector_weight * ort_detector + classifier_weight * ort_classifier
        error = abs(pt_fused - ort_fused)
        maximum_error = max(maximum_error, error)
        pt_label = int(pt_fused >= plan.threshold)
        ort_label = int(ort_fused >= plan.threshold)
        if error > plan.spec.alignment.score_atol:
            raise OnnxBundleError(
                f"final fused score mismatch for {image.filename}: {error} > "
                f"{plan.spec.alignment.score_atol}"
            )
        if pt_label != ort_label:
            raise OnnxBundleError(
                f"final fused label mismatch for {image.filename}: {pt_label} vs {ort_label}"
            )
        rows.append(
            {
                "filename": image.filename,
                "pytorch_detector_mean": pt_detector,
                "ort_detector_mean": ort_detector,
                "pytorch_classifier_mean": pt_classifier,
                "ort_classifier_mean": ort_classifier,
                "pytorch_fused_score": pt_fused,
                "ort_fused_score": ort_fused,
                "fused_score_abs_error": error,
                "pytorch_label": pt_label,
                "ort_label": ort_label,
            }
        )
    if plan.tta is None:
        detector_reduction = "mean(max(full_score, slice_score))"
        classifier_reduction = "mean(positive_class_probability)"
        view_aggregation = "original_only"
    else:
        detector_reduction = (
            "mean(arithmetic_mean(max(original_full,original_slice),"
            "max(horizontal_flip_full,horizontal_flip_slice)))"
        )
        classifier_reduction = (
            "mean(arithmetic_mean(original_probability,horizontal_flip_probability))"
        )
        view_aggregation = "arithmetic_mean(original,horizontal_flip)"
    return {
        "status": "passed",
        "method": "weighted_average",
        "detector_weight": 1.0 - plan.selection.classifier_weight,
        "classifier_weight": plan.selection.classifier_weight,
        "view_aggregation": view_aggregation,
        "detector_fold_reduction": detector_reduction,
        "classifier_fold_reduction": classifier_reduction,
        "threshold_profile": plan.spec.alignment.threshold_profile,
        "threshold": plan.threshold,
        "max_fused_score_abs_error": maximum_error,
        "labels_match": True,
        "images": rows,
    }


def _representative_records(plan: OnnxBundlePlan) -> list[dict[str, object]]:
    return [
        {
            "path": _portable_path(image.path, plan.spec.project_root),
            "filename": image.filename,
            "sha256": image.sha256,
            "width": image.width,
            "height": image.height,
        }
        for image in plan.representative_images
    ]


def _tta_manifest_record(plan: OnnxBundlePlan) -> dict[str, object] | None:
    if plan.tta is None:
        return None
    return {
        **plan.tta.manifest_record(),
        "config": {
            "path": _portable_path(plan.tta.config_path, plan.spec.project_root),
            "sha256": plan.tta.config_sha256,
        },
    }


def _preprocessing_record(plan: OnnxBundlePlan) -> dict[str, str]:
    if plan.tta is None:
        return {
            "detector": "Ultralytics LetterBox static square, RGB BCHW float32 / 255",
            "classifier": "Ultralytics classify_transforms resize/center-crop/normalize",
        }
    return {
        "detector": (
            "original and explicit horizontal-flip views; each uses Ultralytics LetterBox "
            "static square, RGB BCHW float32 / 255"
        ),
        "classifier": (
            "original and explicit horizontal-flip views; each uses Ultralytics "
            "classify_transforms resize/center-crop/normalize"
        ),
    }


def _build_manifest(
    plan: OnnxBundlePlan,
    records: Mapping[str, Sequence[dict[str, object]]],
    results: Mapping[tuple[str, int], ComponentExportResult],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": plan.prediction.schema_version,
        "kind": "onnx_bundle",
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "export_config": {
                "path": _portable_path(plan.spec.config_path, plan.spec.project_root),
                "sha256": plan.spec.config_sha256,
            },
            "prediction_config": {
                "path": _portable_path(plan.prediction.config_path, plan.spec.project_root),
                "sha256": sha256_file(plan.prediction.config_path),
            },
            "selection_manifest": {
                "path": _portable_path(plan.spec.selection_manifest, plan.spec.project_root),
                "sha256": plan.selection.manifest_sha256,
            },
            "candidate": plan.selection.candidate,
            "detector_inference_config": {
                "path": _portable_path(plan.detector_inference.config_path, plan.spec.project_root),
                "sha256": plan.detector_inference.config_sha256,
            },
        },
        "export": {
            "ultralytics_version": plan.prediction.ultralytics_version,
            "device": plan.spec.requested_device,
            "parameters": plan.spec.export.to_dict(),
            "output_directory": _portable_path(plan.spec.output_dir, plan.spec.project_root),
        },
        "alignment_contract": plan.spec.alignment.to_dict(),
        "preprocessing": _preprocessing_record(plan),
        "representative_images": _representative_records(plan),
        "components": {
            "detector": list(records["detector"]),
            "classifier": list(records["classifier"]),
        },
        "fusion_alignment": _fusion_alignment(results, plan),
    }
    tta_record = _tta_manifest_record(plan)
    if tta_record is not None:
        payload["tta"] = tta_record
    return payload


def export_onnx_bundle(
    plan: OnnxBundlePlan,
    *,
    runtime: ComponentRuntime | None = None,
) -> dict[str, object]:
    """Export, align, verify, and atomically publish ten ONNX components."""

    output = plan.spec.output_dir
    if output.exists():
        raise OnnxBundleError(
            f"ONNX bundle output already exists and will not be overwritten: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    selected_runtime = runtime or UltralyticsOnnxRuntime(plan)
    records: dict[str, list[dict[str, object]]] = {"detector": [], "classifier": []}
    results: dict[tuple[str, int], ComponentExportResult] = {}
    try:
        (staging / "models").mkdir(parents=True)
        groups = (plan.selection.detector_models, plan.selection.classifier_models)
        for artifacts in groups:
            for artifact in artifacts:
                destination = staging / "models" / f"{artifact.task}_fold{artifact.fold}.onnx"
                result = selected_runtime.export_component(
                    artifact=artifact,
                    destination=destination,
                    images=plan.representative_images,
                    prediction=plan.prediction,
                    detector_inference=plan.detector_inference,
                    export=plan.spec.export,
                    alignment=plan.spec.alignment,
                    requested_device=plan.spec.requested_device,
                    ultralytics_device=plan.spec.ultralytics_device,
                )
                if not destination.is_file() or destination.stat().st_size <= 0:
                    raise OnnxBundleError(
                        f"{artifact.task} Fold {artifact.fold} did not produce an ONNX file"
                    )
                records[artifact.task].append(
                    _artifact_record(artifact, result, destination, plan=plan)
                )
                results[(artifact.task, artifact.fold)] = result
        if set(results) != {
            (task, fold) for task in ("detector", "classifier") for fold in EXPECTED_FOLDS
        }:
            raise OnnxBundleError("ONNX export did not produce exactly ten Fold components")
        payload = _build_manifest(plan, records, results)
        atomic_write_json(staging / plan.spec.manifest_filename, payload)
        _check_bundle_directory(
            plan,
            staging,
            graph_inspector=selected_runtime.inspect_graph,
        )
        staging.replace(output)
        return payload
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


GraphInspector = Callable[..., GraphContract]


def _relative_bundle_file(bundle: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise OnnxBundleError(f"{field} must be a non-empty relative path")
    raw = value.strip()
    portable = PurePosixPath(raw.replace("\\", "/"))
    if portable.is_absolute() or PureWindowsPath(raw).is_absolute() or ".." in portable.parts:
        raise OnnxBundleError(f"{field} must remain inside the ONNX bundle")
    path = (bundle / Path(*portable.parts)).resolve()
    try:
        path.relative_to(bundle.resolve())
    except ValueError as exc:
        raise OnnxBundleError(f"{field} escapes the ONNX bundle") from exc
    return path


def _component_rows(payload: Mapping[str, Any], task: str) -> list[Mapping[str, Any]]:
    components = payload.get("components")
    rows = components.get(task) if isinstance(components, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 5:
        raise OnnxBundleError(f"components.{task} must contain exactly five records")
    typed: list[Mapping[str, Any]] = []
    for item in rows:
        if not isinstance(item, Mapping):
            raise OnnxBundleError(f"components.{task} must contain exactly five records")
        typed.append(item)
    folds = {row.get("fold") for row in typed}
    if folds != EXPECTED_FOLDS:
        raise OnnxBundleError(f"components.{task} must contain Fold 0..4")
    return typed


def _check_bound_record(
    value: object,
    *,
    expected_path: Path,
    expected_sha256: str,
    root: Path,
    field: str,
) -> None:
    if not isinstance(value, Mapping):
        raise OnnxBundleError(f"{field} must be an object")
    if (
        value.get("path") != _portable_path(expected_path, root)
        or value.get("sha256") != expected_sha256
    ):
        raise OnnxBundleError(f"{field} binding changed")


def _finite_manifest_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OnnxBundleError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OnnxBundleError(f"{field} must be finite")
    return result


def _manifest_probability(value: object, field: str) -> float:
    result = _finite_manifest_number(value, field)
    if not 0.0 <= result <= 1.0:
        raise OnnxBundleError(f"{field} must be in [0, 1]")
    return result


def _require_close(actual: float, expected: float, field: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
        raise OnnxBundleError(f"{field} is inconsistent: {actual} != {expected}")


def _check_detector_view_scores(
    image_row: Mapping[str, Any],
    *,
    prefix: str,
    combined_key: str,
    field: str,
) -> None:
    full = _manifest_probability(image_row.get(f"pytorch_{prefix}full_score"), f"{field} PT full")
    sliced = _manifest_probability(
        image_row.get(f"pytorch_{prefix}slice_score"), f"{field} PT slice"
    )
    ort_full = _manifest_probability(image_row.get(f"ort_{prefix}full_score"), f"{field} ORT full")
    ort_sliced = _manifest_probability(
        image_row.get(f"ort_{prefix}slice_score"), f"{field} ORT slice"
    )
    _require_close(
        _manifest_probability(image_row.get(f"pytorch_{combined_key}"), f"{field} PT reduced"),
        max(full, sliced),
        f"{field} PT max(full,slice)",
    )
    _require_close(
        _manifest_probability(image_row.get(f"ort_{combined_key}"), f"{field} ORT reduced"),
        max(ort_full, ort_sliced),
        f"{field} ORT max(full,slice)",
    )


def _check_component_image_alignment(
    image_row: Mapping[str, Any],
    *,
    plan: OnnxBundlePlan,
    task: str,
    fold: int,
) -> tuple[float, float, float]:
    field = f"{task} Fold {fold} image"
    pt_score = _manifest_probability(image_row.get("pytorch_score"), f"{field} pytorch_score")
    ort_score = _manifest_probability(image_row.get("ort_score"), f"{field} ort_score")
    score_error = _finite_manifest_number(image_row.get("score_abs_error"), f"{field} score error")
    tensor_error = _finite_manifest_number(
        image_row.get("compared_tensor_max_abs_error"), f"{field} tensor error"
    )
    coordinate_error = _finite_manifest_number(
        image_row.get("coordinate_max_abs_error"), f"{field} coordinate error"
    )
    if min(score_error, tensor_error, coordinate_error) < 0.0:
        raise OnnxBundleError(f"{field} alignment errors must be non-negative")
    _require_close(score_error, abs(pt_score - ort_score), f"{field} score_abs_error")
    if score_error > plan.spec.alignment.score_atol:
        raise OnnxBundleError(f"{field} score error exceeds tolerance")

    if plan.tta is None:
        if image_row.get("view_aggregation") != "original_only":
            raise OnnxBundleError(f"{field} must use only the original view")
        if any("hflip" in key for key in image_row):
            raise OnnxBundleError(f"{field} schema-v1 alignment unexpectedly contains TTA scores")
        if "pytorch_original_score" in image_row or "ort_original_score" in image_row:
            pt_original = _manifest_probability(
                image_row.get("pytorch_original_score"), f"{field} PT original"
            )
            ort_original = _manifest_probability(
                image_row.get("ort_original_score"), f"{field} ORT original"
            )
            _require_close(pt_score, pt_original, f"{field} PT original reduction")
            _require_close(ort_score, ort_original, f"{field} ORT original reduction")
        if task == "detector":
            if image_row.get("detector_reduction") != "max(full_score, slice_score)":
                raise OnnxBundleError(f"{field} original detector reduction changed")
            _check_detector_view_scores(
                image_row,
                prefix="",
                combined_key="score",
                field=f"{field} original view",
            )
        return score_error, tensor_error, coordinate_error

    if image_row.get("view_aggregation") != "arithmetic_mean(original,horizontal_flip)":
        raise OnnxBundleError(f"{field} TTA aggregation must be the arithmetic mean")
    pt_original = _manifest_probability(
        image_row.get("pytorch_original_score"), f"{field} PT original"
    )
    ort_original = _manifest_probability(
        image_row.get("ort_original_score"), f"{field} ORT original"
    )
    pt_hflip = _manifest_probability(
        image_row.get("pytorch_hflip_score"), f"{field} PT horizontal flip"
    )
    ort_hflip = _manifest_probability(
        image_row.get("ort_hflip_score"), f"{field} ORT horizontal flip"
    )
    _require_close(
        pt_score,
        arithmetic_mean_score(pt_original, pt_hflip),
        f"{field} PT arithmetic mean",
    )
    _require_close(
        ort_score,
        arithmetic_mean_score(ort_original, ort_hflip),
        f"{field} ORT arithmetic mean",
    )
    if task == "detector":
        if image_row.get("coordinates_restored_to_original_view") is not True:
            raise OnnxBundleError(f"{field} horizontal-flip coordinates were not restored")
        if image_row.get("detector_reduction") != "max(full_score, slice_score)":
            raise OnnxBundleError(f"{field} original detector reduction changed")
        if image_row.get("hflip_detector_reduction") != "max(full_score, slice_score)":
            raise OnnxBundleError(f"{field} horizontal-flip detector reduction changed")
        _check_detector_view_scores(
            image_row,
            prefix="",
            combined_key="original_score",
            field=f"{field} original view",
        )
        _check_detector_view_scores(
            image_row,
            prefix="hflip_",
            combined_key="hflip_score",
            field=f"{field} horizontal-flip view",
        )
    elif "coordinates_restored_to_original_view" in image_row:
        raise OnnxBundleError(f"{field} classifier alignment cannot claim box restoration")
    return score_error, tensor_error, coordinate_error


def _check_bundle_directory(
    plan: OnnxBundlePlan,
    bundle_dir: Path,
    *,
    graph_inspector: GraphInspector,
) -> dict[str, Any]:
    manifest_path = bundle_dir / plan.spec.manifest_filename
    payload = _read_json_object(manifest_path, "ONNX bundle manifest")
    if (
        payload.get("schema_version") != plan.prediction.schema_version
        or payload.get("kind") != "onnx_bundle"
    ):
        raise OnnxBundleError("ONNX bundle manifest schema/kind is invalid")
    if payload.get("status") != "complete":
        raise OnnxBundleError("ONNX bundle manifest status must be complete")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise OnnxBundleError("ONNX bundle source must be an object")
    _check_bound_record(
        source.get("selection_manifest"),
        expected_path=plan.spec.selection_manifest,
        expected_sha256=plan.selection.manifest_sha256,
        root=plan.spec.project_root,
        field="source.selection_manifest",
    )
    _check_bound_record(
        source.get("export_config"),
        expected_path=plan.spec.config_path,
        expected_sha256=plan.spec.config_sha256,
        root=plan.spec.project_root,
        field="source.export_config",
    )
    _check_bound_record(
        source.get("prediction_config"),
        expected_path=plan.prediction.config_path,
        expected_sha256=sha256_file(plan.prediction.config_path),
        root=plan.spec.project_root,
        field="source.prediction_config",
    )
    _check_bound_record(
        source.get("detector_inference_config"),
        expected_path=plan.detector_inference.config_path,
        expected_sha256=plan.detector_inference.config_sha256,
        root=plan.spec.project_root,
        field="source.detector_inference_config",
    )
    if source.get("candidate") != plan.selection.candidate:
        raise OnnxBundleError("ONNX bundle candidate binding changed")
    export_record = payload.get("export")
    if not isinstance(export_record, Mapping):
        raise OnnxBundleError("ONNX bundle export record must be an object")
    if (
        export_record.get("ultralytics_version") != plan.prediction.ultralytics_version
        or export_record.get("device") != plan.spec.requested_device
        or export_record.get("parameters") != plan.spec.export.to_dict()
    ):
        raise OnnxBundleError("ONNX bundle export parameters changed")
    if payload.get("alignment_contract") != plan.spec.alignment.to_dict():
        raise OnnxBundleError("ONNX bundle alignment contract changed")
    if payload.get("preprocessing") != _preprocessing_record(plan):
        raise OnnxBundleError("ONNX bundle preprocessing contract changed")
    expected_tta = _tta_manifest_record(plan)
    if expected_tta is None:
        if "tta" in payload:
            raise OnnxBundleError("schema-v1 ONNX bundle unexpectedly contains a TTA contract")
    elif payload.get("tta") != expected_tta:
        raise OnnxBundleError("schema-v2 ONNX bundle TTA binding or algorithm contract changed")
    representatives = payload.get("representative_images")
    if not isinstance(representatives, list) or len(representatives) != 2:
        raise OnnxBundleError("ONNX bundle must bind exactly two representative images")
    recorded_image_hashes = {
        item.get("filename"): item.get("sha256")
        for item in representatives
        if isinstance(item, Mapping)
    }
    expected_image_hashes = {image.filename: image.sha256 for image in plan.representative_images}
    if recorded_image_hashes != expected_image_hashes:
        raise OnnxBundleError("ONNX bundle representative image binding changed")

    expected_artifacts = {
        (artifact.task, artifact.fold): artifact
        for artifact in (*plan.selection.detector_models, *plan.selection.classifier_models)
    }
    onnx_files: set[Path] = set()
    component_scores: dict[tuple[str, int, str], tuple[float, float]] = {}
    for task in ("detector", "classifier"):
        for row in _component_rows(payload, task):
            fold = row.get("fold")
            if type(fold) is not int:
                raise OnnxBundleError(f"components.{task} fold must be an integer")
            artifact = expected_artifacts[(task, fold)]
            _check_bound_record(
                row.get("source_weights"),
                expected_path=artifact.weights_path,
                expected_sha256=artifact.weights_sha256,
                root=plan.spec.project_root,
                field=f"{task} Fold {fold} source_weights",
            )
            _check_bound_record(
                row.get("source_run_manifest"),
                expected_path=artifact.run_manifest_path,
                expected_sha256=artifact.run_manifest_sha256,
                root=plan.spec.project_root,
                field=f"{task} Fold {fold} source_run_manifest",
            )
            if (
                row.get("task") != task
                or row.get("model_name") != artifact.model_name
                or row.get("imgsz") != artifact.imgsz
                or row.get("seed") != artifact.seed
            ):
                raise OnnxBundleError(f"{task} Fold {fold} model metadata changed")
            onnx_record = row.get("onnx")
            if not isinstance(onnx_record, Mapping):
                raise OnnxBundleError(f"{task} Fold {fold} ONNX record is invalid")
            onnx_path = _relative_bundle_file(
                bundle_dir, onnx_record.get("path"), f"{task} Fold {fold} onnx.path"
            )
            if not onnx_path.is_file():
                raise OnnxBundleError(f"{task} Fold {fold} ONNX file is missing")
            digest = onnx_record.get("sha256")
            if not isinstance(digest, str) or len(digest) != SHA256_LENGTH:
                raise OnnxBundleError(f"{task} Fold {fold} ONNX SHA-256 is invalid")
            if sha256_file(onnx_path) != digest:
                raise OnnxBundleError(f"{task} Fold {fold} ONNX SHA-256 drifted")
            if onnx_record.get("size_bytes") != onnx_path.stat().st_size:
                raise OnnxBundleError(f"{task} Fold {fold} ONNX size changed")
            graph = graph_inspector(
                onnx_path,
                task=task,
                imgsz=artifact.imgsz,
                ultralytics_version=plan.prediction.ultralytics_version,
            )
            if row.get("graph") != graph.to_dict():
                raise OnnxBundleError(f"{task} Fold {fold} ONNX graph contract drifted")
            provider = row.get("provider")
            session_providers = (
                provider.get("session_providers") if isinstance(provider, Mapping) else None
            )
            if (
                not isinstance(provider, Mapping)
                or provider.get("requested") != plan.spec.alignment.provider
                or provider.get("device_id") != plan.spec.alignment.provider_device_id
                or provider.get("use_tf32") is not False
                or provider.get("node_providers") != [plan.spec.alignment.provider]
                or not isinstance(session_providers, list)
                or not session_providers
                or session_providers[0] != plan.spec.alignment.provider
            ):
                raise OnnxBundleError(f"{task} Fold {fold} provider evidence is invalid")
            alignment = row.get("alignment")
            if not isinstance(alignment, Mapping) or alignment.get("status") != "passed":
                raise OnnxBundleError(f"{task} Fold {fold} alignment did not pass")
            expected_view_aggregation = (
                "original_only" if plan.tta is None else "arithmetic_mean(original,horizontal_flip)"
            )
            if alignment.get("view_aggregation") != expected_view_aggregation:
                raise OnnxBundleError(f"{task} Fold {fold} view aggregation changed")
            maximum_score_error = _finite_manifest_number(
                alignment.get("max_score_abs_error"),
                f"{task} Fold {fold} max_score_abs_error",
            )
            maximum_tensor_error = _finite_manifest_number(
                alignment.get("max_compared_tensor_abs_error"),
                f"{task} Fold {fold} max_compared_tensor_abs_error",
            )
            maximum_coordinate_error = _finite_manifest_number(
                alignment.get("max_coordinate_abs_error"),
                f"{task} Fold {fold} max_coordinate_abs_error",
            )
            if min(maximum_score_error, maximum_tensor_error, maximum_coordinate_error) < 0.0:
                raise OnnxBundleError(f"{task} Fold {fold} alignment errors must be non-negative")
            if maximum_score_error > plan.spec.alignment.score_atol:
                raise OnnxBundleError(f"{task} Fold {fold} score error exceeds tolerance")
            image_rows = alignment.get("images")
            if not isinstance(image_rows, list) or len(image_rows) != 2:
                raise OnnxBundleError(f"{task} Fold {fold} alignment images are incomplete")
            image_names = {item.get("filename") for item in image_rows if isinstance(item, Mapping)}
            if image_names != set(expected_image_hashes):
                raise OnnxBundleError(f"{task} Fold {fold} alignment image names changed")
            image_score_errors: list[float] = []
            image_tensor_errors: list[float] = []
            image_coordinate_errors: list[float] = []
            for image_row in image_rows:
                if not isinstance(image_row, Mapping):
                    raise OnnxBundleError(f"{task} Fold {fold} alignment image is invalid")
                score_error, tensor_error, coordinate_error = _check_component_image_alignment(
                    image_row,
                    plan=plan,
                    task=task,
                    fold=fold,
                )
                filename = image_row.get("filename")
                if not isinstance(filename, str):
                    raise OnnxBundleError(f"{task} Fold {fold} alignment filename is invalid")
                component_scores[(task, fold, filename)] = (
                    _manifest_probability(
                        image_row.get("pytorch_score"),
                        f"{task} Fold {fold} image PyTorch score",
                    ),
                    _manifest_probability(
                        image_row.get("ort_score"), f"{task} Fold {fold} image ORT score"
                    ),
                )
                image_score_errors.append(score_error)
                image_tensor_errors.append(tensor_error)
                image_coordinate_errors.append(coordinate_error)
            _require_close(
                maximum_score_error,
                max(image_score_errors, default=0.0),
                f"{task} Fold {fold} max_score_abs_error",
            )
            _require_close(
                maximum_tensor_error,
                max(image_tensor_errors, default=0.0),
                f"{task} Fold {fold} max_compared_tensor_abs_error",
            )
            _require_close(
                maximum_coordinate_error,
                max(image_coordinate_errors, default=0.0),
                f"{task} Fold {fold} max_coordinate_abs_error",
            )
            onnx_files.add(onnx_path)
    actual_files = {path.resolve() for path in bundle_dir.rglob("*") if path.is_file()}
    expected_files = onnx_files | {manifest_path.resolve()}
    if actual_files != expected_files:
        extra = sorted(str(path) for path in actual_files - expected_files)
        missing = sorted(str(path) for path in expected_files - actual_files)
        raise OnnxBundleError(f"ONNX bundle file closure failed; extra={extra}, missing={missing}")
    fusion = payload.get("fusion_alignment")
    if not isinstance(fusion, Mapping) or fusion.get("status") != "passed":
        raise OnnxBundleError("ONNX bundle final fusion alignment did not pass")
    if (
        fusion.get("labels_match") is not True
        or fusion.get("method") != "weighted_average"
        or fusion.get("detector_weight") != 1.0 - plan.selection.classifier_weight
        or fusion.get("classifier_weight") != plan.selection.classifier_weight
        or fusion.get("threshold_profile") != plan.spec.alignment.threshold_profile
        or fusion.get("threshold") != plan.threshold
    ):
        raise OnnxBundleError("ONNX bundle final labels do not match")
    if plan.tta is None:
        expected_fusion_aggregation = {
            "view_aggregation": "original_only",
            "detector_fold_reduction": "mean(max(full_score, slice_score))",
            "classifier_fold_reduction": "mean(positive_class_probability)",
        }
    else:
        expected_fusion_aggregation = {
            "view_aggregation": "arithmetic_mean(original,horizontal_flip)",
            "detector_fold_reduction": (
                "mean(arithmetic_mean(max(original_full,original_slice),"
                "max(horizontal_flip_full,horizontal_flip_slice)))"
            ),
            "classifier_fold_reduction": (
                "mean(arithmetic_mean(original_probability,horizontal_flip_probability))"
            ),
        }
    for field, expected in expected_fusion_aggregation.items():
        if fusion.get(field) != expected:
            raise OnnxBundleError(f"ONNX bundle final fusion {field} changed")
    maximum_fused_error = _finite_manifest_number(
        fusion.get("max_fused_score_abs_error"),
        "fusion_alignment.max_fused_score_abs_error",
    )
    if maximum_fused_error < 0.0 or maximum_fused_error > plan.spec.alignment.score_atol:
        raise OnnxBundleError("ONNX bundle final fused score error exceeds tolerance")
    fusion_images = fusion.get("images")
    if not isinstance(fusion_images, list) or len(fusion_images) != 2:
        raise OnnxBundleError("ONNX bundle final fusion images are incomplete")
    fusion_names: set[str] = set()
    fused_errors: list[float] = []
    for image_row in fusion_images:
        if not isinstance(image_row, Mapping):
            raise OnnxBundleError("ONNX bundle final fusion image is invalid")
        if image_row.get("pytorch_label") != image_row.get("ort_label"):
            raise OnnxBundleError("ONNX bundle final fused label changed")
        if type(image_row.get("pytorch_label")) is not int or image_row.get(
            "pytorch_label"
        ) not in {0, 1}:
            raise OnnxBundleError("ONNX bundle final label must be integer 0/1")
        filename = image_row.get("filename")
        if not isinstance(filename, str) or filename not in expected_image_hashes:
            raise OnnxBundleError("ONNX bundle final fusion filename changed")
        fusion_names.add(filename)
        error = _finite_manifest_number(
            image_row.get("fused_score_abs_error"), "fusion image fused_score_abs_error"
        )
        if error < 0.0 or error > plan.spec.alignment.score_atol:
            raise OnnxBundleError("ONNX bundle final image score error exceeds tolerance")
        fused_errors.append(error)
        pt_detector = _manifest_probability(
            image_row.get("pytorch_detector_mean"), "fusion image PyTorch detector mean"
        )
        ort_detector = _manifest_probability(
            image_row.get("ort_detector_mean"), "fusion image ORT detector mean"
        )
        pt_classifier = _manifest_probability(
            image_row.get("pytorch_classifier_mean"), "fusion image PyTorch classifier mean"
        )
        ort_classifier = _manifest_probability(
            image_row.get("ort_classifier_mean"), "fusion image ORT classifier mean"
        )
        expected_pt_detector = (
            math.fsum(
                component_scores[("detector", fold, filename)][0] for fold in sorted(EXPECTED_FOLDS)
            )
            / 5.0
        )
        expected_ort_detector = (
            math.fsum(
                component_scores[("detector", fold, filename)][1] for fold in sorted(EXPECTED_FOLDS)
            )
            / 5.0
        )
        expected_pt_classifier = (
            math.fsum(
                component_scores[("classifier", fold, filename)][0]
                for fold in sorted(EXPECTED_FOLDS)
            )
            / 5.0
        )
        expected_ort_classifier = (
            math.fsum(
                component_scores[("classifier", fold, filename)][1]
                for fold in sorted(EXPECTED_FOLDS)
            )
            / 5.0
        )
        _require_close(pt_detector, expected_pt_detector, "fusion image PyTorch detector mean")
        _require_close(ort_detector, expected_ort_detector, "fusion image ORT detector mean")
        _require_close(
            pt_classifier, expected_pt_classifier, "fusion image PyTorch classifier mean"
        )
        _require_close(ort_classifier, expected_ort_classifier, "fusion image ORT classifier mean")
        pt_fused = _manifest_probability(
            image_row.get("pytorch_fused_score"), "fusion image PyTorch fused score"
        )
        ort_fused = _manifest_probability(
            image_row.get("ort_fused_score"), "fusion image ORT fused score"
        )
        classifier_weight = plan.selection.classifier_weight
        detector_weight = 1.0 - classifier_weight
        _require_close(
            pt_fused,
            detector_weight * pt_detector + classifier_weight * pt_classifier,
            "fusion image PyTorch weighted average",
        )
        _require_close(
            ort_fused,
            detector_weight * ort_detector + classifier_weight * ort_classifier,
            "fusion image ORT weighted average",
        )
        _require_close(error, abs(pt_fused - ort_fused), "fusion image score error")
        expected_pt_label = int(pt_fused >= plan.threshold)
        expected_ort_label = int(ort_fused >= plan.threshold)
        if (
            image_row.get("pytorch_label") != expected_pt_label
            or image_row.get("ort_label") != expected_ort_label
        ):
            raise OnnxBundleError("ONNX bundle final label does not use the frozen threshold")
    if fusion_names != set(expected_image_hashes):
        raise OnnxBundleError("ONNX bundle final fusion image set changed")
    _require_close(
        maximum_fused_error,
        max(fused_errors, default=0.0),
        "fusion_alignment.max_fused_score_abs_error",
    )
    return payload


def check_onnx_bundle(
    plan: OnnxBundlePlan,
    *,
    graph_inspector: GraphInspector = inspect_onnx_graph,
) -> dict[str, Any]:
    """Recheck current source bindings, graph contracts, hashes, providers, and file closure."""

    if not plan.spec.output_dir.is_dir():
        raise OnnxBundleError(f"ONNX bundle output does not exist: {plan.spec.output_dir}")
    return _check_bundle_directory(
        plan,
        plan.spec.output_dir,
        graph_inspector=graph_inspector,
    )
