"""GPU runtime benchmark for the seven fixed pre-official ablation candidates.

The benchmark is deliberately separate from the OOF diagnostics bundle.  It
measures six atomic inference components with the frozen five-fold checkpoints,
then composes the per-image timings into the seven already-defined candidates.
No labels, training entry points, or parameter fitting are involved.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from fire_detection.analysis.oof_diagnostics import CANDIDATE_ORDER
from fire_detection.config import (
    ConfigError,
    load_yaml_mapping,
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
from fire_detection.inference.slicing import generate_tile_windows, load_image_tiles
from fire_detection.inference.test_input import OfficialTestInputError, inspect_image_file
from fire_detection.inference.test_prediction import (
    FinalSelectionContract,
    ModelArtifact,
    TestImage,
    TestPredictionSpec,
    UltralyticsPredictionRunner,
    load_final_selection_contract,
    load_test_prediction_spec,
)
from fire_detection.inference.tta import image_tiles, load_horizontal_flipped_image
from fire_detection.models.features import (
    classifier_positive_probability,
    extract_ultralytics_detection_result,
    tensor_to_list,
)
from fire_detection.models.offline import sha256_file
from fire_detection.utils.io import atomic_write_csv, atomic_write_json

DETECTOR_ORIGINAL_FULL = "detector_original_full"
DETECTOR_ORIGINAL_SLICE = "detector_original_slice"
DETECTOR_HFLIP_FULL = "detector_hflip_full"
DETECTOR_HFLIP_SLICE = "detector_hflip_slice"
CLASSIFIER_ORIGINAL = "classifier_original"
CLASSIFIER_HFLIP = "classifier_hflip"

DETECTOR_COMPONENTS = (
    DETECTOR_ORIGINAL_FULL,
    DETECTOR_ORIGINAL_SLICE,
    DETECTOR_HFLIP_FULL,
    DETECTOR_HFLIP_SLICE,
)
CLASSIFIER_COMPONENTS = (CLASSIFIER_ORIGINAL, CLASSIFIER_HFLIP)
COMPONENT_ORDER = (*DETECTOR_COMPONENTS, *CLASSIFIER_COMPONENTS)

CANDIDATE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "detector_full_original": (DETECTOR_ORIGINAL_FULL,),
    "classifier_original": (CLASSIFIER_ORIGINAL,),
    "fusion_full_original": (DETECTOR_ORIGINAL_FULL, CLASSIFIER_ORIGINAL),
    "fusion_original_with_slicing": (
        DETECTOR_ORIGINAL_FULL,
        DETECTOR_ORIGINAL_SLICE,
        CLASSIFIER_ORIGINAL,
    ),
    "fusion_full_tta": (
        DETECTOR_ORIGINAL_FULL,
        DETECTOR_HFLIP_FULL,
        CLASSIFIER_ORIGINAL,
        CLASSIFIER_HFLIP,
    ),
    "fusion_slicing_tta_lofo": COMPONENT_ORDER,
    "frozen_winner": COMPONENT_ORDER,
}


class RuntimeBenchmarkError(ValueError):
    """Raised when the fixed runtime benchmark contract cannot be satisfied."""


@dataclass(frozen=True)
class RuntimeBenchmarkSpec:
    """Strict configuration for the fixed 100-image engineering benchmark."""

    project_root: Path
    config_path: Path
    config_sha256: str
    experiment_name: str
    prediction_config: Path
    processed_manifest: Path
    images_dir: Path
    sampling_method: str
    sample_count: int
    tile_count_tie_breaker: str
    warmup_count: int
    repeats: int
    ram_poll_interval_seconds: float
    output_dir: Path


@dataclass(frozen=True)
class BenchmarkImage:
    """One label-free engineering image and its deterministic sampling provenance."""

    image: TestImage
    tile_count: int
    population_rank: int
    quantile_index: int
    quantile_midpoint: float

    def row(self) -> dict[str, object]:
        """Return one stable sample-manifest row."""

        return {
            "quantile_index": self.quantile_index,
            "quantile_midpoint": self.quantile_midpoint,
            "population_rank": self.population_rank,
            "filename": self.image.filename,
            "sha256": self.image.sha256,
            "width": self.image.width,
            "height": self.image.height,
            "tile_count": self.tile_count,
        }


class ComponentBackend(Protocol):
    """Injectable boundary for real YOLO26 component execution."""

    def activate(self, model: ModelArtifact) -> None: ...

    def run_component(self, component: str, image: TestImage) -> None: ...

    def synchronize(self) -> None: ...

    def reset_gpu_peaks(self) -> None: ...

    def gpu_peaks(self) -> tuple[int, int]: ...

    def environment(self) -> Mapping[str, object]: ...

    def close(self) -> None: ...


class PeakMonitor(Protocol):
    """Context manager exposing peak process-tree resident memory."""

    peak_process_tree_ram_bytes: int

    def __enter__(self) -> PeakMonitor: ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ConfigError(
            f"{name} keys mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def load_runtime_benchmark_spec(path: Path, *, project_root: Path) -> RuntimeBenchmarkSpec:
    """Load the benchmark YAML and reject every implicit protocol choice."""

    root = project_root.resolve()
    config_path = path.resolve() if path.is_absolute() else (root / path).resolve()
    raw = load_yaml_mapping(config_path)
    _exact_keys(
        raw,
        {"schema_version", "experiment", "inputs", "sampling", "benchmark", "output"},
        "runtime_benchmark",
    )
    if require_int(raw, "schema_version", context="runtime_benchmark", minimum=1) != 1:
        raise ConfigError("runtime_benchmark.schema_version must be 1")
    experiment = require_mapping(raw, "experiment", context="runtime_benchmark")
    inputs = require_mapping(raw, "inputs", context="runtime_benchmark")
    sampling = require_mapping(raw, "sampling", context="runtime_benchmark")
    benchmark = require_mapping(raw, "benchmark", context="runtime_benchmark")
    output = require_mapping(raw, "output", context="runtime_benchmark")
    _exact_keys(experiment, {"name"}, "runtime_benchmark.experiment")
    _exact_keys(
        inputs,
        {"prediction_config", "processed_manifest", "images"},
        "runtime_benchmark.inputs",
    )
    _exact_keys(
        sampling,
        {"method", "sample_count", "tile_count_tie_breaker"},
        "runtime_benchmark.sampling",
    )
    _exact_keys(
        benchmark,
        {"warmup_count", "repeats", "ram_poll_interval_seconds"},
        "runtime_benchmark.benchmark",
    )
    _exact_keys(output, {"directory"}, "runtime_benchmark.output")

    method = require_string(sampling, "method", context="runtime_benchmark.sampling")
    if method != "midpoint_tile_count_quantiles":
        raise ConfigError("runtime_benchmark.sampling.method must be midpoint_tile_count_quantiles")
    tie_breaker = require_string(
        sampling, "tile_count_tie_breaker", context="runtime_benchmark.sampling"
    )
    if tie_breaker != "filename_ascending":
        raise ConfigError(
            "runtime_benchmark.sampling.tile_count_tie_breaker must be filename_ascending"
        )
    sample_count = require_int(
        sampling, "sample_count", context="runtime_benchmark.sampling", minimum=1
    )
    warmup_count = require_int(
        benchmark, "warmup_count", context="runtime_benchmark.benchmark", minimum=1
    )
    repeats = require_int(benchmark, "repeats", context="runtime_benchmark.benchmark", minimum=1)
    poll_interval = require_float(
        benchmark,
        "ram_poll_interval_seconds",
        context="runtime_benchmark.benchmark",
        minimum=0.001,
    )
    if sample_count != 100 or warmup_count != 10 or repeats != 3:
        raise ConfigError(
            "the pre-official runtime protocol is frozen at sample_count=100, "
            "warmup_count=10, repeats=3"
        )
    if warmup_count > sample_count:
        raise ConfigError("runtime benchmark warmup_count cannot exceed sample_count")
    prediction_config = resolve_project_path(
        root,
        require_string(inputs, "prediction_config", context="runtime_benchmark.inputs"),
    ).resolve()
    processed_manifest = resolve_project_path(
        root,
        require_string(inputs, "processed_manifest", context="runtime_benchmark.inputs"),
    ).resolve()
    images_dir = resolve_project_path(
        root, require_string(inputs, "images", context="runtime_benchmark.inputs")
    ).resolve()
    output_dir = resolve_project_path(
        root, require_string(output, "directory", context="runtime_benchmark.output")
    ).resolve()
    frozen_paths = {
        "inputs.prediction_config": (
            prediction_config,
            (root / "configs/predict_test_round3_tta.yaml").resolve(),
        ),
        "inputs.processed_manifest": (
            processed_manifest,
            (root / "data/processed/v3/manifest.json").resolve(),
        ),
        "inputs.images": (images_dir, (root / "data/processed/v3/images").resolve()),
        "output.directory": (
            output_dir,
            (root / "runs/analysis/pre_official/ablation_runtime").resolve(),
        ),
    }
    for field, (actual, expected) in frozen_paths.items():
        if actual != expected:
            raise ConfigError(
                f"runtime_benchmark.{field} is frozen at {expected.relative_to(root).as_posix()}"
            )
    return RuntimeBenchmarkSpec(
        project_root=root,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        experiment_name=require_string(experiment, "name", context="runtime_benchmark.experiment"),
        prediction_config=prediction_config,
        processed_manifest=processed_manifest,
        images_dir=images_dir,
        sampling_method=method,
        sample_count=sample_count,
        tile_count_tie_breaker=tie_breaker,
        warmup_count=warmup_count,
        repeats=repeats,
        ram_poll_interval_seconds=poll_interval,
        output_dir=output_dir,
    )


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise RuntimeBenchmarkError(f"{name} contains non-standard JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeBenchmarkError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    import json

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeBenchmarkError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeBenchmarkError(f"{name} must contain a JSON object")
    return value


def _manifest_image_rows(path: Path, *, images_dir: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        raise RuntimeBenchmarkError(f"processed manifest does not exist: {path}")
    if not images_dir.is_dir():
        raise RuntimeBenchmarkError(f"processed image directory does not exist: {images_dir}")
    payload = _load_json_object(path, name="processed manifest")
    raw_images = payload.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise RuntimeBenchmarkError("processed manifest images must be a non-empty list")
    if payload.get("image_count") != len(raw_images):
        raise RuntimeBenchmarkError("processed manifest image_count does not match images")
    rows: list[dict[str, object]] = []
    names: set[str] = set()
    for index, value in enumerate(raw_images):
        if not isinstance(value, Mapping):
            raise RuntimeBenchmarkError(f"processed manifest images[{index}] must be a mapping")
        filename = value.get("filename")
        digest = value.get("processed_sha256")
        width = value.get("width")
        height = value.get("height")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise RuntimeBenchmarkError(
                f"processed manifest images[{index}].filename must be a plain filename"
            )
        if filename in names:
            raise RuntimeBenchmarkError(f"processed manifest repeats filename {filename!r}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeBenchmarkError(
                f"processed manifest images[{index}].processed_sha256 is invalid"
            )
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise RuntimeBenchmarkError(
                f"processed manifest images[{index}] dimensions must be positive integers"
            )
        rows.append(
            {
                "filename": filename,
                "processed_sha256": digest,
                "width": width,
                "height": height,
                "path": (images_dir / filename).resolve(),
            }
        )
        names.add(filename)
    return tuple(rows)


def _validate_processed_manifest_binding(
    *,
    selection_manifest: Path,
    processed_manifest: Path,
    project_root: Path,
) -> None:
    """Require the engineering population to be the manifest frozen by the winner."""

    selection = _load_json_object(selection_manifest, name="final selection manifest")
    artifacts = selection.get("artifacts")
    inputs = artifacts.get("inputs") if isinstance(artifacts, Mapping) else None
    record = inputs.get("processed_manifest") if isinstance(inputs, Mapping) else None
    if not isinstance(record, Mapping):
        raise RuntimeBenchmarkError("final selection omits processed_manifest binding")
    raw_path = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        raise RuntimeBenchmarkError("final selection processed_manifest path is invalid")
    bound_path = (project_root / Path(raw_path)).resolve()
    if bound_path != processed_manifest.resolve():
        raise RuntimeBenchmarkError(
            "benchmark processed manifest differs from the frozen winner binding"
        )
    if not isinstance(expected_hash, str) or sha256_file(processed_manifest) != expected_hash:
        raise RuntimeBenchmarkError("benchmark processed manifest hash drifted")


def select_quantile_images(
    manifest_rows: Sequence[Mapping[str, object]],
    *,
    sample_count: int,
    tile_size: int,
    tile_stride: int,
) -> tuple[BenchmarkImage, ...]:
    """Select one midpoint from each tile-count quantile with filename tie-breaking."""

    if type(sample_count) is not int or sample_count < 1:
        raise RuntimeBenchmarkError("sample_count must be a positive integer")
    if len(manifest_rows) < sample_count:
        raise RuntimeBenchmarkError(
            f"cannot select {sample_count} unique images from {len(manifest_rows)} rows"
        )
    ranked: list[dict[str, object]] = []
    for row in manifest_rows:
        width = row.get("width")
        height = row.get("height")
        filename = row.get("filename")
        if type(width) is not int or type(height) is not int or not isinstance(filename, str):
            raise RuntimeBenchmarkError("manifest sampling rows have invalid fields")
        tile_count = len(
            generate_tile_windows(
                width=width,
                height=height,
                size=tile_size,
                stride=tile_stride,
            )
        )
        ranked.append({**row, "tile_count": tile_count})
    ranked.sort(
        key=lambda row: (
            cast(int, row["tile_count"]),
            cast(str, row["filename"]),
        )
    )
    population_count = len(ranked)
    selected: list[BenchmarkImage] = []
    used_ranks: set[int] = set()
    for quantile_index in range(sample_count):
        midpoint = (quantile_index + 0.5) / sample_count
        population_rank = min(
            ((2 * quantile_index + 1) * population_count) // (2 * sample_count),
            population_count - 1,
        )
        if population_rank in used_ranks:
            raise RuntimeBenchmarkError("midpoint quantile selection produced a duplicate rank")
        row = ranked[population_rank]
        path = row.get("path")
        if not isinstance(path, Path) or not path.is_file():
            raise RuntimeBenchmarkError(f"selected image does not exist: {path}")
        try:
            inspected = inspect_image_file(path)
        except OfficialTestInputError as exc:
            raise RuntimeBenchmarkError(str(exc)) from exc
        expected_hash = str(row["processed_sha256"])
        if inspected["sha256"] != expected_hash:
            raise RuntimeBenchmarkError(f"selected image hash drifted: {row['filename']}")
        if inspected["width"] != row["width"] or inspected["height"] != row["height"]:
            raise RuntimeBenchmarkError(f"selected image dimensions drifted: {row['filename']}")
        test_image = TestImage(
            filename=str(row["filename"]),
            path=path,
            sha256=expected_hash,
            width=cast(int, row["width"]),
            height=cast(int, row["height"]),
        )
        selected.append(
            BenchmarkImage(
                image=test_image,
                tile_count=cast(int, row["tile_count"]),
                population_rank=population_rank,
                quantile_index=quantile_index,
                quantile_midpoint=midpoint,
            )
        )
        used_ranks.add(population_rank)
    return tuple(selected)


class ProcessTreePeakMonitor(AbstractContextManager["ProcessTreePeakMonitor"]):
    """Poll RSS for the current process and all descendants during one measured block."""

    def __init__(self, poll_interval_seconds: float) -> None:
        self._poll_interval_seconds = poll_interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_process_tree_ram_bytes = 0

    @staticmethod
    def _rss() -> int:
        try:
            import psutil
        except ImportError:
            return 0
        try:
            root = psutil.Process()
            processes = [root, *root.children(recursive=True)]
            total = 0
            for process in processes:
                try:
                    total += int(process.memory_info().rss)
                except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
            return total
        except (OSError, psutil.Error):
            return 0

    def _poll(self) -> None:
        while not self._stop.wait(self._poll_interval_seconds):
            self.peak_process_tree_ram_bytes = max(self.peak_process_tree_ram_bytes, self._rss())

    def __enter__(self) -> ProcessTreePeakMonitor:
        self.peak_process_tree_ram_bytes = self._rss()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._poll_interval_seconds * 4.0))
        self.peak_process_tree_ram_bytes = max(self.peak_process_tree_ram_bytes, self._rss())


class UltralyticsComponentBackend:
    """Execute individual frozen YOLO26 inference components without retaining predictions."""

    def __init__(self, spec: TestPredictionSpec, detector: DetectorInferenceSpec) -> None:
        if spec.ultralytics_device == "cpu":
            raise RuntimeBenchmarkError("GPU runtime benchmark forbids CPU execution")
        self._spec = spec
        self._detector = detector
        self._runner = UltralyticsPredictionRunner(spec)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeBenchmarkError("CUDA is unavailable for the GPU runtime benchmark")
        self._torch = torch
        self._device = int(spec.ultralytics_device)
        self._model: Any | None = None
        self._artifact: ModelArtifact | None = None

    def activate(self, model: ModelArtifact) -> None:
        """Load one verified checkpoint; load time is excluded from candidate latency."""

        self._artifact = model
        self._model = self._runner._activate_model(  # noqa: SLF001
            model, detector=model.task == "detector"
        )

    def _one_result(self, *, source: object, detector: bool) -> Any:
        if self._model is None or self._artifact is None:
            raise RuntimeBenchmarkError("benchmark component ran without an active model")
        arguments: dict[str, object] = {
            "source": source,
            "stream": True,
            "augment": False,
            "device": self._spec.ultralytics_device,
            "verbose": False,
            "save": False,
        }
        if detector:
            arguments.update(
                {
                    "imgsz": self._detector.inference_imgsz,
                    "conf": self._detector.conf,
                    "max_det": self._detector.max_det,
                }
            )
        else:
            arguments["imgsz"] = self._spec.classifier_imgsz
        results = list(self._model.predict(**arguments))
        if len(results) != 1:
            raise RuntimeBenchmarkError(
                f"benchmark {self._artifact.task} Fold {self._artifact.fold} "
                f"returned {len(results)} results"
            )
        return results[0]

    @staticmethod
    def _validate_detector_result(result: Any) -> None:
        _filename, _features, boxes = extract_ultralytics_detection_result(result)
        if any(box.class_id != 0 for box in boxes):
            raise RuntimeBenchmarkError("benchmark detector produced a non-fire class")

    def _validate_classifier_result(self, result: Any) -> None:
        if result.probs is None:
            raise RuntimeBenchmarkError("benchmark classifier result has no probabilities")
        classifier_positive_probability(
            names=result.names,
            probabilities=tensor_to_list(result.probs.data),
            positive_name=self._spec.positive_class_name,
        )

    def _detector_slice(self, image: TestImage, *, horizontal_flip: bool) -> None:
        if horizontal_flip:
            _shape, opened = load_horizontal_flipped_image(image.path)
            try:
                tiles = image_tiles(
                    opened,
                    size=self._detector.slicing.size,
                    stride=self._detector.slicing.stride,
                )
                try:
                    for tile in tiles:
                        result = self._one_result(source=tile.image, detector=True)
                        self._validate_detector_result(result)
                finally:
                    for tile in tiles:
                        tile.image.close()
            finally:
                opened.close()
            return
        _shape, tiles = load_image_tiles(
            image.path,
            size=self._detector.slicing.size,
            stride=self._detector.slicing.stride,
        )
        try:
            for tile in tiles:
                result = self._one_result(source=tile.image, detector=True)
                self._validate_detector_result(result)
        finally:
            for tile in tiles:
                tile.image.close()

    def run_component(self, component: str, image: TestImage) -> None:
        """Run one atomic view/task component for one image."""

        if self._artifact is None:
            raise RuntimeBenchmarkError("benchmark component ran without an active artifact")
        expected_task = "detector" if component in DETECTOR_COMPONENTS else "classifier"
        if self._artifact.task != expected_task:
            raise RuntimeBenchmarkError(
                f"component {component} cannot use {self._artifact.task} checkpoint"
            )
        if component == DETECTOR_ORIGINAL_FULL:
            self._validate_detector_result(self._one_result(source=str(image.path), detector=True))
        elif component == DETECTOR_ORIGINAL_SLICE:
            self._detector_slice(image, horizontal_flip=False)
        elif component == DETECTOR_HFLIP_FULL:
            _shape, flipped = load_horizontal_flipped_image(image.path)
            try:
                self._validate_detector_result(self._one_result(source=flipped, detector=True))
            finally:
                flipped.close()
        elif component == DETECTOR_HFLIP_SLICE:
            self._detector_slice(image, horizontal_flip=True)
        elif component == CLASSIFIER_ORIGINAL:
            self._validate_classifier_result(
                self._one_result(source=str(image.path), detector=False)
            )
        elif component == CLASSIFIER_HFLIP:
            _shape, flipped = load_horizontal_flipped_image(image.path)
            try:
                self._validate_classifier_result(self._one_result(source=flipped, detector=False))
            finally:
                flipped.close()
        else:
            raise RuntimeBenchmarkError(f"unknown runtime component {component!r}")

    def synchronize(self) -> None:
        self._torch.cuda.synchronize(self._device)

    def reset_gpu_peaks(self) -> None:
        self._torch.cuda.reset_peak_memory_stats(self._device)

    def gpu_peaks(self) -> tuple[int, int]:
        return (
            int(self._torch.cuda.max_memory_allocated(self._device)),
            int(self._torch.cuda.max_memory_reserved(self._device)),
        )

    def environment(self) -> Mapping[str, object]:
        return self._runner.environment()

    def close(self) -> None:
        self._runner.close()
        self._model = None
        self._artifact = None


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise RuntimeBenchmarkError("cannot summarize an empty timing sequence")
    return float(np.percentile(np.asarray(values, dtype=float), quantile, method="linear"))


def _component_models(
    contract: FinalSelectionContract, component: str
) -> tuple[ModelArtifact, ...]:
    if component in DETECTOR_COMPONENTS:
        return contract.detector_models
    return contract.classifier_models


def measure_components(
    *,
    spec: RuntimeBenchmarkSpec,
    contract: FinalSelectionContract,
    sample: Sequence[BenchmarkImage],
    backend: ComponentBackend,
    clock: Callable[[], float] = time.perf_counter,
    monitor_factory: Callable[[], PeakMonitor] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure six atomic components for all five folds, images, and repeats."""

    if len(sample) != spec.sample_count:
        raise RuntimeBenchmarkError(
            f"benchmark sample has {len(sample)} rows, expected {spec.sample_count}"
        )
    names = [item.image.filename for item in sample]
    if len(set(names)) != len(names):
        raise RuntimeBenchmarkError("benchmark sample filenames are not unique")
    if len(contract.detector_models) != 5 or len(contract.classifier_models) != 5:
        raise RuntimeBenchmarkError(
            "runtime benchmark requires five detector and classifier models"
        )
    monitor_factory = monitor_factory or (
        lambda: ProcessTreePeakMonitor(spec.ram_poll_interval_seconds)
    )
    rows: list[dict[str, object]] = []
    resource_rows: list[dict[str, object]] = []
    warmup = tuple(sample[: spec.warmup_count])
    try:
        for component in COMPONENT_ORDER:
            for model in _component_models(contract, component):
                backend.activate(model)
                for item in warmup:
                    backend.run_component(component, item.image)
                backend.synchronize()
                for repeat in range(1, spec.repeats + 1):
                    backend.synchronize()
                    backend.reset_gpu_peaks()
                    with monitor_factory() as monitor:
                        for item in sample:
                            backend.synchronize()
                            started = clock()
                            backend.run_component(component, item.image)
                            backend.synchronize()
                            elapsed = clock() - started
                            if not math.isfinite(elapsed) or elapsed < 0.0:
                                raise RuntimeBenchmarkError(
                                    f"non-finite/negative timing for {component}"
                                )
                            rows.append(
                                {
                                    "component": component,
                                    "task": model.task,
                                    "fold": model.fold,
                                    "repeat": repeat,
                                    "filename": item.image.filename,
                                    "tile_count": item.tile_count,
                                    "seconds": elapsed,
                                }
                            )
                    allocated, reserved = backend.gpu_peaks()
                    if allocated < 0 or reserved < allocated:
                        raise RuntimeBenchmarkError("backend returned invalid GPU memory peaks")
                    resource_rows.append(
                        {
                            "component": component,
                            "task": model.task,
                            "fold": model.fold,
                            "repeat": repeat,
                            "peak_gpu_memory_bytes": allocated,
                            "peak_gpu_reserved_memory_bytes": reserved,
                            "peak_process_tree_ram_bytes": int(monitor.peak_process_tree_ram_bytes),
                        }
                    )
    finally:
        backend.close()
    timings = pd.DataFrame(rows)
    resources = pd.DataFrame(resource_rows)
    expected_timing_rows = len(COMPONENT_ORDER) * 5 * spec.repeats * spec.sample_count
    expected_resource_rows = len(COMPONENT_ORDER) * 5 * spec.repeats
    if len(timings) != expected_timing_rows or len(resources) != expected_resource_rows:
        raise RuntimeBenchmarkError("component benchmark row closure failed")
    return timings, resources


def compose_candidate_resources(
    timings: pd.DataFrame,
    component_resources: pd.DataFrame,
    *,
    sample_count: int,
    warmup_count: int,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compose per-image five-fold component timings into the seven candidates."""

    if tuple(CANDIDATE_COMPONENTS) != CANDIDATE_ORDER:
        raise RuntimeBenchmarkError("candidate/component mapping drifted from fixed OOF order")
    required_timing = {"component", "fold", "repeat", "filename", "seconds"}
    required_resources = {
        "component",
        "fold",
        "repeat",
        "peak_gpu_memory_bytes",
        "peak_gpu_reserved_memory_bytes",
        "peak_process_tree_ram_bytes",
    }
    if not required_timing.issubset(timings.columns):
        raise RuntimeBenchmarkError("component timings omit required columns")
    if not required_resources.issubset(component_resources.columns):
        raise RuntimeBenchmarkError("component resources omit required columns")
    timing_keys = ["component", "fold", "repeat", "filename"]
    if timings.duplicated(timing_keys).any():
        raise RuntimeBenchmarkError("component timings contain duplicate rows")
    resource_keys = ["component", "fold", "repeat"]
    if component_resources.duplicated(resource_keys).any():
        raise RuntimeBenchmarkError("component resources contain duplicate rows")
    expected_components = set(COMPONENT_ORDER)
    if set(timings["component"]) != expected_components:
        raise RuntimeBenchmarkError("component timing set is incomplete")
    if set(component_resources["component"]) != expected_components:
        raise RuntimeBenchmarkError("component resource set is incomplete")
    detail_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for order, candidate in enumerate(CANDIDATE_ORDER, start=1):
        components = CANDIDATE_COMPONENTS[candidate]
        selected = timings[timings["component"].isin(components)]
        detail = (
            selected.groupby(["repeat", "filename"], as_index=False, sort=True)["seconds"]
            .sum()
            .sort_values(["repeat", "filename"], kind="stable")
            .reset_index(drop=True)
        )
        if len(detail) != sample_count * repeats:
            raise RuntimeBenchmarkError(f"candidate {candidate} timing closure failed")
        detail.insert(0, "candidate", candidate)
        detail.insert(0, "candidate_order", order)
        detail_frames.append(detail)
        values = detail["seconds"].astype(float).tolist()
        resources = component_resources[component_resources["component"].isin(components)]
        summaries.append(
            {
                "candidate_order": order,
                "candidate": candidate,
                "status": "complete",
                "components": "+".join(components),
                "sample_count": sample_count,
                "warmup_count_per_model_component": warmup_count,
                "repeats": repeats,
                "measured_image_repeats": len(values),
                "total_seconds": float(math.fsum(values)),
                "mean_seconds_per_image": float(math.fsum(values) / len(values)),
                "p50_seconds_per_image": _percentile(values, 50.0),
                "p95_seconds_per_image": _percentile(values, 95.0),
                "peak_gpu_memory_bytes": int(resources["peak_gpu_memory_bytes"].max()),
                "peak_gpu_reserved_memory_bytes": int(
                    resources["peak_gpu_reserved_memory_bytes"].max()
                ),
                "peak_process_tree_ram_bytes": int(resources["peak_process_tree_ram_bytes"].max()),
                "five_fold_inference": True,
                "model_load_time_included": False,
            }
        )
    detail_output = pd.concat(detail_frames, ignore_index=True)
    summary_output = pd.DataFrame(summaries)
    sixth = detail_output[detail_output["candidate"] == "fusion_slicing_tta_lofo"].reset_index(
        drop=True
    )
    seventh = detail_output[detail_output["candidate"] == "frozen_winner"].reset_index(drop=True)
    if not sixth[["repeat", "filename", "seconds"]].equals(
        seventh[["repeat", "filename", "seconds"]]
    ):
        raise RuntimeBenchmarkError("candidate 6 and frozen winner must have identical runtime")
    return summary_output, detail_output


def _file_record(path: Path, *, root: Path) -> dict[str, object]:
    try:
        portable = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        portable = path.resolve().as_posix()
    return {"path": portable, "sha256": sha256_file(path)}


def _git_snapshot(root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        status = run("status", "--porcelain=v1", "--untracked-files=all")
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeBenchmarkError(f"cannot record Git snapshot: {exc}") from exc


def build_runtime_report(
    *,
    spec: RuntimeBenchmarkSpec,
    prediction: TestPredictionSpec,
    detector: DetectorInferenceSpec,
    contract: FinalSelectionContract,
    sample: Sequence[BenchmarkImage],
    candidate_summary: pd.DataFrame,
    component_resources: pd.DataFrame,
    environment: Mapping[str, object],
    git: Mapping[str, object],
) -> dict[str, object]:
    """Build a strict complete-only resource report from measured rows."""

    if len(candidate_summary) != len(CANDIDATE_ORDER):
        raise RuntimeBenchmarkError("runtime report requires all seven candidate measurements")
    if set(candidate_summary["status"]) != {"complete"}:
        raise RuntimeBenchmarkError("runtime report cannot publish incomplete measurements")
    histogram = Counter(item.tile_count for item in sample)
    return {
        "schema_version": 1,
        "kind": "pre_official_ablation_runtime_benchmark",
        "status": "complete",
        "engineering_only": True,
        "training_performed": False,
        "experiment": spec.experiment_name,
        "git": dict(git),
        "protocol": {
            "sampling": spec.sampling_method,
            "sample_count": spec.sample_count,
            "warmup_count_per_model_component": spec.warmup_count,
            "repeats": spec.repeats,
            "five_fold_inference": True,
            "timing_clock": "time.perf_counter_wall_clock_with_cuda_synchronize",
            "candidate_latency": "sum_of_measured_atomic_component_times",
            "model_load_time_included": False,
            "gpu_peak": "torch_cuda_peak_memory_allocated_max_over_required_components",
            "ram_peak": "polled_process_tree_rss_max_over_required_components",
        },
        "inputs": {
            "benchmark_config": _file_record(spec.config_path, root=spec.project_root),
            "prediction_config": _file_record(prediction.config_path, root=spec.project_root),
            "processed_manifest": _file_record(spec.processed_manifest, root=spec.project_root),
            "selection_manifest": {
                **_file_record(prediction.selection_manifest, root=spec.project_root),
                "validated_sha256": contract.manifest_sha256,
            },
            "detector_inference_config": _file_record(detector.config_path, root=spec.project_root),
            "models": {
                "detector": [
                    model.manifest_record(spec.project_root) for model in contract.detector_models
                ],
                "classifier": [
                    model.manifest_record(spec.project_root) for model in contract.classifier_models
                ],
            },
        },
        "sample": {
            "population_count": 1100,
            "selected_count": len(sample),
            "tile_size": detector.slicing.size,
            "tile_stride": detector.slicing.stride,
            "tie_breaker": spec.tile_count_tie_breaker,
            "tile_count_histogram": {str(key): histogram[key] for key in sorted(histogram)},
            "labels_consumed": False,
        },
        "environment": dict(environment),
        "measurements": candidate_summary.to_dict("records"),
        "component_resource_blocks": len(component_resources),
        "limitations": [
            "the 100-image set is an engineering latency set, not an accuracy estimate",
            "candidate latency is composed from identically measured atomic components",
            "candidate 6 and frozen winner use identical components and therefore identical cost",
            "checkpoint loading and benchmark warmup are excluded from candidate latency",
            "process-tree RSS is polling-based and may miss peaks shorter than the poll interval",
            "no official test labels, leaderboard scores, training, or parameter fitting are used",
        ],
    }


def publish_runtime_benchmark(
    *,
    spec: RuntimeBenchmarkSpec,
    sample: Sequence[BenchmarkImage],
    component_timings: pd.DataFrame,
    component_resources: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    candidate_timings: pd.DataFrame,
    report: Mapping[str, object],
) -> Path:
    """Atomically publish a measured bundle and refuse every existing destination."""

    if spec.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite runtime benchmark: {spec.output_dir}")
    spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = spec.output_dir.parent / f".{spec.output_dir.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        sample_frame = pd.DataFrame([item.row() for item in sample])
        frames = {
            "benchmark_sample.csv": sample_frame,
            "component_measurements.csv": component_timings,
            "component_resource_blocks.csv": component_resources,
            "candidate_measurements.csv": candidate_timings,
            "ablation_resource_report.csv": candidate_summary,
        }
        outputs: dict[str, dict[str, object]] = {}
        for filename, frame in frames.items():
            path = staging / filename
            atomic_write_csv(path, frame)
            outputs[filename] = {
                "path": (spec.output_dir / filename)
                .resolve()
                .relative_to(spec.project_root)
                .as_posix(),
                "sha256": sha256_file(path),
                "rows": len(frame),
            }
        report_path = staging / "ablation_resource_report.json"
        atomic_write_json(report_path, report)
        outputs[report_path.name] = {
            "path": (spec.output_dir / report_path.name)
            .resolve()
            .relative_to(spec.project_root)
            .as_posix(),
            "sha256": sha256_file(report_path),
        }
        manifest = {
            "schema_version": 1,
            "kind": "pre_official_ablation_runtime_benchmark_manifest",
            "status": "complete",
            "engineering_only": True,
            "config_sha256": spec.config_sha256,
            "candidate_order": list(CANDIDATE_ORDER),
            "outputs": outputs,
        }
        atomic_write_json(staging / "benchmark_manifest.json", manifest)
        staging.replace(spec.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return spec.output_dir


def prepare_runtime_benchmark(
    spec: RuntimeBenchmarkSpec,
) -> tuple[
    TestPredictionSpec,
    DetectorInferenceSpec,
    FinalSelectionContract,
    tuple[BenchmarkImage, ...],
]:
    """Validate the frozen predictor, ten checkpoints, and deterministic sample."""

    prediction = load_test_prediction_spec(spec.prediction_config, project_root=spec.project_root)
    if prediction.tta is None:
        raise RuntimeBenchmarkError("runtime benchmark requires the frozen horizontal-flip TTA")
    if prediction.classifier_weight != 0.45:
        raise RuntimeBenchmarkError("runtime benchmark requires classifier_weight=0.45")
    detector = load_detector_inference_spec(prediction.detector_inference_config)
    if not (
        detector.inference_imgsz == 640
        and detector.slicing.enabled
        and detector.slicing.size == 640
        and detector.slicing.stride == 512
        and math.isclose(detector.slicing.overlap, 0.2, rel_tol=0.0, abs_tol=1e-12)
        and prediction.classifier_imgsz == 384
    ):
        raise RuntimeBenchmarkError("runtime benchmark inference settings drifted from winner")
    contract = load_final_selection_contract(prediction)
    if contract.stable_threshold != 0.475:
        raise RuntimeBenchmarkError("runtime benchmark requires frozen stable threshold 0.475")
    _validate_processed_manifest_binding(
        selection_manifest=prediction.selection_manifest,
        processed_manifest=spec.processed_manifest,
        project_root=spec.project_root,
    )
    rows = _manifest_image_rows(spec.processed_manifest, images_dir=spec.images_dir)
    if len(rows) != 1100:
        raise RuntimeBenchmarkError(
            f"runtime benchmark requires 1100 canonical images, found {len(rows)}"
        )
    sample = select_quantile_images(
        rows,
        sample_count=spec.sample_count,
        tile_size=detector.slicing.size,
        tile_stride=detector.slicing.stride,
    )
    return prediction, detector, contract, sample


def run_runtime_benchmark(
    spec: RuntimeBenchmarkSpec,
    *,
    backend_factory: Callable[
        [TestPredictionSpec, DetectorInferenceSpec], ComponentBackend
    ] = UltralyticsComponentBackend,
) -> Path:
    """Run the complete GPU benchmark and atomically publish only complete results."""

    if spec.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite runtime benchmark: {spec.output_dir}")
    prediction, detector, contract, sample = prepare_runtime_benchmark(spec)
    backend = backend_factory(prediction, detector)
    environment = dict(backend.environment())
    component_timings, component_resources = measure_components(
        spec=spec,
        contract=contract,
        sample=sample,
        backend=backend,
    )
    candidate_summary, candidate_timings = compose_candidate_resources(
        component_timings,
        component_resources,
        sample_count=spec.sample_count,
        warmup_count=spec.warmup_count,
        repeats=spec.repeats,
    )
    report = build_runtime_report(
        spec=spec,
        prediction=prediction,
        detector=detector,
        contract=contract,
        sample=sample,
        candidate_summary=candidate_summary,
        component_resources=component_resources,
        environment=environment,
        git=_git_snapshot(spec.project_root),
    )
    return publish_runtime_benchmark(
        spec=spec,
        sample=sample,
        component_timings=component_timings,
        component_resources=component_resources,
        candidate_summary=candidate_summary,
        candidate_timings=candidate_timings,
        report=report,
    )


__all__ = [
    "CANDIDATE_COMPONENTS",
    "CLASSIFIER_COMPONENTS",
    "COMPONENT_ORDER",
    "DETECTOR_COMPONENTS",
    "BenchmarkImage",
    "ComponentBackend",
    "ProcessTreePeakMonitor",
    "RuntimeBenchmarkError",
    "RuntimeBenchmarkSpec",
    "UltralyticsComponentBackend",
    "build_runtime_report",
    "compose_candidate_resources",
    "load_runtime_benchmark_spec",
    "measure_components",
    "prepare_runtime_benchmark",
    "publish_runtime_benchmark",
    "run_runtime_benchmark",
    "select_quantile_images",
]
