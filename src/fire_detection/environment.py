"""Runtime environment collection and validation utilities."""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

EXPECTED_PYTHON = (3, 11)
EXPECTED_TORCH = "2.11.0+cu128"
EXPECTED_TORCHVISION = "0.26.0+cu128"
EXPECTED_TORCH_CUDA = "12.8"
EXPECTED_ULTRALYTICS = "8.4.92"
EXPECTED_ONNXRUNTIME_GPU = "1.26.0"
EXPECTED_DEVICE_FRAGMENT = "RTX 5080"
EXPECTED_CAPABILITY = (12, 0)
EXPECTED_ARCH = "sm_120"

_OPENCV_DISTRIBUTIONS = {
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
}
_ORT_DISTRIBUTIONS = {"onnxruntime", "onnxruntime-gpu"}
_PROHIBITED_DISTRIBUTIONS = {
    "albumentations",
    "torchaudio",
    "onnxsim",
}


def _canonicalize_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _installed_distributions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        raw_name = distribution.metadata["Name"]
        if raw_name:
            packages[_canonicalize_distribution_name(raw_name)] = distribution.version
    return packages


def _find_yolo26_configs() -> tuple[str | None, str | None]:
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.submodule_search_locations is None:
        return None, None

    package_root = Path(next(iter(spec.submodule_search_locations)))
    config_root = package_root / "cfg" / "models" / "26"
    detect = config_root / "yolo26.yaml"
    classify = config_root / "yolo26-cls.yaml"
    detect_name = detect.relative_to(package_root).as_posix() if detect.is_file() else None
    classify_name = classify.relative_to(package_root).as_posix() if classify.is_file() else None
    return detect_name, classify_name


@dataclass(frozen=True)
class EnvironmentReport:
    """Serializable snapshot of the active Python and accelerator environment."""

    python_version: str
    python_executable: str
    python_prefix: str
    platform: str
    torch_version: str | None
    torchvision_version: str | None
    torch_cuda_version: str | None
    cuda_available: bool
    cuda_device_name: str | None
    cuda_capability: tuple[int, int] | None
    cuda_arch_list: tuple[str, ...]
    ultralytics_version: str | None
    onnxruntime_version: str | None
    onnxruntime_distributions: tuple[str, ...]
    ort_available_providers: tuple[str, ...]
    opencv_distributions: tuple[str, ...]
    prohibited_distributions_present: tuple[str, ...]
    yolo26_detect_config: str | None
    yolo26_classify_config: str | None
    collection_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the report as a JSON-serializable mapping."""

        return asdict(self)


@dataclass(frozen=True)
class EnvironmentValidation:
    """Result of checking an :class:`EnvironmentReport` against this project."""

    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the validation result as a JSON-serializable mapping."""

        return asdict(self)


def collect_environment_report() -> EnvironmentReport:
    """Collect package, CUDA, ORT, OpenCV, and installed YOLO26 configuration state."""

    packages = _installed_distributions()
    errors: list[str] = []
    cuda_available = False
    cuda_device_name: str | None = None
    cuda_capability: tuple[int, int] | None = None
    cuda_arch_list: tuple[str, ...] = ()
    torch_cuda_version: str | None = None

    try:
        import torch

        torch_cuda_version = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device_name = torch.cuda.get_device_name(0)
            cuda_capability = torch.cuda.get_device_capability(0)
            cuda_arch_list = tuple(torch.cuda.get_arch_list())
    except (ImportError, OSError, RuntimeError) as exc:
        errors.append(f"PyTorch collection failed: {exc}")

    ort_available_providers: tuple[str, ...] = ()
    try:
        import onnxruntime as ort

        ort_available_providers = tuple(ort.get_available_providers())
    except (ImportError, OSError, RuntimeError) as exc:
        errors.append(f"ONNX Runtime collection failed: {exc}")

    detect_config, classify_config = _find_yolo26_configs()
    ort_distributions = tuple(sorted(_ORT_DISTRIBUTIONS.intersection(packages)))
    opencv_distributions = tuple(sorted(_OPENCV_DISTRIBUTIONS.intersection(packages)))
    prohibited = tuple(sorted(_PROHIBITED_DISTRIBUTIONS.intersection(packages)))

    return EnvironmentReport(
        python_version=platform.python_version(),
        python_executable=sys.executable,
        python_prefix=sys.prefix,
        platform=platform.platform(),
        torch_version=packages.get("torch"),
        torchvision_version=packages.get("torchvision"),
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        cuda_capability=cuda_capability,
        cuda_arch_list=cuda_arch_list,
        ultralytics_version=packages.get("ultralytics"),
        onnxruntime_version=packages.get("onnxruntime-gpu") or packages.get("onnxruntime"),
        onnxruntime_distributions=ort_distributions,
        ort_available_providers=ort_available_providers,
        opencv_distributions=opencv_distributions,
        prohibited_distributions_present=prohibited,
        yolo26_detect_config=detect_config,
        yolo26_classify_config=classify_config,
        collection_errors=tuple(errors),
    )


def validate_environment(
    report: EnvironmentReport,
    *,
    require_cuda: bool = False,
    require_ort_cuda: bool = False,
    check_yolo: bool = False,
) -> EnvironmentValidation:
    """Validate fixed versions and optional CUDA/YOLO26 requirements."""

    errors = list(report.collection_errors)
    warnings: list[str] = []

    version_parts = tuple(int(part) for part in report.python_version.split(".")[:2])
    if version_parts != EXPECTED_PYTHON:
        errors.append(f"Python must be 3.11, found {report.python_version}")
    if report.torch_version != EXPECTED_TORCH:
        errors.append(f"torch must be {EXPECTED_TORCH}, found {report.torch_version}")
    if report.torchvision_version != EXPECTED_TORCHVISION:
        errors.append(
            f"torchvision must be {EXPECTED_TORCHVISION}, found {report.torchvision_version}"
        )
    if report.torch_cuda_version != EXPECTED_TORCH_CUDA:
        errors.append(
            f"PyTorch CUDA runtime must be {EXPECTED_TORCH_CUDA}, found {report.torch_cuda_version}"
        )
    if report.ultralytics_version != EXPECTED_ULTRALYTICS:
        errors.append(
            f"ultralytics must be {EXPECTED_ULTRALYTICS}, found {report.ultralytics_version}"
        )
    if report.onnxruntime_version != EXPECTED_ONNXRUNTIME_GPU:
        errors.append(
            "onnxruntime-gpu must be "
            f"{EXPECTED_ONNXRUNTIME_GPU}, found {report.onnxruntime_version}"
        )
    if report.onnxruntime_distributions != ("onnxruntime-gpu",):
        errors.append(
            "Exactly one ORT distribution is required: onnxruntime-gpu; found "
            f"{report.onnxruntime_distributions}"
        )
    if report.opencv_distributions != ("opencv-python",):
        errors.append(
            "Exactly one OpenCV distribution is required: opencv-python; found "
            f"{report.opencv_distributions}"
        )
    if report.prohibited_distributions_present:
        errors.append(
            "Prohibited baseline distributions are installed: "
            f"{report.prohibited_distributions_present}"
        )

    if require_cuda:
        if not report.cuda_available:
            errors.append("CUDA is required but torch.cuda.is_available() is false")
        if (
            report.cuda_device_name is None
            or EXPECTED_DEVICE_FRAGMENT not in report.cuda_device_name
        ):
            errors.append(f"Expected an RTX 5080, found {report.cuda_device_name}")
        if report.cuda_capability != EXPECTED_CAPABILITY:
            errors.append(
                f"Expected compute capability {EXPECTED_CAPABILITY}, found {report.cuda_capability}"
            )
        if EXPECTED_ARCH not in report.cuda_arch_list:
            errors.append(f"PyTorch wheel does not contain {EXPECTED_ARCH}")
    elif not report.cuda_available:
        warnings.append("CUDA is unavailable; CPU-only health reporting remains available")

    if require_ort_cuda and "CUDAExecutionProvider" not in report.ort_available_providers:
        errors.append("CUDAExecutionProvider is required but unavailable")

    if check_yolo:
        if report.yolo26_detect_config is None:
            errors.append("Installed Ultralytics package has no YOLO26 detection YAML")
        if report.yolo26_classify_config is None:
            errors.append("Installed Ultralytics package has no YOLO26 classification YAML")

    return EnvironmentValidation(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))
