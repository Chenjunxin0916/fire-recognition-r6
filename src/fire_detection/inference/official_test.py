"""Resumable, contract-bound execution for official or engineering test inference."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from fire_detection.config import ConfigError, load_yaml_mapping
from fire_detection.inference.submission import (
    load_submission_strict,
    predictions_to_submission,
    validate_submission_payload,
)
from fire_detection.inference.submission_candidates import generate_submission_candidates
from fire_detection.inference.test_input import (
    OfficialTestInputError,
    validate_test_images,
    write_test_input_manifest,
)
from fire_detection.inference.test_prediction import (
    ClassifierFoldPrediction,
    DetectorFoldPrediction,
    FinalSelectionContract,
    ModelArtifact,
    PredictionRunner,
    SelectionValidator,
    TestImage,
    TestPredictionError,
    TestPredictionSpec,
    UltralyticsPredictionRunner,
    _close_prediction_runner,
    _fusion_manifest,
    _normalize_result_mapping,
    _portable_path,
    _raw_box_payload,
    _runner_environment,
    _test_bundle_sha256,
    _tta_manifest_record,
    _validate_inference_spec,
    _validate_prediction_frame,
    aggregate_fold_predictions,
    load_final_selection_contract,
    load_test_prediction_spec,
    validate_test_inference_manifest,
)
from fire_detection.models.offline import OfflinePreparationError, sha256_file
from fire_detection.utils import atomic_write_csv, atomic_write_json
from fire_detection.utils.io import atomic_write_text

OFFICIAL_COUNT = 1198
OFFICIAL_PREDICTION_CONFIG = Path("configs/predict_test_round3_tta.yaml")
OFFICIAL_OUTPUT = Path("runs/inference/round3_test")
OFFICIAL_SUBMISSIONS = Path("submissions/round3_tta")
PROFILE_NAMES = ("stable", "recall", "precision")


class OfficialTestError(RuntimeError):
    """Base class carrying a stable command-line exit code."""

    exit_code = 4


class OfficialInputError(OfficialTestError):
    """CLI, directory, lock, or input-contract failure."""

    exit_code = 2


class OfficialContractError(OfficialTestError):
    """Frozen config, manifest, model, or hash failure."""

    exit_code = 3


class OfficialRuntimeError(OfficialTestError):
    """GPU, disk, OOM, or prediction failure."""

    exit_code = 4


class OfficialValidationError(OfficialTestError):
    """Final inference or submission validation failure."""

    exit_code = 5


@dataclass(frozen=True)
class OfficialTestSpec:
    """Strict wrapper around the frozen prediction config."""

    project_root: Path
    config_path: Path
    mode: str
    prediction: TestPredictionSpec
    input_manifest_filename: str
    submissions_dir: Path
    shard_size: int
    minimum_free_bytes: int
    archive: Path | None

    @property
    def output_dir(self) -> Path:
        return self.prediction.output_dir

    @property
    def work_dir(self) -> Path:
        return self.output_dir.parent / f".{self.output_dir.name}.official-inprogress"

    @property
    def lock_path(self) -> Path:
        return self.output_dir.parent / f".{self.output_dir.name}.official.lock"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ConfigError(
            f"{name} keys mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _plain_filename(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ConfigError(f"{name} must be a plain filename")
    if "/" in value or "\\" in value:
        raise ConfigError(f"{name} must be a plain filename")
    return value


def _project_path(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be non-empty text")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigError(f"{name} must stay inside the project root") from exc
    return resolved


def load_official_test_spec(path: Path, *, project_root: Path) -> OfficialTestSpec:
    """Load a wrapper config and enforce non-overridable official values."""

    root = project_root.resolve()
    config = path.resolve() if path.is_absolute() else (root / path).resolve()
    raw = load_yaml_mapping(config)
    _exact_keys(
        raw,
        {
            "schema_version",
            "mode",
            "prediction_config",
            "test_images",
            "expected_count",
            "input_manifest_filename",
            "output_directory",
            "submissions_directory",
            "shard_size",
            "minimum_free_bytes",
            "archive",
        },
        "official_test",
    )
    if raw.get("schema_version") != 1:
        raise ConfigError("official_test.schema_version must be 1")
    mode = raw.get("mode")
    if mode not in {"official", "mock"}:
        raise ConfigError("official_test.mode must be official or mock")
    expected_count = raw.get("expected_count")
    shard_size = raw.get("shard_size")
    minimum_free_bytes = raw.get("minimum_free_bytes")
    for value, name, minimum in (
        (expected_count, "expected_count", 1),
        (shard_size, "shard_size", 1),
        (minimum_free_bytes, "minimum_free_bytes", 0),
    ):
        if type(value) is not int or value < minimum:
            raise ConfigError(f"official_test.{name} must be an integer >= {minimum}")
    prediction_config = _project_path(root, raw.get("prediction_config"), "prediction_config")
    test_images = _project_path(root, raw.get("test_images"), "test_images")
    output = _project_path(root, raw.get("output_directory"), "output_directory")
    submissions = _project_path(root, raw.get("submissions_directory"), "submissions_directory")
    archive_value = raw.get("archive")
    archive = None
    if archive_value is not None:
        archive = _project_path(root, archive_value, "archive")
    manifest_filename = _plain_filename(
        raw.get("input_manifest_filename"), "input_manifest_filename"
    )
    if mode == "official":
        required = {
            "prediction_config": root / OFFICIAL_PREDICTION_CONFIG,
            "test_images": root / "data/test/images",
            "output_directory": root / OFFICIAL_OUTPUT,
            "submissions_directory": root / OFFICIAL_SUBMISSIONS,
        }
        actual = {
            "prediction_config": prediction_config,
            "test_images": test_images,
            "output_directory": output,
            "submissions_directory": submissions,
        }
        for name, required_path in required.items():
            if actual[name] != required_path.resolve():
                raise ConfigError(f"official mode fixes {name} to {required_path}")
        if expected_count != OFFICIAL_COUNT or shard_size != 16:
            raise ConfigError("official mode fixes expected_count=1198 and shard_size=16")
    else:
        forbidden = {
            (root / "data/test").resolve(),
            (root / OFFICIAL_OUTPUT).resolve(),
            (root / OFFICIAL_SUBMISSIONS).resolve(),
        }
        for candidate in (test_images, output, submissions):
            if any(candidate == item or item in candidate.parents for item in forbidden):
                raise ConfigError("mock mode must not use an official input or output namespace")
    try:
        prediction = load_test_prediction_spec(
            prediction_config,
            project_root=root,
            test_dir_override=test_images,
            output_dir_override=output,
            expected_count_override=expected_count,
        )
    except ConfigError as exc:
        raise OfficialContractError(f"frozen prediction config is invalid: {exc}") from exc
    if mode == "official" and prediction.tta is None:
        raise ConfigError("official mode requires the frozen explicit horizontal-flip TTA")
    reserved_names = {
        prediction.predictions_filename,
        prediction.boxes_filename,
        prediction.manifest_filename,
        "run_contract.json",
        "progress_manifest.json",
        "failure.json",
        "events.jsonl",
        "official_test_runtime_report.json",
        "mock_test_1198_runtime_report.json",
        "shards",
        "quarantine",
    }
    if manifest_filename in reserved_names:
        raise ConfigError("input_manifest_filename collides with a reserved run artifact")

    def overlaps(first: Path, second: Path) -> bool:
        return first == second or first in second.parents or second in first.parents

    if overlaps(output, submissions):
        raise ConfigError("inference and submission output directories must not overlap")
    if overlaps(test_images, output) or overlaps(test_images, submissions):
        raise ConfigError("test input and output directories must not overlap")
    return OfficialTestSpec(
        project_root=root,
        config_path=config,
        mode=str(mode),
        prediction=prediction,
        input_manifest_filename=manifest_filename,
        submissions_dir=submissions,
        shard_size=cast(int, shard_size),
        minimum_free_bytes=cast(int, minimum_free_bytes),
        archive=archive,
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _git_evidence(project_root: Path) -> dict[str, object]:
    """Capture the exact execution HEAD and dirty paths without mutating Git."""

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
        dirty_paths = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OfficialContractError(f"cannot capture execution Git state: {exc}") from exc
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise OfficialContractError("execution Git HEAD is not a lowercase 40-character SHA-1")
    return {"commit": commit, "dirty": bool(dirty_paths), "dirty_paths": dirty_paths}


def _digest_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _stable_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "created_at"}


def _audit_images(
    spec: OfficialTestSpec, *, existing_manifest: Path | None = None
) -> tuple[dict[str, Any], tuple[TestImage, ...]]:
    try:
        current = validate_test_images(
            spec.prediction.test_images,
            expected_count=spec.prediction.expected_count,
            archive=spec.archive,
        )
    except (OfficialTestInputError, OSError) as exc:
        raise OfficialInputError(str(exc)) from exc
    payload = current
    if existing_manifest is not None:
        try:
            saved = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OfficialContractError(f"cannot read saved input manifest: {exc}") from exc
        if not isinstance(saved, dict) or _stable_audit(saved) != _stable_audit(current):
            raise OfficialContractError("test input drifted from the saved hashed manifest")
        payload = saved
    records = payload.get("images")
    if not isinstance(records, list) or len(records) != spec.prediction.expected_count:
        raise OfficialContractError("input manifest image coverage is invalid")
    images: list[TestImage] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise OfficialContractError(f"input manifest image {index} must be an object")
        filename = record.get("filename")
        digest = record.get("sha256")
        width = record.get("width")
        height = record.get("height")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(digest, str)
            or len(digest) != 64
            or type(width) is not int
            or width <= 0
            or type(height) is not int
            or height <= 0
        ):
            raise OfficialContractError(f"input manifest image {index} is malformed")
        images.append(
            TestImage(
                filename=filename,
                path=(spec.prediction.test_images / filename).resolve(),
                sha256=digest,
                width=width,
                height=height,
            )
        )
    if [image.filename for image in images] != sorted(image.filename for image in images):
        raise OfficialContractError("input manifest filenames are not sorted")
    return payload, tuple(images)


def _contract_payload(
    spec: OfficialTestSpec,
    selection: FinalSelectionContract,
    *,
    input_manifest_sha256: str,
    environment: Mapping[str, object],
    git: Mapping[str, object],
) -> dict[str, Any]:
    prediction = spec.prediction
    lock = spec.project_root / "requirements/lock-win-py311-cu128.txt"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "official_test_run_contract",
        "mode": spec.mode,
        "wrapper_config": {
            "path": _portable_path(spec.config_path, spec.project_root),
            "sha256": sha256_file(spec.config_path),
        },
        "prediction_config": {
            "path": _portable_path(prediction.config_path, spec.project_root),
            "sha256": sha256_file(prediction.config_path),
        },
        "selection_manifest": {
            "path": _portable_path(prediction.selection_manifest, spec.project_root),
            "sha256": selection.manifest_sha256,
        },
        "detector_inference_config": {
            "path": _portable_path(prediction.detector_inference_config, spec.project_root),
            "sha256": sha256_file(prediction.detector_inference_config),
        },
        "offline_manifest": {
            "path": _portable_path(prediction.offline_manifest, spec.project_root),
            "sha256": sha256_file(prediction.offline_manifest),
        },
        "requirements_lock": None
        if not lock.is_file()
        else {"path": _portable_path(lock, spec.project_root), "sha256": sha256_file(lock)},
        "input_manifest_sha256": input_manifest_sha256,
        "expected_count": prediction.expected_count,
        "shard_size": spec.shard_size,
        "candidate": selection.candidate,
        "classifier_weight": selection.classifier_weight,
        "stable_threshold": selection.stable_threshold,
        "tta": None
        if prediction.tta is None
        else _tta_manifest_record(prediction.tta, root=spec.project_root),
        "device": prediction.requested_device,
        "environment": dict(environment),
        "git": dict(git),
        "models": {
            "detector": [
                model.manifest_record(spec.project_root) for model in selection.detector_models
            ],
            "classifier": [
                model.manifest_record(spec.project_root) for model in selection.classifier_models
            ],
        },
        "outputs": {
            "inference": _portable_path(spec.output_dir, spec.project_root),
            "submissions": _portable_path(spec.submissions_dir, spec.project_root),
        },
    }
    return payload


def _write_run_contract(path: Path, payload: Mapping[str, Any]) -> str:
    digest = _digest_json(payload)
    atomic_write_json(path, {"contract_sha256": digest, "contract": dict(payload)})
    return digest


def _load_run_contract(path: Path) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfficialContractError(f"cannot read run contract: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("contract"), dict):
        raise OfficialContractError("run contract is malformed")
    payload = dict(value["contract"])
    digest = _digest_json(payload)
    if value.get("contract_sha256") != digest:
        raise OfficialContractError("run contract SHA-256 is inconsistent")
    return payload, digest


class _RunLock:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.resume = resume
        self.acquired = False

    @staticmethod
    def _alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            import psutil
        except ImportError:
            if os.name == "nt":
                raise OfficialInputError(
                    "psutil is required for safe run-lock validation on Windows"
                ) from None
        else:
            try:
                process = psutil.Process(pid)
                return bool(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                return False
            except psutil.AccessDenied:
                return True
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def __enter__(self) -> _RunLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                owner = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(owner.get("pid", -1))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pid = -1
            if self._alive(pid):
                raise OfficialInputError(f"official test run is already active under PID {pid}")
            if not self.resume:
                raise OfficialInputError(
                    "stale run lock exists; use --resume or cleanup explicitly"
                )
            self.path.unlink(missing_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise OfficialInputError("official test run lock was acquired concurrently") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"pid": os.getpid(), "started_unix": time.time()}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.acquired = True
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


def _event(path: Path, event: str, **fields: object) -> None:
    payload = {"time_unix": time.time(), "event": event, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _shards(images: Sequence[TestImage], size: int) -> tuple[tuple[TestImage, ...], ...]:
    return tuple(tuple(images[start : start + size]) for start in range(0, len(images), size))


def _score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OfficialRuntimeError(f"{name} must be a real probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise OfficialRuntimeError(f"{name} must be finite and in [0, 1]")
    return result


def _validate_shard_payload(
    payload: object,
    *,
    contract_sha256: str,
    task: str,
    fold: int,
    index: int,
    filenames: Sequence[str],
    tta_enabled: bool,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OfficialContractError("checkpoint shard must be a JSON object")
    saved_digest = payload.get("payload_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if not isinstance(saved_digest, str) or saved_digest != _digest_json(unsigned):
        raise OfficialContractError("checkpoint shard payload SHA-256 is inconsistent")
    expected_scalars = {
        "schema_version": 1,
        "contract_sha256": contract_sha256,
        "task": task,
        "fold": fold,
        "shard_index": index,
        "tta_enabled": tta_enabled,
    }
    for field, expected in expected_scalars.items():
        if type(payload.get(field)) is not type(expected) or payload.get(field) != expected:
            raise OfficialContractError(f"checkpoint shard {field} drifted")
    rows = payload.get("rows")
    if not isinstance(rows, list) or [
        row.get("filename") for row in rows if isinstance(row, dict)
    ] != list(filenames):
        raise OfficialContractError("checkpoint shard filename coverage drifted")
    for row in rows:
        if not isinstance(row, dict):
            raise OfficialContractError("checkpoint shard row must be an object")
        if task == "detector":
            for field in ("full_score", "slice_score"):
                _score(row.get(field), f"detector shard {field}")
            for field in ("hflip_full_score", "hflip_slice_score"):
                if tta_enabled and row.get(field) is None:
                    raise OfficialContractError(f"detector TTA shard omits {field}")
                if not tta_enabled and row.get(field) is not None:
                    raise OfficialContractError(f"non-TTA detector shard contains {field}")
                if row.get(field) is not None:
                    _score(row.get(field), f"detector shard {field}")
            if not isinstance(row.get("raw_box_payload"), dict):
                raise OfficialContractError("detector shard omits raw box payload")
        else:
            _score(row.get("original_score"), "classifier original score")
            if tta_enabled and row.get("hflip_score") is None:
                raise OfficialContractError("classifier TTA shard omits hflip_score")
            if not tta_enabled and row.get("hflip_score") is not None:
                raise OfficialContractError("non-TTA classifier shard contains hflip_score")
            if row.get("hflip_score") is not None:
                _score(row.get("hflip_score"), "classifier hflip score")
    elapsed = payload.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise OfficialContractError("checkpoint shard elapsed_seconds is invalid")
    if not math.isfinite(float(elapsed)) or float(elapsed) < 0.0:
        raise OfficialContractError("checkpoint shard elapsed_seconds is invalid")
    resources = payload.get("resource_snapshot")
    if not isinstance(resources, dict):
        raise OfficialContractError("checkpoint shard resource_snapshot is invalid")
    for field in (
        "workspace_bytes_current",
        "process_tree_rss_bytes_current",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
    ):
        value = resources.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise OfficialContractError(f"checkpoint shard resource {field} is invalid")
    return payload


def _read_shard(
    path: Path,
    *,
    contract_sha256: str,
    task: str,
    fold: int,
    index: int,
    filenames: Sequence[str],
    tta_enabled: bool,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfficialContractError(f"cannot read checkpoint shard {path.name}: {exc}") from exc
    return _validate_shard_payload(
        value,
        contract_sha256=contract_sha256,
        task=task,
        fold=fold,
        index=index,
        filenames=filenames,
        tta_enabled=tta_enabled,
    )


def _write_shard(path: Path, payload: Mapping[str, Any], **validation: object) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    sealed = dict(payload)
    sealed.pop("payload_sha256", None)
    sealed["payload_sha256"] = _digest_json(sealed)
    atomic_write_json(temporary, sealed)
    _read_shard(temporary, **validation)  # type: ignore[arg-type]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(path)
    return sha256_file(path)


def _checkpoint_path(root: Path, task: str, fold: int, index: int) -> Path:
    return root / "shards" / f"{task}_fold{fold}_shard{index:04d}.json"


def _quarantine(path: Path, root: Path) -> None:
    quarantine = root / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    path.replace(quarantine / f"{path.name}.invalid-{uuid.uuid4().hex}")


def _detector_rows(
    *,
    runner: PredictionRunner,
    model: ModelArtifact,
    images: Sequence[TestImage],
    prediction: TestPredictionSpec,
    inference: Any,
) -> list[dict[str, Any]]:
    values = runner.predict_detector(model, images, inference, device=prediction.ultralytics_device)
    normalized = _normalize_result_mapping(
        values,
        expected_names=[image.filename for image in images],
        name=f"detector Fold {model.fold}",
    )
    rows: list[dict[str, Any]] = []
    for image in images:
        item = normalized[image.filename]
        if not isinstance(item, DetectorFoldPrediction):
            raise OfficialRuntimeError("detector runner returned the wrong result type")
        rows.append(
            {
                "filename": image.filename,
                "full_score": _score(item.full_score, "detector full score"),
                "slice_score": _score(item.slice_score, "detector slice score"),
                "hflip_full_score": None
                if item.hflip_full_score is None
                else _score(item.hflip_full_score, "detector hflip full score"),
                "hflip_slice_score": None
                if item.hflip_slice_score is None
                else _score(item.hflip_slice_score, "detector hflip slice score"),
                "raw_box_payload": _raw_box_payload(
                    image.filename,
                    model,
                    item,
                    inference,
                    project_root=prediction.project_root,
                    tta=prediction.tta,
                ),
            }
        )
    return rows


def _classifier_rows(
    *,
    runner: PredictionRunner,
    model: ModelArtifact,
    images: Sequence[TestImage],
    prediction: TestPredictionSpec,
) -> list[dict[str, Any]]:
    values = runner.predict_classifier(
        model,
        images,
        device=prediction.ultralytics_device,
        imgsz=prediction.classifier_imgsz,
        positive_class_name=prediction.positive_class_name,
        augment=False,
    )
    normalized = _normalize_result_mapping(
        values,
        expected_names=[image.filename for image in images],
        name=f"classifier Fold {model.fold}",
    )
    rows: list[dict[str, Any]] = []
    for image in images:
        item = normalized[image.filename]
        if isinstance(item, ClassifierFoldPrediction):
            original = item.original_score
            hflip: float | None = item.hflip_score
        else:
            original = item
            hflip = None
        rows.append(
            {
                "filename": image.filename,
                "original_score": _score(original, "classifier original score"),
                "hflip_score": None if hflip is None else _score(hflip, "classifier hflip score"),
            }
        )
    return rows


def _workspace_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _resource_snapshot(work: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"workspace_bytes_current": _workspace_bytes(work)}
    try:
        import psutil

        process = psutil.Process()
        processes = [process, *process.children(recursive=True)]
        rss = 0
        for item in processes:
            try:
                if item.is_running():
                    rss += int(item.memory_info().rss)
            except (OSError, psutil.Error):
                continue
        result["process_tree_rss_bytes_current"] = rss
    except (ImportError, OSError):
        result["process_tree_rss_bytes_current"] = None
    try:
        import torch

        if torch.cuda.is_available():
            result["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            result["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        else:
            result["gpu_peak_allocated_bytes"] = None
            result["gpu_peak_reserved_bytes"] = None
    except (ImportError, RuntimeError):
        result["gpu_peak_allocated_bytes"] = None
        result["gpu_peak_reserved_bytes"] = None
    return result


def _resource_report(root: Path, completed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    final_snapshot = _resource_snapshot(root)
    snapshots: list[Mapping[str, Any]] = [final_snapshot]
    for row in completed:
        snapshot = row.get("resource_snapshot")
        if isinstance(snapshot, Mapping):
            snapshots.append(snapshot)

    def maximum(field: str) -> int | None:
        values = [value for row in snapshots if type(value := row.get(field)) is int]
        return max(values) if values else None

    return {
        "sampling": "after_each_completed_shard_and_at_final_aggregation",
        "final_snapshot": final_snapshot,
        "max_observed": {
            "workspace_bytes": maximum("workspace_bytes_current"),
            "process_tree_rss_bytes": maximum("process_tree_rss_bytes_current"),
            "gpu_peak_allocated_bytes": maximum("gpu_peak_allocated_bytes"),
            "gpu_peak_reserved_bytes": maximum("gpu_peak_reserved_bytes"),
        },
        "peak_semantics": {
            "workspace_and_rss": "maximum of discrete shard-boundary snapshots",
            "gpu": "maximum of torch CUDA peak-memory counters observed at shard boundaries",
        },
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def _recovery_evidence(root: Path, completed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    resumed = [row for row in completed if row.get("resumed") is True]
    prior_failure_path = root / "failure.json"
    prior_failure: dict[str, Any] | None = None
    if prior_failure_path.is_file():
        try:
            value = json.loads(prior_failure_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                failure_digest = sha256_file(prior_failure_path)
                recovery_dir = root / "recovery"
                recovery_dir.mkdir(parents=True, exist_ok=True)
                archived_failure = recovery_dir / f"failure-{failure_digest[:16]}.json"
                if archived_failure.exists():
                    if sha256_file(archived_failure) != failure_digest:
                        raise OSError("recovery failure archive collides")
                    prior_failure_path.unlink()
                else:
                    prior_failure_path.replace(archived_failure)
                prior_failure = {
                    "path": archived_failure.relative_to(root).as_posix(),
                    "sha256": failure_digest,
                    "exit_code": value.get("exit_code"),
                    "exception_type": value.get("exception_type"),
                }
        except (OSError, UnicodeError, json.JSONDecodeError):
            prior_failure = None
    return {
        "resumed_shards": len(resumed),
        "new_shards": len(completed) - len(resumed),
        "resumed_rows": sum(int(row["rows"]) for row in resumed),
        "resume_verified": bool(resumed),
        "controlled_interrupt_observed": (
            prior_failure is not None and prior_failure.get("exit_code") == 130
        ),
        "prior_failure": prior_failure,
    }


def _progress_payload(
    root: Path, contract_sha256: str, completed: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "in_progress",
        "contract_sha256": contract_sha256,
        "completed_shards": completed,
        "resource_snapshot": _resource_snapshot(root),
    }


def _run_shards(
    *,
    spec: OfficialTestSpec,
    root: Path,
    images: Sequence[TestImage],
    selection: FinalSelectionContract,
    contract_sha256: str,
    runner: PredictionRunner,
    inference: Any,
) -> list[dict[str, Any]]:
    event_log = root / "events.jsonl"
    chunks = _shards(images, spec.shard_size)
    completed: list[dict[str, Any]] = []
    tasks = (
        ("detector", selection.detector_models),
        ("classifier", selection.classifier_models),
    )
    for task, models in tasks:
        for model in models:
            for index, chunk in enumerate(chunks):
                filenames = [image.filename for image in chunk]
                target = _checkpoint_path(root, task, model.fold, index)
                if target.is_file():
                    try:
                        saved = _read_shard(
                            target,
                            contract_sha256=contract_sha256,
                            task=task,
                            fold=model.fold,
                            index=index,
                            filenames=filenames,
                            tta_enabled=spec.prediction.tta is not None,
                        )
                    except OfficialContractError:
                        _quarantine(target, root)
                    else:
                        completed.append(
                            {
                                "id": f"{task}:{model.fold}:{index}",
                                "path": target.relative_to(root).as_posix(),
                                "sha256": sha256_file(target),
                                "rows": len(saved["rows"]),
                                "filenames": filenames,
                                "elapsed_seconds": float(saved.get("elapsed_seconds", 0.0)),
                                "resource_snapshot": dict(saved["resource_snapshot"]),
                                "resumed": True,
                            }
                        )
                        continue
                started = time.perf_counter()
                _event(event_log, "shard_started", task=task, fold=model.fold, shard=index)
                if task == "detector":
                    rows = _detector_rows(
                        runner=runner,
                        model=model,
                        images=chunk,
                        prediction=spec.prediction,
                        inference=inference,
                    )
                else:
                    rows = _classifier_rows(
                        runner=runner,
                        model=model,
                        images=chunk,
                        prediction=spec.prediction,
                    )
                elapsed = time.perf_counter() - started
                resource_snapshot = _resource_snapshot(root)
                payload = {
                    "schema_version": 1,
                    "contract_sha256": contract_sha256,
                    "task": task,
                    "fold": model.fold,
                    "shard_index": index,
                    "tta_enabled": spec.prediction.tta is not None,
                    "rows": rows,
                    "elapsed_seconds": elapsed,
                    "resource_snapshot": resource_snapshot,
                }
                digest = _write_shard(
                    target,
                    payload,
                    contract_sha256=contract_sha256,
                    task=task,
                    fold=model.fold,
                    index=index,
                    filenames=filenames,
                    tta_enabled=spec.prediction.tta is not None,
                )
                record = {
                    "id": f"{task}:{model.fold}:{index}",
                    "path": target.relative_to(root).as_posix(),
                    "sha256": digest,
                    "rows": len(rows),
                    "filenames": filenames,
                    "elapsed_seconds": elapsed,
                    "resource_snapshot": resource_snapshot,
                    "resumed": False,
                }
                completed.append(record)
                atomic_write_json(
                    root / "progress_manifest.json",
                    _progress_payload(root, contract_sha256, completed),
                )
                _event(
                    event_log,
                    "shard_completed",
                    task=task,
                    fold=model.fold,
                    shard=index,
                    elapsed_seconds=elapsed,
                    sha256=digest,
                )
    return completed


def _collect_shards(
    *,
    spec: OfficialTestSpec,
    root: Path,
    images: Sequence[TestImage],
    selection: FinalSelectionContract,
    contract_sha256: str,
) -> tuple[
    dict[int, dict[str, DetectorFoldPrediction]],
    dict[int, dict[str, float | ClassifierFoldPrediction]],
    list[dict[str, Any]],
]:
    chunks = _shards(images, spec.shard_size)
    detector: dict[int, dict[str, DetectorFoldPrediction]] = {
        model.fold: {} for model in selection.detector_models
    }
    classifier: dict[int, dict[str, float | ClassifierFoldPrediction]] = {
        model.fold: {} for model in selection.classifier_models
    }
    raw_boxes: list[dict[str, Any]] = []
    for task, models in (
        ("detector", selection.detector_models),
        ("classifier", selection.classifier_models),
    ):
        for model in models:
            for index, chunk in enumerate(chunks):
                path = _checkpoint_path(root, task, model.fold, index)
                payload = _read_shard(
                    path,
                    contract_sha256=contract_sha256,
                    task=task,
                    fold=model.fold,
                    index=index,
                    filenames=[image.filename for image in chunk],
                    tta_enabled=spec.prediction.tta is not None,
                )
                for row in payload["rows"]:
                    filename = str(row["filename"])
                    if task == "detector":
                        detector[model.fold][filename] = DetectorFoldPrediction(
                            full_score=float(row["full_score"]),
                            slice_score=float(row["slice_score"]),
                            hflip_full_score=None
                            if row["hflip_full_score"] is None
                            else float(row["hflip_full_score"]),
                            hflip_slice_score=None
                            if row["hflip_slice_score"] is None
                            else float(row["hflip_slice_score"]),
                        )
                        raw_boxes.append(dict(row["raw_box_payload"]))
                    elif spec.prediction.tta is None:
                        classifier[model.fold][filename] = float(row["original_score"])
                    else:
                        classifier[model.fold][filename] = ClassifierFoldPrediction(
                            original_score=float(row["original_score"]),
                            hflip_score=float(row["hflip_score"]),
                        )
    return detector, classifier, raw_boxes


def _inference_manifest(
    *,
    spec: OfficialTestSpec,
    selection: FinalSelectionContract,
    images: Sequence[TestImage],
    inference: Any,
    environment: Mapping[str, object],
    git: Mapping[str, object],
    artifact_root: Path,
    predictions_path: Path,
    boxes_path: Path,
    contract_sha256: str,
    input_manifest_sha256: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    prediction = spec.prediction
    manifest: dict[str, Any] = {
        "schema_version": prediction.schema_version,
        "status": "complete",
        "task": "five_fold_test_prediction",
        "experiment_name": prediction.experiment_name,
        "seed": prediction.seed,
        "candidate": selection.candidate,
        "git": dict(git),
        "config": {
            "path": _portable_path(prediction.config_path, spec.project_root),
            "sha256": sha256_file(prediction.config_path),
        },
        "selection_manifest": {
            "path": _portable_path(prediction.selection_manifest, spec.project_root),
            "sha256": selection.manifest_sha256,
        },
        "detector_inference_config": {
            "path": _portable_path(prediction.detector_inference_config, spec.project_root),
            "sha256": inference.config_sha256,
            **inference.provenance(),
        },
        "input_manifest": {
            "path": _portable_path(artifact_root / spec.input_manifest_filename, spec.project_root),
            "sha256": input_manifest_sha256,
        },
        "run_contract": {
            "path": _portable_path(artifact_root / "run_contract.json", spec.project_root),
            "contract_sha256": contract_sha256,
        },
        "models": {
            "detector": [
                model.manifest_record(spec.project_root) for model in selection.detector_models
            ],
            "classifier": [
                model.manifest_record(spec.project_root) for model in selection.classifier_models
            ],
        },
        "test_images": {
            "directory": _portable_path(prediction.test_images, spec.project_root),
            "count": len(images),
            "bundle_sha256": _test_bundle_sha256(images),
            "files": [image.manifest_record() for image in images],
        },
        "fusion": _fusion_manifest(prediction, selection),
        "runtime": {
            "requested_device": prediction.requested_device,
            "ultralytics_device": prediction.ultralytics_device,
            "elapsed_seconds": elapsed_seconds,
            "environment": dict(environment),
        },
        "outputs": {
            "test_predictions_csv": {
                "path": _portable_path(
                    artifact_root / prediction.predictions_filename, spec.project_root
                ),
                "sha256": sha256_file(predictions_path),
                "rows": len(images),
            },
            "detector_boxes_jsonl": {
                "path": _portable_path(
                    artifact_root / prediction.boxes_filename, spec.project_root
                ),
                "sha256": sha256_file(boxes_path),
                "rows": len(images) * 5,
            },
        },
    }
    if prediction.tta is not None:
        manifest["tta"] = _tta_manifest_record(prediction.tta, root=spec.project_root)
    return manifest


def _validate_official_inference_manifest(
    *,
    spec: OfficialTestSpec,
    artifact_root: Path,
    contract_sha256: str,
    input_manifest_sha256: str,
    environment: Mapping[str, object],
    git: Mapping[str, object],
    selection_validator: SelectionValidator | None,
) -> dict[str, Any]:
    """Run the full generic validator plus official input/contract bindings."""

    manifest_path = artifact_root / spec.prediction.manifest_filename
    try:
        manifest, _digest = validate_test_inference_manifest(
            manifest_path,
            spec.project_root,
            selection_validator=selection_validator,
        )
        input_record = _mapping(manifest.get("input_manifest"), "input_manifest")
        expected_input_path = _portable_path(
            artifact_root / spec.input_manifest_filename, spec.project_root
        )
        if input_record != {
            "path": expected_input_path,
            "sha256": input_manifest_sha256,
        }:
            raise OfficialContractError("inference input-manifest binding drifted")
        if sha256_file(artifact_root / spec.input_manifest_filename) != input_manifest_sha256:
            raise OfficialContractError("inference input manifest content drifted")

        contract_record = _mapping(manifest.get("run_contract"), "run_contract")
        expected_contract_path = _portable_path(
            artifact_root / "run_contract.json", spec.project_root
        )
        if contract_record != {
            "path": expected_contract_path,
            "contract_sha256": contract_sha256,
        }:
            raise OfficialContractError("inference run-contract binding drifted")
        contract_payload, saved_contract_sha256 = _load_run_contract(
            artifact_root / "run_contract.json"
        )
        if saved_contract_sha256 != contract_sha256:
            raise OfficialContractError("inference run contract SHA-256 drifted")
        if contract_payload.get("input_manifest_sha256") != input_manifest_sha256:
            raise OfficialContractError("run contract input-manifest SHA-256 drifted")
        if _canonical_json(contract_payload.get("environment")) != _canonical_json(environment):
            raise OfficialContractError("run contract environment drifted")
        if _canonical_json(contract_payload.get("git")) != _canonical_json(git):
            raise OfficialContractError("run contract Git evidence drifted")

        runtime = _mapping(manifest.get("runtime"), "runtime")
        if _canonical_json(runtime.get("environment")) != _canonical_json(environment):
            raise OfficialContractError("inference runtime environment drifted")
        if _canonical_json(manifest.get("git")) != _canonical_json(git):
            raise OfficialContractError("inference Git evidence drifted")
    except (ConfigError, OfficialContractError, TestPredictionError, OSError) as exc:
        raise OfficialValidationError(f"official inference validation failed: {exc}") from exc
    return manifest


def _publish_inference(
    *,
    spec: OfficialTestSpec,
    root: Path,
    images: Sequence[TestImage],
    selection: FinalSelectionContract,
    contract_sha256: str,
    input_manifest_sha256: str,
    environment: Mapping[str, object],
    git: Mapping[str, object],
    inference: Any,
    completed: Sequence[Mapping[str, Any]],
    selection_validator: SelectionValidator | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    detector, classifier, raw_boxes = _collect_shards(
        spec=spec,
        root=root,
        images=images,
        selection=selection,
        contract_sha256=contract_sha256,
    )
    predictions = aggregate_fold_predictions(
        [image.filename for image in images],
        detector,
        classifier,
        classifier_weight=selection.classifier_weight,
        tta=spec.prediction.tta,
        threshold=selection.stable_threshold,
    )
    _validate_prediction_frame(
        predictions,
        expected_rows=len(images),
        classifier_weight=selection.classifier_weight,
        tta=spec.prediction.tta,
        threshold=selection.stable_threshold,
    )
    predictions_path = root / spec.prediction.predictions_filename
    boxes_path = root / spec.prediction.boxes_filename
    atomic_write_csv(predictions_path, predictions)
    atomic_write_text(
        boxes_path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in sorted(
                raw_boxes, key=lambda row: (str(row["filename"]), int(row["model_fold"]))
            )
        ),
    )
    elapsed_values = [float(row["elapsed_seconds"]) for row in completed]
    per_image_totals = {image.filename: 0.0 for image in images}
    per_image_contributions = {image.filename: 0 for image in images}
    for row in completed:
        filenames = row.get("filenames")
        if not isinstance(filenames, list):
            raise OfficialValidationError("completed shard omits its filename coverage")
        shard_per_image = float(row["elapsed_seconds"]) / max(1, int(row["rows"]))
        for filename in filenames:
            if filename not in per_image_totals:
                raise OfficialValidationError("completed shard contains an unknown filename")
            per_image_totals[filename] += shard_per_image
            per_image_contributions[filename] += 1
    per_image = list(per_image_totals.values())
    expected_contributions = len(selection.detector_models) + len(selection.classifier_models)
    if set(per_image_contributions.values()) != {expected_contributions}:
        raise OfficialValidationError("runtime per-image shard coverage is incomplete")
    recovery = _recovery_evidence(root, completed)
    shard_elapsed_total = math.fsum(elapsed_values)
    end_to_end_model_seconds_mean = shard_elapsed_total / len(images)
    runtime_report = {
        "schema_version": 1,
        "kind": "mock_test_1198_runtime_report"
        if spec.mode == "mock"
        else "official_test_runtime_report",
        "engineering_only": spec.mode == "mock",
        "status": "complete",
        "exit_code": 0,
        "count": len(images),
        "success_count": len(images),
        "failure_count": 0,
        "completed_shards": len(completed),
        "shard_elapsed_seconds_total": shard_elapsed_total,
        "end_to_end_model_seconds_per_image_mean": end_to_end_model_seconds_mean,
        "per_image_seconds_mean": statistics.fmean(per_image) if per_image else 0.0,
        "per_image_seconds_p50": _percentile(per_image, 0.50),
        "per_image_seconds_p95": _percentile(per_image, 0.95),
        "per_image_distribution_method": "sum_of_task_fold_shard_average_seconds",
        "per_image_latency_exact": False,
        "per_image_latency_note": (
            "mean equals total model-shard seconds/count; P50/P95 are shard-normalized "
            "estimates because the runner does not expose exact per-image timings"
        ),
        "aggregation_seconds": time.perf_counter() - started,
        "resume_verified": recovery["resume_verified"],
        "resumed_shards": recovery["resumed_shards"],
        "controlled_interrupt_observed": recovery["controlled_interrupt_observed"],
        "recovery": recovery,
        "resources": _resource_report(root, completed),
    }
    runtime_name = (
        "mock_test_1198_runtime_report.json"
        if spec.mode == "mock"
        else "official_test_runtime_report.json"
    )
    atomic_write_json(root / runtime_name, runtime_report)
    manifest = _inference_manifest(
        spec=spec,
        selection=selection,
        images=images,
        inference=inference,
        environment=environment,
        git=git,
        artifact_root=root,
        predictions_path=predictions_path,
        boxes_path=boxes_path,
        contract_sha256=contract_sha256,
        input_manifest_sha256=input_manifest_sha256,
        elapsed_seconds=math.fsum(elapsed_values),
    )
    atomic_write_json(root / spec.prediction.manifest_filename, manifest)
    atomic_write_json(
        root / "progress_manifest.json",
        {
            "schema_version": 1,
            "status": "inference_complete",
            "contract_sha256": contract_sha256,
            "completed_shards": list(completed),
            "resources": _resource_snapshot(root),
        },
    )
    _validate_official_inference_manifest(
        spec=spec,
        artifact_root=root,
        contract_sha256=contract_sha256,
        input_manifest_sha256=input_manifest_sha256,
        environment=environment,
        git=git,
        selection_validator=selection_validator,
    )
    if root != spec.output_dir:
        published_manifest = _inference_manifest(
            spec=spec,
            selection=selection,
            images=images,
            inference=inference,
            environment=environment,
            git=git,
            artifact_root=spec.output_dir,
            predictions_path=predictions_path,
            boxes_path=boxes_path,
            contract_sha256=contract_sha256,
            input_manifest_sha256=input_manifest_sha256,
            elapsed_seconds=math.fsum(elapsed_values),
        )
        atomic_write_json(root / spec.prediction.manifest_filename, published_manifest)
        root.replace(spec.output_dir)
    try:
        validated = _validate_official_inference_manifest(
            spec=spec,
            artifact_root=spec.output_dir,
            contract_sha256=contract_sha256,
            input_manifest_sha256=input_manifest_sha256,
            environment=environment,
            git=git,
            selection_validator=selection_validator,
        )
    except OfficialValidationError as exc:
        recovery_path = spec.work_dir
        if not recovery_path.exists() and spec.output_dir.exists():
            spec.output_dir.replace(recovery_path)
        raise OfficialValidationError(f"published inference failed validation: {exc}") from exc
    return validated


def _selection_thresholds(
    spec: OfficialTestSpec, selection: FinalSelectionContract
) -> dict[str, float]:
    try:
        payload = json.loads(spec.prediction.selection_manifest.read_text(encoding="utf-8"))
        profiles = _mapping(payload.get("threshold_profiles"), "threshold_profiles")
        thresholds = {
            profile: _score(
                _mapping(profiles.get(profile), f"threshold profile {profile}").get("threshold"),
                f"threshold profile {profile}",
            )
            for profile in PROFILE_NAMES
        }
    except (OSError, UnicodeError, json.JSONDecodeError, ConfigError, OfficialRuntimeError) as exc:
        raise OfficialContractError(f"cannot load frozen submission thresholds: {exc}") from exc
    if not thresholds["recall"] <= thresholds["stable"] <= thresholds["precision"]:
        raise OfficialContractError("frozen thresholds must satisfy recall <= stable <= precision")
    if (
        selection.stable_threshold is not None
        and thresholds["stable"] != selection.stable_threshold
    ):
        raise OfficialContractError("submission stable threshold differs from final selection")
    return thresholds


def _filename_inventory_sha256(filenames: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{filename}\n" for filename in filenames).encode("utf-8")
    ).hexdigest()


def _validate_submissions(
    spec: OfficialTestSpec, *, thresholds: Mapping[str, float]
) -> dict[str, Any]:
    manifest_path = spec.submissions_dir / "submission_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfficialValidationError(f"cannot read submission manifest: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != "three_threshold_submission_candidates"
        or manifest.get("primary") != "stable"
    ):
        raise OfficialValidationError("submission manifest must select stable as primary")
    inference_path = spec.output_dir / spec.prediction.manifest_filename
    predictions_path = spec.output_dir / spec.prediction.predictions_filename
    expected_bindings = {
        "selection_manifest": {
            "path": _portable_path(spec.prediction.selection_manifest, spec.project_root),
            "sha256": sha256_file(spec.prediction.selection_manifest),
        },
        "inference_manifest": {
            "path": _portable_path(inference_path, spec.project_root),
            "sha256": sha256_file(inference_path),
        },
        "predictions": {
            "path": _portable_path(predictions_path, spec.project_root),
            "sha256": sha256_file(predictions_path),
            "rows": spec.prediction.expected_count,
        },
    }
    for field, expected in expected_bindings.items():
        if manifest.get(field) != expected:
            raise OfficialValidationError(f"submission manifest {field} binding drifted")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(PROFILE_NAMES):
        raise OfficialValidationError("submission manifest must contain three profiles")
    try:
        input_payload = json.loads(
            (spec.output_dir / spec.input_manifest_filename).read_text(encoding="utf-8")
        )
        image_records = input_payload["images"]
        if not isinstance(image_records, list):
            raise TypeError("images must be a list")
        filenames = tuple(str(image["filename"]) for image in image_records)
        frame = pd.read_csv(predictions_path)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        pd.errors.ParserError,
    ) as exc:
        raise OfficialValidationError(f"cannot load submission validation inputs: {exc}") from exc
    expected_test_images = {
        "path": _portable_path(spec.prediction.test_images, spec.project_root),
        "count": len(filenames),
        "filenames_sha256": _filename_inventory_sha256(filenames),
    }
    if manifest.get("test_images") != expected_test_images:
        raise OfficialValidationError("submission manifest test-image binding drifted")
    report_outputs: dict[str, Any] = {}
    for profile in PROFILE_NAMES:
        record = outputs[profile]
        if not isinstance(record, dict):
            raise OfficialValidationError(f"submission {profile} record is invalid")
        path = spec.submissions_dir / f"submission_{profile}.json"
        try:
            payload = load_submission_strict(path)
            validate_submission_payload(
                payload, filenames, expected_count=spec.prediction.expected_count
            )
            expected_payload = predictions_to_submission(
                frame,
                expected_filenames=filenames,
                threshold=thresholds[profile],
            )
        except (OSError, ValueError) as exc:
            raise OfficialValidationError(f"submission {profile} is invalid: {exc}") from exc
        if payload != expected_payload:
            raise OfficialValidationError(
                f"submission {profile} values differ from frozen-threshold probabilities"
            )
        expected_record = {
            "path": _portable_path(path, spec.project_root),
            "sha256": sha256_file(path),
            "threshold": thresholds[profile],
            "rows": len(payload),
        }
        if record != expected_record:
            raise OfficialValidationError(f"submission {profile} manifest hash/rows drifted")
        report_outputs[profile] = {
            "path": _portable_path(path, spec.project_root),
            "sha256": sha256_file(path),
            "rows": len(payload),
            "threshold": record.get("threshold"),
            "utf8_bom": False,
            "integer_binary_values": True,
        }
    report = {
        "schema_version": 1,
        "status": "valid",
        "primary": "stable",
        "expected_count": spec.prediction.expected_count,
        "submission_manifest": {
            "path": _portable_path(manifest_path, spec.project_root),
            "sha256": sha256_file(manifest_path),
        },
        "outputs": report_outputs,
    }
    atomic_write_json(spec.submissions_dir / "submission_validation_report.json", report)
    return report


def _failure(root: Path, exc: BaseException, *, contract_sha256: str | None) -> None:
    if not root.exists():
        return
    code = (
        exc.exit_code
        if isinstance(exc, OfficialTestError)
        else 130
        if isinstance(exc, KeyboardInterrupt)
        else 4
    )
    atomic_write_json(
        root / "failure.json",
        {
            "schema_version": 1,
            "status": "failed",
            "exit_code": code,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "contract_sha256": contract_sha256,
        },
    )


def run_official_test(
    spec: OfficialTestSpec,
    *,
    resume: bool = False,
    runner: PredictionRunner | None = None,
    selection_validator: SelectionValidator | None = None,
    submission_generator: Callable[..., dict[str, Any]] = generate_submission_candidates,
) -> dict[str, Any]:
    """Run or resume all shards, then atomically publish and validate three JSON files."""

    contract_sha256: str | None = None
    active_root = spec.output_dir if spec.output_dir.exists() else spec.work_dir
    with _RunLock(spec.lock_path, resume=resume):
        active_runner: PredictionRunner | None = None
        try:
            if spec.output_dir.exists() and not resume:
                raise OfficialInputError(
                    "official output exists; use --resume for idempotent validation"
                )
            if spec.work_dir.exists() and not resume:
                raise OfficialInputError(
                    "in-progress state exists; use --resume or cleanup explicitly"
                )
            if spec.submissions_dir.exists() and not resume:
                raise OfficialInputError("submission output exists; refusing to overwrite")
            if not spec.output_dir.exists() and not spec.work_dir.exists():
                spec.work_dir.mkdir(parents=True)
            active_root = spec.output_dir if spec.output_dir.exists() else spec.work_dir
            free = shutil.disk_usage(active_root.parent).free
            if free < spec.minimum_free_bytes:
                raise OfficialRuntimeError(
                    f"insufficient free disk: {free} < {spec.minimum_free_bytes} bytes"
                )
            input_path = active_root / spec.input_manifest_filename
            audit, images = _audit_images(
                spec, existing_manifest=input_path if input_path.is_file() else None
            )
            if not input_path.is_file():
                write_test_input_manifest(input_path, audit)
            input_sha = sha256_file(input_path)
            try:
                selection = load_final_selection_contract(
                    spec.prediction, validator=selection_validator
                )
                inference = _validate_inference_spec(spec.prediction.detector_inference_config)
                thresholds = _selection_thresholds(spec, selection)
            except (TestPredictionError, ConfigError, OSError) as exc:
                raise OfficialContractError(str(exc)) from exc
            git = _git_evidence(spec.project_root)
            if spec.mode == "official" and git["dirty"] is not False:
                raise OfficialContractError("official mode requires a clean execution worktree")
            try:
                active_runner = runner or UltralyticsPredictionRunner(spec.prediction)
            except (OfflinePreparationError, ImportError) as exc:
                raise OfficialContractError(f"offline runtime contract is invalid: {exc}") from exc
            except TestPredictionError as exc:
                raise OfficialRuntimeError(str(exc)) from exc
            environment = _runner_environment(active_runner)
            expected_contract = _contract_payload(
                spec,
                selection,
                input_manifest_sha256=input_sha,
                environment=environment,
                git=git,
            )
            contract_path = active_root / "run_contract.json"
            if contract_path.is_file():
                saved_contract, contract_sha256 = _load_run_contract(contract_path)
                if _canonical_json(saved_contract) != _canonical_json(expected_contract):
                    raise OfficialContractError("--resume contract differs from the frozen run")
            else:
                contract_sha256 = _write_run_contract(contract_path, expected_contract)
            _event(
                active_root / "events.jsonl", "contract_validated", contract_sha256=contract_sha256
            )

            if spec.output_dir.exists():
                _validate_official_inference_manifest(
                    spec=spec,
                    artifact_root=spec.output_dir,
                    contract_sha256=contract_sha256,
                    input_manifest_sha256=input_sha,
                    environment=environment,
                    git=git,
                    selection_validator=selection_validator,
                )
            else:
                try:
                    completed = _run_shards(
                        spec=spec,
                        root=active_root,
                        images=images,
                        selection=selection,
                        contract_sha256=contract_sha256,
                        runner=active_runner,
                        inference=inference,
                    )
                finally:
                    _close_prediction_runner(active_runner)
                _publish_inference(
                    spec=spec,
                    root=active_root,
                    images=images,
                    selection=selection,
                    contract_sha256=contract_sha256,
                    input_manifest_sha256=input_sha,
                    environment=environment,
                    git=git,
                    inference=inference,
                    completed=completed,
                    selection_validator=selection_validator,
                )
                active_root = spec.output_dir

            if spec.submissions_dir.exists():
                if not resume:
                    raise OfficialInputError("submission output exists; refusing to overwrite")
            else:
                try:
                    submission_generator(
                        predictions_path=spec.output_dir / spec.prediction.predictions_filename,
                        inference_manifest=spec.output_dir / spec.prediction.manifest_filename,
                        selection_manifest=spec.prediction.selection_manifest,
                        test_dir=spec.prediction.test_images,
                        output_dir=spec.submissions_dir,
                        expected_count=spec.prediction.expected_count,
                        project_root=spec.project_root,
                    )
                except (OSError, ValueError, TestPredictionError) as exc:
                    raise OfficialValidationError(f"submission generation failed: {exc}") from exc
            report = _validate_submissions(spec, thresholds=thresholds)
            (active_root / "failure.json").unlink(missing_ok=True)
            _event(active_root / "events.jsonl", "run_completed", contract_sha256=contract_sha256)
            return {
                "status": "complete",
                "mode": spec.mode,
                "contract_sha256": contract_sha256,
                "inference_manifest": _portable_path(
                    spec.output_dir / spec.prediction.manifest_filename, spec.project_root
                ),
                "submissions": _portable_path(spec.submissions_dir, spec.project_root),
                "submission_validation": report,
            }
        except KeyboardInterrupt as exc:
            _failure(active_root, exc, contract_sha256=contract_sha256)
            raise
        except OfficialTestError as exc:
            _failure(active_root, exc, contract_sha256=contract_sha256)
            raise
        except BaseException as exc:
            wrapped = OfficialRuntimeError(str(exc))
            _failure(active_root, wrapped, contract_sha256=contract_sha256)
            raise wrapped from exc
        finally:
            if active_runner is not None:
                _close_prediction_runner(active_runner)


__all__ = [
    "OfficialContractError",
    "OfficialInputError",
    "OfficialRuntimeError",
    "OfficialTestError",
    "OfficialTestSpec",
    "OfficialValidationError",
    "load_official_test_spec",
    "run_official_test",
]
