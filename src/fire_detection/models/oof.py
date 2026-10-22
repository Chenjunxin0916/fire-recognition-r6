"""Strict OOF export and provenance validation for trained YOLO26 Fold models."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from fire_detection.models.features import RawBox
from fire_detection.models.offline import OfflinePreparationError, sha256_file
from fire_detection.models.training import (
    FoldRecord,
    TrainingConfigurationError,
    TrainingSpec,
    _sha256_training_dataset,
    load_fold_records,
)

BASE_COLUMNS = ("filename", "label", "fold", "group_id")
DETECTOR_BASE_FEATURE_COLUMNS = (
    "det_max_conf",
    "det_top3_mean",
    "det_box_count",
    "det_largest_box_ratio",
    "det_total_box_ratio",
)
DETECTOR_SLICE_FEATURE_COLUMNS = (
    "slice_max_conf",
    "full_slice_max_conf",
    "tile_count",
)
DETECTOR_FEATURE_COLUMNS = (*DETECTOR_BASE_FEATURE_COLUMNS, *DETECTOR_SLICE_FEATURE_COLUMNS)
DETECTOR_INFERENCE_COLUMNS = (
    "inference_config_sha256",
    "inference_imgsz",
    "slice_enabled",
    "slice_size",
    "slice_overlap",
    "slice_stride",
)
TTA_PROVENANCE_COLUMNS = (
    "tta_config_sha256",
    "tta_transform",
    "tta_aggregation",
)
DETECTOR_TTA_FEATURE_COLUMNS = (
    "det_original_full_slice_conf",
    "det_hflip_max_conf",
    "slice_hflip_max_conf",
    "det_hflip_full_slice_conf",
    "det_tta_mean_conf",
)
DETECTOR_TTA_COLUMNS = (
    *DETECTOR_TTA_FEATURE_COLUMNS,
    *TTA_PROVENANCE_COLUMNS,
    "tta_detector_view_reduction",
)
CLASSIFIER_TTA_FEATURE_COLUMNS = (
    "cls_original_probability",
    "cls_hflip_probability",
    "cls_tta_mean_probability",
)
CLASSIFIER_TTA_COLUMNS = (
    *CLASSIFIER_TTA_FEATURE_COLUMNS,
    *TTA_PROVENANCE_COLUMNS,
    "tta_classifier_view_score",
)
MODEL_COLUMNS = (
    "model_name",
    "weights_path",
    "weights_sha256",
    "imgsz",
    "seed",
    "folds_sha256",
    "processed_manifest_sha256",
)
DETECTOR_COLUMNS = (
    *BASE_COLUMNS,
    *DETECTOR_FEATURE_COLUMNS,
    *MODEL_COLUMNS,
    *DETECTOR_INFERENCE_COLUMNS,
    "end2end",
)
CLASSIFIER_COLUMNS = (*BASE_COLUMNS, "cls_probability", *MODEL_COLUMNS)
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
EXPECTED_FOLDS = frozenset(range(5))


def repository_relative(path: Path, root: Path) -> str:
    """Return a portable repository-relative path when possible."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _required_repository_path(path: Path, root: Path, field: str) -> str:
    """Return one repository-relative manifest path or reject an external path."""

    value = repository_relative(path, root)
    portable = PurePosixPath(value.replace("\\", "/"))
    if portable.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in portable.parts:
        raise TrainingConfigurationError(f"{field} must resolve inside the project root: {path}")
    return portable.as_posix()


def _require_manifest_equal(
    manifest: Mapping[str, Any], field: str, expected: Any, manifest_path: Path
) -> None:
    """Require one JSON manifest field to have the exact expected type and value."""

    actual = manifest.get(field)

    def strictly_equal(left: Any, right: Any) -> bool:
        if type(left) is not type(right):
            return False
        if isinstance(right, list):
            return len(left) == len(right) and all(
                strictly_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        return bool(left == right)

    if not strictly_equal(actual, expected):
        raise TrainingConfigurationError(
            f"run manifest {field} mismatch for {manifest_path}: "
            f"expected {expected!r}, found {actual!r}"
        )


def _required_file_sha256(path: Path, field: str) -> str:
    """Hash a required run input and normalize failures to the training error contract."""

    try:
        return sha256_file(path)
    except OfflinePreparationError as exc:
        raise TrainingConfigurationError(f"cannot verify run manifest {field}: {exc}") from exc


def load_run_manifest(spec: TrainingSpec, fold: int) -> tuple[Path, dict[str, Any]]:
    """Load and verify that one fixed-epoch ``last.pt`` belongs to its held-out Fold."""

    if spec.fold != fold:
        raise TrainingConfigurationError(
            f"training spec Fold {spec.fold} does not match requested OOF Fold {fold}"
        )
    if spec.outer_fold is not None:
        raise TrainingConfigurationError("formal five-Fold OOF cannot load a nested miner run")
    run_name = f"{spec.experiment_name}_fold{fold}"
    if spec.run_name != run_name:
        raise TrainingConfigurationError(
            f"OOF requires formal run name {run_name!r}, found {spec.run_name!r}"
        )
    run_dir = spec.output_dir / run_name
    manifest_path = run_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingConfigurationError(
            f"cannot read run manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise TrainingConfigurationError(f"run manifest must be an object: {manifest_path}")

    expected_training_folds = list(spec.training_folds)
    scalar_fields: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "task": spec.task,
        "experiment_name": spec.experiment_name,
        "run_name": run_name,
        "validation_fold": fold,
        "training_folds": expected_training_folds,
        "seed": spec.seed,
        "model_name": spec.model_name,
        "requested_device": spec.requested_device,
        "ultralytics_device": spec.ultralytics_device,
        "imgsz": spec.imgsz,
        "batch": spec.batch,
        "epochs": spec.epochs,
    }
    for field, expected in scalar_fields.items():
        _require_manifest_equal(manifest, field, expected, manifest_path)

    path_fields = {
        "initial_weights": spec.weights_path,
        "config_path": spec.config_path,
        "folds_path": spec.folds_path,
        "processed_manifest": spec.processed_manifest,
        "dataset_path": spec.dataset_path,
    }
    for field, path in path_fields.items():
        expected = _required_repository_path(path, spec.project_root, field)
        _require_manifest_equal(manifest, field, expected, manifest_path)

    hash_fields = {
        "initial_weights_sha256": _required_file_sha256(
            spec.weights_path, "initial_weights_sha256"
        ),
        "config_sha256": _required_file_sha256(spec.config_path, "config_sha256"),
        "folds_sha256": _required_file_sha256(spec.folds_path, "folds_sha256"),
        "processed_manifest_sha256": _required_file_sha256(
            spec.processed_manifest, "processed_manifest_sha256"
        ),
        "dataset_sha256": _sha256_training_dataset(spec),
    }
    for field, expected in hash_fields.items():
        _require_manifest_equal(manifest, field, expected, manifest_path)

    if spec.hard_negative_manifest is not None:
        from fire_detection.models.hard_negative import load_hard_negative_replay

        records = load_fold_records(spec.folds_path, expected_count=1100)
        replay = load_hard_negative_replay(
            path=spec.hard_negative_manifest,
            repeat_factor=spec.hard_negative_repeat_factor,
            validation_fold=fold,
            records=records,
            folds_path=spec.folds_path,
            processed_manifest=spec.processed_manifest,
            project_root=spec.project_root,
            expected_seed=spec.seed,
        )
        _require_manifest_equal(
            manifest,
            "hard_negative_replay",
            replay.manifest_section(spec.project_root),
            manifest_path,
        )
    elif "hard_negative_replay" in manifest:
        raise TrainingConfigurationError(
            f"run manifest has unexpected hard_negative_replay for {manifest_path}"
        )

    if spec.hard_sample_replay_manifest is not None:
        if spec.hard_sample_replay_config is None or spec.hard_sample_replay_variant is None:
            raise TrainingConfigurationError("incomplete Round-3 hard-sample replay configuration")
        from fire_detection.models.round3_replay import load_hard_sample_replay

        records = load_fold_records(spec.folds_path, expected_count=1100)
        hard_sample_replay = load_hard_sample_replay(
            config_path=spec.hard_sample_replay_config,
            manifest_path=spec.hard_sample_replay_manifest,
            variant=spec.hard_sample_replay_variant,
            repeat_factor=spec.hard_sample_repeat_factor,
            validation_fold=fold,
            records=records,
        )
        _require_manifest_equal(
            manifest,
            "hard_sample_replay",
            hard_sample_replay.manifest_section(spec.project_root),
            manifest_path,
        )
    elif "hard_sample_replay" in manifest:
        raise TrainingConfigurationError(
            f"run manifest has unexpected hard_sample_replay for {manifest_path}"
        )

    weights_path = run_dir / "weights" / "last.pt"
    if not weights_path.is_file():
        raise TrainingConfigurationError(f"fixed-epoch last checkpoint is missing: {weights_path}")
    expected_record = _required_repository_path(weights_path, spec.project_root, "last_weights")
    _require_manifest_equal(manifest, "last_weights", expected_record, manifest_path)
    _require_manifest_equal(
        manifest,
        "last_weights_sha256",
        _required_file_sha256(weights_path, "last_weights_sha256"),
        manifest_path,
    )
    return weights_path, manifest


def model_provenance(spec: TrainingSpec, weights_path: Path) -> dict[str, str | int]:
    """Return the repeated provenance columns for one Fold model."""

    return {
        "model_name": spec.model_name,
        "weights_path": repository_relative(weights_path, spec.project_root),
        "weights_sha256": sha256_file(weights_path),
        "imgsz": spec.imgsz,
        "seed": spec.seed,
        "folds_sha256": sha256_file(spec.folds_path),
        "processed_manifest_sha256": sha256_file(spec.processed_manifest),
    }


def raw_box_record(
    *,
    base: Mapping[str, Any],
    boxes: Sequence[RawBox],
    provenance: Mapping[str, Any],
    spec: TrainingSpec,
    inference: Mapping[str, Any] | None = None,
    tiles: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build one JSONL record describing official end-to-end postprocessed boxes."""

    inference_record = (
        dict(inference)
        if inference is not None
        else {
            "conf": spec.inference_conf,
            "max_det": spec.inference_max_det,
            "augment": spec.inference_augment,
            "end2end": True,
            "postprocess": "official_yolo26_topk_conf_filter",
        }
    )
    record = {
        **dict(base),
        **dict(provenance),
        "inference": inference_record,
        "boxes": [box.to_dict() for box in boxes],
    }
    if inference is not None:
        record["tiles"] = [dict(tile) for tile in tiles]
    return record


def _validate_slice_oof_row(row: Mapping[str, Any], filename: str) -> None:
    """Validate the optional second-round sliced-inference OOF extension."""

    for column in ("slice_max_conf", "full_slice_max_conf", "slice_overlap"):
        value = row.get(column)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TrainingConfigurationError(f"{column} is not numeric for {filename}")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise TrainingConfigurationError(f"{column} is invalid for {filename}")
    for column in ("tile_count", "inference_imgsz", "slice_size", "slice_stride"):
        value = row.get(column)
        minimum = 0 if column == "tile_count" else 1
        if type(value) is not int or value < minimum:
            raise TrainingConfigurationError(f"{column} is invalid for {filename}")
    if type(row.get("slice_enabled")) is not bool:
        raise TrainingConfigurationError(f"slice_enabled is invalid for {filename}")
    digest = row.get("inference_config_sha256")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise TrainingConfigurationError(f"inference_config_sha256 is invalid for {filename}")

    slice_score = float(row["slice_max_conf"])
    combined_score = float(row["full_slice_max_conf"])
    expected_combined = max(float(row["det_max_conf"]), slice_score)
    if not math.isclose(combined_score, expected_combined, rel_tol=0.0, abs_tol=1e-12):
        raise TrainingConfigurationError(f"full_slice_max_conf is inconsistent for {filename}")
    enabled = bool(row["slice_enabled"])
    tile_count = int(row["tile_count"])
    slice_size = int(row["slice_size"])
    slice_stride = int(row["slice_stride"])
    overlap = float(row["slice_overlap"])
    if (
        overlap >= 1.0
        or slice_stride > slice_size
        or not math.isclose(
            slice_stride,
            slice_size * (1.0 - overlap),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise TrainingConfigurationError(f"slice stride/overlap is inconsistent for {filename}")
    if enabled and tile_count < 1:
        raise TrainingConfigurationError(f"enabled slicing produced no tiles for {filename}")
    if not enabled and (tile_count != 0 or slice_score != 0.0):
        raise TrainingConfigurationError(f"disabled slicing has tile results for {filename}")


def _validated_probability(row: Mapping[str, Any], column: str, filename: str) -> float:
    value = row.get(column)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigurationError(f"{column} is not numeric for {filename}")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise TrainingConfigurationError(f"{column} is invalid for {filename}")
    return result


def _validate_tta_provenance(row: Mapping[str, Any], filename: str, *, task: str) -> None:
    digest = row.get("tta_config_sha256")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise TrainingConfigurationError(f"tta_config_sha256 is invalid for {filename}")
    if row.get("tta_transform") != "horizontal_flip":
        raise TrainingConfigurationError(f"tta_transform is invalid for {filename}")
    if row.get("tta_aggregation") != "arithmetic_mean":
        raise TrainingConfigurationError(
            f"tta_aggregation must be arithmetic_mean for {filename}; TTA max is forbidden"
        )
    if task == "detect" and row.get("tta_detector_view_reduction") != "max_full_slice":
        raise TrainingConfigurationError(f"tta_detector_view_reduction is invalid for {filename}")
    if task == "classify" and row.get("tta_classifier_view_score") != "positive_class_probability":
        raise TrainingConfigurationError(f"tta_classifier_view_score is invalid for {filename}")


def _validate_detector_tta_row(row: Mapping[str, Any], filename: str) -> None:
    """Validate explicit original/flip detector views and their arithmetic mean."""

    values = {
        column: _validated_probability(row, column, filename)
        for column in DETECTOR_TTA_FEATURE_COLUMNS
    }
    original_expected = max(float(row["det_max_conf"]), float(row["slice_max_conf"]))
    if not math.isclose(
        values["det_original_full_slice_conf"],
        original_expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TrainingConfigurationError(
            f"det_original_full_slice_conf is inconsistent for {filename}"
        )
    flipped_expected = max(values["det_hflip_max_conf"], values["slice_hflip_max_conf"])
    if not math.isclose(
        values["det_hflip_full_slice_conf"],
        flipped_expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TrainingConfigurationError(
            f"det_hflip_full_slice_conf is inconsistent for {filename}"
        )
    mean_expected = math.fsum((original_expected, flipped_expected)) / 2.0
    if not math.isclose(values["det_tta_mean_conf"], mean_expected, rel_tol=0.0, abs_tol=1e-12):
        raise TrainingConfigurationError(
            f"det_tta_mean_conf must be the arithmetic mean for {filename}"
        )
    _validate_tta_provenance(row, filename, task="detect")


def _validate_classifier_tta_row(row: Mapping[str, Any], filename: str) -> None:
    """Validate explicit original/flip classifier views and their arithmetic mean."""

    values = {
        column: _validated_probability(row, column, filename)
        for column in CLASSIFIER_TTA_FEATURE_COLUMNS
    }
    if not math.isclose(
        values["cls_original_probability"],
        float(row["cls_probability"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TrainingConfigurationError(f"cls_original_probability is inconsistent for {filename}")
    mean_expected = (
        math.fsum((values["cls_original_probability"], values["cls_hflip_probability"])) / 2.0
    )
    if not math.isclose(
        values["cls_tta_mean_probability"],
        mean_expected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TrainingConfigurationError(
            f"cls_tta_mean_probability must be the arithmetic mean for {filename}"
        )
    _validate_tta_provenance(row, filename, task="classify")


def validate_oof_rows(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[FoldRecord],
    *,
    task: str,
) -> None:
    """Require one prediction per fixed Fold row with exact label and group provenance."""

    if task not in {"detect", "classify"}:
        raise TrainingConfigurationError(f"unsupported OOF task {task!r}")
    expected = {record.filename: record for record in records}
    if len(rows) != len(expected):
        raise TrainingConfigurationError(
            f"{task} OOF has {len(rows)} rows, expected {len(expected)}"
        )
    seen: set[str] = set()
    score_column = "det_max_conf" if task == "detect" else "cls_probability"
    slice_extension_columns = (*DETECTOR_SLICE_FEATURE_COLUMNS, *DETECTOR_INFERENCE_COLUMNS)
    has_slice_extension = task == "detect" and any(
        any(column in row for column in slice_extension_columns) for row in rows
    )
    tta_columns = DETECTOR_TTA_COLUMNS if task == "detect" else CLASSIFIER_TTA_COLUMNS
    has_tta_extension = any(any(column in row for column in tta_columns) for row in rows)
    if task == "detect" and has_tta_extension and not has_slice_extension:
        raise TrainingConfigurationError(
            "detector horizontal-flip TTA requires the full sliced-inference extension"
        )
    for row in rows:
        filename = row.get("filename")
        if not isinstance(filename, str) or filename not in expected:
            raise TrainingConfigurationError(
                f"{task} OOF contains unexpected filename {filename!r}"
            )
        if filename in seen:
            raise TrainingConfigurationError(f"{task} OOF contains duplicate filename {filename!r}")
        record = expected[filename]
        if row.get("label") != record.label or row.get("fold") != record.fold:
            raise TrainingConfigurationError(f"{task} OOF label/Fold mismatch for {filename}")
        if row.get("group_id") != record.group_id:
            raise TrainingConfigurationError(f"{task} OOF group mismatch for {filename}")
        score = row.get(score_column)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TrainingConfigurationError(f"{task} OOF score is not numeric for {filename}")
        if not 0.0 <= float(score) <= 1.0:
            raise TrainingConfigurationError(f"{task} OOF score is outside [0, 1] for {filename}")
        if task == "detect":
            for column in ("det_top3_mean", "det_largest_box_ratio"):
                value = row.get(column)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TrainingConfigurationError(f"{column} is not numeric for {filename}")
                if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                    raise TrainingConfigurationError(f"{column} is invalid for {filename}")
            count = row.get("det_box_count")
            if type(count) is not int or count < 0:
                raise TrainingConfigurationError(f"det_box_count is invalid for {filename}")
            total_ratio = row.get("det_total_box_ratio")
            if (
                isinstance(total_ratio, bool)
                or not isinstance(total_ratio, (int, float))
                or not math.isfinite(float(total_ratio))
                or float(total_ratio) < 0.0
            ):
                raise TrainingConfigurationError(f"det_total_box_ratio is invalid for {filename}")
            if row.get("end2end") is not True:
                raise TrainingConfigurationError(f"detector OOF is not end-to-end for {filename}")
            if has_slice_extension:
                missing = [column for column in slice_extension_columns if column not in row]
                if missing:
                    raise TrainingConfigurationError(
                        "detector OOF sliced-inference extension is incomplete for "
                        f"{filename}: {missing}"
                    )
                _validate_slice_oof_row(row, filename)
        if has_tta_extension:
            missing_tta = [column for column in tta_columns if column not in row]
            if missing_tta:
                raise TrainingConfigurationError(
                    f"{task} OOF TTA extension is incomplete for {filename}: {missing_tta}"
                )
            if task == "detect":
                _validate_detector_tta_row(row, filename)
            else:
                _validate_classifier_tta_row(row, filename)
        for column in ("model_name", "weights_path"):
            value = row.get(column)
            if not isinstance(value, str) or not value.strip():
                raise TrainingConfigurationError(f"{column} is empty for {filename}")
        raw_weights_path = str(row["weights_path"])
        normalized_weights_path = raw_weights_path.replace("\\", "/")
        portable_weights_path = PurePosixPath(normalized_weights_path)
        if (
            portable_weights_path.is_absolute()
            or PureWindowsPath(raw_weights_path).is_absolute()
            or ".." in portable_weights_path.parts
        ):
            raise TrainingConfigurationError(
                f"weights_path must be repository-relative for {filename}"
            )
        if portable_weights_path.name != "last.pt":
            raise TrainingConfigurationError(
                f"weights_path must identify a last.pt checkpoint for {filename}"
            )
        for column in ("weights_sha256", "folds_sha256", "processed_manifest_sha256"):
            value = row.get(column)
            if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
                raise TrainingConfigurationError(f"{column} is invalid for {filename}")
        if type(row.get("imgsz")) is not int or int(row["imgsz"]) <= 0:
            raise TrainingConfigurationError(f"imgsz is invalid for {filename}")
        if type(row.get("seed")) is not int or int(row["seed"]) < 0:
            raise TrainingConfigurationError(f"seed is invalid for {filename}")
        seen.add(filename)
    missing = sorted(set(expected) - seen)
    if missing:
        raise TrainingConfigurationError(f"{task} OOF is missing images: {missing[:5]}")
    _validate_oof_provenance(rows, task=task, has_tta_extension=has_tta_extension)


def _validate_oof_provenance(
    rows: Sequence[Mapping[str, Any]], *, task: str, has_tta_extension: bool = False
) -> None:
    """Validate global and per-Fold checkpoint invariants after row-level checks."""

    folds = frozenset(int(row["fold"]) for row in rows)
    if folds != EXPECTED_FOLDS:
        raise TrainingConfigurationError(
            f"{task} OOF must contain exactly Fold 0..4, found {sorted(folds)}"
        )
    for column in (
        "model_name",
        "imgsz",
        "seed",
        "folds_sha256",
        "processed_manifest_sha256",
    ):
        if len({row[column] for row in rows}) != 1:
            raise TrainingConfigurationError(
                f"{task} OOF {column} must be identical across all Folds"
            )
    if task == "detect" and "inference_config_sha256" in rows[0]:
        for column in DETECTOR_INFERENCE_COLUMNS:
            if len({row[column] for row in rows}) != 1:
                raise TrainingConfigurationError(
                    f"detector OOF {column} must be identical across all Folds"
                )
    if has_tta_extension:
        task_column = (
            "tta_detector_view_reduction" if task == "detect" else "tta_classifier_view_score"
        )
        for column in (*TTA_PROVENANCE_COLUMNS, task_column):
            if len({row[column] for row in rows}) != 1:
                raise TrainingConfigurationError(
                    f"{task} OOF {column} must be identical across all Folds"
                )

    fold_paths: list[str] = []
    fold_hashes: list[str] = []
    for fold in sorted(EXPECTED_FOLDS):
        fold_rows = [row for row in rows if row["fold"] == fold]
        paths = {str(row["weights_path"]).replace("\\", "/") for row in fold_rows}
        hashes = {str(row["weights_sha256"]).lower() for row in fold_rows}
        if len(paths) != 1 or len(hashes) != 1:
            raise TrainingConfigurationError(
                f"{task} OOF Fold {fold} must identify exactly one last.pt checkpoint"
            )
        fold_paths.append(next(iter(paths)))
        fold_hashes.append(next(iter(hashes)))
    if len(set(fold_paths)) != len(EXPECTED_FOLDS):
        raise TrainingConfigurationError(
            f"{task} OOF must use a different last.pt weights_path for each Fold"
        )
    if len(set(fold_hashes)) != len(EXPECTED_FOLDS):
        raise TrainingConfigurationError(
            f"{task} OOF must use different checkpoint bytes for each Fold"
        )


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """Write a deterministic UTF-8 CSV without partial output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write deterministic JSONL records without partial output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def combine_oof_rows(
    detector_rows: Sequence[Mapping[str, Any]],
    classifier_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Strictly join detector and classifier OOF rows with suffixed model provenance."""

    detector = {str(row["filename"]): row for row in detector_rows}
    classifier = {str(row["filename"]): row for row in classifier_rows}
    if len(detector) != len(detector_rows) or len(classifier) != len(classifier_rows):
        raise TrainingConfigurationError("cannot combine OOF rows with duplicate filenames")
    if set(detector) != set(classifier):
        raise TrainingConfigurationError("detector/classifier OOF filename sets differ")
    rows: list[dict[str, Any]] = []
    for filename in sorted(detector):
        det = detector[filename]
        cls = classifier[filename]
        for column in BASE_COLUMNS:
            if det[column] != cls[column]:
                raise TrainingConfigurationError(
                    f"detector/classifier OOF {column} differs for {filename}"
                )
        for column in ("folds_sha256", "processed_manifest_sha256"):
            if det[column] != cls[column]:
                raise TrainingConfigurationError(
                    f"detector/classifier OOF {column} differs for {filename}"
                )
        row = {column: det[column] for column in BASE_COLUMNS}
        present_detector_features = [column for column in DETECTOR_FEATURE_COLUMNS if column in det]
        row.update({column: det[column] for column in present_detector_features})
        present_detector_tta = [column for column in DETECTOR_TTA_FEATURE_COLUMNS if column in det]
        row.update({column: det[column] for column in present_detector_tta})
        row["cls_probability"] = cls["cls_probability"]
        present_classifier_tta = [
            column for column in CLASSIFIER_TTA_FEATURE_COLUMNS if column in cls
        ]
        row.update({column: cls[column] for column in present_classifier_tta})
        for column in MODEL_COLUMNS:
            row[f"det_{column}"] = det[column]
            row[f"cls_{column}"] = cls[column]
        for column in DETECTOR_INFERENCE_COLUMNS:
            if column in det:
                row[f"det_{column}"] = det[column]
        for column in (*TTA_PROVENANCE_COLUMNS, "tta_detector_view_reduction"):
            if column in det:
                row[f"det_{column}"] = det[column]
        for column in (*TTA_PROVENANCE_COLUMNS, "tta_classifier_view_score"):
            if column in cls:
                row[f"cls_{column}"] = cls[column]
        row["det_end2end"] = det["end2end"]
        rows.append(row)
    detector_feature_columns = tuple(
        column
        for column in DETECTOR_FEATURE_COLUMNS
        if detector_rows and column in detector_rows[0]
    )
    detector_inference = tuple(
        column
        for column in DETECTOR_INFERENCE_COLUMNS
        if detector_rows and column in detector_rows[0]
    )
    detector_tta_features = tuple(
        column
        for column in DETECTOR_TTA_FEATURE_COLUMNS
        if detector_rows and column in detector_rows[0]
    )
    classifier_tta_features = tuple(
        column
        for column in CLASSIFIER_TTA_FEATURE_COLUMNS
        if classifier_rows and column in classifier_rows[0]
    )
    detector_tta_provenance = tuple(
        column
        for column in (*TTA_PROVENANCE_COLUMNS, "tta_detector_view_reduction")
        if detector_rows and column in detector_rows[0]
    )
    classifier_tta_provenance = tuple(
        column
        for column in (*TTA_PROVENANCE_COLUMNS, "tta_classifier_view_score")
        if classifier_rows and column in classifier_rows[0]
    )
    columns = (
        *BASE_COLUMNS,
        *detector_feature_columns,
        *detector_tta_features,
        "cls_probability",
        *classifier_tta_features,
        *(f"det_{column}" for column in MODEL_COLUMNS),
        *(f"det_{column}" for column in detector_inference),
        *(f"det_{column}" for column in detector_tta_provenance),
        *(f"cls_{column}" for column in MODEL_COLUMNS),
        *(f"cls_{column}" for column in classifier_tta_provenance),
        "det_end2end",
    )
    return rows, tuple(columns)
