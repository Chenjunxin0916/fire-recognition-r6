"""Strict OOF prediction validation and detector/classifier joining."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import numpy as np
import pandas as pd

OOFKind = Literal["detector", "classifier"]
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
EXPECTED_FOLDS = frozenset(range(5))

COMMON_COLUMNS = (
    "filename",
    "label",
    "fold",
    "group_id",
    "model_name",
    "weights_path",
    "weights_sha256",
    "imgsz",
    "seed",
    "folds_sha256",
    "processed_manifest_sha256",
)
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
CLASSIFIER_FEATURE_COLUMNS = ("cls_probability",)
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


def canonical_filename(value: object) -> str:
    """Normalize Windows or POSIX input paths to one case-preserving basename."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("filename values must be non-empty strings")
    normalized = value.strip().replace("\\", "/")
    name = PurePosixPath(normalized).name
    if name in {"", ".", ".."}:
        raise ValueError(f"invalid filename: {value!r}")
    return name


def _validate_binary_column(series: pd.Series, *, name: str) -> pd.Series:
    values: list[int] = []
    for index, value in series.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name}[{index}] must be a non-boolean integer")
        integer = int(value)
        if integer not in {0, 1}:
            raise ValueError(f"{name}[{index}] must be 0 or 1")
        values.append(integer)
    return pd.Series(values, index=series.index, dtype="int64")


def _validate_integer_column(series: pd.Series, *, name: str, minimum: int = 0) -> pd.Series:
    values: list[int] = []
    for index, value in series.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name}[{index}] must be a non-boolean integer")
        integer = int(value)
        if integer < minimum:
            raise ValueError(f"{name}[{index}] must be >= {minimum}")
        values.append(integer)
    return pd.Series(values, index=series.index, dtype="int64")


def _validate_probability_column(series: pd.Series, *, name: str) -> pd.Series:
    values: list[float] = []
    for index, value in series.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name}[{index}] must be a non-boolean real number")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name}[{index}] must be finite and in [0, 1]")
        values.append(number)
    return pd.Series(values, index=series.index, dtype="float64")


def _validate_string_column(series: pd.Series, *, name: str) -> pd.Series:
    values: list[str] = []
    for index, value in series.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}[{index}] must be a non-empty string")
        values.append(value.strip())
    return pd.Series(values, index=series.index, dtype="object")


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 digest")
    return value.lower()


def _validate_sha256_column(series: pd.Series, *, name: str) -> pd.Series:
    values = [_validate_sha256(value, name=f"{name}[{index}]") for index, value in series.items()]
    return pd.Series(values, index=series.index, dtype="object")


def _normalize_weights_path(value: object, *, index: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"weights_path[{index}] must be a non-empty string")
    raw = value.strip()
    normalized = raw.replace("\\", "/")
    posix = PurePosixPath(normalized)
    if posix.is_absolute() or PureWindowsPath(raw).is_absolute() or ".." in posix.parts:
        raise ValueError(f"weights_path[{index}] must be a repository-relative path")
    if posix.name != "last.pt":
        raise ValueError(f"weights_path[{index}] must identify a last.pt checkpoint")
    return posix.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot hash provenance file {path}: {exc}") from exc
    return digest.hexdigest()


def _manifest_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"OOF manifest {name} must be a JSON object")
    return value


def _manifest_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"OOF manifest {name} must be an integer >= {minimum}")
    return value


def _validate_manifest_git(value: object) -> dict[str, object]:
    """Validate the Git snapshot required by schema-v3 TTA OOF manifests."""

    record = _manifest_mapping(value, name="git")
    if set(record) != {"commit", "dirty"}:
        raise ValueError("OOF manifest git must contain exactly commit and dirty")
    commit = record.get("commit")
    dirty = record.get("dirty")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("OOF manifest git.commit must be 40 lowercase hexadecimal digits")
    if type(dirty) is not bool:
        raise ValueError("OOF manifest git.dirty must be a boolean")
    return {"commit": commit, "dirty": dirty}


def _resolve_manifest_path(value: object, *, artifact_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"OOF manifest {name}.path must be a non-empty string")
    raw = value.strip()
    portable = PurePosixPath(raw.replace("\\", "/"))
    if portable.is_absolute() or PureWindowsPath(raw).is_absolute() or ".." in portable.parts:
        raise ValueError(f"OOF manifest {name}.path must be repository-relative")
    root = artifact_root.resolve()
    resolved = (root / Path(*portable.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"OOF manifest {name}.path escapes artifact_root") from exc
    return resolved


def _validate_manifest_file(
    value: object,
    *,
    artifact_root: Path,
    name: str,
    expected_path: Path | None = None,
) -> tuple[Path, str, Mapping[str, Any]]:
    record = _manifest_mapping(value, name=name)
    path = _resolve_manifest_path(record.get("path"), artifact_root=artifact_root, name=name)
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"OOF manifest {name}.path does not match the requested file")
    expected_hash = _validate_sha256(record.get("sha256"), name=f"manifest {name}.sha256")
    if not path.is_file():
        raise ValueError(f"OOF manifest {name} file does not exist: {path}")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"OOF manifest {name}.sha256 does not match the actual file")
    return path, actual_hash, record


def _csv_row_count(path: Path) -> int:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            if next(reader, None) is None:
                raise ValueError(f"OOF CSV has no header: {path}")
            return sum(1 for _row in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"cannot count OOF CSV rows in {path}: {exc}") from exc


def _validate_manifest_oof_csv(
    value: object,
    *,
    artifact_root: Path,
    name: str,
    expected_path: Path | None,
    expected_rows: int | None,
) -> tuple[Path, str, int]:
    path, digest, record = _validate_manifest_file(
        value,
        artifact_root=artifact_root,
        name=name,
        expected_path=expected_path,
    )
    rows = _manifest_int(record.get("rows"), name=f"{name}.rows", minimum=1)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"OOF manifest {name}.rows is {rows}, expected {expected_rows}")
    actual_rows = _csv_row_count(path)
    if actual_rows != rows:
        raise ValueError(f"OOF manifest {name}.rows is {rows}, actual CSV has {actual_rows} rows")
    return path, digest, rows


def _validate_manifest_jsonl(
    value: object,
    *,
    artifact_root: Path,
    name: str,
    expected_rows: int | None,
) -> tuple[Path, str, int]:
    path, digest, record = _validate_manifest_file(
        value,
        artifact_root=artifact_root,
        name=name,
    )
    rows = _manifest_int(record.get("rows"), name=f"{name}.rows", minimum=1)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"OOF manifest {name}.rows is {rows}, expected {expected_rows}")
    try:
        with path.open(encoding="utf-8") as handle:
            actual_rows = sum(1 for line in handle if line.strip())
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot count OOF JSONL rows in {path}: {exc}") from exc
    if actual_rows != rows:
        raise ValueError(f"OOF manifest {name}.rows is {rows}, actual JSONL has {actual_rows} rows")
    return path, digest, rows


def validate_oof_manifest(
    manifest_path: Path,
    *,
    artifact_root: Path,
    detector_oof_path: Path | None = None,
    classifier_oof_path: Path | None = None,
    reference_folds_path: Path | None = None,
    processed_manifest_path: Path | None = None,
    expected_rows: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate a canonical ``run_oof --task both`` schema-v2 manifest.

    All recorded input and output hashes are checked against files below
    ``artifact_root``. Optional explicit paths bind a downstream invocation to the
    exact artifacts selected on its command line. The returned digest identifies the
    validated manifest bytes themselves.
    """

    manifest_hash = _sha256_file(manifest_path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read OOF manifest {manifest_path}: {exc}") from exc
    manifest_record = _manifest_mapping(raw, name="root")
    manifest = dict(manifest_record)
    if manifest.get("schema_version") != 2:
        raise ValueError("OOF manifest schema_version must be 2")
    if manifest.get("task") != "both":
        raise ValueError("OOF manifest task must be 'both'")
    if manifest.get("checkpoint_policy") != "last":
        raise ValueError("OOF manifest checkpoint_policy must be 'last'")
    if manifest.get("checkpoint_filename", "last.pt") != "last.pt":
        raise ValueError("OOF manifest checkpoint_filename must be 'last.pt'")

    sections = {
        kind: _manifest_mapping(manifest.get(kind), name=kind)
        for kind in ("detector", "classifier")
    }
    fold_hashes: set[str] = set()
    processed_hashes: set[str] = set()
    row_counts: dict[str, int] = {}
    for kind, section in sections.items():
        _validate_manifest_file(
            section.get("config"),
            artifact_root=artifact_root,
            name=f"{kind}.config",
        )
        _fold_path, fold_hash, _fold_record = _validate_manifest_file(
            section.get("folds"),
            artifact_root=artifact_root,
            name=f"{kind}.folds",
            expected_path=reference_folds_path,
        )
        _processed_path, processed_hash, _processed_record = _validate_manifest_file(
            section.get("processed_manifest"),
            artifact_root=artifact_root,
            name=f"{kind}.processed_manifest",
            expected_path=processed_manifest_path,
        )
        selected_oof_path = detector_oof_path if kind == "detector" else classifier_oof_path
        _oof_path, _oof_hash, rows = _validate_manifest_oof_csv(
            section.get("oof_csv"),
            artifact_root=artifact_root,
            name=f"{kind}.oof_csv",
            expected_path=selected_oof_path,
            expected_rows=expected_rows,
        )
        top_level_rows = _manifest_int(manifest.get(f"{kind}_rows"), name=f"{kind}_rows", minimum=1)
        if top_level_rows != rows:
            raise ValueError(f"OOF manifest {kind}_rows does not match {kind}.oof_csv.rows")
        fold_hashes.add(fold_hash)
        processed_hashes.add(processed_hash)
        row_counts[kind] = rows

    if len(fold_hashes) != 1:
        raise ValueError("detector and classifier OOF manifest Fold hashes differ")
    if len(processed_hashes) != 1:
        raise ValueError("detector and classifier processed manifest hashes differ")
    if row_counts["detector"] != row_counts["classifier"]:
        raise ValueError("detector and classifier OOF manifest row counts differ")

    _combined_path, _combined_hash, combined_rows = _validate_manifest_oof_csv(
        manifest.get("combined_oof"),
        artifact_root=artifact_root,
        name="combined_oof",
        expected_path=None,
        expected_rows=expected_rows,
    )
    if combined_rows != row_counts["detector"]:
        raise ValueError("combined OOF manifest row count differs from task OOF row counts")
    return manifest, manifest_hash


def validate_oof_artifact_manifest(
    manifest_path: Path,
    *,
    artifact_root: Path,
    kind: OOFKind,
    oof_path: Path | None = None,
    reference_folds_path: Path | None = None,
    processed_manifest_path: Path | None = None,
    expected_rows: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate one detector or classifier artifact in an independent OOF manifest.

    Unlike :func:`validate_oof_manifest`, this entry point accepts both standalone
    ``task=detect``/``task=classify`` manifests and the canonical ``task=both``
    manifest. Schema v2 remains compatible with the frozen non-TTA winner. Schema v3
    additionally requires a typed execution Git snapshot and retains its TTA fields.
    """

    manifest_hash = _sha256_file(manifest_path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read OOF manifest {manifest_path}: {exc}") from exc
    manifest_record = _manifest_mapping(raw, name="root")
    manifest = dict(manifest_record)
    schema_version = manifest.get("schema_version")
    if schema_version not in {2, 3}:
        raise ValueError("OOF artifact manifest schema_version must be 2 or 3")
    if schema_version == 3:
        _validate_manifest_git(manifest.get("git"))

    expected_task = "detect" if kind == "detector" else "classify"
    task = manifest.get("task")
    if task not in {expected_task, "both"}:
        raise ValueError(
            f"OOF artifact manifest task must be {expected_task!r} or 'both' for {kind}"
        )
    if manifest.get("checkpoint_policy") != "last":
        raise ValueError("OOF artifact manifest checkpoint_policy must be 'last'")
    if manifest.get("checkpoint_filename", "last.pt") != "last.pt":
        raise ValueError("OOF artifact manifest checkpoint_filename must be 'last.pt'")

    section = _manifest_mapping(manifest.get(kind), name=kind)
    _validate_manifest_file(
        section.get("config"),
        artifact_root=artifact_root,
        name=f"{kind}.config",
    )
    _validate_manifest_file(
        section.get("folds"),
        artifact_root=artifact_root,
        name=f"{kind}.folds",
        expected_path=reference_folds_path,
    )
    _validate_manifest_file(
        section.get("processed_manifest"),
        artifact_root=artifact_root,
        name=f"{kind}.processed_manifest",
        expected_path=processed_manifest_path,
    )
    if "inference_config" in section:
        _validate_manifest_file(
            section["inference_config"],
            artifact_root=artifact_root,
            name=f"{kind}.inference_config",
        )
    section_tta = section.get("tta_config")
    top_level_tta = manifest.get("tta")
    if (section_tta is None) != (top_level_tta is None):
        raise ValueError("OOF manifest TTA config and algorithm contract must be declared together")
    if section_tta is not None:
        _tta_path, tta_hash, _tta_record = _validate_manifest_file(
            section_tta,
            artifact_root=artifact_root,
            name=f"{kind}.tta_config",
        )
        tta = _manifest_mapping(top_level_tta, name="tta")
        if tta.get("transform") != "horizontal_flip":
            raise ValueError("OOF manifest TTA transform must be horizontal_flip")
        if tta.get("aggregation") != "arithmetic_mean":
            raise ValueError("OOF manifest TTA aggregation must be arithmetic_mean")
        if tta.get("views") != ["original", "horizontal_flip"]:
            raise ValueError("OOF manifest TTA views must be [original, horizontal_flip]")
        config = _manifest_mapping(tta.get("config"), name="tta.config")
        if config.get("sha256") != tta_hash:
            raise ValueError("OOF manifest top-level TTA config hash differs from task section")
    _selected_path, _selected_hash, rows = _validate_manifest_oof_csv(
        section.get("oof_csv"),
        artifact_root=artifact_root,
        name=f"{kind}.oof_csv",
        expected_path=oof_path,
        expected_rows=expected_rows,
    )
    top_level_rows = _manifest_int(
        manifest.get(f"{kind}_rows"),
        name=f"{kind}_rows",
        minimum=1,
    )
    if top_level_rows != rows:
        raise ValueError(f"OOF artifact manifest {kind}_rows does not match {kind}.oof_csv.rows")
    if "raw_boxes_jsonl" in section:
        _validate_manifest_jsonl(
            section["raw_boxes_jsonl"],
            artifact_root=artifact_root,
            name=f"{kind}.raw_boxes_jsonl",
            expected_rows=rows,
        )
    return manifest, manifest_hash


def _require_single_value(frame: pd.DataFrame, column: str) -> object:
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"{column} must be identical across the complete OOF table")
    return values[0]


def _validate_fold_provenance(frame: pd.DataFrame) -> None:
    folds = frozenset(int(value) for value in frame["fold"].tolist())
    if folds != EXPECTED_FOLDS:
        raise ValueError(f"OOF must contain exactly Fold 0..4, found {sorted(folds)}")

    for column in (
        "model_name",
        "imgsz",
        "seed",
        "folds_sha256",
        "processed_manifest_sha256",
    ):
        _require_single_value(frame, column)
    if "inference_config_sha256" in frame.columns:
        for column in DETECTOR_INFERENCE_COLUMNS:
            _require_single_value(frame, column)
    if "tta_config_sha256" in frame.columns:
        task_columns = [*TTA_PROVENANCE_COLUMNS]
        if "tta_detector_view_reduction" in frame.columns:
            task_columns.append("tta_detector_view_reduction")
        if "tta_classifier_view_score" in frame.columns:
            task_columns.append("tta_classifier_view_score")
        for column in task_columns:
            _require_single_value(frame, column)

    weights_paths: list[str] = []
    weights_hashes: list[str] = []
    for fold in sorted(EXPECTED_FOLDS):
        fold_rows = frame.loc[frame["fold"] == fold]
        path = _require_single_value(fold_rows, "weights_path")
        digest = _require_single_value(fold_rows, "weights_sha256")
        weights_paths.append(str(path))
        weights_hashes.append(str(digest))
    if len(set(weights_paths)) != len(EXPECTED_FOLDS):
        raise ValueError("each Fold must use a different last.pt weights_path")
    if len(set(weights_hashes)) != len(EXPECTED_FOLDS):
        raise ValueError("each Fold must use a different last.pt weights_sha256")


def _validate_reference_folds(reference_folds: pd.DataFrame) -> pd.DataFrame:
    reference_columns = ["filename", "label", "fold", "group_id"]
    missing = [column for column in reference_columns if column not in reference_folds.columns]
    if missing:
        raise ValueError(f"reference folds missing columns: {missing}")
    reference = reference_folds[reference_columns].copy()
    reference["filename"] = [canonical_filename(value) for value in reference["filename"]]
    if reference["filename"].duplicated().any():
        raise ValueError("reference folds contain duplicate filenames")
    reference["label"] = _validate_binary_column(reference["label"], name="reference.label")
    reference["fold"] = _validate_integer_column(reference["fold"], name="reference.fold")
    reference["group_id"] = _validate_string_column(
        reference["group_id"], name="reference.group_id"
    )
    return reference


def _compare_reference_folds(frame: pd.DataFrame, reference_folds: pd.DataFrame) -> None:
    reference_columns = ["filename", "label", "fold", "group_id"]
    reference = _validate_reference_folds(reference_folds)
    joined = frame[reference_columns].merge(
        reference,
        on="filename",
        how="outer",
        suffixes=("_oof", "_reference"),
        indicator=True,
        validate="one_to_one",
    )
    if (joined["_merge"] != "both").any():
        raise ValueError("OOF filename set does not match the fixed Fold file")
    for column in ("label", "fold", "group_id"):
        if not (joined[f"{column}_oof"] == joined[f"{column}_reference"]).all():
            raise ValueError(f"OOF {column} values do not match the fixed Fold file")


def _verify_weight_artifacts(frame: pd.DataFrame, artifact_root: Path) -> None:
    root = artifact_root.resolve()
    for weights_path, expected_hash in (
        frame[["weights_path", "weights_sha256"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        relative = Path(*PurePosixPath(str(weights_path)).parts)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"weights_path escapes artifact_root: {weights_path}") from exc
        if not candidate.is_file():
            raise ValueError(f"OOF last.pt artifact does not exist: {candidate}")
        actual_hash = _sha256_file(candidate)
        if actual_hash != expected_hash:
            raise ValueError(
                f"OOF weights_sha256 does not match artifact {weights_path}: "
                f"expected {expected_hash}, found {actual_hash}"
            )


def validate_oof_frame(
    frame: pd.DataFrame,
    *,
    kind: OOFKind,
    expected_rows: int | None = None,
    reference_folds: pd.DataFrame | None = None,
    reference_folds_path: Path | None = None,
    expected_folds_sha256: str | None = None,
    expected_processed_manifest_sha256: str | None = None,
    artifact_root: Path | None = None,
) -> pd.DataFrame:
    """Validate one strict five-Fold OOF table and return a normalized copy.

    ``reference_folds_path`` is preferred over an already loaded frame because it binds
    the predictions to the actual Fold-file bytes. When ``artifact_root`` is supplied,
    every repository-relative ``weights_path`` is resolved below that root and hashed.
    """

    if reference_folds is not None and reference_folds_path is not None:
        raise ValueError("pass either reference_folds or reference_folds_path, not both")

    slice_extension = False
    tta_extension = False
    feature_columns: tuple[str, ...]
    tta_feature_columns: tuple[str, ...]
    tta_columns: tuple[str, ...]
    if kind == "detector":
        optional_columns = (*DETECTOR_SLICE_FEATURE_COLUMNS, *DETECTOR_INFERENCE_COLUMNS)
        slice_extension = any(column in frame.columns for column in optional_columns)
        if slice_extension:
            missing_extension = [
                column for column in optional_columns if column not in frame.columns
            ]
            if missing_extension:
                raise ValueError(
                    f"detector sliced-inference OOF extension is incomplete: {missing_extension}"
                )
        feature_columns = (
            DETECTOR_FEATURE_COLUMNS if slice_extension else DETECTOR_BASE_FEATURE_COLUMNS
        )
        tta_feature_columns = DETECTOR_TTA_FEATURE_COLUMNS
        tta_columns = DETECTOR_TTA_COLUMNS
    else:
        feature_columns = CLASSIFIER_FEATURE_COLUMNS
        tta_feature_columns = CLASSIFIER_TTA_FEATURE_COLUMNS
        tta_columns = CLASSIFIER_TTA_COLUMNS
    tta_extension = any(column in frame.columns for column in tta_columns)
    if tta_extension:
        missing_tta = [column for column in tta_columns if column not in frame.columns]
        if missing_tta:
            raise ValueError(f"{kind} TTA OOF extension is incomplete: {missing_tta}")
        if kind == "detector" and not slice_extension:
            raise ValueError(
                "detector horizontal-flip TTA requires the full sliced-inference extension"
            )
    required = (
        *COMMON_COLUMNS,
        *feature_columns,
        *(tta_columns if tta_extension else ()),
        *(DETECTOR_INFERENCE_COLUMNS if slice_extension else ()),
        *(("end2end",) if kind == "detector" else ()),
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required {kind} OOF columns: {missing}")
    if frame.empty:
        raise ValueError("OOF predictions must not be empty")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} OOF rows, found {len(frame)}")

    normalized = frame.copy()
    normalized["filename"] = [canonical_filename(value) for value in normalized["filename"]]
    if normalized["filename"].duplicated().any():
        duplicates = normalized.loc[normalized["filename"].duplicated(False), "filename"].tolist()
        raise ValueError(f"duplicate OOF filenames after normalization: {duplicates[:5]}")
    normalized["label"] = _validate_binary_column(normalized["label"], name="label")
    normalized["fold"] = _validate_integer_column(normalized["fold"], name="fold")
    normalized["imgsz"] = _validate_integer_column(normalized["imgsz"], name="imgsz", minimum=1)
    normalized["seed"] = _validate_integer_column(normalized["seed"], name="seed")
    for column in feature_columns:
        if column in {"det_box_count", "tile_count"}:
            normalized[column] = _validate_integer_column(normalized[column], name=column)
        elif column == "det_total_box_ratio":
            values = pd.to_numeric(normalized[column], errors="raise").astype(float)
            if (~np.isfinite(values)).any() or (values < 0).any():
                raise ValueError(f"{column} must contain finite non-negative values")
            normalized[column] = values
        else:
            normalized[column] = _validate_probability_column(normalized[column], name=column)
    if tta_extension:
        for column in tta_feature_columns:
            normalized[column] = _validate_probability_column(normalized[column], name=column)
        normalized["tta_config_sha256"] = _validate_sha256_column(
            normalized["tta_config_sha256"], name="tta_config_sha256"
        )
        for column in ("tta_transform", "tta_aggregation"):
            normalized[column] = _validate_string_column(normalized[column], name=column)
        if (normalized["tta_transform"] != "horizontal_flip").any():
            raise ValueError("tta_transform must be horizontal_flip")
        if (normalized["tta_aggregation"] != "arithmetic_mean").any():
            raise ValueError("tta_aggregation must be arithmetic_mean; TTA max is forbidden")
        if kind == "detector":
            normalized["tta_detector_view_reduction"] = _validate_string_column(
                normalized["tta_detector_view_reduction"],
                name="tta_detector_view_reduction",
            )
            if (normalized["tta_detector_view_reduction"] != "max_full_slice").any():
                raise ValueError("tta_detector_view_reduction must be max_full_slice")
        else:
            normalized["tta_classifier_view_score"] = _validate_string_column(
                normalized["tta_classifier_view_score"],
                name="tta_classifier_view_score",
            )
            if (normalized["tta_classifier_view_score"] != "positive_class_probability").any():
                raise ValueError("tta_classifier_view_score must be positive_class_probability")

    for column in ("group_id", "model_name"):
        normalized[column] = _validate_string_column(normalized[column], name=column)
    normalized["weights_path"] = pd.Series(
        [
            _normalize_weights_path(value, index=index)
            for index, value in normalized["weights_path"].items()
        ],
        index=normalized.index,
        dtype="object",
    )
    for column in ("weights_sha256", "folds_sha256", "processed_manifest_sha256"):
        normalized[column] = _validate_sha256_column(normalized[column], name=column)
    if kind == "detector":
        for index, value in normalized["end2end"].items():
            if not isinstance(value, (bool, np.bool_)) or not bool(value):
                raise ValueError(f"end2end[{index}] must be true for detector OOF")
        normalized["end2end"] = True
        if slice_extension:
            normalized["inference_config_sha256"] = _validate_sha256_column(
                normalized["inference_config_sha256"],
                name="inference_config_sha256",
            )
            for column in ("inference_imgsz", "slice_size", "slice_stride"):
                normalized[column] = _validate_integer_column(
                    normalized[column], name=column, minimum=1
                )
            normalized["slice_overlap"] = _validate_probability_column(
                normalized["slice_overlap"], name="slice_overlap"
            )
            if (normalized["slice_overlap"] >= 1.0).any():
                raise ValueError("slice_overlap must be finite and in [0, 1)")
            expected_stride = normalized["slice_size"] * (1.0 - normalized["slice_overlap"])
            if (normalized["slice_stride"] > normalized["slice_size"]).any() or not np.allclose(
                normalized["slice_stride"].to_numpy(),
                expected_stride.to_numpy(),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("slice_stride must equal slice_size * (1 - slice_overlap)")
            enabled_values: list[bool] = []
            for index, value in normalized["slice_enabled"].items():
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError(f"slice_enabled[{index}] must be a boolean")
                enabled_values.append(bool(value))
            normalized["slice_enabled"] = pd.Series(
                enabled_values, index=normalized.index, dtype="bool"
            )
            expected_score = np.maximum(
                normalized["det_max_conf"].to_numpy(),
                normalized["slice_max_conf"].to_numpy(),
            )
            if not np.allclose(
                normalized["full_slice_max_conf"].to_numpy(),
                expected_score,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("full_slice_max_conf must equal max(det_max_conf, slice_max_conf)")
            enabled = normalized["slice_enabled"]
            if (normalized.loc[enabled, "tile_count"] < 1).any():
                raise ValueError("enabled sliced inference must produce at least one tile")
            disabled = ~enabled
            if (normalized.loc[disabled, "tile_count"] != 0).any() or (
                normalized.loc[disabled, "slice_max_conf"] != 0.0
            ).any():
                raise ValueError("disabled sliced inference must not contain tile predictions")
    if tta_extension:
        if kind == "detector":
            original = np.maximum(
                normalized["det_max_conf"].to_numpy(),
                normalized["slice_max_conf"].to_numpy(),
            )
            flipped = np.maximum(
                normalized["det_hflip_max_conf"].to_numpy(),
                normalized["slice_hflip_max_conf"].to_numpy(),
            )
            if not np.allclose(
                normalized["det_original_full_slice_conf"].to_numpy(),
                original,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "det_original_full_slice_conf must equal max(original full, slice)"
                )
            if not np.allclose(
                normalized["det_hflip_full_slice_conf"].to_numpy(),
                flipped,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "det_hflip_full_slice_conf must equal max(horizontal-flip full, slice)"
                )
            expected_mean = (original + flipped) / 2.0
            if not np.allclose(
                normalized["det_tta_mean_conf"].to_numpy(),
                expected_mean,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("det_tta_mean_conf must be the arithmetic view mean")
        else:
            if not np.allclose(
                normalized["cls_original_probability"].to_numpy(),
                normalized["cls_probability"].to_numpy(),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("cls_original_probability must equal cls_probability")
            expected_mean = (
                normalized["cls_original_probability"].to_numpy()
                + normalized["cls_hflip_probability"].to_numpy()
            ) / 2.0
            if not np.allclose(
                normalized["cls_tta_mean_probability"].to_numpy(),
                expected_mean,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("cls_tta_mean_probability must be the arithmetic view mean")

    _validate_fold_provenance(normalized)

    actual_folds_hash: str | None = None
    if reference_folds_path is not None:
        actual_folds_hash = _sha256_file(reference_folds_path)
        try:
            reference_folds = pd.read_csv(reference_folds_path)
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise ValueError(
                f"cannot read reference Fold CSV {reference_folds_path}: {exc}"
            ) from exc

    if expected_folds_sha256 is not None:
        expected_folds_sha256 = _validate_sha256(
            expected_folds_sha256, name="expected_folds_sha256"
        )
        if actual_folds_hash is not None and actual_folds_hash != expected_folds_sha256:
            raise ValueError(
                "actual reference Fold file SHA-256 does not match expected_folds_sha256"
            )
    bound_folds_hash = actual_folds_hash or expected_folds_sha256
    if bound_folds_hash is not None:
        row_folds_hash = str(_require_single_value(normalized, "folds_sha256"))
        if row_folds_hash != bound_folds_hash:
            raise ValueError("OOF folds_sha256 does not match the fixed Fold file")

    if expected_processed_manifest_sha256 is not None:
        expected_processed_manifest_sha256 = _validate_sha256(
            expected_processed_manifest_sha256,
            name="expected_processed_manifest_sha256",
        )
        row_processed_hash = str(_require_single_value(normalized, "processed_manifest_sha256"))
        if row_processed_hash != expected_processed_manifest_sha256:
            raise ValueError("OOF processed_manifest_sha256 does not match the expected manifest")

    if reference_folds is not None:
        _compare_reference_folds(normalized, reference_folds)

    if artifact_root is not None:
        _verify_weight_artifacts(normalized, artifact_root)

    return normalized.sort_values("filename", kind="stable").reset_index(drop=True)


def load_oof_csv(
    path: Path,
    *,
    kind: OOFKind,
    expected_rows: int | None = None,
    reference_folds: pd.DataFrame | None = None,
    reference_folds_path: Path | None = None,
    expected_folds_sha256: str | None = None,
    expected_processed_manifest_sha256: str | None = None,
    artifact_root: Path | None = None,
) -> pd.DataFrame:
    """Load and validate an OOF CSV."""

    return validate_oof_frame(
        pd.read_csv(path),
        kind=kind,
        expected_rows=expected_rows,
        reference_folds=reference_folds,
        reference_folds_path=reference_folds_path,
        expected_folds_sha256=expected_folds_sha256,
        expected_processed_manifest_sha256=expected_processed_manifest_sha256,
        artifact_root=artifact_root,
    )


def combine_oof_frames(
    detector: pd.DataFrame,
    classifier: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    reference_folds: pd.DataFrame | None = None,
    reference_folds_path: Path | None = None,
    expected_folds_sha256: str | None = None,
    expected_processed_manifest_sha256: str | None = None,
    artifact_root: Path | None = None,
) -> pd.DataFrame:
    """Strictly join detector and classifier OOF tables one-to-one."""

    det = validate_oof_frame(
        detector,
        kind="detector",
        expected_rows=expected_rows,
        reference_folds=reference_folds,
        reference_folds_path=reference_folds_path,
        expected_folds_sha256=expected_folds_sha256,
        expected_processed_manifest_sha256=expected_processed_manifest_sha256,
        artifact_root=artifact_root,
    )
    cls = validate_oof_frame(
        classifier,
        kind="classifier",
        expected_rows=expected_rows,
        reference_folds=reference_folds,
        reference_folds_path=reference_folds_path,
        expected_folds_sha256=expected_folds_sha256,
        expected_processed_manifest_sha256=expected_processed_manifest_sha256,
        artifact_root=artifact_root,
    )
    for column in ("folds_sha256", "processed_manifest_sha256"):
        if det[column].iat[0] != cls[column].iat[0]:
            raise ValueError(f"detector and classifier OOF {column} values do not match")
    keys = ["filename", "label", "fold", "group_id"]
    det_provenance = [
        column
        for column in det.columns
        if column not in {*keys, *DETECTOR_FEATURE_COLUMNS, *DETECTOR_TTA_FEATURE_COLUMNS}
    ]
    cls_provenance = [
        column
        for column in cls.columns
        if column not in {*keys, *CLASSIFIER_FEATURE_COLUMNS, *CLASSIFIER_TTA_FEATURE_COLUMNS}
    ]
    det = det.rename(columns={column: f"det_{column}" for column in det_provenance})
    cls = cls.rename(columns={column: f"cls_{column}" for column in cls_provenance})
    combined = det.merge(cls, on=keys, how="outer", indicator=True, validate="one_to_one")
    if (combined["_merge"] != "both").any():
        raise ValueError("detector and classifier OOF filename sets do not match")
    return (
        combined.drop(columns="_merge")
        .sort_values("filename", kind="stable")
        .reset_index(drop=True)
    )
