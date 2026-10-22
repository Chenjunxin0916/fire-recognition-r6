"""Transactional builder for normalized YOLO detection/classification data."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps

from fire_detection.data.corrections import (
    AppliedBboxCorrection,
    CorrectionManifest,
    apply_bbox_correction,
    bbox_overflow,
    load_correction_manifest,
)
from fire_detection.data.schema import DatasetIndex, load_dataset_index
from fire_detection.data.similarity import sha256_file


class ProcessedDataError(RuntimeError):
    """Raised when a gate prevents safe derived-data construction."""


@dataclass(frozen=True)
class FixedFoldRow:
    """Validated row from the immutable fold artifact."""

    filename: str
    label: int
    group_id: str
    fold: int


@dataclass(frozen=True)
class ImageBuildRecord:
    """Manifest record for one canonical normalized image."""

    filename: str
    label: int
    fold: int
    source_sha256: str
    processed_sha256: str
    width: int
    height: int
    source_orientation: int | None
    operation: str


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProcessedDataError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProcessedDataError(f"expected JSON object in {path}")
    return value


def require_audit_gate(
    path: Path,
    *,
    images_dir: Path,
    coco_json: Path,
    image_labels_json: Path,
) -> dict[str, Any]:
    """Require a successful audit and prove every audited source byte is unchanged."""

    report = _load_json_object(path)
    if report.get("build_allowed") is not True:
        raise ProcessedDataError(
            f"data audit did not explicitly allow a build: {path}; resolve unknown issues first"
        )
    manifest_name = report.get("checksum_manifest")
    if not isinstance(manifest_name, str) or not manifest_name:
        raise ProcessedDataError("audit report does not identify its checksum manifest")
    manifest_path = path.parent / manifest_name
    expected: dict[tuple[str, str], tuple[int, str]] = {}
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"source_kind", "path", "size_bytes", "sha256"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ProcessedDataError("audit checksum manifest has invalid columns")
            for row in reader:
                kind = row.get("source_kind", "")
                source_path = row.get("path", "")
                key = (kind, source_path)
                if not kind or not source_path or key in expected:
                    raise ProcessedDataError("audit checksum manifest has invalid/duplicate rows")
                expected[key] = (int(row.get("size_bytes", "")), row.get("sha256", ""))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ProcessedDataError(f"cannot verify audit checksum manifest: {exc}") from exc

    actual_paths: dict[tuple[str, str], Path] = {
        ("coco_json", coco_json.name): coco_json,
        ("image_labels_json", image_labels_json.name): image_labels_json,
    }
    if not images_dir.is_dir():
        raise ProcessedDataError(f"source image directory does not exist: {images_dir}")
    for source in sorted(images_dir.iterdir(), key=lambda item: item.name):
        if source.is_file():
            actual_paths[("image", f"images/{source.name}")] = source
    if set(expected) != set(actual_paths):
        raise ProcessedDataError("audited/current source file sets differ")
    changed = [
        source.name
        for key, source in actual_paths.items()
        if expected[key] != (source.stat().st_size, sha256_file(source))
    ]
    if changed:
        raise ProcessedDataError(f"source files changed after audit: {changed[:10]}")
    return report


def load_fixed_folds(path: Path, index: DatasetIndex) -> tuple[FixedFoldRow, ...]:
    """Strictly read the fixed fold CSV and verify one-to-one source coverage."""

    result: list[FixedFoldRow] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"filename", "label", "group_id", "fold"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ProcessedDataError(f"fold CSV must contain {sorted(required)}")
        for csv_row in reader:
            filename = csv_row.get("filename", "")
            if not filename:
                raise ProcessedDataError("fold filename must not be empty")
            if filename in seen:
                raise ProcessedDataError(f"duplicate fold filename: {filename}")
            try:
                label = int(csv_row.get("label", ""))
                fold = int(csv_row.get("fold", ""))
            except ValueError as exc:
                raise ProcessedDataError(f"invalid label/fold for {filename}") from exc
            group_id = csv_row.get("group_id", "")
            if label not in {0, 1} or fold not in range(5) or not group_id:
                raise ProcessedDataError(f"invalid fold row for {filename}: {csv_row}")
            seen.add(filename)
            result.append(FixedFoldRow(filename, label, group_id, fold))
    expected = set(index.labels)
    if seen != expected:
        raise ProcessedDataError(
            "fold/source filename mismatch; "
            f"missing={sorted(expected - seen)[:10]}, extra={sorted(seen - expected)[:10]}"
        )
    for fold_row in result:
        if index.labels[fold_row.filename] != fold_row.label:
            raise ProcessedDataError(f"fold label mismatch for {fold_row.filename}")
    group_folds: dict[str, set[int]] = {}
    for fold_row in result:
        group_folds.setdefault(fold_row.group_id, set()).add(fold_row.fold)
    split = sorted(group_id for group_id, folds in group_folds.items() if len(folds) != 1)
    if split:
        raise ProcessedDataError(f"groups cross folds: {split[:10]}")
    return tuple(sorted(result, key=lambda item: item.filename))


def require_fold_metadata(folds_csv: Path, metadata_path: Path) -> dict[str, Any]:
    """Verify the fold file against its immutable metadata digest and constants."""

    metadata = _load_json_object(metadata_path)
    if metadata.get("status") == "superseded":
        raise ProcessedDataError("fold metadata is superseded")
    if metadata.get("strategy") != "StratifiedGroupKFold":
        raise ProcessedDataError("unexpected fold strategy")
    if metadata.get("n_splits") != 5 or metadata.get("seed") != 42:
        raise ProcessedDataError("fold metadata must declare n_splits=5 and seed=42")
    artifacts = metadata.get("artifact_sha256")
    if not isinstance(artifacts, dict) or artifacts.get("folds.csv") != sha256_file(folds_csv):
        raise ProcessedDataError("folds.csv does not match folds.meta.json")
    expected_groups = artifacts.get("groups.csv")
    if expected_groups is not None:
        groups_csv = folds_csv.with_name("groups.csv")
        if not groups_csv.is_file() or expected_groups != sha256_file(groups_csv):
            raise ProcessedDataError("groups.csv does not match folds.meta.json")
    return metadata


def _link_or_copy(source: Path, destination: Path) -> str:
    """Create a task-view file from an already safe canonical image."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "byte_copy"


def _copy_raw_to_canonical(source: Path, destination: Path) -> str:
    """Copy raw bytes so mutations under processed/ can never alter raw data."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    try:
        if os.path.samefile(source, destination):
            raise ProcessedDataError(f"canonical image aliases raw source: {source}")
    except OSError as exc:
        raise ProcessedDataError(f"cannot verify canonical byte copy for {source}: {exc}") from exc
    return "byte_copy"


def _save_exif_normalized_jpeg(source: Path, destination: Path, expected: int) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        orientation = image.getexif().get(274)
        if orientation != expected:
            raise ProcessedDataError(
                f"{source.name} EXIF orientation changed: expected {expected}, got {orientation}"
            )
        normalized = ImageOps.exif_transpose(image)
        exif = normalized.getexif()
        if 274 in exif:
            del exif[274]
        image_format = image.format
        if image_format is None:
            raise ProcessedDataError(f"source image format is unknown: {source}")
        normalized.format = image_format
        icc_profile = image.info.get("icc_profile")
        try:
            if image_format == "JPEG":
                normalized.save(
                    destination,
                    format=image_format,
                    exif=exif.tobytes(),
                    icc_profile=icc_profile,
                    quality="keep",
                    subsampling="keep",
                    qtables="keep",
                )
            else:
                normalized.save(
                    destination,
                    format=image_format,
                    exif=exif.tobytes(),
                    icc_profile=icc_profile,
                )
        except (OSError, ValueError) as exc:
            raise ProcessedDataError(f"cannot normalize {source}: {exc}") from exc
        return normalized.size


def _build_canonical_images(
    staging: Path,
    images_dir: Path,
    index: DatasetIndex,
    folds: tuple[FixedFoldRow, ...],
    corrections: CorrectionManifest,
) -> tuple[tuple[ImageBuildRecord, ...], dict[str, str]]:
    fold_map = {row.filename: row for row in folds}
    operations: dict[str, str] = {}
    records: list[ImageBuildRecord] = []
    encountered_orientation: set[str] = set()
    for image_record in index.images:
        source = images_dir / image_record.file_name
        destination = staging / "images" / image_record.file_name
        if not source.is_file():
            raise ProcessedDataError(f"source image is missing: {source}")
        try:
            with Image.open(source) as image:
                image.load()
                orientation_raw = image.getexif().get(274)
                orientation = orientation_raw if isinstance(orientation_raw, int) else None
                visual_size = ImageOps.exif_transpose(image).size
        except OSError as exc:
            raise ProcessedDataError(f"cannot decode source image {source}: {exc}") from exc
        if visual_size != (image_record.width, image_record.height):
            raise ProcessedDataError(
                f"visual dimensions for {image_record.file_name} are {visual_size}, "
                f"COCO declares {(image_record.width, image_record.height)}"
            )
        orientation_fix = corrections.orientation.get(image_record.file_name)
        if orientation not in {None, 1} and orientation_fix is None:
            raise ProcessedDataError(
                f"unreviewed EXIF orientation {orientation} on {image_record.file_name}"
            )
        if orientation_fix is not None:
            encountered_orientation.add(image_record.file_name)
            size = _save_exif_normalized_jpeg(
                source, destination, orientation_fix.expected_orientation
            )
            operation = "exif_transpose_reencode"
        else:
            operation = _copy_raw_to_canonical(source, destination)
            size = visual_size
        if size != (image_record.width, image_record.height):
            raise ProcessedDataError(f"processed dimensions are wrong for {image_record.file_name}")
        operations[image_record.file_name] = operation
        row = fold_map[image_record.file_name]
        records.append(
            ImageBuildRecord(
                image_record.file_name,
                row.label,
                row.fold,
                sha256_file(source),
                sha256_file(destination),
                size[0],
                size[1],
                orientation,
                operation,
            )
        )
    missing_orientation = sorted(set(corrections.orientation) - encountered_orientation)
    if missing_orientation:
        raise ProcessedDataError(
            f"orientation manifest entries were not applied: {missing_orientation}"
        )
    return tuple(records), operations


def _validate_and_correct_boxes(
    index: DatasetIndex, corrections: CorrectionManifest
) -> tuple[dict[int, tuple[float, float, float, float]], tuple[AppliedBboxCorrection, ...]]:
    images = index.images_by_id
    annotations_by_image = index.annotations_by_image
    corrected: dict[int, tuple[float, float, float, float]] = {}
    applications: list[AppliedBboxCorrection] = []
    seen_corrections: set[int] = set()
    if {(category.id, category.name) for category in index.categories} != {(0, "fire")}:
        raise ProcessedDataError("derived YOLO data requires the reviewed category mapping 0=fire")
    for image in index.images:
        label = index.labels[image.file_name]
        annotation_count = len(annotations_by_image[image.id])
        if label == 0 and annotation_count:
            raise ProcessedDataError(
                f"negative image {image.file_name} has {annotation_count} fire boxes"
            )
        if label == 1 and not annotation_count:
            raise ProcessedDataError(f"positive image {image.file_name} has no fire box")
    for annotation in index.annotations:
        image = images[annotation.image_id]
        x, y, width, height = annotation.bbox
        if width <= 0.0 or height <= 0.0 or annotation.area <= 0.0:
            raise ProcessedDataError(f"annotation {annotation.id} has non-positive area")
        if annotation.category_id != 0:
            raise ProcessedDataError(
                f"annotation {annotation.id} has unexpected category {annotation.category_id}"
            )
        correction = corrections.bbox.get(annotation.id)
        overflow = bbox_overflow(annotation.bbox, image.width, image.height)
        if correction is not None:
            application = apply_bbox_correction(annotation, image, correction)
            corrected[annotation.id] = application.corrected_bbox
            applications.append(application)
            seen_corrections.add(annotation.id)
        elif overflow > 1e-9:
            raise ProcessedDataError(
                f"annotation {annotation.id} has unreviewed boundary overflow {overflow:.17g} px"
            )
        else:
            corrected[annotation.id] = annotation.bbox
    missing = sorted(set(corrections.bbox) - seen_corrections)
    if missing:
        raise ProcessedDataError(f"bbox manifest entries were not applied: {missing}")
    return corrected, tuple(sorted(applications, key=lambda item: item.annotation_id))


def _write_yolo_labels(
    staging: Path,
    index: DatasetIndex,
    boxes: dict[int, tuple[float, float, float, float]],
) -> None:
    annotations_by_image = index.annotations_by_image
    stems = [Path(image.file_name).stem for image in index.images]
    if len(stems) != len(set(stems)):
        raise ProcessedDataError("image stems must be unique before creating YOLO label filenames")
    for image in index.images:
        lines: list[str] = []
        for annotation in annotations_by_image[image.id]:
            x, y, width, height = boxes[annotation.id]
            center_x = (x + width / 2.0) / image.width
            center_y = (y + height / 2.0) / image.height
            normalized_width = width / image.width
            normalized_height = height / image.height
            lines.append(
                "0 "
                + " ".join(
                    format(value, ".12g")
                    for value in (center_x, center_y, normalized_width, normalized_height)
                )
            )
        label_path = staging / "labels" / f"{Path(image.file_name).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_detection_folds(staging: Path, folds: tuple[FixedFoldRow, ...]) -> None:
    for held_out in range(5):
        fold_dir = staging / "detection" / f"fold_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        training = [row for row in folds if row.fold != held_out]
        validation = [row for row in folds if row.fold == held_out]
        for name, rows in (("train.txt", training), ("val.txt", validation)):
            # Ultralytics resolves only ``./``-prefixed list entries relative to the list file.
            content = "".join(f"./../../images/{row.filename}\n" for row in rows)
            (fold_dir / name).write_text(content, encoding="utf-8", newline="\n")
        dataset_yaml = {
            "train": "train.txt",
            "val": "val.txt",
            "names": {0: "fire"},
        }
        (fold_dir / "data.yaml").write_text(
            yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )


def _write_classification_folds(
    staging: Path, folds: tuple[FixedFoldRow, ...]
) -> tuple[dict[str, object], ...]:
    operations: list[dict[str, object]] = []
    for held_out in range(5):
        for row in folds:
            split = "val" if row.fold == held_out else "train"
            source = staging / "images" / row.filename
            destination = (
                staging
                / "classification"
                / f"fold_{held_out}"
                / split
                / str(row.label)
                / row.filename
            )
            operation = _link_or_copy(source, destination)
            operations.append(
                {
                    "fold": held_out,
                    "split": split,
                    "filename": row.filename,
                    "label": row.label,
                    "operation": operation,
                }
            )
    return tuple(operations)


def _atomic_manifest(path: Path, value: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def build_processed_dataset(
    *,
    images_dir: Path,
    coco_json: Path,
    image_labels_json: Path,
    folds_csv: Path,
    folds_metadata: Path,
    audit_report: Path,
    corrections_yaml: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Build one complete derived dataset in a staging directory, then publish it."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite processed dataset: {output_dir}")
    audit_metadata = require_audit_gate(
        audit_report,
        images_dir=images_dir,
        coco_json=coco_json,
        image_labels_json=image_labels_json,
    )
    index = load_dataset_index(coco_json, image_labels_json)
    folds = load_fixed_folds(folds_csv, index)
    fold_metadata = require_fold_metadata(folds_csv, folds_metadata)
    version_match = re.fullmatch(r"v([1-9]\d*)", output_dir.name)
    requires_provenance = version_match is not None and int(version_match.group(1)) >= 2
    if requires_provenance:
        if (
            fold_metadata.get("fold_version") != output_dir.name
            or fold_metadata.get("status") != "locked"
        ):
            raise ProcessedDataError(
                f"processed {output_dir.name} requires matching active locked Fold metadata"
            )
        provenance = fold_metadata.get("provenance")
        source_hashes = audit_metadata.get("source_hashes")
        manifest_name = audit_metadata.get("checksum_manifest")
        if not isinstance(provenance, dict) or not isinstance(source_hashes, dict):
            raise ProcessedDataError("Fold metadata is missing verified source provenance")
        if not isinstance(manifest_name, str) or not manifest_name:
            raise ProcessedDataError("audit report does not identify checksum_manifest")
        current_provenance = {
            "audit_report": sha256_file(audit_report),
            "checksum_manifest": sha256_file(audit_report.parent / manifest_name),
            "raw_images_tree": source_hashes.get("images_tree"),
            "coco_json": sha256_file(coco_json),
            "image_labels_json": sha256_file(image_labels_json),
        }
        if any(provenance.get(key) != value for key, value in current_provenance.items()):
            raise ProcessedDataError("processed sources do not match locked Fold provenance")
    corrections = load_correction_manifest(corrections_yaml)
    boxes, bbox_applications = _validate_and_correct_boxes(index, corrections)

    staging = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        image_records, _ = _build_canonical_images(staging, images_dir, index, folds, corrections)
        _write_yolo_labels(staging, index, boxes)
        _write_detection_folds(staging, folds)
        classification_operations = _write_classification_folds(staging, folds)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "processed_version": output_dir.name,
            "correction_version": corrections.version,
            "image_count": len(index.images),
            "annotation_count": len(index.annotations),
            "negative_count": sum(label == 0 for label in index.labels.values()),
            "positive_count": sum(label == 1 for label in index.labels.values()),
            "source_sha256": {
                "train_coco.json": sha256_file(coco_json),
                "train_image.json": sha256_file(image_labels_json),
                "audit_report": sha256_file(audit_report),
                "data_corrections.yaml": sha256_file(corrections_yaml),
                "folds.csv": sha256_file(folds_csv),
                "folds.meta.json": sha256_file(folds_metadata),
            },
            "fold_version": fold_metadata.get("fold_version"),
            "images": [record.__dict__ for record in image_records],
            "bbox_corrections": [record.__dict__ for record in bbox_applications],
            "classification_files": list(classification_operations),
        }
        _atomic_manifest(staging / "manifest.json", manifest)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
