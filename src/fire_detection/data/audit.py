"""Read-only dataset auditing and reproducible report generation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from PIL import Image, ImageOps, UnidentifiedImageError

from fire_detection.data.corrections import (
    CorrectionManifest,
    load_correction_manifest,
)
from fire_detection.data.schema import (
    AnnotationRecord,
    DatasetIndex,
    DataValidationError,
    load_dataset_index,
)

MACHINE_EPSILON_PX = 1e-9
FLOAT_BOUNDARY_TOLERANCE_PX = 1e-3
SMALL_BOX_AREA_RATIO = 1e-3
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class AuditIssue:
    """One structured finding that is never silently repaired."""

    code: str
    severity: str
    message: str
    filename: str | None = None
    image_id: int | None = None
    annotation_id: int | None = None
    acknowledged: bool = False
    details: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "filename": self.filename,
            "image_id": self.image_id,
            "annotation_id": self.annotation_id,
            "acknowledged": self.acknowledged,
            "details": self.details or {},
        }


@dataclass(frozen=True)
class AuditReport:
    """Complete audit result and downstream build gate."""

    build_allowed: bool
    summary: dict[str, object]
    issues: tuple[AuditIssue, ...]
    source_hashes: dict[str, str]
    checksum_manifest: str
    correction_manifest_version: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "build_allowed": self.build_allowed,
            "summary": self.summary,
            "issues": [issue.to_dict() for issue in self.issues],
            "source_hashes": self.source_hashes,
            "checksum_manifest": self.checksum_manifest,
            "correction_manifest_version": self.correction_manifest_version,
        }


@dataclass(frozen=True)
class _ChecksumRow:
    source_kind: str
    path: str
    size_bytes: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_rows(
    images_dir: Path,
    coco_json: Path,
    image_labels_json: Path,
) -> tuple[list[_ChecksumRow], list[Path]]:
    rows: list[_ChecksumRow] = []
    for kind, path in (("coco_json", coco_json), ("image_labels_json", image_labels_json)):
        if path.is_file():
            rows.append(_ChecksumRow(kind, path.name, path.stat().st_size, _sha256(path)))

    files = (
        sorted(
            (path for path in images_dir.iterdir() if path.is_file()),
            key=lambda path: path.name,
        )
        if images_dir.is_dir()
        else []
    )
    for path in files:
        rows.append(
            _ChecksumRow("image", f"images/{path.name}", path.stat().st_size, _sha256(path))
        )
    return rows, files


def _validate_unique_json_keys(path: Path) -> None:
    class DuplicateKeyError(ValueError):
        pass

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as exc:
        raise DataValidationError(f"cannot strictly parse JSON {path}: {exc}") from exc


def _write_checksum_manifest(path: Path, rows: list[_ChecksumRow]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=("source_kind", "path", "size_bytes", "sha256"),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "source_kind": row.source_kind,
                "path": row.path,
                "size_bytes": row.size_bytes,
                "sha256": row.sha256,
            }
        )
    _atomic_write_text(path, buffer.getvalue())


def _source_hashes(rows: list[_ChecksumRow]) -> dict[str, str]:
    hashes = {
        row.source_kind: row.sha256
        for row in rows
        if row.source_kind in {"coco_json", "image_labels_json"}
    }
    tree = hashlib.sha256()
    for row in rows:
        if row.source_kind == "image":
            tree.update(f"{row.path}\t{row.size_bytes}\t{row.sha256}\n".encode())
    hashes["images_tree"] = tree.hexdigest()
    return hashes


def _bbox_overflow(annotation: AnnotationRecord, width: int, height: int) -> float:
    x, y, box_width, box_height = annotation.bbox
    return max(0.0, -x, -y, x + box_width - width, y + box_height - height)


def _correction_matches(
    annotation: AnnotationRecord,
    corrections: CorrectionManifest | None,
) -> tuple[bool, dict[str, object]]:
    if corrections is None or annotation.id not in corrections.bbox:
        return False, {}
    correction = corrections.bbox[annotation.id]
    matches = all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(annotation.bbox, correction.expected_bbox, strict=True)
    )
    return matches, {
        "correction_action": correction.action,
        "correction_reason": correction.reason,
        "correction_expected_bbox": list(correction.expected_bbox),
    }


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _area_distribution(values: list[float]) -> dict[str, object]:
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "p25": _quantile(values, 0.25),
        "median": median(values) if values else None,
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "p99": _quantile(values, 0.99),
        "maximum": max(values) if values else None,
        "below_0.1_percent": sum(value < SMALL_BOX_AREA_RATIO for value in values),
    }


def _issue_sort_key(issue: AuditIssue) -> tuple[object, ...]:
    rank = {"error": 0, "warning": 1, "info": 2}
    return (
        rank.get(issue.severity, 3),
        issue.code,
        issue.filename or "",
        issue.annotation_id if issue.annotation_id is not None else -1,
    )


def _write_issues(path: Path, issues: tuple[AuditIssue, ...]) -> None:
    buffer = io.StringIO(newline="")
    fields = (
        "severity",
        "code",
        "acknowledged",
        "filename",
        "image_id",
        "annotation_id",
        "message",
        "details",
    )
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for issue in issues:
        row = issue.to_dict()
        row["details"] = json.dumps(issue.details or {}, ensure_ascii=False, sort_keys=True)
        writer.writerow({field: row[field] for field in fields})
    _atomic_write_text(path, buffer.getvalue())


def _summary_markdown(report: AuditReport) -> str:
    summary = report.summary
    status = "PASS" if report.build_allowed else "BLOCKED"
    raw_severity = summary.get("issues_by_severity")
    severity = raw_severity if isinstance(raw_severity, dict) else {}
    return "\n".join(
        (
            "# Dataset audit summary",
            "",
            f"- Build gate: **{status}**",
            f"- Files on disk: {summary.get('image_files', 0)}",
            f"- COCO images: {summary.get('coco_images', 0)}",
            f"- Image labels: {summary.get('image_labels', 0)}",
            f"- Negative / positive: {summary.get('negative_images', 0)} / "
            f"{summary.get('positive_images', 0)}",
            f"- Annotations: {summary.get('annotations', 0)}",
            f"- Small boxes (<0.1%): {summary.get('small_box_annotations', 0)}",
            f"- Issues (error / warning / info): "
            f"{severity.get('error', 0)} / {severity.get('warning', 0)} / "
            f"{severity.get('info', 0)}",
            "",
            "All findings remain in `issues.csv`; acknowledged corrections are applied only "
            "when derived data is built.",
            "",
        )
    )


def _write_report(output_dir: Path, report: AuditReport) -> None:
    issues = tuple(sorted(report.issues, key=_issue_sort_key))
    normalized = AuditReport(
        report.build_allowed,
        report.summary,
        issues,
        report.source_hashes,
        report.checksum_manifest,
        report.correction_manifest_version,
    )
    _atomic_write_text(
        output_dir / "report.json",
        json.dumps(normalized.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_issues(output_dir / "issues.csv", issues)
    _atomic_write_text(output_dir / "summary.md", _summary_markdown(normalized))


def _schema_failure_report(
    output_dir: Path,
    rows: list[_ChecksumRow],
    actual_files: list[Path],
    message: str,
) -> AuditReport:
    issue = AuditIssue("dataset_schema_invalid", "error", message)
    summary: dict[str, object] = {
        "image_files": sum(path.suffix.lower() in _IMAGE_SUFFIXES for path in actual_files),
        "coco_images": 0,
        "image_labels": 0,
        "negative_images": 0,
        "positive_images": 0,
        "annotations": 0,
        "small_box_annotations": 0,
        "issues_by_severity": {"error": 1, "warning": 0, "info": 0},
        "issues_by_code": {issue.code: 1},
    }
    report = AuditReport(
        False,
        summary,
        (issue,),
        _source_hashes(rows),
        "sha256_manifest.csv",
        None,
    )
    _write_report(output_dir, report)
    return report


def audit_dataset(
    images_dir: Path,
    coco_json: Path,
    image_labels_json: Path,
    output_dir: Path,
    *,
    corrections_path: Path | None = None,
    expected_image_count: int | None = None,
    expected_category_id: int = 0,
    expected_category_name: str = "fire",
) -> AuditReport:
    """Audit source data without modifying it and write deterministic artifacts."""

    images_dir = Path(images_dir)
    coco_json = Path(coco_json)
    image_labels_json = Path(image_labels_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, actual_files = _manifest_rows(images_dir, coco_json, image_labels_json)
    _write_checksum_manifest(output_dir / "sha256_manifest.csv", rows)

    try:
        _validate_unique_json_keys(coco_json)
        _validate_unique_json_keys(image_labels_json)
        index = load_dataset_index(coco_json, image_labels_json)
    except DataValidationError as exc:
        return _schema_failure_report(output_dir, rows, actual_files, str(exc))

    corrections: CorrectionManifest | None = None
    if corrections_path is not None:
        try:
            corrections = load_correction_manifest(Path(corrections_path))
        except DataValidationError as exc:
            return _schema_failure_report(
                output_dir,
                rows,
                actual_files,
                f"invalid correction manifest: {exc}",
            )

    issues: list[AuditIssue] = []
    actual_images = {
        path.name: path for path in actual_files if path.suffix.lower() in _IMAGE_SUFFIXES
    }
    unsupported = [path.name for path in actual_files if path.suffix.lower() not in _IMAGE_SUFFIXES]
    for filename in unsupported:
        issues.append(
            AuditIssue(
                "unsupported_file",
                "warning",
                "Non-image file is present in the image directory.",
                filename=filename,
            )
        )

    casefolded: dict[str, str] = {}
    for filename in sorted(actual_images):
        key = filename.casefold()
        if key in casefolded and casefolded[key] != filename:
            issues.append(
                AuditIssue(
                    "filename_case_collision",
                    "error",
                    f"Filename collides case-insensitively with {casefolded[key]!r}.",
                    filename=filename,
                )
            )
        casefolded[key] = filename

    metadata_names = set(index.images_by_name)
    disk_names = set(actual_images)
    for filename in sorted(metadata_names - disk_names):
        image = index.images_by_name[filename]
        issues.append(
            AuditIssue(
                "image_missing_on_disk",
                "error",
                "COCO image is missing from the image directory.",
                filename=filename,
                image_id=image.id,
            )
        )
    for filename in sorted(disk_names - metadata_names):
        issues.append(
            AuditIssue(
                "image_not_in_coco",
                "error",
                "Image file is not referenced by COCO metadata.",
                filename=filename,
            )
        )

    if expected_image_count is not None and len(metadata_names) != expected_image_count:
        issues.append(
            AuditIssue(
                "unexpected_image_count",
                "error",
                f"Expected {expected_image_count} COCO images, found {len(metadata_names)}.",
                details={"expected": expected_image_count, "actual": len(metadata_names)},
            )
        )

    expected_categories = {(expected_category_id, expected_category_name)}
    actual_categories = {(category.id, category.name) for category in index.categories}
    if actual_categories != expected_categories:
        issues.append(
            AuditIssue(
                "category_mapping_mismatch",
                "error",
                "COCO category mapping does not match the configured fire category.",
                details={
                    "expected": sorted(expected_categories),
                    "actual": sorted(actual_categories),
                },
            )
        )

    decoded = 0
    orientation_count = 0
    seen_orientation_corrections: set[str] = set()
    for image in index.images:
        path = actual_images.get(image.file_name)
        if path is None:
            continue
        try:
            with Image.open(path) as opened:
                opened.load()
                raw_size = opened.size
                orientation_value = opened.getexif().get(274, 1)
                orientation = orientation_value if type(orientation_value) is int else None
                normalized_size = ImageOps.exif_transpose(opened).size
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            issues.append(
                AuditIssue(
                    "image_decode_failed",
                    "error",
                    f"Image cannot be fully decoded: {exc}",
                    filename=image.file_name,
                    image_id=image.id,
                )
            )
            continue
        decoded += 1
        if normalized_size != (image.width, image.height):
            issues.append(
                AuditIssue(
                    "image_dimension_mismatch",
                    "error",
                    "EXIF-normalized pixels do not match COCO width and height.",
                    filename=image.file_name,
                    image_id=image.id,
                    details={
                        "raw_size": list(raw_size),
                        "normalized_size": list(normalized_size),
                        "coco_size": [image.width, image.height],
                        "orientation": orientation,
                    },
                )
            )
        if orientation not in {None, 1}:
            orientation_count += 1
            correction = (
                corrections.orientation.get(image.file_name) if corrections is not None else None
            )
            acknowledged = correction is not None and correction.expected_orientation == orientation
            details: dict[str, object] = {
                "orientation": orientation,
                "raw_size": list(raw_size),
                "normalized_size": list(normalized_size),
            }
            if correction is not None:
                seen_orientation_corrections.add(image.file_name)
                details.update(
                    {
                        "correction_action": correction.action,
                        "correction_reason": correction.reason,
                        "correction_expected_orientation": correction.expected_orientation,
                    }
                )
            issues.append(
                AuditIssue(
                    "exif_orientation",
                    "warning",
                    "Image has a non-default EXIF orientation; derived pixels must be transposed.",
                    filename=image.file_name,
                    image_id=image.id,
                    acknowledged=acknowledged,
                    details=details,
                )
            )
            if correction is not None and not acknowledged:
                issues.append(
                    AuditIssue(
                        "orientation_correction_mismatch",
                        "error",
                        "EXIF orientation no longer matches the reviewed correction manifest.",
                        filename=image.file_name,
                        image_id=image.id,
                        details=details,
                    )
                )
        elif orientation is None:
            issues.append(
                AuditIssue(
                    "invalid_exif_orientation",
                    "error",
                    "EXIF Orientation must be an integer.",
                    filename=image.file_name,
                    image_id=image.id,
                    details={"orientation": repr(orientation_value)},
                )
            )

    annotations_by_image = index.annotations_by_image
    bbox_ratios: list[float] = []
    seen_bbox_corrections: set[int] = set()
    for image in index.images:
        duplicates: dict[tuple[int, float, float, float, float], int] = {}
        for annotation in annotations_by_image[image.id]:
            x, y, box_width, box_height = annotation.bbox
            if annotation.category_id != expected_category_id:
                issues.append(
                    AuditIssue(
                        "annotation_category_mismatch",
                        "error",
                        f"Annotation category must be {expected_category_id}.",
                        filename=image.file_name,
                        image_id=image.id,
                        annotation_id=annotation.id,
                        details={"category_id": annotation.category_id},
                    )
                )
            if annotation.iscrowd != 0:
                issues.append(
                    AuditIssue(
                        "unsupported_crowd_annotation",
                        "error",
                        "Baseline annotations must have iscrowd=0.",
                        filename=image.file_name,
                        image_id=image.id,
                        annotation_id=annotation.id,
                    )
                )
            if box_width <= 0.0 or box_height <= 0.0:
                issues.append(
                    AuditIssue(
                        "bbox_non_positive_extent",
                        "error",
                        "Bounding-box width and height must be positive.",
                        filename=image.file_name,
                        image_id=image.id,
                        annotation_id=annotation.id,
                        details={"bbox": list(annotation.bbox)},
                    )
                )
            else:
                bbox_ratios.append((box_width * box_height) / (image.width * image.height))
            if annotation.area <= 0.0:
                issues.append(
                    AuditIssue(
                        "annotation_non_positive_area",
                        "error",
                        "COCO annotation area must be positive.",
                        filename=image.file_name,
                        image_id=image.id,
                        annotation_id=annotation.id,
                        details={"area": annotation.area},
                    )
                )
            elif (
                box_width > 0.0
                and box_height > 0.0
                and not math.isclose(
                    annotation.area,
                    box_width * box_height,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            ):
                issues.append(
                    AuditIssue(
                        "annotation_area_mismatch",
                        "warning",
                        "COCO area differs from bbox width multiplied by height.",
                        filename=image.file_name,
                        image_id=image.id,
                        annotation_id=annotation.id,
                        details={
                            "area": annotation.area,
                            "bbox_area": box_width * box_height,
                        },
                    )
                )

            duplicate_key = (
                annotation.category_id,
                round(x, 9),
                round(y, 9),
                round(box_width, 9),
                round(box_height, 9),
            )
            if duplicate_key in duplicates:
                issues.append(
                    AuditIssue(
                        "duplicate_bbox",
                        "error",
                        "Duplicate category and bbox coordinates occur on the same image.",
                        filename=image.file_name,
                        image_id=image.id,
                        annotation_id=annotation.id,
                        details={"first_annotation_id": duplicates[duplicate_key]},
                    )
                )
            else:
                duplicates[duplicate_key] = annotation.id

            overflow = _bbox_overflow(annotation, image.width, image.height)
            if overflow <= 0.0:
                continue
            if overflow <= MACHINE_EPSILON_PX:
                severity, code = "info", "bbox_machine_precision"
            elif overflow <= FLOAT_BOUNDARY_TOLERANCE_PX:
                severity, code = "warning", "bbox_float_overflow"
            else:
                severity, code = "error", "bbox_out_of_bounds"
            acknowledged, correction_details = _correction_matches(annotation, corrections)
            correction_configured = annotation.id in (
                corrections.bbox if corrections is not None else {}
            )
            if correction_configured:
                seen_bbox_corrections.add(annotation.id)
            details = {
                "bbox": list(annotation.bbox),
                "image_size": [image.width, image.height],
                "overflow_px": overflow,
                **correction_details,
            }
            issues.append(
                AuditIssue(
                    code,
                    severity,
                    f"Bounding box exceeds the image by {overflow:.17g} px.",
                    filename=image.file_name,
                    image_id=image.id,
                    annotation_id=annotation.id,
                    acknowledged=acknowledged,
                    details=details,
                )
            )
            if correction_configured and not acknowledged:
                issues.append(
                    AuditIssue(
                        "bbox_correction_mismatch",
                        "error",
                        "Bounding box no longer matches the reviewed correction manifest.",
                        filename=image.file_name,
                        image_id=image.id,
                        annotation_id=annotation.id,
                        details=details,
                    )
                )

    for image in index.images:
        label = index.labels[image.file_name]
        annotations = annotations_by_image[image.id]
        if label == 0 and annotations:
            issues.append(
                AuditIssue(
                    "negative_image_has_bbox",
                    "error",
                    "Image-level label is 0 but COCO contains fire boxes.",
                    filename=image.file_name,
                    image_id=image.id,
                    details={"annotation_ids": [item.id for item in annotations]},
                )
            )
        if label == 1 and not annotations:
            issues.append(
                AuditIssue(
                    "positive_image_has_no_bbox",
                    "error",
                    "Image-level label is 1 but COCO contains no fire box.",
                    filename=image.file_name,
                    image_id=image.id,
                )
            )

    if corrections is not None:
        for annotation_id in sorted(set(corrections.bbox) - seen_bbox_corrections):
            issues.append(
                AuditIssue(
                    "bbox_correction_not_applicable",
                    "error",
                    "Configured bbox correction is missing, changed, or no longer out of bounds.",
                    annotation_id=annotation_id,
                )
            )
        for filename in sorted(set(corrections.orientation) - seen_orientation_corrections):
            issues.append(
                AuditIssue(
                    "orientation_correction_not_applicable",
                    "error",
                    "Configured orientation correction is missing or no longer matches EXIF.",
                    filename=filename,
                )
            )

    issue_severity = Counter(issue.severity for issue in issues)
    issue_codes = Counter(issue.code for issue in issues)
    build_allowed = not any(
        issue.severity == "error" and not issue.acknowledged for issue in issues
    )
    summary = {
        "image_files": len(actual_images),
        "decoded_images": decoded,
        "coco_images": len(index.images),
        "image_labels": len(index.labels),
        "negative_images": sum(label == 0 for label in index.labels.values()),
        "positive_images": sum(label == 1 for label in index.labels.values()),
        "annotations": len(index.annotations),
        "categories": [{"id": category.id, "name": category.name} for category in index.categories],
        "exif_oriented_images": orientation_count,
        "multi_box_images": sum(len(annotations_by_image[image.id]) > 1 for image in index.images),
        "small_box_annotations": sum(value < SMALL_BOX_AREA_RATIO for value in bbox_ratios),
        "bbox_area_ratio_distribution": _area_distribution(bbox_ratios),
        "issues_by_severity": {
            level: issue_severity.get(level, 0) for level in ("error", "warning", "info")
        },
        "issues_by_code": dict(sorted(issue_codes.items())),
        "unacknowledged_blocking_errors": sum(
            issue.severity == "error" and not issue.acknowledged for issue in issues
        ),
    }
    report = AuditReport(
        build_allowed,
        summary,
        tuple(issues),
        _source_hashes(rows),
        "sha256_manifest.csv",
        corrections.version if corrections is not None else None,
    )
    _write_report(output_dir, report)
    return report


def load_audit_issues(path: Path) -> tuple[AuditIssue, ...]:
    """Load structured issues from an audit ``report.json`` file."""

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read audit report {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("issues"), list):
        raise DataValidationError("audit report must contain an issues list")
    issues: list[AuditIssue] = []
    for offset, item in enumerate(raw["issues"]):
        if not isinstance(item, dict):
            raise DataValidationError(f"issues[{offset}] must be an object")
        code = item.get("code")
        severity = item.get("severity")
        message = item.get("message")
        if not isinstance(code, str) or severity not in {"info", "warning", "error"}:
            raise DataValidationError(f"issues[{offset}] has invalid code or severity")
        if not isinstance(message, str):
            raise DataValidationError(f"issues[{offset}].message must be a string")
        details = item.get("details")
        issues.append(
            AuditIssue(
                code,
                severity,
                message,
                item.get("filename") if isinstance(item.get("filename"), str) else None,
                item.get("image_id") if type(item.get("image_id")) is int else None,
                item.get("annotation_id") if type(item.get("annotation_id")) is int else None,
                item.get("acknowledged") is True,
                details if isinstance(details, dict) else {},
            )
        )
    return tuple(issues)


def index_for_visualization(coco_json: Path, image_labels_json: Path) -> DatasetIndex:
    """Load the shared strict index for visualization callers."""

    return load_dataset_index(Path(coco_json), Path(image_labels_json))
