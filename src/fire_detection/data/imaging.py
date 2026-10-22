"""Read-only COCO annotation rendering and paginated contact sheets."""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from fire_detection.data.audit import SMALL_BOX_AREA_RATIO, AuditIssue
from fire_detection.data.schema import (
    AnnotationRecord,
    DatasetIndex,
    DataValidationError,
    load_dataset_index,
)

_CONTACT_COLUMNS = 4
_CONTACT_ROWS = 5
_TILE_WIDTH = 360
_TILE_HEIGHT = 280
_CAPTION_HEIGHT = 42


@dataclass(frozen=True)
class VisualizationResult:
    """Counts and output locations for a completed visualization run."""

    rendered_images: int
    contact_items: dict[str, int]
    contact_pages: dict[str, int]
    annotations_dir: Path
    contact_sheets_dir: Path


@dataclass(frozen=True)
class _ContactItem:
    group: str
    filename: str
    title: str
    annotation_id: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    bbox_area_ratio: float | None = None


def _display_text(value: str) -> str:
    return value.encode("ascii", errors="replace").decode("ascii")


def _load_normalized_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as opened:
            opened.load()
            return ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise DataValidationError(f"cannot render image {path}: {exc}") from exc


def _annotate(
    image: Image.Image,
    annotations: tuple[AnnotationRecord, ...],
    image_label: int,
) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    line_width = max(2, round(min(rendered.size) / 300))
    for annotation in annotations:
        x, y, width, height = annotation.bbox
        xyxy = (x, y, x + width, y + height)
        color = (255, 40, 40)
        draw.rectangle(xyxy, outline=color, width=line_width)
        caption = f"fire id={annotation.id}"
        text_bbox = draw.textbbox((0, 0), caption)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_y = max(0.0, y - text_height - 4)
        draw.rectangle((x, text_y, x + text_width + 4, text_y + text_height + 4), fill=color)
        draw.text((x + 2, text_y + 2), caption, fill=(255, 255, 255))
    header = f"label={image_label} boxes={len(annotations)}"
    header_bbox = draw.textbbox((0, 0), header)
    draw.rectangle(
        (0, 0, header_bbox[2] - header_bbox[0] + 8, header_bbox[3] - header_bbox[1] + 8),
        fill=(0, 0, 0),
    )
    draw.text((4, 4), header, fill=(255, 255, 255))
    return rendered


def _atomic_save(image: Image.Image, path: Path, image_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        options: dict[str, object] = {}
        if image_format == "JPEG":
            options = {"quality": 92, "subsampling": 0}
        image.save(temporary, format=image_format, **options)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fit_tile(image: Image.Image, bbox: tuple[float, float, float, float] | None) -> Image.Image:
    if bbox is not None:
        x, y, width, height = bbox
        center_x = x + width / 2
        center_y = y + height / 2
        context_width = max(width * 6, image.width * 0.12)
        context_height = max(height * 6, image.height * 0.12)
        target_ratio = _TILE_WIDTH / (_TILE_HEIGHT - _CAPTION_HEIGHT)
        if context_width / context_height < target_ratio:
            context_width = context_height * target_ratio
        else:
            context_height = context_width / target_ratio
        left = max(0.0, center_x - context_width / 2)
        top = max(0.0, center_y - context_height / 2)
        right = min(float(image.width), center_x + context_width / 2)
        bottom = min(float(image.height), center_y + context_height / 2)
        image = image.crop((left, top, right, bottom))
    fitted = ImageOps.contain(
        image,
        (_TILE_WIDTH, _TILE_HEIGHT - _CAPTION_HEIGHT),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (_TILE_WIDTH, _TILE_HEIGHT - _CAPTION_HEIGHT), (24, 24, 24))
    offset = ((canvas.width - fitted.width) // 2, (canvas.height - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _contact_page(
    items: list[_ContactItem],
    images_dir: Path,
    index: DatasetIndex,
) -> Image.Image:
    page = Image.new(
        "RGB",
        (_CONTACT_COLUMNS * _TILE_WIDTH, _CONTACT_ROWS * _TILE_HEIGHT),
        (245, 245, 245),
    )
    annotations_by_image = index.annotations_by_image
    images_by_name = index.images_by_name
    for position, item in enumerate(items):
        metadata = images_by_name[item.filename]
        source = _load_normalized_rgb(images_dir / item.filename)
        annotated = _annotate(
            source,
            annotations_by_image[metadata.id],
            index.labels[item.filename],
        )
        tile_image = _fit_tile(annotated, item.bbox)
        tile = Image.new("RGB", (_TILE_WIDTH, _TILE_HEIGHT), (255, 255, 255))
        tile.paste(tile_image, (0, 0))
        draw = ImageDraw.Draw(tile)
        draw.multiline_text(
            (5, _TILE_HEIGHT - _CAPTION_HEIGHT + 4),
            _display_text(item.title)[:110],
            fill=(0, 0, 0),
            spacing=2,
        )
        column = position % _CONTACT_COLUMNS
        row = position // _CONTACT_COLUMNS
        page.paste(tile, (column * _TILE_WIDTH, row * _TILE_HEIGHT))
    return page


def _issue_contact_items(issues: tuple[AuditIssue, ...]) -> list[_ContactItem]:
    items: list[_ContactItem] = []
    for issue in issues:
        if issue.filename is None:
            continue
        bbox: tuple[float, float, float, float] | None = None
        raw_bbox = (issue.details or {}).get("bbox")
        if (
            isinstance(raw_bbox, list)
            and len(raw_bbox) == 4
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in raw_bbox
            )
        ):
            bbox = tuple(float(value) for value in raw_bbox)  # type: ignore[assignment]
        if issue.code.startswith("bbox_"):
            group = "bbox_issues"
        elif "exif" in issue.code or "orientation" in issue.code:
            group = "exif_orientation"
        else:
            group = "issues"
        items.append(
            _ContactItem(
                group,
                issue.filename,
                f"{issue.severity}:{issue.code} ann={issue.annotation_id}",
                issue.annotation_id,
                bbox,
            )
        )
    return items


def _dataset_contact_items(index: DatasetIndex) -> list[_ContactItem]:
    items: list[_ContactItem] = []
    annotations_by_image = index.annotations_by_image
    for image in index.images:
        annotations = annotations_by_image[image.id]
        if len(annotations) > 1:
            items.append(
                _ContactItem(
                    "multiple_boxes",
                    image.file_name,
                    f"{image.file_name} boxes={len(annotations)}",
                )
            )
        for annotation in annotations:
            ratio = annotation.bbox[2] * annotation.bbox[3] / (image.width * image.height)
            if ratio < SMALL_BOX_AREA_RATIO:
                items.append(
                    _ContactItem(
                        "small_boxes",
                        image.file_name,
                        f"{image.file_name} ann={annotation.id} ratio={ratio:.8f}",
                        annotation.id,
                        annotation.bbox,
                        ratio,
                    )
                )
    return items


def _write_contact_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    buffer = io.StringIO(newline="")
    fields = ("group", "page", "filename", "annotation_id", "bbox_area_ratio", "title")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(buffer.getvalue(), encoding="utf-8", newline="")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def visualize_annotations(
    images_dir: Path,
    coco_json: Path,
    image_labels_json: Path,
    output_dir: Path,
    *,
    issues: tuple[AuditIssue, ...] = (),
) -> VisualizationResult:
    """Render every image and all small-box, multi-box, and audit-issue contact sheets."""

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    index = load_dataset_index(Path(coco_json), Path(image_labels_json))
    annotations_dir = output_dir / "annotations"
    contacts_dir = output_dir / "contact_sheets"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    contacts_dir.mkdir(parents=True, exist_ok=True)

    rendered_images = 0
    for metadata in index.images:
        path = images_dir / metadata.file_name
        if not path.is_file():
            raise DataValidationError(f"cannot visualize missing image: {metadata.file_name}")
        source = _load_normalized_rgb(path)
        if source.size != (metadata.width, metadata.height):
            raise DataValidationError(
                f"normalized image size mismatch for {metadata.file_name}: "
                f"pixels={source.size}, COCO={(metadata.width, metadata.height)}"
            )
        rendered = _annotate(
            source,
            index.annotations_by_image[metadata.id],
            index.labels[metadata.file_name],
        )
        if Path(metadata.file_name).suffix.lower() in {".jpg", ".jpeg"}:
            output_name = f"{metadata.file_name}.annotated.jpg"
            output_format = "JPEG"
        else:
            output_name = f"{metadata.file_name}.annotated.png"
            output_format = "PNG"
        _atomic_save(rendered, annotations_dir / output_name, output_format)
        rendered_images += 1

    contact_items = _dataset_contact_items(index) + _issue_contact_items(issues)
    grouped: dict[str, list[_ContactItem]] = {}
    for item in contact_items:
        if item.filename not in index.images_by_name:
            continue
        grouped.setdefault(item.group, []).append(item)
    for items in grouped.values():
        items.sort(key=lambda item: (item.filename, item.annotation_id or -1, item.title))

    per_page = _CONTACT_COLUMNS * _CONTACT_ROWS
    pages_by_group: dict[str, int] = {}
    manifest_rows: list[dict[str, object]] = []
    for group in sorted(grouped):
        items = grouped[group]
        pages = (len(items) + per_page - 1) // per_page
        pages_by_group[group] = pages
        for page_number in range(1, pages + 1):
            start = (page_number - 1) * per_page
            page_items = items[start : start + per_page]
            page = _contact_page(page_items, images_dir, index)
            _atomic_save(page, contacts_dir / f"{group}_{page_number:03d}.jpg", "JPEG")
            for item in page_items:
                manifest_rows.append(
                    {
                        "group": group,
                        "page": page_number,
                        "filename": item.filename,
                        "annotation_id": item.annotation_id,
                        "bbox_area_ratio": item.bbox_area_ratio,
                        "title": item.title,
                    }
                )
    _write_contact_manifest(contacts_dir / "manifest.csv", manifest_rows)
    return VisualizationResult(
        rendered_images,
        {group: len(items) for group, items in sorted(grouped.items())},
        pages_by_group,
        annotations_dir,
        contacts_dir,
    )
