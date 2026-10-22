"""Cross-fitted confusion tables and image contact sheets."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from fire_detection.inference.oof import canonical_filename
from fire_detection.utils import atomic_write_csv, atomic_write_json

SMALL_FIRE_AREA_RATIO = 0.001


def load_gt_box_summary(
    coco_path: Path,
    *,
    small_fire_area_ratio: float = SMALL_FIRE_AREA_RATIO,
) -> pd.DataFrame:
    """Return both fixed small-fire definitions for every COCO image.

    ``contains_small_fire`` means that at least one GT box is smaller than the
    threshold. ``all_boxes_small`` means that the image has at least one GT box and
    every GT box is smaller than the threshold.  The historical
    ``all_gt_boxes_small`` column remains as an exact compatibility alias.
    """

    if (
        isinstance(small_fire_area_ratio, bool)
        or not isinstance(small_fire_area_ratio, Real)
        or not math.isfinite(float(small_fire_area_ratio))
        or not 0.0 < float(small_fire_area_ratio) <= 1.0
    ):
        raise ValueError("small_fire_area_ratio must be in (0, 1]")
    threshold = float(small_fire_area_ratio)

    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload["images"]}
    ratios: dict[int, list[float]] = {image_id: [] for image_id in images}
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        image = images[image_id]
        width = float(annotation["bbox"][2])
        height = float(annotation["bbox"][3])
        ratios[image_id].append(width * height / (float(image["width"]) * float(image["height"])))
    rows: list[dict[str, Any]] = []
    for image_id, image in images.items():
        values = ratios[image_id]
        minimum = min(values, default=0.0)
        maximum = max(values, default=0.0)
        contains_small_fire = bool(values) and minimum < threshold
        all_boxes_small = bool(values) and maximum < threshold
        rows.append(
            {
                "filename": canonical_filename(image["file_name"]),
                "gt_box_count": len(values),
                "min_gt_box_ratio": minimum,
                "max_gt_box_ratio": maximum,
                "contains_small_fire": contains_small_fire,
                "all_boxes_small": all_boxes_small,
                "all_gt_boxes_small": all_boxes_small,
            }
        )
    return pd.DataFrame(rows)


def _render_contact_sheet(
    frame: pd.DataFrame,
    *,
    images_dir: Path,
    output: Path,
    limit: int,
) -> None:
    rows = frame.head(limit).to_dict(orient="records")
    if not rows:
        return
    columns = 5
    tile_width, tile_height = 260, 205
    sheet = Image.new(
        "RGB",
        (columns * tile_width, ((len(rows) + columns - 1) // columns) * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        path = images_dir / str(row["filename"])
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((tile_width - 10, 160), Image.Resampling.LANCZOS)
                sheet.paste(image, (x + (tile_width - image.width) // 2, y + 5))
        except (OSError, ValueError):
            draw.rectangle((x + 5, y + 5, x + tile_width - 5, y + 165), outline="red")
        caption = (
            f"{row['filename']}\nscore={float(row['score']):.4f} thr={float(row['threshold']):.4f}"
        )
        draw.multiline_text((x + 5, y + 168), caption, fill="black", font=font, spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=90, optimize=False, progressive=False)


def generate_error_report(
    predictions: pd.DataFrame,
    *,
    output_dir: Path,
    images_dir: Path | None = None,
    coco_path: Path | None = None,
    contact_limit: int = 100,
) -> dict[str, int]:
    """Write exhaustive confusion CSVs, threshold-near rows, and optional contact sheets."""

    required = {"filename", "label", "fold", "score", "threshold", "y_pred", "margin"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"cross-fitted predictions missing columns: {missing}")
    frame = predictions.copy()
    if "confusion" not in frame:
        mapping = {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}
        frame["confusion"] = [
            mapping[(int(label), int(prediction))]
            for label, prediction in zip(frame["label"], frame["y_pred"], strict=True)
        ]
    if coco_path is not None:
        frame = frame.merge(
            load_gt_box_summary(coco_path),
            on="filename",
            how="left",
            validate="one_to_one",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    order = {
        "FP": ("margin", False),
        "FN": ("margin", True),
        "TP": ("margin", False),
        "TN": ("margin", True),
    }
    counts: dict[str, int] = {}
    for confusion, (column, ascending) in order.items():
        subset = frame[frame["confusion"] == confusion].sort_values(column, ascending=ascending)
        counts[confusion.lower()] = len(subset)
        atomic_write_csv(output_dir / f"{confusion.lower()}.csv", subset)
        if images_dir is not None:
            _render_contact_sheet(
                subset,
                images_dir=images_dir,
                output=output_dir / f"{confusion.lower()}_contact.jpg",
                limit=contact_limit,
            )
    transition_reports = {
        "TP->FN": ("new_fn_tp_to_fn", "margin", True),
        "FN->TP": ("recovered_fn_fn_to_tp", "margin", False),
        "TN->FP": ("new_fp_tn_to_fp", "margin", False),
        "FP->TN": ("recovered_fp_fp_to_tn", "margin", True),
    }
    if "transition" in frame:
        for transition, (stem, column, ascending) in transition_reports.items():
            subset = frame[frame["transition"] == transition].sort_values(
                column,
                ascending=ascending,
            )
            counts[stem] = len(subset)
            atomic_write_csv(output_dir / f"{stem}.csv", subset)
            if images_dir is not None:
                _render_contact_sheet(
                    subset,
                    images_dir=images_dir,
                    output=output_dir / f"{stem}_contact.jpg",
                    limit=contact_limit,
                )
    near = frame.assign(abs_margin=frame["margin"].abs()).sort_values("abs_margin")
    atomic_write_csv(output_dir / "threshold_near.csv", near)
    if images_dir is not None:
        _render_contact_sheet(
            near,
            images_dir=images_dir,
            output=output_dir / "threshold_near_contact.jpg",
            limit=contact_limit,
        )
    small_fire_frames: dict[str, pd.DataFrame] = {}
    for criterion in ("contains_small_fire", "all_boxes_small"):
        if criterion not in frame:
            continue
        subset = frame[
            (frame["confusion"] == "FN") & frame[criterion].fillna(False).astype(bool)
        ].sort_values("margin")
        small_fire_frames[criterion] = subset
        counts[f"{criterion}_fn"] = len(subset)
        atomic_write_csv(output_dir / f"{criterion}_fn.csv", subset)
        if images_dir is not None:
            _render_contact_sheet(
                subset,
                images_dir=images_dir,
                output=output_dir / f"{criterion}_fn_contact.jpg",
                limit=contact_limit,
            )

    # Keep the original all-boxes-small report names stable for downstream tooling.
    legacy_small_fn = small_fire_frames.get("all_boxes_small")
    if legacy_small_fn is None and "all_gt_boxes_small" in frame:
        legacy_small_fn = frame[
            (frame["confusion"] == "FN") & frame["all_gt_boxes_small"].fillna(False).astype(bool)
        ].sort_values("margin")
    if legacy_small_fn is not None:
        counts["small_fire_fn"] = len(legacy_small_fn)
        atomic_write_csv(output_dir / "small_fire_fn.csv", legacy_small_fn)
        if images_dir is not None:
            _render_contact_sheet(
                legacy_small_fn,
                images_dir=images_dir,
                output=output_dir / "small_fire_fn_contact.jpg",
                limit=contact_limit,
            )
    atomic_write_json(output_dir / "summary.json", counts)
    return counts
