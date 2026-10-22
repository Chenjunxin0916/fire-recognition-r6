"""Evidence-rich FP/FN reporting for the frozen Round-3 selection.

The report is derived exclusively from validated cross-fitted artifacts.  Human
semantic fields are emitted empty (or pending) on purpose: this module must not
invent scene labels or remediation decisions.
"""

from __future__ import annotations

import json
import math
import shutil
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from fire_detection.ensemble.selection import validate_final_selection_manifest
from fire_detection.utils import atomic_write_csv, atomic_write_json

SelectionValidator = Callable[..., tuple[dict[str, Any], str]]

_DETECTOR_COLUMNS = {
    "filename",
    "label",
    "fold",
    "group_id",
    "det_max_conf",
    "slice_max_conf",
    "det_original_full_slice_conf",
    "det_hflip_max_conf",
    "slice_hflip_max_conf",
    "det_hflip_full_slice_conf",
    "det_tta_mean_conf",
}
_CLASSIFIER_COLUMNS = {
    "filename",
    "label",
    "fold",
    "group_id",
    "cls_original_probability",
    "cls_hflip_probability",
    "cls_tta_mean_probability",
}
_SELECTED_COLUMNS = {
    "filename",
    "label",
    "fold",
    "score",
    "threshold",
    "y_pred",
    "confusion",
    "margin",
}
_ALIGNED_COLUMNS = {
    "comparison_candidate",
    "filename",
    "candidate_score",
    "candidate_threshold",
    "candidate_y_pred",
    "candidate_confusion",
}


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _resolve_artifact(root: Path, record: object, name: str) -> Path:
    value = _mapping(record, name)
    raw = value.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name}.path must be non-empty text")
    path = (root / Path(raw)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name}.path escapes the project root") from exc
    if not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    return path


def _read_csv(
    path: Path,
    required: set[str],
    name: str,
    *,
    require_unique_filenames: bool = True,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"cannot read {name} {path}: {exc}") from exc
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if require_unique_filenames and frame["filename"].duplicated().any():
        raise ValueError(f"{name} contains duplicate filenames")
    return frame


def _component_prediction_rows(
    aligned: pd.DataFrame,
    *,
    candidate: str,
    prefix: str,
) -> pd.DataFrame:
    rows = aligned[aligned["comparison_candidate"] == candidate].copy()
    if rows.empty:
        raise ValueError(f"aligned predictions omit {candidate}")
    if rows["filename"].duplicated().any():
        raise ValueError(f"aligned predictions duplicate {candidate} filenames")
    return rows[
        [
            "filename",
            "candidate_score",
            "candidate_threshold",
            "candidate_y_pred",
            "candidate_confusion",
        ]
    ].rename(
        columns={
            "candidate_score": f"{prefix}_independent_score",
            "candidate_threshold": f"{prefix}_independent_threshold",
            "candidate_y_pred": f"{prefix}_independent_prediction",
            "candidate_confusion": f"{prefix}_independent_confusion",
        }
    )


def _validate_probability_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"{column} contains non-finite values")
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{column} must stay in [0, 1]")
        frame[column] = values.astype(float)


def _error_attribution(row: pd.Series) -> str:
    detector_wrong = bool(row["detector_independent_error"])
    classifier_wrong = bool(row["classifier_independent_error"])
    if detector_wrong and classifier_wrong:
        return "both_components_wrong"
    if detector_wrong:
        return "detector_only_wrong"
    if classifier_wrong:
        return "classifier_only_wrong"
    return "fusion_only_wrong"


def build_winner_error_frame(
    manifest: Mapping[str, Any],
    *,
    project_root: Path,
) -> pd.DataFrame:
    """Join validated Round-3 component evidence and derive counterfactual scores."""

    root = project_root.resolve()
    artifacts = _mapping(manifest.get("artifacts"), "selection artifacts")
    inputs = _mapping(artifacts.get("inputs"), "selection artifacts.inputs")
    outputs = _mapping(artifacts.get("outputs"), "selection artifacts.outputs")
    components = _mapping(manifest.get("components"), "selection components")
    detector = _mapping(components.get("detector"), "selection detector")
    classifier = _mapping(components.get("classifier"), "selection classifier")
    comparison_outputs = _mapping(inputs.get("comparison_outputs"), "selection comparison outputs")

    selected_path = _resolve_artifact(
        root, outputs.get("selected_cross_fitted"), "selected cross-fitted"
    )
    detector_path = _resolve_artifact(root, detector.get("oof"), "detector OOF")
    classifier_path = _resolve_artifact(root, classifier.get("oof"), "classifier OOF")
    aligned_path = _resolve_artifact(
        root,
        comparison_outputs.get("aligned_predictions.csv"),
        "aligned predictions",
    )
    selected = _read_csv(selected_path, _SELECTED_COLUMNS, "selected predictions")
    detector_frame = _read_csv(detector_path, _DETECTOR_COLUMNS, "detector OOF")
    classifier_frame = _read_csv(classifier_path, _CLASSIFIER_COLUMNS, "classifier OOF")
    aligned = _read_csv(
        aligned_path,
        _ALIGNED_COLUMNS,
        "aligned predictions",
        require_unique_filenames=False,
    )

    detector_source = detector.get("source")
    classifier_source = classifier.get("source")
    if not isinstance(detector_source, str) or not isinstance(classifier_source, str):
        raise ValueError("selection component source names must be text")
    detector_predictions = _component_prediction_rows(
        aligned,
        candidate=f"detector:{detector_source}",
        prefix="detector",
    )
    classifier_predictions = _component_prediction_rows(
        aligned,
        candidate=f"classifier:{classifier_source}",
        prefix="classifier",
    )

    joined = selected.merge(
        detector_frame[list(_DETECTOR_COLUMNS)],
        on="filename",
        how="inner",
        suffixes=("", "_detector"),
        validate="one_to_one",
    ).merge(
        classifier_frame[list(_CLASSIFIER_COLUMNS)],
        on="filename",
        how="inner",
        suffixes=("", "_classifier"),
        validate="one_to_one",
    )
    joined = joined.merge(detector_predictions, on="filename", validate="one_to_one")
    joined = joined.merge(classifier_predictions, on="filename", validate="one_to_one")
    if len(joined) != len(selected):
        raise ValueError("winner evidence filename sets differ")
    labels_and_folds_match = all(
        joined["label"].astype(int).equals(joined[f"label_{suffix}"].astype(int))
        and joined["fold"].astype(int).equals(joined[f"fold_{suffix}"].astype(int))
        for suffix in ("detector", "classifier")
    )
    groups_match = joined["group_id"].astype(str).equals(joined["group_id_classifier"].astype(str))
    if not labels_and_folds_match or not groups_match:
        raise ValueError("winner component label, Fold, or group provenance differs")

    probability_columns = [
        "score",
        "threshold",
        "det_max_conf",
        "slice_max_conf",
        "det_original_full_slice_conf",
        "det_hflip_max_conf",
        "slice_hflip_max_conf",
        "det_hflip_full_slice_conf",
        "det_tta_mean_conf",
        "cls_original_probability",
        "cls_hflip_probability",
        "cls_tta_mean_probability",
        "detector_independent_score",
        "detector_independent_threshold",
        "classifier_independent_score",
        "classifier_independent_threshold",
    ]
    _validate_probability_columns(joined, probability_columns)

    fusion = _mapping(manifest.get("fusion"), "selection fusion")
    detector_weight = _finite_float(fusion.get("detector_weight"), "detector weight")
    classifier_weight = _finite_float(fusion.get("classifier_weight"), "classifier weight")
    if not math.isclose(detector_weight + classifier_weight, 1.0, abs_tol=1e-12):
        raise ValueError("selection fusion weights do not sum to one")

    joined["det_full_tta_mean_conf"] = (joined["det_max_conf"] + joined["det_hflip_max_conf"]) / 2.0
    joined["fused_no_slice_no_tta"] = (
        detector_weight * joined["det_max_conf"]
        + classifier_weight * joined["cls_original_probability"]
    )
    joined["fused_slice_no_tta"] = (
        detector_weight * joined["det_original_full_slice_conf"]
        + classifier_weight * joined["cls_original_probability"]
    )
    joined["fused_no_slice_tta"] = (
        detector_weight * joined["det_full_tta_mean_conf"]
        + classifier_weight * joined["cls_tta_mean_probability"]
    )
    joined["fused_slice_tta"] = (
        detector_weight * joined["det_tta_mean_conf"]
        + classifier_weight * joined["cls_tta_mean_probability"]
    )
    if not ((joined["fused_slice_tta"] - joined["score"]).abs() <= 1e-12).all():
        raise ValueError("winner score does not equal the declared fixed fusion")

    threshold = joined["threshold"]
    for stem in (
        "fused_no_slice_no_tta",
        "fused_slice_no_tta",
        "fused_no_slice_tta",
        "fused_slice_tta",
    ):
        joined[f"{stem}_prediction"] = (joined[stem] >= threshold).astype(int)
    joined["slice_score_delta"] = joined["fused_slice_tta"] - joined["fused_no_slice_tta"]
    joined["tta_score_delta"] = joined["fused_slice_tta"] - joined["fused_slice_no_tta"]
    joined["slice_transition"] = (
        joined["fused_no_slice_tta_prediction"].astype(str)
        + "->"
        + joined["fused_slice_tta_prediction"].astype(str)
    )
    joined["tta_transition"] = (
        joined["fused_slice_no_tta_prediction"].astype(str)
        + "->"
        + joined["fused_slice_tta_prediction"].astype(str)
    )
    joined["is_boundary_sample"] = joined["margin"].abs() <= 0.05
    joined["detector_independent_error"] = joined["detector_independent_prediction"].astype(
        int
    ) != joined["label"].astype(int)
    joined["classifier_independent_error"] = joined["classifier_independent_prediction"].astype(
        int
    ) != joined["label"].astype(int)
    joined["both_components_error"] = (
        joined["detector_independent_error"] & joined["classifier_independent_error"]
    )
    joined["mechanical_attribution"] = joined.apply(_error_attribution, axis=1)
    joined["slice_changed_decision"] = (
        joined["fused_no_slice_tta_prediction"] != joined["fused_slice_tta_prediction"]
    )
    joined["tta_changed_decision"] = (
        joined["fused_slice_no_tta_prediction"] != joined["fused_slice_tta_prediction"]
    )
    joined["uses_slicing"] = True
    joined["uses_tta"] = True
    joined["error_category"] = ""
    joined["suspected_label_issue"] = ""
    joined["suggested_action"] = ""
    joined["reviewer"] = ""
    joined["review_status"] = "pending"
    drop_columns = [
        "label_detector",
        "fold_detector",
        "label_classifier",
        "fold_classifier",
        "group_id_classifier",
    ]
    return joined.drop(columns=drop_columns).sort_values("filename", kind="stable")


def _render_contact_sheet(frame: pd.DataFrame, images_dir: Path, output: Path) -> None:
    if frame.empty:
        return
    columns = 4
    tile_width, tile_height = 320, 230
    rows = frame.to_dict(orient="records")
    sheet = Image.new(
        "RGB",
        (columns * tile_width, ((len(rows) + columns - 1) // columns) * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        x = index % columns * tile_width
        y = index // columns * tile_height
        try:
            with Image.open(images_dir / str(row["filename"])) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((tile_width - 10, 175), Image.Resampling.LANCZOS)
                sheet.paste(image, (x + (tile_width - image.width) // 2, y + 4))
        except (OSError, ValueError):
            draw.rectangle((x + 4, y + 4, x + tile_width - 4, y + 178), outline="red")
        caption = (
            f"{row['filename']}\nfold={int(row['fold'])} {row['confusion']} "
            f"score={float(row['score']):.4f} thr={float(row['threshold']):.4f}\n"
            f"det={float(row['det_tta_mean_conf']):.4f} "
            f"cls={float(row['cls_tta_mean_probability']):.4f} "
            f"{row['mechanical_attribution']}"
        )
        draw.multiline_text((x + 4, y + 181), caption, fill="black", font=font, spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=90, optimize=False, progressive=False)


def generate_winner_error_report(
    selection_manifest: Path,
    *,
    project_root: Path,
    output_dir: Path,
    images_dir: Path | None = None,
    selection_validator: SelectionValidator = validate_final_selection_manifest,
) -> dict[str, Any]:
    """Validate the winner and atomically publish its exhaustive FP/FN evidence."""

    root = project_root.resolve()
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite winner error report: {destination}")
    manifest, manifest_sha = selection_validator(selection_manifest, project_root=root)
    frame = build_winner_error_frame(manifest, project_root=root)
    fp = frame[frame["confusion"] == "FP"].sort_values("margin", ascending=False)
    fn = frame[frame["confusion"] == "FN"].sort_values("margin", ascending=True)
    metrics = _mapping(
        _mapping(manifest.get("metrics"), "selection metrics").get("cross_fitted"),
        "selection cross-fitted metrics",
    )
    expected_fp = int(metrics.get("fp", -1))
    expected_fn = int(metrics.get("fn", -1))
    if len(fp) != expected_fp or len(fn) != expected_fn:
        raise ValueError(
            f"winner error counts differ from selection: FP {len(fp)}/{expected_fp}, "
            f"FN {len(fn)}/{expected_fn}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        atomic_write_csv(staging / "final_fp_report.csv", fp)
        atomic_write_csv(staging / "final_fn_report.csv", fn)
        summary: dict[str, Any] = {
            "schema_version": 1,
            "kind": "round3_winner_error_report",
            "status": "mechanical_review_pending",
            "selection_manifest": {
                "path": selection_manifest.as_posix(),
                "sha256": manifest_sha,
            },
            "counts": {"fp": len(fp), "fn": len(fn)},
            "mechanical_attribution": {
                str(key): int(value)
                for key, value in pd.concat([fp, fn])["mechanical_attribution"]
                .value_counts()
                .sort_index()
                .items()
            },
            "boundary": {
                "definition": "abs(margin) <= 0.05",
                "fp": int(fp["is_boundary_sample"].sum()),
                "fn": int(fn["is_boundary_sample"].sum()),
            },
            "decision_changes": {
                "slice_fp": int(fp["slice_changed_decision"].sum()),
                "slice_fn": int(fn["slice_changed_decision"].sum()),
                "tta_fp": int(fp["tta_changed_decision"].sum()),
                "tta_fn": int(fn["tta_changed_decision"].sum()),
            },
            "human_review": {
                "required_rows": len(fp) + len(fn),
                "completed_rows": 0,
                "semantic_fields_are_intentionally_blank": True,
            },
        }
        atomic_write_json(staging / "error_summary.json", summary)
        if images_dir is not None:
            images = images_dir if images_dir.is_absolute() else root / images_dir
            _render_contact_sheet(fp, images, staging / "final_fp_contact.jpg")
            _render_contact_sheet(fn, images, staging / "final_fn_contact.jpg")
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def summary_json(summary: Mapping[str, Any]) -> str:
    """Return deterministic CLI JSON without NaN values."""

    return json.dumps(dict(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
