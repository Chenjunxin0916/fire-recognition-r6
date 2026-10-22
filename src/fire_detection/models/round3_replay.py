"""Versioned, leakage-safe hard-sample replay for the Round-3 classifiers.

This module deliberately does not change the historical HN2 contract.  It validates
that contract as an input, adds independently selected HP2, Small2, and Smoke2
samples, and emits one deterministic manifest per outer Fold.  A filename selected
by more than one source is still replayed only once, so its effective factor never
exceeds two.
"""

from __future__ import annotations

import json
import math
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from fire_detection.ensemble.error_report import load_gt_box_summary
from fire_detection.models.hard_negative import (
    EXPECTED_FOLDS,
    load_hard_negative_replay,
    load_nested_oof_csv,
    validate_nested_oof_rows,
)
from fire_detection.models.offline import sha256_file
from fire_detection.models.training import (
    FoldRecord,
    TrainingConfigurationError,
    load_fold_records,
    write_json_atomic,
)

ROUND3_VARIANTS: dict[str, tuple[str, ...]] = {
    "hn2_hp2": ("hn2", "hp2"),
    "hn2_small2": ("hn2", "small2"),
    "hn2_smoke2": ("hn2", "smoke2"),
    "hn2_hp2_smoke2": ("hn2", "hp2", "smoke2"),
}
SMOKE_CATEGORIES = (
    "smoke_or_steam",
    "light_or_reflection",
    "sunset_or_afterglow",
    "welding_or_sparks",
    "red_object",
    "warm_light",
    "cloud_or_fog",
    "other_negative",
)
SMOKE_REVIEW_V2_POLICY = "v2_complete_bundle_approved_training_eligible_only"
SMOKE_REVIEW_V2_DISPOSITIONS = (
    "confirmed_no_fire",
    "uncertain",
    "suspected_label_error",
)
SMOKE_REVIEW_V2_TAGS = (
    "smoke_or_steam",
    "light",
    "high_brightness",
    "sunset_or_afterglow",
    "red_or_orange_object",
    "reflection",
    "welding_or_sparks",
    "cloud_or_fog",
    "duplicate_or_near_duplicate",
    "other",
)


@dataclass(frozen=True)
class Round3ReplaySpec:
    """Validated paths and constants for the isolated Round-3 replay campaign."""

    project_root: Path
    config_path: Path
    experiment_name: str
    seed: int
    folds_path: Path
    processed_manifest: Path
    coco_json: Path
    hn2_manifest_template: str
    smoke_review: Path
    output_root: Path
    repeat_factor: int
    expected_count: int
    expected_validation_count: int
    expected_negative_count: int
    hp2_rule: str
    small2_all_boxes_threshold: float
    small2_max_box_ratio: float
    smoke2_group_rule: str
    smoke_review_policy: str
    variants: dict[str, tuple[str, ...]]

    def hn2_manifest(self, outer_fold: int) -> Path:
        """Resolve the historical clean HN2 manifest for one outer Fold."""

        return _template_path(
            self.project_root,
            self.hn2_manifest_template,
            field="sources.hn2_manifest",
            fold=outer_fold,
        )

    def replay_manifest(self, variant: str, outer_fold: int) -> Path:
        """Return the canonical Round-3 replay manifest path."""

        _variant_sources(self, variant)
        _require_outer_fold(outer_fold)
        return self.output_root / variant / f"outer_{outer_fold}.json"

    def campaign_manifest(self, variant: str) -> Path:
        """Return the canonical five-Fold campaign-manifest path."""

        _variant_sources(self, variant)
        return self.output_root / variant / "campaign_manifest.json"


@dataclass(frozen=True)
class ReplaySelection:
    """One source-specific replay selection and its audit evidence."""

    record: FoldRecord
    source: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class HardSampleReplayContract:
    """Validated Round-3 replay inputs consumed by one classifier Fold."""

    config_path: Path
    config_sha256: str
    manifest_path: Path
    manifest_sha256: str
    variant: str
    outer_fold: int
    repeat_factor: int
    replay_samples: tuple[tuple[str, int], ...]
    original_train_count: int
    effective_train_count: int
    original_train_class_counts: dict[str, int]
    effective_train_class_counts: dict[str, int]
    validation_unique_count: int
    validation_class_counts: dict[str, int]

    def manifest_section(self, project_root: Path) -> dict[str, Any]:
        """Return stable run-manifest fields for this replay contract."""

        return {
            "campaign_config": _repository_relative(self.config_path, project_root),
            "campaign_config_sha256": self.config_sha256,
            "manifest": _repository_relative(self.manifest_path, project_root),
            "manifest_sha256": self.manifest_sha256,
            "variant": self.variant,
            "repeat_factor": self.repeat_factor,
            "replay_unique_count": len(self.replay_samples),
            "original_train_count": self.original_train_count,
            "effective_train_count": self.effective_train_count,
            "original_train_class_counts": self.original_train_class_counts,
            "effective_train_class_counts": self.effective_train_class_counts,
            "validation_unique_count": self.validation_unique_count,
            "validation_class_counts": self.validation_class_counts,
        }


@dataclass(frozen=True)
class SmokeReviewV2Summary:
    """Validated completeness and eligibility counts for one Smoke2 v2 bundle."""

    total: int
    pending: int
    secondary_required: int
    complete: int
    training_eligible: int
    excluded: int
    final_dispositions: dict[str, int]
    tags: dict[str, int]
    folds: dict[str, dict[str, int]]
    eligible_tags: dict[str, tuple[str, ...]]

    @property
    def remaining(self) -> int:
        """Return rows that still require a primary or secondary human decision."""

        return self.pending + self.secondary_required


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingConfigurationError(f"{field} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, field: str) -> Any:
    if key not in mapping:
        raise TrainingConfigurationError(f"missing required configuration field {field}.{key}")
    return mapping[key]


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigurationError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TrainingConfigurationError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_ratio(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigurationError(f"{field} must be a finite number in (0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise TrainingConfigurationError(f"{field} must be a finite number in (0, 1]")
    return result


def _path(project_root: Path, value: Any, field: str) -> Path:
    raw = _string(value, field)
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _template_path(project_root: Path, template: str, *, field: str, fold: int) -> Path:
    _require_outer_fold(fold)
    try:
        raw = template.format(fold=fold)
    except (KeyError, ValueError) as exc:
        raise TrainingConfigurationError(f"invalid Fold template in {field}: {template!r}") from exc
    if "{" in raw or "}" in raw:
        raise TrainingConfigurationError(f"unresolved Fold template in {field}: {template!r}")
    return _path(project_root, raw, field)


def _repository_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _require_outer_fold(outer_fold: int) -> None:
    if type(outer_fold) is not int or outer_fold not in EXPECTED_FOLDS:
        raise TrainingConfigurationError("outer_fold must be in 0..4")


def _class_counts(records: Sequence[FoldRecord]) -> dict[str, int]:
    return {
        "0": sum(record.label == 0 for record in records),
        "1": sum(record.label == 1 for record in records),
    }


def _artifact_record(path: Path, project_root: Path) -> dict[str, str]:
    if not path.is_file():
        raise TrainingConfigurationError(f"required Round-3 artifact is missing: {path}")
    return {
        "path": _repository_relative(path, project_root),
        "sha256": sha256_file(path),
    }


def _load_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingConfigurationError(f"cannot read {field} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingConfigurationError(f"{field} must be a JSON object: {path}")
    return payload


def _bound_artifact(
    section: Any,
    *,
    project_root: Path,
    field: str,
) -> Path:
    record = _mapping(section, field)
    raw = _string(record.get("path"), f"{field}.path").replace("\\", "/")
    portable = PurePosixPath(raw)
    if portable.is_absolute() or PureWindowsPath(raw).is_absolute() or ".." in portable.parts:
        raise TrainingConfigurationError(f"{field}.path must be repository-relative")
    path = (project_root / Path(*portable.parts)).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise TrainingConfigurationError(f"{field}.path escapes the project root") from exc
    expected = record.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise TrainingConfigurationError(f"{field}.sha256 must be a SHA-256 digest")
    if not path.is_file() or sha256_file(path) != expected.lower():
        raise TrainingConfigurationError(f"{field} artifact is missing or changed: {path}")
    return path


def _variant_sources(spec: Round3ReplaySpec, variant: str) -> tuple[str, ...]:
    sources = spec.variants.get(variant)
    if sources is None:
        raise TrainingConfigurationError(
            f"unknown Round-3 replay variant {variant!r}; expected {sorted(spec.variants)}"
        )
    return sources


def load_round3_replay_spec(path: Path) -> Round3ReplaySpec:
    """Load the decision-complete, isolated Round-3 replay configuration."""

    config_path = path.resolve()
    project_root = config_path.parent.parent
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TrainingConfigurationError(
            f"cannot read Round-3 config {config_path}: {exc}"
        ) from exc
    root = _mapping(raw, "Round-3 replay config")
    experiment = _mapping(_required(root, "experiment", "config"), "experiment")
    data = _mapping(_required(root, "data", "config"), "data")
    sources = _mapping(_required(root, "sources", "config"), "sources")
    outputs = _mapping(_required(root, "outputs", "config"), "outputs")
    replay = _mapping(_required(root, "replay", "config"), "replay")
    selection = _mapping(_required(root, "selection", "config"), "selection")

    seed = _integer(_required(experiment, "seed", "experiment"), "experiment.seed")
    if seed != 42:
        raise TrainingConfigurationError("Round-3 replay requires seed 42")
    repeat_factor = _integer(
        _required(replay, "repeat_factor", "replay"), "replay.repeat_factor", minimum=2
    )
    if repeat_factor != 2:
        raise TrainingConfigurationError("Round-3 replay.repeat_factor must be exactly 2")
    variants_raw = _mapping(_required(replay, "variants", "replay"), "replay.variants")
    variants: dict[str, tuple[str, ...]] = {}
    for name, expected in ROUND3_VARIANTS.items():
        value = variants_raw.get(name)
        if value != list(expected):
            raise TrainingConfigurationError(
                f"replay.variants.{name} must be exactly {list(expected)}"
            )
        variants[name] = expected
    if set(variants_raw) != set(ROUND3_VARIANTS):
        raise TrainingConfigurationError(
            f"replay.variants must contain exactly {sorted(ROUND3_VARIANTS)}"
        )

    hp2 = _mapping(_required(selection, "hp2", "selection"), "selection.hp2")
    small2 = _mapping(_required(selection, "small2", "selection"), "selection.small2")
    smoke2 = _mapping(_required(selection, "smoke2", "selection"), "selection.smoke2")
    hp2_rule = _string(_required(hp2, "group_rule", "selection.hp2"), "selection.hp2.group_rule")
    if hp2_rule != "lowest_margin_then_filename":
        raise TrainingConfigurationError(
            "selection.hp2.group_rule must be lowest_margin_then_filename"
        )
    small_rule = _string(
        _required(small2, "group_rule", "selection.small2"),
        "selection.small2.group_rule",
    )
    if small_rule != "smallest_max_box_ratio_then_filename":
        raise TrainingConfigurationError(
            "selection.small2.group_rule must be smallest_max_box_ratio_then_filename"
        )
    smoke_rule = _string(
        _required(smoke2, "group_rule", "selection.smoke2"),
        "selection.smoke2.group_rule",
    )
    if smoke_rule != "filename_ascending":
        raise TrainingConfigurationError("selection.smoke2.group_rule must be filename_ascending")
    review_policy = _string(
        _required(smoke2, "review_policy", "selection.smoke2"),
        "selection.smoke2.review_policy",
    )
    if review_policy not in {
        "all_236_rows_and_bundle_approved",
        SMOKE_REVIEW_V2_POLICY,
    }:
        raise TrainingConfigurationError(
            "selection.smoke2.review_policy must be the historical v1 policy or "
            f"{SMOKE_REVIEW_V2_POLICY}"
        )

    output_root = _path(
        project_root, _required(outputs, "manifests", "outputs"), "outputs.manifests"
    )
    output_directory = (
        "runs/round3/replay_smoke_v2"
        if review_policy == SMOKE_REVIEW_V2_POLICY
        else "runs/round3/replay"
    )
    allowed_output = (project_root / output_directory).resolve()
    if output_root != allowed_output:
        raise TrainingConfigurationError(f"outputs.manifests must be {output_directory}")
    hn2_template = _string(_required(sources, "hn2_manifest", "sources"), "sources.hn2_manifest")
    if "{fold}" not in hn2_template:
        raise TrainingConfigurationError("sources.hn2_manifest must contain {fold}")

    return Round3ReplaySpec(
        project_root=project_root,
        config_path=config_path,
        experiment_name=_string(_required(experiment, "name", "experiment"), "experiment.name"),
        seed=seed,
        folds_path=_path(project_root, _required(data, "folds", "data"), "data.folds"),
        processed_manifest=_path(
            project_root,
            _required(data, "processed_manifest", "data"),
            "data.processed_manifest",
        ),
        coco_json=_path(project_root, _required(data, "coco_json", "data"), "data.coco_json"),
        hn2_manifest_template=hn2_template,
        smoke_review=_path(
            project_root,
            _required(sources, "smoke_review", "sources"),
            "sources.smoke_review",
        ),
        output_root=output_root,
        repeat_factor=repeat_factor,
        expected_count=_integer(
            _required(replay, "expected_count", "replay"),
            "replay.expected_count",
            minimum=1,
        ),
        expected_validation_count=_integer(
            _required(replay, "expected_validation_count", "replay"),
            "replay.expected_validation_count",
            minimum=1,
        ),
        expected_negative_count=_integer(
            _required(replay, "expected_negative_count", "replay"),
            "replay.expected_negative_count",
            minimum=1,
        ),
        hp2_rule=hp2_rule,
        small2_all_boxes_threshold=_finite_ratio(
            _required(small2, "all_boxes_small_threshold", "selection.small2"),
            "selection.small2.all_boxes_small_threshold",
        ),
        small2_max_box_ratio=_finite_ratio(
            _required(small2, "max_gt_box_ratio", "selection.small2"),
            "selection.small2.max_gt_box_ratio",
        ),
        smoke2_group_rule=smoke_rule,
        smoke_review_policy=review_policy,
        variants=variants,
    )


def create_smoke_review_template(
    spec: Round3ReplaySpec,
    records: Sequence[FoldRecord],
) -> dict[str, Any]:
    """Write a complete but deliberately unapproved 236-row semantic review template."""

    negatives = sorted(
        (record for record in records if record.label == 0), key=lambda x: x.filename
    )
    if len(negatives) != spec.expected_negative_count:
        raise TrainingConfigurationError(
            f"Smoke2 expects {spec.expected_negative_count} negatives, found {len(negatives)}"
        )
    if spec.smoke_review.exists():
        raise FileExistsError(f"refusing to overwrite Smoke2 review: {spec.smoke_review}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "round3_smoke_negative_semantic_review",
        "status": "pending",
        "reviewer": None,
        "reviewed_at": None,
        "folds": _artifact_record(spec.folds_path, spec.project_root),
        "expected_negative_count": spec.expected_negative_count,
        "allowed_categories": list(SMOKE_CATEGORIES),
        "records": [
            {
                "filename": record.filename,
                "label": 0,
                "fold": record.fold,
                "group_id": record.group_id,
                "category": None,
                "review_status": "pending",
                "notes": "",
            }
            for record in negatives
        ],
    }
    write_json_atomic(spec.smoke_review, payload)
    return payload


def _approved_timestamp(value: Any) -> str:
    raw = _string(value, "Smoke2 review.reviewed_at")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TrainingConfigurationError(
            "Smoke2 review.reviewed_at must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TrainingConfigurationError("Smoke2 review.reviewed_at must include a timezone")
    return raw


def validate_smoke_review(
    payload: Mapping[str, Any],
    records: Sequence[FoldRecord],
    *,
    folds_path: Path,
    project_root: Path,
    expected_negative_count: int,
) -> dict[str, str]:
    """Validate full human sign-off and return filename-to-category mappings.

    Pending, rejected, or uncertain rows block the entire Smoke2 source.  Categories
    are never inferred from filenames, pixels, model scores, or group neighbours.
    """

    if payload.get("schema_version") != 1 or payload.get("kind") != (
        "round3_smoke_negative_semantic_review"
    ):
        raise TrainingConfigurationError("Smoke2 review schema/kind mismatch")
    if payload.get("status") != "approved":
        raise TrainingConfigurationError(
            "Smoke2 review is not approved; pending/uncertain reviews block replay"
        )
    _string(payload.get("reviewer"), "Smoke2 review.reviewer")
    _approved_timestamp(payload.get("reviewed_at"))
    if payload.get("expected_negative_count") != expected_negative_count:
        raise TrainingConfigurationError("Smoke2 review expected_negative_count mismatch")
    if payload.get("allowed_categories") != list(SMOKE_CATEGORIES):
        raise TrainingConfigurationError("Smoke2 review allowed_categories drifted")
    folds = _mapping(payload.get("folds"), "Smoke2 review.folds")
    if folds != _artifact_record(folds_path, project_root):
        raise TrainingConfigurationError("Smoke2 review Fold provenance drifted")

    expected = {record.filename: record for record in records if record.label == 0}
    if len(expected) != expected_negative_count:
        raise TrainingConfigurationError(
            "fixed Fold file contains "
            f"{len(expected)} negatives, expected {expected_negative_count}"
        )
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != expected_negative_count:
        raise TrainingConfigurationError(
            f"Smoke2 review must contain exactly {expected_negative_count} rows"
        )
    categories: dict[str, str] = {}
    allowed = set(SMOKE_CATEGORIES)
    for offset, raw in enumerate(rows):
        row = _mapping(raw, f"Smoke2 review.records[{offset}]")
        filename = _string(row.get("filename"), f"Smoke2 review.records[{offset}].filename")
        if filename in categories:
            raise TrainingConfigurationError(f"duplicate Smoke2 review filename: {filename}")
        record = expected.get(filename)
        if record is None:
            raise TrainingConfigurationError(f"unknown/non-negative Smoke2 filename: {filename}")
        if (
            row.get("label") != 0
            or row.get("fold") != record.fold
            or row.get("group_id") != record.group_id
        ):
            raise TrainingConfigurationError(f"Smoke2 review provenance mismatch for {filename}")
        status = row.get("review_status")
        if status != "approved":
            raise TrainingConfigurationError(
                f"Smoke2 review row {filename} is {status!r}; all rows must be approved"
            )
        category = row.get("category")
        if category not in allowed:
            raise TrainingConfigurationError(
                f"Smoke2 review row {filename} has invalid/uncertain category {category!r}"
            )
        notes = row.get("notes")
        if not isinstance(notes, str):
            raise TrainingConfigurationError(f"Smoke2 review notes must be text for {filename}")
        categories[filename] = str(category)
    if set(categories) != set(expected):
        missing = sorted(set(expected) - set(categories))
        raise TrainingConfigurationError(f"Smoke2 review is missing negatives: {missing[:5]}")
    return categories


def _empty_smoke_review_v2_row(record: FoldRecord) -> dict[str, Any]:
    return {
        "filename": record.filename,
        "label": 0,
        "fold": record.fold,
        "group_id": record.group_id,
        "disposition": None,
        "tags": [],
        "primary_reviewer": None,
        "primary_reviewed_at": None,
        "primary_notes": "",
        "secondary_reviewer": None,
        "secondary_reviewed_at": None,
        "secondary_disposition": None,
        "secondary_notes": "",
        "revision": 0,
        "training_eligible": False,
    }


def _smoke_review_v2_template(
    spec: Round3ReplaySpec,
    records: Sequence[FoldRecord],
    *,
    migration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    negatives = sorted(
        (record for record in records if record.label == 0), key=lambda item: item.filename
    )
    if len(negatives) != spec.expected_negative_count:
        raise TrainingConfigurationError(
            f"Smoke2 expects {spec.expected_negative_count} negatives, found {len(negatives)}"
        )
    return {
        "schema_version": 2,
        "kind": "round3_smoke_negative_semantic_review",
        "status": "pending",
        "finalized_by": None,
        "finalized_at": None,
        "folds": _artifact_record(spec.folds_path, spec.project_root),
        "expected_negative_count": spec.expected_negative_count,
        "allowed_dispositions": list(SMOKE_REVIEW_V2_DISPOSITIONS),
        "allowed_tags": list(SMOKE_REVIEW_V2_TAGS),
        "migration": None if migration is None else dict(migration),
        "records": [_empty_smoke_review_v2_row(record) for record in negatives],
    }


def create_smoke_review_v2_template(
    spec: Round3ReplaySpec,
    records: Sequence[FoldRecord],
) -> dict[str, Any]:
    """Create a separate pending v2 template without touching any v1 review."""

    if spec.smoke_review_policy != SMOKE_REVIEW_V2_POLICY:
        raise TrainingConfigurationError(
            "Smoke2 v2 template requires the isolated v2 replay configuration"
        )
    if spec.smoke_review.exists():
        raise FileExistsError(f"refusing to overwrite Smoke2 v2 review: {spec.smoke_review}")
    payload = _smoke_review_v2_template(spec, records)
    validate_smoke_review_v2_draft(
        payload,
        records,
        folds_path=spec.folds_path,
        project_root=spec.project_root,
        expected_negative_count=spec.expected_negative_count,
    )
    write_json_atomic(spec.smoke_review, payload)
    return payload


def _validate_legacy_smoke_review_for_migration(
    payload: Mapping[str, Any],
    records: Sequence[FoldRecord],
    *,
    folds_path: Path,
    project_root: Path,
    expected_negative_count: int,
) -> dict[str, int]:
    if payload.get("schema_version") != 1 or payload.get("kind") != (
        "round3_smoke_negative_semantic_review"
    ):
        raise TrainingConfigurationError("legacy Smoke2 review schema/kind mismatch")
    if payload.get("status") not in {"pending", "approved"}:
        raise TrainingConfigurationError("legacy Smoke2 bundle status is invalid")
    reviewer = payload.get("reviewer")
    if reviewer is not None and not isinstance(reviewer, str):
        raise TrainingConfigurationError("legacy Smoke2 reviewer must be text or null")
    reviewed_at = payload.get("reviewed_at")
    if payload.get("status") == "pending" and reviewed_at is not None:
        raise TrainingConfigurationError("pending legacy Smoke2 bundle must have reviewed_at=null")
    if payload.get("folds") != _artifact_record(folds_path, project_root):
        raise TrainingConfigurationError("legacy Smoke2 review Fold provenance drifted")
    if payload.get("expected_negative_count") != expected_negative_count:
        raise TrainingConfigurationError("legacy Smoke2 expected_negative_count mismatch")
    if payload.get("allowed_categories") != list(SMOKE_CATEGORIES):
        raise TrainingConfigurationError("legacy Smoke2 categories drifted")
    expected = {record.filename: record for record in records if record.label == 0}
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != expected_negative_count:
        raise TrainingConfigurationError(
            f"legacy Smoke2 review must contain exactly {expected_negative_count} rows"
        )
    counts = {"approved": 0, "uncertain": 0, "pending": 0}
    seen: set[str] = set()
    for offset, raw in enumerate(rows):
        row = _mapping(raw, f"legacy Smoke2 review.records[{offset}]")
        filename = _string(row.get("filename"), f"legacy records[{offset}].filename")
        record = expected.get(filename)
        if record is None or filename in seen:
            raise TrainingConfigurationError(
                f"legacy Smoke2 review contains unknown/duplicate filename: {filename}"
            )
        if (
            type(row.get("label")) is not int
            or row.get("label") != 0
            or type(row.get("fold")) is not int
            or row.get("fold") != record.fold
            or row.get("group_id") != record.group_id
        ):
            raise TrainingConfigurationError(
                f"legacy Smoke2 review provenance mismatch for {filename}"
            )
        status = row.get("review_status")
        if status not in counts:
            raise TrainingConfigurationError(
                f"legacy Smoke2 row {filename} has invalid status {status!r}"
            )
        category = row.get("category")
        if category is not None and category not in SMOKE_CATEGORIES:
            raise TrainingConfigurationError(
                f"legacy Smoke2 row {filename} has invalid category {category!r}"
            )
        if status == "approved" and category is None:
            raise TrainingConfigurationError(
                f"legacy approved Smoke2 row {filename} has no category"
            )
        if not isinstance(row.get("notes"), str):
            raise TrainingConfigurationError(f"legacy Smoke2 notes must be text for {filename}")
        counts[str(status)] += 1
        seen.add(filename)
    if seen != set(expected):
        raise TrainingConfigurationError("legacy Smoke2 review does not cover the negative set")
    if payload.get("status") == "approved":
        validate_smoke_review(
            payload,
            records,
            folds_path=folds_path,
            project_root=project_root,
            expected_negative_count=expected_negative_count,
        )
    return counts


def migrate_smoke_review_v1_to_v2(
    spec: Round3ReplaySpec,
    records: Sequence[FoldRecord],
    *,
    legacy_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a pending v2 bundle while preserving, but never asserting, v1 decisions.

    V1 lacks per-row reviewer identity and timestamps.  Consequently, migration binds
    its provenance for audit but deliberately leaves every v2 row pending for a fresh
    explicit primary review.  No category-to-tag or disposition inference is performed.
    """

    if spec.smoke_review_policy != SMOKE_REVIEW_V2_POLICY:
        raise TrainingConfigurationError(
            "Smoke2 v2 migration requires the isolated v2 replay configuration"
        )
    source = legacy_path.resolve()
    legacy = _load_json_object(source, "legacy Smoke2 semantic review")
    counts = _validate_legacy_smoke_review_for_migration(
        legacy,
        records,
        folds_path=spec.folds_path,
        project_root=spec.project_root,
        expected_negative_count=spec.expected_negative_count,
    )
    migration = {
        "source_review": _artifact_record(source, spec.project_root),
        "policy": "provenance_only_all_rows_require_fresh_v2_human_review",
        "semantic_inference": "forbidden",
    }
    payload = _smoke_review_v2_template(spec, records, migration=migration)
    summary = validate_smoke_review_v2_draft(
        payload,
        records,
        folds_path=spec.folds_path,
        project_root=spec.project_root,
        expected_negative_count=spec.expected_negative_count,
    )
    report = {
        "schema_version": 1,
        "kind": "round3_smoke_review_v1_to_v2_migration",
        "source_review": migration["source_review"],
        "target_review": _repository_relative(spec.smoke_review, spec.project_root),
        "policy": migration["policy"],
        "legacy_status_counts": counts,
        "migrated_human_decisions": 0,
        "rows_requiring_primary_review": summary.pending,
        "semantic_inference": "forbidden",
    }
    return payload, report


def _review_timestamp(value: Any, field: str) -> str:
    raw = _string(value, field)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TrainingConfigurationError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TrainingConfigurationError(f"{field} must include a timezone")
    return raw


def validate_smoke_review_v2_draft(
    payload: Mapping[str, Any],
    records: Sequence[FoldRecord],
    *,
    folds_path: Path,
    project_root: Path,
    expected_negative_count: int,
) -> SmokeReviewV2Summary:
    """Validate v2 provenance, human audit fields, completeness, and derived eligibility."""

    if payload.get("schema_version") != 2 or payload.get("kind") != (
        "round3_smoke_negative_semantic_review"
    ):
        raise TrainingConfigurationError("Smoke2 v2 review schema/kind mismatch")
    bundle_status = payload.get("status")
    if bundle_status not in {"pending", "approved"}:
        raise TrainingConfigurationError("Smoke2 v2 bundle status must be pending or approved")
    if payload.get("folds") != _artifact_record(folds_path, project_root):
        raise TrainingConfigurationError("Smoke2 v2 review Fold provenance drifted")
    if payload.get("expected_negative_count") != expected_negative_count:
        raise TrainingConfigurationError("Smoke2 v2 expected_negative_count mismatch")
    if payload.get("allowed_dispositions") != list(SMOKE_REVIEW_V2_DISPOSITIONS):
        raise TrainingConfigurationError("Smoke2 v2 allowed_dispositions drifted")
    if payload.get("allowed_tags") != list(SMOKE_REVIEW_V2_TAGS):
        raise TrainingConfigurationError("Smoke2 v2 allowed_tags drifted")
    migration = payload.get("migration")
    if migration is not None:
        migration_map = _mapping(migration, "Smoke2 v2 migration")
        if (
            migration_map.get("policy")
            != ("provenance_only_all_rows_require_fresh_v2_human_review")
            or migration_map.get("semantic_inference") != "forbidden"
        ):
            raise TrainingConfigurationError("Smoke2 v2 migration policy drifted")
        _bound_artifact(
            migration_map.get("source_review"),
            project_root=project_root,
            field="Smoke2 v2 migration.source_review",
        )
    finalized_by = payload.get("finalized_by")
    finalized_at = payload.get("finalized_at")
    if bundle_status == "pending":
        if finalized_by is not None or finalized_at is not None:
            raise TrainingConfigurationError(
                "pending Smoke2 v2 bundle must not contain finalization fields"
            )
    else:
        _string(finalized_by, "Smoke2 v2 finalized_by")
        _review_timestamp(finalized_at, "Smoke2 v2 finalized_at")

    expected = {record.filename: record for record in records if record.label == 0}
    if len(expected) != expected_negative_count:
        raise TrainingConfigurationError(
            "fixed Fold file contains "
            f"{len(expected)} negatives, expected {expected_negative_count}"
        )
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != expected_negative_count:
        raise TrainingConfigurationError(
            f"Smoke2 v2 review must contain exactly {expected_negative_count} rows"
        )

    pending = 0
    secondary_required = 0
    complete = 0
    eligible_count = 0
    excluded = 0
    final_counts = {value: 0 for value in SMOKE_REVIEW_V2_DISPOSITIONS}
    tag_counts = {value: 0 for value in SMOKE_REVIEW_V2_TAGS}
    fold_counts: dict[str, dict[str, int]] = {}
    eligible_tags: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    tag_order = {tag: index for index, tag in enumerate(SMOKE_REVIEW_V2_TAGS)}
    for offset, raw in enumerate(rows):
        row = _mapping(raw, f"Smoke2 v2 records[{offset}]")
        filename = _string(row.get("filename"), f"Smoke2 v2 records[{offset}].filename")
        record = expected.get(filename)
        if record is None or filename in seen:
            raise TrainingConfigurationError(
                f"Smoke2 v2 contains unknown/duplicate filename: {filename}"
            )
        if (
            type(row.get("label")) is not int
            or row.get("label") != 0
            or type(row.get("fold")) is not int
            or row.get("fold") != record.fold
            or row.get("group_id") != record.group_id
        ):
            raise TrainingConfigurationError(f"Smoke2 v2 provenance mismatch for {filename}")
        revision = row.get("revision")
        if type(revision) is not int or revision < 0:
            raise TrainingConfigurationError(
                f"Smoke2 v2 revision must be a non-negative integer for {filename}"
            )
        tags_raw = row.get("tags")
        if not isinstance(tags_raw, list) or any(not isinstance(tag, str) for tag in tags_raw):
            raise TrainingConfigurationError(f"Smoke2 v2 tags must be a list for {filename}")
        if len(tags_raw) != len(set(tags_raw)) or any(tag not in tag_order for tag in tags_raw):
            raise TrainingConfigurationError(f"Smoke2 v2 tags are invalid for {filename}")
        tags = tuple(sorted(tags_raw, key=tag_order.__getitem__))
        if list(tags) != tags_raw:
            raise TrainingConfigurationError(f"Smoke2 v2 tags are not canonical for {filename}")
        primary_notes = row.get("primary_notes")
        secondary_notes = row.get("secondary_notes")
        if not isinstance(primary_notes, str) or not isinstance(secondary_notes, str):
            raise TrainingConfigurationError(f"Smoke2 v2 notes must be text for {filename}")
        disposition = row.get("disposition")
        secondary = row.get("secondary_disposition")
        training_eligible = row.get("training_eligible")
        if type(training_eligible) is not bool:
            raise TrainingConfigurationError(
                f"Smoke2 v2 training_eligible must be boolean for {filename}"
            )

        row_complete = False
        final_disposition: str | None = None
        if disposition is None:
            if (
                tags
                or row.get("primary_reviewer") is not None
                or row.get("primary_reviewed_at") is not None
                or primary_notes
                or row.get("secondary_reviewer") is not None
                or row.get("secondary_reviewed_at") is not None
                or secondary is not None
                or secondary_notes
                or revision != 0
            ):
                raise TrainingConfigurationError(
                    f"pending Smoke2 v2 row contains partial review data: {filename}"
                )
            pending += 1
        else:
            if disposition not in SMOKE_REVIEW_V2_DISPOSITIONS:
                raise TrainingConfigurationError(
                    f"Smoke2 v2 row {filename} has invalid disposition {disposition!r}"
                )
            _string(row.get("primary_reviewer"), f"Smoke2 v2 {filename}.primary_reviewer")
            _review_timestamp(
                row.get("primary_reviewed_at"),
                f"Smoke2 v2 {filename}.primary_reviewed_at",
            )
            if revision < 1 or not tags:
                raise TrainingConfigurationError(
                    f"reviewed Smoke2 v2 row requires revision >= 1 and tags: {filename}"
                )
            if disposition != "confirmed_no_fire" and not primary_notes.strip():
                raise TrainingConfigurationError(
                    f"uncertain/label-error Smoke2 v2 row requires primary notes: {filename}"
                )
            secondary_values = (
                row.get("secondary_reviewer"),
                row.get("secondary_reviewed_at"),
                secondary,
            )
            secondary_absent = (
                all(value is None for value in secondary_values) and not secondary_notes
            )
            if disposition == "confirmed_no_fire":
                if not secondary_absent:
                    raise TrainingConfigurationError(
                        f"confirmed Smoke2 v2 row must not contain secondary review: {filename}"
                    )
                row_complete = True
                final_disposition = disposition
            elif secondary_absent:
                secondary_required += 1
            else:
                secondary_reviewer = _string(
                    row.get("secondary_reviewer"),
                    f"Smoke2 v2 {filename}.secondary_reviewer",
                )
                primary_reviewer = _string(
                    row.get("primary_reviewer"),
                    f"Smoke2 v2 {filename}.primary_reviewer",
                )
                if secondary_reviewer == primary_reviewer:
                    raise TrainingConfigurationError(
                        f"Smoke2 v2 secondary reviewer must differ for {filename}"
                    )
                _review_timestamp(
                    row.get("secondary_reviewed_at"),
                    f"Smoke2 v2 {filename}.secondary_reviewed_at",
                )
                if secondary not in SMOKE_REVIEW_V2_DISPOSITIONS:
                    raise TrainingConfigurationError(
                        f"Smoke2 v2 row {filename} has invalid secondary disposition"
                    )
                if not secondary_notes.strip():
                    raise TrainingConfigurationError(
                        f"Smoke2 v2 secondary review requires notes: {filename}"
                    )
                if revision < 2:
                    raise TrainingConfigurationError(
                        f"second-reviewed Smoke2 v2 row requires revision >= 2: {filename}"
                    )
                row_complete = True
                final_disposition = str(secondary)

        expected_eligible = (
            bundle_status == "approved"
            and row_complete
            and final_disposition == "confirmed_no_fire"
        )
        if training_eligible != expected_eligible:
            raise TrainingConfigurationError(
                f"Smoke2 v2 training_eligible is not derived correctly for {filename}"
            )
        fold_key = str(record.fold)
        fold_summary = fold_counts.setdefault(
            fold_key,
            {"total": 0, "complete": 0, "training_eligible": 0, "excluded": 0},
        )
        fold_summary["total"] += 1
        for tag in tags:
            tag_counts[tag] += 1
        if row_complete and final_disposition is not None:
            complete += 1
            fold_summary["complete"] += 1
            final_counts[final_disposition] += 1
            if training_eligible:
                eligible_count += 1
                fold_summary["training_eligible"] += 1
                eligible_tags[filename] = tags
            elif final_disposition != "confirmed_no_fire":
                excluded += 1
                fold_summary["excluded"] += 1
        seen.add(filename)

    if seen != set(expected):
        raise TrainingConfigurationError("Smoke2 v2 review does not cover the negative set")
    if bundle_status == "approved" and pending + secondary_required:
        raise TrainingConfigurationError(
            "approved Smoke2 v2 bundle must have no pending/secondary-required rows"
        )
    return SmokeReviewV2Summary(
        total=len(rows),
        pending=pending,
        secondary_required=secondary_required,
        complete=complete,
        training_eligible=eligible_count,
        excluded=excluded,
        final_dispositions=final_counts,
        tags=tag_counts,
        folds={key: fold_counts[key] for key in sorted(fold_counts, key=int)},
        eligible_tags=eligible_tags,
    )


def validate_smoke_review_v2(
    payload: Mapping[str, Any],
    records: Sequence[FoldRecord],
    *,
    folds_path: Path,
    project_root: Path,
    expected_negative_count: int,
) -> dict[str, tuple[str, ...]]:
    """Require a complete approved v2 bundle and return only eligible negative rows."""

    summary = validate_smoke_review_v2_draft(
        payload,
        records,
        folds_path=folds_path,
        project_root=project_root,
        expected_negative_count=expected_negative_count,
    )
    if payload.get("status") != "approved":
        raise TrainingConfigurationError("Smoke2 v2 review is not approved; replay remains blocked")
    if summary.remaining:
        raise TrainingConfigurationError("Smoke2 v2 review is incomplete")
    return summary.eligible_tags


def select_hp2_samples(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[FoldRecord],
    *,
    outer_fold: int,
    thresholds_by_fold: Mapping[int, float],
) -> tuple[ReplaySelection, ...]:
    """Select inner-OOF false negatives, one lowest-margin image per scene group."""

    _require_outer_fold(outer_fold)
    expected = {record.filename: record for record in records if record.fold != outer_fold}
    training_folds = set(EXPECTED_FOLDS - {outer_fold})
    if set(thresholds_by_fold) != training_folds:
        raise TrainingConfigurationError("HP2 thresholds must cover all four outer-training Folds")
    seen: set[str] = set()
    candidates: list[ReplaySelection] = []
    for raw in rows:
        filename = raw.get("filename")
        if not isinstance(filename, str) or filename not in expected or filename in seen:
            raise TrainingConfigurationError(f"HP2 contains unexpected/duplicate row {filename!r}")
        record = expected[filename]
        if (
            raw.get("label") != record.label
            or raw.get("fold") != record.fold
            or raw.get("group_id") != record.group_id
            or raw.get("outer_fold") != outer_fold
        ):
            raise TrainingConfigurationError(f"HP2 OOF provenance mismatch for {filename}")
        score_raw = raw.get("cls_probability")
        if isinstance(score_raw, bool) or not isinstance(score_raw, (int, float)):
            raise TrainingConfigurationError(f"HP2 score is not numeric for {filename}")
        score = float(score_raw)
        threshold = float(thresholds_by_fold[record.fold])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise TrainingConfigurationError(f"HP2 score is invalid for {filename}")
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise TrainingConfigurationError(f"HP2 threshold is invalid for Fold {record.fold}")
        if record.label == 1 and score < threshold:
            candidates.append(
                ReplaySelection(
                    record=record,
                    source="hp2",
                    evidence={
                        "score": score,
                        "threshold": threshold,
                        "margin": score - threshold,
                        "selection": "lowest_margin_then_filename_per_group",
                    },
                )
            )
        seen.add(filename)
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise TrainingConfigurationError(f"HP2 OOF is incomplete: {missing[:5]}")

    selected: list[ReplaySelection] = []
    by_group: dict[str, list[ReplaySelection]] = {}
    for candidate in candidates:
        by_group.setdefault(candidate.record.group_id, []).append(candidate)
    for group_id in sorted(by_group):
        selected.append(
            min(
                by_group[group_id],
                key=lambda item: (float(item.evidence["margin"]), item.record.filename),
            )
        )
    return tuple(sorted(selected, key=lambda item: item.record.filename))


def select_small2_samples(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[FoldRecord],
    *,
    outer_fold: int,
    all_boxes_small_threshold: float,
    max_gt_box_ratio: float,
) -> tuple[ReplaySelection, ...]:
    """Select all-boxes-small positives below 0.0005, one image per group."""

    _require_outer_fold(outer_fold)
    if not math.isclose(all_boxes_small_threshold, 0.001, abs_tol=0.0):
        raise TrainingConfigurationError("Small2 all_boxes_small threshold must be 0.001")
    if not math.isclose(max_gt_box_ratio, 0.0005, abs_tol=0.0):
        raise TrainingConfigurationError("Small2 max_gt_box_ratio must be 0.0005")
    expected = {record.filename: record for record in records}
    summaries: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        filename = raw.get("filename")
        if not isinstance(filename, str) or filename not in expected or filename in summaries:
            raise TrainingConfigurationError(
                f"Small2 GT summary contains unexpected/duplicate filename {filename!r}"
            )
        summaries[filename] = raw
    if set(summaries) != set(expected):
        missing = sorted(set(expected) - set(summaries))
        raise TrainingConfigurationError(f"Small2 GT summary is incomplete: {missing[:5]}")

    candidates: list[ReplaySelection] = []
    for filename, record in expected.items():
        row = summaries[filename]
        count = row.get("gt_box_count")
        ratio_raw = row.get("max_gt_box_ratio")
        all_small = row.get("all_boxes_small")
        if type(count) is not int or count < 0 or type(all_small) is not bool:
            raise TrainingConfigurationError(f"Small2 GT summary is malformed for {filename}")
        if isinstance(ratio_raw, bool) or not isinstance(ratio_raw, (int, float)):
            raise TrainingConfigurationError(f"Small2 ratio is malformed for {filename}")
        ratio = float(ratio_raw)
        if not math.isfinite(ratio) or ratio < 0.0:
            raise TrainingConfigurationError(f"Small2 ratio is invalid for {filename}")
        if record.label == 0 and (count != 0 or all_small or ratio != 0.0):
            raise TrainingConfigurationError(
                f"negative image has GT fire boxes in Small2: {filename}"
            )
        if (
            record.fold != outer_fold
            and record.label == 1
            and all_small
            and count > 0
            and ratio < max_gt_box_ratio
        ):
            candidates.append(
                ReplaySelection(
                    record=record,
                    source="small2",
                    evidence={
                        "all_boxes_small": True,
                        "all_boxes_small_threshold": all_boxes_small_threshold,
                        "max_gt_box_ratio": ratio,
                        "max_gt_box_ratio_limit": max_gt_box_ratio,
                        "selection": "smallest_max_box_ratio_then_filename_per_group",
                    },
                )
            )
    by_group: dict[str, list[ReplaySelection]] = {}
    for candidate in candidates:
        by_group.setdefault(candidate.record.group_id, []).append(candidate)
    selected = [
        min(
            by_group[group_id],
            key=lambda item: (float(item.evidence["max_gt_box_ratio"]), item.record.filename),
        )
        for group_id in sorted(by_group)
    ]
    return tuple(sorted(selected, key=lambda item: item.record.filename))


def select_smoke2_samples(
    review_evidence: Mapping[str, str | Sequence[str]],
    records: Sequence[FoldRecord],
    *,
    outer_fold: int,
) -> tuple[ReplaySelection, ...]:
    """Select one deterministic representative per approved outer-train negative group."""

    _require_outer_fold(outer_fold)
    expected_negative_names = {record.filename for record in records if record.label == 0}
    if not set(review_evidence) <= expected_negative_names:
        raise TrainingConfigurationError("Smoke2 review contains a non-negative/unknown row")
    by_group: dict[str, list[FoldRecord]] = {}
    for record in records:
        if record.label == 0 and record.fold != outer_fold and record.filename in review_evidence:
            by_group.setdefault(record.group_id, []).append(record)
    selected: list[ReplaySelection] = []
    for group_id in sorted(by_group):
        record = min(by_group[group_id], key=lambda item: item.filename)
        review_value = review_evidence[record.filename]
        human_evidence: dict[str, Any]
        if isinstance(review_value, str):
            human_evidence = {"category": review_value, "review_status": "approved"}
        else:
            human_evidence = {
                "tags": list(review_value),
                "review_schema": 2,
                "bundle_status": "approved",
                "training_eligible": True,
            }
        selected.append(
            ReplaySelection(
                record=record,
                source="smoke2",
                evidence={
                    **human_evidence,
                    "selection": "filename_ascending_per_group",
                },
            )
        )
    return tuple(sorted(selected, key=lambda item: item.record.filename))


def merge_replay_selections(
    selections: Mapping[str, Sequence[ReplaySelection]],
    records: Sequence[FoldRecord],
    *,
    outer_fold: int,
) -> tuple[dict[str, Any], ...]:
    """Merge source selections by filename while retaining all source evidence."""

    _require_outer_fold(outer_fold)
    by_name = {record.filename: record for record in records}
    held_out_groups = {record.group_id for record in records if record.fold == outer_fold}
    merged: dict[str, dict[str, Any]] = {}
    for source in sorted(selections):
        seen_source: set[str] = set()
        for selection in selections[source]:
            record = selection.record
            if selection.source != source:
                raise TrainingConfigurationError("Round-3 replay source/evidence mismatch")
            if record.filename in seen_source:
                raise TrainingConfigurationError(
                    f"duplicate {source} replay filename: {record.filename}"
                )
            expected = by_name.get(record.filename)
            if expected != record or record.fold == outer_fold:
                raise TrainingConfigurationError(
                    f"{source} selected held-out/unknown replay image {record.filename}"
                )
            if record.group_id in held_out_groups:
                raise TrainingConfigurationError(
                    f"{source} selected a group overlapping outer validation: {record.filename}"
                )
            row = merged.setdefault(
                record.filename,
                {
                    "filename": record.filename,
                    "label": record.label,
                    "fold": record.fold,
                    "group_id": record.group_id,
                    "sources": [],
                    "evidence": {},
                },
            )
            row["sources"].append(source)
            row["evidence"][source] = dict(selection.evidence)
            seen_source.add(record.filename)
    for row in merged.values():
        row["sources"].sort()
        row["evidence"] = {source: row["evidence"][source] for source in sorted(row["evidence"])}
    return tuple(merged[name] for name in sorted(merged))


def build_hard_sample_replay_indices(
    samples: Sequence[Sequence[Any]],
    replay_samples: Sequence[tuple[str, int]],
    *,
    repeat_factor: int,
) -> tuple[int, ...]:
    """Append exactly one logical index per merged replay sample."""

    if repeat_factor != 2:
        raise TrainingConfigurationError("Round-3 hard-sample repeat_factor must be exactly 2")
    by_name: dict[str, tuple[int, int]] = {}
    for index, dataset_sample in enumerate(samples):
        if len(dataset_sample) < 2:
            raise TrainingConfigurationError("classification samples must expose path and class")
        filename = Path(str(dataset_sample[0])).name
        try:
            label = int(dataset_sample[1])
        except (TypeError, ValueError) as exc:
            raise TrainingConfigurationError(f"invalid class index for {filename}") from exc
        if filename in by_name:
            raise TrainingConfigurationError(f"duplicate classifier training filename: {filename}")
        by_name[filename] = (index, label)
    declared: dict[str, int] = {}
    for filename, label in replay_samples:
        if filename in declared:
            raise TrainingConfigurationError(f"duplicate merged replay filename: {filename}")
        if label not in {0, 1}:
            raise TrainingConfigurationError(f"invalid replay label for {filename}")
        declared[filename] = label
    replay_indices: list[int] = []
    for filename in sorted(declared):
        indexed_sample = by_name.get(filename)
        if indexed_sample is None:
            raise TrainingConfigurationError(
                f"hard-sample replay image is absent from the training split: {filename}"
            )
        index, actual_label = indexed_sample
        if actual_label != declared[filename]:
            raise TrainingConfigurationError(
                f"hard-sample replay class mismatch for {filename}: "
                f"expected {declared[filename]}, found {actual_label}"
            )
        replay_indices.append(index)
    return tuple([*range(len(samples)), *replay_indices])


def _validated_hn2_context(
    spec: Round3ReplaySpec,
    records: Sequence[FoldRecord],
    *,
    outer_fold: int,
) -> tuple[dict[str, Any], tuple[str, ...], list[dict[str, Any]], dict[int, float], Path, Path]:
    hn2_path = spec.hn2_manifest(outer_fold)
    contract = load_hard_negative_replay(
        path=hn2_path,
        repeat_factor=2,
        validation_fold=outer_fold,
        records=records,
        folds_path=spec.folds_path,
        processed_manifest=spec.processed_manifest,
        project_root=spec.project_root,
        expected_seed=spec.seed,
    )
    payload = _load_json_object(hn2_path, "clean HN2 manifest")
    nested_oof = _bound_artifact(
        payload.get("nested_oof"), project_root=spec.project_root, field="HN2 nested_oof"
    )
    campaign = _bound_artifact(
        payload.get("nested_campaign_manifest"),
        project_root=spec.project_root,
        field="HN2 nested_campaign_manifest",
    )
    rows = validate_nested_oof_rows(
        load_nested_oof_csv(nested_oof),
        records,
        outer_fold=outer_fold,
        folds_path=spec.folds_path,
        processed_manifest=spec.processed_manifest,
        campaign_manifest=campaign,
        expected_seed=spec.seed,
        artifact_root=spec.project_root,
    )
    thresholds_raw = payload.get("per_inner_fold_thresholds")
    if not isinstance(thresholds_raw, list):
        raise TrainingConfigurationError("HN2 per_inner_fold_thresholds must be a list")
    thresholds: dict[int, float] = {}
    for raw in thresholds_raw:
        row = _mapping(raw, "HN2 threshold row")
        fold = row.get("validation_fold")
        value = row.get("threshold")
        if type(fold) is not int or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TrainingConfigurationError("HN2 threshold row is malformed")
        thresholds[fold] = float(value)
    return payload, contract.hard_negative_filenames, rows, thresholds, nested_oof, campaign


def _hn2_selections(
    names: Sequence[str],
    records: Sequence[FoldRecord],
    *,
    outer_fold: int,
    manifest_sha256: str,
) -> tuple[ReplaySelection, ...]:
    by_name = {record.filename: record for record in records}
    selections: list[ReplaySelection] = []
    for filename in sorted(names):
        record = by_name.get(filename)
        if record is None or record.label != 0 or record.fold == outer_fold:
            raise TrainingConfigurationError(f"invalid clean HN2 selection: {filename}")
        selections.append(
            ReplaySelection(
                record=record,
                source="hn2",
                evidence={
                    "source_manifest_sha256": manifest_sha256,
                    "selection": "validated_clean_hn2_cross_fitted_fp",
                },
            )
        )
    return tuple(selections)


def build_round3_replay_payload(
    spec: Round3ReplaySpec,
    records: Sequence[FoldRecord],
    *,
    variant: str,
    outer_fold: int,
) -> dict[str, Any]:
    """Rebuild one deterministic Round-3 replay manifest from frozen source artifacts."""

    sources = _variant_sources(spec, variant)
    _require_outer_fold(outer_fold)
    if len(records) != spec.expected_count:
        raise TrainingConfigurationError(
            f"Round-3 replay expects {spec.expected_count} Fold rows, found {len(records)}"
        )
    validation = [record for record in records if record.fold == outer_fold]
    training = [record for record in records if record.fold != outer_fold]
    if len(validation) != spec.expected_validation_count:
        raise TrainingConfigurationError(
            f"outer Fold {outer_fold} must contain {spec.expected_validation_count} unique rows"
        )
    group_folds: dict[str, set[int]] = {}
    for record in records:
        group_folds.setdefault(record.group_id, set()).add(record.fold)
    crossing = sorted(group for group, folds in group_folds.items() if len(folds) != 1)
    if crossing:
        raise TrainingConfigurationError(f"scene groups cross fixed Folds: {crossing[:5]}")

    hn_payload, hn_names, nested_rows, thresholds, nested_oof, campaign = _validated_hn2_context(
        spec, records, outer_fold=outer_fold
    )
    hn2_path = spec.hn2_manifest(outer_fold)
    selected: dict[str, Sequence[ReplaySelection]] = {
        "hn2": _hn2_selections(
            hn_names,
            records,
            outer_fold=outer_fold,
            manifest_sha256=sha256_file(hn2_path),
        )
    }
    policies: dict[str, Any] = {
        "hn2": {
            "selection": "validated_clean_hn2_cross_fitted_negative_fp",
            "repeat_factor": 2,
        }
    }
    inputs: dict[str, Any] = {
        "campaign_config": _artifact_record(spec.config_path, spec.project_root),
        "folds": _artifact_record(spec.folds_path, spec.project_root),
        "processed_manifest": _artifact_record(spec.processed_manifest, spec.project_root),
        "hn2_manifest": _artifact_record(hn2_path, spec.project_root),
        "nested_oof": _artifact_record(nested_oof, spec.project_root),
        "nested_campaign_manifest": _artifact_record(campaign, spec.project_root),
    }
    if "hp2" in sources:
        selected["hp2"] = select_hp2_samples(
            nested_rows,
            records,
            outer_fold=outer_fold,
            thresholds_by_fold=thresholds,
        )
        policies["hp2"] = {
            "candidate": "positive inner-OOF false negative",
            "thresholds": "clean HN2 per-inner LOFO thresholds",
            "group_rule": spec.hp2_rule,
        }
    if "small2" in sources:
        try:
            gt_frame = load_gt_box_summary(
                spec.coco_json,
                small_fire_area_ratio=spec.small2_all_boxes_threshold,
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise TrainingConfigurationError(f"cannot build Small2 GT summary: {exc}") from exc
        gt_rows = gt_frame.to_dict("records")
        selected["small2"] = select_small2_samples(
            gt_rows,
            records,
            outer_fold=outer_fold,
            all_boxes_small_threshold=spec.small2_all_boxes_threshold,
            max_gt_box_ratio=spec.small2_max_box_ratio,
        )
        inputs["coco_json"] = _artifact_record(spec.coco_json, spec.project_root)
        policies["small2"] = {
            "candidate": "positive all_boxes_small and max_gt_box_ratio < 0.0005",
            "all_boxes_small_threshold": spec.small2_all_boxes_threshold,
            "max_gt_box_ratio": spec.small2_max_box_ratio,
            "comparison": "strictly_less_than",
            "group_rule": "smallest_max_box_ratio_then_filename",
        }
    if "smoke2" in sources:
        review_payload = _load_json_object(spec.smoke_review, "Smoke2 semantic review")
        if spec.smoke_review_policy == SMOKE_REVIEW_V2_POLICY:
            review_evidence: Mapping[str, str | Sequence[str]] = validate_smoke_review_v2(
                review_payload,
                records,
                folds_path=spec.folds_path,
                project_root=spec.project_root,
                expected_negative_count=spec.expected_negative_count,
            )
            policy_description = "complete v2 bundle approved; only derived training_eligible rows"
        else:
            review_evidence = validate_smoke_review(
                review_payload,
                records,
                folds_path=spec.folds_path,
                project_root=spec.project_root,
                expected_negative_count=spec.expected_negative_count,
            )
            policy_description = "historical v1 all rows and bundle approved"
        selected["smoke2"] = select_smoke2_samples(review_evidence, records, outer_fold=outer_fold)
        inputs["smoke_review"] = _artifact_record(spec.smoke_review, spec.project_root)
        policies["smoke2"] = {
            "candidate": "human-reviewed negative",
            "review_policy": policy_description,
            "category_inference": "forbidden",
            "group_rule": spec.smoke2_group_rule,
        }
    if tuple(sorted(selected)) != tuple(sorted(sources)):
        raise TrainingConfigurationError("Round-3 variant sources were not built exactly")

    merged = merge_replay_selections(selected, records, outer_fold=outer_fold)
    original_counts = _class_counts(training)
    validation_counts = _class_counts(validation)
    replay_counts = {
        "0": sum(int(row["label"]) == 0 for row in merged),
        "1": sum(int(row["label"]) == 1 for row in merged),
    }
    effective_counts = {
        label: original_counts[label] + replay_counts[label] for label in ("0", "1")
    }
    return {
        "schema_version": 1,
        "kind": "round3_hard_sample_replay",
        "experiment_name": spec.experiment_name,
        "seed": spec.seed,
        "variant": variant,
        "sources": list(sources),
        "outer_fold": outer_fold,
        "training_folds": sorted(EXPECTED_FOLDS - {outer_fold}),
        "repeat_factor": 2,
        "maximum_effective_factor": 2,
        "inputs": inputs,
        "source_policies": {source: policies[source] for source in sources},
        "counts": {
            "outer_train_unique": len(training),
            "outer_validation_unique": len(validation),
            "outer_train_class_counts": original_counts,
            "outer_validation_class_counts": validation_counts,
            "selected_by_source_before_merge": {
                source: len(selected[source]) for source in sources
            },
            "merged_replay_unique": len(merged),
            "merged_overlap_count": sum(len(row["sources"]) - 1 for row in merged),
            "replay_class_counts": replay_counts,
            "effective_train_count": len(training) + len(merged),
            "effective_train_class_counts": effective_counts,
        },
        "samples": list(merged),
        "hn2_source_experiment": hn_payload.get("experiment_name"),
    }


def _contract_from_payload(
    spec: Round3ReplaySpec,
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    variant: str,
    outer_fold: int,
) -> HardSampleReplayContract:
    counts = _mapping(payload.get("counts"), "Round-3 replay counts")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise TrainingConfigurationError("Round-3 replay samples must be a list")
    replay_samples = tuple((str(row["filename"]), int(row["label"])) for row in samples)
    return HardSampleReplayContract(
        config_path=spec.config_path,
        config_sha256=sha256_file(spec.config_path),
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        variant=variant,
        outer_fold=outer_fold,
        repeat_factor=2,
        replay_samples=replay_samples,
        original_train_count=int(counts["outer_train_unique"]),
        effective_train_count=int(counts["effective_train_count"]),
        original_train_class_counts=dict(counts["outer_train_class_counts"]),
        effective_train_class_counts=dict(counts["effective_train_class_counts"]),
        validation_unique_count=int(counts["outer_validation_unique"]),
        validation_class_counts=dict(counts["outer_validation_class_counts"]),
    )


def load_hard_sample_replay(
    *,
    config_path: Path,
    manifest_path: Path,
    variant: str,
    repeat_factor: int,
    validation_fold: int,
    records: Sequence[FoldRecord],
) -> HardSampleReplayContract:
    """Strictly rebuild and validate one Round-3 replay manifest before training."""

    spec = load_round3_replay_spec(config_path)
    if repeat_factor != 2:
        raise TrainingConfigurationError("Round-3 hard-sample repeat_factor must be exactly 2")
    canonical = spec.replay_manifest(variant, validation_fold).resolve()
    if manifest_path.resolve() != canonical:
        raise TrainingConfigurationError(
            f"Round-3 replay manifest must be canonical: expected {canonical}"
        )
    actual = _load_json_object(manifest_path, "Round-3 replay manifest")
    expected = build_round3_replay_payload(
        spec, records, variant=variant, outer_fold=validation_fold
    )
    if actual != expected:
        raise TrainingConfigurationError(
            "Round-3 replay manifest differs from deterministic source selection"
        )
    return _contract_from_payload(
        spec,
        actual,
        manifest_path=manifest_path.resolve(),
        variant=variant,
        outer_fold=validation_fold,
    )


def _campaign_payload(spec: Round3ReplaySpec, variant: str) -> dict[str, Any]:
    manifests = []
    for outer_fold in sorted(EXPECTED_FOLDS):
        path = spec.replay_manifest(variant, outer_fold)
        payload = _load_json_object(path, "Round-3 replay manifest")
        manifests.append(
            {
                "outer_fold": outer_fold,
                **_artifact_record(path, spec.project_root),
                "replay_unique": payload["counts"]["merged_replay_unique"],
            }
        )
    return {
        "schema_version": 1,
        "kind": "round3_hard_sample_replay_campaign",
        "experiment_name": spec.experiment_name,
        "variant": variant,
        "seed": spec.seed,
        "fold_count": 5,
        "repeat_factor": 2,
        "campaign_config": _artifact_record(spec.config_path, spec.project_root),
        "folds": _artifact_record(spec.folds_path, spec.project_root),
        "processed_manifest": _artifact_record(spec.processed_manifest, spec.project_root),
        "manifests": manifests,
    }


def prepare_round3_replay_campaign(spec: Round3ReplaySpec, *, variant: str) -> dict[str, Any]:
    """Atomically publish all five deterministic replay manifests for one variant."""

    _variant_sources(spec, variant)
    records = load_fold_records(spec.folds_path, expected_count=spec.expected_count)
    payloads = {
        fold: build_round3_replay_payload(spec, records, variant=variant, outer_fold=fold)
        for fold in sorted(EXPECTED_FOLDS)
    }
    output = spec.output_root / variant
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Round-3 replay campaign: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for fold, payload in payloads.items():
            write_json_atomic(staging / f"outer_{fold}.json", payload)
        staging.replace(output)
        campaign = _campaign_payload(spec, variant)
        write_json_atomic(output / "campaign_manifest.json", campaign)
    except BaseException:
        if output.exists() and not (output / "campaign_manifest.json").exists():
            shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_round3_replay_campaign(spec, variant=variant)


def validate_round3_replay_campaign(spec: Round3ReplaySpec, *, variant: str) -> dict[str, Any]:
    """Rebuild all source selections and validate the five-Fold campaign hashes."""

    records = load_fold_records(spec.folds_path, expected_count=spec.expected_count)
    for outer_fold in sorted(EXPECTED_FOLDS):
        path = spec.replay_manifest(variant, outer_fold)
        load_hard_sample_replay(
            config_path=spec.config_path,
            manifest_path=path,
            variant=variant,
            repeat_factor=2,
            validation_fold=outer_fold,
            records=records,
        )
    campaign_path = spec.campaign_manifest(variant)
    actual = _load_json_object(campaign_path, "Round-3 campaign manifest")
    expected = _campaign_payload(spec, variant)
    if actual != expected:
        raise TrainingConfigurationError("Round-3 campaign manifest provenance drifted")
    return actual
