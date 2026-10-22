# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 ChenJunxin
# Candidate-only notice (2026-10-18): applies to the author's original HN
# research implementation under the author's recorded license decision.
# It does not relicense separately owned Ultralytics or Focal source code.
"""Leakage-safe nested hard-negative mining and deterministic replay contracts."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from fire_detection.metrics import compute_binary_metrics
from fire_detection.models.offline import sha256_file
from fire_detection.models.training import (
    FoldRecord,
    TrainingConfigurationError,
    TrainingSpec,
    _sha256_training_dataset,
    search_image_metrics,
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_FOLDS = frozenset(range(5))
NESTED_OOF_COLUMNS = (
    "filename",
    "label",
    "fold",
    "group_id",
    "outer_fold",
    "cls_probability",
    "model_name",
    "weights_path",
    "weights_sha256",
    "imgsz",
    "seed",
    "folds_sha256",
    "processed_manifest_sha256",
    "nested_split_manifest_sha256",
)


@dataclass(frozen=True)
class HardNegativeMiningSpec:
    """Validated paths and constants for the fixed nested-mining campaign."""

    project_root: Path
    config_path: Path
    experiment_name: str
    seed: int
    folds_path: Path
    processed_manifest: Path
    images_path: Path
    nested_views_dir: Path
    classifier_config: Path
    nested_oof_dir: Path
    hard_manifests_dir: Path
    score_column: str
    threshold_start: float
    threshold_end: float
    threshold_step: float


@dataclass(frozen=True)
class NestedSplitContract:
    """One outer/inner split derived only from immutable Fold-v3 assignments."""

    outer_fold: int
    inner_fold: int
    training_folds: tuple[int, ...]
    train_records: tuple[FoldRecord, ...]
    validation_records: tuple[FoldRecord, ...]
    excluded_records: tuple[FoldRecord, ...]


@dataclass(frozen=True)
class ReplayContract:
    """Validated in-memory replay inputs for one formal outer Fold."""

    manifest_path: Path
    manifest_sha256: str
    outer_fold: int
    repeat_factor: int
    hard_negative_filenames: tuple[str, ...]
    original_train_count: int
    effective_train_count: int
    original_train_class_counts: dict[str, int]
    effective_train_class_counts: dict[str, int]
    validation_unique_count: int
    validation_class_counts: dict[str, int]

    def manifest_section(self, project_root: Path) -> dict[str, Any]:
        """Return stable run-manifest fields for this replay contract."""

        return {
            "manifest": _repository_relative(self.manifest_path, project_root),
            "manifest_sha256": self.manifest_sha256,
            "repeat_factor": self.repeat_factor,
            "hard_negative_unique_count": len(self.hard_negative_filenames),
            "original_train_count": self.original_train_count,
            "effective_train_count": self.effective_train_count,
            "original_train_class_counts": self.original_train_class_counts,
            "effective_train_class_counts": self.effective_train_class_counts,
            "validation_unique_count": self.validation_unique_count,
            "validation_class_counts": self.validation_class_counts,
        }


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


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigurationError(f"{field} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise TrainingConfigurationError(f"{field} must be a finite number in [0, 1]")
    return result


def _path(project_root: Path, value: Any, field: str) -> Path:
    raw = _string(value, field)
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _repository_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _manifest_artifact(section: Mapping[str, Any], *, project_root: Path, field: str) -> Path:
    raw = _string(section.get("path"), f"{field}.path").replace("\\", "/")
    portable = PurePosixPath(raw)
    if portable.is_absolute() or PureWindowsPath(raw).is_absolute() or ".." in portable.parts:
        raise TrainingConfigurationError(f"{field}.path must be repository-relative")
    candidate = (project_root.resolve() / Path(*portable.parts)).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise TrainingConfigurationError(f"{field}.path escapes the project root") from exc
    if not candidate.is_file():
        raise TrainingConfigurationError(f"{field} artifact is missing: {candidate}")
    expected = _require_sha256(section.get("sha256"), f"{field}.sha256")
    if sha256_file(candidate) != expected:
        raise TrainingConfigurationError(f"{field} artifact hash drift: {candidate}")
    return candidate


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value.lower()) is None:
        raise TrainingConfigurationError(f"{field} must be a SHA-256 digest")
    return value.lower()


def _load_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingConfigurationError(f"cannot read {field} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingConfigurationError(f"{field} must be a JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_hard_negative_mining_spec(path: Path) -> HardNegativeMiningSpec:
    """Load the decision-complete nested-mining campaign configuration."""

    config_path = path.resolve()
    project_root = config_path.parent.parent
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TrainingConfigurationError(f"cannot read mining config {config_path}: {exc}") from exc
    root = _mapping(raw, "hard-negative config")
    experiment = _mapping(_required(root, "experiment", "config"), "experiment")
    data = _mapping(_required(root, "data", "config"), "data")
    models = _mapping(_required(root, "models", "config"), "models")
    outputs = _mapping(_required(root, "outputs", "config"), "outputs")
    mining = _mapping(_required(root, "mining", "config"), "mining")
    seed = _integer(_required(experiment, "seed", "experiment"), "experiment.seed")
    if seed != 42:
        raise TrainingConfigurationError("nested hard-negative mining requires seed 42")
    outer_folds = _required(mining, "outer_folds", "mining")
    if outer_folds != [0, 1, 2, 3, 4]:
        raise TrainingConfigurationError("mining.outer_folds must be exactly [0, 1, 2, 3, 4]")
    if _string(_required(mining, "inner_policy", "mining"), "mining.inner_policy") != (
        "remaining_fold_v3"
    ):
        raise TrainingConfigurationError("mining.inner_policy must be remaining_fold_v3")
    score_column = _string(_required(mining, "score_column", "mining"), "mining.score_column")
    if score_column != "cls_probability":
        raise TrainingConfigurationError("mining.score_column must be cls_probability")
    start = _probability(_required(mining, "threshold_start", "mining"), "mining.threshold_start")
    end = _probability(_required(mining, "threshold_end", "mining"), "mining.threshold_end")
    step = _probability(_required(mining, "threshold_step", "mining"), "mining.threshold_step")
    if step <= 0.0 or start > end:
        raise TrainingConfigurationError("invalid nested-mining threshold grid")
    checkpoint = _string(_required(mining, "checkpoint", "mining"), "mining.checkpoint")
    if checkpoint != "last.pt":
        raise TrainingConfigurationError("nested hard-negative mining must use last.pt")
    nested_views_dir = _path(
        project_root,
        _required(data, "nested_views", "data"),
        "data.nested_views",
    )
    processed_root = (project_root / "data/processed").resolve()
    try:
        nested_views_dir.relative_to(processed_root)
    except ValueError as exc:
        raise TrainingConfigurationError(
            "data.nested_views must stay inside the repository data/processed directory"
        ) from exc
    allowed_nested_views = {
        (processed_root / "round2/hard_negative/nested").resolve(),
        (processed_root / "final_clean/hard_negative/nested").resolve(),
    }
    if nested_views_dir not in allowed_nested_views:
        raise TrainingConfigurationError(
            "data.nested_views must be either the historical "
            "data/processed/round2/hard_negative/nested namespace or the isolated "
            "data/processed/final_clean/hard_negative/nested namespace; frozen v3 and "
            "other historical outputs are read-only"
        )
    return HardNegativeMiningSpec(
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
        images_path=_path(project_root, _required(data, "images", "data"), "data.images"),
        nested_views_dir=nested_views_dir,
        classifier_config=_path(
            project_root,
            _required(models, "classifier_config", "models"),
            "models.classifier_config",
        ),
        nested_oof_dir=_path(
            project_root,
            _required(outputs, "nested_oof", "outputs"),
            "outputs.nested_oof",
        ),
        hard_manifests_dir=_path(
            project_root,
            _required(outputs, "hard_manifests", "outputs"),
            "outputs.hard_manifests",
        ),
        score_column=score_column,
        threshold_start=start,
        threshold_end=end,
        threshold_step=step,
    )


def build_nested_split_contract(
    records: Sequence[FoldRecord], *, outer_fold: int, inner_fold: int
) -> NestedSplitContract:
    """Derive one 3-Fold train, 1-Fold inner-val, 1-Fold excluded split."""

    if outer_fold not in EXPECTED_FOLDS or inner_fold not in EXPECTED_FOLDS:
        raise TrainingConfigurationError("outer_fold and inner_fold must be in 0..4")
    if outer_fold == inner_fold:
        raise TrainingConfigurationError("inner_fold must differ from outer_fold")
    if not records:
        raise TrainingConfigurationError("fixed Fold records must not be empty")
    names: set[str] = set()
    group_folds: dict[str, set[int]] = {}
    observed_folds: set[int] = set()
    for record in records:
        if record.filename in names:
            raise TrainingConfigurationError(f"duplicate Fold filename: {record.filename}")
        if record.label not in {0, 1} or record.fold not in EXPECTED_FOLDS:
            raise TrainingConfigurationError(f"invalid fixed Fold row: {record}")
        names.add(record.filename)
        observed_folds.add(record.fold)
        group_folds.setdefault(record.group_id, set()).add(record.fold)
    if observed_folds != set(EXPECTED_FOLDS):
        raise TrainingConfigurationError(
            f"nested mining requires Fold 0..4, found {sorted(observed_folds)}"
        )
    split_groups = sorted(group for group, folds in group_folds.items() if len(folds) != 1)
    if split_groups:
        raise TrainingConfigurationError(f"groups cross fixed Folds: {split_groups[:5]}")
    training_folds = tuple(
        fold for fold in sorted(EXPECTED_FOLDS) if fold not in {outer_fold, inner_fold}
    )
    train = tuple(
        sorted((r for r in records if r.fold in training_folds), key=lambda r: r.filename)
    )
    validation = tuple(
        sorted((r for r in records if r.fold == inner_fold), key=lambda r: r.filename)
    )
    excluded = tuple(sorted((r for r in records if r.fold == outer_fold), key=lambda r: r.filename))
    named_splits = (
        ("train", train),
        ("inner validation", validation),
        ("outer excluded", excluded),
    )
    for name, rows in named_splits:
        if {row.label for row in rows} != {0, 1}:
            raise TrainingConfigurationError(f"nested {name} split must contain both classes")
    if {r.group_id for r in train} & {r.group_id for r in validation + excluded}:
        raise TrainingConfigurationError("nested train groups overlap validation/excluded groups")
    if {r.group_id for r in validation} & {r.group_id for r in excluded}:
        raise TrainingConfigurationError("inner validation groups overlap outer excluded groups")
    return NestedSplitContract(
        outer_fold,
        inner_fold,
        training_folds,
        train,
        validation,
        excluded,
    )


def _processed_image_hashes(
    processed_manifest: Path, records: Sequence[FoldRecord]
) -> dict[str, str]:
    manifest = _load_json_object(processed_manifest, "processed-data manifest")
    rows = manifest.get("images")
    if not isinstance(rows, list):
        raise TrainingConfigurationError("processed-data manifest images must be a list")
    expected = {record.filename: record for record in records}
    result: dict[str, str] = {}
    for offset, raw in enumerate(rows):
        row = _mapping(raw, f"processed images[{offset}]")
        filename = _string(row.get("filename"), f"processed images[{offset}].filename")
        if filename in result:
            raise TrainingConfigurationError(f"duplicate processed image {filename}")
        record = expected.get(filename)
        if record is None:
            raise TrainingConfigurationError(f"unknown processed image {filename}")
        if row.get("label") != record.label or row.get("fold") != record.fold:
            raise TrainingConfigurationError(f"processed label/Fold mismatch for {filename}")
        result[filename] = _require_sha256(
            row.get("processed_sha256"), f"processed image {filename}.processed_sha256"
        )
    if set(result) != set(expected):
        missing = sorted(set(expected) - set(result))
        raise TrainingConfigurationError(f"processed manifest is missing images: {missing[:5]}")
    return result


def _class_counts(records: Sequence[FoldRecord]) -> dict[str, int]:
    return {
        "0": sum(record.label == 0 for record in records),
        "1": sum(record.label == 1 for record in records),
    }


def _split_manifest_payload(
    *,
    spec: HardNegativeMiningSpec,
    contract: NestedSplitContract,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for split, rows in (("train", contract.train_records), ("val", contract.validation_records)):
        records.extend(
            {
                "filename": row.filename,
                "label": row.label,
                "group_id": row.group_id,
                "fold": row.fold,
                "split": split,
                "operation": "hardlink",
                "source_sha256": hashes[row.filename],
            }
            for row in rows
        )
    records.sort(key=lambda row: (str(row["split"]), str(row["filename"])))
    return {
        "schema_version": 1,
        "kind": "nested_classifier_split",
        "seed": spec.seed,
        "outer_fold": contract.outer_fold,
        "inner_fold": contract.inner_fold,
        "training_folds": list(contract.training_folds),
        "validation_folds": [contract.inner_fold],
        "excluded_folds": [contract.outer_fold],
        "folds": {
            "path": _repository_relative(spec.folds_path, spec.project_root),
            "sha256": sha256_file(spec.folds_path),
        },
        "processed_manifest": {
            "path": _repository_relative(spec.processed_manifest, spec.project_root),
            "sha256": sha256_file(spec.processed_manifest),
        },
        "images": {"path": _repository_relative(spec.images_path, spec.project_root)},
        "counts": {
            "train_unique": len(contract.train_records),
            "validation_unique": len(contract.validation_records),
            "excluded_unique": len(contract.excluded_records),
            "train_class_counts": _class_counts(contract.train_records),
            "validation_class_counts": _class_counts(contract.validation_records),
        },
        "records": records,
    }


def prepare_nested_classification_views(
    spec: HardNegativeMiningSpec, records: Sequence[FoldRecord]
) -> dict[str, Any]:
    """Atomically create all 20 hard-linked nested classification views."""

    try:
        spec.nested_views_dir.resolve().relative_to(spec.project_root.resolve())
    except ValueError as exc:
        raise TrainingConfigurationError(
            "nested views output must stay inside the project root"
        ) from exc
    frozen_v3 = (spec.project_root / "data/processed/v3").resolve()
    try:
        spec.nested_views_dir.resolve().relative_to(frozen_v3)
    except ValueError:
        pass
    else:
        raise TrainingConfigurationError(
            "nested views must not be written inside the frozen data/processed/v3 tree"
        )
    if spec.nested_views_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite nested classification views: {spec.nested_views_dir}"
        )
    if not spec.images_path.is_dir():
        raise TrainingConfigurationError(
            f"processed image directory is missing: {spec.images_path}"
        )
    hashes = _processed_image_hashes(spec.processed_manifest, records)
    staging = spec.nested_views_dir.parent / (
        f".{spec.nested_views_dir.name}.building-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    splits: list[dict[str, Any]] = []
    try:
        for outer_fold in sorted(EXPECTED_FOLDS):
            for inner_fold in sorted(EXPECTED_FOLDS - {outer_fold}):
                contract = build_nested_split_contract(
                    records, outer_fold=outer_fold, inner_fold=inner_fold
                )
                dataset = staging / f"outer_{outer_fold}" / f"inner_{inner_fold}"
                for split, split_records in (
                    ("train", contract.train_records),
                    ("val", contract.validation_records),
                ):
                    for record in split_records:
                        source = spec.images_path / record.filename
                        if not source.is_file() or sha256_file(source) != hashes[record.filename]:
                            raise TrainingConfigurationError(
                                f"processed image is missing or changed: {source}"
                            )
                        destination = dataset / split / str(record.label) / record.filename
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.link(source, destination)
                        except OSError as exc:
                            raise TrainingConfigurationError(
                                f"cannot create hard-linked nested view {destination}: {exc}"
                            ) from exc
                        if not os.path.samefile(source, destination):
                            raise TrainingConfigurationError(
                                "nested view is not a hard link to its processed source: "
                                f"{destination}"
                            )
                split_manifest = dataset / "split_manifest.json"
                _write_json(
                    split_manifest,
                    _split_manifest_payload(spec=spec, contract=contract, hashes=hashes),
                )
                relative_dataset = (
                    spec.nested_views_dir / f"outer_{outer_fold}" / f"inner_{inner_fold}"
                )
                splits.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "training_folds": list(contract.training_folds),
                        "validation_fold": inner_fold,
                        "excluded_fold": outer_fold,
                        "dataset_path": _repository_relative(relative_dataset, spec.project_root),
                        "split_manifest_path": _repository_relative(
                            relative_dataset / "split_manifest.json", spec.project_root
                        ),
                        "split_manifest_sha256": sha256_file(split_manifest),
                        "train_unique": len(contract.train_records),
                        "validation_unique": len(contract.validation_records),
                        "excluded_unique": len(contract.excluded_records),
                    }
                )
        campaign = {
            "schema_version": 1,
            "kind": "nested_hard_negative_campaign",
            "experiment_name": spec.experiment_name,
            "seed": spec.seed,
            "folds": {
                "path": _repository_relative(spec.folds_path, spec.project_root),
                "sha256": sha256_file(spec.folds_path),
            },
            "processed_manifest": {
                "path": _repository_relative(spec.processed_manifest, spec.project_root),
                "sha256": sha256_file(spec.processed_manifest),
            },
            "split_count": len(splits),
            "outer_fold_count": 5,
            "inner_folds_per_outer": 4,
            "splits": splits,
        }
        _write_json(staging / "campaign_manifest.json", campaign)
        spec.nested_views_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(spec.nested_views_dir)
        return campaign
    except BaseException:
        if staging.exists():
            import shutil

            shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_nested_split_view(
    *,
    spec: TrainingSpec,
    records: Sequence[FoldRecord],
) -> NestedSplitContract:
    """Validate one materialized nested view and its complete provenance manifest."""

    if spec.outer_fold is None or spec.split_manifest is None:
        raise TrainingConfigurationError(
            "nested split validation requires outer_fold and split_manifest"
        )
    contract = build_nested_split_contract(
        records, outer_fold=spec.outer_fold, inner_fold=spec.fold
    )
    if spec.split_manifest.parent.resolve() != spec.dataset_path.resolve():
        raise TrainingConfigurationError("nested split manifest must be inside its dataset root")
    manifest = _load_json_object(spec.split_manifest, "nested split manifest")
    exact = {
        "schema_version": 1,
        "kind": "nested_classifier_split",
        "seed": spec.seed,
        "outer_fold": spec.outer_fold,
        "inner_fold": spec.fold,
        "training_folds": list(contract.training_folds),
        "validation_folds": [spec.fold],
        "excluded_folds": [spec.outer_fold],
    }
    for field, expected in exact.items():
        if type(manifest.get(field)) is not type(expected) or manifest.get(field) != expected:
            raise TrainingConfigurationError(
                f"nested split manifest {field} mismatch: expected {expected!r}, "
                f"found {manifest.get(field)!r}"
            )
    provenance_paths = (
        ("folds", spec.folds_path),
        ("processed_manifest", spec.processed_manifest),
    )
    for field, path in provenance_paths:
        section = _mapping(manifest.get(field), f"nested split manifest {field}")
        if section.get("sha256") != sha256_file(path):
            raise TrainingConfigurationError(f"nested split manifest {field} hash drift")
    counts = _mapping(manifest.get("counts"), "nested split manifest counts")
    expected_counts = {
        "train_unique": len(contract.train_records),
        "validation_unique": len(contract.validation_records),
        "excluded_unique": len(contract.excluded_records),
        "train_class_counts": _class_counts(contract.train_records),
        "validation_class_counts": _class_counts(contract.validation_records),
    }
    if counts != expected_counts:
        raise TrainingConfigurationError(
            f"nested split manifest counts mismatch: expected {expected_counts}, found {counts}"
        )
    hashes = _processed_image_hashes(spec.processed_manifest, records)
    expected_rows = {(row.filename, "train"): row for row in contract.train_records} | {
        (row.filename, "val"): row for row in contract.validation_records
    }
    raw_rows = manifest.get("records")
    if not isinstance(raw_rows, list):
        raise TrainingConfigurationError("nested split manifest records must be a list")
    seen: set[tuple[str, str]] = set()
    expected_files: set[Path] = set()
    for offset, raw in enumerate(raw_rows):
        row = _mapping(raw, f"nested split records[{offset}]")
        filename = _string(row.get("filename"), f"nested split records[{offset}].filename")
        split = _string(row.get("split"), f"nested split records[{offset}].split")
        key = (filename, split)
        if key in seen or key not in expected_rows:
            raise TrainingConfigurationError(f"duplicate/unexpected nested split row: {key}")
        expected = expected_rows[key]
        if (
            row.get("label") != expected.label
            or row.get("fold") != expected.fold
            or row.get("group_id") != expected.group_id
            or row.get("operation") != "hardlink"
            or row.get("source_sha256") != hashes[filename]
        ):
            raise TrainingConfigurationError(f"nested split provenance mismatch for {filename}")
        destination = spec.dataset_path / split / str(expected.label) / filename
        source = spec.images_path / filename
        if not destination.is_file() or not source.is_file():
            raise TrainingConfigurationError(f"nested split image is missing: {destination}")
        try:
            same_file = os.path.samefile(source, destination)
        except OSError as exc:
            raise TrainingConfigurationError(
                f"cannot verify nested hard link {destination}: {exc}"
            ) from exc
        if not same_file:
            raise TrainingConfigurationError(
                f"nested split image is not a hard link to processed source: {destination}"
            )
        expected_files.add(destination.resolve())
        seen.add(key)
    if seen != set(expected_rows):
        missing = sorted(set(expected_rows) - seen)
        raise TrainingConfigurationError(f"nested split manifest is incomplete: {missing[:5]}")
    actual_files = {
        path.resolve()
        for split in ("train", "val")
        for path in (spec.dataset_path / split).rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise TrainingConfigurationError("nested split filesystem differs from split manifest")
    return contract


def validate_outer_classification_view(spec: TrainingSpec, records: Sequence[FoldRecord]) -> None:
    """Require an exact 4-Fold train/1-Fold validation classification view."""

    if spec.task != "classify" or spec.outer_fold is not None:
        raise TrainingConfigurationError(
            "outer classification view validation requires a formal classifier Fold"
        )
    expected: set[Path] = set()
    for record in records:
        split = "val" if record.fold == spec.fold else "train"
        expected.add((spec.dataset_path / split / str(record.label) / record.filename).resolve())
    actual = {
        path.resolve()
        for split in ("train", "val")
        for path in (spec.dataset_path / split).rglob("*")
        if path.is_file()
    }
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise TrainingConfigurationError(
            "formal classifier dataset does not match fixed Fold assignments; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def validate_nested_campaign_views(
    spec: HardNegativeMiningSpec, records: Sequence[FoldRecord]
) -> dict[str, Any]:
    """Validate all 20 nested views against their campaign and split manifests."""

    campaign_path = spec.nested_views_dir / "campaign_manifest.json"
    expected_hashes = {
        (outer_fold, inner_fold): digest
        for outer_fold in sorted(EXPECTED_FOLDS)
        for inner_fold, digest in _campaign_split_hashes(
            campaign_path,
            outer_fold=outer_fold,
            folds_path=spec.folds_path,
            processed_manifest=spec.processed_manifest,
            project_root=spec.project_root,
        ).items()
    }
    validated: list[dict[str, Any]] = []
    from fire_detection.models.training import load_training_spec

    for outer_fold, inner_fold in sorted(expected_hashes):
        training_spec = load_training_spec(
            spec.classifier_config,
            expected_task="classify",
            fold_override=inner_fold,
            outer_fold_override=outer_fold,
        )
        contract = validate_nested_split_view(spec=training_spec, records=records)
        actual_hash = sha256_file(training_spec.split_manifest)  # type: ignore[arg-type]
        if actual_hash != expected_hashes[(outer_fold, inner_fold)]:
            raise TrainingConfigurationError(
                f"campaign split hash drift for outer {outer_fold}, inner {inner_fold}"
            )
        validated.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "training_folds": list(contract.training_folds),
                "train_unique": len(contract.train_records),
                "validation_unique": len(contract.validation_records),
                "excluded_unique": len(contract.excluded_records),
                "split_manifest_sha256": actual_hash,
            }
        )
    return {
        "status": "validated",
        "campaign_manifest": _repository_relative(campaign_path, spec.project_root),
        "campaign_manifest_sha256": sha256_file(campaign_path),
        "split_count": len(validated),
        "splits": validated,
    }


def build_replay_indices(
    samples: Sequence[Sequence[Any]],
    hard_negative_filenames: Sequence[str],
    *,
    repeat_factor: int,
) -> tuple[int, ...]:
    """Return deterministic logical indices without creating duplicate image files."""

    if repeat_factor not in {2, 3}:
        raise TrainingConfigurationError("hard-negative repeat_factor must be 2 or 3")
    by_name: dict[str, tuple[int, int]] = {}
    for index, sample in enumerate(samples):
        if len(sample) < 2:
            raise TrainingConfigurationError("classification samples must expose path and class")
        filename = Path(str(sample[0])).name
        try:
            label = int(sample[1])
        except (TypeError, ValueError) as exc:
            raise TrainingConfigurationError(f"invalid class index for {filename}") from exc
        if filename in by_name:
            raise TrainingConfigurationError(f"duplicate classifier training filename: {filename}")
        by_name[filename] = (index, label)
    hard_names = tuple(sorted(hard_negative_filenames))
    if len(set(hard_names)) != len(hard_names):
        raise TrainingConfigurationError("hard-negative replay filenames must be unique")
    replay_indices: list[int] = []
    for filename in hard_names:
        replay_sample = by_name.get(filename)
        if replay_sample is None:
            raise TrainingConfigurationError(
                f"hard-negative replay image is absent from the training split: {filename}"
            )
        index, label = replay_sample
        if label != 0:
            raise TrainingConfigurationError(
                f"hard-negative replay image must have class index 0: {filename}"
            )
        replay_indices.append(index)
    base = list(range(len(samples)))
    for _ in range(repeat_factor - 1):
        base.extend(replay_indices)
    return tuple(base)


def load_hard_negative_replay(
    *,
    path: Path,
    repeat_factor: int,
    validation_fold: int,
    records: Sequence[FoldRecord],
    folds_path: Path,
    processed_manifest: Path,
    project_root: Path,
    expected_seed: int,
) -> ReplayContract:
    """Validate one per-outer hard-negative manifest before formal training."""

    if repeat_factor not in {2, 3}:
        raise TrainingConfigurationError("hard-negative repeat_factor must be 2 or 3")
    manifest = _load_json_object(path, "hard-negative manifest")
    expected_training_folds = [fold for fold in sorted(EXPECTED_FOLDS) if fold != validation_fold]
    exact = {
        "schema_version": 1,
        "kind": "cross_fitted_hard_negatives",
        "seed": expected_seed,
        "outer_fold": validation_fold,
        "training_folds": expected_training_folds,
        "score_column": "cls_probability",
    }
    for field, expected in exact.items():
        if type(manifest.get(field)) is not type(expected) or manifest.get(field) != expected:
            raise TrainingConfigurationError(
                f"hard-negative manifest {field} mismatch: expected {expected!r}, "
                f"found {manifest.get(field)!r}"
            )
    for field, required_path in (("folds", folds_path), ("processed_manifest", processed_manifest)):
        section = _mapping(manifest.get(field), f"hard-negative manifest {field}")
        if section.get("sha256") != sha256_file(required_path):
            raise TrainingConfigurationError(f"hard-negative manifest {field} hash drift")
    campaign_section = _mapping(
        manifest.get("nested_campaign_manifest"),
        "hard-negative manifest nested_campaign_manifest",
    )
    campaign_path = _manifest_artifact(
        campaign_section,
        project_root=project_root,
        field="nested_campaign_manifest",
    )
    oof_section = _mapping(manifest.get("nested_oof"), "hard-negative manifest nested_oof")
    oof_path = _manifest_artifact(
        oof_section,
        project_root=project_root,
        field="nested_oof",
    )
    nested_rows = validate_nested_oof_rows(
        load_nested_oof_csv(oof_path),
        records,
        outer_fold=validation_fold,
        folds_path=folds_path,
        processed_manifest=processed_manifest,
        campaign_manifest=campaign_path,
        expected_seed=expected_seed,
        artifact_root=project_root,
    )
    if oof_section.get("rows") != len(nested_rows):
        raise TrainingConfigurationError("hard-negative manifest nested_oof row count mismatch")
    grid = _mapping(manifest.get("threshold_grid"), "hard-negative manifest threshold_grid")
    try:
        threshold_start = float(_required(grid, "start", "threshold_grid"))
        threshold_end = float(_required(grid, "end", "threshold_grid"))
        threshold_step = float(_required(grid, "step", "threshold_grid"))
    except (TypeError, ValueError) as exc:
        raise TrainingConfigurationError("hard-negative threshold_grid is invalid") from exc
    for field, value in (
        ("start", threshold_start),
        ("end", threshold_end),
        ("step", threshold_step),
    ):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise TrainingConfigurationError(f"hard-negative threshold_grid.{field} is invalid")
    if threshold_step <= 0.0 or threshold_start > threshold_end:
        raise TrainingConfigurationError("hard-negative threshold_grid is invalid")
    thresholds = manifest.get("per_inner_fold_thresholds")
    if not isinstance(thresholds, list) or len(thresholds) != 4:
        raise TrainingConfigurationError(
            "hard-negative manifest must contain four inner-fold thresholds"
        )
    observed_inner: set[int] = set()
    thresholds_by_fold: dict[int, float] = {}
    for offset, raw in enumerate(thresholds):
        row = _mapping(raw, f"per_inner_fold_thresholds[{offset}]")
        inner = row.get("validation_fold")
        if (
            type(inner) is not int
            or inner not in expected_training_folds
            or inner in observed_inner
        ):
            raise TrainingConfigurationError("invalid/duplicate inner threshold validation_fold")
        expected_fit = [fold for fold in expected_training_folds if fold != inner]
        if row.get("fit_folds") != expected_fit:
            raise TrainingConfigurationError(
                f"inner threshold for Fold {inner} must fit Folds {expected_fit}"
            )
        threshold = _probability(row.get("threshold"), f"inner threshold Fold {inner}")
        fit = [nested for nested in nested_rows if nested["fold"] in expected_fit]
        expected_summary = search_image_metrics(
            [int(nested["label"]) for nested in fit],
            [float(nested["cls_probability"]) for nested in fit],
            threshold_start=threshold_start,
            threshold_end=threshold_end,
            threshold_step=threshold_step,
        )
        if not math.isclose(threshold, expected_summary.threshold, abs_tol=1e-12):
            raise TrainingConfigurationError(
                f"inner threshold for Fold {inner} was not fit on the declared LOFO rows"
            )
        fit_metrics = _mapping(row.get("fit_metrics"), f"inner threshold Fold {inner} metrics")
        if fit_metrics != expected_summary.to_dict():
            raise TrainingConfigurationError(f"inner threshold fit metrics drift for Fold {inner}")
        observed_inner.add(inner)
        thresholds_by_fold[inner] = threshold
    hard_rows = manifest.get("hard_negatives")
    if not isinstance(hard_rows, list):
        raise TrainingConfigurationError("hard-negative manifest hard_negatives must be a list")
    by_name = {record.filename: record for record in records}
    held_out_groups = {record.group_id for record in records if record.fold == validation_fold}
    nested_by_name = {str(row["filename"]): row for row in nested_rows}
    expected_hard_names = {
        str(row["filename"])
        for row in nested_rows
        if row["label"] == 0
        and float(row["cls_probability"]) >= thresholds_by_fold[int(row["fold"])]
    }
    hard_names: list[str] = []
    for offset, raw in enumerate(hard_rows):
        row = _mapping(raw, f"hard_negatives[{offset}]")
        filename = _string(row.get("filename"), f"hard_negatives[{offset}].filename")
        if filename in hard_names:
            raise TrainingConfigurationError(f"duplicate hard-negative filename: {filename}")
        record = by_name.get(filename)
        if record is None:
            raise TrainingConfigurationError(f"unknown hard-negative filename: {filename}")
        if record.label != 0 or record.fold == validation_fold:
            raise TrainingConfigurationError(
                f"hard-negative must be a negative outer-training image: {filename}"
            )
        if record.group_id in held_out_groups:
            raise TrainingConfigurationError(
                f"hard-negative group overlaps held-out outer Fold: {filename}"
            )
        if (
            row.get("label") != record.label
            or row.get("fold") != record.fold
            or row.get("group_id") != record.group_id
        ):
            raise TrainingConfigurationError(f"hard-negative provenance mismatch for {filename}")
        score = _probability(row.get("score"), f"hard-negative score {filename}")
        threshold = _probability(row.get("threshold"), f"hard-negative threshold {filename}")
        if score < threshold:
            raise TrainingConfigurationError(f"hard-negative is below its threshold: {filename}")
        nested = nested_by_name[filename]
        score_matches = math.isclose(score, float(nested["cls_probability"]), abs_tol=1e-12)
        threshold_matches = math.isclose(threshold, thresholds_by_fold[record.fold], abs_tol=1e-12)
        if not score_matches or not threshold_matches:
            raise TrainingConfigurationError(
                f"hard-negative score/threshold does not match nested OOF for {filename}"
            )
        hard_names.append(filename)
    if set(hard_names) != expected_hard_names:
        raise TrainingConfigurationError(
            "hard-negative filename set does not match cross-fitted negative FPs"
        )
    counts = _mapping(manifest.get("counts"), "hard-negative manifest counts")
    if counts.get("hard_negative_unique") != len(hard_names):
        raise TrainingConfigurationError("hard-negative manifest count mismatch")
    training = [record for record in records if record.fold != validation_fold]
    validation = [record for record in records if record.fold == validation_fold]
    original_counts = _class_counts(training)
    validation_counts = _class_counts(validation)
    expected_counts = {
        "outer_train_unique": len(training),
        "outer_validation_unique": len(validation),
        "outer_train_class_counts": original_counts,
        "outer_validation_class_counts": validation_counts,
        "hard_negative_unique": len(hard_names),
        "hard_negative_by_inner_fold": {
            str(fold): sum(by_name[name].fold == fold for name in hard_names)
            for fold in expected_training_folds
        },
    }
    if counts != expected_counts:
        raise TrainingConfigurationError(
            f"hard-negative manifest counts mismatch: expected {expected_counts}, found {counts}"
        )
    effective_counts = dict(original_counts)
    effective_counts["0"] += len(hard_names) * (repeat_factor - 1)
    return ReplayContract(
        path.resolve(),
        sha256_file(path),
        validation_fold,
        repeat_factor,
        tuple(sorted(hard_names)),
        len(training),
        len(training) + len(hard_names) * (repeat_factor - 1),
        original_counts,
        effective_counts,
        len(validation),
        validation_counts,
    )


def _portable_last_weights(value: Any, field: str) -> str:
    path = _string(value, field).replace("\\", "/")
    portable = PurePosixPath(path)
    if portable.is_absolute() or PureWindowsPath(path).is_absolute() or ".." in portable.parts:
        raise TrainingConfigurationError(f"{field} must be repository-relative")
    if portable.name != "last.pt":
        raise TrainingConfigurationError(f"{field} must identify last.pt")
    return portable.as_posix()


def load_nested_run_checkpoint(
    spec: TrainingSpec, records: Sequence[FoldRecord]
) -> tuple[Path, dict[str, Any]]:
    """Verify one completed nested miner run and return its fixed-epoch checkpoint."""

    contract = validate_nested_split_view(spec=spec, records=records)
    manifest_path = spec.run_dir / "run_manifest.json"
    manifest = _load_json_object(manifest_path, "nested miner run manifest")
    expected = {
        "schema_version": 1,
        "status": "complete",
        "task": "classify",
        "experiment_name": spec.experiment_name,
        "run_name": spec.run_name,
        "outer_fold": spec.outer_fold,
        "validation_fold": spec.fold,
        "training_folds": list(contract.training_folds),
        "excluded_folds": [spec.outer_fold],
        "seed": spec.seed,
        "model_name": spec.model_name,
        "requested_device": spec.requested_device,
        "ultralytics_device": spec.ultralytics_device,
        "imgsz": spec.imgsz,
        "batch": spec.batch,
        "epochs": spec.epochs,
    }
    for field, value in expected.items():
        if type(manifest.get(field)) is not type(value) or manifest.get(field) != value:
            raise TrainingConfigurationError(
                f"nested run manifest {field} mismatch for {manifest_path}"
            )
    split_manifest = spec.split_manifest
    if split_manifest is None:
        raise TrainingConfigurationError("nested miner spec has no split_manifest")
    path_fields = {
        "initial_weights": spec.weights_path,
        "config_path": spec.config_path,
        "folds_path": spec.folds_path,
        "processed_manifest": spec.processed_manifest,
        "dataset_path": spec.dataset_path,
        "split_manifest": split_manifest,
    }
    for field, path in path_fields.items():
        if manifest.get(field) != _repository_relative(path, spec.project_root):
            raise TrainingConfigurationError(f"nested run manifest {field} mismatch")
    hash_fields = {
        "initial_weights_sha256": sha256_file(spec.weights_path),
        "config_sha256": sha256_file(spec.config_path),
        "folds_sha256": sha256_file(spec.folds_path),
        "processed_manifest_sha256": sha256_file(spec.processed_manifest),
        "dataset_sha256": _sha256_training_dataset(spec),
        "split_manifest_sha256": sha256_file(split_manifest),
    }
    for field, value in hash_fields.items():
        if manifest.get(field) != value:
            raise TrainingConfigurationError(f"nested run manifest {field} drift")
    last = spec.run_dir / "weights" / "last.pt"
    if not last.is_file():
        raise TrainingConfigurationError(f"nested miner last.pt is missing: {last}")
    expected_path = _repository_relative(last, spec.project_root)
    if _portable_last_weights(manifest.get("last_weights"), "last_weights") != expected_path:
        raise TrainingConfigurationError("nested run manifest last_weights mismatch")
    if manifest.get("last_weights_sha256") != sha256_file(last):
        raise TrainingConfigurationError("nested run manifest last_weights_sha256 drift")
    return last, manifest


def load_nested_oof_csv(path: Path) -> list[dict[str, Any]]:
    """Load typed nested OOF rows from one exporter CSV."""

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not set(NESTED_OOF_COLUMNS).issubset(reader.fieldnames):
                raise TrainingConfigurationError(
                    f"nested OOF must contain {list(NESTED_OOF_COLUMNS)}"
                )
            raw_rows = list(reader)
    except OSError as exc:
        raise TrainingConfigurationError(f"cannot read nested OOF {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for offset, raw in enumerate(raw_rows, start=2):
        try:
            row = dict(raw)
            for field in ("label", "fold", "outer_fold", "imgsz", "seed"):
                value = int(row[field])
                if str(value) != str(row[field]).strip():
                    raise ValueError(field)
                row[field] = value
            row["cls_probability"] = float(row["cls_probability"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainingConfigurationError(f"invalid nested OOF values at line {offset}") from exc
        rows.append(row)
    return rows


def _campaign_split_hashes(
    campaign_manifest: Path,
    *,
    outer_fold: int,
    folds_path: Path,
    processed_manifest: Path,
    project_root: Path | None = None,
) -> dict[int, str]:
    campaign = _load_json_object(campaign_manifest, "nested campaign manifest")
    exact = {
        "schema_version": 1,
        "kind": "nested_hard_negative_campaign",
        "seed": 42,
        "split_count": 20,
        "outer_fold_count": 5,
        "inner_folds_per_outer": 4,
    }
    for field, expected in exact.items():
        if type(campaign.get(field)) is not type(expected) or campaign.get(field) != expected:
            raise TrainingConfigurationError(f"nested campaign manifest {field} mismatch")
    for field, path in (("folds", folds_path), ("processed_manifest", processed_manifest)):
        section = _mapping(campaign.get(field), f"campaign {field}")
        if section.get("sha256") != sha256_file(path):
            raise TrainingConfigurationError(f"nested campaign {field} hash drift")
    splits = campaign.get("splits")
    if not isinstance(splits, list) or len(splits) != 20:
        raise TrainingConfigurationError("nested campaign must describe exactly 20 splits")
    result: dict[int, str] = {}
    seen_pairs: set[tuple[int, int]] = set()
    for raw in splits:
        row = _mapping(raw, "campaign split")
        pair = (row.get("outer_fold"), row.get("inner_fold"))
        if (
            type(pair[0]) is not int
            or type(pair[1]) is not int
            or pair[0] not in EXPECTED_FOLDS
            or pair[1] not in EXPECTED_FOLDS
            or pair[0] == pair[1]
            or pair in seen_pairs
        ):
            raise TrainingConfigurationError(f"invalid/duplicate campaign split {pair}")
        seen_pairs.add(pair)  # type: ignore[arg-type]
        digest = _require_sha256(row.get("split_manifest_sha256"), "campaign split hash")
        if project_root is not None:
            split_path = _manifest_artifact(
                {
                    "path": row.get("split_manifest_path"),
                    "sha256": digest,
                },
                project_root=project_root,
                field="campaign split manifest",
            )
            expected_suffix = Path(f"outer_{pair[0]}") / f"inner_{pair[1]}" / "split_manifest.json"
            if not split_path.as_posix().endswith(expected_suffix.as_posix()):
                raise TrainingConfigurationError(
                    f"campaign split manifest path does not match split {pair}"
                )
        if pair[0] == outer_fold:
            result[int(pair[1])] = digest
    if set(result) != set(EXPECTED_FOLDS - {outer_fold}):
        raise TrainingConfigurationError(f"campaign is incomplete for outer Fold {outer_fold}")
    return result


def validate_nested_oof_rows(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[FoldRecord],
    *,
    outer_fold: int,
    folds_path: Path,
    processed_manifest: Path,
    campaign_manifest: Path,
    expected_seed: int = 42,
    artifact_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Require one cross-fitted prediction for every outer-training image."""

    if outer_fold not in EXPECTED_FOLDS:
        raise TrainingConfigurationError("outer_fold must be in 0..4")
    expected = {record.filename: record for record in records if record.fold != outer_fold}
    if len(rows) != len(expected):
        raise TrainingConfigurationError(
            f"nested OOF outer Fold {outer_fold} has {len(rows)} rows, expected {len(expected)}"
        )
    split_hashes = _campaign_split_hashes(
        campaign_manifest,
        outer_fold=outer_fold,
        folds_path=folds_path,
        processed_manifest=processed_manifest,
        project_root=(artifact_root.resolve() if artifact_root is not None else None),
    )
    folds_hash = sha256_file(folds_path)
    processed_hash = sha256_file(processed_manifest)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_fold_paths: dict[int, set[str]] = {}
    per_fold_hashes: dict[int, set[str]] = {}
    for raw in rows:
        row = dict(raw)
        filename = row.get("filename")
        if not isinstance(filename, str) or filename not in expected or filename in seen:
            raise TrainingConfigurationError(
                f"nested OOF contains duplicate/unexpected filename {filename!r}"
            )
        record = expected[filename]
        if (
            row.get("label") != record.label
            or row.get("fold") != record.fold
            or row.get("group_id") != record.group_id
            or row.get("outer_fold") != outer_fold
        ):
            raise TrainingConfigurationError(f"nested OOF provenance mismatch for {filename}")
        score = _probability(row.get("cls_probability"), f"nested OOF score {filename}")
        if row.get("seed") != expected_seed or type(row.get("imgsz")) is not int:
            raise TrainingConfigurationError(
                f"nested OOF runtime provenance mismatch for {filename}"
            )
        if int(row["imgsz"]) <= 0:
            raise TrainingConfigurationError(f"nested OOF imgsz is invalid for {filename}")
        if row.get("folds_sha256") != folds_hash:
            raise TrainingConfigurationError(f"nested OOF Fold hash drift for {filename}")
        if row.get("processed_manifest_sha256") != processed_hash:
            raise TrainingConfigurationError(
                f"nested OOF processed-manifest hash drift for {filename}"
            )
        if row.get("nested_split_manifest_sha256") != split_hashes[record.fold]:
            raise TrainingConfigurationError(f"nested split hash mismatch for {filename}")
        weights_path = _portable_last_weights(
            row.get("weights_path"), f"nested OOF weights_path {filename}"
        )
        weights_hash = _require_sha256(
            row.get("weights_sha256"), f"nested OOF weights_sha256 {filename}"
        )
        if artifact_root is not None:
            candidate = (
                artifact_root.resolve() / Path(*PurePosixPath(weights_path).parts)
            ).resolve()
            try:
                candidate.relative_to(artifact_root.resolve())
            except ValueError as exc:
                raise TrainingConfigurationError(
                    f"nested OOF weights_path escapes artifact root: {weights_path}"
                ) from exc
            if not candidate.is_file() or sha256_file(candidate) != weights_hash:
                raise TrainingConfigurationError(
                    f"nested OOF checkpoint is missing or changed: {candidate}"
                )
        if not isinstance(row.get("model_name"), str) or not str(row["model_name"]).strip():
            raise TrainingConfigurationError(f"nested OOF model_name is empty for {filename}")
        per_fold_paths.setdefault(record.fold, set()).add(weights_path)
        per_fold_hashes.setdefault(record.fold, set()).add(weights_hash)
        row["weights_path"] = weights_path
        row["weights_sha256"] = weights_hash
        row["cls_probability"] = score
        normalized.append(row)
        seen.add(filename)
    for inner_fold in sorted(EXPECTED_FOLDS - {outer_fold}):
        if (
            len(per_fold_paths.get(inner_fold, set())) != 1
            or len(per_fold_hashes.get(inner_fold, set())) != 1
        ):
            raise TrainingConfigurationError(
                f"nested OOF inner Fold {inner_fold} must use exactly one checkpoint"
            )
    if (
        len({next(iter(value)) for value in per_fold_paths.values()}) != 4
        or len({next(iter(value)) for value in per_fold_hashes.values()}) != 4
    ):
        raise TrainingConfigurationError("nested OOF must use four distinct inner checkpoints")
    if (
        len({row["model_name"] for row in normalized}) != 1
        or len({row["imgsz"] for row in normalized}) != 1
    ):
        raise TrainingConfigurationError("nested OOF model_name/imgsz must be constant")
    return sorted(normalized, key=lambda row: str(row["filename"]))


def select_cross_fitted_hard_negatives(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[FoldRecord],
    *,
    spec: HardNegativeMiningSpec,
    outer_fold: int,
    predictions_path: Path,
    campaign_manifest: Path,
) -> dict[str, Any]:
    """Fit four LOFO thresholds and select only cross-fitted negative FPs."""

    normalized = validate_nested_oof_rows(
        rows,
        records,
        outer_fold=outer_fold,
        folds_path=spec.folds_path,
        processed_manifest=spec.processed_manifest,
        campaign_manifest=campaign_manifest,
        expected_seed=spec.seed,
        artifact_root=spec.project_root,
    )
    inner_folds = sorted(EXPECTED_FOLDS - {outer_fold})
    predictions: list[int] = []
    labels: list[int] = []
    hard_rows: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    for validation_fold in inner_folds:
        fit_folds = [fold for fold in inner_folds if fold != validation_fold]
        fit = [row for row in normalized if row["fold"] in fit_folds]
        held_out = [row for row in normalized if row["fold"] == validation_fold]
        summary = search_image_metrics(
            [int(row["label"]) for row in fit],
            [float(row[spec.score_column]) for row in fit],
            threshold_start=spec.threshold_start,
            threshold_end=spec.threshold_end,
            threshold_step=spec.threshold_step,
        )
        hard_count = 0
        for row in held_out:
            label = int(row["label"])
            score = float(row[spec.score_column])
            prediction = int(score >= summary.threshold)
            labels.append(label)
            predictions.append(prediction)
            if label == 0 and prediction == 1:
                hard_count += 1
                hard_rows.append(
                    {
                        "filename": row["filename"],
                        "label": 0,
                        "fold": validation_fold,
                        "group_id": row["group_id"],
                        "score": score,
                        "threshold": summary.threshold,
                        "margin": score - summary.threshold,
                    }
                )
        thresholds.append(
            {
                "validation_fold": validation_fold,
                "fit_folds": fit_folds,
                "threshold": summary.threshold,
                "fit_metrics": summary.to_dict(),
                "validation_unique": len(held_out),
                "hard_negative_count": hard_count,
            }
        )
    metrics = compute_binary_metrics(labels, predictions, 0.5)
    training = [record for record in records if record.fold != outer_fold]
    validation = [record for record in records if record.fold == outer_fold]
    hard_rows.sort(key=lambda row: str(row["filename"]))
    return {
        "schema_version": 1,
        "kind": "cross_fitted_hard_negatives",
        "experiment_name": spec.experiment_name,
        "seed": spec.seed,
        "outer_fold": outer_fold,
        "training_folds": inner_folds,
        "score_column": spec.score_column,
        "threshold_grid": {
            "start": str(spec.threshold_start),
            "end": str(spec.threshold_end),
            "step": str(spec.threshold_step),
        },
        "folds": {
            "path": _repository_relative(spec.folds_path, spec.project_root),
            "sha256": sha256_file(spec.folds_path),
        },
        "processed_manifest": {
            "path": _repository_relative(spec.processed_manifest, spec.project_root),
            "sha256": sha256_file(spec.processed_manifest),
        },
        "nested_campaign_manifest": {
            "path": _repository_relative(campaign_manifest, spec.project_root),
            "sha256": sha256_file(campaign_manifest),
        },
        "nested_oof": {
            "path": _repository_relative(predictions_path, spec.project_root),
            "sha256": sha256_file(predictions_path),
            "rows": len(normalized),
        },
        "per_inner_fold_thresholds": thresholds,
        "cross_fitted_inner_metrics": {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "tn": metrics.tn,
        },
        "counts": {
            "outer_train_unique": len(training),
            "outer_validation_unique": len(validation),
            "outer_train_class_counts": _class_counts(training),
            "outer_validation_class_counts": _class_counts(validation),
            "hard_negative_unique": len(hard_rows),
            "hard_negative_by_inner_fold": {
                str(fold): sum(row["fold"] == fold for row in hard_rows) for fold in inner_folds
            },
        },
        "hard_negatives": hard_rows,
    }
