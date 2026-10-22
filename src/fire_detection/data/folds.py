"""Immutable, gate-checked StratifiedGroupKFold artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

from fire_detection.data.annotation_review import (
    AnnotationReviewError,
    AnnotationReviewEvidence,
    validate_approved_annotation_review,
)
from fire_detection.data.candidate_metadata import (
    REQUIRED_CANDIDATE_CODE_PATHS,
    CandidateMetadataError,
    candidate_version_number,
)
from fire_detection.data.grouping import (
    GroupRecord,
    read_candidates,
    read_decisions,
    read_signatures,
    resolve_scene_groups,
)
from fire_detection.data.reviewer import validate_review_audit_for_lock
from fire_detection.data.schema import DatasetIndex, load_dataset_index
from fire_detection.data.similarity import sha256_file

_VERSION_NAME = re.compile(r"^v[1-9]\d*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SMALL_FIRE_RATIO = 0.001


class FoldGateError(RuntimeError):
    """Raised when a proposed fixed fold assignment violates a quality gate."""


@dataclass(frozen=True)
class FoldRecord:
    """One immutable image-to-fold assignment."""

    filename: str
    label: int
    group_id: str
    fold: int


@dataclass(frozen=True)
class FoldStats:
    """Class distribution summary for one validation fold."""

    fold: int
    count: int
    negative: int
    positive: int
    positive_ratio: float
    small_fire: int = 0
    small_fire_ratio: float = 0.0
    small_fire_positive_ratio: float = 0.0


def _strict_json(path: Path) -> dict[str, Any]:
    """Read a JSON object while rejecting duplicate keys."""

    class DuplicateKeyError(ValueError):
        pass

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(key)
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as exc:
        raise FoldGateError(f"cannot strictly read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FoldGateError(f"expected JSON object in {path}")
    return value


def _tree_sha256(images_dir: Path) -> str:
    """Return the audit-compatible deterministic hash of a flat image directory."""

    digest = hashlib.sha256()
    if not images_dir.is_dir():
        raise FoldGateError(f"image directory does not exist: {images_dir}")
    for path in sorted(
        (item for item in images_dir.iterdir() if item.is_file()), key=lambda item: item.name
    ):
        digest.update(f"images/{path.name}\t{path.stat().st_size}\t{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _checksum_rows(path: Path) -> dict[tuple[str, str], tuple[int, str]]:
    """Read an audit checksum manifest into a duplicate-free mapping."""

    rows: dict[tuple[str, str], tuple[int, str]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"source_kind", "path", "size_bytes", "sha256"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise FoldGateError("audit checksum manifest has invalid columns")
            for row in reader:
                kind = row.get("source_kind", "")
                source = row.get("path", "")
                digest = row.get("sha256", "")
                key = (kind, source)
                if not kind or not source or key in rows or not _SHA256.fullmatch(digest):
                    raise FoldGateError("audit checksum manifest has invalid/duplicate rows")
                rows[key] = (int(row.get("size_bytes", "")), digest)
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, FoldGateError):
            raise
        raise FoldGateError(f"cannot read audit checksum manifest {path}: {exc}") from exc
    return rows


def validate_source_provenance(
    *,
    audit_report: Path,
    images_dir: Path,
    coco_json: Path,
    image_labels_json: Path,
) -> dict[str, str]:
    """Verify audit, raw image bytes, and both source JSON files are unchanged."""

    report = _strict_json(audit_report)
    if report.get("build_allowed") is not True:
        raise FoldGateError(f"audit report does not allow fold locking: {audit_report}")
    manifest_name = report.get("checksum_manifest")
    if not isinstance(manifest_name, str) or not manifest_name:
        raise FoldGateError("audit report does not identify checksum_manifest")
    manifest_path = audit_report.parent / manifest_name
    expected = _checksum_rows(manifest_path)
    if not images_dir.is_dir():
        raise FoldGateError(f"image directory does not exist: {images_dir}")
    actual: dict[tuple[str, str], Path] = {
        ("coco_json", coco_json.name): coco_json,
        ("image_labels_json", image_labels_json.name): image_labels_json,
    }
    missing_required = [path for path in (coco_json, image_labels_json) if not path.is_file()]
    if missing_required:
        raise FoldGateError(f"source JSON files do not exist: {missing_required}")
    actual.update(
        {("image", f"images/{path.name}"): path for path in images_dir.iterdir() if path.is_file()}
    )
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise FoldGateError(
            f"audit/current source file sets differ; missing={missing}, extra={extra}"
        )
    changed = [
        source.name
        for key, source in actual.items()
        if expected[key] != (source.stat().st_size, sha256_file(source))
    ]
    if changed:
        raise FoldGateError(f"source files changed after audit: {changed[:10]}")
    source_hashes = report.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise FoldGateError("audit report does not contain source_hashes")
    actual_tree = _tree_sha256(images_dir)
    audited_sources = {
        "images_tree": actual_tree,
        "coco_json": sha256_file(coco_json),
        "image_labels_json": sha256_file(image_labels_json),
    }
    for key, actual_hash in audited_sources.items():
        if source_hashes.get(key) != actual_hash:
            raise FoldGateError(f"{key} does not match audit report")
    return {
        "audit_report": sha256_file(audit_report),
        "checksum_manifest": sha256_file(manifest_path),
        "raw_images_tree": actual_tree,
        "coco_json": audited_sources["coco_json"],
        "image_labels_json": audited_sources["image_labels_json"],
    }


def _metadata_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FoldGateError(f"candidate metadata field {field} must be an object")
    return value


def _metadata_hash(mapping: Mapping[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FoldGateError(f"candidate metadata field {field} must be a SHA-256")
    return value


def _metadata_path(mapping: Mapping[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise FoldGateError(f"candidate metadata field {field} must be a non-empty path")
    return value


def _resolve_metadata_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def validate_candidates_metadata(
    path: Path,
    *,
    image_hashes: Path,
    candidates: Path,
    source_hashes: Mapping[str, str] | None = None,
    config_path: Path | None = None,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """Validate a v2+ candidate manifest and all hashes it claims to own."""

    metadata = _strict_json(path)
    if metadata.get("status") == "superseded" or metadata.get("superseded") is True:
        raise FoldGateError(f"candidate metadata is superseded: {path}")
    if metadata.get("status") != "active":
        raise FoldGateError("candidate metadata must declare status=active")
    candidate_version = metadata.get("candidate_version")
    directory_version = path.parent.name
    try:
        candidate_version_number(candidate_version)
        candidate_version_number(directory_version)
        if expected_version is not None:
            candidate_version_number(expected_version)
    except CandidateMetadataError as exc:
        raise FoldGateError(str(exc)) from exc
    if candidate_version != directory_version:
        raise FoldGateError(
            "candidate metadata version must match its directory: "
            f"{candidate_version!r} != {directory_version!r}"
        )
    if expected_version is not None and candidate_version != expected_version:
        raise FoldGateError(
            "candidate metadata version must match the requested Fold version: "
            f"{candidate_version!r} != {expected_version!r}"
        )
    if metadata.get("schema_version", 0) != 2:
        raise FoldGateError("candidate metadata must declare schema_version=2")
    inputs = _metadata_mapping(metadata.get("inputs"), "inputs")
    artifacts = _metadata_mapping(metadata.get("artifact_sha256"), "artifact_sha256")
    for name, artifact in (("image_hashes.csv", image_hashes), ("candidates.csv", candidates)):
        if artifact.parent.resolve() != path.parent.resolve():
            raise FoldGateError(f"{name} must be in candidate version directory {path.parent}")
        expected = _metadata_hash(artifacts, name)
        if expected != sha256_file(artifact):
            raise FoldGateError(f"{name} does not match candidates.meta.json")

    git = _metadata_mapping(metadata.get("git"), "git")
    commit = git.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise FoldGateError("candidate metadata git.commit is invalid")
    if type(git.get("dirty")) is not bool:
        raise FoldGateError("candidate metadata git.dirty must be boolean")
    code_hashes = _metadata_mapping(metadata.get("code_sha256"), "code_sha256")
    if not code_hashes:
        raise FoldGateError("candidate metadata code_sha256 must not be empty")
    missing_code = sorted(set(REQUIRED_CANDIDATE_CODE_PATHS) - set(code_hashes))
    if missing_code:
        raise FoldGateError(
            f"candidate metadata code manifest omits required files: {missing_code}"
        )
    thresholds = _metadata_mapping(metadata.get("thresholds"), "thresholds")
    counts = _metadata_mapping(metadata.get("counts"), "counts")
    if not thresholds or not counts:
        raise FoldGateError("candidate metadata thresholds/counts must not be empty")
    large_component_size = thresholds.get("large_component_size")
    if type(large_component_size) is not int or large_component_size < 2:
        raise FoldGateError(
            "candidate metadata thresholds.large_component_size must be an integer >= 2"
        )

    project_root: Path | None = None
    if config_path is not None:
        project_root = (
            config_path.parent.parent
            if config_path.parent.name == "configs"
            else config_path.parent
        )
        for relative, expected_value in code_hashes.items():
            if not isinstance(relative, str):
                raise FoldGateError("candidate metadata code path must be a string")
            if not isinstance(expected_value, str) or not _SHA256.fullmatch(expected_value):
                raise FoldGateError(f"candidate metadata code hash is invalid: {relative}")
            code_path = _resolve_metadata_path(relative, project_root)
            if not code_path.is_file() or sha256_file(code_path) != expected_value:
                raise FoldGateError(f"candidate-generation code hash mismatch: {relative}")

    embedding = _metadata_mapping(metadata.get("scene_embedding"), "scene_embedding")
    if embedding.get("architecture") != "resnet50":
        raise FoldGateError("candidate metadata scene embedding must use resnet50")
    if embedding.get("weights") != "IMAGENET1K_V2":
        raise FoldGateError("candidate metadata scene embedding must use IMAGENET1K_V2")
    if embedding.get("asset_name") != "resnet50_imagenet1k_v2.pth":
        raise FoldGateError("candidate metadata scene embedding asset_name is invalid")
    embedding_path = _metadata_path(embedding, "path")
    embedding_hash = _metadata_hash(embedding, "sha256")
    embedding_size = embedding.get("size_bytes")
    feature_dimension = embedding.get("feature_dimension")
    if type(embedding_size) is not int or embedding_size <= 0:
        raise FoldGateError("candidate metadata scene embedding size_bytes is invalid")
    if type(feature_dimension) is not int or feature_dimension != 2048:
        raise FoldGateError("candidate metadata scene embedding feature_dimension is invalid")
    if project_root is not None:
        weight_path = _resolve_metadata_path(embedding_path, project_root)
        if not weight_path.is_file():
            raise FoldGateError(f"scene embedding weight is missing: {weight_path}")
        if (
            weight_path.stat().st_size != embedding_size
            or sha256_file(weight_path) != embedding_hash
        ):
            raise FoldGateError("scene embedding weight differs from candidates.meta.json")

    if source_hashes is not None:
        images = _metadata_mapping(inputs.get("images"), "inputs.images")
        _metadata_path(images, "path")
        if _metadata_hash(images, "tree_sha256") != source_hashes["raw_images_tree"]:
            raise FoldGateError("candidate metadata hash mismatch for raw image tree")
        coco = _metadata_mapping(inputs.get("coco_json"), "inputs.coco_json")
        labels = _metadata_mapping(inputs.get("image_labels_json"), "inputs.image_labels_json")
        _metadata_path(coco, "path")
        _metadata_path(labels, "path")
        if _metadata_hash(coco, "sha256") != source_hashes["coco_json"]:
            raise FoldGateError("candidate metadata hash mismatch for coco_json")
        if _metadata_hash(labels, "sha256") != source_hashes["image_labels_json"]:
            raise FoldGateError("candidate metadata hash mismatch for image_labels_json")
        audit = _metadata_mapping(inputs.get("audit"), "inputs.audit")
        audit_report_meta = _metadata_mapping(audit.get("report"), "inputs.audit.report")
        checksum_meta = _metadata_mapping(
            audit.get("checksum_manifest"), "inputs.audit.checksum_manifest"
        )
        audit_sources = _metadata_mapping(audit.get("source_hashes"), "inputs.audit.source_hashes")
        _metadata_path(audit_report_meta, "path")
        _metadata_path(checksum_meta, "path")
        if _metadata_hash(audit_report_meta, "sha256") != source_hashes["audit_report"]:
            raise FoldGateError("candidate metadata hash mismatch for audit report")
        if _metadata_hash(checksum_meta, "sha256") != source_hashes["checksum_manifest"]:
            raise FoldGateError("candidate metadata hash mismatch for audit checksum manifest")
        for key, source_key in (
            ("coco_json", "coco_json"),
            ("image_labels_json", "image_labels_json"),
            ("images_tree", "raw_images_tree"),
        ):
            if _metadata_hash(audit_sources, key) != source_hashes[source_key]:
                raise FoldGateError(f"candidate audit source hash mismatch for {key}")
        if config_path is not None:
            config = _metadata_mapping(inputs.get("config"), "inputs.config")
            _metadata_path(config, "path")
            if _metadata_hash(config, "sha256") != sha256_file(config_path):
                raise FoldGateError("candidate metadata config hash mismatch")
            offline = _metadata_mapping(inputs.get("offline_manifest"), "inputs.offline_manifest")
            offline_path = _resolve_metadata_path(
                _metadata_path(offline, "path"), project_root or path.parent
            )
            if not offline_path.is_file() or _metadata_hash(offline, "sha256") != sha256_file(
                offline_path
            ):
                raise FoldGateError("candidate metadata offline manifest hash mismatch")
    return metadata


def small_fire_filenames(
    index: DatasetIndex, threshold: float = _SMALL_FIRE_RATIO
) -> frozenset[str]:
    """Return positive image names containing at least one box below the area threshold."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("small-fire area-ratio threshold must be between 0 and 1")
    by_image = index.annotations_by_image
    result: set[str] = set()
    for image in index.images:
        if index.labels[image.file_name] != 1:
            continue
        image_area = float(image.width * image.height)
        if any(
            (annotation.bbox[2] * annotation.bbox[3]) / image_area < threshold
            for annotation in by_image[image.id]
        ):
            result.add(image.file_name)
    return frozenset(result)


def assign_stratified_group_folds(
    groups: tuple[GroupRecord, ...],
    *,
    small_fire_images: Iterable[str] = (),
    expected_count: int | None = 1100,
    min_negative: int = 30,
    max_positive_ratio_delta: float = 0.05,
    max_small_fire_ratio_delta: float = 0.05,
) -> tuple[tuple[FoldRecord, ...], tuple[FoldStats, ...]]:
    """Assign the approved 5-fold split using negative/positive/small-fire strata."""

    if not groups:
        raise FoldGateError("cannot create folds from an empty group table")
    if expected_count is not None and len(groups) != expected_count:
        raise FoldGateError(f"expected {expected_count} images, got {len(groups)}")
    filenames = [row.filename for row in groups]
    if len(filenames) != len(set(filenames)):
        raise FoldGateError("group table contains duplicate filenames")
    labels = np.asarray([row.label for row in groups], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise FoldGateError("group table must contain both integer labels 0 and 1")
    small_fire = frozenset(small_fire_images)
    unknown_small = sorted(small_fire - set(filenames))
    if unknown_small:
        raise FoldGateError(f"small-fire set references unknown images: {unknown_small[:10]}")
    invalid_small = sorted(
        row.filename for row in groups if row.filename in small_fire and row.label != 1
    )
    if invalid_small:
        raise FoldGateError(f"negative images cannot be small-fire positives: {invalid_small[:10]}")
    strata = np.asarray(
        [2 if row.filename in small_fire else row.label for row in groups], dtype=np.int64
    )
    group_ids = np.asarray([row.group_id for row in groups], dtype=object)
    ordered_indices = np.arange(len(groups), dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_by_index: dict[int, int] = {}
    for fold, (_, validation) in enumerate(
        splitter.split(ordered_indices, strata, groups=group_ids)
    ):
        for index in validation.tolist():
            if index in fold_by_index:
                raise FoldGateError(f"image index assigned more than once: {index}")
            fold_by_index[index] = fold
    if set(fold_by_index) != set(range(len(groups))):
        raise FoldGateError("fold split did not cover every image exactly once")

    records = tuple(
        FoldRecord(row.filename, row.label, row.group_id, fold_by_index[index])
        for index, row in enumerate(groups)
    )
    group_folds: dict[str, set[int]] = {}
    for record in records:
        group_folds.setdefault(record.group_id, set()).add(record.fold)
    split_groups = sorted(group_id for group_id, folds in group_folds.items() if len(folds) != 1)
    if split_groups:
        raise FoldGateError(f"scene groups cross validation folds: {split_groups[:10]}")

    overall_ratio = float(labels.mean())
    overall_small_ratio = len(small_fire) / len(groups)
    stats: list[FoldStats] = []
    failures: list[str] = []
    for fold in range(5):
        fold_labels = [record.label for record in records if record.fold == fold]
        negative = fold_labels.count(0)
        positive = fold_labels.count(1)
        ratio = positive / len(fold_labels) if fold_labels else 0.0
        fold_small = sum(record.filename in small_fire for record in records if record.fold == fold)
        small_ratio = fold_small / len(fold_labels) if fold_labels else 0.0
        small_positive_ratio = fold_small / positive if positive else 0.0
        stats.append(
            FoldStats(
                fold,
                len(fold_labels),
                negative,
                positive,
                ratio,
                fold_small,
                small_ratio,
                small_positive_ratio,
            )
        )
        if negative == 0 or positive == 0:
            failures.append(f"fold {fold} does not contain both classes")
        if negative < min_negative:
            failures.append(f"fold {fold} has {negative} negatives (< {min_negative})")
        if abs(ratio - overall_ratio) > max_positive_ratio_delta:
            failures.append(
                f"fold {fold} positive ratio delta {abs(ratio - overall_ratio):.6f} "
                f"exceeds {max_positive_ratio_delta:.6f}"
            )
        if abs(small_ratio - overall_small_ratio) > max_small_fire_ratio_delta:
            failures.append(
                f"fold {fold} small-fire ratio delta "
                f"{abs(small_ratio - overall_small_ratio):.6f} exceeds "
                f"{max_small_fire_ratio_delta:.6f}"
            )
    if failures:
        raise FoldGateError("; ".join(failures))
    return tuple(sorted(records, key=lambda item: item.filename)), tuple(stats)


def _atomic_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_fixed_fold_artifacts(
    output_dir: Path,
    groups: tuple[GroupRecord, ...],
    folds: tuple[FoldRecord, ...],
    stats: tuple[FoldStats, ...],
    *,
    input_paths: dict[str, Path],
    provenance: Mapping[str, object] | None = None,
) -> None:
    """Publish groups/folds/meta as one immutable directory transaction."""

    if output_dir.name == "locked" and _VERSION_NAME.fullmatch(output_dir.parent.name):
        fold_version = output_dir.parent.name
    elif _VERSION_NAME.fullmatch(output_dir.name):
        fold_version = output_dir.name
    else:
        raise ValueError("fold output must be <version> or <version>/locked, such as v3/locked")
    if fold_version == "v1":
        raise FoldGateError("candidate v1 is superseded and cannot be locked")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fixed fold artifacts: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        groups_path = staging / "groups.csv"
        folds_path = staging / "folds.csv"
        metadata_path = staging / "folds.meta.json"
        _atomic_csv(
            groups_path,
            (
                "filename",
                "label",
                "sha256",
                "phash",
                "width",
                "height",
                "group_id",
                "group_size",
            ),
            (row.__dict__ for row in groups),
        )
        _atomic_csv(
            folds_path,
            ("filename", "label", "group_id", "fold"),
            (row.__dict__ for row in folds),
        )
        metadata = {
            "schema_version": 2,
            "status": "locked",
            "fold_version": fold_version,
            "strategy": "StratifiedGroupKFold",
            "stratification": ["negative", "positive", "small_fire_positive"],
            "small_fire_area_ratio_threshold": _SMALL_FIRE_RATIO,
            "n_splits": 5,
            "shuffle": True,
            "seed": 42,
            "image_count": len(folds),
            "negative_count": sum(record.label == 0 for record in folds),
            "positive_count": sum(record.label == 1 for record in folds),
            "small_fire_count": sum(row.small_fire for row in stats),
            "group_count": len({record.group_id for record in folds}),
            "quality_gates": {
                "minimum_negative_per_fold": 30,
                "maximum_positive_ratio_delta": 0.05,
                "maximum_small_fire_ratio_delta": 0.05,
            },
            "folds": [row.__dict__ for row in stats],
            "input_sha256": {key: sha256_file(path) for key, path in sorted(input_paths.items())},
            "artifact_sha256": {
                "groups.csv": sha256_file(groups_path),
                "folds.csv": sha256_file(folds_path),
            },
        }
        if provenance is not None:
            metadata["provenance"] = dict(provenance)
        _atomic_json(metadata_path, metadata)
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite fixed fold artifacts: {output_dir}")
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def finalize_fold_version(
    output_dir: Path,
    *,
    image_hashes: Path,
    candidates: Path,
    decisions: Path,
    coco_json: Path,
    image_labels_json: Path,
    images_dir: Path | None = None,
    audit_report: Path | None = None,
    candidates_metadata: Path | None = None,
    candidates_meta: Path | None = None,
    annotation_review: Path | None = None,
    config_path: Path | None = None,
    expected_count: int | None = 1100,
    check_only: bool = False,
) -> tuple[tuple[GroupRecord, ...], tuple[FoldRecord, ...]]:
    """Resolve review, validate provenance and either preflight or lock one v2+ split."""

    if candidates_metadata is not None and candidates_meta is not None:
        if candidates_metadata != candidates_meta:
            raise FoldGateError("conflicting candidates metadata paths were provided")
    metadata_path = candidates_metadata or candidates_meta
    version = output_dir.parent.name if output_dir.name == "locked" else output_dir.name
    try:
        candidate_version_number(version)
    except CandidateMetadataError as exc:
        raise FoldGateError(str(exc)) from exc
    source_hashes: dict[str, str] | None = None
    candidate_metadata_payload: dict[str, Any] | None = None
    annotation_evidence: AnnotationReviewEvidence | None = None
    if any(value is not None for value in (images_dir, audit_report, metadata_path)):
        if images_dir is None or audit_report is None or metadata_path is None:
            raise FoldGateError(
                "v2+ provenance validation requires images_dir, audit_report, and "
                "candidates_metadata"
            )
        source_hashes = validate_source_provenance(
            audit_report=audit_report,
            images_dir=images_dir,
            coco_json=coco_json,
            image_labels_json=image_labels_json,
        )
        candidate_metadata_payload = validate_candidates_metadata(
            metadata_path,
            image_hashes=image_hashes,
            candidates=candidates,
            source_hashes=source_hashes,
            config_path=config_path,
            expected_version=version,
        )
    else:
        raise FoldGateError(
            f"{version} locking requires candidates.meta.json and source provenance"
        )

    if annotation_review is None or config_path is None:
        raise FoldGateError(
            f"{version} locking requires an approved annotation contact-sheet review"
        )
    project_root = (
        config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    )
    try:
        annotation_evidence = validate_approved_annotation_review(
            annotation_review,
            project_root=project_root,
        )
    except AnnotationReviewError as exc:
        raise FoldGateError(f"annotation review gate failed: {exc}") from exc

    _strict_json(coco_json)
    _strict_json(image_labels_json)
    index = load_dataset_index(coco_json, image_labels_json)
    signatures, labels = read_signatures(image_hashes)
    if set(labels) != set(index.labels):
        raise FoldGateError("image_hashes.csv does not cover the current JSON image set")
    if labels != index.labels:
        mismatched = sorted(name for name in labels if labels[name] != index.labels[name])
        if mismatched:
            raise FoldGateError(f"image_hashes.csv label mismatch: {mismatched[:10]}")
    if images_dir is not None:
        signatures_by_name = {signature.filename: signature for signature in signatures}
        for image in index.images:
            source = images_dir / image.file_name
            signature = signatures_by_name.get(image.file_name)
            if signature is None:
                raise FoldGateError(f"missing image signature: {image.file_name}")
            if sha256_file(source) != signature.sha256:
                raise FoldGateError(
                    f"raw image hash differs from image_hashes.csv: {image.file_name}"
                )
            try:
                with Image.open(source) as opened:
                    visual_size = ImageOps.exif_transpose(opened).size
            except OSError as exc:
                raise FoldGateError(f"cannot inspect raw image {source}: {exc}") from exc
            if visual_size != (signature.width, signature.height):
                raise FoldGateError(
                    f"EXIF-normalized image dimensions differ from signature: {image.file_name}"
                )
    large_component_size: int | None = None
    if candidate_metadata_payload is not None:
        metadata_thresholds = _metadata_mapping(
            candidate_metadata_payload.get("thresholds"), "thresholds"
        )
        threshold_value = metadata_thresholds.get("large_component_size")
        if type(threshold_value) is not int or threshold_value < 2:
            raise FoldGateError(
                "candidate metadata thresholds.large_component_size must be an integer >= 2"
            )
        large_component_size = threshold_value
    candidate_rows = read_candidates(candidates, large_component_size=large_component_size)
    if candidate_metadata_payload is not None:
        counts = _metadata_mapping(candidate_metadata_payload.get("counts"), "counts")
        expected_counts = {
            "images": len(signatures),
            "candidates": len(candidate_rows),
            "mandatory_reviews": sum(candidate.requires_review for candidate in candidate_rows),
            "embedding_candidates": sum(
                "scene_embedding" in candidate.sources for candidate in candidate_rows
            ),
        }
        for key, expected in expected_counts.items():
            if counts.get(key) != expected:
                raise FoldGateError(f"candidate metadata count mismatch for {key}")
        actual_source_counts: dict[str, int] = {}
        for candidate in candidate_rows:
            for candidate_source in candidate.sources:
                actual_source_counts[candidate_source] = (
                    actual_source_counts.get(candidate_source, 0) + 1
                )
        if counts.get("source_counts") != actual_source_counts:
            raise FoldGateError("candidate metadata source_counts mismatch")
    signatures_by_name = {signature.filename: signature for signature in signatures}
    for candidate in candidate_rows:
        left = signatures_by_name.get(candidate.filename_a)
        right = signatures_by_name.get(candidate.filename_b)
        if left is None or right is None:
            raise FoldGateError(f"candidate references unknown image: {candidate.pair_id}")
        if (candidate.sha256_a, candidate.sha256_b) != (left.sha256, right.sha256):
            raise FoldGateError(f"candidate image hashes are stale: {candidate.pair_id}")
    review_history = decisions.with_name("review_history.csv")
    try:
        validate_review_audit_for_lock(
            candidates,
            decisions,
            review_history,
            large_component_size=large_component_size or 10,
        )
    except (OSError, TimeoutError, ValueError) as exc:
        raise FoldGateError(f"review audit gate failed: {exc}") from exc
    decision_rows = read_decisions(decisions)
    groups = resolve_scene_groups(signatures, labels, candidate_rows, decision_rows)
    small_fire = small_fire_filenames(index)
    folds, stats = assign_stratified_group_folds(
        groups,
        small_fire_images=small_fire,
        expected_count=expected_count,
    )
    if check_only:
        return groups, folds
    input_paths = {
        "image_hashes.csv": image_hashes,
        "candidates.csv": candidates,
        "review_decisions.csv": decisions,
        "train_coco.json": coco_json,
        "train_image.json": image_labels_json,
    }
    if review_history.is_file():
        input_paths["review_history.csv"] = review_history
    if audit_report is not None:
        input_paths["audit_report"] = audit_report
    if metadata_path is not None:
        input_paths["candidates.meta.json"] = metadata_path
    if config_path is not None:
        input_paths["data_config"] = config_path
    if annotation_evidence is not None:
        input_paths["annotation_review.json"] = annotation_evidence.review_path
        input_paths["annotation_manifest.csv"] = annotation_evidence.manifest_path
    locked_provenance: dict[str, object] = dict(source_hashes or {})
    if annotation_evidence is not None:
        locked_provenance["annotation_review"] = {
            "review_sha256": annotation_evidence.review_sha256,
            "manifest_sha256": annotation_evidence.manifest_sha256,
            "page_bundle_sha256": annotation_evidence.page_bundle_sha256,
            "page_count": annotation_evidence.page_count,
            "reviewer": annotation_evidence.reviewer,
            "reviewed_at": annotation_evidence.reviewed_at,
            "disposition": annotation_evidence.disposition,
            "issue_count": annotation_evidence.issue_count,
        }
    write_fixed_fold_artifacts(
        output_dir,
        groups,
        folds,
        stats,
        input_paths=input_paths,
        provenance=locked_provenance,
    )
    return groups, folds
