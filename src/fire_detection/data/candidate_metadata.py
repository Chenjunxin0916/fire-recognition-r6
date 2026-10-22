"""Provenance and write-once metadata for scene-candidate artifacts."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from fire_detection.data.grouping import CandidatePair
from fire_detection.data.scene_embeddings import (
    SCENE_BACKBONE_ARCHITECTURE,
    SCENE_BACKBONE_ASSET_NAME,
    SCENE_BACKBONE_WEIGHTS,
    SCENE_EMBEDDING_DIMENSION,
)
from fire_detection.data.similarity import ImageSignature, sha256_file
from fire_detection.models.offline import AssetRecord


class CandidateMetadataError(ValueError):
    """Raised when candidate provenance is absent or has drifted."""


_CANDIDATE_VERSION = re.compile(r"^v([1-9]\d*)$")


REQUIRED_CANDIDATE_CODE_PATHS = (
    "scripts/detect_duplicates.py",
    "src/fire_detection/config.py",
    "src/fire_detection/data/candidate_metadata.py",
    "src/fire_detection/data/folds.py",
    "src/fire_detection/data/grouping.py",
    "src/fire_detection/data/processed.py",
    "src/fire_detection/data/scene_embeddings.py",
    "src/fire_detection/data/schema.py",
    "src/fire_detection/data/similarity.py",
    "src/fire_detection/models/offline.py",
)


def candidate_version_number(value: object, *, allow_v1: bool = False) -> int:
    """Return the numeric candidate version, rejecting malformed/legacy versions."""

    match = _CANDIDATE_VERSION.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise CandidateMetadataError("candidate version must be named vN, such as v2 or v3")
    number = int(match.group(1))
    if number == 1 and not allow_v1:
        raise CandidateMetadataError("candidate v1 is superseded and cannot be used")
    return number


def required_candidate_code_paths(project_root: Path) -> tuple[Path, ...]:
    """Return the complete local-code provenance contract for Fold candidates."""

    return tuple(project_root / relative for relative in REQUIRED_CANDIDATE_CODE_PATHS)


def _require_complete_code_manifest(
    code_hashes: Mapping[str, object], *, project_root: Path
) -> None:
    missing = sorted(set(REQUIRED_CANDIDATE_CODE_PATHS) - set(code_hashes))
    if missing:
        raise CandidateMetadataError(
            f"candidate metadata code manifest omits required files: {missing}"
        )
    for raw_path, expected in code_hashes.items():
        path = _resolve_recorded_path(raw_path, project_root, "code_sha256")
        if not isinstance(expected, str) or len(expected) != 64 or not path.is_file():
            raise CandidateMetadataError(
                f"candidate metadata code input is unavailable: {raw_path}"
            )
        actual = sha256_file(path)
        if actual != expected:
            raise CandidateMetadataError(
                f"candidate metadata code hash changed for {raw_path}: "
                f"expected {expected}, found {actual}"
            )


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _git_state(project_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CandidateMetadataError(f"cannot determine Git state: {exc}") from exc
    return {"commit": commit, "dirty": dirty}


def _audit_inputs(audit_report: Path, project_root: Path) -> dict[str, object]:
    try:
        report = json.loads(audit_report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateMetadataError(f"cannot read audit report {audit_report}: {exc}") from exc
    if not isinstance(report, dict) or report.get("build_allowed") is not True:
        raise CandidateMetadataError("candidate generation requires an allowed audit report")
    manifest_name = report.get("checksum_manifest")
    source_hashes = report.get("source_hashes")
    if not isinstance(manifest_name, str) or not manifest_name:
        raise CandidateMetadataError("audit report has no checksum_manifest")
    if not isinstance(source_hashes, dict):
        raise CandidateMetadataError("audit report has no source_hashes object")
    manifest = audit_report.parent / manifest_name
    if not manifest.is_file():
        raise CandidateMetadataError(f"audit checksum manifest is missing: {manifest}")
    return {
        "report": {
            "path": _relative(audit_report, project_root),
            "sha256": sha256_file(audit_report),
        },
        "checksum_manifest": {
            "path": _relative(manifest, project_root),
            "sha256": sha256_file(manifest),
        },
        "source_hashes": {str(key): str(value) for key, value in sorted(source_hashes.items())},
    }


def build_candidate_metadata(
    *,
    project_root: Path,
    candidate_version: str,
    audit_report: Path,
    images_dir: Path,
    coco_json: Path,
    image_labels_json: Path,
    config_path: Path,
    offline_manifest: Path,
    scene_weight: AssetRecord,
    thresholds: Mapping[str, int | float],
    signatures: tuple[ImageSignature, ...],
    labels: Mapping[str, int],
    candidates: tuple[CandidatePair, ...],
    code_paths: Iterable[Path],
    supersedes: str | None = None,
) -> dict[str, Any]:
    """Build a complete, deterministic provenance payload for v2+ artifacts."""

    root = project_root.resolve()
    candidate_version_number(candidate_version)
    paths = {
        "images": images_dir,
        "coco_json": coco_json,
        "image_labels_json": image_labels_json,
        "config": config_path,
        "offline_manifest": offline_manifest,
    }
    missing = [str(path) for path in paths.values() if not path.is_file() and path != images_dir]
    if not images_dir.is_dir():
        missing.append(str(images_dir))
    if missing:
        raise CandidateMetadataError(f"candidate metadata input is missing: {missing[:5]}")
    audit = _audit_inputs(audit_report, root)
    resolved_code_paths = sorted({path.resolve() for path in code_paths}, key=str)
    missing_code = [str(path) for path in resolved_code_paths if not path.is_file()]
    if missing_code:
        raise CandidateMetadataError(f"candidate metadata code file is missing: {missing_code[:5]}")
    code_hashes = {_relative(path, root): sha256_file(path) for path in resolved_code_paths}
    _require_complete_code_manifest(code_hashes, project_root=root)
    source_paths: dict[str, Any] = {
        key: {"path": _relative(path, root), "sha256": sha256_file(path)}
        for key, path in paths.items()
        if key != "images"
    }
    source_hashes = audit["source_hashes"]
    assert isinstance(source_hashes, dict)
    source_paths["images"] = {
        "path": _relative(images_dir, root),
        "tree_sha256": source_hashes.get("images_tree", ""),
    }
    source_paths["audit"] = audit
    source_paths["offline_manifest"] = {
        "path": _relative(offline_manifest, root),
        "sha256": sha256_file(offline_manifest),
    }
    if set(labels) != {item.filename for item in signatures} or set(labels.values()) != {0, 1}:
        raise CandidateMetadataError("candidate signatures must contain both labels")
    source_counts: dict[str, int] = {}
    for candidate in candidates:
        for source in candidate.sources:
            source_counts[source] = source_counts.get(source, 0) + 1
    payload: dict[str, Any] = {
        "schema_version": 2,
        "candidate_version": candidate_version,
        "status": "active",
        "supersedes": supersedes,
        "git": _git_state(root),
        "code_sha256": code_hashes,
        "inputs": source_paths,
        "scene_embedding": {
            "architecture": SCENE_BACKBONE_ARCHITECTURE,
            "weights": SCENE_BACKBONE_WEIGHTS,
            "asset_name": SCENE_BACKBONE_ASSET_NAME,
            "path": _relative(scene_weight.path, root),
            "sha256": scene_weight.sha256,
            "size_bytes": scene_weight.size_bytes,
            "feature_dimension": SCENE_EMBEDDING_DIMENSION,
        },
        "thresholds": {str(key): thresholds[key] for key in sorted(thresholds)},
        "counts": {
            "images": len(signatures),
            "candidates": len(candidates),
            "mandatory_reviews": sum(item.requires_review for item in candidates),
            "embedding_candidates": sum("scene_embedding" in item.sources for item in candidates),
            "source_counts": source_counts,
        },
        "artifact_sha256": {},
    }
    return payload


def load_candidate_metadata(path: Path) -> dict[str, Any]:
    """Read and minimally validate a candidate metadata object."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateMetadataError(f"cannot read candidate metadata {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CandidateMetadataError("candidate metadata must be a JSON object")
    if raw.get("schema_version") != 2:
        raise CandidateMetadataError("candidate metadata schema_version must be 2")
    candidate_version = raw.get("candidate_version")
    candidate_version_number(candidate_version, allow_v1=True)
    directory_version = path.parent.name
    candidate_version_number(directory_version, allow_v1=True)
    if candidate_version != directory_version:
        raise CandidateMetadataError(
            "candidate metadata version does not match its directory: "
            f"{candidate_version!r} != {directory_version!r}"
        )
    if raw.get("status") not in {"active", "superseded"}:
        raise CandidateMetadataError("candidate metadata status is invalid")
    if raw.get("status") == "active":
        candidate_version_number(candidate_version)
    if not isinstance(raw.get("artifact_sha256"), dict):
        raise CandidateMetadataError("candidate metadata has no artifact_sha256 object")
    return raw


def verify_candidate_artifact_hashes(metadata_path: Path) -> None:
    """Verify sibling CSV hashes recorded in metadata."""

    metadata = load_candidate_metadata(metadata_path)
    if metadata.get("status") == "superseded":
        return
    hashes = metadata["artifact_sha256"]
    assert isinstance(hashes, dict)
    for name in ("image_hashes.csv", "candidates.csv"):
        expected = hashes.get(name)
        artifact = metadata_path.parent / name
        if not isinstance(expected, str) or not artifact.is_file():
            raise CandidateMetadataError(f"candidate artifact hash is missing for {name}")
        actual = sha256_file(artifact)
        if actual != expected:
            raise CandidateMetadataError(
                f"candidate artifact hash changed for {name}: expected {expected}, found {actual}"
            )


def _resolve_recorded_path(raw_path: object, project_root: Path, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise CandidateMetadataError(f"candidate metadata {field} path is invalid")
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def verify_candidate_input_hashes(metadata_path: Path, *, project_root: Path) -> None:
    """Reject changed config, annotations, audit, code, manifest, or scene weights."""

    metadata = load_candidate_metadata(metadata_path)
    if metadata.get("status") == "superseded":
        raise CandidateMetadataError("superseded candidate metadata cannot authorize a fold")
    inputs = metadata.get("inputs")
    code_hashes = metadata.get("code_sha256")
    scene = metadata.get("scene_embedding")
    if (
        not isinstance(inputs, dict)
        or not isinstance(code_hashes, dict)
        or not isinstance(scene, dict)
    ):
        raise CandidateMetadataError("candidate metadata provenance objects are missing")
    records: list[tuple[str, dict[str, Any]]] = []
    for key in ("coco_json", "image_labels_json", "config", "offline_manifest"):
        record = inputs.get(key)
        if not isinstance(record, dict):
            raise CandidateMetadataError(f"candidate metadata input {key} is missing")
        records.append((f"inputs.{key}", record))
    audit = inputs.get("audit")
    if not isinstance(audit, dict):
        raise CandidateMetadataError("candidate metadata audit input is missing")
    for key in ("report", "checksum_manifest"):
        record = audit.get(key)
        if not isinstance(record, dict):
            raise CandidateMetadataError(f"candidate metadata audit.{key} is missing")
        records.append((f"inputs.audit.{key}", record))
    records.append(("scene_embedding", scene))
    for field, record in records:
        path = _resolve_recorded_path(record.get("path"), project_root, field)
        expected = record.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64 or not path.is_file():
            raise CandidateMetadataError(f"candidate metadata {field} input is unavailable")
        actual = sha256_file(path)
        if actual != expected:
            raise CandidateMetadataError(
                f"candidate metadata input hash changed for {field}: "
                f"expected {expected}, found {actual}"
            )
    _require_complete_code_manifest(code_hashes, project_root=project_root)


def write_superseded_metadata(
    version_dir: Path,
    *,
    superseded_by: str,
) -> dict[str, Any]:
    """Mark an older publication superseded without touching its artifacts."""

    source_number = candidate_version_number(version_dir.name, allow_v1=True)
    target_number = candidate_version_number(superseded_by)
    if target_number <= source_number:
        raise CandidateMetadataError(
            f"superseded_by must be newer than {version_dir.name}: {superseded_by}"
        )
    metadata_path = version_dir / "candidates.meta.json"
    required = [version_dir / "image_hashes.csv", version_dir / "candidates.csv"]
    if any(not path.is_file() for path in required):
        raise CandidateMetadataError(
            f"cannot supersede incomplete candidate directory: {version_dir}"
        )
    if metadata_path.exists():
        existing = load_candidate_metadata(metadata_path)
        if existing.get("status") == "superseded":
            if existing.get("superseded_by") != superseded_by:
                raise CandidateMetadataError(
                    f"existing superseded metadata conflicts with requested state: {metadata_path}"
                )
            return existing
        payload = dict(existing)
        payload["status"] = "superseded"
        payload["superseded_by"] = superseded_by
        recorded_hashes = payload.get("artifact_sha256")
        actual_hashes = {path.name: sha256_file(path) for path in required}
        if not isinstance(recorded_hashes, dict) or any(
            recorded_hashes.get(name) != digest for name, digest in actual_hashes.items()
        ):
            raise CandidateMetadataError(
                f"candidate artifacts changed before supersede: {version_dir}"
            )
    else:
        payload = {
            "schema_version": 2,
            "candidate_version": version_dir.name,
            "status": "superseded",
            "superseded_by": superseded_by,
            "provenance_complete": False,
            "artifact_sha256": {path.name: sha256_file(path) for path in required},
        }
    version_dir.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(metadata_path)
    return payload
