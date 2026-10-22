"""Manifest-closed, hash-verified backups for a frozen final selection."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

from fire_detection.utils.io import atomic_write_json


class ArtifactBackupError(RuntimeError):
    """Raised when a frozen-artifact backup cannot be trusted."""


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactBackupError(f"cannot read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactBackupError(f"JSON root must be an object: {path}")
    return payload


def _artifact_references(value: Any) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        path = value.get("path")
        digest = value.get("sha256")
        if isinstance(path, str) and isinstance(digest, str) and len(digest) == 64:
            yield path, digest.lower()
        for child in value.values():
            yield from _artifact_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_references(child)


def collect_selection_artifacts(
    selection_manifest: Path,
    *,
    project_root: Path,
) -> tuple[dict[str, object], ...]:
    """Resolve every unique ``path``/``sha256`` reference in a selection manifest."""

    root = project_root.resolve()
    manifest = selection_manifest.resolve()
    try:
        manifest.relative_to(root)
    except ValueError as exc:
        raise ArtifactBackupError("selection manifest must be inside project_root") from exc
    payload = _load_json(manifest)
    if payload.get("kind") != "final_selection" or payload.get("status") != "complete":
        raise ArtifactBackupError("selection manifest is not a completed final selection")

    expected: dict[str, str] = {}
    for raw_path, digest in _artifact_references(payload):
        pure_path = PurePosixPath(raw_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ArtifactBackupError(f"artifact path must be repository-relative: {raw_path}")
        normalized = pure_path.as_posix()
        previous = expected.setdefault(normalized, digest)
        if previous != digest:
            raise ArtifactBackupError(f"conflicting hashes for artifact: {normalized}")

    manifest_relative = manifest.relative_to(root).as_posix()
    expected[manifest_relative] = sha256_file(manifest)
    artifacts: list[dict[str, object]] = []
    for relative, expected_hash in sorted(expected.items()):
        source = (root / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ArtifactBackupError(f"artifact escapes project root: {relative}") from exc
        if not source.is_file():
            raise ArtifactBackupError(f"artifact is missing: {relative}")
        actual_hash = sha256_file(source)
        if actual_hash != expected_hash:
            raise ArtifactBackupError(
                f"artifact hash mismatch: {relative}; expected {expected_hash}, got {actual_hash}"
            )
        artifacts.append(
            {
                "path": relative,
                "sha256": actual_hash,
                "size_bytes": source.stat().st_size,
            }
        )
    return tuple(artifacts)


def create_selection_backup(
    selection_manifest: Path,
    destination: Path,
    *,
    project_root: Path,
    backup_name: str,
) -> Path:
    """Copy a frozen selection closure and atomically publish a verified backup."""

    if not backup_name or any(character in backup_name for character in "\\/:"):
        raise ArtifactBackupError("backup_name must be one portable path component")
    root = project_root.resolve()
    destination_root = destination.resolve()
    final = destination_root / backup_name
    if final.exists():
        raise ArtifactBackupError(f"backup destination already exists: {final}")
    artifacts = collect_selection_artifacts(selection_manifest, project_root=root)
    staging = destination_root / f".{backup_name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for artifact in artifacts:
            relative = str(artifact["path"])
            source = root / Path(*PurePosixPath(relative).parts)
            target = staging / Path(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_hash = sha256_file(target)
            if copied_hash != artifact["sha256"]:
                raise ArtifactBackupError(f"copied artifact hash mismatch: {relative}")

        selection_relative = selection_manifest.resolve().relative_to(root).as_posix()
        payload = {
            "schema_version": 1,
            "kind": "final_selection_backup",
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "source": {
                "project_root_name": root.name,
                "selection_manifest": selection_relative,
                "selection_manifest_sha256": sha256_file(selection_manifest),
            },
            "counts": {
                "files": len(artifacts),
                "bytes": sum(cast(int, item["size_bytes"]) for item in artifacts),
            },
            "artifacts": list(artifacts),
        }
        atomic_write_json(staging / "backup_manifest.json", payload)
        destination_root.mkdir(parents=True, exist_ok=True)
        staging.replace(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verify_selection_backup(final)
    return final


def verify_selection_backup(backup_dir: Path) -> Mapping[str, Any]:
    """Verify every file and reject missing, changed, or extra backup artifacts."""

    root = backup_dir.resolve()
    manifest_path = root / "backup_manifest.json"
    payload = _load_json(manifest_path)
    if payload.get("kind") != "final_selection_backup" or payload.get("status") != "complete":
        raise ArtifactBackupError("backup manifest is not complete")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ArtifactBackupError("backup manifest artifacts must be a non-empty list")

    expected_paths: set[str] = set()
    total_bytes = 0
    for index, value in enumerate(raw_artifacts):
        if not isinstance(value, dict):
            raise ArtifactBackupError(f"artifacts[{index}] must be an object")
        relative = value.get("path")
        digest = value.get("sha256")
        size = value.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not isinstance(size, int)
        ):
            raise ArtifactBackupError(f"artifacts[{index}] has invalid fields")
        pure_path = PurePosixPath(relative)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise ArtifactBackupError(f"backup artifact path is unsafe: {relative}")
        if relative in expected_paths:
            raise ArtifactBackupError(f"duplicate backup artifact: {relative}")
        expected_paths.add(relative)
        artifact = (root / Path(*pure_path.parts)).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ArtifactBackupError(f"backup artifact escapes root: {relative}") from exc
        if not artifact.is_file() or artifact.stat().st_size != size:
            raise ArtifactBackupError(f"backup artifact is missing or has wrong size: {relative}")
        if sha256_file(artifact) != digest:
            raise ArtifactBackupError(f"backup artifact hash mismatch: {relative}")
        total_bytes += size

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ArtifactBackupError(f"backup file set mismatch; missing={missing}, extra={extra}")
    counts = payload.get("counts")
    if not isinstance(counts, dict) or counts.get("files") != len(expected_paths):
        raise ArtifactBackupError("backup file count does not match manifest")
    if counts.get("bytes") != total_bytes:
        raise ArtifactBackupError("backup byte count does not match manifest")
    return payload
