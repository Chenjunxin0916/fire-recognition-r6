"""Build a label-free 1198-image engineering set for official-pipeline pressure tests."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fire_detection.inference.slicing import generate_tile_windows
from fire_detection.inference.submission import IMAGE_SUFFIXES
from fire_detection.inference.test_input import inspect_image_file
from fire_detection.models.offline import sha256_file
from fire_detection.utils import atomic_write_json


class MockTestBuildError(ValueError):
    """Raised when the engineering-only set cannot be built deterministically."""


def _project_path(path: Path, root: Path, name: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise MockTestBuildError(f"{name} must stay inside the project root") from exc


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
        return "copy"
    return "hardlink"


def _duplicate_name(index: int, source: Path) -> str:
    special_suffixes = (".JPG", ".JPEG", ".PNG")
    if index < len(special_suffixes):
        return f"工程 复本 {index:04d}{special_suffixes[index]}"
    return f"engineering duplicate {index:04d}{source.suffix.lower()}"


def build_mock_test_set(
    *,
    source_dir: Path,
    output_dir: Path,
    project_root: Path,
    expected_source_count: int = 1100,
    duplicate_count: int = 98,
    tile_size: int = 640,
    tile_stride: int = 512,
    processed_manifest: Path | None = None,
) -> dict[str, Any]:
    """Hardlink/copy canonical images plus deterministic high-tile-count duplicates."""

    root = project_root.resolve()
    source = source_dir.resolve()
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite mock test set: {destination}")
    if (
        destination == (root / "data/test").resolve()
        or (root / "data/test").resolve() in destination.parents
    ):
        raise MockTestBuildError("mock test set must never be written under data/test")
    if type(expected_source_count) is not int or expected_source_count <= 0:
        raise MockTestBuildError("expected_source_count must be a positive integer")
    if type(duplicate_count) is not int or duplicate_count <= 0:
        raise MockTestBuildError("duplicate_count must be a positive integer")
    if duplicate_count > expected_source_count:
        raise MockTestBuildError("duplicate_count cannot exceed the canonical source count")
    if not source.is_dir():
        raise MockTestBuildError(f"canonical source directory does not exist: {source}")
    unsupported = [
        path.name
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() not in IMAGE_SUFFIXES
    ]
    if unsupported:
        raise MockTestBuildError(f"canonical source contains unsupported files: {unsupported[:5]}")
    files = sorted(
        (path for path in source.iterdir() if path.is_file()), key=lambda path: path.name
    )
    if len(files) != expected_source_count:
        raise MockTestBuildError(
            f"expected {expected_source_count} canonical images, found {len(files)}"
        )
    if len({path.name.casefold() for path in files}) != len(files):
        raise MockTestBuildError("canonical filenames collide case-insensitively")

    inspected: list[tuple[Path, dict[str, Any], int]] = []
    for path in files:
        record = inspect_image_file(path)
        tile_count = len(
            generate_tile_windows(
                width=int(record["width"]),
                height=int(record["height"]),
                size=tile_size,
                stride=tile_stride,
            )
        )
        inspected.append((path, record, tile_count))
    ranked = sorted(inspected, key=lambda row: (-row[2], row[0].name))

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    images_dir = staging / "images"
    images_dir.mkdir(parents=True)
    mapping: list[dict[str, Any]] = []
    try:
        for source_path, source_record, tile_count in inspected:
            target = images_dir / source_path.name
            mode = _link_or_copy(source_path, target)
            mapping.append(
                {
                    "destination_filename": target.name,
                    "source_filename": source_path.name,
                    "source_sha256": source_record["sha256"],
                    "content_sha256": sha256_file(target),
                    "tile_count": tile_count,
                    "duplicate": False,
                    "materialization": mode,
                }
            )
        for index, (source_path, source_record, tile_count) in enumerate(ranked[:duplicate_count]):
            target = images_dir / _duplicate_name(index, source_path)
            if target.exists() or target.name.casefold() in {
                row["destination_filename"].casefold() for row in mapping
            }:
                raise MockTestBuildError(f"mock duplicate filename collides: {target.name}")
            mode = _link_or_copy(source_path, target)
            mapping.append(
                {
                    "destination_filename": target.name,
                    "source_filename": source_path.name,
                    "source_sha256": source_record["sha256"],
                    "content_sha256": sha256_file(target),
                    "tile_count": tile_count,
                    "duplicate": True,
                    "rank": index,
                    "materialization": mode,
                }
            )
        names = [str(row["destination_filename"]) for row in mapping]
        if len(names) != expected_source_count + duplicate_count or len(set(names)) != len(names):
            raise MockTestBuildError("mock inventory count or uniqueness is invalid")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "kind": "pre_official_mock_test_set",
            "engineering_only": True,
            "contains_labels": False,
            "source": {
                "directory": _project_path(source, root, "source_dir"),
                "count": expected_source_count,
                "processed_manifest": None,
            },
            "selection": {
                "duplicate_count": duplicate_count,
                "ranking": "tile_count_desc_then_filename_asc",
                "tile_size": tile_size,
                "tile_stride": tile_stride,
            },
            "output": {
                "images": _project_path(destination / "images", root, "output images"),
                "count": len(mapping),
            },
            "files": mapping,
        }
        if processed_manifest is not None:
            processed = processed_manifest.resolve()
            if not processed.is_file():
                raise MockTestBuildError(f"processed manifest does not exist: {processed}")
            manifest["source"]["processed_manifest"] = {
                "path": _project_path(processed, root, "processed_manifest"),
                "sha256": sha256_file(processed),
            }
        atomic_write_json(staging / "mock_test_manifest.json", manifest)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


__all__ = ["MockTestBuildError", "build_mock_test_set"]
