"""Read-only validation and hashing for an official competition test image directory."""

from __future__ import annotations

import hashlib
import warnings
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from fire_detection.inference.submission import IMAGE_SUFFIXES
from fire_detection.utils.io import atomic_write_json


class OfficialTestInputError(ValueError):
    """Raised when the official test input contract is not satisfied."""


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image_file(path: Path) -> dict[str, Any]:
    """Fully decode one image and return its immutable input-contract record.

    ``Image.verify`` alone does not force pixel decoding.  This shared routine performs
    both structural verification and a complete load, so the audit and GPU predictor
    reject the same corrupt inputs.  Pillow decompression-bomb warnings are promoted to
    hard errors; ordinary legal high-resolution images remain allowed.
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image_format = str(image.format or "unknown").upper()
                image.verify()
            with Image.open(path) as image:
                image.load()
                normalized = ImageOps.exif_transpose(image)
                width, height = normalized.size
    except (
        OSError,
        UnidentifiedImageError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise OfficialTestInputError(
            f"test image cannot be decoded (cannot fully decode pixels): {path.name}: {exc}"
        ) from exc
    if width <= 0 or height <= 0:
        raise OfficialTestInputError(f"test image has invalid dimensions: {path.name}")
    return {
        "filename": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "format": image_format,
        "extension": path.suffix.lower(),
    }


def validate_test_images(
    test_dir: Path,
    *,
    expected_count: int,
    archive: Path | None = None,
) -> dict[str, Any]:
    """Validate count, names, decoding, dimensions, formats, and SHA-256 hashes."""

    if type(expected_count) is not int or expected_count <= 0:
        raise OfficialTestInputError("expected_count must be a positive integer")
    root = test_dir.resolve()
    if not root.is_dir():
        raise OfficialTestInputError(f"test image directory does not exist: {test_dir}")
    nested = sorted(path for path in root.rglob("*") if path.is_file() and path.parent != root)
    if nested:
        raise OfficialTestInputError(f"test image directory contains nested files: {nested[0]}")
    files = sorted((path for path in root.iterdir() if path.is_file()), key=lambda path: path.name)
    unsupported = [path.name for path in files if path.suffix.lower() not in IMAGE_SUFFIXES]
    if unsupported:
        raise OfficialTestInputError(
            f"test image directory contains unsupported files: {unsupported[:5]}"
        )
    if len(files) != expected_count:
        raise OfficialTestInputError(f"expected {expected_count} test images, found {len(files)}")
    names = [path.name for path in files]
    if len(names) != len(set(names)):
        raise OfficialTestInputError("test filenames are not unique")
    folded: dict[str, list[str]] = {}
    for name in names:
        folded.setdefault(name.casefold(), []).append(name)
    case_conflicts = [values for values in folded.values() if len(values) > 1]
    if case_conflicts:
        raise OfficialTestInputError(
            f"test filenames collide case-insensitively: {case_conflicts[0]}"
        )

    images: list[dict[str, Any]] = []
    extensions: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    for path in files:
        record = inspect_image_file(path)
        extension = str(record["extension"])
        extensions[extension] += 1
        formats[str(record["format"])] += 1
        images.append(record)

    archive_payload: dict[str, Any] | None = None
    if archive is not None:
        archive_path = archive.resolve()
        if not archive_path.is_file():
            raise OfficialTestInputError(f"test archive does not exist: {archive}")
        archive_payload = {
            "name": archive_path.name,
            "sha256": _sha256(archive_path),
            "size_bytes": archive_path.stat().st_size,
        }
    return {
        "schema_version": 1,
        "kind": "official_test_input_manifest",
        "status": "valid",
        "created_at": datetime.now(UTC).isoformat(),
        "expected_count": expected_count,
        "count": len(images),
        "total_bytes": sum(int(image["size_bytes"]) for image in images),
        "extensions": dict(sorted(extensions.items())),
        "formats": dict(sorted(formats.items())),
        "archive": archive_payload,
        "images": images,
    }


def write_test_input_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a successful test-input validation manifest."""

    if payload.get("kind") != "official_test_input_manifest" or payload.get("status") != "valid":
        raise OfficialTestInputError("only a valid official test input manifest may be written")
    atomic_write_json(path, payload)
