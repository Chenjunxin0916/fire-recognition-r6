"""Offline asset preparation and runtime preflight for Ultralytics training."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ASSET_RELEASE = "v8.4.0"
ASSET_BASE_URL = f"https://github.com/ultralytics/assets/releases/download/{ASSET_RELEASE}"
SCENE_BACKBONE_ASSET_NAME = "resnet50_imagenet1k_v2.pth"
SCENE_BACKBONE_SOURCE_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
REQUIRED_WEIGHT_NAMES = ("yolo26s.pt", "yolo26s-cls.pt", "yolo26n.pt", SCENE_BACKBONE_ASSET_NAME)
DEFAULT_ULTRALYTICS_VERSION = "8.4.92"


class OfflinePreparationError(RuntimeError):
    """Raised when offline assets are missing, altered, or ambiguously configured."""


@dataclass(frozen=True)
class AssetRecord:
    """Verified local asset and its recorded provenance."""

    name: str
    path: Path
    sha256: str
    size_bytes: int
    source_url: str


@dataclass(frozen=True)
class OfflineRuntime:
    """Validated paths required before importing Ultralytics."""

    manifest_path: Path
    yolo_config_dir: Path
    weights_dir: Path
    assets: dict[str, AssetRecord]
    ultralytics_version: str

    def require(self, name: str) -> AssetRecord:
        """Return a required asset or fail with a contextual message."""

        try:
            return self.assets[name]
        except KeyError as exc:
            raise OfflinePreparationError(f"offline manifest has no asset named {name!r}") from exc


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise OfflinePreparationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def configure_offline_environment(yolo_config_dir: Path) -> None:
    """Set network and telemetry guards before any Ultralytics import."""

    target = str(yolo_config_dir.resolve())
    if "ultralytics" in sys.modules:
        current_dir = os.environ.get("YOLO_CONFIG_DIR")
        current_offline = os.environ.get("YOLO_OFFLINE", "").lower()
        if current_dir != target or current_offline != "true":
            raise OfflinePreparationError(
                "Ultralytics was imported before the required offline environment was configured"
            )
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_CONFIG_DIR"] = target
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")


def _as_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OfflinePreparationError(f"{field} must be a JSON object")
    return value


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OfflinePreparationError(f"cannot read offline manifest {path}: {exc}") from exc
    return _as_mapping(raw, "offline manifest")


def _validate_asset_record(
    *, project_root: Path, raw_assets: dict[str, Any], name: str
) -> AssetRecord:
    """Validate one manifest row and return its immutable local record."""

    row = _as_mapping(raw_assets.get(name), f"asset {name!r}")
    raw_path = row.get("path")
    recorded_hash = row.get("sha256")
    source_url = row.get("source_url")
    size_bytes = row.get("size_bytes")
    if not isinstance(raw_path, str) or not raw_path:
        raise OfflinePreparationError(f"asset {name!r} has no path")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise OfflinePreparationError(f"asset {name!r} has an invalid SHA-256")
    if not isinstance(source_url, str) or not source_url:
        raise OfflinePreparationError(f"asset {name!r} has no source_url")
    if type(size_bytes) is not int or size_bytes <= 0:
        raise OfflinePreparationError(f"asset {name!r} has an invalid size_bytes")
    asset_path = Path(raw_path)
    if not asset_path.is_absolute():
        asset_path = project_root.resolve() / asset_path
    asset_path = asset_path.resolve()
    if not asset_path.is_file():
        raise OfflinePreparationError(f"offline asset is missing: {asset_path}")
    actual_size = asset_path.stat().st_size
    if actual_size != size_bytes:
        raise OfflinePreparationError(
            f"offline asset size changed for {name}: expected {size_bytes}, found {actual_size}"
        )
    actual_hash = sha256_file(asset_path)
    if actual_hash != recorded_hash:
        raise OfflinePreparationError(
            f"offline asset hash changed for {name}: expected {recorded_hash}, found {actual_hash}"
        )
    return AssetRecord(name, asset_path, actual_hash, actual_size, source_url)


def validate_offline_asset(
    *,
    project_root: Path,
    manifest_path: Path,
    asset_name: str,
    expected_ultralytics_version: str = DEFAULT_ULTRALYTICS_VERSION,
) -> AssetRecord:
    """Validate one explicitly named local asset without importing a model."""

    manifest = _load_manifest(manifest_path)
    if manifest.get("schema_version") != 1:
        raise OfflinePreparationError("offline manifest schema_version must be 1")
    if manifest.get("ultralytics_version") != expected_ultralytics_version:
        raise OfflinePreparationError(
            "offline manifest expects ultralytics "
            f"{manifest.get('ultralytics_version')!r}, required {expected_ultralytics_version!r}"
        )
    raw_assets = _as_mapping(manifest.get("assets"), "offline manifest assets")
    return _validate_asset_record(
        project_root=project_root.resolve(), raw_assets=raw_assets, name=asset_name
    )


def validate_offline_runtime(
    *,
    project_root: Path,
    manifest_path: Path,
    yolo_config_dir: Path,
    required_assets: tuple[str, ...],
    expected_ultralytics_version: str = DEFAULT_ULTRALYTICS_VERSION,
) -> OfflineRuntime:
    """Validate recorded weights and font without importing Ultralytics."""

    configure_offline_environment(yolo_config_dir)
    root = project_root.resolve()
    manifest = _load_manifest(manifest_path)
    if manifest.get("schema_version") != 1:
        raise OfflinePreparationError("offline manifest schema_version must be 1")
    version = manifest.get("ultralytics_version")
    if version != expected_ultralytics_version:
        raise OfflinePreparationError(
            "offline manifest expects ultralytics "
            f"{version!r}, required {expected_ultralytics_version!r}"
        )
    raw_assets = _as_mapping(manifest.get("assets"), "offline manifest assets")
    assets: dict[str, AssetRecord] = {}
    for name in required_assets:
        assets[name] = _validate_asset_record(project_root=root, raw_assets=raw_assets, name=name)

    font = assets.get("Arial.ttf")
    expected_font = yolo_config_dir.resolve() / "Ultralytics" / "Arial.ttf"
    if font is not None and font.path != expected_font:
        raise OfflinePreparationError(
            f"Arial.ttf must be installed at {expected_font}, manifest points to {font.path}"
        )
    weight_parents = {
        assets[name].path.parent for name in required_assets if name.endswith((".pt", ".pth"))
    }
    if weight_parents and len(weight_parents) != 1:
        raise OfflinePreparationError("all offline Ultralytics weights must share one directory")
    weights_dir = next(iter(weight_parents), root / "weights" / "pretrained")
    return OfflineRuntime(
        manifest_path.resolve(),
        yolo_config_dir.resolve(),
        weights_dir,
        assets,
        expected_ultralytics_version,
    )


def configure_imported_ultralytics(runtime: OfflineRuntime) -> None:
    """Point an already imported Ultralytics package at verified local assets only."""

    import importlib.metadata as metadata

    version = metadata.version("ultralytics")
    if version != runtime.ultralytics_version:
        raise OfflinePreparationError(
            f"ultralytics version changed: expected {runtime.ultralytics_version}, found {version}"
        )
    from ultralytics import SETTINGS

    SETTINGS.update({"weights_dir": str(runtime.weights_dir), "sync": False})


def _relative_manifest_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _download_file(url: str, destination: Path, *, force: bool) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if partial.stat().st_size <= 100_000:
            raise OfflinePreparationError(f"downloaded asset is unexpectedly small: {url}")
        os.replace(partial, destination)
    except (OSError, urllib.error.URLError) as exc:
        partial.unlink(missing_ok=True)
        raise OfflinePreparationError(f"failed to download {url}: {exc}") from exc


def prepare_offline_assets(
    *,
    project_root: Path,
    weights_dir: Path,
    yolo_config_dir: Path,
    manifest_path: Path,
    font_source: Path,
    ultralytics_version: str = DEFAULT_ULTRALYTICS_VERSION,
    force: bool = False,
) -> dict[str, Any]:
    """Download official weights, install a local font, and write a hash manifest."""

    try:
        installed_version = metadata.version("ultralytics")
    except metadata.PackageNotFoundError as exc:
        raise OfflinePreparationError(
            "Ultralytics must be installed before preparing its offline assets"
        ) from exc
    if installed_version != ultralytics_version:
        raise OfflinePreparationError(
            f"installed ultralytics is {installed_version}, expected {ultralytics_version}"
        )
    root = project_root.resolve()
    assets: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_WEIGHT_NAMES:
        destination = weights_dir / name
        source_url = (
            SCENE_BACKBONE_SOURCE_URL
            if name == SCENE_BACKBONE_ASSET_NAME
            else f"{ASSET_BASE_URL}/{name}"
        )
        _download_file(source_url, destination, force=force)
        if not destination.is_file() or destination.stat().st_size <= 100_000:
            raise OfflinePreparationError(
                f"weight asset {destination} is missing or unexpectedly small"
            )
        assets[name] = {
            "path": _relative_manifest_path(destination, root),
            "source_url": source_url,
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }

    if not font_source.is_file():
        raise OfflinePreparationError(f"font source does not exist: {font_source}")
    font_destination = yolo_config_dir / "Ultralytics" / "Arial.ttf"
    font_destination.parent.mkdir(parents=True, exist_ok=True)
    if force or not font_destination.exists():
        shutil.copy2(font_source, font_destination)
    assets["Arial.ttf"] = {
        "path": _relative_manifest_path(font_destination, root),
        "source_url": font_source.resolve().as_uri(),
        "sha256": sha256_file(font_destination),
        "size_bytes": font_destination.stat().st_size,
    }
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "ultralytics_version": ultralytics_version,
        "asset_release": ASSET_RELEASE,
        "assets": assets,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return payload


def find_default_windows_font() -> Path:
    """Return the expected local Windows Arial font path."""

    windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
    path = windows_root / "Fonts" / "arial.ttf"
    if not path.is_file():
        raise OfflinePreparationError(
            f"Arial font was not found at {path}; pass --font-source explicitly"
        )
    return path
