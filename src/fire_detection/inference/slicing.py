"""Deterministic overlapping image tiling and detector box coordinate mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from fire_detection.models.features import RawBox


class SliceInferenceError(ValueError):
    """Raised when an image or tile result cannot satisfy the slicing contract."""


@dataclass(frozen=True)
class TileWindow:
    """One half-open crop window in original-image pixel coordinates."""

    index: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_dict(self) -> dict[str, int]:
        """Return stable JSON fields for tile provenance."""

        return asdict(self)


@dataclass(frozen=True)
class LoadedTile:
    """A detached PIL crop paired with its original-image window."""

    window: TileWindow
    image: Image.Image


@dataclass(frozen=True)
class MappedTileBox:
    """One tile-local detection mapped back to original-image pixels."""

    tile_xyxy: tuple[float, float, float, float]
    xyxy: tuple[float, float, float, float]
    conf: float
    class_id: int

    def to_dict(self) -> dict[str, object]:
        """Return both local and mapped coordinates for raw-result audit."""

        return {
            "tile_xyxy": list(self.tile_xyxy),
            "xyxy": list(self.xyxy),
            "conf": self.conf,
            "class_id": self.class_id,
        }


def _axis_starts(length: int, size: int, stride: int) -> tuple[int, ...]:
    if type(length) is not int or length <= 0:
        raise SliceInferenceError("image dimensions must be positive integers")
    if type(size) is not int or size <= 0:
        raise SliceInferenceError("tile size must be a positive integer")
    if type(stride) is not int or stride <= 0 or stride > size:
        raise SliceInferenceError("tile stride must be a positive integer no greater than size")
    if length <= size:
        return (0,)
    final = length - size
    starts = list(range(0, final + 1, stride))
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def generate_tile_windows(
    *, width: int, height: int, size: int, stride: int
) -> tuple[TileWindow, ...]:
    """Generate row-major tiles with the final row and column flush to image edges."""

    x_starts = _axis_starts(width, size, stride)
    y_starts = _axis_starts(height, size, stride)
    windows: list[TileWindow] = []
    for y1 in y_starts:
        for x1 in x_starts:
            windows.append(
                TileWindow(
                    index=len(windows),
                    x1=x1,
                    y1=y1,
                    x2=min(x1 + size, width),
                    y2=min(y1 + size, height),
                )
            )
    return tuple(windows)


def load_image_tiles(
    path: Path, *, size: int, stride: int
) -> tuple[tuple[int, int], tuple[LoadedTile, ...]]:
    """Read one image safely and return detached RGB crops, including Unicode paths."""

    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            windows = generate_tile_windows(width=width, height=height, size=size, stride=stride)
            tiles = tuple(
                LoadedTile(
                    window,
                    image.crop((window.x1, window.y1, window.x2, window.y2)),
                )
                for window in windows
            )
    except (OSError, UnidentifiedImageError) as exc:
        raise SliceInferenceError(f"cannot read image for sliced inference {path}: {exc}") from exc
    return (height, width), tiles


def map_tile_boxes(
    window: TileWindow,
    boxes: tuple[RawBox, ...],
    *,
    image_shape: tuple[int, int],
) -> tuple[MappedTileBox, ...]:
    """Translate tile-local boxes into clipped original-image coordinates."""

    height, width = image_shape
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise SliceInferenceError("image_shape must contain positive integer height and width")
    if not (0 <= window.x1 < window.x2 <= width and 0 <= window.y1 < window.y2 <= height):
        raise SliceInferenceError("tile window lies outside the original image")
    mapped: list[MappedTileBox] = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        clipped_local = (
            max(0.0, min(float(window.width), x1)),
            max(0.0, min(float(window.height), y1)),
            max(0.0, min(float(window.width), x2)),
            max(0.0, min(float(window.height), y2)),
        )
        global_xyxy = (
            clipped_local[0] + window.x1,
            clipped_local[1] + window.y1,
            clipped_local[2] + window.x1,
            clipped_local[3] + window.y1,
        )
        mapped.append(MappedTileBox(box.xyxy, global_xyxy, box.conf, box.class_id))
    return tuple(mapped)


def raw_tile_record(window: TileWindow, boxes: tuple[MappedTileBox, ...]) -> dict[str, object]:
    """Build one deterministic raw tile result record without applying cross-tile NMS."""

    return {
        "tile_index": window.index,
        "window": window.to_dict(),
        "boxes": [box.to_dict() for box in boxes],
    }
