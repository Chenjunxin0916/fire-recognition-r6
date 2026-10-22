"""Offline ResNet-50 scene embeddings and deterministic neighbor selection."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageOps

SCENE_BACKBONE_ARCHITECTURE = "resnet50"
SCENE_BACKBONE_WEIGHTS = "IMAGENET1K_V2"
SCENE_BACKBONE_ASSET_NAME = "resnet50_imagenet1k_v2.pth"
SCENE_BACKBONE_SOURCE_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
SCENE_EMBEDDING_DIMENSION = 2048


class SceneEmbeddingError(RuntimeError):
    """Raised when local scene features cannot be produced without guessing."""


@dataclass(frozen=True)
class SceneEmbeddingPair:
    """One unordered scene-neighbor edge selected for human review."""

    filename_a: str
    filename_b: str
    cosine_similarity: float


def _normalized_embeddings(embeddings: Mapping[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = sorted(embeddings)
    if len(names) < 2:
        return names, np.empty((len(names), 0), dtype=np.float64)
    dimensions = {np.asarray(embeddings[name]).size for name in names}
    if len(dimensions) != 1 or next(iter(dimensions)) <= 0:
        raise ValueError("scene embeddings must have one shared positive dimension")
    matrix = np.stack(
        [np.asarray(embeddings[name], dtype=np.float64).reshape(-1) for name in names]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("scene embeddings must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("scene embeddings must have non-zero norms")
    return names, cast(np.ndarray, matrix / norms[:, None])


def select_scene_embedding_pairs(
    embeddings: Mapping[str, np.ndarray],
    *,
    cosine_threshold: float = 0.85,
    mutual_top_k: int = 5,
) -> tuple[SceneEmbeddingPair, ...]:
    """Select thresholded mutual top-k edges plus top-1 edges for isolated samples.

    A sample is isolated when it participates in no thresholded mutual-top-k edge.
    Its highest-scoring neighbor is retained when that score also meets the same
    threshold. Filename order resolves exact score ties, making the result stable.
    """

    if not -1.0 <= cosine_threshold <= 1.0:
        raise ValueError("cosine_threshold must be between -1 and 1")
    if mutual_top_k < 1:
        raise ValueError("mutual_top_k must be at least 1")
    names, matrix = _normalized_embeddings(embeddings)
    count = len(names)
    if count < 2:
        return ()

    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -np.inf)
    neighbor_count = min(mutual_top_k, count - 1)
    neighbors: list[tuple[int, ...]] = []
    for row_index in range(count):
        order = np.lexsort((np.arange(count, dtype=np.int64), -similarity[row_index]))
        neighbors.append(
            tuple(int(index) for index in order if int(index) != row_index)[:neighbor_count]
        )

    selected: dict[tuple[str, str], float] = {}
    connected: set[int] = set()
    neighbor_sets = [set(row) for row in neighbors]
    for left in range(count):
        for right in neighbors[left]:
            if left >= right or left not in neighbor_sets[right]:
                continue
            score = float(similarity[left, right])
            if score < cosine_threshold:
                continue
            key = (names[left], names[right])
            selected[key] = score
            connected.update((left, right))

    for left in range(count):
        if left in connected:
            continue
        right = neighbors[left][0]
        score = float(similarity[left, right])
        if score < cosine_threshold:
            continue
        first, second = sorted((names[left], names[right]))
        key = (first, second)
        selected[key] = max(selected.get(key, -1.0), score)

    return tuple(
        SceneEmbeddingPair(left, right, selected[(left, right)]) for left, right in sorted(selected)
    )


def _load_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()
    except (OSError, ValueError) as exc:
        raise SceneEmbeddingError(f"cannot load scene image {path}: {exc}") from exc


def extract_resnet50_scene_embeddings(
    image_paths: Sequence[Path],
    *,
    weights_path: Path,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Extract normalized IMAGENET1K_V2 features from an explicit local checkpoint.

    The model is always constructed with ``weights=None``. The only state loaded is
    the caller-provided, previously hashed checkpoint, so this function has no code
    path that asks torchvision or torch.hub to download an asset.
    """

    if batch_size < 1:
        raise ValueError("scene embedding batch_size must be at least 1")
    if not weights_path.is_file():
        raise FileNotFoundError(f"scene backbone checkpoint does not exist: {weights_path}")
    names = [path.name for path in image_paths]
    if len(names) != len(set(names)):
        raise ValueError("scene embedding image basenames must be unique")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50
    except ImportError as exc:
        raise SceneEmbeddingError(
            "torch and torchvision are required for scene embeddings"
        ) from exc

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise SceneEmbeddingError(f"CUDA scene embedding device is unavailable: {device}")
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise SceneEmbeddingError("scene backbone checkpoint is not a state dictionary")
        model = resnet50(weights=None)
        model.load_state_dict(state, strict=True)
        model.fc = torch.nn.Identity()
        model.eval().to(target)
        transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, SceneEmbeddingError):
            raise
        raise SceneEmbeddingError(f"cannot initialize offline ResNet-50: {exc}") from exc

    ordered_paths = sorted(image_paths, key=lambda path: path.name)
    result: dict[str, np.ndarray] = {}
    try:
        with torch.inference_mode():
            for offset in range(0, len(ordered_paths), batch_size):
                batch_paths = ordered_paths[offset : offset + batch_size]
                tensors = [transform(_load_rgb(path)) for path in batch_paths]
                batch = torch.stack(tensors).to(target)
                features = torch.nn.functional.normalize(model(batch), dim=1).cpu().numpy()
                if features.shape != (len(batch_paths), SCENE_EMBEDDING_DIMENSION):
                    raise SceneEmbeddingError(
                        f"unexpected ResNet-50 feature shape: {tuple(features.shape)}"
                    )
                for path, feature in zip(batch_paths, features, strict=True):
                    result[path.name] = np.asarray(feature, dtype=np.float32)
    except RuntimeError as exc:
        raise SceneEmbeddingError(f"scene embedding inference failed: {exc}") from exc
    return result
