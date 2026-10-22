"""Deterministic image fingerprints and full-resolution SSIM."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn
from scipy.ndimage import correlate1d


@dataclass(frozen=True)
class ImageSignature:
    """Content and perceptual fingerprints for one normalized image."""

    filename: str
    sha256: str
    phash: str
    width: int
    height: int


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading a file at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_grayscale(path: Path) -> Image.Image:
    with Image.open(path) as image:
        normalized = ImageOps.exif_transpose(image)
        return normalized.convert("L").copy()


def perceptual_hash(path: Path) -> str:
    """Compute the approved 64-bit, imagehash-compatible perceptual hash.

    Images are EXIF-normalized, converted to grayscale, resized to 32x32 with
    LANCZOS, transformed by an orthonormal 2-D DCT, and thresholded against the
    median of the top-left 8x8 coefficients (including the DC coefficient).
    """

    image = _normalized_grayscale(path)
    resized = image.resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float64)
    low_frequency = dctn(pixels, type=2, norm="ortho")[:8, :8]
    bits = (low_frequency > np.median(low_frequency)).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(left: str, right: str) -> int:
    """Return the Hamming distance between two 64-bit hexadecimal hashes."""

    if len(left) != 16 or len(right) != 16:
        raise ValueError("perceptual hashes must contain exactly 16 hexadecimal digits")
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError as exc:
        raise ValueError("perceptual hashes must be hexadecimal") from exc


def grayscale_array(path: Path) -> np.ndarray:
    """Load an EXIF-normalized image as a full-resolution float64 array."""

    return np.asarray(_normalized_grayscale(path), dtype=np.float64)


def gaussian_window(*, size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Build the normalized Gaussian window used by the SSIM reference."""

    if size <= 0 or size % 2 == 0:
        raise ValueError("Gaussian window size must be a positive odd integer")
    if sigma <= 0.0:
        raise ValueError("Gaussian sigma must be positive")
    coordinates = np.arange(size, dtype=np.float64) - size // 2
    kernel_1d = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    window = np.outer(kernel_1d, kernel_1d)
    return cast(np.ndarray, window / window.sum())


def _valid_gaussian_filter(image: np.ndarray) -> np.ndarray:
    coordinates = np.arange(11, dtype=np.float64) - 5
    kernel = np.exp(-(coordinates**2) / (2.0 * 1.5**2))
    kernel /= kernel.sum()
    filtered = correlate1d(image, kernel, axis=0, mode="constant", cval=0.0)
    filtered = correlate1d(filtered, kernel, axis=1, mode="constant", cval=0.0)
    return cast(np.ndarray, filtered[5:-5, 5:-5])


def structural_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Compute standard valid-window SSIM for equal-size grayscale images."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 2 or right_array.ndim != 2:
        raise ValueError("SSIM inputs must be two-dimensional grayscale arrays")
    if left_array.shape != right_array.shape:
        raise ValueError("SSIM inputs must have identical dimensions")
    if left_array.shape[0] < 11 or left_array.shape[1] < 11:
        raise ValueError("SSIM inputs must be at least 11x11 pixels")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("SSIM inputs must contain only finite values")

    mean_left = _valid_gaussian_filter(left_array)
    mean_right = _valid_gaussian_filter(right_array)
    mean_left_sq = mean_left * mean_left
    mean_right_sq = mean_right * mean_right
    mean_product = mean_left * mean_right
    variance_left = _valid_gaussian_filter(left_array * left_array) - mean_left_sq
    variance_right = _valid_gaussian_filter(right_array * right_array) - mean_right_sq
    covariance = _valid_gaussian_filter(left_array * right_array) - mean_product

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mean_product + c1) * (2.0 * covariance + c2)
    denominator = (mean_left_sq + mean_right_sq + c1) * (variance_left + variance_right + c2)
    return float(np.mean(numerator / denominator, dtype=np.float64))


def image_signature(path: Path) -> ImageSignature:
    """Compute all deterministic signatures for one image path."""

    image = _normalized_grayscale(path)
    return ImageSignature(
        filename=path.name,
        sha256=sha256_file(path),
        phash=perceptual_hash(path),
        width=image.width,
        height=image.height,
    )
