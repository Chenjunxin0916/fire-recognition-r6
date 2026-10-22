"""Deterministic evaluation perturbations derived only from existing source images.

These operations produce in-memory views of the same sample. They never create
independent observations, change source pixels, or fit parameters to test data.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

AVAILABLE_CONDITIONS = (
    "clean",
    "brightness_0.7",
    "brightness_0.4",
    "blur_1",
    "blur_2",
    "noise_0.02",
    "noise_0.05",
)
_PARAMETERS: dict[str, dict[str, Any]] = {
    "clean": {"operation": "identity"},
    "brightness_0.7": {"operation": "brightness", "factor": 0.7},
    "brightness_0.4": {"operation": "brightness", "factor": 0.4},
    "blur_1": {"operation": "gaussian_blur", "radius_pixels": 1.0},
    "blur_2": {"operation": "gaussian_blur", "radius_pixels": 2.0},
    "noise_0.02": {"operation": "additive_gaussian_noise", "sigma_unit_rgb": 0.02},
    "noise_0.05": {"operation": "additive_gaussian_noise", "sigma_unit_rgb": 0.05},
}
_NOISE_NAMESPACE = "direction_a_noise_v1"


def condition_parameters(condition: str) -> dict[str, Any]:
    """Return a detached JSON-safe manifest description for one allowed condition."""
    if not isinstance(condition, str) or condition not in _PARAMETERS:
        raise ValueError(f"Unknown perturbation condition: {condition!r}")
    parameters = {
        "condition": condition,
        "input_mode": "RGB",
        "pixel_dtype": "uint8",
        "preserves_source_identity": True,
        "independent_sample": False,
        **_PARAMETERS[condition],
    }
    if condition.startswith("noise_"):
        parameters.update(
            rng="numpy.Generator(PCG64)",
            seed_namespace=_NOISE_NAMESPACE,
            seed_fields=["namespace", "seed", "source_filename"],
            seed_derivation="SHA256 canonical UTF-8 JSON; first 16 bytes as unsigned big-endian",
            json_serialization="sort_keys=True,ensure_ascii=False,separators=(',',':')",
            noise_severity_pairing="same standard normal field across noise conditions",
            output_quantization="numpy.rint followed by clip to [0,255] and uint8 cast",
        )
    return parameters


def source_noise_seed(seed: int, source_filename: str) -> int:
    """Derive a source-specific seed without mutable RNG state or call-order effects."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer, not bool")
    if not isinstance(source_filename, str) or not source_filename.strip():
        raise ValueError("source_filename must be a non-empty string")
    if any(character in source_filename for character in ("/", "\\", "\r", "\n", "\x00")):
        raise ValueError("source_filename must be a source identity, without directory separators")
    payload = json.dumps(
        {"namespace": _NOISE_NAMESPACE, "seed": seed, "source_filename": source_filename},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], byteorder="big", signed=False)


def apply_perturbation(
    image: Image.Image,
    condition: str = "clean",
    *,
    seed: int,
    source_filename: str,
) -> Image.Image:
    """Return one RGB view; keep the source image, shape and identity unchanged.

    Noise sigma is defined on RGB values divided by 255. The source identity and
    global seed determine its realization, so every method receives identical
    perturbations and severity levels share the same underlying normal field.
    Call this on each original decoded RGB image, never on another noisy view.
    """
    parameters = condition_parameters(condition)
    derived_seed = source_noise_seed(seed, source_filename)
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("image must be a PIL image already decoded in RGB mode")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image dimensions must be positive")
    source = image.copy()
    operation = parameters["operation"]
    if operation == "identity":
        return source
    if operation == "brightness":
        return ImageEnhance.Brightness(source).enhance(parameters["factor"])
    if operation == "gaussian_blur":
        return source.filter(ImageFilter.GaussianBlur(radius=parameters["radius_pixels"]))
    generator = np.random.Generator(np.random.PCG64(derived_seed))
    pixels = np.asarray(source, dtype=np.float64)
    normal_field = generator.standard_normal(pixels.shape)
    perturbed = np.rint(pixels + normal_field * (255.0 * parameters["sigma_unit_rgb"]))
    return Image.fromarray(np.clip(perturbed, 0, 255).astype(np.uint8))
