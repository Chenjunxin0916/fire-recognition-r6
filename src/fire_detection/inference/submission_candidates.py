"""Generate the three frozen-threshold competition submission candidates."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from fire_detection.ensemble.selection import validate_final_selection_manifest
from fire_detection.inference.submission import (
    list_test_filenames,
    load_submission_strict,
    predictions_to_submission,
    validate_submission_payload,
    write_submission,
)
from fire_detection.inference.test_prediction import validate_test_inference_manifest
from fire_detection.models.offline import sha256_file
from fire_detection.utils import atomic_write_json

PROFILE_NAMES = ("stable", "recall", "precision")


class SubmissionCandidateError(ValueError):
    """Raised when frozen predictions cannot produce strict candidate JSON files."""


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SubmissionCandidateError(f"{name} must be a JSON object")
    return value


def _threshold(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubmissionCandidateError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise SubmissionCandidateError(f"{name} must be finite and in [0, 1]")
    return result


def _validated_manifest_mapping(result: object) -> Mapping[str, Any]:
    value = result[0] if isinstance(result, tuple) else result
    return _mapping(value, name="validated final selection manifest")


def _validated_manifest_digest(result: object, *, name: str) -> str:
    if not isinstance(result, tuple) or len(result) != 2:
        raise SubmissionCandidateError(f"{name} validator must return manifest and SHA-256")
    digest = result[1]
    if not isinstance(digest, str) or len(digest) != 64:
        raise SubmissionCandidateError(f"{name} validator returned an invalid SHA-256")
    return digest


def _thresholds_from_manifest(manifest: Mapping[str, Any]) -> dict[str, float]:
    if manifest.get("status") != "complete":
        raise SubmissionCandidateError("final selection manifest status must be complete")
    raw_profiles = _mapping(
        manifest.get("threshold_profiles"),
        name="final selection threshold_profiles",
    )
    thresholds: dict[str, float] = {}
    for profile in PROFILE_NAMES:
        row = _mapping(raw_profiles.get(profile), name=f"threshold profile {profile}")
        thresholds[profile] = _threshold(
            row.get("threshold"),
            name=f"threshold profile {profile}.threshold",
        )
    if not thresholds["recall"] <= thresholds["stable"] <= thresholds["precision"]:
        raise SubmissionCandidateError(
            "threshold profiles must satisfy recall <= stable <= precision"
        )
    return thresholds


def load_submission_thresholds(
    selection_manifest: Path,
    *,
    project_root: Path,
) -> dict[str, float]:
    """Read stable/recall/precision thresholds from a revalidated final selection."""

    validated = validate_final_selection_manifest(
        selection_manifest,
        project_root=project_root,
    )
    return _thresholds_from_manifest(_validated_manifest_mapping(validated))


def _repository_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise SubmissionCandidateError(
            f"submission artifact must be inside the project root: {path}"
        ) from exc


def _filename_inventory_sha256(filenames: tuple[str, ...]) -> str:
    payload = "".join(f"{filename}\n" for filename in filenames).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_submission_candidates(
    *,
    predictions_path: Path,
    inference_manifest: Path,
    selection_manifest: Path,
    test_dir: Path,
    output_dir: Path,
    expected_count: int,
    project_root: Path,
) -> dict[str, Any]:
    """Atomically publish stable, recall, and precision JSON submissions.

    Thresholds are always read from the validated clean OOF selection manifest. The
    probability CSV is never converted with an implicit or caller-provided threshold.
    """

    if type(expected_count) is not int or expected_count <= 0:
        raise SubmissionCandidateError("expected_count must be a positive integer")
    root = project_root.resolve()
    predictions = predictions_path.resolve()
    inference = inference_manifest.resolve()
    selection = selection_manifest.resolve()
    images = test_dir.resolve()
    destination = output_dir.resolve()
    for path in (predictions, inference, selection):
        if not path.is_file():
            raise FileNotFoundError(path)
    _repository_path(predictions, root)
    _repository_path(inference, root)
    _repository_path(selection, root)
    _repository_path(images, root)
    _repository_path(destination, root)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite submission directory: {destination}")

    selection_result = validate_final_selection_manifest(selection, project_root=root)
    selection_payload = _validated_manifest_mapping(selection_result)
    selection_sha256 = _validated_manifest_digest(
        selection_result,
        name="final selection manifest",
    )
    thresholds = _thresholds_from_manifest(selection_payload)
    inference_result = validate_test_inference_manifest(inference, root)
    inference_payload = _validated_manifest_mapping(inference_result)
    inference_sha256 = _validated_manifest_digest(
        inference_result,
        name="test inference manifest",
    )
    inference_outputs = _mapping(
        inference_payload.get("outputs"),
        name="test inference outputs",
    )
    prediction_record = _mapping(
        inference_outputs.get("test_predictions_csv"),
        name="test inference predictions",
    )
    if prediction_record.get("path") != _repository_path(predictions, root):
        raise SubmissionCandidateError(
            "predictions path does not match the validated test inference manifest"
        )
    if prediction_record.get("sha256") != sha256_file(predictions):
        raise SubmissionCandidateError(
            "predictions SHA-256 does not match the validated test inference manifest"
        )
    inference_selection = _mapping(
        inference_payload.get("selection_manifest"),
        name="test inference selection manifest",
    )
    if (
        inference_selection.get("path") != _repository_path(selection, root)
        or inference_selection.get("sha256") != selection_sha256
    ):
        raise SubmissionCandidateError(
            "selection manifest does not match the validated test inference manifest"
        )
    if selection_payload.get("status") != "complete":
        raise SubmissionCandidateError("final selection manifest status must be complete")
    filenames = list_test_filenames(images, expected_count=expected_count)
    inference_images = _mapping(
        inference_payload.get("test_images"),
        name="test inference images",
    )
    image_records = inference_images.get("files")
    if not isinstance(image_records, list):
        raise SubmissionCandidateError("test inference image files must be a list")
    recorded_filenames = tuple(
        str(_mapping(record, name="test inference image record").get("filename"))
        for record in image_records
    )
    if recorded_filenames != filenames:
        raise SubmissionCandidateError(
            "test image filenames do not match the validated inference manifest"
        )
    try:
        frame = pd.read_csv(predictions)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise SubmissionCandidateError(f"cannot read predictions CSV {predictions}: {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    outputs: dict[str, dict[str, object]] = {}
    try:
        for profile in PROFILE_NAMES:
            payload = predictions_to_submission(
                frame,
                expected_filenames=filenames,
                threshold=thresholds[profile],
            )
            validate_submission_payload(
                payload,
                filenames,
                expected_count=expected_count,
            )
            filename = f"submission_{profile}.json"
            staged_path = staging / filename
            write_submission(staged_path, payload)
            reparsed = load_submission_strict(staged_path)
            validate_submission_payload(
                reparsed,
                filenames,
                expected_count=expected_count,
            )
            outputs[profile] = {
                "path": _repository_path(destination / filename, root),
                "sha256": sha256_file(staged_path),
                "threshold": thresholds[profile],
                "rows": len(payload),
            }

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "kind": "three_threshold_submission_candidates",
            "primary": "stable",
            "selection_manifest": {
                "path": _repository_path(selection, root),
                "sha256": selection_sha256,
            },
            "inference_manifest": {
                "path": _repository_path(inference, root),
                "sha256": inference_sha256,
            },
            "predictions": {
                "path": _repository_path(predictions, root),
                "sha256": sha256_file(predictions),
                "rows": len(frame),
            },
            "test_images": {
                "path": _repository_path(images, root),
                "count": len(filenames),
                "filenames_sha256": _filename_inventory_sha256(filenames),
            },
            "outputs": outputs,
        }
        atomic_write_json(staging / "submission_manifest.json", manifest)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def load_submission_candidate_manifest(path: Path) -> dict[str, Any]:
    """Load a generated candidate manifest for lightweight downstream inspection."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SubmissionCandidateError(f"cannot read submission manifest {path}: {exc}") from exc
    return dict(_mapping(value, name="submission candidate manifest"))
