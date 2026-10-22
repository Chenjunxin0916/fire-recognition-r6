"""Strict final-candidate freezing from a completed fixed OOF comparison."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import numpy as np
import pandas as pd

from fire_detection.config import (
    ConfigError,
    load_yaml_mapping,
    require_float,
    require_int,
    require_mapping,
    require_string,
    resolve_project_path,
)
from fire_detection.inference.oof import load_oof_csv, validate_oof_artifact_manifest
from fire_detection.inference.tta import load_horizontal_flip_tta_spec
from fire_detection.metrics import compute_binary_metrics
from fire_detection.models.offline import sha256_file
from fire_detection.utils import atomic_write_csv, atomic_write_json

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_FOLDS = frozenset(range(5))
OOFKind = Literal["detector", "classifier"]


@dataclass(frozen=True)
class FinalSelectionSpec:
    """Validated, decision-complete contract for freezing the final clean winner."""

    project_root: Path
    config_path: Path
    name: str
    candidate: str
    baseline: str
    detector_source: str
    classifier_source: str
    detector_score_column: str
    classifier_score_column: str
    comparison_dir: Path
    artifact_root: Path
    expected_count: int
    output_dir: Path
    classifier_weight: float
    minimum_f1_gain: float
    minimum_folds_not_worse: int
    maximum_single_fold_drop: float
    reference_f1: float
    reference_recall: float
    f1_tolerance: float


def _probability(value: float, *, name: str, allow_zero: bool = True) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be finite and in [0, 1]")
    if not allow_zero and value <= 0.0:
        raise ConfigError(f"{name} must be greater than zero")
    return value


def load_final_selection_spec(path: Path, *, project_root: Path) -> FinalSelectionSpec:
    """Load final-selection policy without silent paths, gates, or metric defaults."""

    config_path = path if path.is_absolute() else project_root / path
    raw = load_yaml_mapping(config_path)
    selection = require_mapping(raw, "selection", context="final_selection")
    fusion = require_mapping(raw, "fusion", context="final_selection")
    promotion = require_mapping(raw, "promotion", context="final_selection")
    profiles = require_mapping(raw, "threshold_profiles", context="final_selection")

    if require_string(fusion, "method", context="final_selection.fusion") != ("weighted_average"):
        raise ConfigError("final_selection.fusion.method must be weighted_average")
    minimum_folds = require_int(
        promotion,
        "minimum_folds_not_worse",
        context="final_selection.promotion",
        minimum=0,
    )
    if minimum_folds > 5:
        raise ConfigError("final_selection.promotion.minimum_folds_not_worse must be <= 5")

    classifier_weight = _probability(
        require_float(
            fusion,
            "classifier_weight",
            context="final_selection.fusion",
        ),
        name="final_selection.fusion.classifier_weight",
    )
    minimum_gain = _probability(
        require_float(
            promotion,
            "minimum_f1_gain",
            context="final_selection.promotion",
        ),
        name="final_selection.promotion.minimum_f1_gain",
    )
    maximum_drop = _probability(
        require_float(
            promotion,
            "maximum_single_fold_drop",
            context="final_selection.promotion",
        ),
        name="final_selection.promotion.maximum_single_fold_drop",
    )
    reference_f1 = _probability(
        require_float(promotion, "reference_f1", context="final_selection.promotion"),
        name="final_selection.promotion.reference_f1",
    )
    reference_recall = _probability(
        require_float(
            promotion,
            "reference_recall",
            context="final_selection.promotion",
        ),
        name="final_selection.promotion.reference_recall",
    )
    f1_tolerance = _probability(
        require_float(
            profiles,
            "f1_tolerance",
            context="final_selection.threshold_profiles",
        ),
        name="final_selection.threshold_profiles.f1_tolerance",
        allow_zero=False,
    )

    candidate = require_string(selection, "candidate", context="final_selection.selection")
    detector_source = require_string(
        selection, "detector_source", context="final_selection.selection"
    )
    classifier_source = require_string(
        selection, "classifier_source", context="final_selection.selection"
    )
    expected_candidate = f"fusion:{detector_source}+{classifier_source}"
    if candidate != expected_candidate:
        raise ConfigError(
            "final_selection.selection.candidate must equal "
            f"{expected_candidate!r} for the declared component sources"
        )

    return FinalSelectionSpec(
        project_root=project_root.resolve(),
        config_path=config_path.resolve(),
        name=require_string(selection, "name", context="final_selection.selection"),
        candidate=candidate,
        baseline=require_string(selection, "baseline", context="final_selection.selection"),
        detector_source=detector_source,
        classifier_source=classifier_source,
        detector_score_column=require_string(
            selection,
            "detector_score_column",
            context="final_selection.selection",
        ),
        classifier_score_column=require_string(
            selection,
            "classifier_score_column",
            context="final_selection.selection",
        ),
        comparison_dir=resolve_project_path(
            project_root,
            require_string(selection, "comparison_dir", context="final_selection.selection"),
        ).resolve(),
        artifact_root=resolve_project_path(
            project_root,
            require_string(selection, "artifact_root", context="final_selection.selection"),
        ).resolve(),
        expected_count=require_int(
            selection,
            "expected_count",
            context="final_selection.selection",
            minimum=1,
        ),
        output_dir=resolve_project_path(
            project_root,
            require_string(selection, "output_dir", context="final_selection.selection"),
        ).resolve(),
        classifier_weight=classifier_weight,
        minimum_f1_gain=minimum_gain,
        minimum_folds_not_worse=minimum_folds,
        maximum_single_fold_drop=maximum_drop,
        reference_f1=reference_f1,
        reference_recall=reference_recall,
        f1_tolerance=f1_tolerance,
    )


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {name} {path}: {exc}") from exc
    return dict(_mapping(value, name=name))


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"final selection artifact must be inside project root: {path}") from exc


def _resolve_portable_path(value: object, *, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}.path must be a non-empty string")
    raw = value.strip()
    portable = PurePosixPath(raw.replace("\\", "/"))
    if portable.is_absolute() or PureWindowsPath(raw).is_absolute() or ".." in portable.parts:
        raise ValueError(f"{name}.path must be repository-relative")
    resolved = (root / Path(*portable.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name}.path escapes the project root") from exc
    return resolved


def _verified_file_record(
    value: object,
    *,
    root: Path,
    name: str,
    expected_path: Path | None = None,
) -> tuple[Path, dict[str, str]]:
    record = _mapping(value, name=name)
    path = _resolve_portable_path(record.get("path"), root=root, name=name)
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{name}.path does not match the required artifact")
    expected_hash = _sha256(record.get("sha256"), name=f"{name}.sha256")
    if not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{name}.sha256 drifted: expected {expected_hash}, found {actual_hash}")
    return path, {"path": _portable_path(path, root), "sha256": actual_hash}


def _csv_row_count(path: Path) -> int:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            if next(reader, None) is None:
                raise ValueError(f"CSV has no header: {path}")
            return sum(1 for _row in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"cannot count CSV rows in {path}: {exc}") from exc


def _verify_comparison_outputs(
    metrics: Mapping[str, Any],
    *,
    spec: FinalSelectionSpec,
) -> dict[str, dict[str, object]]:
    outputs = _mapping(metrics.get("outputs"), name="comparison metrics.outputs")
    verified: dict[str, dict[str, object]] = {}
    for filename, raw_record in outputs.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("comparison output keys must be plain filenames")
        expected_path = spec.comparison_dir / filename
        path, record = _verified_file_record(
            raw_record,
            root=spec.project_root,
            name=f"comparison output {filename}",
            expected_path=expected_path,
        )
        output_record = _mapping(raw_record, name=f"comparison output {filename}")
        rows = _strict_int(output_record.get("rows"), name=f"comparison output {filename}.rows")
        if _csv_row_count(path) != rows:
            raise ValueError(f"comparison output {filename}.rows does not match the CSV")
        verified[filename] = {**record, "rows": rows}
    required = {
        "candidate_metrics.csv",
        "comparison_fold_metrics.csv",
        "comparison_summary.csv",
        "aligned_predictions.csv",
        "threshold_curves.csv",
    }
    missing = sorted(required.difference(verified))
    if missing:
        raise ValueError(f"comparison metrics omit required outputs: {missing}")
    return verified


def _source_record(
    metrics: Mapping[str, Any],
    *,
    kind: OOFKind,
    source_name: str,
) -> Mapping[str, Any]:
    matrix = _mapping(metrics.get("fixed_matrix"), name="comparison metrics.fixed_matrix")
    key = "detectors" if kind == "detector" else "classifiers"
    raw_sources = matrix.get(key)
    if not isinstance(raw_sources, list):
        raise ValueError(f"comparison metrics.fixed_matrix.{key} must be a list")
    matches = [
        _mapping(item, name=f"fixed_matrix.{key} item")
        for item in raw_sources
        if isinstance(item, Mapping) and item.get("name") == source_name
    ]
    if len(matches) != 1:
        raise ValueError(f"fixed matrix must contain source {source_name!r} exactly once")
    source = matches[0]
    if source.get("kind") != kind:
        raise ValueError(f"fixed-matrix source {source_name!r} has the wrong kind")
    return source


def _validated_component_source(
    metrics: Mapping[str, Any],
    *,
    spec: FinalSelectionSpec,
    kind: OOFKind,
    source_name: str,
    score_column: str,
    folds_path: Path,
    processed_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str], dict[str, str], dict[str, str]]:
    source = _source_record(metrics, kind=kind, source_name=source_name)
    if source.get("score_column") != score_column:
        raise ValueError(
            f"fixed-matrix source {source_name!r} score column is not {score_column!r}"
        )
    oof_path, oof_record = _verified_file_record(
        source.get("oof"), root=spec.project_root, name=f"{kind} source OOF"
    )
    source_oof = _mapping(source.get("oof"), name=f"{kind} source OOF")
    if _strict_int(source_oof.get("rows"), name=f"{kind} source OOF.rows", minimum=1) != (
        spec.expected_count
    ):
        raise ValueError(f"{kind} source OOF row count does not equal expected_count")
    manifest_path, manifest_record = _verified_file_record(
        source.get("manifest"),
        root=spec.project_root,
        name=f"{kind} source OOF manifest",
    )
    manifest, manifest_hash = validate_oof_artifact_manifest(
        manifest_path,
        artifact_root=spec.artifact_root,
        kind=kind,
        oof_path=oof_path,
        reference_folds_path=folds_path,
        processed_manifest_path=processed_manifest_path,
        expected_rows=spec.expected_count,
    )
    if manifest_hash != manifest_record["sha256"]:
        raise ValueError(f"{kind} OOF manifest hash changed during validation")
    frame = load_oof_csv(
        oof_path,
        kind=kind,
        expected_rows=spec.expected_count,
        reference_folds_path=folds_path,
        expected_folds_sha256=sha256_file(folds_path),
        expected_processed_manifest_sha256=sha256_file(processed_manifest_path),
        artifact_root=spec.artifact_root,
    )
    if score_column not in frame:
        raise ValueError(f"{kind} OOF does not contain selected score column {score_column!r}")
    section = _mapping(manifest.get(kind), name=f"{kind} OOF manifest section")
    config_path, config_record = _verified_file_record(
        section.get("config"), root=spec.project_root, name=f"{kind} training config"
    )
    config_record = {"path": _portable_path(config_path, spec.project_root), **config_record}
    return frame, manifest, oof_record, manifest_record, config_record


def _selected_tta_contract(
    detector_manifest: Mapping[str, Any],
    classifier_manifest: Mapping[str, Any],
    *,
    spec: FinalSelectionSpec,
) -> dict[str, object] | None:
    """Bind a shared detector/classifier TTA config when both selected OOFs use it."""

    detector_section = _mapping(
        detector_manifest.get("detector"), name="detector OOF manifest detector"
    )
    classifier_section = _mapping(
        classifier_manifest.get("classifier"), name="classifier OOF manifest classifier"
    )
    detector_tta = detector_section.get("tta_config")
    classifier_tta = classifier_section.get("tta_config")
    if detector_tta is None and classifier_tta is None:
        return None
    if detector_tta is None or classifier_tta is None:
        raise ValueError("selected detector/classifier OOF must declare TTA together")
    detector_path, detector_record = _verified_file_record(
        detector_tta,
        root=spec.project_root,
        name="selected detector TTA config",
    )
    classifier_path, classifier_record = _verified_file_record(
        classifier_tta,
        root=spec.project_root,
        name="selected classifier TTA config",
    )
    if detector_path != classifier_path or detector_record != classifier_record:
        raise ValueError("selected detector/classifier OOF TTA configs differ")
    tta_spec = load_horizontal_flip_tta_spec(detector_path)
    if tta_spec.config_sha256 != detector_record["sha256"]:
        raise ValueError("selected TTA config hash changed during validation")
    expected_contract = tta_spec.manifest_record()
    for kind, manifest in (
        ("detector", detector_manifest),
        ("classifier", classifier_manifest),
    ):
        git = _mapping(manifest.get("git"), name=f"{kind} OOF manifest git")
        commit = git.get("commit")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError(f"{kind} TTA OOF manifest git.commit is invalid")
        if git.get("dirty") is not False:
            raise ValueError(f"{kind} TTA OOF manifest is dirty")
        recorded = _mapping(manifest.get("tta"), name=f"{kind} OOF manifest TTA")
        without_config = {key: value for key, value in recorded.items() if key != "config"}
        if without_config != expected_contract:
            raise ValueError(f"{kind} OOF manifest TTA algorithm contract drifted")
        recorded_config = _mapping(recorded.get("config"), name=f"{kind} OOF manifest TTA config")
        if recorded_config.get("sha256") != detector_record["sha256"]:
            raise ValueError(f"{kind} OOF manifest TTA config hash drifted")
    return {
        **expected_contract,
        "config": {
            "path": _portable_path(detector_path, spec.project_root),
            **detector_record,
        },
    }


def _clean_git_record(manifest: Mapping[str, Any], *, name: str) -> dict[str, object]:
    git = _mapping(manifest.get("git"), name=f"{name}.git")
    if git.get("dirty") is not False:
        raise ValueError(f"{name} is dirty")
    commit = git.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"{name}.git.commit is invalid")
    return {"commit": commit, "dirty": False}


def _checkpoint_file_record(
    checkpoint: Mapping[str, Any],
    *,
    path_field: str,
    hash_field: str,
    root: Path,
    name: str,
) -> tuple[Path, dict[str, str]]:
    path = _resolve_portable_path(checkpoint.get(path_field), root=root, name=name)
    expected_hash = _sha256(checkpoint.get(hash_field), name=f"{name}.{hash_field}")
    if not path.is_file():
        raise ValueError(f"{name} is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{name} hash drifted")
    return path, {"path": _portable_path(path, root), "sha256": actual_hash}


def _clean_nested_provenance(
    replay_path: Path,
    *,
    outer_fold: int,
    spec: FinalSelectionSpec,
    folds_record: Mapping[str, str],
    processed_record: Mapping[str, str],
) -> dict[str, Any]:
    """Follow one replay manifest through its nested OOF to four clean miners."""

    replay = _load_json(replay_path, name=f"classifier Fold {outer_fold} replay manifest")
    if replay.get("schema_version") != 1 or replay.get("kind") != ("cross_fitted_hard_negatives"):
        raise ValueError(f"classifier Fold {outer_fold} replay manifest schema/kind mismatch")
    if replay.get("outer_fold") != outer_fold:
        raise ValueError(f"classifier Fold {outer_fold} replay outer_fold mismatch")
    campaign_path, campaign_record = _verified_file_record(
        replay.get("nested_campaign_manifest"),
        root=spec.project_root,
        name=f"classifier Fold {outer_fold} nested campaign",
    )
    nested_oof_path, nested_oof_record = _verified_file_record(
        replay.get("nested_oof"),
        root=spec.project_root,
        name=f"classifier Fold {outer_fold} nested OOF",
    )
    replay_oof = _mapping(replay.get("nested_oof"), name=f"classifier Fold {outer_fold} nested OOF")
    nested_rows = _strict_int(
        replay_oof.get("rows"),
        name=f"classifier Fold {outer_fold} nested OOF.rows",
        minimum=1,
    )
    nested_manifest_path = nested_oof_path.with_name("oof_manifest.json")
    nested_manifest = _load_json(
        nested_manifest_path, name=f"classifier Fold {outer_fold} nested OOF manifest"
    )
    expected_scalars: dict[str, object] = {
        "schema_version": 1,
        "kind": "nested_classifier_oof",
        "outer_fold": outer_fold,
        "rows": nested_rows,
        "checkpoint_policy": "last",
        "folds_sha256": folds_record["sha256"],
        "processed_manifest_sha256": processed_record["sha256"],
        "campaign_manifest_sha256": campaign_record["sha256"],
    }
    for field, expected in expected_scalars.items():
        if type(nested_manifest.get(field)) is not type(expected) or (
            nested_manifest.get(field) != expected
        ):
            raise ValueError(f"classifier Fold {outer_fold} nested OOF manifest {field} mismatch")
    classifier_oof = _mapping(
        nested_manifest.get("classifier_oof"),
        name=f"classifier Fold {outer_fold} nested OOF manifest classifier_oof",
    )
    declared_oof_path = _resolve_portable_path(
        classifier_oof.get("path"),
        root=spec.project_root,
        name=f"classifier Fold {outer_fold} nested OOF manifest classifier_oof",
    )
    if (
        declared_oof_path != nested_oof_path
        or classifier_oof.get("sha256") != (nested_oof_record["sha256"])
    ):
        raise ValueError(f"classifier Fold {outer_fold} nested OOF manifest does not bind its CSV")

    raw_checkpoints = nested_manifest.get("checkpoints")
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != 4:
        raise ValueError(f"classifier Fold {outer_fold} nested OOF must bind four checkpoints")
    expected_inner = EXPECTED_FOLDS - {outer_fold}
    seen_inner: set[int] = set()
    miners: list[dict[str, Any]] = []
    for offset, raw_checkpoint in enumerate(raw_checkpoints):
        checkpoint = _mapping(
            raw_checkpoint,
            name=f"classifier Fold {outer_fold} nested checkpoint {offset}",
        )
        inner_fold = _strict_int(
            checkpoint.get("inner_fold"),
            name=f"classifier Fold {outer_fold} nested checkpoint inner_fold",
        )
        if inner_fold not in expected_inner or inner_fold in seen_inner:
            raise ValueError(f"classifier Fold {outer_fold} has invalid/duplicate inner Fold")
        seen_inner.add(inner_fold)
        run_path, run_record = _checkpoint_file_record(
            checkpoint,
            path_field="run_manifest",
            hash_field="run_manifest_sha256",
            root=spec.project_root,
            name=f"miner outer {outer_fold} inner {inner_fold} run manifest",
        )
        weights_path, weights_record = _checkpoint_file_record(
            checkpoint,
            path_field="last_weights",
            hash_field="last_weights_sha256",
            root=spec.project_root,
            name=f"miner outer {outer_fold} inner {inner_fold} weights",
        )
        split_path, split_record = _checkpoint_file_record(
            checkpoint,
            path_field="split_manifest",
            hash_field="split_manifest_sha256",
            root=spec.project_root,
            name=f"miner outer {outer_fold} inner {inner_fold} split manifest",
        )
        run = _load_json(run_path, name=f"miner outer {outer_fold} inner {inner_fold} run")
        run_expected: dict[str, object] = {
            "schema_version": 1,
            "status": "complete",
            "task": "classify",
            "outer_fold": outer_fold,
            "validation_fold": inner_fold,
            "folds_path": folds_record["path"],
            "folds_sha256": folds_record["sha256"],
            "processed_manifest": processed_record["path"],
            "processed_manifest_sha256": processed_record["sha256"],
            "last_weights": weights_record["path"],
            "last_weights_sha256": weights_record["sha256"],
            "split_manifest": split_record["path"],
            "split_manifest_sha256": split_record["sha256"],
        }
        for field, expected in run_expected.items():
            if type(run.get(field)) is not type(expected) or run.get(field) != expected:
                raise ValueError(
                    f"miner outer {outer_fold} inner {inner_fold} run {field} mismatch"
                )
        git_record = _clean_git_record(
            run, name=f"miner outer {outer_fold} inner {inner_fold} run manifest"
        )
        config_path = _resolve_portable_path(
            run.get("config_path"),
            root=spec.project_root,
            name=f"miner outer {outer_fold} inner {inner_fold} config",
        )
        config_hash = _sha256(
            run.get("config_sha256"),
            name=f"miner outer {outer_fold} inner {inner_fold} config_sha256",
        )
        if not config_path.is_file() or sha256_file(config_path) != config_hash:
            raise ValueError(f"miner outer {outer_fold} inner {inner_fold} config drifted")
        if weights_path.name != "last.pt" or split_path.name != "split_manifest.json":
            raise ValueError(
                f"miner outer {outer_fold} inner {inner_fold} artifact names are invalid"
            )
        miners.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "git": git_record,
                "config": {
                    "path": _portable_path(config_path, spec.project_root),
                    "sha256": config_hash,
                },
                "run_manifest": run_record,
                "weights": weights_record,
                "split_manifest": split_record,
            }
        )
    if seen_inner != expected_inner:
        raise ValueError(
            f"classifier Fold {outer_fold} nested OOF inner Fold coverage is incomplete"
        )
    return {
        "campaign_manifest": {
            "path": _portable_path(campaign_path, spec.project_root),
            **campaign_record,
        },
        "nested_oof": {**nested_oof_record, "rows": nested_rows},
        "nested_oof_manifest": {
            "path": _portable_path(nested_manifest_path, spec.project_root),
            "sha256": sha256_file(nested_manifest_path),
        },
        "miners": sorted(miners, key=lambda item: int(item["inner_fold"])),
    }


def _model_records(
    frame: pd.DataFrame,
    *,
    spec: FinalSelectionSpec,
    task: OOFKind,
    training_config: Mapping[str, str],
    folds_record: Mapping[str, str],
    processed_record: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Bind five OOF checkpoints to complete, clean formal training manifests."""

    models: list[dict[str, Any]] = []
    seen_weight_hashes: set[str] = set()
    for fold in sorted(EXPECTED_FOLDS):
        fold_rows = frame.loc[frame["fold"] == fold]
        if fold_rows.empty:
            raise ValueError(f"{task} OOF has no rows for Fold {fold}")
        weight_paths = fold_rows["weights_path"].drop_duplicates().tolist()
        weight_hashes = fold_rows["weights_sha256"].drop_duplicates().tolist()
        if len(weight_paths) != 1 or len(weight_hashes) != 1:
            raise ValueError(f"{task} OOF Fold {fold} must bind exactly one checkpoint")
        weight_path = _resolve_portable_path(
            weight_paths[0], root=spec.artifact_root, name=f"{task} Fold {fold} weights"
        )
        weight_hash = _sha256(weight_hashes[0], name=f"{task} Fold {fold} weights_sha256")
        if not weight_path.is_file() or sha256_file(weight_path) != weight_hash:
            raise ValueError(f"{task} Fold {fold} checkpoint is missing or changed")
        if weight_hash in seen_weight_hashes:
            raise ValueError(f"{task} Fold checkpoints must have five distinct hashes")
        seen_weight_hashes.add(weight_hash)

        run_manifest_path = weight_path.parent.parent / "run_manifest.json"
        run_manifest = _load_json(run_manifest_path, name=f"{task} Fold {fold} run manifest")
        expected_scalars: dict[str, object] = {
            "schema_version": 1,
            "status": "complete",
            "task": "detect" if task == "detector" else "classify",
            "validation_fold": fold,
            "config_path": training_config["path"],
            "config_sha256": training_config["sha256"],
            "folds_path": folds_record["path"],
            "folds_sha256": folds_record["sha256"],
            "processed_manifest": processed_record["path"],
            "processed_manifest_sha256": processed_record["sha256"],
            "last_weights": _portable_path(weight_path, spec.project_root),
            "last_weights_sha256": weight_hash,
        }
        for field, expected in expected_scalars.items():
            if type(run_manifest.get(field)) is not type(expected) or (
                run_manifest.get(field) != expected
            ):
                raise ValueError(
                    f"{task} Fold {fold} run manifest {field} mismatch: "
                    f"expected {expected!r}, found {run_manifest.get(field)!r}"
                )
        git_record = _clean_git_record(run_manifest, name=f"{task} Fold {fold} formal run manifest")

        model: dict[str, Any] = {
            "fold": fold,
            "git": git_record,
            "run_manifest": {
                "path": _portable_path(run_manifest_path, spec.project_root),
                "sha256": sha256_file(run_manifest_path),
            },
            "weights": {
                "path": _portable_path(weight_path, spec.project_root),
                "sha256": weight_hash,
            },
        }
        replay_value = run_manifest.get("hard_negative_replay")
        if replay_value is not None:
            replay = _mapping(replay_value, name=f"{task} Fold {fold} hard_negative_replay")
            replay_path = _resolve_portable_path(
                replay.get("manifest"),
                root=spec.project_root,
                name=f"{task} Fold {fold} hard-negative replay",
            )
            replay_hash = _sha256(
                replay.get("manifest_sha256"),
                name=f"{task} Fold {fold} hard-negative replay SHA-256",
            )
            if not replay_path.is_file() or sha256_file(replay_path) != replay_hash:
                raise ValueError(f"{task} Fold {fold} hard-negative replay is missing or changed")
            model["hard_negative_replay"] = {
                "path": _portable_path(replay_path, spec.project_root),
                "sha256": replay_hash,
                "repeat_factor": _strict_int(
                    replay.get("repeat_factor"),
                    name=f"{task} Fold {fold} hard-negative repeat_factor",
                    minimum=2,
                ),
                **_clean_nested_provenance(
                    replay_path,
                    outer_fold=fold,
                    spec=spec,
                    folds_record=folds_record,
                    processed_record=processed_record,
                ),
            }
        models.append(model)
    return models


def _metric_counts(
    frame: pd.DataFrame,
    *,
    prediction_column: str = "y_pred",
) -> dict[str, float | int]:
    metrics = compute_binary_metrics(
        frame["label"].astype(int).tolist(),
        frame[prediction_column].astype(int).tolist(),
        0.5,
    )
    return asdict(metrics)


def _selected_cross_fitted(
    aligned: pd.DataFrame,
    *,
    candidate: str,
    expected_count: int,
) -> pd.DataFrame:
    required = {
        "comparison_candidate",
        "filename",
        "baseline_score",
        "baseline_threshold",
        "baseline_y_pred",
        "baseline_confusion",
        "label",
        "fold",
        "candidate_score",
        "candidate_threshold",
        "candidate_y_pred",
        "candidate_confusion",
        "transition",
        "contains_small_fire",
        "all_boxes_small",
    }
    missing = sorted(required.difference(aligned.columns))
    if missing:
        raise ValueError(f"aligned predictions omit required columns: {missing}")
    selected = aligned.loc[aligned["comparison_candidate"] == candidate].copy()
    if len(selected) != expected_count:
        raise ValueError(
            f"selected candidate has {len(selected)} aligned rows, expected {expected_count}"
        )
    selected = selected.rename(
        columns={
            "candidate_score": "score",
            "candidate_threshold": "threshold",
            "candidate_y_pred": "y_pred",
            "candidate_confusion": "confusion",
        }
    )[
        [
            "filename",
            "label",
            "fold",
            "score",
            "threshold",
            "y_pred",
            "confusion",
            "baseline_score",
            "baseline_threshold",
            "baseline_y_pred",
            "baseline_confusion",
            "transition",
            "contains_small_fire",
            "all_boxes_small",
        ]
    ]
    if selected["filename"].duplicated().any():
        raise ValueError("selected cross-fitted predictions contain duplicate filenames")
    if set(int(value) for value in selected["fold"].unique()) != EXPECTED_FOLDS:
        raise ValueError("selected cross-fitted predictions must contain Fold 0..4")
    for column in ("score", "threshold", "baseline_score", "baseline_threshold"):
        values = pd.to_numeric(selected[column], errors="raise").astype(float)
        if (~np.isfinite(values)).any() or ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"selected cross-fitted {column} must be finite and in [0, 1]")
        selected[column] = values
    labels = pd.to_numeric(selected["label"], errors="raise").astype(int)
    predictions = pd.to_numeric(selected["y_pred"], errors="raise").astype(int)
    baseline_predictions = pd.to_numeric(selected["baseline_y_pred"], errors="raise").astype(int)
    if (
        not labels.isin([0, 1]).all()
        or not predictions.isin([0, 1]).all()
        or not baseline_predictions.isin([0, 1]).all()
    ):
        raise ValueError("selected labels and predictions must be integer 0/1")
    expected_predictions = (selected["score"] >= selected["threshold"]).astype(int)
    if not (predictions == expected_predictions).all():
        raise ValueError("selected y_pred does not match score >= per-Fold threshold")
    expected_baseline_predictions = (
        selected["baseline_score"] >= selected["baseline_threshold"]
    ).astype(int)
    if not (baseline_predictions == expected_baseline_predictions).all():
        raise ValueError("baseline_y_pred does not match baseline_score >= threshold")
    confusion = [
        {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}[(label, pred)]
        for label, pred in zip(labels, predictions, strict=True)
    ]
    if selected["confusion"].tolist() != confusion:
        raise ValueError("selected confusion labels do not match label/y_pred")
    baseline_confusion = [
        {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}[(label, pred)]
        for label, pred in zip(labels, baseline_predictions, strict=True)
    ]
    if selected["baseline_confusion"].tolist() != baseline_confusion:
        raise ValueError("baseline confusion labels do not match label/y_pred")
    transitions = [
        f"{baseline}->{candidate}"
        for baseline, candidate in zip(baseline_confusion, confusion, strict=True)
    ]
    if selected["transition"].tolist() != transitions:
        raise ValueError("selected transitions do not match baseline/candidate confusion")
    for column in ("contains_small_fire", "all_boxes_small"):
        boolean_values: list[bool] = []
        for index, value in selected[column].items():
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"selected {column}[{index}] must be boolean")
            boolean_values.append(bool(value))
        selected[column] = boolean_values
    selected["label"] = labels
    selected["fold"] = pd.to_numeric(selected["fold"], errors="raise").astype(int)
    selected["y_pred"] = predictions
    selected["baseline_y_pred"] = baseline_predictions
    selected["margin"] = selected["score"] - selected["threshold"]
    columns = [
        "filename",
        "label",
        "fold",
        "score",
        "threshold",
        "y_pred",
        "confusion",
        "margin",
        "baseline_score",
        "baseline_threshold",
        "baseline_y_pred",
        "baseline_confusion",
        "transition",
        "contains_small_fire",
        "all_boxes_small",
    ]
    return selected[columns].sort_values("filename", kind="stable").reset_index(drop=True)


def _threshold_point(row: pd.Series, *, rule: str) -> dict[str, object]:
    return {
        "threshold": _finite_float(row["threshold"], name="threshold profile threshold"),
        "rule": rule,
        "pooled_precision": _finite_float(row["precision"], name="threshold precision"),
        "pooled_recall": _finite_float(row["recall"], name="threshold recall"),
        "pooled_f1": _finite_float(row["f1"], name="threshold F1"),
    }


def build_threshold_profiles(
    curve: pd.DataFrame,
    *,
    candidate: str,
    classifier_weight: float,
    stable_threshold: float,
    f1_tolerance: float,
) -> dict[str, object]:
    """Return best pooled threshold and the low/high F1-band endpoints."""

    required = {
        "candidate",
        "classifier_weight",
        "threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    }
    missing = sorted(required.difference(curve.columns))
    if missing:
        raise ValueError(f"threshold curve omits required columns: {missing}")
    selected = curve.loc[curve["candidate"] == candidate].copy()
    if selected.empty:
        raise ValueError(f"threshold curve has no rows for {candidate!r}")
    for column in ("classifier_weight", "threshold", "precision", "recall", "f1"):
        selected[column] = pd.to_numeric(selected[column], errors="raise").astype(float)
        if (~np.isfinite(selected[column])).any():
            raise ValueError(f"threshold curve {column} contains non-finite values")
    for column in ("tp", "fp", "fn", "tn"):
        values = pd.to_numeric(selected[column], errors="raise")
        if (
            (~np.isfinite(values)).any()
            or (values < 0).any()
            or (values.astype(float) % 1.0 != 0.0).any()
        ):
            raise ValueError(f"threshold curve {column} must contain non-negative integers")
        selected[column] = values.astype(int)
    if selected["threshold"].duplicated().any():
        raise ValueError("selected threshold curve contains duplicate thresholds")
    weights = selected["classifier_weight"].drop_duplicates().tolist()
    if len(weights) != 1 or weights[0] != classifier_weight:
        raise ValueError("selected threshold curve classifier weight does not match final fusion")
    best_f1 = float(selected["f1"].max())
    f1_winners = selected.loc[selected["f1"].map(lambda value: _is_close(float(value), best_f1))]
    best_recall = float(f1_winners["recall"].max())
    recall_winners = f1_winners.loc[
        f1_winners["recall"].map(lambda value: _is_close(float(value), best_recall))
    ]
    plateau_centers: list[pd.Series] = []
    for _, plateau in recall_winners.groupby(["tp", "fp", "fn", "tn"], sort=False):
        ordered = plateau.sort_values("threshold", kind="stable").reset_index(drop=True)
        plateau_centers.append(ordered.iloc[(len(ordered) - 1) // 2])
    expected_stable = min(
        plateau_centers,
        key=lambda row: _finite_float(row["threshold"], name="stable threshold winner"),
    )
    expected_stable_threshold = _finite_float(
        expected_stable["threshold"], name="stable threshold winner"
    )
    if not _is_close(stable_threshold, expected_stable_threshold):
        raise ValueError(
            "pooled stable threshold does not match the independently recomputed F1/Recall winner"
        )
    stable_rows = selected.loc[
        selected["threshold"].map(lambda value: _is_close(float(value), stable_threshold))
    ]
    if len(stable_rows) != 1:
        raise ValueError("pooled stable threshold is absent or duplicated in threshold curve")
    stable_row = stable_rows.iloc[0]
    band = selected.loc[selected["f1"] >= best_f1 - f1_tolerance]
    if band.empty:
        raise ValueError("F1 tolerance band is unexpectedly empty")
    recall_row = band.loc[band["threshold"].idxmin()]
    precision_row = band.loc[band["threshold"].idxmax()]
    return {
        "f1_tolerance": f1_tolerance,
        "pooled_best_f1": best_f1,
        "stable": _threshold_point(
            stable_row,
            rule="pooled_max_f1_then_recall_then_lower_middle_prediction_plateau",
        ),
        "recall": _threshold_point(
            recall_row,
            rule="lowest_threshold_with_pooled_f1_at_least_best_minus_tolerance",
        ),
        "precision": _threshold_point(
            precision_row,
            rule="highest_threshold_with_pooled_f1_at_least_best_minus_tolerance",
        ),
    }


def _is_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _single_row(frame: pd.DataFrame, *, column: str, value: str, name: str) -> pd.Series:
    selected = frame.loc[frame[column] == value]
    if len(selected) != 1:
        raise ValueError(f"{name} must contain {value!r} exactly once")
    return selected.iloc[0]


def _metric_record(row: pd.Series, *, name: str) -> dict[str, float | int]:
    return {
        "precision": _finite_float(row["precision"], name=f"{name}.precision"),
        "recall": _finite_float(row["recall"], name=f"{name}.recall"),
        "f1": _finite_float(row["f1"], name=f"{name}.f1"),
        "tp": _strict_int(row["tp"], name=f"{name}.tp"),
        "fp": _strict_int(row["fp"], name=f"{name}.fp"),
        "fn": _strict_int(row["fn"], name=f"{name}.fn"),
        "tn": _strict_int(row["tn"], name=f"{name}.tn"),
    }


def _verify_promotion_and_metrics(
    metrics: Mapping[str, Any],
    *,
    spec: FinalSelectionSpec,
) -> tuple[dict[str, float | int], dict[str, object], float, dict[str, object]]:
    if metrics.get("schema_version") != 1 or metrics.get("evaluation_contract") != (
        "cross_fitted_leave_one_fold_out"
    ):
        raise ValueError("comparison metrics schema/evaluation contract mismatch")
    baseline = _mapping(metrics.get("baseline"), name="comparison metrics.baseline")
    if baseline.get("name") != spec.baseline:
        raise ValueError("comparison baseline does not match final-selection policy")
    gates = _mapping(metrics.get("promotion_gates"), name="comparison promotion_gates")
    gate_values = {
        "minimum_f1_gain": spec.minimum_f1_gain,
        "minimum_folds_not_worse": spec.minimum_folds_not_worse,
        "maximum_single_fold_drop": spec.maximum_single_fold_drop,
    }
    for field, expected in gate_values.items():
        actual = gates.get(field)
        if isinstance(expected, int):
            if type(actual) is not int or actual != expected:
                raise ValueError(f"comparison promotion gate {field} drifted")
        elif not _is_close(_finite_float(actual, name=f"promotion gate {field}"), expected):
            raise ValueError(f"comparison promotion gate {field} drifted")

    candidate_metrics = pd.read_csv(spec.comparison_dir / "candidate_metrics.csv")
    candidate = _single_row(
        candidate_metrics,
        column="candidate",
        value=spec.candidate,
        name="candidate_metrics.csv",
    )
    if (
        candidate.get("method") != "weighted_average"
        or candidate.get("detector") != spec.detector_source
        or candidate.get("classifier") != spec.classifier_source
    ):
        raise ValueError("selected candidate component declaration drifted")
    classifier_weight = _finite_float(
        candidate["pooled_classifier_weight"], name="pooled classifier weight"
    )
    if not _is_close(classifier_weight, spec.classifier_weight):
        raise ValueError("selected pooled classifier weight does not match the frozen weight")
    stable_threshold = _finite_float(candidate["pooled_threshold"], name="pooled stable threshold")
    _probability(stable_threshold, name="pooled stable threshold")
    cross_fitted = _metric_record(candidate, name="selected cross-fitted metrics")

    summary = pd.read_csv(spec.comparison_dir / "comparison_summary.csv")
    decision = _single_row(
        summary,
        column="candidate",
        value=spec.candidate,
        name="comparison_summary.csv",
    )
    if not _strict_bool(decision["promoted"], name="selected promotion decision"):
        raise ValueError("selected candidate was not promoted by the comparison")
    baseline_cross = _mapping(baseline.get("cross_fitted"), name="comparison baseline.cross_fitted")
    baseline_f1 = _finite_float(baseline_cross.get("f1"), name="baseline F1")
    expected_gain = float(cross_fitted["f1"]) - baseline_f1

    fold_metrics = pd.read_csv(spec.comparison_dir / "comparison_fold_metrics.csv")
    selected_folds = fold_metrics.loc[fold_metrics["comparison_candidate"] == spec.candidate].copy()
    if len(selected_folds) != 5:
        raise ValueError("selected candidate must have exactly five comparison Fold rows")
    fold_ids = pd.to_numeric(selected_folds["fold"], errors="raise").astype(int)
    if set(fold_ids.tolist()) != EXPECTED_FOLDS:
        raise ValueError("selected candidate comparison must cover Fold 0..4")
    deltas = pd.to_numeric(selected_folds["f1_delta"], errors="raise").astype(float)
    if (~np.isfinite(deltas)).any():
        raise ValueError("selected candidate Fold F1 deltas must be finite")
    folds_not_worse = int((deltas >= 0.0).sum())
    worst_fold_drop = float(deltas.min())
    recorded_gain = _finite_float(decision["f1_gain"], name="recorded F1 gain")
    recorded_folds = _strict_int(decision["folds_not_worse"], name="recorded folds_not_worse")
    recorded_worst = _finite_float(decision["worst_fold_drop"], name="recorded worst_fold_drop")
    if not _is_close(recorded_gain, expected_gain):
        raise ValueError("selected candidate F1 gain does not match candidate/baseline metrics")
    if recorded_folds != folds_not_worse or not _is_close(recorded_worst, worst_fold_drop):
        raise ValueError("selected candidate Fold promotion statistics drifted")
    expected_promoted = (
        expected_gain >= spec.minimum_f1_gain
        and folds_not_worse >= spec.minimum_folds_not_worse
        and worst_fold_drop >= -spec.maximum_single_fold_drop
    )
    if not expected_promoted:
        raise ValueError("selected candidate does not independently pass all promotion gates")

    selected_f1 = float(cross_fitted["f1"])
    selected_recall = float(cross_fitted["recall"])
    if selected_f1 < spec.reference_f1 and not _is_close(selected_f1, spec.reference_f1):
        raise ValueError("clean selected F1 is below the frozen reference F1")
    if _is_close(selected_f1, spec.reference_f1) and (
        selected_recall < spec.reference_recall
        and not _is_close(selected_recall, spec.reference_recall)
    ):
        raise ValueError("clean selected Recall is below the frozen reference Recall at tied F1")

    curve = pd.read_csv(spec.comparison_dir / "threshold_curves.csv")
    profiles = build_threshold_profiles(
        curve,
        candidate=spec.candidate,
        classifier_weight=spec.classifier_weight,
        stable_threshold=stable_threshold,
        f1_tolerance=spec.f1_tolerance,
    )
    promotion = {
        "baseline": spec.baseline,
        "f1_gain": expected_gain,
        "folds_not_worse": folds_not_worse,
        "worst_fold_drop": worst_fold_drop,
        "gates": gate_values,
        "reference": {"f1": spec.reference_f1, "recall": spec.reference_recall},
        "promoted": True,
    }
    return cross_fitted, promotion, stable_threshold, profiles


def _verify_metric_values(
    recorded: Mapping[str, Any],
    calculated: Mapping[str, float | int],
    *,
    name: str,
) -> None:
    for field, expected in calculated.items():
        actual = recorded.get(field)
        if isinstance(expected, int):
            if _strict_int(actual, name=f"{name}.{field}") != expected:
                raise ValueError(f"{name}.{field} differs from aligned predictions")
        elif not _is_close(_finite_float(actual, name=f"{name}.{field}"), float(expected)):
            raise ValueError(f"{name}.{field} differs from aligned predictions")


def _verify_selected_baseline_and_promotion(
    metrics: Mapping[str, Any],
    selected: pd.DataFrame,
    *,
    cross_fitted: Mapping[str, float | int],
    promotion: Mapping[str, object],
    spec: FinalSelectionSpec,
) -> None:
    """Recompute baseline and promotion evidence from the aligned selected rows."""

    baseline_predictions = (
        selected["baseline_score"].astype(float) >= selected["baseline_threshold"].astype(float)
    ).astype(int)
    if not baseline_predictions.equals(selected["baseline_y_pred"].astype(int)):
        raise ValueError("baseline predictions drifted from aligned score/threshold values")

    calculated_candidate = _metric_counts(selected)
    _verify_metric_values(
        cross_fitted,
        calculated_candidate,
        name="selected cross-fitted metrics",
    )
    calculated_baseline = _metric_counts(selected, prediction_column="baseline_y_pred")
    baseline = _mapping(metrics.get("baseline"), name="comparison metrics.baseline")
    baseline_cross = _mapping(baseline.get("cross_fitted"), name="comparison baseline.cross_fitted")
    _verify_metric_values(
        baseline_cross,
        calculated_baseline,
        name="baseline cross-fitted metrics",
    )

    fold_metrics = pd.read_csv(spec.comparison_dir / "comparison_fold_metrics.csv")
    selected_folds = fold_metrics.loc[fold_metrics["comparison_candidate"] == spec.candidate].copy()
    fold_ids = pd.to_numeric(selected_folds["fold"], errors="raise").astype(int)
    if len(selected_folds) != 5 or set(fold_ids.tolist()) != EXPECTED_FOLDS:
        raise ValueError("selected candidate comparison must contain one row for each Fold 0..4")
    selected_folds["fold"] = fold_ids

    calculated_deltas: list[float] = []
    for fold in sorted(EXPECTED_FOLDS):
        fold_rows = selected.loc[selected["fold"] == fold]
        if fold_rows.empty:
            raise ValueError(f"selected aligned predictions omit Fold {fold}")
        baseline_fold = _metric_counts(fold_rows, prediction_column="baseline_y_pred")
        candidate_fold = _metric_counts(fold_rows)
        comparison_row = selected_folds.loc[selected_folds["fold"] == fold]
        if len(comparison_row) != 1:
            raise ValueError(f"selected comparison Fold {fold} must appear exactly once")
        row = comparison_row.iloc[0]
        baseline_f1 = float(baseline_fold["f1"])
        candidate_f1 = float(candidate_fold["f1"])
        delta = candidate_f1 - baseline_f1
        if not _is_close(
            _finite_float(row.get("baseline_f1"), name=f"Fold {fold} baseline_f1"),
            baseline_f1,
        ):
            raise ValueError(f"Fold {fold} baseline_f1 differs from aligned predictions")
        if not _is_close(
            _finite_float(row.get("candidate_f1"), name=f"Fold {fold} candidate_f1"),
            candidate_f1,
        ):
            raise ValueError(f"Fold {fold} candidate_f1 differs from aligned predictions")
        if not _is_close(
            _finite_float(row.get("f1_delta"), name=f"Fold {fold} f1_delta"),
            delta,
        ):
            raise ValueError(f"Fold {fold} f1_delta differs from aligned predictions")
        calculated_deltas.append(delta)

    expected_gain = float(calculated_candidate["f1"]) - float(calculated_baseline["f1"])
    folds_not_worse = sum(delta >= 0.0 for delta in calculated_deltas)
    worst_fold_drop = min(calculated_deltas)
    if not _is_close(
        _finite_float(promotion.get("f1_gain"), name="promotion f1_gain"),
        expected_gain,
    ):
        raise ValueError("promotion f1_gain differs from aligned predictions")
    if (
        _strict_int(promotion.get("folds_not_worse"), name="promotion folds_not_worse")
        != folds_not_worse
    ):
        raise ValueError("promotion folds_not_worse differs from aligned predictions")
    if not _is_close(
        _finite_float(promotion.get("worst_fold_drop"), name="promotion worst_fold_drop"),
        worst_fold_drop,
    ):
        raise ValueError("promotion worst_fold_drop differs from aligned predictions")

    if not (
        expected_gain >= spec.minimum_f1_gain
        and folds_not_worse >= spec.minimum_folds_not_worse
        and worst_fold_drop >= -spec.maximum_single_fold_drop
    ):
        raise ValueError("aligned predictions do not independently pass all promotion gates")


def _verify_clean_miner_coverage(classifier_models: list[dict[str, Any]]) -> None:
    pairs: set[tuple[int, int]] = set()
    run_paths: set[str] = set()
    weight_paths: set[str] = set()
    for model in classifier_models:
        replay = model.get("hard_negative_replay")
        if not isinstance(replay, Mapping):
            raise ValueError("every selected HN2 classifier Fold must bind hard-negative replay")
        miners = replay.get("miners")
        if not isinstance(miners, list) or len(miners) != 4:
            raise ValueError("every selected HN2 classifier Fold must bind four miners")
        for raw_miner in miners:
            miner = _mapping(raw_miner, name="final selection miner")
            pair = (
                _strict_int(miner.get("outer_fold"), name="miner outer_fold"),
                _strict_int(miner.get("inner_fold"), name="miner inner_fold"),
            )
            run = _mapping(miner.get("run_manifest"), name="miner run_manifest")
            weights = _mapping(miner.get("weights"), name="miner weights")
            run_path = str(run.get("path"))
            weights_path = str(weights.get("path"))
            if pair in pairs or run_path in run_paths or weights_path in weight_paths:
                raise ValueError(
                    "final selection miners must have 20 unique Fold/run/weight records"
                )
            pairs.add(pair)
            run_paths.add(run_path)
            weight_paths.add(weights_path)
    expected_pairs = {
        (outer, inner) for outer in EXPECTED_FOLDS for inner in EXPECTED_FOLDS if inner != outer
    }
    if pairs != expected_pairs:
        raise ValueError("final selection does not cover the complete 5x4 clean miner campaign")


def _selected_fold_parameters(spec: FinalSelectionSpec) -> pd.DataFrame:
    """Load the five held-out Fold parameters selected without each held-out Fold."""

    fold_metrics = pd.read_csv(spec.comparison_dir / "fold_metrics.csv")
    required = {"candidate", "method", "fold", "classifier_weight", "threshold"}
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        raise ValueError(f"fold_metrics.csv omits required columns: {missing}")
    selected = fold_metrics.loc[fold_metrics["candidate"] == spec.candidate].copy()
    if len(selected) != 5:
        raise ValueError("selected candidate must have exactly five fold_metrics rows")
    if not (selected["method"] == "weighted_average").all():
        raise ValueError("selected candidate Fold method must be weighted_average")

    raw_folds = pd.to_numeric(selected["fold"], errors="raise").astype(float)
    if (~np.isfinite(raw_folds)).any() or (raw_folds % 1.0 != 0.0).any():
        raise ValueError("selected candidate fold_metrics Fold IDs must be finite integers")
    selected["fold"] = raw_folds.astype(int)
    if selected["fold"].duplicated().any() or set(selected["fold"].tolist()) != EXPECTED_FOLDS:
        raise ValueError("selected candidate fold_metrics must contain Fold 0..4 exactly once")

    for column in ("classifier_weight", "threshold"):
        values = pd.to_numeric(selected[column], errors="raise").astype(float)
        if (~np.isfinite(values)).any() or ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(
                f"selected candidate Fold {column} values must be finite and in [0, 1]"
            )
        selected[column] = values
    return selected.set_index("fold").sort_index()


def _verify_pooled_threshold_curve(
    components: pd.DataFrame,
    *,
    spec: FinalSelectionSpec,
) -> None:
    """Recompute every pooled final-weight curve point from raw component OOF scores."""

    detector_scores = components[spec.detector_score_column].to_numpy(dtype=float)
    classifier_scores = components[spec.classifier_score_column].to_numpy(dtype=float)
    pooled_scores = (
        1.0 - spec.classifier_weight
    ) * detector_scores + spec.classifier_weight * classifier_scores
    if (~np.isfinite(pooled_scores)).any() or ((pooled_scores < 0.0) | (pooled_scores > 1.0)).any():
        raise ValueError("pooled final-weight component scores must be finite and in [0, 1]")

    curve = pd.read_csv(spec.comparison_dir / "threshold_curves.csv")
    required = {
        "candidate",
        "classifier_weight",
        "threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    }
    missing = sorted(required.difference(curve.columns))
    if missing:
        raise ValueError(f"threshold curve omits required columns: {missing}")
    selected = curve.loc[curve["candidate"] == spec.candidate].copy()
    if selected.empty:
        raise ValueError(f"threshold curve has no rows for {spec.candidate!r}")

    labels = components["label_detector"].astype(int).tolist()
    for index, row in selected.iterrows():
        weight = _probability(
            _finite_float(row["classifier_weight"], name="pooled curve classifier weight"),
            name="pooled curve classifier weight",
        )
        if not _is_close(weight, spec.classifier_weight):
            raise ValueError("pooled threshold curve classifier weight does not match final fusion")
        threshold = _probability(
            _finite_float(row["threshold"], name="pooled curve threshold"),
            name="pooled curve threshold",
        )
        calculated = asdict(compute_binary_metrics(labels, pooled_scores.tolist(), threshold))
        _verify_metric_values(
            row.to_dict(),
            calculated,
            name=f"pooled threshold curve row {index}",
        )


def _verify_selected_formula(
    selected: pd.DataFrame,
    detector: pd.DataFrame,
    classifier: pd.DataFrame,
    *,
    spec: FinalSelectionSpec,
) -> None:
    components = detector[["filename", "label", "fold", spec.detector_score_column]].merge(
        classifier[["filename", "label", "fold", spec.classifier_score_column]],
        on="filename",
        how="inner",
        suffixes=("_detector", "_classifier"),
        validate="one_to_one",
    )
    if len(components) != spec.expected_count:
        raise ValueError("selected detector/classifier OOF filename sets differ")
    if not (
        components["label_detector"].equals(components["label_classifier"])
        and components["fold_detector"].equals(components["fold_classifier"])
    ):
        raise ValueError("selected detector/classifier OOF label or Fold provenance differs")
    components = components.sort_values("filename", kind="stable").reset_index(drop=True)
    ordered = selected.sort_values("filename", kind="stable").reset_index(drop=True)
    if components["filename"].tolist() != ordered["filename"].tolist():
        raise ValueError("selected cross-fitted filenames differ from component OOF")
    if not (
        ordered["label"].astype(int).equals(components["label_detector"].astype(int))
        and ordered["fold"].astype(int).equals(components["fold_detector"].astype(int))
    ):
        raise ValueError("selected cross-fitted label or Fold differs from component OOF")

    fold_parameters = _selected_fold_parameters(spec)
    classifier_weights = ordered["fold"].map(fold_parameters["classifier_weight"])
    thresholds = ordered["fold"].map(fold_parameters["threshold"])
    if classifier_weights.isna().any() or thresholds.isna().any():
        raise ValueError("selected cross-fitted rows reference an unknown Fold parameter")
    detector_scores = components[spec.detector_score_column].to_numpy(dtype=float)
    classifier_scores = components[spec.classifier_score_column].to_numpy(dtype=float)
    weight_values = classifier_weights.to_numpy(dtype=float)
    expected_score = (1.0 - weight_values) * detector_scores + weight_values * classifier_scores
    if not np.allclose(
        ordered["score"].to_numpy(dtype=float),
        expected_score,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "selected cross-fitted score does not equal the per-Fold LOFO weighted average"
        )
    if not np.allclose(
        ordered["threshold"].to_numpy(dtype=float),
        thresholds.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "selected cross-fitted threshold does not match the per-Fold LOFO threshold"
        )
    _verify_pooled_threshold_curve(components, spec=spec)


def _evaluate_final_selection(
    spec: FinalSelectionSpec,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics_path = spec.comparison_dir / "metrics.json"
    metrics = _load_json(metrics_path, name="comparison metrics")
    comparison_outputs = _verify_comparison_outputs(metrics, spec=spec)
    inputs = _mapping(metrics.get("inputs"), name="comparison metrics.inputs")
    input_records: dict[str, dict[str, str]] = {}
    input_paths: dict[str, Path] = {}
    for key in ("config", "folds", "processed_manifest", "coco_json"):
        path, record = _verified_file_record(
            inputs.get(key), root=spec.project_root, name=f"comparison input {key}"
        )
        input_paths[key] = path
        input_records[key] = record
    folds_path = input_paths["folds"]
    processed_path = input_paths["processed_manifest"]
    cross_fitted, promotion, _stable_threshold, profiles = _verify_promotion_and_metrics(
        metrics, spec=spec
    )

    detector_frame, detector_manifest, detector_oof, detector_oof_manifest, detector_config = (
        _validated_component_source(
            metrics,
            spec=spec,
            kind="detector",
            source_name=spec.detector_source,
            score_column=spec.detector_score_column,
            folds_path=folds_path,
            processed_manifest_path=processed_path,
        )
    )
    (
        classifier_frame,
        classifier_manifest,
        classifier_oof,
        classifier_oof_manifest,
        (classifier_config),
    ) = _validated_component_source(
        metrics,
        spec=spec,
        kind="classifier",
        source_name=spec.classifier_source,
        score_column=spec.classifier_score_column,
        folds_path=folds_path,
        processed_manifest_path=processed_path,
    )
    tta_contract = _selected_tta_contract(
        detector_manifest,
        classifier_manifest,
        spec=spec,
    )
    detector_section = _mapping(
        detector_manifest.get("detector"), name="detector OOF manifest detector"
    )
    inference_path, inference_record = _verified_file_record(
        detector_section.get("inference_config"),
        root=spec.project_root,
        name="selected detector inference config",
    )

    detector_models = _model_records(
        detector_frame,
        spec=spec,
        task="detector",
        training_config=detector_config,
        folds_record=input_records["folds"],
        processed_record=input_records["processed_manifest"],
    )
    classifier_models = _model_records(
        classifier_frame,
        spec=spec,
        task="classifier",
        training_config=classifier_config,
        folds_record=input_records["folds"],
        processed_record=input_records["processed_manifest"],
    )
    _verify_clean_miner_coverage(classifier_models)

    aligned = pd.read_csv(spec.comparison_dir / "aligned_predictions.csv")
    selected = _selected_cross_fitted(
        aligned, candidate=spec.candidate, expected_count=spec.expected_count
    )
    _verify_selected_formula(selected, detector_frame, classifier_frame, spec=spec)
    calculated = _metric_counts(selected)
    for field, expected in cross_fitted.items():
        actual = calculated[field]
        if isinstance(expected, int):
            if actual != expected:
                raise ValueError(f"selected cross-fitted {field} differs from candidate metrics")
        elif not _is_close(float(actual), float(expected)):
            raise ValueError(f"selected cross-fitted {field} differs from candidate metrics")
    _verify_selected_baseline_and_promotion(
        metrics,
        selected,
        cross_fitted=cross_fitted,
        promotion=promotion,
        spec=spec,
    )

    selection_config = {
        "path": _portable_path(spec.config_path, spec.project_root),
        "sha256": sha256_file(spec.config_path),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "final_selection",
        "status": "complete",
        "name": spec.name,
        "selection": {"candidate": spec.candidate, "baseline": spec.baseline},
        "artifacts": {
            "inputs": {
                "selection_config": selection_config,
                "comparison_metrics": {
                    "path": _portable_path(metrics_path, spec.project_root),
                    "sha256": sha256_file(metrics_path),
                },
                "comparison_outputs": comparison_outputs,
                "comparison_config": input_records["config"],
                "folds": input_records["folds"],
                "processed_manifest": input_records["processed_manifest"],
                "coco_json": input_records["coco_json"],
            },
            "outputs": {},
        },
        "components": {
            "detector": {
                "source": spec.detector_source,
                "score_column": spec.detector_score_column,
                "training_config": detector_config,
                "inference_config": {
                    "path": _portable_path(inference_path, spec.project_root),
                    **inference_record,
                },
                "oof": {**detector_oof, "rows": spec.expected_count},
                "oof_manifest": detector_oof_manifest,
                "models": detector_models,
            },
            "classifier": {
                "source": spec.classifier_source,
                "score_column": spec.classifier_score_column,
                "training_config": classifier_config,
                "oof": {**classifier_oof, "rows": spec.expected_count},
                "oof_manifest": classifier_oof_manifest,
                "models": classifier_models,
            },
        },
        "fusion": {
            "method": "weighted_average",
            "detector_weight": 1.0 - spec.classifier_weight,
            "classifier_weight": spec.classifier_weight,
            "detector_score_column": spec.detector_score_column,
            "classifier_score_column": spec.classifier_score_column,
        },
        "metrics": {
            "evaluation_contract": "cross_fitted_leave_one_fold_out",
            "cross_fitted": cross_fitted,
            "promotion": promotion,
        },
        "threshold_profiles": profiles,
    }
    if tta_contract is not None:
        payload["tta"] = tta_contract
    return payload, selected


def _with_selected_output(
    payload: dict[str, Any],
    *,
    selected_path: Path,
    published_path: Path,
    spec: FinalSelectionSpec,
    rows: int,
) -> dict[str, Any]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("final selection artifacts must be a JSON object")
    outputs = artifacts.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("final selection artifacts.outputs must be a JSON object")
    outputs["selected_cross_fitted"] = {
        "path": _portable_path(published_path, spec.project_root),
        "sha256": sha256_file(selected_path),
        "rows": rows,
    }
    return payload


def freeze_final_selection(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Validate and atomically freeze the clean final candidate without overwriting output."""

    spec = load_final_selection_spec(config_path, project_root=project_root)
    if spec.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite final selection output: {spec.output_dir}")
    spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = spec.output_dir.parent / f".{spec.output_dir.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        payload, selected = _evaluate_final_selection(spec)
        selected_staging = staging / "selected_cross_fitted.csv"
        atomic_write_csv(selected_staging, selected)
        payload = _with_selected_output(
            payload,
            selected_path=selected_staging,
            published_path=spec.output_dir / "selected_cross_fitted.csv",
            spec=spec,
            rows=len(selected),
        )
        atomic_write_json(staging / "final_selection_manifest.json", payload)
        staging.replace(spec.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest, _manifest_hash = validate_final_selection_manifest(
        spec.output_dir / "final_selection_manifest.json",
        project_root=spec.project_root,
    )
    return manifest


def _validate_selected_frame(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    if actual.columns.tolist() != expected.columns.tolist() or len(actual) != len(expected):
        raise ValueError("published selected_cross_fitted CSV schema/row count drifted")
    string_columns = [
        "filename",
        "confusion",
        "baseline_confusion",
        "transition",
    ]
    integer_columns = ["label", "fold", "y_pred", "baseline_y_pred"]
    float_columns = [
        "score",
        "threshold",
        "margin",
        "baseline_score",
        "baseline_threshold",
    ]
    for column in string_columns:
        if actual[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ValueError(f"published selected_cross_fitted {column} drifted")
    for column in integer_columns:
        if actual[column].astype(int).tolist() != expected[column].astype(int).tolist():
            raise ValueError(f"published selected_cross_fitted {column} drifted")
    for column in float_columns:
        if not np.allclose(
            pd.to_numeric(actual[column], errors="raise").to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"published selected_cross_fitted {column} drifted")
    for column in ("contains_small_fire", "all_boxes_small"):
        normalized: list[bool] = []
        for value in actual[column].tolist():
            if isinstance(value, (bool, np.bool_)):
                normalized.append(bool(value))
            elif isinstance(value, str) and value in {"True", "False"}:
                normalized.append(value == "True")
            else:
                raise ValueError(f"published selected_cross_fitted {column} is not boolean")
        if normalized != expected[column].astype(bool).tolist():
            raise ValueError(f"published selected_cross_fitted {column} drifted")


def validate_final_selection_manifest(
    path: Path,
    *,
    project_root: Path,
) -> tuple[dict[str, Any], str]:
    """Recompute and validate a final-selection manifest; return payload and SHA-256."""

    root = project_root.resolve()
    manifest_path = path if path.is_absolute() else root / path
    manifest = _load_json(manifest_path, name="final selection manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "final_selection"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("final selection manifest schema/kind/status mismatch")
    artifacts = _mapping(manifest.get("artifacts"), name="final selection artifacts")
    inputs = _mapping(artifacts.get("inputs"), name="final selection artifacts.inputs")
    config_path, config_record = _verified_file_record(
        inputs.get("selection_config"), root=root, name="final selection config"
    )
    spec = load_final_selection_spec(config_path, project_root=root)
    expected_manifest_path = spec.output_dir / "final_selection_manifest.json"
    if manifest_path.resolve() != expected_manifest_path.resolve():
        raise ValueError("final selection manifest path does not match selection configuration")
    if config_record["sha256"] != sha256_file(spec.config_path):
        raise ValueError("final selection configuration changed during validation")

    expected_payload, expected_selected = _evaluate_final_selection(spec)
    outputs = _mapping(artifacts.get("outputs"), name="final selection artifacts.outputs")
    selected_path, selected_record = _verified_file_record(
        outputs.get("selected_cross_fitted"),
        root=root,
        name="final selection selected_cross_fitted",
        expected_path=spec.output_dir / "selected_cross_fitted.csv",
    )
    selected_output = _mapping(
        outputs.get("selected_cross_fitted"), name="final selection selected_cross_fitted"
    )
    rows = _strict_int(selected_output.get("rows"), name="selected_cross_fitted.rows", minimum=1)
    if rows != spec.expected_count or _csv_row_count(selected_path) != rows:
        raise ValueError("final selection selected_cross_fitted row count mismatch")
    try:
        actual_selected = pd.read_csv(selected_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"cannot read selected_cross_fitted CSV {selected_path}: {exc}") from exc
    _validate_selected_frame(actual_selected, expected_selected)
    expected_payload = _with_selected_output(
        expected_payload,
        selected_path=selected_path,
        published_path=selected_path,
        spec=spec,
        rows=rows,
    )
    if (
        selected_record["sha256"]
        != expected_payload["artifacts"]["outputs"]["selected_cross_fitted"]["sha256"]
    ):
        raise ValueError("selected_cross_fitted hash changed during validation")
    if manifest != expected_payload:
        raise ValueError("final selection manifest fields differ from recomputed provenance")
    return manifest, sha256_file(manifest_path)
