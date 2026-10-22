"""Direction A orchestration: plan, train, calibrate, then evaluate untouched outer rows."""

from __future__ import annotations

import gc
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

from fire_detection.ensemble.evaluation import _select_weight_and_threshold
from fire_detection.inference.slicing import generate_tile_windows
from fire_detection.inference.tta import arithmetic_mean_score, reduce_detector_view
from fire_detection.metrics.binary import compute_binary_metrics
from fire_detection.metrics.thresholds import search_thresholds
from fire_detection.models.features import (
    extract_detection_features,
    extract_ultralytics_classification_result,
    tensor_to_list,
)
from fire_detection.models.offline import sha256_file
from fire_detection.models.training import write_json_atomic
from fire_detection.research.protocol import (
    ARMS,
    build_partition,
    configured_evaluation_methods,
    configured_model_names,
    digest_object,
    is_scheme_b,
    project_path,
    publish_contract,
    publish_hash_sidecar,
    scheme_b_contract,
    validate_formal_training_authorization,
    validate_hash_sidecar,
    verify_source,
)
from fire_detection.research.training import (
    configure_runtime,
    resolve_complete_training_attempt,
    train_model,
    training_contract,
)

EVALUATION_SCHEMA_VERSION = 2
CAMPAIGN_CONTRACT_SCHEMA_VERSION = 2
FROZEN_CAMPAIGN_IDENTITIES = frozenset(
    {
        "main_v1",
        "main_v2",
        "main_v3",
        "main_v4",
        "main_v5",
        "smoke_v1",
        "smoke_v2",
        "smoke_v3",
        "smoke_v4",
        "smoke_v5",
        "smoke_v6",
        "smoke_v7",
        "smoke_v8",
    }
)


class IncompleteEvaluation(ValueError):
    """Raised when a cell has no validated evaluation completion pointer."""

# The full evaluation matrix is part of the Direction A contract.  Keeping it
# in one place prevents a completed manifest from being accepted merely
# because it claims that a subset of methods ran.
EXPECTED_EVALUATION_METHODS = (
    "cls_hn",
    "cls_random_negative",
    "cls_uniform_negative",
    "cls_uniform_all",
    "cls_hn_tta",
    "det_full",
    "det_no_slice",
    "det_no_tta",
    "fusion_hn_full",
    "fusion_random_negative_full",
    "fusion_uniform_negative_full",
    "fusion_uniform_all_full",
    "fusion_hn_no_slice",
    "fusion_hn_no_tta",
)
_EVALUATION_REQUIRED_COLUMNS = {
    "experiment_id",
    "method",
    "outer_fold",
    "seed",
    "filename",
    "label",
    "group_id",
    "score",
    "threshold",
    "classifier_weight",
    "y_pred",
    "split",
    "condition",
    "evidence_status",
    "calibration_fold",
}


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    """Read one strict object and turn malformed recovery state into a hard failure."""
    if not path.is_file():
        raise ValueError(f"missing {context}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {context}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object: {path}")
    return value


def _immutable_json(path: Path, payload: Mapping[str, Any], *, context: str) -> None:
    """Publish a JSON artifact once, rejecting byte/content drift on recovery."""
    if path.exists():
        try:
            existing = _read_json(path, context=context)
        except ValueError:
            raise
        if existing != dict(payload):
            raise ValueError(f"refusing to overwrite changed {context}: {path}")
        return
    write_json_atomic(path, payload)


def _campaign_snapshot_inputs(
    config: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, tuple[Path, str]]:
    """Map every source/dependency artifact into the new campaign snapshot."""
    root = Path(config["_root"])
    inputs: dict[str, tuple[Path, str]] = {
        relative: (root / "src" / "fire_detection" / relative, digest)
        for relative, digest in source["implementation_sha256"].items()
    }
    inputs.update(
        {
            f"entrypoints/{relative}": (root / relative, digest)
            for relative, digest in source["dependency_snapshot"]["entrypoints_sha256"].items()
        }
    )
    inputs.update(
        {
            f"dependencies/{relative}": (root / relative, digest)
            for relative, digest in source["dependency_snapshot"]["files_sha256"].items()
        }
    )
    inputs["config.yaml"] = (Path(config["_config_path"]), config["_config_sha256"])
    return inputs


def _expected_partition_records(
    records: Sequence[Any],
    contract: Mapping[str, Any],
    *,
    outer_folds: Sequence[int],
    seeds: Sequence[int],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Build immutable partition expectations before writing any campaign files."""
    source_digest = digest_object(contract)
    expected: dict[tuple[int, int], dict[str, Any]] = {}
    for outer in outer_folds:
        for seed in seeds:
            partition = build_partition(records, outer, seed)
            partition["source_contract_sha256"] = source_digest
            partition["counts"] = {key: len(rows) for key, rows in partition["splits"].items()}
            if sum(partition["counts"].values()) != len(records):
                raise ValueError("fit/cal/test partition does not cover the configured dataset")
            if any(count <= 0 for count in partition["counts"].values()):
                raise ValueError("fit/cal/test partitions must all be nonempty")
            expected[(outer, seed)] = partition
    return expected


def plan_campaign(
    config: Mapping[str, Any], *, outer_folds: Sequence[int], seeds: Sequence[int]
) -> tuple[Path, dict[str, Any]]:
    """Verify input lineage and persist immutable plans before training."""
    if config.get("campaign") in FROZEN_CAMPAIGN_IDENTITIES:
        raise ValueError(
            "frozen legacy campaign identity cannot be used after contract repair; "
            "use a new campaign identity"
        )
    records, source = verify_source(config)
    destination = Path(config["_root"]) / "runs" / "direction_a" / config["campaign"]
    contract = {
        "schema_version": CAMPAIGN_CONTRACT_SCHEMA_VERSION,
        "kind": "internal_fit_cal_test_direction_a",
        "campaign_identity": config["campaign"],
        "config_sha256": config["_config_sha256"],
        "source": source,
        "engineering_only": config["engineering_only"],
        "external_samples": 0,
        "expected_outer_folds": config["outer_folds"],
        "expected_seeds": config["seeds"],
        "arms": config["arms"],
        "training_source": config["data"].get("training_source"),
        "original_training_data_allowed": config["data"].get(
            "original_training_data_allowed"
        ),
        "historical_data_reuse": bool(
            config["data"].get("original_training_data_allowed", True)
        ),
        "external_evaluation_samples": int(
            config["data"].get("external_evaluation_count", 0)
        ),
    }
    snapshot_inputs = _campaign_snapshot_inputs(config, source)
    expected_partitions = _expected_partition_records(
        records, contract, outer_folds=outer_folds, seeds=seeds
    )

    # A legacy campaign is deliberately not upgraded in place.  In particular,
    # editing the old JSON hash cannot make it eligible: the old snapshot lacks
    # the dependency/entry-point closure and is checked before any write.
    if destination.exists():
        existing_contract = _read_json(
            destination / "campaign_contract.json", context="campaign contract"
        )
        if (
            existing_contract.get("schema_version") != CAMPAIGN_CONTRACT_SCHEMA_VERSION
            or existing_contract.get("campaign_identity") != config["campaign"]
            or "source" not in existing_contract
            or "dependency_snapshot" not in existing_contract.get("source", {})
        ):
            raise ValueError(
                "legacy or incomplete campaign contract cannot be resumed; "
                "use a new campaign identity"
            )
        if existing_contract != contract:
            raise ValueError(
                "existing campaign contract differs; do not edit its old hash; "
                "use a new campaign identity"
            )
        snapshot = destination / "source_snapshot"
        validate_hash_sidecar(destination / "campaign_contract.json")
        for relative, (_, expected_digest) in snapshot_inputs.items():
            target = snapshot / relative
            if not target.is_file() or sha256_file(target) != expected_digest:
                raise ValueError(f"source/dependency snapshot is incomplete or drifted: {target}")
        for (outer, seed), partition in expected_partitions.items():
            path = destination / f"outer_{outer}" / f"seed_{seed}" / "partition.json"
            if _read_json(path, context="partition contract") != partition:
                raise ValueError(f"partition contract is incomplete or drifted: {path}")
            validate_hash_sidecar(path)
        return destination, contract

    publish_contract(destination / "campaign_contract.json", contract)
    publish_hash_sidecar(destination / "campaign_contract.json")
    snapshot = destination / "source_snapshot"
    for relative, (source_path, expected_digest) in snapshot_inputs.items():
        target = snapshot / relative
        if not source_path.is_file() or sha256_file(source_path) != expected_digest:
            raise ValueError(f"source/dependency artifact drifted before snapshot: {source_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())
        if sha256_file(target) != expected_digest:
            raise ValueError(f"source snapshot hash drift: {target}")
    for (outer, seed), partition in expected_partitions.items():
        publish_contract(
            destination / f"outer_{outer}" / f"seed_{seed}" / "partition.json", partition
        )
        publish_hash_sidecar(
            destination / f"outer_{outer}" / f"seed_{seed}" / "partition.json"
        )
    return destination, contract


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    # Existing contract artifacts are immutable.  A byte-identical rerun is
    # safe and idempotent; any changed output indicates an incompatible
    # recovery or protocol drift and must fail closed.
    if path.exists():
        if temporary.read_bytes() != path.read_bytes():
            temporary.unlink(missing_ok=True)
            raise ValueError(f"refusing to overwrite changed contract artifact: {path}")
        temporary.unlink(missing_ok=True)
        return
    temporary.replace(path)


def _write_immutable_bytes(path: Path, data: bytes, *, context: str) -> None:
    """Write a byte artifact once; identical recovery is a no-op."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing to overwrite changed {context}: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_rejection_ledger(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Persist detector geometry rejects with provenance and return its hash."""
    required = {
        "filename",
        "condition",
        "view",
        "input",
        "tile",
        "tile_index",
        "tile_window",
        "coordinate_space",
        "box_index",
        "xyxy",
        "confidence",
        "reason",
        "would_change_max_valid_score",
        "config_sha256",
        "checkpoint_sha256",
        "partition_sha256",
    }
    encoded_rows: list[bytes] = []
    for row in rows:
        if not isinstance(row, Mapping) or not required <= set(row):
            raise ValueError("detector rejection ledger row lacks audit provenance")
        encoded_rows.append(
            (
                json.dumps(dict(row), ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            ).encode("utf-8")
        )
    data = b"".join(encoded_rows)
    _write_immutable_bytes(path, data, context="detector rejection ledger")
    return sha256_file(path)


def _training_artifacts(
    config: Mapping[str, Any],
    cell: Path,
    partition: Mapping[str, Any],
    model_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Validate all model manifests before any evaluation-side write."""
    partition_hash = digest_object(partition)
    result: dict[str, dict[str, Any]] = {}
    for name in model_names:
        task_dir = cell / "models" / name
        run, manifest = resolve_complete_training_attempt(
            task_dir, context=f"{name} training run"
        )
        training_contract = manifest["contract"]
        if training_contract.get("source_contract_sha256") != partition_hash:
            raise ValueError(f"unverified training partition: {name}")
        if training_contract.get("task") not in {"classifier", "detector"}:
            raise ValueError(f"invalid training task in {name}")
        result[name] = {
            "manifest": manifest,
            "run_dir": run,
            "manifest_sha256": sha256_file(run / "research_manifest.json"),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "contract_sha256": manifest["contract_sha256"],
        }
    return result


def _evaluation_contract(
    config: Mapping[str, Any],
    cell: Path,
    partition: Mapping[str, Any],
    model_names: Sequence[str],
    training: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the immutable input contract for calibration/test evaluation."""
    campaign_contract_path = cell.parents[1] / "campaign_contract.json"
    validate_hash_sidecar(campaign_contract_path)
    campaign_contract = _read_json(campaign_contract_path, context="campaign contract")
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "campaign_identity": config["campaign"],
        "engineering_only": bool(config["engineering_only"]),
        "campaign_contract_sha256": sha256_file(campaign_contract_path),
        "campaign_contract_digest": digest_object(campaign_contract),
        "config_sha256": config["_config_sha256"],
        "partition_sha256": digest_object(partition),
        "model_names": list(model_names),
        "method_names": list(configured_evaluation_methods(config)),
        "models": {
            name: {
                "manifest_sha256": training[name]["manifest_sha256"],
                "contract_sha256": training[name]["contract_sha256"],
                "checkpoint_sha256": training[name]["checkpoint_sha256"],
            }
            for name in model_names
        },
        "inference": dict(config["inference"]),
        "evaluation": dict(config["evaluation"]),
        "split_roles": {"cal": "calibration_only", "test": "untouched_outer_test"},
    }


def _expected_raw_sidecars(cell: Path, model_names: Sequence[str]) -> list[Path]:
    return [
        cell / "raw" / f"{name}_{split}.manifest.json"
        for name in model_names
        for split in ("cal", "test")
    ]


def _validate_complete_evaluation(
    cell: Path,
    manifest: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    model_names: Sequence[str],
) -> dict[str, Any]:
    """Validate a completed evaluation without touching its timestamp or bytes."""
    if not isinstance(manifest, Mapping):
        raise ValueError("evaluation manifest must be an object")
    if manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("evaluation manifest schema is not resumable")
    if manifest.get("status") != "cell_complete":
        raise ValueError("evaluation output is incomplete; use a new campaign")
    if manifest.get("contract") != dict(expected_contract):
        raise ValueError("evaluation contract differs; use a new campaign")
    if manifest.get("contract_sha256") != digest_object(expected_contract):
        raise ValueError("evaluation contract hash differs")
    required = (
        "predictions_sha256",
        "metrics_sha256",
        "calibration_sha256",
        "raw_manifests",
        "training_manifests",
        "completed_at_unix",
        "rows",
        "methods",
        "evidence_status",
    )
    if any(key not in manifest for key in required):
        raise ValueError("evaluation manifest is incomplete; use a new campaign")
    completed_at = manifest["completed_at_unix"]
    # bool is an int subclass, but it is not a valid completion timestamp.
    if type(completed_at) not in (int, float) or not math.isfinite(float(completed_at)):
        raise ValueError("evaluation completion timestamp is invalid")
    if float(completed_at) <= 0:
        raise ValueError("evaluation completion timestamp is invalid")
    expected_engineering = expected_contract.get("engineering_only")
    if expected_engineering is not None:
        if type(manifest.get("engineering_only")) is not bool:
            raise ValueError("evaluation evidence status is incomplete")
        if manifest["engineering_only"] != expected_engineering:
            raise ValueError("evaluation evidence status differs")

    # Validate immutable partition metadata before trusting manifest row counts.
    partition_path = cell / "partition.json"
    validate_hash_sidecar(partition_path)
    partition = _read_json(partition_path, context="partition contract")
    splits = partition.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"fit", "cal", "test"}:
        raise ValueError("evaluation partition is incomplete")
    if manifest.get("outer_fold") != partition.get("outer_fold"):
        raise ValueError("evaluation outer fold differs from its partition")
    if manifest.get("seed") != partition.get("seed"):
        raise ValueError("evaluation seed differs from its partition")
    split_rows: dict[str, list[Mapping[str, Any]]] = {}
    for split in ("cal", "test"):
        rows = splits.get(split)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"evaluation {split} partition is incomplete")
        if not all(isinstance(row, Mapping) for row in rows):
            raise ValueError(f"evaluation {split} partition rows are invalid")
        names = [row.get("filename") for row in rows]
        if any(not isinstance(name, str) or not name for name in names) or len(names) != len(
            set(names)
        ):
            raise ValueError(f"evaluation {split} partition identities are invalid")
        split_rows[split] = rows
    declared_methods = expected_contract.get("method_names", EXPECTED_EVALUATION_METHODS)
    if not isinstance(declared_methods, list | tuple) or not declared_methods:
        raise ValueError("evaluation contract method matrix is invalid")
    expected_methods = set(declared_methods)
    if len(expected_methods) != len(declared_methods):
        raise ValueError("evaluation contract method matrix contains duplicates")
    methods = manifest.get("methods")
    if (
        not isinstance(methods, list)
        or len(methods) != len(expected_methods)
        or set(methods) != expected_methods
        or methods != sorted(methods)
    ):
        raise ValueError("evaluation method matrix is incomplete")
    expected_rows = sum(len(rows) for rows in split_rows.values()) * len(expected_methods)
    rows_value = manifest.get("rows")
    if type(rows_value) is not int or rows_value != expected_rows:
        raise ValueError("evaluation row count is incomplete or inconsistent")
    for key, split in (("cal_unique_images", "cal"), ("test_unique_images", "test")):
        value = manifest.get(key)
        if type(value) is not int or value != len(split_rows[split]):
            raise ValueError(f"evaluation {key} is incomplete or inconsistent")
    predictions = cell / "predictions.csv"
    metrics = cell / "metrics.csv"
    calibration = cell / "calibration.json"
    for path, label in (
        (predictions, "predictions"),
        (metrics, "metrics"),
        (calibration, "calibration"),
    ):
        if not path.is_file():
            raise ValueError(f"completed evaluation is missing {label}: {path}")
    if manifest["predictions_sha256"] != sha256_file(predictions):
        raise ValueError("completed evaluation predictions changed")
    if manifest["metrics_sha256"] != sha256_file(metrics):
        raise ValueError("completed evaluation metrics changed")
    if manifest["calibration_sha256"] != sha256_file(calibration):
        raise ValueError("completed evaluation calibration changed")

    try:
        prediction_frame = pd.read_csv(predictions, float_precision="round_trip")
        metrics_frame = pd.read_csv(metrics, float_precision="round_trip")
        calibration_payload = _read_json(calibration, context="calibration contract")
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError("completed evaluation output is unreadable") from exc
    if len(prediction_frame) != expected_rows:
        raise ValueError("completed evaluation predictions row count differs")
    if not _EVALUATION_REQUIRED_COLUMNS <= set(prediction_frame.columns):
        raise ValueError("completed evaluation predictions are missing required columns")
    if prediction_frame[list(_EVALUATION_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("completed evaluation predictions contain missing identities")
    if set(prediction_frame["method"]) != expected_methods or set(
        prediction_frame["split"]
    ) != {"cal", "test"}:
        raise ValueError("completed evaluation method/split matrix differs")
    if prediction_frame.duplicated(["method", "split", "filename"]).any():
        raise ValueError("completed evaluation predictions contain duplicate identities")
    expected_evidence = (
        "engineering_only"
        if bool(expected_contract.get("engineering_only", False))
        else "internal_revalidation"
    )
    if manifest.get("evidence_status") != expected_evidence:
        raise ValueError("evaluation manifest evidence status differs")
    if set(prediction_frame["condition"]) != {"clean"} or set(
        prediction_frame["evidence_status"]
    ) != {expected_evidence}:
        raise ValueError("completed evaluation evidence/condition differs")
    if prediction_frame["experiment_id"].astype(str).str.strip().eq("").any():
        raise ValueError("completed evaluation experiment identities are empty")
    for column in ("score", "threshold", "classifier_weight"):
        numeric = pd.to_numeric(prediction_frame[column], errors="coerce")
        if numeric.isna().any() or not numeric.between(0, 1).all():
            raise ValueError(f"completed evaluation {column} is invalid")
    for column in ("outer_fold", "seed", "label", "y_pred", "calibration_fold"):
        numeric = pd.to_numeric(prediction_frame[column], errors="coerce")
        if numeric.isna().any() or (numeric != numeric.astype(int)).any():
            raise ValueError(f"completed evaluation {column} is invalid")
    if not prediction_frame["outer_fold"].eq(partition["outer_fold"]).all() or not prediction_frame[
        "seed"
    ].eq(partition["seed"]).all():
        raise ValueError("completed evaluation fold/seed identities differ")
    if not prediction_frame["calibration_fold"].eq((partition["outer_fold"] + 1) % 5).all():
        raise ValueError("completed evaluation calibration fold differs")
    if not prediction_frame["label"].isin([0, 1]).all() or not prediction_frame["y_pred"].isin(
        [0, 1]
    ).all():
        raise ValueError("completed evaluation labels or predictions are not binary")
    if not prediction_frame["y_pred"].eq(
        (prediction_frame["score"] >= prediction_frame["threshold"]).astype(int)
    ).all():
        raise ValueError("completed evaluation predictions do not use frozen thresholds")
    for split, rows in split_rows.items():
        expected_identity = sorted(
            (row["filename"], row.get("label"), row.get("group_id")) for row in rows
        )
        for method in sorted(expected_methods):
            actual = prediction_frame.loc[
                (prediction_frame["method"] == method) & (prediction_frame["split"] == split)
            ]
            if len(actual) != len(rows):
                raise ValueError(f"completed evaluation {method}/{split} row count differs")
            actual_identity = sorted(
                zip(actual["filename"], actual["label"], actual["group_id"], strict=True)
            )
            if actual_identity != expected_identity:
                raise ValueError(f"completed evaluation {method}/{split} identities differ")

    metric_required = {"method", "split", "f1", "recall", "fpr", "tp", "fp", "fn", "tn"}
    if not metric_required <= set(metrics_frame.columns):
        raise ValueError("completed evaluation metrics are incomplete")
    expected_metric_keys = {
        (method, split) for method in expected_methods for split in ("cal", "test")
    }
    actual_metric_keys = set(zip(metrics_frame["method"], metrics_frame["split"], strict=False))
    if (
        len(metrics_frame) != len(expected_metric_keys)
        or actual_metric_keys != expected_metric_keys
    ):
        raise ValueError("completed evaluation metrics matrix differs")
    if metrics_frame.duplicated(["method", "split"]).any():
        raise ValueError("completed evaluation metrics contain duplicate identities")
    for column in ("f1", "recall", "fpr"):
        values = pd.to_numeric(metrics_frame[column], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            raise ValueError(f"completed evaluation metric {column} is invalid")
    for column in ("tp", "fp", "fn", "tn"):
        values = pd.to_numeric(metrics_frame[column], errors="coerce")
        if values.isna().any() or (values < 0).any() or (values != values.astype(int)).any():
            raise ValueError(f"completed evaluation metric {column} is invalid")
    for _, metric_row in metrics_frame.iterrows():
        expected_n = len(split_rows[str(metric_row["split"])])
        if int(metric_row[["tp", "fp", "fn", "tn"]].sum()) != expected_n:
            raise ValueError("completed evaluation metric counts differ from partition rows")
    if not isinstance(calibration_payload.get("methods"), Mapping) or set(
        calibration_payload["methods"]
    ) != expected_methods:
        raise ValueError("completed evaluation calibration matrix is incomplete")
    if calibration_payload.get("partition_sha256") != digest_object(partition):
        raise ValueError("completed evaluation calibration partition differs")
    cal_raw_hashes = calibration_payload.get("cal_raw_sha256")
    if not isinstance(cal_raw_hashes, Mapping) or set(cal_raw_hashes) != set(model_names):
        raise ValueError("completed evaluation calibration raw-manifest index is incomplete")
    for name in model_names:
        cal_csv = cell / "raw" / f"{name}_cal.csv"
        if not cal_csv.is_file():
            raise ValueError(f"completed evaluation calibration raw prediction is missing: {name}")
        if cal_raw_hashes[name] != sha256_file(cal_csv):
            raise ValueError(f"completed evaluation calibration raw prediction changed: {name}")
    for method, params in calibration_payload["methods"].items():
        if not isinstance(params, Mapping) or set(params) != {"threshold", "classifier_weight"}:
            raise ValueError(f"completed evaluation calibration is invalid: {method}")
        for key in ("threshold", "classifier_weight"):
            value = params[key]
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"completed evaluation calibration is invalid: {method}/{key}")
        rows_for_method = prediction_frame.loc[prediction_frame["method"] == method]
        if not rows_for_method["threshold"].eq(params["threshold"]).all() or not rows_for_method[
            "classifier_weight"
        ].eq(params["classifier_weight"]).all():
            raise ValueError(
                f"completed evaluation calibration does not match predictions: {method}"
            )

    integrity = cell / "evaluation_manifest.sha256"
    expected_integrity = (sha256_file(cell / "evaluation_manifest.json") + "\n").encode("ascii")
    if not integrity.is_file() or integrity.read_bytes() != expected_integrity:
        raise ValueError("evaluation manifest content hash is missing or changed")
    completion_record_path = cell / "evaluation_completion.json"
    completion_record = _read_json(
        completion_record_path, context="evaluation completion record"
    )
    if completion_record != {
        "manifest_sha256": sha256_file(cell / "evaluation_manifest.json"),
        "completed_at_unix": completed_at,
    }:
        raise ValueError("evaluation completion timestamp or content hash changed")
    expected_sidecars = _expected_raw_sidecars(cell, model_names)
    recorded_sidecars = manifest["raw_manifests"]
    if not isinstance(recorded_sidecars, dict):
        raise ValueError("evaluation raw-manifest index is invalid")
    if set(recorded_sidecars) != {sidecar.name for sidecar in expected_sidecars}:
        raise ValueError("evaluation raw-manifest index is incomplete")
    for sidecar in expected_sidecars:
        if not sidecar.is_file() or sidecar.name not in recorded_sidecars:
            raise ValueError(f"evaluation raw output is incomplete: {sidecar}")
        if recorded_sidecars[sidecar.name] != sha256_file(sidecar):
            raise ValueError(f"evaluation raw manifest changed: {sidecar}")
        raw = _read_json(sidecar, context="raw prediction manifest")
        raw_contract = raw.get("contract")
        if not isinstance(raw_contract, Mapping):
            raise ValueError(f"evaluation raw prediction contract is invalid: {sidecar}")
        csv_path = sidecar.with_name(sidecar.name.replace(".manifest.json", ".csv"))
        if not csv_path.is_file() or raw.get("csv_sha256") != sha256_file(csv_path):
            raise ValueError(f"evaluation raw prediction changed: {csv_path}")
        split = "cal" if sidecar.name.endswith("_cal.manifest.json") else "test"
        name = sidecar.name.rsplit("_", 1)[0]
        expected_task = "detector" if name == "detector" else "classifier"
        expected_model = expected_contract.get("models", {}).get(name)
        if not isinstance(expected_model, Mapping):
            raise ValueError(f"evaluation raw prediction model is unexpected: {sidecar}")
        expected_raw_contract = {
            "config_sha256": expected_contract.get("config_sha256"),
            "checkpoint_sha256": expected_model.get("checkpoint_sha256"),
            "rows_sha256": digest_object(split_rows[split]),
            "task": expected_task,
            "model_name": name,
            "split": split,
            "tta": True,
            "condition": "clean",
            "inference": expected_contract.get("inference"),
            "partition_sha256": expected_contract.get("partition_sha256"),
        }
        if dict(raw_contract) != expected_raw_contract:
            raise ValueError(f"evaluation raw prediction contract differs: {sidecar}")
        if raw.get("row_count") != len(split_rows[split]):
            raise ValueError(f"evaluation raw prediction row count differs: {csv_path}")
        if raw_contract.get("condition") != "clean":
            raise ValueError(f"evaluation raw prediction condition differs: {csv_path}")
        if raw_contract.get("split") != split:
            raise ValueError(f"evaluation raw prediction split differs: {csv_path}")
        raw_frame = pd.read_csv(csv_path, float_precision="round_trip")
        if "filename" not in raw_frame or raw_frame["filename"].tolist() != [
            row["filename"] for row in split_rows[split]
        ]:
            raise ValueError(f"evaluation raw prediction identities differ: {csv_path}")
        ledger = sidecar.with_name(sidecar.name.replace(".manifest.json", ".invalid_boxes.jsonl"))
        if sidecar.name.startswith("detector_"):
            if not ledger.is_file() or raw.get("invalid_boxes_sha256") != sha256_file(ledger):
                raise ValueError(f"evaluation detector rejection ledger changed: {ledger}")
        elif ledger.exists():
            raise ValueError(f"unexpected detector rejection ledger: {ledger}")
    recorded_training = manifest["training_manifests"]
    if not isinstance(recorded_training, dict) or set(recorded_training) != set(model_names):
        raise ValueError("evaluation training-manifest index is incomplete")
    for name in model_names:
        task_dir = cell / "models" / name
        legacy_path = task_dir / "research_manifest.json"
        # Schema-v2 evaluation manifests already seal the exact training
        # manifest bytes.  Keep validating pre-attempt legacy fixtures and
        # historical cells that lack an embedded training contract, while all
        # new attempt-based tasks still require their strict completion pointer.
        if legacy_path.is_file():
            legacy = _read_json(legacy_path, context=f"{name} training manifest")
            if "contract" not in legacy:
                run = task_dir
            else:
                run, _ = resolve_complete_training_attempt(
                    task_dir, context=f"{name} training run"
                )
        else:
            run, _ = resolve_complete_training_attempt(
                task_dir, context=f"{name} training run"
            )
        path = run / "research_manifest.json"
        if recorded_training[name] != sha256_file(path):
            raise ValueError(f"evaluation training manifest changed: {path}")
    return dict(manifest)


def valid_detector_score(result: Any) -> tuple[float, list[dict[str, Any]]]:
    """Reject non-positive-area boxes with a ledger; numerical failure remains fatal."""
    if result.boxes is None:
        return 0.0, []
    coordinates = tensor_to_list(result.boxes.xyxy)
    confidences = tensor_to_list(result.boxes.conf)
    classes = tensor_to_list(result.boxes.cls)
    if not len(coordinates) == len(confidences) == len(classes):
        raise ValueError("detector output column lengths differ")
    accepted_coordinates = []
    accepted_confidences = []
    accepted_classes = []
    rejected = []
    for box_index, (box, confidence, class_id) in enumerate(
        zip(coordinates, confidences, classes, strict=True)
    ):
        if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
            raise ValueError("non-finite or malformed detector coordinates")
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("non-finite or invalid detector confidence")
        if class_id != 0:
            raise ValueError("research detector must emit only the validated fire class 0")
        if box[2] <= box[0] or box[3] <= box[1]:
            rejected.append(
                {
                    "box_index": box_index,
                    "xyxy": box,
                    "confidence": confidence,
                    "reason": "non_positive_area",
                }
            )
        else:
            accepted_coordinates.append(box)
            accepted_confidences.append(float(confidence))
            accepted_classes.append(0)
    features, _ = extract_detection_features(
        xyxy=accepted_coordinates,
        confidences=accepted_confidences,
        class_ids=accepted_classes,
        image_shape=tuple(result.orig_shape),
    )
    # Record whether a rejected detection would have changed the image-level
    # maximum had its invalid geometry been accepted.  The valid score is
    # computed from accepted boxes only; no coordinate repair is performed.
    valid_score = float(features.det_max_conf)
    for item in rejected:
        item["would_change_max_valid_score"] = float(item["confidence"]) > valid_score
    return valid_score, rejected


def predict_model(
    config: Mapping[str, Any],
    cell: Path,
    *,
    task: str,
    model_name: str,
    checkpoint: Path,
    rows: Sequence[Mapping[str, Any]],
    split: str,
    tta: bool = True,
    condition: str = "clean",
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Save original/flip and full/slice scores with checkpoint/source hashes."""
    if condition != "clean" and split != "test":
        raise ValueError("perturbations are test-only; calibration must remain clean")
    suffix = "" if condition == "clean" else f"_{condition}"
    if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_." for c in condition):
        raise ValueError("invalid perturbation condition")
    destination = (output_dir or (cell / "raw")) / f"{model_name}_{split}{suffix}.csv"
    sidecar = destination.with_suffix(".manifest.json")
    invalid_ledger = destination.with_suffix(".invalid_boxes.jsonl")
    validate_hash_sidecar(cell / "partition.json")
    contract = {
        "config_sha256": config["_config_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "rows_sha256": digest_object(rows),
        "task": task,
        "model_name": model_name,
        "split": split,
        "tta": tta,
        "condition": condition,
        "inference": config["inference"],
        # Bind raw predictions to the canonical partition object, matching the
        # evaluation contract rather than the incidental JSON byte encoding.
        "partition_sha256": digest_object(
            _read_json(cell / "partition.json", context="partition contract")
        ),
    }
    if sidecar.exists():
        recorded = _read_json(sidecar, context="prediction manifest")
        if not destination.is_file():
            raise ValueError(f"prediction manifest exists without its CSV: {destination}")
        if recorded.get("contract") != contract or recorded.get("csv_sha256") != sha256_file(
            destination
        ):
            raise ValueError(f"cached prediction contract/hash drift: {destination}")
        if recorded.get("row_count") != len(rows):
            raise ValueError(f"cached prediction row count drift: {destination}")
        if task == "detector":
            if not invalid_ledger.is_file() or recorded.get("invalid_boxes_sha256") != sha256_file(
                invalid_ledger
            ):
                raise ValueError("detector rejected-box ledger hash drift")
        elif invalid_ledger.exists():
            raise ValueError("unexpected detector rejected-box ledger for classifier output")
        frame = pd.read_csv(destination, float_precision="round_trip")
        if frame["filename"].tolist() != [row["filename"] for row in rows]:
            raise ValueError("prediction row identity differs from input list")
        return frame
    if destination.exists() or invalid_ledger.exists():
        raise ValueError(f"partial predictions without valid manifest: {destination}")
    configure_runtime(config)
    import torch
    from ultralytics import YOLO

    model = YOLO(str(checkpoint), verbose=False)
    image_root = project_path(Path(config["_root"]), config["data"]["images"])
    result_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    args = {
        "imgsz": config[task]["imgsz"],
        "device": config["training"]["device"],
        "verbose": False,
        "augment": False,
        "save": False,
    }
    if task == "detector":
        args.update(conf=config["inference"]["confidence"], max_det=config["inference"]["max_det"])
    started = time.time()
    try:
        for offset in range(
            0, len(rows), config["inference"]["batch"] if task == "classifier" else 1
        ):
            chunk = rows[
                offset : offset + (config["inference"]["batch"] if task == "classifier" else 1)
            ]
            images: list[Image.Image] = []
            for row in chunk:
                with Image.open(image_root / row["filename"]) as opened:
                    source_image = ImageOps.exif_transpose(opened).convert("RGB")
                if condition != "clean":
                    from fire_detection.research.perturbations import apply_perturbation

                    seed = config["inference"]["perturbation_seed"]
                    transformed = apply_perturbation(
                        source_image, condition, seed=seed, source_filename=row["filename"]
                    )
                    source_image.close()
                    source_image = transformed
                images.append(source_image)
            if task == "classifier":
                original = model.predict(images, **args)
                flipped_images = [ImageOps.mirror(image) for image in images] if tta else []
                flipped = model.predict(flipped_images, **args) if tta else original
                if len(original) != len(chunk) or len(flipped) != len(chunk):
                    raise ValueError("classifier inference cardinality mismatch")
                for row, first, second in zip(chunk, original, flipped, strict=True):
                    score = extract_ultralytics_classification_result(first)[1]
                    flip_score = extract_ultralytics_classification_result(second)[1]
                    result_rows.append(
                        {
                            **dict(row),
                            "original": score,
                            "flip": flip_score,
                            "tta": arithmetic_mean_score(score, flip_score),
                        }
                    )
                for flipped_image in flipped_images:
                    flipped_image.close()
            else:
                image = images[0]
                view_scores: list[tuple[float, float]] = []
                views = [image, ImageOps.mirror(image)] if tta else [image]
                for view_index, view in enumerate(views):
                    windows = generate_tile_windows(
                        width=view.width,
                        height=view.height,
                        size=config["inference"]["tile_size"],
                        stride=round(
                            config["inference"]["tile_size"]
                            * (1.0 - config["inference"]["overlap"])
                        ),
                    )
                    crops = [view.crop((w.x1, w.y1, w.x2, w.y2)) for w in windows]
                    predictions = model.predict([view, *crops], **args)
                    if len(predictions) != len(crops) + 1:
                        raise ValueError("detector full/slice cardinality mismatch")
                    scores = []
                    for prediction_index, result in enumerate(predictions):
                        score, rejected = valid_detector_score(result)
                        scores.append(score)
                        invalid_rows.extend(
                            {
                                "filename": chunk[0]["filename"],
                                "condition": condition,
                                "view": "original" if view_index == 0 else "horizontal_flip",
                                "input": "full" if prediction_index == 0 else "tile",
                                "tile": None
                                if prediction_index == 0
                                else prediction_index - 1,
                                "tile_index": None
                                if prediction_index == 0
                                else prediction_index - 1,
                                "tile_window": None
                                if prediction_index == 0
                                else windows[prediction_index - 1].to_dict(),
                                "coordinate_space": "view_pixels",
                                "config_sha256": contract["config_sha256"],
                                "checkpoint_sha256": contract["checkpoint_sha256"],
                                "partition_sha256": contract["partition_sha256"],
                                **box,
                            }
                            for box in rejected
                        )
                    view_scores.append((scores[0], max(scores[1:], default=0.0)))
                    for crop in crops:
                        crop.close()
                if tta:
                    views[1].close()
                else:
                    view_scores.append(view_scores[0])
                (full, sliced), (full_flip, slice_flip) = view_scores
                result_rows.append(
                    {
                        **dict(chunk[0]),
                        "full": full,
                        "slice": sliced,
                        "full_flip": full_flip,
                        "slice_flip": slice_flip,
                        "original": reduce_detector_view(full, sliced),
                        "no_slice": arithmetic_mean_score(full, full_flip),
                        "tta": arithmetic_mean_score(
                            reduce_detector_view(full, sliced),
                            reduce_detector_view(full_flip, slice_flip),
                        ),
                    }
                )
            for image in images:
                image.close()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pd.DataFrame(result_rows)
    if frame["filename"].tolist() != [row["filename"] for row in rows]:
        raise ValueError("prediction coverage mismatch")
    _write_csv(destination, frame)
    geometry_audit: dict[str, Any] = {}
    if task == "detector":
        invalid_ledger_hash = _write_rejection_ledger(invalid_ledger, invalid_rows)
        geometry_audit = {
            "invalid_geometry_policy": "exclude_non_positive_area_and_audit; non_finite_fatal",
            "invalid_boxes": len(invalid_rows),
            "max_invalid_confidence": max(
                (row["confidence"] for row in invalid_rows), default=None
            ),
            "invalid_would_change_max_valid_score": any(
                row.get("would_change_max_valid_score", False) for row in invalid_rows
            ),
            "invalid_boxes_sha256": invalid_ledger_hash,
        }
    _immutable_json(
        sidecar,
        {
            "contract": contract,
            "csv_sha256": sha256_file(destination),
            "elapsed_seconds": time.time() - started,
            "row_count": len(frame),
            **geometry_audit,
        },
        context="prediction manifest",
    )
    return frame


def training_task_names(
    config: Mapping[str, Any], partition: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return the exact queue-addressable training matrix for one cell."""
    miners = tuple(f"miner_inner_{item['validation_fold']}" for item in partition["miners"])
    classifiers = tuple(f"cls_{arm}" for arm in config["arms"])
    if is_scheme_b(config):
        mining_seed = int(scheme_b_contract(config)["mining_seed"])
        return (miners if int(partition["seed"]) == mining_seed else ()) + classifiers
    return miners + classifiers + ("detector",)


def detector_training_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep valid detector positives and all valid fire-negative images."""
    return [dict(row) for row in rows if bool(row.get("detector_eligible", True))]


def _cell_training_context(
    cell: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int, str]:
    partition_path = cell / "partition.json"
    validate_hash_sidecar(partition_path)
    partition = _read_json(partition_path, context="partition contract")
    fit_rows = partition["splits"]["fit"]
    cal_rows = partition["splits"]["cal"]
    seed = partition["seed"]
    source_hash = digest_object(partition)
    return partition, fit_rows, cal_rows, seed, source_hash


def _hard_negative_selection(
    config: Mapping[str, Any],
    cell: Path,
    partition: Mapping[str, Any],
    fit_rows: Sequence[Mapping[str, Any]],
    seed: int,
    source_hash: str,
) -> list[str]:
    """Freeze one fit-only OOF selection and reuse it across training seeds."""
    mining_cell = cell
    mining_partition = partition
    mining_fit_rows = list(fit_rows)
    mining_source_hash = source_hash
    mining_seed = seed
    if is_scheme_b(config):
        mining_seed = int(scheme_b_contract(config)["mining_seed"])
        mining_cell = cell.parent / f"seed_{mining_seed}"
        mining_partition, mining_fit_rows, _, actual_seed, mining_source_hash = (
            _cell_training_context(mining_cell)
        )
        if actual_seed != mining_seed:
            raise ValueError("Scheme B mining cell has the wrong seed identity")
        comparable_keys = (
            "outer_fold",
            "calibration_fold",
            "fit_folds",
            "splits",
            "miners",
            "source_contract_sha256",
        )
        if any(partition.get(key) != mining_partition.get(key) for key in comparable_keys):
            raise ValueError("Scheme B partitions differ beyond their training seed")
    selection_path = mining_cell / "hard_negative_selection.json"
    mining_path = mining_cell / "fit_mining_oof.csv"
    if selection_path.is_file():
        selection = _read_json(selection_path, context="hard-negative selection")
        if selection.get("partition_sha256") != mining_source_hash:
            raise ValueError("hard-negative selection partition hash differs")
        if selection.get("mining_seed", mining_seed) != mining_seed:
            raise ValueError("hard-negative selection mining seed differs")
        hard_names = selection.get("hard_names")
        if not isinstance(hard_names, list) or len(hard_names) != len(set(hard_names)):
            raise ValueError("hard-negative selection identities are invalid")
        eligible = {row["filename"] for row in mining_fit_rows if row["label"] == 0}
        if not set(hard_names) <= eligible or selection.get("K") != len(hard_names):
            raise ValueError("hard-negative selection is outside the frozen fit negatives")
        if not mining_path.is_file():
            raise ValueError("hard-negative selection is missing its OOF evidence table")
        mining = pd.read_csv(mining_path, float_precision="round_trip")
        if selection.get("mining_csv_sha256") != digest_object(mining.to_dict("records")):
            raise ValueError("hard-negative OOF evidence changed")
        return sorted(hard_names)
    mining_frames: list[pd.DataFrame] = []
    for miner in mining_partition["miners"]:
        inner = miner["validation_fold"]
        name = f"miner_inner_{inner}"
        val_rows = [row for row in mining_fit_rows if row["fold"] == inner]
        miner_run, _ = resolve_complete_training_attempt(mining_cell / "models" / name)
        mining_frames.append(
            predict_model(
                config,
                mining_cell,
                task="classifier",
                model_name=name,
                checkpoint=miner_run / "weights" / "last.pt",
                rows=val_rows,
                split="fit_oof",
                tta=False,
                output_dir=miner_run / "predictions",
            )
        )
    mining = pd.concat(mining_frames, ignore_index=True).sort_values("filename", kind="stable")
    mining_names = set(mining["filename"])
    expected_names = {row["filename"] for row in mining_fit_rows}
    if mining_names != expected_names or len(mining) != len(mining_fit_rows):
        raise ValueError("fit-only miner OOF must cover exactly all fit rows")
    threshold = search_thresholds(
        mining["label"].tolist(), mining["original"].tolist(), **config["evaluation"]["threshold"]
    ).best.threshold
    hard_names = sorted(
        mining.loc[(mining["label"] == 0) & (mining["original"] >= threshold), "filename"].tolist()
    )
    selection = {
        "policy": "fit_only_oof_negative_fp_at_fit_pooled_f1_threshold",
        "partition_sha256": mining_source_hash,
        "mining_seed": mining_seed,
        "threshold": threshold,
        "K": len(hard_names),
        "hard_names": hard_names,
        "status": "selected" if hard_names else "no_hard_negatives",
        "no_hard_negative_action": "keep_K_zero_and_report_H1_unidentifiable"
        if not hard_names
        else None,
        "mining_csv_sha256": digest_object(mining.to_dict("records")),
    }
    publish_contract(selection_path, selection)
    _write_csv(mining_path, mining)
    return hard_names


def train_task(config: Mapping[str, Any], cell: Path, task_name: str) -> dict[str, Any]:
    """Run or strictly skip exactly one queue-addressable training task."""
    if config.get("campaign") in FROZEN_CAMPAIGN_IDENTITIES:
        raise ValueError("frozen campaign is read-only; use a new campaign identity")
    validate_formal_training_authorization(config, cell.parents[1])
    partition, fit_rows, cal_rows, seed, source_hash = _cell_training_context(cell)
    valid = training_task_names(config, partition)
    if task_name not in valid:
        raise ValueError(f"unknown training task {task_name!r}; expected one of {valid}")
    if task_name.startswith("miner_inner_"):
        inner = int(task_name.removeprefix("miner_inner_"))
        miner = next(
            (item for item in partition["miners"] if item["validation_fold"] == inner), None
        )
        if miner is None:
            raise ValueError(f"miner task is not declared by partition: {task_name}")
        train_rows = [row for row in fit_rows if row["fold"] in miner["training_folds"]]
        val_rows = [row for row in fit_rows if row["fold"] == inner]
        manifest = train_model(
            config,
            cell / "models" / task_name,
            task="classifier",
            train_rows=train_rows,
            val_rows=val_rows,
            seed=seed,
            source_contract_sha256=source_hash,
            epochs=int(scheme_b_contract(config)["miner_epochs"])
            if is_scheme_b(config)
            else None,
        )
        run, _ = resolve_complete_training_attempt(
            cell / "models" / task_name, manifest["contract"]
        )
        predict_model(
            config,
            cell,
            task="classifier",
            model_name=task_name,
            checkpoint=run / "weights" / "last.pt",
            rows=val_rows,
            split="fit_oof",
            tta=False,
            output_dir=run / "predictions",
        )
    elif task_name.startswith("cls_"):
        arm = task_name.removeprefix("cls_")
        hard_names = _hard_negative_selection(
            config, cell, partition, fit_rows, seed, source_hash
        )
        train_model(
            config,
            cell / "models" / task_name,
            task="classifier",
            train_rows=fit_rows,
            val_rows=cal_rows,
            seed=seed,
            source_contract_sha256=source_hash,
            arm=arm,
            hard_names=hard_names,
        )
    else:
        detector_fit_rows = detector_training_rows(fit_rows)
        detector_cal_rows = detector_training_rows(cal_rows)
        train_model(
            config,
            cell / "models" / "detector",
            task="detector",
            train_rows=detector_fit_rows,
            val_rows=detector_cal_rows,
            seed=seed,
            source_contract_sha256=source_hash,
        )
    run, manifest = resolve_complete_training_attempt(cell / "models" / task_name)
    return {
        "stage": "train_task",
        "task": task_name,
        "status": "complete",
        "attempt": str(run.relative_to(cell)).replace("\\", "/"),
        "manifest_sha256": sha256_file(run / "research_manifest.json"),
        "contract_sha256": manifest["contract_sha256"],
    }


def validate_training_task(
    config: Mapping[str, Any], cell: Path, task_name: str
) -> dict[str, Any]:
    """Validate task completion against the current config and partition hashes."""
    partition, fit_rows, cal_rows, seed, source_hash = _cell_training_context(cell)
    if task_name not in training_task_names(config, partition):
        raise ValueError(f"task is outside the partition matrix: {task_name}")
    if task_name.startswith("miner_inner_"):
        inner = int(task_name.removeprefix("miner_inner_"))
        miner = next(item for item in partition["miners"] if item["validation_fold"] == inner)
        rows = [row for row in fit_rows if row["fold"] in miner["training_folds"]]
        val = [row for row in fit_rows if row["fold"] == inner]
        arm, hard_names, task = "hn", (), "classifier"
    elif task_name.startswith("cls_"):
        rows, val = fit_rows, cal_rows
        arm = task_name.removeprefix("cls_")
        hard_names = _hard_negative_selection(
            config, cell, partition, fit_rows, seed, source_hash
        )
        task = "classifier"
    else:
        rows, val = detector_training_rows(fit_rows), detector_training_rows(cal_rows)
        arm, hard_names, task = "hn", (), "detector"
    expected = training_contract(
        config,
        task=task,
        train_rows=rows,
        val_rows=val,
        seed=seed,
        source_contract_sha256=source_hash,
        arm=arm,
        hard_names=hard_names,
        epochs=(
            int(scheme_b_contract(config)["miner_epochs"])
            if is_scheme_b(config) and task_name.startswith("miner_inner_")
            else None
        ),
    )
    run, manifest = resolve_complete_training_attempt(
        cell / "models" / task_name, expected, context=f"{task_name} training task"
    )
    return {
        "task": task_name,
        "run_dir": str(run),
        "manifest_sha256": sha256_file(run / "research_manifest.json"),
        "contract_sha256": manifest["contract_sha256"],
    }


def train_cell(
    config: Mapping[str, Any], cell: Path, *, arms: Sequence[str], include_detector: bool = True
) -> dict[str, Any]:
    """Train fit-only miners, freeze K, then train matched classifier arms."""
    partition, fit_rows, _, _, source_hash = _cell_training_context(cell)
    for task_name in training_task_names(config, partition):
        if not task_name.startswith("miner_inner_"):
            continue
        train_task(config, cell, task_name)
    hard_names = _hard_negative_selection(
        config, cell, partition, fit_rows, int(partition["seed"]), source_hash
    )
    for arm in arms:
        train_task(config, cell, f"cls_{arm}")
    if include_detector:
        train_task(config, cell, "detector")
    status = {
        "stage": "train",
        "requested_arms": list(arms),
        "K": len(hard_names),
        "detector": include_detector,
        "status": "requested_models_complete",
        "full_arm_coverage": set(arms) == set(config["arms"]),
    }
    write_json_atomic(cell / "training_status.json", status)
    return status


def _method_scores(
    raw: Mapping[str, pd.DataFrame], arms: Sequence[str], *, include_detector: bool
) -> dict[str, tuple[list[float], list[float] | None]]:
    methods: dict[str, tuple[list[float], list[float] | None]] = {}
    for arm in arms:
        methods[f"cls_{arm}"] = (raw[f"cls_{arm}"]["original"].tolist(), None)
    if "hn" in arms and include_detector:
        methods["cls_hn_tta"] = (raw["cls_hn"]["tta"].tolist(), None)
    if include_detector:
        det = raw["detector"]
        methods["det_full"] = (det["tta"].tolist(), None)
        methods["det_no_slice"] = (det["no_slice"].tolist(), None)
        methods["det_no_tta"] = (det["original"].tolist(), None)
        for arm in arms:
            methods[f"fusion_{arm}_full"] = (det["tta"].tolist(), raw[f"cls_{arm}"]["tta"].tolist())
        if "hn" in arms:
            methods["fusion_hn_no_slice"] = (
                det["no_slice"].tolist(),
                raw["cls_hn"]["tta"].tolist(),
            )
            methods["fusion_hn_no_tta"] = (
                det["original"].tolist(),
                raw["cls_hn"]["original"].tolist(),
            )
    return methods


def _calibrate(
    labels: Sequence[int],
    first: Sequence[float],
    second: Sequence[float] | None,
    config: Mapping[str, Any],
) -> dict[str, float]:
    threshold = config["evaluation"]["threshold"]
    if second is None:
        selected = search_thresholds(labels, first, **threshold)
        return {"threshold": selected.best.threshold, "classifier_weight": 0.0}
    weight, selected = _select_weight_and_threshold(
        labels,
        first,
        second,
        weight_start=Decimal("0"),
        weight_end=Decimal("1"),
        weight_step=Decimal(str(config["evaluation"]["weight_step"])),
        threshold_start=Decimal(str(threshold["start"])),
        threshold_end=Decimal(str(threshold["end"])),
        threshold_step=Decimal(str(threshold["step"])),
    )
    return {"threshold": selected.best.threshold, "classifier_weight": weight}


def _evaluation_inputs(
    config: Mapping[str, Any],
    cell: Path,
    *,
    arms: Sequence[str],
    include_detector: bool,
) -> tuple[dict[str, Any], list[str], dict[str, Any], Path]:
    """Load immutable evaluation inputs without writing any cell outputs."""
    expected_arms = tuple(config.get("arms", ARMS))
    expected_detector = not is_scheme_b(config)
    if (
        set(arms) != set(expected_arms)
        or len(arms) != len(expected_arms)
        or include_detector != expected_detector
    ):
        raise ValueError(
            "evaluation request differs from the configured method matrix; legacy evaluation "
            "requires all four arms and the detector"
        )
    arms = expected_arms
    partition_path = cell / "partition.json"
    validate_hash_sidecar(partition_path)
    partition = _read_json(partition_path, context="partition contract")
    campaign_contract_path = cell.parents[1] / "campaign_contract.json"
    validate_hash_sidecar(campaign_contract_path)
    campaign_contract = _read_json(campaign_contract_path, context="campaign contract")
    if campaign_contract.get("schema_version") != CAMPAIGN_CONTRACT_SCHEMA_VERSION:
        raise ValueError("legacy campaign cannot be evaluated; use a new campaign identity")
    if partition.get("source_contract_sha256") != digest_object(campaign_contract):
        raise ValueError("partition is not bound to the immutable campaign contract")
    model_names = list(configured_model_names(config))
    training = _training_artifacts(config, cell, partition, model_names)
    evaluation_contract = _evaluation_contract(config, cell, partition, model_names, training)
    return partition, model_names, evaluation_contract, cell / "evaluation_manifest.json"


def validate_evaluation_cell(
    config: Mapping[str, Any],
    cell: Path,
    *,
    arms: Sequence[str] | None = None,
    include_detector: bool | None = None,
) -> dict[str, Any]:
    """Validate a completed evaluation without launching inference."""
    requested_arms = tuple(config["arms"]) if arms is None else tuple(arms)
    requested_detector = (not is_scheme_b(config)) if include_detector is None else include_detector
    _partition, model_names, evaluation_contract, evaluation_path = _evaluation_inputs(
        config, cell, arms=requested_arms, include_detector=requested_detector
    )
    if not evaluation_path.is_file():
        raise IncompleteEvaluation(f"evaluation has no completion manifest: {cell}")
    existing = _read_json(evaluation_path, context="evaluation manifest")
    return _validate_complete_evaluation(cell, existing, evaluation_contract, model_names)


def evaluate_cell(
    config: Mapping[str, Any], cell: Path, *, arms: Sequence[str], include_detector: bool = True
) -> dict[str, Any]:
    """Freeze calibration parameters before loading or predicting any test image."""
    try:
        return validate_evaluation_cell(
            config, cell, arms=arms, include_detector=include_detector
        )
    except IncompleteEvaluation as exc:
        if config.get("campaign") in FROZEN_CAMPAIGN_IDENTITIES:
            raise ValueError(
                "frozen campaign is read-only; use a new campaign identity"
            ) from exc
    partition, model_names, evaluation_contract, evaluation_path = _evaluation_inputs(
        config, cell, arms=arms, include_detector=include_detector
    )
    arms = tuple(config["arms"])
    training = _training_artifacts(config, cell, partition, model_names)
    partial_evaluation_paths = [
        cell / "calibration.json",
        cell / "predictions.csv",
        cell / "metrics.csv",
        cell / "evaluation_manifest.sha256",
        cell / "evaluation_completion.json",
    ]
    for sidecar in _expected_raw_sidecars(cell, model_names):
        partial_evaluation_paths.extend(
            [sidecar, sidecar.with_name(sidecar.name.replace(".manifest.json", ".csv"))]
        )
    if any(path.exists() for path in partial_evaluation_paths):
        raise ValueError("incomplete evaluation outputs exist; use a new campaign identity")
    all_rows: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for split in ("cal", "test"):
        rows = partition["splits"][split]
        raw = {
            name: predict_model(
                config,
                cell,
                task="detector" if name == "detector" else "classifier",
                model_name=name,
                checkpoint=Path(training[name]["run_dir"]) / "weights" / "last.pt",
                rows=rows,
                split=split,
            )
            for name in model_names
        }
        methods = _method_scores(raw, arms, include_detector=include_detector)
        if split == "cal":
            selection = {
                name: _calibrate([row["label"] for row in rows], first, second, config)
                for name, (first, second) in methods.items()
            }
            publish_contract(
                cell / "calibration.json",
                {
                    "policy": "cal_only_before_test_predictions",
                    "partition_sha256": digest_object(partition),
                    "methods": selection,
                    "cal_raw_sha256": {
                        name: sha256_file(cell / "raw" / f"{name}_cal.csv") for name in model_names
                    },
                },
            )
        for method, (first, second) in methods.items():
            params = selection[method]
            weight = params["classifier_weight"]
            scores = (
                first
                if second is None
                else [(1 - weight) * x + weight * y for x, y in zip(first, second, strict=True)]
            )
            for row, score in zip(rows, scores, strict=True):
                all_rows.append(
                    {
                        "experiment_id": (
                            f"{config['campaign']}_outer{partition['outer_fold']}"
                            f"_seed{partition['seed']}"
                        ),
                        "method": method,
                        "outer_fold": partition["outer_fold"],
                        "seed": partition["seed"],
                        "filename": row["filename"],
                        "label": row["label"],
                        "group_id": row["group_id"],
                        "score": score,
                        "threshold": params["threshold"],
                        "classifier_weight": weight,
                        "y_pred": int(score >= params["threshold"]),
                        "split": split,
                        "condition": "clean",
                        "evidence_status": "engineering_only"
                        if config["engineering_only"]
                        else "internal_revalidation",
                        "calibration_fold": partition["calibration_fold"],
                    }
                )
    predictions = pd.DataFrame(all_rows)
    output = cell / "predictions.csv"
    _write_csv(output, predictions)
    metrics = []
    for (method, split), frame in predictions.groupby(["method", "split"], sort=True):
        result = compute_binary_metrics(frame["label"].tolist(), frame["y_pred"].tolist(), 0.5)
        metrics.append(
            {
                "method": method,
                "split": split,
                **asdict(result),
                "fpr": result.fp / (result.fp + result.tn)
                if result.fp + result.tn
                else 0.0,
            }
        )
    _write_csv(cell / "metrics.csv", pd.DataFrame(metrics))
    raw_manifests = {
        sidecar.name: sha256_file(sidecar)
        for sidecar in _expected_raw_sidecars(cell, model_names)
    }
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "cell_complete",
        "contract": evaluation_contract,
        "contract_sha256": digest_object(evaluation_contract),
        "engineering_only": config["engineering_only"],
        "predictions_sha256": sha256_file(output),
        "metrics_sha256": sha256_file(cell / "metrics.csv"),
        "rows": len(predictions),
        "methods": sorted(selection),
        "test_unique_images": len(partition["splits"]["test"]),
        "cal_unique_images": len(partition["splits"]["cal"]),
        "outer_fold": partition["outer_fold"],
        "seed": partition["seed"],
        "completed_at_unix": time.time(),
        "calibration_sha256": sha256_file(cell / "calibration.json"),
        "raw_manifests": raw_manifests,
        "training_manifests": {
            name: training[name]["manifest_sha256"] for name in model_names
        },
        "evidence_status": (
            "engineering_only" if config["engineering_only"] else "internal_revalidation"
        ),
        "limitations": [
            "internal D-Fire evaluation does not establish external generalization",
            "one cell does not complete the campaign",
            "predeclared synthetic perturbations require the separate robustness chain",
        ],
    }
    _immutable_json(evaluation_path, manifest, context="evaluation manifest")
    _write_immutable_bytes(
        cell / "evaluation_manifest.sha256",
        (sha256_file(evaluation_path) + "\n").encode("ascii"),
        context="evaluation manifest integrity sidecar",
    )
    _immutable_json(
        cell / "evaluation_completion.json",
        {
            "manifest_sha256": sha256_file(evaluation_path),
            "completed_at_unix": manifest["completed_at_unix"],
        },
        context="evaluation completion record",
    )
    return manifest
