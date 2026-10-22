"""Auditable group partitions and matched exposure contracts for Direction A."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from fire_detection.models.offline import sha256_file
from fire_detection.models.training import FoldRecord, load_fold_records, write_json_atomic
from .immutable_inputs import snapshot_identity

ARMS = ("hn", "random_negative", "uniform_negative", "uniform_all")
SCHEME_B_PROFILE = "dfire_split_v2_frozen_author_final_v1"
SCHEME_B_ARMS = ("hn", "random_negative")
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# These files are part of the executable contract, not optional documentation.
# A campaign can only be resumed when the same dependency declarations and
# verified local-asset manifest are still present.  They are copied into a new
# campaign snapshot; legacy campaigns which do not contain them are never
# upgraded in place.
DEPENDENCY_FILES = (
    "requirements/lock-win-py311-cu128.txt",
    "environment/fire26-conda-history.yml",
    "weights/pretrained/manifest.json",
)
ENTRYPOINT_FILES = (
    "scripts/run_direction_a.py",
    "scripts/run_direction_a_queue.py",
    "scripts/collect_direction_a.py",
    "scripts/analyze_direction_a.py",
    "scripts/run_direction_a_robustness.py",
    "scripts/run_direction_a_sensitivity.py",
    "scripts/audit_direction_a.py",
    "scripts/build_dfire_scheme_b_pool.py",
    "scripts/run_scheme_b_fast_canary.py",
)
RUNTIME_PACKAGES = ("torch", "ultralytics", "numpy", "pandas", "Pillow", "PyYAML")


def digest_object(value: Any) -> str:
    """Hash canonical JSON without accepting non-finite numbers."""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def project_path(root: Path, value: str) -> Path:
    """Resolve an existing-project path and reject escape or network paths."""
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Direction A paths must stay inside the project: {value}")
    return path


def existing_campaign_identities(root: Path = PROJECT_ROOT) -> set[str]:
    """Collect campaign identities from configs and existing run directories."""
    identities: set[str] = set()
    config_dir = root / "configs" / "direction_a"
    if config_dir.is_dir():
        for path in sorted(config_dir.glob("*.yaml")):
            try:
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError):
                continue
            if isinstance(payload, dict):
                campaign = payload.get("campaign")
                if isinstance(campaign, str) and campaign:
                    identities.add(campaign)
    runs = root / "runs" / "direction_a"
    if runs.is_dir():
        for child in runs.iterdir():
            if not child.is_dir():
                continue
            if (child / "campaign_contract.json").is_file() or child.name.startswith(
                ("main", "smoke")
            ):
                identities.add(child.name)
    return identities


def next_unique_campaign_identity(family: str, root: Path = PROJECT_ROOT) -> str:
    """Return the next unused main_vN or smoke_vN identity."""
    if family not in {"main", "smoke"}:
        raise ValueError("campaign family must be 'main' or 'smoke'")
    used = existing_campaign_identities(root)
    number = 1
    while True:
        candidate = f"{family}_v{number}"
        if candidate not in used:
            return candidate
        number += 1
        if number > 10_000:
            raise RuntimeError(f"campaign identity space exhausted for {family}")


def _dependency_snapshot(root: Path) -> dict[str, Any]:
    """Return immutable dependency and entry-point evidence for a campaign."""
    files: dict[str, str] = {}
    for relative in DEPENDENCY_FILES:
        path = project_path(root, relative)
        if not path.is_file():
            raise ValueError(f"required dependency declaration is missing: {relative}")
        files[relative] = sha256_file(path)
    entrypoints: dict[str, str] = {}
    for relative in ENTRYPOINT_FILES:
        path = project_path(root, relative)
        if not path.is_file():
            raise ValueError(f"required Direction A entrypoint is missing: {relative}")
        entrypoints[relative] = sha256_file(path)
    packages: dict[str, str | None] = {}
    for package in RUNTIME_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "files_sha256": files,
        "entrypoints_sha256": entrypoints,
        "python": platform.python_version(),
        "packages": packages,
        "executable": str(Path(sys.executable).resolve()),
    }


def load_config(path: Path, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Read the research contract, keeping every artifact in an isolated namespace."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "campaign",
        "engineering_only",
        "data",
        "offline",
        "seeds",
        "outer_folds",
        "arms",
        "training",
        "classifier",
        "detector",
        "inference",
        "evaluation",
    }
    optional = {"scheme_b", "budget_policy", "budget_policy_version", "training_protocol_version"}
    if (
        not isinstance(config, dict)
        or not required <= set(config)
        or set(config) - required - optional
    ):
        raise ValueError(
            f"research config keys must contain {sorted(required)} and may include "
            f"{sorted(optional)}"
        )
    if config["schema_version"] != 1 or type(config["engineering_only"]) is not bool:
        raise ValueError("schema_version=1 and a boolean engineering_only are required")
    campaign = config["campaign"]
    if (
        not isinstance(campaign, str)
        or not campaign
        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in campaign)
    ):
        raise ValueError("campaign must contain only lowercase ASCII letters, digits, '_' or '-'")
    profile = config["data"].get("dataset_profile", "legacy_v3_1100")
    expected_arms = list(SCHEME_B_ARMS) if profile == SCHEME_B_PROFILE else list(ARMS)
    if config["arms"] != expected_arms:
        raise ValueError(f"arms must be exactly {expected_arms} for {profile}")
    if config["inference"]["invalid_geometry"] != "exclude_non_positive_area_and_audit":
        raise ValueError("detector geometry policy must be explicit and fixed")
    if any(type(seed) is not int or seed < 0 for seed in config["seeds"]) or not config["seeds"]:
        raise ValueError("seeds must be nonempty nonnegative integers")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("duplicate seeds are forbidden")
    if (
        any(type(fold) is not int or fold not in range(5) for fold in config["outer_folds"])
        or not config["outer_folds"]
    ):
        raise ValueError("outer_folds must contain values in 0..4")
    if len(set(config["outer_folds"])) != len(config["outer_folds"]):
        raise ValueError("duplicate outer_folds are forbidden")
    train = config["training"]
    execution_profile = train.get("execution_profile", "frozen_reference_v1")
    if execution_profile == "frozen_reference_v1":
        if train["workers"] != 0:
            raise ValueError("frozen_reference_v1 requires workers=0")
        if train["amp"] is not False:
            raise ValueError("frozen_reference_v1 requires amp=false")
    elif execution_profile in {"scheme_b_fast_v1", "scheme_b_fp32_v1"}:
        if profile != SCHEME_B_PROFILE:
            raise ValueError("Scheme B execution profiles require dfire_split_v2_frozen_author_final_v1")
        if type(train["workers"]) is not int or not 1 <= train["workers"] <= 8:
            raise ValueError("Scheme B execution-profile workers must be in 1..8")
        expected_amp = execution_profile == "scheme_b_fast_v1"
        if train["amp"] is not expected_amp:
            raise ValueError(
                f"{execution_profile} requires amp={str(expected_amp).lower()}"
            )
        if train.get("cache") is not False or train.get("compile") is not False:
            raise ValueError("Scheme B execution profiles require cache=false and compile=false")
    else:
        raise ValueError(f"unsupported training.execution_profile: {execution_profile!r}")
    if train["optimizer"] == "auto":
        raise ValueError("optimizer must be pinned")
    if train["deterministic"] is not True or train["patience"] != 0:
        raise ValueError("deterministic=true and patience=0 are required")
    if (
        config["classifier"]["weights"] != "weights/pretrained/yolo26s-cls.pt"
        or config["detector"]["weights"] != "weights/pretrained/yolo26s.pt"
    ):
        raise ValueError("only existing validated YOLO26-S pretrained assets are allowed")
    for task in ("classifier", "detector"):
        for key in ("epochs", "batch", "imgsz"):
            if type(config[task][key]) is not int or config[task][key] <= 0:
                raise ValueError(f"{task}.{key} must be a positive integer")
        if config[task]["imgsz"] % 32:
            raise ValueError("image sizes must be multiples of 32")
    if not config["engineering_only"] and config["classifier"]["epochs"] < 2:
        raise ValueError("one-epoch runs must be engineering_only")
    if (
        not config["engineering_only"]
        and profile != SCHEME_B_PROFILE
        and config["detector"]["epochs"] < 2
    ):
        raise ValueError("one-epoch detector runs must be engineering_only")
    data = config["data"]
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping")
    required_data = {"folds", "images", "labels", "processed_manifest"}
    if not required_data <= set(data):
        raise ValueError(f"data must contain {sorted(required_data)}")
    profile = data.get("dataset_profile", "legacy_v3_1100")
    if profile == "legacy_v3_1100":
        if (
            data["folds"] != "data/folds/v3/locked/folds.csv"
            or data["images"] != "data/processed/v3/images"
        ):
            raise ValueError("legacy_v3_1100 must use the frozen historical paths")
        data.setdefault("expected_count", 1100)
        data.setdefault("training_source", "historical_1100")
        data.setdefault("original_training_data_allowed", True)
    elif profile == "dfire_only_v2":
        expected_paths = {
            "folds": "data/processed/dfire_v2/folds.csv",
            "images": "data/processed/dfire_v2/images",
            "labels": "data/processed/dfire_v2/labels",
            "processed_manifest": "data/processed/dfire_v2/manifest.json",
        }
        if any(data.get(key) != value for key, value in expected_paths.items()):
            raise ValueError("dfire_only_v2 must use isolated D-Fire copy paths")
        if data.get("expected_count") != 21509:
            raise ValueError("dfire_only_v2 must contain exactly 21,509 eligible images")
        if data.get("training_source") != "D-Fire":
            raise ValueError("dfire_only_v2 training_source must be D-Fire")
        if data.get("original_training_data_allowed") is not False:
            raise ValueError("dfire_only_v2 must forbid the original 1,100 images from training")
        if data.get("dataset_id") != "dfire_master_4bf9c31_20260916":
            raise ValueError("unexpected D-Fire dataset identity")
    elif profile == SCHEME_B_PROFILE:
        common_snapshot = data.get("input_snapshot_manifest") is not None
        folder = "data/processed/dfire_eoi_snapshot_20261020_v1" if common_snapshot else "data/processed/dfire_split_v2_frozen_author_final_v1"
        expected_paths = {
            "folds": folder + "/folds.csv",
            "images": folder + "/images",
            "labels": folder + "/labels",
            "processed_manifest": folder + "/manifest.json",
        }
        if common_snapshot and (data["input_snapshot_manifest"] != folder + "/input_snapshot.json" or not isinstance(data.get("input_snapshot_manifest_sha256"), str) or len(data["input_snapshot_manifest_sha256"]) != 64):
            raise ValueError("New common snapshot identity/path/hash must be explicit")
        if any(data.get(key) != value for key, value in expected_paths.items()):
            raise ValueError("dfire_split_v2_frozen_author_final_v1 must use its isolated processed paths")
        if data.get("expected_count") != 1100:
            raise ValueError("dfire_split_v2_frozen_author_final_v1 must contain exactly 1,100 images")
        if data.get("training_source") != "D-Fire":
            raise ValueError("dfire_split_v2_frozen_author_final_v1 training_source must be D-Fire")
        if data.get("original_training_data_allowed") is not False:
            raise ValueError("dfire_split_v2_frozen_author_final_v1 forbids legacy images")
        expected_dataset_id = "dfire_1100_components427_eoi_snapshot_20261020_v1" if common_snapshot else "dfire_1100_components427_final_20261020_v1"
        if data.get("dataset_id") != expected_dataset_id:
            raise ValueError("unexpected Scheme B dataset identity")
    else:
        raise ValueError(f"unsupported data.dataset_profile: {profile!r}")
    if profile == SCHEME_B_PROFILE:
        scheme_b = config.get("scheme_b")
        if scheme_b is None:
            # Existing engineering canaries predate the formal Scheme B
            # contract.  Their single declared seed remains their mining seed;
            # no formal campaign may rely on these defaults.
            if not config["engineering_only"]:
                raise ValueError("formal Scheme B configs require an explicit scheme_b contract")
            scheme_b = {
                "mining_seed": config["seeds"][0],
                "random_negative_identity_seed": config["inference"]["perturbation_seed"],
                "miner_epochs": config["classifier"]["epochs"],
            }
            config["scheme_b"] = scheme_b
        required_scheme_b = {
            "mining_seed",
            "random_negative_identity_seed",
            "miner_epochs",
        }
        if not isinstance(scheme_b, dict) or set(scheme_b) != required_scheme_b:
            raise ValueError(f"scheme_b keys must be exactly {sorted(required_scheme_b)}")
        if (
            type(scheme_b["mining_seed"]) is not int
            or scheme_b["mining_seed"] not in config["seeds"]
        ):
            raise ValueError("scheme_b.mining_seed must be one of the declared training seeds")
        if config["seeds"][0] != scheme_b["mining_seed"]:
            raise ValueError("scheme_b.mining_seed must be first so queue dependencies are ordered")
        if (
            type(scheme_b["random_negative_identity_seed"]) is not int
            or scheme_b["random_negative_identity_seed"] < 0
        ):
            raise ValueError("scheme_b.random_negative_identity_seed must be nonnegative")
        if type(scheme_b["miner_epochs"]) is not int or scheme_b["miner_epochs"] <= 0:
            raise ValueError("scheme_b.miner_epochs must be a positive integer")
        if not config["engineering_only"] and scheme_b["miner_epochs"] < 2:
            raise ValueError("one-epoch miners must be engineering_only")
    if type(data.get("expected_count")) is not int or data["expected_count"] <= 0:
        raise ValueError("data.expected_count must be a positive integer")
    for key in required_data | {"source_contract", "overlap_exclusion_contract"}:
        value = data.get(key)
        if value is not None:
            project_path(root, value)
    if config.get("budget_policy") != "B_FULL_40_EPOCHS" or config.get("budget_policy_version") != "FULL-STRATEGY-B-40E-20261020-V1" or config.get("training_protocol_version") != "FULL-STRATEGY-B-40E-20261020-V1":
        raise ValueError("New release requires an explicit versioned budget policy")
    config["_root"] = str(root.resolve())
    config["_config_path"] = str(path.resolve())
    config["_config_sha256"] = sha256_file(path)
    return config


def is_scheme_b(config: Mapping[str, Any]) -> bool:
    """Return whether a loaded configuration uses the frozen Scheme B profile."""
    data = config.get("data")
    return isinstance(data, Mapping) and data.get("dataset_profile") == SCHEME_B_PROFILE


def scheme_b_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the validated Scheme B controls for a loaded configuration."""
    if not is_scheme_b(config):
        raise ValueError("Scheme B controls requested for a different dataset profile")
    value = config.get("scheme_b")
    if not isinstance(value, Mapping):
        raise ValueError("loaded Scheme B config has no validated scheme_b contract")
    return value


def configured_model_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the models that must be evaluated for one completed cell."""
    classifiers = tuple(f"cls_{arm}" for arm in config["arms"])
    return classifiers if is_scheme_b(config) else classifiers + ("detector",)


def configured_evaluation_methods(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact paper-facing method matrix for this protocol."""
    if is_scheme_b(config):
        return tuple(f"cls_{arm}" for arm in SCHEME_B_ARMS)
    return (
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


def replay_seed(config: Mapping[str, Any], *, arm: str, training_seed: int) -> int:
    """Decouple the fixed random-negative identity from model-training randomness."""
    if is_scheme_b(config) and arm == "random_negative":
        return int(scheme_b_contract(config)["random_negative_identity_seed"])
    return training_seed


def formal_training_authorization_path(campaign: Path) -> Path:
    """Return the only accepted location for an explicit author decision record."""
    return campaign / "authorization" / "formal_training.json"


def validate_formal_training_authorization(
    config: Mapping[str, Any], campaign: Path
) -> dict[str, Any] | None:
    """Fail closed before formal Scheme B training until the author approves it.

    Planning, queue construction, validation, collection, and analysis remain
    available without this record.  Only model-training launch paths call this
    gate.  The record is intentionally outside the immutable queue/config so a
    later author decision does not rewrite the already-reviewed protocol.
    """
    if config.get("engineering_only", True) or not is_scheme_b(config):
        return None
    path = formal_training_authorization_path(campaign)
    if not path.is_file():
        raise PermissionError(
            "formal Scheme B training is not authorized; missing author decision record"
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PermissionError("formal training authorization record is unreadable") from exc
    contract_path = campaign / "campaign_contract.json"
    required = {
        "schema_version": 1,
        "decision": "AUTHOR_APPROVED_FORMAL_TRAINING",
        "formal_training_authorized": True,
        "campaign_identity": config["campaign"],
        "config_sha256": config["_config_sha256"],
        "campaign_contract_sha256": sha256_file(contract_path),
    }
    if not isinstance(record, dict) or any(
        record.get(key) != value for key, value in required.items()
    ):
        raise PermissionError("formal training authorization does not match this campaign")
    confirmation = record.get("author_confirmation")
    if not isinstance(confirmation, str) or not confirmation.strip():
        raise PermissionError("formal training authorization lacks author confirmation evidence")
    return record


def build_partition(records: Sequence[FoldRecord], outer_fold: int, seed: int) -> dict[str, Any]:
    """Rotate fixed groups into fit3/cal1/test1; miners never see cal or test."""
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be 0..4")
    names = [r.filename for r in records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate source filenames")
    groups: dict[str, set[int]] = {}
    for record in records:
        if record.label not in (0, 1) or record.fold not in range(5):
            raise ValueError("invalid label or fold")
        groups.setdefault(record.group_id, set()).add(record.fold)
    if any(len(folds) != 1 for folds in groups.values()):
        raise ValueError("scene group crosses locked folds")
    cal_fold = (outer_fold + 1) % 5
    fit_folds = sorted(set(range(5)) - {outer_fold, cal_fold})
    splits = {
        "fit": [asdict(r) for r in records if r.fold in fit_folds],
        "cal": [asdict(r) for r in records if r.fold == cal_fold],
        "test": [asdict(r) for r in records if r.fold == outer_fold],
    }
    if any({r["label"] for r in rows} != {0, 1} for rows in splits.values()):
        raise ValueError("fit/cal/test must each contain both classes")
    miners = [
        {"validation_fold": fold, "training_folds": [f for f in fit_folds if f != fold]}
        for fold in fit_folds
    ]
    return {
        "outer_fold": outer_fold,
        "calibration_fold": cal_fold,
        "fit_folds": fit_folds,
        "seed": seed,
        "splits": splits,
        "miners": miners,
    }


def replay_indices(
    filenames: Sequence[str],
    labels: Sequence[int],
    hard_names: Sequence[str],
    *,
    arm: str,
    seed: int,
    epoch: int,
) -> tuple[int, ...]:
    """Return N+K indices; rotating controls distribute extras uniformly across epochs."""
    if arm not in ARMS or epoch < 0 or len(filenames) != len(labels):
        raise ValueError("invalid replay request")
    if len(set(filenames)) != len(filenames) or len(set(hard_names)) != len(hard_names):
        raise ValueError("duplicate filenames or hard negatives")
    by_name = {name: i for i, name in enumerate(filenames)}
    if any(name not in by_name or labels[by_name[name]] != 0 for name in hard_names):
        raise ValueError("hard negatives must be negative fit samples")
    # Replay selection is defined over canonical filenames, then mapped back to
    # the caller's positional order.  Ultralytics classification datasets are
    # grouped by class directory, whereas the frozen contract uses filename
    # order; selecting positional indices directly would therefore produce a
    # different per-file ledger with the same total exposure.
    canonical_names = sorted(filenames)
    canonical_negative_names = [
        name for name in canonical_names if labels[by_name[name]] == 0
    ]
    k = len(hard_names)
    if k > len(canonical_negative_names):
        raise ValueError("K exceeds available unique negatives")
    if arm == "hn":
        extra_names = sorted(hard_names)
    elif arm == "random_negative":
        extra_names = random.Random(seed).sample(canonical_negative_names, k)
    else:
        population = (
            canonical_negative_names
            if arm == "uniform_negative"
            else canonical_names
        )
        random.Random(seed).shuffle(population)
        extra_names = [population[(epoch * k + j) % len(population)] for j in range(k)]
    extras = [by_name[name] for name in extra_names]
    return tuple([*range(len(filenames)), *extras])


def expected_budget(n: int, k: int, batch: int, epochs: int) -> dict[str, int]:
    """Count exact exposures/steps with nbs=batch and no drop-last or AMP skips."""
    return {
        "unique_fit_samples": n,
        "extra_exposures_per_epoch": k,
        "effective_samples_per_epoch": n + k,
        "sample_exposures": (n + k) * epochs,
        "optimizer_steps": math.ceil((n + k) / batch) * epochs,
        "epochs": epochs,
        "batch": batch,
    }


def expected_exposure_counts(
    filenames: Sequence[str],
    labels: Sequence[int],
    hard_names: Sequence[str],
    *,
    arm: str,
    seed: int,
    epochs: int,
) -> dict[str, int]:
    """Derive the exact per-source exposure contract across every training epoch."""
    counts: Counter[str] = Counter()
    for epoch in range(epochs):
        indices = replay_indices(filenames, labels, hard_names, arm=arm, seed=seed, epoch=epoch)
        counts.update(filenames[index] for index in indices)
    return dict(sorted(counts.items()))


def verify_source(config: Mapping[str, Any]) -> tuple[tuple[FoldRecord, ...], dict[str, Any]]:
    """Verify the configured immutable dataset without allowing source substitution."""
    root = Path(config["_root"])
    data = config["data"]
    records = load_fold_records(
        project_path(root, data["folds"]), expected_count=int(data["expected_count"])
    )
    manifest_path = project_path(root, data["processed_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = str(data.get("dataset_profile", "legacy_v3_1100"))
    if manifest.get("dataset_profile", profile) != profile:
        raise ValueError("processed manifest dataset profile differs from config")
    if profile in {"dfire_only_v2", "dfire_split_v2_frozen_author_final_v1"}:
        if manifest.get("dataset_id") != data["dataset_id"]:
            raise ValueError("D-Fire manifest dataset identity differs")
        if manifest.get("training_source") != "D-Fire":
            raise ValueError("D-Fire manifest does not lock D-Fire as the training source")
        if manifest.get("original_training_data_allowed") is not False:
            raise ValueError("D-Fire manifest permits forbidden original training data")
    manifest_rows = {row["filename"]: row for row in manifest["images"]}
    if set(manifest_rows) != {record.filename for record in records}:
        raise ValueError("processed manifest and locked fold filenames differ")
    image_hashes: dict[str, str] = {}
    label_hashes: dict[str, str] = {}
    for record in records:
        row = manifest_rows[record.filename]
        if (row["label"], row["fold"], row["group_id"]) != (record.label, record.fold, record.group_id):
            raise ValueError(f"processed label/fold mismatch: {record.filename}")
        if bool(row.get("detector_eligible", True)) != record.detector_eligible:
            raise ValueError(f"detector eligibility mismatch: {record.filename}")
        actual = sha256_file(project_path(root, data["images"]) / record.filename)
        if actual != row["processed_sha256"]:
            raise ValueError(f"processed image hash drift: {record.filename}")
        image_hashes[record.filename] = actual
        label_hashes[record.filename] = sha256_file(
            project_path(root, data["labels"]) / f"{Path(record.filename).stem}.txt"
        )
    if profile == SCHEME_B_PROFILE:
        from .split_release_binding import verify_binding
        release_binding = verify_binding(config, records, manifest, image_hashes, label_hashes)
    else:
        release_binding = None
    implementation_paths = (
        "research/immutable_inputs.py",
        "research/frozen_dataset.py",
        "research/node_logging.py",
        "source_provenance.py",
        "research/split_release_binding.py",
        "research/split_role_guard.py",
        "research/three_arm_budget.py",
        "research/three_arm_execution.py",
        "research/three_arm_recovery.py",
        "research/three_arm_evaluation.py",
        "research/route_b_evaluation.py",
        "research/route_b_preflight.py",
        "research/focal_sampling.py",
        "research/focal_training.py",
        "research/focal_loader.py",
        "research/focal_audit.py",
        "research/strict_release_guard.py",
        "research/protocol.py",
        "research/training.py",
        "research/pipeline.py",
        "research/statistics.py",
        "research/plots.py",
        "research/__init__.py",
        "research/perturbations.py",
        "models/offline.py",
        "models/features.py",
        "models/training.py",
        "inference/slicing.py",
        "inference/tta.py",
        "ensemble/evaluation.py",
        "metrics/thresholds.py",
        "metrics/binary.py",
    )
    code = {
        relative: sha256_file(root / "src" / "fire_detection" / relative)
        for relative in implementation_paths
    }
    source_contract_hashes = {
        key: sha256_file(project_path(root, data[key]))
        for key in ("source_contract", "overlap_exclusion_contract")
        if data.get(key) is not None
    }
    return records, {
        "release_binding": release_binding,
        "dataset_profile": profile,
        "dataset_id": data.get("dataset_id"),
        "training_source": data.get("training_source"),
        "original_training_data_allowed": data.get("original_training_data_allowed"),
        "expected_count": data["expected_count"],
        "folds_sha256": sha256_file(project_path(root, data["folds"])),
        "input_snapshot": snapshot_identity(config),
        "processed_manifest_sha256": sha256_file(manifest_path),
        "source_contracts_sha256": source_contract_hashes,
        "image_hashes": image_hashes,
        "label_hashes": label_hashes,
        "implementation_sha256": code,
        "dependency_snapshot": _dependency_snapshot(root),
        "initial_weights_sha256": {
            task: sha256_file(project_path(root, config[task]["weights"]))
            for task in ("classifier", "detector")
        },
    }


def publish_contract(path: Path, payload: Mapping[str, Any]) -> None:
    """Idempotently publish immutable JSON; reject drift instead of overwriting it."""
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"existing contract is unreadable: {path}") from exc
        if not isinstance(existing, dict) or existing != dict(payload):
            raise ValueError(f"existing contract differs; use a new campaign namespace: {path}")
        return
    write_json_atomic(path, payload)


def publish_hash_sidecar(path: Path) -> None:
    """Seal a contract's exact bytes without rewriting an existing seal."""
    if not path.is_file():
        raise ValueError(f"cannot seal missing contract: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    expected = (sha256_file(path) + "\n").encode("ascii")
    if sidecar.exists():
        if sidecar.read_bytes() != expected:
            raise ValueError(f"contract content hash differs: {path}")
        return
    sidecar.write_bytes(expected)


def validate_hash_sidecar(path: Path) -> None:
    """Require an existing exact-byte seal; never repair a completed contract."""
    if not path.is_file():
        raise ValueError(f"contract is missing: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    expected = (sha256_file(path) + "\n").encode("ascii")
    if not sidecar.is_file() or sidecar.read_bytes() != expected:
        raise ValueError(f"contract content hash is missing or changed: {path}")
