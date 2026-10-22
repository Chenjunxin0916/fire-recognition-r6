"""Three-arm adapter for frozen cal-only thresholds and paired test summaries.

No training occurs here. Synthetic providers are restricted to engineering paths.
Each calibration/evaluation call maps to one existing DAG node (two views).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from fire_detection.metrics.thresholds import search_thresholds
from fire_detection.models.features import classifier_positive_probability
from fire_detection.models.offline import sha256_file
from fire_detection.research.protocol import (
    digest_object,
    publish_contract,
    publish_hash_sidecar,
    validate_hash_sidecar,
)
from fire_detection.research.statistics import (
    binary_metrics,
    paired_cluster_bootstrap,
    summarize_predictions,
    validate_predictions,
)

METHODS = ("standard", "hn", "focal")
IDENTITIES = ["filename", "label", "group_id", "fold"]
VIEWS = {"original": "", "tta": "_tta"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _destination(path, engineering_only):
    path = Path(path).resolve()
    if engineering_only and not any(
        value in str(path).lower() for value in ("engineering", "acceptance", "pytest", "tmp")
    ):
        raise ValueError("Engineering predictions require an isolated engineering/acceptance path")
    if engineering_only and any(
        value in str(path).lower() for value in ("/paper/results", "\\paper\\results")
    ):
        raise ValueError("Synthetic predictions cannot be written into paper results")
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_partition(partition, *, engineering_only=False):
    """Validate sample IDs, labels, groups and explicit split/fold membership."""
    if partition["calibration_fold"] != (partition["outer_fold"] + 1) % 5:
        raise ValueError("Frozen calibration-fold rotation differs")
    if not engineering_only and set(partition["fit_folds"]) != (
        set(range(5)) - {partition["outer_fold"], partition["calibration_fold"]}
    ):
        raise ValueError("Frozen fit folds differ")
    parts = {}
    for split in ("fit", "cal", "test"):
        part = pd.DataFrame(partition["splits"][split])
        if not set(IDENTITIES) <= set(part) or part[IDENTITIES].isna().any().any():
            raise ValueError("Partition identities missing")
        if part.empty or part.filename.duplicated().any():
            raise ValueError("Partition empty or duplicate filename")
        if not part.label.isin([0, 1]).all():
            raise ValueError("Partition labels must be binary")
        for col in ("filename", "group_id"):
            if part[col].astype(str).str.strip().eq("").any():
                raise ValueError("Empty partition identity")
        expected_folds = (
            partition["fit_folds"]
            if split == "fit"
            else [partition["calibration_fold"] if split == "cal" else partition["outer_fold"]]
        )
        if not part.fold.isin(expected_folds).all():
            raise ValueError("Partition split/fold identity differs")
        if not engineering_only and len(part) != (660 if split == "fit" else 220):
            raise ValueError("Formal partition cardinality differs")
        parts[split] = part
    for left, right in (("fit", "cal"), ("fit", "test"), ("cal", "test")):
        for col in ("filename", "group_id"):
            if set(parts[left][col]) & set(parts[right][col]):
                raise ValueError(f"Partition {col} leakage: {left}/{right}")
    return parts


def positive_probability(names, probabilities):
    """Only two named classes 0 and 1 are allowed; positive is the name '1'."""
    names = dict(names) if isinstance(names, Mapping) else dict(enumerate(names))
    if set(names) != {0, 1} or set(map(str, names.values())) != {"0", "1"}:
        raise ValueError("Expected exact binary checkpoint class names 0 and 1")
    if len(probabilities) != 2:
        raise ValueError("Expected two probabilities")
    return classifier_positive_probability(names=names, probabilities=probabilities)


def resolve_artifact(config, partition, model_root, method):
    """Read actual completion metadata and validate checkpoint, config and fit/cal IDs."""
    from fire_detection.research.focal_audit import audit_attempt
    from fire_detection.research.training import resolve_complete_training_attempt

    root = Path(model_root)
    if method not in (*METHODS, "random_negative"):
        raise ValueError("Unknown method")
    if method == "focal":
        pointers = list(root.glob("completed_*.json"))
        if len(pointers) != 1:
            raise ValueError("Focal requires one unambiguous completed attempt")
        attempt = (root / _read(pointers[0])["attempt"]).resolve()
        if not attempt.is_relative_to(root.resolve()):
            raise ValueError("Focal attempt escapes model root")
        audit_attempt(attempt)
        contract = _read(root / "focal_contract.json")
        manifest_path = attempt / "focal_manifest.json"
        if contract["splits"] != partition["splits"] or contract["fold"] != partition["outer_fold"]:
            raise ValueError("Focal partition differs")
        if contract["settings"] != {"enabled": True, "at_epoch": 10, "gamma": 0.5}:
            raise ValueError("Focal sampling definition differs")
        training, classifier = contract["training"], contract["classifier"]
    else:
        attempt, manifest = resolve_complete_training_attempt(root)
        manifest_path = attempt / "research_manifest.json"
        contract = manifest["contract"]
        if (
            contract["train"] != partition["splits"]["fit"]
            or contract["validation"] != partition["splits"]["cal"]
        ):
            raise ValueError("Training fit/cal partition differs")
        expected_arm = "random_negative" if method == "random_negative" else "hn"
        if contract["task"] != "classifier" or contract["arm"] != expected_arm:
            raise ValueError("Wrong classifier training arm")
        if method == "standard" and contract["hard_names"]:
            raise ValueError("Standard cannot use replay samples")
        hyper = contract["hyperparameters"]
        training = {k: hyper[k] for k in config["training"]}
        classifier = {k: hyper[k] for k in config["classifier"]}
    manifest = _read(manifest_path)
    if (
        contract["seed"] != partition["seed"]
        or training != config["training"]
        or classifier != config["classifier"]
    ):
        raise ValueError("Training configuration/seed differs")
    if contract["engineering_only"] != config["engineering_only"]:
        raise ValueError("Training evidence class differs")
    checkpoint = attempt / "weights" / "last.pt"
    if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metadata": str(manifest_path.resolve()),
        "metadata_sha256": sha256_file(manifest_path),
        "contract_sha256": manifest["contract_sha256"],
    }


def strict_predictor_class(*, engineering_cpu=False):
    """Build a testable predictor with explicit precision at the backend boundary."""
    import torch
    from ultralytics.models.yolo.classify import ClassificationPredictor
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.utils.torch_utils import select_device

    from fire_detection.research.strict_release_guard import check_precision

    class StrictPredictor(ClassificationPredictor):
        def setup_model(self, model, verbose=True):
            self.model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device, verbose=False),
                fp16=False,
                fuse=False,
                verbose=False,
            )
            self.device = self.model.device
            self.args.half = False
            self.args.quantize = None
            required_device = "cpu" if engineering_cpu else "cuda"
            if self.device.type != required_device or self.model.fp16:
                raise ValueError("Strict inference requires CUDA FP32")
            if any(p.dtype != torch.float32 for p in self.model.model.parameters()):
                raise ValueError("Non-FP32 inference parameters")
            self.model.eval()

        def preprocess(self, images):
            check_precision(torch)
            result = super().preprocess(images)
            required_device = "cpu" if engineering_cpu else "cuda"
            if result.dtype != torch.float32 or result.device.type != required_device:
                raise ValueError("Non-FP32 inference inputs")
            return result

        def inference(self, im, *args, **kwargs):
            check_precision(torch)
            with torch.autocast(device_type=self.device.type, enabled=False):
                return super().inference(im, *args, **kwargs)

    return StrictPredictor


from .immutable_inputs import guarded_inputs, binding, verified_read


@guarded_inputs
def strict_predict_classifier(config, checkpoint, rows, split, *, tta=True, engineering_cpu=False):
    """Real original/flip inference; pinned backend fusion and AMP are disabled."""
    import torch
    from PIL import Image, ImageOps

    from fire_detection.research.strict_release_guard import check_precision, set_precision

    if engineering_cpu and not (config["engineering_only"] is True and config["training"]["device"] == "cpu" and split == "engineering_cpu_probe"):
        raise PermissionError("CPU inference is restricted to explicit engineering probes")
    set_precision(torch)
    predictor = strict_predictor_class(engineering_cpu=engineering_cpu)(
        overrides={
            "model": str(checkpoint),
            "device": config["training"]["device"],
            "imgsz": config["classifier"]["imgsz"],
            "half": False,
            "quantize": None,
            "compile": False,
            "augment": False,
            "save": False,
            "verbose": False,
        }
    )
    image_root = Path(config["_root"]) / config["data"]["images"]
    _, _, snapshot = binding(config)
    expected_images = {r['filename']:r for r in snapshot['images']}
    output = []
    batch = config["inference"]["batch"]
    for offset in range(0, len(rows), batch):
        chunk = rows[offset : offset + batch]
        images, flips = [], []
        try:
            for row in chunk:
                expected = expected_images[row['filename']]
                if any(row[key] != expected[key] for key in IDENTITIES):
                    raise ValueError('Prediction row differs from frozen input identity')
                path = image_root / row['filename']
                with verified_read(path, expected['sha256']):
                    with Image.open(path) as image:
                        images.append(ImageOps.exif_transpose(image).convert("RGB"))
            flips = [ImageOps.mirror(image) for image in images] if tta else []
            original = list(predictor(images))
            flipped = list(predictor(flips)) if tta else original
            if len(original) != len(chunk) or len(flipped) != len(chunk):
                raise ValueError("Prediction cardinality mismatch")
            for row, first, second in zip(chunk, original, flipped, strict=True):
                p = positive_probability(first.names, first.probs.data.detach().cpu().tolist())
                q = positive_probability(second.names, second.probs.data.detach().cpu().tolist())
                output.append({**row, "original": p, "flip": q, "tta": (p + q) / 2})
        finally:
            for image in images + flips:
                image.close()
    check_precision(torch)
    return pd.DataFrame(output)


def _raw(frame, rows):
    expected = pd.DataFrame(rows)[IDENTITIES].reset_index(drop=True)
    if not set(IDENTITIES + ["original", "flip", "tta"]) <= set(frame):
        raise ValueError("Raw probability schema incomplete")
    actual = frame[IDENTITIES].reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Raw sample identity/order mismatch")
    probabilities = frame[["original", "flip", "tta"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid probabilities")
    if not np.allclose(frame.tta, (frame.original + frame.flip) / 2, atol=1e-12, rtol=0):
        raise ValueError("TTA must be original/flip arithmetic mean")
    return frame


def _context(config, partition, model_root, method, engineering_only, predictor):
    validate_partition(partition, engineering_only=engineering_only)
    if not engineering_only and config["engineering_only"]:
        raise ValueError("Engineering training cannot be promoted to formal evidence")
    if method not in (*METHODS, "random_negative"):
        raise ValueError("Unknown method")
    if predictor is not None and not engineering_only:
        raise ValueError("Injected predictions are engineering-only")
    # Tests can use a fake model root containing a sealed synthetic metadata pair.
    if predictor is not None:
        model_root = Path(model_root)
        artifact = {
            "checkpoint": str(model_root / "last.pt"),
            "checkpoint_sha256": sha256_file(model_root / "last.pt"),
            "metadata_sha256": sha256_file(model_root / "metadata.json"),
            "synthetic": True,
        }
    else:
        artifact = resolve_artifact(config, partition, model_root, method)
    return {
        "schema_version": 1,
        "method": method,
        "outer_fold": partition["outer_fold"],
        "seed": partition["seed"],
        "partition_sha256": digest_object(partition),
        "config_sha256": digest_object({k: v for k, v in config.items() if not k.startswith("_")}),
        "artifact": artifact,
        "engineering_only": engineering_only,
        "positive_class": "1",
        "score": "P(class_name=1)",
        "threshold_grid": config["evaluation"]["threshold"],
        "inference_policy": {"half": False, "quantize": None, "fuse": False, "tf32": False},
    }


def _save_csv(path, frame):
    if path.exists():
        raise ValueError(f"Refusing overwrite: {path}")
    temp = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def calibrate_method(
    config, partition, model_root, output_dir, method, *, predictor=None, engineering_only=False
):
    """Calibrate original and TTA for a single method using designated cal only."""
    destination = _destination(output_dir, engineering_only)
    context = _context(config, partition, model_root, method, engineering_only, predictor)
    target = destination / "calibration.json"
    raw_path = destination / "cal_predictions.csv"
    if target.exists():
        validate_hash_sidecar(target)
        payload = _read(target)
        if payload["context"] != context or payload["raw_sha256"] != sha256_file(raw_path):
            raise ValueError("Calibration input/output drift")
        return [target, target.with_name(target.name + ".sha256"), raw_path]
    rows = partition["splits"]["cal"]
    frame = _raw(
        (predictor or strict_predict_classifier)(
            config, context["artifact"]["checkpoint"], rows, "cal"
        ),
        rows,
    )
    grid = config["evaluation"]["threshold"]
    if grid != {"start": 0.01, "end": 0.99, "step": 0.005}:
        raise ValueError("Frozen threshold grid differs")
    thresholds = {
        view: search_thresholds(frame.label.tolist(), frame[view].tolist(), **grid).best.threshold
        for view in VIEWS
    }
    _save_csv(raw_path, frame)
    publish_contract(
        target,
        {
            "context": context,
            "thresholds": thresholds,
            "raw_sha256": sha256_file(raw_path),
            "fit_split": "cal",
        },
    )
    publish_hash_sidecar(target)
    return [target, target.with_name(target.name + ".sha256"), raw_path]


def evaluate_method(
    config,
    partition,
    model_root,
    output_dir,
    method,
    *,
    calibration_paths,
    required_methods=None,
    predictor=None,
    engineering_only=False,
):
    """Require the release's frozen calibration artifacts before test inference."""
    context = _context(config, partition, model_root, method, engineering_only, predictor)
    if set(calibration_paths) != set(METHODS if required_methods is None else required_methods):
        raise ValueError("Required calibrations must be frozen before test")
    cal_hashes = {}
    for name, path in calibration_paths.items():
        path = Path(path)
        validate_hash_sidecar(path)
        payload = _read(path)
        peer = payload["context"]
        for key in ("partition_sha256", "config_sha256", "engineering_only", "seed", "outer_fold"):
            if peer[key] != context[key]:
                raise ValueError("Cross-method calibration pairing/config differs")
        if peer["method"] != name or payload["fit_split"] != "cal":
            raise ValueError("Calibration method/split differs")
        if payload["raw_sha256"] != sha256_file(path.parent / "cal_predictions.csv"):
            raise ValueError("Calibration raw predictions changed")
        cal_hashes[name] = sha256_file(path)
    own = _read(calibration_paths[method])
    if own["context"] != context:
        raise ValueError("Checkpoint or metadata changed since calibration")
    destination = _destination(output_dir, engineering_only)
    completion = destination / "evaluation.json"
    predictions = destination / "predictions.csv"
    metrics_path = destination / "metrics.json"
    raw_path = destination / "test_raw.csv"
    if completion.exists():
        validate_hash_sidecar(completion)
        existing = _read(completion)
        if existing["context"] != context or existing["calibrations_sha256"] != cal_hashes:
            raise ValueError("Evaluation input drift")
        for path in (predictions, metrics_path, raw_path):
            if existing["outputs"][path.name] != sha256_file(path):
                raise ValueError("Evaluation output drift")
        return [
            predictions,
            metrics_path,
            raw_path,
            completion,
            completion.with_name(completion.name + ".sha256"),
        ]
    rows = partition["splits"]["test"]
    raw = _raw(
        (predictor or strict_predict_classifier)(
            config, context["artifact"]["checkpoint"], rows, "test"
        ),
        rows,
    )
    records, metrics = [], {}
    for view, suffix in VIEWS.items():
        frame = raw[IDENTITIES].copy()
        frame["score"] = raw[view]
        frame["threshold"] = own["thresholds"][view]
        frame["y_pred"] = (frame.score >= frame.threshold).astype(int)
        frame["method"] = f"cls_{method}{suffix}"
        frame["experiment_id"] = config["campaign"]
        frame["outer_fold"], frame["seed"] = partition["outer_fold"], partition["seed"]
        frame["split"], frame["condition"] = "test", "clean"
        frame["evidence_status"] = (
            "engineering_only" if engineering_only else "internal_revalidation"
        )
        frame["checkpoint_sha256"] = context["artifact"]["checkpoint_sha256"]
        frame["calibration_sha256"] = cal_hashes[method]
        frame["metadata_sha256"] = context["artifact"]["metadata_sha256"]
        frame["partition_sha256"] = context["partition_sha256"]
        record = binary_metrics(
            frame.label.to_numpy(), frame.score.to_numpy(), frame.y_pred.to_numpy()
        )
        record["fnr"] = record["fn"] / record["n_positive"] if record["n_positive"] else None
        record["confusion_matrix"] = [[record["tn"], record["fp"]], [record["fn"], record["tp"]]]
        metrics[frame.method.iloc[0]] = record
        records.append(frame)
    combined = validate_predictions(pd.concat(records, ignore_index=True))
    _save_csv(raw_path, raw)
    _save_csv(predictions, combined)
    publish_contract(metrics_path, metrics)
    publish_contract(
        completion,
        {
            "context": context,
            "calibrations_sha256": cal_hashes,
            "outputs": {p.name: sha256_file(p) for p in (predictions, metrics_path, raw_path)},
        },
    )
    publish_hash_sidecar(completion)
    return [
        predictions,
        metrics_path,
        raw_path,
        completion,
        completion.with_name(completion.name + ".sha256"),
    ]


def summarize_three_arm(predictions, output_dir, spec, *, engineering_only=False):
    """Pool five folds within seed; paired scene-bootstrap is not a seed/fold t-test."""
    data = validate_predictions(predictions)
    expected = {f"cls_{method}{suffix}" for method in METHODS for suffix in VIEWS.values()}
    if set(data.method) != expected or not data.split.eq("test").all():
        raise ValueError("Complete six-method test matrix required")
    status = "engineering_only" if engineering_only else "internal_revalidation"
    if set(data.evidence_status) != {status} or data.experiment_id.nunique() != 1:
        raise ValueError("Mixed campaign/evidence forbidden")
    if not engineering_only:
        for _, part in data.groupby(["method", "seed"]):
            if set(part.outer_fold) != set(range(5)) or len(part) != 1100:
                raise ValueError("Formal fold/test matrix incomplete")
        if set(data.seed) != {42, 2026, 3407}:
            raise ValueError("Formal seed matrix incomplete")
    if spec["bootstrap"]["iterations"] != 10000 or spec["bootstrap"]["seed"] != 20260914:
        raise ValueError("Frozen bootstrap configuration differs")
    seed, fold, summary = summarize_predictions(data)
    for frame in (seed, fold):
        frame["fnr"] = frame.fn / frame.n_positive.replace(0, np.nan)
    for index, row in summary.iterrows():
        values = seed.loc[seed.method.eq(row.method), "fnr"]
        summary.loc[index, "fnr_mean"] = values.mean() if values.notna().all() else np.nan
        summary.loc[index, "fnr_std"] = values.std(ddof=1) if values.notna().all() else np.nan
    paired = []
    for suffix in VIEWS.values():
        subset = data.loc[data.method.isin([f"cls_{m}{suffix}" for m in METHODS])]
        for reference, wanted in (("standard", ["hn", "focal"]), ("focal", ["hn"])):
            result = paired_cluster_bootstrap(
                subset, f"cls_{reference}{suffix}", iterations=10000, analysis_seed=20260914
            )
            paired.append(result.loc[result.method.isin([f"cls_{m}{suffix}" for m in wanted])])
    destination = _destination(output_dir, engineering_only)
    paths = []
    for name, frame in (
        ("predictions", data),
        ("per_seed", seed),
        ("per_fold", fold),
        ("summary", summary),
        ("paired_ci", pd.concat(paired, ignore_index=True)),
    ):
        path = destination / f"{name}.csv"
        _save_csv(path, frame)
        paths.append(path)
    manifest = destination / "summary_manifest.json"
    publish_contract(
        manifest,
        {
            "engineering_only": engineering_only,
            "spec_sha256": digest_object(spec),
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 20260914,
            "unit": "scene groups within outer folds; same draws across methods and seeds",
            "scope": "conditional on frozen models and thresholds; not training uncertainty",
            "inference": "descriptive paired effects and percentile CIs; no significance claim",
            "outputs": {p.name: sha256_file(p) for p in paths},
        },
    )
    paths.append(manifest)
    return paths
