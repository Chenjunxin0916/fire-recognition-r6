"""Plan, smoke-test, or explicitly train the independent Focal baseline."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_detection.research.focal_sampling import digest, validate_split_rows  # noqa: E402
from fire_detection.research.focal_training import load_focal_config, train_focal  # noqa: E402
from fire_detection.research.protocol import (  # noqa: E402
    build_partition,
    publish_contract,
    publish_hash_sidecar,
    verify_source,
)


def balanced_subset(rows: list[dict], each: int) -> list[dict]:
    return [r for label in (0, 1) for r in [x for x in rows if x["label"] == label][:each]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/focal_sampling/dfire.yaml"))
    parser.add_argument("--stage", choices=("plan", "smoke", "train"), default="plan")
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", help="smoke only; formal device comes from frozen base config")
    parser.add_argument("--smoke-id", default="v2", help="separate engineering run identity")
    parser.add_argument(
        "--workers", type=int, help="smoke only; CUDA default uses formal worker count"
    )
    parser.add_argument("--execute-formal", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "train" and not args.execute_formal:
        parser.error("formal training requires --execute-formal; default stage only plans")
    if (args.device or args.workers is not None) and args.stage != "smoke":
        parser.error("--device/--workers are only engineering smoke overrides")
    if not args.smoke_id or not all(
        c.isascii() and (c.isalnum() or c == "_") for c in args.smoke_id
    ):
        parser.error("smoke ID must contain ASCII letters, digits or underscores")
    if args.workers is not None and args.workers < 0:
        parser.error("workers must be nonnegative")
    config, settings = load_focal_config(ROOT / args.config, ROOT)
    folds = (
        args.folds
        if args.folds is not None
        else ([0] if args.stage == "smoke" else config["outer_folds"])
    )
    seeds = (
        args.seeds
        if args.seeds is not None
        else ([42] if args.stage == "smoke" else config["seeds"])
    )
    if len(set(folds)) != len(folds) or not set(folds) <= set(config["outer_folds"]):
        parser.error("folds must be unique members of the frozen protocol")
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(config["seeds"]):
        parser.error("seeds must be unique members of the frozen protocol")
    records, source = verify_source(config)
    # Record runtime dependencies per attempt, outside the portable dataset identity.
    portable_source = {k: v for k, v in source.items() if k != "dependency_snapshot"}
    if args.stage == "smoke":
        config = copy.deepcopy(config)
        config["campaign"] += "_smoke_" + args.smoke_id
        config["engineering_only"] = True
        device = args.device or "cpu"
        workers = (
            args.workers
            if args.workers is not None
            else (0 if device == "cpu" else config["training"]["workers"])
        )
        config["training"].update(device=device, workers=workers)
        config["classifier"].update(epochs=3, batch=4, imgsz=64)
        settings = {**settings, "at_epoch": 0}
    campaign = ROOT / "runs" / "direction_a" / config["campaign"]
    publish_contract(campaign / "source.json", portable_source)
    source_hash = digest(portable_source)
    publish_contract(
        campaign / "plan.json",
        {
            "source_sha256": source_hash,
            "focal_sampling": settings,
            "base_config_sha256": config["_focal_base_sha256"],
            "classifier": config["classifier"],
            "training": config["training"],
            "engineering_only": config["engineering_only"],
            "formal_folds": config["outer_folds"],
            "formal_seeds": config["seeds"],
            "checkpoint_policy": "fixed_epochs_last",
        },
    )
    for fold in folds:
        for seed in seeds:
            partition = build_partition(records, fold, seed)
            if args.stage == "smoke":
                partition["splits"]["fit"] = balanced_subset(partition["splits"]["fit"], 4)
                partition["splits"]["cal"] = balanced_subset(partition["splits"]["cal"], 2)
                partition["engineering_only"] = True
            validate_split_rows(partition["splits"])
            cell = campaign / f"outer_{fold}" / f"seed_{seed}"
            publish_contract(cell / "partition.json", partition)
            publish_hash_sidecar(cell / "partition.json")
            print(
                json.dumps(
                    {
                        "cell": str(cell),
                        "stage": args.stage,
                        "counts": {k: len(v) for k, v in partition["splits"].items()},
                        "focal": settings,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.stage != "plan":
                result = train_focal(
                    config,
                    settings,
                    cell / "focal",
                    splits=partition["splits"],
                    seed=seed,
                    fold=fold,
                    source_sha256=source_hash,
                )
                print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
