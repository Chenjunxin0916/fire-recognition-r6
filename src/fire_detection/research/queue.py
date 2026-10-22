"""Immutable, fail-fast, single-process queue for Direction A formal campaigns."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fire_detection.models.offline import sha256_file
from fire_detection.research.pipeline import (
    IncompleteEvaluation,
    plan_campaign,
    training_task_names,
    validate_evaluation_cell,
    validate_training_task,
)
from fire_detection.research.protocol import (
    publish_contract,
    publish_hash_sidecar,
    validate_formal_training_authorization,
    validate_hash_sidecar,
)
from fire_detection.research.training import IncompleteTrainingTask


def build_queue(config: Mapping[str, Any], campaign: Path) -> tuple[Path, dict[str, Any]]:
    """Create the immutable configured training/evaluation queue."""
    items: list[dict[str, Any]] = []
    config_path = str(Path(config["_config_path"]).resolve())
    entrypoint = str((Path(config["_root"]) / "scripts" / "run_direction_a.py").resolve())
    for outer in config["outer_folds"]:
        for seed in config["seeds"]:
            cell = campaign / f"outer_{outer}" / f"seed_{seed}"
            partition = json.loads((cell / "partition.json").read_text(encoding="utf-8"))
            for task_name in training_task_names(config, partition):
                identity = f"outer_{outer}__seed_{seed}__{task_name}"
                items.append(
                    {
                        "id": identity,
                        "kind": "training",
                        "outer_fold": outer,
                        "seed": seed,
                        "task": task_name,
                        "command": [
                            sys.executable,
                            entrypoint,
                            "--config",
                            config_path,
                            "--stage",
                            "task",
                            "--outer-folds",
                            str(outer),
                            "--seeds",
                            str(seed),
                            "--task-name",
                            task_name,
                        ],
                    }
                )
            items.append(
                {
                    "id": f"outer_{outer}__seed_{seed}__evaluate",
                    "kind": "evaluation",
                    "outer_fold": outer,
                    "seed": seed,
                    "task": "evaluate",
                    "command": [
                        sys.executable,
                        entrypoint,
                        "--config",
                        config_path,
                        "--stage",
                        "evaluate",
                        "--outer-folds",
                        str(outer),
                        "--seeds",
                        str(seed),
                    ],
                }
            )
    payload = {
        "schema_version": 1,
        "campaign": config["campaign"],
        "campaign_contract_sha256": sha256_file(campaign / "campaign_contract.json"),
        "config_sha256": config["_config_sha256"],
        "execution": "single_gpu_serial_fail_fast",
        "training_task_count": sum(item["kind"] == "training" for item in items),
        "evaluation_task_count": sum(item["kind"] == "evaluation" for item in items),
        "items": items,
    }
    path = campaign / "queue" / "queue.json"
    publish_contract(path, payload)
    publish_hash_sidecar(path)
    return path, payload


def validate_queue(config: Mapping[str, Any], queue_path: Path) -> dict[str, Any]:
    """Dry-run all identities, commands, hashes, ordering, and task counts."""
    validate_hash_sidecar(queue_path)
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    campaign, _ = plan_campaign(
        config, outer_folds=config["outer_folds"], seeds=config["seeds"]
    )
    if payload.get("campaign") != config["campaign"]:
        raise ValueError("queue campaign differs")
    if payload.get("config_sha256") != config["_config_sha256"]:
        raise ValueError("queue config hash differs")
    if payload.get("campaign_contract_sha256") != sha256_file(
        campaign / "campaign_contract.json"
    ):
        raise ValueError("queue campaign contract hash differs")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("queue items must be a list")
    identities = [item.get("id") for item in items]
    if len(set(identities)) != len(identities):
        raise ValueError("queue contains duplicate task identities")
    expected_training: list[tuple[int, int, str]] = []
    expected_evaluation: list[tuple[int, int, str]] = []
    for outer in config["outer_folds"]:
        for seed in config["seeds"]:
            partition = json.loads(
                (campaign / f"outer_{outer}" / f"seed_{seed}" / "partition.json").read_text(
                    encoding="utf-8"
                )
            )
            expected_training.extend(
                (outer, seed, task) for task in training_task_names(config, partition)
            )
            expected_evaluation.append((outer, seed, "evaluate"))
    actual_training = [
        (item.get("outer_fold"), item.get("seed"), item.get("task"))
        for item in items
        if item.get("kind") == "training"
    ]
    actual_evaluation = [
        (item.get("outer_fold"), item.get("seed"), item.get("task"))
        for item in items
        if item.get("kind") == "evaluation"
    ]
    if actual_training != expected_training:
        raise ValueError("queue training-task matrix differs")
    if actual_evaluation != expected_evaluation:
        raise ValueError("queue evaluation-task matrix differs")
    if payload.get("training_task_count") != len(expected_training):
        raise ValueError("queue training-task count differs")
    if payload.get("evaluation_task_count") != len(expected_evaluation):
        raise ValueError("queue evaluation-task count differs")
    if len(items) != len(expected_training) + len(expected_evaluation):
        raise ValueError("queue contains undeclared tasks")
    for item in items:
        command = item.get("command")
        if not isinstance(command, list) or not command or command[0] != sys.executable:
            raise ValueError(f"queue command interpreter differs: {item.get('id')}")
    return payload


def _execution_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1_000_000):
        path = root / f"attempt_{number:03d}"
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise RuntimeError("queue execution attempt space exhausted")


def run_queue(config: Mapping[str, Any], queue_path: Path) -> int:
    """Run at most one child at a time and stop at the first nonzero exit."""
    payload = validate_queue(config, queue_path)
    campaign = queue_path.parents[1]
    validate_formal_training_authorization(config, campaign)
    environment = os.environ.copy()
    environment.update(
        YOLO_OFFLINE="true", HF_HUB_OFFLINE="1", WANDB_DISABLED="true", PYTHONUTF8="1"
    )
    for item in payload["items"]:
        # Revalidate frozen inputs before every possible process launch.
        plan_campaign(config, outer_folds=config["outer_folds"], seeds=config["seeds"])
        cell = campaign / f"outer_{item['outer_fold']}" / f"seed_{item['seed']}"
        if item["kind"] == "training":
            try:
                validate_training_task(config, cell, item["task"])
            except IncompleteTrainingTask:
                pass
            else:
                continue
        elif item["kind"] == "evaluation":
            try:
                validate_evaluation_cell(config, cell)
            except IncompleteEvaluation:
                pass
            else:
                continue
        else:
            raise ValueError(f"unknown queue item kind: {item.get('id')}")
        attempt = _execution_attempt(campaign / "queue" / "executions" / item["id"])
        command_path = attempt / "command.json"
        publish_contract(
            command_path,
            {"command": item["command"], "queue_sha256": sha256_file(queue_path)},
        )
        started = time.time()
        with (attempt / "stdout.log").open("wb") as stdout, (attempt / "stderr.log").open(
            "wb"
        ) as stderr:
            process = subprocess.run(
                item["command"], cwd=config["_root"], env=environment, stdout=stdout, stderr=stderr
            )
        record = {
            "schema_version": 1,
            "command_sha256": sha256_file(command_path),
            "stdout_sha256": sha256_file(attempt / "stdout.log"),
            "stderr_sha256": sha256_file(attempt / "stderr.log"),
            "exit_code": process.returncode,
            "elapsed_seconds": time.time() - started,
        }
        publish_contract(attempt / "exit.json", record)
        if process.returncode != 0:
            return process.returncode
    return 0

