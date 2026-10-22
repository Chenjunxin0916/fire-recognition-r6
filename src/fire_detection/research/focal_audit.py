"""Recheck completed Focal evidence after transfer, without running training."""

from __future__ import annotations

import json
from pathlib import Path

from fire_detection.models.offline import sha256_file
from fire_detection.research.focal_sampling import FocalState, digest, source_name


def audit_attempt(attempt: Path) -> dict:
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    run = attempt.parent.parent
    contract = read(run / "focal_contract.json")
    manifest = read(attempt / "focal_manifest.json")
    pointer = read(run / f"completed_{attempt.name}.json")
    if pointer != {
        "attempt": f"attempts/{attempt.name}",
        "manifest_sha256": sha256_file(attempt / "focal_manifest.json"),
    }:
        raise ValueError("completion pointer/manifest drift")
    if manifest["status"] != "complete" or manifest["contract_sha256"] != digest(contract):
        raise ValueError("incomplete or changed contract")
    for file, key in (("weights/last.pt", "checkpoint_sha256"), ("args.yaml", "args_sha256")):
        if sha256_file(attempt / file) != manifest[key]:
            raise ValueError(f"saved artifact drift: {file}")
    sampling = attempt / "sampling"
    hashes = manifest["sampling_files_sha256"]
    if set(hashes) != {p.name for p in sampling.iterdir() if p.is_file()}:
        raise ValueError("missing/extra sampling evidence")
    for name, expected in hashes.items():
        if source_name(name) != name or sha256_file(sampling / name) != expected:
            raise ValueError("sampling evidence hash mismatch")
    index = read(sampling / "source_index.json")
    expected_rows = {r["filename"]: r["label"] for r in contract["splits"]["fit"]}
    if dict(zip(index["names"], index["labels"], strict=True)) != expected_rows:
        raise ValueError("source index differs from fit split")
    settings = contract["settings"]
    state = FocalState(
        index["names"],
        index["labels"],
        seed=contract["seed"],
        fold=contract["fold"],
        at_epoch=settings["at_epoch"],
        gamma=settings["gamma"],
        epochs=contract["classifier"]["epochs"],
        batch=contract["classifier"]["batch"],
    )
    score_records = [
        json.loads(line) for line in (sampling / "training_scores.jsonl").read_text().splitlines()
    ]
    cursor = 0
    for epoch in range(state.epochs):
        planned = read(sampling / f"epoch_{epoch:03d}_plan.json")
        ids = state.begin_epoch(epoch)
        if planned["source_indices"] != ids:
            raise ValueError("sampling sequence is not seed-reproducible")
        for batch_id, start in enumerate(range(0, len(ids), state.batch)):
            selected = ids[start : start + state.batch]
            p = None
            if epoch <= state.at_epoch:
                record = score_records[cursor]
                cursor += 1
                if (record["epoch_zero_based"], record["batch"], record["ids"]) != (
                    epoch,
                    batch_id,
                    selected,
                ):
                    raise ValueError("score provenance/order differs")
                p = record["p_true"]
            state.observe(selected, [state.labels[i] for i in selected], p)
        observed = read(sampling / f"epoch_{epoch:03d}_observed.json")
        recreated = state.finish_epoch(observed["optimizer_steps"], observed["optimizer_attempts"])
        if observed != recreated:
            raise ValueError("observed ledger differs from reproduced ledger")
    if cursor != len(score_records):
        raise ValueError("unexpected scoring after freeze")
    frozen = read(sampling / "frozen_distribution.json")
    if (
        frozen["probabilities"] != list(state.probabilities)
        or frozen["sha256"] != state.frozen_sha256
    ):
        raise ValueError("frozen probabilities differ from training score reconstruction")
    if manifest["audit"] != state.summary():
        raise ValueError("aggregate budget differs from epoch evidence")
    for key, expected in contract["expected_budget"].items():
        if manifest["audit"][key] != expected:
            raise ValueError("aggregate budget differs from contract")
    return {
        "status": "passed",
        "attempt": str(attempt),
        "engineering_only": contract["engineering_only"],
        "audit": state.summary(),
    }
