# Public candidate adaptation notice (2026-10-18): this local implementation
# adapts the Focal Sampling method of Alessio Quercia and collaborators to the
# study's deterministic, FIT-only Ultralytics classification workflow.
# Upstream: https://github.com/AlessioQuercia/sgd_biased at
# 438dbe399ca1f7f623879ece45f361d3ba906e8c (Apache-2.0).
# Relevant upstream callbacks.py credits Copyright 2024 Forschungszentrum
# Juelich GmbH and file author Alessio Quercia. See FOCAL_ADAPTATION_NOTICE.md.
"""Auditable, classification-only Focal Sampling state (KAIS 2025 / ICDM 2023).

The upstream core uses last training-batch p(true class), freezes once at the
end of zero-based at_epoch, and samples N examples with replacement thereafter.
No OOF scores, negative-only restriction, Focal Loss, or extra scoring passes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any

UPSTREAM_COMMIT = "438dbe399ca1f7f623879ece45f361d3ba906e8c"


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    """Never replace already-published sampling evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_name(path: str) -> str:
    """Handle native paths as well as Windows paths in transferred metadata."""
    return PureWindowsPath(path).name if "\\" in path else Path(path).name


def epoch_seed(seed: int, fold: int, epoch: int) -> int:
    return int(digest(["focal_sampling_v1", seed, fold, epoch])[:15], 16)


def validate_split_rows(splits: dict[str, list[dict[str, Any]]]) -> None:
    names: set[str] = set()
    groups: set[str] = set()
    for role in ("fit", "cal", "test"):
        rows = splits[role]
        local_names = [r["filename"] for r in rows]
        local_groups = {r["group_id"] for r in rows}
        if not rows or len(set(local_names)) != len(local_names):
            raise ValueError(f"empty/duplicate rows in {role}")
        if any(source_name(n) != n or n in (".", "..") for n in local_names):
            raise ValueError("filenames must be basenames")
        if names.intersection(local_names) or groups.intersection(local_groups):
            raise ValueError("fit/cal/test leakage by filename or scene group")
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"{role} must contain both binary classes")
        names.update(local_names)
        groups.update(local_groups)


class FocalState:
    """Main-process state; workers never receive scores, logs, or model objects."""

    def __init__(
        self,
        names: list[str],
        labels: list[int],
        *,
        seed: int,
        fold: int,
        at_epoch: int,
        gamma: float,
        epochs: int,
        batch: int,
        log_dir: Path | None = None,
    ) -> None:
        if not names or len(names) != len(labels) or len(set(names)) != len(names):
            raise ValueError("invalid sample identities")
        if set(labels) != {0, 1} or any(type(y) is not int for y in labels):
            raise ValueError("binary classification labels required")
        if type(at_epoch) is not int or not 0 <= at_epoch < epochs - 1:
            raise ValueError("at_epoch must leave at least one post-switch epoch")
        if not math.isfinite(gamma) or gamma <= 0 or batch < 1:
            raise ValueError("invalid gamma/batch")
        self.names, self.labels = tuple(names), tuple(labels)
        self.seed, self.fold = seed, fold
        self.at_epoch, self.gamma, self.epochs, self.batch = at_epoch, gamma, epochs, batch
        self.log_dir = log_dir
        self.scores: list[float | None] = [None] * len(names)
        self.probabilities: tuple[float, ...] | None = None
        self.frozen_sha256: str | None = None
        self.current_epoch = -1
        self.active = False
        self.indices: list[int] = []
        self.observed: list[int] = []
        self.batches = 0
        self.exposures: Counter[str] = Counter()
        self.completed_epochs: list[dict[str, Any]] = []
        if self.log_dir:
            write_new_json(
                self.log_dir / "source_index.json",
                {
                    "names": self.names,
                    "labels": self.labels,
                    "ordered_names_sha256": digest(self.names),
                },
            )

    def begin_epoch(self, epoch: int) -> list[int]:
        import torch

        if self.active or epoch != self.current_epoch + 1 or epoch >= self.epochs:
            raise ValueError("non-sequential or repeated epoch; restart in a new attempt")
        generator = torch.Generator(device="cpu").manual_seed(
            epoch_seed(self.seed, self.fold, epoch)
        )
        if epoch <= self.at_epoch:
            indices = torch.randperm(len(self.names), generator=generator).tolist()
        else:
            if self.probabilities is None or digest(self.probabilities) != self.frozen_sha256:
                raise ValueError("missing or changed frozen distribution")
            indices = torch.multinomial(
                torch.tensor(self.probabilities, dtype=torch.float64),
                len(self.names),
                replacement=True,
                generator=generator,
            ).tolist()
        self.current_epoch, self.active = epoch, True
        self.indices, self.observed, self.batches = indices, [], 0
        if self.log_dir:
            write_new_json(
                self.log_dir / f"epoch_{epoch:03d}_plan.json",
                {
                    "epoch_zero_based": epoch,
                    "sampling_seed": generator.initial_seed(),
                    "phase": "uniform_without_replacement"
                    if epoch <= self.at_epoch
                    else "focal_with_replacement",
                    "source_indices": indices,
                    "ordered_names_sha256": digest(self.names),
                    "frozen_distribution_sha256": self.frozen_sha256,
                    "expected_exposures": len(indices),
                    "expected_optimizer_steps": math.ceil(len(indices) / self.batch),
                },
            )
        return indices

    def observe(self, ids: list[int], labels: list[int], probabilities: list[float] | None) -> None:
        if not self.active:
            raise ValueError("scoring outside fit training is forbidden")
        start = len(self.observed)
        if not ids or ids != self.indices[start : start + len(ids)]:
            raise ValueError("batch IDs do not match the planned sampling sequence")
        if labels != [self.labels[i] for i in ids]:
            raise ValueError("sample ID / label mismatch")
        if self.current_epoch <= self.at_epoch:
            if probabilities is None or len(probabilities) != len(ids):
                raise ValueError("missing per-sample training probabilities")
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
                raise ValueError("invalid probability")
            for i, p in zip(ids, probabilities, strict=True):
                self.scores[i] = float(p)
            if self.log_dir:
                with (self.log_dir / "training_scores.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "epoch_zero_based": self.current_epoch,
                                "batch": self.batches,
                                "ids": ids,
                                "p_true": probabilities,
                            }
                        )
                        + "\n"
                    )
        self.observed.extend(ids)
        self.exposures.update(self.names[i] for i in ids)
        self.batches += 1

    def finish_epoch(self, optimizer_steps: int, optimizer_attempts: int) -> dict[str, Any]:
        expected_steps = math.ceil(len(self.names) / self.batch)
        if not self.active or self.observed != self.indices or self.batches != expected_steps:
            raise ValueError("incomplete/extra training batches")
        if optimizer_steps != expected_steps or optimizer_attempts != expected_steps:
            raise ValueError("optimizer update budget differs or steps were skipped")
        self.active = False
        if self.current_epoch == self.at_epoch:
            if any(p is None for p in self.scores):
                raise ValueError("cannot freeze: some fit samples were never scored")
            weights = [(1.0 - float(p)) ** self.gamma for p in self.scores]
            total = math.fsum(weights)
            if total <= 0 or not math.isfinite(total):
                raise ValueError("degenerate Focal weights; no silent uniform fallback")
            self.probabilities = tuple(w / total for w in weights)
            self.frozen_sha256 = digest(self.probabilities)
            if self.log_dir:
                write_new_json(
                    self.log_dir / "frozen_distribution.json",
                    {
                        "names": self.names,
                        "labels": self.labels,
                        "p_true": self.scores,
                        "weights": weights,
                        "probabilities": self.probabilities,
                        "sha256": self.frozen_sha256,
                        "completed_epochs": self.at_epoch + 1,
                        "first_weighted_epoch_zero_based": self.at_epoch + 1,
                        "score_source": "last_observed_training_batch_true_class_probability",
                        "upstream_commit": UPSTREAM_COMMIT,
                    },
                )
        record = {
            "epoch_zero_based": self.current_epoch,
            "sample_exposures": len(self.observed),
            "batches": self.batches,
            "optimizer_steps": optimizer_steps,
            "optimizer_attempts": optimizer_attempts,
            "per_filename_exposures": dict(Counter(self.names[i] for i in self.observed)),
            "sequence_sha256": digest(self.observed),
            "distribution_sha256": self.frozen_sha256,
        }
        self.completed_epochs.append(record)
        if self.log_dir:
            write_new_json(self.log_dir / f"epoch_{self.current_epoch:03d}_observed.json", record)
        return record

    def summary(self) -> dict[str, Any]:
        if len(self.completed_epochs) != self.epochs or self.active or self.probabilities is None:
            raise ValueError("incomplete Focal training")
        return {
            "completed_epochs": self.epochs,
            "sample_exposures": sum(self.exposures.values()),
            "optimizer_steps": sum(x["optimizer_steps"] for x in self.completed_epochs),
            "batches": sum(x["batches"] for x in self.completed_epochs),
            "per_filename_exposures": dict(sorted(self.exposures.items())),
            "distribution_sha256": self.frozen_sha256,
            "switch_after_completed_epochs": self.at_epoch + 1,
        }


class IndexedClassificationDataset:
    """Top-level, spawn-pickleable wrapper; stable source IDs independent of order."""

    def __init__(self, dataset: Any, rows: list[dict[str, Any]]) -> None:
        self.dataset = dataset
        self.names = [source_name(str(s[0])) for s in dataset.samples]
        self.labels = [int(s[1]) for s in dataset.samples]
        expected = {r["filename"]: r["label"] for r in rows}
        if len(expected) != len(rows) or len(self.names) != len(set(self.names)):
            raise ValueError("duplicate fit identity")
        if dict(zip(self.names, self.labels, strict=True)) != expected:
            raise ValueError("loaded fit data do not match the frozen partition")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = dict(self.dataset[index])
        result["source_id"] = index
        result["im_file"] = str(self.dataset.samples[index][0])
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "dataset"), name)


class EpochPlanSampler:
    """Iterating never advances epochs: prefetch cannot change sampling state."""

    def __init__(self, size: int) -> None:
        self.indices = list(range(size))

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        return iter(tuple(self.indices))


def seed_worker(worker_id: int) -> None:
    import random

    import numpy as np
    import torch

    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


class ScoringCrossEntropy:
    """Same CE mean and gradients as Ultralytics; detached scores only."""

    def __init__(self, state: FocalState) -> None:
        self.state = state

    def __call__(self, preds: Any, batch: dict[str, Any]):
        import torch
        from torch.nn import functional as F

        logits = preds[1] if isinstance(preds, (tuple, list)) else preds
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("Focal adapter supports binary classification logits only")
        loss = F.cross_entropy(logits, batch["cls"], reduction="mean")
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite loss")
        p = None
        if self.state.current_epoch <= self.state.at_epoch:
            p = (
                logits.detach()
                .softmax(dim=1)
                .gather(1, batch["cls"].view(-1, 1))
                .flatten()
                .cpu()
                .tolist()
            )
        self.state.observe(batch["source_id"].cpu().tolist(), batch["cls"].cpu().tolist(), p)
        return loss, loss.detach()
