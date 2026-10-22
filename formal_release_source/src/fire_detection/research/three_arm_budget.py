"""Fail-closed HN budget adapter; no changes to replay or mining policy.

The identity-only CPU probe executes the installed replay dataset and loader.
It is engineering evidence, never a model run or a measured GPU cost.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace


class BudgetBlocked(RuntimeError):
    """Unknown H, invalid identity, or a forbidden training budget."""


class IdentityDataset:
    """No image I/O, augmentation, model, gradients, or CUDA tensors."""

    def __init__(self, rows):
        self.samples = [(str(r["filename"]), int(r["label"])) for r in rows]
        self.base = SimpleNamespace(classes=["0", "1"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {"source_index": index, "cls": self.samples[index][1]}


def probe_replay_loader(fit_rows, hard_names, *, seed=42, batch=64):
    """Enumerate actual Ultralytics sampler/loader and existing ReplayDataset."""
    from ultralytics.data.build import build_dataloader

    from fire_detection.research.training import ReplayDataset

    dataset = ReplayDataset(IdentityDataset(fit_rows), arm="hn", hard_names=hard_names, seed=seed)
    loader = build_dataloader(dataset, batch=batch, workers=0, rank=-1, pin_memory=False)
    sizes, counts = [], Counter()
    for item in loader:
        sizes.append(len(item["im_file"]))
        counts.update(Path(name).name for name in item["im_file"])
    expected = Counter(str(row["filename"]) for row in fit_rows)
    expected.update(hard_names)
    if counts != expected or len(sizes) != len(loader):
        raise BudgetBlocked("Actual loader exposure identities differ from unchanged N+K replay")
    return {
        "dataset_size": len(dataset),
        "batches_per_epoch": len(loader),
        "batch_sizes": sizes,
        "exposures_per_epoch": sum(sizes),
        "per_filename_exposures_per_epoch": dict(sorted(counts.items())),
        "sampler": type(loader.sampler).__name__,
        "loader": type(loader).__name__,
        "drop_last": loader.drop_last,
        "probe_kind": "IDENTITY_ONLY_CPU_NO_TRAINING",
    }


def require_hn_budget(
    fit_rows,
    hard_names,
    *,
    seed,
    epochs=40,
    batch=64,
    nbs=64,
    warmup_epochs=0.0,
    compile=False,
    max_updates=None,
    policy="B_FULL_40_EPOCHS",
):
    """Called after selection AND before HN training; None is never K=0."""
    if hard_names is None:
        raise BudgetBlocked("Actual K is unknown; formal HN task remains BLOCKED")
    if policy != "B_FULL_40_EPOCHS":
        raise BudgetBlocked("Explicit A/B policy required")
    admitted_max = epochs * ((len(fit_rows) + len({r["filename"] for r in fit_rows if r["label"] == 0}) + batch - 1) // batch)
    if max_updates is not None and max_updates != admitted_max:
        raise BudgetBlocked("Max updates cannot override the selected policy")
    max_updates = admitted_max
    if len(fit_rows) != 660:
        raise BudgetBlocked("Repaired split requires exactly 660 unique FIT images")
    if (epochs, batch, nbs, warmup_epochs, compile) != (40, 64, 64, 0.0, False):
        raise BudgetBlocked(
            "Frozen 40 epoch / batch64 / nbs64 / warmup0 / compileFalse / B policy drift"
        )
    names = [str(r["filename"]) for r in fit_rows]
    if not names or len(set(names)) != len(names):
        raise BudgetBlocked("Fit sample identities are empty or duplicated")
    if any(
        r["label"] not in (0, 1) or Path(str(r["filename"])).name != r["filename"] for r in fit_rows
    ):
        raise BudgetBlocked("Invalid fit labels or filenames")
    if not isinstance(hard_names, (list, tuple)) or len(set(hard_names)) != len(hard_names):
        raise BudgetBlocked("Hard-negative identities must be a unique list")
    negatives = {r["filename"] for r in fit_rows if r["label"] == 0}
    if not set(hard_names) <= negatives:
        raise BudgetBlocked("H must contain only actual fit negatives")
    probe = probe_replay_loader(fit_rows, hard_names, seed=seed, batch=batch)
    updates = epochs * probe["batches_per_epoch"]
    if updates > max_updates:
        raise BudgetBlocked(
            f"HN budget NO-GO: K={len(hard_names)}, "
            f"actual loader requires {updates}>{max_updates} updates under {policy}; preserve H"
        )
    return {
        **probe,
        "budget_policy": policy,
        "budget_policy_version": "FULL-STRATEGY-B-40E-20261020-V1",
        "strict_equal_optimizer_updates_claim_allowed": policy == "A_STRICT_440",
        "extra_updates_vs_standard": updates - 440,
        "extra_exposures_vs_standard": epochs * len(hard_names),
        "fit_size": len(fit_rows),
        "K": len(hard_names),
        "epochs": epochs,
        "batch": batch,
        "nbs": nbs,
        "seed": seed,
        "optimizer_updates": updates,
        "sample_exposures": epochs * probe["exposures_per_epoch"],
        "max_updates": updates,
        "maximum_feasible_updates_if_all_FIT_negatives": max_updates,
        "budget_gate": "PASS",
        "actual_training_executed": False,
    }


def check_runtime_budget(trainer, budget, *, completed=False):
    """Inspect the real training loader after setup, then measured completion."""
    from torch.utils.data import RandomSampler

    from fire_detection.research.training import ReplayDataset

    loader = trainer.train_loader
    args = trainer.args
    if (args.epochs, args.batch, args.nbs, args.warmup_epochs, args.compile, args.resume) != (
        budget["epochs"],
        64,
        64,
        0.0,
        False,
        False,
    ):
        raise BudgetBlocked("Runtime frozen budget settings drift")
    if trainer.batch_size != 64 or trainer.accumulate != 1 or trainer.amp:
        raise BudgetBlocked("Runtime batch/accumulation/AMP drift")
    if len(loader) != budget["batches_per_epoch"] or len(loader.dataset) != budget["dataset_size"]:
        raise BudgetBlocked("Runtime loader length differs from admitted budget")
    if loader.batch_size != 64 or loader.drop_last:
        raise BudgetBlocked("Runtime loader batch/drop_last drift")
    if not isinstance(loader.dataset, ReplayDataset):
        raise BudgetBlocked("Runtime loader must retain existing HN ReplayDataset")
    if not isinstance(loader.sampler, RandomSampler) or loader.sampler.replacement:
        raise BudgetBlocked("Runtime sampler differs from the pinned non-replacement sampler")
    dataset = loader.dataset
    actual = Counter(Path(str(dataset.samples[i][0])).name for i in dataset.indices)
    if actual != budget["per_filename_exposures_per_epoch"]:
        raise BudgetBlocked("Runtime replay identities differ from admitted fit + H")
    audit = trainer.research_audit
    if (
        audit["optimizer_steps"] > budget["max_updates"]
        or audit["sample_exposures"] > budget["sample_exposures"]
    ):
        raise BudgetBlocked("Runtime measured budget exceeded")
    if completed:
        if (
            audit["optimizer_steps"] != budget["optimizer_updates"]
            or audit["sample_exposures"] != budget["sample_exposures"]
        ):
            raise BudgetBlocked("Incomplete actual optimizer/exposure budget")
        expected = {name: count * budget["epochs"] for name, count in actual.items()}
        if dict(trainer.exposure_counts) != expected:
            raise BudgetBlocked("Measured per-filename exposure ledger differs")
    return True


def install_budget_callbacks(trainer, budget):
    """Install after _setup_train, before any forward/backward or optimizer step."""
    check_runtime_budget(trainer, budget)
    trainer.add_callback("on_train_epoch_start", lambda t: check_runtime_budget(t, budget))
    trainer.add_callback("on_train_epoch_end", lambda t: check_runtime_budget(t, budget))

    def before_optimizer(optimizer, args, kwargs):
        check_runtime_budget(trainer, budget)
        if trainer.research_audit["optimizer_steps"] >= budget["optimizer_updates"]:
            raise BudgetBlocked("Optimizer update would exceed admitted budget")

    return trainer.optimizer.register_step_pre_hook(before_optimizer)
