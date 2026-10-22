"""Synthetic checks for the later fixed random-negative replay contract."""

from __future__ import annotations

from fire_detection.research.protocol import expected_budget, replay_indices


def test_fixed_random_replay_preserves_budget_and_negative_identity() -> None:
    names = [f"positive_{i:03d}.jpg" for i in range(520)] + [
        f"negative_{i:03d}.jpg" for i in range(140)
    ]
    labels = [1] * 520 + [0] * 140
    hard_names = names[520:632]
    first = replay_indices(
        names, labels, hard_names, arm="random_negative", seed=20260914, epoch=0
    )
    again = replay_indices(
        names, labels, hard_names, arm="random_negative", seed=20260914, epoch=39
    )
    assert first == again
    assert first[:660] == tuple(range(660))
    selected = [names[i] for i in first[660:]]
    assert len(selected) == len(set(selected)) == 112
    assert all(labels[i] == 0 for i in first[660:])
    assert len(set(selected) & set(hard_names)) >= 84
    budget = expected_budget(660, 112, 64, 40)
    assert budget["optimizer_steps"] == 520
    assert budget["sample_exposures"] == 30_880
