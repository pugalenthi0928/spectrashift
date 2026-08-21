from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SceneRecord:
    sample_id: str
    scene_group: str


def _stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def grouped_split(
    records: Iterable[SceneRecord],
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, list[SceneRecord]]:
    """Split whole scene groups deterministically, never individual neighboring samples."""

    records = list(records)
    if not records:
        raise ValueError("records must not be empty")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("sample identifiers must be unique")

    groups = sorted({record.scene_group for record in records}, key=lambda x: _stable_order(x, seed))
    if len(groups) < 3:
        raise ValueError("at least three scene groups are required")
    train_cut = max(1, round(len(groups) * train_fraction))
    validation_count = max(1, round(len(groups) * validation_fraction))
    validation_cut = min(len(groups) - 1, train_cut + validation_count)
    train_groups = set(groups[:train_cut])
    validation_groups = set(groups[train_cut:validation_cut])
    test_groups = set(groups[validation_cut:])
    if not validation_groups or not test_groups:
        raise ValueError("fractions produce an empty validation or test group set")

    result = {"train": [], "validation": [], "test": []}
    for record in records:
        if record.scene_group in train_groups:
            result["train"].append(record)
        elif record.scene_group in validation_groups:
            result["validation"].append(record)
        elif record.scene_group in test_groups:
            result["test"].append(record)
    assert_group_disjoint(result)
    return result


def assert_group_disjoint(splits: dict[str, list[SceneRecord]]) -> None:
    group_sets = {
        name: {record.scene_group for record in records} for name, records in splits.items()
    }
    names = list(group_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = group_sets[left] & group_sets[right]
            if overlap:
                raise ValueError(f"scene-group leakage between {left} and {right}: {sorted(overlap)}")

