import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.token_label_downsampling_utils import (
    apply_token_label_downsampling_to_dataset,
    count_token_class_labels,
)


class SimpleDataset:
    def __init__(self, rows):
        self.rows = rows

    @classmethod
    def from_labels(cls, labels):
        return cls(
            [
                {
                    "seq": "A",
                    "labels": example_labels,
                }
                for example_labels in labels
            ]
        )

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, column):
        return [row[column] for row in self.rows]

    def map(self, function, batched=False, **kwargs):
        if not batched:
            return SimpleDataset([function(row) for row in self.rows])

        batch = {key: [row[key] for row in self.rows] for key in self.rows[0]}
        mapped_batch = function(batch)
        new_rows = []
        for row_index, row in enumerate(self.rows):
            new_row = row.copy()
            for key, values in mapped_batch.items():
                new_row[key] = values[row_index]
            new_rows.append(new_row)
        return SimpleDataset(new_rows)

    def filter(self, function, **kwargs):
        return SimpleDataset([row for row in self.rows if function(row)])


def _dataset(labels):
    return SimpleDataset.from_labels(labels)


def _task_config(ratio, enabled=True, seed=None):
    config = {
        "num_labels": 3,
        "token_label_downsampling": {
            "enabled": enabled,
            "minority_to_majority_ratio": ratio,
        },
    }
    if seed is not None:
        config["token_label_downsampling"]["seed"] = seed
    return config


def test_disabled_token_label_downsampling_leaves_labels_unchanged():
    dataset = _dataset([[0, 0, 1, 2, -100], [2, 1, 0]])

    downsampled = apply_token_label_downsampling_to_dataset(
        dataset,
        _task_config(1.0, enabled=False),
        seed=7,
    )

    assert downsampled["labels"] == dataset["labels"]


def test_ratio_one_balances_token_classes_by_masking_majority_labels():
    dataset = _dataset([[0, 0, 0, 0, 1, 2], [1, 2, 2, 2]])

    downsampled = apply_token_label_downsampling_to_dataset(
        dataset,
        _task_config(1.0),
        seed=7,
    )

    assert dict(count_token_class_labels(downsampled, class_ids=[0, 1, 2])) == {
        0: 2,
        1: 2,
        2: 2,
    }


def test_ratio_half_keeps_majority_classes_at_most_twice_the_minority_class():
    dataset = _dataset(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1],
            [2, 2, 2, 2, 2, 2],
        ]
    )

    downsampled = apply_token_label_downsampling_to_dataset(
        dataset,
        _task_config(0.5),
        seed=7,
    )

    assert dict(count_token_class_labels(downsampled, class_ids=[0, 1, 2])) == {
        0: 4,
        1: 2,
        2: 4,
    }


def test_existing_blank_labels_stay_ignored():
    dataset = _dataset([[-100, 0, 1, 2]])

    downsampled = apply_token_label_downsampling_to_dataset(
        dataset,
        _task_config(1.0),
        seed=7,
    )

    assert downsampled["labels"] == [[-100, 0, 1, 2]]


def test_same_seed_gives_deterministic_token_label_downsampling():
    dataset = _dataset(
        [
            [0, 0, 0, 0, 0, 1, 2],
            [1, 2, 2, 2, 2],
        ]
    )

    first = apply_token_label_downsampling_to_dataset(dataset, _task_config(1.0), seed=7)
    second = apply_token_label_downsampling_to_dataset(dataset, _task_config(1.0), seed=7)

    assert first["labels"] == second["labels"]


def test_missing_class_raises_clear_error_when_downsampling_is_enabled():
    dataset = _dataset([[0, 0, 1, 1]])

    with pytest.raises(ValueError, match="missing from the training labels"):
        apply_token_label_downsampling_to_dataset(
            dataset,
            _task_config(1.0),
            seed=7,
        )
