import os
import sys

import numpy as np
import pytest

pytest.importorskip("datasets")
pytest.importorskip("imblearn")

from datasets import Dataset, load_from_disk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.downsample_token_classification_dataset import (  # noqa: E402
    _label_counts,
    downsample_dataset,
    main,
)


WANDB_LIVER_TRAIN_DATASETS = [
    "/sci/labs/michall/roeizucker/roei_test/token_cls_datasets2/_token_cls_Liver-Hepatocytes_kmer/"
    "Z000000R3_retrain_token_cls_Liver-Hepatocytes_kmer_seq_5400_datasets_train",
    "/sci/labs/michall/roeizucker/roei_test/token_cls_datasets2/_token_cls_Liver-Hepatocytes_kmer/"
    "Z000000T3_retrain_token_cls_Liver-Hepatocytes_kmer_seq_5400_datasets_train",
    "/sci/labs/michall/roeizucker/roei_test/token_cls_datasets2/_token_cls_Liver-Hepatocytes_kmer/"
    "Z0000043Q_retrain_token_cls_Liver-Hepatocytes_kmer_seq_5400_datasets_train",
    "/sci/labs/michall/roeizucker/roei_test/token_cls_datasets2/_token_cls_Liver-Hepatocytes_kmer/"
    "Z0000044H_retrain_token_cls_Liver-Hepatocytes_kmer_seq_5400_datasets_train",
    "/sci/labs/michall/roeizucker/roei_test/token_cls_datasets2/_token_cls_Liver-Hepatocytes_kmer/"
    "Z0000044M_retrain_token_cls_Liver-Hepatocytes_kmer_seq_5400_datasets_train",
    "/sci/labs/michall/roeizucker/roei_test/token_cls_datasets2/_token_cls_Liver-Hepatocytes_kmer/"
    "Z00000431_retrain_token_cls_Liver-Hepatocytes_kmer_seq_5400_datasets_train",
]


def _dataset(labels):
    return Dataset.from_dict(
        {
            "seq": ["A" * len(row) for row in labels],
            "labels": labels,
            "window_id": [f"window_{i}" for i in range(len(labels))],
        }
    )


def test_ratio_one_balances_classes_with_random_under_sampler():
    dataset = _dataset(
        [
            [0, 0, 0, 0, 1, 2, -100],
            [0, 0, 1, 1, 2, 2, -100],
            [0, 0, 0, 1, 2, -100, -100],
        ]
    )

    downsampled = downsample_dataset(dataset, ratio=1.0, seed=7)

    assert _label_counts(np.array(downsampled["labels"]), -100) == {0: 4, 1: 4, 2: 4}
    for original, sampled in zip(dataset["labels"], downsampled["labels"]):
        for original_label, sampled_label in zip(original, sampled):
            if original_label == -100:
                assert sampled_label == -100


def test_ratio_half_caps_majority_classes_at_twice_minority_count():
    dataset = _dataset(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, -100, -100],
            [2, 2, 2, 2, 2, 2],
        ]
    )

    downsampled = downsample_dataset(dataset, ratio=0.5, seed=7)

    assert _label_counts(np.array(downsampled["labels"]), -100) == {0: 4, 1: 2, 2: 4}


@pytest.mark.parametrize(
    ("ratio", "expected_counts"),
    [
        (1.0, {0: 3, 1: 3, 2: 3}),
        (0.5, {0: 6, 1: 3, 2: 6}),
        (0.25, {0: 10, 1: 3, 2: 10}),
        (0.05, {0: 10, 1: 3, 2: 10}),
    ],
)
def test_multiple_ratios_use_expected_multiclass_sampling_strategy(ratio, expected_counts):
    dataset = _dataset(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [1, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, -100],
        ]
    )

    downsampled = downsample_dataset(dataset, ratio=ratio, seed=7)

    assert _label_counts(np.array(downsampled["labels"]), -100) == expected_counts


def test_same_seed_is_deterministic():
    dataset = _dataset(
        [
            [0, 0, 0, 0, 0, 1, 2],
            [1, 2, 2, 2, 2, 0, 0],
        ]
    )

    first = downsample_dataset(dataset, ratio=1.0, seed=11)
    second = downsample_dataset(dataset, ratio=1.0, seed=11)

    assert first["labels"] == second["labels"]


def test_filter_empty_examples_removes_rows_that_have_no_trainable_labels():
    dataset = _dataset([[-100, -100], [0, 1], [2, 2]])

    downsampled = downsample_dataset(dataset, ratio=1.0, seed=7)

    assert len(downsampled) == 2
    assert all(any(label != -100 for label in row) for row in downsampled["labels"])


def test_main_saves_reloadable_downsampled_dataset(tmp_path, monkeypatch):
    input_path = tmp_path / "input_ds"
    output_path = tmp_path / "output_ds"
    _dataset([[0, 0, 0, 1, 2], [0, 0, 1, 2, 2]]).save_to_disk(str(input_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "downsample_token_classification_dataset.py",
            "--input-dataset-path",
            str(input_path),
            "--output-dataset-path",
            str(output_path),
            "--minority-to-majority-ratio",
            "1.0",
        ],
    )

    main()

    saved = load_from_disk(str(output_path))
    assert _label_counts(np.array(saved["labels"]), -100) == {0: 2, 1: 2, 2: 2}


def test_main_refuses_to_overwrite_without_flag(tmp_path, monkeypatch):
    input_path = tmp_path / "input_ds"
    output_path = tmp_path / "output_ds"
    _dataset([[0, 1, 2]]).save_to_disk(str(input_path))
    _dataset([[0, 1, 2]]).save_to_disk(str(output_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "downsample_token_classification_dataset.py",
            "--input-dataset-path",
            str(input_path),
            "--output-dataset-path",
            str(output_path),
            "--minority-to-majority-ratio",
            "1.0",
        ],
    )

    with pytest.raises(FileExistsError):
        main()


def test_main_refuses_same_input_and_output_path(tmp_path, monkeypatch):
    input_path = tmp_path / "input_ds"
    _dataset([[0, 1, 2]]).save_to_disk(str(input_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "downsample_token_classification_dataset.py",
            "--input-dataset-path",
            str(input_path),
            "--output-dataset-path",
            str(input_path),
            "--minority-to-majority-ratio",
            "1.0",
            "--overwrite",
        ],
    )

    with pytest.raises(ValueError, match="must be different"):
        main()


def test_all_blank_dataset_raises_clear_error():
    dataset = _dataset([[-100, -100], [-100, -100]])

    with pytest.raises(ValueError, match="Cannot downsample"):
        downsample_dataset(dataset, ratio=1.0, seed=7)


@pytest.mark.integration
@pytest.mark.parametrize("ratio", [1.0, 0.5, 0.25, 0.05])
@pytest.mark.parametrize("dataset_path", WANDB_LIVER_TRAIN_DATASETS)
def test_wandb_liver_train_dataset_top_rows_smoke(dataset_path, ratio, tmp_path):
    if not os.path.exists(dataset_path):
        pytest.skip(f"real WandB train dataset is not available: {dataset_path}")

    top_rows = int(os.environ.get("DOWNSAMPLE_REAL_TOP_ROWS", "1000"))
    dataset = load_from_disk(dataset_path)
    dataset = dataset.select(range(min(top_rows, len(dataset))))
    before = _label_counts(np.array(dataset["labels"]), -100)

    downsampled = downsample_dataset(dataset, ratio=ratio, seed=42)
    after = _label_counts(np.array(downsampled["labels"]), -100)
    downsampled.save_to_disk(str(tmp_path / "downsampled"))
    reloaded = load_from_disk(str(tmp_path / "downsampled"))

    assert before
    assert after == _label_counts(np.array(reloaded["labels"]), -100)
    assert min(after.values()) > 0
    assert max(after.values()) <= int(min(after.values()) / ratio)
