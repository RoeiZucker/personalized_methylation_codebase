import os
import sys
from types import SimpleNamespace

import pytest

datasets = pytest.importorskip("datasets")
from datasets import load_from_disk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import token_classification_data_extractor as extractor


class TinyTokenizer:
    def __call__(self, seq):
        return {
            "input_ids": [ord(base) % 5 for base in seq],
            "attention_mask": [1] * len(seq),
        }


def _chr10_dataset(size, prefix):
    return datasets.Dataset.from_dict(
        {
            "seq": ["ACGTAC"] * size,
            "labels": [[-100, 0, 1, 2] for _ in range(size)],
            "chrom": ["chr10"] * size,
            "start": list(range(size)),
            "end": [value + 6 for value in range(size)],
            "window_id": [f"{prefix}_chr10_window_{value}" for value in range(size)],
        }
    )


def _patch_token_dataset_loading(monkeypatch, train_dataset, test_dataset):
    def load_dataset(task, dataframe_path, kmer_size, seq_size):
        if dataframe_path == "train.csv":
            return train_dataset
        if dataframe_path == "test.csv":
            return test_dataset
        raise AssertionError(f"unexpected dataframe path: {dataframe_path}")

    monkeypatch.setattr(extractor, "_load_token_dataset", load_dataset)
    monkeypatch.setattr(
        extractor,
        "resolve_token_label_binning",
        lambda task, train_dataset=None: SimpleNamespace(
            method="none",
            low=None,
            high=None,
            resolved_low=None,
            resolved_high=None,
            blank_label=-100,
        ),
    )
    monkeypatch.setattr(extractor, "apply_token_label_binning_to_dataset", lambda dataset, resolved: dataset)
    monkeypatch.setattr(
        extractor,
        "apply_token_label_downsampling_to_dataset",
        lambda dataset, task, blank_label, seed, verbose: dataset,
    )


def _paths(tmp_path, with_eval=True):
    paths = {
        "intermediate_train_data_path": "train.csv",
        "intermediate_test_data_path": "test.csv",
        "hf_dataset_train_path": str(tmp_path / "train_hf"),
        "hf_dataset_test_path": str(tmp_path / "test_hf"),
    }
    if with_eval:
        paths["hf_dataset_eval_path"] = str(tmp_path / "eval_hf")
    return paths


def _create_datasets(monkeypatch, tmp_path, eval_size):
    train_dataset = _chr10_dataset(4, "train")
    test_dataset = _chr10_dataset(10, "test")
    _patch_token_dataset_loading(monkeypatch, train_dataset, test_dataset)
    paths = _paths(tmp_path, with_eval=eval_size > 0)

    extractor._create_token_classification_hf_datasets(
        task={},
        paths=paths,
        tokenizer=TinyTokenizer(),
        train_test_seperation_type=None,
        blank_label=-100,
        test_size=0.2,
        eval_size=eval_size,
        random_state=42,
        kmer_size=6,
        seq_size=6,
        train_only=False,
        verbose=False,
    )
    return paths


def test_existing_token_extraction_still_writes_only_train_and_test(tmp_path, monkeypatch):
    paths = _create_datasets(monkeypatch, tmp_path, eval_size=0)

    train_dataset = load_from_disk(paths["hf_dataset_train_path"])
    test_dataset = load_from_disk(paths["hf_dataset_test_path"])

    assert len(train_dataset) == 4
    assert len(test_dataset) == 10
    assert not (tmp_path / "eval_hf").exists()
    assert set(test_dataset["chrom"]) == {"chr10"}
    assert "seq" not in test_dataset.column_names
    assert "input_ids" in test_dataset.column_names


def test_token_extraction_can_split_test_rows_into_eval_dataset(tmp_path, monkeypatch):
    paths = _create_datasets(monkeypatch, tmp_path, eval_size=0.2)

    test_dataset = load_from_disk(paths["hf_dataset_test_path"])
    eval_dataset = load_from_disk(paths["hf_dataset_eval_path"])

    assert len(test_dataset) == 8
    assert len(eval_dataset) == 2
    assert set(test_dataset["chrom"]) == {"chr10"}
    assert set(eval_dataset["chrom"]) == {"chr10"}

    test_windows = set(test_dataset["window_id"])
    eval_windows = set(eval_dataset["window_id"])
    assert test_windows.isdisjoint(eval_windows)
    assert test_windows | eval_windows == {f"test_chr10_window_{value}" for value in range(10)}


def test_eval_split_is_row_level_and_keeps_complete_label_arrays(tmp_path, monkeypatch):
    paths = _create_datasets(monkeypatch, tmp_path, eval_size=0.2)

    test_dataset = load_from_disk(paths["hf_dataset_test_path"])
    eval_dataset = load_from_disk(paths["hf_dataset_eval_path"])

    for dataset in [test_dataset, eval_dataset]:
        for labels in dataset["labels"]:
            assert labels == [-100, 0, 1, 2]


def test_eval_size_requires_eval_output_path(tmp_path, monkeypatch):
    train_dataset = _chr10_dataset(4, "train")
    test_dataset = _chr10_dataset(10, "test")
    _patch_token_dataset_loading(monkeypatch, train_dataset, test_dataset)

    with pytest.raises(ValueError, match="hf_dataset_eval_path"):
        extractor._create_token_classification_hf_datasets(
            task={},
            paths=_paths(tmp_path, with_eval=False),
            tokenizer=TinyTokenizer(),
            train_test_seperation_type=None,
            blank_label=-100,
            test_size=0.2,
            eval_size=0.2,
            random_state=42,
            kmer_size=6,
            seq_size=6,
            train_only=False,
            verbose=False,
        )


@pytest.mark.parametrize("eval_size", [-0.1, 1.0])
def test_eval_size_must_be_a_fraction_smaller_than_one(eval_size):
    with pytest.raises(ValueError, match="eval_size"):
        extractor._validate_eval_dataset_params({"hf_dataset_eval_path": "/tmp/eval"}, eval_size)
