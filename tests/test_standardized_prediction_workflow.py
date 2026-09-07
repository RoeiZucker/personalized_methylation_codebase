"""Tests for the standardized prediction-only eval/test workflow."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

datasets = pytest.importorskip("datasets")
from datasets import Dataset, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_standardized_prediction_datasets import (
    create_datasets,
    manifest_rows,
    partition_windows,
    select_windows,
)
from scripts.create_standardized_prediction_jobs import (
    ARRAY_RUNNER,
    CONFIG_ROOT,
    TASK_FILE,
    RunInfo,
    VERSION,
    array_submission_command,
    build_prediction_config,
    prediction_task,
)
from scripts import run_pretrain_checkpoint_prediction as prediction_runner
from scripts.create_chr12_prediction_dataset_jobs import (
    SampleInfo,
    build_config as build_chr12_config,
)
from src import token_classification_data_extractor as extraction_runner


def test_common_window_partition_is_exact_and_deterministic():
    common = {f"chr1:{start}-{start + 100}" for start in range(0, 1000, 100)}

    first_eval, first_test = partition_windows(common, 0.2, 42)
    second_eval, second_test = partition_windows(set(reversed(sorted(common))), 0.2, 42)

    assert first_eval == second_eval
    assert first_test == second_test
    assert len(first_eval) == 2
    assert len(first_test) == 8
    assert set(first_eval).isdisjoint(first_test)
    assert set(first_eval) | set(first_test) == common
    assert manifest_rows(first_eval, first_test) == sorted(
        [(window_id, "eval") for window_id in first_eval]
        + [(window_id, "test") for window_id in first_test]
    )


def test_select_windows_uses_manifest_order_and_needs_no_train_dataset(tmp_path):
    source = Dataset.from_dict(
        {
            "window_id": ["w3", "w1", "w2"],
            "input_ids": [[3], [1], [2]],
            "labels": [[0], [1], [2]],
        }
    )

    selected = select_windows(source, ["w1", "w2"], tmp_path / "source")

    assert selected["window_id"] == ["w1", "w2"]
    assert selected["input_ids"] == [[1], [2]]


def test_dataset_builder_writes_identical_ordered_splits(tmp_path):
    runs_root = tmp_path / "runs"
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    tissue_dir = "_token_cls_Tissue_kmer"
    sample_windows = {
        "S1": ["chr12:w0", "chr12:w1", "chr12:w2", "chr12:w3", "chr12:w4"],
        "S2": ["chr12:w1", "chr12:w2", "chr12:w3", "chr12:w4", "chr12:w5"],
    }
    for sample, ids in sample_windows.items():
        run = runs_root / tissue_dir / f"{sample}_epoch-10-step-1_downsampled_ratio_0p25_run"
        run.mkdir(parents=True)
        source = (
            input_root
            / tissue_dir
            / f"{sample}_retrain_token_cls_Tissue_kmer_seq_5400_datasets_test"
        )
        Dataset.from_dict(
            {"window_id": ids, "input_ids": [[index] for index in range(len(ids))]}
        ).save_to_disk(str(source))
    args = SimpleNamespace(
        runs_root=str(runs_root),
        input_root=str(input_root),
        output_root=str(output_root),
        eval_fraction=0.5,
        seed=42,
        skip_existing=False,
        overwrite=False,
        dry_run=False,
    )

    create_datasets(args)

    observed = []
    for sample in sample_windows:
        prefix = f"{sample}_retrain_token_cls_Tissue_kmer_seq_5400_datasets_"
        eval_dataset = load_from_disk(str(output_root / tissue_dir / f"{prefix}eval"))
        test_dataset = load_from_disk(str(output_root / tissue_dir / f"{prefix}test"))
        observed.append((eval_dataset["window_id"], test_dataset["window_id"]))
    assert observed[0] == observed[1]
    assert set(observed[0][0]) | set(observed[0][1]) == {"chr12:w1", "chr12:w2", "chr12:w3", "chr12:w4"}

    args.skip_existing = True
    create_datasets(args)


def fake_run(tmp_path: Path) -> RunInfo:
    run_path = tmp_path / "run"
    pretrain = tmp_path / "epoch-10-step-100"
    checkpoints = tuple(tmp_path / f"checkpoint-{step}" for step in (10, 20, 30))
    return RunInfo(
        "Z000000AA",
        "_token_cls_Tissue_kmer",
        "Tissue",
        run_path,
        pretrain,
        checkpoints,
    )


def test_prediction_config_has_no_training_dataset(tmp_path):
    run = fake_run(tmp_path)
    source = {
        "paths": {"train_dataset_path": "/old/train", "test_dataset_path": "/old/test"},
        "task": {"analysis_name": "old", "top_rows": 1},
        "wandb": {"enabled": True},
        "train": {"load_dataset_to_memory": True, "per_device_eval_batch_size": 14},
    }
    split_path = tmp_path / "standardized_eval"

    config = build_prediction_config(source, run, split_path, "eval")

    assert config["paths"]["train_dataset_path"] is None
    assert config["paths"]["eval_dataset_path"] == str(split_path)
    assert config["paths"]["test_dataset_path"] is None
    assert config["paths"]["trained_model_path"] == str(run.pretrain_path)
    assert config["wandb"] == {"enabled": False}
    assert VERSION in config["task"]["analysis_name"]
    assert "train" not in config
    assert config["prediction"] == {
        "load_dataset_to_memory": True,
        "per_device_eval_batch_size": 14,
    }


def test_array_submission_is_short_and_preserves_task_arguments(tmp_path):
    run = fake_run(tmp_path)
    target = run.run_path / "checkpoint-10"
    config_path = CONFIG_ROOT / "sample.yaml"

    task = prediction_task(
        run,
        target,
        run.checkpoints[0],
        "test",
        config_path,
        Path("/sci/nosnap/project"),
    )
    command = array_submission_command(
        248,
        TASK_FILE,
        ARRAY_RUNNER,
        Path("/sci/nosnap/project"),
        20,
    )

    assert command.startswith("sbatch --array=0-247%20 ")
    assert "run_downsampled_prediction_chr12_standardized_v1_array.sh" in command
    assert "downsampled_prediction_chr12_standardized_v1_array_tasks.txt" in command
    assert "--wrap" not in command
    assert task.config_path == Path("/sci/nosnap/project/configs/downsampled_runs/prediction_chr12_standardized_v1/sample.yaml")
    assert task.checkpoint_path == run.checkpoints[0]
    assert task.output_path == target / VERSION / "test_predictions.csv.gitbackup"
    runner = ARRAY_RUNNER.read_text()
    assert "#SBATCH --gres=gg:g4:1" in runner
    assert "#SBATCH --constraint=firefoot|cyril" in runner
    assert '--checkpoint-path "$CHECKPOINT_PATH"' in runner


def test_prediction_trainer_does_not_attach_a_training_dataset(monkeypatch, tmp_path):
    prediction_dataset = object()
    captured = {}
    monkeypatch.setattr(
        prediction_runner.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        prediction_runner,
        "get_fine_tuned_model",
        lambda *args, **kwargs: object(),
    )

    def capture_trainer(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(prediction_runner, "get_trainer", capture_trainer)
    config = {
        "model": {
            "model_repo": "repo",
            "model_name": "model",
            "model_type": "classification_analysis",
            "num_labels": 3,
        },
        "prediction": {"per_device_eval_batch_size": 14},
    }

    prediction_runner._create_prediction_trainer(
        config, tmp_path / "checkpoint", prediction_dataset, tmp_path / "output"
    )

    assert captured["dataset"] is None
    assert captured["eval_dataset"] is prediction_dataset
    assert captured["per_device_eval_batch_size"] == 14
    assert prediction_runner._prediction_settings({"train": {"legacy": True}})["legacy"]



def test_chr12_extraction_config_is_prediction_only(tmp_path):
    source = {
        "paths": {
            "raw_data_path": "/raw/sample.bigwig",
            "hf_dataset_train_path": "/old/train",
            "hf_dataset_test_path": "/old/test",
        },
        "task": {
            "chromosomes": ["chr1"],
            "test_size": 0.2,
            "token_label_binning": {"method": "fixed", "low": 0.2, "high": 0.8},
        },
    }
    info = SampleInfo("S1", "_token_cls_Tissue_kmer", "Tissue")

    config = build_chr12_config(source, tmp_path, info)

    assert config["task"]["chromosomes"] == ["chr12"]
    assert config["task"]["prediction_only"] is True
    assert config["task"]["test_size"] == 0
    assert "hf_dataset_train_path" not in config["paths"]
    assert config["paths"]["hf_dataset_test_path"].endswith("datasets_test")


def test_prediction_only_extractor_never_uses_train_hf_path(monkeypatch, tmp_path):
    captured = {}
    cfg = {
        "paths": {
            "intermediate_train_data_path": str(tmp_path / "source.csv"),
            "intermediate_test_data_path": str(tmp_path / "empty.csv"),
            "hf_dataset_test_path": str(tmp_path / "chr12_test"),
        },
        "task": {
            "prediction_only": True,
            "test_size": 0,
            "output_hf_dataset": True,
            "override_dataset": False,
            "tokenizer_name": "tokenizer",
            "seq_size": 5400,
        },
    }
    monkeypatch.setattr(extraction_runner, "_maybe_create_intermediate_data", lambda config: None)
    monkeypatch.setattr(
        extraction_runner.AutoTokenizer,
        "from_pretrained",
        lambda name: object(),
    )

    def capture(*args):
        captured["paths"] = args[1]
        return SimpleNamespace(
            method="fixed",
            low=0.2,
            high=0.8,
            resolved_low=0.2,
            resolved_high=0.8,
        )

    monkeypatch.setattr(extraction_runner, "_create_prediction_only_hf_dataset", capture)

    extraction_runner.encode_cpg_token_classification_extraction(cfg)

    assert captured["paths"]["hf_dataset_test_path"].endswith("chr12_test")
    assert "hf_dataset_train_path" not in captured["paths"]
