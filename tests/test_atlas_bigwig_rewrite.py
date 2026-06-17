import os
import sys
from pathlib import Path

import importlib
import types

import pandas as pd
import pyBigWig
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))


def _install_evaluate_stub():
    module = types.ModuleType("evaluate")

    class Metric:
        def __init__(self, name):
            self.name = name

        def compute(self, predictions=None, references=None, prediction_scores=None):
            import numpy as np

            if self.name == "pearsonr":
                predictions_arr = np.asarray(predictions, dtype=float)
                references_arr = np.asarray(references, dtype=float)
                if len(predictions_arr) <= 1:
                    return {"pearsonr": np.nan}
                return {"pearsonr": float(np.corrcoef(predictions_arr, references_arr)[0, 1])}
            if self.name == "mse":
                predictions_arr = np.asarray(predictions, dtype=float)
                references_arr = np.asarray(references, dtype=float)
                return {"mse": float(np.mean((predictions_arr - references_arr) ** 2))}
            if self.name == "mae":
                predictions_arr = np.asarray(predictions, dtype=float)
                references_arr = np.asarray(references, dtype=float)
                return {"mae": float(np.mean(np.abs(predictions_arr - references_arr)))}
            if self.name == "accuracy":
                predictions_arr = np.asarray(predictions)
                references_arr = np.asarray(references)
                return {"accuracy": float(np.mean(predictions_arr == references_arr))}
            if self.name == "matthews_correlation":
                return {"matthews_correlation": 0.0}
            if self.name == "roc_auc":
                return {"roc_auc": 0.0}
            if self.name == "f1":
                return {"f1": 0.0}
            if self.name == "precision":
                return {"precision": 0.0}
            if self.name == "recall":
                return {"recall": 0.0}
            return {self.name: 0.0}

    module.load = lambda name: Metric(name)
    sys.modules["evaluate"] = module


_install_evaluate_stub()

from src.atlas_evaluation_creator_script import create_atlas_eval_configs
from src.utils.atlas_bigwig_utils import build_atlas_position_dataframe, evaluate_atlas_from_bigwigs


def _write_bigwig(path: Path, starts, values, chrom="chr1", chrom_length=100):
    bw = pyBigWig.open(str(path), "w")
    try:
        bw.addHeader([(chrom, chrom_length)])
        bw.addEntries(
            [chrom] * len(starts),
            starts,
            ends=[start + 1 for start in starts],
            values=[float(value) for value in values],
        )
    finally:
        bw.close()


def _create_sample_bigwigs(tmp_path: Path):
    starts = [0, 6, 12, 18, 24, 30]
    target_path = tmp_path / "target.bw"
    atlas_a_path = tmp_path / "atlas_a.bw"
    atlas_b_path = tmp_path / "atlas_b.bw"

    _write_bigwig(target_path, starts, [0.1, 0.3, 0.5, 0.7, 0.55, 0.7])
    _write_bigwig(atlas_a_path, starts, [0.1, 0.2, 0.9, 0.8, 0.95, 0.85])
    _write_bigwig(atlas_b_path, starts, [0.1, 0.4, 0.1, 0.6, 0.15, 0.55])
    return target_path, [atlas_a_path, atlas_b_path]


def _install_datasets_stub():
    module = types.ModuleType("datasets")

    class Dataset:  # pragma: no cover - compatibility shim for import-time only
        pass

    module.Dataset = Dataset
    module.DatasetDict = dict
    module.Sequence = list
    module.Value = object
    module.concatenate_datasets = lambda *args, **kwargs: None
    module.load_dataset = lambda *args, **kwargs: None
    module.load_from_disk = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_from_disk should not be called during atlas tests"))
    sys.modules["datasets"] = module


def _install_runtime_stubs():
    _install_datasets_stub()

    transformers_module = types.ModuleType("transformers")

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - smoke-test stub only
            return cls()

    transformers_module.AutoTokenizer = _Tokenizer
    sys.modules["transformers"] = transformers_module

    trainer_utils_module = types.ModuleType("src.utils.trainer_utils")
    trainer_utils_module.get_compute_func = lambda *args, **kwargs: None
    trainer_utils_module.get_trainer = lambda *args, **kwargs: None
    trainer_utils_module.get_trainer_type = lambda *args, **kwargs: None
    sys.modules["src.utils.trainer_utils"] = trainer_utils_module

    dataset_utils_module = types.ModuleType("src.utils.dataset_utils")
    dataset_utils_module.keep_batch = lambda *args, **kwargs: None
    sys.modules["src.utils.dataset_utils"] = dataset_utils_module

    model_utils_module = types.ModuleType("src.utils.model_utils")
    model_utils_module.get_fine_tuned_model = lambda *args, **kwargs: None
    sys.modules["src.utils.model_utils"] = model_utils_module

    tissue_model_utils_module = types.ModuleType("src.utils.tissue_model_utils")
    tissue_model_utils_module.get_fine_tuned_model = lambda *args, **kwargs: None
    sys.modules["src.utils.tissue_model_utils"] = tissue_model_utils_module

    tissue_trainer_utils_module = types.ModuleType("src.utils.tissue_trainer_utils")
    tissue_trainer_utils_module.get_trainer = lambda *args, **kwargs: None
    sys.modules["src.utils.tissue_trainer_utils"] = tissue_trainer_utils_module


def _import_module(module_name: str):
    _install_runtime_stubs()
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_atlas_bigwig_helper_recreates_variability_and_debug_controls(tmp_path):
    target_path, atlas_paths = _create_sample_bigwigs(tmp_path)

    matched_df = build_atlas_position_dataframe(
        target_bigwig_path=str(target_path),
        atlas_bigwig_paths=[str(path) for path in atlas_paths],
        number_of_bins=2,
        chroms=["chr1"],
        verbose=False,
    )

    assert list(matched_df["start"]) == [0, 6, 12, 18, 24, 30]
    assert matched_df["atlas_mean"].tolist() == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.55, 0.7], abs=1e-3)
    assert matched_df["target_value"].tolist() == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.55, 0.7], abs=1e-3)
    assert sorted(matched_df["std_bin"].astype(str).value_counts().tolist()) == [2, 4]

    top_rows_df = build_atlas_position_dataframe(
        target_bigwig_path=str(target_path),
        atlas_bigwig_paths=[str(path) for path in atlas_paths],
        number_of_bins=2,
        chroms=["chr1"],
        top_rows=4,
        verbose=False,
    )
    assert list(top_rows_df["start"]) == [0, 6, 12, 18]

    sampled_df = build_atlas_position_dataframe(
        target_bigwig_path=str(target_path),
        atlas_bigwig_paths=[str(path) for path in atlas_paths],
        number_of_bins=2,
        chroms=["chr1"],
        test_mode=True,
        jump_sample=2,
        verbose=False,
    )
    assert list(sampled_df["start"]) == [0, 12, 24]

    results = evaluate_atlas_from_bigwigs(
        target_bigwig_path=str(target_path),
        atlas_bigwig_paths=[str(path) for path in atlas_paths],
        number_of_bins=2,
        chroms=["chr1"],
        verbose=False,
    )
    assert len(results) == 2
    for _, pearson_result, mse_result, mae_result in results:
        assert pearson_result["pearsonr"] == pytest.approx(1.0)
        assert mse_result["mse"] == pytest.approx(0.0, abs=1e-6)
        assert mae_result["mae"] == pytest.approx(0.0, abs=5e-4)


def test_atlas_eval_config_generation_uses_bigwig_paths(tmp_path):
    created_configs_path = tmp_path / "generated"
    project_config_path = tmp_path / "project.yaml"
    project_config = {
        "params": {
            "names": ["R3", "T3", "Q3"],
            "bigwig_files": ["/data/R3.bw", "/data/T3.bw", "/data/Q3.bw"],
            "created_configs_path": str(created_configs_path),
            "dataset_base_dir": "/datasets",
            "project_suffix": "atlas_project",
            "base_suffix": "atlas_project",
            "tokenizer_name": "repo/model",
            "model_type": "regression",
            "seq_size": 600,
            "number_of_bins": 4,
            "chromosomes": ["chr1", "chr2"],
        }
    }
    project_config_path.write_text(yaml.safe_dump(project_config), encoding="utf-8")

    written_paths = create_atlas_eval_configs(str(project_config_path))
    assert len(written_paths) == 3

    r3_config = yaml.safe_load(Path(written_paths[0]).read_text(encoding="utf-8"))
    assert r3_config["paths"]["target_bigwig_path"] == "/data/R3.bw"
    assert r3_config["paths"]["atlas_bigwig_paths"] == ["/data/T3.bw", "/data/Q3.bw"]
    assert r3_config["task"]["chromosomes"] == ["chr1", "chr2"]
    assert "dataset_path" not in r3_config["paths"]
    assert "atlas_dataset_paths" not in r3_config["paths"]


def test_atlas_evaluation_skips_hf_dataset_loading(tmp_path, monkeypatch):
    target_path, atlas_paths = _create_sample_bigwigs(tmp_path)

    def _fail_load(*args, **kwargs):
        raise AssertionError("load_from_disk should not be called in atlas mode")

    cfg = {
        "paths": {
            "target_bigwig_path": str(target_path),
            "atlas_bigwig_paths": [str(path) for path in atlas_paths],
        },
        "task": {
            "sub_task": "atlas_evaluation",
            "number_of_bins": 2,
            "chromosomes": ["chr1"],
            "top_rows": -1,
        },
        "model": {},
        "testing_params": {
            "test_mode": True,
            "jump_sample": 2,
        },
        "verbose": False,
    }

    evaluator_module = _import_module("src.evaluator")

    monkeypatch.setattr(evaluator_module, "load_from_disk", _fail_load)
    result_df = evaluator_module.perform_evaluation(cfg)
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 1


def test_tissue_atlas_evaluation_skips_hf_dataset_loading(tmp_path, monkeypatch):
    target_path, atlas_paths = _create_sample_bigwigs(tmp_path)

    def _fail_load(*args, **kwargs):
        raise AssertionError("load_from_disk should not be called in tissue atlas mode")

    cfg = {
        "paths": {
            "target_bigwig_path": str(target_path),
            "atlas_bigwig_paths": [str(path) for path in atlas_paths],
        },
        "task": {
            "sub_task": "atlas_evaluation",
            "number_of_bins": 2,
            "chromosomes": ["chr1"],
            "top_rows": -1,
        },
        "model": {
            "tissue_prompt": {"task_level": "token"},
            "model_repo": "unused",
            "model_name": "unused",
        },
        "testing_params": {
            "test_mode": False,
            "jump_sample": -1,
        },
        "verbose": False,
    }

    tissue_module = _import_module("src.tissue_evaluator")

    monkeypatch.setattr(tissue_module, "load_from_disk", _fail_load)
    result_df = tissue_module.perform_tissue_evaluation(cfg)
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2
