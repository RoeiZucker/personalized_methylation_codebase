import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(
    0,
    os.path.abspath("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code"),
)
sys.path.insert(
    0,
    os.path.abspath("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/src"),
)

import src.utils.variability_utils as variability_utils
from src.config_manager import create_project_config


def test_load_variability_dict_skips_csv_load_when_filtering_disabled(monkeypatch):
    def fail_read_csv(*args, **kwargs):
        raise AssertionError("read_csv should not be called when filtering is disabled")

    monkeypatch.setattr(variability_utils.pd, "read_csv", fail_read_csv)

    assert variability_utils.load_variability_dict(None, use_variant_filtering=False) is None


def test_load_variability_dict_requires_path_when_filtering_enabled():
    with pytest.raises(ValueError, match="variant_file_path is required"):
        variability_utils.load_variability_dict(None, use_variant_filtering=True)


def test_create_project_config_omits_pretrain_variability_path_when_filtering_disabled(tmp_path):
    created_configs_path = tmp_path / "generated_configs"
    dataset_base_dir = tmp_path / "datasets"
    base_model_location = tmp_path / "models"

    create_project_config(
        project_suffix="_unit_no_var_pretrain",
        bigwig_files=["/tmp/A.bigwig", "/tmp/B.bigwig"],
        names=["A", "B"],
        created_configs_path=str(created_configs_path),
        tokenizer_name="InstaDeepAI/nucleotide-transformer-500m-1000g",
        dataset_base_dir=str(dataset_base_dir),
        base_model_location=str(base_model_location),
        model_type="regression_analysis",
        use_lora=False,
        freeze_model=False,
        num_labels=1,
        load_best_model_at_end=False,
        num_train_epoch=1,
        num_pretrain_epoch=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-6,
        metric_for_best_model="mse",
        save_stratagy="epoch",
        number_of_steps=10,
        save_total_limit=1,
        add_epoch_end_save_callback=False,
        save_at_end=False,
        continue_from_last=False,
        use_variant_filtering=False,
        variant_filtering_upper_bound=-1,
        variant_filtering_lower_bound=-1,
        chromosomes=["chr1"],
        seq_size=600,
        number_of_bins=5,
        test_size=0.2,
        load_dataset_to_memory=False,
        override_dataset=True,
    )

    pretrain_configs = sorted((created_configs_path / "pretrain_extraction").glob("*.yaml"))
    variability_configs = sorted((created_configs_path / "variability_extraction").glob("*.yaml"))

    assert pretrain_configs, "expected pretrain extraction configs to be generated"
    assert variability_configs, "expected downstream variability configs to still be generated"

    for config_path in pretrain_configs:
        cfg = yaml.safe_load(Path(config_path).read_text())
        assert cfg["variant_filtering"]["use_variant_filtering"] is False
        assert "variant_file_path" not in cfg["paths"]
