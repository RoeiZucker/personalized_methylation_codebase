"""Create W&B configs for liver epoch-10 token-classification retraining.

Example run:
python3 scripts/create_wandb_liver_epoch10_retrain_configs.py

Example run with a custom output config directory:
python3 scripts/create_wandb_liver_epoch10_retrain_configs.py --config-dir configs/my_liver_epoch10_configs
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import yaml

PARTICIPANTS = [
    "Z000000R3",
    "Z000000T3",
    "Z0000043Q",
    "Z0000044H",
    "Z0000044M",
    "Z00000431",
]

PROJECT_NAME = "token_cls_Liver-Hepatocytes_kmer"
CONFIG_DIR = Path("configs/wandb_liver_token_cls_epoch10_single_retrain_epoch")
DATASET_BASE_DIR = Path("/sci/labs/michall/roeizucker/roei_test/token_cls_datasets2/_token_cls_Liver-Hepatocytes_kmer")
MODEL_BASE_DIR = Path("/sci/labs/michall/roeizucker/roei_test/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer")
OUTPUT_BASE_DIR = Path("/sci/labs/michall/roeizucker/roei_test/wandb_token_cls_models/liver_epoch10_single_retrain_epoch")
WANDB_DIR = "/sci/labs/michall/roeizucker/roei_test/wandb_runs/liver_epoch10_single_retrain_epoch"
TRAIN_SCRIPT = "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/run_training_params.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create liver epoch-10 W&B retraining configs.")
    parser.add_argument("--config-dir", default=str(CONFIG_DIR), help="Directory for generated YAML files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated configs.")
    return parser.parse_args()


def checkpoint_path(participant: str) -> str:
    pattern = MODEL_BASE_DIR / f"{participant}_pretrain_{PROJECT_NAME}_lr_1e-06_bs_14_seq_5400" / "epoch-10-step-*"
    matches = sorted(glob.glob(str(pattern)))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one epoch-10 checkpoint for {participant}, found {len(matches)}: {matches}")
    return matches[0]


def dataset_path(participant: str, split: str) -> str:
    path = DATASET_BASE_DIR / f"{participant}_retrain_{PROJECT_NAME}_seq_5400_datasets_{split}"
    if not path.exists():
        raise FileNotFoundError(f"missing {split} dataset for {participant}: {path}")
    return str(path)


def analysis_name(participant: str, checkpoint: str) -> str:
    checkpoint_name = Path(checkpoint).name
    return f"{participant}_{checkpoint_name}_wandb_retrain_{PROJECT_NAME}_lr_1e-06_bs_14_seq_5400_single_epoch"


def config_for_participant(participant: str) -> dict[str, Any]:
    checkpoint = checkpoint_path(participant)
    name = analysis_name(participant, checkpoint)
    return {
        "paths": {
            "assemblies": {
                "HG38": "/sci/archive/michall/roeizucker/reference_genome/hg38.fa",
                "MM10": "/sci/archive/michall/roeizucker/reference_genome/mm10.fa",
            },
            "generetor_cache_dir": "/sci/archive/michall/roeizucker/huggingface_modles_cache/datasets/generator",
            "train_dataset_path": [dataset_path(participant, "train")],
            "eval_dataset_path": dataset_path(participant, "test"),
            "trained_model_path": checkpoint,
            "output_dir": str(OUTPUT_BASE_DIR / name),
        },
        "task": {
            "task_type": "cpg_retraining",
            "analysis_name": name,
            "top_rows": -1,
            "label_transform": "none",
        },
        "testing_params": {"test_mode": False},
        "random_state": 42,
        "verbose": True,
        "model": {
            "model_name": "nucleotide-transformer-2.5b-multi-species",
            "model_repo": "InstaDeepAI",
            "model_type": "classification_analysis",
            "blank_label": -100,
            "use_lora": False,
            "freeze_model": False,
            "num_labels": 3,
            "lora_over_finetuned": False,
        },
        "train": {
            "load_best_model_at_end": False,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 14,
            "per_device_eval_batch_size": 14,
            "learning_rate": 1.0e-6,
            "metric_for_best_model": "mcc",
            "save_stratagy": "steps",
            "lora_over_finetuned": False,
            "load_dataset_to_memory": True,
            "min_number_of_cpg_sites": 10,
            "max_grad_norm": 1,
            "number_of_steps": 10000,
            "eval_steps": 1000,
            "logging_steps": 500,
            "save_total_limit": 2,
            "add_epoch_end_save_callback": False,
            "add_epoch_end_prediction": True,
            "save_at_end": False,
            "continue_from_last": True,
        },
        "wandb": {
            "enabled": True,
            "project": "personalized-methylation-liver-retraining",
            "entity": None,
            "name": name,
            "group": "liver-token-cls-epoch10-single-retrain-epoch",
            "tags": [
                "liver",
                "token-classification",
                "retraining",
                "epoch-10-pretraining",
                "single-retrain-epoch",
                participant,
            ],
            "mode": "online",
            "dir": WANDB_DIR,
        },
    }


def validate_config(config: dict[str, Any], filename: str) -> None:
    text_values = [filename, config["task"]["analysis_name"], config["paths"]["trained_model_path"]]
    if any("no_pretraining" in value for value in text_values):
        raise ValueError(f"no_pretraining is not allowed in {filename}")
    if "pretrain_prediction" in config["paths"]["trained_model_path"]:
        raise ValueError(f"pretrain_prediction checkpoint is not allowed in {filename}")
    if config["train"]["num_train_epochs"] != 1:
        raise ValueError(f"expected a single retraining epoch in {filename}")


def write_config(config_dir: Path, participant: str, overwrite: bool) -> Path:
    config = config_for_participant(participant)
    path = config_dir / f"{config['task']['analysis_name']}.yaml"
    validate_config(config, path.name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"config already exists: {path}. Use --overwrite to replace it.")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def write_submit_script(config_dir: Path, config_paths: list[Path]) -> Path:
    script_path = config_dir / "submit_epoch10_single_retrain_epoch.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for path in config_paths:
        lines.append(
            "sbatch -A tomhope --mem=45g -c15 --gres=gg:g4:6 --time=5-23 "
            f"--wrap=\"{TRAIN_SCRIPT} {path}\""
        )
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = [write_config(config_dir, participant, args.overwrite) for participant in PARTICIPANTS]
    submit_script = write_submit_script(config_dir, paths)
    for path in paths:
        print(path)
    print(submit_script)


if __name__ == "__main__":
    main()
