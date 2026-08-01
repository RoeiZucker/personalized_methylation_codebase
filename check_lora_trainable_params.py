# Check trainable parameter counts for the LoRA-over-finetuned training path.
# Example local run:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/check_lora_trainable_params.py
# Example Slurm run:
# sbatch --mem=45g -c15 --gres=gg:g4:1 --constraint=firefoot --time=02:00:00 --output=/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/scripts/outputs/slurm-%j.out --wrap='. /cs/usr/roeizucker/new_storage/new_python_env/bin/activate; cd /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code; python3 check_lora_trainable_params.py'

import os
import sys

import yaml


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(
    REPO_ROOT,
    "configs",
    "wandb_classification_full_training_r3_epoch10_lora_10k_limited_run.yaml",
)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.model_utils import get_base_model


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def count_trainable_parameters(model):
    trainable = 0
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return trainable, total


def print_trainable_summary(model):
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    trainable, total = count_trainable_parameters(model)
    percent = 100 * trainable / total if total else 0
    print(f"trainable parameters: {trainable:,}", flush=True)
    print(f"total parameters:     {total:,}", flush=True)
    print(f"trainable percent:    {percent:.6f}%", flush=True)


def main():
    cfg = load_config()
    model_cfg = cfg["model"]
    trained_model_path = cfg["paths"]["trained_model_path"]

    model = get_base_model(
        use_lora=True,
        freeze_model=True,
        num_labels=model_cfg["num_labels"],
        base_model_name=trained_model_path,
        lora_config=model_cfg.get("lora"),
    )
    print_trainable_summary(model)


if __name__ == "__main__":
    main()
