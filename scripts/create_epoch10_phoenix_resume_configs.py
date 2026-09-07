# Create Phoenix-local configs for resuming every ratio-0.25 run to epoch 10.
# Example:
# python3 scripts/create_epoch10_phoenix_resume_configs.py

from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "configs" / "downsampled_runs" / "generated"
OUTPUT_ROOT = REPO_ROOT / "configs" / "downsampled_runs" / "epoch10_resume_phoenix"
TASK_FILE = (
    REPO_ROOT
    / "configs"
    / "downsampled_runs"
    / "epoch10_resume_phoenix_array_tasks.txt"
)
TRAIN_DATASET_ROOT = Path(
    "/sci/labs/michall/roeizucker/roei_test/token_cls_datasets_downsampled"
)
PRETRAINED_MODEL_ROOT = Path(
    "/sci/labs/michall/roeizucker/roei_test/token_cls_models2"
)
OUTPUT_MODEL_ROOT = Path(
    "/sci/labs/michall/roeizucker/roei_test/token_cls_models_downsampled"
)
HG38_PATH = Path("/sci/archive/michall/roeizucker/reference_genome/hg38.fa")


def load_config(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def local_train_path(paths, tissue):
    dataset_name = Path(paths["train_dataset_path"][0]).name
    return TRAIN_DATASET_ROOT / tissue / dataset_name


def local_pretrained_path(paths, tissue):
    source = Path(paths["trained_model_path"])
    return PRETRAINED_MODEL_ROOT / tissue / source.parent.name / source.name


def local_output_path(paths, tissue):
    return OUTPUT_MODEL_ROOT / tissue / Path(paths["output_dir"]).name


def set_local_paths(config, tissue):
    paths = config["paths"]
    paths["assemblies"]["HG38"] = str(HG38_PATH)
    paths["train_dataset_path"] = [str(local_train_path(paths, tissue))]
    paths["trained_model_path"] = str(local_pretrained_path(paths, tissue))
    paths["output_dir"] = str(local_output_path(paths, tissue))
    paths["eval_dataset_path"] = None
    paths["test_dataset_path"] = None


def set_epoch10_resume_training(config):
    train = config["train"]
    train["num_train_epochs"] = 10
    train["save_stratagy"] = "epoch"
    train["save_total_limit"] = 10
    train["number_of_steps"] = -1
    train["eval_steps"] = -1
    train["continue_from_last"] = True
    train["add_epoch_end_prediction"] = False
    config["wandb"] = {"enabled": False}


def build_config(source_path):
    config = deepcopy(load_config(source_path))
    set_local_paths(config, source_path.parent.name)
    set_epoch10_resume_training(config)
    return config


def write_config(source_path, config):
    destination = OUTPUT_ROOT / source_path.parent.name / source_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return destination


def create_configs():
    destinations = []
    for source_path in sorted(SOURCE_ROOT.rglob("*.yaml")):
        destinations.append(write_config(source_path, build_config(source_path)))
    return destinations


def write_task_file(config_paths):
    lines = [str(path.resolve()) for path in config_paths]
    TASK_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    config_paths = create_configs()
    write_task_file(config_paths)
    print(f"created {len(config_paths)} configs")
    print(f"wrote task list: {TASK_FILE}")


if __name__ == "__main__":
    main()
