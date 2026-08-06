# Create downsampled epoch-10 retraining configs for the IUCC ratio-0.25 runs.
# Example, preview configs without writing:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_downsampled_retraining_configs.py --dry-run
# Example, write configs:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_downsampled_retraining_configs.py

import argparse
import copy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = Path("/sci/labs/michall/roeizucker/roei_test/token_cls_project_configs2")
LOCAL_DOWNSAMPLED_ROOT = Path("/sci/labs/michall/roeizucker/roei_test/token_cls_datasets_downsampled")
IUCC_DOWNSAMPLED_ROOT = Path("/shared/cycle1_huji_linial_prj/roei_test/downsampled_datasets_all")
EVAL_TEST_ROOT = Path("/home/users/roeizucker/tests/token_cls_datasets2_eval_test_split_all")
MODEL_OUTPUT_ROOT = Path("/shared/cycle1_huji_linial_prj/roei_test/token_cls_models_downsampled")
CONFIG_OUTPUT_DIR = REPO_ROOT / "configs" / "downsampled_runs" / "generated"
RATIO = "0.25"
LEARNING_RATE = 3e-5
EPOCHS = 3
BATCH_SIZE = 14
SAVE_TOTAL_LIMIT = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Create downsampled retraining YAML configs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def ratio_tag():
    return RATIO.replace(".", "p")


def lr_tag():
    return f"{LEARNING_RATE:.0e}"


def sample_from_dataset(dataset_dir):
    return dataset_dir.name.split("_retrain_token_cls_", 1)[0]


def tissue_from_dir(tissue_dir):
    name = tissue_dir.name.removeprefix("_token_cls_")
    return name.removeprefix("_").removesuffix("_kmer")


def find_downsampled_datasets():
    pattern = f"*datasets_train_ratio_{ratio_tag()}"
    return sorted(LOCAL_DOWNSAMPLED_ROOT.glob(f"_token_cls_*_kmer/{pattern}"))


def config_key(sample, tissue_dir):
    return sample, tissue_from_dir(tissue_dir)


def index_epoch10_configs():
    configs = {}
    pattern = "*epoch-10-step-*retrain_training_token_cls*_lr_1e-06_bs_14_seq_5400.yaml"
    for path in SOURCE_CONFIG_ROOT.rglob(pattern):
        tissue_dir = path.parent.parent.parent.name
        sample = path.name.split("_epoch-10-step-", 1)[0]
        configs[config_key(sample, Path(tissue_dir))] = path
    return configs


def dataset_name(sample, tissue, split):
    return f"{sample}_retrain_token_cls_{tissue}_kmer_seq_5400_datasets_{split}"


def run_name(sample, tissue, checkpoint_name):
    return (
        f"{sample}_{checkpoint_name}_downsampled_ratio_{ratio_tag()}_"
        f"retrain_token_cls_{tissue}_kmer_lr_{lr_tag()}_bs_{BATCH_SIZE}_seq_5400"
    )


def set_paths(cfg, dataset_dir, source_cfg):
    paths = cfg["paths"]
    sample = sample_from_dataset(dataset_dir)
    tissue_dir = dataset_dir.parent.name
    tissue = tissue_from_dir(dataset_dir.parent)
    checkpoint_name = Path(source_cfg["paths"]["trained_model_path"]).name
    name = run_name(sample, tissue, checkpoint_name)

    paths["train_dataset_path"] = [str(IUCC_DOWNSAMPLED_ROOT / tissue_dir / dataset_dir.name)]
    paths["eval_dataset_path"] = str(EVAL_TEST_ROOT / tissue_dir / dataset_name(sample, tissue, "eval"))
    paths["test_dataset_path"] = str(EVAL_TEST_ROOT / tissue_dir / dataset_name(sample, tissue, "test"))
    paths["output_dir"] = str(MODEL_OUTPUT_ROOT / tissue_dir / name)
    return name


def set_training(cfg):
    train = cfg["train"]
    train["num_train_epochs"] = EPOCHS
    train["per_device_train_batch_size"] = BATCH_SIZE
    train["per_device_eval_batch_size"] = BATCH_SIZE
    train["learning_rate"] = LEARNING_RATE
    train["save_stratagy"] = "epoch"
    train["number_of_steps"] = -1
    train["save_total_limit"] = SAVE_TOTAL_LIMIT
    train["add_epoch_end_prediction"] = False
    train["continue_from_last"] = False
    train["eval_steps"] = -1


def set_wandb(cfg):
    cfg["wandb"] = {"enabled": False}


def build_config(source_cfg, dataset_dir):
    cfg = copy.deepcopy(source_cfg)
    name = set_paths(cfg, dataset_dir, source_cfg)
    cfg["task"]["analysis_name"] = name
    cfg["task"]["top_rows"] = -1
    set_training(cfg)
    set_wandb(cfg)
    return cfg, name


def output_path(dataset_dir, source_cfg):
    sample = sample_from_dataset(dataset_dir)
    tissue = tissue_from_dir(dataset_dir.parent)
    checkpoint_name = Path(source_cfg["paths"]["trained_model_path"]).name
    name = run_name(sample, tissue, checkpoint_name)
    return CONFIG_OUTPUT_DIR / dataset_dir.parent.name / f"{name}.yaml"


def create_configs(args):
    source_configs = index_epoch10_configs()
    created = []
    skipped = []

    for dataset_dir in find_downsampled_datasets():
        sample = sample_from_dataset(dataset_dir)
        key = config_key(sample, dataset_dir.parent)
        source_path = source_configs.get(key)
        if source_path is None:
            skipped.append((dataset_dir, "missing epoch-10 source config"))
            continue
        source_cfg = load_yaml(source_path)
        cfg, name = build_config(source_cfg, dataset_dir)
        path = output_path(dataset_dir, source_cfg)
        if not args.dry_run:
            write_yaml(path, cfg)
        created.append((path, name))

    return created, skipped


def print_summary(created, skipped, dry_run):
    action = "would create" if dry_run else "created"
    print(f"{action}: {len(created)} configs", flush=True)
    for path, _ in created:
        print(path, flush=True)
    if skipped:
        print(f"skipped: {len(skipped)}", flush=True)
        for path, reason in skipped:
            print(f"{path}: {reason}", flush=True)


def main():
    args = parse_args()
    created, skipped = create_configs(args)
    print_summary(created, skipped, args.dry_run)


if __name__ == "__main__":
    main()
