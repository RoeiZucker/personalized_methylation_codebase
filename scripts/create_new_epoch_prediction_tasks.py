"""Create Phoenix prediction tasks for new model-only epoch saves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(
    "/sci/labs/michall/roeizucker/roei_test/token_cls_models_downsampled"
)
CONFIG_ROOT = REPO_ROOT / "configs/downsampled_runs/prediction"
TASK_FILE = REPO_ROOT / "scripts/downsampled_new_epoch_prediction_tasks.tsv"
SPLITS = ("eval", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create eval/test prediction tasks for model-only epoch saves."
    )
    parser.add_argument("--include-existing", action="store_true")
    return parser.parse_args()


def model_shards(model_dir: Path) -> list[Path]:
    single_file = model_dir / "model.safetensors"
    if single_file.is_file():
        return [single_file]

    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.is_file():
        return []
    index = json.loads(index_file.read_text(encoding="utf-8"))
    filenames = sorted(set(index.get("weight_map", {}).values()))
    return [model_dir / filename for filename in filenames]


def is_complete_model(model_dir: Path) -> bool:
    shards = model_shards(model_dir)
    return bool(shards) and all(path.is_file() and path.stat().st_size > 0 for path in shards)


def epoch_sort_key(model_dir: Path) -> tuple[str, str, int, int]:
    epoch_text, step_text = model_dir.name.removeprefix("epoch-").split("-step-")
    sample = model_dir.parent.name.split("_epoch-10-step-", 1)[0]
    return model_dir.parent.parent.name, sample, int(epoch_text), int(step_text)


def discover_models() -> list[Path]:
    candidates = MODELS_ROOT.glob("_token_cls_*/*downsampled_ratio_0p25*/epoch-*-step-*")
    return sorted((path for path in candidates if is_complete_model(path)), key=epoch_sort_key)


def prediction_config(sample: str, split: str) -> Path:
    return CONFIG_ROOT / (
        f"{sample}_downsampled_ratio_0p25_lr_3e-05_predict_{split}.yaml"
    )


def validate_config(config_path: Path) -> None:
    if not config_path.is_file():
        raise FileNotFoundError(f"missing prediction config: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset_path = Path(config["paths"]["eval_dataset_path"])
    required_files = (dataset_path / "dataset_info.json", dataset_path / "state.json")
    if not dataset_path.is_dir() or not all(path.is_file() for path in required_files):
        raise FileNotFoundError(f"missing or incomplete prediction dataset: {dataset_path}")


def create_tasks(models: list[Path], include_existing: bool) -> list[tuple[str, ...]]:
    tasks = []
    for model_dir in models:
        sample = model_dir.parent.name.split("_epoch-10-step-", 1)[0]
        epoch = model_dir.name.split("-step-", 1)[0].removeprefix("epoch-")
        for split in SPLITS:
            config_path = prediction_config(sample, split)
            validate_config(config_path)
            output_path = model_dir / f"{split}_predictions.csv.gitbackup"
            if output_path.is_file() and output_path.stat().st_size > 0 and not include_existing:
                continue
            label = f"new:{sample[-2:]}:e{epoch}:{split}"
            tasks.append((str(len(tasks)), label, str(config_path), str(model_dir), str(output_path)))
    return tasks


def write_tasks(tasks: list[tuple[str, ...]]) -> None:
    with TASK_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("task_id", "job_label", "config_path", "checkpoint_path", "output_path"))
        writer.writerows(tasks)


def main() -> None:
    args = parse_args()
    models = discover_models()
    tasks = create_tasks(models, args.include_existing)
    write_tasks(tasks)
    print(f"validated model directories: {len(models)}")
    print(f"prediction tasks written: {len(tasks)}")
    print(f"task file: {TASK_FILE}")


if __name__ == "__main__":
    main()


# Example, create tasks only for missing eval/test predictions:
# python3 scripts/create_new_epoch_prediction_tasks.py
#
# Example, recreate tasks for every valid epoch model, including existing outputs:
# python3 scripts/create_new_epoch_prediction_tasks.py --include-existing
