"""Create 4,000-step resume configs for the three uncovered training runs."""

from copy import deepcopy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "configs/downsampled_runs/epoch10_resume_phoenix"
OUTPUT_ROOT = REPO_ROOT / "configs/downsampled_runs/save_steps_4000_resume"
TASK_FILE = (
    REPO_ROOT
    / "configs/downsampled_runs/save_steps_4000_resume_missing_array_tasks.txt"
)
SAMPLES = ("Z0000041W", "Z0000043R", "Z0000044M")


def source_config(sample: str) -> Path:
    matches = list(SOURCE_ROOT.rglob(f"{sample}_*.yaml"))
    if len(matches) != 1:
        raise ValueError(f"expected one source config for {sample}, found {len(matches)}")
    return matches[0]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_step_and_epoch_saves(config: dict) -> None:
    train = config["train"]
    train["num_train_epochs"] = 10
    train["save_stratagy"] = "steps"
    train["number_of_steps"] = 4000
    train.pop("save_total_limit", None)
    train["add_epoch_end_save_callback"] = True
    train["add_epoch_end_prediction"] = False
    train["save_at_end"] = False
    train["continue_from_last"] = True
    train["eval_steps"] = -1
    config["paths"]["eval_dataset_path"] = None
    config["paths"]["test_dataset_path"] = None
    config["wandb"] = {"enabled": False}


def build_config(path: Path) -> dict:
    config = deepcopy(load_config(path))
    configure_step_and_epoch_saves(config)
    name = config["task"]["analysis_name"]
    config["task"]["analysis_name"] = f"{name}_save_steps_4000"
    return config


def write_config(source: Path) -> Path:
    destination = OUTPUT_ROOT / source.relative_to(SOURCE_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(build_config(source), handle, sort_keys=False)
    return destination


def main() -> None:
    configs = [write_config(source_config(sample)) for sample in SAMPLES]
    TASK_FILE.write_text(
        "\n".join(str(path.resolve()) for path in configs) + "\n",
        encoding="utf-8",
    )
    print(f"created {len(configs)} configs")
    print(f"wrote task list: {TASK_FILE}")


if __name__ == "__main__":
    main()


# Example, create the three missing 4,000-step resume configs and task list:
# python3 scripts/create_missing_save_steps_4000_configs.py
