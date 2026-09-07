"""Create prediction configs and a Phoenix array job for standardized chr12 splits.

Example, preview the number of configs and missing prediction jobs:
python3 scripts/create_standardized_prediction_jobs.py --dry-run

Example, write configs, the array task table, and the single sbatch command:
python3 scripts/create_standardized_prediction_jobs.py

Example, require every standardized dataset to exist before writing jobs:
python3 scripts/create_standardized_prediction_jobs.py --require-datasets

Example, include jobs whose versioned prediction output already exists:
python3 scripts/create_standardized_prediction_jobs.py --include-existing
"""

from __future__ import annotations

import argparse
import csv
import copy
from dataclasses import dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PHOENIX_REPO_ROOT = Path(
    "/sci/nosnap/michall/roeizucker/jupyter_notebooks/"
    "Tom_Hope_Project/refactored_code"
)
RUNS_ROOT = Path("/sci/labs/michall/roeizucker/roei_test/token_cls_models_downsampled")
PRETRAIN_ROOT = Path("/sci/labs/michall/roeizucker/roei_test/token_cls_models2")
DATASET_ROOT = Path(
    "/sci/labs/michall/roeizucker/roei_test/"
    "token_cls_datasets2_chr12_eval_test_standardized_v1"
)
SOURCE_CONFIG_ROOT = REPO_ROOT / "configs/downsampled_runs/prediction"
CONFIG_ROOT = REPO_ROOT / "configs/downsampled_runs/prediction_chr12_standardized_v1"
COMMAND_FILE = REPO_ROOT / "scripts/downsampled_prediction_chr12_standardized_v1_sbatch_commands.sh"
TASK_FILE = REPO_ROOT / "scripts/downsampled_prediction_chr12_standardized_v1_array_tasks.txt"
ARRAY_RUNNER = REPO_ROOT / "scripts/run_downsampled_prediction_chr12_standardized_v1_array.sh"
WRAPPER = PHOENIX_REPO_ROOT / "scripts/run_downsample_checkpoint_prediction_params.sh"
VERSION = "chr12_standardized_split_v1"
STANDARD_H_CHECKPOINTS = {"checkpoint-16448", "checkpoint-32896", "checkpoint-49344"}
DEFAULT_MAX_CONCURRENT = 20


@dataclass(frozen=True)
class RunInfo:
    sample: str
    tissue_dir: str
    tissue: str
    run_path: Path
    pretrain_path: Path
    checkpoints: tuple[Path, ...]


@dataclass(frozen=True)
class PredictionTask:
    job_label: str
    config_path: Path
    checkpoint_path: Path
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create standardized chr12 configs and one Phoenix array job."
    )
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--pretrain-root", default=str(PRETRAIN_ROOT))
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--source-config-root", default=str(SOURCE_CONFIG_ROOT))
    parser.add_argument("--config-root", default=str(CONFIG_ROOT))
    parser.add_argument("--command-file", default=str(COMMAND_FILE))
    parser.add_argument("--task-file", default=str(TASK_FILE))
    parser.add_argument("--array-runner", default=str(ARRAY_RUNNER))
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--phoenix-repo-root", default=str(PHOENIX_REPO_ROOT))
    parser.add_argument("--require-datasets", action="store_true")
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def tissue_name(tissue_dir: str) -> str:
    return tissue_dir.removeprefix("_token_cls_").removesuffix("_kmer")


def numeric_checkpoints(run_path: Path, sample: str) -> tuple[Path, ...]:
    checkpoints = sorted(
        (path for path in run_path.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if sample == "Z0000044H":
        checkpoints = [path for path in checkpoints if path.name in STANDARD_H_CHECKPOINTS]
    if len(checkpoints) != 3:
        raise ValueError(f"expected three standardized checkpoints in {run_path}")
    return tuple(checkpoints)


def find_pretrain_checkpoint(
    pretrain_root: Path, tissue_dir: str, tissue: str, sample: str
) -> Path:
    model_dir = (
        pretrain_root
        / tissue_dir
        / f"{sample}_pretrain_token_cls_{tissue}_kmer_lr_1e-06_bs_14_seq_5400"
    )
    matches = sorted(path for path in model_dir.glob("epoch-10-step-*") if path.is_dir())
    if len(matches) != 1:
        raise ValueError(f"expected one epoch-10 checkpoint for {sample}, found {len(matches)}")
    return matches[0]


def discover_runs(runs_root: Path, pretrain_root: Path) -> list[RunInfo]:
    runs = []
    seen_samples = set()
    for run_path in sorted(runs_root.glob("_token_cls_*/*downsampled_ratio_0p25*")):
        if not run_path.is_dir() or "smoke_top400" in run_path.name or "bad_partial" in run_path.name:
            continue
        sample = run_path.name.split("_epoch-10-step-", 1)[0]
        if sample in seen_samples:
            raise ValueError(f"duplicate downsampled run for sample {sample}")
        seen_samples.add(sample)
        tissue_dir = run_path.parent.name
        tissue = tissue_name(tissue_dir)
        runs.append(
            RunInfo(
                sample,
                tissue_dir,
                tissue,
                run_path,
                find_pretrain_checkpoint(pretrain_root, tissue_dir, tissue, sample),
                numeric_checkpoints(run_path, sample),
            )
        )
    if not runs:
        raise ValueError(f"no ratio-0.25 downsampled runs found under {runs_root}")
    return runs


def dataset_path(dataset_root: Path, run: RunInfo, split: str) -> Path:
    name = (
        f"{run.sample}_retrain_token_cls_{run.tissue}_kmer_"
        f"seq_5400_datasets_{split}"
    )
    return dataset_root / run.tissue_dir / name


def config_name(sample: str, split: str) -> str:
    return (
        f"{sample}_downsampled_ratio_0p25_lr_3e-05_"
        f"{VERSION}_predict_{split}.yaml"
    )


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_prediction_config(source: dict, run: RunInfo, split_path: Path, split: str) -> dict:
    config = copy.deepcopy(source)
    training = config.pop("train", {})
    paths = config["paths"]
    paths["train_dataset_path"] = None
    paths["eval_dataset_path"] = str(split_path)
    paths["test_dataset_path"] = None
    paths["trained_model_path"] = str(run.pretrain_path)
    paths["output_dir"] = str(run.run_path)
    config["task"]["analysis_name"] = (
        f"{run.sample}_downsampled_ratio_0p25_lr_3e-05_{VERSION}_predict_{split}"
    )
    config["task"]["top_rows"] = -1
    config["prediction"] = {
        "load_dataset_to_memory": training.get("load_dataset_to_memory", False),
        "per_device_eval_batch_size": training.get("per_device_eval_batch_size", 1),
    }
    config["wandb"] = {"enabled": False}
    return config


def write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def source_config_path(source_root: Path, sample: str) -> Path:
    return source_root / f"{sample}_downsampled_ratio_0p25_lr_3e-05_predict_eval.yaml"


def create_configs(
    runs: list[RunInfo], dataset_root: Path, source_root: Path, config_root: Path, dry_run: bool
) -> dict[tuple[str, str], Path]:
    paths = {}
    for run in runs:
        source_path = source_config_path(source_root, run.sample)
        if not source_path.exists():
            raise FileNotFoundError(f"missing source prediction config: {source_path}")
        source = load_yaml(source_path)
        for split in ("eval", "test"):
            output_path = config_root / config_name(run.sample, split)
            split_path = dataset_path(dataset_root, run, split)
            config = build_prediction_config(source, run, split_path, split)
            if not dry_run:
                write_yaml(output_path, config)
            paths[(run.sample, split)] = output_path
    return paths


def prediction_output(target: Path, split: str) -> Path:
    return target / VERSION / f"{split}_predictions.csv.gitbackup"


def short_sample(sample: str) -> str:
    return sample[-2:]


def job_name(run: RunInfo, target: Path, split: str) -> str:
    stage = "pre" if target.name == "pretrain_epoch10" else target.name.split("-")[-1]
    return f"std:{short_sample(run.sample)}:{stage}:{split}"


def phoenix_path(path: Path, phoenix_repo_root: Path) -> Path:
    return phoenix_repo_root / path.relative_to(REPO_ROOT)


def prediction_task(
    run: RunInfo,
    target: Path,
    checkpoint: Path,
    split: str,
    config_path: Path,
    phoenix_repo_root: Path,
) -> PredictionTask:
    return PredictionTask(
        job_label=job_name(run, target, split),
        config_path=phoenix_path(config_path, phoenix_repo_root),
        checkpoint_path=checkpoint,
        output_path=prediction_output(target, split),
    )


def run_targets(run: RunInfo) -> list[tuple[Path, Path]]:
    targets = [(run.run_path / "pretrain_epoch10", run.pretrain_path)]
    targets.extend((checkpoint, checkpoint) for checkpoint in run.checkpoints)
    return targets


def create_tasks(
    runs: list[RunInfo],
    config_paths: dict[tuple[str, str], Path],
    phoenix_repo_root: Path,
    include_existing: bool,
) -> list[PredictionTask]:
    tasks = []
    for run in runs:
        for target, checkpoint in run_targets(run):
            for split in ("eval", "test"):
                output_path = prediction_output(target, split)
                if output_path.exists() and not include_existing:
                    continue
                tasks.append(
                    prediction_task(
                        run,
                        target,
                        checkpoint,
                        split,
                        config_paths[(run.sample, split)],
                        phoenix_repo_root,
                    )
                )
    return tasks


def write_task_file(path: Path, tasks: list[PredictionTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["task_id", "job_label", "config_path", "checkpoint_path", "output_path"]
        )
        for task_id, task in enumerate(tasks):
            writer.writerow(
                [
                    task_id,
                    task.job_label,
                    task.config_path,
                    task.checkpoint_path,
                    task.output_path,
                ]
            )


def array_submission_command(
    task_count: int,
    task_file: Path,
    array_runner: Path,
    phoenix_repo_root: Path,
    max_concurrent: int,
) -> str | None:
    if task_count == 0:
        return None
    if max_concurrent < 1:
        raise ValueError("--max-concurrent must be at least 1")
    phoenix_tasks = phoenix_path(task_file, phoenix_repo_root)
    phoenix_runner = phoenix_path(array_runner, phoenix_repo_root)
    array_range = f"0-{task_count - 1}%{max_concurrent}"
    return f"sbatch --array={array_range} {phoenix_runner} {phoenix_tasks}"


def write_command_file(path: Path, command: str | None) -> None:
    header = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# One Phoenix array submission for {VERSION}; generated by {Path(__file__).name}.",
    ]
    lines = header + ([command] if command else ["# No unfinished prediction tasks."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def require_datasets(runs: list[RunInfo], dataset_root: Path) -> None:
    missing = [
        dataset_path(dataset_root, run, split)
        for run in runs
        for split in ("eval", "test")
        if not dataset_path(dataset_root, run, split).is_dir()
    ]
    if missing:
        details = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing {len(missing)} standardized datasets:\n{details}")


def create_jobs(args: argparse.Namespace) -> None:
    runs = discover_runs(Path(args.runs_root), Path(args.pretrain_root))
    dataset_root = Path(args.dataset_root)
    if args.require_datasets:
        require_datasets(runs, dataset_root)
    config_paths = create_configs(
        runs,
        dataset_root,
        Path(args.source_config_root),
        Path(args.config_root),
        args.dry_run,
    )
    tasks = create_tasks(
        runs,
        config_paths,
        Path(args.phoenix_repo_root),
        args.include_existing,
    )
    print(f"runs={len(runs)} configs={len(config_paths)} array_tasks={len(tasks)}", flush=True)
    if not args.dry_run:
        task_file = Path(args.task_file)
        command = array_submission_command(
            len(tasks),
            task_file,
            Path(args.array_runner),
            Path(args.phoenix_repo_root),
            args.max_concurrent,
        )
        write_task_file(task_file, tasks)
        write_command_file(Path(args.command_file), command)
        print(f"wrote array tasks: {task_file}", flush=True)
        print(f"wrote array submission: {args.command_file}", flush=True)


def main() -> None:
    create_jobs(parse_args())


if __name__ == "__main__":
    main()
