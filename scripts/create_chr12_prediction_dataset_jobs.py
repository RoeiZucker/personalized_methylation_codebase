"""Create chr12-only prediction-dataset configs and Phoenix jobs for every sample.

Example, inspect how many configs and jobs would be created:
python3 scripts/create_chr12_prediction_dataset_jobs.py --dry-run

Example, write all chr12 extraction configs and the explicit sbatch command file:
python3 scripts/create_chr12_prediction_dataset_jobs.py

Example, verify that every raw BigWig is visible before writing:
python3 scripts/create_chr12_prediction_dataset_jobs.py --require-raw-data

Example, include jobs for chr12 datasets that already exist:
python3 scripts/create_chr12_prediction_dataset_jobs.py --include-existing
"""

from __future__ import annotations

import argparse
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
SOURCE_CONFIG_ROOT = Path(
    "/sci/labs/michall/roeizucker/roei_test/token_cls_project_configs2"
)
DATASET_ROOT = Path(
    "/sci/labs/michall/roeizucker/roei_test/"
    "token_cls_datasets2_chr12_prediction_v1"
)
CONFIG_ROOT = REPO_ROOT / "configs/downsampled_runs/chr12_prediction_dataset_v1"
COMMAND_FILE = REPO_ROOT / "scripts/downsampled_chr12_prediction_dataset_v1_sbatch_commands.sh"
VERSION = "chr12_prediction_v1"
RAW_DATA_ROOT = Path("/sci/labs/michall/roeizucker/roei_test/downloaded_datasets")
HG38_FASTA = Path("/sci/archive/michall/roeizucker/reference_genome/hg38.fa")


@dataclass(frozen=True)
class SampleInfo:
    sample: str
    tissue_dir: str
    tissue: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create chr12-only prediction dataset configs and Phoenix jobs."
    )
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--source-config-root", default=str(SOURCE_CONFIG_ROOT))
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--config-root", default=str(CONFIG_ROOT))
    parser.add_argument("--command-file", default=str(COMMAND_FILE))
    parser.add_argument("--phoenix-repo-root", default=str(PHOENIX_REPO_ROOT))
    parser.add_argument("--require-raw-data", action="store_true")
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def tissue_name(tissue_dir: str) -> str:
    return tissue_dir.removeprefix("_token_cls_").removesuffix("_kmer")


def discover_samples(runs_root: Path) -> list[SampleInfo]:
    samples = []
    seen = set()
    for run in sorted(runs_root.glob("_token_cls_*/*downsampled_ratio_0p25*")):
        if not run.is_dir() or "smoke_top400" in run.name or "bad_partial" in run.name:
            continue
        sample = run.name.split("_epoch-10-step-", 1)[0]
        if sample in seen:
            raise ValueError(f"duplicate downsampled run for sample {sample}")
        seen.add(sample)
        tissue_dir = run.parent.name
        samples.append(SampleInfo(sample, tissue_dir, tissue_name(tissue_dir)))
    if not samples:
        raise ValueError(f"no ratio-0.25 downsampled runs found under {runs_root}")
    return samples


def source_config_path(root: Path, info: SampleInfo) -> Path:
    project = f"_token_cls_{info.tissue}_kmer_lr_1e-06_bs_14_seq_5400"
    name = (
        f"{info.sample}_retrain_data_extraction_token_cls_"
        f"{info.tissue}_kmer_lr_1e-06_bs_14_seq_5400.yaml"
    )
    return root / f"_{info.tissue}_kmer" / project / "retrain_extraction" / name


def dataset_name(info: SampleInfo) -> str:
    return (
        f"{info.sample}_retrain_token_cls_{info.tissue}_kmer_"
        "seq_5400_datasets_test"
    )


def dataset_path(root: Path, info: SampleInfo) -> Path:
    return root / info.tissue_dir / dataset_name(info)


def intermediate_path(root: Path, info: SampleInfo, suffix: str) -> Path:
    name = (
        f"{info.sample}_retrain_token_cls_{info.tissue}_kmer_seq_5400_"
        f"{VERSION}_{suffix}.csv"
    )
    return root / info.tissue_dir / name


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_fixed_binning(config: dict, path: Path) -> None:
    method = config["task"].get("token_label_binning", {}).get("method")
    if method != "fixed":
        raise ValueError(f"chr12 prediction extraction requires fixed label bins: {path}")


def build_config(source: dict, root: Path, info: SampleInfo) -> dict:
    config = copy.deepcopy(source)
    paths = config["paths"]
    paths["raw_data_path"] = str(RAW_DATA_ROOT / Path(paths["raw_data_path"]).name)
    paths.setdefault("assemblies", {})["HG38"] = str(HG38_FASTA)
    paths["intermediate_train_data_path"] = str(
        intermediate_path(root, info, "intermediate")
    )
    paths["intermediate_test_data_path"] = str(intermediate_path(root, info, "empty"))
    paths["hf_dataset_test_path"] = str(dataset_path(root, info))
    paths.pop("hf_dataset_train_path", None)
    paths.pop("hf_dataset_eval_path", None)

    task = config["task"]
    task["chromosomes"] = ["chr12"]
    task["prediction_only"] = True
    task["test_size"] = 0
    task["eval_size"] = 0
    task["output_preprocessed_data"] = False
    task["output_intermediate_data"] = True
    task["output_hf_dataset"] = True
    task["override_dataset"] = False
    task["clear_generator_cache"] = False
    return config


def config_name(info: SampleInfo) -> str:
    return f"{info.sample}_{VERSION}_data_extraction.yaml"


def write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def phoenix_path(path: Path, phoenix_root: Path) -> Path:
    return phoenix_root / path.relative_to(REPO_ROOT)


def job_name(info: SampleInfo) -> str:
    return f"chr12:{info.sample[-2:]}"


def sbatch_command(info: SampleInfo, config_path: Path, phoenix_root: Path) -> str:
    wrapper = phoenix_root / "scripts/run_chr12_prediction_dataset_creation.sh"
    config = phoenix_path(config_path, phoenix_root)
    return (
        "sbatch --mem=64g -c8 --time=1-00 --killable --requeue "
        f'--job-name="{job_name(info)}" --wrap="{wrapper} {config}"'
    )


def write_commands(path: Path, commands: list[str]) -> None:
    header = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Explicit Phoenix jobs for {VERSION}; generated by {Path(__file__).name}.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + commands) + "\n", encoding="utf-8")
    path.chmod(0o755)


def require_raw_data(config: dict, source_path: Path) -> None:
    raw_path = Path(config["paths"]["raw_data_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"raw BigWig from {source_path} is not visible: {raw_path}")


def create_jobs(args: argparse.Namespace) -> None:
    samples = discover_samples(Path(args.runs_root))
    source_root = Path(args.source_config_root)
    dataset_root = Path(args.dataset_root)
    config_root = Path(args.config_root)
    phoenix_root = Path(args.phoenix_repo_root)
    configs = []
    commands = []

    for info in samples:
        source_path = source_config_path(source_root, info)
        if not source_path.is_file():
            raise FileNotFoundError(f"missing extraction config: {source_path}")
        source = load_yaml(source_path)
        validate_fixed_binning(source, source_path)
        config = build_config(source, dataset_root, info)
        if args.require_raw_data:
            require_raw_data(config, source_path)
        output_config = config_root / config_name(info)
        configs.append(output_config)
        if not args.dry_run:
            write_yaml(output_config, config)
        output_dataset = dataset_path(dataset_root, info)
        if args.include_existing or not output_dataset.is_dir():
            commands.append(sbatch_command(info, output_config, phoenix_root))

    print(
        f"samples={len(samples)} configs={len(configs)} jobs={len(commands)}",
        flush=True,
    )
    if not args.dry_run:
        write_commands(Path(args.command_file), commands)
        print(f"wrote commands: {args.command_file}", flush=True)


def main() -> None:
    create_jobs(parse_args())


if __name__ == "__main__":
    main()
