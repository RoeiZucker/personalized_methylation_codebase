"""Split existing Hugging Face test datasets into separate eval and test datasets.

Example run for the liver W&B epoch-10 configs:
python3 scripts/split_test_datasets_to_eval_and_test.py

Example dry run:
python3 scripts/split_test_datasets_to_eval_and_test.py --dry-run

Example run with custom paths:
python3 scripts/split_test_datasets_to_eval_and_test.py --config-dir configs/my_configs --output-root /sci/labs/michall/roeizucker/roei_test/my_eval_test_split
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from datasets import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs/wandb_liver_token_cls_epoch10_single_retrain_epoch"
DEFAULT_OUTPUT_ROOT = Path("/sci/labs/michall/roeizucker/roei_test/token_cls_datasets2_eval_test_split")
DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split config eval/test datasets into separate eval and test datasets.")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR), help="Directory containing YAML configs to read.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for the new split datasets.")
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION, help="Fraction of rows to save as eval.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic row split seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output datasets if they already exist.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip datasets when both split outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned split paths without writing datasets.")
    return parser.parse_args()


def read_eval_dataset_path(config_path: Path) -> Path | None:
    config = yaml.safe_load(config_path.read_text())
    dataset_path = config.get("paths", {}).get("eval_dataset_path")
    if dataset_path is None:
        return None
    if isinstance(dataset_path, list):
        raise ValueError(f"expected one eval_dataset_path in {config_path}, got a list")
    return Path(dataset_path)


def config_dataset_paths(config_dir: Path) -> list[Path]:
    paths = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        dataset_path = read_eval_dataset_path(config_path)
        if dataset_path is not None:
            paths.append(dataset_path)
    return sorted(set(paths))


def output_paths(input_path: Path, output_root: Path) -> tuple[Path, Path]:
    if input_path.name.endswith("_datasets_test"):
        base_name = input_path.name[: -len("_datasets_test")]
        test_name = f"{base_name}_datasets_test"
        eval_name = f"{base_name}_datasets_eval"
    else:
        test_name = f"{input_path.name}_test"
        eval_name = f"{input_path.name}_eval"
    parent_name = input_path.parent.name
    output_dir = output_root / parent_name
    return output_dir / test_name, output_dir / eval_name


def validate_args(eval_fraction: float) -> None:
    if not 0 < eval_fraction < 1:
        raise ValueError("--eval-fraction must be between 0 and 1")


def existing_outputs(test_path: Path, eval_path: Path) -> list[Path]:
    return [path for path in (test_path, eval_path) if path.exists()]


def ensure_outputs_available(test_path: Path, eval_path: Path, overwrite: bool, skip_existing: bool) -> bool:
    existing = existing_outputs(test_path, eval_path)
    if not existing:
        return True
    if skip_existing and len(existing) == 2:
        return False
    if not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output dataset already exists: {paths}. Use --overwrite or --skip-existing.")
    for path in existing:
        shutil.rmtree(path)
    return True


def split_dataset(dataset: "Dataset", eval_fraction: float, seed: int) -> tuple["Dataset", "Dataset"]:
    split = dataset.train_test_split(test_size=eval_fraction, seed=seed)
    return split["train"], split["test"]


def split_one_dataset(
    input_path: Path,
    output_root: Path,
    eval_fraction: float,
    seed: int,
    overwrite: bool,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"missing input dataset: {input_path}")
    test_path, eval_path = output_paths(input_path, output_root)
    existing = existing_outputs(test_path, eval_path)

    print(f"{input_path}")
    print(f"  test -> {test_path}")
    print(f"  eval -> {eval_path}")
    if existing:
        print(f"  existing outputs: {len(existing)}/2")
    if dry_run:
        return
    should_write = ensure_outputs_available(test_path, eval_path, overwrite, skip_existing)
    if not should_write:
        print("  skipped existing split outputs")
        return

    from datasets import load_from_disk

    dataset = load_from_disk(str(input_path))
    test_dataset, eval_dataset = split_dataset(dataset, eval_fraction, seed)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_dataset.save_to_disk(str(test_path))
    eval_dataset.save_to_disk(str(eval_path))
    print(f"  rows: original={len(dataset)} test={len(test_dataset)} eval={len(eval_dataset)}")


def main() -> None:
    args = parse_args()
    validate_args(args.eval_fraction)
    input_paths = config_dataset_paths(Path(args.config_dir))
    if not input_paths:
        raise ValueError(f"no eval_dataset_path entries found in {args.config_dir}")
    for input_path in input_paths:
        split_one_dataset(
            input_path,
            Path(args.output_root),
            args.eval_fraction,
            args.seed,
            args.overwrite,
            args.skip_existing,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
