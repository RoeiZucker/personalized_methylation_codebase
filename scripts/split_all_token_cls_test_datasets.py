"""Split every token-classification test dataset under a root into test/eval datasets.

Example dry run:
python3 scripts/split_all_token_cls_test_datasets.py --dry-run

Example run for all token-classification datasets:
python3 scripts/split_all_token_cls_test_datasets.py --skip-existing

Example run with custom paths:
python3 scripts/split_all_token_cls_test_datasets.py --input-root /sci/labs/michall/roeizucker/roei_test/token_cls_datasets2 --output-root /sci/labs/michall/roeizucker/roei_test/token_cls_datasets2_eval_test_split_all --skip-existing
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset


DEFAULT_INPUT_ROOT = Path("/sci/labs/michall/roeizucker/roei_test/token_cls_datasets2")
DEFAULT_OUTPUT_ROOT = Path("/sci/labs/michall/roeizucker/roei_test/token_cls_datasets2_eval_test_split_all")
DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split all token-classification test datasets into test/eval datasets.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Root containing token-classification datasets.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root directory for the new split datasets.")
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION, help="Fraction of rows to save as eval.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic row split seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output datasets if they already exist.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip datasets when both split outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned split paths without writing datasets.")
    return parser.parse_args()


def validate_args(input_root: Path, eval_fraction: float) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"missing input root: {input_root}")
    if not 0 < eval_fraction < 1:
        raise ValueError("--eval-fraction must be between 0 and 1")


def is_huggingface_dataset(path: Path) -> bool:
    return (path / "dataset_info.json").exists() or (path / "state.json").exists()


def test_dataset_paths(input_root: Path) -> list[Path]:
    paths = []
    for path in sorted(input_root.glob("*/*_datasets_test")):
        if path.is_dir() and is_huggingface_dataset(path):
            paths.append(path)
    return paths


def split_output_paths(input_path: Path, input_root: Path, output_root: Path) -> tuple[Path, Path]:
    relative_parent = input_path.parent.relative_to(input_root)
    base_name = input_path.name[: -len("_datasets_test")]
    output_dir = output_root / relative_parent
    return output_dir / f"{base_name}_datasets_test", output_dir / f"{base_name}_datasets_eval"


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
    input_root: Path,
    output_root: Path,
    eval_fraction: float,
    seed: int,
    overwrite: bool,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    test_path, eval_path = split_output_paths(input_path, input_root, output_root)
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


def split_all_datasets(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    validate_args(input_root, args.eval_fraction)

    paths = test_dataset_paths(input_root)
    if not paths:
        raise ValueError(f"no Hugging Face *_datasets_test directories found under {input_root}")

    print(f"found {len(paths)} test datasets")
    for input_path in paths:
        split_one_dataset(
            input_path,
            input_root,
            output_root,
            args.eval_fraction,
            args.seed,
            args.overwrite,
            args.skip_existing,
            args.dry_run,
        )


def main() -> None:
    split_all_datasets(parse_args())


if __name__ == "__main__":
    main()
