"""Create task rows for downsampling saved token-classification datasets.

Example retrain ratio 0.25 task file:
python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_downsample_dataset_tasks.py --ratio 0.25 --task-filter retrain --skip-existing --output-task-file /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/downsample_ratio_0p25_retrain_tasks.tsv

Example all train datasets ratio 0.25 task file:
python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_downsample_dataset_tasks.py --ratio 0.25 --task-filter all --skip-existing --output-task-file /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/downsample_ratio_0p25_all_tasks.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("/sci/labs/michall/roeizucker/roei_test/token_cls_datasets2")
DEFAULT_OUTPUT_ROOT = Path("/sci/labs/michall/roeizucker/roei_test/token_cls_datasets_downsampled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create token-classification downsample task rows.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-task-file", required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--task-filter", choices=["retrain", "pretrain", "all"], default="retrain")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def ratio_tag(ratio: float) -> str:
    return f"{ratio:g}".replace(".", "p")


def is_huggingface_dataset(path: Path) -> bool:
    return (path / "dataset_info.json").exists() or (path / "state.json").exists()


def wanted_dataset(path: Path, task_filter: str) -> bool:
    if task_filter == "all":
        return True
    return f"_{task_filter}_token_cls_" in path.name


def train_dataset_paths(input_root: Path, task_filter: str) -> list[Path]:
    return [
        path
        for path in sorted(input_root.glob("*/*_datasets_train"))
        if path.is_dir() and is_huggingface_dataset(path) and wanted_dataset(path, task_filter)
    ]


def output_path(input_path: Path, input_root: Path, output_root: Path, ratio: float) -> Path:
    relative_path = input_path.relative_to(input_root)
    return output_root / relative_path.parent / f"{relative_path.name}_ratio_{ratio_tag(ratio)}"


def task_line(input_path: Path, output_path: Path, ratio: float) -> str:
    return "\t".join([str(input_path), str(output_path), f"{ratio:g}"])


def create_task_lines(args: argparse.Namespace) -> list[str]:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    lines = []
    for input_path in train_dataset_paths(input_root, args.task_filter):
        destination = output_path(input_path, input_root, output_root, args.ratio)
        if args.skip_existing and destination.exists():
            continue
        lines.append(task_line(input_path, destination, args.ratio))
    return lines


def write_task_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    lines = create_task_lines(args)
    write_task_file(Path(args.output_task_file), lines)
    print({"task_file": args.output_task_file, "tasks": len(lines)}, flush=True)


if __name__ == "__main__":
    main()
