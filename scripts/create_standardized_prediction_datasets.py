"""Create identical eval/test window splits from chr12-only prediction datasets.

Example, inspect the 31-run common-window split without writing datasets:
python3 scripts/create_standardized_prediction_datasets.py --dry-run

Example, create the standardized datasets and reuse matching existing outputs:
python3 scripts/create_standardized_prediction_datasets.py --skip-existing

Example, replace an incomplete standardized dataset build:
python3 scripts/create_standardized_prediction_datasets.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset


RUNS_ROOT = Path("/sci/labs/michall/roeizucker/roei_test/token_cls_models_downsampled")
INPUT_ROOT = Path(
    "/sci/labs/michall/roeizucker/roei_test/"
    "token_cls_datasets2_chr12_prediction_v1"
)
OUTPUT_ROOT = Path(
    "/sci/labs/michall/roeizucker/roei_test/"
    "token_cls_datasets2_chr12_eval_test_standardized_v1"
)
MANIFEST_NAME = "window_split_manifest.tsv"
SUMMARY_NAME = "standardized_split_summary.tsv"
DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_SEED = 42
EXPECTED_CHROMOSOME = "chr12"


@dataclass(frozen=True)
class DatasetSource:
    sample: str
    tissue_dir: str
    tissue: str
    input_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one exact common-window chr12 eval/test split for all ratio-0.25 runs."
    )
    parser.add_argument("--runs-root", default=str(RUNS_ROOT))
    parser.add_argument("--input-root", default=str(INPUT_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def tissue_name(tissue_dir: str) -> str:
    return tissue_dir.removeprefix("_token_cls_").removesuffix("_kmer")


def source_dataset_path(input_root: Path, sample: str, tissue_dir: str) -> Path:
    tissue = tissue_name(tissue_dir)
    name = f"{sample}_retrain_token_cls_{tissue}_kmer_seq_5400_datasets_test"
    return input_root / tissue_dir / name


def discover_sources(runs_root: Path, input_root: Path) -> list[DatasetSource]:
    sources = []
    seen_samples = set()
    for run in sorted(runs_root.glob("_token_cls_*/*downsampled_ratio_0p25*")):
        if not run.is_dir() or "smoke_top400" in run.name or "bad_partial" in run.name:
            continue
        sample = run.name.split("_epoch-10-step-", 1)[0]
        if sample in seen_samples:
            raise ValueError(f"duplicate downsampled run for sample {sample}")
        seen_samples.add(sample)
        tissue_dir = run.parent.name
        input_path = source_dataset_path(input_root, sample, tissue_dir)
        if not input_path.is_dir():
            raise FileNotFoundError(f"missing source test dataset: {input_path}")
        sources.append(
            DatasetSource(sample, tissue_dir, tissue_name(tissue_dir), input_path)
        )
    if not sources:
        raise ValueError(f"no ratio-0.25 downsampled runs found under {runs_root}")
    return sources


def load_dataset(path: Path) -> "Dataset":
    from datasets import load_from_disk

    return load_from_disk(str(path))


def window_ids(dataset: "Dataset", path: Path) -> list[str]:
    if "window_id" not in dataset.column_names:
        raise ValueError(f"dataset has no window_id column: {path}")
    values = list(dataset["window_id"])
    if len(values) != len(set(values)):
        raise ValueError(f"dataset has duplicate window_id values: {path}")
    return values


def common_windows(sources: list[DatasetSource]) -> tuple[set[str], dict[str, int]]:
    common = None
    sizes = {}
    for source in sources:
        ids = set(window_ids(load_dataset(source.input_path), source.input_path))
        unexpected = [value for value in ids if not value.startswith(f"{EXPECTED_CHROMOSOME}:")]
        if unexpected:
            raise ValueError(
                f"source dataset contains non-{EXPECTED_CHROMOSOME} windows: "
                f"{source.input_path}"
            )
        sizes[source.sample] = len(ids)
        common = ids if common is None else common & ids
        print(f"indexed {source.sample}: {len(ids)} windows", flush=True)
    if not common:
        raise ValueError("the source test datasets have no common window IDs")
    return common, sizes


def window_score(window_id: str, seed: int) -> bytes:
    payload = f"{seed}\0{window_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def partition_windows(
    common: set[str], eval_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    if not 0 < eval_fraction < 1:
        raise ValueError("--eval-fraction must be between 0 and 1")
    ranked = sorted(common, key=lambda value: (window_score(value, seed), value))
    eval_count = math.ceil(len(ranked) * eval_fraction)
    eval_set = set(ranked[:eval_count])
    eval_ids = sorted(eval_set)
    test_ids = sorted(common - eval_set)
    return eval_ids, test_ids


def manifest_rows(eval_ids: list[str], test_ids: list[str]) -> list[tuple[str, str]]:
    rows = [(window_id, "eval") for window_id in eval_ids]
    rows.extend((window_id, "test") for window_id in test_ids)
    return sorted(rows)


def read_manifest(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [(row["window_id"], row["split"]) for row in reader]


def write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["window_id", "split"])
        writer.writerows(rows)


def ensure_manifest(
    path: Path, rows: list[tuple[str, str]], overwrite: bool, dry_run: bool
) -> None:
    if path.exists() and not overwrite:
        if read_manifest(path) != rows:
            raise ValueError(f"existing manifest disagrees with the requested split: {path}")
        print(f"reusing matching manifest: {path}", flush=True)
        return
    if dry_run:
        print(f"would write manifest: {path}", flush=True)
        return
    write_manifest(path, rows)
    print(f"wrote manifest: {path}", flush=True)


def output_path(output_root: Path, source: DatasetSource, split: str) -> Path:
    name = (
        f"{source.sample}_retrain_token_cls_{source.tissue}_kmer_"
        f"seq_5400_datasets_{split}"
    )
    return output_root / source.tissue_dir / name


def select_windows(dataset: "Dataset", wanted_ids: list[str], source_path: Path) -> "Dataset":
    ids = window_ids(dataset, source_path)
    index = {window_id: row for row, window_id in enumerate(ids)}
    missing = [window_id for window_id in wanted_ids if window_id not in index]
    if missing:
        raise ValueError(f"{source_path} is missing {len(missing)} manifest windows")
    return dataset.select([index[window_id] for window_id in wanted_ids])


def validate_dataset(path: Path, expected_ids: list[str]) -> None:
    actual_ids = window_ids(load_dataset(path), path)
    if actual_ids != expected_ids:
        raise ValueError(f"saved dataset has unexpected window order or membership: {path}")


def should_write(
    path: Path, expected_ids: list[str], overwrite: bool, skip_existing: bool
) -> bool:
    if not path.exists():
        return True
    if skip_existing:
        validate_dataset(path, expected_ids)
        print(f"reused validated dataset: {path}", flush=True)
        return False
    if not overwrite:
        raise FileExistsError(f"output dataset already exists: {path}")
    shutil.rmtree(path)
    return True


def save_dataset_atomic(
    dataset: "Dataset", wanted_ids: list[str], source_path: Path, output_path: Path
) -> None:
    temporary_path = output_path.with_name(output_path.name + ".incomplete")
    if temporary_path.exists():
        shutil.rmtree(temporary_path)
    selected = select_windows(dataset, wanted_ids, source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.save_to_disk(str(temporary_path))
    validate_dataset(temporary_path, wanted_ids)
    temporary_path.rename(output_path)
    validate_dataset(output_path, wanted_ids)


def save_source_splits(
    source: DatasetSource,
    output_root: Path,
    eval_ids: list[str],
    test_ids: list[str],
    overwrite: bool,
    skip_existing: bool,
    dry_run: bool,
) -> tuple[Path, Path]:
    eval_path = output_path(output_root, source, "eval")
    test_path = output_path(output_root, source, "test")
    print(f"{source.sample}: eval -> {eval_path}", flush=True)
    print(f"{source.sample}: test -> {test_path}", flush=True)
    if dry_run:
        return eval_path, test_path
    write_eval = should_write(eval_path, eval_ids, overwrite, skip_existing)
    write_test = should_write(test_path, test_ids, overwrite, skip_existing)
    if not write_eval and not write_test:
        return eval_path, test_path
    dataset = load_dataset(source.input_path)
    if write_eval:
        save_dataset_atomic(dataset, eval_ids, source.input_path, eval_path)
    if write_test:
        save_dataset_atomic(dataset, test_ids, source.input_path, test_path)
    return eval_path, test_path


def write_summary(
    path: Path,
    sources: list[DatasetSource],
    sizes: dict[str, int],
    eval_ids: list[str],
    test_ids: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["sample", "tissue", "source_windows", "eval_windows", "test_windows"]
        )
        for source in sources:
            writer.writerow(
                [source.sample, source.tissue, sizes[source.sample], len(eval_ids), len(test_ids)]
            )


def create_datasets(args: argparse.Namespace) -> None:
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite are mutually exclusive")
    runs_root = Path(args.runs_root)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    sources = discover_sources(runs_root, input_root)
    common, sizes = common_windows(sources)
    eval_ids, test_ids = partition_windows(common, args.eval_fraction, args.seed)
    rows = manifest_rows(eval_ids, test_ids)
    print(
        f"sources={len(sources)} common={len(common)} "
        f"eval={len(eval_ids)} test={len(test_ids)}",
        flush=True,
    )
    ensure_manifest(output_root / MANIFEST_NAME, rows, args.overwrite, args.dry_run)
    for source in sources:
        save_source_splits(
            source,
            output_root,
            eval_ids,
            test_ids,
            args.overwrite,
            args.skip_existing,
            args.dry_run,
        )
    if not args.dry_run:
        write_summary(output_root / SUMMARY_NAME, sources, sizes, eval_ids, test_ids)


def main() -> None:
    create_datasets(parse_args())


if __name__ == "__main__":
    main()
