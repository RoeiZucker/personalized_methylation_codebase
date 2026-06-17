import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pyBigWig

try:
    from .utils.bigwig_utils import get_preprocessed_encode_cpg_chrom_dataframe_from_bw
except ImportError:
    from utils.bigwig_utils import get_preprocessed_encode_cpg_chrom_dataframe_from_bw

DEFAULT_CHROMS = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr18", "chr19", "chr20", "chr21", "chr22"]
DEFAULT_GROUP_MIN_SIZE = 4
DEFAULT_NUM_BINS = 5
DEFAULT_OUTPUT_CSV = "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/results/tissue_group_atlas_all_results_chr1_5_18_22.csv"
DEFAULT_SUMMARY_CSV = "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/results/tissue_group_atlas_summary_chr1_5_18_22.csv"
DEFAULT_UBER_SCRIPT_PATH = "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/src/uber_project_creator_script.py"


@dataclass
class OnlineMetricAccumulator:
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0
    sum_sqerr: float = 0.0
    sum_abserr: float = 0.0

    def update(self, predictions: np.ndarray, references: np.ndarray) -> None:
        if predictions.size == 0:
            return
        predictions = predictions.astype(np.float64, copy=False)
        references = references.astype(np.float64, copy=False)
        diff = predictions - references
        self.n += int(predictions.size)
        self.sum_x += float(predictions.sum())
        self.sum_y += float(references.sum())
        self.sum_x2 += float(np.square(predictions).sum())
        self.sum_y2 += float(np.square(references).sum())
        self.sum_xy += float((predictions * references).sum())
        self.sum_sqerr += float(np.square(diff).sum())
        self.sum_abserr += float(np.abs(diff).sum())

    def finalize(self) -> Dict[str, float]:
        if self.n == 0:
            return {"pearsonr": np.nan, "mse": np.nan, "mae": np.nan, "n_positions": 0}

        pearson = np.nan
        if self.n > 1:
            numerator = (self.n * self.sum_xy) - (self.sum_x * self.sum_y)
            denom_x = (self.n * self.sum_x2) - (self.sum_x ** 2)
            denom_y = (self.n * self.sum_y2) - (self.sum_y ** 2)
            denom = math.sqrt(max(denom_x, 0.0) * max(denom_y, 0.0))
            if denom > 0:
                pearson = numerator / denom

        mse = self.sum_sqerr / self.n
        mae = self.sum_abserr / self.n
        return {"pearsonr": pearson, "mse": mse, "mae": mae, "n_positions": self.n}


def _extract_files_names_block(script_path: str) -> List[str]:
    text = Path(script_path).read_text(encoding="utf-8")
    start_token = "FILES_NAMES ='''''"
    end_token = "'''.split(\"\\n\")"
    start = text.index(start_token) + len(start_token)
    end = text.index(end_token)
    return [line for line in text[start:end].split("\n") if line.strip()]


def _sample_name_from_file_name(file_name: str) -> str:
    return file_name.replace(".hg38.bigwig", "").split("-")[-1]


def load_grouped_tissue_files(script_path: str, min_group_size: int) -> Dict[str, List[str]]:
    files = _extract_files_names_block(script_path)
    grouped: Dict[str, List[str]] = {}
    for file_name in files:
        suffix_source = "_".join(file_name.split("_")[1:])
        group_name = "-".join(suffix_source.split("-")[:-1])
        grouped.setdefault(group_name, []).append(file_name)
    return {group_name: sorted(file_names) for group_name, file_names in grouped.items() if len(file_names) >= min_group_size}


def _full_position_series(df: pd.DataFrame) -> pd.Series:
    return df["chrom"].astype(str) + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)


def load_group_matrix_for_chrom(bws: Sequence[pyBigWig.pyBigWig], sample_names: Sequence[str], chrom: str) -> pd.DataFrame:
    merged = None
    for sample_name, bw in zip(sample_names, bws):
        df = get_preprocessed_encode_cpg_chrom_dataframe_from_bw(bw, chrom)
        if df.empty:
            return pd.DataFrame(columns=["full_position", *sample_names])
        curr = pd.DataFrame({"full_position": _full_position_series(df), sample_name: df["methyl_rate"].astype(np.float64)})
        merged = curr if merged is None else merged.merge(curr, on="full_position", how="inner")
        if merged.empty:
            return pd.DataFrame(columns=["full_position", *sample_names])
    return merged


def _shared_chromosomes(bws: Sequence[pyBigWig.pyBigWig], requested_chroms: Sequence[str]) -> List[str]:
    available = set(requested_chroms)
    for bw in bws:
        available &= set(bw.chroms().keys())
    return [chrom for chrom in requested_chroms if chrom in available]


def _compute_max_std_for_group(bws: Sequence[pyBigWig.pyBigWig], sample_names: Sequence[str], chroms: Sequence[str], verbose: bool) -> Dict[str, float]:
    max_std = {sample_name: 0.0 for sample_name in sample_names}
    for chrom_index, chrom in enumerate(chroms, start=1):
        if verbose:
            print(f"pass1 {chrom_index}/{len(chroms)} {chrom}", flush=True)
        matrix_df = load_group_matrix_for_chrom(bws, sample_names, chrom)
        if matrix_df.empty:
            continue
        values = matrix_df[sample_names].to_numpy(dtype=np.float64)
        for sample_index, sample_name in enumerate(sample_names):
            ref_values = np.delete(values, sample_index, axis=1)
            std_values = ref_values.std(axis=1, ddof=1)
            curr_max = float(std_values.max()) if std_values.size else 0.0
            if curr_max > max_std[sample_name]:
                max_std[sample_name] = curr_max
    return max_std


def _bin_edges(max_std: float, number_of_bins: int) -> np.ndarray:
    if max_std <= 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    return np.linspace(0.0, max_std, number_of_bins + 1, dtype=np.float64)


def _assign_bin_indices(std_values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(edges) <= 2 and edges[0] == edges[-1]:
        return np.zeros(std_values.shape[0], dtype=np.int64)
    indices = np.searchsorted(edges, std_values, side="right") - 1
    return np.clip(indices, 0, len(edges) - 2)


def evaluate_group(group_name: str, file_names: Sequence[str], base_file_path: str, chroms: Sequence[str], number_of_bins: int, verbose: bool) -> List[Dict[str, object]]:
    sample_names = [_sample_name_from_file_name(file_name) for file_name in file_names]
    file_paths = [os.path.join(base_file_path, file_name) for file_name in file_names]
    bws = [pyBigWig.open(file_path) for file_path in file_paths]
    try:
        shared_chroms = _shared_chromosomes(bws, chroms)
        if verbose:
            print(f"group {group_name}: {len(sample_names)} samples, chroms={shared_chroms}", flush=True)
        max_std = _compute_max_std_for_group(bws, sample_names, shared_chroms, verbose)
        edges_by_sample = {sample_name: _bin_edges(max_std[sample_name], number_of_bins) for sample_name in sample_names}
        accumulators = {sample_name: [OnlineMetricAccumulator() for _ in range(max(1, len(edges_by_sample[sample_name]) - 1))] for sample_name in sample_names}

        for chrom_index, chrom in enumerate(shared_chroms, start=1):
            if verbose:
                print(f"pass2 {chrom_index}/{len(shared_chroms)} {chrom}", flush=True)
            matrix_df = load_group_matrix_for_chrom(bws, sample_names, chrom)
            if matrix_df.empty:
                continue
            values = matrix_df[sample_names].to_numpy(dtype=np.float64)
            for sample_index, sample_name in enumerate(sample_names):
                target_values = values[:, sample_index]
                ref_values = np.delete(values, sample_index, axis=1)
                atlas_mean = ref_values.mean(axis=1)
                std_values = ref_values.std(axis=1, ddof=1)
                edges = edges_by_sample[sample_name]
                bin_indices = _assign_bin_indices(std_values, edges)
                for bin_index, accumulator in enumerate(accumulators[sample_name]):
                    mask = bin_indices == bin_index
                    if np.any(mask):
                        accumulator.update(atlas_mean[mask], target_values[mask])

        rows = []
        for sample_name in sample_names:
            edges = edges_by_sample[sample_name]
            for bin_rank, accumulator in enumerate(accumulators[sample_name], start=1):
                metrics = accumulator.finalize()
                lower = float(edges[bin_rank - 1]) if len(edges) > 1 else 0.0
                upper = float(edges[bin_rank]) if len(edges) > 1 else 0.0
                rows.append({
                    "group_name": group_name,
                    "held_out_sample": sample_name,
                    "group_sample_count": len(sample_names),
                    "chromosomes": ",".join(shared_chroms),
                    "bin_rank": bin_rank,
                    "bin_label": f"{lower}-{upper}",
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "pearsonr": metrics["pearsonr"],
                    "mse": metrics["mse"],
                    "mae": metrics["mae"],
                    "n_positions": metrics["n_positions"],
                })
        return rows
    finally:
        for bw in bws:
            bw.close()


def build_summary_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    summary = (
        raw_df.groupby(["group_name", "group_sample_count", "bin_rank"], as_index=False)
        .agg(
            pearsonr_mean=("pearsonr", "mean"),
            pearsonr_std=("pearsonr", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_positions_mean=("n_positions", "mean"),
        )
    )
    return summary.sort_values(["group_name", "bin_rank"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute leave-one-out atlas baseline results for each tissue group.")
    parser.add_argument("--uber-script-path", default=DEFAULT_UBER_SCRIPT_PATH)
    parser.add_argument("--base-file-path", default="/sci/archive/michall/roeizucker/downloaded_datasets")
    parser.add_argument("--group-min-size", type=int, default=DEFAULT_GROUP_MIN_SIZE)
    parser.add_argument("--number-of-bins", type=int, default=DEFAULT_NUM_BINS)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--chromosomes", nargs="*", default=DEFAULT_CHROMS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    grouped_files = load_grouped_tissue_files(args.uber_script_path, args.group_min_size)
    if args.groups:
        requested = set(args.groups)
        grouped_files = {group: files for group, files in grouped_files.items() if group in requested}

    all_rows: List[Dict[str, object]] = []
    for group_index, group_name in enumerate(sorted(grouped_files), start=1):
        if args.verbose:
            print(f"[{group_index}/{len(grouped_files)}] evaluating {group_name}", flush=True)
        all_rows.extend(evaluate_group(group_name, grouped_files[group_name], args.base_file_path, args.chromosomes, args.number_of_bins, args.verbose))

    raw_df = pd.DataFrame(all_rows).sort_values(["group_name", "held_out_sample", "bin_rank"]).reset_index(drop=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(args.output_csv, index=False)

    summary_df = build_summary_dataframe(raw_df)
    summary_df.to_csv(args.summary_csv, index=False)

    print(f"wrote raw results to {args.output_csv}")
    print(f"wrote summary results to {args.summary_csv}")


if __name__ == "__main__":
    main()
