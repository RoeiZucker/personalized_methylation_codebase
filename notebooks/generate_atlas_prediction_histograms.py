#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.utils.atlas_distribution_utils import (
    DEFAULT_UBER_SCRIPT_PATH,
    build_histogram_table,
    load_variability_std_dataframe,
    resolve_atlas_job_inputs,
    resolve_group_bigwig_paths,
    summarize_distribution_values,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build atlas histograms. By default the standard-deviation histogram is loaded directly "
            "from the saved variability CSV. Optionally, a slower BigWig pass can also compute the "
            "atlas-mean methylation histogram."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--atlas-config", type=Path)
    source_group.add_argument("--group-name", type=str)

    parser.add_argument("--held-out-sample", type=str)
    parser.add_argument("--uber-script-path", type=Path, default=Path(DEFAULT_UBER_SCRIPT_PATH))
    parser.add_argument("--base-file-path", type=Path)
    parser.add_argument("--variability-csv", type=Path)
    parser.add_argument("--chromosomes", nargs="+", default=None)
    parser.add_argument("--number-of-bins", type=int, default=None)
    parser.add_argument("--top-rows", type=int, default=None)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--jump-sample", type=int, default=None)
    parser.add_argument("--hist-bins", type=int, default=50)
    parser.add_argument("--compute-atlas-mean-histogram", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.group_name and not args.held_out_sample:
        parser.error("--held-out-sample is required when using --group-name")
    return args


def _slugify(text: str | None) -> str:
    if not text:
        return "atlas"
    return text.lower().replace(" ", "_").replace("/", "_")


def resolve_inputs(args: argparse.Namespace) -> dict:
    if args.atlas_config is not None:
        resolved = resolve_atlas_job_inputs(
            atlas_config_path=args.atlas_config,
            uber_script_path=args.uber_script_path,
            base_file_path=args.base_file_path,
        )
    else:
        resolved = resolve_group_bigwig_paths(
            group_name=args.group_name,
            held_out_sample=args.held_out_sample,
            uber_script_path=args.uber_script_path,
            base_file_path=args.base_file_path,
        )
        resolved.update(
            {
                "variant_file_path": None,
                "chromosomes": None,
                "number_of_bins": 5,
                "top_rows": -1,
                "test_mode": False,
                "jump_sample": -1,
            }
        )

    if args.variability_csv is not None:
        resolved["variant_file_path"] = str(args.variability_csv)
    if args.chromosomes:
        resolved["chromosomes"] = args.chromosomes
    if args.number_of_bins is not None:
        resolved["number_of_bins"] = args.number_of_bins
    if args.top_rows is not None:
        resolved["top_rows"] = args.top_rows
    if args.jump_sample is not None:
        resolved["jump_sample"] = args.jump_sample
    if args.test_mode:
        resolved["test_mode"] = True
    return resolved


def plot_histograms(
    std_values: pd.Series,
    out_path: Path,
    title_prefix: str,
    hist_bins: int,
    atlas_mean_values: pd.Series | None = None,
) -> None:
    has_atlas_mean = atlas_mean_values is not None and len(atlas_mean_values) > 0
    ncols = 2 if has_atlas_mean else 1
    fig_width = 13.5 if has_atlas_mean else 6.8
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 5.4))
    if ncols == 1:
        axes = [axes]

    std_array = pd.to_numeric(std_values, errors="coerce").dropna().to_numpy(dtype=float)
    axes[0].hist(std_array, bins=hist_bins, color="#4C78A8", alpha=0.85, edgecolor="white")
    axes[0].set_title(
        "Reference-sample standard deviation\n"
        f"mean={std_array.mean():.4f}, median={pd.Series(std_array).median():.4f}"
    )
    axes[0].set_xlabel("Standard deviation across reference samples")
    axes[0].set_ylabel("Variant count")
    axes[0].grid(True, axis="y", alpha=0.25)

    if has_atlas_mean:
        atlas_mean_array = pd.to_numeric(atlas_mean_values, errors="coerce").dropna().to_numpy(dtype=float)
        axes[1].hist(
            atlas_mean_array,
            bins=hist_bins,
            range=(0.0, 1.0),
            color="#F58518",
            alpha=0.75,
            edgecolor="white",
        )
        axes[1].set_title(
            "Atlas mean methylation-rate distribution\n"
            f"mean={atlas_mean_array.mean():.4f}, median={pd.Series(atlas_mean_array).median():.4f}"
        )
        axes[1].set_xlabel("Methylation rate")
        axes[1].set_ylabel("Variant count")
        axes[1].set_xlim(0.0, 1.0)
        axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(title_prefix, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if not plt.isinteractive():
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plt.style.use("seaborn-v0_8-whitegrid")

    resolved = resolve_inputs(args)
    variability_path = resolved.get("variant_file_path")
    if variability_path is None:
        raise ValueError(
            "A variability CSV is required for the atlas standard-deviation histogram. "
            "Use --atlas-config with variant_file_path or pass --variability-csv explicitly."
        )

    variability_df = load_variability_std_dataframe(
        variability_path=variability_path,
        top_rows=int(resolved["top_rows"]),
        test_mode=bool(resolved["test_mode"]),
        jump_sample=int(resolved["jump_sample"]),
    )
    if variability_df.empty:
        raise RuntimeError(f"No usable std values were found in {variability_path}.")

    group_slug = _slugify(resolved.get("group_name"))
    held_out_sample = resolved.get("held_out_sample") or "atlas"
    prefix = f"{group_slug}_{held_out_sample}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variability_rows_csv_path = args.output_dir / f"{prefix}_atlas_variability_rows.csv"
    summary_csv_path = args.output_dir / f"{prefix}_atlas_distribution_summary.csv"
    std_hist_csv_path = args.output_dir / f"{prefix}_atlas_std_histogram_bins.csv"
    plot_path = args.output_dir / f"{prefix}_atlas_distribution_histograms.png"

    variability_df.to_csv(variability_rows_csv_path, index=False)
    summary_frames = [summarize_distribution_values(variability_df["std"], "reference_std")]
    build_histogram_table(variability_df["std"], bins=args.hist_bins, count_column_name="std_count").to_csv(
        std_hist_csv_path, index=False
    )

    atlas_mean_values = None
    if args.compute_atlas_mean_histogram:
        from src.utils.atlas_bigwig_utils import build_atlas_position_dataframe

        matched_df = build_atlas_position_dataframe(
            target_bigwig_path=resolved["target_bigwig_path"],
            atlas_bigwig_paths=resolved["atlas_bigwig_paths"],
            number_of_bins=int(resolved["number_of_bins"]),
            chroms=resolved["chromosomes"],
            top_rows=int(resolved["top_rows"]),
            test_mode=bool(resolved["test_mode"]),
            jump_sample=int(resolved["jump_sample"]),
            verbose=bool(args.verbose),
        )
        if matched_df.empty:
            raise RuntimeError("No matched atlas positions were found for the requested inputs.")
        atlas_mean_values = matched_df["atlas_mean"]
        summary_frames.append(summarize_distribution_values(atlas_mean_values, "atlas_mean_prediction"))

        atlas_mean_rows_csv_path = args.output_dir / f"{prefix}_atlas_mean_rows.csv"
        methyl_hist_csv_path = args.output_dir / f"{prefix}_atlas_methylation_histogram_bins.csv"
        matched_df.to_csv(atlas_mean_rows_csv_path, index=False)
        build_histogram_table(
            atlas_mean_values,
            bins=args.hist_bins,
            hist_range=(0.0, 1.0),
            count_column_name="atlas_mean_count",
        ).to_csv(methyl_hist_csv_path, index=False)
        print("wrote:", atlas_mean_rows_csv_path)
        print("wrote:", methyl_hist_csv_path)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_df.to_csv(summary_csv_path, index=False)

    title_prefix = (
        f"{resolved.get('group_name', 'Atlas')} | held out: {held_out_sample} | "
        f"n={len(variability_df):,} variability rows"
    )
    plot_histograms(
        std_values=variability_df["std"],
        out_path=plot_path,
        title_prefix=title_prefix,
        hist_bins=args.hist_bins,
        atlas_mean_values=atlas_mean_values,
    )

    print("wrote:", variability_rows_csv_path)
    print("wrote:", summary_csv_path)
    print("wrote:", std_hist_csv_path)
    print("wrote:", plot_path)
    if not args.compute_atlas_mean_histogram:
        print(
            "skipped atlas mean methylation histogram; pass --compute-atlas-mean-histogram to build it from BigWigs."
        )

    if args.show_plots:
        plt.show()


if __name__ == "__main__":
    main()
