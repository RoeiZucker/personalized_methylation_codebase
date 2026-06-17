#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULT_RE = re.compile(
    r"^(?P<sample>[^_]+)_(?:(?P<checkpoint>epoch-\d+-step-\d+)|(?P<no_pretraining>no_pretraining))"
    r"_eval_(?P<tissue>.+)_(?P<split>kmer|window)_lr_(?P<lr>[^_]+)_bs_(?P<bs>[^_]+)_seq_(?P<seq>\d+)_testsize_(?P<test>[^_]+)_result\.csv(?:\.gitbackup)?$"
)
METRIC_COL_RE = re.compile(r"^(?P<bin>.+)_(?P<metric>pearsonr|mse|mae)$")

NO_PRETRAIN_COLOR = "#4C78A8"
PRETRAINED_COLOR = "#F58518"
EPOCH2_COLOR = "#54A24B"
ATLAS_COLOR = "red"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--group-name", type=str, required=True)
    parser.add_argument("--atlas-summary-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variability-base-dir", type=Path, required=True)
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()


def preferred_csv_like_paths(result_dir: Path) -> list[Path]:
    candidates = sorted(result_dir.glob("*.csv")) + sorted(result_dir.glob("*.csv.gitbackup"))
    preferred: dict[str, Path] = {}
    for path in candidates:
        key = path.name.removesuffix(".gitbackup")
        current = preferred.get(key)
        if current is None or (current.name.endswith(".gitbackup") and not path.name.endswith(".gitbackup")):
            preferred[key] = path
    return [preferred[k] for k in sorted(preferred)]


def parse_result_path(path: Path) -> dict | None:
    match = RESULT_RE.match(path.name)
    if not match:
        return None
    groups = match.groupdict()
    checkpoint = groups["checkpoint"]
    if groups.get("no_pretraining"):
        pretraining_mode = "no_pretraining"
        pretraining_bucket = "no_pretraining"
    else:
        pretraining_mode = "pretrained"
        epoch_match = re.search(r"epoch-(\d+)", checkpoint or "")
        epoch_num = int(epoch_match.group(1)) if epoch_match else 0
        pretraining_bucket = f"epoch_{epoch_num}_pretraining"
    return {
        "file_path": path,
        "held_out_sample": groups["sample"],
        "checkpoint": checkpoint,
        "tissue_name": groups["tissue"],
        "split_type": groups["split"],
        "seq_size": int(groups["seq"]),
        "learning_rate": groups["lr"],
        "batch_size": int(groups["bs"]),
        "test_size": float(groups["test"]),
        "pretraining_mode": pretraining_mode,
        "pretraining_bucket": pretraining_bucket,
    }


def add_std_bins_to_dataframe(number_of_bins: int, variant_file_dataframe: pd.DataFrame) -> None:
    max_val = variant_file_dataframe["std"].max()
    edges = list(np.linspace(0, max_val, number_of_bins + 1))
    labels = [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges) - 1)]
    variant_file_dataframe["std_bin"] = pd.cut(
        variant_file_dataframe["std"],
        bins=edges,
        labels=labels,
        right=True,
        include_lowest=True,
    )


def load_variability_counts(variability_path: Path, number_of_bins: int = 5) -> pd.DataFrame:
    std_df = pd.read_csv(variability_path, usecols=['std']).dropna(subset=['std']).copy()
    if std_df.empty:
        return pd.DataFrame(columns=['bin_rank', 'n_positions'])
    std_values = pd.to_numeric(std_df['std'], errors='coerce').dropna().to_numpy()
    if std_values.size == 0:
        return pd.DataFrame(columns=['bin_rank', 'n_positions'])
    max_val = float(std_values.max())
    edges = np.linspace(0.0, max_val, number_of_bins + 1)
    counts, _ = np.histogram(std_values, bins=edges)
    return pd.DataFrame({'bin_rank': np.arange(1, number_of_bins + 1), 'n_positions': counts.astype(int)})


def resolve_variability_path(base_dir: Path, metadata: dict) -> Path:
    tissue_dir = base_dir / f"_{metadata['tissue_name']}_{metadata['split_type']}"
    pattern = f"{metadata['held_out_sample']}_per_varaint_variability_{metadata['tissue_name']}_{metadata['split_type']}_seq_*_datasets.csv"
    matches = sorted(tissue_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No variability file matched {pattern} in {tissue_dir}")
    preferred = [m for m in matches if "_seq_5400_" in m.name]
    if len(preferred) == 1:
        return preferred[0]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"Ambiguous variability files for {metadata['held_out_sample']}: {[m.name for m in matches]}")


def extract_final_metrics(result_path: Path) -> pd.DataFrame:
    df = pd.read_csv(result_path)
    final_row = df.iloc[-1]
    metric_rows = []
    for col in df.columns:
        match = METRIC_COL_RE.match(col)
        if not match:
            continue
        metric_rows.append(
            {
                "bin_label": match.group("bin"),
                "metric": match.group("metric"),
                "value": pd.to_numeric(final_row[col], errors="coerce"),
            }
        )
    metric_df = pd.DataFrame(metric_rows)
    bounds = metric_df["bin_label"].str.extract(r"^(?P<lower>[0-9.]+)-(?P<upper>[0-9.]+)$")
    metric_df["bin_lower"] = pd.to_numeric(bounds["lower"], errors="coerce")
    metric_df = metric_df.sort_values(["bin_lower", "metric"]).reset_index(drop=True)
    label_order = {label: idx + 1 for idx, label in enumerate(metric_df.drop_duplicates("bin_label")["bin_label"].tolist())}
    metric_df["bin_rank"] = metric_df["bin_label"].map(label_order)
    return metric_df[["bin_rank", "bin_label", "metric", "value"]]


def build_run_level_df(result_dir: Path, variability_base_dir: Path) -> pd.DataFrame:
    rows = []
    variability_counts_cache: dict[Path, pd.DataFrame] = {}
    for path in preferred_csv_like_paths(result_dir):
        metadata = parse_result_path(path)
        if metadata is None:
            continue
        metrics_df = extract_final_metrics(path)
        variability_path = resolve_variability_path(variability_base_dir, metadata)
        variability_counts = variability_counts_cache.get(variability_path)
        if variability_counts is None:
            variability_counts = load_variability_counts(variability_path)
            variability_counts_cache[variability_path] = variability_counts
        merged = metrics_df.merge(variability_counts, on="bin_rank", how="left")
        for record in merged.to_dict("records"):
            rows.append({**metadata, **record})
    if not rows:
        raise RuntimeError(f"No matching result files found in {result_dir}")
    return pd.DataFrame(rows)


def summarize_mode(run_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    metric_df = run_df[run_df["metric"] == metric].copy()
    summary = (
        metric_df.groupby(["tissue_name", "split_type", "seq_size", "pretraining_mode", "bin_rank"], as_index=False)
        .agg(
            value=("value", "mean"),
            value_std=("value", "std"),
            n_runs=("file_path", "nunique"),
            n_samples=("held_out_sample", "nunique"),
        )
        .sort_values(["seq_size", "pretraining_mode", "bin_rank"])
    )
    return summary


def summarize_bucket(run_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    metric_df = run_df[run_df["metric"] == metric].copy()
    value_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    summary = (
        metric_df.groupby(["seq_size", "bin_rank", "pretraining_bucket"], as_index=False)
        .agg(
            **{
                value_col: ("value", "mean"),
                std_col: ("value", "std"),
                "n_positions_mean": ("n_positions", "mean"),
                "run_count": ("file_path", "nunique"),
                "sample_count": ("held_out_sample", "nunique"),
                "avg_n_positions": ("n_positions", "mean"),
            }
        )
        .sort_values(["seq_size", "bin_rank", "pretraining_bucket"])
    )
    label_map = {
        "no_pretraining": "no pretraining",
        "epoch_1_pretraining": "1 epoch pretraining",
        "epoch_2_pretraining": "2 epoch pretraining",
    }
    summary["pretraining_label"] = summary["pretraining_bucket"].map(label_map).fillna(summary["pretraining_bucket"])
    column_order = [
        "seq_size",
        "bin_rank",
        "pretraining_bucket",
        "pretraining_label",
        value_col,
        std_col,
        "n_positions_mean",
        "run_count",
        "sample_count",
        "avg_n_positions",
    ]
    return summary[column_order]


def atlas_overlay_rows(atlas_summary_path: Path, group_name: str, metric: str) -> pd.DataFrame:
    atlas_df = pd.read_csv(atlas_summary_path)
    subset = atlas_df[atlas_df["group_name"] == group_name].copy().sort_values("bin_rank")
    value_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    return subset[["group_name", "bin_rank", value_col, std_col, "n_positions_mean"]]


def plot_mode_summary(mode_df: pd.DataFrame, out_path: Path) -> None:
    seq_sizes = sorted(mode_df["seq_size"].unique().tolist())
    fig, axes = plt.subplots(1, len(seq_sizes), figsize=(6.8 * len(seq_sizes), 5.2), sharey=True)
    if len(seq_sizes) == 1:
        axes = [axes]
    fig.suptitle(
        f"All participants: final-epoch mean Pearson by variability level ({mode_df['tissue_name'].iloc[0]}, {mode_df['split_type'].iloc[0]})",
        y=0.98,
    )
    color_map = {"no_pretraining": NO_PRETRAIN_COLOR, "pretrained": PRETRAINED_COLOR}
    label_map = {"no_pretraining": "no pretraining", "pretrained": "pretrained"}
    y_min = min(-0.12, float(mode_df['value'].min()) - 0.05)
    y_max = max(1.0, float(mode_df['value'].max()) + 0.05)

    for ax, seq in zip(axes, seq_sizes):
        sub = mode_df[mode_df["seq_size"] == seq].copy()
        levels = sorted(sub["bin_rank"].unique().tolist())
        x = np.arange(len(levels))
        width = 0.34
        for i, mode in enumerate(["no_pretraining", "pretrained"]):
            mode_sub = sub[sub["pretraining_mode"] == mode].sort_values("bin_rank")
            vals = [mode_sub.loc[mode_sub["bin_rank"] == level, "value"].iloc[0] if level in mode_sub["bin_rank"].values else np.nan for level in levels]
            errs = [mode_sub.loc[mode_sub["bin_rank"] == level, "value_std"].iloc[0] if level in mode_sub["bin_rank"].values else np.nan for level in levels]
            bars = ax.bar(x + (i - 0.5) * width, vals, width, color=color_map[mode], label=label_map[mode], yerr=errs, capsize=4, ecolor="#222222")
            for bar, val in zip(bars, vals):
                if pd.isna(val):
                    continue
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha='center', va='bottom', fontsize=8, rotation=90)
        ax.set_title(f"seq_{seq}")
        ax.set_xlabel("Variability level")
        ax.set_xticks(x)
        ax.set_xticklabels([str(level) for level in levels])
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis='y', alpha=0.25)
    axes[0].set_ylabel("Mean Final-Epoch Pearsonr")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Pretraining mode", loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=True)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_bucket_summary(bucket_df: pd.DataFrame, atlas_df: pd.DataFrame, out_path: Path, metric: str) -> None:
    seq_sizes = sorted(bucket_df["seq_size"].unique().tolist())
    fig, axes = plt.subplots(1, len(seq_sizes), figsize=(6.8 * len(seq_sizes), 5.8), sharey=True)
    if len(seq_sizes) == 1:
        axes = [axes]
    title_metric = "Pearson" if metric == "pearsonr" else "MSE"
    tissue_slug = out_path.stem.split('_')[0]
    display_group = tissue_slug.replace('-', ' ').title().replace(' ', '-')
    fig.suptitle(
        f"{display_group} kmer: final selected-epoch mean {title_metric} by variability\nAverage positions per bin shown under each variability level",
        y=0.98,
    )
    bucket_order = ["no_pretraining", "epoch_1_pretraining", "epoch_2_pretraining"]
    label_map = {
        "no_pretraining": "no pretraining",
        "epoch_1_pretraining": "1 epoch pretraining",
        "epoch_2_pretraining": "2 epoch pretraining",
    }
    color_map = {
        "no_pretraining": NO_PRETRAIN_COLOR,
        "epoch_1_pretraining": PRETRAINED_COLOR,
        "epoch_2_pretraining": EPOCH2_COLOR,
    }
    value_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    y_min = -0.12 if metric == 'pearsonr' else 0.0
    y_max = max((1.02 if metric == 'pearsonr' else float(bucket_df[value_col].max()) + 0.06), float(atlas_df[value_col].max()) + 0.06 if not atlas_df.empty else 0)

    for ax, seq in zip(axes, seq_sizes):
        sub = bucket_df[bucket_df["seq_size"] == seq].copy()
        levels = sorted(sub["bin_rank"].unique().tolist())
        x = np.arange(len(levels))
        width = 0.22
        avg_positions = sub.groupby('bin_rank', as_index=False)['avg_n_positions'].mean().sort_values('bin_rank')
        for i, bucket in enumerate(bucket_order):
            bucket_sub = sub[sub["pretraining_bucket"] == bucket].sort_values("bin_rank")
            vals = [bucket_sub.loc[bucket_sub["bin_rank"] == level, value_col].iloc[0] if level in bucket_sub["bin_rank"].values else np.nan for level in levels]
            errs = [bucket_sub.loc[bucket_sub["bin_rank"] == level, std_col].iloc[0] if level in bucket_sub["bin_rank"].values else np.nan for level in levels]
            bars = ax.bar(x + (i - 1) * width, vals, width, color=color_map[bucket], label=label_map[bucket], yerr=errs, capsize=4, ecolor="#222222")
            for bar, val in zip(bars, vals):
                if pd.isna(val):
                    continue
                ax.text(bar.get_x() + bar.get_width()/2, val + (0.02 if metric == 'pearsonr' else 0.004), f"{val:.2f}", ha='center', va='bottom', fontsize=9)
        if not atlas_df.empty:
            atlas_vals = [atlas_df.loc[atlas_df['bin_rank'] == level, value_col].iloc[0] if level in atlas_df['bin_rank'].values else np.nan for level in levels]
            ax.scatter(x, atlas_vals, color=ATLAS_COLOR, marker='D', s=70, edgecolors='black', linewidths=0.5, zorder=5, label='Atlas summary')
            for xi, val in zip(x, atlas_vals):
                if pd.isna(val):
                    continue
                ax.text(xi, val + (0.02 if metric == 'pearsonr' else 0.004), f"{val:.2f}", color=ATLAS_COLOR, ha='center', va='bottom', fontsize=9)
        ax.set_title(f"seq {seq}")
        ax.set_xlabel("Variability bin")
        xticklabels = []
        for level in levels:
            npos = avg_positions.loc[avg_positions['bin_rank'] == level, 'avg_n_positions'].iloc[0]
            if npos >= 1000:
                n_text = f"n~{npos/1000:.1f}K"
            else:
                n_text = f"n~{int(round(npos))}"
            xticklabels.append(f"{level}\n({n_text})")
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis='y', alpha=0.25)
    axes[0].set_ylabel(f"Mean {title_metric}")
    handles, labels = axes[0].get_legend_handles_labels()
    seen = {}
    dedup_handles, dedup_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            dedup_handles.append(h)
            dedup_labels.append(l)
    fig.legend(dedup_handles, dedup_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=min(4, len(dedup_labels)), frameon=True)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt.style.use('seaborn-v0_8-whitegrid')

    run_df = build_run_level_df(args.result_dir, args.variability_base_dir)
    pear_mode_df = summarize_mode(run_df, 'pearsonr')
    pear_bucket_df = summarize_bucket(run_df, 'pearsonr')
    mse_bucket_df = summarize_bucket(run_df, 'mse')
    atlas_pear_df = atlas_overlay_rows(args.atlas_summary_path, args.group_name, 'pearsonr')
    atlas_mse_df = atlas_overlay_rows(args.atlas_summary_path, args.group_name, 'mse')

    slug = args.group_name.lower()
    output_dir = args.output_dir
    plot_dir = output_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    pear_mode_path = output_dir / f'{slug}_kmer_final_epoch_mean_by_variability_rows.csv'
    pear_bucket_path = output_dir / f'{slug}_kmer_final_epoch_mean_by_variability_pretraining_split_rows.csv'
    mse_bucket_path = output_dir / f'{slug}_kmer_final_epoch_mean_by_variability_pretraining_split_mse_rows.csv'
    atlas_pear_path = output_dir / f'{slug}_atlas_overlay_rows.csv'
    atlas_mse_path = output_dir / f'{slug}_atlas_overlay_mse_rows.csv'

    pear_mode_df.to_csv(pear_mode_path, index=False)
    pear_bucket_df.to_csv(pear_bucket_path, index=False)
    mse_bucket_df.to_csv(mse_bucket_path, index=False)
    atlas_pear_df.to_csv(atlas_pear_path, index=False)
    atlas_mse_df.to_csv(atlas_mse_path, index=False)

    plot_mode_summary(pear_mode_df, plot_dir / f'{slug}_kmer_all_participants_final_epoch_mean_by_variability.png')
    plot_bucket_summary(pear_bucket_df, atlas_pear_df, plot_dir / f'{slug}_kmer_all_participants_final_epoch_mean_by_variability_pretraining_split.png', 'pearsonr')
    plot_bucket_summary(mse_bucket_df, atlas_mse_df, plot_dir / f'{slug}_kmer_all_participants_final_epoch_mean_by_variability_pretraining_split_mse.png', 'mse')

    print('wrote:', pear_mode_path)
    print('wrote:', pear_bucket_path)
    print('wrote:', mse_bucket_path)
    print('wrote:', atlas_pear_path)
    print('wrote:', atlas_mse_path)

    if args.show_plots:
        plt.show()


if __name__ == '__main__':
    main()
