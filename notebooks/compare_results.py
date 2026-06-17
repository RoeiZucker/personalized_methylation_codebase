#!/usr/bin/env python3
"""Compare result CSVs across two experiment directories.

Default directories:
- left:  window_split_with_600
- right: _different_epoch_length_dataset_override_test

The script pairs files by normalized filename key that ignores project token
between `eval...` and `_lr_...`, so counterpart files from different project
folders can still be matched.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


FILENAME_RE = re.compile(
    r"^(?P<prefix>.+?)_eval(?P<variant>_lora_over_lora|_lora)?(?P<project>.+?)"
    r"_lr_(?P<lr>[^_]+)_bs_(?P<bs>[^_]+)_seq_(?P<seq>\d+)_testsize_(?P<testsize>[^_]+)_result\.csv$"
)
METRIC_COL_RE = re.compile(r"^(?P<bin>.+)_(?P<metric>pearsonr|mse|mae)$")
VARIANT_TITLES = {
    "full_finetune": "regular",
    "lora": "lora",
    "lora_over_lora": "lora_over_lora",
}
TRAINING_PHASE_ORDER = [
    "first_phase_only_epoch1",
    "second_phase_from_epoch1",
    "first_phase_only_epoch2",
    "second_phase_from_epoch2",
    "no_pretrain",
]
TRAINING_PHASE_LABELS = {
    "first_phase_only_epoch1": "1st phase only (e1)",
    "second_phase_from_epoch1": "2nd phase from e1",
    "first_phase_only_epoch2": "1st phase only (e2)",
    "second_phase_from_epoch2": "2nd phase from e2",
    "no_pretrain": "No pretrain",
}
TRAINING_PHASE_COLORS = {
    "first_phase_only_epoch1": "#9ecae1",
    "second_phase_from_epoch1": "#3182bd",
    "first_phase_only_epoch2": "#a1d99b",
    "second_phase_from_epoch2": "#31a354",
    "no_pretrain": "#ff7f0e",
}
EPOCH1_PHASE_COMPARE_COLORS = {
    "first_phase_only_epoch1": "#7DB7D5",
    "second_phase_from_epoch1": "#E6862A",
    "no_pretrain": "#5B9A57",
}
SOURCE_DISPLAY_NAMES = {
    "window_split_with_600": "window based split",
    "_different_epoch_length_dataset_override_test": "kmer based split",
    "_sixfold_mayhem": "sixfold mayhem",
}


def variant_file_tag(eval_variant: str) -> str:
    return "non_lora" if eval_variant == "full_finetune" else eval_variant


def source_display_name(source_label: str) -> str:
    return SOURCE_DISPLAY_NAMES.get(source_label, source_label.lstrip("_"))


@dataclass(frozen=True)
class FileMeta:
    path: Path
    file_name: str
    key: str
    prefix: str
    person_id: str
    model_param: str
    eval_variant: str
    seq_len: int


def canonical_model_param(prefix: str) -> str:
    """Normalize model label for plotting (ignore step id)."""
    tail = prefix.split("_", 1)[1] if "_" in prefix else prefix
    low = tail.lower()

    if "no_pretraining" in low or "no_pretrain" in low or "no-pretrain" in low:
        return "no_pretrain"

    m = re.search(r"epoch-(\d+)-step-\d+", low)
    if m:
        return f"epoch {int(m.group(1))}"

    m = re.search(r"epoch-(\d+)", low) or re.search(r"epoch(\d+)", low)
    if m:
        return f"epoch {int(m.group(1))}"

    return tail


def parse_filename(path: Path) -> Optional[FileMeta]:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None

    d = m.groupdict()
    variant_raw = d["variant"] or ""
    if variant_raw == "_lora":
        eval_variant = "lora"
    elif variant_raw == "_lora_over_lora":
        eval_variant = "lora_over_lora"
    else:
        eval_variant = "full_finetune"

    # Counterpart key intentionally ignores `project` token.
    key = "|".join([
        d["prefix"],
        eval_variant,
        d["lr"],
        d["bs"],
        d["seq"],
        d["testsize"],
    ])

    return FileMeta(
        path=path,
        file_name=path.name,
        key=key,
        prefix=d["prefix"],
        person_id=d["prefix"].split("_", 1)[0],
        model_param=canonical_model_param(d["prefix"]),
        eval_variant=eval_variant,
        seq_len=int(d["seq"]),
    )


def build_index(directory: Path) -> Tuple[Dict[str, List[FileMeta]], List[Path]]:
    index: Dict[str, List[FileMeta]] = {}
    unparsed: List[Path] = []

    for path in sorted(directory.glob("*.csv")):
        meta = parse_filename(path)
        if meta is None:
            unparsed.append(path)
            continue
        index.setdefault(meta.key, []).append(meta)

    return index, unparsed


def normalize_result_csv(path: Path, source_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    value_cols = [c for c in df.columns if METRIC_COL_RE.match(c)]
    if not value_cols:
        raise ValueError(f"No metric columns found in {path}")

    stem = path.name.replace("_result.csv", "")
    train_mode = "no_pretraining" if "no_pretraining" in stem else "pretrained"

    row_df = df.copy()
    row_df["row_in_file"] = range(len(row_df))
    row_df["eval_path"] = row_df["paths"].astype(str)
    row_df["eval_checkpoint"] = row_df["eval_path"].str.findall(r"epoch-\d+-step-\d+").str[-1].fillna("unknown")
    row_df["eval_epoch"] = pd.to_numeric(
        row_df["eval_checkpoint"].str.extract(r"epoch-(\d+)-step-\d+")[0], errors="coerce"
    )

    # Keep consistency with previous notebook logic.
    if train_mode == "pretrained":
        row_df.loc[row_df["row_in_file"] == 0, "eval_epoch"] = 0

    long_df = row_df.melt(
        id_vars=["row_in_file", "eval_epoch", "eval_path"],
        value_vars=value_cols,
        var_name="metric_col",
        value_name="value",
    )
    parsed = long_df["metric_col"].str.extract(METRIC_COL_RE)
    long_df["bin_range"] = parsed["bin"]
    long_df["metric"] = parsed["metric"]

    bounds = long_df["bin_range"].str.extract(r"^(?P<bin_start>[0-9.]+)-(?P<bin_end>[0-9.]+)$")
    long_df["bin_start"] = pd.to_numeric(bounds["bin_start"], errors="coerce")
    long_df["variability_level"] = (
        long_df.groupby("row_in_file")["bin_start"].rank(method="dense", ascending=True).astype("Int64")
    )

    long_df["source"] = source_label
    return long_df[["source", "eval_epoch", "metric", "variability_level", "value"]].copy()


def summarize_pair(left_meta: FileMeta, right_meta: FileMeta) -> Tuple[pd.DataFrame, pd.DataFrame]:
    left = normalize_result_csv(left_meta.path, "left")
    right = normalize_result_csv(right_meta.path, "right")

    merged = left.merge(
        right,
        on=["eval_epoch", "metric", "variability_level"],
        how="inner",
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        summary = pd.DataFrame(
            [
                {
                    "pair_key": left_meta.key,
                    "left_file": left_meta.file_name,
                    "right_file": right_meta.file_name,
                    "metric": "<none>",
                    "n_points": 0,
                    "mean_delta": math.nan,
                    "mae_delta": math.nan,
                    "rmse_delta": math.nan,
                }
            ]
        )
        return merged, summary

    merged["delta_left_minus_right"] = merged["value_left"] - merged["value_right"]

    summary = (
        merged.groupby("metric", as_index=False)
        .agg(
            n_points=("delta_left_minus_right", "size"),
            mean_delta=("delta_left_minus_right", "mean"),
            mae_delta=("delta_left_minus_right", lambda s: s.abs().mean()),
            rmse_delta=("delta_left_minus_right", lambda s: (s.pow(2).mean()) ** 0.5),
        )
        .sort_values("metric")
    )
    summary.insert(0, "right_file", right_meta.file_name)
    summary.insert(0, "left_file", left_meta.file_name)
    summary.insert(0, "pair_key", left_meta.key)

    merged.insert(0, "right_file", right_meta.file_name)
    merged.insert(0, "left_file", left_meta.file_name)
    merged.insert(0, "pair_key", left_meta.key)
    return merged, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare result CSVs across two directories.")
    parser.add_argument(
        "--left-dir",
        type=Path,
        default=Path(
            "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/results/window_split_with_600"
        ),
    )
    parser.add_argument(
        "--right-dir",
        type=Path,
        default=Path(
            "/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/results/_different_epoch_length_dataset_override_test"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "compare_results_output",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display each generated figure with plt.show() in addition to saving it.",
    )
    parser.add_argument(
        "--single-project-dir",
        type=Path,
        default=None,
        help="Generate averaged plots for a single project directory instead of comparing two directories.",
    )
    parser.add_argument(
        "--run_indvidual",
        "--run-individual",
        action="store_true",
        dest="run_indvidual",
        help="Also generate individual participant plots/summaries. Default is all-participants only.",
    )
    return parser.parse_args()


def collect_person_pearson(index: Dict[str, List[FileMeta]], source_label: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for metas in index.values():
        for meta in metas:
            norm = normalize_result_csv(meta.path, source_label)
            pear = norm[norm["metric"] == "pearsonr"].copy()
            if pear.empty:
                continue
            pear["person_id"] = meta.person_id
            pear["seq_len"] = meta.seq_len
            pear["eval_variant"] = meta.eval_variant
            pear["model_param"] = meta.model_param
            pear["file_name"] = meta.file_name
            rows.append(
                pear[
                    [
                        "source",
                        "person_id",
                        "seq_len",
                        "eval_variant",
                        "model_param",
                        "file_name",
                        "eval_epoch",
                        "variability_level",
                        "value",
                    ]
                ]
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "person_id",
                "seq_len",
                "eval_variant",
                "model_param",
                "file_name",
                "eval_epoch",
                "variability_level",
                "value",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def select_best_epoch_per_model(pear_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """For each source/person/seq/variant/model, select the epoch with best mean Pearsonr.

    Returns:
    - selected_rows: row-level values for the selected epoch (used for bars by variability level).
    - chosen_epochs: one row per model with chosen epoch and its mean score.
    """
    if pear_df.empty:
        empty_selected = pear_df.copy()
        empty_epochs = pd.DataFrame(
            columns=[
                "source",
                "person_id",
                "seq_len",
                "eval_variant",
                "model_param",
                "chosen_epoch",
                "chosen_epoch_mean",
            ]
        )
        return empty_selected, empty_epochs

    epoch_mean = (
        pear_df.groupby(
            ["source", "person_id", "seq_len", "eval_variant", "model_param", "eval_epoch"], as_index=False
        )["value"]
        .mean()
        .rename(columns={"value": "epoch_mean_pearson"})
    )

    # Pick max mean Pearsonr; tie-break by larger epoch index.
    chosen = (
        epoch_mean.sort_values(
            ["source", "person_id", "seq_len", "eval_variant", "model_param", "epoch_mean_pearson", "eval_epoch"],
            ascending=[True, True, True, True, True, False, False],
        )
        .drop_duplicates(["source", "person_id", "seq_len", "eval_variant", "model_param"])
        .rename(columns={"eval_epoch": "chosen_epoch", "epoch_mean_pearson": "chosen_epoch_mean"})
    )

    selected_rows = pear_df.merge(
        chosen[
            [
                "source",
                "person_id",
                "seq_len",
                "eval_variant",
                "model_param",
                "chosen_epoch",
                "chosen_epoch_mean",
            ]
        ],
        left_on=["source", "person_id", "seq_len", "eval_variant", "model_param", "eval_epoch"],
        right_on=["source", "person_id", "seq_len", "eval_variant", "model_param", "chosen_epoch"],
        how="inner",
    )

    return selected_rows, chosen


def select_final_epoch_per_model(pear_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """For each source/person/seq/variant/model, select the final available eval epoch."""
    if pear_df.empty:
        empty_selected = pear_df.copy()
        empty_epochs = pd.DataFrame(
            columns=[
                "source",
                "person_id",
                "seq_len",
                "eval_variant",
                "model_param",
                "chosen_epoch",
                "chosen_epoch_mean",
            ]
        )
        return empty_selected, empty_epochs

    epoch_mean = (
        pear_df.groupby(
            ["source", "person_id", "seq_len", "eval_variant", "model_param", "eval_epoch"], as_index=False
        )["value"]
        .mean()
        .rename(columns={"value": "epoch_mean_pearson"})
    )

    chosen = (
        epoch_mean.sort_values(
            ["source", "person_id", "seq_len", "eval_variant", "model_param", "eval_epoch"],
            ascending=[True, True, True, True, True, False],
        )
        .drop_duplicates(["source", "person_id", "seq_len", "eval_variant", "model_param"])
        .rename(columns={"eval_epoch": "chosen_epoch", "epoch_mean_pearson": "chosen_epoch_mean"})
    )

    selected_rows = pear_df.merge(
        chosen[
            [
                "source",
                "person_id",
                "seq_len",
                "eval_variant",
                "model_param",
                "chosen_epoch",
                "chosen_epoch_mean",
            ]
        ],
        left_on=["source", "person_id", "seq_len", "eval_variant", "model_param", "eval_epoch"],
        right_on=["source", "person_id", "seq_len", "eval_variant", "model_param", "chosen_epoch"],
        how="inner",
    )

    return selected_rows, chosen


def run_sanity_checks(
    pear_df: pd.DataFrame,
    selected_rows: pd.DataFrame,
    chosen_epochs: pd.DataFrame,
    selection_label: str = "best",
) -> None:
    """Validate selected epochs and plotted values."""
    if pear_df.empty:
        return

    if selection_label == "best":
        # Chosen epoch must equal max mean epoch per model group.
        epoch_mean = (
            pear_df.groupby(
                ["source", "person_id", "seq_len", "eval_variant", "model_param", "eval_epoch"], as_index=False
            )["value"]
            .mean()
            .rename(columns={"value": "epoch_mean_pearson"})
        )
        max_mean = (
            epoch_mean.groupby(["source", "person_id", "seq_len", "eval_variant", "model_param"], as_index=False)[
                "epoch_mean_pearson"
            ]
            .max()
            .rename(columns={"epoch_mean_pearson": "max_epoch_mean"})
        )
        chk = chosen_epochs.merge(
            max_mean,
            on=["source", "person_id", "seq_len", "eval_variant", "model_param"],
            how="left",
        )
        bad = chk[(chk["chosen_epoch_mean"] - chk["max_epoch_mean"]).abs() > 1e-10]
        if not bad.empty:
            raise AssertionError(f"Sanity check failed: chosen epoch is not max mean for {len(bad)} groups")
    elif selection_label == "final":
        max_epoch = (
            pear_df.groupby(["source", "person_id", "seq_len", "eval_variant", "model_param"], as_index=False)[
                "eval_epoch"
            ]
            .max()
            .rename(columns={"eval_epoch": "max_eval_epoch"})
        )
        chk = chosen_epochs.merge(
            max_epoch,
            on=["source", "person_id", "seq_len", "eval_variant", "model_param"],
            how="left",
        )
        bad = chk[chk["chosen_epoch"] != chk["max_eval_epoch"]]
        if not bad.empty:
            raise AssertionError(f"Sanity check failed: chosen epoch is not final epoch for {len(bad)} groups")
    else:
        raise ValueError(f"Unsupported selection_label: {selection_label}")

    # Selected rows values must match raw rows for chosen epoch keys.
    selected_keys = selected_rows[
        ["source", "person_id", "seq_len", "eval_variant", "model_param", "eval_epoch", "variability_level", "value"]
    ].copy()
    raw_keys = pear_df[
        ["source", "person_id", "seq_len", "eval_variant", "model_param", "eval_epoch", "variability_level", "value"]
    ].copy()
    merged = selected_keys.merge(
        raw_keys,
        on=["source", "person_id", "seq_len", "eval_variant", "model_param", "eval_epoch", "variability_level"],
        suffixes=("_sel", "_raw"),
        how="left",
    )
    bad_rows = merged[(merged["value_sel"] - merged["value_raw"]).abs() > 1e-10]
    if not bad_rows.empty:
        raise AssertionError(f"Sanity check failed: selected values mismatch for {len(bad_rows)} rows")


def make_person_2x2_plots(
    selected_df: pd.DataFrame,
    output_dir: Path,
    left_name: str,
    right_name: str,
    show_plots: bool = False,
    selection_label: str = "best",
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if selected_df.empty:
        print("No data available for person 2x2 best-pearson plots.")
        return

    persons = sorted(selected_df["person_id"].dropna().unique().tolist())
    variants = ["full_finetune", "lora", "lora_over_lora"]
    variant_title = {
        "full_finetune": "regular",
        "lora": "lora",
        "lora_over_lora": "lora_over_lora",
    }
    seq_cols = [600, 5400]
    source_rows = [left_name, right_name]
    ylabel = "Best Pearsonr"
    title_metric = "best Pearsonr by variability level"
    out_suffix = "best_pearson"
    if selection_label == "final":
        ylabel = "Final-Epoch Pearsonr"
        title_metric = "final-epoch Pearsonr by variability level"
        out_suffix = "final_epoch_pearson"

    for person in persons:
        for variant in variants:
            person_variant = selected_df[
                (selected_df["person_id"] == person) & (selected_df["eval_variant"] == variant)
            ].copy()
            if person_variant.empty:
                continue

            model_params = sorted(person_variant["model_param"].dropna().unique().tolist())
            if not model_params:
                continue
            palette = plt.get_cmap("tab20")
            color_map = {m: palette(i % 20) for i, m in enumerate(model_params)}

            fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
            fig.suptitle(f"{person}: {title_metric} ({variant_title[variant]})", y=0.98)

            for r, source in enumerate(source_rows):
                for c, seq in enumerate(seq_cols):
                    ax = axes[r, c]
                    sub = selected_df[
                        (selected_df["person_id"] == person)
                        & (selected_df["source"] == source)
                        & (selected_df["seq_len"] == seq)
                        & (selected_df["eval_variant"] == variant)
                    ].copy()
                    ax.set_title(f"{source_display_name(source)} | seq_{seq}")
                    ax.set_xlabel("Variability level")
                    if c == 0:
                        ax.set_ylabel(ylabel)
                    ax.set_ylim(-0.1, 1.0)

                    if sub.empty:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                        continue

                    # Multiple bars per variability level: one bar per model_param.
                    levels = sorted(sub["variability_level"].dropna().astype(int).unique().tolist())
                    x = np.arange(len(levels), dtype=float)
                    n_models = len(model_params)
                    width = min(0.8 / max(n_models, 1), 0.25)
                    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

                    for i_model, model in enumerate(model_params):
                        model_sub = sub[sub["model_param"] == model].copy()
                        if model_sub.empty:
                            continue

                        level_to_val = {
                            int(v): float(y)
                            for v, y in zip(model_sub["variability_level"].tolist(), model_sub["value"].tolist())
                        }
                        heights = [level_to_val.get(lvl, np.nan) for lvl in levels]
                        bars = ax.bar(
                            x + offsets[i_model],
                            heights,
                            width=width * 0.95,
                            color=color_map[model],
                            alpha=0.9,
                            label=model,
                        )
                        for b in bars:
                            h = b.get_height()
                            if np.isnan(h):
                                continue
                            ax.text(
                                b.get_x() + b.get_width() / 2,
                                h + 0.015,
                                f"{h:.3f}",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                            )

                    ax.set_xticks(x)
                    ax.set_xticklabels([str(lvl) for lvl in levels])

            # Always show a figure-level legend with model-color mapping.
            legend_handles = [Patch(facecolor=color_map[m], edgecolor="none", label=m) for m in model_params]
            fig.legend(
                handles=legend_handles,
                title="Model parameter (color mapping)",
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=min(4, len(model_params)),
                frameon=True,
            )
            fig.tight_layout(rect=[0, 0.12, 1, 0.96])
            out_path = plots_dir / f"{person}_2x2_{out_suffix}_{variant}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close(fig)
            print(f"Wrote plot: {out_path}")


def make_single_project_all_participants_mean_plots(
    selected_df: pd.DataFrame,
    output_dir: Path,
    source_name: str,
    show_plots: bool = False,
    selection_label: str = "best",
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if selected_df.empty:
        print("No data available for single-project mean-pearson plots.")
        return

    avg_df = (
        selected_df.groupby(["seq_len", "eval_variant", "model_param", "variability_level"], as_index=False)
        .agg(value=("value", "mean"), n_participants=("person_id", "nunique"))
        .copy()
    )
    rows_name = "single_project_mean_pearson_rows.csv"
    if selection_label == "final":
        rows_name = "single_project_mean_pearson_final_epoch_rows.csv"
    avg_df.to_csv(output_dir / rows_name, index=False)

    seqs = sorted(avg_df["seq_len"].dropna().astype(int).unique().tolist())
    variants = ["full_finetune", "lora", "lora_over_lora"]

    for variant in variants:
        var_df = avg_df[avg_df["eval_variant"] == variant].copy()
        if var_df.empty:
            continue

        model_params = sorted(var_df["model_param"].dropna().unique().tolist())
        if not model_params:
            continue

        palette = plt.get_cmap("tab20")
        color_map = {m: palette(i % 20) for i, m in enumerate(model_params)}

        fig, axes = plt.subplots(1, len(seqs), figsize=(5.5 * len(seqs), 5.5), sharey=True)
        if len(seqs) == 1:
            axes = [axes]
        ylabel = "Mean Best Pearsonr"
        title_metric = "mean Pearsonr by variability level"
        out_suffix = "mean_pearson"
        if selection_label == "final":
            ylabel = "Mean Final-Epoch Pearsonr"
            title_metric = "mean final-epoch Pearsonr by variability level"
            out_suffix = "mean_pearson_final_epoch"
        fig.suptitle(
            f"All participants: {title_metric} ({source_display_name(source_name)} | {VARIANT_TITLES[variant]})",
            y=0.98,
        )

        for ax, seq in zip(axes, seqs):
            sub = var_df[var_df["seq_len"] == seq].copy()
            ax.set_title(f"seq_{seq}")
            ax.set_xlabel("Variability level")
            ax.set_ylabel(ylabel)
            ax.set_ylim(-0.1, 1.0)

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                continue

            levels = sorted(sub["variability_level"].dropna().astype(int).unique().tolist())
            x = np.arange(len(levels), dtype=float)
            n_models = len(model_params)
            width = min(0.8 / max(n_models, 1), 0.25)
            offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

            for i_model, model in enumerate(model_params):
                model_sub = sub[sub["model_param"] == model].copy()
                level_to_val = {
                    int(v): float(y)
                    for v, y in zip(model_sub["variability_level"].tolist(), model_sub["value"].tolist())
                }
                heights = [level_to_val.get(lvl, np.nan) for lvl in levels]
                bars = ax.bar(
                    x + offsets[i_model],
                    heights,
                    width=width * 0.95,
                    color=color_map[model],
                    alpha=0.9,
                    label=model,
                )
                for b in bars:
                    h = b.get_height()
                    if np.isnan(h):
                        continue
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        h + 0.015,
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([str(lvl) for lvl in levels])

        legend_handles = [Patch(facecolor=color_map[m], edgecolor="none", label=m) for m in model_params]
        fig.legend(
            handles=legend_handles,
            title="Model parameter (color mapping)",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(5, len(model_params)),
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 0.95])
        out_path = plots_dir / f"ALL_PARTICIPANTS_{source_name}_{out_suffix}_{variant_file_tag(variant)}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)
        print(f"Wrote plot: {out_path}")


def make_single_project_person_mean_plots(
    selected_df: pd.DataFrame,
    output_dir: Path,
    source_name: str,
    show_plots: bool = False,
    selection_label: str = "best",
) -> None:
    if selected_df.empty:
        print("No data available for single-project per-person mean-pearson plots.")
        return

    persons = sorted(selected_df["person_id"].dropna().unique().tolist())
    if not persons:
        print("No participant IDs found for single-project per-person mean-pearson plots.")
        return

    rows_name = "person_mean_pearson_rows.csv"
    ylabel = "Mean Best Pearsonr"
    title_metric = "mean Pearsonr by variability level"
    out_suffix = "mean_pearson"
    if selection_label == "final":
        rows_name = "person_mean_pearson_final_epoch_rows.csv"
        ylabel = "Mean Final-Epoch Pearsonr"
        title_metric = "mean final-epoch Pearsonr by variability level"
        out_suffix = "mean_pearson_final_epoch"

    variants = ["full_finetune", "lora", "lora_over_lora"]

    for person_id in persons:
        person_dir = output_dir / "per_person" / person_id
        plots_dir = person_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        person_selected = selected_df[selected_df["person_id"] == person_id].copy()
        if person_selected.empty:
            continue

        avg_df = (
            person_selected.groupby(["seq_len", "eval_variant", "model_param", "variability_level"], as_index=False)
            .agg(value=("value", "mean"))
            .copy()
        )
        avg_df.to_csv(person_dir / rows_name, index=False)

        seqs = sorted(avg_df["seq_len"].dropna().astype(int).unique().tolist())
        if not seqs:
            continue

        for variant in variants:
            var_df = avg_df[avg_df["eval_variant"] == variant].copy()
            if var_df.empty:
                continue

            model_params = sorted(var_df["model_param"].dropna().unique().tolist())
            if not model_params:
                continue

            palette = plt.get_cmap("tab20")
            color_map = {m: palette(i % 20) for i, m in enumerate(model_params)}

            fig, axes = plt.subplots(1, len(seqs), figsize=(5.5 * len(seqs), 5.5), sharey=True)
            if len(seqs) == 1:
                axes = [axes]

            fig.suptitle(
                f"{person_id}: {title_metric} ({source_display_name(source_name)} | {VARIANT_TITLES[variant]})",
                y=0.98,
            )

            for ax, seq in zip(axes, seqs):
                sub = var_df[var_df["seq_len"] == seq].copy()
                ax.set_title(f"seq_{seq}")
                ax.set_xlabel("Variability level")
                ax.set_ylabel(ylabel)
                ax.set_ylim(-0.1, 1.0)

                if sub.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                    continue

                levels = sorted(sub["variability_level"].dropna().astype(int).unique().tolist())
                x = np.arange(len(levels), dtype=float)
                n_models = len(model_params)
                width = min(0.8 / max(n_models, 1), 0.25)
                offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

                for i_model, model in enumerate(model_params):
                    model_sub = sub[sub["model_param"] == model].copy()
                    level_to_val = {
                        int(v): float(y)
                        for v, y in zip(model_sub["variability_level"].tolist(), model_sub["value"].tolist())
                    }
                    heights = [level_to_val.get(lvl, np.nan) for lvl in levels]
                    bars = ax.bar(
                        x + offsets[i_model],
                        heights,
                        width=width * 0.95,
                        color=color_map[model],
                        alpha=0.9,
                        label=model,
                    )
                    for b in bars:
                        h = b.get_height()
                        if np.isnan(h):
                            continue
                        ax.text(
                            b.get_x() + b.get_width() / 2,
                            h + 0.015,
                            f"{h:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

                ax.set_xticks(x)
                ax.set_xticklabels([str(lvl) for lvl in levels])

            legend_handles = [Patch(facecolor=color_map[m], edgecolor="none", label=m) for m in model_params]
            fig.legend(
                handles=legend_handles,
                title="Model parameter (color mapping)",
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=min(5, len(model_params)),
                frameon=True,
            )
            fig.tight_layout(rect=[0, 0.10, 1, 0.95])
            out_path = plots_dir / f"{person_id}_{source_name}_{out_suffix}_{variant_file_tag(variant)}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close(fig)
            print(f"Wrote plot: {out_path}")


def make_single_project_all_participants_epoch_trend_plots(
    pear_df: pd.DataFrame, output_dir: Path, source_name: str, eval_variant: str = "full_finetune", show_plots: bool = False
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    variant_title = VARIANT_TITLES.get(eval_variant, eval_variant)
    variant_tag = variant_file_tag(eval_variant)

    if pear_df.empty:
        print(f"No data available for single-project {variant_tag} epoch trend plots.")
        return

    all_df = pear_df[
        (pear_df["source"] == source_name)
        & (pear_df["eval_variant"] == eval_variant)
        & (pear_df["eval_epoch"].notna())
        & (pear_df["variability_level"].notna())
    ].copy()
    if all_df.empty:
        print(f"No {eval_variant} rows found for single-project epoch trend plots.")
        return

    all_df["eval_epoch"] = all_df["eval_epoch"].astype(int)
    all_df["variability_level"] = all_df["variability_level"].astype(int)

    trend = (
        all_df.groupby(["seq_len", "model_param", "eval_epoch", "variability_level"], as_index=False)
        .agg(value=("value", "mean"), n_participants=("person_id", "nunique"), n_files=("file_name", "nunique"))
    )
    trend.to_csv(output_dir / f"single_project_{variant_tag}_epoch_trend_rows.csv", index=False)

    seqs = sorted(trend["seq_len"].dropna().astype(int).unique().tolist())
    model_params = sorted(trend["model_param"].dropna().unique().tolist())
    palette = plt.get_cmap("tab20")
    color_map = {m: palette(i % 20) for i, m in enumerate(model_params)}

    for seq in seqs:
        sub = trend[trend["seq_len"] == seq].copy()
        levels = sorted(sub["variability_level"].dropna().astype(int).unique().tolist())
        if not levels:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
        fig.suptitle(
            f"All participants: mean Pearsonr by epoch and variability level ({source_display_name(source_name)} | seq_{seq} | {variant_title})",
            y=0.98,
        )
        ax_list = axes.flatten()

        for i, lvl in enumerate(levels[:6]):
            ax = ax_list[i]
            lvl_sub = sub[sub["variability_level"] == lvl].copy()
            ax.set_title(f"Level {lvl}")
            ax.set_xlabel("Epoch")
            if i % 3 == 0:
                ax.set_ylabel("Mean Pearsonr")
            ax.set_ylim(-0.1, 1.0)

            for model in model_params:
                msub = lvl_sub[lvl_sub["model_param"] == model].sort_values("eval_epoch")
                if msub.empty:
                    continue
                ax.plot(
                    msub["eval_epoch"].tolist(),
                    msub["value"].tolist(),
                    marker="o",
                    linewidth=2,
                    color=color_map[model],
                    label=model,
                )
                ax.set_xticks(sorted(msub["eval_epoch"].unique().tolist()))

            if lvl_sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)

        for j in range(len(levels), 6):
            ax_list[j].axis("off")

        legend_handles = [Patch(facecolor=color_map[m], edgecolor="none", label=m) for m in model_params]
        fig.legend(
            handles=legend_handles,
            title="Model parameter (line color)",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(5, len(model_params)),
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 0.96])
        out_path = plots_dir / f"ALL_PARTICIPANTS_{source_name}_epoch_trend_{variant_tag}_seq_{seq}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)
        print(f"Wrote plot: {out_path}")


def make_single_project_all_participants_strategy_best_plot(
    pear_df: pd.DataFrame, output_dir: Path, source_name: str, show_plots: bool = False
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cand = _strategy_candidates(pear_df)
    cand = cand[cand["source"] == source_name].copy()
    if cand.empty:
        print("No data available for single-project strategy comparison plot.")
        return

    per_person_best = (
        cand.groupby(["person_id", "seq_len", "variability_level", "strategy"], as_index=False)["value"]
        .max()
        .rename(columns={"value": "best_value"})
    )
    avg_best = (
        per_person_best.groupby(["seq_len", "variability_level", "strategy"], as_index=False)
        .agg(value=("best_value", "mean"), n_participants=("person_id", "nunique"))
        .copy()
    )
    avg_best.to_csv(output_dir / "single_project_best_strategy_by_variability_rows.csv", index=False)

    seqs = sorted(avg_best["seq_len"].dropna().astype(int).unique().tolist())
    strategies = ["full_finetune", "lora", "lora_over_lora"]
    color_map = {
        "full_finetune": "#ff7f0e",
        "lora": "#1f77b4",
        "lora_over_lora": "#2ca02c",
    }

    fig, axes = plt.subplots(1, len(seqs), figsize=(5.5 * len(seqs), 5.5), sharey=True)
    if len(seqs) == 1:
        axes = [axes]
    fig.suptitle(f"All participants: best strategy comparison by variability level ({source_display_name(source_name)})", y=0.98)

    for ax, seq in zip(axes, seqs):
        sub = avg_best[avg_best["seq_len"] == seq].copy()
        ax.set_title(f"seq_{seq}")
        ax.set_xlabel("Variability level")
        ax.set_ylabel("Mean Best Pearsonr")
        ax.set_ylim(-0.1, 1.0)

        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
            continue

        levels = sorted(sub["variability_level"].dropna().astype(int).unique().tolist())
        x = np.arange(len(levels), dtype=float)
        n_strat = len(strategies)
        width = min(0.8 / max(n_strat, 1), 0.25)
        offsets = (np.arange(n_strat) - (n_strat - 1) / 2.0) * width

        for i_s, strat in enumerate(strategies):
            ssub = sub[sub["strategy"] == strat].copy()
            lvl_to_val = {int(v): float(y) for v, y in zip(ssub["variability_level"], ssub["value"])}
            heights = [lvl_to_val.get(lvl, np.nan) for lvl in levels]
            bars = ax.bar(
                x + offsets[i_s],
                heights,
                width=width * 0.95,
                color=color_map[strat],
                alpha=0.9,
                label=strat,
            )
            for b in bars:
                h = b.get_height()
                if np.isnan(h):
                    continue
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + 0.015,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in levels])

    legend_handles = [Patch(facecolor=color_map[s], edgecolor="none", label=s) for s in strategies]
    fig.legend(
        handles=legend_handles,
        title="Strategy",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    out_path = plots_dir / f"ALL_PARTICIPANTS_{source_name}_best_strategy_compare_by_variability.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def make_all_participants_2x2_plots(
    selected_df: pd.DataFrame,
    output_dir: Path,
    left_name: str,
    right_name: str,
    show_plots: bool = False,
    selection_label: str = "best",
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if selected_df.empty:
        print("No data available for all-participants 2x2 mean-pearson plots.")
        return

    avg_df = (
        selected_df.groupby(
            ["source", "seq_len", "eval_variant", "model_param", "variability_level"], as_index=False
        )
        .agg(
            value=("value", "mean"),
            n_participants=("person_id", "nunique"),
        )
        .copy()
    )
    rows_name = "mean_pearson_across_participants_rows.csv"
    title_metric = "mean Pearsonr by variability level"
    ylabel = "Mean Best Pearsonr"
    out_suffix = "mean_pearson"
    if selection_label == "final":
        rows_name = "mean_pearson_across_participants_final_epoch_rows.csv"
        title_metric = "mean final-epoch Pearsonr by variability level"
        ylabel = "Mean Final-Epoch Pearsonr"
        out_suffix = "mean_pearson_final_epoch"
    avg_df.to_csv(output_dir / rows_name, index=False)

    variants = ["full_finetune", "lora", "lora_over_lora"]
    variant_title = {
        "full_finetune": "regular",
        "lora": "lora",
        "lora_over_lora": "lora_over_lora",
    }
    seq_cols = [600, 5400]
    source_rows = [left_name, right_name]

    for variant in variants:
        var_df = avg_df[avg_df["eval_variant"] == variant].copy()
        if var_df.empty:
            continue

        model_params = sorted(var_df["model_param"].dropna().unique().tolist())
        if not model_params:
            continue
        palette = plt.get_cmap("tab20")
        color_map = {m: palette(i % 20) for i, m in enumerate(model_params)}

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
        fig.suptitle(f"All participants: {title_metric} ({variant_title[variant]})", y=0.98)

        for r, source in enumerate(source_rows):
            for c, seq in enumerate(seq_cols):
                ax = axes[r, c]
                sub = var_df[(var_df["source"] == source) & (var_df["seq_len"] == seq)].copy()
                ax.set_title(f"{source_display_name(source)} | seq_{seq}")
                ax.set_xlabel("Variability level")
                if c == 0:
                    ax.set_ylabel(ylabel)
                ax.set_ylim(-0.1, 1.0)

                if sub.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                    continue

                levels = sorted(sub["variability_level"].dropna().astype(int).unique().tolist())
                x = np.arange(len(levels), dtype=float)
                n_models = len(model_params)
                width = min(0.8 / max(n_models, 1), 0.25)
                offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

                for i_model, model in enumerate(model_params):
                    model_sub = sub[sub["model_param"] == model].copy()
                    if model_sub.empty:
                        continue

                    level_to_val = {
                        int(v): float(y)
                        for v, y in zip(model_sub["variability_level"].tolist(), model_sub["value"].tolist())
                    }
                    heights = [level_to_val.get(lvl, np.nan) for lvl in levels]
                    bars = ax.bar(
                        x + offsets[i_model],
                        heights,
                        width=width * 0.95,
                        color=color_map[model],
                        alpha=0.9,
                        label=model,
                    )
                    for b in bars:
                        h = b.get_height()
                        if np.isnan(h):
                            continue
                        ax.text(
                            b.get_x() + b.get_width() / 2,
                            h + 0.015,
                            f"{h:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

                ax.set_xticks(x)
                ax.set_xticklabels([str(lvl) for lvl in levels])

        legend_handles = [Patch(facecolor=color_map[m], edgecolor="none", label=m) for m in model_params]
        fig.legend(
            handles=legend_handles,
            title="Model parameter (color mapping)",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(4, len(model_params)),
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.12, 1, 0.96])
        out_path = plots_dir / f"ALL_PARTICIPANTS_2x2_{out_suffix}_{variant}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)
        print(f"Wrote plot: {out_path}")


def _assign_training_phase_group(pear_df: pd.DataFrame) -> pd.DataFrame:
    if pear_df.empty:
        return pear_df.copy()

    df = pear_df.copy()
    df["training_phase_group"] = np.select(
        [
            (df["model_param"] == "epoch 1") & (df["eval_epoch"] == 0),
            (df["model_param"] == "epoch 1") & (df["eval_epoch"] > 0),
            (df["model_param"] == "epoch 2") & (df["eval_epoch"] == 0),
            (df["model_param"] == "epoch 2") & (df["eval_epoch"] > 0),
            df["model_param"] == "no_pretrain",
        ],
        TRAINING_PHASE_ORDER,
        default=None,
    )
    return df[df["training_phase_group"].notna()].copy()


def _select_last_epoch_values(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Select rows from the last available eval epoch within each group."""
    if df.empty:
        return df.copy()

    last_epoch = (
        df.groupby(group_cols, as_index=False)["eval_epoch"]
        .max()
        .rename(columns={"eval_epoch": "selected_eval_epoch"})
    )
    return (
        df.merge(
            last_epoch,
            left_on=group_cols + ["eval_epoch"],
            right_on=group_cols + ["selected_eval_epoch"],
            how="inner",
        )
        .groupby(group_cols + ["selected_eval_epoch"], as_index=False)
        .agg(value=("value", "mean"))
    )


def make_all_participants_strategy_training_phase_plots(
    pear_df: pd.DataFrame, output_dir: Path, left_name: str, right_name: str, show_plots: bool = False
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped_df = _assign_training_phase_group(pear_df)
    if grouped_df.empty:
        print("No data available for strategy/training-phase plots.")
        return

    group_cols = ["person_id", "source", "seq_len", "eval_variant", "variability_level", "training_phase_group"]
    per_person_last = _select_last_epoch_values(grouped_df, group_cols).rename(columns={"value": "last_value"})
    avg_df = (
        per_person_last.groupby(
            ["source", "seq_len", "eval_variant", "variability_level", "training_phase_group"], as_index=False
        )
        .agg(
            value=("last_value", "mean"),
            n_participants=("person_id", "nunique"),
        )
        .copy()
    )
    avg_df = avg_df[avg_df["eval_variant"] == "full_finetune"].copy()
    avg_df.to_csv(output_dir / "all_participants_non_lora_training_phase_by_variability_rows.csv", index=False)

    levels = sorted(avg_df["variability_level"].dropna().astype(int).unique().tolist())
    if not levels:
        print("No variability levels available for non-lora training-phase plots.")
        return

    sources = [left_name, right_name]
    seq_cols = [600, 5400]
    strategies = ["full_finetune"]

    for lvl in levels:
        lvl_df = avg_df[avg_df["variability_level"] == lvl].copy()
        level_min = float(lvl_df["value"].min())
        level_max = float(lvl_df["value"].max())
        if level_min == level_max:
            level_pad = 0.05
        else:
            level_pad = max(0.03, (level_max - level_min) * 0.08)
        y_min = level_min - level_pad
        y_max = level_max + level_pad
        label_offset = max(0.008, (y_max - y_min) * 0.02)

        fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
        fig.suptitle(f"All participants: mean last-epoch Pearsonr by training phase (Level {lvl} | regular)", y=0.98)

        for r, source in enumerate(sources):
            for c, seq in enumerate(seq_cols):
                ax = axes[r, c]
                sub = avg_df[
                    (avg_df["source"] == source)
                    & (avg_df["seq_len"] == seq)
                    & (avg_df["variability_level"] == lvl)
                ].copy()
                ax.set_title(f"{source_display_name(source)} | seq_{seq}")
                ax.set_xlabel("Strategy")
                if c == 0:
                    ax.set_ylabel("Mean Last-Epoch Pearsonr")
                ax.set_ylim(y_min, y_max)

                if sub.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                    continue

                x = np.arange(len(strategies), dtype=float)
                n_phases = len(TRAINING_PHASE_ORDER)
                width = min(0.8 / max(n_phases, 1), 0.18)
                offsets = (np.arange(n_phases) - (n_phases - 1) / 2.0) * width

                for i_phase, phase in enumerate(TRAINING_PHASE_ORDER):
                    phase_sub = sub[sub["training_phase_group"] == phase].copy()
                    strat_to_val = {
                        variant: float(val)
                        for variant, val in zip(phase_sub["eval_variant"].tolist(), phase_sub["value"].tolist())
                    }
                    heights = [strat_to_val.get(strategy, np.nan) for strategy in strategies]
                    bars = ax.bar(
                        x + offsets[i_phase],
                        heights,
                        width=width * 0.95,
                        color=TRAINING_PHASE_COLORS[phase],
                        alpha=0.9,
                        label=TRAINING_PHASE_LABELS[phase],
                    )
                    for b in bars:
                        h = b.get_height()
                        if np.isnan(h):
                            continue
                        ax.text(
                            b.get_x() + b.get_width() / 2,
                            h + label_offset,
                            f"{h:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                        )

                ax.set_xticks(x)
                ax.set_xticklabels([VARIANT_TITLES[strategy] for strategy in strategies])

        legend_handles = [
            Patch(facecolor=TRAINING_PHASE_COLORS[phase], edgecolor="none", label=TRAINING_PHASE_LABELS[phase])
            for phase in TRAINING_PHASE_ORDER
        ]
        fig.legend(
            handles=legend_handles,
            title="Training Phase",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 0.96])
        out_path = plots_dir / f"ALL_PARTICIPANTS_2x2_non_lora_training_phase_level_{lvl}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)
        print(f"Wrote plot: {out_path}")




def make_all_participants_epoch1_phase_comparison_plot(
    pear_df: pd.DataFrame, output_dir: Path, left_name: str, right_name: str, show_plots: bool = False
) -> None:
    """2x2 grouped bars by variability level for epoch-1 phase comparison across all participants."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped_df = _assign_training_phase_group(pear_df)
    if grouped_df.empty:
        print("No data available for epoch-1 phase comparison plot.")
        return

    comparison_labels = {
        "first_phase_only_epoch1": "1st phase only (e1)",
        "second_phase_from_epoch1": "1st + 2nd phase",
        "no_pretrain": "2nd phase only",
    }
    comparison_order = ["first_phase_only_epoch1", "second_phase_from_epoch1", "no_pretrain"]

    comp_df = grouped_df[
        (grouped_df["eval_variant"] == "full_finetune")
        & (grouped_df["training_phase_group"].isin(comparison_order))
    ].copy()
    if comp_df.empty:
        print("No non-lora rows available for epoch-1 phase comparison plot.")
        return

    group_cols = ["person_id", "source", "seq_len", "variability_level", "training_phase_group"]
    per_person_last = _select_last_epoch_values(comp_df, group_cols).rename(columns={"value": "selected_value"})
    avg_df = (
        per_person_last.groupby(["source", "seq_len", "variability_level", "training_phase_group"], as_index=False)
        .agg(
            value=("selected_value", "mean"),
            n_participants=("person_id", "nunique"),
        )
        .copy()
    )
    avg_df["comparison_label"] = avg_df["training_phase_group"].map(comparison_labels)
    avg_df.to_csv(output_dir / "all_participants_non_lora_epoch1_phase_compare_rows.csv", index=False)

    if avg_df.empty:
        print("No aggregated rows available for epoch-1 phase comparison plot.")
        return

    fig_min = float(avg_df["value"].min())
    fig_max = float(avg_df["value"].max())
    if fig_min == fig_max:
        fig_pad = 0.05
    else:
        fig_pad = max(0.03, (fig_max - fig_min) * 0.08)
    y_min = fig_min - fig_pad
    y_max = fig_max + fig_pad
    label_offset = max(0.008, (y_max - y_min) * 0.02)

    sources = [left_name, right_name]
    seqs = [600, 5400]
    levels = sorted(avg_df["variability_level"].dropna().astype(int).unique().tolist())
    color_map = EPOCH1_PHASE_COMPARE_COLORS

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
    fig.suptitle("All participants: epoch 1 phase comparison by variability level (regular)", y=0.98)

    for r, source in enumerate(sources):
        for c, seq in enumerate(seqs):
            ax = axes[r, c]
            sub = avg_df[(avg_df["source"] == source) & (avg_df["seq_len"] == seq)].copy()
            ax.set_title(f"{source_display_name(source)} | seq_{seq}")
            ax.set_xlabel("Variability level")
            if c == 0:
                ax.set_ylabel("Mean Pearsonr")
            ax.set_ylim(y_min, y_max)

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                continue

            x = np.arange(len(levels), dtype=float)
            n_bars = len(comparison_order)
            width = min(0.8 / max(n_bars, 1), 0.25)
            offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * width

            for i_group, group_name in enumerate(comparison_order):
                group_sub = sub[sub["training_phase_group"] == group_name].copy()
                lvl_to_val = {
                    int(v): float(y)
                    for v, y in zip(group_sub["variability_level"].tolist(), group_sub["value"].tolist())
                }
                heights = [lvl_to_val.get(lvl, np.nan) for lvl in levels]
                bars = ax.bar(
                    x + offsets[i_group],
                    heights,
                    width=width * 0.95,
                    color=color_map[group_name],
                    alpha=0.9,
                    label=comparison_labels[group_name],
                )
                for b in bars:
                    h = b.get_height()
                    if np.isnan(h):
                        continue
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        h + label_offset,
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([str(level) for level in levels])

    legend_handles = [
        Patch(facecolor=color_map[group_name], edgecolor="none", label=comparison_labels[group_name])
        for group_name in comparison_order
    ]
    fig.legend(
        handles=legend_handles,
        title="Training setup",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    out_path = plots_dir / "ALL_PARTICIPANTS_2x2_non_lora_epoch1_phase_compare.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"Wrote plot: {out_path}")




def make_all_participants_epoch1_phase_comparison_without_second_phase_only_plot(
    pear_df: pd.DataFrame, output_dir: Path, left_name: str, right_name: str, show_plots: bool = False
) -> None:
    """2x2 grouped bars by variability level for epoch-1 comparison without no-pretrain bar."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped_df = _assign_training_phase_group(pear_df)
    if grouped_df.empty:
        print("No data available for epoch-1 phase comparison without second-phase-only plot.")
        return

    comparison_labels = {
        "first_phase_only_epoch1": "1st phase only (e1)",
        "second_phase_from_epoch1": "1st + 2nd phase",
    }
    comparison_order = ["first_phase_only_epoch1", "second_phase_from_epoch1"]

    comp_df = grouped_df[
        (grouped_df["eval_variant"] == "full_finetune")
        & (grouped_df["training_phase_group"].isin(comparison_order))
    ].copy()
    if comp_df.empty:
        print("No non-lora rows available for epoch-1 phase comparison without second-phase-only plot.")
        return

    group_cols = ["person_id", "source", "seq_len", "variability_level", "training_phase_group"]
    per_person_last = _select_last_epoch_values(comp_df, group_cols).rename(columns={"value": "selected_value"})
    avg_df = (
        per_person_last.groupby(["source", "seq_len", "variability_level", "training_phase_group"], as_index=False)
        .agg(
            value=("selected_value", "mean"),
            n_participants=("person_id", "nunique"),
        )
        .copy()
    )
    avg_df["comparison_label"] = avg_df["training_phase_group"].map(comparison_labels)
    avg_df.to_csv(output_dir / "all_participants_non_lora_epoch1_phase_compare_without_second_phase_only_rows.csv", index=False)

    if avg_df.empty:
        print("No aggregated rows available for epoch-1 phase comparison without second-phase-only plot.")
        return

    fig_min = float(avg_df["value"].min())
    fig_max = float(avg_df["value"].max())
    if fig_min == fig_max:
        fig_pad = 0.05
    else:
        fig_pad = max(0.03, (fig_max - fig_min) * 0.08)
    y_min = fig_min - fig_pad
    y_max = fig_max + fig_pad
    label_offset = max(0.008, (y_max - y_min) * 0.02)

    sources = [left_name, right_name]
    seqs = [600, 5400]
    levels = sorted(avg_df["variability_level"].dropna().astype(int).unique().tolist())
    color_map = {
        key: EPOCH1_PHASE_COMPARE_COLORS[key]
        for key in comparison_order
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True)
    fig.suptitle(
        "All participants: epoch 1 phase comparison without 2nd phase only by variability level (regular)", y=0.98
    )

    for r, source in enumerate(sources):
        for c, seq in enumerate(seqs):
            ax = axes[r, c]
            sub = avg_df[(avg_df["source"] == source) & (avg_df["seq_len"] == seq)].copy()
            ax.set_title(f"{source_display_name(source)} | seq_{seq}")
            ax.set_xlabel("Variability level")
            if c == 0:
                ax.set_ylabel("Mean Pearsonr")
            ax.set_ylim(y_min, y_max)

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                continue

            x = np.arange(len(levels), dtype=float)
            n_bars = len(comparison_order)
            width = min(0.8 / max(n_bars, 1), 0.30)
            offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * width

            for i_group, group_name in enumerate(comparison_order):
                group_sub = sub[sub["training_phase_group"] == group_name].copy()
                lvl_to_val = {
                    int(v): float(y)
                    for v, y in zip(group_sub["variability_level"].tolist(), group_sub["value"].tolist())
                }
                heights = [lvl_to_val.get(lvl, np.nan) for lvl in levels]
                bars = ax.bar(
                    x + offsets[i_group],
                    heights,
                    width=width * 0.95,
                    color=color_map[group_name],
                    alpha=0.9,
                    label=comparison_labels[group_name],
                )
                for b in bars:
                    h = b.get_height()
                    if np.isnan(h):
                        continue
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        h + label_offset,
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([str(level) for level in levels])

    legend_handles = [
        Patch(facecolor=color_map[group_name], edgecolor="none", label=comparison_labels[group_name])
        for group_name in comparison_order
    ]
    fig.legend(
        handles=legend_handles,
        title="Training setup",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    out_path = plots_dir / "ALL_PARTICIPANTS_2x2_non_lora_epoch1_phase_compare_without_second_phase_only.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def make_person_non_lora_epoch_trend_plots(
    pear_df: pd.DataFrame,
    person_id: str,
    output_dir: Path,
    left_name: str,
    right_name: str,
    eval_variant: str = "full_finetune",
    show_plots: bool = False,
) -> None:
    """Plot person Pearsonr trends over eval epochs for one eval variant.

    For each (source, seq_len), generate a figure with one subplot per variability level.
    Each line is a model parameter family (no_pretrain / epoch 1 / epoch 2), with values
    averaged when multiple files map to the same key.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    variant_title = VARIANT_TITLES.get(eval_variant, eval_variant)
    variant_tag = variant_file_tag(eval_variant)

    if pear_df.empty:
        print(f"No data available for {person_id} {variant_tag} epoch trend plots.")
        return

    person_df = pear_df[
        (pear_df["person_id"] == person_id)
        & (pear_df["eval_variant"] == eval_variant)
        & (pear_df["eval_epoch"].notna())
        & (pear_df["variability_level"].notna())
    ].copy()
    if person_df.empty:
        print(f"No {person_id} {eval_variant} rows found for epoch trend plots.")
        return

    person_df["eval_epoch"] = person_df["eval_epoch"].astype(int)
    person_df["variability_level"] = person_df["variability_level"].astype(int)

    trend = (
        person_df.groupby(
            ["source", "seq_len", "model_param", "eval_epoch", "variability_level"], as_index=False
        ).agg(value=("value", "mean"), n_files=("file_name", "nunique"))
    )
    trend.to_csv(output_dir / f"{person_id}_{variant_tag}_epoch_trend_rows.csv", index=False)

    sources = [left_name, right_name]
    seqs = [600, 5400]
    model_params = sorted(trend["model_param"].dropna().unique().tolist())
    palette = plt.get_cmap("tab20")
    color_map = {m: palette(i % 20) for i, m in enumerate(model_params)}

    for source in sources:
        for seq in seqs:
            sub = trend[(trend["source"] == source) & (trend["seq_len"] == seq)].copy()
            if sub.empty:
                continue

            levels = sorted(sub["variability_level"].dropna().astype(int).unique().tolist())
            if not levels:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
            fig.suptitle(
                f"{person_id}: Pearsonr by epoch and variability level "
                f"({source_display_name(source)} | seq_{seq} | {variant_title})",
                y=0.98,
            )
            ax_list = axes.flatten()

            for i, lvl in enumerate(levels[:6]):
                ax = ax_list[i]
                lvl_sub = sub[sub["variability_level"] == lvl].copy()
                ax.set_title(f"Level {lvl}")
                ax.set_xlabel("Epoch")
                if i % 3 == 0:
                    ax.set_ylabel("Pearsonr")
                ax.set_ylim(-0.1, 1.0)

                for model in model_params:
                    msub = lvl_sub[lvl_sub["model_param"] == model].sort_values("eval_epoch")
                    if msub.empty:
                        continue
                    ax.plot(
                        msub["eval_epoch"].tolist(),
                        msub["value"].tolist(),
                        marker="o",
                        linewidth=2,
                        color=color_map[model],
                        label=model,
                    )
                    ax.set_xticks(sorted(msub["eval_epoch"].unique().tolist()))

                if lvl_sub.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)

            for j in range(len(levels), 6):
                ax_list[j].axis("off")

            legend_handles = [Patch(facecolor=color_map[m], edgecolor="none", label=m) for m in model_params]
            fig.legend(
                handles=legend_handles,
                title="Model parameter (line color)",
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=min(4, len(model_params)),
                frameon=True,
            )
            fig.tight_layout(rect=[0, 0.10, 1, 0.96])

            out_path = plots_dir / f"{person_id}_epoch_trend_{variant_tag}_{source}_seq_{seq}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close(fig)
            print(f"Wrote plot: {out_path}")


def make_person_non_lora_2x2_by_variability_plots(
    pear_df: pd.DataFrame,
    person_id: str,
    output_dir: Path,
    left_name: str,
    right_name: str,
    eval_variant: str = "full_finetune",
    show_plots: bool = False,
) -> None:
    """For each variability level, create a 2x2 plot across source x seq_len for one person/variant."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    variant_title = VARIANT_TITLES.get(eval_variant, eval_variant)
    variant_tag = variant_file_tag(eval_variant)

    if pear_df.empty:
        print(f"No data available for {person_id} {variant_tag} 2x2-by-variability plots.")
        return

    person_df = pear_df[
        (pear_df["person_id"] == person_id)
        & (pear_df["eval_variant"] == eval_variant)
        & (pear_df["eval_epoch"].notna())
        & (pear_df["variability_level"].notna())
    ].copy()
    if person_df.empty:
        print(f"No {person_id} {eval_variant} rows found for 2x2-by-variability plots.")
        return

    person_df["eval_epoch"] = person_df["eval_epoch"].astype(int)
    person_df["variability_level"] = person_df["variability_level"].astype(int)

    trend = (
        person_df.groupby(
            ["source", "seq_len", "model_param", "eval_epoch", "variability_level"], as_index=False
        ).agg(value=("value", "mean"), n_files=("file_name", "nunique"))
    )

    trend.to_csv(output_dir / f"{person_id}_{variant_tag}_2x2_by_variability_rows.csv", index=False)

    sources = [left_name, right_name]
    seqs = [600, 5400]
    model_params = sorted(trend["model_param"].dropna().unique().tolist())
    levels = sorted(trend["variability_level"].dropna().astype(int).unique().tolist())

    if not levels or not model_params:
        print(f"No levels/model parameters available for {person_id} 2x2-by-variability plots.")
        return

    palette = plt.get_cmap("tab20")
    color_map = {m: palette(i % 20) for i, m in enumerate(model_params)}

    for lvl in levels:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        fig.suptitle(f"{person_id}: Pearsonr by epoch (Level {lvl} | {variant_title})", y=0.98)

        for r, source in enumerate(sources):
            for c, seq in enumerate(seqs):
                ax = axes[r, c]
                sub = trend[
                    (trend["source"] == source)
                    & (trend["seq_len"] == seq)
                    & (trend["variability_level"] == lvl)
                ].copy()

                ax.set_title(f"{source_display_name(source)} | seq_{seq}")
                ax.set_xlabel("Epoch")
                if c == 0:
                    ax.set_ylabel("Pearsonr")
                ax.set_ylim(-0.1, 1.0)

                if sub.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                    continue

                epoch_ticks = sorted(sub["eval_epoch"].dropna().astype(int).unique().tolist())
                for model in model_params:
                    msub = sub[sub["model_param"] == model].sort_values("eval_epoch")
                    if msub.empty:
                        continue
                    ax.plot(
                        msub["eval_epoch"].tolist(),
                        msub["value"].tolist(),
                        marker="o",
                        linewidth=2,
                        color=color_map[model],
                        label=model,
                    )
                if epoch_ticks:
                    ax.set_xticks(epoch_ticks)

        legend_handles = [Patch(facecolor=color_map[m], edgecolor="none", label=m) for m in model_params]
        fig.legend(
            handles=legend_handles,
            title="Model parameter (line color)",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(4, len(model_params)),
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 0.96])
        out_path = plots_dir / f"{person_id}_2x2_epoch_trend_{variant_tag}_level_{lvl}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)
        print(f"Wrote plot: {out_path}")


def make_all_participants_non_lora_epoch_trend_plots(
    pear_df: pd.DataFrame,
    output_dir: Path,
    left_name: str,
    right_name: str,
    eval_variant: str = "full_finetune",
    show_plots: bool = False,
) -> None:
    """Plot mean Pearsonr trends over eval epochs across all participants for one eval variant."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    variant_title = VARIANT_TITLES.get(eval_variant, eval_variant)
    variant_tag = variant_file_tag(eval_variant)

    if pear_df.empty:
        print(f"No data available for all-participants {variant_tag} epoch trend plots.")
        return

    all_df = pear_df[
        (pear_df["eval_variant"] == eval_variant)
        & (pear_df["eval_epoch"].notna())
        & (pear_df["variability_level"].notna())
    ].copy()
    if all_df.empty:
        print(f"No {eval_variant} rows found for all-participants epoch trend plots.")
        return

    all_df["eval_epoch"] = all_df["eval_epoch"].astype(int)
    all_df["variability_level"] = all_df["variability_level"].astype(int)

    trend = (
        all_df.groupby(
            ["source", "seq_len", "model_param", "eval_epoch", "variability_level"], as_index=False
        ).agg(
            value=("value", "mean"),
            n_participants=("person_id", "nunique"),
            n_files=("file_name", "nunique"),
        )
    )
    trend.to_csv(output_dir / f"all_participants_{variant_tag}_epoch_trend_rows.csv", index=False)

    sources = [left_name, right_name]
    seqs = [600, 5400]
    model_params = sorted(trend["model_param"].dropna().unique().tolist())
    palette = plt.get_cmap("tab20")
    color_map = {m: palette(i % 20) for i, m in enumerate(model_params)}

    for source in sources:
        for seq in seqs:
            sub = trend[(trend["source"] == source) & (trend["seq_len"] == seq)].copy()
            if sub.empty:
                continue

            levels = sorted(sub["variability_level"].dropna().astype(int).unique().tolist())
            if not levels:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
            fig.suptitle(
                f"All participants: mean Pearsonr by epoch and variability level "
                f"({source_display_name(source)} | seq_{seq} | {variant_title})",
                y=0.98,
            )
            ax_list = axes.flatten()

            for i, lvl in enumerate(levels[:6]):
                ax = ax_list[i]
                lvl_sub = sub[sub["variability_level"] == lvl].copy()
                ax.set_title(f"Level {lvl}")
                ax.set_xlabel("Epoch")
                if i % 3 == 0:
                    ax.set_ylabel("Mean Pearsonr")
                ax.set_ylim(-0.1, 1.0)

                for model in model_params:
                    msub = lvl_sub[lvl_sub["model_param"] == model].sort_values("eval_epoch")
                    if msub.empty:
                        continue
                    ax.plot(
                        msub["eval_epoch"].tolist(),
                        msub["value"].tolist(),
                        marker="o",
                        linewidth=2,
                        color=color_map[model],
                        label=model,
                    )
                    ax.set_xticks(sorted(msub["eval_epoch"].unique().tolist()))

                if lvl_sub.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)

            for j in range(len(levels), 6):
                ax_list[j].axis("off")

            legend_handles = [Patch(facecolor=color_map[m], edgecolor="none", label=m) for m in model_params]
            fig.legend(
                handles=legend_handles,
                title="Model parameter (line color)",
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=min(4, len(model_params)),
                frameon=True,
            )
            fig.tight_layout(rect=[0, 0.10, 1, 0.96])

            out_path = plots_dir / f"ALL_PARTICIPANTS_epoch_trend_{variant_tag}_{source}_seq_{seq}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close(fig)
            print(f"Wrote plot: {out_path}")


def make_all_participants_non_lora_2x2_by_variability_plots(
    pear_df: pd.DataFrame,
    output_dir: Path,
    left_name: str,
    right_name: str,
    eval_variant: str = "full_finetune",
    show_plots: bool = False,
) -> None:
    """For each variability level, create a 2x2 plot across source x seq_len using participant mean."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    variant_title = VARIANT_TITLES.get(eval_variant, eval_variant)
    variant_tag = variant_file_tag(eval_variant)

    if pear_df.empty:
        print(f"No data available for all-participants {variant_tag} 2x2-by-variability plots.")
        return

    all_df = pear_df[
        (pear_df["eval_variant"] == eval_variant)
        & (pear_df["eval_epoch"].notna())
        & (pear_df["variability_level"].notna())
    ].copy()
    if all_df.empty:
        print(f"No {eval_variant} rows found for all-participants 2x2-by-variability plots.")
        return

    all_df["eval_epoch"] = all_df["eval_epoch"].astype(int)
    all_df["variability_level"] = all_df["variability_level"].astype(int)

    trend = (
        all_df.groupby(
            ["source", "seq_len", "model_param", "eval_epoch", "variability_level"], as_index=False
        ).agg(
            value=("value", "mean"),
            n_participants=("person_id", "nunique"),
            n_files=("file_name", "nunique"),
        )
    )
    trend.to_csv(output_dir / f"all_participants_{variant_tag}_2x2_by_variability_rows.csv", index=False)

    sources = [left_name, right_name]
    seqs = [600, 5400]
    model_params = sorted(trend["model_param"].dropna().unique().tolist())
    levels = sorted(trend["variability_level"].dropna().astype(int).unique().tolist())

    if not levels or not model_params:
        print("No levels/model parameters available for all-participants 2x2-by-variability plots.")
        return

    palette = plt.get_cmap("tab20")
    color_map = {m: palette(i % 20) for i, m in enumerate(model_params)}

    for lvl in levels:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        fig.suptitle(f"All participants: mean Pearsonr by epoch (Level {lvl} | {variant_title})", y=0.98)

        for r, source in enumerate(sources):
            for c, seq in enumerate(seqs):
                ax = axes[r, c]
                sub = trend[
                    (trend["source"] == source)
                    & (trend["seq_len"] == seq)
                    & (trend["variability_level"] == lvl)
                ].copy()

                ax.set_title(f"{source_display_name(source)} | seq_{seq}")
                ax.set_xlabel("Epoch")
                if c == 0:
                    ax.set_ylabel("Mean Pearsonr")
                ax.set_ylim(-0.1, 1.0)

                if sub.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                    continue

                epoch_ticks = sorted(sub["eval_epoch"].dropna().astype(int).unique().tolist())
                for model in model_params:
                    msub = sub[sub["model_param"] == model].sort_values("eval_epoch")
                    if msub.empty:
                        continue
                    ax.plot(
                        msub["eval_epoch"].tolist(),
                        msub["value"].tolist(),
                        marker="o",
                        linewidth=2,
                        color=color_map[model],
                        label=model,
                    )
                if epoch_ticks:
                    ax.set_xticks(epoch_ticks)

        legend_handles = [Patch(facecolor=color_map[m], edgecolor="none", label=m) for m in model_params]
        fig.legend(
            handles=legend_handles,
            title="Model parameter (line color)",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(4, len(model_params)),
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 0.96])
        out_path = plots_dir / f"ALL_PARTICIPANTS_2x2_epoch_trend_{variant_tag}_level_{lvl}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)
        print(f"Wrote plot: {out_path}")


def make_all_participants_stacked_epoch_trend_by_level(
    pear_df: pd.DataFrame,
    output_dir: Path,
    left_name: str,
    right_name: str,
    eval_variant: str = "full_finetune",
    show_plots: bool = False,
) -> None:
    """Plot 5 stacked variability-level subplots with one line per source/seq combo.

    For each source/seq/epoch/level, the plotted value is the best mean Pearsonr
    across model families, which compresses the older 2x2 panel view into a
    single figure without losing the epoch trend.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    variant_title = VARIANT_TITLES.get(eval_variant, eval_variant)
    variant_tag = variant_file_tag(eval_variant)
    left_display = source_display_name(left_name)
    right_display = source_display_name(right_name)

    if pear_df.empty:
        print(f"No data available for all-participants {variant_tag} stacked epoch trend plots.")
        return

    all_df = pear_df[
        (pear_df["eval_variant"] == eval_variant)
        & (pear_df["eval_epoch"].notna())
        & (pear_df["variability_level"].notna())
    ].copy()
    if all_df.empty:
        print(f"No {eval_variant} rows found for all-participants stacked epoch trend plots.")
        return

    all_df["eval_epoch"] = all_df["eval_epoch"].astype(int)
    all_df["variability_level"] = all_df["variability_level"].astype(int)

    mean_by_model = (
        all_df.groupby(
            ["source", "seq_len", "model_param", "eval_epoch", "variability_level"], as_index=False
        ).agg(
            value=("value", "mean"),
            n_participants=("person_id", "nunique"),
            n_files=("file_name", "nunique"),
        )
    )
    best_by_combo = (
        mean_by_model.groupby(["source", "seq_len", "eval_epoch", "variability_level"], as_index=False)
        .agg(value=("value", "max"))
        .copy()
    )
    best_by_combo["combo_label"] = best_by_combo.apply(
        lambda row: f"{source_display_name(str(row['source']))} | seq_{int(row['seq_len'])}", axis=1
    )
    best_by_combo.to_csv(output_dir / f"all_participants_{variant_tag}_stacked_epoch_by_level_rows.csv", index=False)

    levels = sorted(best_by_combo["variability_level"].dropna().astype(int).unique().tolist())
    if not levels:
        print(f"No variability levels available for all-participants {variant_tag} stacked epoch trend plots.")
        return

    sources = [left_name, right_name]
    seqs = [600, 5400]
    combos = [(source, seq) for source in sources for seq in seqs]
    combo_styles = {
        (left_name, 600): {"color": "#1f77b4", "linestyle": "-"},
        (left_name, 5400): {"color": "#1f77b4", "linestyle": "--"},
        (right_name, 600): {"color": "#d62728", "linestyle": "-"},
        (right_name, 5400): {"color": "#d62728", "linestyle": "--"},
    }

    fig, axes = plt.subplots(len(levels), 1, figsize=(12, max(12, len(levels) * 2.6)), sharex=True, sharey=False)
    if len(levels) == 1:
        axes = [axes]
    fig.suptitle(
        f"All participants: epoch trend across levels ({variant_title})",
        y=0.995,
    )

    for ax, lvl in zip(axes, levels):
        lvl_sub = best_by_combo[best_by_combo["variability_level"] == lvl].copy()
        ax.set_title(f"Level {lvl}")
        ax.set_ylabel("Mean Best Pearsonr")

        if lvl_sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
            continue

        level_min = float(lvl_sub["value"].min())
        level_max = float(lvl_sub["value"].max())
        if level_min == level_max:
            level_pad = 0.05
        else:
            level_pad = max(0.03, (level_max - level_min) * 0.08)
        ax.set_ylim(level_min - level_pad, level_max + level_pad)

        epoch_ticks: List[int] = []
        for source, seq in combos:
            combo_sub = lvl_sub[(lvl_sub["source"] == source) & (lvl_sub["seq_len"] == seq)].sort_values("eval_epoch")
            if combo_sub.empty:
                continue
            epoch_vals = combo_sub["eval_epoch"].astype(int).tolist()
            epoch_ticks.extend(epoch_vals)
            style = combo_styles[(source, seq)]
            ax.plot(
                epoch_vals,
                combo_sub["value"].tolist(),
                marker="o",
                linewidth=2,
                color=style["color"],
                linestyle=style["linestyle"],
                label=f"{source_display_name(source)} | seq_{seq}",
            )

        if epoch_ticks:
            ax.set_xticks(sorted(set(epoch_ticks)))

    axes[-1].set_xlabel("Epoch")
    legend_handles = []
    for source, seq in combos:
        style = combo_styles[(source, seq)]
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker="o",
                linewidth=2,
                label=f"{left_display if source == left_name else right_display} | seq_{seq}",
            )
        )

    fig.legend(
        handles=legend_handles,
        title="Source | Sequence Length",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.975])
    out_path = plots_dir / f"ALL_PARTICIPANTS_stacked_epoch_trend_{variant_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def _strategy_candidates(pear_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows used for strategy comparison and assign strategy label."""
    if pear_df.empty:
        return pear_df.copy()

    df = pear_df.copy()
    # Strategy mapping:
    # - full_finetune: all full-finetune runs (pretrained/no_pretrain/epoch variants)
    # - lora
    # - lora_over_lora
    df["strategy"] = np.select(
        [
            df["eval_variant"] == "full_finetune",
            df["eval_variant"] == "lora_over_lora",
            df["eval_variant"] == "lora",
        ],
        [
            "full_finetune",
            "lora_over_lora",
            "lora",
        ],
        default=None,
    )
    keep = df[df["strategy"].notna() & df["variability_level"].notna()].copy()
    return keep


def make_all_participants_strategy_best_2x2_plot(
    pear_df: pd.DataFrame, output_dir: Path, left_name: str, right_name: str, show_plots: bool = False
) -> None:
    """2x2 grouped bars comparing best strategy score per variability level across all participants."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cand = _strategy_candidates(pear_df)
    if cand.empty:
        print("No data available for all-participants strategy comparison plot.")
        return

    # Best per participant first, then average participants for fair weighting.
    per_person_best = (
        cand.groupby(
            ["person_id", "source", "seq_len", "variability_level", "strategy"], as_index=False
        )["value"]
        .max()
        .rename(columns={"value": "best_value"})
    )
    avg_best = (
        per_person_best.groupby(["source", "seq_len", "variability_level", "strategy"], as_index=False)
        .agg(value=("best_value", "mean"), n_participants=("person_id", "nunique"))
        .copy()
    )
    avg_best.to_csv(output_dir / "all_participants_best_strategy_by_variability_rows.csv", index=False)

    sources = [left_name, right_name]
    seqs = [600, 5400]
    strategies = ["full_finetune", "lora", "lora_over_lora"]
    color_map = {
        "full_finetune": "#ff7f0e",
        "lora": "#1f77b4",
        "lora_over_lora": "#2ca02c",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    fig.suptitle("All participants: best strategy comparison by variability level", y=0.98)

    for r, source in enumerate(sources):
        for c, seq in enumerate(seqs):
            ax = axes[r, c]
            sub = avg_best[(avg_best["source"] == source) & (avg_best["seq_len"] == seq)].copy()
            ax.set_title(f"{source_display_name(source)} | seq_{seq}")
            ax.set_xlabel("Variability level")
            if c == 0:
                ax.set_ylabel("Mean Best Pearsonr")
            ax.set_ylim(-0.1, 1.0)

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                continue

            levels = sorted(sub["variability_level"].dropna().astype(int).unique().tolist())
            x = np.arange(len(levels), dtype=float)
            n_strat = len(strategies)
            width = min(0.8 / max(n_strat, 1), 0.25)
            offsets = (np.arange(n_strat) - (n_strat - 1) / 2.0) * width

            for i_s, strat in enumerate(strategies):
                ssub = sub[sub["strategy"] == strat].copy()
                lvl_to_val = {int(v): float(y) for v, y in zip(ssub["variability_level"], ssub["value"])}
                heights = [lvl_to_val.get(lvl, np.nan) for lvl in levels]
                bars = ax.bar(
                    x + offsets[i_s],
                    heights,
                    width=width * 0.95,
                    color=color_map[strat],
                    alpha=0.9,
                    label=strat,
                )
                for b in bars:
                    h = b.get_height()
                    if np.isnan(h):
                        continue
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        h + 0.015,
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in levels])

    legend_handles = [Patch(facecolor=color_map[s], edgecolor="none", label=s) for s in strategies]
    fig.legend(
        handles=legend_handles,
        title="Strategy",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    out_path = plots_dir / "ALL_PARTICIPANTS_2x2_best_strategy_compare_by_variability.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def make_person_strategy_best_2x2_plot(
    pear_df: pd.DataFrame,
    person_id: str,
    output_dir: Path,
    left_name: str,
    right_name: str,
    show_plots: bool = False,
) -> None:
    """2x2 grouped bars comparing best strategy score per variability level for one participant."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cand = _strategy_candidates(pear_df)
    cand = cand[cand["person_id"] == person_id].copy()
    if cand.empty:
        print(f"No data available for {person_id} strategy comparison plot.")
        return

    best_df = (
        cand.groupby(["source", "seq_len", "variability_level", "strategy"], as_index=False)["value"]
        .max()
        .rename(columns={"value": "best_value"})
    )
    best_df.to_csv(output_dir / f"{person_id}_best_strategy_by_variability_rows.csv", index=False)

    sources = [left_name, right_name]
    seqs = [600, 5400]
    strategies = ["full_finetune", "lora", "lora_over_lora"]
    color_map = {
        "full_finetune": "#ff7f0e",
        "lora": "#1f77b4",
        "lora_over_lora": "#2ca02c",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    fig.suptitle(f"{person_id}: best strategy comparison by variability level", y=0.98)

    for r, source in enumerate(sources):
        for c, seq in enumerate(seqs):
            ax = axes[r, c]
            sub = best_df[(best_df["source"] == source) & (best_df["seq_len"] == seq)].copy()
            ax.set_title(f"{source_display_name(source)} | seq_{seq}")
            ax.set_xlabel("Variability level")
            if c == 0:
                ax.set_ylabel("Best Pearsonr")
            ax.set_ylim(-0.1, 1.0)

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                continue

            levels = sorted(sub["variability_level"].dropna().astype(int).unique().tolist())
            x = np.arange(len(levels), dtype=float)
            n_strat = len(strategies)
            width = min(0.8 / max(n_strat, 1), 0.25)
            offsets = (np.arange(n_strat) - (n_strat - 1) / 2.0) * width

            for i_s, strat in enumerate(strategies):
                ssub = sub[sub["strategy"] == strat].copy()
                lvl_to_val = {int(v): float(y) for v, y in zip(ssub["variability_level"], ssub["best_value"])}
                heights = [lvl_to_val.get(lvl, np.nan) for lvl in levels]
                bars = ax.bar(
                    x + offsets[i_s],
                    heights,
                    width=width * 0.95,
                    color=color_map[strat],
                    alpha=0.9,
                    label=strat,
                )
                for b in bars:
                    h = b.get_height()
                    if np.isnan(h):
                        continue
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        h + 0.015,
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in levels])

    legend_handles = [Patch(facecolor=color_map[s], edgecolor="none", label=s) for s in strategies]
    fig.legend(
        handles=legend_handles,
        title="Strategy",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.96])
    out_path = plots_dir / f"{person_id}_2x2_best_strategy_compare_by_variability.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def main() -> None:
    args = parse_args()
    if args.single_project_dir is not None:
        source_dir: Path = args.single_project_dir
        output_dir: Path = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        index, unparsed = build_index(source_dir)
        print("=== Single Project Results ===")
        print(f"source dir: {source_dir}")
        print(f"parsed files: {sum(len(v) for v in index.values())}")
        print(f"unparsed files: {len(unparsed)}")
        pd.DataFrame({"unparsed_file": [p.name for p in unparsed]}).to_csv(
            output_dir / "single_project_unparsed_files.csv", index=False
        )

        source_name = source_dir.name
        all_pear = collect_person_pearson(index, source_name)
        if all_pear.empty:
            print("No Pearson rows were collected for the single-project run.")
            return

        selected_rows, chosen_epochs = select_best_epoch_per_model(all_pear)
        final_selected_rows, final_chosen_epochs = select_final_epoch_per_model(all_pear)
        run_sanity_checks(all_pear, selected_rows, chosen_epochs)
        selected_rows.to_csv(output_dir / "single_project_best_pearson_selected_rows.csv", index=False)
        chosen_epochs.to_csv(output_dir / "single_project_best_epoch_per_model.csv", index=False)
        final_selected_rows.to_csv(output_dir / "single_project_final_epoch_selected_rows.csv", index=False)
        final_chosen_epochs.to_csv(output_dir / "single_project_final_epoch_per_model.csv", index=False)

        make_single_project_all_participants_mean_plots(
            selected_rows, output_dir, source_name, show_plots=args.show_plots
        )
        make_single_project_all_participants_mean_plots(
            final_selected_rows,
            output_dir,
            source_name,
            show_plots=args.show_plots,
            selection_label="final",
        )
        make_single_project_person_mean_plots(
            final_selected_rows,
            output_dir,
            source_name,
            show_plots=args.show_plots,
            selection_label="final",
        )
        for person_id in sorted(final_selected_rows["person_id"].dropna().unique().tolist()):
            person_dir = output_dir / "per_person" / person_id
            person_dir.mkdir(parents=True, exist_ok=True)
            final_selected_rows[final_selected_rows["person_id"] == person_id].to_csv(
                person_dir / "person_final_epoch_selected_rows.csv",
                index=False,
            )
            final_chosen_epochs[final_chosen_epochs["person_id"] == person_id].to_csv(
                person_dir / "person_final_epoch_per_model.csv",
                index=False,
            )
        make_single_project_all_participants_strategy_best_plot(
            all_pear, output_dir, source_name, show_plots=args.show_plots
        )
        for eval_variant in ["full_finetune", "lora", "lora_over_lora"]:
            make_single_project_all_participants_epoch_trend_plots(
                all_pear,
                output_dir,
                source_name,
                eval_variant=eval_variant,
                show_plots=args.show_plots,
            )

        print(f"\nWrote single-project outputs to: {output_dir}")
        return

    left_dir: Path = args.left_dir
    right_dir: Path = args.right_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    left_index, left_unparsed = build_index(left_dir)
    right_index, right_unparsed = build_index(right_dir)

    left_keys = set(left_index)
    right_keys = set(right_index)
    common_keys = sorted(left_keys & right_keys)
    left_only = sorted(left_keys - right_keys)
    right_only = sorted(right_keys - left_keys)

    print("=== Compare Results ===")
    print(f"left dir:  {left_dir}")
    print(f"right dir: {right_dir}")
    print(f"left parsed files:  {sum(len(v) for v in left_index.values())}")
    print(f"right parsed files: {sum(len(v) for v in right_index.values())}")
    print(f"unparsed left files:  {len(left_unparsed)}")
    print(f"unparsed right files: {len(right_unparsed)}")
    print(f"common keys: {len(common_keys)}")
    print(f"left-only keys: {len(left_only)}")
    print(f"right-only keys: {len(right_only)}")

    # Save mapping artifacts.
    pd.DataFrame({"left_unparsed_file": [p.name for p in left_unparsed]}).to_csv(
        output_dir / "left_unparsed_files.csv", index=False
    )
    pd.DataFrame({"right_unparsed_file": [p.name for p in right_unparsed]}).to_csv(
        output_dir / "right_unparsed_files.csv", index=False
    )
    pd.DataFrame({"left_only_key": left_only}).to_csv(output_dir / "left_only_keys.csv", index=False)
    pd.DataFrame({"right_only_key": right_only}).to_csv(output_dir / "right_only_keys.csv", index=False)

    pair_rows = []
    point_rows = []
    summary_rows = []

    for key in common_keys:
        # If duplicates exist in one side, compare all combinations.
        for lmeta in left_index[key]:
            for rmeta in right_index[key]:
                pair_rows.append(
                    {
                        "pair_key": key,
                        "left_file": lmeta.file_name,
                        "right_file": rmeta.file_name,
                    }
                )
                points, summary = summarize_pair(lmeta, rmeta)
                point_rows.append(points)
                summary_rows.append(summary)

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(output_dir / "file_pairs.csv", index=False)

    if point_rows:
        all_points = pd.concat(point_rows, ignore_index=True)
        all_points.to_csv(output_dir / "point_level_diffs.csv", index=False)
    else:
        all_points = pd.DataFrame()

    if summary_rows:
        all_summary = pd.concat(summary_rows, ignore_index=True)
        all_summary.to_csv(output_dir / "metric_summary.csv", index=False)
        print("\n=== Metric Summary (mean over pairs) ===")
        overall = (
            all_summary[all_summary["metric"] != "<none>"]
            .groupby("metric", as_index=False)
            .agg(
                total_points=("n_points", "sum"),
                avg_mean_delta=("mean_delta", "mean"),
                avg_mae_delta=("mae_delta", "mean"),
                avg_rmse_delta=("rmse_delta", "mean"),
            )
        )
        if overall.empty:
            print("No overlapping points across paired files.")
        else:
            print(overall.to_string(index=False))
            overall.to_csv(output_dir / "overall_metric_summary.csv", index=False)
            
    else:
        print("No comparable pairs were found.")

    # Per-person 2x2 bar plots: rows are directories, columns are seq_600/seq_5400.
    left_source_name = left_dir.name
    right_source_name = right_dir.name
    left_pear = collect_person_pearson(left_index, left_source_name)
    right_pear = collect_person_pearson(right_index, right_source_name)
    all_pear = pd.concat([left_pear, right_pear], ignore_index=True)
    final_selected_rows, final_chosen_epochs = select_final_epoch_per_model(all_pear)
    run_sanity_checks(all_pear, final_selected_rows, final_chosen_epochs, selection_label="final")
    final_selected_rows.to_csv(output_dir / "final_epoch_selected_rows.csv", index=False)
    final_chosen_epochs.to_csv(output_dir / "final_epoch_per_model.csv", index=False)

    if args.run_indvidual:
        make_person_2x2_plots(
            final_selected_rows,
            output_dir,
            left_source_name,
            right_source_name,
            show_plots=args.show_plots,
            selection_label="final",
        )
    make_all_participants_2x2_plots(
        final_selected_rows,
        output_dir,
        left_source_name,
        right_source_name,
        show_plots=args.show_plots,
        selection_label="final",
    )
    make_all_participants_strategy_training_phase_plots(
        all_pear, output_dir, left_source_name, right_source_name, show_plots=args.show_plots
    )
    make_all_participants_epoch1_phase_comparison_plot(
        all_pear, output_dir, left_source_name, right_source_name, show_plots=args.show_plots
    )
    make_all_participants_epoch1_phase_comparison_without_second_phase_only_plot(
        all_pear, output_dir, left_source_name, right_source_name, show_plots=args.show_plots
    )
    if args.run_indvidual:
        people = sorted(all_pear["person_id"].dropna().unique().tolist())
        for person_id in people:
            make_person_strategy_best_2x2_plot(
                all_pear,
                person_id,
                output_dir,
                left_source_name,
                right_source_name,
                show_plots=args.show_plots,
            )
        epoch_trend_variants = ["full_finetune", "lora", "lora_over_lora"]
        for person_id in people:
            for eval_variant in epoch_trend_variants:
                make_person_non_lora_epoch_trend_plots(
                    all_pear,
                    person_id,
                    output_dir,
                    left_source_name,
                    right_source_name,
                    eval_variant=eval_variant,
                    show_plots=args.show_plots,
                )
                make_person_non_lora_2x2_by_variability_plots(
                    all_pear,
                    person_id,
                    output_dir,
                    left_source_name,
                    right_source_name,
                    eval_variant=eval_variant,
                    show_plots=args.show_plots,
                )
    else:
        print("Skipping individual participant plots (set --run_indvidual to enable).")
    make_all_participants_strategy_best_2x2_plot(
        all_pear, output_dir, left_source_name, right_source_name, show_plots=args.show_plots
    )
    for eval_variant in ["full_finetune", "lora", "lora_over_lora"]:
        make_all_participants_non_lora_epoch_trend_plots(
            all_pear,
            output_dir,
            left_source_name,
            right_source_name,
            eval_variant=eval_variant,
            show_plots=args.show_plots,
        )
        make_all_participants_non_lora_2x2_by_variability_plots(
            all_pear,
            output_dir,
            left_source_name,
            right_source_name,
            eval_variant=eval_variant,
            show_plots=args.show_plots,
        )
        make_all_participants_stacked_epoch_trend_by_level(
            all_pear,
            output_dir,
            left_source_name,
            right_source_name,
            eval_variant=eval_variant,
            show_plots=args.show_plots,
        )

    print(f"\nWrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
