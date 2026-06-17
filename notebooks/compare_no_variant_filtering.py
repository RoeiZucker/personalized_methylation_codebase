#!/usr/bin/env python3
"""Compare _no_variant_filtering vs _different_epoch_length_dataset_override_test.

This script matches compatible CSVs by:
- person_id
- seq_len
- eval_variant
- strategy (model family: epoch 1 / epoch 2 / no_pretrain)

Then it compares Pearsonr values by matched eval_epoch + variability_level and
generates summary tables and variability-level plots.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FILENAME_RE = re.compile(
    r"^(?P<prefix>.+?)_eval(?P<variant>_lora_over_lora|_lora)?(?P<project>.+?)"
    r"_lr_(?P<lr>[^_]+)_bs_(?P<bs>[^_]+)_seq_(?P<seq>\d+)_testsize_(?P<testsize>[^_]+)_result\.csv$"
)
METRIC_COL_RE = re.compile(r"^(?P<bin>.+)_(?P<metric>pearsonr|mse|mae)$")


@dataclass(frozen=True)
class FileMeta:
    path: Path
    file_name: str
    person_id: str
    seq_len: int
    eval_variant: str
    model_param: str


def canonical_model_param(prefix: str) -> str:
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
    return FileMeta(
        path=path,
        file_name=path.name,
        person_id=d["prefix"].split("_", 1)[0],
        seq_len=int(d["seq"]),
        eval_variant=eval_variant,
        model_param=canonical_model_param(d["prefix"]),
    )


def load_metas(directory: Path) -> List[FileMeta]:
    metas: List[FileMeta] = []
    for path in sorted(directory.glob("*.csv")):
        meta = parse_filename(path)
        if meta is not None:
            metas.append(meta)
    return metas


def normalize_result_csv(path: Path) -> pd.DataFrame:
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
    if train_mode == "pretrained":
        row_df.loc[row_df["row_in_file"] == 0, "eval_epoch"] = 0

    long_df = row_df.melt(
        id_vars=["row_in_file", "eval_epoch"],
        value_vars=value_cols,
        var_name="metric_col",
        value_name="value",
    )
    parsed = long_df["metric_col"].str.extract(METRIC_COL_RE)
    long_df["metric"] = parsed["metric"]
    long_df["bin_range"] = parsed["bin"]

    bounds = long_df["bin_range"].str.extract(r"^(?P<bin_start>[0-9.]+)-(?P<bin_end>[0-9.]+)$")
    long_df["bin_start"] = pd.to_numeric(bounds["bin_start"], errors="coerce")
    long_df["variability_level"] = (
        long_df.groupby("row_in_file")["bin_start"].rank(method="dense", ascending=True).astype("Int64")
    )
    return long_df[["eval_epoch", "metric", "variability_level", "value"]].copy()


def choose_compatible_pairs(left_metas: List[FileMeta], right_metas: List[FileMeta]) -> pd.DataFrame:
    def build_map(metas: List[FileMeta]) -> Dict[Tuple[str, int, str, str], List[FileMeta]]:
        out: Dict[Tuple[str, int, str, str], List[FileMeta]] = {}
        for m in metas:
            key = (m.person_id, m.seq_len, m.eval_variant, m.model_param)
            out.setdefault(key, []).append(m)
        return out

    left_map = build_map(left_metas)
    right_map = build_map(right_metas)
    common_keys = sorted(set(left_map) & set(right_map))

    rows = []
    for key in common_keys:
        l_items = sorted(left_map[key], key=lambda x: x.file_name)
        r_items = sorted(right_map[key], key=lambda x: x.file_name)
        rows.append(
            {
                "person_id": key[0],
                "seq_len": key[1],
                "eval_variant": key[2],
                "model_param": key[3],
                "left_file": l_items[0].file_name,
                "right_file": r_items[0].file_name,
                "left_path": str(l_items[0].path),
                "right_path": str(r_items[0].path),
                "left_candidates": len(l_items),
                "right_candidates": len(r_items),
            }
        )
    return pd.DataFrame(rows)


def compute_pairwise_pearson(pair_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: List[dict] = []
    level_rows: List[pd.DataFrame] = []

    for _, row in pair_df.iterrows():
        ldf = normalize_result_csv(Path(row["left_path"]))
        rdf = normalize_result_csv(Path(row["right_path"]))
        lpear = ldf[ldf["metric"] == "pearsonr"].copy()
        rpear = rdf[rdf["metric"] == "pearsonr"].copy()
        merged = lpear.merge(rpear, on=["eval_epoch", "variability_level"], how="inner", suffixes=("_left", "_right"))
        if merged.empty:
            continue

        merged["delta_left_minus_right"] = merged["value_left"] - merged["value_right"]

        pair_rows.append(
            {
                "person_id": row["person_id"],
                "seq_len": row["seq_len"],
                "eval_variant": row["eval_variant"],
                "model_param": row["model_param"],
                "left_file": row["left_file"],
                "right_file": row["right_file"],
                "n_points": len(merged),
                "left_mean_pearson": merged["value_left"].mean(),
                "right_mean_pearson": merged["value_right"].mean(),
                "mean_delta_left_minus_right": merged["delta_left_minus_right"].mean(),
                "max_abs_delta": merged["delta_left_minus_right"].abs().max(),
            }
        )

        by_level = (
            merged.groupby("variability_level", as_index=False)
            .agg(
                left_mean=("value_left", "mean"),
                right_mean=("value_right", "mean"),
                mean_delta=("delta_left_minus_right", "mean"),
                n_points=("delta_left_minus_right", "size"),
            )
            .sort_values("variability_level")
        )
        by_level.insert(0, "model_param", row["model_param"])
        by_level.insert(0, "eval_variant", row["eval_variant"])
        by_level.insert(0, "seq_len", row["seq_len"])
        by_level.insert(0, "person_id", row["person_id"])
        level_rows.append(by_level)

    pair_summary = pd.DataFrame(pair_rows).sort_values(["person_id", "model_param"])
    if level_rows:
        level_summary = pd.concat(level_rows, ignore_index=True)
    else:
        level_summary = pd.DataFrame(
            columns=[
                "person_id",
                "seq_len",
                "eval_variant",
                "model_param",
                "variability_level",
                "left_mean",
                "right_mean",
                "mean_delta",
                "n_points",
            ]
        )
    return pair_summary, level_summary


def make_strategy_2x2_plots(
    level_summary: pd.DataFrame,
    output_dir: Path,
    left_label: str,
    right_label: str,
    show_plots: bool = False,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if level_summary.empty:
        return
    left_display = left_label.lstrip("_")
    right_display = right_label.lstrip("_")

    persons = sorted(level_summary["person_id"].dropna().unique().tolist())
    strategies = sorted(level_summary["model_param"].dropna().unique().tolist())

    for strategy in strategies:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        fig.suptitle(f"{strategy}: Pearsonr by variability level ({left_label} vs {right_label})", y=0.98)
        ax_list = axes.flatten()

        for i, person in enumerate(persons[:4]):
            ax = ax_list[i]
            sub = level_summary[(level_summary["person_id"] == person) & (level_summary["model_param"] == strategy)]
            ax.set_title(person)
            ax.set_xlabel("Variability level")
            if i % 2 == 0:
                ax.set_ylabel("Mean Pearsonr")
            ax.set_ylim(-0.1, 1.0)

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                continue

            x = sub["variability_level"].astype(int).tolist()
            ax.plot(x, sub["left_mean"], marker="o", linewidth=2, color="tab:blue", label=left_display)
            ax.plot(x, sub["right_mean"], marker="o", linewidth=2, color="tab:orange", label=right_display)
            ax.set_xticks(x)

        for j in range(len(persons), 4):
            ax_list[j].axis("off")

        legend_handles = [
            Line2D([0], [0], color="tab:blue", marker="o", linewidth=2, label=left_display),
            Line2D([0], [0], color="tab:orange", marker="o", linewidth=2, label=right_display),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
        out_path = plots_dir / f"strategy_{strategy.replace(' ', '_')}_2x2_persons.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)


def make_person_2x2_plots(
    level_summary: pd.DataFrame,
    output_dir: Path,
    left_label: str,
    right_label: str,
    show_plots: bool = False,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if level_summary.empty:
        return
    left_display = left_label.lstrip("_")
    right_display = right_label.lstrip("_")

    persons = sorted(level_summary["person_id"].dropna().unique().tolist())
    strategies = sorted(level_summary["model_param"].dropna().unique().tolist())

    for person in persons:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        fig.suptitle(f"{person}: Pearsonr by variability level ({left_label} vs {right_label})", y=0.98)
        ax_list = axes.flatten()

        for i, strategy in enumerate(strategies[:4]):
            ax = ax_list[i]
            sub = level_summary[(level_summary["person_id"] == person) & (level_summary["model_param"] == strategy)]
            ax.set_title(strategy)
            ax.set_xlabel("Variability level")
            if i % 2 == 0:
                ax.set_ylabel("Mean Pearsonr")
            ax.set_ylim(-0.1, 1.0)

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, alpha=0.7)
                continue

            x = sub["variability_level"].astype(int).tolist()
            ax.plot(x, sub["left_mean"], marker="o", linewidth=2, color="tab:blue", label=left_display)
            ax.plot(x, sub["right_mean"], marker="o", linewidth=2, color="tab:orange", label=right_display)
            ax.set_xticks(x)

        for j in range(len(strategies), 4):
            ax_list[j].axis("off")

        legend_handles = [
            Line2D([0], [0], color="tab:blue", marker="o", linewidth=2, label=left_display),
            Line2D([0], [0], color="tab:orange", marker="o", linewidth=2, label=right_display),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
        out_path = plots_dir / f"person_{person}_2x2_strategies.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare compatible CSVs for two result directories.")
    parser.add_argument(
        "--left-dir",
        type=Path,
        default=Path("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/results/_no_variant_filtering"),
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
        default=Path(__file__).resolve().parent / "compare_no_variant_filtering_output",
    )
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    left_metas = load_metas(args.left_dir)
    right_metas = load_metas(args.right_dir)

    pair_df = choose_compatible_pairs(left_metas, right_metas)
    pair_df.to_csv(output_dir / "compatible_pairs.csv", index=False)

    if pair_df.empty:
        print("No compatible pairs found.")
        print(f"left: {args.left_dir}")
        print(f"right: {args.right_dir}")
        print(f"Wrote: {output_dir / 'compatible_pairs.csv'}")
        return

    pair_summary, level_summary = compute_pairwise_pearson(pair_df)
    pair_summary.to_csv(output_dir / "pair_summary_pearson.csv", index=False)
    level_summary.to_csv(output_dir / "variability_level_summary_pearson.csv", index=False)

    left_label = args.left_dir.name
    right_label = args.right_dir.name
    make_strategy_2x2_plots(level_summary, output_dir, left_label, right_label, args.show_plots)
    make_person_2x2_plots(level_summary, output_dir, left_label, right_label, args.show_plots)

    print("Compatible pairs:", len(pair_df))
    print("Pair summary rows:", len(pair_summary))
    print("Variability-level rows:", len(level_summary))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
