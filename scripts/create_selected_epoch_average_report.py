#!/usr/bin/env python3
# Create the all-sample and tissue-average PDF after EVAL-bin epoch selection:
# Bin-4 selection:
#   python3 scripts/create_selected_epoch_average_report.py \
#     --selection-dir scripts/results/downsampled_standard_ratio_0p25_lr_3e-05/eval_bin4_selected_epoch_all_tissues \
#     --output scripts/results/downsampled_standard_ratio_0p25_lr_3e-05/eval_bin4_selected_epoch_all_tissues/selected_eval_bin4_epoch_test_weighted_f1_average_and_tissue_report.pdf
# Bin-5 selection:
#   python3 scripts/create_selected_epoch_average_report.py \
#     --selection-dir scripts/results/downsampled_standard_ratio_0p25_lr_3e-05/eval_bin5_selected_epoch_all_tissues \
#     --output scripts/results/downsampled_standard_ratio_0p25_lr_3e-05/eval_bin5_selected_epoch_all_tissues/selected_eval_bin5_epoch_test_weighted_f1_average_and_tissue_report.pdf

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


SECTIONS = ["all_results", "bin_1", "bin_2", "bin_3", "bin_4", "bin_5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the selected-epoch TEST weighted-F1 average report."
    )
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_inputs(selection_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    selections = pd.read_csv(selection_dir / "selected_epochs.csv")
    comparisons = pd.read_csv(selection_dir / "selected_test_comparisons.csv")
    return selections, comparisons


def selected_bin(selections: pd.DataFrame) -> int:
    if "selection_bin" not in selections:
        return 4
    values = selections["selection_bin"].dropna().unique()
    if len(values) != 1:
        raise ValueError("selected_epochs.csv must contain exactly one selection bin")
    return int(values[0])


def selection_metric_column(selections: pd.DataFrame) -> str:
    generic = "eval_selection_weighted_f1"
    if generic in selections:
        return generic
    return f"eval_bin{selected_bin(selections)}_weighted_f1"


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for section, group in rows.groupby("section_rank", sort=False):
        record = {"section_rank": section, "participants": group["participant"].nunique()}
        for label, column in [
            ("selected", "selected_weighted_f1"),
            ("fixed_e3", "fixed_e3_weighted_f1"),
            ("pretrain", "pretrain_weighted_f1"),
            ("atlas", "atlas_weighted_f1"),
        ]:
            record[f"{label}_mean"] = group[column].mean()
            record[f"{label}_std"] = group[column].std(ddof=1)
            record[f"{label}_n"] = group[column].notna().sum()
        records.append(record)
    result = pd.DataFrame(records)
    order = {section: index for index, section in enumerate(SECTIONS)}
    return result.sort_values("section_rank", key=lambda values: values.map(order))


def section_title(section: str) -> str:
    if section == "all_results":
        return "All sites"
    return section.replace("_", " ").title()


def limits(row: pd.Series) -> tuple[float, float]:
    names = ["selected", "fixed_e3", "pretrain", "atlas"]
    lows = [row[f"{name}_mean"] - row[f"{name}_std"] for name in names]
    highs = [row[f"{name}_mean"] + row[f"{name}_std"] for name in names]
    low = max(0.0, min(lows) - 0.04)
    high = min(1.0, max(highs) + 0.04)
    if high - low < 0.12:
        center = (high + low) / 2
        low = max(0.0, center - 0.06)
        high = min(1.0, center + 0.06)
    return low, high


def draw_panel(axis: plt.Axes, row: pd.Series) -> None:
    selected = row["selected_mean"]
    selected_std = row["selected_std"]
    axis.bar(
        [0], [selected], yerr=[selected_std], width=0.46,
        color="#4C78A8", edgecolor="#2F4B68", capsize=5,
        label=f"EVAL-selected epoch (n={int(row['selected_n'])})",
    )
    axis.axhline(
        row["fixed_e3_mean"], color="#555555", linestyle="--", linewidth=1.8,
        label=f"Fixed E3 (n={int(row['fixed_e3_n'])})",
    )
    axis.axhline(
        row["pretrain_mean"], color="#D04A4A", linestyle=":", linewidth=2.0,
        label=f"Pretrain E10 (n={int(row['pretrain_n'])})",
    )
    axis.axhline(
        row["atlas_mean"], color="#2E8B57", linestyle="-.", linewidth=1.8,
        label=f"Atlas mean-label (n={int(row['atlas_n'])})",
    )
    axis.errorbar(
        0.30, row["fixed_e3_mean"], yerr=row["fixed_e3_std"],
        color="#555555", marker="o", markersize=3, capsize=4,
    )
    axis.errorbar(
        -0.42, row["pretrain_mean"], yerr=row["pretrain_std"],
        color="#D04A4A", marker="o", markersize=3, capsize=4,
    )
    axis.errorbar(
        0.46, row["atlas_mean"], yerr=row["atlas_std"],
        color="#2E8B57", marker="o", markersize=3, capsize=4,
    )
    axis.text(0, selected, f"{selected:.3f}", ha="center", va="bottom", fontsize=8)
    axis.set_title(f"{section_title(row['section_rank'])} (n={int(row['participants'])})", fontsize=10)
    axis.set_xticks([0], ["Selected"])
    axis.set_xlim(-0.65, 0.65)
    axis.set_ylim(*limits(row))
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylabel("Weighted F1")


def result_page(
    pdf: PdfPages,
    rows: pd.DataFrame,
    title: str,
    participants: int,
    total: int,
) -> None:
    summary = aggregate(rows)
    figure, axes = plt.subplots(3, 2, figsize=(11, 8.5), constrained_layout=True)
    status = "complete" if participants == total else f"incomplete: {participants}/{total} selected TEST predictions"
    figure.suptitle(f"{title}\n{status}", fontsize=14)
    for axis, section in zip(axes.flat, SECTIONS):
        section_rows = summary.loc[summary["section_rank"].eq(section)]
        if section_rows.empty:
            axis.axis("off")
            continue
        draw_panel(axis, section_rows.iloc[0])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    pdf.savefig(figure)
    plt.close(figure)


def cover_page(pdf: PdfPages, selections: pd.DataFrame) -> None:
    selection_bin = selected_bin(selections)
    available = int(selections["selected_test_available"].sum())
    total = len(selections)
    epoch_counts = selections["selected_epoch"].value_counts().sort_index()
    missing = selections.loc[~selections["selected_test_available"], "participant"].tolist()
    ignored = selections.loc[
        selections["ignored_incomplete_epochs"].fillna("").ne(""),
        ["participant", "ignored_incomplete_epochs"],
    ]
    ignored_text = "; ".join(
        f"{row.participant} ({row.ignored_incomplete_epochs})"
        for row in ignored.itertuples(index=False)
    )
    lines = [
        f"EVAL-bin-{selection_bin}-selected epoch: TEST weighted-F1 report",
        "All samples and tissue averages",
        "",
        "Selection rule",
        f"For each participant, consider only epochs with both EVAL and TEST predictions.",
        f"Among those complete pairs, choose the epoch with the highest EVAL bin-{selection_bin}",
        "weighted F1, then report that fixed epoch on TEST. TEST metric values never influence",
        "checkpoint selection. Pretrain E10 and atlas mean-label are retained as baselines.",
        "",
        f"Participants with selected TEST predictions: {available}/{total}",
        "Selected epochs: " + ", ".join(f"{epoch}={count}" for epoch, count in epoch_counts.items()),
        f"Missing selected TEST predictions: {', '.join(missing) if missing else 'none'}",
        f"Ignored incomplete epoch pairs: {ignored_text if ignored_text else 'none'}",
        "",
        "Plot conventions",
        "Blue bar: equal-participant mean for the selected retraining checkpoints.",
        "Gray dashed line: fixed-E3 equal-participant mean.",
        "Red dotted line: pretrain-E10 equal-participant mean.",
        "Green dash-dot line: atlas mean-label equal-participant mean.",
        "Whiskers show between-participant standard deviation.",
        "",
        "Every selected bar uses complete EVAL/TEST pairs; baseline legend labels show their n.",
    ]
    figure, axis = plt.subplots(figsize=(11, 8.5))
    axis.axis("off")
    axis.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11, linespacing=1.35)
    pdf.savefig(figure)
    plt.close(figure)


def selection_page(pdf: PdfPages, selections: pd.DataFrame) -> None:
    selection_bin = selected_bin(selections)
    metric_column = selection_metric_column(selections)
    columns = [
        "tissue", "participant", "selected_epoch", metric_column, "selected_test_available"
    ]
    table_rows = selections[columns].copy()
    table_rows[metric_column] = table_rows[metric_column].map(lambda value: f"{value:.6f}")
    table_rows["selected_test_available"] = table_rows["selected_test_available"].map({True: "yes", False: "no"})
    table_rows.columns = ["Tissue", "Participant", "Epoch", f"EVAL bin-{selection_bin} F1", "TEST available"]
    figure, axis = plt.subplots(figsize=(11, 8.5))
    axis.axis("off")
    axis.set_title("Participant-specific checkpoint selections", fontsize=14, pad=12)
    table = axis.table(cellText=table_rows.values, colLabels=table_rows.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1, 1.05)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#D9E2EC")
            cell.set_text_props(weight="bold")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def create_report(output: Path, selections: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        cover_page(pdf, selections)
        selection_page(pdf, selections)
        result_page(pdf, comparisons, "All tissues: equal-sample TEST average", comparisons["participant"].nunique(), len(selections))
        totals = selections.groupby("tissue")["participant"].nunique()
        for tissue in sorted(selections["tissue"].unique()):
            rows = comparisons.loc[comparisons["tissue"].eq(tissue)]
            result_page(pdf, rows, f"{tissue}: equal-sample TEST average", rows["participant"].nunique(), int(totals[tissue]))


def main() -> None:
    args = parse_args()
    selections, comparisons = load_inputs(args.selection_dir)
    create_report(args.output, selections, comparisons)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
