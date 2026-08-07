#!/usr/bin/env python3
# Examples:
#   python3 scripts/plot_single_prediction_evaluations.py \
#     scripts/results/downsampled_ratio_0p25_lr_3e-05_Z00000431_eval_variability_bins/eval_eval_objects_dict.txt \
#     --plot-title "Z00000431 ratio=0.25 lr=3e-5 eval" \
#     --output-dir scripts/results/downsampled_ratio_0p25_lr_3e-05_Z00000431_eval_variability_bins/plots
#
#   python3 scripts/plot_single_prediction_evaluations.py \
#     scripts/results/downsampled_ratio_0p25_lr_3e-05_Z00000431_eval_variability_bins/eval_eval_objects_dict.txt \
#     scripts/results/downsampled_ratio_0p25_lr_3e-05_Z0000044H_eval_test_variability_bins/eval_eval_objects_dict.txt \
#     --labels "Z00000431,Z0000044H" \
#     --plot-title "Downsampled ratio=0.25 lr=3e-5 eval comparison" \
#     --output-dir scripts/results/downsampled_ratio_0p25_lr_3e-05_eval_comparison_plots
#
#   python3 scripts/plot_single_prediction_evaluations.py \
#     scripts/results/downsampled_ratio_0p25_lr_3e-05_Z00000431_eval_variability_bins/eval_eval_objects_dict.txt \
#     --labels "checkpoint-14870,checkpoint-29740,checkpoint-44610" \
#     --plot-title "Z00000431 ratio=0.25 lr=3e-5 eval" \
#     --output-dir scripts/results/downsampled_ratio_0p25_lr_3e-05_Z00000431_eval_variability_bins/plots

from argparse import ArgumentParser
import ast
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = [0, 1, 2]
METRICS = ["f1", "precision", "recall"]


def parse_args():
    parser = ArgumentParser(
        description="Plot weighted metrics from one or more eval_objects_dict files."
    )
    parser.add_argument("evaluation_files", nargs="+", type=Path)
    parser.add_argument("--plot-title", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Comma-separated labels. Can be one label per input eval file, "
            "or one label per prediction/checkpoint inside the eval files."
        ),
    )
    return parser.parse_args()


def read_eval_file(path):
    return ast.literal_eval(path.read_text(encoding="utf-8"))


def input_labels(args):
    if args.labels is None:
        return None
    return [label.strip() for label in args.labels.split(",") if label.strip()]


def section_sort_key(section):
    if section == "all_results":
        return (-1, 0.0)
    return (0, float(section.split("-", 1)[0]))


def section_display_name(section, bin_index):
    if section == "all_results":
        return "All sites"
    low, high = section.split("-", 1)
    return f"Bin {bin_index} std {float(low):.4f}-{float(high):.4f}"


def prediction_label(prediction_path, fallback_label):
    if fallback_label:
        return fallback_label
    parent_name = Path(prediction_path).parent.name
    epoch_match = re.search(r"(?:epoch|checkpoint)-(?P<epoch>\d+)(?:-step-(?P<step>\d+))?", parent_name)
    if epoch_match is not None:
        label = f"E{epoch_match.group('epoch')}"
        if parent_name.startswith("checkpoint-"):
            label = f"checkpoint {epoch_match.group('epoch')}"
        return label
    return parent_name


def confusion_to_array(confusion_matrix):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=float)
    for predicted_label, true_counts in confusion_matrix.items():
        predicted_label = int(predicted_label)
        if predicted_label not in LABELS:
            continue
        for true_label, count in true_counts.items():
            true_label = int(true_label)
            if true_label in LABELS:
                matrix[true_label, predicted_label] = float(count)
    return matrix


def weighted_metrics(section_object, prediction_type):
    matrix = confusion_to_array(section_object[f"{prediction_type}_confusion_matrix"])
    support = matrix.sum(axis=1)
    total_support = support.sum()
    metric_values = {metric: [] for metric in METRICS}
    for label in LABELS:
        precision = float(section_object[prediction_type][label]["Precision"])
        recall = float(section_object[prediction_type][label]["recall"])
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        metric_values["precision"].append(precision)
        metric_values["recall"].append(recall)
        metric_values["f1"].append(f1)
    return {
        metric: float(np.average(values, weights=support)) if total_support else np.nan
        for metric, values in metric_values.items()
    } | {"support": int(total_support)}


def collect_prediction_records(evaluation_files):
    records = []
    for file_index, evaluation_file in enumerate(evaluation_files):
        payload = read_eval_file(evaluation_file)
        for sample_name, sample_results in payload.items():
            for prediction_path, prediction_eval in sample_results.items():
                records.append(
                    {
                        "file_index": file_index,
                        "evaluation_file": evaluation_file,
                        "sample_name": sample_name,
                        "prediction_path": prediction_path,
                        "prediction_eval": prediction_eval,
                    }
                )
    return records


def choose_record_label(record, labels, evaluation_file_count, prediction_index):
    if labels is None:
        return prediction_label(record["prediction_path"], None)
    if len(labels) == evaluation_file_count:
        return prediction_label(record["prediction_path"], labels[record["file_index"]])
    if len(labels) == prediction_index["total"]:
        return prediction_label(record["prediction_path"], labels[prediction_index["current"]])
    raise ValueError(
        "--labels must contain either one label per evaluation file "
        "or one label per prediction/checkpoint"
    )


def collect_rows(evaluation_files, labels):
    records = collect_prediction_records(evaluation_files)
    rows = []
    prediction_index = {"current": 0, "total": len(records)}
    for record in records:
        label = choose_record_label(record, labels, len(evaluation_files), prediction_index)
        prediction_index["current"] += 1
        for section, section_object in record["prediction_eval"].items():
            for prediction_type in ["predicted_class", "mean_label"]:
                if f"{prediction_type}_confusion_matrix" not in section_object:
                    continue
                values = weighted_metrics(section_object, prediction_type)
                for metric in METRICS:
                    rows.append(
                        {
                            "input_order": prediction_index["current"] - 1,
                            "evaluation_file": str(record["evaluation_file"]),
                            "sample": record["sample_name"],
                            "prediction_path": record["prediction_path"],
                            "label": label,
                            "section": section,
                            "prediction_type": prediction_type,
                            "metric": metric,
                            "value": values[metric],
                            "support": values["support"],
                        }
                    )
    return pd.DataFrame(rows)


def add_bar_labels(axis, bars):
    for bar in bars:
        height = bar.get_height()
        if np.isfinite(height):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.006,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )


def add_mean_line(axis, section_rows, metric):
    means = section_rows[
        (section_rows["prediction_type"] == "mean_label")
        & (section_rows["metric"] == metric)
    ]["value"].drop_duplicates()
    if means.empty:
        return
    mean_value = float(means.iloc[0])
    axis.axhline(mean_value, color="black", linestyle="--", linewidth=2.5, alpha=0.9)
    axis.annotate(
        f"mean {mean_value:.3f}",
        xy=(0.99, mean_value),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color="black",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "black", "alpha": 0.85},
    )


def plot_metric(rows, metric, output_dir, plot_title):
    sections = sorted(rows["section"].unique(), key=section_sort_key)
    labels = rows[rows["prediction_type"] == "predicted_class"]["label"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(sections),
        1,
        figsize=(max(8, 1.8 * len(labels)), 3.0 * len(sections)),
        sharex=True,
    )
    if len(sections) == 1:
        axes = [axes]

    bin_index = 0
    for axis, section in zip(axes, sections):
        if section != "all_results":
            bin_index += 1
        section_rows = rows[rows["section"] == section]
        plot_rows = section_rows[
            (section_rows["prediction_type"] == "predicted_class")
            & (section_rows["metric"] == metric)
        ].copy()
        plot_rows = plot_rows.sort_values(["input_order", "label"])
        values = plot_rows["value"].tolist()
        x_positions = np.arange(len(values))
        bars = axis.bar(x_positions, values, color="#4C78A8", width=0.55)
        add_bar_labels(axis, bars)
        add_mean_line(axis, section_rows, metric)
        comparison_values = list(plot_rows["value"].dropna())
        comparison_values.extend(
            section_rows[
                (section_rows["prediction_type"] == "mean_label")
                & (section_rows["metric"] == metric)
            ]["value"].dropna()
        )
        if comparison_values:
            axis.set_ylim(
                max(0, min(comparison_values) - 0.08),
                min(1.05, max(comparison_values) + 0.10),
            )
        support = plot_rows["support"].iloc[0] if not plot_rows.empty else 0
        axis.set_title(
            f"{section_display_name(section, bin_index)} (n={int(support):,})",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel(metric.upper())
        axis.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(np.arange(len(labels)))
    axes[-1].set_xticklabels(labels, rotation=35, ha="right")
    figure.suptitle(f"{plot_title}: weighted {metric}", y=0.995)
    figure.tight_layout(rect=[0, 0, 1, 0.985])
    output_path = output_dir / f"{safe_filename(plot_title)}_weighted_{metric}.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def main():
    args = parse_args()
    output_dir = args.output_dir or args.evaluation_files[0].parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.evaluation_files, input_labels(args))
    summary_path = output_dir / f"{safe_filename(args.plot_title)}_weighted_metrics.csv"
    rows.to_csv(summary_path, index=False)
    written = [summary_path]
    for metric in METRICS:
        written.append(plot_metric(rows, metric, output_dir, args.plot_title))
    print("Wrote:", flush=True)
    for path in written:
        print(path, flush=True)


if __name__ == "__main__":
    main()
