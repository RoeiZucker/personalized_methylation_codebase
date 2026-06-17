#!/usr/bin/env python3
"""Plot metrics from cls_downsample_liver eval_objects_dict output."""

import argparse
import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


BIN_RE = re.compile(
    r"^(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"-(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)
EPOCH_RE = re.compile(r"epoch-(\d+)-step-(\d+)")
METHODS = ["predicted_class", "mean_label"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "cls_downsample_liver_eval_objects_dict.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "cls_downsample_liver_plots",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_data(path):
    return ast.literal_eval(path.read_text(encoding="utf-8"))


def parse_bin(bin_key):
    match = BIN_RE.fullmatch(str(bin_key).strip())
    if match is None:
        return None
    low, high = map(float, match.groups())
    return low, high


def value_for_class(mapping, class_id):
    return next((value for key, value in mapping.items() if str(key) == class_id), 0)


def f1_score(precision, recall):
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def sample_from_path(path):
    run_name = Path(path).parents[1].name
    return run_name.split("_", 1)[0]


def checkpoint_from_path(path):
    match = EPOCH_RE.search(str(path))
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def class_support(confusion, class_id):
    return sum(
        float(value_for_class(predicted_counts, class_id))
        for predicted_counts in confusion.values()
    )


def rows_for_result(sample, path, bin_rank, bin_key, result):
    bounds = parse_bin(bin_key)
    if bounds is None:
        return []

    low, high = bounds
    epoch, step = checkpoint_from_path(path)
    rows = []
    for method in METHODS:
        if method not in result or f"{method}_confusion_matrix" not in result:
            continue

        class_rows = []
        confusion = result[f"{method}_confusion_matrix"]
        for raw_class, values in result[method].items():
            class_id = str(raw_class)
            precision = float(values["Precision"])
            recall = float(values["recall"])
            row = {
                "sample": sample,
                "result_path": path,
                "epoch": epoch,
                "step": step,
                "bin_rank": bin_rank,
                "bin": bin_key,
                "bin_low": low,
                "bin_high": high,
                "method": method,
                "class": class_id,
                "support": class_support(confusion, class_id),
                "precision": precision,
                "recall": recall,
                "f1": f1_score(precision, recall),
            }
            class_rows.append(row)
            rows.append(row)

        total_support = sum(row["support"] for row in class_rows)
        if total_support:
            rows.append(
                {
                    "sample": sample,
                    "result_path": path,
                    "epoch": epoch,
                    "step": step,
                    "bin_rank": bin_rank,
                    "bin": bin_key,
                    "bin_low": low,
                    "bin_high": high,
                    "method": method,
                    "class": "weighted",
                    "support": total_support,
                    "precision": sum(
                        row["precision"] * row["support"] for row in class_rows
                    )
                    / total_support,
                    "recall": sum(row["recall"] * row["support"] for row in class_rows)
                    / total_support,
                    "f1": sum(row["f1"] * row["support"] for row in class_rows)
                    / total_support,
                }
            )

    return rows


def extract_metrics(data):
    rows = []
    for sample, sample_results in data.items():
        for path, result_by_bin in sample_results.items():
            bin_keys = [
                key for key in result_by_bin if key != "all_results" and parse_bin(key)
            ]
            bin_keys = sorted(bin_keys, key=lambda key: parse_bin(key)[0])
            for bin_rank, bin_key in enumerate(bin_keys, start=1):
                rows.extend(
                    rows_for_result(sample, path, bin_rank, bin_key, result_by_bin[bin_key])
                )

    if not rows:
        raise ValueError("No variability-bin metrics found.")
    return pd.DataFrame(rows)


def rows_for_all_result(sample, path, result):
    epoch, step = checkpoint_from_path(path)
    rows = []
    for method in METHODS:
        if method not in result or f"{method}_confusion_matrix" not in result:
            continue

        class_rows = []
        confusion = result[f"{method}_confusion_matrix"]
        for raw_class, values in result[method].items():
            class_id = str(raw_class)
            precision = float(values["Precision"])
            recall = float(values["recall"])
            row = {
                "sample": sample,
                "result_path": path,
                "epoch": epoch,
                "step": step,
                "method": method,
                "class": class_id,
                "support": class_support(confusion, class_id),
                "precision": precision,
                "recall": recall,
                "f1": f1_score(precision, recall),
            }
            class_rows.append(row)
            rows.append(row)

        total_support = sum(row["support"] for row in class_rows)
        if total_support:
            rows.append(
                {
                    "sample": sample,
                    "result_path": path,
                    "epoch": epoch,
                    "step": step,
                    "method": method,
                    "class": "weighted",
                    "support": total_support,
                    "precision": sum(row["precision"] * row["support"] for row in class_rows) / total_support,
                    "recall": sum(row["recall"] * row["support"] for row in class_rows) / total_support,
                    "f1": sum(row["f1"] * row["support"] for row in class_rows) / total_support,
                }
            )
    return rows


def extract_all_results_metrics(data):
    rows = []
    for sample, sample_results in data.items():
        for path, result_by_bin in sample_results.items():
            if "all_results" in result_by_bin:
                rows.extend(rows_for_all_result(sample, path, result_by_bin["all_results"]))
    if not rows:
        raise ValueError("No all_results metrics found.")
    return pd.DataFrame(rows)


def plot_weighted_metric_lines(metrics, output_dir, dpi):
    weighted = metrics[metrics["class"] == "weighted"]
    plot_data = (
        weighted.groupby(["method", "bin_rank"], as_index=False)[["precision", "recall", "f1"]]
        .mean()
        .sort_values("bin_rank")
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)

    for axis, metric in zip(axes, ["f1", "precision", "recall"]):
        for method in METHODS:
            method_data = plot_data[plot_data["method"] == method]
            axis.plot(
                method_data["bin_rank"],
                method_data[metric],
                marker="o",
                linewidth=2,
                label=method.replace("_", " ").title(),
            )
        axis.set_title(metric.title())
        axis.set_xlabel("Variability bin")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean weighted score across samples")
    axes[-1].legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "weighted_metrics_by_variability_bin.png", dpi=dpi)
    plt.close(figure)


def plot_non_weighted_metric_lines(metrics, output_dir, dpi):
    class_metrics = metrics[metrics["class"] != "weighted"]
    plot_data = (
        class_metrics.groupby(["method", "bin_rank"], as_index=False)[
            ["precision", "recall", "f1"]
        ]
        .mean()
        .sort_values("bin_rank")
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)

    for axis, metric in zip(axes, ["f1", "precision", "recall"]):
        for method in METHODS:
            method_data = plot_data[plot_data["method"] == method]
            axis.plot(
                method_data["bin_rank"],
                method_data[metric],
                marker="o",
                linewidth=2,
                label=method.replace("_", " " ).title(),
            )
        axis.set_title(metric.title())
        axis.set_xlabel("Variability bin")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean non-weighted score across classes and samples")
    axes[-1].legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "non_weighted_metrics_by_variability_bin.png", dpi=dpi)
    plt.close(figure)


def plot_sample_non_weighted_metric_lines(metrics, output_dir, dpi):
    class_metrics = metrics[metrics["class"] != "weighted"]
    plot_data = (
        class_metrics.groupby(["sample", "method", "bin_rank"], as_index=False)[
            ["precision", "recall", "f1"]
        ]
        .mean()
        .sort_values(["sample", "bin_rank"])
    )
    samples = sorted(plot_data["sample"].unique())

    for metric in ["f1", "precision", "recall"]:
        figure, axes = plt.subplots(
            len(samples),
            1,
            figsize=(8, 2.5 * len(samples)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for axis, sample in zip(axes[:, 0], samples):
            sample_data = plot_data[plot_data["sample"] == sample]
            for method in METHODS:
                method_data = sample_data[sample_data["method"] == method].sort_values(
                    "bin_rank"
                )
                axis.plot(
                    method_data["bin_rank"],
                    method_data[metric],
                    marker="o",
                    label=method.replace("_", " " ).title(),
                )
            axis.set_title(sample)
            axis.set_ylabel(metric.title())
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.3)

        axes[-1, 0].set_xlabel("Variability bin")
        axes[0, 0].legend(loc="lower right")
        figure.tight_layout()
        figure.savefig(output_dir / f"non_weighted_{metric}_by_sample_and_bin.png", dpi=dpi)
        plt.close(figure)


def class_sort_key(class_id):
    try:
        return int(class_id)
    except ValueError:
        return class_id


def plot_per_class_metric_lines(metrics, output_dir, dpi):
    class_metrics = metrics[metrics["class"] != "weighted"]
    plot_data = (
        class_metrics.groupby(["class", "method", "bin_rank"], as_index=False)[
            ["precision", "recall", "f1"]
        ]
        .mean()
        .sort_values(["class", "bin_rank"])
    )

    for class_id in sorted(plot_data["class"].unique(), key=class_sort_key):
        class_data = plot_data[plot_data["class"] == class_id]
        figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)

        for axis, metric in zip(axes, ["f1", "precision", "recall"]):
            for method in METHODS:
                method_data = class_data[class_data["method"] == method].sort_values(
                    "bin_rank"
                )
                axis.plot(
                    method_data["bin_rank"],
                    method_data[metric],
                    marker="o",
                    linewidth=2,
                    label=method.replace("_", " " ).title(),
                )
            axis.set_title(metric.title())
            axis.set_xlabel("Variability bin")
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.3)

        axes[0].set_ylabel(f"Class {class_id} score")
        axes[-1].legend(loc="lower right")
        figure.suptitle(f"Class {class_id}: metrics by variability bin", y=0.98)
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        figure.savefig(
            output_dir / f"class_{class_id}_metrics_by_variability_bin.png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.15,
        )
        plt.close(figure)


def plot_all_results_class_metrics(all_metrics, output_dir, dpi):
    class_metrics = all_metrics[all_metrics["class"] != "weighted"]
    plot_data = (
        class_metrics.groupby(["class", "method"], as_index=False)[
            ["precision", "recall", "f1"]
        ]
        .mean()
        .sort_values("class")
    )
    classes = sorted(plot_data["class"].unique(), key=class_sort_key)
    x = range(len(classes))
    width = 0.35
    offsets = {METHODS[0]: -width / 2, METHODS[1]: width / 2}

    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, metric in zip(axes, ["f1", "precision", "recall"]):
        for method in METHODS:
            method_data = plot_data[plot_data["method"] == method].set_index("class")
            values = [method_data.loc[class_id, metric] for class_id in classes]
            positions = [value + offsets[method] for value in x]
            axis.bar(positions, values, width=width, label=method.replace("_", " " ).title())
            for position, value in zip(positions, values):
                axis.text(position, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        axis.set_title(metric.title())
        axis.set_xticks(list(x))
        axis.set_xticklabels([f"Class {class_id}" for class_id in classes])
        axis.set_ylim(0, 1.08)
        axis.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean score across samples")
    axes[-1].legend(loc="lower right")
    figure.suptitle("All positions: class metrics without variability bins", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(
        output_dir / "all_results_class_metrics.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(figure)


def plot_all_results_weighted_metrics(all_metrics, output_dir, dpi):
    weighted = all_metrics[all_metrics["class"] == "weighted"]
    plot_data = weighted.groupby("method", as_index=False)[["precision", "recall", "f1"]].mean()
    metrics = ["f1", "precision", "recall"]
    x = range(len(metrics))
    width = 0.35
    offsets = {METHODS[0]: -width / 2, METHODS[1]: width / 2}

    figure, axis = plt.subplots(figsize=(8, 4))
    for method in METHODS:
        method_data = plot_data[plot_data["method"] == method].iloc[0]
        values = [method_data[metric] for metric in metrics]
        positions = [value + offsets[method] for value in x]
        axis.bar(positions, values, width=width, label=method.replace("_", " " ).title())
        for position, value in zip(positions, values):
            axis.text(position, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    axis.set_xticks(list(x))
    axis.set_xticklabels([metric.title() for metric in metrics])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Mean weighted score across samples")
    axis.set_title("All positions: weighted metrics without variability bins")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "all_results_weighted_metrics.png", dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def plot_sample_weighted_f1(metrics, output_dir, dpi):
    weighted = metrics[metrics["class"] == "weighted"]
    samples = sorted(weighted["sample"].unique())
    figure, axes = plt.subplots(
        len(samples),
        1,
        figsize=(8, 2.5 * len(samples)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for axis, sample in zip(axes[:, 0], samples):
        sample_data = weighted[weighted["sample"] == sample]
        for method in METHODS:
            method_data = sample_data[sample_data["method"] == method].sort_values(
                "bin_rank"
            )
            axis.plot(
                method_data["bin_rank"],
                method_data["f1"],
                marker="o",
                label=method.replace("_", " ").title(),
            )
        axis.set_title(sample)
        axis.set_ylabel("Weighted F1")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.3)

    axes[-1, 0].set_xlabel("Variability bin")
    axes[0, 0].legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "weighted_f1_by_sample_and_bin.png", dpi=dpi)
    plt.close(figure)


def plot_class_f1_heatmaps(metrics, output_dir, dpi):
    class_metrics = metrics[metrics["class"] != "weighted"]
    for method in METHODS:
        method_data = class_metrics[class_metrics["method"] == method]
        pivot = method_data.pivot_table(
            index="class",
            columns="bin_rank",
            values="f1",
            aggfunc="mean",
        ).sort_index()

        figure, axis = plt.subplots(figsize=(8, 3.5))
        image = axis.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns)
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels([f"Class {value}" for value in pivot.index])
        axis.set_xlabel("Variability bin")
        axis.set_title(f"Mean class F1: {method.replace('_', ' ').title()}")

        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                value = pivot.iloc[row, col]
                axis.text(col, row, f"{value:.3f}", ha="center", va="center", color="white")

        figure.colorbar(image, ax=axis, label="F1")
        figure.tight_layout()
        figure.savefig(output_dir / f"class_f1_heatmap_{method}.png", dpi=dpi)
        plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.input_file)
    metrics = extract_metrics(data)
    all_metrics = extract_all_results_metrics(data)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    all_metrics.to_csv(args.output_dir / "all_results_metrics.csv", index=False)
    plot_all_results_class_metrics(all_metrics, args.output_dir, args.dpi)
    plot_all_results_weighted_metrics(all_metrics, args.output_dir, args.dpi)
    plot_weighted_metric_lines(metrics, args.output_dir, args.dpi)
    plot_non_weighted_metric_lines(metrics, args.output_dir, args.dpi)
    plot_sample_non_weighted_metric_lines(metrics, args.output_dir, args.dpi)
    plot_per_class_metric_lines(metrics, args.output_dir, args.dpi)
    plot_sample_weighted_f1(metrics, args.output_dir, args.dpi)
    plot_class_f1_heatmaps(metrics, args.output_dir, args.dpi)
    print(f"Saved plots and metrics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
