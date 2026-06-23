#!/usr/bin/env python3
"""Plot metrics from grouped eval_objects_dict output."""

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
METHODS = ["predicted_class", "mean_label", "all_two"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / "eval_objects_dict.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "prediction_metric_plots",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--final-epoch-only",
        action="store_true",
        help="For each sample, keep only the highest retraining epoch prediction file.",
    )
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


def method_offsets(width, methods):
    center = (len(methods) - 1) / 2
    return {method: (index - center) * width for index, method in enumerate(methods)}


def class_support(confusion, class_id):
    return sum(
        float(value_for_class(predicted_counts, class_id))
        for predicted_counts in confusion.values()
    )


def filter_final_epoch_per_sample(data):
    filtered = {}
    for sample, sample_results in data.items():
        best_path = None
        best_key = (-1, -1)
        for path in sample_results:
            epoch, step = checkpoint_from_path(path)
            key = (epoch if epoch is not None else -1, step if step is not None else -1)
            if key > best_key:
                best_key = key
                best_path = path
        if best_path is not None:
            filtered[sample] = {best_path: sample_results[best_path]}
    return filtered


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


def plot_weighted_f1_bar_by_variability_bin(metrics, output_dir, dpi):
    weighted = metrics[metrics["class"] == "weighted"]
    grouped = weighted.groupby(["method", "bin_rank"])["f1"]
    plot_data = grouped.mean().reset_index().sort_values("bin_rank")
    error_data = grouped.std().fillna(0)

    bins = sorted(plot_data["bin_rank"].unique())
    methods = [method for method in METHODS if method in plot_data["method"].unique()]
    x = list(range(len(bins)))
    width = min(0.8 / max(len(methods), 1), 0.35)
    offsets = method_offsets(width, methods)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    for method in methods:
        method_data = plot_data[plot_data["method"] == method].set_index("bin_rank")
        values = [method_data.loc[bin_rank, "f1"] if bin_rank in method_data.index else 0 for bin_rank in bins]
        errors = [
            error_data.loc[(method, bin_rank)]
            if (method, bin_rank) in error_data.index
            else 0
            for bin_rank in bins
        ]
        positions = [value + offsets[method] for value in x]
        axis.bar(
            positions,
            values,
            width=width,
            yerr=errors,
            capsize=4,
            error_kw={"elinewidth": 1, "capthick": 1},
            label=method.replace("_", " " ).title(),
        )
        for position, value, error in zip(positions, values, errors):
            axis.text(
                position,
                min(value + error + 0.015, 1.06),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_xticks(x)
    axis.set_xticklabels([f"Bin {bin_rank}" for bin_rank in bins])
    axis.set_ylim(0, 1.1)
    axis.set_ylabel("Mean weighted F1 (+/- SD)")
    axis.set_xlabel("Variability bin")
    axis.set_title("Weighted F1 by variability bin")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(
        output_dir / "weighted_f1_bar_by_variability_bin.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
    )
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
    grouped = class_metrics.groupby(["class", "method"])[
        ["precision", "recall", "f1"]
    ]
    plot_data = grouped.mean().reset_index().sort_values("class")
    error_data = grouped.std().fillna(0)
    classes = sorted(plot_data["class"].unique(), key=class_sort_key)
    x = range(len(classes))
    methods = [method for method in METHODS if method in plot_data["method"].unique()]
    width = min(0.8 / len(methods), 0.35)
    offsets = method_offsets(width, methods)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, metric in zip(axes, ["f1", "precision", "recall"]):
        for method in methods:
            method_data = plot_data[plot_data["method"] == method].set_index("class")
            values = [method_data.loc[class_id, metric] for class_id in classes]
            errors = [error_data.loc[(class_id, method), metric] for class_id in classes]
            positions = [value + offsets[method] for value in x]
            axis.bar(
                positions,
                values,
                width=width,
                yerr=errors,
                capsize=4,
                error_kw={"elinewidth": 1, "capthick": 1},
                label=method.replace("_", " " ).title(),
            )
            for position, value, error in zip(positions, values, errors):
                axis.text(
                    position,
                    min(value + error + 0.015, 1.06),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.set_title(metric.title())
        axis.set_xticks(list(x))
        axis.set_xticklabels([f"Class {class_id}" for class_id in classes])
        axis.set_ylim(0, 1.1)
        axis.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Mean score across samples (+/- SD)")
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
    grouped = weighted.groupby("method")[["precision", "recall", "f1"]]
    plot_data = grouped.mean()
    error_data = grouped.std().fillna(0)
    metrics = ["f1", "precision", "recall"]
    x = range(len(metrics))
    methods = [method for method in METHODS if method in plot_data.index]
    width = min(0.8 / len(methods), 0.35)
    offsets = method_offsets(width, methods)

    figure, axis = plt.subplots(figsize=(8, 4))
    for method in methods:
        values = [plot_data.loc[method, metric] for metric in metrics]
        errors = [error_data.loc[method, metric] for metric in metrics]
        positions = [value + offsets[method] for value in x]
        axis.bar(
            positions,
            values,
            width=width,
            yerr=errors,
            capsize=4,
            error_kw={"elinewidth": 1, "capthick": 1},
            label=method.replace("_", " " ).title(),
        )
        for position, value, error in zip(positions, values, errors):
            axis.text(
                position,
                min(value + error + 0.015, 1.06),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_xticks(list(x))
    axis.set_xticklabels([metric.title() for metric in metrics])
    axis.set_ylim(0, 1.1)
    axis.set_ylabel("Mean weighted score across samples (+/- SD)")
    axis.set_title("All positions: weighted metrics without variability bins")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "all_results_weighted_metrics.png", dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def add_confusion_counts(total, confusion):
    for predicted_class, true_counts in confusion.items():
        predicted_class = str(predicted_class)
        for true_class, count in true_counts.items():
            total[(str(true_class), predicted_class)] = total.get((str(true_class), predicted_class), 0) + int(count)


def extract_all_results_confusion_matrices(data):
    matrices = {}
    for sample_results in data.values():
        for result_by_bin in sample_results.values():
            result = result_by_bin.get("all_results")
            if result is None:
                continue
            for method in METHODS:
                key = f"{method}_confusion_matrix"
                if key not in result:
                    continue
                matrices.setdefault(method, {})
                add_confusion_counts(matrices[method], result[key])
    return matrices


def confusion_dataframe(counts):
    labels = sorted(
        {label for pair in counts for label in pair},
        key=class_sort_key,
    )
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for (true_class, predicted_class), count in counts.items():
        matrix.loc[true_class, predicted_class] = count
    return matrix


def plot_all_results_confusion_matrices(data, output_dir, dpi):
    matrices = extract_all_results_confusion_matrices(data)
    methods = [method for method in METHODS if method in matrices]
    if not methods:
        return

    figure, axes = plt.subplots(
        1,
        len(methods),
        figsize=(4.2 * len(methods), 4),
        squeeze=False,
    )

    for axis, method in zip(axes[0], methods):
        matrix = confusion_dataframe(matrices[method])
        row_sums = matrix.sum(axis=1).replace(0, pd.NA)
        normalized = matrix.div(row_sums, axis=0).fillna(0)
        image = axis.imshow(normalized.values, vmin=0, vmax=1, cmap="Blues")

        axis.set_title(method.replace("_", " " ).title())
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(range(len(matrix.columns)))
        axis.set_xticklabels(matrix.columns)
        axis.set_yticks(range(len(matrix.index)))
        axis.set_yticklabels(matrix.index)

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                frac = normalized.iloc[row, col]
                count = matrix.iloc[row, col]
                color = "white" if frac >= 0.5 else "black"
                axis.text(
                    col,
                    row,
                    f"{frac:.2f}\n{count:,}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

    figure.subplots_adjust(left=0.07, right=0.88, bottom=0.16, top=0.82, wspace=0.35)
    colorbar_axis = figure.add_axes([0.91, 0.18, 0.015, 0.58])
    figure.colorbar(image, cax=colorbar_axis, label="Row-normalized fraction")
    figure.suptitle("All positions: confusion matrices without variability bins", y=0.96)
    figure.savefig(
        output_dir / "all_results_confusion_matrices.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(figure)


def plot_variability_bin_class_metric_grid(metrics, output_dir, dpi):
    class_metrics = metrics[metrics["class"] != "weighted"]
    grouped = class_metrics.groupby(["bin_rank", "class", "method"])[
        ["precision", "recall", "f1"]
    ]
    plot_data = grouped.mean().reset_index()
    error_data = grouped.std().fillna(0)

    bins = (
        plot_data[["bin_rank"]]
        .drop_duplicates()
        .sort_values("bin_rank")
    )
    classes = sorted(plot_data["class"].unique(), key=class_sort_key)
    methods = [method for method in METHODS if method in plot_data["method"].unique()]
    metric_names = ["f1", "precision", "recall"]

    x = list(range(len(classes)))
    width = min(0.8 / max(len(methods), 1), 0.35)
    offsets = method_offsets(width, methods)
    figure, axes = plt.subplots(
        len(bins),
        len(metric_names),
        figsize=(4.8 * len(metric_names), 2.9 * len(bins)),
        sharey=True,
        squeeze=False,
    )

    for row, bin_info in enumerate(bins.itertuples(index=False)):
        bin_data = plot_data[plot_data["bin_rank"] == bin_info.bin_rank]
        for col, metric in enumerate(metric_names):
            axis = axes[row, col]
            for method in methods:
                method_data = bin_data[bin_data["method"] == method].set_index("class")
                values = [method_data.loc[class_id, metric] if class_id in method_data.index else 0 for class_id in classes]
                errors = [
                    error_data.loc[(bin_info.bin_rank, class_id, method), metric]
                    if (bin_info.bin_rank, class_id, method) in error_data.index
                    else 0
                    for class_id in classes
                ]
                positions = [value + offsets[method] for value in x]
                axis.bar(
                    positions,
                    values,
                    width=width,
                    yerr=errors,
                    capsize=3,
                    error_kw={"elinewidth": 1, "capthick": 1},
                    label=method.replace("_", " " ).title(),
                )
            if row == 0:
                axis.set_title(metric.title())
            if col == 0:
                axis.set_ylabel(f"Variability bin {bin_info.bin_rank}")
            axis.set_xticks(x)
            axis.set_xticklabels([f"Class {class_id}" for class_id in classes])
            axis.set_ylim(0, 1.1)
            axis.grid(axis="y", alpha=0.3)

    handles, labels = axes[0, -1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=max(len(methods), 1))
    figure.suptitle("Class metrics by variability bin", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(
        output_dir / "class_metrics_by_variability_bin_grid.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.15,
    )
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
    if args.final_epoch_only:
        data = filter_final_epoch_per_sample(data)
    all_metrics = extract_all_results_metrics(data)
    all_metrics.to_csv(args.output_dir / "all_results_metrics.csv", index=False)
    plot_all_results_class_metrics(all_metrics, args.output_dir, args.dpi)
    plot_all_results_weighted_metrics(all_metrics, args.output_dir, args.dpi)
    plot_all_results_confusion_matrices(data, args.output_dir, args.dpi)

    try:
        metrics = extract_metrics(data)
    except ValueError as exc:
        print(f"Skipping variability-bin plots: {exc}")
    else:
        metrics.to_csv(args.output_dir / "metrics.csv", index=False)
        plot_weighted_metric_lines(metrics, args.output_dir, args.dpi)
        plot_weighted_f1_bar_by_variability_bin(metrics, args.output_dir, args.dpi)
        plot_non_weighted_metric_lines(metrics, args.output_dir, args.dpi)
        plot_sample_non_weighted_metric_lines(metrics, args.output_dir, args.dpi)
        plot_per_class_metric_lines(metrics, args.output_dir, args.dpi)
        plot_variability_bin_class_metric_grid(metrics, args.output_dir, args.dpi)
        plot_sample_weighted_f1(metrics, args.output_dir, args.dpi)
        plot_class_f1_heatmaps(metrics, args.output_dir, args.dpi)

    print(f"Saved plots and metrics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
