#!/usr/bin/env python3
# Examples:
#   python3 scripts/plot_group_comparison.py --input-root scripts/results/grouped_evaluations/variability_free --output-dir scripts/results/grouped_evaluations/variability_free/comparison_plots
#   python3 scripts/plot_group_comparison.py --input-root scripts/results/grouped_evaluations/variability_free --output-dir scripts/results/grouped_evaluations/variability_free/comparison_plots_no_all_two --exclude-method all_two
#   python3 scripts/plot_group_comparison.py --input-root scripts/results/grouped_evaluations/variability_bins --output-dir scripts/results/grouped_evaluations/variability_bins/comparison_plots --variability-bins
from argparse import ArgumentParser
from pathlib import Path
import ast
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ROOT = (
    Path(__file__).resolve().parent
    / "results"
    / "grouped_evaluations"
    / "variability_free"
)
GROUP_ORDER = [
    "no_pretraining",
    "first_phase_epoch_1",
    "first_phase_epoch_2",
    "first_phase_epoch_3",
    "first_phase_epoch_4",
    "first_phase_epoch_5",
    "first_phase_epoch_6",
    "epoch_1_pretraining",
    "epoch_2_pretraining",
    "epoch_3_pretraining",
    "epoch_4_pretraining",
    "epoch_5_pretraining",
    "epoch_6_pretraining",
]
GROUP_LABELS = {
    "no_pretraining": "No pretrain",
    "first_phase_epoch_1": "Pretrain E1",
    "first_phase_epoch_2": "Pretrain E2",
    "first_phase_epoch_3": "Pretrain E3",
    "first_phase_epoch_4": "Pretrain E4",
    "first_phase_epoch_5": "Pretrain E5",
    "first_phase_epoch_6": "Pretrain E6",
    "epoch_1_pretraining": "Retrain E1",
    "epoch_2_pretraining": "Retrain E2",
    "epoch_3_pretraining": "Retrain E3",
    "epoch_4_pretraining": "Retrain E4",
    "epoch_5_pretraining": "Retrain E5",
    "epoch_6_pretraining": "Retrain E6",
}
METHODS = ["predicted_class", "mean_label", "all_two"]
EPOCH_RE = re.compile(r"/epoch-(\d+)-step-\d+/")
BIN_RE = re.compile(r"^(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)-(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")


def group_label(group):
    if group in GROUP_LABELS:
        return GROUP_LABELS[group]
    match = re.fullmatch(r"first_phase_epoch_(\d+)", group)
    if match is not None:
        return f"Pretrain E{int(match.group(1))}"
    match = re.fullmatch(r"epoch_(\d+)_pretraining", group)
    if match is not None:
        return f"Retrain E{int(match.group(1))}"
    return group


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "comparison_plots",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--variability-bins", action="store_true")
    parser.add_argument(
        "--group",
        action="append",
        dest="groups",
        default=[],
        help="Group directory to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="Comma-separated group directories to include, in plot order.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods to plot. Defaults to every available method.",
    )
    parser.add_argument(
        "--exclude-method",
        action="append",
        default=[],
        help="Method to exclude from plots and the output metrics CSV. Can be passed multiple times.",
    )
    return parser.parse_args()


def parse_groups(args):
    groups = list(args.groups)
    if args.groups:
        groups.extend(group.strip() for group in args.groups.split(",") if group.strip())
    return groups or GROUP_ORDER


def parse_methods(args, metrics):
    available_methods = [method for method in METHODS if method in set(metrics["method"])]
    if args.methods:
        requested_methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    else:
        requested_methods = available_methods
    excluded_methods = set(args.exclude_method)
    return [method for method in requested_methods if method in available_methods and method not in excluded_methods]


def f1_score(precision, recall):
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def class_sort_key(class_id):
    try:
        return int(class_id)
    except ValueError:
        return class_id


def method_offsets(width, methods):
    center = (len(methods) - 1) / 2
    return {method: (index - center) * width for index, method in enumerate(methods)}


def value_for_class(mapping, class_id):
    return next((value for key, value in mapping.items() if str(key) == class_id), 0)


def class_support(confusion, class_id):
    return sum(
        float(value_for_class(predicted_counts, class_id))
        for predicted_counts in confusion.values()
    )


def epoch_num(path):
    match = EPOCH_RE.search(path)
    return int(match.group(1)) if match else -1


def parse_bin_key(key):
    match = BIN_RE.fullmatch(str(key).strip())
    if match is None:
        return None
    low, high = map(float, match.groups())
    return low, high


def rows_for_result(group, sample, result_path, result, bin_rank=None, bin_label=None):
    rows = []
    for method in METHODS:
        if method not in result or f"{method}_confusion_matrix" not in result:
            continue
        confusion = result[f"{method}_confusion_matrix"]
        class_rows = []
        for raw_class, values in result[method].items():
            class_id = str(raw_class)
            precision = float(values["Precision"])
            recall = float(values["recall"])
            row = {
                "group": group,
                "group_label": group_label(group),
                "sample": sample,
                "result_path": result_path,
                "epoch": epoch_num(result_path),
                "bin_rank": bin_rank,
                "bin_label": bin_label,
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
                    "group": group,
                    "group_label": group_label(group),
                    "sample": sample,
                    "result_path": result_path,
                    "epoch": epoch_num(result_path),
                    "bin_rank": bin_rank,
                    "bin_label": bin_label,
                    "method": method,
                    "class": "weighted",
                    "support": total_support,
                    "precision": sum(row["precision"] * row["support"] for row in class_rows) / total_support,
                    "recall": sum(row["recall"] * row["support"] for row in class_rows) / total_support,
                    "f1": sum(row["f1"] * row["support"] for row in class_rows) / total_support,
                }
            )
    return rows


def load_group_rows(input_root, group, final_only, variability_bins):
    path = input_root / group / "eval_objects_dict.txt"
    if not path.exists():
        return []
    data = ast.literal_eval(path.read_text(encoding="utf-8"))
    rows = []
    for sample, sample_results in data.items():
        items = list(sample_results.items())
        if final_only and items:
            items = [max(items, key=lambda item: epoch_num(item[0]))]
        for result_path, result_by_bin in items:
            if variability_bins:
                bin_keys = [key for key in result_by_bin if parse_bin_key(key) is not None]
                bin_keys = sorted(bin_keys, key=lambda key: parse_bin_key(key)[0])
                for bin_rank, bin_key in enumerate(bin_keys, start=1):
                    rows.extend(
                        rows_for_result(
                            group,
                            sample,
                            result_path,
                            result_by_bin[bin_key],
                            bin_rank=bin_rank,
                            bin_label=str(bin_key),
                        )
                    )
            elif "all_results" in result_by_bin:
                rows.extend(rows_for_result(group, sample, result_path, result_by_bin["all_results"]))
    return rows


def load_metrics(input_root, final_only, variability_bins, groups):
    rows = []
    for group in groups:
        rows.extend(load_group_rows(input_root, group, final_only, variability_bins))
    if not rows:
        raise ValueError(f"No eval_objects_dict.txt files found under {input_root}")
    return pd.DataFrame(rows)


def plot_weighted_metrics(metrics, output_dir, dpi, groups, methods):
    weighted = metrics[metrics["class"] == "weighted"]
    grouped = weighted.groupby(["group", "group_label", "method"])[["precision", "recall", "f1"]]
    mean_data = grouped.mean().reset_index()
    std_data = grouped.std().fillna(0)
    present_groups = [group for group in groups if group in mean_data["group"].unique()]
    methods = [method for method in methods if method in mean_data["method"].unique()]
    metrics_to_plot = ["f1", "precision", "recall"]
    x = list(range(len(present_groups)))
    width = min(0.8 / max(len(methods), 1), 0.28)
    offsets = method_offsets(width, methods)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for axis, metric in zip(axes, metrics_to_plot):
        for method in methods:
            method_data = mean_data[mean_data["method"] == method].set_index("group")
            values = [method_data.loc[group, metric] if group in method_data.index else 0 for group in present_groups]
            errors = [
                std_data.loc[(group, group_label(group), method), metric]
                if (group, group_label(group), method) in std_data.index
                else 0
                for group in present_groups
            ]
            positions = [value + offsets[method] for value in x]
            axis.bar(positions, values, width=width, yerr=errors, capsize=4, label=method.replace("_", " ").title())
            for position, value, error in zip(positions, values, errors):
                axis.text(
                    position,
                    min(value + error + 0.015, 1.07),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
        axis.set_title(metric.title())
        axis.set_xticks(x)
        axis.set_xticklabels([group_label(group) for group in present_groups], rotation=30, ha="right")
        axis.set_ylim(0, 1.1)
        axis.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean weighted score (+/- SD)")
    axes[-1].legend(loc="lower right")
    figure.suptitle("Group comparison: weighted metrics")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_dir / "group_comparison_weighted_metrics.png", dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def plot_class_f1(metrics, output_dir, dpi, groups, methods, output_name="group_comparison_class_f1.png", title="Group comparison: class F1"):
    class_metrics = metrics[metrics["class"] != "weighted"]
    grouped = class_metrics.groupby(["group", "group_label", "method", "class"])["f1"]
    mean_data = grouped.mean().reset_index()
    std_data = grouped.std().fillna(0)
    present_groups = [group for group in groups if group in mean_data["group"].unique()]
    methods = [method for method in methods if method in mean_data["method"].unique()]
    classes = sorted(mean_data["class"].unique(), key=class_sort_key)
    x = list(range(len(present_groups)))
    width = min(0.8 / max(len(methods), 1), 0.28)
    offsets = method_offsets(width, methods)

    figure, axes = plt.subplots(len(classes), 1, figsize=(12, 3.2 * len(classes)), sharex=True, sharey=True, squeeze=False)
    for axis, class_id in zip(axes[:, 0], classes):
        class_data = mean_data[mean_data["class"] == class_id]
        for method in methods:
            method_data = class_data[class_data["method"] == method].set_index("group")
            values = [method_data.loc[group, "f1"] if group in method_data.index else 0 for group in present_groups]
            errors = [
                std_data.loc[(group, group_label(group), method, class_id)]
                if (group, group_label(group), method, class_id) in std_data.index
                else 0
                for group in present_groups
            ]
            positions = [value + offsets[method] for value in x]
            axis.bar(positions, values, width=width, yerr=errors, capsize=4, label=method.replace("_", " ").title())
            for position, value, error in zip(positions, values, errors):
                axis.text(
                    position,
                    min(value + error + 0.015, 1.07),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
        axis.set_title(f"Class {class_id} F1")
        axis.set_ylim(0, 1.1)
        axis.grid(axis="y", alpha=0.3)
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels([group_label(group) for group in present_groups], rotation=30, ha="right")
    axes[0, 0].legend(loc="lower right")
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_dir / output_name, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def plot_class_f1_by_variability_bin(metrics, output_dir, dpi, groups, methods):
    if "bin_rank" not in metrics.columns:
        return
    bin_ranks = sorted(rank for rank in metrics["bin_rank"].dropna().unique())
    for bin_rank in bin_ranks:
        bin_metrics = metrics[metrics["bin_rank"] == bin_rank]
        if bin_metrics.empty:
            continue
        suffix = int(bin_rank) if float(bin_rank).is_integer() else bin_rank
        plot_class_f1(
            bin_metrics,
            output_dir,
            dpi,
            groups,
            methods,
            output_name=f"group_comparison_class_f1_bin_{suffix}.png",
            title=f"Group comparison: class F1, variability bin {suffix}",
        )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = parse_groups(args)
    metrics = load_metrics(args.input_root, args.final_only, args.variability_bins, groups)
    methods = parse_methods(args, metrics)
    if not methods:
        raise ValueError("No methods left to plot after applying --methods/--exclude-method")
    metrics = metrics[metrics["method"].isin(methods)]
    metrics.to_csv(args.output_dir / "group_comparison_metrics.csv", index=False)
    plot_weighted_metrics(metrics, args.output_dir, args.dpi, groups, methods)
    plot_class_f1(metrics, args.output_dir, args.dpi, groups, methods)
    if args.variability_bins:
        plot_class_f1_by_variability_bin(metrics, args.output_dir, args.dpi, groups, methods)
    print(f"Saved comparison plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
