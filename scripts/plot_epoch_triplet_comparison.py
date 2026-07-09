#!/usr/bin/env python3
# Examples:
#   python3 scripts/plot_epoch_triplet_comparison.py --metrics-csv scripts/results/all_tissues_grouped_evaluations/combined_as_single_tissue_all_groups/variability_free/comparison_plots_all_groups/group_comparison_metrics.csv --output-dir scripts/results/all_tissues_grouped_evaluations/combined_as_single_tissue_all_groups/variability_free/epoch_triplet_plots_equal_weighted_complete_pretrain
#   python3 scripts/plot_epoch_triplet_comparison.py --metrics-csv scripts/results/all_tissues_grouped_evaluations/combined_as_single_tissue_all_groups/variability_free/comparison_plots_all_groups/group_comparison_metrics.csv --output-dir scripts/results/all_tissues_grouped_evaluations/combined_as_single_tissue_all_groups/variability_free/epoch_triplet_plots_split_by_tissue_cohort --split-by-tissue-cohort
#   python3 scripts/plot_epoch_triplet_comparison.py --metrics-csv scripts/results/all_tissues_grouped_evaluations/combined_as_single_tissue_all_groups/variability_free/comparison_plots_all_groups/group_comparison_metrics.csv --output-dir scripts/results/all_tissues_grouped_evaluations/combined_as_single_tissue_all_groups/variability_free/epoch_triplet_plots_complete_through_e10_e12 --complete-through-epochs 10,12
#   python3 scripts/plot_epoch_triplet_comparison.py --metrics-csv scripts/results/all_tissues_grouped_evaluations/combined_as_single_tissue_all_groups/variability_free/comparison_plots_all_groups/group_comparison_metrics.csv --output-dir scripts/results/all_tissues_grouped_evaluations/combined_as_single_tissue_all_groups/variability_free/epoch_triplet_plots_final_finetune_complete_through_e8_e10_e12 --complete-through-epochs 8,10,12 --final-finetune-only
#   python3 scripts/plot_epoch_triplet_comparison.py --metrics-csv scripts/results/all_tissues_grouped_evaluations/Liver-Hepatocytes/variability_free/comparison_plots_all_groups/group_comparison_metrics.csv --output-dir scripts/results/all_tissues_grouped_evaluations/Liver-Hepatocytes/variability_free/epoch_triplet_plots_equal_weighted_complete_pretrain --title-prefix Liver-Hepatocytes
from argparse import ArgumentParser
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BAR_ORDER = ["mean_label", "pretrain_prediction", "retrain_prediction"]
BAR_LABELS = {
    "mean_label": "Mean Label",
    "pretrain_prediction": "Pretrain Prediction",
    "retrain_prediction": "Retrain Prediction",
}
BAR_COLORS = {
    "mean_label": "#4c78a8",
    "pretrain_prediction": "#f58518",
    "retrain_prediction": "#54a24b",
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", default="")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--allow-incomplete-pretrain-epochs",
        action="store_true",
        help="Plot tissue/epoch pairs even when not every person has the pretrain prediction for that epoch.",
    )
    parser.add_argument(
        "--split-by-tissue-cohort",
        action="store_true",
        help="Create separate plots for epoch ranges that share the same included tissue set.",
    )
    parser.add_argument(
        "--complete-through-epochs",
        default=None,
        help="Comma-separated max epochs. For each N, plot E1..EN using only tissues complete for every pretrain epoch through N.",
    )
    parser.add_argument(
        "--final-finetune-only",
        action="store_true",
        help="For retrain prediction bars, use only the highest fine-tune checkpoint epoch per tissue/person/pretraining epoch.",
    )
    return parser.parse_args()


def group_epoch(group):
    match = re.fullmatch(r"first_phase_epoch_(\d+)", str(group))
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"epoch_(\d+)_pretraining", str(group))
    if match:
        return int(match.group(1))
    return None


def prediction_kind(row):
    if row["method"] == "mean_label":
        return "mean_label"
    if row["method"] != "predicted_class":
        return None
    group = str(row["group"])
    if group.startswith("first_phase_epoch_"):
        return "pretrain_prediction"
    if re.fullmatch(r"epoch_\d+_pretraining", group):
        return "retrain_prediction"
    return None


def split_tissue_sample(sample):
    sample = str(sample)
    if "::" in sample:
        return sample.split("::", 1)
    return "single_tissue", sample


def load_epoch_metrics(metrics_csv):
    metrics = pd.read_csv(metrics_csv)
    metrics = metrics[metrics["group"] != "no_pretraining"].copy()
    metrics["epoch_number"] = metrics["group"].map(group_epoch)
    metrics["bar_kind"] = metrics.apply(prediction_kind, axis=1)
    metrics = metrics.dropna(subset=["epoch_number", "bar_kind"])
    metrics["epoch_number"] = metrics["epoch_number"].astype(int)
    tissue_sample = metrics["sample"].map(split_tissue_sample)
    metrics["tissue"] = tissue_sample.map(lambda value: value[0])
    metrics["person"] = tissue_sample.map(lambda value: value[1])
    return metrics


def filter_final_finetune_only(metrics):
    retrain_metrics = metrics[metrics["bar_kind"] == "retrain_prediction"].copy()
    other_metrics = metrics[metrics["bar_kind"] != "retrain_prediction"].copy()
    if retrain_metrics.empty:
        return metrics.copy()
    max_epoch_by_person = retrain_metrics.groupby(
        ["epoch_number", "tissue", "person"]
    )["epoch"].transform("max")
    final_retrain_metrics = retrain_metrics[retrain_metrics["epoch"] == max_epoch_by_person].copy()
    return pd.concat([other_metrics, final_retrain_metrics], ignore_index=True)


def complete_pretrain_tissue_epochs(metrics):
    people_per_tissue = metrics.groupby("tissue")["person"].nunique()
    pretrain_metrics = metrics[metrics["bar_kind"] == "pretrain_prediction"]
    pretrain_people = (
        pretrain_metrics.groupby(["tissue", "epoch_number"])["person"]
        .nunique()
        .reset_index(name="pretrain_people")
    )
    pretrain_people["total_people"] = pretrain_people["tissue"].map(people_per_tissue)
    pretrain_people["is_complete"] = (
        pretrain_people["pretrain_people"] == pretrain_people["total_people"]
    )
    return pretrain_people


def filter_complete_pretrain_epochs(metrics):
    completeness = complete_pretrain_tissue_epochs(metrics)
    complete_pairs = {
        (row.tissue, int(row.epoch_number))
        for row in completeness.itertuples(index=False)
        if row.is_complete
    }
    filtered_metrics = metrics[
        metrics.apply(
            lambda row: (row["tissue"], int(row["epoch_number"])) in complete_pairs,
            axis=1,
        )
    ].copy()
    return filtered_metrics, completeness


def aggregate_metric(metrics, metric_name, class_id):
    subset = metrics[metrics["class"].astype(str) == str(class_id)].copy()
    person_level = (
        subset.groupby(["epoch_number", "bar_kind", "tissue", "person"], as_index=False)[metric_name]
        .mean()
    )
    tissue_level = (
        person_level.groupby(["epoch_number", "bar_kind", "tissue"], as_index=False)[metric_name]
        .mean()
    )
    grouped = tissue_level.groupby(["epoch_number", "bar_kind"])[metric_name]
    return grouped.agg(["mean", "std"]).reset_index().fillna({"std": 0})


def draw_grouped_bars(axis, data, metric_name, epochs):
    width = 0.24
    offsets = {kind: (index - 1) * width for index, kind in enumerate(BAR_ORDER)}
    x = list(range(len(epochs)))
    data_by_key = data.set_index(["epoch_number", "bar_kind"])
    for kind in BAR_ORDER:
        values = []
        errors = []
        for epoch in epochs:
            key = (epoch, kind)
            if key in data_by_key.index:
                values.append(float(data_by_key.loc[key, "mean"]))
                errors.append(float(data_by_key.loc[key, "std"]))
            else:
                values.append(0.0)
                errors.append(0.0)
        positions = [position + offsets[kind] for position in x]
        axis.bar(
            positions,
            values,
            width=width,
            yerr=errors,
            capsize=4,
            label=BAR_LABELS[kind],
            color=BAR_COLORS[kind],
        )
        for position, value, error in zip(positions, values, errors):
            if value == 0 and error == 0:
                continue
            axis.text(
                position,
                min(value + error + 0.015, 1.07),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    axis.set_xticks(x)
    axis.set_xticklabels([f"E{epoch}" for epoch in epochs], rotation=30, ha="right")
    axis.set_ylim(0, 1.1)
    axis.grid(axis="y", alpha=0.3)
    axis.set_ylabel(metric_name.title())


def wrapped_list(values):
    return ", ".join(values) if values else "none"


def add_cohort_note(figure, included_tissues, exclusion_note):
    note = f"Included tissues: {wrapped_list(included_tissues)}"
    if exclusion_note:
        note = f"{note}\nExcluded: {exclusion_note}"
    figure.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8)


def output_stem(prefix, output_name):
    path = Path(output_name)
    return f"{prefix}_{path.stem}{path.suffix}"


def plot_weighted(metrics, output_dir, dpi, title_prefix, included_tissues=None, exclusion_note="", output_prefix=""):
    epochs = sorted(metrics["epoch_number"].unique())
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for axis, metric_name in zip(axes, ["f1", "precision", "recall"]):
        data = aggregate_metric(metrics, metric_name, "weighted")
        draw_grouped_bars(axis, data, metric_name, epochs)
        axis.set_title(f"Weighted {metric_name.title()}")
    axes[-1].legend(loc="lower right")
    title = "Epoch comparison: weighted metrics"
    if title_prefix:
        title = f"{title_prefix}: {title}"
    figure.suptitle(title)
    if included_tissues is not None:
        add_cohort_note(figure, included_tissues, exclusion_note)
    figure.tight_layout(rect=(0, 0.05 if included_tissues is not None else 0, 1, 0.93))
    output_name = output_stem(output_prefix, "epoch_triplet_weighted_metrics.png") if output_prefix else "epoch_triplet_weighted_metrics.png"
    figure.savefig(output_dir / output_name, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def plot_class_metric(metrics, output_dir, dpi, title_prefix, metric_name, included_tissues=None, exclusion_note="", output_prefix=""):
    class_metrics = metrics[metrics["class"].astype(str) != "weighted"]
    classes = sorted(class_metrics["class"].astype(str).unique(), key=lambda value: int(value) if value.isdigit() else value)
    epochs = sorted(metrics["epoch_number"].unique())
    figure, axes = plt.subplots(len(classes), 1, figsize=(15, 3.5 * len(classes)), sharex=True, sharey=True, squeeze=False)
    for axis, class_id in zip(axes[:, 0], classes):
        data = aggregate_metric(class_metrics, metric_name, class_id)
        draw_grouped_bars(axis, data, metric_name, epochs)
        axis.set_title(f"Class {class_id} {metric_name.title()}")
    axes[0, 0].legend(loc="lower right")
    title = f"Epoch comparison: class {metric_name.title()}"
    if title_prefix:
        title = f"{title_prefix}: {title}"
    figure.suptitle(title)
    if included_tissues is not None:
        add_cohort_note(figure, included_tissues, exclusion_note)
    figure.tight_layout(rect=(0, 0.06 if included_tissues is not None else 0, 1, 0.95))
    output_name = f"epoch_triplet_class_{metric_name}.png"
    if output_prefix:
        output_name = output_stem(output_prefix, output_name)
    figure.savefig(output_dir / output_name, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def format_epoch_range(epochs):
    epochs = sorted(epochs)
    if not epochs:
        return "none"
    ranges = []
    start = previous = epochs[0]
    for epoch in epochs[1:]:
        if epoch == previous + 1:
            previous = epoch
            continue
        ranges.append((start, previous))
        start = previous = epoch
    ranges.append((start, previous))
    return "_".join(f"e{low}" if low == high else f"e{low}-e{high}" for low, high in ranges)


def exclusion_reasons(all_tissues, included_tissues, epochs, completeness):
    rows = []
    completeness_by_key = completeness.set_index(["tissue", "epoch_number"])
    for tissue in all_tissues:
        if tissue in included_tissues:
            continue
        reasons = []
        for epoch in epochs:
            key = (tissue, epoch)
            if key not in completeness_by_key.index:
                reasons.append(f"E{epoch}: no pretrain prediction")
                continue
            row = completeness_by_key.loc[key]
            if not bool(row["is_complete"]):
                reasons.append(
                    f"E{epoch}: {int(row['pretrain_people'])}/{int(row['total_people'])} people"
                )
        if reasons:
            rows.append(f"{tissue} ({'; '.join(reasons)})")
    return "; ".join(rows)


def cohort_infos(metrics, completeness):
    all_tissues = sorted(metrics["tissue"].unique())
    complete_rows = completeness[completeness["is_complete"]].copy()
    included_by_epoch = (
        complete_rows.groupby("epoch_number")["tissue"]
        .apply(lambda values: tuple(sorted(values)))
        .to_dict()
    )
    cohorts = {}
    for epoch, tissues in included_by_epoch.items():
        if not tissues:
            continue
        cohorts.setdefault(tissues, []).append(int(epoch))
    infos = []
    for tissues, epochs in cohorts.items():
        epochs = sorted(epochs)
        included_tissues = list(tissues)
        infos.append(
            {
                "epochs": epochs,
                "included_tissues": included_tissues,
                "output_prefix": f"{format_epoch_range(epochs)}_{len(included_tissues)}tissues",
                "exclusion_note": exclusion_reasons(all_tissues, included_tissues, epochs, completeness),
            }
        )
    return sorted(infos, key=lambda info: info["epochs"][0])


def plot_all_metrics(metrics, output_dir, dpi, title_prefix, included_tissues=None, exclusion_note="", output_prefix=""):
    plot_weighted(metrics, output_dir, dpi, title_prefix, included_tissues, exclusion_note, output_prefix)
    for metric_name in ["f1", "precision", "recall"]:
        plot_class_metric(
            metrics,
            output_dir,
            dpi,
            title_prefix,
            metric_name,
            included_tissues,
            exclusion_note,
            output_prefix,
        )


def plot_cohort_split(metrics, completeness, output_dir, dpi, title_prefix):
    summaries = []
    for info in cohort_infos(metrics, completeness):
        cohort_metrics = metrics[
            metrics["epoch_number"].isin(info["epochs"])
            & metrics["tissue"].isin(info["included_tissues"])
        ].copy()
        if cohort_metrics.empty:
            continue
        title = f"{title_prefix}: {wrapped_list(info['included_tissues'])}" if title_prefix else wrapped_list(info["included_tissues"])
        plot_all_metrics(
            cohort_metrics,
            output_dir,
            dpi,
            title,
            included_tissues=info["included_tissues"],
            exclusion_note=info["exclusion_note"],
            output_prefix=info["output_prefix"],
        )
        cohort_metrics.to_csv(output_dir / f"{info['output_prefix']}_epoch_triplet_metrics.csv", index=False)
        summaries.append(
            {
                "epochs": ",".join(str(epoch) for epoch in info["epochs"]),
                "included_tissues": wrapped_list(info["included_tissues"]),
                "excluded_tissues_and_reasons": info["exclusion_note"],
                "output_prefix": info["output_prefix"],
            }
        )
    pd.DataFrame(summaries).to_csv(output_dir / "cohort_split_summary.csv", index=False)


def parse_complete_through_epochs(value):
    if value is None:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def tissues_complete_through_epoch(all_tissues, max_epoch, completeness):
    complete = completeness.set_index(["tissue", "epoch_number"])
    included = []
    for tissue in all_tissues:
        is_complete = True
        for epoch in range(1, max_epoch + 1):
            key = (tissue, epoch)
            if key not in complete.index or not bool(complete.loc[key, "is_complete"]):
                is_complete = False
                break
        if is_complete:
            included.append(tissue)
    return included


def plot_complete_through_ranges(metrics, completeness, max_epochs, output_dir, dpi, title_prefix):
    all_tissues = sorted(metrics["tissue"].unique())
    summaries = []
    for max_epoch in max_epochs:
        epochs = list(range(1, max_epoch + 1))
        included_tissues = tissues_complete_through_epoch(all_tissues, max_epoch, completeness)
        if not included_tissues:
            summaries.append(
                {
                    "epochs": f"1-{max_epoch}",
                    "included_tissues": "none",
                    "excluded_tissues_and_reasons": exclusion_reasons(all_tissues, [], epochs, completeness),
                    "output_prefix": f"e1-e{max_epoch}_0tissues",
                }
            )
            continue
        cohort_metrics = metrics[
            metrics["epoch_number"].isin(epochs)
            & metrics["tissue"].isin(included_tissues)
        ].copy()
        if cohort_metrics.empty:
            continue
        output_prefix = f"e1-e{max_epoch}_{len(included_tissues)}tissues_complete_through_e{max_epoch}"
        exclusion_note = exclusion_reasons(all_tissues, included_tissues, epochs, completeness)
        title = (
            f"{title_prefix}: E1-E{max_epoch}, {wrapped_list(included_tissues)}"
            if title_prefix
            else f"E1-E{max_epoch}, {wrapped_list(included_tissues)}"
        )
        plot_all_metrics(
            cohort_metrics,
            output_dir,
            dpi,
            title,
            included_tissues=included_tissues,
            exclusion_note=exclusion_note,
            output_prefix=output_prefix,
        )
        cohort_metrics.to_csv(output_dir / f"{output_prefix}_epoch_triplet_metrics.csv", index=False)
        summaries.append(
            {
                "epochs": f"1-{max_epoch}",
                "included_tissues": wrapped_list(included_tissues),
                "excluded_tissues_and_reasons": exclusion_note,
                "output_prefix": output_prefix,
            }
        )
    pd.DataFrame(summaries).to_csv(output_dir / "complete_through_summary.csv", index=False)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_epoch_metrics(args.metrics_csv)
    if args.final_finetune_only:
        metrics = filter_final_finetune_only(metrics)
    completeness = complete_pretrain_tissue_epochs(metrics)
    if not args.allow_incomplete_pretrain_epochs:
        metrics, completeness = filter_complete_pretrain_epochs(metrics)
    completeness.to_csv(args.output_dir / "pretrain_epoch_completeness.csv", index=False)
    if metrics.empty:
        raise ValueError("No metrics left after applying complete-pretrain-epoch filtering")
    metrics.to_csv(args.output_dir / "epoch_triplet_metrics.csv", index=False)
    complete_through_epochs = parse_complete_through_epochs(args.complete_through_epochs)
    if complete_through_epochs:
        plot_complete_through_ranges(
            metrics,
            completeness,
            complete_through_epochs,
            args.output_dir,
            args.dpi,
            args.title_prefix,
        )
    elif args.split_by_tissue_cohort:
        plot_cohort_split(metrics, completeness, args.output_dir, args.dpi, args.title_prefix)
    else:
        included_tissues = sorted(metrics["tissue"].unique())
        plot_all_metrics(metrics, args.output_dir, args.dpi, args.title_prefix, included_tissues=included_tissues)
    print(f"Saved epoch triplet plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
