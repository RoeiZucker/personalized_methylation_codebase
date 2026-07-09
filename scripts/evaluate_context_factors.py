#!/usr/bin/env python3
# Examples:
#   python3 scripts/evaluate_context_factors.py \
#     --results-dir /sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer \
#     --output-dir scripts/results/context_factors/Liver-Hepatocytes \
#     --tissue-name Liver-Hepatocytes \
#     --groups epoch_10_pretraining,no_pretraining \
#     --gene-bed /path/to/genes.bed \
#     --cpg-island-bed /path/to/cpg_islands.bed \
#     --final-only
#   python3 scripts/evaluate_context_factors.py \
#     --results-dir /sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer \
#     --output-dir scripts/results/context_factors/Liver-Hepatocytes_model_vs_mean \
#     --tissue-name Liver-Hepatocytes \
#     --groups first_phase_epoch_10,epoch_10_pretraining,no_pretraining \
#     --gene-bed annotations/hg38/hg38_refGene_gene_intervals.bed \
#     --cpg-island-bed annotations/hg38/hg38_cpgIslandExt.bed \
#     --comparison-dicts-dir /sci/labs/michall/roeizucker/comparison_dicts/Liver-Hepatocytes \
#     --include-mean-label \
#     --final-only
#   python3 scripts/evaluate_context_factors.py \
#     --results-dir /sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer \
#     --output-dir scripts/results/context_factors/Liver-Hepatocytes_smoke \
#     --tissue-name Liver-Hepatocytes \
#     --pretraining-group epoch_10_pretraining \
#     --gene-bed /path/to/genes.bed \
#     --cpg-island-bed /path/to/cpg_islands.bed \
#     --max-files 1

from argparse import ArgumentParser
from pathlib import Path
import pickle
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


NO_PRETRAINING_RUN_RE = re.compile(r"^[^_]+_no_pretraining_retrain_")
RETRAIN_RUN_RE = re.compile(r"^[^_]+_epoch-(?P<epoch>\d+)-step-\d+_retrain_")
FIRST_PHASE_RUN_RE = re.compile(r"^[^_]+_epoch-(?P<epoch>\d+)-step-\d+_pretrain_prediction$")
EPOCH_RE = re.compile(r"/epoch-(?P<epoch>\d+)-step-\d+/")
DEFAULT_LABELS = [0, 1, 2]
PREDICTION_COLUMNS = [
    "window_id",
    "window_start",
    "window_end",
    "base_position_in_window",
    "genomic_position",
    "label",
    "predicted_class",
]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tissue-name", required=True)
    parser.add_argument("--gene-bed", type=Path, required=True)
    parser.add_argument("--cpg-island-bed", type=Path, required=True)
    parser.add_argument("--pretraining-group", default=None)
    parser.add_argument("--groups", default=None)
    parser.add_argument("--batch-size", default="14")
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--include-mean-label", action="store_true")
    parser.add_argument("--comparison-dicts-dir", type=Path, default=None)
    parser.add_argument("--ranges", default="0,0.3333333333333333,0.6666666666666666,1")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def parse_ranges(value):
    ranges = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(ranges) < 2:
        raise ValueError("--ranges must contain at least two numeric boundaries")
    return ranges


def parse_groups(args):
    groups = []
    if args.pretraining_group:
        groups.append(args.pretraining_group)
    if args.groups:
        groups.extend(group.strip() for group in args.groups.split(",") if group.strip())
    return groups


def group_name_from_run_dir(run_dir_name):
    if NO_PRETRAINING_RUN_RE.search(run_dir_name):
        return "no_pretraining"
    match = RETRAIN_RUN_RE.search(run_dir_name)
    if match is not None:
        return f"epoch_{int(match.group('epoch'))}_pretraining"
    match = FIRST_PHASE_RUN_RE.search(run_dir_name)
    if match is not None:
        return f"first_phase_epoch_{int(match.group('epoch'))}"
    return None


def sample_name_from_result_path(result_file_path):
    return result_file_path.parents[1].name.split("_", 1)[0]


def epoch_num(path):
    match = EPOCH_RE.search(str(path))
    return int(match.group("epoch")) if match else -1


def discover_result_files(results_dir, groups, batch_size):
    selected_groups = set(groups)
    rows = []
    for result_file_path in sorted(results_dir.glob("*/epoch-*/eval_predictions.csv.gitbackup")):
        run_dir_name = result_file_path.parents[1].name
        if "_low_var" in run_dir_name:
            continue
        group = group_name_from_run_dir(run_dir_name)
        if group is None:
            continue
        if selected_groups and group not in selected_groups:
            continue
        if not group.startswith("first_phase_") and f"_bs_{batch_size}_" not in run_dir_name:
            continue
        rows.append(
            {
                "group": group,
                "sample": sample_name_from_result_path(result_file_path),
                "path": result_file_path,
                "epoch": epoch_num(result_file_path),
            }
        )
    return rows


def keep_final_files(rows):
    if not rows:
        return rows
    frame = pd.DataFrame(rows)
    idx = frame.groupby(["group", "sample"])["epoch"].idxmax()
    return frame.loc[idx].sort_values(["group", "sample", "path"]).to_dict("records")


def read_bed(path):
    bed = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )
    bed["start"] = pd.to_numeric(bed["start"], errors="coerce")
    bed["end"] = pd.to_numeric(bed["end"], errors="coerce")
    bed = bed.dropna(subset=["chrom", "start", "end"]).copy()
    bed["start"] = bed["start"].astype(int)
    bed["end"] = bed["end"].astype(int)
    return bed[bed["end"] > bed["start"]]


def build_interval_index(bed):
    index = {}
    for chrom, chrom_bed in bed.groupby("chrom", sort=False):
        sorted_bed = chrom_bed.sort_values("start")
        starts = sorted_bed["start"].to_numpy(dtype=np.int64)
        ends = sorted_bed["end"].to_numpy(dtype=np.int64)
        index[chrom] = (starts, np.maximum.accumulate(ends))
    return index


def interval_overlap_flags(chroms, starts, ends, interval_index):
    flags = np.zeros(len(starts), dtype=bool)
    chrom_series = pd.Series(np.asarray(chroms), index=range(len(starts)))
    starts_series = pd.Series(np.asarray(starts), index=range(len(starts)), dtype="int64")
    ends_series = pd.Series(np.asarray(ends), index=range(len(starts)), dtype="int64")
    for chrom, row_indexes in chrom_series.groupby(chrom_series).groups.items():
        if chrom not in interval_index:
            continue
        bed_starts, cumulative_ends = interval_index[chrom]
        query_starts = starts_series.iloc[list(row_indexes)].to_numpy(dtype=np.int64)
        query_ends = ends_series.iloc[list(row_indexes)].to_numpy(dtype=np.int64)
        interval_indexes = np.searchsorted(bed_starts, query_ends, side="left") - 1
        valid = interval_indexes >= 0
        overlap = np.zeros(len(query_starts), dtype=bool)
        overlap[valid] = cumulative_ends[interval_indexes[valid]] > query_starts[valid]
        flags[np.asarray(list(row_indexes), dtype=int)] = overlap
    return flags


def load_prediction_dataframe(path):
    prediction = pd.read_csv(path, usecols=PREDICTION_COLUMNS)
    prediction["label"] = pd.to_numeric(prediction["label"], errors="coerce")
    prediction["predicted_class"] = pd.to_numeric(prediction["predicted_class"], errors="coerce")
    prediction["genomic_position"] = pd.to_numeric(prediction["genomic_position"], errors="coerce")
    prediction["base_position_in_window"] = pd.to_numeric(
        prediction["base_position_in_window"],
        errors="coerce",
    )
    prediction = prediction.dropna(
        subset=["window_id", "genomic_position", "base_position_in_window", "label", "predicted_class"]
    ).copy()
    prediction = prediction[prediction["label"] != -100].copy()
    prediction["label"] = prediction["label"].astype(int)
    prediction["predicted_class"] = prediction["predicted_class"].astype(int)
    prediction["genomic_position"] = prediction["genomic_position"].astype(int)
    prediction["base_position_in_window"] = prediction["base_position_in_window"].astype(int)
    prediction["chrom"] = prediction["window_id"].astype(str).str.split(":", n=1).str[0]
    return prediction


def load_comparison_dicts(comparison_dicts_dir, sample):
    path = comparison_dicts_dir / f"{sample}_comparison_dicts.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison dicts for {sample}: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["compare_dicts"]


def create_position_mean_lookup(compare_dicts):
    lookup = {}
    for chrom, positions in compare_dicts.items():
        position_means = {}
        for position, values_by_key in positions.items():
            methyl_values = [
                value
                for key, value in values_by_key.items()
                if "methyl_rate_ind" in key
            ]
            position_means[position] = np.mean(methyl_values) if methyl_values else np.nan
        lookup[chrom] = pd.Series(position_means, dtype="float64")
    return lookup


def create_mean_values(result_file, position_mean_lookup):
    mean_values = pd.Series(np.nan, index=result_file.index, dtype="float64")
    for chrom, chrom_rows in result_file.groupby("chrom", sort=False):
        if chrom not in position_mean_lookup:
            continue
        chrom_position_means = position_mean_lookup[chrom]
        summed_values = pd.Series(0.0, index=chrom_rows.index)
        matched_counts = pd.Series(0, index=chrom_rows.index)
        has_nan_value = pd.Series(False, index=chrom_rows.index)
        genomic_positions = chrom_rows["genomic_position"]

        for offset in range(-1, 6):
            candidate_positions = genomic_positions + offset
            matched_positions = candidate_positions.isin(chrom_position_means.index)
            candidate_means = candidate_positions.map(chrom_position_means)
            summed_values += candidate_means.where(matched_positions, 0).fillna(0)
            matched_counts += matched_positions.astype(int)
            has_nan_value |= matched_positions & candidate_means.isna()

        valid_values = (matched_counts > 0) & ~has_nan_value
        mean_values.loc[chrom_rows.index[valid_values]] = (
            summed_values.loc[valid_values] / matched_counts.loc[valid_values]
        )
    return mean_values


def create_mean_labels(mean_values, ranges):
    values = pd.Series(mean_values)
    ranges_array = np.asarray(ranges)
    value_array = values.to_numpy()
    labels = np.searchsorted(ranges_array[1:], value_array, side="left")
    in_range = (value_array >= ranges_array[0]) & (value_array <= ranges_array[-1])
    valid_values = values.notna().to_numpy() & in_range
    result = pd.Series(np.nan, index=values.index)
    result.iloc[valid_values] = labels[valid_values]
    return result


def add_mean_label(prediction, compare_dicts, ranges):
    annotated = prediction.copy()
    lookup = create_position_mean_lookup(compare_dicts)
    annotated["mean_value"] = create_mean_values(annotated, lookup)
    annotated["mean_label"] = create_mean_labels(annotated["mean_value"], ranges)
    annotated = annotated.dropna(subset=["mean_label"]).copy()
    annotated["mean_label"] = annotated["mean_label"].astype(int)
    return annotated


def add_context_factors(prediction, gene_index, cpg_island_index):
    annotated = prediction.copy()
    cpg_starts = annotated["genomic_position"].to_numpy(dtype=np.int64)
    cpg_ends = cpg_starts + 2
    annotated["gene_overlap"] = np.where(
        interval_overlap_flags(annotated["chrom"], cpg_starts, cpg_ends, gene_index),
        "in_gene",
        "not_in_gene",
    )
    annotated["cpg_island_overlap"] = np.where(
        interval_overlap_flags(annotated["chrom"], cpg_starts, cpg_ends, cpg_island_index),
        "in_cpg_island",
        "not_in_cpg_island",
    )
    annotated["window_position"] = np.where(
        (annotated["base_position_in_window"] < 1080)
        | (annotated["base_position_in_window"] >= 4320),
        "edge",
        "middle",
    )
    cpg_counts = annotated.groupby("window_id")["window_id"].transform("size")
    annotated["window_cpg_count"] = cpg_counts.astype(int)
    annotated["window_cpg_count_bin"] = pd.cut(
        annotated["window_cpg_count"],
        bins=[0, 6, 12, np.inf],
        labels=["low", "medium", "high"],
        right=True,
    ).astype(str)
    return annotated


def f1_score(precision, recall):
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def metrics_for_rows(rows, prediction_column="predicted_class", labels=DEFAULT_LABELS):
    output_rows = []
    true_labels = rows["label"]
    predicted_labels = rows[prediction_column]
    class_rows = []
    for label in labels:
        specific_true = true_labels == label
        specific_predicted = predicted_labels == label
        precision = precision_score(specific_true, specific_predicted, zero_division=0)
        recall = recall_score(specific_true, specific_predicted, zero_division=0)
        support = int(specific_true.sum())
        class_row = {
            "class": str(label),
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
            "support": support,
        }
        class_rows.append(class_row)
        output_rows.append(class_row)

    total_support = sum(row["support"] for row in class_rows)
    if total_support:
        output_rows.append(
            {
                "class": "weighted",
                "precision": sum(row["precision"] * row["support"] for row in class_rows) / total_support,
                "recall": sum(row["recall"] * row["support"] for row in class_rows) / total_support,
                "f1": sum(row["f1"] * row["support"] for row in class_rows) / total_support,
                "support": total_support,
            }
        )
    return output_rows


def prediction_columns_for_metrics(prediction):
    columns = [("model_prediction", "predicted_class")]
    if "mean_label" in prediction.columns:
        columns.append(("mean_prediction", "mean_label"))
    return columns


def per_file_metric_rows(prediction, metadata):
    factors = {
        "gene_overlap": "gene_overlap",
        "cpg_island_overlap": "cpg_island_overlap",
        "window_position": "window_position",
        "window_cpg_count": "window_cpg_count_bin",
    }
    rows = []
    for factor, column in factors.items():
        for factor_value, factor_rows in prediction.groupby(column, sort=True):
            if factor_rows.empty:
                continue
            for prediction_type, prediction_column in prediction_columns_for_metrics(factor_rows):
                for metrics in metrics_for_rows(factor_rows, prediction_column=prediction_column):
                    rows.append(
                        {
                            **metadata,
                            "prediction_type": prediction_type,
                            "factor": factor,
                            "factor_value": str(factor_value),
                            "n_windows": int(factor_rows["window_id"].nunique()),
                            "avg_window_cpg_count": float(factor_rows["window_cpg_count"].mean()),
                            **metrics,
                        }
                    )
    return rows


def aggregate_metrics(per_file_metrics):
    sample_level = (
        per_file_metrics.groupby(
            ["tissue", "group", "sample", "prediction_type", "factor", "factor_value", "class"],
            as_index=False,
        )
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            support=("support", "mean"),
            n_windows=("n_windows", "mean"),
            avg_window_cpg_count=("avg_window_cpg_count", "mean"),
            prediction_file_count=("prediction_path", "nunique"),
        )
    )
    return (
        sample_level.groupby(["tissue", "group", "prediction_type", "factor", "factor_value", "class"], as_index=False)
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            support_mean=("support", "mean"),
            n_windows_mean=("n_windows", "mean"),
            avg_window_cpg_count=("avg_window_cpg_count", "mean"),
            n_samples=("sample", "nunique"),
            prediction_file_count=("prediction_file_count", "sum"),
        )
    )


def group_label(group):
    if group == "no_pretraining":
        return "No pretrain"
    match = re.fullmatch(r"first_phase_epoch_(\d+)", group)
    if match:
        return f"Pretrain E{int(match.group(1))}"
    match = re.fullmatch(r"epoch_(\d+)_pretraining", group)
    if match:
        return f"Retrain E{int(match.group(1))}"
    return group


def comparison_label(row):
    if row.get("prediction_type") == "mean_prediction":
        return f"{group_label(row['group'])} mean"
    return f"{group_label(row['group'])} model"


def plot_factor(summary, factor, output_path, dpi):
    data = summary[(summary["factor"] == factor) & (summary["class"] == "weighted")].copy()
    if data.empty:
        return
    if "prediction_type" not in data.columns:
        data["prediction_type"] = "model_prediction"
    data["group_label"] = data.apply(comparison_label, axis=1)
    pivot = data.pivot_table(index="factor_value", columns="group_label", values="f1", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(max(10, len(pivot) * 2.2), 5))
    ax.set_title(f"{factor}: weighted F1 by group and prediction type")
    ax.set_xlabel(factor)
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_combined(summary, output_path, dpi):
    data = summary[summary["class"] == "weighted"].copy()
    if data.empty:
        return
    if "prediction_type" not in data.columns:
        data["prediction_type"] = "model_prediction"
    data["label"] = data["factor"] + "\n" + data["factor_value"]
    data["group_label"] = data.apply(comparison_label, axis=1)
    pivot = data.pivot_table(index="label", columns="group_label", values="f1", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(max(12, len(pivot) * 0.8), 6))
    ax.set_title("Genomic context factors: weighted F1")
    ax.set_xlabel("Factor group")
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def run(args):
    groups = parse_groups(args)
    if args.include_mean_label and args.comparison_dicts_dir is None:
        raise ValueError("--comparison-dicts-dir is required with --include-mean-label")
    ranges = parse_ranges(args.ranges)
    comparison_dicts_cache = {}
    result_rows = discover_result_files(args.results_dir, groups, args.batch_size)
    if args.final_only:
        result_rows = keep_final_files(result_rows)
    if args.max_files is not None:
        result_rows = result_rows[: args.max_files]
    if not result_rows:
        raise ValueError("No prediction files found for the requested inputs.")

    gene_index = build_interval_index(read_bed(args.gene_bed))
    cpg_island_index = build_interval_index(read_bed(args.cpg_island_bed))

    all_rows = []
    for row in result_rows:
        print(f"Evaluating {row['group']} {row['sample']}: {row['path']}", flush=True)
        prediction = load_prediction_dataframe(row["path"])
        if args.include_mean_label:
            if row["sample"] not in comparison_dicts_cache:
                comparison_dicts_cache[row["sample"]] = load_comparison_dicts(
                    args.comparison_dicts_dir,
                    row["sample"],
                )
            prediction = add_mean_label(prediction, comparison_dicts_cache[row["sample"]], ranges)
        annotated = add_context_factors(prediction, gene_index, cpg_island_index)
        all_rows.extend(
            per_file_metric_rows(
                annotated,
                {
                    "tissue": args.tissue_name,
                    "group": row["group"],
                    "sample": row["sample"],
                    "prediction_path": str(row["path"]),
                    "epoch": row["epoch"],
                },
            )
        )

    per_file = pd.DataFrame(all_rows)
    summary = aggregate_metrics(per_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_file.to_csv(args.output_dir / "context_factor_per_file_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "context_factor_metrics.csv", index=False)

    for factor in sorted(summary["factor"].unique()):
        plot_factor(summary, factor, args.output_dir / f"{factor}_weighted_f1.png", args.dpi)
    plot_combined(summary, args.output_dir / "context_factors_weighted_f1.png", args.dpi)
    print(f"Wrote {args.output_dir}", flush=True)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
