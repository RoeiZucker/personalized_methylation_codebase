#!/usr/bin/env python3
# Create bin-rank-corrected PDF reports for the standard downsampled runs.
# Example:
# python3 scripts/create_downsampled_standard_report.py

import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "scripts/results/downsampled_standard_ratio_0p25_lr_3e-05"
OUT = ROOT / "report"
LABELS = [0, 1, 2]
EXCLUDED_SAMPLES = set()
SAMPLES = {}
SECTIONS = ["all_results", "bin_1", "bin_2", "bin_3", "bin_4", "bin_5"]
WEIGHTED_METRICS = ["weighted_f1", "weighted_precision", "weighted_recall"]
MACRO_METRICS = ["macro_f1", "macro_precision", "macro_recall"]
CLASS_METRICS = [
    f"class_{label}_{metric}"
    for label in LABELS
    for metric in ["f1", "precision", "recall"]
]
MAIN_METRICS = WEIGHTED_METRICS + CLASS_METRICS
METRICS = MAIN_METRICS + MACRO_METRICS


def read_eval(path):
    return ast.literal_eval(path.read_text(encoding="utf-8"))


def discover_samples():
    samples = {}
    for sample_dir in sorted(ROOT.iterdir()):
        if not sample_dir.is_dir() or sample_dir.name in EXCLUDED_SAMPLES:
            continue
        eval_file = sample_dir / "eval_eval_objects_dict.txt"
        test_file = sample_dir / "test_eval_objects_dict.txt"
        if not eval_file.exists() or not test_file.exists():
            continue
        sample = sample_dir.name
        payload = read_eval(eval_file)
        if sample not in payload:
            continue
        tissue = infer_tissue(payload[sample])
        samples[sample] = tissue
    return samples


def infer_tissue(sample_results):
    for prediction_path in sample_results:
        match = re.search(r"_token_cls_(.+?)_kmer/", prediction_path)
        if match:
            return match.group(1)
    return "Unknown"


def stage_from_path(path):
    parent = Path(path).parent.name
    if parent.startswith("pretrain"):
        return "pretrain"
    checkpoint = re.match(r"checkpoint-(\d+)", parent)
    if checkpoint:
        return "checkpoint-" + checkpoint.group(1)
    epoch = re.match(r"epoch-(\d+)-step-", parent)
    if epoch:
        return "E" + epoch.group(1)
    return parent


def stage_number(stage):
    if stage.startswith("checkpoint-"):
        return int(stage.split("-", 1)[1])
    if stage.startswith("E"):
        return int(stage[1:])
    return 0


def stage_map(paths):
    stages = sorted((stage_from_path(path) for path in paths if stage_from_path(path) != "pretrain"), key=stage_number)
    mapping = {"pretrain": "pretrain"}
    for index, stage in enumerate(stages, start=1):
        mapping[stage] = f"E{index}"
    return mapping


def raw_section_key(section):
    if section == "all_results":
        return (-1, 0.0)
    return (0, float(section.split("-", 1)[0]))


def section_map(prediction_eval):
    mapping = {}
    bin_index = 0
    for section in sorted(prediction_eval, key=raw_section_key):
        if section == "all_results":
            mapping[section] = "all_results"
        else:
            bin_index += 1
            mapping[section] = f"bin_{bin_index}"
    return mapping


def confusion_array(confusion):
    matrix = np.zeros((3, 3), dtype=float)
    for predicted, true_counts in confusion.items():
        for true, count in true_counts.items():
            predicted = int(predicted)
            true = int(true)
            if predicted in LABELS and true in LABELS:
                matrix[true, predicted] = float(count)
    return matrix


def summarize(section_object, prediction_type):
    matrix = confusion_array(section_object[f"{prediction_type}_confusion_matrix"])
    support = matrix.sum(axis=1)
    total = support.sum()
    output = {"support": int(total)}
    values_by_metric = {"precision": [], "recall": [], "f1": []}
    for label in LABELS:
        precision = float(section_object[prediction_type][label]["Precision"])
        recall = float(section_object[prediction_type][label]["recall"])
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        values_by_metric["precision"].append(precision)
        values_by_metric["recall"].append(recall)
        values_by_metric["f1"].append(f1)
        output[f"class_{label}_precision"] = precision
        output[f"class_{label}_recall"] = recall
        output[f"class_{label}_f1"] = f1
    for metric, values in values_by_metric.items():
        output[f"weighted_{metric}"] = float(np.average(values, weights=support)) if total else np.nan
        output[f"macro_{metric}"] = float(np.mean(values))
    return output


def prediction_rows(sample, tissue, split, prediction_path, prediction_eval, stages):
    rows = []
    sections = section_map(prediction_eval)
    stage = stages[stage_from_path(prediction_path)]
    for raw_section, section_object in prediction_eval.items():
        for prediction_type in ["predicted_class", "mean_label"]:
            rows.append(
                {
                    "sample": sample,
                    "tissue": tissue,
                    "split": split,
                    "stage": stage,
                    "prediction_type": prediction_type,
                    "section_rank": sections[raw_section],
                    "prediction_path": prediction_path,
                }
                | summarize(section_object, prediction_type)
            )
    return rows


def collect_rows():
    rows = []
    for sample, tissue in SAMPLES.items():
        for split in ["eval", "test"]:
            sample_results = read_eval(ROOT / sample / f"{split}_eval_objects_dict.txt")[sample]
            stages = stage_map(sample_results)
            for prediction_path, prediction_eval in sample_results.items():
                rows.extend(prediction_rows(sample, tissue, split, prediction_path, prediction_eval, stages))
    return pd.DataFrame(rows)


def aggregate(rows, group_cols=None):
    value_cols = [col for col in rows if col.startswith(("weighted_", "macro_", "class_"))]
    if group_cols is None:
        group_cols = ["split", "stage", "prediction_type", "section_rank"]
    grouped = rows.groupby(group_cols, as_index=False)
    averages = grouped[value_cols].mean()
    stds = grouped[value_cols].std(ddof=1).rename(columns={col: f"{col}_std" for col in value_cols})
    support = grouped["support"].mean().rename(columns={"support": "avg_support"})
    return averages.merge(stds, on=group_cols).merge(support, on=group_cols)


def individual_rows(rows, sample):
    sample_rows = rows[rows["sample"] == sample].copy()
    return sample_rows.rename(columns={"support": "avg_support"})


def axis_limits(values):
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return 0, 1
    low = max(0, min(values) - 0.07)
    high = min(1.05, max(values) + 0.09)
    if high - low < 0.12:
        mid = (low + high) / 2
        return max(0, mid - 0.06), min(1.05, mid + 0.06)
    return low, high


def section_title(section, support):
    label = "All sites" if section == "all_results" else "Bin " + section.split("_")[1]
    return f"{label} | avg n={support:,.0f}"


def sorted_retrain(predicted):
    retrain = predicted[predicted["stage"].isin(["E1", "E2", "E3"])].copy()
    retrain["stage"] = pd.Categorical(retrain["stage"], ["E1", "E2", "E3"], ordered=True)
    return retrain.sort_values("stage")


def add_baseline(axis, rows, metric, color, label, values, x_position):
    if rows.empty:
        return
    baseline_rows = rows[rows["stage"] == "pretrain"] if label.startswith("Mean") else rows
    selected = baseline_rows if not baseline_rows.empty else rows
    value = float(selected[metric].iloc[0])
    std = baseline_std(selected, metric)
    values.append(value)
    axis.axhline(value, color=color, linestyle=":", linewidth=3, label=label)
    add_baseline_whisker(axis, x_position, value, std, color, values)
    va = "bottom" if color == "black" else "top"
    axis.text(0.99, value, f" {label.lower()} {value:.3f}", transform=axis.get_yaxis_transform(), ha="right", va=va, fontsize=8, color=color)


def baseline_std(rows, metric):
    std_col = f"{metric}_std"
    if std_col not in rows.columns:
        return None
    std = rows[std_col].iloc[0]
    if not np.isfinite(std):
        return None
    return float(std)


def add_baseline_whisker(axis, x_position, value, std, color, values):
    if std is None or std <= 0:
        return
    values.extend([value - std, value + std])
    axis.errorbar(
        [x_position],
        [value],
        yerr=[[std], [std]],
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=2,
        capsize=5,
        markersize=4,
        zorder=5,
    )


def draw_section(axis, averages, split, metric, section):
    subset = averages[(averages["split"] == split) & (averages["section_rank"] == section)]
    predicted = subset[subset["prediction_type"] == "predicted_class"]
    mean_rows = subset[subset["prediction_type"] == "mean_label"]
    retrain = sorted_retrain(predicted)
    yerr = None
    std_col = f"{metric}_std"
    if std_col in retrain.columns:
        yerr = retrain[std_col].fillna(0).to_numpy()
    bars = axis.bar(
        np.arange(len(retrain)),
        retrain[metric],
        yerr=yerr,
        capsize=5 if yerr is not None else 0,
        ecolor="#222222",
        color="#4C78A8",
        width=0.55,
        label="Retrain prediction",
    )
    values = list(retrain[metric].dropna())
    if yerr is not None:
        values.extend((retrain[metric] + yerr).dropna())
        values.extend((retrain[metric] - yerr).dropna())
    for bar in bars:
        height = bar.get_height()
        axis.text(bar.get_x() + bar.get_width() / 2, height + 0.006, f"{height:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
    add_baseline(axis, mean_rows, metric, "black", "Mean label", values, len(retrain) - 0.12)
    add_baseline(axis, predicted[predicted["stage"] == "pretrain"], metric, "#D62728", "Pretrain E10", values, len(retrain) - 0.32)
    support = float(retrain["avg_support"].mean()) if not retrain.empty else np.nan
    axis.set_title(section_title(section, support), loc="left", fontsize=10)
    axis.set_ylim(*axis_limits(values))
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best", fontsize=8)


def draw_page(pdf, averages, split, metric, title):
    figure, axes = plt.subplots(len(SECTIONS), 1, figsize=(11, 2.5 * len(SECTIONS)), sharex=True)
    for axis, section in zip(axes, SECTIONS):
        draw_section(axis, averages, split, metric, section)
    axes[-1].set_xticks(np.arange(3))
    axes[-1].set_xticklabels(["Retrain E1", "Retrain E2", "Retrain E3"])
    figure.suptitle(f"{title} ({split}): {metric.replace('_', ' ').title()}\nBins grouped by rank", y=0.997, fontsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.965])
    pdf.savefig(figure)
    plt.close(figure)


def draw_title(pdf):
    figure, axis = plt.subplots(figsize=(11, 8.5))
    axis.axis("off")
    lines = [
        f"Downsampled ratio=0.25, lr=3e-5 report ({len(SAMPLES)} samples)",
        "Bin-rank corrected",
        "",
        "Included samples:",
        *[f"  {sample}: {tissue}" for sample, tissue in SAMPLES.items()],
        "",
        "Each sample contributes pretrain_epoch10 and retrain E1-E3 for eval and test.",
        "Averages use equal sample weight.",
        "A second PDF contains non-averaged pages for each sample.",
        "Split-specific PDFs are also created for eval-only and test-only views.",
        "Class metrics are per-class, not weighted by other classes.",
        "Variability bins are grouped by bin order, not exact numeric boundaries.",
        "Dotted black line: mean-label baseline. Dotted red line: pretrain-E10 baseline.",
    ]
    axis.text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=14)
    pdf.savefig(figure)
    plt.close(figure)


def main():
    global SAMPLES
    OUT.mkdir(parents=True, exist_ok=True)
    SAMPLES = discover_samples()
    rows = collect_rows()
    sample_averages = aggregate(rows)
    tissue_averages = aggregate(rows, ["tissue", "split", "stage", "prediction_type", "section_rank"])
    rows.to_csv(OUT / "downsampled_standard_all_samples_metric_rows_binrank_corrected.csv", index=False)
    sample_averages.to_csv(OUT / "downsampled_standard_all_samples_average_metrics_binrank_corrected.csv", index=False)
    tissue_averages.to_csv(OUT / "downsampled_standard_tissue_average_metrics_binrank_corrected.csv", index=False)
    paths = report_paths()
    write_metric_pdf(paths["average_main"], sample_averages, MAIN_METRICS, f"Average across {len(SAMPLES)} samples")
    write_metric_pdf(paths["average_macro"], sample_averages, MACRO_METRICS, f"Average across {len(SAMPLES)} samples")
    write_tissue_pdf(paths["tissue_main"], tissue_averages, MAIN_METRICS)
    write_tissue_pdf(paths["tissue_macro"], tissue_averages, MACRO_METRICS)
    write_individual_pdf(paths["individual"], rows)
    write_split_pdfs(paths, sample_averages, tissue_averages, rows)
    for path in paths.values():
        print(path)


def report_paths():
    paths = {
        "average_main": OUT / "downsampled_standard_all_samples_average_main_report_binrank_corrected.pdf",
        "average_macro": OUT / "downsampled_standard_all_samples_average_macro_report_binrank_corrected.pdf",
        "tissue_main": OUT / "downsampled_standard_tissue_average_main_report_binrank_corrected.pdf",
        "tissue_macro": OUT / "downsampled_standard_tissue_average_macro_report_binrank_corrected.pdf",
        "individual": OUT / "downsampled_standard_all_samples_individual_report_binrank_corrected.pdf",
    }
    for split in ["eval", "test"]:
        paths[f"{split}_average_main"] = OUT / f"downsampled_standard_{split}_all_samples_average_main_report_binrank_corrected.pdf"
        paths[f"{split}_average_macro"] = OUT / f"downsampled_standard_{split}_all_samples_average_macro_report_binrank_corrected.pdf"
        paths[f"{split}_tissue_main"] = OUT / f"downsampled_standard_{split}_tissue_average_main_report_binrank_corrected.pdf"
        paths[f"{split}_tissue_macro"] = OUT / f"downsampled_standard_{split}_tissue_average_macro_report_binrank_corrected.pdf"
        paths[f"{split}_individual"] = OUT / f"downsampled_standard_{split}_all_samples_individual_report_binrank_corrected.pdf"
    return paths


def write_split_pdfs(paths, sample_averages, tissue_averages, rows):
    for split in ["eval", "test"]:
        split_sample_averages = sample_averages[sample_averages["split"] == split]
        split_tissue_averages = tissue_averages[tissue_averages["split"] == split]
        split_rows = rows[rows["split"] == split]
        title = f"{split} average across {len(SAMPLES)} samples"
        write_metric_pdf(paths[f"{split}_average_main"], split_sample_averages, MAIN_METRICS, title, [split])
        write_metric_pdf(paths[f"{split}_average_macro"], split_sample_averages, MACRO_METRICS, title, [split])
        write_tissue_pdf(paths[f"{split}_tissue_main"], split_tissue_averages, MAIN_METRICS, [split])
        write_tissue_pdf(paths[f"{split}_tissue_macro"], split_tissue_averages, MACRO_METRICS, [split])
        write_individual_pdf(paths[f"{split}_individual"], split_rows, [split])


def write_metric_pdf(path, averages, metrics, title, splits=None):
    if splits is None:
        splits = ["eval", "test"]
    with PdfPages(path) as pdf:
        draw_title(pdf)
        for split in splits:
            for metric in metrics:
                draw_page(pdf, averages, split, metric, title)


def write_tissue_pdf(path, tissue_averages, metrics, splits=None):
    if splits is None:
        splits = ["eval", "test"]
    with PdfPages(path) as pdf:
        draw_title(pdf)
        for tissue in sorted(tissue_averages["tissue"].unique()):
            tissue_rows = tissue_averages[tissue_averages["tissue"] == tissue]
            sample_count = sum(1 for sample_tissue in SAMPLES.values() if sample_tissue == tissue)
            title = f"{tissue} average across {sample_count} samples"
            for split in splits:
                for metric in metrics:
                    draw_page(pdf, tissue_rows, split, metric, title)


def write_individual_pdf(path, rows, splits=None):
    if splits is None:
        splits = ["eval", "test"]
    with PdfPages(path) as pdf:
        draw_title(pdf)
        for sample, tissue in SAMPLES.items():
            sample_data = individual_rows(rows, sample)
            if sample_data.empty:
                continue
            for split in splits:
                for metric in METRICS:
                    draw_page(pdf, sample_data, split, metric, f"{sample} ({tissue})")


if __name__ == "__main__":
    main()
