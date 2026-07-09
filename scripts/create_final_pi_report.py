#!/usr/bin/env python3
"""Create a PI-facing report for variability-free and variability-bin evaluations."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from xml.sax.saxutils import escape
import ast
import math
import re
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from PIL import Image

PROJECT = Path("/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code")
DEFAULT_OUTPUT_DIR = PROJECT / "scripts/results/final_pi_report"
VARFREE_ROOT = PROJECT / "scripts/results/all_tissues_grouped_evaluations"
VARBINS_ROOT = PROJECT / "scripts/results/all_tissues_variability_bins"
EPOCH_RE = re.compile(r"/epoch-(\d+)-step-\d+/")
FIRST_PHASE_RE = re.compile(r"^first_phase_epoch_(\d+)$")
RETRAIN_RE = re.compile(r"^epoch_(\d+)_pretraining$")
BIN_RE = re.compile(r"^(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)-(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")
METRICS = ["f1", "precision", "recall"]
BIN_COLORS = {
    1: "#4C78A8",
    2: "#F58518",
    3: "#54A24B",
    4: "#B279A2",
    5: "#E45756",
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-finetune-epoch", type=int, default=10)
    parser.add_argument(
        "--require-full-finetune",
        action="store_true",
        help="Select best pretrain epoch only among groups whose retrain outputs include every fine-tune epoch up to --max-finetune-epoch.",
    )
    parser.add_argument(
        "--best-pretrain-epoch",
        type=int,
        default=None,
        help="Use one shared pretrain epoch for every tissue. If omitted, choose the best overall pretrain epoch with equal tissue weighting.",
    )
    parser.add_argument(
        "--skip-individual-pages",
        action="store_true",
        help="Do not add per-person variability-free and variability-bin progression pages.",
    )
    parser.add_argument(
        "--zoom-variability-free-y-axis",
        action="store_true",
        help="Use tight per-panel y-axis limits for variability-free progression plots.",
    )
    return parser.parse_args()


def group_epoch(group: str):
    match = FIRST_PHASE_RE.fullmatch(group)
    if match:
        return int(match.group(1))
    match = RETRAIN_RE.fullmatch(group)
    if match:
        return int(match.group(1))
    return None


def finetune_epoch(result_path: str):
    match = EPOCH_RE.search(result_path)
    return int(match.group(1)) if match else -1


def parse_bin_key(key):
    match = BIN_RE.fullmatch(str(key).strip())
    if match is None:
        return None
    return tuple(map(float, match.groups()))


def value_for_class(mapping, class_id):
    return next((value for key, value in mapping.items() if str(key) == str(class_id)), 0)


def class_support(confusion, class_id):
    return sum(float(value_for_class(predicted_counts, class_id)) for predicted_counts in confusion.values())


def f1_score(precision, recall):
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def metric_rows(eval_object, method):
    if method not in eval_object or f"{method}_confusion_matrix" not in eval_object:
        return []
    confusion = eval_object[f"{method}_confusion_matrix"]
    rows = []
    total_support = 0.0
    totals = {metric: 0.0 for metric in METRICS}
    for raw_class, values in eval_object[method].items():
        precision = float(values["Precision"])
        recall = float(values["recall"])
        f1 = f1_score(precision, recall)
        support = class_support(confusion, raw_class)
        total_support += support
        totals["precision"] += precision * support
        totals["recall"] += recall * support
        totals["f1"] += f1 * support
        rows.append({
            "class": str(raw_class),
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    if total_support:
        rows.append({
            "class": "weighted",
            "support": total_support,
            **{metric: totals[metric] / total_support for metric in METRICS},
        })
    return rows


def read_eval_file(path):
    return ast.literal_eval(path.read_text(encoding="utf-8"))


def eval_path(root, tissue, mode_dir, group):
    return root / tissue / mode_dir / group / "eval_objects_dict.txt"


def load_variability_free_rows(tissues):
    rows = []
    for tissue in tissues:
        tissue_root = VARFREE_ROOT / tissue / "variability_free"
        if not tissue_root.exists():
            continue
        for group_dir in sorted(tissue_root.iterdir()):
            group = group_dir.name
            pretraining_epoch = group_epoch(group)
            if pretraining_epoch is None:
                if group != "no_pretraining":
                    continue
                pretraining_epoch = 0
                kind = "no_pretraining"
            else:
                kind = "pretrain" if group.startswith("first_phase") else "retrain"
            path = group_dir / "eval_objects_dict.txt"
            if not path.exists() or path.stat().st_size == 0:
                continue
            for person, results in read_eval_file(path).items():
                for result_path, result_by_section in results.items():
                    if "_low_var" in str(result_path):
                        continue
                    if "all_results" not in result_by_section:
                        continue
                    for method in ["predicted_class", "mean_label"]:
                        for values in metric_rows(result_by_section["all_results"], method):
                            rows.append({
                                "tissue": tissue,
                                "person": person,
                                "pretraining_epoch": pretraining_epoch,
                                "kind": kind,
                                "finetune_epoch": finetune_epoch(result_path),
                                "method": method,
                                **values,
                            })
    return pd.DataFrame(rows)


def load_variability_bin_rows(tissues):
    rows = []
    for tissue in tissues:
        tissue_root = VARBINS_ROOT / tissue / "variability_bins"
        if not tissue_root.exists():
            continue
        for group_dir in sorted(tissue_root.iterdir()):
            group = group_dir.name
            pretraining_epoch = group_epoch(group)
            if pretraining_epoch is None:
                if group != "no_pretraining":
                    continue
                pretraining_epoch = 0
                kind = "no_pretraining"
            else:
                kind = "pretrain" if group.startswith("first_phase") else "retrain"
            path = group_dir / "eval_objects_dict.txt"
            if not path.exists() or path.stat().st_size == 0:
                continue
            for person, results in read_eval_file(path).items():
                for result_path, result_by_bin in results.items():
                    if "_low_var" in str(result_path):
                        continue
                    bin_keys = [key for key in result_by_bin if parse_bin_key(key) is not None]
                    bin_keys = sorted(bin_keys, key=lambda key: parse_bin_key(key)[0])
                    for bin_rank, bin_key in enumerate(bin_keys, start=1):
                        for method in ["predicted_class", "mean_label"]:
                            for values in metric_rows(result_by_bin[bin_key], method):
                                rows.append({
                                    "tissue": tissue,
                                    "person": person,
                                    "pretraining_epoch": pretraining_epoch,
                                    "kind": kind,
                                    "finetune_epoch": finetune_epoch(result_path),
                                    "bin_rank": bin_rank,
                                    "bin_label": str(bin_key),
                                    "method": method,
                                    **values,
                                })
    return pd.DataFrame(rows)


def available_tissues():
    tissues = []
    for tissue_dir in sorted(VARFREE_ROOT.iterdir()):
        if not tissue_dir.is_dir() or tissue_dir.name.startswith("combined_as_single"):
            continue
        vf = tissue_dir / "variability_free"
        has_pretrain = any((vf / f"first_phase_epoch_{epoch}" / "eval_objects_dict.txt").exists() for epoch in range(1, 20))
        has_retrain = any((vf / f"epoch_{epoch}_pretraining" / "eval_objects_dict.txt").exists() for epoch in range(1, 20))
        if has_pretrain and has_retrain:
            tissues.append(tissue_dir.name)
    return tissues


def full_finetune_pairs(varfree_rows, max_finetune_epoch):
    retrain = varfree_rows[
        (varfree_rows["kind"] == "retrain")
        & (varfree_rows["method"] == "predicted_class")
        & (varfree_rows["class"] == "weighted")
    ].copy()
    required = set(range(1, max_finetune_epoch + 1))
    complete = []
    for (tissue, pretraining_epoch), group in retrain.groupby(["tissue", "pretraining_epoch"]):
        observed = set(int(epoch) for epoch in group["finetune_epoch"].dropna().unique())
        if required.issubset(observed):
            complete.append((tissue, int(pretraining_epoch)))
    return set(complete)


def pretrain_epoch_scores(varfree_rows, require_full_finetune=False, max_finetune_epoch=10):
    pretrain = varfree_rows[
        (varfree_rows["kind"] == "pretrain")
        & (varfree_rows["method"] == "predicted_class")
        & (varfree_rows["class"] == "weighted")
    ].copy()
    if require_full_finetune:
        complete_pairs = full_finetune_pairs(varfree_rows, max_finetune_epoch)
        pretrain = pretrain[
            pretrain.apply(
                lambda row: (row["tissue"], int(row["pretraining_epoch"])) in complete_pairs,
                axis=1,
            )
        ].copy()
    person = pretrain.groupby(["tissue", "person", "pretraining_epoch"], as_index=False)["f1"].mean()
    tissue_epoch = person.groupby(["tissue", "pretraining_epoch"], as_index=False)["f1"].mean()
    overall_epoch = tissue_epoch.groupby("pretraining_epoch", as_index=False)["f1"].mean()
    return tissue_epoch.rename(columns={"f1": "tissue_pretrain_weighted_f1"}), overall_epoch.rename(columns={"f1": "overall_pretrain_weighted_f1"})


def best_pretrain_epochs(varfree_rows, require_full_finetune=False, max_finetune_epoch=10, selected_epoch=None):
    tissue_epoch, overall_epoch = pretrain_epoch_scores(varfree_rows, require_full_finetune, max_finetune_epoch)
    if selected_epoch is None:
        selected = overall_epoch.sort_values(["overall_pretrain_weighted_f1", "pretraining_epoch"], ascending=[False, True]).iloc[0]
        selected_epoch = int(selected["pretraining_epoch"])
    selected_scores = tissue_epoch[tissue_epoch["pretraining_epoch"] == selected_epoch].copy()
    overall_score = overall_epoch[overall_epoch["pretraining_epoch"] == selected_epoch]["overall_pretrain_weighted_f1"].iloc[0]
    selected_scores["best_pretrain_weighted_f1"] = selected_scores["tissue_pretrain_weighted_f1"]
    selected_scores["overall_pretrain_weighted_f1"] = overall_score
    return selected_scores[["tissue", "pretraining_epoch", "best_pretrain_weighted_f1", "overall_pretrain_weighted_f1"]]


def combine_progression_rows(rows, best_epochs, bins=None, max_finetune_epoch=10, class_id="weighted"):
    frames = []
    selected_tissues = set(best_epochs["tissue"])
    for best in best_epochs.itertuples(index=False):
        tissue_rows = rows[
            (rows["tissue"] == best.tissue)
            & (rows["pretraining_epoch"] == int(best.pretraining_epoch))
            & (rows["method"] == "predicted_class")
            & (rows["class"].astype(str) == str(class_id))
        ].copy()
        if bins is not None:
            tissue_rows = tissue_rows[tissue_rows["bin_rank"].isin(bins)]
        frames.append(tissue_rows)

    no_pretraining = rows[
        (rows["tissue"].isin(selected_tissues))
        & (rows["kind"] == "no_pretraining")
        & (rows["method"] == "predicted_class")
        & (rows["class"].astype(str) == str(class_id))
    ].copy()
    if bins is not None:
        no_pretraining = no_pretraining[no_pretraining["bin_rank"].isin(bins)]
    frames.append(no_pretraining)

    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    data = data[
        (data["kind"] == "pretrain")
        | (
            data["kind"].isin(["retrain", "no_pretraining"])
            & data["finetune_epoch"].between(1, max_finetune_epoch)
        )
    ]
    return data


def mean_label_rows(rows, best_epochs, bins=None, class_id="weighted"):
    frames = []
    for best in best_epochs.itertuples(index=False):
        tissue_rows = rows[
            (rows["tissue"] == best.tissue)
            & (rows["pretraining_epoch"] == int(best.pretraining_epoch))
            & (rows["method"] == "mean_label")
            & (rows["class"].astype(str) == str(class_id))
        ].copy()
        if bins is not None:
            tissue_rows = tissue_rows[tissue_rows["bin_rank"].isin(bins)]
        frames.append(tissue_rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_progression(data, mean_rows, tissue=None, person=None, bins=None):
    if tissue is not None:
        data = data[data["tissue"] == tissue]
        mean_rows = mean_rows[mean_rows["tissue"] == tissue]
    if person is not None:
        data = data[data["person"] == person]
        mean_rows = mean_rows[mean_rows["person"] == person]
    if bins is not None and "bin_rank" in data.columns:
        data = data[data["bin_rank"].isin(bins)]
        mean_rows = mean_rows[mean_rows["bin_rank"].isin(bins)]
    summaries = {}
    for metric in METRICS:
        pre = data[data["kind"] == "pretrain"].groupby(["tissue", "person"], as_index=False)[metric].mean()
        retrain_person = data[data["kind"] == "retrain"].groupby(["tissue", "person", "finetune_epoch"], as_index=False)[metric].mean()
        no_pretraining_person = data[data["kind"] == "no_pretraining"].groupby(["tissue", "person", "finetune_epoch"], as_index=False)[metric].mean()
        mean_person = mean_rows.groupby(["tissue", "person"], as_index=False)[metric].mean()
        if tissue is None:
            pre_tissue = pre.groupby("tissue", as_index=False)[metric].mean()
            pre_mean = pre_tissue[metric].mean() if not pre_tissue.empty else math.nan
            retrain_tissue = retrain_person.groupby(["tissue", "finetune_epoch"], as_index=False)[metric].mean()
            retrain = retrain_tissue.groupby("finetune_epoch")[metric].agg(["mean", "std"]).reset_index().fillna(0)
            no_pretraining_tissue = no_pretraining_person.groupby(["tissue", "finetune_epoch"], as_index=False)[metric].mean()
            no_pretraining = no_pretraining_tissue.groupby("finetune_epoch")[metric].agg(["mean", "std"]).reset_index().fillna(0)
            mean_tissue = mean_person.groupby("tissue", as_index=False)[metric].mean()
            mean_label = mean_tissue[metric].mean() if not mean_tissue.empty else math.nan
        else:
            pre_mean = pre[metric].mean() if not pre.empty else math.nan
            retrain = retrain_person.groupby("finetune_epoch")[metric].agg(["mean", "std"]).reset_index().fillna(0)
            no_pretraining = no_pretraining_person.groupby("finetune_epoch")[metric].agg(["mean", "std"]).reset_index().fillna(0)
            mean_label = mean_person[metric].mean() if not mean_person.empty else math.nan
        summaries[metric] = {"pretrain": pre_mean, "mean_label": mean_label, "retrain": retrain, "no_pretraining": no_pretraining}
    return summaries


def args_max_finetune_epoch_for_plot(retrain):
    if retrain.empty:
        return 10
    return int(retrain["finetune_epoch"].max())




def tissue_sample_counts(rows, tissues):
    counts = (
        rows[rows["tissue"].isin(tissues)]
        .groupby("tissue")["person"]
        .nunique()
        .to_dict()
    )
    return {tissue: int(counts.get(tissue, 0)) for tissue in tissues}


def combined_sample_count(counts):
    return sum(counts.values())


def zoomed_metric_ylim(values, min_span=0.025, pad_fraction=0.18):
    values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not values:
        return 0, 1.05
    low = min(values)
    high = max(values)
    span = max(high - low, min_span)
    center = (low + high) / 2
    low = center - span / 2
    high = center + span / 2
    pad = span * pad_fraction
    return max(0, low - pad), min(1.05, high + pad)


def progression_metric_values(item):
    values = []
    for key in ["pretrain", "mean_label"]:
        if key in item and not math.isnan(item[key]):
            values.append(item[key])
    for table_key in ["retrain", "no_pretraining"]:
        table = item.get(table_key)
        if table is not None and not table.empty:
            values.extend(table["mean"].tolist())
            if "std" in table.columns:
                values.extend((table["mean"] - table["std"]).tolist())
                values.extend((table["mean"] + table["std"]).tolist())
    return values


def plot_progression(summary, title, output_path, zoom_y_axis=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for axis, metric in zip(axes, METRICS):
        item = summary[metric]
        retrain = item["retrain"]
        no_pretraining = item["no_pretraining"]
        if not no_pretraining.empty:
            axis.errorbar(
                no_pretraining["finetune_epoch"],
                no_pretraining["mean"],
                yerr=no_pretraining["std"],
                marker="s",
                linestyle="-",
                color="#7F7F7F",
                alpha=0.85,
                label="No pretraining",
            )
        if not retrain.empty:
            axis.errorbar(retrain["finetune_epoch"], retrain["mean"], yerr=retrain["std"], marker="o", color="#54A24B", label="Retrain")
        axis.axhline(item["pretrain"], color="#F58518", linestyle="--", linewidth=2, label="Best pretrain")
        if not math.isnan(item["mean_label"]):
            side_x = args_max_finetune_epoch_for_plot(retrain) + 0.7
            axis.scatter(
                [side_x],
                [item["mean_label"]],
                marker="D",
                s=80,
                color="#4C78A8",
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
                label="Mean label",
            )
            axis.text(side_x + 0.08, item["mean_label"], f"{item['mean_label']:.3f}", va="center", fontsize=8)
        axis.set_title(metric.title())
        axis.set_xlabel("Fine-tune epoch")
        if zoom_y_axis:
            axis.set_ylim(*zoomed_metric_ylim(progression_metric_values(item)))
        else:
            axis.set_ylim(0, 1.05)
        axis.set_xlim(0.5, args_max_finetune_epoch_for_plot(retrain) + 1.4)
        axis.grid(alpha=0.3)
        axis.text(0.02, 0.04, f"pre={item['pretrain']:.3f}\nmean={item['mean_label']:.3f}", transform=axis.transAxes, fontsize=9, va="bottom")
    axes[0].set_ylabel("Weighted score")
    axes[-1].legend(loc="lower right")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def draw_progression_axis(axis, item, metric, ylabel=None, show_legend=False, zoom_y_axis=False):
    retrain = item["retrain"]
    no_pretraining = item["no_pretraining"]
    max_epoch = args_max_finetune_epoch_for_plot(retrain)
    if not no_pretraining.empty:
        axis.errorbar(
            no_pretraining["finetune_epoch"],
            no_pretraining["mean"],
            yerr=no_pretraining["std"],
            marker="s",
            linestyle="-",
            color="#7F7F7F",
            alpha=0.85,
            label="No pretraining",
        )
    if not retrain.empty:
        axis.errorbar(
            retrain["finetune_epoch"],
            retrain["mean"],
            yerr=retrain["std"],
            marker="o",
            color="#54A24B",
            label="Retrain",
        )
    if not math.isnan(item["pretrain"]):
        axis.axhline(item["pretrain"], color="#F58518", linestyle="--", linewidth=1.8, label="Best pretrain")
    if not math.isnan(item["mean_label"]):
        side_x = max_epoch + 0.7
        axis.scatter([side_x], [item["mean_label"]], marker="D", s=55, color="#4C78A8", edgecolor="black", linewidth=0.6, zorder=5, label="Mean label")
        axis.text(side_x + 0.05, item["mean_label"], f"{item['mean_label']:.2f}", va="center", fontsize=7)
    axis.set_title(metric.title())
    if zoom_y_axis:
        axis.set_ylim(*zoomed_metric_ylim(progression_metric_values(item)))
    else:
        axis.set_ylim(0, 1.05)
    axis.set_xlim(0.5, max_epoch + 1.4)
    axis.grid(alpha=0.3)
    if ylabel:
        axis.set_ylabel(ylabel)
    if show_legend:
        axis.legend(loc="lower right", fontsize=8)


def plot_class_progression_grid(rows, mean_rows_source, best_epochs, title, output_path, tissue=None, bins=None, max_finetune_epoch=10, zoom_y_axis=False):
    classes = ["0", "1", "2"]
    fig, axes = plt.subplots(len(classes), len(METRICS), figsize=(15, 10), sharex=True, sharey=True)
    for row_index, class_id in enumerate(classes):
        data = combine_progression_rows(rows, best_epochs, bins=bins, max_finetune_epoch=max_finetune_epoch, class_id=class_id)
        mean_rows = mean_label_rows(mean_rows_source, best_epochs, bins=bins, class_id=class_id)
        summary = summarize_progression(data, mean_rows, tissue=tissue, bins=bins)
        for col_index, metric in enumerate(METRICS):
            ylabel = f"Class {class_id}" if col_index == 0 else None
            draw_progression_axis(
                axes[row_index, col_index],
                summary[metric],
                metric,
                ylabel=ylabel,
                show_legend=(row_index == 0 and col_index == len(METRICS) - 1),
                zoom_y_axis=zoom_y_axis,
            )
            if row_index == len(classes) - 1:
                axes[row_index, col_index].set_xlabel("Fine-tune epoch")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)




def average_sites_by_bin(data):
    if data.empty or "bin_rank" not in data.columns or "support" not in data.columns:
        return {}
    weighted = data[data["class"].astype(str) == "weighted"] if "class" in data.columns else data
    return {int(rank): value for rank, value in weighted.groupby("bin_rank")["support"].mean().items()}


def bin_legend_label(bin_rank, avg_sites):
    sites = avg_sites.get(int(bin_rank))
    if sites is None or math.isnan(sites):
        return f"Bin {int(bin_rank)}"
    return f"Bin {int(bin_rank)} (avg sites={sites:.0f})"


def plot_bin_progression(data, mean_rows, title, output_path, tissue=None, person=None, max_finetune_epoch=10):
    if tissue is not None:
        data = data[data["tissue"] == tissue]
        mean_rows = mean_rows[mean_rows["tissue"] == tissue]
    if person is not None:
        data = data[data["person"] == person]
        mean_rows = mean_rows[mean_rows["person"] == person]
    if data.empty:
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.text(0.05, 0.85, title, fontsize=20, weight="bold")
        fig.text(0.05, 0.72, "No variability-bin rows available for this selection.", fontsize=14)
        fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        return
    avg_sites = average_sites_by_bin(data)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), sharey=True)
    for axis, metric in zip(axes, METRICS):
        for bin_rank in sorted(data["bin_rank"].dropna().unique()):
            bin_rank = int(bin_rank)
            color = BIN_COLORS.get(bin_rank, None)
            bin_data = data[data["bin_rank"] == bin_rank]
            pre = bin_data[bin_data["kind"] == "pretrain"].groupby(["tissue", "person"], as_index=False)[metric].mean()
            retrain_person = bin_data[(bin_data["kind"] == "retrain") & (bin_data["finetune_epoch"].between(1, max_finetune_epoch))].groupby(["tissue", "person", "finetune_epoch"], as_index=False)[metric].mean()
            no_pretraining_person = bin_data[(bin_data["kind"] == "no_pretraining") & (bin_data["finetune_epoch"].between(1, max_finetune_epoch))].groupby(["tissue", "person", "finetune_epoch"], as_index=False)[metric].mean()
            mean_person = mean_rows[mean_rows["bin_rank"] == bin_rank].groupby(["tissue", "person"], as_index=False)[metric].mean()
            if tissue is None:
                pre_value = pre.groupby("tissue", as_index=False)[metric].mean()[metric].mean()
                retrain_tissue = retrain_person.groupby(["tissue", "finetune_epoch"], as_index=False)[metric].mean()
                retrain = retrain_tissue.groupby("finetune_epoch")[metric].mean().reset_index()
                no_pretraining_tissue = no_pretraining_person.groupby(["tissue", "finetune_epoch"], as_index=False)[metric].mean()
                no_pretraining = no_pretraining_tissue.groupby("finetune_epoch")[metric].mean().reset_index()
                mean_value = mean_person.groupby("tissue", as_index=False)[metric].mean()[metric].mean()
            else:
                pre_value = pre[metric].mean()
                retrain = retrain_person.groupby("finetune_epoch")[metric].mean().reset_index()
                no_pretraining = no_pretraining_person.groupby("finetune_epoch")[metric].mean().reset_index()
                mean_value = mean_person[metric].mean()
            if not no_pretraining.empty:
                axis.plot(no_pretraining["finetune_epoch"], no_pretraining[metric], marker="s", linestyle="-.", color=color, alpha=0.38, label="_nolegend_")
            if not retrain.empty:
                axis.plot(retrain["finetune_epoch"], retrain[metric], marker="o", color=color, label=bin_legend_label(bin_rank, avg_sites))
            if not math.isnan(pre_value):
                axis.scatter([0.78], [pre_value], marker="*", s=145, color=color, edgecolor="black", linewidth=0.65, zorder=6, label="_nolegend_")
            if not math.isnan(mean_value):
                side_x = max_finetune_epoch + 0.7
                axis.scatter(
                    [side_x],
                    [mean_value],
                    marker="D",
                    s=55,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=5,
                )
        axis.set_title(metric.title())
        axis.set_xlabel("Fine-tune epoch")
        axis.set_ylim(0, 1.05)
        axis.set_xlim(0.45, max_finetune_epoch + 1.4)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Weighted score")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, title="Retrain lines")
    fig.suptitle(title + "\nsolid=retrain, dash-dot=no pretraining, ★=selected pretrain-only score, right diamond=mean label")
    fig.tight_layout(rect=(0, 0, 0.84, 0.90))
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def selected_people(rows, tissues):
    people_by_tissue = {}
    filtered = rows[rows["tissue"].isin(tissues)]
    for tissue, tissue_rows in filtered.groupby("tissue"):
        people_by_tissue[tissue] = sorted(str(person) for person in tissue_rows["person"].dropna().unique())
    return people_by_tissue


def plot_individual_progressions(varfree_progress, varfree_mean, varbins_progress, varbins_mean, tissues, people_by_tissue, sample_counts, plot_dir, max_finetune_epoch, zoom_variability_free_y_axis=False):
    individual_plots = []
    for tissue in tissues:
        for person in people_by_tissue.get(tissue, []):
            safe_person = re.sub(r"[^A-Za-z0-9_.-]+", "_", person)
            vf_path = plot_dir / f"{tissue}_{safe_person}_individual_variability_free_progression.png"
            plot_progression(
                summarize_progression(varfree_progress, varfree_mean, tissue=tissue, person=person),
                f"{tissue} {person}: individual progression (variability-free)",
                vf_path,
                zoom_y_axis=zoom_variability_free_y_axis,
            )
            vb_path = plot_dir / f"{tissue}_{safe_person}_individual_variability_bins_progression.png"
            plot_bin_progression(
                varbins_progress,
                varbins_mean,
                f"{tissue} {person}: individual progression by variability bin",
                vb_path,
                tissue=tissue,
                person=person,
                max_finetune_epoch=max_finetune_epoch,
            )
            title_prefix = f"{tissue} {person} (tissue n={sample_counts.get(tissue, 0)}):"
            individual_plots.append((tissue, person, title_prefix, vf_path, vb_path))
    return individual_plots


def text_page(pdf, title, lines):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.text(0.05, 0.90, title, fontsize=26, weight="bold", va="top")
    y = 0.78
    for line in lines:
        fig.text(0.07, y, line, fontsize=15, va="top")
        y -= 0.065
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def image_page(pdf, image_path, title=None):
    image = Image.open(image_path)
    fig = plt.figure(figsize=(13.33, 7.5))
    if title:
        fig.text(0.04, 0.96, title, fontsize=18, weight="bold", va="top")
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.86])
    else:
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.imshow(image)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_file_to_zip(zf, path, data):
    zf.writestr(path, data)


def pptx_xml_escape(text):
    return escape(str(text))


def create_pptx(slides, output_path):
    width = 12192000
    height = 6858000
    image_dir = output_path.parent / "pptx_media"
    image_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        add_file_to_zip(zf, "[Content_Types].xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
''' + "".join(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>\n' for i in range(1, len(slides)+1)) + "</Types>")
        add_file_to_zip(zf, "_rels/.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>''')
        pres_rels = ''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>\n' for i in range(1, len(slides)+1))
        pres_rels += f'<Relationship Id="rId{len(slides)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
        add_file_to_zip(zf, "ppt/_rels/presentation.xml.rels", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{pres_rels}</Relationships>''')
        slide_ids = ''.join(f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, len(slides)+1))
        add_file_to_zip(zf, "ppt/presentation.xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{len(slides)+1}"/></p:sldMasterIdLst><p:sldIdLst>{slide_ids}</p:sldIdLst><p:sldSz cx="{width}" cy="{height}" type="wide"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>''')
        add_file_to_zip(zf, "ppt/slideMasters/slideMaster1.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld><p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>''')
        add_file_to_zip(zf, "ppt/slideMasters/_rels/slideMaster1.xml.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>''')
        add_file_to_zip(zf, "ppt/slideLayouts/slideLayout1.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld></p:sldLayout>''')
        for idx, slide in enumerate(slides, start=1):
            rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>']
            pic_xml = ""
            if slide.get("image"):
                image_path = Path(slide["image"])
                media_name = f"image{idx}.png"
                zf.write(image_path, f"ppt/media/{media_name}")
                rels.append(f'<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{media_name}"/>')
                pic_xml = f'''<p:pic><p:nvPicPr><p:cNvPr id="4" name="Picture {idx}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="457200" y="914400"/><a:ext cx="11277600" cy="5486400"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'''
            title = pptx_xml_escape(slide.get("title", ""))
            text = pptx_xml_escape(slide.get("text", ""))
            title_xml = f'''<p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="457200" y="182880"/><a:ext cx="11277600" cy="548640"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr sz="2800" b="1"/><a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>'''
            body_xml = ""
            if text:
                lines = ''.join(f'<a:p><a:r><a:rPr sz="1800"/><a:t>{pptx_xml_escape(line)}</a:t></a:r></a:p>' for line in str(slide["text"]).split("\n"))
                body_xml = f'''<p:sp><p:nvSpPr><p:cNvPr id="3" name="Body"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="685800" y="1219200"/><a:ext cx="10972800" cy="4572000"/></a:xfrm></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/>{lines}</p:txBody></p:sp>'''
            slide_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>{title_xml}{body_xml}{pic_xml}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'''
            add_file_to_zip(zf, f"ppt/slides/slide{idx}.xml", slide_xml)
            add_file_to_zip(zf, f"ppt/slides/_rels/slide{idx}.xml.rels", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>''')


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    tissues = available_tissues()
    varfree_rows = load_variability_free_rows(tissues)
    varbins_rows = load_variability_bin_rows(tissues)
    best = best_pretrain_epochs(varfree_rows, args.require_full_finetune, args.max_finetune_epoch, args.best_pretrain_epoch)
    _, overall_pretrain_scores = pretrain_epoch_scores(varfree_rows, args.require_full_finetune, args.max_finetune_epoch)
    best.to_csv(args.output_dir / "selected_overall_pretrain_epoch_by_tissue.csv", index=False)
    overall_pretrain_scores.to_csv(args.output_dir / "overall_pretrain_epoch_scores.csv", index=False)

    slides = []
    pdf_path = args.output_dir / "final_pi_report.pdf"
    pptx_path = args.output_dir / "final_pi_report.pptx"

    varfree_progress = combine_progression_rows(varfree_rows, best, max_finetune_epoch=args.max_finetune_epoch)
    varfree_mean = mean_label_rows(varfree_rows, best)
    varbins_progress = combine_progression_rows(varbins_rows, best, max_finetune_epoch=args.max_finetune_epoch)
    varbins_mean = mean_label_rows(varbins_rows, best)

    sample_counts = tissue_sample_counts(varfree_rows, tissues)
    combined_n = combined_sample_count(sample_counts)
    combined_vf = plot_dir / "combined_variability_free_best_pretrain_retrain_progression.png"
    plot_progression(
        summarize_progression(varfree_progress, varfree_mean),
        "Combined tissues: selected pretrain epoch and retrain progression (variability-free)",
        combined_vf,
        zoom_y_axis=args.zoom_variability_free_y_axis,
    )
    combined_bins = plot_dir / "combined_variability_bins_best_pretrain_retrain_progression.png"
    plot_bin_progression(
        varbins_progress,
        varbins_mean,
        "Combined tissues: best pretrain epoch and retrain progression by variability bin",
        combined_bins,
        max_finetune_epoch=args.max_finetune_epoch,
    )
    combined_vf_classes = plot_dir / "combined_variability_free_per_class_progression.png"
    plot_class_progression_grid(
        varfree_rows,
        varfree_rows,
        best,
        "Combined tissues: per-class progression (variability-free)",
        combined_vf_classes,
        max_finetune_epoch=args.max_finetune_epoch,
        zoom_y_axis=args.zoom_variability_free_y_axis,
    )
    combined_bins_classes = plot_dir / "combined_variability_bins_all_bins_per_class_progression.png"
    plot_class_progression_grid(
        varbins_rows,
        varbins_rows,
        best,
        "Combined tissues: per-class progression, all variability bins",
        combined_bins_classes,
        max_finetune_epoch=args.max_finetune_epoch,
    )

    tissue_plots = []
    for tissue in tissues:
        epoch = int(best[best["tissue"] == tissue]["pretraining_epoch"].iloc[0])
        vf_path = plot_dir / f"{tissue}_variability_free_bestE{epoch}_retrain_progression.png"
        plot_progression(
            summarize_progression(varfree_progress, varfree_mean, tissue=tissue),
            f"{tissue}: best pretrain E{epoch}, retrain progression (variability-free)",
            vf_path,
            zoom_y_axis=args.zoom_variability_free_y_axis,
        )
        vb_path = plot_dir / f"{tissue}_variability_bins_bestE{epoch}_retrain_progression.png"
        plot_bin_progression(
            varbins_progress,
            varbins_mean,
            f"{tissue}: best pretrain E{epoch}, retrain progression by variability bin",
            vb_path,
            tissue=tissue,
            max_finetune_epoch=args.max_finetune_epoch,
        )
        vf_class_path = plot_dir / f"{tissue}_variability_free_bestE{epoch}_per_class_progression.png"
        plot_class_progression_grid(
            varfree_rows,
            varfree_rows,
            best,
            f"{tissue}: per-class progression (variability-free)",
            vf_class_path,
            tissue=tissue,
            max_finetune_epoch=args.max_finetune_epoch,
            zoom_y_axis=args.zoom_variability_free_y_axis,
        )
        vb_class_path = plot_dir / f"{tissue}_variability_bins_all_bins_E{epoch}_per_class_progression.png"
        plot_class_progression_grid(
            varbins_rows,
            varbins_rows,
            best,
            f"{tissue}: per-class progression, all variability bins",
            vb_class_path,
            tissue=tissue,
            max_finetune_epoch=args.max_finetune_epoch,
        )
        tissue_plots.append((tissue, epoch, vf_path, vb_path, vf_class_path, vb_class_path))

    individual_plots = []
    if not args.skip_individual_pages:
        people_by_tissue = selected_people(varfree_progress, tissues)
        individual_plots = plot_individual_progressions(
            varfree_progress,
            varfree_mean,
            varbins_progress,
            varbins_mean,
            tissues,
            people_by_tissue,
            sample_counts,
            plot_dir,
            args.max_finetune_epoch,
            zoom_variability_free_y_axis=args.zoom_variability_free_y_axis,
        )

    best_lines = [
        f"{row.tissue}: best pretrain E{int(row.pretraining_epoch)} (weighted F1={row.best_pretrain_weighted_f1:.3f})"
        for row in best.itertuples(index=False)
    ]
    summary_lines = [
        "Scope: completed variability-free and variability-bin evaluations.",
        "Metric shown in weighted progression plots: weighted F1, weighted precision, weighted recall.",
        "Per-class slides show class 0/1/2 F1, precision, and recall; variability-bin class slides include all bins.",
        "Individual-level pages show each person's weighted F1, precision, and recall progression for variability-free and variability-bin evaluations.",
        (
            f"A single shared pretrain-only epoch E{int(best['pretraining_epoch'].iloc[0])} was selected by overall variability-free weighted F1 with equal tissue weighting"
            + (" restricted to retrain groups with all requested fine-tune epochs" if args.require_full_finetune else "")
            + "."
        ),
        "Retrain plots show no-pretraining and selected-pretrain-initialized models across fine-tuning epochs 1-10.",
        "Variability-bin plots: solid=retrain, dash-dot=no pretraining, star=selected pretrain-only score, right-side diamond=mean-label baseline; bin legend includes average sites per bin.",
    ]

    combined_title = f"Combined (n={combined_n}):"
    with PdfPages(pdf_path) as pdf:
        text_page(pdf, "Token Classification Report", ["Prepared from completed grouped evaluations", "", *summary_lines])
        text_page(pdf, "Shared Overall Pretrain Epoch Used For All Tissues", best_lines)
        image_page(pdf, combined_vf, f"{combined_title} variability-free progression")
        image_page(pdf, combined_bins, f"{combined_title} variability-bin progression")
        image_page(pdf, combined_vf_classes, f"{combined_title} per-class variability-free progression")
        image_page(pdf, combined_bins_classes, f"{combined_title} per-class all variability bins progression")
        for tissue, epoch, vf_path, vb_path, vf_class_path, vb_class_path in tissue_plots:
            tissue_title = f"{tissue} (n={sample_counts.get(tissue, 0)}):"
            image_page(pdf, vf_path, f"{tissue_title} variability-free")
            image_page(pdf, vb_path, f"{tissue_title} variability bins")
            image_page(pdf, vf_class_path, f"{tissue_title} per-class variability-free")
            image_page(pdf, vb_class_path, f"{tissue_title} per-class all variability bins")
        if individual_plots:
            text_page(
                pdf,
                "Individual-Level Progression Pages",
                [
                    "Each following person has two weighted-metric pages:",
                    "1. Variability-free F1, precision, and recall across fine-tune epochs.",
                    "2. Variability-bin F1, precision, and recall across fine-tune epochs.",
                    "These pages use the same selected shared pretrain epoch and mean-label baseline as the aggregate report.",
                ],
            )
            for tissue, person, title_prefix, vf_path, vb_path in individual_plots:
                image_page(pdf, vf_path, f"{title_prefix} variability-free individual progression")
                image_page(pdf, vb_path, f"{title_prefix} variability-bin individual progression")

    slides.append({"title": "Token Classification Report", "text": "Prepared from completed grouped evaluations\n\n" + "\n".join(summary_lines)})
    slides.append({"title": "Shared Overall Pretrain Epoch Used For All Tissues", "text": "\n".join(best_lines)})
    slides.append({"title": f"{combined_title} variability-free progression", "image": str(combined_vf)})
    slides.append({"title": f"{combined_title} variability-bin progression", "image": str(combined_bins)})
    slides.append({"title": f"{combined_title} per-class variability-free progression", "image": str(combined_vf_classes)})
    slides.append({"title": f"{combined_title} per-class all variability bins progression", "image": str(combined_bins_classes)})
    for tissue, epoch, vf_path, vb_path, vf_class_path, vb_class_path in tissue_plots:
        tissue_title = f"{tissue} (n={sample_counts.get(tissue, 0)}):"
        slides.append({"title": f"{tissue_title} variability-free", "image": str(vf_path)})
        slides.append({"title": f"{tissue_title} variability bins", "image": str(vb_path)})
        slides.append({"title": f"{tissue_title} per-class variability-free", "image": str(vf_class_path)})
        slides.append({"title": f"{tissue_title} per-class all variability bins", "image": str(vb_class_path)})
    if individual_plots:
        slides.append({
            "title": "Individual-Level Progression Pages",
            "text": "Each following person has variability-free and variability-bin weighted progression pages.\nSame selected shared pretrain epoch and mean-label baseline as aggregate report.",
        })
        for tissue, person, title_prefix, vf_path, vb_path in individual_plots:
            slides.append({"title": f"{title_prefix} variability-free individual progression", "image": str(vf_path)})
            slides.append({"title": f"{title_prefix} variability-bin individual progression", "image": str(vb_path)})
    create_pptx(slides, pptx_path)

    print(pdf_path)
    print(pptx_path)
    print(args.output_dir / "selected_overall_pretrain_epoch_by_tissue.csv")
    print(args.output_dir / "overall_pretrain_epoch_scores.csv")


if __name__ == "__main__":
    main()

# Examples:
# /sci/nosnap/michall/roeizucker/new_python_env/bin/python3 scripts/create_final_pi_report.py \
#   --output-dir scripts/results/final_pi_report_with_individuals_zoomed_varfree \
#   --best-pretrain-epoch 10 \
#   --zoom-variability-free-y-axis
# /sci/nosnap/michall/roeizucker/new_python_env/bin/python3 scripts/create_final_pi_report.py \
#   --output-dir scripts/results/final_pi_report \
#   --skip-individual-pages
