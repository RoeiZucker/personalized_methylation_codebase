#!/usr/bin/env python3
"""Visualize shard-project results with a small, explicit pipeline.

Current scope:
- load paired shard result directories for the same tissue
- compare kmer and window split strategies
- plot all-participants final-epoch mean Pearson by variability level
- overlay atlas-evaluation means where they align with the standard evaluation
- plot all-participants epoch trends

The script stays intentionally narrow so it is easy to inspect and extend.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


RESULT_FILE_RE = re.compile(
    r"^(?P<prefix>.+?)_eval(?P<variant>_lora_over_lora|_lora)?(?P<project>.+?)"
    r"_lr_(?P<lr>[^_]+)_bs_(?P<bs>[^_]+)_seq_(?P<seq>\d+)_testsize_(?P<testsize>[^_]+)_result\.csv(?:\.gitbackup)?$"
)
ATLAS_RESULT_FILE_RE = re.compile(
    r"^(?P<person>.+?)_atlas_eval_(?P<project>.+?)_seq_(?P<seq>\d+)(?P<small>_small)?_result\.csv(?:\.gitbackup)?$"
)
METRIC_COL_RE = re.compile(r"^(?P<bin>.+)_(?P<metric>pearsonr|mse|mae)$")
PROJECT_PREFIX = "sixteen_shards_of_adonalsium_"
DEFAULT_VARIANT = "full_finetune"
DEFAULT_PROJECT_DIRS = [
    Path(__file__).resolve().parents[2] / "results" / "_Bladder-Epithelial_kmer",
    Path(__file__).resolve().parents[2] / "results" / "_Bladder-Epithelial_window",
    Path(__file__).resolve().parents[2] / "results" / "_Cortex-Neuron_kmer",
    Path(__file__).resolve().parents[2] / "results" / "_Cortex-Neuron_window",
]
PROJECT_TISSUE_OVERRIDES = {
    "_sixteen_Heart_Cardiomyocyte_window": "Heart Cardiomyocyte",
    "_sixteen_shards_of_adonalsium_Heart_Cardiomyocyte": "Heart Cardiomyocyte",
    "_sixteen_Fibroblasts_window": "Heart Fibroblasts",
    "_sixteen_shards_of_adonalsium_Heart_Fibroblasts": "Heart Fibroblasts",
    "_sixteen_liver_window": "Liver",
    "_sixteen_shards_of_adonalsium_liver": "Liver",
    "_Bladder-Epithelial_kmer": "Bladder-Epithelial",
    "_Bladder-Epithelial_window": "Bladder-Epithelial",
    "_Cortex-Neuron_kmer": "Cortex-Neuron",
    "_Cortex-Neuron_window": "Cortex-Neuron",
}
SPLIT_ORDER = ["kmer", "window"]
SPLIT_HATCHES = {"kmer": "", "window": "//"}
SPLIT_LINESTYLES = {"kmer": "-", "window": "--"}
ATLAS_MARKER_COLOR = "red"


@dataclass(frozen=True)
class ResultFile:
    file_path: Path
    file_name: str
    project_name: str
    tissue: str
    split_strategy: str
    person_id: str
    seq_len: int
    eval_variant: str
    model_param: str


@dataclass(frozen=True)
class AtlasResultFile:
    file_path: Path
    file_name: str
    project_name: str
    tissue: str
    split_strategy: str
    person_id: str
    seq_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize shard-project evaluation results.")
    parser.add_argument(
        "--project-dirs",
        type=Path,
        nargs="+",
        default=DEFAULT_PROJECT_DIRS,
        help="One or more result directories to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "kol_kore_visualize",
        help="Directory where plots and tidy CSV outputs will be written.",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        default=None,
        help="Fallback split-strategy label if it cannot be inferred from a directory name.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display figures with plt.show() in addition to saving them.",
    )
    return parser.parse_args()


def infer_tissue_label(project_dir_name: str) -> str:
    if project_dir_name in PROJECT_TISSUE_OVERRIDES:
        return PROJECT_TISSUE_OVERRIDES[project_dir_name]
    label = project_dir_name.lstrip("_")
    if label.startswith(PROJECT_PREFIX):
        label = label[len(PROJECT_PREFIX) :]
    elif label.startswith("sixteen_"):
        label = label[len("sixteen_") :]
    for suffix in ("_window", "_kmer"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    if label == "Fibroblasts":
        label = "Heart_Fibroblasts"
    label = label.replace("_", " ").strip()
    return label.title()


def infer_split_strategy(project_dir_name: str, fallback_split_strategy: Optional[str] = None) -> str:
    lowered = project_dir_name.lower()
    if "window" in lowered:
        return "window"
    if "shards_of_adonalsium" in lowered or lowered.endswith("_kmer"):
        return "kmer"
    return fallback_split_strategy or "unknown"


def tissue_slug(tissue: str) -> str:
    return tissue.lower().replace(" ", "_")


def canonical_model_param(prefix: str) -> str:
    tail = prefix.split("_", 1)[1] if "_" in prefix else prefix
    low = tail.lower()

    if "no_pretraining" in low or "no_pretrain" in low or "no-pretrain" in low:
        return "no_pretrain"

    match = re.search(r"epoch-(\d+)-step-\d+", low)
    if match:
        return f"epoch {int(match.group(1))}"

    match = re.search(r"epoch-(\d+)", low) or re.search(r"epoch(\d+)", low)
    if match:
        return f"epoch {int(match.group(1))}"

    return tail


def parse_result_file(file_path: Path, fallback_split_strategy: Optional[str] = None) -> Optional[ResultFile]:
    match = RESULT_FILE_RE.match(file_path.name)
    if not match:
        return None

    groups = match.groupdict()
    variant_raw = groups["variant"] or ""
    eval_variant = {
        "": "full_finetune",
        "_lora": "lora",
        "_lora_over_lora": "lora_over_lora",
    }[variant_raw]

    project_name = file_path.parent.name
    return ResultFile(
        file_path=file_path,
        file_name=file_path.name,
        project_name=project_name,
        tissue=infer_tissue_label(project_name),
        split_strategy=infer_split_strategy(project_name, fallback_split_strategy),
        person_id=groups["prefix"].split("_", 1)[0],
        seq_len=int(groups["seq"]),
        eval_variant=eval_variant,
        model_param=canonical_model_param(groups["prefix"]),
    )


def parse_atlas_result_file(
    file_path: Path,
    fallback_split_strategy: Optional[str] = None,
) -> Optional[AtlasResultFile]:
    match = ATLAS_RESULT_FILE_RE.match(file_path.name)
    if not match:
        return None

    groups = match.groupdict()
    if groups.get("small"):
        return None

    project_name = file_path.parent.name
    return AtlasResultFile(
        file_path=file_path,
        file_name=file_path.name,
        project_name=project_name,
        tissue=infer_tissue_label(project_name),
        split_strategy=infer_split_strategy(project_name, fallback_split_strategy),
        person_id=groups["person"],
        seq_len=int(groups["seq"]),
    )


def preferred_csv_like_paths(project_dir: Path) -> List[Path]:
    candidates = sorted(project_dir.glob("*.csv")) + sorted(project_dir.glob("*.csv.gitbackup"))
    preferred: dict[str, Path] = {}
    for file_path in candidates:
        canonical_name = file_path.name.removesuffix(".gitbackup")
        current = preferred.get(canonical_name)
        if current is None or (current.name.endswith(".gitbackup") and not file_path.name.endswith(".gitbackup")):
            preferred[canonical_name] = file_path
    return [preferred[name] for name in sorted(preferred)]


def discover_project_files(
    project_dir: Path,
    fallback_split_strategy: Optional[str] = None,
) -> tuple[List[ResultFile], List[AtlasResultFile], List[Path]]:
    parsed: List[ResultFile] = []
    atlas_parsed: List[AtlasResultFile] = []
    unparsed: List[Path] = []

    for file_path in preferred_csv_like_paths(project_dir):
        atlas_file = parse_atlas_result_file(file_path, fallback_split_strategy=fallback_split_strategy)
        if atlas_file is not None:
            atlas_parsed.append(atlas_file)
            continue

        parsed_file = parse_result_file(file_path, fallback_split_strategy=fallback_split_strategy)
        if parsed_file is None:
            unparsed.append(file_path)
            continue
        parsed.append(parsed_file)

    return parsed, atlas_parsed, unparsed


def normalize_result_csv(result_file: ResultFile) -> pd.DataFrame:
    df = pd.read_csv(result_file.file_path)
    metric_cols = [col for col in df.columns if METRIC_COL_RE.match(col)]
    if not metric_cols:
        raise ValueError(f"No metric columns found in {result_file.file_path}")

    row_df = df.copy()
    row_df["row_in_file"] = range(len(row_df))
    row_df["eval_path"] = row_df["paths"].astype(str)
    row_df["eval_checkpoint"] = row_df["eval_path"].str.findall(r"epoch-\d+-step-\d+").str[-1].fillna("unknown")
    row_df["eval_epoch"] = pd.to_numeric(
        row_df["eval_checkpoint"].str.extract(r"epoch-(\d+)-step-\d+")[0],
        errors="coerce",
    )

    if result_file.model_param != "no_pretrain":
        row_df.loc[row_df["row_in_file"] == 0, "eval_epoch"] = 0

    long_df = row_df.melt(
        id_vars=["row_in_file", "eval_epoch"],
        value_vars=metric_cols,
        var_name="metric_col",
        value_name="value",
    )
    metric_parts = long_df["metric_col"].str.extract(METRIC_COL_RE)
    long_df["bin_range"] = metric_parts["bin"]
    long_df["metric"] = metric_parts["metric"]

    bounds = long_df["bin_range"].str.extract(r"^(?P<bin_start>[0-9.]+)-(?P<bin_end>[0-9.]+)$")
    long_df["bin_start"] = pd.to_numeric(bounds["bin_start"], errors="coerce")
    long_df["variability_level"] = (
        long_df.groupby("row_in_file")["bin_start"].rank(method="dense", ascending=True).astype("Int64")
    )

    long_df["project_name"] = result_file.project_name
    long_df["tissue"] = result_file.tissue
    long_df["split_strategy"] = result_file.split_strategy
    long_df["person_id"] = result_file.person_id
    long_df["seq_len"] = result_file.seq_len
    long_df["eval_variant"] = result_file.eval_variant
    long_df["model_param"] = result_file.model_param
    long_df["file_name"] = result_file.file_name
    long_df["file_path"] = str(result_file.file_path)

    return long_df[
        [
            "project_name",
            "tissue",
            "split_strategy",
            "person_id",
            "seq_len",
            "eval_variant",
            "model_param",
            "file_name",
            "file_path",
            "eval_epoch",
            "metric",
            "variability_level",
            "value",
        ]
    ].copy()


def parse_metric_cell(cell: object, expected_metric: str) -> float:
    if isinstance(cell, dict):
        return float(cell.get(expected_metric, np.nan))
    if pd.isna(cell):
        return float("nan")
    parsed = ast.literal_eval(str(cell))
    if not isinstance(parsed, dict):
        return float("nan")
    return float(parsed.get(expected_metric, np.nan))


def normalize_atlas_result_csv(result_file: AtlasResultFile) -> pd.DataFrame:
    df = pd.read_csv(result_file.file_path)
    if df.shape[1] < 4:
        raise ValueError(f"Unexpected atlas CSV shape in {result_file.file_path}")

    row_df = pd.DataFrame(
        {
            "bin_range": df.iloc[:, 0].astype(str),
            "pearsonr": df.iloc[:, 1].apply(lambda value: parse_metric_cell(value, "pearsonr")),
            "mse": df.iloc[:, 2].apply(lambda value: parse_metric_cell(value, "mse")),
            "mae": df.iloc[:, 3].apply(lambda value: parse_metric_cell(value, "mae")),
        }
    )
    bounds = row_df["bin_range"].str.extract(r"^(?P<bin_start>[0-9.]+)-(?P<bin_end>[0-9.]+)$")
    row_df["bin_start"] = pd.to_numeric(bounds["bin_start"], errors="coerce")
    row_df["variability_level"] = row_df["bin_start"].rank(method="dense", ascending=True).astype("Int64")

    long_df = row_df.melt(
        id_vars=["bin_range", "variability_level"],
        value_vars=["pearsonr", "mse", "mae"],
        var_name="metric",
        value_name="value",
    )
    long_df["project_name"] = result_file.project_name
    long_df["tissue"] = result_file.tissue
    long_df["split_strategy"] = result_file.split_strategy
    long_df["person_id"] = result_file.person_id
    long_df["seq_len"] = result_file.seq_len
    long_df["file_name"] = result_file.file_name
    long_df["file_path"] = str(result_file.file_path)

    return long_df[
        [
            "project_name",
            "tissue",
            "split_strategy",
            "person_id",
            "seq_len",
            "file_name",
            "file_path",
            "metric",
            "variability_level",
            "value",
        ]
    ].copy()


def load_projects(
    project_dirs: Iterable[Path],
    fallback_split_strategy: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parsed_rows: List[pd.DataFrame] = []
    atlas_rows: List[pd.DataFrame] = []
    file_index_rows: List[dict] = []
    unparsed_rows: List[dict] = []

    for project_dir in project_dirs:
        parsed_files, atlas_files, unparsed_files = discover_project_files(
            project_dir,
            fallback_split_strategy=fallback_split_strategy,
        )
        for result_file in parsed_files:
            parsed_rows.append(normalize_result_csv(result_file))
            file_index_rows.append(
                {
                    "project_name": result_file.project_name,
                    "tissue": result_file.tissue,
                    "split_strategy": result_file.split_strategy,
                    "person_id": result_file.person_id,
                    "seq_len": result_file.seq_len,
                    "eval_variant": result_file.eval_variant,
                    "model_param": result_file.model_param,
                    "analysis_type": "standard_eval",
                    "file_name": result_file.file_name,
                    "file_path": str(result_file.file_path),
                }
            )
        for atlas_file in atlas_files:
            atlas_rows.append(normalize_atlas_result_csv(atlas_file))
            file_index_rows.append(
                {
                    "project_name": atlas_file.project_name,
                    "tissue": atlas_file.tissue,
                    "split_strategy": atlas_file.split_strategy,
                    "person_id": atlas_file.person_id,
                    "seq_len": atlas_file.seq_len,
                    "analysis_type": "atlas_eval",
                    "file_name": atlas_file.file_name,
                    "file_path": str(atlas_file.file_path),
                }
            )
        for file_path in unparsed_files:
            unparsed_rows.append(
                {
                    "project_name": project_dir.name,
                    "analysis_type": "unparsed",
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                }
            )

    tidy_df = (
        pd.concat(parsed_rows, ignore_index=True)
        if parsed_rows
        else pd.DataFrame(
            columns=[
                "project_name",
                "tissue",
                "split_strategy",
                "person_id",
                "seq_len",
                "eval_variant",
                "model_param",
                "file_name",
                "file_path",
                "eval_epoch",
                "metric",
                "variability_level",
                "value",
            ]
        )
    )
    atlas_df = (
        pd.concat(atlas_rows, ignore_index=True)
        if atlas_rows
        else pd.DataFrame(
            columns=[
                "project_name",
                "tissue",
                "split_strategy",
                "person_id",
                "seq_len",
                "file_name",
                "file_path",
                "metric",
                "variability_level",
                "value",
            ]
        )
    )
    file_index_df = pd.DataFrame(file_index_rows)
    if unparsed_rows:
        file_index_df = pd.concat([file_index_df, pd.DataFrame(unparsed_rows)], ignore_index=True, sort=False)

    return tidy_df, atlas_df, file_index_df


def select_final_epoch_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    group_cols = [
        "project_name",
        "tissue",
        "split_strategy",
        "person_id",
        "seq_len",
        "eval_variant",
        "model_param",
    ]
    chosen = (
        df.groupby(group_cols, as_index=False)["eval_epoch"]
        .max()
        .rename(columns={"eval_epoch": "chosen_epoch"})
    )
    final_rows = df.merge(
        chosen,
        left_on=group_cols + ["eval_epoch"],
        right_on=group_cols + ["chosen_epoch"],
        how="inner",
    )
    return final_rows


def compute_padded_ylim(values: pd.Series) -> tuple[float, float]:
    if values.empty:
        return -0.1, 1.0

    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        pad = 0.05 if np.isclose(vmax, 0.0) else abs(vmax) * 0.1
        return vmin - pad, vmax + pad

    pad = max((vmax - vmin) * 0.08, 0.03)
    return vmin - pad, vmax + pad


def ordered_splits(values: Iterable[str]) -> List[str]:
    present = list(dict.fromkeys(values))
    return [split for split in SPLIT_ORDER if split in present] + [
        split for split in present if split not in SPLIT_ORDER
    ]


def plot_final_epoch_mean_by_variability(
    pear_df: pd.DataFrame,
    atlas_df: pd.DataFrame,
    tissue: str,
    output_dir: Path,
    show_plots: bool = False,
) -> None:
    project_df = pear_df[
        (pear_df["tissue"] == tissue)
        & (pear_df["eval_variant"] == DEFAULT_VARIANT)
        & (pear_df["metric"] == "pearsonr")
    ].copy()
    if project_df.empty:
        print(f"No Pearson rows available for {tissue} final-epoch variability plot.")
        return

    final_rows = select_final_epoch_rows(project_df)
    avg_df = (
        final_rows.groupby(["split_strategy", "seq_len", "model_param", "variability_level"], as_index=False)
        .agg(value=("value", "mean"), n_participants=("person_id", "nunique"))
        .copy()
    )
    avg_df.to_csv(output_dir / f"{tissue_slug(tissue)}_final_epoch_mean_by_variability_rows.csv", index=False)

    atlas_overlay_df = (
        atlas_df[
            (atlas_df["tissue"] == tissue)
            & (atlas_df["metric"] == "pearsonr")
            & (atlas_df["split_strategy"] == "window")
        ]
        .groupby(["split_strategy", "seq_len", "variability_level"], as_index=False)
        .agg(value=("value", "mean"), n_participants=("person_id", "nunique"))
        .copy()
    )
    if not atlas_overlay_df.empty:
        atlas_overlay_df.to_csv(output_dir / f"{tissue_slug(tissue)}_atlas_mean_by_variability_rows.csv", index=False)

    seqs = sorted(avg_df["seq_len"].dropna().astype(int).unique().tolist())
    model_params = sorted(avg_df["model_param"].dropna().unique().tolist())
    split_strategies = ordered_splits(avg_df["split_strategy"].dropna().tolist())
    if not seqs or not model_params or not split_strategies:
        print(f"Not enough data to plot final-epoch variability means for {tissue}.")
        return

    palette = plt.get_cmap("tab10")
    color_map = {model: palette(i % 10) for i, model in enumerate(model_params)}
    fig, axes = plt.subplots(1, len(seqs), figsize=(6 * len(seqs), 5.5), sharey=True)
    if len(seqs) == 1:
        axes = [axes]

    fig.suptitle(
        f"All participants: final-epoch mean Pearson by variability level ({tissue})",
        y=0.98,
    )
    ylim_values = avg_df["value"]
    if not atlas_overlay_df.empty:
        ylim_values = pd.concat([ylim_values, atlas_overlay_df["value"]], ignore_index=True)
    y_min, y_max = compute_padded_ylim(ylim_values)
    split_display = {split: split.title() for split in split_strategies}
    has_atlas_overlay = False

    for ax, seq in zip(axes, seqs):
        sub = avg_df[avg_df["seq_len"] == seq].copy()
        ax.set_title(f"seq_{seq}")
        ax.set_xlabel("Variability level")
        ax.set_ylabel("Mean Final-Epoch Pearsonr")
        ax.set_ylim(y_min, y_max)

        levels = sorted(sub["variability_level"].dropna().astype(int).unique().tolist())
        x = np.arange(len(levels), dtype=float)
        bar_keys = [(split, model) for split in split_strategies for model in model_params]
        width = min(0.84 / max(len(bar_keys), 1), 0.16)
        offsets = (np.arange(len(bar_keys)) - (len(bar_keys) - 1) / 2.0) * width

        for i_key, (split, model) in enumerate(bar_keys):
            combo_sub = sub[(sub["split_strategy"] == split) & (sub["model_param"] == model)].copy()
            level_to_val = {
                int(v): float(y)
                for v, y in zip(combo_sub["variability_level"].tolist(), combo_sub["value"].tolist())
            }
            heights = [level_to_val.get(level, np.nan) for level in levels]
            bars = ax.bar(
                x + offsets[i_key],
                heights,
                width=width * 0.95,
                color=color_map[model],
                hatch=SPLIT_HATCHES.get(split, ""),
                alpha=0.95,
                edgecolor="black",
                linewidth=0.4,
                label=f"{split_display[split]} | {model}",
            )
            for bar in bars:
                height = bar.get_height()
                if np.isnan(height):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (y_max - y_min) * 0.015,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([str(level) for level in levels])

        atlas_seq_df = atlas_overlay_df[atlas_overlay_df["seq_len"] == seq].copy()
        if not atlas_seq_df.empty:
            atlas_level_to_val = {
                int(v): float(y)
                for v, y in zip(atlas_seq_df["variability_level"].tolist(), atlas_seq_df["value"].tolist())
            }
            atlas_values = [atlas_level_to_val.get(level, np.nan) for level in levels]
            ax.scatter(
                x,
                atlas_values,
                color=ATLAS_MARKER_COLOR,
                marker="D",
                s=42,
                edgecolors="black",
                linewidths=0.4,
                zorder=5,
            )
            has_atlas_overlay = True

    model_handles = [Patch(facecolor=color_map[m], edgecolor="none", label=m) for m in model_params]
    split_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=SPLIT_HATCHES.get(split, ""), label=split_display[split])
        for split in split_strategies
    ]
    fig.legend(
        handles=model_handles,
        title="Model parameter",
        loc="lower center",
        bbox_to_anchor=(0.24, 0.02),
        ncol=min(4, len(model_params)),
        frameon=True,
    )
    fig.legend(
        handles=split_handles,
        title="Split strategy",
        loc="lower center",
        bbox_to_anchor=(0.76, 0.02),
        ncol=min(2, len(split_strategies)),
        frameon=True,
    )
    if has_atlas_overlay:
        fig.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    marker="D",
                    color=ATLAS_MARKER_COLOR,
                    markerfacecolor=ATLAS_MARKER_COLOR,
                    markeredgecolor="black",
                    linewidth=0,
                    label="Atlas eval",
                )
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            frameon=True,
        )
    fig.tight_layout(rect=[0, 0.14, 1, 0.94])

    out_path = output_dir / "plots" / f"{tissue_slug(tissue)}_all_participants_final_epoch_mean_by_variability.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def plot_epoch_trend(
    pear_df: pd.DataFrame,
    tissue: str,
    output_dir: Path,
    show_plots: bool = False,
) -> None:
    project_df = pear_df[
        (pear_df["tissue"] == tissue)
        & (pear_df["eval_variant"] == DEFAULT_VARIANT)
        & (pear_df["metric"] == "pearsonr")
        & (pear_df["eval_epoch"].notna())
        & (pear_df["variability_level"].notna())
    ].copy()
    if project_df.empty:
        print(f"No Pearson rows available for {tissue} epoch-trend plot.")
        return

    project_df["eval_epoch"] = project_df["eval_epoch"].astype(int)
    project_df["variability_level"] = project_df["variability_level"].astype(int)

    trend_df = (
        project_df.groupby(["split_strategy", "seq_len", "model_param", "eval_epoch", "variability_level"], as_index=False)
        .agg(value=("value", "mean"), n_participants=("person_id", "nunique"))
        .copy()
    )
    trend_df.to_csv(output_dir / f"{tissue_slug(tissue)}_epoch_trend_rows.csv", index=False)

    model_params = sorted(trend_df["model_param"].dropna().unique().tolist())
    split_strategies = ordered_splits(trend_df["split_strategy"].dropna().tolist())
    palette = plt.get_cmap("tab10")
    color_map = {model: palette(i % 10) for i, model in enumerate(model_params)}

    for seq in sorted(trend_df["seq_len"].dropna().astype(int).unique().tolist()):
        seq_df = trend_df[trend_df["seq_len"] == seq].copy()
        levels = sorted(seq_df["variability_level"].dropna().astype(int).unique().tolist())
        fig, axes = plt.subplots(len(levels), 1, figsize=(9, 3.2 * len(levels)), sharex=True, sharey=True)
        if len(levels) == 1:
            axes = [axes]

        fig.suptitle(
            f"All participants: epoch trend ({tissue} | seq_{seq})",
            y=0.995,
        )
        y_min, y_max = compute_padded_ylim(seq_df["value"])
        split_display = {split: split.title() for split in split_strategies}

        for ax, level in zip(axes, levels):
            level_df = seq_df[seq_df["variability_level"] == level].copy()
            ax.set_title(f"Level {level}")
            ax.set_ylabel("Mean Pearsonr")
            ax.set_ylim(y_min, y_max)
            ax.grid(True, axis="y", alpha=0.3)

            for split in split_strategies:
                for model in model_params:
                    model_df = level_df[
                        (level_df["split_strategy"] == split) & (level_df["model_param"] == model)
                    ].sort_values("eval_epoch")
                    if model_df.empty:
                        continue
                    ax.plot(
                        model_df["eval_epoch"],
                        model_df["value"],
                        marker="o",
                        linewidth=2,
                        markersize=5,
                        color=color_map[model],
                        linestyle=SPLIT_LINESTYLES.get(split, "-"),
                        label=f"{split_display[split]} | {model}",
                    )

        axes[-1].set_xlabel("Eval epoch")
        model_handles = [Patch(facecolor=color_map[m], edgecolor="none", label=m) for m in model_params]
        split_handles = [
            plt.Line2D([0], [0], color="black", linestyle=SPLIT_LINESTYLES.get(split, "-"), label=split_display[split])
            for split in split_strategies
        ]
        fig.legend(
            handles=model_handles,
            title="Model parameter",
            loc="lower center",
            bbox_to_anchor=(0.28, 0.01),
            ncol=min(4, len(model_params)),
            frameon=True,
        )
        fig.legend(
            handles=split_handles,
            title="Split strategy",
            loc="lower center",
            bbox_to_anchor=(0.78, 0.01),
            ncol=min(2, len(split_strategies)),
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])

        out_path = output_dir / "plots" / f"{tissue_slug(tissue)}_all_participants_epoch_trend_seq_{seq}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        if show_plots:
            plt.show()
        plt.close(fig)
        print(f"Wrote plot: {out_path}")


def main() -> None:
    args = parse_args()
    plt.style.use("seaborn-v0_8-whitegrid")

    tidy_df, atlas_df, file_index_df = load_projects(args.project_dirs, fallback_split_strategy=args.split_strategy)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tidy_df.to_csv(args.output_dir / "tidy_results.csv", index=False)
    atlas_df.to_csv(args.output_dir / "atlas_tidy_results.csv", index=False)
    file_index_df.to_csv(args.output_dir / "file_index.csv", index=False)

    print("=== Shard Visualization ===")
    print(f"project dirs: {len(args.project_dirs)}")
    print(f"parsed rows: {len(tidy_df)}")
    print(f"atlas rows: {len(atlas_df)}")
    print(f"unique projects: {tidy_df['project_name'].nunique() if not tidy_df.empty else 0}")

    if tidy_df.empty:
        print("No parsed result rows were collected.")
        return

    tissue_meta = tidy_df[["tissue"]].drop_duplicates().sort_values(["tissue"])

    for row in tissue_meta.itertuples(index=False):
        tissue_output_dir = args.output_dir / tissue_slug(row.tissue)
        tissue_output_dir.mkdir(parents=True, exist_ok=True)

        plot_final_epoch_mean_by_variability(
            tidy_df,
            atlas_df,
            tissue=row.tissue,
            output_dir=tissue_output_dir,
            show_plots=args.show_plots,
        )
        plot_epoch_trend(
            tidy_df,
            tissue=row.tissue,
            output_dir=tissue_output_dir,
            show_plots=args.show_plots,
        )

    print(f"\nWrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
