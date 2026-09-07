#!/usr/bin/env python3
"""Generate browser tracks directly from labeled 6-mer prediction rows.

Progress is printed and flushed after every completed sample and major output stage.

Examples:
  # Discover standard test predictions and create priority high-variability tracks:
  python3 scripts/generate_methylation_kmer_tracks.py \
    --results-root scripts/results/downsampled_standard_ratio_0p25_lr_3e-05 \
    --output-dir scripts/results/methylation_kmer_tracks

  # Participant/chromosome pilot with participant-specific categorical tracks:
  python3 scripts/generate_methylation_kmer_tracks.py \
    --results-root scripts/results/downsampled_standard_ratio_0p25_lr_3e-05 \
    --output-dir scripts/results/methylation_kmer_tracks_Z0000044P_chr21 \
    --participant Z0000044P --chrom chr21 --participant-tracks

  # Atlas-versus-retraining participant/chromosome pilot:
  python3 scripts/generate_methylation_kmer_tracks.py \
    --results-root scripts/results/downsampled_standard_ratio_0p25_lr_3e-05 \
    --output-dir scripts/results/methylation_kmer_tracks_Z0000044P_chr21_retrained_vs_atlas \
    --participant Z0000044P --chrom chr21 --participant-tracks --baseline atlas

  # Tissue-level atlas comparison requiring at least three participants per k-mer:
  python3 scripts/generate_methylation_kmer_tracks.py \
    --results-root scripts/results/downsampled_standard_ratio_0p25_lr_3e-05 \
    --output-dir analysis/Heart-Cardiomyocyte_chr21_retrained_vs_atlas \
    --tissue Heart-Cardiomyocyte --chrom chr21 --baseline atlas \
    --min-participants-per-kmer 3

  # Separate tissue tracks plus an aggregate across all samples:
  python3 scripts/generate_methylation_kmer_tracks.py \
    --results-root scripts/results/downsampled_standard_ratio_0p25_lr_3e-05 \
    --output-dir analysis/all_tissues_and_combined_chr21 \
    --chrom chr21 --baseline atlas --combined-samples-track \
    --min-participants-per-kmer 3 --workers 4

  # All available chromosomes, each tissue plus all samples, versus atlas:
  python3 scripts/generate_methylation_kmer_tracks.py \
    --results-root scripts/results/downsampled_standard_ratio_0p25_lr_3e-05 \
    --output-dir /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/methylation_kmer_tracks_all_tissues_and_combined_all_chromosomes_retrained_vs_atlas \
    --baseline atlas --combined-samples-track \
    --min-participants-per-kmer 3 --workers 4 --ucsc-item-details

  # All available chromosomes, each tissue plus all samples, versus pretrained:
  python3 scripts/generate_methylation_kmer_tracks.py \
    --results-root scripts/results/downsampled_standard_ratio_0p25_lr_3e-05 \
    --output-dir /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/methylation_kmer_tracks_all_tissues_and_combined_all_chromosomes_retrained_vs_pretrained \
    --baseline pretrained --combined-samples-track \
    --min-participants-per-kmer 3 --workers 4 --ucsc-item-details

  # Combined standardized chr12 eval+test tracks versus atlas:
  python3 scripts/generate_methylation_kmer_tracks.py \
    --results-root scripts/results/chr12_standardized_results \
    --split combined --variability-root /path/to/chr12_variability \
    --chrom chr12 --output-dir analysis/chr12_combined_vs_atlas \
    --baseline atlas --combined-samples-track \
    --min-participants-per-kmer 3 --workers 4 --ucsc-item-details

  # Combined standardized chr12 eval+test tracks versus pretrained:
  python3 scripts/generate_methylation_kmer_tracks.py \
    --results-root scripts/results/chr12_standardized_results \
    --split combined --variability-root /path/to/chr12_variability \
    --chrom chr12 --output-dir analysis/chr12_combined_vs_pretrained \
    --baseline pretrained --combined-samples-track \
    --min-participants-per-kmer 3 --workers 4 --ucsc-item-details

  # Explicit manifest and selected variability views:
  python3 scripts/generate_methylation_kmer_tracks.py \
    --manifest analysis/track_manifest.tsv --output-dir analysis/kmer_tracks \
    --bin-group bin4 --bin-group bin5 --bin-group bins4-5
"""

from __future__ import annotations

import argparse
import gzip
from concurrent.futures import as_completed, ThreadPoolExecutor
from functools import partial
import html
from pathlib import Path
from typing import Iterable, TextIO

import numpy as np
import pandas as pd

try:
    from generate_methylation_browser_tracks import (
        DEFAULT_VARIABILITY_ROOT,
        add_variability,
        add_window_coordinates,
        chromosome_key,
        discover_manifest,
        expand_bin_groups,
        filter_manifest,
        load_manifest,
        prepare_output_dir,
        read_table,
        safe_name,
        validate_numeric_values,
    )
except ModuleNotFoundError:
    from scripts.generate_methylation_browser_tracks import (
        DEFAULT_VARIABILITY_ROOT,
        add_variability,
        add_window_coordinates,
        chromosome_key,
        discover_manifest,
        expand_bin_groups,
        filter_manifest,
        load_manifest,
        prepare_output_dir,
        read_table,
        safe_name,
        validate_numeric_values,
    )

PAIR_KEYS = ["window_id", "genomic_position"]
DEFAULT_BIN_GROUPS = [
    "bin1", "bin2", "bin3", "bin4", "bin5", "bins3-5", "bins4-5"
]
BASELINES = ("pretrained", "atlas")
PREDICTION_COLUMNS = [
    "window_id", "window_start", "window_end", "token_index",
    "token_position_in_window", "base_position_in_window", "genomic_position",
    "label", "predicted_class", "probability_class_0",
    "probability_class_1", "probability_class_2",
]
STATUS_COLORS = {
    "rescued": "0,140,0", "harmed": "190,30,45",
    "both_correct": "50,100,200", "both_wrong": "90,90,90",
    "mixed_or_tied": "128,128,128", "favors_retrained": "0,140,0",
    "favors_baseline": "190,30,45",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--results-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split", choices=("eval", "test", "combined"), default="test"
    )
    parser.add_argument("--tissue", action="append", dest="tissues")
    parser.add_argument("--participant", action="append", dest="participants")
    parser.add_argument("--chrom", action="append", dest="chroms")
    parser.add_argument("--bin-group", action="append", dest="bin_groups")
    parser.add_argument(
        "--baseline", action="append", dest="baselines", choices=BASELINES,
        help="Comparison baseline; repeat to select both (default: all available).",
    )
    parser.add_argument("--kmer-size", type=int, default=6)
    parser.add_argument("--min-participants-per-kmer", type=int, default=1)
    parser.add_argument("--min-variability-match-rate", type=float, default=0.5)
    parser.add_argument(
        "--variability-root", type=Path, default=DEFAULT_VARIABILITY_ROOT,
        help="Root containing participant variability CSVs grouped by tissue",
    )
    parser.add_argument("--participant-tracks", action="store_true")
    parser.add_argument("--combined-samples-track", action="store_true")
    parser.add_argument(
        "--ucsc-item-details",
        action="store_true",
        help="Write compressed per-track bedDetail files with click descriptions.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--genome", default="hg38")
    return parser.parse_args()


def read_prediction_columns(path: Path) -> pd.DataFrame:
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    available = pd.read_csv(path, nrows=0).columns
    missing = set(PREDICTION_COLUMNS).difference(available)
    if missing:
        raise ValueError(f"{path} is missing k-mer columns: {sorted(missing)}")
    return pd.read_csv(path, usecols=PREDICTION_COLUMNS)


def integer_column(frame: pd.DataFrame, column: str, path: Path) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = values.isna() | ~np.isfinite(values) | (values % 1 != 0)
    if invalid.any():
        raise ValueError(f"{path} contains invalid integer {column} values")
    return values.astype("int64")


def true_class_probability(frame: pd.DataFrame, path: Path) -> pd.Series:
    labels = integer_column(frame, "label", path)
    columns = ["probability_class_0", "probability_class_1", "probability_class_2"]
    validate_numeric_values(frame, columns, path)
    if (~labels.isin(range(len(columns)))).any():
        raise ValueError(f"{path} contains labels without a probability column")
    probabilities = frame[columns].to_numpy(dtype=float)
    values = probabilities[np.arange(len(frame)), labels.to_numpy()]
    return pd.Series(values, index=frame.index, dtype=float)


def validate_kmer_coordinates(frame: pd.DataFrame, path: Path, kmer_size: int) -> None:
    token_position = integer_column(frame, "token_position_in_window", path)
    base_position = integer_column(frame, "base_position_in_window", path)
    genomic_position = integer_column(frame, "genomic_position", path)
    if not (base_position == token_position * kmer_size).all():
        raise ValueError(f"{path} token/base positions do not describe {kmer_size}-mers")
    if not (genomic_position == frame["window_start"] + base_position).all():
        raise ValueError(f"{path} genomic positions disagree with window/token positions")
    if (genomic_position + kmer_size > frame["window_end"]).any():
        raise ValueError(f"{path} contains k-mers extending beyond their window")


def load_kmer_predictions(path: Path, kmer_size: int) -> pd.DataFrame:
    frame = add_window_coordinates(read_prediction_columns(path), path)
    validate_numeric_values(frame, ["label", "predicted_class"], path)
    validate_kmer_coordinates(frame, path, kmer_size)
    if frame.duplicated(PAIR_KEYS).any():
        raise ValueError(f"{path} has duplicate window/genomic-position keys")
    frame["genomic_position"] = integer_column(frame, "genomic_position", path)
    frame["label"] = integer_column(frame, "label", path)
    frame["predicted_class"] = integer_column(frame, "predicted_class", path)
    frame["true_class_probability"] = true_class_probability(frame, path)
    frame["kmer_start"] = frame["genomic_position"]
    frame["kmer_end"] = frame["kmer_start"] + kmer_size
    return frame


def same_values(left: pd.Series, right: pd.Series) -> bool:
    return np.array_equal(left.to_numpy(), right.to_numpy())


def retrained_prediction_frame(row: pd.Series, kmer_size: int) -> pd.DataFrame:
    retrained = load_kmer_predictions(row["retrained_path"], kmer_size)
    columns = PAIR_KEYS + [
        "chrom", "window_start", "window_end", "kmer_start", "kmer_end", "label",
        "predicted_class", "true_class_probability",
    ]
    result = retrained[columns].rename(columns={
        "predicted_class": "retrained_prediction",
        "true_class_probability": "retrained_true_probability",
    })
    result["participant"] = row["participant"]
    result["tissue"] = row["tissue"]
    return result


def attach_pretrained_predictions(
    retrained: pd.DataFrame, row: pd.Series, kmer_size: int
) -> pd.DataFrame:
    pretrained = load_kmer_predictions(row["pretrained_path"], kmer_size)
    columns = PAIR_KEYS + ["label", "predicted_class", "true_class_probability"]
    baseline = pretrained[columns].rename(columns={
        "label": "pretrained_label", "predicted_class": "pretrained_prediction",
        "true_class_probability": "pretrained_true_probability",
    })
    paired = retrained.merge(
        baseline, on=PAIR_KEYS, how="outer", indicator=True, validate="one_to_one"
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError(f"Prediction keys differ for {row['participant']}")
    paired = paired.drop(columns="_merge")
    if not same_values(paired["label"], paired["pretrained_label"]):
        raise ValueError(f"Prediction labels differ for {row['participant']}")
    return paired.drop(columns="pretrained_label")


def paired_prediction_frame(
    row: pd.Series, kmer_size: int, include_pretrained: bool = True
) -> pd.DataFrame:
    retrained = retrained_prediction_frame(row, kmer_size)
    if not include_pretrained:
        return retrained
    return attach_pretrained_predictions(retrained, row, kmer_size)


def mean_baseline_values(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Missing mean-baseline file: {path}")
    frame = read_table(path)
    required = set(PAIR_KEYS + ["mean_label"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks mean-baseline columns: {sorted(missing)}")
    if frame.duplicated(PAIR_KEYS).any():
        raise ValueError(f"{path} has duplicate mean-baseline keys")
    validate_numeric_values(frame, ["mean_label"], path)
    columns = PAIR_KEYS + ["mean_label"]
    if "label" in frame:
        columns.append("label")
    return frame[columns].rename(columns={"label": "mean_truth"})


def attach_mean_baseline(frame: pd.DataFrame, path: Path | None) -> None:
    baseline = mean_baseline_values(path)
    if baseline is None:
        frame["mean_prediction"] = np.nan
        return
    merged = frame[PAIR_KEYS + ["label"]].merge(
        baseline, on=PAIR_KEYS, how="left", validate="one_to_one"
    )
    if "mean_truth" in merged:
        matched = merged["mean_truth"].notna()
        if not same_values(merged.loc[matched, "label"], merged.loc[matched, "mean_truth"]):
            raise ValueError(f"Labels differ from mean baseline {path}")
    frame["mean_prediction"] = merged["mean_label"].to_numpy()


def status_values(retrained_correct: pd.Series, baseline_correct: pd.Series) -> pd.Series:
    conditions = [
        retrained_correct & ~baseline_correct,
        ~retrained_correct & baseline_correct,
        retrained_correct & baseline_correct,
    ]
    choices = ["rescued", "harmed", "both_correct"]
    return pd.Series(
        np.select(conditions, choices, default="both_wrong"),
        index=retrained_correct.index,
    )


def baseline_column(baseline: str) -> str:
    return "mean_prediction" if baseline == "atlas" else "pretrained_prediction"


def comparison_name(baseline: str) -> str:
    return "retrained_vs_atlas" if baseline == "atlas" else "retrained_minus_pretrained"


def comparison_frame(frame: pd.DataFrame, baseline: str) -> pd.DataFrame:
    column = baseline_column(baseline)
    result = frame.loc[frame[column].notna()].copy()
    retrained_correct = result["retrained_prediction"].eq(result["label"])
    baseline_correct = result[column].eq(result["label"])
    result["comparison"] = comparison_name(baseline)
    result["baseline_prediction"] = result[column]
    result["correctness_delta"] = retrained_correct.astype(int) - baseline_correct.astype(int)
    result["status"] = status_values(retrained_correct, baseline_correct)
    if baseline == "pretrained":
        result["true_probability_delta"] = (
            result["retrained_true_probability"] - result["pretrained_true_probability"]
        )
    else:
        result["true_probability_delta"] = np.nan
    return result


def comparison_rows(
    frame: pd.DataFrame, baselines: list[str] | None = None
) -> pd.DataFrame:
    selected = baselines or list(BASELINES)
    pieces = []
    for baseline in selected:
        if frame[baseline_column(baseline)].notna().any():
            pieces.append(comparison_frame(frame, baseline))
        elif baselines is not None:
            raise ValueError(f"No {baseline} baseline predictions are available")
    return pd.concat(pieces, ignore_index=True)


def participant_qc(
    frame: pd.DataFrame, row: pd.Series, minimum_rate: float
) -> dict[str, object]:
    matched = frame["variability_bin"].notna()
    rate = float(matched.mean()) if len(frame) else 0.0
    if rate < minimum_rate:
        raise ValueError(
            f"Variability match rate {rate:.4f} below {minimum_rate:.4f} "
            f"for {row['participant']}"
        )
    return {
        "participant": row["participant"], "tissue": row["tissue"],
        "split": row.get("split", "unspecified"),
        "paired_kmers": len(frame),
        "matched_variability_kmers": int(matched.sum()),
        "unmatched_variability_kmers": int((~matched).sum()),
        "variability_match_rate": rate,
        "mean_baseline_kmers": int(frame["mean_prediction"].notna().sum()),
    }


def participant_kmers(
    row: pd.Series,
    kmer_size: int,
    chroms: list[str] | None,
    minimum_rate: float,
    baselines: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    include_pretrained = baselines is None or "pretrained" in baselines
    paired = paired_prediction_frame(row, kmer_size, include_pretrained)
    if chroms:
        paired = paired[paired["chrom"].isin(chroms)].copy()
    if paired.empty:
        raise ValueError(f"No k-mers remain for {row['participant']} after chromosome filtering")
    attach_mean_baseline(paired, row.get("mean_path"))
    variable = add_variability(paired, row["variability_path"], chroms)
    qc = participant_qc(variable, row, minimum_rate)
    sites = variable.dropna(subset=["variability_bin"])
    return comparison_rows(sites, baselines), qc


def add_indicator_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for status in ("rescued", "harmed", "both_correct", "both_wrong"):
        result[f"n_{status}"] = result["status"].eq(status).astype(int)
    for label in (0, 1, 2):
        result[f"n_label_{label}"] = result["label"].eq(label).astype(int)
    return result


def aggregate_kmers(frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_indicator_columns(frame)
    keys = ["tissue", "chrom", "kmer_start", "kmer_end", "bin_group", "comparison"]
    aggregate = frame.groupby(keys, observed=True, as_index=False).agg(
        correctness_delta=("correctness_delta", "mean"),
        correctness_standard_error=("correctness_delta", "std"),
        true_probability_delta=("true_probability_delta", "mean"),
        probability_standard_error=("true_probability_delta", "std"),
        n_probability_observations=("true_probability_delta", "count"),
        n_individuals=("participant", "nunique"),
        n_observations=("correctness_delta", "size"),
        n_rescued=("n_rescued", "sum"), n_harmed=("n_harmed", "sum"),
        n_both_correct=("n_both_correct", "sum"),
        n_both_wrong=("n_both_wrong", "sum"),
        n_label_0=("n_label_0", "sum"), n_label_1=("n_label_1", "sum"),
        n_label_2=("n_label_2", "sum"),
    )
    aggregate["correctness_standard_error"] /= np.sqrt(aggregate["n_observations"])
    probability_count = aggregate["n_probability_observations"].replace(0, np.nan)
    aggregate["probability_standard_error"] /= np.sqrt(probability_count)
    aggregate = aggregate.drop(columns="n_probability_observations")
    return sort_kmers(aggregate)


def filter_by_participant_coverage(
    summary: pd.DataFrame, minimum_participants: int
) -> pd.DataFrame:
    if minimum_participants < 1:
        raise ValueError("minimum participants per k-mer must be positive")
    result = summary[summary["n_individuals"] >= minimum_participants].copy()
    if result.empty:
        raise ValueError(
            f"No k-mers have at least {minimum_participants} contributing participants"
        )
    return result


def aggregate_with_combined_samples(
    frame: pd.DataFrame, include_combined: bool
) -> pd.DataFrame:
    summary = aggregate_kmers(frame)
    if not include_combined:
        return summary
    combined = frame.copy()
    combined["tissue"] = "All-samples"
    return sort_kmers(pd.concat([summary, aggregate_kmers(combined)], ignore_index=True))


def sort_kmers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_chrom_key"] = result["chrom"].map(chromosome_key)
    columns = [
        "tissue", "comparison", "bin_group", "_chrom_key", "kmer_start", "kmer_end"
    ]
    return result.sort_values(columns, kind="stable").drop(columns="_chrom_key")


def aggregate_class(value: float) -> str:
    if np.isclose(value, 0.0):
        return "mixed_or_tied"
    return "favors_retrained" if value > 0 else "favors_baseline"


def write_bedgraph(group: pd.DataFrame, path: Path, column: str) -> None:
    selected = group.dropna(subset=[column])
    with path.open("w", encoding="utf-8") as handle:
        previous: dict[str, int] = {}
        for row in selected.itertuples():
            start, end = int(row.kmer_start), int(row.kmer_end)
            if start < previous.get(row.chrom, -1):
                raise ValueError(f"Overlapping or unsorted k-mers in {path}")
            handle.write(f"{row.chrom}\t{start}\t{end}\t{getattr(row, column):.8g}\n")
            previous[row.chrom] = end


def aggregate_bed_name(row) -> str:
    return f"n{int(row.n_individuals)}_r{int(row.n_rescued)}_h{int(row.n_harmed)}"


def aggregate_bed_description(row) -> str:
    outcomes = {
        "mixed_or_tied": "Same accuracy",
        "favors_retrained": "Retraining better",
        "favors_baseline": "Baseline better",
    }
    outcome = outcomes[aggregate_class(row.correctness_delta)]
    return (
        f"{outcome} | Participants: {int(row.n_individuals)}; "
        f"Rescued: {int(row.n_rescued)}; "
        f"Harmed: {int(row.n_harmed)}; Both correct: {int(row.n_both_correct)}; "
        f"Both wrong: {int(row.n_both_wrong)}"
    )


def aggregate_bed_fields(row) -> list[object]:
    result_class = aggregate_class(row.correctness_delta)
    score = int(round(min(abs(row.correctness_delta), 1.0) * 1000))
    return [
        row.chrom, int(row.kmer_start), int(row.kmer_end),
        aggregate_bed_name(row), score, ".", int(row.kmer_start),
        int(row.kmer_end), STATUS_COLORS[result_class],
    ]


def write_aggregate_bed(group: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in group.itertuples():
            handle.write("\t".join(map(str, aggregate_bed_fields(row))) + "\n")


def display_prediction(value: object) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"


def baseline_item_key(comparison: str) -> str:
    return "atlas" if comparison == "retrained_vs_atlas" else "pre"


def write_participant_bed(group: pd.DataFrame, path: Path) -> None:
    group = sort_kmers(group)
    with path.open("w", encoding="utf-8") as handle:
        for row in group.itertuples():
            score = 1000 if row.status in {"rescued", "harmed"} else 0
            baseline_key = baseline_item_key(row.comparison)
            name = safe_name(
                f"{row.status}|label={display_prediction(row.label)}|"
                f"{baseline_key}={display_prediction(row.baseline_prediction)}|"
                f"retrain={display_prediction(row.retrained_prediction)}"
            )
            fields = [
                row.chrom, int(row.kmer_start), int(row.kmer_end), name, score, ".",
                int(row.kmer_start), int(row.kmer_end), STATUS_COLORS[row.status],
            ]
            handle.write("\t".join(map(str, fields)) + "\n")


def write_tissue_tracks(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    tracks_dir = output_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    keys = ["tissue", "comparison", "bin_group"]
    for values, group in summary.groupby(keys, observed=True):
        tissue, comparison, bin_group = values
        group = sort_kmers(group)
        prefix = f"{safe_name(tissue)}.{comparison}.{bin_group}"
        correctness = tracks_dir / f"{prefix}.correctness_delta.bedGraph"
        probability = tracks_dir / f"{prefix}.true_probability_delta.bedGraph"
        statuses = tracks_dir / f"{prefix}.status.bed"
        write_bedgraph(group, correctness, "correctness_delta")
        write_aggregate_bed(group, statuses)
        paths.extend([correctness, statuses])
        if group["true_probability_delta"].notna().any():
            write_bedgraph(group, probability, "true_probability_delta")
            paths.append(probability)
    return paths


def write_ucsc_item_details(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    directory = output_dir / "ucsc_status_tracks"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    keys = ["tissue", "comparison", "bin_group"]
    for values, group in summary.groupby(keys, observed=True):
        tissue, comparison, bin_group = values
        prefix = f"{safe_name(tissue)}.{comparison}.{bin_group}"
        path = directory / f"{prefix}.status.bedDetail.txt.gz"
        track_name = f"{prefix}.status"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(
                f'track type=bedDetail name="{track_name}" '
                'visibility=pack itemRgb=On\n'
            )
            for row in sort_kmers(group).itertuples():
                item_id = f"{row.chrom}:{int(row.kmer_start)}-{int(row.kmer_end)}"
                fields = aggregate_bed_fields(row)
                fields.extend([item_id, aggregate_bed_description(row)])
                handle.write("\t".join(map(str, fields)) + "\n")
        paths.append(path)
    return paths


def write_participant_tracks(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    directory = output_dir / "participant_tracks"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    keys = ["participant", "tissue", "comparison", "bin_group"]
    for values, group in frame.groupby(keys, observed=True):
        participant, tissue, comparison, bin_group = values
        name = f"{safe_name(tissue)}.{participant}.{comparison}.{bin_group}.status.bed"
        path = directory / name
        write_participant_bed(group, path)
        paths.append(path)
    return paths


def write_igv_session(paths: Iterable[Path], output_dir: Path, genome: str) -> Path:
    resources = "\n".join(
        f'    <Resource path="{html.escape(path.relative_to(output_dir).as_posix())}" />'
        for path in paths
    )
    content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<Session genome="{html.escape(genome)}" version="8">
  <Resources>
{resources}
  </Resources>
</Session>
"""
    path = output_dir / "igv_session.xml"
    path.write_text(content, encoding="utf-8")
    return path


def custom_track_header(path: Path) -> str:
    name = path.stem
    if path.suffix == ".bed":
        return f'track type=bed name="{name}" visibility=pack itemRgb=On\n'
    return (
        f'track type=bedGraph name="{name}" visibility=full '
        'color=0,140,0 altColor=190,30,45\n'
    )


def copy_track_rows(path: Path, output: TextIO) -> None:
    expected = 9 if path.suffix == ".bed" else 4
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != expected:
                raise ValueError(
                    f"{path}:{line_number} has {len(fields)} fields, expected {expected}"
                )
            output.write(line)


def write_ucsc_custom_tracks(
    paths: Iterable[Path],
    output_dir: Path,
    filename: str = "ucsc_custom_tracks.txt",
) -> Path:
    destination = output_dir / filename
    with destination.open("w", encoding="utf-8") as output:
        for path in paths:
            output.write(custom_track_header(path))
            copy_track_rows(path, output)
    return destination


def report_text(
    summary: pd.DataFrame,
    qc: pd.DataFrame,
    kmer_size: int,
    groups: list[str],
) -> str:
    chromosomes = sorted(summary["chrom"].unique(), key=chromosome_key)
    return "\n".join([
        "# K-mer methylation browser tracks", "",
        f"Each browser interval is one validated {kmer_size}-base token: "
        "`[genomic_position, genomic_position + kmer_size)`. The containing 5.4 kb "
        "window is context only and is not the aggregation unit.", "",
        "Correctness delta is retrained correctness minus baseline correctness: +1 rescued, "
        "-1 harmed, and 0 means both models were correct or both were wrong. Tissue tracks "
        "show the mean across available participants at the exact k-mer coordinate.", "",
        "True-probability delta is retrained minus pretrained probability assigned to the "
        "observed class. It is unavailable for the atlas (mean-label) baseline.", "",
        f"Variability views: {', '.join(groups)}.",
        f"Chromosome scope: {', '.join(chromosomes)}. This is input/filter coverage, not whole-genome coverage.",
        f"Participants: {qc['participant'].nunique()}; tissues: {qc['tissue'].nunique()}; "
        f"aggregated k-mer rows: {len(summary)}.", "",
        f"Browser coverage filter: at least "
        f"{int(qc['minimum_participants_per_kmer'].iloc[0])} participants per k-mer; "
        f"displayed rows: {int(qc['browser_kmer_rows'].iloc[0])}.", "",
        "`kmer_summary.tsv.gz` contains the complete rescued, harmed, both-correct, and "
        "both-wrong counts plus uncertainty. Aggregate status BED names use `n#_r#_h#`, "
        "where n is participants, r is rescued, and h is harmed. Green items have r > h, "
        "red have h > r, and gray have r = h. These are model-result tracks, not "
        "methylation beta values.",
    ])


def report_progress(message: str) -> None:
    print(f"[kmer-tracks] {message}", flush=True)


def run_participant_jobs(manifest: pd.DataFrame, worker, workers: int):
    rows = [row for _, row in manifest.iterrows()]
    total = len(rows)
    report_progress(f"Processing {total} samples with {workers} worker(s)")
    if workers == 1:
        results = []
        for completed, row in enumerate(rows, start=1):
            results.append(worker(row))
            report_progress(
                f"Samples completed: {completed}/{total} ({row['participant']})"
            )
        return results
    ordered_results = [None] * total
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(worker, row): (position, row["participant"])
            for position, row in enumerate(rows)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            position, participant = futures[future]
            ordered_results[position] = future.result()
            report_progress(
                f"Samples completed: {completed}/{total} ({participant})"
            )
    return ordered_results


def validate_disjoint_input_windows(results) -> None:
    seen: dict[tuple[str, str], set[str]] = {}
    for frame, _ in results:
        identities = frame.groupby(["participant", "tissue"], observed=True)
        for identity, group in identities:
            window_ids = set(group["window_id"].astype(str).unique())
            overlap = seen.setdefault(identity, set()).intersection(window_ids)
            if overlap:
                raise ValueError(
                    f"Input splits overlap by {len(overlap)} windows for "
                    f"{identity[0]} ({identity[1]})"
                )
            seen[identity].update(window_ids)


def build_tracks(
    manifest: pd.DataFrame,
    output_dir: Path,
    kmer_size: int = 6,
    chroms: list[str] | None = None,
    bin_groups: list[str] | None = None,
    baselines: list[str] | None = None,
    minimum_participants_per_kmer: int = 1,
    minimum_match_rate: float = 0.5,
    participant_tracks: bool = False,
    combined_samples_track: bool = False,
    ucsc_item_details: bool = False,
    workers: int = 1,
    genome: str = "hg38",
) -> None:
    if kmer_size <= 0:
        raise ValueError("k-mer size must be positive")
    if not 0 <= minimum_match_rate <= 1:
        raise ValueError("minimum variability match rate must be between zero and one")
    if workers < 1:
        raise ValueError("workers must be positive")
    groups = bin_groups or DEFAULT_BIN_GROUPS
    chromosome_scope = ", ".join(chroms) if chroms else "all available chromosomes"
    report_progress(
        f"Starting {len(manifest)} samples across "
        f"{manifest['tissue'].nunique()} tissues; scope: {chromosome_scope}"
    )
    prepare_output_dir(output_dir)
    worker = partial(
        participant_kmers,
        kmer_size=kmer_size,
        chroms=chroms,
        minimum_rate=minimum_match_rate,
        baselines=baselines,
    )
    results = run_participant_jobs(manifest, worker, workers)
    validate_disjoint_input_windows(results)
    report_progress("Expanding variability-bin views")
    frames = [expand_bin_groups(sites, groups) for sites, _ in results]
    qc_rows = [qc for _, qc in results]
    participant_frame = pd.concat(frames, ignore_index=True)
    duplicate_keys = [
        "participant", "tissue", "chrom", "kmer_start", "bin_group", "comparison"
    ]
    if participant_frame.duplicated(duplicate_keys).any():
        raise ValueError("A participant has duplicate k-mer loci after bin expansion")
    report_progress("Aggregating tissue and combined-sample summaries")
    summary = aggregate_with_combined_samples(
        participant_frame, combined_samples_track
    )
    browser_summary = filter_by_participant_coverage(
        summary, minimum_participants_per_kmer
    )
    qc = pd.DataFrame(qc_rows)
    qc["minimum_participants_per_kmer"] = minimum_participants_per_kmer
    qc["browser_kmer_rows"] = len(browser_summary)
    report_progress("Writing summary tables")
    summary.to_csv(
        output_dir / "kmer_summary.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    browser_summary.to_csv(
        output_dir / "browser_kmer_summary.tsv.gz", sep="\t", index=False,
        compression="gzip",
    )
    qc.to_csv(output_dir / "qc_summary.tsv", sep="\t", index=False)
    manifest.to_csv(output_dir / "selected_manifest.tsv", sep="\t", index=False)
    report_progress("Writing tissue browser tracks")
    paths = write_tissue_tracks(browser_summary, output_dir)
    report_progress(f"Wrote {len(paths)} tissue track files")
    if ucsc_item_details:
        report_progress("Writing compressed UCSC item-detail subtracks")
        detail_paths = write_ucsc_item_details(browser_summary, output_dir)
        report_progress(f"Wrote {len(detail_paths)} UCSC item-detail subtracks")
    categorical_paths = [path for path in paths if path.suffix == ".bed"]
    if participant_tracks:
        report_progress("Writing participant summaries and tracks")
        participant_frame.to_csv(
            output_dir / "participant_kmer_summary.tsv.gz", sep="\t", index=False,
            compression="gzip",
        )
        participant_paths = write_participant_tracks(participant_frame, output_dir)
        paths.extend(participant_paths)
        categorical_paths = participant_paths
        report_progress(f"Wrote {len(participant_paths)} participant track files")
    report_progress("Writing IGV session and UCSC custom-track bundles")
    write_igv_session(paths, output_dir, genome)
    write_ucsc_custom_tracks(paths, output_dir)
    write_ucsc_custom_tracks(
        categorical_paths,
        output_dir,
        filename="ucsc_categorical_tracks.txt",
    )
    (output_dir / "REPORT.md").write_text(
        report_text(summary, qc, kmer_size, groups), encoding="utf-8"
    )
    report_progress(f"Finished writing outputs to {output_dir}")


def selected_manifest(args: argparse.Namespace) -> pd.DataFrame:
    manifest = (
        load_manifest(args.manifest)
        if args.manifest
        else discover_manifest(args.results_root, args.split, args.variability_root)
    )
    return filter_manifest(manifest, args.tissues, args.participants)


def main() -> None:
    args = parse_args()
    build_tracks(
        manifest=selected_manifest(args),
        output_dir=args.output_dir,
        kmer_size=args.kmer_size,
        chroms=args.chroms,
        bin_groups=args.bin_groups,
        baselines=args.baselines,
        minimum_participants_per_kmer=args.min_participants_per_kmer,
        minimum_match_rate=args.min_variability_match_rate,
        participant_tracks=args.participant_tracks,
        combined_samples_track=args.combined_samples_track,
        ucsc_item_details=args.ucsc_item_details,
        workers=args.workers,
        genome=args.genome,
    )
    print(f"Wrote k-mer browser tracks to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
