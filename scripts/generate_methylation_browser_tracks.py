#!/usr/bin/env python3
"""Create genome-browser tracks from paired methylation predictions.

Examples:
  # Full comparison, including a mean-baseline CSV or pickle in each manifest row:
  python3 scripts/generate_methylation_browser_tracks.py \
    --manifest analysis/track_manifest.tsv --output-dir analysis/browser_tracks

  # Discover standard test results; latest numeric checkpoint is selected without test metrics:
  python3 scripts/generate_methylation_browser_tracks.py \
    --results-root scripts/results/downsampled_standard_ratio_0p25_lr_3e-05 \
    --split test --output-dir scripts/results/methylation_browser_tracks

  # Pretrained comparison only (the manifest has no mean_path column):
  python3 scripts/generate_methylation_browser_tracks.py \
    --manifest analysis/track_manifest_no_mean.tsv --output-dir analysis/browser_tracks

  # Treat predictions as continuous values and compare absolute error:
  python3 scripts/generate_methylation_browser_tracks.py \
    --manifest analysis/track_manifest.tsv --output-dir analysis/browser_tracks_mae \
    --metric mae --prediction-column prediction --genome hg38

  # Limit the metadata BED while retaining complete compressed audits and bedGraphs:
  python3 scripts/generate_methylation_browser_tracks.py \
    --manifest analysis/track_manifest.tsv --output-dir analysis/browser_tracks_bounded \
    --max-regions-per-track 500
"""

from __future__ import annotations

import argparse
import ast
import html
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_MANIFEST_COLUMNS = {
    "participant",
    "tissue",
    "retrained_path",
    "pretrained_path",
    "variability_path",
}
KEY_COLUMNS = ["window_id", "genomic_position"]
EXACT_BIN_GROUPS = {f"bin{i}": {i} for i in range(1, 6)}
COMBINED_BIN_GROUPS = {"bins3-5": {3, 4, 5}, "bins4-5": {4, 5}, "all": {1, 2, 3, 4, 5}}
BED_COLUMNS = [
    "chrom",
    "start",
    "end",
    "name",
    "score",
    "strand",
    "thickStart",
    "thickEnd",
    "itemRgb",
    "tissue",
    "bin_group",
    "comparison",
    "improvement",
    "standard_error",
    "p_value",
    "q_value",
    "n_individuals",
    "n_observations",
    "n_sites",
    "result_class",
    "evidence_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--results-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("eval", "test"), default="test")
    parser.add_argument("--tissue", action="append", dest="tissues")
    parser.add_argument("--participant", action="append", dest="participants")
    parser.add_argument("--chrom", action="append", dest="chroms")
    parser.add_argument("--bin-group", action="append", dest="bin_groups")
    parser.add_argument("--metric", choices=("accuracy", "mae"), default="accuracy")
    parser.add_argument("--prediction-column", default="predicted_class")
    parser.add_argument("--genome", default="hg38", help="IGV genome identifier")
    parser.add_argument("--min-variability-match-rate", type=float, default=0.5)
    parser.add_argument(
        "--max-regions-per-track",
        type=int,
        default=1000,
        help="Maximum BED regions per tissue/bin/comparison; audits remain complete",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path)


def read_prediction_table(path: Path, prediction_column: str) -> pd.DataFrame:
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    columns = pd.read_csv(path, nrows=0).columns
    wanted = set(KEY_COLUMNS + ["label", prediction_column, "window_start", "window_end", "mean_label"])
    return pd.read_csv(path, usecols=[column for column in columns if column in wanted])


def resolve_path(value: object, base_dir: Path) -> Path | None:
    if pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, sep=None, engine="python", dtype=str)
    missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("Manifest has no participant rows")
    if manifest.duplicated(["participant", "tissue"]).any():
        raise ValueError("Manifest participant/tissue pairs must be unique")
    base_dir = path.resolve().parent
    for column in ("retrained_path", "pretrained_path", "variability_path", "mean_path"):
        if column in manifest:
            manifest[column] = manifest[column].map(lambda value: resolve_path(value, base_dir))
    return manifest


def checkpoint_number(path: str) -> int | None:
    match = re.search(r"(?:^|/)checkpoint-(\d+)(?:/|$)", path)
    return int(match.group(1)) if match else None


def tissue_from_path(path: str) -> str:
    match = re.search(r"/_token_cls_(?P<tissue>.+?)_kmer/", path)
    if match is None:
        raise ValueError(f"Could not infer tissue from prediction path: {path}")
    return match.group("tissue")


def variability_path(participant: str, tissue: str) -> Path:
    root = Path("/sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir")
    name = f"{participant}_per_varaint_variability_{tissue}_kmer_seq_5400_datasets.csv"
    return root / f"_{tissue}_kmer" / name


def discover_result_row(eval_path: Path, split: str) -> dict[str, object]:
    payload = ast.literal_eval(eval_path.read_text(encoding="utf-8"))
    if len(payload) != 1:
        raise ValueError(f"Expected one participant in {eval_path}")
    participant, evaluations = next(iter(payload.items()))
    prediction_paths = list(evaluations)
    pretrain = [path for path in prediction_paths if "/pretrain_" in path]
    checkpoints = [(checkpoint_number(path), path) for path in prediction_paths]
    checkpoints = [(number, path) for number, path in checkpoints if number is not None]
    if len(pretrain) != 1 or not checkpoints:
        raise ValueError(f"Expected one pretrain and at least one checkpoint in {eval_path}")
    selected_checkpoint, retrained = max(checkpoints, key=lambda item: item[0])
    tissue = tissue_from_path(retrained)
    mean_name = f"{split}_eval_objects_dict_mean_label_cache.pkl"
    mean_path = eval_path.parent / "mean_label_caches" / mean_name
    return {
        "participant": participant,
        "tissue": tissue,
        "retrained_path": Path(retrained),
        "pretrained_path": Path(pretrain[0]),
        "variability_path": variability_path(participant, tissue),
        "mean_path": mean_path,
        "split": split,
        "selected_checkpoint": selected_checkpoint,
        "selection_rule": "largest_numeric_checkpoint",
    }


def discover_manifest(results_root: Path, split: str) -> pd.DataFrame:
    paths = sorted(results_root.glob(f"*/{split}_eval_objects_dict.txt"))
    if not paths:
        raise FileNotFoundError(f"No {split} evaluation dictionaries under {results_root}")
    return pd.DataFrame(discover_result_row(path, split) for path in paths)


def filter_manifest(
    manifest: pd.DataFrame, tissues: list[str] | None, participants: list[str] | None
) -> pd.DataFrame:
    result = manifest
    if tissues:
        result = result[result["tissue"].isin(tissues)]
    if participants:
        result = result[result["participant"].isin(participants)]
    if result.empty:
        raise ValueError("No manifest rows matched the tissue/participant filters")
    return result.reset_index(drop=True)


def parse_window_ids(values: pd.Series) -> pd.DataFrame:
    return values.astype(str).str.extract(
        r"^(?P<chrom>[^:]+):(?P<parsed_start>\d+)-(?P<parsed_end>\d+)$"
    )


def add_window_coordinates(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    if "window_id" not in frame:
        raise ValueError(f"{source} has no window_id column")
    parsed = parse_window_ids(frame["window_id"])
    if parsed.isna().any().any():
        raise ValueError(f"{source} contains malformed window_id values")
    parsed_start = parsed["parsed_start"].astype("int64")
    parsed_end = parsed["parsed_end"].astype("int64")
    result = frame.copy()
    result["chrom"] = parsed["chrom"]
    result["window_start"] = numeric_coordinate(result, "window_start", parsed_start, source)
    result["window_end"] = numeric_coordinate(result, "window_end", parsed_end, source)
    if not result["window_start"].equals(parsed_start) or not result["window_end"].equals(parsed_end):
        raise ValueError(f"{source} window_start/window_end disagree with window_id")
    if (result["window_start"] < 0).any() or (result["window_end"] <= result["window_start"]).any():
        raise ValueError(f"{source} contains invalid half-open window coordinates")
    return result


def numeric_coordinate(
    frame: pd.DataFrame, column: str, fallback: pd.Series, source: Path
) -> pd.Series:
    if column not in frame:
        return fallback
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or (values % 1 != 0).any():
        raise ValueError(f"{source} contains non-integer {column} values")
    return values.astype("int64")


def load_predictions(path: Path, prediction_column: str) -> pd.DataFrame:
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    frame = add_window_coordinates(read_prediction_table(path, prediction_column), path)
    required = set(KEY_COLUMNS + ["label", prediction_column])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    empty_fallback = pd.Series(dtype="int64")
    frame["genomic_position"] = numeric_coordinate(
        frame, "genomic_position", empty_fallback, path
    )
    if (frame["genomic_position"] < frame["window_start"]).any():
        raise ValueError(f"{path} has genomic positions before their window")
    if (frame["genomic_position"] >= frame["window_end"]).any():
        raise ValueError(f"{path} has genomic positions outside their half-open window")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{path} has duplicate window/position keys")
    validate_numeric_values(frame, ["label", prediction_column], path)
    return frame


def validate_numeric_values(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"{source} contains non-finite {column} values")


def baseline_values(frame: pd.DataFrame, prediction_column: str, source: Path) -> pd.DataFrame:
    candidates = ["mean_label", prediction_column, "prediction"]
    value_column = next((column for column in candidates if column in frame), None)
    if value_column is None:
        raise ValueError(f"{source} has no usable mean-baseline prediction column")
    required = set(KEY_COLUMNS + [value_column])
    if required.difference(frame.columns):
        raise ValueError(f"{source} mean baseline must include {KEY_COLUMNS}")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{source} has duplicate mean-baseline keys")
    validate_numeric_values(frame, [value_column], source)
    columns = KEY_COLUMNS + [value_column]
    if "label" in frame:
        columns.append("label")
    return frame[columns].rename(
        columns={value_column: "mean_prediction", "label": "mean_label_truth"}
    )


def assert_same_keys(left: pd.DataFrame, right: pd.DataFrame, description: str) -> None:
    keys = left[KEY_COLUMNS].merge(right[KEY_COLUMNS], on=KEY_COLUMNS, how="outer", indicator=True)
    if not (keys["_merge"] == "both").all():
        counts = keys["_merge"].value_counts().to_dict()
        raise ValueError(f"Mismatched {description} genomic keys: {counts}")


def pair_predictions(row: pd.Series, prediction_column: str) -> pd.DataFrame:
    retrained_path = row["retrained_path"]
    pretrained_path = row["pretrained_path"]
    retrained = load_predictions(retrained_path, prediction_column)
    pretrained = load_predictions(pretrained_path, prediction_column)
    assert_same_keys(retrained, pretrained, "retrained/pretrained")
    keep = KEY_COLUMNS + ["chrom", "window_start", "window_end", "label", prediction_column]
    paired = retrained[keep].rename(columns={prediction_column: "retrained_prediction"})
    pretrained_keep = pretrained[KEY_COLUMNS + ["label", prediction_column]].rename(
        columns={"label": "pretrained_label", prediction_column: "pretrained_prediction"}
    )
    paired = paired.merge(pretrained_keep, on=KEY_COLUMNS, validate="one_to_one")
    if not numeric_equal(paired["label"], paired["pretrained_label"]):
        raise ValueError(f"Labels differ between {retrained_path} and {pretrained_path}")
    paired = paired.drop(columns="pretrained_label")
    attach_mean_baseline(paired, retrained, row.get("mean_path"), prediction_column)
    paired["participant"] = row["participant"]
    paired["tissue"] = row["tissue"]
    return paired


def numeric_equal(left: pd.Series, right: pd.Series) -> bool:
    left_values = pd.to_numeric(left, errors="coerce")
    right_values = pd.to_numeric(right, errors="coerce")
    return left_values.notna().all() and np.array_equal(left_values.to_numpy(), right_values.to_numpy())


def attach_mean_baseline(
    paired: pd.DataFrame,
    retrained: pd.DataFrame,
    mean_path: Path | None,
    prediction_column: str,
) -> None:
    source = mean_path
    if mean_path is not None:
        if not mean_path.exists():
            raise FileNotFoundError(f"Missing mean-baseline file: {mean_path}")
        mean = baseline_values(read_table(mean_path), prediction_column, mean_path)
    elif "mean_label" in retrained:
        source = Path("mean_label column in retrained prediction file")
        mean = baseline_values(retrained, prediction_column, source)
    else:
        paired["mean_prediction"] = np.nan
        return
    merged = paired[KEY_COLUMNS + ["label"]].merge(
        mean, on=KEY_COLUMNS, how="left", validate="one_to_one"
    )
    if "mean_label_truth" in merged:
        matched = merged["mean_label_truth"].notna()
        if not numeric_equal(merged.loc[matched, "label"], merged.loc[matched, "mean_label_truth"]):
            raise ValueError(f"Labels differ between predictions and mean baseline {source}")
    paired["mean_prediction"] = merged["mean_prediction"].to_numpy()


def chromosome_variability_rows(
    path: Path, chroms: list[str], chunk_size: int = 500_000
) -> tuple[pd.DataFrame, float]:
    prefixes = tuple(f"{chrom}:" for chrom in chroms)
    selected: list[pd.DataFrame] = []
    std_max = np.nan
    chunks = pd.read_csv(
        path, usecols=["full_position", "std"], chunksize=chunk_size
    )
    for chunk in chunks:
        values = pd.to_numeric(chunk["std"], errors="coerce")
        valid = values.dropna()
        if (valid < 0).any():
            raise ValueError(f"{path} contains negative variability values")
        if not valid.empty:
            chunk_max = float(valid.max())
            std_max = chunk_max if np.isnan(std_max) else max(std_max, chunk_max)
        keep = chunk["full_position"].astype(str).str.startswith(prefixes)
        selected.append(chunk.loc[keep].assign(std=values.loc[keep]))
    frame = pd.concat(selected, ignore_index=True)
    return frame, std_max


def load_variability(
    path: Path, chroms: list[str] | None = None
) -> tuple[dict[str, pd.Series], float]:
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing variability file: {path}")
    if chroms:
        frame, std_max = chromosome_variability_rows(path, chroms)
    else:
        frame = pd.read_csv(path, usecols=["full_position", "std"])
        std_max = np.nan
    parsed = frame["full_position"].astype(str).str.extract(
        r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$"
    )
    frame = frame.assign(
        chrom=parsed["chrom"],
        start=pd.to_numeric(parsed["start"]),
        std=pd.to_numeric(frame["std"], errors="coerce"),
    )
    frame = frame.dropna(subset=["chrom", "start", "std"])
    if frame.empty or (frame["std"] < 0).any():
        raise ValueError(f"{path} has no valid non-negative variability values")
    frame["start"] = frame["start"].astype("int64")
    lookup = {
        chrom: group.groupby("start")["std"].mean()
        for chrom, group in frame.groupby("chrom")
    }
    maximum = float(frame["std"].max()) if np.isnan(std_max) else float(std_max)
    return lookup, maximum


def variability_values(frame: pd.DataFrame, lookup: dict[str, pd.Series]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for chrom, rows in frame.groupby("chrom", sort=False):
        chrom_lookup = lookup.get(chrom)
        if chrom_lookup is None:
            continue
        matches = pd.concat(
            [(rows["genomic_position"] + offset).map(chrom_lookup) for offset in range(-1, 6)],
            axis=1,
        )
        result.loc[rows.index] = matches.mean(axis=1, skipna=True)
    return result


def assign_bins(values: pd.Series, std_max: float) -> pd.Series:
    if std_max == 0:
        return pd.Series(1, index=values.index, dtype="Int64").where(values.notna())
    bins = np.ceil(values / (std_max / 5.0)).clip(lower=1, upper=5)
    return bins.astype("Int64")


def add_variability(frame: pd.DataFrame, path: Path, chroms: list[str] | None) -> pd.DataFrame:
    if chroms:
        frame = frame[frame["chrom"].isin(chroms)].copy()
    lookup, std_max = load_variability(path, chroms)
    result = frame.copy()
    result["variability_std"] = variability_values(result, lookup)
    result["variability_bin"] = assign_bins(result["variability_std"], std_max)
    return result


def filter_chromosomes(frame: pd.DataFrame, chroms: list[str] | None) -> pd.DataFrame:
    if not chroms:
        return frame
    return frame[frame["chrom"].isin(chroms)].copy()


def count_windows(frame: pd.DataFrame) -> int:
    return int(frame["window_id"].nunique())


def qc_row(frame: pd.DataFrame, metadata: pd.Series, minimum_rate: float) -> dict[str, object]:
    matched = frame["variability_bin"].notna()
    mean_matched = frame["mean_prediction"].notna()
    site_count = len(frame)
    if site_count == 0:
        raise ValueError(f"No paired sites for {metadata['participant']}")
    match_rate = float(matched.mean())
    if match_rate < minimum_rate:
        raise ValueError(
            f"Variability match rate {match_rate:.4f} below {minimum_rate:.4f} "
            f"for {metadata['participant']}"
        )
    return {
        "participant": metadata["participant"],
        "tissue": metadata["tissue"],
        "input_retrained_sites": site_count,
        "input_pretrained_sites": site_count,
        "input_model_windows": count_windows(frame),
        "paired_sites": site_count,
        "paired_windows": count_windows(frame),
        "variability_matched_sites": int(matched.sum()),
        "variability_unmatched_sites": int((~matched).sum()),
        "variability_matched_windows": count_windows(frame[matched]),
        "variability_unmatched_site_windows": count_windows(frame[~matched]),
        "variability_match_rate": match_rate,
        "mean_baseline_matched_sites": int(mean_matched.sum()),
        "mean_baseline_missing_sites": int((~mean_matched).sum()),
        "mean_baseline_match_rate": float(mean_matched.mean()),
        "qc_status": "pass",
    }


def loss(label: pd.Series, prediction: pd.Series, metric: str) -> pd.Series:
    actual = pd.to_numeric(label, errors="coerce")
    predicted = pd.to_numeric(prediction, errors="coerce")
    if metric == "mae":
        return (actual - predicted).abs()
    return (actual != predicted).astype(float)


def comparison_rows(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    retrained_loss = loss(frame["label"], frame["retrained_prediction"], metric)
    pieces = []
    for baseline in ("pretrained", "mean"):
        column = f"{baseline}_prediction"
        available = frame[column].notna()
        if not available.any():
            continue
        piece = frame.loc[available].copy()
        piece["comparison"] = f"retrained_minus_{baseline}"
        piece["improvement"] = loss(piece["label"], piece[column], metric) - retrained_loss[available]
        pieces.append(piece)
    if not pieces:
        raise ValueError("No baseline predictions were available")
    return pd.concat(pieces, ignore_index=True)


def expand_bin_groups(frame: pd.DataFrame, selected_groups: list[str] | None) -> pd.DataFrame:
    pieces = []
    groups = {**EXACT_BIN_GROUPS, **COMBINED_BIN_GROUPS}
    if selected_groups:
        unknown = set(selected_groups).difference(groups)
        if unknown:
            raise ValueError(f"Unknown bin groups: {sorted(unknown)}")
        groups = {name: groups[name] for name in selected_groups}
    for name, bins in groups.items():
        selected = frame[frame["variability_bin"].isin(np.atleast_1d(list(bins) if isinstance(bins, set) else bins))].copy()
        selected["bin_group"] = name
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


def participant_windows(sites: pd.DataFrame, bin_groups: list[str] | None) -> pd.DataFrame:
    expanded = expand_bin_groups(sites, bin_groups)
    keys = ["participant", "tissue", "chrom", "window_start", "window_end", "bin_group", "comparison"]
    return expanded.groupby(keys, observed=True, as_index=False).agg(
        improvement=("improvement", "mean"), n_sites=("improvement", "size")
    )


def standard_error(values: pd.Series) -> float:
    return float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else math.nan


def sign_flip_p(values: pd.Series) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return math.nan
    observed = abs(clean.mean())
    if np.isclose(observed, 0.0):
        return 1.0
    if len(clean) > 20:
        rng = np.random.default_rng(20260815)
        signs = rng.choice((-1, 1), size=(100_000, len(clean)))
    else:
        indices = np.arange(2 ** len(clean), dtype=np.uint64)[:, None]
        bits = (indices >> np.arange(len(clean), dtype=np.uint64)) & 1
        signs = bits.astype(np.int8) * 2 - 1
    permuted = np.abs((signs * clean).mean(axis=1))
    extreme = int(np.sum(permuted >= observed - 1e-15))
    if len(clean) > 20:
        return float((extreme + 1) / (len(permuted) + 1))
    return float(extreme / len(permuted))


def aggregate_windows(windows: pd.DataFrame) -> pd.DataFrame:
    keys = ["tissue", "chrom", "window_start", "window_end", "bin_group", "comparison"]
    result = windows.groupby(keys, observed=True, as_index=False).agg(
        improvement=("improvement", "mean"),
        standard_error=("improvement", standard_error),
        p_value=("improvement", sign_flip_p),
        n_individuals=("participant", "nunique"),
        n_observations=("improvement", "size"),
        n_sites=("n_sites", "sum"),
    )
    result["q_value"] = grouped_fdr(result, ["tissue", "comparison", "bin_group"])
    return result


def participant_summaries(windows: pd.DataFrame) -> pd.DataFrame:
    keys = ["participant", "tissue", "bin_group", "comparison"]
    return windows.groupby(keys, observed=True, as_index=False).agg(
        improvement=("improvement", "mean"),
        n_windows=("improvement", "size"),
        n_sites=("n_sites", "sum"),
    )


def aggregate_claims(participants: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    keys = ["tissue", "bin_group", "comparison"]
    result = participants.groupby(keys, observed=True, as_index=False).agg(
        improvement=("improvement", "mean"),
        standard_error=("improvement", standard_error),
        p_value=("improvement", sign_flip_p),
        n_individuals=("participant", "nunique"),
        n_observations=("improvement", "size"),
        n_participant_windows=("n_windows", "sum"),
        n_sites=("n_sites", "sum"),
    )
    unique = windows.drop_duplicates(keys + ["chrom", "window_start", "window_end"])
    counts = unique.groupby(keys, observed=True, as_index=False).size()
    counts = counts.rename(columns={"size": "n_windows"})
    result = result.merge(counts, on=keys, validate="one_to_one")
    result["q_value"] = benjamini_hochberg(result["p_value"])
    return result


def direction_counts(windows: pd.DataFrame) -> pd.DataFrame:
    keys = ["tissue", "chrom", "window_start", "window_end", "bin_group", "comparison"]
    grouped = windows.groupby(keys, observed=True)["improvement"]
    return grouped.agg(
        n_favors_retrained=lambda values: int((values > 0).sum()),
        n_favors_baseline=lambda values: int((values < 0).sum()),
        n_tie=lambda values: int(np.isclose(values, 0.0).sum()),
    ).reset_index()


def candidate_regions(aggregates: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    high_groups = {"bin3", "bin4", "bin5", "bins3-5", "bins4-5"}
    selected = aggregates[aggregates["bin_group"].isin(high_groups)].copy()
    selected = selected.merge(direction_counts(windows), validate="one_to_one")
    selected["fraction_favors_retrained"] = (
        selected["n_favors_retrained"] / selected["n_observations"]
    )
    selected["fraction_favors_baseline"] = (
        selected["n_favors_baseline"] / selected["n_observations"]
    )
    selected["evidence_status"] = selected.apply(evidence_status, axis=1)
    selected["_baseline_priority"] = selected["comparison"].ne(
        "retrained_minus_pretrained"
    ).astype(int)
    selected["_q"] = selected["q_value"].fillna(1.0)
    selected["_effect"] = selected["improvement"].abs()
    selected = selected.sort_values(
        ["_baseline_priority", "n_individuals", "_q", "_effect"],
        ascending=[True, False, True, False],
        kind="stable",
    ).drop(columns=["_baseline_priority", "_q", "_effect"])
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    return selected


def evidence_rank(values: pd.Series) -> pd.Series:
    order = {"supported": 0, "uncertain": 1, "not_testable": 2}
    return values.map(order).fillna(3)


def rank_candidates(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if maximum <= 0:
        raise ValueError("max regions per track must be positive")
    keys = ["tissue", "bin_group", "comparison"]
    ranked = frame.copy()
    ranked["_evidence_rank"] = evidence_rank(ranked["evidence_status"])
    ranked["_q_rank"] = pd.to_numeric(
        ranked["q_value"], errors="coerce"
    ).fillna(1.0)
    ranked["_effect_rank"] = pd.to_numeric(
        ranked["improvement"], errors="coerce"
    ).abs()
    ranked["_chrom_rank"] = ranked["chrom"].map(chromosome_key)
    sort_columns = keys + [
        "n_individuals",
        "_evidence_rank",
        "_q_rank",
        "_effect_rank",
        "_chrom_rank",
        "window_start",
        "window_end",
    ]
    ranked = ranked.sort_values(
        sort_columns,
        ascending=[True, True, True, False, True, True, False, True, True, True],
        kind="stable",
    )
    ranked = ranked.groupby(keys, observed=True, sort=False).head(maximum).copy()
    ranked["rank_within_track"] = (
        ranked.groupby(keys, observed=True, sort=False).cumcount() + 1
    )
    helper_columns = [
        "_evidence_rank",
        "_q_rank",
        "_effect_rank",
        "_chrom_rank",
    ]
    return ranked.drop(columns=helper_columns)


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    adjusted = valid * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted.iloc[::-1])[::-1].clip(upper=1.0)
    result.loc[valid.index] = adjusted
    return result


def grouped_fdr(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby(columns, observed=True):
        result.loc[group.index] = benjamini_hochberg(group["p_value"])
    return result


def chromosome_key(chrom: str) -> tuple[int, int | str]:
    value = chrom.removeprefix("chr")
    return (0, int(value)) if value.isdigit() else (1, value)


def output_chromosomes(frame: pd.DataFrame) -> list[str]:
    chroms = frame["chrom"].dropna().astype(str).unique()
    return sorted(chroms, key=chromosome_key)


def sort_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_chrom_key"] = result["chrom"].map(chromosome_key)
    result = result.sort_values(["_chrom_key", "window_start", "window_end"], kind="stable")
    return result.drop(columns="_chrom_key")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def result_class(value: float) -> str:
    if np.isclose(value, 0.0, atol=1e-12):
        return "no_difference"
    return "favors_retrained" if value > 0 else "favors_baseline"


def evidence_status(row: pd.Series) -> str:
    if pd.isna(row["p_value"]):
        return "not_testable"
    if row["q_value"] <= 0.05:
        return "supported"
    return "uncertain"


def bed_frame(windows: pd.DataFrame) -> pd.DataFrame:
    frame = windows.copy()
    frame["result_class"] = frame["improvement"].map(result_class)
    frame["evidence_status"] = frame.apply(evidence_status, axis=1)
    frame["name"] = frame.apply(
        lambda row: safe_name(f"{row.tissue}|{row.bin_group}|{row.comparison}|{row.result_class}"), axis=1
    )
    frame["score"] = (frame["improvement"].abs() * 1000).round().clip(0, 1000).astype(int)
    frame["strand"] = "."
    frame["thickStart"] = frame["window_start"]
    frame["thickEnd"] = frame["window_end"]
    frame["itemRgb"] = frame["result_class"].map(
        {"favors_retrained": "0,140,0", "favors_baseline": "190,30,45", "no_difference": "128,128,128"}
    )
    frame = frame.rename(columns={"window_start": "start", "window_end": "end"})
    for column in ("improvement", "standard_error", "p_value", "q_value"):
        frame[column] = frame[column].map(format_number)
    return frame[BED_COLUMNS]


def write_ranked_regions(frame: pd.DataFrame, output_dir: Path) -> Path:
    table_path = output_dir / "candidate_regions_top.tsv"
    bed_path = output_dir / "candidate_regions_top.bed"
    frame.to_csv(table_path, sep="\t", index=False)
    bed_frame(sort_intervals(frame)).to_csv(
        bed_path, sep="\t", index=False, header=False, na_rep="."
    )
    return bed_path


def format_number(value: float) -> str:
    return "." if pd.isna(value) else f"{value:.8g}"


def disjoint_segments(frame: pd.DataFrame) -> list[tuple[str, int, int, float]]:
    segments = []
    for chrom, rows in frame.groupby("chrom", sort=False):
        events: dict[int, list[tuple[int, float]]] = {}
        for row in rows.itertuples():
            events.setdefault(int(row.window_start), []).append((1, float(row.improvement)))
            events.setdefault(int(row.window_end), []).append((-1, float(row.improvement)))
        active: list[float] = []
        previous = None
        for position in sorted(events):
            if previous is not None and position > previous and active:
                segments.append((chrom, previous, position, float(np.mean(active))))
            for direction, value in events[position]:
                if direction == 1:
                    active.append(value)
                else:
                    active.remove(value)
            previous = position
    return segments


def write_bedgraphs(windows: pd.DataFrame, output_dir: Path) -> list[Path]:
    tracks_dir = output_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for keys, group in windows.groupby(["tissue", "comparison", "bin_group"], observed=True):
        tissue, comparison, bin_group = keys
        path = tracks_dir / f"{safe_name(tissue)}.{comparison}.{bin_group}.bedGraph"
        with path.open("w", encoding="utf-8") as handle:
            for chrom, start, end, value in disjoint_segments(sort_intervals(group)):
                handle.write(f"{chrom}\t{start}\t{end}\t{value:.8g}\n")
        paths.append(path)
    return paths


def write_ucsc_custom_track(
    paths: Iterable[Path], output_dir: Path, bed_path: Path
) -> Path:
    destination = output_dir / "ucsc_custom_tracks.txt"
    with destination.open("w", encoding="utf-8") as output:
        output.write(
            'track type=bed name="methylation_windows" description="window evidence" '
            'visibility=pack itemRgb=On\n'
        )
        write_ucsc_bed_rows(bed_path, output)
        for path in paths:
            name = path.stem
            color = "0,140,0" if "minus" in name else "60,60,180"
            output.write(f'track type=bedGraph name="{name}" description="positive favors retraining" visibility=full color={color} altColor=190,30,45\n')
            output.write(path.read_text(encoding="utf-8"))
    return destination


def write_ucsc_bed_rows(source: Path, output) -> None:
    """Write BED9 rows accepted by UCSC while retaining BED9+ for IGV."""
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                raise ValueError(
                    f"{source}:{line_number} has {len(fields)} BED fields; "
                    "expected at least 9"
                )
            output.write("\t".join(fields[:9]) + "\n")


def write_igv_session(paths: Iterable[Path], output_dir: Path, genome: str) -> Path:
    destination = output_dir / "igv_session.xml"
    resources = "\n".join(
        f'    <Resource path="{html.escape(path.relative_to(output_dir).as_posix())}" />' for path in paths
    )
    content = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<Session genome="{html.escape(genome)}" version="8">
  <Resources>
{resources}
  </Resources>
</Session>
'''
    destination.write_text(content, encoding="utf-8")
    return destination


def descriptive_lines(claims: pd.DataFrame, participants: pd.DataFrame) -> list[str]:
    combined = claims[claims["bin_group"].isin(["bins3-5", "bins4-5"])]
    lines = [
        "",
        "## Observed combined-bin effects",
        "",
        "| Tissue | Bins | Comparison | Effect | SE | p | q | Individuals | Windows | Participant-windows | Sites |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined.sort_values(["tissue", "bin_group", "comparison"]).itertuples():
        lines.append(
            f"| {row.tissue} | {row.bin_group} | {row.comparison} | "
            f"{row.improvement:.4g} | {row.standard_error:.4g} | "
            f"{format_number(row.p_value)} | {format_number(row.q_value)} | "
            f"{row.n_individuals} | {row.n_windows} | {row.n_participant_windows} | {row.n_sites} |"
        )
    lines.extend(["", "## Observed person-level heterogeneity", ""])
    subset = participants[
        participants["bin_group"].isin(["bins3-5", "bins4-5"])
        & participants["comparison"].eq("retrained_minus_pretrained")
    ]
    for keys, group in subset.groupby(["tissue", "bin_group"], observed=True):
        low = group.loc[group["improvement"].idxmin()]
        high = group.loc[group["improvement"].idxmax()]
        favors = int((group["improvement"] > 0).sum())
        lines.append(
            f"- {keys[0]} {keys[1]}: {favors}/{len(group)} individuals favored retraining; "
            f"range {low['participant']}={low['improvement']:.4g} to "
            f"{high['participant']}={high['improvement']:.4g}."
        )
    lines.extend(
        [
            "",
            "## Hypothesis-generation notes",
            "",
            "Candidate regions are exploratory nominations, not mechanistic findings. "
            "Regions with consistent participant directions can be cross-referenced with "
            "genes and regulatory annotations in follow-up work; no biological cause is "
            "inferred here. Small tissue cohorts limit exact sign-flip test resolution.",
        ]
    )
    return lines


def top_candidate_lines(candidates: pd.DataFrame) -> list[str]:
    selected = candidates[
        candidates["bin_group"].eq("bins4-5")
        & candidates["comparison"].eq("retrained_minus_pretrained")
        & candidates["rank_within_track"].le(3)
    ]
    lines = [
        "",
        "## Top bins4-5 pretrained-comparison candidates",
        "",
        "| Tissue | Region | Effect | SE | p | q | Individuals | Participant-windows | Sites | Favors retrained | Favors baseline | Ties | Direction | Evidence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in selected.sort_values(
        ["tissue", "rank_within_track"], kind="stable"
    ).itertuples():
        lines.append(
            f"| {row.tissue} | {row.chrom}:{row.window_start}-{row.window_end} | "
            f"{row.improvement:.4g} | {format_number(row.standard_error)} | "
            f"{format_number(row.p_value)} | {format_number(row.q_value)} | "
            f"{row.n_individuals} | {row.n_observations} | {row.n_sites} | "
            f"{row.n_favors_retrained} | {row.n_favors_baseline} | "
            f"{row.n_tie} | {result_class(row.improvement)} | "
            f"{row.evidence_status} |"
        )
    if selected.empty:
        lines.append(
            "| No bins4-5 pretrained-comparison candidates were generated. "
            "| | | | | | | | | | | | | |"
        )
    lines.extend(
        [
            "",
            "This table is an exploratory browser shortlist, not evidence of a "
            "biological mechanism. Ranking and direction should be interpreted with "
            "the participant count and evidence status.",
        ]
    )
    return lines


def report_text(
    claims: pd.DataFrame,
    participants: pd.DataFrame,
    candidates: pd.DataFrame,
    metric: str,
    has_mean: bool,
    maximum: int,
    chromosomes: list[str],
) -> str:
    significant = claims[(claims["q_value"] <= 0.05) & claims["q_value"].notna()]
    lines = [
        "# Methylation browser-track report",
        "",
        "## What was generated",
        "",
        "`tracks/*.bedGraph` contains continuous retraining improvement scores; positive values favor retraining and negative values favor the named baseline. `candidate_regions_top.bed` is BED9+ with metadata and green/red/gray favors-retrained/favors-baseline/no-difference coloring. `ucsc_custom_tracks.txt` can be loaded as one UCSC custom-track file. `igv_session.xml` loads the bounded BED and all bedGraph resources in IGV when kept beside the `tracks` directory.",
        "",
        f"Requested variability views are emitted separately (see `aggregate_summary.tsv` for the exact set). The metadata BED is capped at {maximum} ranked regions per tissue/bin/comparison. Ranking favors greater participant coverage, supported evidence, lower q-values, and larger absolute effects. `candidate_regions.tsv.gz` and `window_summary.tsv.gz` preserve the complete audits. Overlapping source windows are converted to disjoint bedGraph segments by averaging active window scores, and bedGraphs retain every aggregate window.",
        "",
        "Discovery mode records the split, numeric checkpoint, and selection rule in `selected_manifest.tsv`; it always selects the largest numeric checkpoint and never ranks checkpoints on evaluation or test performance.",
        "",
        "## Coordinates and scores",
        "",
        "The source generation pipeline targets hg38. This exporter treats `window_start`/`window_end` as 0-based, half-open BED coordinates and validates them against `window_id` and token bounds; this convention remains an explicit provenance assumption because no independent known-locus validation is performed here. `genomic_position` is used only to join prediction and variability records; no claim is made here about whether it denotes a cytosine or another token anchor. Variability matching reproduces the current evaluator's -1 through +5 bp candidate lookup. Five bins are equal-width from zero to the maximum `std` in each participant's variability file, so bin boundaries can differ among participants.",
        "",
        "Chromosome scope represented in this output: "
        + ", ".join(f"`{chrom}`" for chrom in chromosomes)
        + ". This is the observed input/filter coverage, not an assertion of "
        "whole-genome coverage.",
        "",
        f"The selected metric is `{metric}`. Improvement is baseline loss minus retrained loss, so positive is better for retraining. Accuracy loss is 0/1 misclassification; MAE loss is absolute numeric error. Window values first average sites within each participant, then average participants so participants, rather than CpG-rich windows, are the replicate unit. Standard error is SD/sqrt(n) across participants.",
        "",
        "## Statistical evidence",
        "",
        "Aggregate two-sided sign-flip tests use one mean effect per participant. Tests are exact for up to 20 participants (a fixed-seed 100,000-draw approximation is used above 20). Benjamini-Hochberg q-values cover all tissue/bin/comparison aggregate claims. Window-level tests also use participant effects and are corrected within each tissue/bin/comparison track. Rows with fewer than two participants have no p-value. These are exploratory tests: genomic windows and bins are correlated, and no covariate, batch, or repeated-measures model is fitted.",
        "",
        f"Mean-baseline comparison available: **{'yes' if has_mean else 'no'}**. The pretrained model is always the primary baseline; a missing mean baseline is never imputed.",
        "",
        "`qc_summary.tsv` reports model pairing, variability coverage, and mean-baseline coverage for every participant; generation stops below the configured variability match-rate threshold.",
        "",
        "## Observed aggregate patterns",
        "",
    ]
    if significant.empty:
        lines.append("No tissue/bin/comparison aggregate passed exploratory FDR q <= 0.05. This is an observed null result, not evidence of equivalence.")
    else:
        for row in significant.sort_values("q_value").itertuples():
            direction = "favored retraining" if row.improvement > 0 else "favored the baseline"
            lines.append(
                f"- {row.tissue}, {row.bin_group}, {row.comparison}: {direction}; mean={row.improvement:.4g}, SE={row.standard_error:.4g}, q={row.q_value:.3g}, {row.n_windows} distinct windows / {row.n_participant_windows} participant-window observations from {row.n_individuals} individuals ({row.n_sites} sites)."
            )
    lines.extend(descriptive_lines(claims, participants))
    lines.extend(top_candidate_lines(candidates))
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Tracks nominate regions for biological follow-up; they do not establish mechanism or generalization. Results reflect the supplied checkpoint(s), without implying hyperparameter optimality. A favors-retrained window may have a small or uncertain effect; use the metadata, participant counts, SE, p/q values, and per-participant table before interpreting it. Brain-derived tissues are kept separate by their manifest tissue name rather than pooled with other tissues.",
            "",
            "See `aggregate_summary.tsv`, `participant_summary.tsv`, `window_summary.tsv.gz`, `candidate_regions.tsv.gz`, and `candidate_regions_top.tsv` for the values behind every statement and track.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    windows: pd.DataFrame,
    participant_windows_frame: pd.DataFrame,
    participants: pd.DataFrame,
    claims: pd.DataFrame,
    qc: pd.DataFrame,
    output_dir: Path,
    metric: str,
    genome: str,
    maximum_regions: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sort_intervals(windows).to_csv(
        output_dir / "window_summary.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    participants.to_csv(output_dir / "participant_summary.tsv", sep="\t", index=False)
    claims.to_csv(output_dir / "aggregate_summary.tsv", sep="\t", index=False)
    qc.to_csv(output_dir / "qc_summary.tsv", sep="\t", index=False)
    candidates = candidate_regions(windows, participant_windows_frame)
    candidates.to_csv(
        output_dir / "candidate_regions.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    top_candidates = rank_candidates(candidates, maximum_regions)
    bed_path = write_ranked_regions(top_candidates, output_dir)
    paths = write_bedgraphs(windows, output_dir)
    write_ucsc_custom_track(paths, output_dir, bed_path)
    write_igv_session([bed_path, *paths], output_dir, genome)
    has_mean = windows["comparison"].eq("retrained_minus_mean").any()
    report = report_text(
        claims, participants, top_candidates, metric, has_mean, maximum_regions,
        output_chromosomes(windows),
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def build_tracks(
    manifest: pd.DataFrame,
    output_dir: Path,
    metric: str,
    prediction_column: str,
    genome: str,
    chroms: list[str] | None,
    bin_groups: list[str] | None,
    minimum_match_rate: float = 0.5,
    maximum_regions: int = 1000,
) -> None:
    if not 0 <= minimum_match_rate <= 1:
        raise ValueError("minimum variability match rate must be between 0 and 1")
    if maximum_regions <= 0:
        raise ValueError("max regions per track must be positive")
    prepare_output_dir(output_dir)
    participant_window_frames = []
    qc_rows = []
    for _, row in manifest.iterrows():
        paired = filter_chromosomes(pair_predictions(row, prediction_column), chroms)
        variable = add_variability(paired, row["variability_path"], None)
        qc_rows.append(qc_row(variable, row, minimum_match_rate))
        matched = variable.dropna(subset=["variability_bin"])
        compared = comparison_rows(matched, metric)
        participant_window_frames.append(participant_windows(compared, bin_groups))
    if not participant_window_frames:
        raise ValueError("No prediction sites matched variability records")
    participant_window_frame = pd.concat(participant_window_frames, ignore_index=True)
    window_frame = aggregate_windows(participant_window_frame)
    participant_frame = participant_summaries(participant_window_frame)
    claim_frame = aggregate_claims(participant_frame, participant_window_frame)
    manifest.to_csv(output_dir / "selected_manifest.tsv", sep="\t", index=False)
    write_outputs(
        window_frame,
        participant_window_frame,
        participant_frame,
        claim_frame,
        pd.DataFrame(qc_rows),
        output_dir,
        metric,
        genome,
        maximum_regions,
    )


def main() -> None:
    args = parse_args()
    if args.manifest:
        manifest = load_manifest(args.manifest)
    else:
        manifest = discover_manifest(args.results_root, args.split)
    manifest = filter_manifest(manifest, args.tissues, args.participants)
    build_tracks(
        manifest, args.output_dir, args.metric, args.prediction_column,
        args.genome, args.chroms, args.bin_groups, args.min_variability_match_rate,
        args.max_regions_per_track,
    )
    print(f"Wrote browser tracks and report to {args.output_dir}")


if __name__ == "__main__":
    main()
