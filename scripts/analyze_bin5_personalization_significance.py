#!/usr/bin/env python3
# TEST-only variability-bin paired resampling analysis.
# Standard bin-5 run:
#   python3 scripts/analyze_bin5_personalization_significance.py
# Bin-4 run:
#   python3 scripts/analyze_bin5_personalization_significance.py --bin-rank 4
# Faster smoke run:
#   python3 scripts/analyze_bin5_personalization_significance.py \
#     --bin-rank 4 --samples Z000000QM --bootstrap-reps 200 --permutation-reps 500

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "scripts/results/downsampled_standard_ratio_0p25_lr_3e-05"
DEFAULT_TABLE = DEFAULT_ROOT / "report/downsampled_standard_all_samples_metric_rows_binrank_corrected.csv"
DEFAULT_VARIABILITY_ROOT = Path(
    "/sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir"
)
STAGES = ["E1", "E2", "E3"]
METRICS = [
    "weighted_f1",
    "weighted_precision",
    "weighted_recall",
    "macro_f1",
    "accuracy",
]
PRIMARY_STAGE = "E3"
PRIMARY_METRIC = "weighted_f1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TEST-only paired variability-bin bootstrap and permutation analyses."
    )
    parser.add_argument("--metric-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--variability-root", type=Path, default=DEFAULT_VARIABILITY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bin-rank", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--permutation-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def load_manifest(path: Path, samples: list[str] | None, bin_rank: int) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows[
        (rows["split"] == "test")
        & (rows["section_rank"] == f"bin_{bin_rank}")
        & (rows["prediction_type"] == "predicted_class")
    ].copy()
    if samples:
        rows = rows[rows["sample"].isin(samples)].copy()
    required = {"pretrain", *STAGES}
    counts = rows.groupby("sample")["stage"].agg(set)
    complete = counts[counts.map(lambda value: required.issubset(value))].index
    rows = rows[rows["sample"].isin(complete)].copy()
    if rows.empty:
        raise ValueError(f"No complete TEST bin-{bin_rank} sample sets were found")
    return rows.sort_values(["tissue", "sample", "stage"])


def sample_paths(rows: pd.DataFrame, sample: str) -> dict[str, Path]:
    subset = rows[rows["sample"] == sample]
    return {
        stage: Path(subset.loc[subset["stage"] == stage, "prediction_path"].iloc[0])
        for stage in ["pretrain", *STAGES]
    }


def variability_path(root: Path, sample: str, tissue: str) -> Path:
    return (
        root
        / f"_{tissue}_kmer"
        / f"{sample}_per_varaint_variability_{tissue}_kmer_seq_5400_datasets.csv"
    )


def read_predictions(path: Path, prediction_name: str) -> pd.DataFrame:
    columns = ["window_id", "genomic_position", "label", "predicted_class"]
    rows = pd.read_csv(path, usecols=columns)
    rows = rows.dropna(subset=columns).copy()
    rows["genomic_position"] = rows["genomic_position"].astype(np.int64)
    rows["label"] = rows["label"].astype(np.int8)
    rows[prediction_name] = rows.pop("predicted_class").astype(np.int8)
    validate_prediction_rows(rows, path)
    return rows


def validate_prediction_rows(rows: pd.DataFrame, path: Path) -> None:
    keys = ["window_id", "genomic_position"]
    if rows.duplicated(keys).any():
        raise ValueError(f"Duplicate prediction keys in {path}")
    if not rows["label"].isin([0, 1, 2]).all():
        raise ValueError(f"Invalid labels in {path}")


def read_variability(path: Path) -> tuple[dict[str, pd.Series], float]:
    rows = pd.read_csv(path, usecols=["full_position", "std"])
    rows = rows.dropna(subset=["full_position", "std"]).copy()
    parts = rows["full_position"].astype(str).str.extract(
        r"^(?P<chrom>[^:]+):(?P<position>\d+)-\d+$"
    )
    rows["chrom"] = parts["chrom"]
    rows["position"] = pd.to_numeric(parts["position"], errors="coerce")
    rows["std"] = pd.to_numeric(rows["std"], errors="coerce")
    rows = rows.dropna(subset=["chrom", "position", "std"]).copy()
    rows["position"] = rows["position"].astype(np.int64)
    lookup = {
        chrom: chrom_rows.groupby("position")["std"].mean()
        for chrom, chrom_rows in rows.groupby("chrom", sort=False)
    }
    return lookup, float(rows["std"].max())


def chromosome_from_window(window_ids: pd.Series) -> pd.Series:
    return window_ids.astype(str).str.split(":", n=1).str[0]


def assign_variability(pretrain: pd.DataFrame, lookup: dict[str, pd.Series]) -> pd.Series:
    values = pd.Series(np.nan, index=pretrain.index, dtype=float)
    chroms = chromosome_from_window(pretrain["window_id"])
    for chrom, indices in chroms.groupby(chroms).groups.items():
        if chrom not in lookup:
            continue
        positions = pretrain.loc[indices, "genomic_position"]
        summed = pd.Series(0.0, index=indices)
        matched = pd.Series(0, index=indices, dtype=np.int16)
        chrom_lookup = lookup[chrom]
        for offset in range(-1, 6):
            candidates = positions + offset
            candidate_values = candidates.map(chrom_lookup)
            present = candidate_values.notna()
            summed += candidate_values.fillna(0.0)
            matched += present.astype(np.int16)
        valid = matched > 0
        values.loc[valid.index[valid]] = summed.loc[valid] / matched.loc[valid]
    return values


def variability_bin_mask(values: pd.Series, maximum: float, bin_rank: int) -> pd.Series:
    bin_width = maximum / 5
    lower = bin_width * (bin_rank - 1)
    upper = bin_width * bin_rank
    if bin_rank == 1:
        return (values >= lower) & (values <= upper)
    return (values > lower) & (values <= upper)


def align_predictions(pretrain: pd.DataFrame, retrain: pd.DataFrame) -> pd.DataFrame:
    keys = ["window_id", "genomic_position"]
    aligned = pretrain.merge(
        retrain.rename(columns={"label": "retrain_label"}),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != len(pretrain) or len(aligned) != len(retrain):
        raise ValueError("Pretrain and retrain prediction keys do not match exactly")
    if not np.array_equal(aligned["label"], aligned["retrain_label"]):
        raise ValueError("Pretrain and retrain labels differ")
    return aligned.drop(columns="retrain_label")


def window_confusions(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, window_codes = np.unique(rows["window_id"].to_numpy(), return_inverse=True)
    pretrain = np.zeros((len(windows), 9), dtype=np.int64)
    retrain = np.zeros((len(windows), 9), dtype=np.int64)
    labels = rows["label"].to_numpy(dtype=np.int64)
    pretrain_cells = labels * 3 + rows["pretrain_prediction"].to_numpy(dtype=np.int64)
    retrain_cells = labels * 3 + rows["retrain_prediction"].to_numpy(dtype=np.int64)
    np.add.at(pretrain, (window_codes, pretrain_cells), 1)
    np.add.at(retrain, (window_codes, retrain_cells), 1)
    return windows, pretrain, retrain


def metric_values(flat_confusions: np.ndarray) -> dict[str, np.ndarray]:
    matrices = np.asarray(flat_confusions, dtype=float).reshape(-1, 3, 3)
    support = matrices.sum(axis=2)
    predicted = matrices.sum(axis=1)
    true_positive = np.diagonal(matrices, axis1=1, axis2=2)
    precision = safe_divide(true_positive, predicted)
    recall = safe_divide(true_positive, support)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    total = support.sum(axis=1)
    return {
        "weighted_f1": safe_divide((f1 * support).sum(axis=1), total),
        "weighted_precision": safe_divide((precision * support).sum(axis=1), total),
        "weighted_recall": safe_divide((recall * support).sum(axis=1), total),
        "macro_f1": f1.mean(axis=1),
        "accuracy": safe_divide(true_positive.sum(axis=1), total),
    }


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.zeros_like(np.asarray(numerator, dtype=float))
    return np.divide(numerator, denominator, out=output, where=denominator != 0)


def observed_metrics(pretrain: np.ndarray, retrain: np.ndarray) -> dict[str, tuple[float, float]]:
    pretrain_values = metric_values(pretrain.sum(axis=0, keepdims=True))
    retrain_values = metric_values(retrain.sum(axis=0, keepdims=True))
    return {
        metric: (float(pretrain_values[metric][0]), float(retrain_values[metric][0]))
        for metric in METRICS
    }


def bootstrap_deltas(
    pretrain: np.ndarray,
    retrain: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    output = {metric: np.empty(repetitions) for metric in METRICS}
    window_count = len(pretrain)
    for start in range(0, repetitions, 250):
        stop = min(start + 250, repetitions)
        indices = rng.integers(0, window_count, size=(stop - start, window_count))
        pretrain_values = metric_values(pretrain[indices].sum(axis=1))
        retrain_values = metric_values(retrain[indices].sum(axis=1))
        for metric in METRICS:
            output[metric][start:stop] = retrain_values[metric] - pretrain_values[metric]
    return output


def permutation_deltas(
    pretrain: np.ndarray,
    retrain: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    output = {metric: np.empty(repetitions) for metric in METRICS}
    pretrain_total = pretrain.sum(axis=0)
    retrain_total = retrain.sum(axis=0)
    pretrain_minus_retrain = pretrain - retrain
    for start in range(0, repetitions, 500):
        stop = min(start + 500, repetitions)
        swaps = rng.integers(0, 2, size=(stop - start, len(pretrain)), dtype=np.int8)
        adjustment = swaps @ pretrain_minus_retrain
        permuted_retrain = retrain_total + adjustment
        permuted_pretrain = pretrain_total - adjustment
        pretrain_values = metric_values(permuted_pretrain)
        retrain_values = metric_values(permuted_retrain)
        for metric in METRICS:
            output[metric][start:stop] = retrain_values[metric] - pretrain_values[metric]
    return output


def resampling_rows(
    sample: str,
    tissue: str,
    stage: str,
    sites: int,
    windows: int,
    observed: dict[str, tuple[float, float]],
    bootstrap: dict[str, np.ndarray],
    permutation: dict[str, np.ndarray],
    bin_rank: int,
) -> list[dict[str, object]]:
    rows = []
    for metric, (pretrain_value, retrain_value) in observed.items():
        difference = retrain_value - pretrain_value
        rows.append(
            {
                "sample": sample,
                "tissue": tissue,
                "split": "test",
                "bin": f"bin_{bin_rank}",
                "stage": stage,
                "metric": metric,
                "sites": sites,
                "windows": windows,
                "pretrain_value": pretrain_value,
                "retrain_value": retrain_value,
                "difference": difference,
                "bootstrap_ci_low": np.quantile(bootstrap[metric], 0.025),
                "bootstrap_ci_high": np.quantile(bootstrap[metric], 0.975),
                "bootstrap_win_probability": np.mean(bootstrap[metric] > 0),
                "bootstrap_tie_probability": np.mean(bootstrap[metric] == 0),
                "permutation_p_one_sided": monte_carlo_p(permutation[metric], difference, False),
                "permutation_p_two_sided": monte_carlo_p(permutation[metric], difference, True),
            }
        )
    return rows


def monte_carlo_p(null_values: np.ndarray, observed: float, two_sided: bool) -> float:
    if two_sided:
        exceedances = np.count_nonzero(np.abs(null_values) >= abs(observed))
    else:
        exceedances = np.count_nonzero(null_values >= observed)
    return float((exceedances + 1) / (len(null_values) + 1))


def report_metric_match(
    manifest: pd.DataFrame,
    sample: str,
    stage: str,
    metric: str,
    observed_value: float,
) -> float:
    expected = manifest.loc[
        (manifest["sample"] == sample) & (manifest["stage"] == stage), metric
    ].iloc[0]
    return abs(float(expected) - observed_value)


def analyze_sample(
    manifest: pd.DataFrame,
    variability_root: Path,
    sample: str,
    bootstrap_reps: int,
    permutation_reps: int,
    rng: np.random.Generator,
    bin_rank: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    sample_rows = manifest[manifest["sample"] == sample]
    tissue = str(sample_rows["tissue"].iloc[0])
    paths = sample_paths(manifest, sample)
    pretrain = read_predictions(paths["pretrain"], "pretrain_prediction")
    lookup, maximum = read_variability(variability_path(variability_root, sample, tissue))
    pretrain["variability_std"] = assign_variability(pretrain, lookup)
    mask = variability_bin_mask(pretrain["variability_std"], maximum, bin_rank)
    pretrain = pretrain[mask].copy()
    pretrain = pretrain.drop(columns="variability_std")
    if pretrain.empty:
        raise ValueError(f"No bin-{bin_rank} TEST rows for {sample}")
    window_rows = [
        {"sample": sample, "tissue": tissue, "window_id": window_id}
        for window_id in sorted(pretrain["window_id"].unique())
    ]

    results = []
    qc_rows = []
    for stage in STAGES:
        retrain = read_predictions(paths[stage], "retrain_prediction")
        aligned = align_predictions(pretrain, retrain.merge(
            pretrain[["window_id", "genomic_position"]],
            on=["window_id", "genomic_position"],
            how="inner",
            validate="one_to_one",
        ))
        windows, pretrain_confusions, retrain_confusions = window_confusions(aligned)
        observed = observed_metrics(pretrain_confusions, retrain_confusions)
        bootstrap = bootstrap_deltas(pretrain_confusions, retrain_confusions, bootstrap_reps, rng)
        permutation = permutation_deltas(pretrain_confusions, retrain_confusions, permutation_reps, rng)
        results.extend(
            resampling_rows(
                sample,
                tissue,
                stage,
                len(aligned),
                len(windows),
                observed,
                bootstrap,
                permutation,
                bin_rank,
            )
        )
        for metric, (_, retrain_value) in observed.items():
            qc_rows.append(
                {
                    "sample": sample,
                    "stage": stage,
                    "metric": metric,
                    "absolute_difference_from_existing_report": report_metric_match(
                        manifest, sample, stage, metric, retrain_value
                    ) if metric in manifest.columns else np.nan,
                }
            )
        if stage == STAGES[0]:
            for metric, (pretrain_value, _) in observed.items():
                qc_rows.append(
                    {
                        "sample": sample,
                        "stage": "pretrain",
                        "metric": metric,
                        "absolute_difference_from_existing_report": report_metric_match(
                            manifest, sample, "pretrain", metric, pretrain_value
                        ) if metric in manifest.columns else np.nan,
                    }
                )
    return results, qc_rows, window_rows


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    ranks = np.arange(1, len(valid) + 1)
    adjusted = valid.to_numpy() * len(valid) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def add_person_fdr(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["permutation_q_one_sided_all_person_stage_metric_tests"] = benjamini_hochberg(
        rows["permutation_p_one_sided"]
    )
    primary = (rows["stage"] == PRIMARY_STAGE) & (rows["metric"] == PRIMARY_METRIC)
    rows["permutation_q_one_sided_primary_person_tests"] = np.nan
    rows.loc[primary, "permutation_q_one_sided_primary_person_tests"] = benjamini_hochberg(
        rows.loc[primary, "permutation_p_one_sided"]
    )
    return rows


def exact_sign_flip_p(differences: np.ndarray, two_sided: bool) -> float:
    observed = float(np.mean(differences))
    signs = np.asarray(list(product([-1.0, 1.0], repeat=len(differences))))
    null_values = (signs * differences).mean(axis=1)
    if two_sided:
        return float(np.mean(np.abs(null_values) >= abs(observed)))
    return float(np.mean(null_values >= observed))


def bootstrap_people(
    differences: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = rng.integers(0, len(differences), size=(repetitions, len(differences)))
    return differences[indices].mean(axis=1)


def tissue_rows(
    person_rows: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    output = []
    for keys, group in person_rows.groupby(["tissue", "stage", "metric"], sort=True):
        tissue, stage, metric = keys
        differences = group["difference"].to_numpy(dtype=float)
        bootstrap = bootstrap_people(differences, repetitions, rng)
        output.append(
            {
                "tissue": tissue,
                "stage": stage,
                "metric": metric,
                "individuals": group["sample"].nunique(),
                "person_window_instances": int(group["windows"].sum()),
                "min_windows_per_person": int(group["windows"].min()),
                "max_windows_per_person": int(group["windows"].max()),
                "sites": int(group["sites"].sum()),
                "mean_difference": float(differences.mean()),
                "median_difference": float(np.median(differences)),
                "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                "bootstrap_win_probability": float(np.mean(bootstrap > 0)),
                "exact_sign_flip_p_one_sided": exact_sign_flip_p(differences, False),
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(differences, True),
            }
        )
    rows = pd.DataFrame(output)
    primary = (rows["stage"] == PRIMARY_STAGE) & (rows["metric"] == PRIMARY_METRIC)
    rows["bh_q_one_sided_primary_six_tissues"] = np.nan
    rows.loc[primary, "bh_q_one_sided_primary_six_tissues"] = benjamini_hochberg(
        rows.loc[primary, "exact_sign_flip_p_one_sided"]
    )
    return rows


def add_unique_window_counts(
    tissue_results: pd.DataFrame,
    window_membership: pd.DataFrame,
) -> pd.DataFrame:
    counts = (
        window_membership.groupby("tissue")["window_id"]
        .nunique()
        .rename("unique_genomic_windows")
        .reset_index()
    )
    return tissue_results.merge(counts, on="tissue", how="left")


def combined_statistic(frame: pd.DataFrame, signs: np.ndarray | None = None) -> float:
    values = frame[["tissue", "difference"]].copy()
    if signs is not None:
        values["difference"] = values["difference"].to_numpy() * signs
    return float(values.groupby("tissue")["difference"].mean().mean())


def combined_bootstrap(
    frame: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    tissues = [group["difference"].to_numpy() for _, group in frame.groupby("tissue")]
    output = np.empty(repetitions)
    for index in range(repetitions):
        tissue_means = []
        for values in tissues:
            selected = rng.integers(0, len(values), size=len(values))
            tissue_means.append(values[selected].mean())
        output[index] = np.mean(tissue_means)
    return output


def combined_permutation(
    frame: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.empty(repetitions)
    for index in range(repetitions):
        signs = rng.choice([-1.0, 1.0], size=len(frame))
        output[index] = combined_statistic(frame, signs)
    return output


def combined_rows(
    person_rows: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    output = []
    for (stage, metric), group in person_rows.groupby(["stage", "metric"], sort=True):
        observed = combined_statistic(group)
        bootstrap = combined_bootstrap(group, repetitions, rng)
        permutation = combined_permutation(group, repetitions, rng)
        output.append(
            {
                "stage": stage,
                "metric": metric,
                "tissues": group["tissue"].nunique(),
                "individuals": group["sample"].nunique(),
                "person_window_instances": int(group["windows"].sum()),
                "sites": int(group["sites"].sum()),
                "equal_tissue_mean_difference": observed,
                "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                "bootstrap_win_probability": float(np.mean(bootstrap > 0)),
                "permutation_p_one_sided": monte_carlo_p(permutation, observed, False),
                "permutation_p_two_sided": monte_carlo_p(permutation, observed, True),
            }
        )
    return pd.DataFrame(output)


def coverage_rows(person_rows: pd.DataFrame) -> pd.DataFrame:
    primary = person_rows[
        (person_rows["stage"] == PRIMARY_STAGE) & (person_rows["metric"] == PRIMARY_METRIC)
    ]
    return primary[
        ["sample", "tissue", "sites", "windows", "pretrain_value", "retrain_value", "difference"]
    ].sort_values(["tissue", "sample"])


def draw_paired_plot(person_rows: pd.DataFrame, path: Path, bin_rank: int) -> None:
    rows = person_rows[
        (person_rows["stage"] == PRIMARY_STAGE) & (person_rows["metric"] == PRIMARY_METRIC)
    ]
    tissues = sorted(rows["tissue"].unique())
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    for axis, tissue in zip(axes.flat, tissues):
        tissue_rows = rows[rows["tissue"] == tissue]
        for _, row in tissue_rows.iterrows():
            color = "#2E7D32" if row["difference"] > 0 else "#C62828"
            axis.plot([0, 1], [row["pretrain_value"], row["retrain_value"]], color=color, alpha=0.75)
            axis.scatter([0, 1], [row["pretrain_value"], row["retrain_value"]], color=color, s=28)
        axis.set_xticks([0, 1], ["Pretrain E10", "Retrain E3"])
        axis.set_title(f"{tissue} (n={len(tissue_rows)})")
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylabel(f"TEST bin {bin_rank} weighted F1")
    figure.suptitle(f"Paired individual TEST bin {bin_rank} results", fontsize=15)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def draw_forest_plot(
    tissues: pd.DataFrame, combined: pd.DataFrame, path: Path, bin_rank: int
) -> None:
    tissue_rows = tissues[
        (tissues["stage"] == PRIMARY_STAGE) & (tissues["metric"] == PRIMARY_METRIC)
    ].copy()
    combined_row = combined[
        (combined["stage"] == PRIMARY_STAGE) & (combined["metric"] == PRIMARY_METRIC)
    ].iloc[0]
    labels = tissue_rows["tissue"].tolist() + ["Combined (equal tissue weight)"]
    effects = tissue_rows["mean_difference"].tolist() + [combined_row["equal_tissue_mean_difference"]]
    lows = tissue_rows["bootstrap_ci_low"].tolist() + [combined_row["bootstrap_ci_low"]]
    highs = tissue_rows["bootstrap_ci_high"].tolist() + [combined_row["bootstrap_ci_high"]]
    y = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.errorbar(
        effects,
        y,
        xerr=[np.asarray(effects) - np.asarray(lows), np.asarray(highs) - np.asarray(effects)],
        fmt="o",
        color="#2F5597",
        ecolor="#2F5597",
        capsize=4,
    )
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Retrain E3 minus pretrain E10 weighted F1")
    axis.set_title(f"TEST bin {bin_rank} tissue effects with individual-bootstrap 95% CIs")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def format_number(value: float) -> str:
    return f"{value:.6f}"


def write_report(
    path: Path,
    person_rows: pd.DataFrame,
    tissue_results: pd.DataFrame,
    combined: pd.DataFrame,
    qc: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    primary_people = person_rows[
        (person_rows["stage"] == PRIMARY_STAGE) & (person_rows["metric"] == PRIMARY_METRIC)
    ]
    primary_tissues = tissue_results[
        (tissue_results["stage"] == PRIMARY_STAGE)
        & (tissue_results["metric"] == PRIMARY_METRIC)
    ]
    primary_combined = combined[
        (combined["stage"] == PRIMARY_STAGE) & (combined["metric"] == PRIMARY_METRIC)
    ].iloc[0]
    lines = [
        f"# TEST bin {args.bin_rank} personalization significance analysis",
        "",
        "## Scope",
        "",
        "- TEST split only.",
        "- Primary comparison: final retrain E3 versus pretrained epoch 10.",
        "- Primary metric: weighted F1.",
        "- E1/E2 and additional metrics are exploratory.",
        "- Per-person inference resamples paired genomic windows; tissue inference uses people as independent units.",
        "- Directional one-sided p-values are supplemental; the frozen confirmatory report uses two-sided tests.",
        f"- Bootstrap repetitions: {args.bootstrap_reps:,}; permutation repetitions: {args.permutation_reps:,}.",
        "",
        "## Combined primary result",
        "",
        f"Equal-tissue-weight mean difference: {format_number(primary_combined['equal_tissue_mean_difference'])}.",
        f"Individual-bootstrap 95% CI: {format_number(primary_combined['bootstrap_ci_low'])} to {format_number(primary_combined['bootstrap_ci_high'])}.",
        f"One-sided sign-flip Monte Carlo p: {format_number(primary_combined['permutation_p_one_sided'])}.",
        f"Two-sided sign-flip Monte Carlo p: {format_number(primary_combined['permutation_p_two_sided'])}.",
        f"Contributors: {int(primary_combined['individuals'])} people across {int(primary_combined['tissues'])} tissues, "
        f"{int(primary_combined['person_window_instances']):,} person-window instances and {int(primary_combined['sites']):,} CpG rows.",
        "",
        "## Tissue primary results",
        "",
        "| Tissue | People | Unique windows | Person-windows | CpGs | Mean difference | 95% CI | One-sided exact p | BH q |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in primary_tissues.sort_values("tissue").iterrows():
        lines.append(
            f"| {row['tissue']} | {int(row['individuals'])} | {int(row['unique_genomic_windows']):,} | "
            f"{int(row['person_window_instances']):,} | "
            f"{int(row['sites']):,} | {row['mean_difference']:.6f} | "
            f"{row['bootstrap_ci_low']:.6f} to {row['bootstrap_ci_high']:.6f} | "
            f"{row['exact_sign_flip_p_one_sided']:.6f} | "
            f"{row['bh_q_one_sided_primary_six_tissues']:.6f} |"
        )
    significant_people = int(
        (primary_people["permutation_q_one_sided_primary_person_tests"] < 0.05).sum()
    )
    lines.extend(
        [
            "",
            "## Per-person summary",
            "",
            f"Retraining improved weighted F1 in {(primary_people['difference'] > 0).sum()} of {len(primary_people)} people.",
            f"After BH correction across the {len(primary_people)} primary per-person tests, {significant_people} had q < 0.05.",
            "Per-person p-values describe stability across that person's genomic windows; they do not establish population-level generalization.",
            "",
            "## Validation",
            "",
            f"Largest absolute difference from the existing bin-{args.bin_rank} report: {qc['absolute_difference_from_existing_report'].max():.3g}.",
            "No EVAL rows were read by this analysis.",
            "",
            "## Files",
            "",
            f"- `person_bin{args.bin_rank}_resampling_results.csv`: per-person bootstrap and permutation results.",
            f"- `tissue_bin{args.bin_rank}_results.csv`: exact tissue sign-flip tests and individual bootstrap intervals.",
            f"- `combined_bin{args.bin_rank}_results.csv`: equal-tissue-weight combined analysis.",
            f"- `bin{args.bin_rank}_coverage_by_person.csv`: person, window, and CpG counts.",
            f"- `bin{args.bin_rank}_window_membership.csv`: auditable person-to-window membership.",
            "- `metric_reproduction_qc.csv`: comparison with the existing evaluation summaries.",
            "",
            "## Interpretation cautions",
            "",
            "The exact tissue tests have coarse p-value resolution because tissues contain only 4-8 people. "
            "Window-level results can be precise even when population-level evidence is weak. "
            "Nearby or overlapping windows may remain correlated, so the per-person window permutation results are secondary to the person-level tissue analysis.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_ROOT / f"bin{args.bin_rank}_significance_test"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.metric_table, args.samples, args.bin_rank)
    rng = np.random.default_rng(args.seed)
    result_rows = []
    qc_rows = []
    window_rows = []
    samples = manifest[["tissue", "sample"]].drop_duplicates().sort_values(["tissue", "sample"])
    for index, row in enumerate(samples.itertuples(index=False), start=1):
        print(f"[{index}/{len(samples)}] {row.tissue} {row.sample}", flush=True)
        sample_results, sample_qc, sample_windows = analyze_sample(
            manifest,
            args.variability_root,
            row.sample,
            args.bootstrap_reps,
            args.permutation_reps,
            rng,
            args.bin_rank,
        )
        result_rows.extend(sample_results)
        qc_rows.extend(sample_qc)
        window_rows.extend(sample_windows)

    person_results = add_person_fdr(pd.DataFrame(result_rows))
    qc = pd.DataFrame(qc_rows)
    window_membership = pd.DataFrame(window_rows)
    tissue_results = add_unique_window_counts(
        tissue_rows(person_results, args.bootstrap_reps, rng),
        window_membership,
    )
    combined = combined_rows(person_results, args.permutation_reps, rng)
    coverage = coverage_rows(person_results)

    bin_tag = f"bin{args.bin_rank}"
    person_results.to_csv(args.output_dir / f"person_{bin_tag}_resampling_results.csv", index=False)
    tissue_results.to_csv(args.output_dir / f"tissue_{bin_tag}_results.csv", index=False)
    combined.to_csv(args.output_dir / f"combined_{bin_tag}_results.csv", index=False)
    coverage.to_csv(args.output_dir / f"{bin_tag}_coverage_by_person.csv", index=False)
    window_membership.to_csv(args.output_dir / f"{bin_tag}_window_membership.csv", index=False)
    qc.to_csv(args.output_dir / "metric_reproduction_qc.csv", index=False)
    draw_paired_plot(
        person_results,
        args.output_dir / f"{bin_tag}_e3_weighted_f1_paired_by_tissue.png",
        args.bin_rank,
    )
    draw_forest_plot(
        tissue_results,
        combined,
        args.output_dir / f"{bin_tag}_e3_weighted_f1_forest.png",
        args.bin_rank,
    )
    write_report(
        args.output_dir / "REPORT.md",
        person_results,
        tissue_results,
        combined,
        qc,
        args,
    )
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
