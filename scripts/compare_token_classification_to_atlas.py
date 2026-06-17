#!/usr/bin/env python3
"""Compare token-classification predictions against an atlas baseline on matched token/atlas positions."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.atlas_bigwig_utils import build_atlas_position_dataframe
from src.utils.atlas_distribution_utils import DEFAULT_BIGWIG_BASE_PATH, resolve_group_bigwig_paths
from src.utils.token_label_binning_utils import (
    CLOSE_TO_ONE_CLASS_ID,
    CLOSE_TO_ZERO_CLASS_ID,
    IN_BETWEEN_CLASS_ID,
)


PREDICTION_PARENT_PATTERN = re.compile(r"^(?P<sample_id>Z[0-9A-Z]+)_")
TOKEN_GROUP_PATTERN = re.compile(r"^_token_cls_(?P<group_name>.+)$")
CLASS_IDS = [CLOSE_TO_ZERO_CLASS_ID, IN_BETWEEN_CLASS_ID, CLOSE_TO_ONE_CLASS_ID]
CLASS_LABELS = {
    CLOSE_TO_ZERO_CLASS_ID: "close_to_0",
    IN_BETWEEN_CLASS_ID: "middle",
    CLOSE_TO_ONE_CLASS_ID: "close_to_1",
}
JOIN_MODE_CHOICES = ["exact", "token_contains", "token_overlaps"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a token-classification eval_predictions.csv.gitbackup file to an atlas baseline "
            "using exact-site or token-overlap matching."
        )
    )
    parser.add_argument("prediction_path", type=Path, help="Path to token-classification eval_predictions.csv.gitbackup")
    parser.add_argument(
        "--group-name",
        default=None,
        help="Tissue group name for atlas lookup, for example Bladder-Epithelial or Bladder-Epithelial_kmer.",
    )
    parser.add_argument(
        "--held-out-sample",
        default=None,
        help="Held-out sample ID, for example Z0000043F. Inferred from the prediction path when omitted.",
    )
    parser.add_argument(
        "--uber-script-path",
        type=Path,
        default=REPO_ROOT / "src" / "uber_project_creator_script.py",
        help="Path to uber_project_creator_script.py used for atlas sample lookup.",
    )
    parser.add_argument(
        "--bigwig-base-path",
        type=Path,
        default=Path(DEFAULT_BIGWIG_BASE_PATH),
        help="Base directory containing the tissue BigWig files.",
    )
    parser.add_argument(
        "--atlas-low-threshold",
        type=float,
        default=0.2,
        help="Lower atlas methylation threshold mapped to class 0.",
    )
    parser.add_argument(
        "--atlas-high-threshold",
        type=float,
        default=0.8,
        help="Upper atlas methylation threshold mapped to class 2.",
    )
    parser.add_argument(
        "--token-size-bp",
        type=int,
        default=6,
        help="Token width in base pairs for token overlap matching.",
    )
    parser.add_argument(
        "--join-mode",
        choices=JOIN_MODE_CHOICES,
        default="token_contains",
        help="How to match prediction rows to atlas rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for saved CSV outputs. Defaults to a sibling folder next to the prediction file.",
    )
    parser.add_argument(
        "--write-joined-sites",
        action="store_true",
        help="Write the joined token/atlas comparison table.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra progress details.")
    return parser.parse_args()


def infer_sample_id(prediction_path: Path) -> str | None:
    for candidate in prediction_path.parents:
        match = PREDICTION_PARENT_PATTERN.match(candidate.name)
        if match is not None:
            return match.group("sample_id")
    return None


def infer_group_name(prediction_path: Path) -> str | None:
    for candidate in prediction_path.parents:
        match = TOKEN_GROUP_PATTERN.match(candidate.name)
        if match is not None:
            return match.group("group_name")
    return None


def load_prediction_dataframe(prediction_path: Path, token_size_bp: int = 6) -> pd.DataFrame:
    if token_size_bp <= 0:
        raise ValueError(f"token_size_bp must be positive, got {token_size_bp}")

    prediction_df = pd.read_csv(prediction_path)
    required_columns = {"window_id", "genomic_position", "label"}
    missing_columns = required_columns.difference(prediction_df.columns)
    if missing_columns:
        raise ValueError(f"{prediction_path} is missing columns: {sorted(missing_columns)}")

    prediction_df = prediction_df.copy()
    prediction_df["label"] = pd.to_numeric(prediction_df["label"], errors="coerce")
    prediction_df = prediction_df.dropna(subset=["label"])
    prediction_df = prediction_df[prediction_df["label"] != -100].copy()
    prediction_df["label"] = prediction_df["label"].astype(int)

    if "predicted_class" in prediction_df.columns:
        prediction_df["model_class"] = pd.to_numeric(prediction_df["predicted_class"], errors="coerce")
    elif "prediction" in prediction_df.columns:
        prediction_df["model_class"] = pd.to_numeric(prediction_df["prediction"], errors="coerce")
    else:
        raise ValueError(f"{prediction_path} is missing both predicted_class and prediction columns")

    prediction_df["genomic_position"] = pd.to_numeric(prediction_df["genomic_position"], errors="coerce")
    prediction_df = prediction_df.dropna(subset=["model_class", "genomic_position"]).copy()
    prediction_df["model_class"] = prediction_df["model_class"].astype(int)
    prediction_df["genomic_position"] = prediction_df["genomic_position"].astype(int)
    prediction_df["chrom"] = prediction_df["window_id"].astype(str).str.split(":", n=1).str[0]
    prediction_df["site_full_position"] = (
        prediction_df["chrom"].astype(str)
        + ":"
        + prediction_df["genomic_position"].astype(str)
        + "-"
        + (prediction_df["genomic_position"] + 2).astype(str)
    )
    prediction_df["token_start"] = prediction_df["genomic_position"].astype(int)
    prediction_df["token_end"] = prediction_df["token_start"] + int(token_size_bp)
    prediction_df["token_full_position"] = (
        prediction_df["chrom"].astype(str)
        + ":"
        + prediction_df["token_start"].astype(str)
        + "-"
        + prediction_df["token_end"].astype(str)
    )
    if not prediction_df["token_full_position"].is_unique:
        duplicate_count = int(prediction_df["token_full_position"].duplicated().sum())
        raise ValueError(f"{prediction_path} contains {duplicate_count} duplicated token positions")

    keep_columns = [
        column
        for column in [
            "window_id",
            "window_start",
            "window_end",
            "token_index",
            "token_position_in_window",
            "base_position_in_window",
            "genomic_position",
            "chrom",
            "site_full_position",
            "token_start",
            "token_end",
            "token_full_position",
            "label",
            "model_class",
        ]
        if column in prediction_df.columns
    ]
    optional_columns = [col for col in prediction_df.columns if col.startswith("probability_class_")]
    return prediction_df[keep_columns + optional_columns].copy()


def classify_continuous_values(values: pd.Series, low_threshold: float, high_threshold: float) -> pd.Series:
    if low_threshold >= high_threshold:
        raise ValueError(
            f"atlas thresholds must satisfy low < high, got {low_threshold} >= {high_threshold}"
        )
    values = pd.to_numeric(values, errors="coerce")
    classes = np.select(
        [values <= low_threshold, values >= high_threshold],
        [CLOSE_TO_ZERO_CLASS_ID, CLOSE_TO_ONE_CLASS_ID],
        default=IN_BETWEEN_CLASS_ID,
    )
    return pd.Series(classes, index=values.index, dtype="int64")


def build_atlas_site_dataframe(
    prediction_df: pd.DataFrame,
    group_name: str,
    held_out_sample: str,
    uber_script_path: Path,
    bigwig_base_path: Path,
    verbose: bool,
) -> pd.DataFrame:
    resolved = resolve_group_bigwig_paths(
        group_name=group_name,
        held_out_sample=held_out_sample,
        uber_script_path=uber_script_path,
        base_file_path=bigwig_base_path,
    )
    chroms = sorted(prediction_df["chrom"].dropna().unique().tolist())
    atlas_df = build_atlas_position_dataframe(
        target_bigwig_path=resolved["target_bigwig_path"],
        atlas_bigwig_paths=resolved["atlas_bigwig_paths"],
        number_of_bins=5,
        chroms=chroms,
        top_rows=-1,
        test_mode=False,
        jump_sample=-1,
        verbose=verbose,
    )
    if atlas_df.empty:
        raise RuntimeError(
            f"Atlas reference build returned zero rows for sample={held_out_sample}, group={group_name}, chroms={chroms}"
        )
    return atlas_df[["full_position", "chrom", "start", "end", "atlas_mean", "target_value", "std"]].copy()


def join_prediction_tokens_to_atlas_sites(
    prediction_df: pd.DataFrame,
    atlas_df: pd.DataFrame,
    join_mode: str = "token_contains",
) -> pd.DataFrame:
    if join_mode not in JOIN_MODE_CHOICES:
        raise ValueError(f"Unsupported join_mode: {join_mode}")

    atlas_join_df = atlas_df.rename(
        columns={
            "full_position": "atlas_full_position",
            "chrom": "atlas_chrom",
            "start": "atlas_start",
            "end": "atlas_end",
        }
    ).copy()

    if join_mode == "exact":
        joined_df = prediction_df.merge(
            atlas_join_df,
            left_on=["chrom", "site_full_position"],
            right_on=["atlas_chrom", "atlas_full_position"],
            how="inner",
            validate="one_to_one",
        )
        return joined_df

    # Keep "token_contains" as a backwards-compatible alias, but match on any
    # interval overlap, including boundary-touching cases.
    joined_chunks: list[pd.DataFrame] = []
    shared_chroms = sorted(set(prediction_df["chrom"].dropna()) & set(atlas_join_df["atlas_chrom"].dropna()))
    for chrom in shared_chroms:
        pred_subset = (
            prediction_df[prediction_df["chrom"] == chrom]
            .sort_values(["token_start", "token_end", "token_full_position"])
            .reset_index(drop=True)
        )
        atlas_subset = (
            atlas_join_df[atlas_join_df["atlas_chrom"] == chrom]
            .sort_values(["atlas_start", "atlas_end", "atlas_full_position"])
            .reset_index(drop=True)
        )
        if pred_subset.empty or atlas_subset.empty:
            continue

        token_starts = pred_subset["token_start"].to_numpy()
        token_ends = pred_subset["token_end"].to_numpy()
        atlas_starts = atlas_subset["atlas_start"].to_numpy()
        atlas_ends = atlas_subset["atlas_end"].to_numpy()

        first_overlap_indices = np.searchsorted(token_ends, atlas_starts, side="left")
        after_overlap_indices = np.searchsorted(token_starts, atlas_ends, side="right")
        overlap_mask = first_overlap_indices < after_overlap_indices
        if not np.any(overlap_mask):
            continue

        overlap_prediction_indices: list[int] = []
        overlap_atlas_indices: list[int] = []
        overlapping_atlas_row_indices = np.flatnonzero(overlap_mask)
        for atlas_row_index, left_index, right_index in zip(
            overlapping_atlas_row_indices,
            first_overlap_indices[overlap_mask],
            after_overlap_indices[overlap_mask],
        ):
            overlap_prediction_indices.extend(range(int(left_index), int(right_index)))
            overlap_atlas_indices.extend([int(atlas_row_index)] * int(right_index - left_index))

        if not overlap_prediction_indices:
            continue

        matched_prediction_rows = pred_subset.iloc[overlap_prediction_indices].reset_index(drop=True)
        matched_atlas_rows = atlas_subset.iloc[overlap_atlas_indices].reset_index(drop=True)
        joined_chunks.append(pd.concat([matched_prediction_rows, matched_atlas_rows], axis=1))

    if not joined_chunks:
        joined_columns = list(prediction_df.columns) + list(atlas_join_df.columns)
        return pd.DataFrame(columns=joined_columns)

    joined_df = pd.concat(joined_chunks, ignore_index=True)
    return joined_df


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, source_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true_values = y_true.astype(int).to_numpy()
    y_pred_values = y_pred.astype(int).to_numpy()

    overall_df = pd.DataFrame(
        [
            {
                "source": source_name,
                "n_sites": int(len(y_true_values)),
                "accuracy": float(accuracy_score(y_true_values, y_pred_values)),
                "macro_precision": float(
                    precision_score(y_true_values, y_pred_values, labels=CLASS_IDS, average="macro", zero_division=0)
                ),
                "macro_recall": float(
                    recall_score(y_true_values, y_pred_values, labels=CLASS_IDS, average="macro", zero_division=0)
                ),
                "macro_f1": float(
                    f1_score(y_true_values, y_pred_values, labels=CLASS_IDS, average="macro", zero_division=0)
                ),
                "weighted_precision": float(
                    precision_score(y_true_values, y_pred_values, labels=CLASS_IDS, average="weighted", zero_division=0)
                ),
                "weighted_recall": float(
                    recall_score(y_true_values, y_pred_values, labels=CLASS_IDS, average="weighted", zero_division=0)
                ),
                "weighted_f1": float(
                    f1_score(y_true_values, y_pred_values, labels=CLASS_IDS, average="weighted", zero_division=0)
                ),
            }
        ]
    )

    precision_values, recall_values, f1_values, support_values = precision_recall_fscore_support(
        y_true_values,
        y_pred_values,
        labels=CLASS_IDS,
        zero_division=0,
    )
    per_class_rows = []
    for idx, class_id in enumerate(CLASS_IDS):
        per_class_rows.append(
            {
                "source": source_name,
                "class_id": class_id,
                "class_label": CLASS_LABELS[class_id],
                "precision": float(precision_values[idx]),
                "recall": float(recall_values[idx]),
                "f1": float(f1_values[idx]),
                "support": int(support_values[idx]),
            }
        )
    return overall_df, pd.DataFrame(per_class_rows)


def confusion_to_dataframe(y_true: pd.Series, y_pred: pd.Series, source_name: str) -> pd.DataFrame:
    confusion = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=CLASS_IDS)
    rows = []
    for true_idx, true_class in enumerate(CLASS_IDS):
        row_total = int(confusion[true_idx].sum())
        for pred_idx, pred_class in enumerate(CLASS_IDS):
            count = int(confusion[true_idx, pred_idx])
            rows.append(
                {
                    "source": source_name,
                    "true_class_id": true_class,
                    "true_class_label": CLASS_LABELS[true_class],
                    "pred_class_id": pred_class,
                    "pred_class_label": CLASS_LABELS[pred_class],
                    "count": count,
                    "row_fraction": float(count / row_total) if row_total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    prediction_path = args.prediction_path.resolve()
    if not prediction_path.exists():
        raise FileNotFoundError(f"prediction file was not found: {prediction_path}")

    held_out_sample = args.held_out_sample or infer_sample_id(prediction_path)
    if held_out_sample is None:
        raise ValueError("Could not infer held-out sample from prediction path. Please provide --held-out-sample.")

    group_name = args.group_name or infer_group_name(prediction_path)
    if group_name is None:
        raise ValueError("Could not infer tissue group from prediction path. Please provide --group-name.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = prediction_path.parent / "atlas_token_classification_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = load_prediction_dataframe(prediction_path, token_size_bp=args.token_size_bp)
    atlas_df = build_atlas_site_dataframe(
        prediction_df=prediction_df,
        group_name=group_name,
        held_out_sample=held_out_sample,
        uber_script_path=args.uber_script_path,
        bigwig_base_path=args.bigwig_base_path,
        verbose=args.verbose,
    )

    joined_df = join_prediction_tokens_to_atlas_sites(
        prediction_df=prediction_df,
        atlas_df=atlas_df,
        join_mode=args.join_mode,
    )
    if joined_df.empty:
        raise RuntimeError("The atlas join produced zero rows; no shared evaluated sites were found.")

    joined_df["atlas_class"] = classify_continuous_values(
        joined_df["atlas_mean"],
        low_threshold=args.atlas_low_threshold,
        high_threshold=args.atlas_high_threshold,
    )
    joined_df["atlas_target_class"] = classify_continuous_values(
        joined_df["target_value"],
        low_threshold=args.atlas_low_threshold,
        high_threshold=args.atlas_high_threshold,
    )

    target_class_match_count = int((joined_df["label"] == joined_df["atlas_target_class"]).sum())
    target_class_mismatch_count = int(len(joined_df) - target_class_match_count)

    overall_frames = []
    per_class_frames = []
    confusion_frames = []
    for source_name, prediction_column in [("model", "model_class"), ("atlas", "atlas_class")]:
        overall_df, per_class_df = compute_metrics(joined_df["label"], joined_df[prediction_column], source_name)
        overall_frames.append(overall_df)
        per_class_frames.append(per_class_df)
        confusion_frames.append(confusion_to_dataframe(joined_df["label"], joined_df[prediction_column], source_name))

    overall_metrics_df = pd.concat(overall_frames, ignore_index=True)
    overall_metrics_df.insert(0, "prediction_path", str(prediction_path))
    overall_metrics_df.insert(1, "held_out_sample", held_out_sample)
    overall_metrics_df.insert(2, "group_name", group_name)
    overall_metrics_df.insert(3, "join_mode", args.join_mode)
    overall_metrics_df.insert(4, "token_size_bp", int(args.token_size_bp))
    overall_metrics_df.insert(5, "atlas_low_threshold", float(args.atlas_low_threshold))
    overall_metrics_df.insert(6, "atlas_high_threshold", float(args.atlas_high_threshold))
    overall_metrics_df["prediction_token_count"] = int(len(prediction_df))
    overall_metrics_df["joined_pair_count"] = int(len(joined_df))
    overall_metrics_df["joined_token_count"] = int(joined_df["token_full_position"].nunique())
    overall_metrics_df["joined_atlas_site_count"] = int(joined_df["atlas_full_position"].nunique())
    overall_metrics_df["joined_pair_fraction_vs_prediction_tokens"] = float(len(joined_df) / len(prediction_df))
    overall_metrics_df["target_class_match_count"] = target_class_match_count
    overall_metrics_df["target_class_mismatch_count"] = target_class_mismatch_count

    per_class_metrics_df = pd.concat(per_class_frames, ignore_index=True)
    per_class_metrics_df.insert(0, "prediction_path", str(prediction_path))
    per_class_metrics_df.insert(1, "held_out_sample", held_out_sample)
    per_class_metrics_df.insert(2, "group_name", group_name)
    per_class_metrics_df.insert(3, "join_mode", args.join_mode)
    per_class_metrics_df.insert(4, "token_size_bp", int(args.token_size_bp))

    confusion_df = pd.concat(confusion_frames, ignore_index=True)
    confusion_df.insert(0, "prediction_path", str(prediction_path))
    confusion_df.insert(1, "held_out_sample", held_out_sample)
    confusion_df.insert(2, "group_name", group_name)
    confusion_df.insert(3, "join_mode", args.join_mode)
    confusion_df.insert(4, "token_size_bp", int(args.token_size_bp))

    overall_metrics_path = output_dir / "overall_metrics.csv"
    per_class_metrics_path = output_dir / "per_class_metrics.csv"
    confusion_path = output_dir / "confusion_counts.csv"
    overall_metrics_df.to_csv(overall_metrics_path, index=False)
    per_class_metrics_df.to_csv(per_class_metrics_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)

    if args.write_joined_sites:
        joined_output_columns = [
            column
            for column in [
                "window_id",
                "window_start",
                "window_end",
                "token_index",
                "token_position_in_window",
                "base_position_in_window",
                "genomic_position",
                "chrom",
                "site_full_position",
                "token_start",
                "token_end",
                "token_full_position",
                "label",
                "model_class",
                "atlas_full_position",
                "atlas_chrom",
                "atlas_start",
                "atlas_end",
                "atlas_mean",
                "atlas_class",
                "atlas_target_class",
                "target_value",
                "std",
            ]
            if column in joined_df.columns
        ]
        optional_columns = [col for col in joined_df.columns if col.startswith("probability_class_")]
        joined_output_df = joined_df[joined_output_columns + optional_columns].copy()
        joined_output_df.to_csv(output_dir / "joined_sites.csv.gz", index=False, compression="gzip")

    print(f"prediction_path: {prediction_path}")
    print(f"held_out_sample: {held_out_sample}")
    print(f"group_name: {group_name}")
    print(f"join_mode: {args.join_mode}")
    print(f"token_size_bp: {args.token_size_bp}")
    print(f"prediction_tokens_in_file: {len(prediction_df):,}")
    print(f"joined_token_atlas_pairs: {len(joined_df):,}")
    print(f"joined_unique_tokens: {joined_df['token_full_position'].nunique():,}")
    print(f"joined_unique_atlas_sites: {joined_df['atlas_full_position'].nunique():,}")
    print(f"atlas_target_class_matches_label: {target_class_match_count:,}")
    print(f"atlas_target_class_mismatches_label: {target_class_mismatch_count:,}")
    print()
    print("overall metrics:")
    print(overall_metrics_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("per-class metrics:")
    print(per_class_metrics_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(f"wrote overall metrics to {overall_metrics_path}")
    print(f"wrote per-class metrics to {per_class_metrics_path}")
    print(f"wrote confusion counts to {confusion_path}")
    if args.write_joined_sites:
        print(f"wrote joined token/atlas table to {output_dir / 'joined_sites.csv.gz'}")


if __name__ == "__main__":
    main()
