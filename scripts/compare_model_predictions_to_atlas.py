"""Compare saved model prediction metrics against atlas evaluation results."""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EPOCH_DIR_PATTERN = re.compile(r"^epoch-(?P<epoch>\d+)-step-(?P<step>\d+)$")
RUN_NAME_PATTERN = re.compile(
    r"^(?P<person_id>[^_]+)_"
    r"(?:(?P<checkpoint>epoch-\d+-step-\d+)|(?P<no_pretraining>no_pretraining))"
    r"_retrain_(?P<tissue>.+)_(?P<split_strategy>kmer|window)"
    r"_lr_(?P<learning_rate>[^_]+)_bs_(?P<batch_size>[^_]+)"
    r"_seq_(?P<seq_len>\d+)_testsize_(?P<test_size>.+)$"
)


def parse_args():
    default_atlas_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "kol_kore_visualize"
        / "atlas_tidy_results.csv"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Read eval_predictions.csv.gitbackup files from a model directory, "
            "score them by variability bin, and compare them to atlas metrics."
        )
    )
    parser.add_argument("models_dir", type=Path, help="Directory containing model run folders.")
    parser.add_argument("variability_dir", type=Path, help="Directory containing per-varaint variability CSV files.")
    parser.add_argument(
        "--atlas-results",
        type=Path,
        default=default_atlas_path if default_atlas_path.exists() else None,
        help=(
            "Optional precomputed atlas tidy/results CSV. Ignored when --target-bigwig "
            "and --atlas-bigwigs are provided. Defaults to the repo atlas_tidy_results.csv when present."
        ),
    )
    parser.add_argument(
        "--target-bigwig",
        type=Path,
        default=None,
        help="Target/sample BigWig path for computing atlas baseline directly.",
    )
    parser.add_argument(
        "--atlas-bigwigs",
        type=Path,
        nargs="+",
        default=None,
        help="Reference atlas BigWig paths used to compute the atlas mean baseline.",
    )
    parser.add_argument(
        "--chromosomes",
        nargs="+",
        default=None,
        help="Chromosomes to evaluate from BigWigs, for example: --chromosomes chr1 chr2 chr3.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_vs_atlas_metrics.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--number-of-bins", type=int, default=5, help="Number of std bins to create.")
    parser.add_argument(
        "--prediction-name",
        default="eval_predictions.csv.gitbackup",
        help="Saved prediction filename to discover under epoch directories.",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Write only the best epoch per run and variability bin by pearsonr.",
    )
    parser.add_argument(
        "--max-prediction-files",
        type=int,
        default=None,
        help="Optional debug limit on discovered prediction files.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress and skipped files.")
    return parser.parse_args()


def parse_epoch_dir(epoch_dir_name):
    match = EPOCH_DIR_PATTERN.match(epoch_dir_name)
    if match is None:
        return None
    return int(match.group("epoch")), int(match.group("step"))


def parse_run_name(run_name):
    match = RUN_NAME_PATTERN.match(run_name)
    if match is None:
        return {
            "person_id": run_name.split("_", 1)[0],
            "run_name": run_name,
            "checkpoint": None,
            "pretraining_mode": None,
            "tissue": None,
            "split_strategy": None,
            "learning_rate": None,
            "batch_size": None,
            "seq_len": None,
            "test_size": None,
        }
    data = match.groupdict()
    data["run_name"] = run_name
    data["pretraining_mode"] = "no_pretraining" if data["no_pretraining"] else "pretrained"
    data.pop("no_pretraining", None)
    data["seq_len"] = int(data["seq_len"])
    return data


def discover_prediction_files(models_dir, prediction_name):
    prediction_files = []
    for path in sorted(models_dir.rglob(prediction_name)):
        epoch_info = parse_epoch_dir(path.parent.name)
        if epoch_info is None:
            continue
        prediction_files.append(path)
    return prediction_files


def find_variability_file(variability_dir, metadata):
    person_id = metadata["person_id"]
    seq_len = metadata.get("seq_len")
    matches = sorted(variability_dir.glob(f"{person_id}_per_varaint_variability_*_datasets.csv"))
    if seq_len is not None:
        seq_matches = [path for path in matches if f"_seq_{seq_len}_" in path.name]
        if seq_matches:
            matches = seq_matches
    if len(matches) == 1:
        return matches[0]
    return None


def add_std_bins(variability_df, number_of_bins):
    if number_of_bins <= 0:
        raise ValueError("--number-of-bins must be positive")
    if variability_df.empty:
        variability_df["std_bin"] = pd.Categorical([])
        return variability_df

    std_values = variability_df["std"].astype(np.float64)
    max_val = std_values.max()
    if pd.isna(max_val):
        variability_df["std_bin"] = pd.Categorical([])
        return variability_df
    if max_val <= 0:
        variability_df["std_bin"] = pd.Categorical(["0.0-0.0"] * len(variability_df), categories=["0.0-0.0"])
        return variability_df

    edges = list(np.linspace(0, max_val, number_of_bins + 1))
    labels = [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]
    variability_df["std_bin"] = pd.cut(
        std_values,
        bins=edges,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    return variability_df


def load_variability_dataframe(variability_path, number_of_bins):
    variability_df = pd.read_csv(variability_path).dropna(subset=["full_position", "std"]).copy()
    full_position_parts = variability_df["full_position"].astype(str).str.extract(
        r"^(?P<chrom>[^:]+):(?P<variant_start>\d+)-(?P<variant_end>\d+)$"
    )
    variability_df["chrom"] = full_position_parts["chrom"]
    variability_df["variant_start"] = pd.to_numeric(full_position_parts["variant_start"], errors="coerce")
    variability_df = variability_df.dropna(subset=["chrom", "variant_start"]).copy()
    variability_df["variant_start"] = variability_df["variant_start"].astype(int)
    return add_std_bins(variability_df, number_of_bins).dropna(subset=["std_bin"])


def load_prediction_dataframe(prediction_path):
    prediction_df = pd.read_csv(prediction_path)
    required_columns = {"window_id", "genomic_position", "label", "prediction"}
    missing_columns = required_columns.difference(prediction_df.columns)
    if missing_columns:
        raise ValueError(f"{prediction_path} is missing columns: {sorted(missing_columns)}")

    prediction_df = prediction_df.copy()
    prediction_df["chrom"] = prediction_df["window_id"].astype(str).str.split(":", n=1).str[0]
    prediction_df["genomic_position"] = pd.to_numeric(prediction_df["genomic_position"], errors="coerce")
    prediction_df["label"] = pd.to_numeric(prediction_df["label"], errors="coerce")
    prediction_df["prediction"] = pd.to_numeric(prediction_df["prediction"], errors="coerce")
    return prediction_df.dropna(subset=["chrom", "genomic_position", "label", "prediction"])


def pearsonr(predictions, labels):
    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(predictions) <= 1:
        return math.nan
    if np.std(predictions) == 0 or np.std(labels) == 0:
        return math.nan
    return float(np.corrcoef(predictions, labels)[0, 1])


def compute_metrics(predictions, labels):
    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return {
        "pearsonr": pearsonr(predictions, labels),
        "mse": float(np.mean((predictions - labels) ** 2)),
        "mae": float(np.mean(np.abs(predictions - labels))),
        "n_positions": int(len(labels)),
    }


def metric_rows_for_prediction(prediction_path, variability_df, metadata, variability_path):
    epoch_num, step_num = parse_epoch_dir(prediction_path.parent.name)
    prediction_df = load_prediction_dataframe(prediction_path)
    merged = prediction_df.merge(
        variability_df[["chrom", "variant_start", "std_bin"]],
        left_on=["chrom", "genomic_position"],
        right_on=["chrom", "variant_start"],
        how="inner",
    )
    merged = merged[merged["label"] != -100].dropna(subset=["label", "prediction", "std_bin"]).copy()
    if merged.empty:
        return []

    categories = list(variability_df["std_bin"].cat.categories)
    rows = []
    for bin_rank, bin_label in enumerate(categories, start=1):
        bin_rows = merged[merged["std_bin"].astype(str) == str(bin_label)]
        if len(bin_rows) <= 1:
            continue
        metrics = compute_metrics(bin_rows["prediction"].to_numpy(), bin_rows["label"].to_numpy())
        rows.append(
            {
                **metadata,
                "epoch": epoch_num,
                "step": step_num,
                "epoch_name": prediction_path.parent.name,
                "prediction_path": str(prediction_path),
                "variability_path": str(variability_path),
                "variability_level": bin_rank,
                "bin_label": str(bin_label),
                **metrics,
            }
        )
    return rows


def load_atlas_results(atlas_path):
    if atlas_path is None or not atlas_path.exists():
        return None

    atlas_df = pd.read_csv(atlas_path)
    if {"metric", "variability_level", "value"}.issubset(atlas_df.columns):
        id_columns = [col for col in ["project_name", "tissue", "split_strategy", "person_id", "seq_len", "variability_level"] if col in atlas_df.columns]
        atlas_wide = (
            atlas_df.pivot_table(index=id_columns, columns="metric", values="value", aggfunc="first")
            .reset_index()
            .rename_axis(None, axis=1)
        )
    else:
        atlas_wide = atlas_df.copy()

    rename_map = {}
    for metric in ["pearsonr", "mse", "mae"]:
        if metric in atlas_wide.columns:
            rename_map[metric] = f"atlas_{metric}"
    atlas_wide = atlas_wide.rename(columns=rename_map)
    keep_columns = [col for col in ["person_id", "seq_len", "variability_level", "atlas_pearsonr", "atlas_mse", "atlas_mae"] if col in atlas_wide.columns]
    return atlas_wide[keep_columns].drop_duplicates()


def atlas_bigwig_results_to_dataframe(atlas_results):
    rows = []
    for bin_rank, (bin_label, pearson_result, mse_result, mae_result) in enumerate(atlas_results, start=1):
        rows.append(
            {
                "variability_level": bin_rank,
                "atlas_bin_label": str(bin_label),
                "atlas_pearsonr": pearson_result.get("pearsonr"),
                "atlas_mse": mse_result.get("mse"),
                "atlas_mae": mae_result.get("mae"),
            }
        )
    return pd.DataFrame(rows)


def compute_atlas_results_from_bigwigs(args):
    refactored_code_dir = Path(__file__).resolve().parents[1]
    if str(refactored_code_dir) not in sys.path:
        sys.path.insert(0, str(refactored_code_dir))

    from src.utils.atlas_bigwig_utils import evaluate_atlas_from_bigwigs

    atlas_results = evaluate_atlas_from_bigwigs(
        target_bigwig_path=str(args.target_bigwig),
        atlas_bigwig_paths=[str(path) for path in args.atlas_bigwigs],
        number_of_bins=args.number_of_bins,
        chroms=args.chromosomes,
        verbose=args.verbose,
    )
    return atlas_bigwig_results_to_dataframe(atlas_results)


def get_atlas_dataframe(args):
    if args.target_bigwig is not None or args.atlas_bigwigs is not None:
        if args.target_bigwig is None or args.atlas_bigwigs is None:
            raise ValueError("--target-bigwig and --atlas-bigwigs must be provided together")
        return compute_atlas_results_from_bigwigs(args)
    return load_atlas_results(args.atlas_results)


def attach_atlas_metrics(model_df, atlas_df):
    if atlas_df is None or atlas_df.empty or model_df.empty:
        return model_df

    join_columns = ["person_id", "seq_len", "variability_level"]
    usable_join_columns = [col for col in join_columns if col in model_df.columns and col in atlas_df.columns]
    if "variability_level" not in usable_join_columns:
        return model_df

    compared = model_df.merge(atlas_df, on=usable_join_columns, how="left")
    for metric in ["pearsonr", "mse", "mae"]:
        atlas_col = f"atlas_{metric}"
        if atlas_col in compared.columns:
            compared[f"{metric}_minus_atlas"] = compared[metric] - compared[atlas_col]
    return compared


def keep_best_rows(df):
    if df.empty:
        return df
    sort_df = df.copy()
    sort_df["_pearsonr_sort"] = sort_df["pearsonr"].fillna(-np.inf)
    sort_df = sort_df.sort_values(
        by=["run_name", "variability_level", "_pearsonr_sort", "epoch", "step"],
        ascending=[True, True, False, False, False],
    )
    return sort_df.drop_duplicates(subset=["run_name", "variability_level"], keep="first").drop(columns=["_pearsonr_sort"])


def main():
    args = parse_args()
    prediction_files = discover_prediction_files(args.models_dir, args.prediction_name)
    if args.max_prediction_files is not None:
        prediction_files = prediction_files[: args.max_prediction_files]
    if args.verbose:
        print(f"found {len(prediction_files)} prediction files under {args.models_dir}", flush=True)

    variability_cache = {}
    rows = []
    skipped = []
    for prediction_path in prediction_files:
        metadata = parse_run_name(prediction_path.parent.parent.name)
        variability_path = find_variability_file(args.variability_dir, metadata)
        if variability_path is None:
            skipped.append((str(prediction_path), "missing or ambiguous variability file"))
            continue
        if variability_path not in variability_cache:
            variability_cache[variability_path] = load_variability_dataframe(variability_path, args.number_of_bins)
        rows.extend(
            metric_rows_for_prediction(
                prediction_path=prediction_path,
                variability_df=variability_cache[variability_path],
                metadata=metadata,
                variability_path=variability_path,
            )
        )

    model_df = pd.DataFrame(rows)
    if args.best_only:
        model_df = keep_best_rows(model_df)

    atlas_df = get_atlas_dataframe(args)
    output_df = attach_atlas_metrics(model_df, atlas_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    print(f"wrote {len(output_df)} rows to {args.output}")
    if args.target_bigwig is not None:
        print(f"atlas target bigwig: {args.target_bigwig}")
        print(f"atlas reference bigwigs: {len(args.atlas_bigwigs)}")
        if args.chromosomes is not None:
            print(f"atlas chromosomes: {', '.join(args.chromosomes)}")
    elif args.atlas_results is not None:
        print(f"atlas results: {args.atlas_results}")
    if skipped and args.verbose:
        print(f"skipped {len(skipped)} prediction files:")
        for path, reason in skipped[:20]:
            print(f"  {reason}: {path}")
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more")


if __name__ == "__main__":
    main()
