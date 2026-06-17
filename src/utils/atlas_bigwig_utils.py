import numpy as np
import pandas as pd
import pyBigWig

try:
    from .bigwig_utils import get_preprocessed_encode_cpg_chrom_dataframe_from_bw
except ImportError:
    from bigwig_utils import get_preprocessed_encode_cpg_chrom_dataframe_from_bw


def _load_metric_objects():
    try:
        from .metrics_utils import mae, mse, pearsonr_eval
    except ImportError:
        from metrics_utils import mae, mse, pearsonr_eval
    return pearsonr_eval, mse, mae


FULL_POSITION_COLUMN_NAME = "full_position"
ATLAS_POSITION_COLUMNS = [
    FULL_POSITION_COLUMN_NAME,
    "chrom",
    "start",
    "end",
    "std",
    "atlas_mean",
    "target_value",
    "std_bin",
]


def _normalize_chromosomes(chroms, target_bw, atlas_bws):
    available = set(target_bw.chroms().keys())
    for bw in atlas_bws:
        available &= set(bw.chroms().keys())

    if chroms is None:
        ordered = [chrom for chrom in target_bw.chroms().keys() if chrom in available]
    else:
        if isinstance(chroms, str):
            chroms = [chroms]
        ordered = [chrom for chrom in chroms if chrom in available]
    return ordered


def _add_full_position_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df = df.copy()
    df[FULL_POSITION_COLUMN_NAME] = (
        df["chrom"].astype(str)
        + ":"
        + df["start"].astype(str)
        + "-"
        + df["end"].astype(str)
    )
    return df


def _load_chrom_dataframe(bw, chrom: str) -> pd.DataFrame:
    return _add_full_position_column(get_preprocessed_encode_cpg_chrom_dataframe_from_bw(bw, chrom))


def _merge_reference_dataframes(reference_dfs):
    if not reference_dfs:
        return pd.DataFrame(columns=[FULL_POSITION_COLUMN_NAME, "chrom", "start", "end", "std", "atlas_mean"])
    if any(df.empty for df in reference_dfs):
        return pd.DataFrame(columns=[FULL_POSITION_COLUMN_NAME, "chrom", "start", "end", "std", "atlas_mean"])

    merged = reference_dfs[0][[FULL_POSITION_COLUMN_NAME, "chrom", "start", "end", "methyl_rate"]].rename(
        columns={"methyl_rate": "methyl_rate_ind_0"}
    )
    for idx, df in enumerate(reference_dfs[1:], start=1):
        renamed = df[[FULL_POSITION_COLUMN_NAME, "methyl_rate"]].rename(
            columns={"methyl_rate": f"methyl_rate_ind_{idx}"}
        )
        merged = merged.merge(renamed, on=FULL_POSITION_COLUMN_NAME, how="inner")
        if merged.empty:
            return pd.DataFrame(columns=[FULL_POSITION_COLUMN_NAME, "chrom", "start", "end", "std", "atlas_mean"])

    value_columns = [col for col in merged.columns if col.startswith("methyl_rate_ind_")]
    merged["std"] = merged[value_columns].std(axis=1)
    merged["atlas_mean"] = merged[value_columns].mean(axis=1)
    return merged[[FULL_POSITION_COLUMN_NAME, "chrom", "start", "end", "std", "atlas_mean"]]


def _apply_debug_controls(matched_df: pd.DataFrame, top_rows: int, test_mode: bool, jump_sample: int) -> pd.DataFrame:
    if matched_df.empty:
        return matched_df
    if top_rows is not None and top_rows > 0:
        matched_df = matched_df.head(top_rows).copy()
    if test_mode and jump_sample is not None and jump_sample > 0:
        matched_df = matched_df.iloc[::jump_sample].reset_index(drop=True)
    return matched_df


def _add_std_bins_to_dataframe(number_of_bins: int, variant_file_dataframe: pd.DataFrame) -> pd.DataFrame:
    if number_of_bins <= 0:
        raise ValueError("number_of_bins must be a positive integer")
    if variant_file_dataframe.empty:
        variant_file_dataframe["std_bin"] = pd.Series(dtype="object")
        return variant_file_dataframe

    std_values = variant_file_dataframe["std"].astype(np.float64)
    max_val = std_values.max()
    if pd.isna(max_val):
        variant_file_dataframe["std_bin"] = pd.Series([pd.NA] * len(variant_file_dataframe), dtype="object")
        return variant_file_dataframe
    if max_val <= 0:
        variant_file_dataframe["std_bin"] = "0.0-0.0"
        return variant_file_dataframe

    edges = list(np.linspace(0, max_val, number_of_bins + 1))
    labels = []
    for i in range(len(edges) - 1):
        labels.append(f"{edges[i]}-{edges[i + 1]}")
    variant_file_dataframe["std_bin"] = pd.cut(
        std_values,
        bins=edges,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    return variant_file_dataframe


def build_atlas_position_dataframe(
    target_bigwig_path: str,
    atlas_bigwig_paths,
    number_of_bins: int,
    chroms=None,
    top_rows: int = -1,
    test_mode: bool = False,
    jump_sample: int = -1,
    verbose: bool = False,
) -> pd.DataFrame:
    if target_bigwig_path is None:
        raise ValueError("atlas evaluation requires a target_bigwig_path")
    if atlas_bigwig_paths is None or len(atlas_bigwig_paths) == 0:
        raise ValueError("atlas evaluation requires atlas_bigwig_paths")

    target_bw = pyBigWig.open(target_bigwig_path)
    atlas_bws = [pyBigWig.open(path) for path in atlas_bigwig_paths]
    try:
        ordered_chroms = _normalize_chromosomes(chroms, target_bw, atlas_bws)
        chrom_order = {chrom: idx for idx, chrom in enumerate(ordered_chroms)}
        matched_dfs = []

        for chrom_index, chrom in enumerate(ordered_chroms):
            if verbose:
                print(f"processing chromosome {chrom_index + 1}/{len(ordered_chroms)}: {chrom}", flush=True)
            reference_dfs = [_load_chrom_dataframe(bw, chrom) for bw in atlas_bws]
            reference_df = _merge_reference_dataframes(reference_dfs)
            if reference_df.empty:
                continue

            target_df = _load_chrom_dataframe(target_bw, chrom)
            if target_df.empty:
                continue

            target_df = target_df[[FULL_POSITION_COLUMN_NAME, "methyl_rate"]].rename(columns={"methyl_rate": "target_value"})
            merged = reference_df.merge(target_df, on=FULL_POSITION_COLUMN_NAME, how="inner")
            if merged.empty:
                continue
            merged["chrom_order"] = chrom_order[chrom]
            matched_dfs.append(merged)

        if not matched_dfs:
            return pd.DataFrame(columns=ATLAS_POSITION_COLUMNS)

        matched_df = pd.concat(matched_dfs, ignore_index=True)
        matched_df = matched_df.dropna(subset=["std", "atlas_mean", "target_value"])
        matched_df = matched_df.sort_values(by=["chrom_order", "start", "end"]).reset_index(drop=True)
        matched_df = _apply_debug_controls(matched_df, top_rows=top_rows, test_mode=test_mode, jump_sample=jump_sample)
        if matched_df.empty:
            return pd.DataFrame(columns=ATLAS_POSITION_COLUMNS)

        matched_df = _add_std_bins_to_dataframe(number_of_bins, matched_df)
        matched_df = matched_df.drop(columns=["chrom_order"], errors="ignore")
        return matched_df[ATLAS_POSITION_COLUMNS]
    finally:
        target_bw.close()
        for bw in atlas_bws:
            bw.close()


def evaluate_atlas_from_bigwigs(
    target_bigwig_path: str,
    atlas_bigwig_paths,
    number_of_bins: int,
    chroms=None,
    top_rows: int = -1,
    test_mode: bool = False,
    jump_sample: int = -1,
    verbose: bool = False,
):
    matched_df = build_atlas_position_dataframe(
        target_bigwig_path=target_bigwig_path,
        atlas_bigwig_paths=atlas_bigwig_paths,
        number_of_bins=number_of_bins,
        chroms=chroms,
        top_rows=top_rows,
        test_mode=test_mode,
        jump_sample=jump_sample,
        verbose=verbose,
    )
    if matched_df.empty:
        return []

    pearsonr_eval, mse, mae = _load_metric_objects()

    res = []
    for curr_bin in pd.unique(matched_df["std_bin"]):
        if pd.isna(curr_bin):
            continue
        curr_df = matched_df[matched_df["std_bin"] == curr_bin]
        if len(curr_df) <= 1:
            continue

        labels_from_dataset = curr_df["target_value"].to_numpy(dtype=np.float64)
        labels_from_prediction = curr_df["atlas_mean"].to_numpy(dtype=np.float64)
        if verbose:
            print("curr_bin:", curr_bin, flush=True)
            print("legnth_labels:", len(labels_from_dataset), flush=True)

        res_r = pearsonr_eval.compute(predictions=labels_from_prediction, references=labels_from_dataset)
        res_mse = mse.compute(predictions=labels_from_prediction, references=labels_from_dataset)
        res_mae = mae.compute(predictions=labels_from_prediction, references=labels_from_dataset)
        bin_results = [str(curr_bin), res_r, res_mse, res_mae]
        if verbose:
            print(bin_results, flush=True)
        res.append(bin_results)
    return res
