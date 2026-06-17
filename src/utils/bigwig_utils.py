import numpy as np
import pandas as pd
import pyBigWig


def get_chrom_intervals(chrom, length, bw):
    all_values = bw.intervals(chrom, 0, length)
    return all_values


def cpg_values_generator(bw, chroms=None, verbose=False):
    for chrom, length in bw.chroms().items():
        if (chroms is not None) and (chrom not in chroms):
            continue
        if verbose:
            print(f"Processing chromosome: {chrom} with length: {length}", flush=True)
        res = get_chrom_intervals(chrom, length, bw)
        if res is not None:
            yield from map(lambda x: list(x) + [chrom], res)


def preprocess_encode_cpg_dataframe(df: pd.DataFrame, remove_under_0=False) -> pd.DataFrame:
    if df.empty:
        return df

    remove_under_0 = "filter"
    df["methyl_rate"] = df["methyl_rate"].astype(np.float16)
    if remove_under_0 == "filter":
        df = df[df["methyl_rate"] > 0]
    elif remove_under_0:
        df.loc[df["methyl_rate"] < 0, "methyl_rate"] = -100.0
    return df


def get_preprocessed_encode_cpg_chrom_dataframe_from_bw(bw, chrom: str, remove_under_0=False) -> pd.DataFrame:
    chrom_length = bw.chroms().get(chrom)
    if chrom_length is None:
        return pd.DataFrame(columns=["start", "end", "methyl_rate", "chrom"])

    intervals = get_chrom_intervals(chrom, chrom_length, bw)
    if intervals is None:
        return pd.DataFrame(columns=["start", "end", "methyl_rate", "chrom"])

    df = pd.DataFrame(intervals, columns=["start", "end", "methyl_rate"])
    df["chrom"] = chrom
    return preprocess_encode_cpg_dataframe(df, remove_under_0=remove_under_0)


def get_preprocessed_encode_cpg_dataframe(bigwig_file_path: str, chroms=None, remove_under_0=False, verbose=False) -> pd.DataFrame:
    bw = pyBigWig.open(bigwig_file_path)
    try:
        df = pd.DataFrame(
            cpg_values_generator(bw, chroms, verbose),
            columns=["start", "end", "methyl_rate", "chrom"],
        )
        return preprocess_encode_cpg_dataframe(df, remove_under_0=remove_under_0)
    finally:
        bw.close()

# TODO: add so that if a single path is given and not a list of paths, load only that path
def load_preprocessed_encode_cpg_dfs(raw_files_paths, chroms, full_position_column_name,verbose):
    dfs = []
    if verbose:
        print("start loading",flush=True)
        print(raw_files_paths,flush=True)
    for path in raw_files_paths:
        dfs.append(get_preprocessed_encode_cpg_dataframe(path,chroms,verbose=verbose))
    if verbose:
        print("stop loading",flush=True)
    for df in dfs:
        # print("count:", len(df))
        # NOTE: This was changed from start to end (commented out line) make sure it works
        # df[full_position_column_name] =  df["chrom"] + ":" + df["start"].astype(str)  + "-" + df["start"].astype(str)
        df[full_position_column_name] =  df["chrom"] + ":" + df["start"].astype(str)  + "-" + df["end"].astype(str)
    return dfs