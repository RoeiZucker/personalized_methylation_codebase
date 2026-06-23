import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from utils.bigwig_utils import load_preprocessed_encode_cpg_dfs
from utils.formatting import combine_cpg_dfs

def evaluate_sample_predictions(result_files_path,
                                chroms,
                                comparison_bigwig_files=None,
                                full_pos_name=None,
                                ranges=None,
                                labels=None,
                                comparison_types=None,
                                all_two=False,
                                verbose=True,
                                comparison_dicts=None,
                                **legacy_kwargs):
    """Evaluate prediction results against dataset.

    Parameters:
        result_files_path: Iterable of CSV file paths with prediction results.
        chroms: Iterable of chromosome names to evaluate.
        comparison_bigwig_files: Files containing preprocessed comparison CpG signals.
        full_pos_name: Column name or key used for genomic position matching.
        ranges: Numeric boundaries for creating mean-based label bins.
        labels: Expected label values, matching the ranges.
        comparison_types: Types of comparisons to include in evaluation.
        all_two: If True, append an all_two comparison type with constant value 2.
        verbose: Whether to print progress information.
        comparison_dicts: Precomputed comparison dictionaries.

    Returns:
        A dictionary of evaluation objects keyed by result file path.
    """
    if "comparison_bigiwg_files" in legacy_kwargs:
        if comparison_bigwig_files is not None:
            raise TypeError("Pass only one of comparison_bigwig_files or comparison_bigiwg_files")
        comparison_bigwig_files = legacy_kwargs.pop("comparison_bigiwg_files")
    if "verbous" in legacy_kwargs:
        verbose = legacy_kwargs.pop("verbous")
    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    if comparison_dicts is None:
        compare_dicts = create_comparison_dicts(comparison_bigwig_files,chroms,full_pos_name)
    else:
        compare_dicts = comparison_dicts
    eval_objects_dict = {}

    for result_file_path in result_files_path:
        if verbose:
            print("curr result file",result_file_path,flush=True)
        curr_result_eval = {}
        eval_objects_dict[result_file_path] = curr_result_eval
        new_result_file = create_result_file_mean_label(result_file_path, compare_dicts,ranges)
        if all_two:
            new_result_file["all_two"] = 2
        curr_comparison_types = list(comparison_types)
        if all_two and "all_two" not in curr_comparison_types:
            curr_comparison_types.append("all_two")
        eval_object = create_eval_object(new_result_file,curr_comparison_types,labels)
        curr_result_eval["all_results"] = eval_object

    return eval_objects_dict

def create_eval_object(new_result_file,comparison_types,labels):
    eval_object = {}
    true_labels = new_result_file["label"]
    for prediction_type in comparison_types:
        predicted_labels = new_result_file[prediction_type]
        eval_object[prediction_type] = {}
        eval_object[prediction_type + "_confusion_matrix"] = pd.crosstab(true_labels, predicted_labels).to_dict()
        for label in labels:
            specific_label = true_labels == label
            specific_type_label = predicted_labels == label
            precision = precision_score(specific_label, specific_type_label, zero_division=0)
            recall = recall_score(specific_label, specific_type_label, zero_division=0)
            eval_object[prediction_type][label] = {
                "Precision": precision,
                "recall": recall,
            }
    return eval_object

def create_comparison_dicts(comparison_bigwig_files,chroms,full_pos_name):
    labels = []
    comparison_dfs = load_preprocessed_encode_cpg_dfs(comparison_bigwig_files,chroms,full_pos_name,False)
    for i in range(len(comparison_dfs)):
        labels.append(f"ind_{i}")
    combined_compare_df = combine_cpg_dfs(full_pos_name,comparison_dfs,labels)
    compare_dicts = {}
    for chrom in chroms:
        compare_dicts[chrom] = combined_compare_df[combined_compare_df["chrom"] == chrom].set_index("start").to_dict(orient='index')
    return compare_dicts

def apply_create_means(row,compare_dicts):
    values = []
    genomic_position = row["genomic_position"]
    chrom = row["chrom"]
    for i in range(genomic_position - 1,genomic_position+6):
        if i in compare_dicts[chrom]:
            temp_values = []
            for key in compare_dicts[chrom][i]:
                if "methyl_rate_ind" in key:
                    temp_values.append(compare_dicts[chrom][i][key])
            values.append(np.mean(temp_values) if temp_values else np.nan)
    if len(values) == 0:
        return None
    return np.mean(values)


def apply_create_labels(row,ranges):
    """Deprecated row-wise label helper kept for regression checks."""
    mean_value = row["mean_value"]
    for i in range(len(ranges) - 1):
        if ranges[i] <= mean_value and mean_value <= ranges[i+1]:
            return i


def create_mean_labels(mean_values,ranges):
    """Create labels for in-range means.

    Threshold values belong to the lower bin, matching apply_create_labels.
    Out-of-range means intentionally stay missing.
    """
    values = pd.Series(mean_values)
    ranges_array = np.asarray(ranges)
    value_array = values.to_numpy()
    labels = np.searchsorted(ranges_array[1:], value_array, side="left")
    in_range = (value_array >= ranges_array[0]) & (value_array <= ranges_array[-1])
    valid_values = values.notna().to_numpy() & in_range
    if valid_values.all():
        return pd.Series(labels.astype(int), index=values.index)
    result = pd.Series(np.nan, index=values.index)
    result.iloc[valid_values] = labels[valid_values]
    return result


def create_result_file_mean_label(result_file_path, compare_dicts,ranges):
    result_file = create_result_file_mean_value(result_file_path, compare_dicts)
    new_result_file = result_file.dropna(subset=["mean_value"]).copy()
    new_result_file["mean_label"] = create_mean_labels(new_result_file["mean_value"],ranges)
    return new_result_file

def create_position_mean_lookup(compare_dicts):
    position_mean_lookup = {}
    for chrom, positions in compare_dicts.items():
        position_means = {}
        for position, values_by_key in positions.items():
            methyl_values = [
                value
                for key, value in values_by_key.items()
                if "methyl_rate_ind" in key
            ]
            position_means[position] = np.mean(methyl_values) if methyl_values else np.nan
        position_mean_lookup[chrom] = pd.Series(position_means, dtype="float64")
    return position_mean_lookup


def create_mean_values(result_file, compare_dicts):
    position_mean_lookup = create_position_mean_lookup(compare_dicts)
    mean_values = pd.Series(np.nan, index=result_file.index, dtype="float64")
    for chrom, chrom_rows in result_file.groupby("chrom", sort=False):
        if chrom not in position_mean_lookup:
            raise KeyError(chrom)
        chrom_position_means = position_mean_lookup[chrom]
        summed_values = pd.Series(0.0, index=chrom_rows.index)
        matched_counts = pd.Series(0, index=chrom_rows.index)
        has_nan_value = pd.Series(False, index=chrom_rows.index)
        genomic_positions = chrom_rows["genomic_position"]

        for offset in range(-1, 6):
            candidate_positions = genomic_positions + offset
            matched_positions = candidate_positions.isin(chrom_position_means.index)
            candidate_means = candidate_positions.map(chrom_position_means)
            summed_values += candidate_means.where(matched_positions, 0).fillna(0)
            matched_counts += matched_positions.astype(int)
            has_nan_value |= matched_positions & candidate_means.isna()

        valid_values = (matched_counts > 0) & ~has_nan_value
        mean_values.loc[chrom_rows.index[valid_values]] = (
            summed_values.loc[valid_values] / matched_counts.loc[valid_values]
        )
    return mean_values


def create_result_file_mean_value(result_file_path, compare_dicts):
    result_file = pd.read_csv(result_file_path)
    result_file["label"] = result_file["label"].astype(int)
    result_file["chrom"] = result_file["window_id"].str.split(":").str[0]
    result_file["mean_value"] = create_mean_values(result_file, compare_dicts)
    return result_file
