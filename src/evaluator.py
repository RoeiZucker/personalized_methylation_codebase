import re
from pathlib import Path

try:
    from datasets import Dataset, load_from_disk
except ImportError:
    Dataset = None
    load_from_disk = None

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score
from scipy.stats import pearsonr
from constants import (
    CPG_EVALUATION_TASK_TYPE
)
try:
    from .variability_free_evaluation import (
        apply_create_labels as variability_free_apply_create_labels,
        apply_create_means as variability_free_apply_create_means,
        create_result_file_mean_label as variability_free_create_result_file_mean_label,
        create_result_file_mean_value as variability_free_create_result_file_mean_value,
    )
except ImportError:
    from variability_free_evaluation import (
        apply_create_labels as variability_free_apply_create_labels,
        apply_create_means as variability_free_apply_create_means,
        create_result_file_mean_label as variability_free_create_result_file_mean_label,
        create_result_file_mean_value as variability_free_create_result_file_mean_value,
    )
class _SimpleMetric:
    def __init__(self, func, key):
        self.func = func
        self.key = key

    def compute(self, predictions, references):
        return {self.key: self.func(predictions, references)}


try:
    from .utils.metrics_utils import(
        pearsonr_eval,
        mse,
        mae
    )
except ImportError:
    def _pearson_metric(predictions, references):
        predictions = np.asarray(predictions)
        references = np.asarray(references)
        if len(predictions) < 2 or len(references) < 2:
            return np.nan
        return pearsonr(predictions, references)[0]

    def _mse_metric(predictions, references):
        return mean_squared_error(references, predictions)

    def _mae_metric(predictions, references):
        return mean_absolute_error(references, predictions)

    pearsonr_eval = _SimpleMetric(_pearson_metric, 'pearsonr')
    mse = _SimpleMetric(_mse_metric, 'mse')
    mae = _SimpleMetric(_mae_metric, 'mae')

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from .utils.trainer_utils import(
        get_compute_func,
        get_trainer,
        get_trainer_type,
    )
    from .utils.dataset_utils import (
        keep_batch
    )
    from .utils.atlas_bigwig_utils import evaluate_atlas_from_bigwigs
    from .utils.label_transform_utils import (
        apply_label_transform_to_dataset,
        get_task_label_transform,
        maybe_decode_regression_predictions,
    )
except ImportError:
    get_compute_func = None
    get_trainer = None
    get_trainer_type = None
    keep_batch = None
    evaluate_atlas_from_bigwigs = None
    apply_label_transform_to_dataset = None
    get_task_label_transform = None
    maybe_decode_regression_predictions = None


# TODO: add evaluate_multiple_checkpoints constant, single as well
try:
    from utils.model_utils import get_fine_tuned_model
except ImportError:
    try:
        from utils.model_utils import get_fine_tuned_model
    except ImportError:
        get_fine_tuned_model = None

from utils.bigwig_utils import load_preprocessed_encode_cpg_dfs
from utils.formatting import combine_cpg_dfs

SAVED_PREDICTIONS_SUBTASK = "evaluate_saved_predictions"
EPOCH_DIR_PATTERN = re.compile(r"^epoch-(?P<epoch>\d+)-step-(?P<step>\d+)$")
RUN_NAME_PATTERN = re.compile(
    r"^(?P<held_out_sample>[^_]+)_"
    r"(?:(?P<checkpoint>epoch-\d+-step-\d+)|(?P<no_pretraining>no_pretraining))"
    r"_retrain_(?P<tissue_name>.+)_(?P<split_type>kmer|window)"
    r"_lr_(?P<learning_rate>[^_]+)_bs_(?P<batch_size>[^_]+)"
    r"_seq_(?P<seq_size>[^_]+)_testsize_(?P<test_size>.+)$"
)
CROSS_FIT_SELECTION_STRATEGY = "two_fold_cross_fit"
PREDICTION_RESULTS_COLUMNS = [
    "family_key",
    "run_name",
    "prediction_path",
    "selected_epoch",
    "selected_epoch_fold_0",
    "selected_epoch_fold_1",
    "selection_score_fold_0",
    "selection_score_fold_1",
    "candidate_epoch_count",
    "selection_strategy",
    "selection_metric",
    "selection_bin_rank",
    "selection_bin_label",
    "held_out_sample",
    "pretraining_mode",
    "pretraining_bucket",
    "tissue_name",
    "split_type",
    "seq_size",
    "learning_rate",
    "batch_size",
    "test_size",
    "bin_rank",
    "bin_label",
    "bin_lower",
    "bin_upper",
    "pearsonr",
    "mse",
    "mae",
    "n_positions",
]
PREDICTION_SUMMARY_COLUMNS = [
    "family_key",
    "selection_strategy",
    "selection_metric",
    "selection_bin_rank",
    "selection_bin_label",
    "pretraining_mode",
    "pretraining_bucket",
    "tissue_name",
    "split_type",
    "seq_size",
    "learning_rate",
    "batch_size",
    "test_size",
    "run_count",
    "sample_count",
    "bin_rank",
    "bin_label",
    "bin_lower",
    "bin_upper",
    "pearsonr_mean",
    "pearsonr_std",
    "mse_mean",
    "mse_std",
    "mae_mean",
    "mae_std",
    "n_positions_mean",
]

def evaluate_sample_predictions(variability_file_path,
                                result_files_path,
                                chroms,
                                comparison_bigwig_files=None,
                                full_pos_name=None,
                                ranges=None,
                                labels=None,
                                label_a=None,
                                label_b=None,
                                number_of_bins=None,
                                comparison_dicts=None,
                                **legacy_kwargs):
    if "comparison_bigiwg_files" in legacy_kwargs:
        if comparison_bigwig_files is not None:
            raise TypeError("Pass only one of comparison_bigwig_files or comparison_bigiwg_files")
        comparison_bigwig_files = legacy_kwargs.pop("comparison_bigiwg_files")
    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    variability_file = pd.read_csv(variability_file_path)
    if comparison_dicts is None:
        compare_dicts = create_comparison_dicts(comparison_bigwig_files,chroms,full_pos_name)
    else:
        compare_dicts = comparison_dicts
    eval_objects_dict = {}

    for result_file_path in result_files_path:
        curr_result_eval = {}
        eval_objects_dict[result_file_path] = curr_result_eval
        new_result_file = create_result_file_mean_label(result_file_path, compare_dicts,ranges)
        eval_object = create_eval_object(new_result_file,label_a,label_b,labels)
        curr_result_eval["all_results"] = eval_object

        std_max = variability_file["std"].max()
        target = number_of_bins
        new_result_file["full_position"] = new_result_file["chrom"] + ":" + new_result_file["genomic_position"].astype(str) + "-"+ (new_result_file["genomic_position"] + 2).astype(str)
        for i in range(target):
            low = (std_max/target) * i
            high = (std_max/target) * (i+1)
            filtered = variability_file[(variability_file["std"] > low) & (variability_file["std"] <= high)]
            positions = set(filtered["full_position"])
            filtered_data = new_result_file[new_result_file["full_position"].isin(positions)]
            eval_object = create_eval_object(filtered_data,label_a,label_b,labels)
            curr_result_eval[f"{low}-{high}"] = eval_object
    return eval_objects_dict

def create_eval_object(new_result_file,label_a,label_b,labels):
    eval_object = {}
    true_labels = new_result_file["label"]
    for prediction_type in [label_a,label_b]:
        predicted_labels = new_result_file[prediction_type]
        eval_object[prediction_type + "_confusion_matrix"] = pd.crosstab(true_labels, predicted_labels).to_dict()
        eval_object[prediction_type] = {}
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
    return variability_free_apply_create_means(row,compare_dicts)


def apply_create_labels(row,ranges):
    return variability_free_apply_create_labels(row,ranges)


def create_result_file_mean_label(result_file_path, compare_dicts,ranges):
    return variability_free_create_result_file_mean_label(result_file_path, compare_dicts,ranges)


def create_result_file_mean_value(result_file_path, compare_dicts):
    return variability_free_create_result_file_mean_value(result_file_path, compare_dicts)

def create_window_id_dataset_dict(dataset,existing_keys = None):
    keys = []
    index = 0
    dataset_dict = {}
    for val in dataset:
        if keys is not None and val["window_id"] not in existing_keys:
            continue
        dataset_dict[val["window_id"]] = val
        keys.append(val["window_id"])
        index+=1
    return dataset_dict, keys

def evaluate_atlas(dataset,dataset_labels,variability_positions_dict,bins_to_use,atlas_datasets_to_use,verbose):
    dataset_dicts_to_use = []
    counter = 0
    for val in atlas_datasets_to_use:
        if verbose:
            print("creating atlas dict:",counter,"/",len(atlas_datasets_to_use),end="\r",flush=True)
        counter+=1
        val = load_from_disk(val)
        dic,temp_keys = create_window_id_dataset_dict(val,variability_positions_dict.keys())
        dataset_dicts_to_use.append(dic)
        del val
    if verbose:
        print("")
    res = []
    
    if verbose:
        print("length of dataset:", len(dataset),flush=True)
    for curr_bin in bins_to_use:
        if verbose:
            print("curr_bin:",curr_bin)
        labels_from_dataset = []
        labels_from_prediction = []
        for curr_dataset in range(len(dataset)):
            curr_window_id =dataset[curr_dataset]["window_id"] 
            if curr_window_id not in variability_positions_dict :
                continue
            if curr_bin not in variability_positions_dict[curr_window_id]:
                continue
            viable_positions = variability_positions_dict[curr_window_id][curr_bin]
            filtered_labels_from_dataset = dataset_labels[curr_window_id][curr_bin]
            other_vals = []
            for dic in dataset_dicts_to_use:
                if curr_window_id in dic:
                    other_vals.append(np.array(dic[curr_window_id]["labels"]))

            if len(other_vals) == 0:
                continue
            other_vals = np.array(other_vals)
            other_vals[other_vals == -100] = np.nan
            if np.isnan(other_vals).all():
                print("all nans!!",flush=True)
            predictions = list(np.nanmean(other_vals,axis=0))
            filtered_from_prediction = [ predictions[i+1] for i in viable_positions]
            labels_from_dataset.extend(filtered_labels_from_dataset)
            labels_from_prediction.extend(filtered_from_prediction)
        
        if len(labels_from_prediction) <= 1:
            continue

        # TODO: extract function
        res_r, res_mse, res_mae = get_cpg_reesults_for_labels(labels_from_dataset, labels_from_prediction,verbose)
        bin_results = [curr_bin,res_r,res_mse,res_mae]
        if verbose:
            print(bin_results,flush=True)
        res.append(bin_results)
    return res

def evaluate_checkpoint(model_repo: str, model_name: str,is_lora: bool,num_labels : int, dataset, 
        task_type,model_path : str,model_type,variability_positions_dict,use_variant_file : bool,
        vriant_grouping_method,bins_to_use,dataset_labels,verbose : bool, per_device_eval_batch_size: int = 1,
        eval_accumulation_steps: int = 4, label_transform="none") -> pd.DataFrame:
    if verbose:
        print("Evaluating model at path:", model_path)
    # model_type = cfg['model']['model_type']
    # TODO: add option to evaluate using cpg file
    # TODO: add option to evaluate using windows file
    base_model_name = model_repo + "/" + model_name
    if use_variant_file:
        # TODO: add non bin option
        prediction = predict_checkpoint(
            dataset,
            dataset,
            model_type,
            is_lora,
            num_labels,
            model_repo,
            model_name,
            model_path,
            # TODO: panic change, make sure it's ok 
            # return batch size to 64 and eval_steps to 10
            per_device_eval_batch_size=8,
            eval_accumulation_steps=2,
        )
        res = []
        if verbose:
            print("length of dataset:", len(dataset),flush=True)
        for curr_bin in bins_to_use:
            if verbose:
                print("curr_bin:",curr_bin)
            labels_from_dataset = []
            labels_from_prediction = []
            for curr_dataset in range(len(prediction.predictions)):
                # filtered_labels_from_dataset = [ dataset[curr_dataset]["labels"][i + 1] for i in viable_positions]   
                # labels_from_dataset.extend(filtered_labels_from_dataset)
                curr_window_id =dataset[curr_dataset]["window_id"] 
                if curr_window_id not in variability_positions_dict :
                    continue
                if curr_bin not in variability_positions_dict[curr_window_id]:
                    continue
                viable_positions = variability_positions_dict[curr_window_id][curr_bin]
                # viable_positions = [((x//6 * 6) - dataset[curr_dataset]["start"])//6 for x in starts_of_viable ]ions)
                # TODO: can move the creation of entire list of labels from dataset section out of function
                filtered_labels_from_dataset = dataset_labels[curr_window_id][curr_bin]
                # filtered_labels_from_dataset = [ dataset[curr_dataset]["labels"][i + 1] for i in viable_positions]   -- old code, myabe important
                labels_from_dataset.extend(filtered_labels_from_dataset)
                # add +1 to labels position because first token is for entire sequence, so all needs to be shifted
                filtered_from_prediction = maybe_decode_regression_predictions(
                    [prediction.predictions[curr_dataset][i + 1][0] for i in viable_positions],
                    label_transform,
                )
                labels_from_prediction.extend(filtered_from_prediction)
            
            if len(labels_from_prediction) <= 1:
                continue

            # TODO: extract function
            res_r, res_mse, res_mae = get_cpg_reesults_for_labels(labels_from_dataset, labels_from_prediction,verbose)
            bin_results = [curr_bin,res_r,res_mse,res_mae]
            if verbose:
                print(bin_results,flush=True)
            res.append(bin_results)
        return res
    model = get_fine_tuned_model(is_lora, num_labels, base_model_name, model_path, for_inference=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    trainer = get_trainer(
        dataset,
        model,
        tokenizer,
        model_type,
        per_device_train_batch_size=per_device_eval_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_accumulation_steps=eval_accumulation_steps,
    )
    eval_results = trainer.evaluate()

    del model
    del trainer
    del tokenizer
    return eval_results

def get_cpg_reesults_for_labels(labels_from_dataset, labels_from_prediction,verbose):

    labels_from_dataset = np.array(labels_from_dataset)
    labels_from_prediction = np.array(labels_from_prediction)
    mask = (labels_from_dataset!=-100)
    labels_from_dataset = labels_from_dataset[mask]
    labels_from_prediction = labels_from_prediction[mask]
    mask = ~(np.isnan(labels_from_dataset) | np.isnan(labels_from_prediction))
            # NOTE: make sure this works
    labels_from_prediction = labels_from_prediction[mask]
    labels_from_dataset = labels_from_dataset[mask]
    if verbose:
        print("legnth_labels:", len(labels_from_dataset),flush=True)
    res_r = pearsonr_eval.compute(predictions=labels_from_prediction,references=labels_from_dataset)
    res_mse = mse.compute(predictions=labels_from_prediction,references=labels_from_dataset)
    res_mae = mae.compute(predictions=labels_from_prediction,references=labels_from_dataset)
    return res_r,res_mse,res_mae


def get_flat_cpg_results_for_labels(labels_from_dataset, labels_from_prediction, verbose):
    res_r, res_mse, res_mae = get_cpg_reesults_for_labels(
        labels_from_dataset,
        labels_from_prediction,
        verbose,
    )
    return {
        "pearsonr": res_r["pearsonr"],
        "mse": res_mse["mse"],
        "mae": res_mae["mae"],
        "n_positions": len(labels_from_prediction),
    }


def create_empty_prediction_results_df():
    return pd.DataFrame(columns=PREDICTION_RESULTS_COLUMNS)


def create_empty_prediction_summary_df():
    return pd.DataFrame(columns=PREDICTION_SUMMARY_COLUMNS)


def parse_epoch_dir_name(epoch_dir_name):
    match = EPOCH_DIR_PATTERN.match(epoch_dir_name)
    if match is None:
        return None
    return int(match.group("epoch")), int(match.group("step"))


def get_pretraining_bucket(checkpoint):
    if checkpoint is None:
        return "no_pretraining"
    checkpoint_epoch_info = parse_epoch_dir_name(checkpoint)
    if checkpoint_epoch_info is None:
        return "pretrained"
    return f"epoch_{checkpoint_epoch_info[0]}_pretraining"


def normalize_saved_prediction_tissue_name(tissue_name):
    known_prefixes = [
        "kol_kora_high_only_",
    ]
    for prefix in known_prefixes:
        if tissue_name.startswith(prefix):
            return tissue_name[len(prefix):]
    return tissue_name


def parse_saved_prediction_run_name(run_name):
    match = RUN_NAME_PATTERN.match(run_name)
    if match is None:
        return None
    metadata = match.groupdict()
    normalized_tissue_name = normalize_saved_prediction_tissue_name(metadata["tissue_name"])
    pretraining_mode = "no_pretraining" if metadata["no_pretraining"] is not None else "pretrained"
    pretraining_bucket = get_pretraining_bucket(metadata["checkpoint"])
    family_key = (
        f"{pretraining_bucket}_retrain_{normalized_tissue_name}_{metadata['split_type']}"
        f"_lr_{metadata['learning_rate']}_bs_{metadata['batch_size']}"
        f"_seq_{metadata['seq_size']}_testsize_{metadata['test_size']}"
    )
    return {
        "held_out_sample": metadata["held_out_sample"],
        "pretraining_mode": pretraining_mode,
        "pretraining_bucket": pretraining_bucket,
        "tissue_name": normalized_tissue_name,
        "split_type": metadata["split_type"],
        "learning_rate": metadata["learning_rate"],
        "batch_size": metadata["batch_size"],
        "seq_size": metadata["seq_size"],
        "test_size": metadata["test_size"],
        "checkpoint": metadata["checkpoint"],
        "run_name": run_name,
        "family_key": family_key,
    }


def collect_saved_prediction_runs(prediction_root_dir, verbose):
    root = Path(prediction_root_dir)
    if not root.exists():
        raise ValueError(f"prediction_root_dir does not exist: {prediction_root_dir}")

    runs = {}
    total_candidates = 0
    for prediction_file in root.rglob("eval_predictions.csv.gitbackup"):
        if len(prediction_file.parents) < 2:
            if verbose:
                print(f"warning: skipping malformed prediction path: {prediction_file}", flush=True)
            continue
        epoch_info = parse_epoch_dir_name(prediction_file.parent.name)
        if epoch_info is None:
            if verbose:
                print(f"warning: skipping prediction file with unrecognized epoch dir: {prediction_file}", flush=True)
            continue
        outer_run_dir = prediction_file.parent.parent
        run_key = str(outer_run_dir)
        runs.setdefault(run_key, {"run_dir": outer_run_dir, "candidates": []})
        runs[run_key]["candidates"].append(
            {
                "prediction_path": prediction_file,
                "epoch_name": prediction_file.parent.name,
                "score": epoch_info,
            }
        )
        total_candidates += 1

    selected_runs = []
    for run in runs.values():
        run["candidates"] = sorted(run["candidates"], key=lambda item: item["score"])
        selected_runs.append(run)

    selected_runs = sorted(selected_runs, key=lambda item: item["run_dir"].name)
    if verbose:
        print(
            f"collected {total_candidates} saved prediction files across {len(selected_runs)} runs from {prediction_root_dir}",
            flush=True,
        )
    return selected_runs


def resolve_variability_file(variability_base_dir, run_metadata, verbose):
    variability_dir = Path(variability_base_dir) / f"_{run_metadata['tissue_name']}_{run_metadata['split_type']}"
    if not variability_dir.exists():
        if verbose:
            print(
                f"warning: variability directory does not exist for run {run_metadata['run_name']}: {variability_dir}",
                flush=True,
            )
        return None

    pattern = (
        f"{run_metadata['held_out_sample']}_per_varaint_variability_"
        f"{run_metadata['tissue_name']}_{run_metadata['split_type']}_seq_*_datasets.csv"
    )
    matches = sorted(variability_dir.glob(pattern))
    if len(matches) == 0:
        if verbose:
            print(
                f"warning: no variability file matched run {run_metadata['run_name']} with pattern {pattern}",
                flush=True,
            )
        return None
    if len(matches) == 1:
        return matches[0]

    preferred = [path for path in matches if "_seq_5400_" in path.name]
    if len(preferred) == 1:
        return preferred[0]

    if verbose:
        print(
            f"warning: ambiguous variability files for run {run_metadata['run_name']}: "
            f"{[str(path) for path in matches]}",
            flush=True,
        )
    return None


def load_prediction_dataframe(prediction_path):
    prediction_df = pd.read_csv(prediction_path)
    required_columns = {"window_id", "genomic_position", "label", "prediction"}
    missing_columns = required_columns.difference(prediction_df.columns)
    if missing_columns:
        raise ValueError(f"prediction file {prediction_path} is missing columns: {sorted(missing_columns)}")
    prediction_df = prediction_df.copy()
    prediction_df["chrom"] = prediction_df["window_id"].astype(str).str.split(":", n=1).str[0]
    prediction_df["genomic_position"] = pd.to_numeric(prediction_df["genomic_position"], errors="coerce")
    prediction_df["label"] = pd.to_numeric(prediction_df["label"], errors="coerce")
    prediction_df["prediction"] = pd.to_numeric(prediction_df["prediction"], errors="coerce")
    return prediction_df


def load_variability_dataframe(variability_path, number_of_bins):
    variability_df = pd.read_csv(variability_path).dropna(subset=["full_position", "window_id", "std"]).copy()
    if variability_df.empty:
        variability_df["std_bin"] = pd.Categorical([])
        variability_df["chrom"] = pd.Series(dtype="object")
        variability_df["variant_start"] = pd.Series(dtype="int64")
        return variability_df
    add_std_bins_to_dataframe(number_of_bins, variability_df)
    full_position_parts = variability_df["full_position"].astype(str).str.extract(
        r"^(?P<chrom>[^:]+):(?P<variant_start>\d+)-(?P<variant_end>\d+)$"
    )
    variability_df["chrom"] = full_position_parts["chrom"]
    variability_df["variant_start"] = pd.to_numeric(full_position_parts["variant_start"], errors="coerce")
    variability_df = variability_df.dropna(subset=["chrom", "variant_start", "std_bin"]).copy()
    variability_df["variant_start"] = variability_df["variant_start"].astype(int)
    return variability_df


def parse_bin_bounds(bin_label):
    left, right = str(bin_label).split("-", 1)
    return float(left), float(right)


def merge_prediction_with_variability(prediction_path, variability_df):
    prediction_df = load_prediction_dataframe(prediction_path)
    merged = prediction_df.merge(
        variability_df[["chrom", "variant_start", "std_bin"]],
        left_on=["chrom", "genomic_position"],
        right_on=["chrom", "variant_start"],
        how="inner",
    )
    merged = merged[merged["label"] != -100]
    merged = merged.dropna(subset=["label", "prediction", "std_bin", "genomic_position"])
    if merged.empty:
        merged["bin_label"] = pd.Series(dtype="object")
        merged["position_key"] = pd.Series(dtype="object")
        return merged
    merged = merged.copy()
    merged["genomic_position"] = merged["genomic_position"].astype(int)
    merged["bin_label"] = merged["std_bin"].astype(str)
    merged["position_key"] = merged["chrom"].astype(str) + ":" + merged["genomic_position"].astype(str)
    return merged


def restrict_candidates_to_common_positions(candidate_entries, run_name, verbose):
    if len(candidate_entries) == 0:
        return [], set()

    common_position_keys = None
    for candidate in candidate_entries:
        candidate_position_keys = set(candidate["merged"]["position_key"].unique())
        if common_position_keys is None:
            common_position_keys = candidate_position_keys
        else:
            common_position_keys = common_position_keys.intersection(candidate_position_keys)

    common_position_keys = set() if common_position_keys is None else common_position_keys
    if len(common_position_keys) == 0:
        if verbose:
            print(
                f"warning: no common prediction positions across candidate epochs for run {run_name}",
                flush=True,
            )
        return [], set()

    filtered_candidates = []
    for candidate in candidate_entries:
        filtered_merged = candidate["merged"][candidate["merged"]["position_key"].isin(common_position_keys)].copy()
        if filtered_merged.empty:
            continue
        updated_candidate = dict(candidate)
        updated_candidate["merged"] = filtered_merged
        filtered_candidates.append(updated_candidate)

    if verbose and filtered_candidates:
        print(
            f"using {len(common_position_keys)} common positions across {len(filtered_candidates)} candidate epochs for run {run_name}",
            flush=True,
        )
    return filtered_candidates, common_position_keys


def build_crossfit_assignment_df(base_merged_df, categories):
    if base_merged_df.empty:
        return pd.DataFrame(columns=["position_key", "fold_id"])

    positions_df = base_merged_df[["position_key", "chrom", "genomic_position", "bin_label"]].drop_duplicates().copy()
    assignment_frames = []
    for bin_label in categories:
        bin_positions = positions_df[positions_df["bin_label"] == str(bin_label)].copy()
        if bin_positions.empty:
            continue
        bin_positions = bin_positions.sort_values(["chrom", "genomic_position", "position_key"]).reset_index(drop=True)
        bin_positions["fold_id"] = np.arange(len(bin_positions)) % 2
        assignment_frames.append(bin_positions[["position_key", "fold_id"]])

    if len(assignment_frames) == 0:
        return pd.DataFrame(columns=["position_key", "fold_id"])
    return pd.concat(assignment_frames, ignore_index=True).drop_duplicates(subset=["position_key"])


def compute_bin_metric_rows(merged_df, categories, verbose):
    rows = []
    for bin_rank, bin_label in enumerate(categories, start=1):
        bin_rows = merged_df[merged_df["bin_label"] == str(bin_label)]
        if len(bin_rows) <= 1:
            continue
        metrics = get_flat_cpg_results_for_labels(
            bin_rows["label"].to_numpy(),
            bin_rows["prediction"].to_numpy(),
            verbose,
        )
        bin_lower, bin_upper = parse_bin_bounds(bin_label)
        rows.append(
            {
                "bin_rank": bin_rank,
                "bin_label": str(bin_label),
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "pearsonr": metrics["pearsonr"],
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "n_positions": metrics["n_positions"],
            }
        )
    return rows


def build_selection_settings(categories, selection_bin_rank):
    selection_metric = "pearsonr"
    if selection_bin_rank is None:
        return {
            "selection_metric": selection_metric,
            "selection_bin_rank": 0,
            "selection_bin_label": "overall",
        }

    try:
        selection_bin_rank = int(selection_bin_rank)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"task.selection_bin_rank must be an integer, got {selection_bin_rank!r}") from exc

    if selection_bin_rank < 1 or selection_bin_rank > len(categories):
        raise ValueError(
            f"task.selection_bin_rank={selection_bin_rank} is out of range for {len(categories)} bins"
        )

    return {
        "selection_metric": selection_metric,
        "selection_bin_rank": selection_bin_rank,
        "selection_bin_label": str(categories[selection_bin_rank - 1]),
    }


def compute_weighted_selection_score(bin_metric_rows):
    valid_rows = [
        row
        for row in bin_metric_rows
        if row["n_positions"] > 1 and pd.notna(row["pearsonr"]) and np.isfinite(row["pearsonr"])
    ]
    if len(valid_rows) == 0:
        return None
    weights = np.array([row["n_positions"] for row in valid_rows], dtype=float)
    scores = np.array([row["pearsonr"] for row in valid_rows], dtype=float)
    return float(np.average(scores, weights=weights))


def compute_selection_score(bin_metric_rows, selection_settings):
    if selection_settings["selection_bin_rank"] == 0:
        return compute_weighted_selection_score(bin_metric_rows)

    for row in bin_metric_rows:
        if row["bin_rank"] != selection_settings["selection_bin_rank"]:
            continue
        if row["n_positions"] <= 1 or pd.isna(row["pearsonr"]) or not np.isfinite(row["pearsonr"]):
            return None
        return float(row["pearsonr"])
    return None


def select_best_candidate_for_report_fold(candidate_entries, categories, report_fold, selection_settings, verbose):
    best_candidate = None
    for candidate in candidate_entries:
        cache_key = (
            f"selection_metrics_for_report_fold_{report_fold}_"
            f"{selection_settings['selection_bin_rank']}_{selection_settings['selection_metric']}"
        )
        if cache_key not in candidate:
            selection_df = candidate["merged"][candidate["merged"]["fold_id"] != report_fold]
            selection_metric_rows = compute_bin_metric_rows(selection_df, categories, verbose=False)
            selection_score = compute_selection_score(selection_metric_rows, selection_settings)
            candidate[cache_key] = {
                "metric_rows": selection_metric_rows,
                "selection_score": selection_score,
            }
        selection_score = candidate[cache_key]["selection_score"]
        if selection_score is None:
            continue
        if verbose:
            print(
                f"selection score for run {candidate['run_name']} candidate {candidate['epoch_name']} "
                f"on report fold {report_fold} using {selection_settings['selection_bin_label']}: {selection_score}",
                flush=True,
            )
        candidate_priority = (selection_score, candidate["score"][0], candidate["score"][1])
        if best_candidate is None or candidate_priority > best_candidate["priority"]:
            best_candidate = {
                "candidate": candidate,
                "selection_score": selection_score,
                "priority": candidate_priority,
            }
    return best_candidate


def evaluate_saved_prediction_run(run_entry, run_metadata, variability_df, selection_bin_rank, verbose):
    categories = [str(category) for category in variability_df["std_bin"].cat.categories]
    selection_settings = build_selection_settings(categories, selection_bin_rank)
    candidate_entries = []
    for candidate in run_entry["candidates"]:
        merged = merge_prediction_with_variability(candidate["prediction_path"], variability_df)
        if verbose:
            print(
                f"joined {len(merged)} prediction rows for run {run_metadata['run_name']} candidate {candidate['epoch_name']}",
                flush=True,
            )
        if merged.empty:
            continue
        candidate_entry = dict(candidate)
        candidate_entry["merged"] = merged
        candidate_entry["run_name"] = run_metadata["run_name"]
        candidate_entries.append(candidate_entry)

    if len(candidate_entries) == 0:
        if verbose:
            print(f"warning: no non-empty prediction candidates for run {run_metadata['run_name']}", flush=True)
        return []

    candidate_entries, _ = restrict_candidates_to_common_positions(candidate_entries, run_metadata["run_name"], verbose)
    if len(candidate_entries) == 0:
        return []

    assignment_df = build_crossfit_assignment_df(candidate_entries[0]["merged"], categories)
    if assignment_df.empty:
        if verbose:
            print(f"warning: could not build cross-fit assignments for run {run_metadata['run_name']}", flush=True)
        return []

    for candidate in candidate_entries:
        candidate["merged"] = candidate["merged"].merge(assignment_df, on="position_key", how="inner")

    selected_for_report_fold = {}
    for report_fold in (0, 1):
        best_candidate = select_best_candidate_for_report_fold(
            candidate_entries,
            categories,
            report_fold,
            selection_settings,
            verbose,
        )
        if best_candidate is None:
            if verbose:
                print(
                    f"warning: could not select a best candidate for run {run_metadata['run_name']} on report fold {report_fold}",
                    flush=True,
                )
            return []
        selected_for_report_fold[report_fold] = best_candidate

    report_frames = []
    for report_fold in (0, 1):
        candidate = selected_for_report_fold[report_fold]["candidate"]
        report_frames.append(candidate["merged"][candidate["merged"]["fold_id"] == report_fold].copy())

    combined_report_df = pd.concat(report_frames, ignore_index=True)
    report_metric_rows = compute_bin_metric_rows(combined_report_df, categories, verbose)
    if len(report_metric_rows) == 0:
        if verbose:
            print(f"warning: no report metrics were produced for run {run_metadata['run_name']}", flush=True)
        return []

    selected_epoch_fold_0 = selected_for_report_fold[0]["candidate"]["epoch_name"]
    selected_epoch_fold_1 = selected_for_report_fold[1]["candidate"]["epoch_name"]
    selected_epoch = f"fold_0={selected_epoch_fold_0};fold_1={selected_epoch_fold_1}"
    prediction_path = (
        f"fold_0={selected_for_report_fold[0]['candidate']['prediction_path']}"
        f"|fold_1={selected_for_report_fold[1]['candidate']['prediction_path']}"
    )

    rows = []
    for metric_row in report_metric_rows:
        rows.append(
            {
                "family_key": run_metadata["family_key"],
                "run_name": run_metadata["run_name"],
                "prediction_path": prediction_path,
                "selected_epoch": selected_epoch,
                "selected_epoch_fold_0": selected_epoch_fold_0,
                "selected_epoch_fold_1": selected_epoch_fold_1,
                "selection_score_fold_0": selected_for_report_fold[0]["selection_score"],
                "selection_score_fold_1": selected_for_report_fold[1]["selection_score"],
                "candidate_epoch_count": len(candidate_entries),
                "selection_strategy": CROSS_FIT_SELECTION_STRATEGY,
                "selection_metric": selection_settings["selection_metric"],
                "selection_bin_rank": selection_settings["selection_bin_rank"],
                "selection_bin_label": selection_settings["selection_bin_label"],
                "held_out_sample": run_metadata["held_out_sample"],
                "pretraining_mode": run_metadata["pretraining_mode"],
                "pretraining_bucket": run_metadata["pretraining_bucket"],
                "tissue_name": run_metadata["tissue_name"],
                "split_type": run_metadata["split_type"],
                "seq_size": run_metadata["seq_size"],
                "learning_rate": run_metadata["learning_rate"],
                "batch_size": run_metadata["batch_size"],
                "test_size": run_metadata["test_size"],
                "bin_rank": metric_row["bin_rank"],
                "bin_label": metric_row["bin_label"],
                "bin_lower": metric_row["bin_lower"],
                "bin_upper": metric_row["bin_upper"],
                "pearsonr": metric_row["pearsonr"],
                "mse": metric_row["mse"],
                "mae": metric_row["mae"],
                "n_positions": metric_row["n_positions"],
            }
        )
    return rows


def build_saved_prediction_summary_df(all_results_df):
    if all_results_df.empty:
        return create_empty_prediction_summary_df()

    group_columns = [
        "family_key",
        "selection_strategy",
        "selection_metric",
        "selection_bin_rank",
        "selection_bin_label",
        "pretraining_mode",
        "pretraining_bucket",
        "tissue_name",
        "split_type",
        "seq_size",
        "learning_rate",
        "batch_size",
        "test_size",
        "bin_rank",
        "bin_label",
        "bin_lower",
        "bin_upper",
    ]
    summary_df = (
        all_results_df.groupby(group_columns, as_index=False)
        .agg(
            run_count=("run_name", "nunique"),
            sample_count=("held_out_sample", "nunique"),
            pearsonr_mean=("pearsonr", "mean"),
            pearsonr_std=("pearsonr", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_positions_mean=("n_positions", "mean"),
        )
    )
    return summary_df[PREDICTION_SUMMARY_COLUMNS]


def evaluate_saved_predictions(cfg, verbose):
    paths = cfg["paths"]
    task = cfg["task"]
    prediction_root_dir = paths.get("prediction_root_dir")
    variability_base_dir = paths.get("variability_base_dir")
    number_of_bins = task.get("number_of_bins", -1)
    selection_bin_rank = task.get("selection_bin_rank")

    if prediction_root_dir is None:
        raise ValueError("missing paths.prediction_root_dir for evaluate_saved_predictions")
    if variability_base_dir is None:
        raise ValueError("missing paths.variability_base_dir for evaluate_saved_predictions")
    if number_of_bins <= 0:
        raise ValueError("task.number_of_bins must be positive for evaluate_saved_predictions")

    saved_prediction_runs = collect_saved_prediction_runs(prediction_root_dir, verbose)
    variability_cache = {}
    all_rows = []

    for run_entry in saved_prediction_runs:
        run_name = run_entry["run_dir"].name
        run_metadata = parse_saved_prediction_run_name(run_name)
        if run_metadata is None:
            if verbose:
                print(f"warning: skipping unrecognized run directory name: {run_name}", flush=True)
            continue

        variability_path = resolve_variability_file(variability_base_dir, run_metadata, verbose)
        if variability_path is None:
            continue

        variability_key = str(variability_path)
        if variability_key not in variability_cache:
            variability_cache[variability_key] = load_variability_dataframe(variability_path, number_of_bins)
        variability_df = variability_cache[variability_key]

        if variability_df.empty:
            if verbose:
                print(f"warning: variability file is empty after loading: {variability_path}", flush=True)
            continue

        rows = evaluate_saved_prediction_run(
            run_entry,
            run_metadata,
            variability_df,
            selection_bin_rank,
            verbose,
        )
        all_rows.extend(rows)

    if len(all_rows) == 0:
        all_results_df = create_empty_prediction_results_df()
    else:
        all_results_df = pd.DataFrame(all_rows)[PREDICTION_RESULTS_COLUMNS]
        all_results_df = all_results_df.sort_values(
            ["family_key", "held_out_sample", "run_name", "bin_rank"]
        ).reset_index(drop=True)

    summary_df = build_saved_prediction_summary_df(all_results_df)
    return {"all_results": all_results_df, "summary": summary_df}




def perform_evaluation(cfg: dict):
    verbose = cfg.get('verbose', False)
    paths = cfg['paths']
    task = cfg['task']
    model = cfg['model']
    if model.get("tissue_prompt") is not None:
        try:
            from .tissue_evaluator import perform_tissue_evaluation
        except ImportError:
            from tissue_evaluator import perform_tissue_evaluation
        return perform_tissue_evaluation(cfg)
    test_params = cfg["testing_params"]

    
    test_mode = test_params.get("test_mode",False)
    jump_sample = test_params.get("jump_sample",-1)
    per_device_eval_batch_size = test_params.get("per_device_eval_batch_size", 1)
    eval_accumulation_steps = test_params.get("eval_accumulation_steps", 4)
    # TODO: extract parameters from files here
    
    dataset_path = paths.get("dataset_path", None)
    variant_file_path = paths.get("variant_file_path",False)
    atlas_dataset_paths = paths.get("atlas_dataset_paths",None)
    target_bigwig_path = paths.get("target_bigwig_path", None)
    atlas_bigwig_paths = paths.get("atlas_bigwig_paths", None)


    sub_task = task['sub_task'] 
    use_variant_file= task.get("use_variant_file",False)
    vriant_grouping_method = task.get("vriant_grouping_method",None) 
    number_of_bins = task.get("number_of_bins", -1)
    top_rows = task.get("top_rows",-1)
    label_transform = get_task_label_transform(task) if sub_task not in ("atlas_evaluation", SAVED_PREDICTIONS_SUBTASK) else None
    
    use_lora = model.get("use_lora", model.get("is_lora", False))
    lora_over_finetuned = model.get("lora_over_finetuned",False)
    model_repo = model.get("model_repo",None)
    model_name = model.get("model_name",None)
    # TODO: add option for multiple files NOTE: why should I need multiple variant files?

    

    
    if verbose: 
        print("performing,", sub_task + " task")
        if sub_task == "atlas_evaluation":
            print("for target bigwig:", target_bigwig_path)
        elif sub_task == SAVED_PREDICTIONS_SUBTASK:
            print("for predictions root:", paths.get("prediction_root_dir"), flush=True)
        else:
            print("for dataset:", dataset_path)
        print("with model:", model_repo, model_name)
    
    if sub_task == "atlas_evaluation":
        if target_bigwig_path is None or atlas_bigwig_paths is None or len(atlas_bigwig_paths) == 0:
            raise ValueError(
                "atlas evaluation now expects paths.target_bigwig_path and paths.atlas_bigwig_paths. "
                "Regenerate the atlas evaluation configs."
            )
        return pd.DataFrame(
            evaluate_atlas_from_bigwigs(
                target_bigwig_path=target_bigwig_path,
                atlas_bigwig_paths=atlas_bigwig_paths,
                number_of_bins=number_of_bins,
                chroms=task.get("chromosomes"),
                top_rows=top_rows,
                test_mode=test_mode,
                jump_sample=jump_sample,
                verbose=verbose,
            )
        )

    if sub_task == SAVED_PREDICTIONS_SUBTASK:
        return evaluate_saved_predictions(cfg, verbose)

    if dataset_path is None:
        raise ValueError("missing dataset_path for non-atlas evaluation")

    dataset = load_from_disk(dataset_path)
    # dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    selected_dataset = dataset.select(range(top_rows)) if top_rows > 0 else dataset
    if test_mode:
        print("running test mode",flush=True)
        if jump_sample > 0:
            print("using jump sample of:",jump_sample,flush=True)
        selected_dataset = dataset.select(range(0,len(selected_dataset) ,jump_sample)) if jump_sample > 0 else selected_dataset 

    if verbose:
        print("finished loading dataset",flush=True)
        print("with size", len(selected_dataset))

    selected_dataset = apply_label_transform_to_dataset(
        selected_dataset,
        label_transform,
        verbose=verbose,
        dataset_name="evaluation dataset",
    )

    variability_positions_dict = {}
    bins_to_use = []
    dataset_labels = None
    if use_variant_file:
        variability_positions_dict = {}
        # TODO: add options other than bin
        variant_file_dataframe = pd.read_csv(variant_file_path).dropna()
        
        if verbose:
            print("loaded variant file",flush=True)
        add_std_bins_to_dataframe(number_of_bins, variant_file_dataframe)
        bins_to_use = variant_file_dataframe["std_bin"].unique()
        window_ids_in_variant_file = set(variant_file_dataframe["window_id"].unique())
        selected_dataset = selected_dataset.filter(lambda example: example["window_id"] in window_ids_in_variant_file)
        variability_positions_dict = create_variability_positions_dict(verbose, selected_dataset, bins_to_use, variant_file_dataframe)
        # set_of_non_empty_window_ids = create_non_empty_window_ids_set(variability_positions_dict)
        # selected_dataset = selected_dataset.filter(lambda example: example["window_id"] in set_of_non_empty_window_ids) # this is probably not needed
        dataset_labels = create_dataset_labels(selected_dataset, variability_positions_dict, bins_to_use,verbose)
        if verbose:
            print("finished filtering dataset",flush=True)
            print("with size", len(selected_dataset))
            
        # assert False
        if verbose:
            print("finished creating variability positions dict",flush=True)
    # TODO: add that if value is multiple paths, combine the deatasets    

    if sub_task == 'evaluate_single_checkpoint':
        res = evaluate_checkpoint( model_repo,model_name ,use_lora,
            cfg["model"]["num_labels"], selected_dataset,cfg['model']['model_type'],cfg['paths']['model_path'],
            cfg['model']['model_type'],variability_positions_dict,use_variant_file,vriant_grouping_method,bins_to_use,dataset_labels,
            verbose, per_device_eval_batch_size=per_device_eval_batch_size,
            eval_accumulation_steps=eval_accumulation_steps, label_transform=label_transform)
        return res
    elif sub_task == 'predict_single_checkpoint':
        model_type = cfg['model']['model_type']
        return predict_checkpoint(
            selected_dataset,
            selected_dataset,
            model_type,
            use_lora,
            cfg["model"]["num_labels"],
            model_repo,
            cfg["model"]["model_name"],
            cfg['paths']['model_path'],
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_accumulation_steps=eval_accumulation_steps,
        )
    elif sub_task == 'evaluate_multiple_checkpoints':
        results = []
        use_non_lora_first_iter = use_lora
        if lora_over_finetuned:
            use_non_lora_first_iter = False
        for model_path in cfg['paths']['model_paths']:
            
            # TODO delete this after find cause of bug
            
            res = evaluate_checkpoint(model_repo, model_name, 
                use_non_lora_first_iter, cfg["model"]["num_labels"], selected_dataset, cfg['model']['model_type'],
                  model_path,cfg['model']['model_type'],variability_positions_dict,use_variant_file,vriant_grouping_method,
                  bins_to_use,dataset_labels,verbose, per_device_eval_batch_size=per_device_eval_batch_size,
                  eval_accumulation_steps=eval_accumulation_steps, label_transform=label_transform)
            use_non_lora_first_iter = use_lora
            results.append(res)
        if use_variant_file:
            
            final_results = []
            for i in range(len(results)):
                model_results = {}
                keys = []
                for bin_value in results[i]:
                    bin_name = bin_value[0]
                    for val in bin_value[1:]:
                        key = list(val)[0]
                        value = val[key]
                        model_results[bin_name+"_"+key]= value
                        keys.append(bin_name+"_"+key)
                final_results.append(model_results)
            results = final_results
        df = pd.DataFrame(results)
        df["paths"] = cfg['paths']['model_paths']
        return df
    else:
        raise ValueError(f"Unknown sub_task: {sub_task}")

def create_non_empty_window_ids_set(variability_positions_dict):
    set_of_non_empty_window_ids = set()
    for window_id in variability_positions_dict:
        for bin in variability_positions_dict[window_id]:
            if len (variability_positions_dict[window_id][bin]):
                set_of_non_empty_window_ids.add(window_id)
    return set_of_non_empty_window_ids

def create_variability_positions_dict(verbose, selected_dataset, bins_to_use, variant_file_dataframe):
    by_window = {wid: g.copy() for wid, g in variant_file_dataframe.groupby("window_id", sort=False)}
    if verbose:
        print("created variants bin based variant dictionary",flush=True)
    variability_positions_dict = {}
    for index in range(len(selected_dataset)):
        if verbose:
            if index % 500 == 0:
                print("mapping viable variants",index,"/",len(selected_dataset),end="\r",flush=True)
        curr_window_id =selected_dataset[index]["window_id"]
        if curr_window_id not in variability_positions_dict:
            variability_positions_dict[curr_window_id] = {}
        if curr_window_id not in by_window:
           continue
        windiw_id_df = by_window[curr_window_id]
        for curr_bin in bins_to_use:
            limited_df = windiw_id_df[windiw_id_df["std_bin"] == curr_bin]
            starts_of_viable = set(limited_df["full_position"].str.split(":").str[1].str.split("-").str[0].astype(int))
            viable_positions = [((x//6 * 6) - selected_dataset[index]["start"])//6 for x in starts_of_viable ]                
            variability_positions_dict[curr_window_id][curr_bin] = list(sorted(viable_positions))
    
    if verbose:
        print()
    return variability_positions_dict

# TODO: maybe move to utils file
def add_std_bins_to_dataframe(number_of_bins, variant_file_dataframe):
    max_val = variant_file_dataframe["std"].max()
    edges = list(np.linspace(0,max_val,number_of_bins + 1))
        # labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    labels = []
    for i in range(len(edges) - 1):
        labels.append(f"{edges[i]}-{edges[i+1]}")
    variant_file_dataframe["std_bin"] = pd.cut(
            variant_file_dataframe["std"],                # your std-dev column
            bins=edges,
            labels=labels,
            right=True,               # (a, b]
            include_lowest=True  
                 # include 0 in first bin
            )

def create_dataset_labels(selected_dataset, variability_positions_dict, bins_to_use,verbose):
    dataset_labels = {}
    for dataset_index in range(len(selected_dataset)):
        if verbose:
            if dataset_index % 500 == 0:
                print("creating dataset labels",dataset_index,"/",len(selected_dataset),end="\r",flush=True)
        curr_window_id =selected_dataset[dataset_index]["window_id"]
        if curr_window_id not in dataset_labels:
            dataset_labels[curr_window_id] = {}
        for curr_bin in bins_to_use:
            if curr_bin not in variability_positions_dict[curr_window_id]:
                continue
            viable_positions = variability_positions_dict[curr_window_id][curr_bin]
                # add +1 to labels position because first token is for entire sequence, so all needs to be shifted
            filtered_labels_from_dataset = [ selected_dataset[dataset_index]["labels"][i + 1] for i in viable_positions]   
            dataset_labels[curr_window_id][curr_bin] = filtered_labels_from_dataset
    if verbose:
        print("\n")
    return dataset_labels

def predict_checkpoint(
    dataset,
    selected_dataset,
    model_type,
    is_lora,
    num_labels,
    model_repo,
    model_name,
    model_path,
    # TODO: fix eval batch size to be parameter, and be far larger
    per_device_eval_batch_size: int = 1024,
    eval_accumulation_steps: int = 128,
):
    base_model_name = model_repo + "/" + model_name
    
    model = get_fine_tuned_model(is_lora, num_labels, base_model_name, model_path, for_inference=True)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    trainer = get_trainer(
        dataset,
        model,
        tokenizer,
        model_type,
        per_device_train_batch_size=per_device_eval_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_accumulation_steps=eval_accumulation_steps,
    )
    return trainer.predict(selected_dataset)
