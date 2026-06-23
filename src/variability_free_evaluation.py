import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score

from utils.bigwig_utils import load_preprocessed_encode_cpg_dfs
from utils.formatting import combine_cpg_dfs

def evaluate_sample_predictions(result_files_path, 
                                chroms, 
                                comparison_bigiwg_files,
                                full_pos_name,
                                ranges, 
                                labels, 
                                comparison_types,
                                all_two,
                                verbous=True,
                                comparison_dicts=None): #TODO: change all_two to something more generic
    if comparison_dicts is None:
        compare_dicts = create_comparison_dicts(comparison_bigiwg_files,chroms,full_pos_name)
    else:
        compare_dicts = comparison_dicts
    eval_objects_dict = {}

    for result_file_path in result_files_path:
        if verbous:
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
    for prediciton_type in comparison_types:
        eval_object[prediciton_type] = {}
        eval_object[prediciton_type + "_confusion_matrix"] = pd.crosstab(new_result_file['label'], new_result_file[prediciton_type]).to_dict()
        for label in labels:
            eval_object[prediciton_type][label] = {}
            df = new_result_file.copy()
            # TODO: make this generic
            df["specific_label"] = df["label"] == label
            df["specific_type_label"] = df[prediciton_type] == label
            precision = precision_score(df['specific_label'], df['specific_type_label'])
            recall = recall_score(df['specific_label'], df['specific_type_label'])
            eval_object[prediciton_type][label]["Precision"] = precision
            eval_object[prediciton_type][label]["recall"] = recall
    
    df = new_result_file.copy()

    return eval_object

def create_comparison_dicts(comparison_bigiwg_files,chroms,full_pos_name):
    labels = []
    comparison_dfs = load_preprocessed_encode_cpg_dfs(comparison_bigiwg_files,chroms,full_pos_name,False)
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
            values.append(np.mean(temp_values))
    if len(values) == 0:
        return None
    return np.mean(values)


def apply_create_labels(row,ranges):
    mean_value = row["mean_value"]
    for i in range(len(ranges) - 1):
        if ranges[i] <= mean_value and mean_value <= ranges[i+1]:
            return i


def create_result_file_mean_label(result_file_path, compare_dicts,ranges):
    result_file = create_result_file_mean_value(result_file_path, compare_dicts)
    new_result_file = result_file.dropna().copy()
    new_result_file["mean_label"] =  new_result_file.apply(lambda x: apply_create_labels(x,ranges),axis=1)
    return new_result_file

def create_result_file_mean_value(result_file_path, compare_dicts):
    result_file = pd.read_csv(result_file_path)
    result_file["label"] = result_file["label"].astype(int)
    result_file["chrom"] = result_file["window_id"].str.split(":").str[0]
    result_file["mean_value"] = result_file.apply(lambda x: apply_create_means(x,compare_dicts),axis=1)
    return result_file