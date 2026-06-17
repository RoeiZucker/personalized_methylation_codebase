# src/data_extractor.py
import sys
import os, shutil
sys.path.insert(0, os.path.abspath("/home/users/roeizucker/tests/jupyter_notebooks/Tom_Hope_Project/refactored_code/src"))
sys.path.insert(0, os.path.abspath("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/src"))

import datasets
import numpy as np
# datasets.disable_caching()
import pandas as pd
from typing import Dict, Tuple
from Bio import SeqIO
from random import randint
from sklearn.model_selection import train_test_split
from utils.formatting import (
    combine_rows_to_multiple_instances_format,
    add_borders_to_multiple_instance_format_df,
    create_labels_formultiple_instance_format_df,
    add_sequence_formultiple_instance_format_df,
    combine_cpg_dfs,
    create_methyl_variability_df,
    create_length_window_id,
    group_methyl_variability_df,
    WINDOW_ID_COLUMN_NAME,
    MAX_DIFF_VARIABILITY_TYPE,  #TODO: these values might need to be defined in a constants file
    STD_VARIABILITY_TYPE,
    QUANTILE_SEPERATION_TYPE,
)
from utils.bigwig_utils import get_preprocessed_encode_cpg_dataframe, load_preprocessed_encode_cpg_dfs
from datasets import Dataset, DatasetDict, load_from_disk
from utils.dataset_utils import dataset_generator_wrapper


from utils.formatting import WINDOW_ID_COLUMN_NAME
from constants import (
    BLANK_LABEL_VALUE,
    SINGLE_INSTANCE_FORMAT_VALUE_NAME,
    HG38_ENCODING,
    HG38_PATH,
    RAW_INPUT_NAME,
    PREPROCESSED_INPUT_NAME,
    INTERMEDIATE_INPUT_NAME,
    DEFAULT_FULL_POSITION_COLUMN_NAME,
    CPG_EXTRACTION_TASK_NAME,
    CPG_SEPERATING_SITES_TASK_NAME,
    CPG_TOKEN_CLASSIFICATION_EXTRACTION_TASK_NAME,
    STANDART_NO_OVERLAP_WINDOW_TYPE,
    WINDOW_NAME_TRAIN_TEST_FILTRATION,
    RANDOM_SAMPLE_TRAIN_TEST_FILTRATION,
    ENCODE_VALUE_NAME,
    INSTADEEP_KMER_SIZE,
    DATASET_BATCH_SIZE,
    KMER_SAMPLE_TRAIN_TEST_FILTRATION
)



def save_cpg_window_dfs(train_df, test_df,train_path,test_path):
    train_df["bin_start"] = train_df["bin_start"].astype(int)
    train_df["bin_end"] = train_df["bin_end"].astype(int)
    test_df["bin_start"] = test_df["bin_start"].astype(int)
    test_df["bin_end"] = test_df["bin_end"].astype(int)
    train_df.to_csv(train_path,index=False)
    test_df.to_csv(test_path,index=False)
    # TODO: train path should not be returned here
    return train_path




def combine_seperated_dfs_to_train_test(test_size, random_state, group_a_df, group_b_df):
    train_df_group_a, test_df_group_a = train_test_split(
            group_a_df,
            test_size   = test_size,
            random_state = random_state,
        )

    train_df_group_b, test_df_group_b = train_test_split(
            group_b_df,
            test_size   = test_size,
            random_state = random_state,
        )
    train_df = pd.concat([train_df_group_a,train_df_group_b])
    test_df = pd.concat([test_df_group_a,test_df_group_b])
    return train_df,test_df

def create_cpg_window_files(raw_files_paths, chroms, full_position_column_name, test_size, random_state, train_path, test_path,
                            variability_type,variant_seperation_type,variant_seperation_threshold,window_seperation_type,
                            window_seperation_threshold, output_per_variant_data, output_window_data,per_variant_data_path,seq_size,verbose):
    if output_per_variant_data and per_variant_data_path is None:
        raise ValueError("no variant data path provided")
    if not output_window_data and not output_per_variant_data:
        raise ValueError("No task provided")
    dfs = load_preprocessed_encode_cpg_dfs(raw_files_paths, chroms, full_position_column_name,verbose)
    labels = []  # same length/order as `kept`
    for i in range(len(dfs)):
        labels.append(f"ind_{i}")

    df = combine_cpg_dfs(full_position_column_name, dfs, labels)
    df = create_methyl_variability_df(len(dfs), df,variability_type,variant_seperation_type,variant_seperation_threshold,seq_size)
    if output_per_variant_data:
        if not os.path.exists(os.path.dirname(per_variant_data_path)):
            os.mkdir(os.path.dirname(per_variant_data_path))
        df[[DEFAULT_FULL_POSITION_COLUMN_NAME,WINDOW_ID_COLUMN_NAME,variability_type,"high_diff"]].to_csv(per_variant_data_path)
        if verbose:
            print("saved cpg variability file to",per_variant_data_path,flush=True)
    
    if not output_window_data:
        return
    grouped_df = group_methyl_variability_df(df)

    if window_seperation_type == QUANTILE_SEPERATION_TYPE:
        high_diff_quantile_value = grouped_df["high_diff"].quantile(window_seperation_threshold)
        if verbose:
            print("high_diff_quantile_value",high_diff_quantile_value,flush=True)
        curr_df = grouped_df.reset_index()
        high_diff_df = curr_df[curr_df["high_diff"] > high_diff_quantile_value]
        low_diff_df = curr_df[curr_df["high_diff"] <= high_diff_quantile_value]
    else:
        raise NotImplementedError()
    train_df, test_df = combine_seperated_dfs_to_train_test(test_size, random_state, high_diff_df, low_diff_df)
    # TODO: train path should not be returned here
    train_path = save_cpg_window_dfs(train_df, test_df,train_path,test_path)
    if verbose:
        print("saved window files to",train_path,test_path,flush=True)


def create_grouped_methyl_dfs(raw_files_paths, chroms, full_position_column_name,variability_type,variant_seperation_type,
                              variant_seperation_threshold,seq_size,verbose):
    dfs = load_preprocessed_encode_cpg_dfs(raw_files_paths, chroms, full_position_column_name,verbose)
    labels = []  # same length/order as `kept`
    for i in range(len(dfs)):
        labels.append(f"ind_{i}")

    df = combine_cpg_dfs(full_position_column_name, dfs, labels)
    df = create_methyl_variability_df(len(dfs), df,variability_type,variant_seperation_type,variant_seperation_threshold,seq_size)
    grouped_df = group_methyl_variability_df(df)
    return grouped_df


def load_fasta(fasta_path: str) -> Dict[str, str]:
    """
    Load all sequences from a FASTA file into a dict.

    Args:
        fasta_path: Path to a FASTA file.

    Returns:
        Dict mapping sequence IDs to sequence strings.
    """
    sequences: Dict[str, str] = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def get_correct_chrom_dict(cfg, task):
    chrom_dict = {}
    if task.get("use_fasta", True):
        asm        = task["assembly"]
        fasta_path = cfg["paths"]["assemblies"][asm]
        chrom_dict = load_fasta(fasta_path)
    return chrom_dict



def load_chrom_dict(cfg: dict, task: dict) -> Dict[str, str]:
    """
    Return a chromosome→sequence dict if task['use_fasta'] is True;
    otherwise return an empty dict.
    """
    if not task.get("use_fasta", True):
        return {}
    asm = task["assembly"]
    path = cfg["paths"]["assemblies"][asm]
    return load_fasta(path)


def group_regions(df):
    grouped = df.groupby(WINDOW_ID_COLUMN_NAME).agg({'start': list, 'end': list, 'methyl_rate': list,'chrom':lambda x: list(x[:1])[0],
                                        'bin_start':lambda x: list(x[:1])[0],'bin_end':lambda x: list(x[:1])[0]})
    val = grouped.reset_index().rename(columns={'start': 'starts', 'end': 'ends','methyl_rate':'values','bin_start':'start','bin_end':'end'})
    return val


def create_intermediate_dataframess_from_preprocessed_dataframe_depricated(chrom_dict, preprocessed_df, test_size, random_state,
                                                                 shuffle, seq_size, test_mode, blank_label):
    inst = combine_rows_to_multiple_instances_format(preprocessed_df, seq_size)
    add_borders_to_multiple_instance_format_df(inst, seq_size,test_mode)
    create_labels_formultiple_instance_format_df(inst, blank_label)
    inst = add_sequence_formultiple_instance_format_df(inst, chrom_dict)

    train_df, test_df = train_test_split(
        inst,
        test_size   = test_size,
        random_state = random_state,
        shuffle      = shuffle
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def extract_data(cfg: dict) -> None:
    """
    Extract data based on the configuration provided in cfg.
    This function handles different tasks such as Encode CpG extraction, creates a hugginfface dataset and saves it to local disk.
    Args:
        cfg: Configuration dictionary containing task details and paths.
    """
    task = cfg["task"]
    paths = cfg["paths"]
    if cfg.get("verbose",True):
        print("started program!!",flush=True)
    # TODO: should this be here?
    if task.get("clear_generator_cache", False):
        shutil.rmtree(paths["generetor_cache_dir"], ignore_errors=True)
    if task["type"] == CPG_EXTRACTION_TASK_NAME:
        # NOTE: This was changed so it only recieves cfg, need to test
        return encode_cpg_extraction(cfg)
    elif task["type"] == CPG_SEPERATING_SITES_TASK_NAME:
        return encode_cpg_window_files_extraction(cfg)
    elif task["type"] == CPG_TOKEN_CLASSIFICATION_EXTRACTION_TASK_NAME:
        try:
            from .token_classification_data_extractor import encode_cpg_token_classification_extraction
        except ImportError:
            from token_classification_data_extractor import encode_cpg_token_classification_extraction
        return encode_cpg_token_classification_extraction(cfg)
    else:
        raise ValueError(f"Unknown task type: {task['type']}")
    

def encode_cpg_window_files_extraction(cfg):
    # TODO: add starts to results DF, will make evaluation much faster!
    paths = cfg["paths"]
    task = cfg["task"]
    raw_files_paths = paths["raw_data_paths"]
    chroms = task.get("chromosomes", None)
    full_position_column_name = DEFAULT_FULL_POSITION_COLUMN_NAME
    test_size = task.get("test_size", 0.2)
    random_state = cfg.get("random_state", 42)
    train_path  = paths.get("train_path",None)
    test_path  = paths.get("test_path",None)
    seq_size = task["seq_size"]
    verbose = cfg.get("verbose",True)
    variability_type = task['variability_type']
    variant_seperation_type = task['variant_seperation_type']
    variant_seperation_threshold = task['variant_seperation_threshold']
    window_seperation_type = task.get('window_seperation_type',None)
    window_seperation_threshold = task.get('window_seperation_threshold',None)
    output_per_variant_data = task.get('output_per_variant_data',None)
    output_window_data = task.get('output_window_data',None)
    
    per_variant_data_path = paths.get("per_variant_data_path",None)
    
    if not (output_per_variant_data or output_window_data):
        raise ValueError("no output requested")
    return create_cpg_window_files(raw_files_paths, chroms, full_position_column_name, test_size, random_state, train_path, test_path,
                            variability_type,variant_seperation_type,variant_seperation_threshold,window_seperation_type,
                            window_seperation_threshold, output_per_variant_data, output_window_data,per_variant_data_path,seq_size,verbose)

def encode_cpg_extraction(cfg):
    task = cfg["task"]
    paths = cfg["paths"]
    variant_filtering = cfg.get("variant_filtering",dict())
    testing_params = cfg["testing_params"]

    train_test_seperation_type = task.get("train_test_seperation",None)
    windowing_type = task.get("window_type",None)
    chrom_dict = load_chrom_dict(cfg, task) #TODO: move to interal function, change so it does not recieve cfg or task
    test_size = task.get("test_size", 0.2)
    test_size = task.get("test_size", 0.2)
    random_state = cfg.get("random_state", 42)
    shuffle = task.get("shuffle", True)
    seq_size = task["seq_size"]
    test_mode = cfg["testing_params"]["test_mode"]
    blank_label = task["blank_label"]
    input_mode = task["input_mode"]

    kmer_size = task.get("kmer_size",INSTADEEP_KMER_SIZE)

    verbose = cfg["verbose"]
    raw_data_path = paths["raw_data_path"]
    chromosomes = task.get("chromosomes", None)
    pre_processed_data_path = paths.get("pre_processed_data_path",None)
    output_preprocessed_data = task.get("output_preprocessed_data",False)
    train_window_names_path = paths.get("train_window_names_path",None)
    test_window_names_path = paths.get("test_window_names_path",None)
    preprocessed_df = None
    output_intermediate_data = task["output_intermediate_data"]
    output_hf_dataset = task["output_hf_dataset"]
    variant_file_path =  paths.get("variant_file_path",None)
    override_dataset = task.get("override_dataset",False)

    use_variant_filtering = variant_filtering.get("use_variant_filtering", False)
    variant_filtering_upper_bound = variant_filtering.get("variant_filtering_upper_bound", -1)
    variant_filtering_lower_bound = variant_filtering.get("variant_filtering_lower_bound", -1)
    if os.path.exists(paths["hf_dataset_train_path"]) and not override_dataset:
        raise ValueError("HF dataset exists and override is False or not supplied" + paths["hf_dataset_train_path"])

    # TODO: add that if window name filtration but no window namfe failes an error is raised
    # TODO: change to parameter
    remove_under_0 = task.get("replace_min1",True)
    test_mode = testing_params["test_mode"]
    raw_data_top_rows = testing_params.get("raw_data_top_rows",-1)

    # TODO: extract to function
    variability_dict = None
    if use_variant_filtering:
        df = pd.read_csv(variant_file_path)
        variability_dict = df[[DEFAULT_FULL_POSITION_COLUMN_NAME,"std"]].set_index("full_position").to_dict(orient="index")
        for key in variability_dict.keys():
            variability_dict[key] = variability_dict[key]["std"]
        if verbose:
            print("finished loading variability dict",flush=True)

    # TODO: make sure this works:
    # train_only = (test_size == 0) and (train_test_seperation_type == RANDOM_SAMPLE_TRAIN_TEST_FILTRATION)
    train_only = (test_size == 0)


    if input_mode == RAW_INPUT_NAME or input_mode == PREPROCESSED_INPUT_NAME:
        preprocessed_df = load_preprocessed_data(input_mode, verbose, raw_data_path, chromosomes, pre_processed_data_path,remove_under_0)
        # TODO: if both input and outpus is preprocessed files, something is wrong and need to raise an error
        if test_mode and raw_data_top_rows > -1:
            preprocessed_df = preprocessed_df.head(raw_data_top_rows)
        if output_preprocessed_data:
            save_preprocessed_data(verbose, pre_processed_data_path, preprocessed_df)
    if not output_intermediate_data and not output_hf_dataset:
            # if no need to output intermidiate or HF dataset, can exit
            return 
    if input_mode == INTERMEDIATE_INPUT_NAME:
        if output_intermediate_data:
            raise ValueError("tried to recreate exactly the same input, probably something went wrong")
        if verbose:
            print(f"Loading intermediate data from already created {paths['intermediate_train_data_path']} and {paths['intermediate_test_data_path']}.",flush=True)
        
    if output_intermediate_data: 
        create_intermediate_data_files(task, paths, train_test_seperation_type, windowing_type, chrom_dict, 
        test_size, random_state, shuffle, seq_size, test_mode, blank_label, verbose, train_window_names_path,
        test_window_names_path, preprocessed_df,variability_dict,variant_filtering_upper_bound,variant_filtering_lower_bound)
    if output_hf_dataset:
        if verbose:
            print("Creating Hugging Face dataset from intermediate data.",flush=True)
        from transformers import AutoTokenizer
        # TODO: chnage to model instead of task
        tokenizer = AutoTokenizer.from_pretrained(task["tokenizer_name"])
        if task.get("tissue_id", None) is not None:
            try:
                from .tissue_data_extractor import create_tissue_huggingface_datasets
            except ImportError:
                from tissue_data_extractor import create_tissue_huggingface_datasets
            return create_tissue_huggingface_datasets(
                intermediate_train_data_path=paths["intermediate_train_data_path"],
                intermediate_test_data_path=paths["intermediate_test_data_path"],
                hf_dataset_train_path=paths["hf_dataset_train_path"],
                hf_dataset_test_path=paths["hf_dataset_test_path"],
                tokenizer=tokenizer,
                train_test_seperation_type=train_test_seperation_type,
                blank_label=blank_label,
                test_size=test_size,
                random_state=random_state,
                kmer_size=kmer_size,
                seq_size=seq_size,
                verbose=verbose,
                train_only=train_only,
                tissue_id=task["tissue_id"],
                map_seperate_window_labels_wrapper=map_seperate_window_labels_wrapper,
            )
        if train_test_seperation_type == KMER_SAMPLE_TRAIN_TEST_FILTRATION:
            dataset = Dataset.from_generator(dataset_generator_wrapper(paths["intermediate_train_data_path"], DATASET_BATCH_SIZE,kmer_size,seq_size))
            new_dataset = dataset.map(map_seperate_window_labels_wrapper(blank_label,1 - test_size,random_state))
            train_ds = new_dataset.remove_columns(["test_labels","labels"]).rename_column("train_labels","labels")
            test_ds = new_dataset.remove_columns(["train_labels","labels"]).rename_column("test_labels","labels")

            # TODO: remove those where all labels are blank
            train_ds = train_ds.filter(lambda example: set(example["labels"]) != {blank_label})
            if test_size > 0:
                test_ds = test_ds.filter(lambda example: set(example["labels"]) != {blank_label})   
            encoded_train = train_ds.map(lambda examples: tokenizer(examples['seq'])).remove_columns(['seq'])
            encoded_test = test_ds.map(lambda examples: tokenizer(examples['seq'])).remove_columns(['seq'])

            # TODO: print paths
            if verbose:
                print("saving train dataset to:",paths["hf_dataset_train_path"],flush=True)
                print("saving test dataset to:",paths["hf_dataset_test_path"],flush=True)
            encoded_train.save_to_disk(paths["hf_dataset_train_path"], num_proc=1)
            if test_size > 0:
                encoded_test.save_to_disk(paths["hf_dataset_test_path"], num_proc=1)
            return
            
        save_huggingface_dataset(paths["intermediate_train_data_path"], paths["hf_dataset_train_path"], tokenizer,kmer_size,seq_size,verbose)
        if not train_only:

            save_huggingface_dataset(paths["intermediate_test_data_path"], paths["hf_dataset_test_path"], tokenizer,kmer_size,seq_size,verbose)

def map_seperate_window_labels_wrapper(empty_label, p,random_seed):
    def apply_seperate_window_labels(row):
        labels = np.array(row["labels"])
        labels_train, labels_test = create_kmer_split_train_test_labels(labels,empty_label,p,random_seed)
        return {"train_labels": labels_train, "test_labels": labels_test}
    return apply_seperate_window_labels


def create_intermediate_data_files( task, paths, train_test_seperation_type, windowing_type, chrom_dict, test_size,
                                    random_state, shuffle, seq_size, test_mode, blank_label, verbose,
                                    train_window_names_path, test_window_names_path, preprocessed_df,variability_dict,
                                    variant_filtering_upper_bound,variant_filtering_lower_bound):
    
    if verbose:
        print("Creating intermediate dataframes from pre-processed data.",flush=True)
    if preprocessed_df is None:
        raise ValueError("To output intermediate data either preprocessed data or raw data should be supplied")
        
        # TODO: should this be here or in a previous section?
    preprocessed_df["value"] = preprocessed_df[task["value_column"]]
        # NOTE: this was depreecated, need to change the name to train_test_seperation, make sure no errors
        # seperation_critiria = task.get("seperation_critiria",None)

    train_df, test_df = create_intermidiate_datasets_from_preprocessed_data(train_test_seperation_type, windowing_type,
                        chrom_dict, test_size, random_state, shuffle,seq_size, test_mode, blank_label, train_window_names_path,
                        test_window_names_path, preprocessed_df,variability_dict,variant_filtering_upper_bound,variant_filtering_lower_bound)
    if verbose:
        print(f"Saving intermediate data to {paths['intermediate_train_data_path']} and {paths['intermediate_test_data_path']}.")
    train_df.to_csv(paths["intermediate_train_data_path"], index=False)
    test_df.to_csv(paths["intermediate_test_data_path"], index=False)

def create_intermidiate_datasets_from_preprocessed_data(train_test_seperation_type, windowing_type, chrom_dict, 
                                                        test_size, random_state, shuffle, seq_size, test_mode, blank_label, 
                                                        train_window_names_path, test_window_names_path, preprocessed_df,
                                                        variability_dict,variant_filtering_upper_bound,variant_filtering_lower_bound):
    # TODO: add train/test seperation method (even if allways test_size)
    if windowing_type is None and train_test_seperation_type is None:
            # TODO: this is legacy, I should allways require windowing and train_test_seperation
        train_df, test_df = create_intermediate_dataframess_from_preprocessed_dataframe_depricated(chrom_dict, preprocessed_df,test_size, 
                                                                                                   random_state, shuffle, seq_size, test_mode,
                                                                                                     blank_label)
    elif windowing_type  is None or train_test_seperation_type is None:
        raise ValueError("need windowing type and filtration type")
    elif windowing_type == STANDART_NO_OVERLAP_WINDOW_TYPE:

        #TODO: in this function, need to add the full position, later it should be aggragated as list
        create_length_window_id(preprocessed_df,seq_size)
        grouped = preprocessed_df.groupby(WINDOW_ID_COLUMN_NAME).agg({'start': list, 'end': list, 'methyl_rate': list,'chrom':lambda x: list(x[:1])[0],
                                        'bin_start':lambda x: list(x[:1])[0],'bin_end':lambda x: list(x[:1])[0]})
        
        
    if train_test_seperation_type == WINDOW_NAME_TRAIN_TEST_FILTRATION:
        train_df, test_df = seperate_train_test_using_window_name(chrom_dict, blank_label, train_window_names_path, test_window_names_path, grouped)
    elif train_test_seperation_type == RANDOM_SAMPLE_TRAIN_TEST_FILTRATION:
        train_df, test_df = random_sample_train_test_filtration_seperation(chrom_dict, test_size, random_state, blank_label, variability_dict, variant_filtering_upper_bound, variant_filtering_lower_bound, grouped)
        # train test split
    elif train_test_seperation_type == KMER_SAMPLE_TRAIN_TEST_FILTRATION:
        df = grouped.copy()
        df = df.reset_index().rename(columns={'start': 'starts', 'end': 'ends','methyl_rate':'values'}).rename(columns={'bin_start':'start','bin_end':'end'})
        train_df, test_df = convert_seperated_dfs_to_intermidiate_format(chrom_dict, 0, blank_label, variability_dict,
                                                     variant_filtering_upper_bound, variant_filtering_lower_bound, df, None)
        # df[["train_labels","test_labels"]] = df.apply(apply_seperate_window_labels_wrapper(blank_label,test_size,random_state,"labels"),axis=1,result_type="expand")
        # # df.to_csv("/sci/archive/michall/roeizucker/huggingface_datasets_dir/temp_kmer_sample_filtration.csv")
        # # print(df[df["train_labels"] != df["test_labels"]])
        # df = df.rename(columns={"labels":"temp_labels"})
        # # print(df["labels"])
        # cols = list(df.columns)
        # train_df = df[cols + ["train_labels"]].rename(columns={'train_labels':'labels'})
        # test_df = df[cols + ["test_labels"]].rename(columns={'test_labels':'labels'})
        # print(test_df["labels"])


        

        # cols = list(df.columns)
        # df[["train_labels","test_labels"]] = df.apply(apply_seperate_window_labels_wrapper(blank_label,test_size,random_state),axis=1,result_type="expand")
        # train_df = df[cols + ["train_labels"]]
        # test_df = df[cols + ["test_labels"]]

        # train_df = train_df.reset_index().rename(columns={'start': 'starts', 'end': 'ends','train_labels':'values'}).rename(columns={'bin_start':'start','bin_end':'end'})
        # if test_size > 0:
        #     test_df = test_df.reset_index().rename(columns={'start': 'starts', 'end': 'ends','test_labels':'values'}).rename(columns={'bin_start':'start','bin_end':'end'})
        # train_df, test_df = convert_seperated_dfs_to_intermidiate_format(chrom_dict, test_size, blank_label, variability_dict,
        #                                           variant_filtering_upper_bound, variant_filtering_lower_bound, train_df, test_df)
    else:
        raise NotImplementedError(f"{train_test_seperation_type} was not implemented")
    return train_df,test_df

# TODO: move to formatting?
def apply_seperate_window_labels_wrapper(empty_label, p,random_seed,col_name):
    def apply_seperate_window_labels(row):
        labels = np.array(row[col_name])
        return create_kmer_split_train_test_labels(labels,empty_label,p,random_seed)
        # return 2,3
    return apply_seperate_window_labels


# TODO: move to formatting 
def create_kmer_split_train_test_labels(lable_array, empty_label, p,random_seed):
    mask = lable_array!=empty_label
    a = mask.copy()
    b = mask.copy()
    idx = np.flatnonzero(mask) 
    rng = np.random.default_rng(random_seed)
    rng.shuffle(idx)
    cut = int(p * idx.size)
    idx_a_false = idx[:cut]
    idx_b_false = idx[cut:]
    a.ravel()[idx_a_false] = False
    b.ravel()[idx_b_false] = False

    arr_train = lable_array.copy()
    arr_test = lable_array.copy()
    arr_train[a] = empty_label
    arr_test[b] = empty_label
    return list(arr_train),list(arr_test)

def random_sample_train_test_filtration_seperation(chrom_dict, test_size, random_state, blank_label, variability_dict, variant_filtering_upper_bound, variant_filtering_lower_bound, grouped):
    if test_size > 0:
        train_df, test_df = train_test_split(
                grouped,
                test_size   = test_size,
                random_state = random_state,
            )
    else:
        train_df = grouped
        test_df = None
        
    train_df = train_df.reset_index().rename(columns={'start': 'starts', 'end': 'ends','methyl_rate':'values'}).rename(columns={'bin_start':'start','bin_end':'end'})
    if test_size > 0:
        test_df = test_df.reset_index().rename(columns={'start': 'starts', 'end': 'ends','methyl_rate':'values'}).rename(columns={'bin_start':'start','bin_end':'end'})

    train_df, test_df = convert_seperated_dfs_to_intermidiate_format(chrom_dict, test_size, blank_label, variability_dict, 
                                                                     variant_filtering_upper_bound, variant_filtering_lower_bound, 
                                                                     train_df, test_df)
    return train_df,test_df

def convert_seperated_dfs_to_intermidiate_format(chrom_dict, test_size, blank_label, variability_dict,
                                                  variant_filtering_upper_bound, variant_filtering_lower_bound, train_df, test_df):
    create_labels_formultiple_instance_format_df(train_df, blank_label,variability_dict,
        variant_filtering_upper_bound,variant_filtering_lower_bound)
    train_df = train_df[train_df["labels"].apply(set).apply(len) > 1]
    if test_size > 0:
            # NOTE: no need for filtering variability for test set, so only None is sent
        create_labels_formultiple_instance_format_df(test_df, blank_label,None,-1, -1)
        test_df = test_df[test_df["labels"].apply(set).apply(len) > 1]
    train_df = add_sequence_formultiple_instance_format_df(train_df, chrom_dict)
    if test_size > 0:
        test_df = add_sequence_formultiple_instance_format_df(test_df, chrom_dict)
        # TODO: fix this crappy code
    if test_size == 0:
        test_df = train_df.head(0)
    return train_df,test_df

def seperate_train_test_using_window_name(chrom_dict, blank_label, train_window_names_path, test_window_names_path, grouped):
    train_windows_df = (pd.read_csv(train_window_names_path))
    test_windows_df = (pd.read_csv(test_window_names_path))
    train_window_ids = set(train_windows_df[WINDOW_ID_COLUMN_NAME].tolist())
    test_window_ids = set(test_windows_df[WINDOW_ID_COLUMN_NAME].tolist())
    # TODO: column names should be values, this needs to be more obvious in the code
    train_df = grouped[grouped.index.isin(train_window_ids)].reset_index().rename(columns={'start': 'starts', 'end': 'ends','methyl_rate':'values'}).rename(columns={'bin_start':'start','bin_end':'end'})
    test_df = grouped[grouped.index.isin(test_window_ids)].reset_index().rename(columns={'start': 'starts', 'end': 'ends','methyl_rate':'values'}).rename(columns={'bin_start':'start','bin_end':'end'})
    # NOTE : should I add variability filtering here?
    create_labels_formultiple_instance_format_df(train_df, blank_label)
    create_labels_formultiple_instance_format_df(test_df, blank_label)
    train_df = add_sequence_formultiple_instance_format_df(train_df, chrom_dict)
    test_df = add_sequence_formultiple_instance_format_df(test_df, chrom_dict)
    return train_df,test_df

def save_preprocessed_data(verbose, pre_processed_data_path, preprocessed_df):
    preprocessed_df.to_csv(pre_processed_data_path, index=False)
    if verbose:
        print(f"Saved pre-processed data to {pre_processed_data_path}.")

def load_preprocessed_data(input_mode, verbose, raw_data_path, chromosomes, pre_processed_data_path,remove_under_0):
    if input_mode == RAW_INPUT_NAME:
        if verbose:
            print("started extracting CpG data", flush=True)
            # if task.get("differentiating_regions",None) is None: # Not sure why this is here
        preprocessed_df = get_preprocessed_encode_cpg_dataframe(raw_data_path,chromosomes,remove_under_0=remove_under_0,verbose=verbose)
        if verbose:
            print(f"Extracted {len(preprocessed_df)} CpG values from raw data.",flush=True)
    elif input_mode == "pre_processed":
        preprocessed_df = pd.read_csv(pre_processed_data_path)
        if verbose:
            print(f"Loaded {len(preprocessed_df)} CpG values from pre-processed data.",flush=True)
    return preprocessed_df

def save_huggingface_dataset(dataframe_path, save_path, tokenizer,kmer_size,seq_size,verbose):
    dataset = Dataset.from_generator(dataset_generator_wrapper(dataframe_path, DATASET_BATCH_SIZE,kmer_size,seq_size))
    encoded_dataset = dataset.map(lambda examples: tokenizer(examples['seq'])).remove_columns(['seq'])
    
    if verbose:
        print("saving dataset to:",save_path,flush=True)
    
    encoded_dataset.save_to_disk(save_path, num_proc=1)
