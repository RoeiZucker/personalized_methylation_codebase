from typing import Dict, Tuple, Callable, Any

import pandas as pd
from random import randint
import numpy as np

# from constants import (
#     BLANK_LABEL_VALUE
# )



# TODO: move to constants file
WINDOW_ID_COLUMN_NAME = "window_id"

MAX_DIFF_VARIABILITY_TYPE = "max_diff"

#TODO: moved to constants file but not deleted 
STD_VARIABILITY_TYPE = "std"
QUANTILE_SEPERATION_TYPE = "quantile"


# TODO: change so it's more generic
def combine_cpg_dfs(full_position_column_name, dfs, labels):
    kept = [
        d for d in dfs
        if (full_position_column_name in d.columns) and ('methyl_rate' in d.columns) and d[full_position_column_name].notna().all()
    ]

    out = pd.concat(
        [
            d[[full_position_column_name, 'methyl_rate']]
            .rename(columns={'methyl_rate': f'methyl_rate_{lab}'})
            .set_index(full_position_column_name)
            for lab, d in zip(labels, kept)
        ],
        axis=1, join='inner'
    ).reset_index()
    df = out.merge(
        kept[0][[full_position_column_name, "start","end","chrom"]].drop_duplicates(full_position_column_name),
        on=full_position_column_name,
        how="left"
    )
    
    return df



# TODO: change so it's more generic

def create_methyl_variability_df(dfs_num, df,variability_type,variant_seperation_type,variant_seperation_threshold,seq_size):
    
    if variability_type == MAX_DIFF_VARIABILITY_TYPE:
        count = 0
        for i in range(dfs_num):
            for j in range(i,dfs_num):
                if i == j:
                    continue
                count+=1
                df[f"{i}-{j}"] = (df[f"methyl_rate_ind_{i}"] - df[f"methyl_rate_ind_{j}"]).abs()
        df[variability_type] = df[df.columns[-count:]].max(axis=1)
    elif variability_type == STD_VARIABILITY_TYPE:
        df[variability_type] = df[df.columns[1:dfs_num+1]].std(axis=1)
    else:
        raise NotImplementedError()
    
    if variant_seperation_type == QUANTILE_SEPERATION_TYPE:
        max_diff_quantile_value = df[variability_type].quantile(variant_seperation_threshold)
        print("max_diff_quantile_value",max_diff_quantile_value)
        # TODO: need to set high_diff as constant
        df["high_diff"] = False
        df.loc[df[variability_type] > max_diff_quantile_value,"high_diff"] = True
    else:
        raise NotImplementedError()
    
    create_length_window_id(df,seq_size) # TODO: change to parameter!
    return df

def group_methyl_variability_df(df):
    grouped = df[[WINDOW_ID_COLUMN_NAME,"high_diff","bin_start","bin_end","chrom"]].groupby(WINDOW_ID_COLUMN_NAME)
    group_agg_df = grouped.agg({'high_diff':'mean', 
                            'bin_start':'mean', 
                            'bin_end':'mean', 
                            'chrom': lambda x: list(x)[0]})
                            
    return group_agg_df


# TODO: change so df isn't changed, or add inplace if want to change
# TODO: add parameter for window_id name
def create_length_window_id(df, window_length):
    pos = df['start'].to_numpy(dtype='int64')
# if your 'start' is 1-based, convert first:
# pos = (df['start'].to_numpy(dtype='int64') - 1)
    bin_index = np.floor_divide(pos, window_length)          # integer division
    df['bin_start'] = (bin_index * window_length).astype('int64')
    df['bin_end']   = df['bin_start'] + window_length
# optional: stable ID (handy for joins/metrics)
    df[WINDOW_ID_COLUMN_NAME] = df['chrom'].astype('string') + ':' + df['bin_start'].astype(str) + '-' + df['bin_end'].astype(str)
    return df

def combine_rows_to_multiple_instances_format(df, maximal_length): \
    # combines rows in a dataframe so that each element that is within range will be combined to a single row
    # columns must be start,end,value,chrom
    # TODO: add remove intersections
    chrom_values = df["chrom"].unique()
    new_vals = []
    for chrom in chrom_values:
        chrom_df = create_chrom_specific_df(chrom, df)
        extract_values_from_chrom_df_to_list(chrom, chrom_df, maximal_length, new_vals)
    return pd.DataFrame(new_vals, columns=["chrom", "starts", "ends", "values"])


def extract_values_from_chrom_df_to_list(chrom, chrom_df, maximal_length, new_vals):
    curr_ends, curr_starts, curr_values = create_clear_element_values()
    last_start = 0
    for index, row in chrom_df.iterrows():
        start = row["start"]
        end = row["end"]
        value = row["value"]
        if start - last_start < maximal_length:
            # print("AAAA")
            # TODO: need to make sure to remove previous if can't ensure no collision
            continue

        if element_addition_fits_within_limits(curr_starts, end, maximal_length, start):
            add_element(curr_ends, curr_starts, curr_values, end, start, value)
            continue
        new_vals.append([chrom, curr_starts, curr_ends, curr_values])
        last_start = curr_starts[0]
        curr_ends, curr_starts, curr_values = create_clear_element_values()
        if (end - start < maximal_length):
            add_element(curr_ends, curr_starts, curr_values, end, start, value)
    if len(curr_starts) != 0:
        new_vals.append([chrom, curr_starts, curr_ends, curr_values])


def create_clear_element_values():
    curr_starts = []
    curr_ends = []
    curr_values = []
    return curr_ends, curr_starts, curr_values


def add_element(curr_ends, curr_starts, curr_values, end, start, value):
    curr_starts.append(start)
    curr_ends.append(end)
    curr_values.append(value)


def element_addition_fits_within_limits(curr_starts, end, maximal_length, start):
    return (len(curr_starts) == 0 and (end - start < maximal_length)) or \
           (len(curr_starts) != 0 and (end - curr_starts[0]) < (maximal_length))


def create_chrom_specific_df(chrom, df):
    chrom_df = df[df["chrom"] == chrom]
    chrom_df = chrom_df.sort_values(by=['chrom', 'start']).reset_index(drop=True)
    return chrom_df
def apply_create_start_for_multiple_instance_format_df(row,seq_size,test_mode=False) -> int:
    # print(row["starts"][0])
    # print(row["ends"][-1])
    length = row["ends"][-1] - row["starts"][0]
    to_fill = seq_size - length
    if test_mode:
        length_before_first_element = to_fill//2
    else:
        length_before_first_element = randint(0,to_fill)
    return row["starts"][0] - length_before_first_element

def add_borders_to_multiple_instance_format_df(df,seq_size,test_mode=False):
    df["start"] = 0
    print(df.apply(lambda row:apply_create_start_for_multiple_instance_format_df(row,seq_size,test_mode),axis=1))
    df["start"] = df.apply(lambda row:apply_create_start_for_multiple_instance_format_df(row,seq_size,test_mode),axis=1)
    df["end"] = df["start"] + seq_size


def apply_create_labels_formultiple_instance_format_df(row,blank_value,variability_dict : dict,variant_filtering_upper_bound,
                                                       variant_filtering_lower_bound):
    # assumes no intersection beteen values in labels
    start = row["start"]
    end = row["end"]
    starts = row["starts"]
    ends = row["ends"]
    # TODO: make sure this is correct
    # for i in range(len(ends)):
    #     ends[i] = ends[i] - 1
    values = row["values"]
    start_labels = [blank_value] * ( starts[0] - start)
    end_labels = [blank_value] *( end - ends[-1])
    mid_labels_list = []
    for i in range(len(starts)):
        variant_id = f'{row["chrom"]}:{starts[i]}-{ends[i]}'
        value = values[i]
        if variability_dict is not None:
            variant_variability = variability_dict.get(variant_id,None)
            if variant_variability is None or \
                ((variant_variability < variant_filtering_lower_bound or variant_variability > variant_filtering_upper_bound)and \
                 (variant_filtering_lower_bound != -1 and variant_filtering_upper_bound != -1)):
                value = blank_value
            
        legnth = ends[i] - starts[i]
        # legnth = ends[i] - starts[i] - 1

        mid_labels_list.extend([value]* legnth)
        # assumes no intersection between labels
        if i+1 < len(starts):
            blank_to_next_lable_length = starts[i+1] - ends[i]
            # blank_to_next_lable_length = starts[i+1] - ends[i] + 1
            mid_labels_list.extend([blank_value] * blank_to_next_lable_length)
    return start_labels + (mid_labels_list) + end_labels
def create_labels_formultiple_instance_format_df(df,blank_value="0",variability_dict = None,variant_filtering_upper_bound = -1,
                                                 variant_filtering_lower_bound = -1):
    
    # TODO add variant filtering here
    df["labels"] = df.apply(lambda row:apply_create_labels_formultiple_instance_format_df(row,blank_value,variability_dict,
                                                variant_filtering_upper_bound,variant_filtering_lower_bound),axis=1)
    # pass



def add_sequence_formultiple_instance_format_df(df,chrom_dict: Dict[str, str]):
    # print(df.apply(apply_add_sequence_formultiple_instance_format_df_wrapper(chrom_dict),axis=1))
    df["seq"] = df.apply(apply_add_sequence_formultiple_instance_format_df_wrapper(chrom_dict),axis=1)
    return df

def apply_add_sequence_formultiple_instance_format_df_wrapper(chrom_dict):
    def apply_add_sequence_formultiple_instance_format_df(row):
        return str(chrom_dict[row["chrom"]][row["start"]:row["end"]]).upper()
    return apply_add_sequence_formultiple_instance_format_df

def filter_dataset_chromosomes(df,chrom_container):
    return df[df["chrom"].isin(chrom_container)]


def convert_string_column_to_list(
    df: pd.DataFrame,
    column_name: str,
    dtype: Callable[[str], Any],
    strip_chars: str = "[]"
) -> pd.Series:
    """
    Convert a DataFrame column of string-encoded lists into a real list type.

    E.g. turn a cell like "[1.0,2.5,3.0]" into [1.0, 2.5, 3.0].

    Args:
      df:           Source DataFrame.
      column_name:  Name of the column containing string-encoded lists.
      dtype:        A callable (e.g. float, int, str) to cast each split element.
      strip_chars:  Characters to strip off the ends before splitting (default "[]").

    Returns:
      A pandas Series where each entry is a list of `dtype`-converted items.
    """
    return (
        df[column_name]
        .str.strip(strip_chars)
        .str.split(",")
        .apply(lambda items: [dtype(item) for item in items if item != ""])
    )
