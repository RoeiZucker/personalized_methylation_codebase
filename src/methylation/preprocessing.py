from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
WINDOW_ID_COLUMN_NAME = "window_id"

def create_length_window_id(df, window_length):
    pos = df['start'].to_numpy(dtype='int64')
    bin_index = np.floor_divide(pos, window_length)
    df['bin_start'] = (bin_index * window_length).astype('int64')
    df['bin_end'] = df['bin_start'] + window_length
    df[WINDOW_ID_COLUMN_NAME] = df['chrom'].astype('string') + ':' + df['bin_start'].astype(str) + '-' + df['bin_end'].astype(str)
    return df

def apply_create_labels_formultiple_instance_format_df(row, blank_value, variability_dict: dict, variant_filtering_upper_bound, variant_filtering_lower_bound):
    start = row['start']
    end = row['end']
    starts = row['starts']
    ends = row['ends']
    values = row['values']
    start_labels = [blank_value] * (starts[0] - start)
    end_labels = [blank_value] * (end - ends[-1])
    mid_labels_list = []
    for i in range(len(starts)):
        variant_id = f"{row['chrom']}:{starts[i]}-{ends[i]}"
        value = values[i]
        if variability_dict is not None:
            variant_variability = variability_dict.get(variant_id, None)
            if variant_variability is None or ((variant_variability < variant_filtering_lower_bound or variant_variability > variant_filtering_upper_bound) and (variant_filtering_lower_bound != -1 and variant_filtering_upper_bound != -1)):
                value = blank_value
        legnth = ends[i] - starts[i]
        mid_labels_list.extend([value] * legnth)
        if i + 1 < len(starts):
            blank_to_next_lable_length = starts[i + 1] - ends[i]
            mid_labels_list.extend([blank_value] * blank_to_next_lable_length)
    return start_labels + mid_labels_list + end_labels

def create_labels_formultiple_instance_format_df(df, blank_value='0', variability_dict=None, variant_filtering_upper_bound=-1, variant_filtering_lower_bound=-1):
    df['labels'] = df.apply(lambda row: apply_create_labels_formultiple_instance_format_df(row, blank_value, variability_dict, variant_filtering_upper_bound, variant_filtering_lower_bound), axis=1)

def add_sequence_formultiple_instance_format_df(df, chrom_dict: Dict[str, str]):
    df['seq'] = df.apply(apply_add_sequence_formultiple_instance_format_df_wrapper(chrom_dict), axis=1)
    return df

def apply_add_sequence_formultiple_instance_format_df_wrapper(chrom_dict):

    def apply_add_sequence_formultiple_instance_format_df(row):
        return str(chrom_dict[row['chrom']][row['start']:row['end']]).upper()
    return apply_add_sequence_formultiple_instance_format_df

def create_kmer_split_train_test_labels(lable_array, empty_label, p, random_seed):
    mask = lable_array != empty_label
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
    return (list(arr_train), list(arr_test))

def dataset_generator_wrapper(file_path: str, chunk_size: int, kmer_size=6, seq_size=5400) -> Dataset:

    def dataset_generator():
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk['labels'] = chunk['labels'].str.strip('[]').str.split(',').apply(lambda x: [float(i) for i in x])
            for row in chunk.itertuples():
                if len(row.seq) == seq_size and len(row.labels) == seq_size and ('N' not in row.seq) and (len(row.labels) > 0):
                    labels_raw = [-100]
                    for i in range(0, len(row.labels), kmer_size):
                        arr = np.array(row.labels[i:i + kmer_size])
                        if arr[0] != -100 and arr[1] == -100:
                            arr[0] = -100
                        if (arr == -100).all():
                            kmer_value = -100
                        else:
                            kmer_value = arr[arr != -100].mean()
                        labels_raw.append(kmer_value)
                    if hasattr(row, 'window_id'):
                        yield {'seq': row.seq, 'labels': np.array(labels_raw, dtype=np.float32), 'start': row.start, 'end': row.end, 'window_id': row.window_id}
                    else:
                        yield {'seq': row.seq, 'labels': np.array(labels_raw, dtype=np.float32), 'start': row.start, 'end': row.end}
    return dataset_generator

def preprocess_encode_cpg_dataframe(df: pd.DataFrame, remove_under_0=False) -> pd.DataFrame:
    if df.empty:
        return df
    remove_under_0 = 'filter'
    df['methyl_rate'] = df['methyl_rate'].astype(np.float16)
    if remove_under_0 == 'filter':
        df = df[df['methyl_rate'] >= 0]
    elif remove_under_0:
        df.loc[df['methyl_rate'] < 0, 'methyl_rate'] = -100.0
    return df
