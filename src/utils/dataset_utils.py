from typing import List, Dict, Optional
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np

def dataset_generator_wrapper(file_path : str,chunk_size : int,kmer_size = 6,seq_size=5400) -> Dataset:
    def dataset_generator():
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk["labels"] = chunk["labels"].str.strip('[]').str.split(',').apply(lambda x: [float(i) for i in x])
            for row in chunk.itertuples():
                if len(row.seq)  == seq_size and len(row.labels) == seq_size and "N" not in row.seq and len(row.labels) > 0:
                    # Here you can process each row as needed
                    # For example, you can yield a dictionary for each row
                    labels_raw = [-100]
                    for i in range(0,len(row.labels),kmer_size):

                        arr = np.array(row.labels[i:i+kmer_size])

                        # TODO: patch fix, change this:
                        if arr[0] != -100 and arr[1] == -100:
                            arr[0] = -100
                        
                        # print(type(arr))
                        # print(curr_label[i:i+kmer_size])
                        if (arr == -100).all():
                            kmer_value = -100
                        else:
                            kmer_value = arr[arr!=-100].mean()
                        labels_raw.append(kmer_value)
                    if hasattr(row, 'window_id'):
                        yield {"seq": row.seq, "labels": np.array(labels_raw, dtype=np.float32), "start": row.start, "end": row.end, "window_id": row.window_id}
                    else:
                        yield {"seq": row.seq, "labels": np.array(labels_raw, dtype=np.float32), "start": row.start, "end": row.end}
    return dataset_generator

def get_dataset_for_paths(path,top_rows = -1,load_dataset_to_memory=False):
    if isinstance(path, list):
        datasets = []
        for path in path:
            # TODO: repeating code, extract to function
            dataset = load_from_disk(path, keep_in_memory=load_dataset_to_memory)
            if top_rows != -1:
                dataset = dataset.select(range(top_rows))
            datasets.append(dataset)
        dataset = concatenate_datasets(datasets)
    else:
        dataset = load_from_disk(path, keep_in_memory=load_dataset_to_memory)
        if top_rows != -1:
            dataset = dataset.select(range(top_rows))
    
    return dataset

def keep_batch(batch, wanted_ids):
# batch is a dict of lists
    w = batch.get("window_id")
    if w is None:  # column missing in this dataset
        # return a mask of all False with the batch length
        batch_len = len(next(iter(batch.values())))
        return [False] * batch_len
    return [wid in wanted_ids for wid in w]
