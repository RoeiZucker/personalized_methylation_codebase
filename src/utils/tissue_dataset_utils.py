from typing import Optional

import numpy as np
import pandas as pd
from datasets import Dataset


def dataset_generator_wrapper(
    file_path: str,
    chunk_size: int,
    kmer_size=6,
    seq_size=5400,
    tissue_id: Optional[int] = None,
) -> Dataset:
    def dataset_generator():
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk["labels"] = chunk["labels"].str.strip("[]").str.split(",").apply(lambda x: [float(i) for i in x])
            for row in chunk.itertuples():
                if len(row.seq) == seq_size and len(row.labels) == seq_size and "N" not in row.seq and len(row.labels) > 0:
                    labels_raw = [-100]
                    for i in range(0, len(row.labels), kmer_size):
                        arr = np.array(row.labels[i : i + kmer_size])

                        if arr[0] != -100 and arr[1] == -100:
                            arr[0] = -100

                        if (arr == -100).all():
                            kmer_value = -100
                        else:
                            kmer_value = arr[arr != -100].mean()
                        labels_raw.append(kmer_value)

                    example = {
                        "seq": row.seq,
                        "labels": np.array(labels_raw, dtype=np.float32),
                        "start": row.start,
                        "end": row.end,
                    }
                    if hasattr(row, "window_id"):
                        example["window_id"] = row.window_id

                    current_tissue_id = tissue_id
                    if current_tissue_id is None and hasattr(row, "tissue_id"):
                        current_tissue_id = row.tissue_id
                    if current_tissue_id is not None:
                        example["tissue_ids"] = int(current_tissue_id)

                    yield example

    return dataset_generator
