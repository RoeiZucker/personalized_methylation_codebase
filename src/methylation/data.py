from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import methylation_class
from .preprocessing import (create_length_window_id, create_labels_formultiple_instance_format_df,
                            add_sequence_formultiple_instance_format_df, dataset_generator_wrapper,
                            create_kmer_split_train_test_labels, preprocess_encode_cpg_dataframe)

MODEL = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
MAIN_CHROMS = [f"chr{i}" for i in [1, 2, 3, 4, 5, 18, 19, 20, 21, 22]]
UNSEEN_CHROMS = [f"chr{i}" for i in range(6, 18)]


def read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def write_table(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def read_track(path, chromosomes=None):
    path = Path(path)
    if path.suffix.lower() in {".bw", ".bigwig"}:
        import pyBigWig
        records = []
        with pyBigWig.open(str(path)) as track:
            for chrom, length in track.chroms().items():
                if chromosomes is None or chrom in chromosomes:
                    records.extend((chrom, start, end, value) for start, end, value in (track.intervals(chrom, 0, length) or []))
        frame = pd.DataFrame(records, columns=["chrom", "start", "end", "methyl_rate"])
    else:
        frame = read_table(path)
    frame = frame[["chrom", "start", "end", "methyl_rate"]].copy()
    if chromosomes is not None:
        frame = frame[frame.chrom.isin(chromosomes)]
    frame = preprocess_encode_cpg_dataframe(frame)
    if not np.isfinite(frame.methyl_rate).all() or frame.methyl_rate.gt(1).any():
        raise ValueError("Invalid methylation track values")
    if frame.duplicated(["chrom", "start", "end"]).any() or frame.end.le(frame.start).any():
        raise ValueError("Invalid or duplicate intervals")
    return frame.reset_index(drop=True)


def prepare(track, fasta, output, donor, mode="personalized", chromosomes=None, model=MODEL, seed=42):
    from Bio import SeqIO
    from datasets import Dataset
    from transformers import AutoTokenizer
    from .sampling import apply_token_label_downsampling_to_dataset

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    chromosomes = chromosomes or (UNSEEN_CHROMS if mode == "chromosomes" else MAIN_CHROMS)
    if mode != "chromosomes" and not set(chromosomes).issubset(MAIN_CHROMS):
        raise ValueError("Supervised datasets must use chromosomes 1–5 and 18–22")
    frame = read_track(track, chromosomes)
    if frame.empty:
        raise ValueError("No measurements on the selected chromosomes")
    create_length_window_id(frame, 5400)
    groups = frame.groupby("window_id").agg(start=("start", list), end=("end", list),
        methyl_rate=("methyl_rate", list), chrom=("chrom", "first"), bin_start=("bin_start", "first"), bin_end=("bin_end", "first"))
    groups = groups.reset_index().rename(columns={"start":"starts", "end":"ends", "methyl_rate":"values", "bin_start":"start", "bin_end":"end"})
    create_labels_formultiple_instance_format_df(groups, -100)
    groups = groups[groups.labels.map(set).map(len).gt(1)].copy()
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta, "fasta") if record.id in chromosomes}
    missing = set(groups.chrom) - sequences.keys()
    if missing:
        raise ValueError(f"Reference FASTA is missing chromosomes: {sorted(missing)}")
    add_sequence_formultiple_instance_format_df(groups, sequences)
    intermediate = output / "windows.csv"
    groups.to_csv(intermediate, index=False)
    dataset = Dataset.from_generator(dataset_generator_wrapper(str(intermediate), 50, 6, 5400))
    if len(dataset) == 0:
        raise ValueError("No complete, unambiguous 5400-base windows")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def classes(example):
        labels = np.asarray(example["labels"])
        valid = labels != -100
        result = np.full(len(labels), -100, dtype=np.int64)
        result[valid] = methylation_class(labels[valid])
        return {"labels": result}

    def tokenize(example):
        encoded = tokenizer(example["seq"])
        if len(encoded["input_ids"]) != len(example["labels"]):
            raise ValueError("Tokenizer does not match six-base tokens plus one special token")
        return encoded

    if mode == "personalized":
        def split(example):
            train, held = create_kmer_split_train_test_labels(np.asarray(example["labels"]), -100, .8, seed)
            return {"train_labels": train, "held_labels": held}
        divided = dataset.map(split)
        train = divided.remove_columns(["labels", "held_labels"]).rename_column("train_labels", "labels")
        held = divided.remove_columns(["labels", "train_labels"]).rename_column("held_labels", "labels")
        train = train.filter(lambda x: any(v != -100 for v in x["labels"])).map(classes)
        held = held.filter(lambda x: any(v != -100 for v in x["labels"])).map(classes)
        train = apply_token_label_downsampling_to_dataset(train, {"token_label_downsampling": {"minority_to_majority_ratio": .25}}, seed=seed)
        train.map(tokenize).remove_columns("seq").save_to_disk(str(output / "TRAIN"))
        held = held.map(tokenize).remove_columns("seq")
        split = held.train_test_split(test_size=.2, seed=seed)
        split["train"].save_to_disk(str(output / "TEST"))
        split["test"].save_to_disk(str(output / "EVAL"))
    else:
        name = "TRAIN" if mode == "reference" else "TEST"
        dataset.map(classes).map(tokenize).remove_columns("seq").save_to_disk(str(output / name))
    (output / "study.json").write_text(json.dumps({"donor": donor, "mode": mode, "chromosomes": chromosomes, "seed": seed, "model": model}, indent=2) + "\n")


def export_training_sites(dataset_path, donor, output):
    from datasets import load_from_disk
    rows = []
    for example in load_from_disk(str(dataset_path)):
        if np.count_nonzero(np.asarray(example["labels"]) != -100) < 10:
            continue
        for token in np.flatnonzero(np.asarray(example["labels"]) != -100):
            if token == 0:
                raise ValueError("Special-token label must be masked")
            rows.append({"donor": donor, "chrom": example["window_id"].split(":")[0],
                         "genomic_position": int(example["start"]) + (int(token) - 1) * 6, "label": int(example["labels"][token])})
    write_table(pd.DataFrame(rows), output)
