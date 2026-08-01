# Downsample a saved Hugging Face token-classification dataset while preserving its schema.
# Example full dataset ratio run:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/downsample_token_classification_dataset.py --input-dataset-path /sci/labs/michall/roeizucker/roei_test/token_cls_datasets2/_token_cls_Liver-Hepatocytes_kmer/Z000000R3_retrain_token_cls_Liver-Hepatocytes_kmer_seq_5400_datasets_train --output-dataset-path /sci/labs/michall/roeizucker/roei_test/token_cls_datasets_downsampled/_token_cls_Liver-Hepatocytes_kmer/Z000000R3_retrain_token_cls_Liver-Hepatocytes_kmer_seq_5400_datasets_train_ratio_1p0 --minority-to-majority-ratio 1.0 --overwrite
# Example short debug run:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/downsample_token_classification_dataset.py --input-dataset-path /sci/labs/michall/roeizucker/roei_test/token_cls_datasets2/_token_cls_Liver-Hepatocytes_kmer/Z000000R3_retrain_token_cls_Liver-Hepatocytes_kmer_seq_5400_datasets_train --output-dataset-path /tmp/Z000000R3_downsample_debug --minority-to-majority-ratio 0.5 --top-rows 1000 --overwrite

import argparse
import os
import shutil
from collections import Counter

import numpy as np
from datasets import load_from_disk
from imblearn.under_sampling import RandomUnderSampler


def _label_counts(labels_arr, blank_label):
    labels = labels_arr[labels_arr != blank_label].astype(int)
    return {int(label): int(count) for label, count in sorted(Counter(labels).items())}


def downsample_dataset(dataset, ratio, seed, label_column="labels", blank_label=-100, filter_empty=True):
    original_features = dataset.features
    labels_arr = np.array(dataset[label_column])
    mask = labels_arr != blank_label
    example_idx, token_idx = np.where(mask)
    x = np.column_stack([example_idx, token_idx])
    y = labels_arr[example_idx, token_idx].astype(int)

    counts = Counter(y)
    if not counts or min(counts.values()) == 0:
        raise ValueError(f"Cannot downsample; label counts are {dict(counts)}")

    target_max = int(min(counts.values()) / ratio)
    sampling_strategy = {label: min(count, target_max) for label, count in counts.items()}
    sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed)
    x_resampled, _ = sampler.fit_resample(x, y)

    downsampled_labels = np.full_like(labels_arr, fill_value=blank_label)
    kept_example_idx = x_resampled[:, 0].astype(int)
    kept_token_idx = x_resampled[:, 1].astype(int)
    downsampled_labels[kept_example_idx, kept_token_idx] = labels_arr[kept_example_idx, kept_token_idx]

    def replace_labels(_, indices):
        return {label_column: [downsampled_labels[i].tolist() for i in indices]}

    dataset = dataset.map(replace_labels, with_indices=True, batched=True, load_from_cache_file=False)
    if filter_empty:
        dataset = dataset.filter(
            lambda row: any(int(label) != blank_label for label in row[label_column]),
            load_from_cache_file=False,
        )
    return dataset.cast(original_features)


def parse_args():
    parser = argparse.ArgumentParser(description="Downsample token-classification labels in a saved HF dataset.")
    parser.add_argument("--input-dataset-path", required=True)
    parser.add_argument("--output-dataset-path", required=True)
    parser.add_argument("--minority-to-majority-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-column", default="labels")
    parser.add_argument("--blank-label", type=int, default=-100)
    parser.add_argument("--top-rows", type=int, default=-1)
    parser.add_argument("--keep-empty-examples", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.minority_to_majority_ratio <= 0 or args.minority_to_majority_ratio > 1:
        raise ValueError("--minority-to-majority-ratio must be in (0, 1]")
    if os.path.abspath(args.input_dataset_path) == os.path.abspath(args.output_dataset_path):
        raise ValueError("Input and output dataset paths must be different")
    if os.path.exists(args.output_dataset_path):
        if not args.overwrite:
            raise FileExistsError(f"Output path already exists: {args.output_dataset_path}")
        shutil.rmtree(args.output_dataset_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_dataset_path)), exist_ok=True)

    dataset = load_from_disk(args.input_dataset_path)
    if args.top_rows != -1:
        dataset = dataset.select(range(min(args.top_rows, len(dataset))))
    before = _label_counts(np.array(dataset[args.label_column]), args.blank_label)
    dataset = downsample_dataset(
        dataset,
        ratio=args.minority_to_majority_ratio,
        seed=args.seed,
        label_column=args.label_column,
        blank_label=args.blank_label,
        filter_empty=not args.keep_empty_examples,
    )
    after = _label_counts(np.array(dataset[args.label_column]), args.blank_label)
    dataset.save_to_disk(args.output_dataset_path, num_proc=1)
    print({"saved_to": args.output_dataset_path, "counts_before": before, "counts_after": after}, flush=True)


if __name__ == "__main__":
    main()
