#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat
import re
import sys

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluator import evaluate_sample_predictions as evaluate_with_variability
from variability_free_evaluation import evaluate_sample_predictions as evaluate_variability_free


RESULTS_DIR = Path("/sci/labs/michall/roeizucker/cls_downsample_liver")
OUTPUT_ROOT = PROJECT_DIR / "scripts" / "results" / "cls_downsample_liver_grouped_evaluations"
VARIABILITY_DIR = Path(
    "/sci/labs/michall/roeizucker/huggingface_datasets_dir/"
    "huggingface_datasets_dir/_Liver-Hepatocytes_kmer"
)
BIGWIG_FILES_BY_SAMPLE = {
    "Z000000R3": "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652233_Liver-Hepatocytes-Z000000R3.hg38.bigwig",
    "Z000000T3": "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652234_Liver-Hepatocytes-Z000000T3.hg38.bigwig",
    "Z0000043Q": "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652235_Liver-Hepatocytes-Z0000043Q.hg38.bigwig",
    "Z0000044H": "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652236_Liver-Hepatocytes-Z0000044H.hg38.bigwig",
    "Z0000044M": "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652237_Liver-Hepatocytes-Z0000044M.hg38.bigwig",
    "Z00000431": "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652238_Liver-Hepatocytes-Z00000431.hg38.bigwig",
}

GROUPS = {
    "no_pretraining": re.compile(r"^[^_]+_no_pretraining_retrain_"),
    "epoch_1_pretraining": re.compile(r"^[^_]+_epoch-1-step-\d+_retrain_"),
    "epoch_2_pretraining": re.compile(r"^[^_]+_epoch-2-step-\d+_retrain_"),
    "epoch_3_pretraining": re.compile(r"^[^_]+_epoch-3-step-\d+_retrain_"),
}

FULL_POS_NAME = "full_pos"
RANGES = [0, 0.2, 0.8, 1]
LABELS = [0, 1, 2]
LABEL_A = "mean_label"
LABEL_B = "predicted_class"
COMPARISON_TYPES = [LABEL_A, LABEL_B]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["variability_free", "variability_bins"],
        required=True,
    )
    parser.add_argument(
        "--pretraining-group",
        choices=sorted(GROUPS),
        required=True,
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", default="14")
    parser.add_argument("--number-of-bins", type=int, default=5)
    parser.add_argument("--all-two", action="store_true", default=True)
    return parser.parse_args()


def sample_name_from_result_path(result_file_path):
    run_name = result_file_path.parents[1].name
    return run_name.split("_", 1)[0]


def group_name_from_run_dir(run_dir_name):
    for group_name, pattern in GROUPS.items():
        if pattern.search(run_dir_name):
            return group_name
    return None


def variability_file_path(sample_name):
    return (
        VARIABILITY_DIR
        / f"{sample_name}_per_varaint_variability_Liver-Hepatocytes_kmer_seq_5400_datasets.csv"
    )


def get_chroms(result_files_path):
    chroms = set()
    for result_file_path in result_files_path:
        window_ids = pd.read_csv(result_file_path, usecols=["window_id"])["window_id"]
        chroms.update(window_ids.str.split(":", n=1).str[0].dropna().unique())
    return sorted(chroms, key=lambda chrom: int(chrom[3:]) if chrom[3:].isdigit() else chrom)


def discover_grouped_result_files(results_dir, batch_size, pretraining_group):
    result_files_by_sample = {}
    for result_file_path in sorted(results_dir.glob("*/epoch-*/eval_predictions.csv.gitbackup")):
        run_dir_name = result_file_path.parents[1].name
        if f"_bs_{batch_size}_" not in run_dir_name:
            continue
        if group_name_from_run_dir(run_dir_name) != pretraining_group:
            continue
        sample_name = sample_name_from_result_path(result_file_path)
        result_files_by_sample.setdefault(sample_name, []).append(str(result_file_path))
    return result_files_by_sample


def evaluate_sample(sample_name, result_files_path, mode, number_of_bins, all_two):
    chroms = get_chroms(result_files_path)
    comparison_bigwig_files = [
        bigwig_file
        for curr_sample, bigwig_file in BIGWIG_FILES_BY_SAMPLE.items()
        if curr_sample != sample_name
    ]

    if mode == "variability_free":
        return evaluate_variability_free(
            result_files_path=result_files_path,
            chroms=chroms,
            comparison_bigiwg_files=comparison_bigwig_files,
            full_pos_name=FULL_POS_NAME,
            ranges=RANGES,
            labels=LABELS,
            comparison_types=COMPARISON_TYPES,
            all_two=all_two,
        )

    return evaluate_with_variability(
        variability_file_path=str(variability_file_path(sample_name)),
        result_files_path=result_files_path,
        chroms=chroms,
        comparison_bigiwg_files=comparison_bigwig_files,
        full_pos_name=FULL_POS_NAME,
        ranges=RANGES,
        labels=LABELS,
        label_a=LABEL_A,
        label_b=LABEL_B,
        number_of_bins=number_of_bins,
    )


def main():
    args = parse_args()
    result_files_by_sample = discover_grouped_result_files(
        args.results_dir,
        args.batch_size,
        args.pretraining_group,
    )
    if not result_files_by_sample:
        raise ValueError(
            f"No result files found for group={args.pretraining_group}, bs={args.batch_size}"
        )

    eval_objects_dict = {}
    for sample_name, result_files_path in sorted(result_files_by_sample.items()):
        print(
            f"{args.mode} {args.pretraining_group}: {sample_name} "
            f"({len(result_files_path)} prediction files)",
            flush=True,
        )
        eval_objects_dict[sample_name] = evaluate_sample(
            sample_name=sample_name,
            result_files_path=result_files_path,
            mode=args.mode,
            number_of_bins=args.number_of_bins,
            all_two=args.all_two,
        )

    output_dir = args.output_root / args.mode / args.pretraining_group
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval_objects_dict.txt"
    output_path.write_text(pformat(eval_objects_dict, sort_dicts=False), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
