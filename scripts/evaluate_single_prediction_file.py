#!/usr/bin/env python3
# Example:
#   python3 scripts/evaluate_single_prediction_file.py \
#     /sci/labs/michall/roeizucker/roei_test/token_cls_models_downsampled/_token_cls_Liver-Hepatocytes_kmer/Z00000431_epoch-10-step-849850_downsampled_ratio_0p25_retrain_token_cls_Liver-Hepatocytes_kmer_lr_3e-05_bs_14_seq_5400/checkpoint-14870/eval_predictions.csv.gitbackup \
#     /sci/labs/michall/roeizucker/roei_test/token_cls_models_downsampled/_token_cls_Liver-Hepatocytes_kmer/Z00000431_epoch-10-step-849850_downsampled_ratio_0p25_retrain_token_cls_Liver-Hepatocytes_kmer_lr_3e-05_bs_14_seq_5400/checkpoint-29740/eval_predictions.csv.gitbackup \
#     /sci/labs/michall/roeizucker/roei_test/token_cls_models_downsampled/_token_cls_Liver-Hepatocytes_kmer/Z00000431_epoch-10-step-849850_downsampled_ratio_0p25_retrain_token_cls_Liver-Hepatocytes_kmer_lr_3e-05_bs_14_seq_5400/checkpoint-44610/eval_predictions.csv.gitbackup \
#     scripts/results/downsampled_ratio_0p25_lr_3e-05_Z00000431_eval_variability_bins/eval_eval_objects_dict.txt

from argparse import ArgumentParser
import ast
from pathlib import Path
from pprint import pformat
import pickle
import re
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluator import evaluate_sample_predictions


COMPARISON_DICTS_ROOT = Path("/sci/labs/michall/roeizucker/comparison_dicts")
HF_DATASETS_ROOT = Path(
    "/sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir"
)
CHROMS = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr18", "chr19", "chr20", "chr21", "chr22"]
RANGES = [0, 0.2, 0.8, 1]
LABELS = [0, 1, 2]


def parse_args():
    parser = ArgumentParser(
        description=(
            "Evaluate one or more token-classification prediction CSVs with "
            "variability bins. The last positional argument is the destination file."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path)
    return parser.parse_args()


def split_input_paths(paths):
    if len(paths) < 2:
        raise ValueError("Pass at least one prediction CSV and one destination file")
    return [path.resolve() for path in paths[:-1]], paths[-1]


def infer_sample_name(prediction_file):
    for parent in prediction_file.parents:
        if parent.name.startswith("checkpoint-") or parent.name.startswith("pretrain_"):
            continue
        match = re.match(r"^(?P<sample>[^_]+)_", parent.name)
        if match is not None:
            return match.group("sample")
    raise ValueError(f"Could not infer sample name from {prediction_file}")


def infer_tissue_name(prediction_file):
    for parent in prediction_file.parents:
        match = re.match(r"^_token_cls_(?P<tissue>.+)_kmer$", parent.name)
        if match is not None:
            return match.group("tissue")
        match = re.search(r"_token_cls_(?P<tissue>.+)_kmer_", parent.name)
        if match is not None:
            return match.group("tissue")
    raise ValueError(f"Could not infer tissue name from {prediction_file}")


def load_comparison_dicts(sample_name, tissue_name):
    path = COMPARISON_DICTS_ROOT / tissue_name / f"{sample_name}_comparison_dicts.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison dict file: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "compare_dicts" in payload:
        return payload["compare_dicts"]
    return payload


def variability_file_path(sample_name, tissue_name):
    return (
        HF_DATASETS_ROOT
        / f"_{tissue_name}_kmer"
        / f"{sample_name}_per_varaint_variability_{tissue_name}_kmer_seq_5400_datasets.csv"
    )


def load_existing(destination_file):
    if not destination_file.exists():
        return {}
    return ast.literal_eval(destination_file.read_text(encoding="utf-8"))


def cache_path(destination_file):
    cache_dir = destination_file.parent / "mean_label_caches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{destination_file.stem}_mean_label_cache.pkl"


def main():
    args = parse_args()
    prediction_files, destination_file = split_input_paths(args.paths)
    destination_file.parent.mkdir(parents=True, exist_ok=True)

    sample_name = infer_sample_name(prediction_files[0])
    tissue_name = infer_tissue_name(prediction_files[0])
    for prediction_file in prediction_files:
        if infer_sample_name(prediction_file) != sample_name:
            raise ValueError("All prediction files must belong to the same sample")
        if infer_tissue_name(prediction_file) != tissue_name:
            raise ValueError("All prediction files must belong to the same tissue")

    comparison_dicts = load_comparison_dicts(sample_name, tissue_name)
    variability_file = variability_file_path(sample_name, tissue_name)
    if not variability_file.exists():
        raise FileNotFoundError(f"Missing variability file: {variability_file}")

    eval_objects = load_existing(destination_file)
    eval_objects.setdefault(sample_name, {})

    result = evaluate_sample_predictions(
        variability_file_path=str(variability_file),
        result_files_path=[str(prediction_file) for prediction_file in prediction_files],
        chroms=CHROMS,
        comparison_bigwig_files=None,
        full_pos_name="full_pos",
        ranges=RANGES,
        labels=LABELS,
        label_a="mean_label",
        label_b="predicted_class",
        number_of_bins=5,
        comparison_dicts=comparison_dicts,
        mean_label_cache_path=cache_path(destination_file),
    )
    eval_objects[sample_name].update(result)
    destination_file.write_text(pformat(eval_objects, sort_dicts=False), encoding="utf-8")
    print(f"Wrote {destination_file}", flush=True)


if __name__ == "__main__":
    main()
