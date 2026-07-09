#!/usr/bin/env python3
# Examples:
#   python3 scripts/evaluate_grouped_predictions.py --mode variability_free --pretraining-group epoch_3_pretraining --comparison-dicts-dir /sci/labs/michall/roeizucker/comparison_dicts/Liver-Hepatocytes --chroms chr1,chr2,chr3
#   python3 scripts/evaluate_grouped_predictions.py --mode variability_bins --pretraining-group epoch_3_pretraining --comparison-dicts-dir /sci/labs/michall/roeizucker/comparison_dicts/Liver-Hepatocytes --mean-label-cache-dir scripts/results/mean_label_caches/Liver-Hepatocytes --chroms chr1,chr2,chr3
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

from evaluator import evaluate_sample_predictions as evaluate_with_variability
from variability_free_evaluation import evaluate_sample_predictions as evaluate_variability_free
from create_comparison_dicts import discover_bigwig_files_by_sample, read_bigwig_map_file


RESULTS_DIR = Path("/sci/labs/michall/roeizucker/cls_downsample_liver")
OUTPUT_ROOT = PROJECT_DIR / "scripts" / "results" / "grouped_evaluations"
DATASETS_DIR = Path("/sci/archive/michall/roeizucker/downloaded_datasets")
HF_DATASETS_ROOT = Path("/sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir")
DEFAULT_TISSUE_NAME = "Liver-Hepatocytes"


NO_PRETRAINING_RUN_RE = re.compile(r"^[^_]+_no_pretraining_retrain_")
RETRAIN_RUN_RE = re.compile(r"^[^_]+_epoch-(?P<epoch>\d+)-step-\d+_retrain_")
FIRST_PHASE_RUN_RE = re.compile(r"^[^_]+_epoch-(?P<epoch>\d+)-step-\d+_pretrain_prediction$")
RETRAIN_GROUP_RE = re.compile(r"^epoch_(?P<epoch>\d+)_pretraining$")
FIRST_PHASE_GROUP_RE = re.compile(r"^first_phase_epoch_(?P<epoch>\d+)$")

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
        required=True,
        help=(
            "Group to evaluate: no_pretraining, epoch_<N>_pretraining, "
            "or first_phase_epoch_<N>."
        ),
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--tissue-name", default=DEFAULT_TISSUE_NAME)
    parser.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--bigwig-map-file", type=Path, default=None)
    parser.add_argument("--variability-dir", type=Path, default=None)
    parser.add_argument("--batch-size", default="14")
    parser.add_argument("--number-of-bins", type=int, default=5)
    parser.add_argument("--comparison-dicts-dir", type=Path, default=None)
    parser.add_argument("--precomputed-scores-dir", type=Path, default=None)
    parser.add_argument(
        "--mean-label-cache-dir",
        type=Path,
        default=None,
        help="Directory for cached sample-level mean_label dataframes used by variability_bins mode.",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Load the existing output file, evaluate only missing result paths, and write the merged dictionary.",
    )
    parser.add_argument(
        "--max-files-per-sample",
        type=int,
        default=None,
        help="Evaluate at most this many prediction files per sample, after sorting.",
    )
    parser.add_argument(
        "--include-low-var-runs",
        action="store_true",
        help="Include run directories whose names contain _low_var. They are skipped by default.",
    )
    parser.add_argument(
        "--low-var-only",
        action="store_true",
        help="Evaluate only run directories whose names contain _low_var.",
    )
    parser.add_argument(
        "--chrom",
        dest="chrom",
        action="append",
        default=[],
        help="Chromosome to evaluate. Can be passed multiple times, e.g. --chrom chr1 --chrom chr2.",
    )
    parser.add_argument(
        "--chroms",
        default=None,
        help="Comma-separated chromosomes to evaluate, e.g. chr1,chr2,chr3.",
    )
    parser.add_argument("--skip-mean-label", action="store_true")
    parser.add_argument("--all-two", dest="all_two", action="store_true", default=True)
    parser.add_argument("--no-all-two", dest="all_two", action="store_false")
    return parser.parse_args()


def sample_name_from_result_path(result_file_path):
    run_name = result_file_path.parents[1].name
    return run_name.split("_", 1)[0]


def group_name_from_run_dir(run_dir_name):
    if NO_PRETRAINING_RUN_RE.search(run_dir_name):
        return "no_pretraining"
    match = RETRAIN_RUN_RE.search(run_dir_name)
    if match is not None:
        return f"epoch_{int(match.group('epoch'))}_pretraining"
    match = FIRST_PHASE_RUN_RE.search(run_dir_name)
    if match is not None:
        return f"first_phase_epoch_{int(match.group('epoch'))}"
    return None


def is_valid_pretraining_group(group_name):
    return (
        group_name == "no_pretraining"
        or RETRAIN_GROUP_RE.fullmatch(group_name) is not None
        or FIRST_PHASE_GROUP_RE.fullmatch(group_name) is not None
    )


def variability_file_path(sample_name, tissue_name, variability_dir):
    if variability_dir is None:
        variability_dir = HF_DATASETS_ROOT / f"_{tissue_name}_kmer"
    return (
        variability_dir
        / f"{sample_name}_per_varaint_variability_{tissue_name}_kmer_seq_5400_datasets.csv"
    )


def parse_chroms(args):
    chroms = list(args.chrom)
    if args.chroms:
        chroms.extend(chrom.strip() for chrom in args.chroms.split(",") if chrom.strip())
    return chroms


def discover_grouped_result_files(results_dir, batch_size, pretraining_group, include_low_var_runs=False, low_var_only=False):
    result_files_by_sample = {}
    for result_file_path in sorted(results_dir.glob("*/epoch-*/eval_predictions.csv.gitbackup")):
        run_dir_name = result_file_path.parents[1].name
        is_low_var_run = "_low_var" in run_dir_name
        if low_var_only and not is_low_var_run:
            continue
        if not low_var_only and not include_low_var_runs and is_low_var_run:
            continue
        group_name = group_name_from_run_dir(run_dir_name)
        if group_name != pretraining_group:
            continue
        if not pretraining_group.startswith("first_phase_") and f"_bs_{batch_size}_" not in run_dir_name:
            continue
        sample_name = sample_name_from_result_path(result_file_path)
        result_files_by_sample.setdefault(sample_name, []).append(str(result_file_path))
    return result_files_by_sample


def load_comparison_dicts(comparison_dicts_dir, sample_name):
    if comparison_dicts_dir is None:
        return None
    path = comparison_dicts_dir / f"{sample_name}_comparison_dicts.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison dict cache for {sample_name}: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "compare_dicts" in payload:
        return payload["compare_dicts"]
    return payload


def load_precomputed_scores(precomputed_scores_dir, sample_name):
    if precomputed_scores_dir is None:
        return None
    path = precomputed_scores_dir / f"{sample_name}_precomputed_scores.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing precomputed scores for {sample_name}: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "scores" in payload:
        return payload["scores"]
    return payload


def load_existing_eval_objects(output_path):
    if not output_path.exists():
        return {}
    return ast.literal_eval(output_path.read_text(encoding="utf-8"))


def filter_missing_result_files(result_files_by_sample, existing_eval_objects):
    missing = {}
    skipped_count = 0
    for sample_name, result_files_path in result_files_by_sample.items():
        existing_paths = set(existing_eval_objects.get(sample_name, {}))
        missing_paths = [
            result_file_path
            for result_file_path in result_files_path
            if result_file_path not in existing_paths
        ]
        skipped_count += len(result_files_path) - len(missing_paths)
        if missing_paths:
            missing[sample_name] = missing_paths
    return missing, skipped_count


def merge_eval_objects(existing_eval_objects, new_eval_objects):
    merged = dict(existing_eval_objects)
    for sample_name, sample_results in new_eval_objects.items():
        merged.setdefault(sample_name, {}).update(sample_results)
    return merged


def evaluate_sample(sample_name, result_files_path, mode, number_of_bins, all_two, comparison_dicts_dir, bigwig_files_by_sample, tissue_name, variability_dir, precomputed_scores_dir, skip_mean_label, chroms, mean_label_cache_dir):
    comparison_bigwig_files = [
        bigwig_file
        for curr_sample, bigwig_file in bigwig_files_by_sample.items()
        if curr_sample != sample_name
    ]
    comparison_dicts = load_comparison_dicts(comparison_dicts_dir, sample_name)
    precomputed_scores = load_precomputed_scores(precomputed_scores_dir, sample_name)

    if mode == "variability_free":
        return evaluate_variability_free(
            result_files_path=result_files_path,
            chroms=chroms,
            comparison_bigwig_files=comparison_bigwig_files,
            full_pos_name=FULL_POS_NAME,
            ranges=RANGES,
            labels=LABELS,
            comparison_types=COMPARISON_TYPES,
            all_two=all_two,
            comparison_dicts=comparison_dicts,
            skip_mean_label=skip_mean_label,
            precomputed_scores=precomputed_scores,
        )

    mean_label_cache_path = None
    if mean_label_cache_dir is not None:
        mean_label_cache_path = mean_label_cache_dir / f"{sample_name}_mean_label_cache.pkl"

    return evaluate_with_variability(
        variability_file_path=str(variability_file_path(sample_name, tissue_name, variability_dir)),
        result_files_path=result_files_path,
        chroms=chroms,
        comparison_bigwig_files=comparison_bigwig_files,
        full_pos_name=FULL_POS_NAME,
        ranges=RANGES,
        labels=LABELS,
        label_a=LABEL_A,
        label_b=LABEL_B,
        number_of_bins=number_of_bins,
        comparison_dicts=comparison_dicts,
        mean_label_cache_path=mean_label_cache_path,
    )


def main():
    args = parse_args()
    if not is_valid_pretraining_group(args.pretraining_group):
        raise ValueError(
            "--pretraining-group must be no_pretraining, "
            "epoch_<N>_pretraining, or first_phase_epoch_<N>"
        )
    chroms = parse_chroms(args)
    output_dir = args.output_root / args.mode / args.pretraining_group
    output_path = output_dir / "eval_objects_dict.txt"
    if not chroms and (args.mode == "variability_bins" or not args.skip_mean_label):
        raise ValueError(
            "Pass chromosomes with --chrom/--chroms when mean_label is calculated."
        )

    result_files_by_sample = discover_grouped_result_files(
        args.results_dir,
        args.batch_size,
        args.pretraining_group,
        include_low_var_runs=args.include_low_var_runs,
        low_var_only=args.low_var_only,
    )
    if not result_files_by_sample:
        raise ValueError(
            f"No result files found for group={args.pretraining_group}, bs={args.batch_size}"
        )
    if args.max_files_per_sample is not None:
        result_files_by_sample = {
            sample_name: result_files_path[:args.max_files_per_sample]
            for sample_name, result_files_path in result_files_by_sample.items()
        }

    existing_eval_objects = {}
    if args.update_existing:
        existing_eval_objects = load_existing_eval_objects(output_path)
        result_files_by_sample, skipped_count = filter_missing_result_files(
            result_files_by_sample,
            existing_eval_objects,
        )
        print(f"Skipping {skipped_count} existing prediction files", flush=True)
        if not result_files_by_sample:
            print(f"No new prediction files to evaluate for {args.pretraining_group}", flush=True)
            return

    if args.comparison_dicts_dir is not None:
        bigwig_files_by_sample = {}
    elif args.bigwig_map_file is not None:
        bigwig_files_by_sample = read_bigwig_map_file(args.bigwig_map_file)
    else:
        bigwig_files_by_sample = discover_bigwig_files_by_sample(args.tissue_name, args.datasets_dir)
    if not bigwig_files_by_sample and args.comparison_dicts_dir is None:
        raise ValueError(
            "No bigwig files found. Pass --bigwig-map-file or --comparison-dicts-dir."
        )

    new_eval_objects_dict = {}
    for sample_name, result_files_path in sorted(result_files_by_sample.items()):
        print(
            f"{args.mode} {args.pretraining_group}: {sample_name} "
            f"({len(result_files_path)} prediction files)",
            flush=True,
        )
        new_eval_objects_dict[sample_name] = evaluate_sample(
            sample_name=sample_name,
            result_files_path=result_files_path,
            mode=args.mode,
            number_of_bins=args.number_of_bins,
            all_two=args.all_two,
            comparison_dicts_dir=args.comparison_dicts_dir,
            bigwig_files_by_sample=bigwig_files_by_sample,
            tissue_name=args.tissue_name,
            variability_dir=args.variability_dir,
            precomputed_scores_dir=args.precomputed_scores_dir,
            skip_mean_label=args.skip_mean_label,
            chroms=chroms,
            mean_label_cache_dir=args.mean_label_cache_dir,
        )

    eval_objects_dict = (
        merge_eval_objects(existing_eval_objects, new_eval_objects_dict)
        if args.update_existing
        else new_eval_objects_dict
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pformat(eval_objects_dict, sort_dicts=False), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()

# Run examples:
#
# 1. Variability-free evaluation with precomputed mean/all_two scores.
#    This is the fast path for updating existing tissue evaluations.
#    python3 scripts/evaluate_grouped_predictions.py \
#      --mode variability_free \
#      --pretraining-group epoch_3_pretraining \
#      --results-dir /sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer \
#      --tissue-name Liver-Hepatocytes \
#      --comparison-dicts-dir /sci/labs/michall/roeizucker/comparison_dicts/Liver-Hepatocytes \
#      --precomputed-scores-dir scripts/results/precomputed_scores/Liver-Hepatocytes \
#      --output-root scripts/results/all_tissues_grouped_evaluations/Liver-Hepatocytes \
#      --chroms chr1,chr2,chr3,chr4,chr5,chr18,chr19,chr20,chr21,chr22 \
#      --skip-mean-label \
#      --update-existing
#
# 2. Variability-free evaluation that computes mean_label during the run.
#    Use this when precomputed scores do not exist yet.
#    python3 scripts/evaluate_grouped_predictions.py \
#      --mode variability_free \
#      --pretraining-group no_pretraining \
#      --results-dir /sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer \
#      --tissue-name Liver-Hepatocytes \
#      --comparison-dicts-dir /sci/labs/michall/roeizucker/comparison_dicts/Liver-Hepatocytes \
#      --output-root scripts/results/all_tissues_grouped_evaluations/Liver-Hepatocytes \
#      --chroms chr1,chr2,chr3,chr4,chr5,chr18,chr19,chr20,chr21,chr22
#
# 3. Variability-bin evaluation with cached sample-level mean_label tables.
#    Reuse the same mean-label-cache-dir across groups for the same tissue.
#    python3 scripts/evaluate_grouped_predictions.py \
#      --mode variability_bins \
#      --pretraining-group epoch_3_pretraining \
#      --results-dir /sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer \
#      --tissue-name Liver-Hepatocytes \
#      --comparison-dicts-dir /sci/labs/michall/roeizucker/comparison_dicts/Liver-Hepatocytes \
#      --mean-label-cache-dir scripts/results/mean_label_caches/Liver-Hepatocytes \
#      --output-root scripts/results/all_tissues_variability_bins/Liver-Hepatocytes \
#      --chroms chr1,chr2,chr3,chr4,chr5,chr18,chr19,chr20,chr21,chr22 \
#      --update-existing
#
# 4. Pretrain-only prediction evaluation.
#    The first_phase_epoch_N groups come from *_pretrain_prediction directories.
#    python3 scripts/evaluate_grouped_predictions.py \
#      --mode variability_free \
#      --pretraining-group first_phase_epoch_3 \
#      --results-dir /sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer \
#      --tissue-name Liver-Hepatocytes \
#      --comparison-dicts-dir /sci/labs/michall/roeizucker/comparison_dicts/Liver-Hepatocytes \
#      --precomputed-scores-dir scripts/results/precomputed_scores/Liver-Hepatocytes \
#      --output-root scripts/results/all_tissues_grouped_evaluations/Liver-Hepatocytes \
#      --chroms chr1,chr2,chr3,chr4,chr5,chr18,chr19,chr20,chr21,chr22 \
#      --skip-mean-label \
#      --update-existing
#
# 5. Quick smoke test on one prediction file per sample.
#    python3 scripts/evaluate_grouped_predictions.py \
#      --mode variability_free \
#      --pretraining-group epoch_1_pretraining \
#      --results-dir /sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer \
#      --tissue-name Liver-Hepatocytes \
#      --comparison-dicts-dir /sci/labs/michall/roeizucker/comparison_dicts/Liver-Hepatocytes \
#      --precomputed-scores-dir scripts/results/precomputed_scores/Liver-Hepatocytes \
#      --output-root /tmp/evaluate_grouped_predictions_smoke \
#      --chroms chr1,chr2,chr3,chr4,chr5,chr18,chr19,chr20,chr21,chr22 \
#      --skip-mean-label \
#      --max-files-per-sample 1
#
# 6. Separate low-variability run analysis.
#    Use --low-var-only to evaluate only directories whose names contain _low_var.
#    python3 scripts/evaluate_grouped_predictions.py \
#      --mode variability_free \
#      --pretraining-group epoch_3_pretraining \
#      --results-dir /sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer \
#      --tissue-name Liver-Hepatocytes \
#      --comparison-dicts-dir /sci/labs/michall/roeizucker/comparison_dicts/Liver-Hepatocytes \
#      --output-root scripts/results/low_var_analysis/Liver-Hepatocytes \
#      --chroms chr1,chr2,chr3,chr4,chr5,chr18,chr19,chr20,chr21,chr22 \
#      --low-var-only \
#      --update-existing
#
# 7. Include low-variability run directories together with regular runs.
#    This is usually not wanted for final analyses, but is useful for diagnostics.
#    python3 scripts/evaluate_grouped_predictions.py \
#      --mode variability_bins \
#      --pretraining-group epoch_3_pretraining \
#      --results-dir /sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer \
#      --tissue-name Liver-Hepatocytes \
#      --comparison-dicts-dir /sci/labs/michall/roeizucker/comparison_dicts/Liver-Hepatocytes \
#      --mean-label-cache-dir scripts/results/mean_label_caches/Liver-Hepatocytes \
#      --output-root scripts/results/all_tissues_variability_bins/Liver-Hepatocytes \
#      --chroms chr1,chr2,chr3,chr4,chr5,chr18,chr19,chr20,chr21,chr22 \
#      --include-low-var-runs
