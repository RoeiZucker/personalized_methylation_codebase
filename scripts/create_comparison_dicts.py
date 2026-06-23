#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import csv
import pickle
import re
import sys

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from variability_free_evaluation import create_comparison_dicts


DEFAULT_DATASETS_DIR = Path("/sci/archive/michall/roeizucker/downloaded_datasets")
DEFAULT_RESULTS_DIRS = [
    Path("/sci/labs/michall/roeizucker/cls_downsample_liver"),
    Path("/sci/labs/michall/roeizucker/token_cls_models2/_token_cls_Liver-Hepatocytes_kmer"),
]
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "scripts" / "results" / "comparison_dicts"
DEFAULT_TISSUE_NAME = "Liver-Hepatocytes"
FULL_POS_NAME = "full_pos"
BIGWIG_SAMPLE_RE = re.compile(r"(?P<sample>Z\d+[A-Z0-9]*)\.hg38\.bigwig$")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        dest="results_dirs",
        default=None,
        help="Only needed if --chrom/--chroms is not provided; used to discover chromosomes from prediction CSVs.",
    )
    parser.add_argument("--tissue-name", default=DEFAULT_TISSUE_NAME)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument(
        "--bigwig-map-file",
        type=Path,
        default=None,
        help="Optional CSV/TSV with columns sample_name,bigwig_path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for {sample}_comparison_dicts.pkl. Defaults to scripts/results/comparison_dicts/<tissue-name>.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        dest="samples",
        default=None,
        help="Person/sample to build. Can be passed multiple times. Defaults to all samples with bigwigs.",
    )
    parser.add_argument(
        "--chrom",
        action="append",
        dest="chrom_list",
        default=None,
        help="Chromosome to include. Can be passed multiple times, e.g. --chrom chr1 --chrom chr2.",
    )
    parser.add_argument(
        "--chroms",
        default=None,
        help="Comma-separated chromosomes, e.g. chr1,chr2,chr3. If omitted, chromosomes are discovered from result files.",
    )
    parser.add_argument("--full-pos-name", default=FULL_POS_NAME)
    return parser.parse_args()


def sample_name_from_result_path(result_file_path):
    run_name = result_file_path.parents[1].name
    return run_name.split("_", 1)[0]


def sample_name_from_bigwig_path(path):
    match = BIGWIG_SAMPLE_RE.search(path.name)
    if match is None:
        return None
    return match.group("sample")


def read_bigwig_map_file(path):
    text = path.read_text().splitlines()
    if not text:
        return {}
    dialect = csv.Sniffer().sniff("\n".join(text[:5]), delimiters=",\t")
    rows = csv.DictReader(text, dialect=dialect)
    mapping = {}
    for row in rows:
        sample_name = row.get("sample_name") or row.get("sample")
        bigwig_path = row.get("bigwig_path") or row.get("path")
        if not sample_name or not bigwig_path:
            raise ValueError(
                f"Bigwig map rows must include sample_name,bigwig_path columns: {path}"
            )
        mapping[sample_name] = bigwig_path
    return mapping


def discover_bigwig_files_by_sample(tissue_name, datasets_dir):
    mapping = {}
    for bigwig_path in sorted(datasets_dir.glob(f"*{tissue_name}*.bigwig")):
        sample_name = sample_name_from_bigwig_path(bigwig_path)
        if sample_name is not None:
            mapping[sample_name] = str(bigwig_path)
    return mapping


def requested_chroms(args):
    chroms = []
    if args.chroms is not None:
        chroms.extend([chrom.strip() for chrom in args.chroms.split(",") if chrom.strip()])
    if args.chrom_list is not None:
        chroms.extend([chrom.strip() for chrom in args.chrom_list if chrom.strip()])
    if not chroms:
        return None
    return sort_chroms(set(chroms))


def chroms_from_result_file(result_file_path):
    window_ids = pd.read_csv(result_file_path, usecols=["window_id"])["window_id"]
    return set(window_ids.str.split(":", n=1).str[0].dropna().unique())


def discover_chroms_by_sample(results_dirs, bigwig_files_by_sample):
    chroms_by_sample = {}
    for results_dir in results_dirs:
        for result_file_path in sorted(results_dir.glob("*/epoch-*/eval_predictions.csv.gitbackup")):
            sample_name = sample_name_from_result_path(result_file_path)
            if sample_name not in bigwig_files_by_sample:
                continue
            chroms_by_sample.setdefault(sample_name, set()).update(
                chroms_from_result_file(result_file_path)
            )
    return chroms_by_sample


def sort_chroms(chroms):
    return sorted(chroms, key=lambda chrom: int(chrom[3:]) if chrom[3:].isdigit() else chrom)


def default_output_dir(tissue_name):
    safe_tissue_name = tissue_name.replace("/", "_").replace(" ", "_")
    return DEFAULT_OUTPUT_ROOT / safe_tissue_name


def main():
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(args.tissue_name)

    if args.bigwig_map_file is not None:
        bigwig_files_by_sample = read_bigwig_map_file(args.bigwig_map_file)
    else:
        bigwig_files_by_sample = discover_bigwig_files_by_sample(
            args.tissue_name,
            args.datasets_dir,
        )
    if not bigwig_files_by_sample:
        raise ValueError(
            f"No bigwig files found for tissue={args.tissue_name}. "
            "Pass --bigwig-map-file or check --datasets-dir."
        )

    explicit_chroms = requested_chroms(args)
    if explicit_chroms is None:
        results_dirs = args.results_dirs if args.results_dirs is not None else DEFAULT_RESULTS_DIRS
        chroms_by_sample = discover_chroms_by_sample(results_dirs, bigwig_files_by_sample)
        samples = args.samples if args.samples is not None else sorted(chroms_by_sample)
    else:
        chroms_by_sample = {}
        samples = args.samples if args.samples is not None else sorted(bigwig_files_by_sample)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"using {len(bigwig_files_by_sample)} bigwig files", flush=True)
    print(f"writing comparison dicts to {output_dir}", flush=True)

    for sample_name in samples:
        if sample_name not in bigwig_files_by_sample:
            raise ValueError(f"Sample {sample_name} does not have a bigwig path")
        if explicit_chroms is None:
            if sample_name not in chroms_by_sample:
                raise ValueError(
                    f"No chromosomes discovered for sample {sample_name}; pass --chroms to skip discovery"
                )
            chroms = sort_chroms(chroms_by_sample[sample_name])
        else:
            chroms = explicit_chroms

        comparison_bigwig_files = [
            bigwig_file
            for curr_sample, bigwig_file in bigwig_files_by_sample.items()
            if curr_sample != sample_name
        ]
        print(
            f"building comparison dicts for {sample_name}: "
            f"{len(chroms)} chromosomes, {len(comparison_bigwig_files)} bigwigs",
            flush=True,
        )
        compare_dicts = create_comparison_dicts(
            comparison_bigwig_files,
            chroms,
            args.full_pos_name,
        )
        output_path = output_dir / f"{sample_name}_comparison_dicts.pkl"
        payload = {
            "sample_name": sample_name,
            "tissue_name": args.tissue_name,
            "chroms": chroms,
            "comparison_bigwig_files": comparison_bigwig_files,
            "compare_dicts": compare_dicts,
        }
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
