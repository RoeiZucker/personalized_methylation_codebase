#!/usr/bin/env python3
# Create leave-one-sample-out variability CSVs from methylation bigWig files.
# Example chr12, one tissue:
#   python3 scripts/create_chr12_variability_files.py \
#     --tissue-name Liver-Hepatocytes \
#     --chroms chr12 \
#     --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1
# Example chr12, selected samples:
#   python3 scripts/create_chr12_variability_files.py \
#     --tissue-name Liver-Hepatocytes --chroms chr12 \
#     --sample Z00000431 --sample Z0000044M \
#     --output-root /sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir_chr12_standardized_split_v1

from argparse import ArgumentParser
from pathlib import Path
import csv
import re
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.bigwig_utils import load_preprocessed_encode_cpg_dfs
from utils.formatting import (
    QUANTILE_SEPERATION_TYPE,
    STD_VARIABILITY_TYPE,
    combine_cpg_dfs,
    create_methyl_variability_df,
)


DEFAULT_DATASETS_DIR = Path("/sci/archive/michall/roeizucker/downloaded_datasets")
DEFAULT_OUTPUT_ROOT = Path(
    "/sci/labs/michall/roeizucker/huggingface_datasets_dir/"
    "huggingface_datasets_dir_chr12_standardized_split_v1"
)
BIGWIG_SAMPLE_RE = re.compile(r"(?P<sample>Z\d+[A-Z0-9]*)\.hg38\.bigwig$")
FULL_POS_NAME = "full_position"
SEQ_SIZE = 5400


def parse_args():
    parser = ArgumentParser(description="Create per-variant variability CSVs for held-out samples.")
    parser.add_argument("--tissue-name", required=True)
    parser.add_argument("--chroms", default="chr12", help="Comma-separated chromosomes, e.g. chr12.")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bigwig-map-file", type=Path, default=None)
    parser.add_argument("--sample", action="append", dest="samples", default=None)
    parser.add_argument("--seq-size", type=int, default=SEQ_SIZE)
    parser.add_argument("--high-diff-quantile", type=float, default=0.5)
    return parser.parse_args()


def parse_chroms(chroms_text):
    chroms = [chrom.strip() for chrom in chroms_text.split(",") if chrom.strip()]
    if not chroms:
        raise ValueError("--chroms must include at least one chromosome")
    return chroms


def read_bigwig_map_file(path):
    text = path.read_text(encoding="utf-8").splitlines()
    if not text:
        return {}
    dialect = csv.Sniffer().sniff("\n".join(text[:5]), delimiters=",\t")
    rows = csv.DictReader(text, dialect=dialect)
    return {sample_name(row): bigwig_path(row) for row in rows}


def sample_name(row):
    value = row.get("sample_name") or row.get("sample")
    if not value:
        raise ValueError("BigWig map rows must include sample_name or sample")
    return value


def bigwig_path(row):
    value = row.get("bigwig_path") or row.get("path")
    if not value:
        raise ValueError("BigWig map rows must include bigwig_path or path")
    return value


def discover_bigwigs(tissue_name, datasets_dir):
    mapping = {}
    for bigwig_path in sorted(datasets_dir.glob(f"*{tissue_name}*.bigwig")):
        sample = sample_from_bigwig(bigwig_path)
        if sample is not None:
            mapping[sample] = str(bigwig_path)
    return mapping


def sample_from_bigwig(path):
    match = BIGWIG_SAMPLE_RE.search(path.name)
    if match is None:
        return None
    return match.group("sample")


def output_dir(output_root, tissue_name):
    path = output_root / f"_{tissue_name}_kmer"
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(output_root, tissue_name, sample):
    return output_dir(output_root, tissue_name) / (
        f"{sample}_per_varaint_variability_{tissue_name}_kmer_seq_5400_datasets.csv"
    )


def comparison_bigwigs(bigwigs_by_sample, held_out_sample):
    return [
        path
        for sample, path in sorted(bigwigs_by_sample.items())
        if sample != held_out_sample
    ]


def create_variability(comparison_paths, chroms, seq_size, high_diff_quantile):
    labels = [f"ind_{index}" for index in range(len(comparison_paths))]
    cpg_dfs = load_preprocessed_encode_cpg_dfs(comparison_paths, chroms, FULL_POS_NAME, False)
    combined = combine_cpg_dfs(FULL_POS_NAME, cpg_dfs, labels)
    variability = create_methyl_variability_df(
        len(cpg_dfs),
        combined,
        STD_VARIABILITY_TYPE,
        QUANTILE_SEPERATION_TYPE,
        high_diff_quantile,
        seq_size,
    )
    return variability[[FULL_POS_NAME, "window_id", "std", "high_diff"]]


def main():
    args = parse_args()
    chroms = parse_chroms(args.chroms)
    if args.bigwig_map_file is None:
        bigwigs_by_sample = discover_bigwigs(args.tissue_name, args.datasets_dir)
    else:
        bigwigs_by_sample = read_bigwig_map_file(args.bigwig_map_file)
    if not bigwigs_by_sample:
        raise ValueError(f"No bigWig files found for {args.tissue_name}")

    samples = args.samples or sorted(bigwigs_by_sample)
    for sample in samples:
        if sample not in bigwigs_by_sample:
            raise ValueError(f"Sample {sample} does not have a bigWig file")
        comparison_paths = comparison_bigwigs(bigwigs_by_sample, sample)
        print(
            f"creating variability for {args.tissue_name} {sample}: "
            f"{len(comparison_paths)} comparison bigWigs, chroms={chroms}",
            flush=True,
        )
        variability = create_variability(
            comparison_paths,
            chroms,
            args.seq_size,
            args.high_diff_quantile,
        )
        destination = output_path(args.output_root, args.tissue_name, sample)
        variability.to_csv(destination)
        print(f"wrote {destination} ({len(variability):,} rows)", flush=True)


if __name__ == "__main__":
    main()
