#!/usr/bin/env python3
"""Validate the file contract of a methylation k-mer browser-track bundle.

Examples:
  # Validate a combined chr12 atlas bundle:
  python3 scripts/validate_methylation_kmer_track_bundle.py \
    /path/to/chr12_atlas_bundle --baseline atlas \
    --expected-participants 31 --expected-split-rows 62 --chrom chr12

  # Validate a combined chr12 pretrained bundle:
  python3 scripts/validate_methylation_kmer_track_bundle.py \
    /path/to/chr12_pretrained_bundle --baseline pretrained \
    --expected-participants 31 --expected-split-rows 62 --chrom chr12
"""

from __future__ import annotations

import argparse
import csv
import gzip
from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


VIEWS = {"bin1", "bin2", "bin3", "bin4", "bin5", "bins3-5", "bins4-5"}
COMPARISONS = {
    "atlas": "retrained_vs_atlas",
    "pretrained": "retrained_minus_pretrained",
}
TOP_LEVEL_FILES = {
    "REPORT.md",
    "browser_kmer_summary.tsv.gz",
    "igv_session.xml",
    "kmer_summary.tsv.gz",
    "qc_summary.tsv",
    "selected_manifest.tsv",
    "ucsc_categorical_tracks.txt",
    "ucsc_custom_tracks.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--baseline", choices=tuple(COMPARISONS), required=True)
    parser.add_argument("--expected-participants", type=int, required=True)
    parser.add_argument("--expected-split-rows", type=int, required=True)
    parser.add_argument("--chrom", required=True)
    return parser.parse_args()


def require_nonempty_files(root: Path, names: set[str]) -> None:
    missing = [name for name in sorted(names) if not (root / name).is_file()]
    empty = [name for name in sorted(names) if (root / name).is_file() and not (root / name).stat().st_size]
    if missing or empty:
        raise ValueError(f"Missing files: {missing}; empty files: {empty}")


def validate_manifest(root: Path, expected_participants: int, expected_rows: int) -> set[str]:
    manifest = pd.read_csv(root / "selected_manifest.tsv", sep="\t")
    if len(manifest) != expected_rows:
        raise ValueError(f"Expected {expected_rows} manifest rows, found {len(manifest)}")
    if manifest["participant"].nunique() != expected_participants:
        raise ValueError("Unexpected participant count in selected_manifest.tsv")
    split_counts = manifest.groupby("participant")["split"].nunique()
    if set(manifest["split"]) != {"eval", "test"} or not split_counts.eq(2).all():
        raise ValueError("Every participant must have one eval and one test input")
    return set(manifest["tissue"]) | {"All-samples"}


def validate_qc(root: Path, expected_rows: int) -> None:
    qc = pd.read_csv(root / "qc_summary.tsv", sep="\t")
    if len(qc) != expected_rows:
        raise ValueError(f"Expected {expected_rows} QC rows, found {len(qc)}")
    if set(qc["split"]) != {"eval", "test"}:
        raise ValueError("QC output does not contain both eval and test")
    if (qc["variability_match_rate"] < 0.5).any():
        raise ValueError("A variability match rate is below 0.5")


def validate_gzip(path: Path) -> None:
    with gzip.open(path, "rb") as handle:
        while handle.read(8 * 1024 * 1024):
            pass


def validate_summary_chrom(path: Path, expected: str) -> None:
    observed = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            observed.add(row["chrom"])
    if observed != {expected}:
        raise ValueError(f"Expected only {expected} in browser summary, found {observed}")


def plain_row_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            count += chunk.count(b"\n")
    return count


def detail_row_count(path: Path) -> int:
    with gzip.open(path, "rb") as handle:
        header = handle.readline()
        first = handle.readline()
        if not header.startswith(b"track type=bedDetail "):
            raise ValueError(f"Invalid bedDetail header in {path}")
        if len(first.rstrip(b"\n").split(b"\t")) != 11:
            raise ValueError(f"Invalid bedDetail columns in {path}")
        count = 1
        while chunk := handle.read(8 * 1024 * 1024):
            count += chunk.count(b"\n")
    return count


def track_map(directory: Path, suffix: str, comparison: str) -> dict[tuple[str, str], Path]:
    tracks = {}
    token = f".{comparison}."
    for path in directory.glob(f"*{suffix}"):
        stem = path.name.removesuffix(suffix)
        if token not in stem:
            continue
        tissue, view = stem.split(token, maxsplit=1)
        tracks[(tissue, view)] = path
    return tracks


def expected_track_keys(tissues: set[str]) -> set[tuple[str, str]]:
    return set(product(tissues, VIEWS))


def require_track_matrix(
    tracks: dict[tuple[str, str], Path], expected: set[tuple[str, str]], label: str
) -> None:
    if set(tracks) != expected:
        missing = sorted(expected.difference(tracks))
        extra = sorted(set(tracks).difference(expected))
        raise ValueError(f"Invalid {label} matrix; missing={missing}, extra={extra}")


def validate_track_rows(
    statuses: dict[tuple[str, str], Path],
    correctness: dict[tuple[str, str], Path],
    details: dict[tuple[str, str], Path],
    probabilities: dict[tuple[str, str], Path],
) -> None:
    for key, status_path in statuses.items():
        rows = plain_row_count(status_path)
        if rows <= 0:
            raise ValueError(f"Empty status track: {status_path}")
        if plain_row_count(correctness[key]) != rows:
            raise ValueError(f"Correctness/status row mismatch for {key}")
        if detail_row_count(details[key]) != rows:
            raise ValueError(f"bedDetail/status row mismatch for {key}")
        if probabilities and plain_row_count(probabilities[key]) != rows:
            raise ValueError(f"Probability/status row mismatch for {key}")


def validate_bundle(args: argparse.Namespace) -> None:
    root = args.output_dir
    require_nonempty_files(root, TOP_LEVEL_FILES)
    tissues = validate_manifest(root, args.expected_participants, args.expected_split_rows)
    validate_qc(root, args.expected_split_rows)
    validate_gzip(root / "kmer_summary.tsv.gz")
    validate_gzip(root / "browser_kmer_summary.tsv.gz")
    ET.parse(root / "igv_session.xml")

    comparison = COMPARISONS[args.baseline]
    expected = expected_track_keys(tissues)
    statuses = track_map(root / "tracks", ".status.bed", comparison)
    correctness = track_map(root / "tracks", ".correctness_delta.bedGraph", comparison)
    details = track_map(root / "ucsc_status_tracks", ".status.bedDetail.txt.gz", comparison)
    probabilities = track_map(root / "tracks", ".true_probability_delta.bedGraph", comparison)
    require_track_matrix(statuses, expected, "status")
    require_track_matrix(correctness, expected, "correctness")
    require_track_matrix(details, expected, "bedDetail")
    if args.baseline == "pretrained":
        require_track_matrix(probabilities, expected, "probability")
    elif probabilities:
        raise ValueError("Atlas bundle unexpectedly contains probability-delta tracks")
    validate_track_rows(statuses, correctness, details, probabilities)

    validate_summary_chrom(root / "browser_kmer_summary.tsv.gz", args.chrom)
    print(
        f"Validated {args.baseline} bundle: {len(statuses)} status, "
        f"{len(details)} bedDetail, {len(correctness)} correctness, "
        f"{len(probabilities)} probability tracks",
        flush=True,
    )


def main() -> None:
    validate_bundle(parse_args())


if __name__ == "__main__":
    main()
