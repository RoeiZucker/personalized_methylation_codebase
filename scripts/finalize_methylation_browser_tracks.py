#!/usr/bin/env python3
"""Finalize bounded browser tracks from completed methylation summaries."""

# Examples:
#   python3 scripts/finalize_methylation_browser_tracks.py \
#     scripts/results/methylation_browser_tracks_full
#   python3 scripts/finalize_methylation_browser_tracks.py \
#     scripts/results/methylation_browser_tracks_full --max-regions-per-track 500

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from generate_methylation_browser_tracks import (
    output_chromosomes,
    rank_candidates,
    report_text,
    write_bedgraphs,
    write_igv_session,
    write_ranked_regions,
    write_ucsc_custom_track,
)


FINAL_PATHS = [
    "candidate_regions_top.tsv",
    "candidate_regions_top.bed",
    "ucsc_custom_tracks.txt",
    "igv_session.xml",
    "REPORT.md",
]


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("summary_dir", type=Path)
    parser.add_argument("--max-regions-per-track", type=int, default=1000)
    parser.add_argument("--genome", default="hg38")
    parser.add_argument("--metric", choices=("accuracy", "mae"), default="accuracy")
    return parser.parse_args()


def read_summary(directory, name):
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"Missing completed summary: {path}")
    return pd.read_csv(path, sep="	")


def candidate_path(directory):
    candidates = [
        directory / "candidate_regions.tsv.gz",
        directory / "candidate_regions.full.tsv.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing full candidate-region audit in {directory}")


def ensure_fresh_finalization(directory):
    stale = [directory / name for name in FINAL_PATHS if (directory / name).exists()]
    tracks = directory / "tracks"
    if tracks.exists() and any(tracks.iterdir()):
        stale.append(tracks)
    if stale:
        names = ", ".join(str(path) for path in stale)
        raise FileExistsError(f"Refusing to overwrite final artifacts: {names}")


def finalize(directory, maximum, genome, metric):
    ensure_fresh_finalization(directory)
    windows = read_summary(directory, "window_summary.tsv.gz")
    participants = read_summary(directory, "participant_summary.tsv")
    claims = read_summary(directory, "aggregate_summary.tsv")
    read_summary(directory, "qc_summary.tsv")
    read_summary(directory, "selected_manifest.tsv")
    candidates = pd.read_csv(candidate_path(directory), sep="	")
    ranked = rank_candidates(candidates, maximum)
    bed_path = write_ranked_regions(ranked, directory)
    track_paths = write_bedgraphs(windows, directory)
    write_ucsc_custom_track(track_paths, directory, bed_path)
    write_igv_session([bed_path, *track_paths], directory, genome)
    has_mean = windows["comparison"].eq("retrained_minus_mean").any()
    report = report_text(
        claims, participants, ranked, metric, has_mean, maximum,
        output_chromosomes(windows),
    )
    (directory / "REPORT.md").write_text(report, encoding="utf-8")


def main():
    args = parse_args()
    finalize(args.summary_dir, args.max_regions_per_track, args.genome, args.metric)
    print(f"Finalized browser tracks in {args.summary_dir}")


if __name__ == "__main__":
    main()
