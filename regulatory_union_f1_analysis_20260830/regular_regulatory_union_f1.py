#!/usr/bin/env python3
"""Analyze one broad regulatory union on ordinary/regular TEST predictions.

Run examples:
  # One-participant smoke test
  python3 regular_regulatory_union_f1.py --participant Z000000QM --output-dir smoke_outputs

  # Full 31-participant run
  python3 regular_regulatory_union_f1.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd


REF_ROOT = Path(
    "/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/"
    "refactored_code"
)
BASE_ANALYSIS = REF_ROOT / "cpg_island_f1_analysis_20260828"
OTHER_ANALYSIS = REF_ROOT / "other_annotation_f1_analysis_20260830"
sys.path[:0] = [str(BASE_ANALYSIS), str(OTHER_ANALYSIS)]

from analyze_cpg_island_f1 import (  # noqa: E402
    DEFAULT_SOURCE_ROOT,
    DTYPES,
    ISLAND_RELATIVE,
    MANIFEST_RELATIVE,
    READ_COLUMNS,
    assert_classes,
    assert_geometry,
    assert_pair,
    build_tasks,
    confusion,
    load_regular_manifest,
    self_check_weighted_f1,
    weighted_f1,
)
from other_annotation_f1_analysis import (  # noqa: E402
    CCRE_RELATIVE,
    GTF_RELATIVE,
    close_writer,
    filtered_frame,
    load_ccres,
    load_gene_annotations,
    merge_frame,
    open_writer,
    overlap_mask,
    summarize_overall,
    summarize_tissues,
)


FILTER = "regulatory_union"
RULE = (
    "overlaps a GENCODE v50 promoter (2 kb upstream to 500 bp downstream of a gene TSS) "
    "or any SCREEN Registry v4 cCRE; each token is retained once"
)
CCRE_BASE_CLASSES = (
    "ccre_pls",
    "ccre_pels",
    "ccre_dels",
    "ccre_ca_ctcf",
    "ccre_ca_h3k4me3",
    "ccre_ca_tf",
    "ccre_ca",
    "ccre_tf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--participant")
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


def regulatory_intervals(source_root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    promoters, _ = load_gene_annotations(source_root / GTF_RELATIVE)
    ccres = load_ccres(source_root / CCRE_RELATIVE)
    rows = []
    for intervals in [promoters, *(ccres[name] for name in CCRE_BASE_CLASSES)]:
        for chrom, (starts, ends) in intervals.items():
            rows.extend(zip(np.repeat(chrom, len(starts)), starts, ends))
    return merge_frame(pd.DataFrame(rows, columns=["chrom", "start", "end"]))


def matrices() -> dict[str, dict[str, np.ndarray]]:
    return {
        name: {role: np.zeros((3, 3), dtype=np.int64) for role in ("e10", "e3")}
        for name in ("unfiltered", FILTER)
    }


def add(matrix, labels, e10, e3, mask) -> None:
    matrix["e10"] += confusion(labels[mask], e10[mask])
    matrix["e3"] += confusion(labels[mask], e3[mask])


def result_row(task, matrix, retained: int, total: int) -> dict[str, object]:
    unfiltered_e10 = weighted_f1(matrix["unfiltered"]["e10"])
    unfiltered_e3 = weighted_f1(matrix["unfiltered"]["e3"])
    filtered_e10 = weighted_f1(matrix[FILTER]["e10"])
    filtered_e3 = weighted_f1(matrix[FILTER]["e3"])
    unfiltered_delta = unfiltered_e3 - unfiltered_e10
    filtered_delta = filtered_e3 - filtered_e10
    return {
        "filter": FILTER,
        "participant": task.participant,
        "tissue": task.tissue,
        "paired_tokens": total,
        "retained_tokens": retained,
        "excluded_tokens": total - retained,
        "retained_fraction": retained / total,
        "unfiltered_weighted_f1_e10": unfiltered_e10,
        "unfiltered_weighted_f1_e3": unfiltered_e3,
        "unfiltered_delta_e3_minus_e10": unfiltered_delta,
        "filtered_weighted_f1_e10": filtered_e10,
        "filtered_weighted_f1_e3": filtered_e3,
        "filtered_delta_e3_minus_e10": filtered_delta,
        "filtered_minus_unfiltered_delta": filtered_delta - unfiltered_delta,
    }


def analyze(task, intervals, writer, write_header: bool) -> tuple[dict[str, object], bool]:
    matrix = matrices()
    retained = 0
    total = 0
    pre_reader = pd.read_csv(task.pretrain_path, usecols=READ_COLUMNS, dtype=DTYPES, chunksize=task.chunksize)
    e3_reader = pd.read_csv(task.personalized_path, usecols=READ_COLUMNS, dtype=DTYPES, chunksize=task.chunksize)
    for index, (pretrain, personalized) in enumerate(zip_longest(pre_reader, e3_reader)):
        context = f"{task.participant} chunk {index}"
        if pretrain is None or personalized is None:
            raise ValueError(f"{context}: E10/E3 chunk counts differ")
        assert_pair(pretrain, personalized, context)
        assert_geometry(pretrain, context)
        labels = assert_classes(pretrain, context)
        assert_classes(personalized, context)
        e10 = pretrain["predicted_class"].to_numpy(np.int64)
        e3 = personalized["predicted_class"].to_numpy(np.int64)
        all_rows = np.ones(len(pretrain), dtype=bool)
        selected = overlap_mask(
            pretrain["window_id"].str.split(":", n=1).str[0].to_numpy(dtype=str),
            pretrain["genomic_position"].to_numpy(np.int64),
            intervals,
        )
        add(matrix["unfiltered"], labels, e10, e3, all_rows)
        add(matrix[FILTER], labels, e10, e3, selected)
        output = filtered_frame(task.participant, task.tissue, pretrain, personalized, selected)
        output.to_csv(writer[2], index=False, header=write_header)
        write_header = False
        retained += len(output)
        total += len(pretrain)
    if retained == 0:
        raise ValueError(f"{task.participant}: regulatory union retained no rows")
    return result_row(task, matrix, retained, total), write_header


def report_text(row) -> str:
    return f"""# Broad-regulatory-union ordinary-TEST weighted-F1 analysis

## Result

- Retained tokens: **{int(row.retained_tokens):,}**.
- Filtered personalization gain (`F1_E3 - F1_E10`): **{row.filtered_delta_e3_minus_e10:+.8f}**.
- Unfiltered gain: **{row.unfiltered_delta_e3_minus_e10:+.8f}**.
- Filtered-minus-unfiltered gain: **{row.filtered_minus_unfiltered_delta:+.8f}**.
- Donor directions for that gain difference: {int(row.gain_difference_positive_donors)}/{int(row.gain_difference_negative_donors)}/{int(row.gain_difference_zero_donors)} positive/negative/zero; nominal unadjusted sign `p={row.gain_difference_nominal_sign_p:.8g}`.

## Filter and calculation

The broad regulatory union {RULE}. Gene bodies and CpG-context annotations are excluded from this union. The union is boolean, so a token overlapping multiple annotations appears once.

Every hg38 prediction token is the BED0 half-open 6-base interval `[genomic_position, genomic_position+6)`. At least one base of overlap is required. E10 and E3 ordinary/regular TEST rows were paired exactly, and sklearn-compatible K=3 class-support-weighted F1 was calculated per participant. Overall results average the 31 participants equally.

This is an exploratory annotation analysis. The nominal donor sign test is descriptive and unadjusted; tokens are not independent biological replicates. Filtering changes class balance, so the personalization-gain comparison is more informative than comparing absolute filtered and unfiltered F1. All pairing, geometry, and class QC passed. Source data and prior outputs were not modified.
"""


def write_metadata(output_dir: Path, per_person: pd.DataFrame) -> None:
    dataset = output_dir / "datasets" / "regulatory_union_filtered_tokens.csv.gz"
    metadata = {
        "scope": "ordinary_regular_TEST_only",
        "participants": int(per_person["participant"].nunique()),
        "source_data_modified": False,
        "coordinate_system": "hg38 BED0 half-open; prediction tokens are 6 bases",
        "filter": FILTER,
        "rule": RULE,
        "dataset": str(dataset),
        "rows": int(per_person["retained_tokens"].sum()),
        "bytes": dataset.stat().st_size,
        "all_pairing_geometry_class_qc_pass": True,
    }
    (output_dir / "METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.chunksize < 1:
        raise ValueError("chunksize must be positive")
    self_check_weighted_f1()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "datasets").mkdir()
    source_root = args.source_root.resolve()
    manifest = load_regular_manifest(source_root / MANIFEST_RELATIVE, args.participant)
    manifest.to_csv(output_dir / "selected_regular_exports.tsv", sep="\t", index=False)
    tasks = build_tasks(manifest, source_root / ISLAND_RELATIVE, args.chunksize)
    intervals = regulatory_intervals(source_root)
    writer = open_writer(output_dir / "datasets" / "regulatory_union_filtered_tokens.csv.gz")
    rows = []
    header = True
    try:
        for task in tasks:
            row, header = analyze(task, intervals, writer, header)
            rows.append(row)
            print(f"complete {task.participant}", flush=True)
    finally:
        close_writer(writer)
    per_person = pd.DataFrame(rows).sort_values("participant")
    tissues = summarize_tissues(per_person)
    overall = summarize_overall(per_person, tissues)
    per_person.to_csv(output_dir / "per_person_comparison.csv", index=False)
    tissues.to_csv(output_dir / "tissue_comparison.csv", index=False)
    overall.to_csv(output_dir / "overall_comparison.csv", index=False)
    donor = overall.loc[overall["summary_weighting"].eq("equal_donor")].iloc[0]
    (output_dir / "REPORT.md").write_text(report_text(donor), encoding="utf-8")
    write_metadata(output_dir, per_person)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
