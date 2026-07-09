from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_context_factors import (  # noqa: E402
    add_context_factors,
    build_interval_index,
    interval_overlap_flags,
    metrics_for_rows,
    read_bed,
    run,
)


def write_bed(path, rows):
    path.write_text(
        "\n".join("\t".join(map(str, row)) for row in rows) + "\n",
        encoding="utf-8",
    )


def prediction_rows(base_positions, window_ids=None, labels=None, predicted=None):
    if window_ids is None:
        window_ids = [f"chr1:{index * 5400}-{(index + 1) * 5400}" for index in range(len(base_positions))]
    if labels is None:
        labels = [0] * len(base_positions)
    if predicted is None:
        predicted = labels
    return pd.DataFrame(
        {
            "window_id": window_ids,
            "window_start": [0] * len(base_positions),
            "window_end": [5400] * len(base_positions),
            "base_position_in_window": base_positions,
            "genomic_position": [100 + index * 10 for index in range(len(base_positions))],
            "label": labels,
            "predicted_class": predicted,
            "chrom": ["chr1"] * len(base_positions),
        }
    )


def test_cpg_interval_overlap_flags_use_half_open_bed_coordinates(tmp_path):
    bed_path = tmp_path / "regions.bed"
    write_bed(bed_path, [("chr1", 100, 102), ("chr1", 200, 205)])
    index = build_interval_index(read_bed(bed_path))

    flags = interval_overlap_flags(
        ["chr1", "chr1", "chr1", "chr2"],
        [98, 102, 204, 100],
        [100, 104, 206, 102],
        index,
    )

    assert flags.tolist() == [False, False, True, False]


def test_context_factor_boundaries_and_overlap_labels(tmp_path):
    gene_bed = tmp_path / "genes.bed"
    island_bed = tmp_path / "islands.bed"
    write_bed(gene_bed, [("chr1", 100, 112)])
    write_bed(island_bed, [("chr1", 130, 132)])
    prediction = prediction_rows([1079, 1080, 4319, 4320])

    annotated = add_context_factors(
        prediction,
        build_interval_index(read_bed(gene_bed)),
        build_interval_index(read_bed(island_bed)),
    )

    assert annotated["window_position"].tolist() == ["edge", "middle", "middle", "edge"]
    assert annotated["gene_overlap"].tolist() == ["in_gene", "in_gene", "not_in_gene", "not_in_gene"]
    assert annotated["cpg_island_overlap"].tolist() == [
        "not_in_cpg_island",
        "not_in_cpg_island",
        "not_in_cpg_island",
        "in_cpg_island",
    ]


@pytest.mark.parametrize(
    ("count", "expected"),
    [(6, "low"), (7, "medium"), (12, "medium"), (13, "high")],
)
def test_window_cpg_count_bins_at_boundaries(tmp_path, count, expected):
    gene_bed = tmp_path / "genes.bed"
    island_bed = tmp_path / "islands.bed"
    write_bed(gene_bed, [("chr1", 0, 1)])
    write_bed(island_bed, [("chr1", 0, 1)])
    prediction = prediction_rows(
        [100] * count,
        window_ids=["chr1:0-5400"] * count,
        labels=[0] * count,
        predicted=[0] * count,
    )

    annotated = add_context_factors(
        prediction,
        build_interval_index(read_bed(gene_bed)),
        build_interval_index(read_bed(island_bed)),
    )

    assert annotated["window_cpg_count"].unique().tolist() == [count]
    assert annotated["window_cpg_count_bin"].unique().tolist() == [expected]


def test_weighted_metrics_match_expected_one_vs_all_values():
    rows = prediction_rows(
        [100, 200, 300, 400],
        labels=[0, 0, 1, 2],
        predicted=[0, 1, 1, 2],
    )

    metrics = {row["class"]: row for row in metrics_for_rows(rows)}

    assert metrics["0"]["precision"] == pytest.approx(1.0)
    assert metrics["0"]["recall"] == pytest.approx(0.5)
    assert metrics["1"]["precision"] == pytest.approx(0.5)
    assert metrics["1"]["recall"] == pytest.approx(1.0)
    assert metrics["2"]["precision"] == pytest.approx(1.0)
    assert metrics["2"]["recall"] == pytest.approx(1.0)
    expected_weighted_f1 = ((2 * (2 / 3)) + (1 * (2 / 3)) + (1 * 1.0)) / 4
    assert metrics["weighted"]["f1"] == pytest.approx(expected_weighted_f1)

def test_run_writes_csvs_and_plots_for_synthetic_prediction_tree(tmp_path):
    results_dir = tmp_path / "results"
    prediction_dir = (
        results_dir
        / "S1_epoch-10-step-100_retrain_Test-Tissue_kmer_lr_1e-06_bs_14_seq_5400"
        / "epoch-2-step-20"
    )
    prediction_dir.mkdir(parents=True)
    prediction_path = prediction_dir / "eval_predictions.csv.gitbackup"
    prediction_rows(
        [100, 1200, 2500, 5000],
        window_ids=["chr1:0-5400", "chr1:0-5400", "chr1:5400-10800", "chr1:5400-10800"],
        labels=[0, 1, 2, 2],
        predicted=[0, 1, 1, 2],
    ).drop(columns=["chrom"]).to_csv(prediction_path, index=False)
    gene_bed = tmp_path / "genes.bed"
    island_bed = tmp_path / "islands.bed"
    write_bed(gene_bed, [("chr1", 100, 102), ("chr1", 7900, 7902)])
    write_bed(island_bed, [("chr1", 6600, 6602)])
    output_dir = tmp_path / "out"

    class Args:
        pass

    args = Args()
    args.results_dir = results_dir
    args.output_dir = output_dir
    args.tissue_name = "Test-Tissue"
    args.gene_bed = gene_bed
    args.cpg_island_bed = island_bed
    args.pretraining_group = "epoch_10_pretraining"
    args.groups = None
    args.batch_size = "14"
    args.final_only = True
    args.max_files = None
    args.dpi = 80

    run(args)

    per_file = pd.read_csv(output_dir / "context_factor_per_file_metrics.csv")
    summary = pd.read_csv(output_dir / "context_factor_metrics.csv")
    assert not per_file.empty
    assert not summary.empty
    assert (output_dir / "context_factors_weighted_f1.png").exists()
    weighted_position = per_file[
        (per_file["factor"] == "window_position") & (per_file["class"] == "weighted")
    ]
    assert weighted_position["support"].sum() == 4

