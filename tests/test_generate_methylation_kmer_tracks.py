"""Tests for direct k-mer methylation browser tracks."""

import gzip
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_methylation_kmer_tracks import (
    DEFAULT_BIN_GROUPS,
    build_tracks,
    filter_by_participant_coverage,
    load_kmer_predictions,
    run_participant_jobs,
    status_values,
)


def prediction_frame(predictions, true_probabilities):
    labels = [1, 2]
    rows = []
    for token_position, (label, prediction, true_probability) in enumerate(
        zip(labels, predictions, true_probabilities)
    ):
        probabilities = [0.05, 0.05, 0.05]
        probabilities[label] = true_probability
        probabilities[(label + 1) % 3] = 1.0 - true_probability - 0.05
        rows.append(
            {
                "window_id": "chr1:100-112",
                "window_start": 100,
                "window_end": 112,
                "token_index": token_position + 1,
                "token_position_in_window": token_position,
                "base_position_in_window": token_position * 6,
                "genomic_position": 100 + token_position * 6,
                "label": label,
                "predicted_class": prediction,
                "probability_class_0": probabilities[0],
                "probability_class_1": probabilities[1],
                "probability_class_2": probabilities[2],
            }
        )
    return pd.DataFrame(rows)


def participant_row(
    tmp_path, participant, retrained, pretrained, retrained_prob, pretrained_prob,
    atlas=None,
    tissue="Heart-Cardiomyocyte",
):
    directory = tmp_path / participant
    directory.mkdir()
    retrained_path = directory / "retrained.csv"
    pretrained_path = directory / "pretrained.csv"
    variability_path = directory / "variability.csv"
    prediction_frame(retrained, retrained_prob).to_csv(retrained_path, index=False)
    prediction_frame(pretrained, pretrained_prob).to_csv(pretrained_path, index=False)
    pd.DataFrame(
        {
            "full_position": ["chr1:100-102", "chr1:106-108"],
            "std": [0.8, 1.0],
        }
    ).to_csv(variability_path, index=False)
    row = {
        "participant": participant,
        "tissue": tissue,
        "retrained_path": retrained_path,
        "pretrained_path": pretrained_path,
        "variability_path": variability_path,
    }
    if atlas is not None:
        mean_path = directory / "mean.csv"
        pd.DataFrame(
            {
                "window_id": ["chr1:100-112", "chr1:100-112"],
                "genomic_position": [100, 106],
                "label": [1, 2],
                "mean_label": atlas,
            }
        ).to_csv(mean_path, index=False)
        row["mean_path"] = mean_path
    return row


def test_builds_kmer_tracks_and_preserves_statuses(tmp_path):
    rows = [
        participant_row(tmp_path, "P1", [1, 1], [2, 2], [0.8, 0.2], [0.1, 0.8]),
        participant_row(tmp_path, "P2", [1, 2], [2, 1], [0.9, 0.9], [0.2, 0.2]),
    ]
    output = tmp_path / "output"
    build_tracks(
        pd.DataFrame(rows), output, chroms=["chr1"], bin_groups=["bins4-5"],
        minimum_match_rate=1.0, participant_tracks=True, ucsc_item_details=True,
    )

    summary = pd.read_csv(output / "kmer_summary.tsv.gz", sep="\t")
    assert list(summary["kmer_start"]) == [100, 106]
    assert list(summary["kmer_end"]) == [106, 112]
    assert list(summary["correctness_delta"]) == [1.0, 0.0]
    assert list(summary["n_rescued"]) == [2, 1]
    assert list(summary["n_harmed"]) == [0, 1]
    assert list(summary["n_individuals"]) == [2, 2]
    assert summary.loc[0, "true_probability_delta"] == pytest.approx(0.7)

    tracks = list((output / "tracks").glob("*"))
    assert len(tracks) == 3
    bed = next((output / "tracks").glob("*.status.bed"))
    bed_rows = [line.split("\t") for line in bed.read_text().splitlines()]
    assert all(len(row) == 9 for row in bed_rows)
    assert [row[3] for row in bed_rows] == ["n2_r2_h0", "n2_r1_h1"]
    assert [row[8] for row in bed_rows] == ["0,140,0", "128,128,128"]
    correctness = next((output / "tracks").glob("*.correctness_delta.bedGraph"))
    detail = next((output / "ucsc_status_tracks").glob("*.bedDetail.txt.gz"))
    with gzip.open(detail, "rt", encoding="utf-8") as handle:
        detail_lines = handle.read().splitlines()
    assert detail_lines[0].startswith("track type=bedDetail ")
    detail_rows = [line.split("\t") for line in detail_lines[1:]]
    assert all(len(row) == 11 for row in detail_rows)
    assert [row[3] for row in detail_rows] == ["n2_r2_h0", "n2_r1_h1"]
    assert detail_rows[1][10] == (
        "Same accuracy | Participants: 2; Rescued: 1; Harmed: 1; Both correct: 0; Both wrong: 0"
    )
    assert correctness.read_text().splitlines() == [
        "chr1\t100\t106\t1", "chr1\t106\t112\t0"
    ]
    participant_beds = list((output / "participant_tracks").glob("*.bed"))
    assert len(participant_beds) == 2
    assert "rescued" in participant_beds[0].read_text()

    ET.parse(output / "igv_session.xml")
    custom = (output / "ucsc_custom_tracks.txt").read_text().splitlines()
    assert any(line.startswith("track type=bed ") for line in custom)
    categorical = (output / "ucsc_categorical_tracks.txt").read_text().splitlines()
    assert categorical
    assert not any(line.startswith("track type=bedGraph ") for line in categorical)
    assert all(
        line.startswith("track type=bed ") or len(line.split("\t")) == 9
        for line in categorical
    )
    assert "window is context only" in (output / "REPORT.md").read_text()
    assert "names use `n#_r#_h#`" in (output / "REPORT.md").read_text()


def test_rejects_non_six_mer_coordinates(tmp_path):
    path = tmp_path / "bad.csv"
    frame = prediction_frame([1, 2], [0.8, 0.8])
    frame.loc[1, "base_position_in_window"] = 7
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="do not describe 6-mers"):
        load_kmer_predictions(path, 6)


def test_builds_atlas_only_tracks_with_explicit_names(tmp_path):
    row = participant_row(
        tmp_path, "P1", [1, 1], [2, 1], [0.8, 0.2], [0.1, 0.8],
        atlas=[2, 2],
    )
    output = tmp_path / "atlas_output"
    build_tracks(
        pd.DataFrame([row]), output, chroms=["chr1"], bin_groups=["bins4-5"],
        baselines=["atlas"], minimum_match_rate=1.0, participant_tracks=True,
    )

    summary = pd.read_csv(output / "participant_kmer_summary.tsv.gz", sep="\t")
    assert set(summary["comparison"]) == {"retrained_vs_atlas"}
    assert summary["status"].tolist() == ["rescued", "harmed"]
    categorical = (output / "ucsc_categorical_tracks.txt").read_text()
    assert "retrained_vs_atlas" in categorical
    assert "atlas_2_retrain_1" in categorical
    assert "_pre_" not in categorical


def test_builds_separate_tissues_and_combined_sample_tracks(tmp_path):
    rows = [
        participant_row(
            tmp_path, "P1", [1, 1], [2, 1], [0.8, 0.2], [0.1, 0.8],
            atlas=[2, 2], tissue="Heart-Cardiomyocyte",
        ),
        participant_row(
            tmp_path, "P2", [1, 2], [2, 1], [0.9, 0.9], [0.2, 0.2],
            atlas=[2, 1], tissue="Liver-Hepatocytes",
        ),
    ]
    output = tmp_path / "combined_output"
    build_tracks(
        pd.DataFrame(rows), output, chroms=["chr1"], bin_groups=["bins4-5"],
        baselines=["atlas"], combined_samples_track=True, workers=2,
    )

    summary = pd.read_csv(output / "kmer_summary.tsv.gz", sep="\t")
    assert set(summary["tissue"]) == {
        "Heart-Cardiomyocyte", "Liver-Hepatocytes", "All-samples"
    }
    combined = summary[summary["tissue"].eq("All-samples")]
    assert set(combined["n_individuals"]) == {2}
    headers = [
        line for line in (output / "ucsc_custom_tracks.txt").read_text().splitlines()
        if line.startswith("track ")
    ]
    assert any('name="All-samples.retrained_vs_atlas' in line for line in headers)


def test_default_groups_add_bins_one_and_two_without_new_combinations():
    assert DEFAULT_BIN_GROUPS == [
        "bin1", "bin2", "bin3", "bin4", "bin5", "bins3-5", "bins4-5"
    ]


def test_reports_sample_progress_and_preserves_manifest_order(capsys):
    manifest = pd.DataFrame(
        {"participant": ["P1", "P2"], "value": [1, 2]}
    )

    results = run_participant_jobs(
        manifest, lambda row: row["value"], workers=2
    )

    output = capsys.readouterr().out
    assert results == [1, 2]
    assert "Processing 2 samples with 2 worker(s)" in output
    assert "Samples completed: 2/2" in output


def test_filters_tissue_kmers_by_participant_coverage():
    summary = pd.DataFrame(
        {"kmer_start": [100, 106, 112], "n_individuals": [1, 3, 5]}
    )
    filtered = filter_by_participant_coverage(summary, 3)
    assert filtered["kmer_start"].tolist() == [106, 112]

    with pytest.raises(ValueError, match="must be positive"):
        filter_by_participant_coverage(summary, 0)
    with pytest.raises(ValueError, match="No k-mers"):
        filter_by_participant_coverage(summary, 6)


def test_status_categories_are_distinct():
    retrained = pd.Series([True, False, True, False])
    baseline = pd.Series([False, True, True, False])
    assert status_values(retrained, baseline).tolist() == [
        "rescued", "harmed", "both_correct", "both_wrong"
    ]
