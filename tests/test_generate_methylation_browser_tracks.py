"""Tests for methylation browser-track generation."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_methylation_browser_tracks import (
    assign_bins,
    baseline_values,
    benjamini_hochberg,
    build_tracks,
    comparison_rows,
    discover_manifest,
    discover_result_row,
    disjoint_segments,
    load_predictions,
    prepare_output_dir,
    register_participant_windows,
    qc_row,
    rank_candidates,
    sign_flip_p,
    write_ranked_regions,
)


def prediction_frame(predictions):
    return pd.DataFrame(
        {
            "window_id": ["chr1:100-200", "chr1:200-300", "chr1:200-300"],
            "window_start": [100, 200, 200],
            "window_end": [200, 300, 300],
            "genomic_position": [110, 210, 220],
            "label": [1, 1, 1],
            "predicted_class": predictions,
            "unused_large_column": ["x", "x", "x"],
        }
    )


def write_participant(tmp_path, participant, retrained, pretrained):
    directory = tmp_path / participant
    directory.mkdir()
    retrained_path = directory / "retrained.csv"
    pretrained_path = directory / "pretrained.csv"
    variability_path = directory / "variability.csv"
    mean_path = directory / "mean.pkl"
    prediction_frame(retrained).to_csv(retrained_path, index=False)
    prediction_frame(pretrained).to_csv(pretrained_path, index=False)
    pd.DataFrame(
        {
            "full_position": ["chr1:110-112", "chr1:210-212", "chr1:220-222"],
            "std": [0.1, 0.4, 0.5],
            "unused": [1, 2, 3],
        }
    ).to_csv(variability_path, index=False)
    prediction_frame([0, 0, 0]).iloc[:2][
        ["window_id", "genomic_position", "label"]
    ].assign(mean_label=0).to_pickle(mean_path)
    return {
        "participant": participant,
        "tissue": "Heart-Cardiomyocyte",
        "retrained_path": retrained_path,
        "pretrained_path": pretrained_path,
        "variability_path": variability_path,
        "mean_path": mean_path,
        "split": "test",
        "selected_checkpoint": 30,
        "selection_rule": "largest_numeric_checkpoint",
    }


def test_builds_browser_tracks_and_allows_partial_mean_cache(tmp_path):
    rows = [
        write_participant(tmp_path, "P1", [1, 1, 1], [0, 0, 0]),
        write_participant(tmp_path, "P2", [0, 1, 0], [1, 0, 1]),
    ]
    output = tmp_path / "output"
    build_tracks(
        pd.DataFrame(rows), output, "accuracy", "predicted_class", "hg38",
        None, ["bin1", "bin4", "bin5", "bins4-5"], maximum_regions=1,
    )

    assert (output / "candidate_regions_top.bed").exists()
    assert not (output / "windows.bed").exists()
    custom_tracks = (output / "ucsc_custom_tracks.txt").read_text()
    assert custom_tracks.startswith("track type=bed")
    assert "itemRgb=On" in custom_tracks
    first_ucsc_bed_row = custom_tracks.splitlines()[1]
    assert len(first_ucsc_bed_row.split("\t")) == 9
    session = (output / "igv_session.xml").read_text()
    assert '<Session genome="hg38"' in session
    assert 'path="candidate_regions_top.bed"' in session
    assert (output / "candidate_regions.tsv.gz").exists()
    assert (output / "window_summary.tsv.gz").exists()
    assert (output / "qc_summary.tsv").exists()
    selected = pd.read_csv(output / "selected_manifest.tsv", sep="\t")
    assert set(selected["selected_checkpoint"]) == {30}
    windows = pd.read_csv(output / "window_summary.tsv.gz", sep="\t")
    mean_rows = windows[windows["comparison"] == "retrained_minus_mean"]
    pretrain_rows = windows[windows["comparison"] == "retrained_minus_pretrained"]
    assert mean_rows["n_sites"].max() < pretrain_rows["n_sites"].max()
    assert set(windows["bin_group"]) == {"bin1", "bin4", "bin5", "bins4-5"}
    top = pd.read_csv(output / "candidate_regions_top.tsv", sep="\t")
    group_keys = ["tissue", "bin_group", "comparison"]
    assert top.groupby(group_keys).size().max() == 1
    assert set(top["rank_within_track"]) == {1}
    report = (output / "REPORT.md").read_text()
    assert "Top bins4-5 pretrained-comparison candidates" in report
    expected_headings = [
        "p",
        "q",
        "Individuals",
        "Participant-windows",
        "Sites",
        "Favors retrained",
        "Favors baseline",
        "Ties",
        "Direction",
        "Evidence",
    ]
    assert all(heading in report for heading in expected_headings)
    candidate = top[
        top["bin_group"].eq("bins4-5")
        & top["comparison"].eq("retrained_minus_pretrained")
    ].iloc[0]
    consistency = (
        f"| {candidate.n_individuals} | {candidate.n_observations} | "
        f"{candidate.n_sites} | {candidate.n_favors_retrained} | "
        f"{candidate.n_favors_baseline} | {candidate.n_tie} |"
    )
    assert consistency in report
    assert "candidate_regions.tsv.gz" in report
    assert "Chromosome scope represented in this output: `chr1`." in report
    validate_bedgraphs(output / "tracks")


def validate_bedgraphs(directory):
    for path in directory.glob("*.bedGraph"):
        previous = {}
        for line in path.read_text().splitlines():
            chrom, start, end, value = line.split("\t")
            assert int(start) < int(end)
            assert int(start) >= previous.get(chrom, -1)
            previous[chrom] = int(end)
            float(value)


def test_nonfinite_mean_baseline_is_rejected(tmp_path):
    frame = pd.DataFrame(
        {
            "window_id": ["chr1:100-200"],
            "genomic_position": [110],
            "mean_label": [float("nan")],
        }
    )
    with pytest.raises(ValueError, match="non-finite mean_label"):
        baseline_values(frame, "predicted_class", tmp_path / "mean.pkl")


def test_prediction_coordinate_mismatch_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    frame = prediction_frame([1, 1, 1])
    frame.loc[0, "window_start"] = 101
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagree"):
        load_predictions(path, "predicted_class")


def test_discovery_selects_largest_numeric_checkpoint(tmp_path):
    result_dir = tmp_path / "P1"
    result_dir.mkdir()
    paths = {
        "/root/_token_cls_Liver-Hepatocytes_kmer/run/pretrain_epoch10/test_predictions.csv": {},
        "/root/_token_cls_Liver-Hepatocytes_kmer/run/checkpoint-30/test_predictions.csv": {},
        "/root/_token_cls_Liver-Hepatocytes_kmer/run/checkpoint-10/test_predictions.csv": {},
        "/root/_token_cls_Liver-Hepatocytes_kmer/run/checkpoint-20/test_predictions.csv": {},
    }
    eval_path = result_dir / "test_eval_objects_dict.txt"
    eval_path.write_text(repr({"P1": paths}))
    row = discover_result_row(eval_path, "test")
    assert row["selected_checkpoint"] == 30
    assert "checkpoint-30" in str(row["retrained_path"])
    assert row["selection_rule"] == "largest_numeric_checkpoint"


def test_combined_discovery_loads_eval_and_test_for_each_participant(tmp_path):
    result_dir = tmp_path / "P1"
    result_dir.mkdir()
    for split in ("eval", "test"):
        paths = {
            f"/root/_token_cls_Liver-Hepatocytes_kmer/run/pretrain_epoch10/{split}_predictions.csv": {},
            f"/root/_token_cls_Liver-Hepatocytes_kmer/run/checkpoint-30/{split}_predictions.csv": {},
        }
        (result_dir / f"{split}_eval_objects_dict.txt").write_text(
            repr({"P1": paths})
        )

    variability_root = tmp_path / "chr12_variability"
    manifest = discover_manifest(tmp_path, "combined", variability_root)

    assert manifest["split"].tolist() == ["eval", "test"]
    assert manifest["participant"].tolist() == ["P1", "P1"]
    assert all(manifest["selected_checkpoint"] == 30)
    assert manifest.iloc[0]["mean_path"].name.startswith("eval_")
    assert manifest.iloc[1]["mean_path"].name.startswith("test_")
    assert all(
        str(path).startswith(str(variability_root))
        for path in manifest["variability_path"]
    )


def test_combined_inputs_must_have_disjoint_windows():
    seen = {}
    metadata = pd.Series({"participant": "P1", "tissue": "T"})
    register_participant_windows(
        pd.DataFrame({"window_id": ["chr12:0-10"]}), metadata, seen
    )
    register_participant_windows(
        pd.DataFrame({"window_id": ["chr12:10-20"]}), metadata, seen
    )
    with pytest.raises(ValueError, match="overlap"):
        register_participant_windows(
            pd.DataFrame({"window_id": ["chr12:0-10"]}), metadata, seen
        )


def test_exact_sign_flip_test_uses_participants_as_replicates():
    assert sign_flip_p(pd.Series([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.125)
    assert sign_flip_p(pd.Series([1.0, -1.0])) == pytest.approx(1.0)


def test_score_sign_bin_boundaries_and_bh():
    frame = pd.DataFrame(
        {
            "label": [1, 1],
            "retrained_prediction": [1, 0],
            "pretrained_prediction": [0, 1],
            "mean_prediction": [0, 1],
        }
    )
    rows = comparison_rows(frame, "accuracy")
    pretrained = rows[rows["comparison"] == "retrained_minus_pretrained"]
    assert pretrained["improvement"].tolist() == [1.0, -1.0]

    values = pd.Series([0.0, 0.2, 0.20001, 0.4, 0.8, 1.0, float("nan")])
    assert assign_bins(values, 1.0).tolist()[:6] == [1, 1, 2, 2, 4, 5]
    adjusted = benjamini_hochberg(
        pd.Series([0.01, 0.04, 0.03, float("nan")])
    )
    assert adjusted.iloc[:3].tolist() == pytest.approx([0.03, 0.04, 0.04])
    assert pd.isna(adjusted.iloc[3])


def test_overlap_segments_and_qc_threshold():
    windows = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "window_start": [0, 5],
            "window_end": [10, 15],
            "improvement": [1.0, -1.0],
        }
    )
    assert disjoint_segments(windows) == [
        ("chr1", 0, 5, 1.0),
        ("chr1", 5, 10, 0.0),
        ("chr1", 10, 15, -1.0),
    ]
    sites = pd.DataFrame(
        {
            "window_id": ["chr1:0-10", "chr1:10-20"],
            "variability_bin": [1, pd.NA],
            "mean_prediction": [1, pd.NA],
        }
    )
    metadata = pd.Series({"participant": "P1", "tissue": "T"})
    assert qc_row(sites, metadata, 0.5)["variability_match_rate"] == 0.5
    with pytest.raises(ValueError, match="below"):
        qc_row(sites, metadata, 0.51)


def test_ranked_bed_is_genomically_sorted_but_tsv_keeps_rank_order(tmp_path):
    frame = pd.DataFrame(
        {
            "tissue": ["T", "T"],
            "chrom": ["chr2", "chr1"],
            "window_start": [200, 100],
            "window_end": [300, 200],
            "bin_group": ["bins4-5", "bins4-5"],
            "comparison": [
                "retrained_minus_pretrained",
                "retrained_minus_pretrained",
            ],
            "improvement": [0.2, 0.1],
            "standard_error": [0.01, 0.02],
            "p_value": [0.1, 0.2],
            "q_value": [0.2, 0.3],
            "n_individuals": [4, 4],
            "n_observations": [4, 4],
            "n_sites": [40, 30],
        }
    )
    write_ranked_regions(frame, tmp_path)

    table = pd.read_csv(tmp_path / "candidate_regions_top.tsv", sep="\t")
    assert table["chrom"].tolist() == ["chr2", "chr1"]
    bed_rows = [
        line.split("\t")[:3]
        for line in (tmp_path / "candidate_regions_top.bed").read_text().splitlines()
    ]
    assert bed_rows == [["chr1", "100", "200"], ["chr2", "200", "300"]]


def test_candidate_cap_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        rank_candidates(pd.DataFrame(), 0)


def test_nonfinite_predictions_and_stale_output_are_rejected(tmp_path):
    prediction_path = tmp_path / "nonfinite.csv"
    prediction_frame([1, float("nan"), 1]).to_csv(
        prediction_path, index=False
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_predictions(prediction_path, "predicted_class")

    output = tmp_path / "existing"
    prepare_output_dir(output)
    (output / "stale.txt").write_text("stale")
    with pytest.raises(FileExistsError, match="Refusing"):
        prepare_output_dir(output)
