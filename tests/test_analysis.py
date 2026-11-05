from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from methylation.annotations import overlaps, bimodal_support
from methylation.baselines import methylation_class, local_class_average, reference_cpgs, attach_reference
from methylation.preprocessing import preprocess_encode_cpg_dataframe, create_kmer_split_train_test_labels
from methylation.statistics import weighted_f1, bootstrap, sign_test, bh_adjust
from methylation.analysis import run_report, donor_matching, chromosome_exclusions

DATA = Path(__file__).resolve().parents[1] / "data"


def test_thresholds_and_zero_measurements():
    assert methylation_class([0, .2, .20001, .79999, .8, 1]).tolist() == [0, 0, 1, 1, 2, 2]
    frame = preprocess_encode_cpg_dataframe(pd.DataFrame({"methyl_rate":[0., .1, -1.]}))
    assert len(frame) == 2 and frame.iloc[0].methyl_rate == 0


def test_label_holdout_is_disjoint_and_repeatable():
    labels = np.array([-100, .1, .2, .3, .4, .5])
    train, held = map(np.asarray, create_kmer_split_train_test_labels(labels, -100, .8, 42))
    assert (train != -100).sum() == 4 and (held != -100).sum() == 1
    assert not ((train != -100) & (held != -100)).any()
    np.testing.assert_array_equal(np.where(train != -100, train, held), labels)
    np.testing.assert_array_equal(train, create_kmer_split_train_test_labels(labels, -100, .8, 42)[0])


def test_known_scores_and_ties():
    assert weighted_f1([0, 1, 2], [0, 1, 2]) == 1
    assert weighted_f1([0, 1], [1, 0]) == 0
    assert sign_test([1] * 5 + [-1, 0, 0])["p_value"] == .21875
    assert sign_test([0, 0])["p_value"] == 1
    np.testing.assert_allclose(bh_adjust([.01, .04, .03]), [.03, .04, .04])


def test_local_average_and_nearest_fallback():
    train = pd.DataFrame({"donor":["A", "A"], "chrom":["chr1", "chr1"], "genomic_position":[0, 20], "label":[0, 2]})
    rows = pd.DataFrame({"donor":["A", "A"], "chrom":["chr1", "chr2"], "genomic_position":[10, 10]})
    assert local_class_average(rows, train, 20).tolist() == [1, 0]
    assert local_class_average(rows, train, 0).tolist() == [0, 0]
    with pytest.raises(ValueError, match="overlap"):
        local_class_average(train, train)


def test_reference_uses_common_coordinates_and_sample_sd():
    a = pd.DataFrame({"chrom":["chr1", "chr1"], "start":[1, 20], "end":[3, 22], "methyl_rate":[.1, .4]})
    b = pd.DataFrame({"chrom":["chr1"], "start":[1], "end":[3], "methyl_rate":[.9]})
    ref = reference_cpgs([a, b])
    assert len(ref) == 1 and ref.iloc[0]["mean"] == .5
    assert ref.iloc[0].sd == pytest.approx(np.std([.1, .9], ddof=1))


def test_frozen_bin_is_not_reconstructed():
    rows = pd.DataFrame({"donor":["A"], "chrom":["chr1"], "window_id":["chr1:0-5400"], "genomic_position":[0], "label":[1]})
    ref = pd.DataFrame({"chrom":["chr1"], "start":[1], "end":[3], "mean":[.8], "sd":[.4]})
    frozen = rows[["donor", "window_id", "genomic_position", "label"]].assign(bin=3)
    result = attach_reference(rows, ref, frozen)
    assert result.iloc[0]["bin"] == 3 and result.iloc[0].atlas == 2
    with pytest.raises(ValueError, match="truth"):
        attach_reference(rows, ref, frozen.assign(label=2))


def test_interval_overlap_and_reference_only_support():
    sites = pd.DataFrame({"donor":["A"], "chrom":["chr1"], "genomic_position":[10]})
    assert not overlaps(sites, pd.DataFrame({"chrom":["chr1"], "start":[16], "end":[18]}))[0]
    catalog = pd.DataFrame({"donor":["A", "B", "B"], "tissue":["T"]*3, "chrom":["chr1"]*3, "start":[10, 12, 13], "end":[20, 20, 20]})
    cohort = pd.DataFrame({"donor":["A", "B"], "tissue":["T", "T"]})
    assert bimodal_support(sites, catalog, cohort).bimodal_support.tolist() == [1]


@pytest.mark.skipif(not (DATA / "cohort.csv").exists(), reason="Study result files supplied separately")
def test_published_means_and_class_counts():
    cohort = pd.read_csv(DATA / "cohort.csv")
    assert len(cohort) == cohort.donor.nunique() == 32
    assert "Z000000R3" in set(cohort.donor)
    assert all(row.donor not in row.reference_donors.split(";") for row in cohort.itertuples())
    scores = pd.read_csv(DATA / "bimodality_donor_scores.csv")
    rows = scores[scores.endpoint.eq("bin_4") & scores["subset"].eq("all")]
    gain = rows.personalized_e3_weighted_f1 - rows.donor_excluded_e10_weighted_f1
    assert gain.mean() == pytest.approx(.0157, abs=5e-5)
    assert (gain > 0).sum() == 21
    inside = scores[scores.endpoint.eq("bin_3") & scores["subset"].eq("inside_bimodal")].sort_values("person")
    delta = inside.personalized_e3_weighted_f1 - inside.local_class_average_weighted_f1
    assert delta.mean() == pytest.approx(.022943, abs=1e-6)
    assert sign_test(delta)["positive"] == 27
    assert bootstrap(delta)["ci_low"] > 0


@pytest.mark.skipif(not (DATA / "chromosome_donor_scores.csv").exists(), reason="Study result files supplied separately")
def test_chromosome_exclusions_use_32_donors():
    scores = pd.read_csv(DATA / "chromosome_donor_scores.csv")
    result = chromosome_exclusions(scores)
    whole = result[result.exclusion.eq("none")]
    assert whole.donors.eq(32).all()
    assert whole.sort_values("bin").positive.tolist() == [29, 28]
    assert result.ci_low.gt(0).all()


def test_row_report_and_matching(tmp_path):
    rows = []
    for donor in ["A", "B"]:
        for bin_id in [3, 4]:
            for i in range(60):
                label = i % 3
                rows.append({"donor":donor, "tissue":"T", "chrom":"chr1", "window_id":f"chr1:{bin_id*5400}-{(bin_id+1)*5400}",
                    "genomic_position":bin_id*5400+i*6, "bin":bin_id, "label":label, "atlas":1,
                    "donor_excluded":(label+1)%3 if i<12 else label, "personalized":(label+1)%3 if i<3 else label,
                    "local_class_average":(label+1)%3 if i<6 else label, "annotation_cpg_island":True, "bimodal_support":i%3})
    rows = pd.DataFrame(rows)
    run_report(rows, tmp_path)
    assert (tmp_path / "support_contrasts.csv").exists()
    matching = []
    for row in rows.to_dict("records"):
        for model_donor in ["A", "B"]:
            matching.append({**row, "model_donor":model_donor, "stage":"personalized", "predicted_class":row["label"] if row["donor"] == model_donor else 1})
    donors, summary = donor_matching(pd.DataFrame(matching))
    assert donors.gain.gt(0).all() and summary.donors.eq(2).all()
