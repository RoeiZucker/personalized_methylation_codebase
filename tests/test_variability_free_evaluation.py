import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import evaluator as variability_bins_evaluator  # noqa: E402

from variability_free_evaluation import (  # noqa: E402
    apply_create_labels,
    apply_create_means,
    create_eval_object,
    create_mean_labels,
    create_mean_values,
    create_result_file_mean_label,
    create_result_file_mean_value,
    evaluate_sample_predictions,
    read_result_file_for_comparison_types,
)


def legacy_create_mean_labels(mean_values, ranges):
    df = pd.DataFrame({"mean_value": mean_values})
    return df.apply(lambda row: apply_create_labels(row, ranges), axis=1)


def legacy_apply_create_means(row, compare_dicts):
    values = []
    genomic_position = row["genomic_position"]
    chrom = row["chrom"]
    for i in range(genomic_position - 1, genomic_position + 6):
        if i in compare_dicts[chrom]:
            temp_values = []
            for key in compare_dicts[chrom][i]:
                if "methyl_rate_ind" in key:
                    temp_values.append(compare_dicts[chrom][i][key])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                values.append(np.mean(temp_values))
    if len(values) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.mean(values)


def assert_labels_match_legacy(mean_values, ranges):
    expected = legacy_create_mean_labels(mean_values, ranges)
    actual = create_mean_labels(mean_values, ranges)
    pd.testing.assert_series_equal(actual, expected, check_dtype=False, check_names=False)


def legacy_create_mean_values(result_file, compare_dicts):
    return result_file.apply(lambda row: apply_create_means(row, compare_dicts), axis=1)


def assert_mean_values_match_legacy(result_file, compare_dicts):
    expected = legacy_create_mean_values(result_file, compare_dicts)
    actual = create_mean_values(result_file, compare_dicts)
    pd.testing.assert_series_equal(actual, expected, check_dtype=False, check_names=False)


def legacy_create_eval_object(new_result_file, comparison_types, labels):
    from sklearn.metrics import precision_score, recall_score

    eval_object = {}
    for prediction_type in comparison_types:
        eval_object[prediction_type] = {}
        eval_object[prediction_type + "_confusion_matrix"] = pd.crosstab(
            new_result_file["label"], new_result_file[prediction_type]
        ).to_dict()
        for label in labels:
            df = new_result_file.copy()
            df["specific_label"] = df["label"] == label
            df["specific_type_label"] = df[prediction_type] == label
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                precision = precision_score(df["specific_label"], df["specific_type_label"])
                recall = recall_score(df["specific_label"], df["specific_type_label"])
            eval_object[prediction_type][label] = {
                "Precision": precision,
                "recall": recall,
            }
    return eval_object


def test_create_mean_labels_matches_legacy_boundaries_and_out_of_range_values():
    ranges = [0, 0.2, 0.8, 1]
    mean_values = pd.Series(
        [-0.1, 0, 0.199999, 0.2, 0.200001, 0.799999, 0.8, 0.800001, 1, 1.1, np.nan]
    )

    assert_labels_match_legacy(mean_values, ranges)

    actual = create_mean_labels(mean_values, ranges)
    assert np.isnan(actual.iloc[0])
    assert actual.iloc[1:9].tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    assert np.isnan(actual.iloc[9])
    assert np.isnan(actual.iloc[10])


def test_create_mean_labels_intentionally_leaves_out_of_range_values_missing():
    labels = create_mean_labels(pd.Series([-0.01, 0.5, 1.01]), [0, 0.2, 0.8, 1])

    assert np.isnan(labels.iloc[0])
    assert labels.iloc[1] == 1
    assert np.isnan(labels.iloc[2])


def test_create_mean_labels_assigns_threshold_values_to_lower_bins():
    labels = create_mean_labels(pd.Series([0.2, 0.8]), [0, 0.2, 0.8, 1])

    assert labels.tolist() == [0, 1]


def test_read_result_file_for_comparison_types_loads_only_needed_columns(tmp_path):
    result_file_path = tmp_path / "eval_predictions_extra_columns.csv"
    pd.DataFrame(
        {
            "label": [0, 1, 2],
            "predicted_class": [0, 2, 2],
            "window_id": ["chr1:10-12", "chr1:20-22", "chr1:30-32"],
            "large_unused_column": ["unused", "unused", "unused"],
        }
    ).to_csv(result_file_path, index=False)

    result = read_result_file_for_comparison_types(
        str(result_file_path),
        ["predicted_class"],
    )

    assert result.columns.tolist() == ["label", "predicted_class"]
    assert result["label"].dtype.kind in {"i", "u"}


def test_create_mean_labels_matches_legacy_on_generated_values():
    ranges = [0, 0.2, 0.8, 1]
    rng = np.random.default_rng(20260623)
    generated_values = pd.Series(rng.uniform(-0.25, 1.25, size=1000))
    generated_values = pd.concat(
        [generated_values, pd.Series([0, 0.2, 0.8, 1, np.nan])],
        ignore_index=True,
    )

    assert_labels_match_legacy(generated_values, ranges)


def test_create_result_file_mean_label_preserves_legacy_labels_with_realistic_generated_data(tmp_path):
    result_file_path = tmp_path / "eval_predictions.csv"
    pd.DataFrame(
        {
            "window_id": ["chr1:10-12", "chr1:20-22", "chr1:30-32", "chr1:40-42"],
            "genomic_position": [10, 20, 30, 40],
            "label": [0, 1, 2, 1],
            "predicted_class": [0, 1, 2, 2],
        }
    ).to_csv(result_file_path, index=False)

    compare_dicts = {"chr1": {}}
    for genomic_position, methyl_rate in [(10, 0.2), (20, 0.5), (30, 0.8)]:
        for position in range(genomic_position - 1, genomic_position + 6):
            compare_dicts["chr1"][position] = {
                "methyl_rate_ind_0": methyl_rate,
                "methyl_rate_ind_1": methyl_rate,
            }

    labeled = create_result_file_mean_label(str(result_file_path), compare_dicts, [0, 0.2, 0.8, 1])
    expected_labels = legacy_create_mean_labels(labeled["mean_value"], [0, 0.2, 0.8, 1])

    assert labeled["genomic_position"].tolist() == [10, 20, 30]
    np.testing.assert_allclose(labeled["mean_value"].to_numpy(), [0.2, 0.5, 0.8])
    pd.testing.assert_series_equal(
        labeled["mean_label"].reset_index(drop=True),
        expected_labels.reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )
    assert labeled["mean_label"].tolist() == [0, 1, 1]


def test_create_result_file_mean_label_keeps_rows_with_unrelated_missing_values(tmp_path):
    result_file_path = tmp_path / "eval_predictions_with_optional_missing.csv"
    pd.DataFrame(
        {
            "window_id": ["chr1:10-12", "chr1:20-22"],
            "genomic_position": [10, 20],
            "label": [0, 1],
            "predicted_class": [0, 1],
            "optional_metadata": [np.nan, "present"],
        }
    ).to_csv(result_file_path, index=False)

    compare_dicts = {"chr1": {}}
    for genomic_position, methyl_rate in [(10, 0.1), (20, 0.5)]:
        for position in range(genomic_position - 1, genomic_position + 6):
            compare_dicts["chr1"][position] = {
                "methyl_rate_ind_0": methyl_rate,
                "methyl_rate_ind_1": methyl_rate,
            }

    labeled = create_result_file_mean_label(str(result_file_path), compare_dicts, [0, 0.2, 0.8, 1])

    assert labeled["genomic_position"].tolist() == [10, 20]
    assert labeled["mean_label"].tolist() == [0, 1]
    assert labeled["optional_metadata"].isna().tolist() == [True, False]


def test_apply_create_means_matches_legacy_missing_data_behavior_on_simulated_rows():
    rows = [
        pd.Series({"chrom": "chr1", "genomic_position": 10}),
        pd.Series({"chrom": "chr1", "genomic_position": 20}),
        pd.Series({"chrom": "chr1", "genomic_position": 30}),
        pd.Series({"chrom": "chr1", "genomic_position": 40}),
    ]
    compare_dicts = {
        "chr1": {
            9: {"methyl_rate_ind_0": 0.2, "methyl_rate_ind_1": 0.4},
            19: {"coverage_ind_0": 15},
            29: {"methyl_rate_ind_0": np.nan, "methyl_rate_ind_1": 0.8},
        }
    }

    for row in rows:
        expected = legacy_apply_create_means(row, compare_dicts)
        actual = apply_create_means(row, compare_dicts)
        if expected is None:
            assert actual is None
        elif np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == expected


def test_create_result_file_mean_label_drops_rows_with_missing_mean_from_empty_methyl_values(tmp_path):
    result_file_path = tmp_path / "eval_predictions_missing_methyl.csv"
    pd.DataFrame(
        {
            "window_id": ["chr1:10-12", "chr1:20-22"],
            "genomic_position": [10, 20],
            "label": [0, 1],
            "predicted_class": [0, 1],
        }
    ).to_csv(result_file_path, index=False)
    compare_dicts = {
        "chr1": {
            9: {"methyl_rate_ind_0": 0.1, "methyl_rate_ind_1": 0.1},
            19: {"coverage_ind_0": 12},
        }
    }

    labeled = create_result_file_mean_label(str(result_file_path), compare_dicts, [0, 0.2, 0.8, 1])

    assert labeled["genomic_position"].tolist() == [10]
    assert labeled["mean_label"].tolist() == [0]


def test_create_result_file_mean_label_with_checked_in_csv_file_io(tmp_path):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    real_csv_path = os.path.join(repo_root, "notebooks", "kol_kore_visualize", "file_index.csv")
    real_rows = pd.read_csv(real_csv_path).head(3)
    result_file_path = tmp_path / "eval_predictions_from_checked_in_csv.csv"
    pd.DataFrame(
        {
            "window_id": [f"chr1:{100 + i * 10}-{102 + i * 10}" for i in range(len(real_rows))],
            "genomic_position": [100 + i * 10 for i in range(len(real_rows))],
            "label": [i % 3 for i in range(len(real_rows))],
            "predicted_class": [i % 3 for i in range(len(real_rows))],
            "source_file_name": real_rows["file_name"].tolist(),
        }
    ).to_csv(result_file_path, index=False)
    compare_dicts = {"chr1": {}}
    for genomic_position, methyl_rate in zip([100, 110, 120], [0.1, 0.5, 0.9]):
        for position in range(genomic_position - 1, genomic_position + 6):
            compare_dicts["chr1"][position] = {
                "methyl_rate_ind_0": methyl_rate,
                "methyl_rate_ind_1": methyl_rate,
            }

    labeled = create_result_file_mean_label(str(result_file_path), compare_dicts, [0, 0.2, 0.8, 1])

    assert labeled["source_file_name"].tolist() == real_rows["file_name"].tolist()
    assert labeled["mean_label"].tolist() == [0, 1, 2]


def test_create_mean_values_matches_rowwise_apply_on_generated_sparse_comparison_dicts():
    rng = np.random.default_rng(20260624)
    result_file = pd.DataFrame(
        {
            "window_id": [
                f"{chrom}:{position}-{position + 2}"
                for chrom in ["chr1", "chr2"]
                for position in range(100, 190, 10)
            ],
            "chrom": [chrom for chrom in ["chr1", "chr2"] for _ in range(100, 190, 10)],
            "genomic_position": [position for _ in ["chr1", "chr2"] for position in range(100, 190, 10)],
            "label": [i % 3 for i in range(18)],
            "predicted_class": [(i + 1) % 3 for i in range(18)],
        }
    )
    compare_dicts = {"chr1": {}, "chr2": {}}
    for chrom in compare_dicts:
        for position in range(95, 195):
            if rng.random() < 0.45:
                values_by_key = {"coverage_ind_0": int(rng.integers(1, 100))}
                if rng.random() < 0.85:
                    values_by_key["methyl_rate_ind_0"] = float(rng.uniform(0, 1))
                    values_by_key["methyl_rate_ind_1"] = float(rng.uniform(0, 1))
                if rng.random() < 0.10:
                    values_by_key["methyl_rate_ind_nan"] = np.nan
                compare_dicts[chrom][position] = values_by_key

    assert_mean_values_match_legacy(result_file, compare_dicts)


def test_create_mean_values_preserves_nan_and_no_match_behavior_from_rowwise_apply():
    result_file = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr2"],
            "genomic_position": [10, 20, 30, 10],
            "label": [0, 1, 2, 0],
            "predicted_class": [0, 1, 2, 0],
        }
    )
    compare_dicts = {
        "chr1": {
            9: {"methyl_rate_ind_0": 0.2, "methyl_rate_ind_1": 0.4},
            19: {"coverage_ind_0": 15},
            29: {"methyl_rate_ind_0": np.nan, "methyl_rate_ind_1": 0.8},
        },
        "chr2": {},
    }

    expected = legacy_create_mean_values(result_file, compare_dicts)
    actual = create_mean_values(result_file, compare_dicts)

    pd.testing.assert_series_equal(actual, expected, check_dtype=False, check_names=False)
    assert actual.iloc[0] == 0.30000000000000004
    assert np.isnan(actual.iloc[1])
    assert np.isnan(actual.iloc[2])
    assert np.isnan(actual.iloc[3])


def test_create_result_file_mean_value_matches_rowwise_apply_from_csv(tmp_path):
    result_file_path = tmp_path / "eval_predictions_for_mean_value.csv"
    raw_result_file = pd.DataFrame(
        {
            "window_id": ["chr1:10-12", "chr1:20-22", "chr2:10-12", "chr2:30-32"],
            "genomic_position": [10, 20, 10, 30],
            "label": [0, 1, 2, 1],
            "predicted_class": [0, 2, 2, 1],
        }
    )
    raw_result_file.to_csv(result_file_path, index=False)
    compare_dicts = {
        "chr1": {
            9: {"methyl_rate_ind_0": 0.1, "methyl_rate_ind_1": 0.3},
            10: {"methyl_rate_ind_0": 0.2, "methyl_rate_ind_1": 0.4},
            19: {"coverage_ind_0": 10},
        },
        "chr2": {
            9: {"methyl_rate_ind_0": 0.7, "methyl_rate_ind_1": 0.9},
        },
    }

    result_file = create_result_file_mean_value(str(result_file_path), compare_dicts)
    expected = legacy_create_mean_values(result_file.drop(columns=["mean_value"]), compare_dicts)

    pd.testing.assert_series_equal(result_file["mean_value"], expected, check_dtype=False, check_names=False)


def test_create_eval_object_matches_legacy_copy_heavy_implementation():
    new_result_file = pd.DataFrame(
        {
            "label": [0, 0, 1, 1, 2, 2, 2],
            "mean_label": [0, 1, 1, 2, 2, 0, 2],
            "predicted_class": [0, 0, 1, 1, 1, 1, 1],
            "all_two": [2, 2, 2, 2, 2, 2, 2],
        }
    )
    comparison_types = ["mean_label", "predicted_class", "all_two"]
    labels = [0, 1, 2]

    expected = legacy_create_eval_object(new_result_file, comparison_types, labels)
    actual = create_eval_object(new_result_file, comparison_types, labels)

    assert actual == expected


def test_evaluate_sample_predictions_accepts_corrected_and_legacy_keyword_names(tmp_path):
    result_file_path = tmp_path / "eval_predictions.csv"
    pd.DataFrame(
        {
            "window_id": ["chr1:10-12", "chr1:20-22", "chr1:30-32"],
            "genomic_position": [10, 20, 30],
            "label": [0, 1, 2],
            "predicted_class": [0, 1, 2],
        }
    ).to_csv(result_file_path, index=False)
    compare_dicts = {"chr1": {}}
    for genomic_position, methyl_rate in [(10, 0.1), (20, 0.5), (30, 0.9)]:
        for position in range(genomic_position - 1, genomic_position + 6):
            compare_dicts["chr1"][position] = {
                "methyl_rate_ind_0": methyl_rate,
                "methyl_rate_ind_1": methyl_rate,
            }

    corrected = evaluate_sample_predictions(
        result_files_path=[str(result_file_path)],
        chroms=["chr1"],
        comparison_bigwig_files=[],
        full_pos_name="full_pos",
        ranges=[0, 0.2, 0.8, 1],
        labels=[0, 1, 2],
        comparison_types=["mean_label", "predicted_class"],
        all_two=True,
        verbose=False,
        comparison_dicts=compare_dicts,
    )
    legacy = evaluate_sample_predictions(
        result_files_path=[str(result_file_path)],
        chroms=["chr1"],
        comparison_bigiwg_files=[],
        full_pos_name="full_pos",
        ranges=[0, 0.2, 0.8, 1],
        labels=[0, 1, 2],
        comparison_types=["mean_label", "predicted_class"],
        all_two=True,
        verbous=False,
        comparison_dicts=compare_dicts,
    )

    assert corrected == legacy
    all_results = corrected[str(result_file_path)]["all_results"]
    assert sorted(all_results) == [
        "all_two",
        "all_two_confusion_matrix",
        "mean_label",
        "mean_label_confusion_matrix",
        "predicted_class",
        "predicted_class_confusion_matrix",
    ]


def test_variability_bins_eval_object_matches_legacy_copy_heavy_implementation():
    new_result_file = pd.DataFrame(
        {
            "label": [0, 0, 1, 1, 2, 2, 2],
            "mean_label": [0, 1, 1, 2, 2, 0, 2],
            "predicted_class": [0, 0, 1, 1, 1, 1, 1],
        }
    )

    expected = legacy_create_eval_object(
        new_result_file,
        ["mean_label", "predicted_class"],
        [0, 1, 2],
    )
    actual = variability_bins_evaluator.create_eval_object(
        new_result_file,
        "mean_label",
        "predicted_class",
        [0, 1, 2],
    )

    assert actual == expected



def test_evaluate_sample_predictions_can_skip_mean_label_with_precomputed_scores(tmp_path):
    result_file_path = tmp_path / "eval_predictions.csv"
    pd.DataFrame(
        {
            "window_id": ["chr1:10-12", "chr1:20-22", "chr1:30-32"],
            "genomic_position": [10, 20, 30],
            "label": [0, 1, 2],
            "predicted_class": [0, 2, 2],
        }
    ).to_csv(result_file_path, index=False)
    precomputed_scores = {
        "mean_label": {
            0: {"Precision": 1.0, "recall": 1.0},
            1: {"Precision": 0.5, "recall": 0.5},
            2: {"Precision": 0.25, "recall": 0.25},
        },
        "mean_label_confusion_matrix": {0: {0: 1}, 1: {1: 1}, 2: {2: 1}},
        "all_two": {
            0: {"Precision": 0.0, "recall": 0.0},
            1: {"Precision": 0.0, "recall": 0.0},
            2: {"Precision": 1 / 3, "recall": 1.0},
        },
        "all_two_confusion_matrix": {2: {0: 1, 1: 1, 2: 1}},
    }

    result = evaluate_sample_predictions(
        result_files_path=[str(result_file_path)],
        chroms=["chr1"],
        comparison_bigwig_files=None,
        full_pos_name="full_pos",
        ranges=[0, 0.2, 0.8, 1],
        labels=[0, 1, 2],
        comparison_types=["mean_label", "predicted_class"],
        all_two=True,
        skip_mean_label=True,
        precomputed_scores=precomputed_scores,
    )

    eval_object = result[str(result_file_path)]["all_results"]
    assert eval_object["mean_label"] == precomputed_scores["mean_label"]
    assert eval_object["all_two"] == precomputed_scores["all_two"]
    assert eval_object["predicted_class"][0]["Precision"] == 1.0
    assert eval_object["predicted_class"][1]["recall"] == 0.0
    assert eval_object["predicted_class"][2]["recall"] == 1.0


def write_variability_bins_result_file(path, predictions):
    pd.DataFrame(
        {
            "window_id": ["chr1:10-12", "chr1:20-22", "chr1:30-32", "chr1:40-42"],
            "genomic_position": [10, 20, 30, 40],
            "label": [0, 1, 2, 1],
            "predicted_class": predictions,
        }
    ).to_csv(path, index=False)


def create_variability_bins_compare_dicts():
    compare_dicts = {"chr1": {}}
    for genomic_position, methyl_rate in [(10, 0.1), (20, 0.5), (30, 0.9), (40, 0.3)]:
        for position in range(genomic_position - 1, genomic_position + 6):
            compare_dicts["chr1"][position] = {
                "methyl_rate_ind_0": methyl_rate,
                "methyl_rate_ind_1": methyl_rate,
            }
    return compare_dicts


def write_variability_file(path):
    pd.DataFrame(
        {
            "full_position": ["chr1:10-12", "chr1:20-22", "chr1:30-32", "chr1:40-42"],
            "std": [0.1, 0.2, 0.7, 0.9],
        }
    ).to_csv(path, index=False)


def create_legacy_variability_bins_expected(result_files_path, variability_file_path, compare_dicts):
    variability_file = pd.read_csv(variability_file_path)
    expected = {}
    std_max = variability_file["std"].max()
    for result_file_path in result_files_path:
        result_file_path = str(result_file_path)
        curr_result_eval = {}
        expected[result_file_path] = curr_result_eval
        new_result_file = variability_bins_evaluator.create_result_file_mean_label(
            result_file_path,
            compare_dicts,
            [0, 0.2, 0.8, 1],
        )
        new_result_file["full_position"] = (
            new_result_file["chrom"]
            + ":"
            + new_result_file["genomic_position"].astype(str)
            + "-"
            + (new_result_file["genomic_position"] + 2).astype(str)
        )
        curr_result_eval["all_results"] = variability_bins_evaluator.create_eval_object(
            new_result_file,
            "mean_label",
            "predicted_class",
            [0, 1, 2],
        )
        for i in range(2):
            low = (std_max / 2) * i
            high = (std_max / 2) * (i + 1)
            filtered = variability_file[(variability_file["std"] > low) & (variability_file["std"] <= high)]
            filtered_data = new_result_file[new_result_file["full_position"].isin(set(filtered["full_position"]))]
            curr_result_eval[f"{low}-{high}"] = variability_bins_evaluator.create_eval_object(
                filtered_data,
                "mean_label",
                "predicted_class",
                [0, 1, 2],
            )
    return expected


def test_variability_bins_evaluation_cache_matches_legacy_per_file_mean_label_calculation(tmp_path):
    first_result_file = tmp_path / "eval_predictions_first.csv"
    second_result_file = tmp_path / "eval_predictions_second.csv"
    variability_file_path = tmp_path / "variability.csv"
    cache_path = tmp_path / "cache" / "sample_mean_label_cache.pkl"
    write_variability_bins_result_file(first_result_file, [0, 1, 2, 2])
    write_variability_bins_result_file(second_result_file, [2, 1, 0, 1])
    write_variability_file(variability_file_path)
    compare_dicts = create_variability_bins_compare_dicts()
    result_files_path = [str(first_result_file), str(second_result_file)]

    expected = create_legacy_variability_bins_expected(
        result_files_path,
        str(variability_file_path),
        compare_dicts,
    )
    actual = variability_bins_evaluator.evaluate_sample_predictions(
        variability_file_path=str(variability_file_path),
        result_files_path=result_files_path,
        chroms=["chr1"],
        comparison_bigwig_files=None,
        full_pos_name="full_pos",
        ranges=[0, 0.2, 0.8, 1],
        labels=[0, 1, 2],
        label_a="mean_label",
        label_b="predicted_class",
        number_of_bins=2,
        comparison_dicts=compare_dicts,
        mean_label_cache_path=cache_path,
    )

    assert actual == expected
    assert cache_path.exists()
    cached = pd.read_pickle(cache_path)
    assert cached["mean_label"].tolist() == [0, 1, 2, 1]
    assert cached["full_position"].tolist() == ["chr1:10-12", "chr1:20-22", "chr1:30-32", "chr1:40-42"]


def test_variability_bins_evaluation_reuses_existing_mean_label_cache_without_recalculating(tmp_path, monkeypatch):
    first_result_file = tmp_path / "eval_predictions_first.csv"
    second_result_file = tmp_path / "eval_predictions_second.csv"
    variability_file_path = tmp_path / "variability.csv"
    cache_path = tmp_path / "cache" / "sample_mean_label_cache.pkl"
    write_variability_bins_result_file(first_result_file, [0, 1, 2, 2])
    write_variability_bins_result_file(second_result_file, [2, 1, 0, 1])
    write_variability_file(variability_file_path)
    compare_dicts = create_variability_bins_compare_dicts()
    result_files_path = [str(first_result_file), str(second_result_file)]

    first_result = variability_bins_evaluator.evaluate_sample_predictions(
        variability_file_path=str(variability_file_path),
        result_files_path=result_files_path,
        chroms=["chr1"],
        comparison_bigwig_files=None,
        full_pos_name="full_pos",
        ranges=[0, 0.2, 0.8, 1],
        labels=[0, 1, 2],
        label_a="mean_label",
        label_b="predicted_class",
        number_of_bins=2,
        comparison_dicts=compare_dicts,
        mean_label_cache_path=cache_path,
    )

    def fail_if_recalculated(*args, **kwargs):
        raise AssertionError("mean_label should have been loaded from cache")

    monkeypatch.setattr(
        variability_bins_evaluator,
        "create_result_file_mean_label",
        fail_if_recalculated,
    )
    second_result = variability_bins_evaluator.evaluate_sample_predictions(
        variability_file_path=str(variability_file_path),
        result_files_path=result_files_path,
        chroms=["chr1"],
        comparison_bigwig_files=None,
        full_pos_name="full_pos",
        ranges=[0, 0.2, 0.8, 1],
        labels=[0, 1, 2],
        label_a="mean_label",
        label_b="predicted_class",
        number_of_bins=2,
        comparison_dicts=compare_dicts,
        mean_label_cache_path=cache_path,
    )

    assert second_result == first_result


def test_variability_bins_cache_reuse_rejects_prediction_files_with_different_window_ids(tmp_path):
    first_result_file = tmp_path / "eval_predictions_first.csv"
    incompatible_result_file = tmp_path / "eval_predictions_incompatible.csv"
    variability_file_path = tmp_path / "variability.csv"
    cache_path = tmp_path / "cache" / "sample_mean_label_cache.pkl"
    write_variability_bins_result_file(first_result_file, [0, 1, 2, 2])
    write_variability_bins_result_file(incompatible_result_file, [0, 1, 2, 2])
    incompatible = pd.read_csv(incompatible_result_file)
    incompatible.loc[2, "window_id"] = "chr1:999-1001"
    incompatible.to_csv(incompatible_result_file, index=False)
    write_variability_file(variability_file_path)
    compare_dicts = create_variability_bins_compare_dicts()

    with pytest.raises(ValueError, match="Cached window IDs do not match"):
        variability_bins_evaluator.evaluate_sample_predictions(
            variability_file_path=str(variability_file_path),
            result_files_path=[str(first_result_file), str(incompatible_result_file)],
            chroms=["chr1"],
            comparison_bigwig_files=None,
            full_pos_name="full_pos",
            ranges=[0, 0.2, 0.8, 1],
            labels=[0, 1, 2],
            label_a="mean_label",
            label_b="predicted_class",
            number_of_bins=2,
            comparison_dicts=compare_dicts,
            mean_label_cache_path=cache_path,
        )


def test_variability_bins_cache_reuse_rejects_prediction_files_with_different_labels(tmp_path):
    first_result_file = tmp_path / "eval_predictions_first.csv"
    incompatible_result_file = tmp_path / "eval_predictions_incompatible.csv"
    variability_file_path = tmp_path / "variability.csv"
    cache_path = tmp_path / "cache" / "sample_mean_label_cache.pkl"
    write_variability_bins_result_file(first_result_file, [0, 1, 2, 2])
    write_variability_bins_result_file(incompatible_result_file, [0, 1, 2, 2])
    incompatible = pd.read_csv(incompatible_result_file)
    incompatible.loc[1, "label"] = 2
    incompatible.to_csv(incompatible_result_file, index=False)
    write_variability_file(variability_file_path)
    compare_dicts = create_variability_bins_compare_dicts()

    with pytest.raises(ValueError, match="Cached labels do not match"):
        variability_bins_evaluator.evaluate_sample_predictions(
            variability_file_path=str(variability_file_path),
            result_files_path=[str(first_result_file), str(incompatible_result_file)],
            chroms=["chr1"],
            comparison_bigwig_files=None,
            full_pos_name="full_pos",
            ranges=[0, 0.2, 0.8, 1],
            labels=[0, 1, 2],
            label_a="mean_label",
            label_b="predicted_class",
            number_of_bins=2,
            comparison_dicts=compare_dicts,
            mean_label_cache_path=cache_path,
        )


def test_evaluate_sample_predictions_does_not_read_synthetic_all_two_from_csv(tmp_path):
    result_file_path = tmp_path / "eval_predictions_without_all_two.csv"
    pd.DataFrame(
        {
            "window_id": ["chr1:10-12", "chr1:20-22", "chr1:30-32"],
            "genomic_position": [10, 20, 30],
            "label": [0, 1, 2],
            "predicted_class": [0, 2, 2],
        }
    ).to_csv(result_file_path, index=False)
    precomputed_scores = {
        "mean_label": {
            0: {"Precision": 1.0, "recall": 1.0},
            1: {"Precision": 1.0, "recall": 1.0},
            2: {"Precision": 1.0, "recall": 1.0},
        },
        "mean_label_confusion_matrix": {0: {0: 1}, 1: {1: 1}, 2: {2: 1}},
    }

    result = evaluate_sample_predictions(
        result_files_path=[str(result_file_path)],
        chroms=["chr1"],
        comparison_bigwig_files=None,
        full_pos_name="full_pos",
        ranges=[0, 0.2, 0.8, 1],
        labels=[0, 1, 2],
        comparison_types=["mean_label", "predicted_class"],
        all_two=True,
        skip_mean_label=True,
        precomputed_scores=precomputed_scores,
    )

    eval_object = result[str(result_file_path)]["all_results"]
    assert eval_object["mean_label"] == precomputed_scores["mean_label"]
    assert eval_object["predicted_class"][2]["recall"] == 1.0
    assert eval_object["all_two"][2]["recall"] == 1.0

