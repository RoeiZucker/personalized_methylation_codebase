from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_cpg_island_personalization_significance import (
    compress_window_patterns,
    equal_tissue_statistic,
    exact_sign_flip_p,
    observed_f1,
    window_confusions,
)


def paired_tokens() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w1", "w1", "w2", "w2"],
            "label": [0, 1, 2, 2],
            "predicted_class_e10": [0, 1, 1, 2],
            "predicted_class_e3": [0, 0, 2, 2],
        }
    )


def test_window_confusions_reproduce_token_level_weighted_f1() -> None:
    e10, e3 = window_confusions(paired_tokens())
    e10_f1, e3_f1 = observed_f1(e10, e3)

    assert e10.shape == (2, 9)
    assert e3.shape == (2, 9)
    assert e10.sum() == 4
    assert e3.sum() == 4
    assert e10_f1 == pytest.approx(0.75)
    assert e3_f1 == pytest.approx(2 / 3)


def test_pattern_compression_preserves_confusion_totals() -> None:
    e10, e3 = window_confusions(paired_tokens())
    duplicated_e10 = np.repeat(e10, [3, 2], axis=0)
    duplicated_e3 = np.repeat(e3, [3, 2], axis=0)

    e10_patterns, e3_patterns, counts = compress_window_patterns(
        duplicated_e10, duplicated_e3
    )

    assert counts.sum() == 5
    np.testing.assert_array_equal(counts @ e10_patterns, duplicated_e10.sum(axis=0))
    np.testing.assert_array_equal(counts @ e3_patterns, duplicated_e3.sum(axis=0))


def test_exact_one_sided_sign_flip_enumerates_all_assignments() -> None:
    assert exact_sign_flip_p(np.array([0.1, 0.2])) == pytest.approx(0.25)


def test_equal_tissue_statistic_gives_tissues_equal_total_weight() -> None:
    people = pd.DataFrame(
        {
            "tissue": ["small", "large", "large", "large"],
            "delta": [0.4, 0.0, 0.0, 0.0],
        }
    )

    assert people["delta"].mean() == pytest.approx(0.1)
    assert equal_tissue_statistic(people, "delta") == pytest.approx(0.2)
