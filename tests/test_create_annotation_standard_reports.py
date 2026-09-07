from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from create_annotation_standard_reports import (
    display_name,
    merge_standard_rows,
    weighted_f1,
)


def test_weighted_f1_uses_confusion_support() -> None:
    matrix = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
        ]
    )

    assert weighted_f1(matrix) == pytest.approx(0.75)


def test_weighted_f1_is_nan_for_empty_bin() -> None:
    assert np.isnan(weighted_f1(np.zeros((3, 3), dtype=int)))


def test_merge_standard_rows_uses_exact_prediction_keys() -> None:
    filtered = pd.DataFrame(
        {
            "window_id": ["chr1:0-100", "chr1:0-100"],
            "token_start": [10, 20],
            "label": [1, 2],
            "predicted_class_e10": [1, 1],
            "predicted_class_e3": [1, 2],
        }
    )
    cache = pd.DataFrame(
        {
            "window_id": ["chr1:0-100"],
            "genomic_position": [20],
            "label": [2],
            "mean_label": [2],
            "section_rank": ["bin_3"],
            "predicted_class_e1": [1],
            "predicted_class_e2": [2],
        }
    )

    result = merge_standard_rows(filtered, cache)

    assert len(result) == 1
    assert result.iloc[0]["genomic_position"] == 20
    assert result.iloc[0]["section_rank"] == "bin_3"


def test_display_name_preserves_annotation_acronyms() -> None:
    assert display_name("ccre_ca_h3k4me3") == "cCRE CA H3K4me3"
    assert display_name("cpg_shore") == "CpG Shore"
