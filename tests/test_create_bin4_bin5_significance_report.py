from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_bin4_bin5_significance_report import (
    build_equal_person_summary,
    equal_tissue_f1,
    monte_carlo_p,
)


def synthetic_people() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bin": [4, 4, 4, 4],
            "tissue": ["small", "large", "large", "large"],
            "sample": ["s1", "l1", "l2", "l3"],
            "pretrain_value": [0.5, 0.5, 0.5, 0.5],
            "retrain_value": [0.9, 0.5, 0.5, 0.5],
            "difference": [0.4, 0.0, 0.0, 0.0],
            "sites": [10, 10, 10, 10],
            "windows": [5, 5, 5, 5],
        }
    )


def test_equal_person_summary_gives_every_person_the_same_weight() -> None:
    rows = synthetic_people()
    summary = build_equal_person_summary(rows).iloc[0]

    assert summary["equal_person_delta_f1"] == pytest.approx(0.1)
    assert summary["equal_person_pretrain_f1"] == pytest.approx(0.5)
    assert summary["equal_person_retrain_f1"] == pytest.approx(0.6)
    assert summary["relative_improvement_percent"] == pytest.approx(20.0)
    assert equal_tissue_f1(rows, "difference") == pytest.approx(0.2)


def test_monte_carlo_p_uses_plus_one_correction() -> None:
    null_values = np.array([0.1, 0.2, 0.3])

    assert monte_carlo_p(null_values, 0.2) == pytest.approx(0.75)
