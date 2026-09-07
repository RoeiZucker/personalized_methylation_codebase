from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_cpg_island_standard_report import (
    aggregate,
    candidate_positions,
    weighted_f1,
)


def test_candidate_positions_cover_evaluator_offsets() -> None:
    rows = pd.DataFrame(
        {
            "row_id": [4],
            "chrom": ["chr12"],
            "genomic_position": [100],
        }
    )

    candidates = candidate_positions(rows)

    assert candidates["row_id"].tolist() == [4] * 7
    assert candidates["full_position"].tolist() == [
        f"chr12:{start}-{start + 2}" for start in range(99, 106)
    ]


def test_weighted_f1_uses_class_support() -> None:
    labels = np.array([0, 1, 2, 2])
    predictions = np.array([0, 1, 1, 2])

    assert weighted_f1(labels, predictions) == pytest.approx(0.75)


def test_aggregate_gives_participants_equal_weight_and_std() -> None:
    rows = pd.DataFrame(
        {
            "stage": ["E3", "E3"],
            "prediction_type": ["predicted_class", "predicted_class"],
            "section_rank": ["all_results", "all_results"],
            "weighted_f1": [0.6, 0.8],
            "support": [10, 1000],
        }
    )

    result = aggregate(rows, ["stage", "prediction_type", "section_rank"]).iloc[0]

    assert result["weighted_f1"] == pytest.approx(0.7)
    assert result["weighted_f1_std"] == pytest.approx(np.sqrt(0.02))
    assert result["avg_support"] == pytest.approx(505)
