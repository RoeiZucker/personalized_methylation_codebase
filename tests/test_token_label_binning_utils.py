import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.token_label_binning_utils import (
    CLOSE_TO_ONE_CLASS_ID,
    CLOSE_TO_ZERO_CLASS_ID,
    IN_BETWEEN_CLASS_ID,
    bin_token_label_values,
    resolve_token_label_binning,
)


def test_fixed_token_label_binning_preserves_blank_labels():
    resolved = resolve_token_label_binning(
        {
            "token_label_binning": {
                "method": "fixed",
                "low": 0.2,
                "high": 0.8,
            }
        }
    )

    labels = bin_token_label_values([0.0, 0.2, 0.5, 0.8, 1.0, -100], resolved)

    assert labels == [
        CLOSE_TO_ZERO_CLASS_ID,
        CLOSE_TO_ZERO_CLASS_ID,
        IN_BETWEEN_CLASS_ID,
        CLOSE_TO_ONE_CLASS_ID,
        CLOSE_TO_ONE_CLASS_ID,
        -100,
    ]


def test_quantile_token_label_binning_uses_train_labels_only():
    train_dataset = [
        {"labels": [-100, 0.0, 0.25, 0.5, 0.75, 1.0]},
    ]
    eval_like_labels = [0.1, 0.3, 0.9]
    resolved = resolve_token_label_binning(
        {
            "token_label_binning": {
                "method": "quantile",
                "low": 0.25,
                "high": 0.75,
            }
        },
        train_dataset=train_dataset,
    )

    labels = bin_token_label_values(eval_like_labels, resolved)

    assert resolved.resolved_low == pytest.approx(0.25)
    assert resolved.resolved_high == pytest.approx(0.75)
    assert labels == [
        CLOSE_TO_ZERO_CLASS_ID,
        IN_BETWEEN_CLASS_ID,
        CLOSE_TO_ONE_CLASS_ID,
    ]


def test_quantile_token_label_binning_rejects_blank_train_labels():
    with pytest.raises(ValueError, match="no non-blank labels"):
        resolve_token_label_binning(
            {
                "token_label_binning": {
                    "method": "quantile",
                    "low": 0.2,
                    "high": 0.8,
                }
            },
            train_dataset=[{"labels": [-100, -100]}],
        )
