import sys
from pathlib import Path

import pandas as pd
import pyBigWig
import pytest
import types
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))


def _install_evaluate_stub():
    module = types.ModuleType("evaluate")

    class Metric:
        def __init__(self, name):
            self.name = name

        def compute(self, predictions=None, references=None):
            return {self.name: 0.0}

    module.load = lambda name: Metric(name)
    sys.modules["evaluate"] = module


_install_evaluate_stub()

from src.utils.atlas_bigwig_utils import build_atlas_position_dataframe
from src.utils.atlas_distribution_utils import (
    load_variability_std_dataframe,
    resolve_atlas_job_inputs,
    summarize_atlas_distribution,
    summarize_distribution_values,
)


def _write_bigwig(path: Path, starts, values, chrom="chr1", chrom_length=100):
    bw = pyBigWig.open(str(path), "w")
    try:
        bw.addHeader([(chrom, chrom_length)])
        bw.addEntries(
            [chrom] * len(starts),
            starts,
            ends=[start + 1 for start in starts],
            values=[float(value) for value in values],
        )
    finally:
        bw.close()


def _create_sample_bigwigs(tmp_path: Path):
    starts = [0, 6, 12, 18, 24, 30]
    target_path = tmp_path / "target.bw"
    atlas_a_path = tmp_path / "atlas_a.bw"
    atlas_b_path = tmp_path / "atlas_b.bw"

    _write_bigwig(target_path, starts, [0.1, 0.3, 0.5, 0.7, 0.55, 0.7])
    _write_bigwig(atlas_a_path, starts, [0.1, 0.2, 0.9, 0.8, 0.95, 0.85])
    _write_bigwig(atlas_b_path, starts, [0.1, 0.4, 0.1, 0.6, 0.15, 0.55])
    return target_path, [atlas_a_path, atlas_b_path]


def test_summarize_atlas_distribution_uses_prediction_mean_and_target_rate(tmp_path):
    target_path, atlas_paths = _create_sample_bigwigs(tmp_path)
    matched_df = build_atlas_position_dataframe(
        target_bigwig_path=str(target_path),
        atlas_bigwig_paths=[str(path) for path in atlas_paths],
        number_of_bins=2,
        chroms=["chr1"],
        verbose=False,
    )

    summary_df = summarize_atlas_distribution(matched_df)
    assert set(summary_df["measure"]) == {
        "reference_std",
        "atlas_mean_prediction",
        "held_out_target_rate",
    }

    atlas_row = summary_df[summary_df["measure"] == "atlas_mean_prediction"].iloc[0]
    target_row = summary_df[summary_df["measure"] == "held_out_target_rate"].iloc[0]
    std_row = summary_df[summary_df["measure"] == "reference_std"].iloc[0]

    assert atlas_row["count"] == 6
    assert atlas_row["mean"] == pytest.approx(matched_df["atlas_mean"].to_numpy(dtype=float).mean(), abs=5e-4)
    assert target_row["mean"] == pytest.approx(matched_df["target_value"].to_numpy(dtype=float).mean(), abs=5e-4)
    assert std_row["mean"] == pytest.approx(matched_df["std"].to_numpy(dtype=float).mean(), abs=5e-4)


def test_load_variability_std_dataframe_uses_saved_csv_and_debug_controls(tmp_path):
    variability_path = tmp_path / "variability.csv"
    pd.DataFrame(
        {
            "full_position": ["chr1:1-2", "chr1:2-3", "chr1:3-4", "chr1:4-5"],
            "window_id": ["w1", "w1", "w2", "w2"],
            "std": [0.1, None, 0.3, 0.5],
            "high_diff": [False, True, True, True],
        }
    ).to_csv(variability_path, index=False)

    loaded_df = load_variability_std_dataframe(variability_path)
    assert loaded_df["std"].tolist() == pytest.approx([0.1, 0.3, 0.5])

    top_rows_df = load_variability_std_dataframe(variability_path, top_rows=2)
    assert top_rows_df["std"].tolist() == pytest.approx([0.1, 0.3])

    sampled_df = load_variability_std_dataframe(variability_path, test_mode=True, jump_sample=2)
    assert sampled_df["std"].tolist() == pytest.approx([0.1, 0.5])

    summary_df = summarize_distribution_values(loaded_df["std"], "reference_std")
    assert summary_df.iloc[0]["measure"] == "reference_std"
    assert summary_df.iloc[0]["count"] == 3


def test_resolve_atlas_job_inputs_supports_legacy_configs_and_variability_path(tmp_path):
    uber_script_path = tmp_path / "uber_project_creator_script.py"
    uber_script_path.write_text(
        "\n".join(
            [
                "FILES_NAMES ='''''GSM1_Group-A-S1.hg38.bigwig",
                "GSM2_Group-A-S2.hg38.bigwig",
                "GSM3_Group-A-S3.hg38.bigwig",
                "'''.split(\"\\n\")",
                'BASE_FILE_PATH = "/tmp/bigwigs"',
            ]
        ),
        encoding="utf-8",
    )

    atlas_config_path = tmp_path / "legacy_atlas_eval.yaml"
    atlas_config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "variant_file_path": "/tmp/variability.csv",
                },
                "task": {
                    "analysis_name": "S1_atlas_eval_Group-A_seq_5400",
                    "base_suffix": "_Group-A",
                    "number_of_bins": 5,
                },
                "testing_params": {
                    "test_mode": True,
                    "jump_sample": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_atlas_job_inputs(atlas_config_path, uber_script_path=uber_script_path)
    assert resolved["held_out_sample"] == "S1"
    assert resolved["group_name"] == "Group-A"
    assert resolved["target_bigwig_path"] == "/tmp/bigwigs/GSM1_Group-A-S1.hg38.bigwig"
    assert resolved["atlas_bigwig_paths"] == [
        "/tmp/bigwigs/GSM2_Group-A-S2.hg38.bigwig",
        "/tmp/bigwigs/GSM3_Group-A-S3.hg38.bigwig",
    ]
    assert resolved["variant_file_path"] == "/tmp/variability.csv"
    assert resolved["test_mode"] is True
    assert resolved["jump_sample"] == 3
