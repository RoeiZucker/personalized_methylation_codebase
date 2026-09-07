from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from create_annotation_personalization_synthesis_report import (
    ANNOTATION_ORDER,
    chr12_overall,
    chr12_tissue,
    consistency_table,
    enrich,
    regular_overall,
    regular_tissue,
)


def test_overall_tables_cover_same_annotations() -> None:
    regular = enrich(regular_overall())
    chr12 = enrich(chr12_overall())

    assert regular["filter"].astype(str).tolist() == ANNOTATION_ORDER
    assert chr12["filter"].astype(str).tolist() == ANNOTATION_ORDER
    assert regular["donors"].eq(31).all()
    assert chr12["donors"].eq(31).all()


def test_cross_setting_summary_matches_source_effects() -> None:
    regular = enrich(regular_overall())
    chr12 = enrich(chr12_overall())
    comparison = consistency_table(regular, chr12).set_index("filter")

    assert comparison["same_direction"].sum() == 9
    assert comparison.loc[
        "cpg_island", "filtered_minus_unfiltered_delta_regular"
    ] == pytest.approx(0.02154272)
    assert comparison.loc[
        "ccre_pls", "filtered_minus_unfiltered_delta_chr12"
    ] == pytest.approx(0.01204955)


def test_tissue_tables_are_complete() -> None:
    regular = regular_tissue()
    chr12 = chr12_tissue()

    assert len(regular) == 16 * 6
    assert len(chr12) == 16 * 6
    assert not regular.isna().any().any()
    assert not chr12.isna().any().any()
