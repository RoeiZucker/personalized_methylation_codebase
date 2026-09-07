#!/usr/bin/env python3
# Create the complete exploratory annotation synthesis:
#   python3 scripts/create_annotation_personalization_synthesis_report.py
# Write the same report to another destination:
#   python3 scripts/create_annotation_personalization_synthesis_report.py --output-dir /tmp/annotation_report

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ISLAND_ROOT = PROJECT / "cpg_island_vs_unfiltered_f1_20260828/outputs"
OTHER_ROOT = PROJECT / "other_annotation_f1_analysis_20260830/outputs"
UNION_ROOT = PROJECT / "regulatory_union_f1_analysis_20260830/outputs"
CHR12_ROOT = PROJECT / "chr12_annotation_f1_analysis_20260830/outputs"
OUTPUT = PROJECT / "annotation_personalization_synthesis_20260831/outputs"

ANNOTATION_ORDER = [
    "cpg_island",
    "cpg_shore",
    "cpg_shelf",
    "open_sea",
    "promoter",
    "gene_body",
    "regulatory_union",
    "ccre_pls",
    "ccre_pels",
    "ccre_dels",
    "ccre_els",
    "ccre_ca",
    "ccre_ca_ctcf",
    "ccre_ca_h3k4me3",
    "ccre_ca_tf",
    "ccre_tf",
]

DISPLAY = {
    "cpg_island": "CpG island",
    "cpg_shore": "CpG shore",
    "cpg_shelf": "CpG shelf",
    "open_sea": "Open Sea",
    "promoter": "Promoter",
    "gene_body": "Gene body",
    "regulatory_union": "Regulatory union",
    "ccre_pls": "cCRE PLS",
    "ccre_pels": "cCRE pELS",
    "ccre_dels": "cCRE dELS",
    "ccre_els": "cCRE ELS",
    "ccre_ca": "cCRE CA",
    "ccre_ca_ctcf": "cCRE CA-CTCF",
    "ccre_ca_h3k4me3": "cCRE CA-H3K4me3",
    "ccre_ca_tf": "cCRE CA-TF",
    "ccre_tf": "cCRE TF",
}

TISSUE_ORDER = [
    "Bladder-Epithelial",
    "Breast-Basal-Epithelial",
    "Cortex-Neuron",
    "Heart-Cardiomyocyte",
    "Heart-Fibroblasts",
    "Liver-Hepatocytes",
]

OVERALL_COLUMNS = [
    "filter",
    "donors",
    "tissues",
    "paired_tokens",
    "retained_tokens",
    "unfiltered_weighted_f1_e10",
    "unfiltered_weighted_f1_e3",
    "unfiltered_delta_e3_minus_e10",
    "filtered_weighted_f1_e10",
    "filtered_weighted_f1_e3",
    "filtered_delta_e3_minus_e10",
    "filtered_minus_unfiltered_delta",
    "gain_difference_positive_donors",
    "gain_difference_negative_donors",
    "gain_difference_zero_donors",
    "gain_difference_nominal_sign_p",
]

TISSUE_COLUMNS = [
    "filter",
    "tissue",
    "donors",
    "paired_tokens",
    "retained_tokens",
    "mean_unfiltered_weighted_f1_e10",
    "mean_unfiltered_weighted_f1_e3",
    "mean_unfiltered_delta_e3_minus_e10",
    "mean_filtered_weighted_f1_e10",
    "mean_filtered_weighted_f1_e3",
    "mean_filtered_delta_e3_minus_e10",
    "mean_filtered_minus_unfiltered_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an exploratory regular-TEST and chr12 annotation synthesis report."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def equal_donor_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    return rows.loc[rows["summary_weighting"].eq("equal_donor")].copy()


def island_overall() -> pd.DataFrame:
    row = equal_donor_rows(ISLAND_ROOT / "overall_comparison.csv").iloc[0]
    return pd.DataFrame(
        [
            {
                "filter": "cpg_island",
                "donors": row["donors"],
                "tissues": row["tissues"],
                "paired_tokens": row["paired_tokens"],
                "retained_tokens": row["island_overlap_tokens"],
                "unfiltered_weighted_f1_e10": row["unfiltered_weighted_f1_e10"],
                "unfiltered_weighted_f1_e3": row["unfiltered_weighted_f1_e3"],
                "unfiltered_delta_e3_minus_e10": row["unfiltered_delta_e3_minus_e10"],
                "filtered_weighted_f1_e10": row["island_weighted_f1_e10"],
                "filtered_weighted_f1_e3": row["island_weighted_f1_e3"],
                "filtered_delta_e3_minus_e10": row["island_delta_e3_minus_e10"],
                "filtered_minus_unfiltered_delta": row["island_minus_unfiltered_delta"],
                "gain_difference_positive_donors": row[
                    "gain_difference_positive_donors"
                ],
                "gain_difference_negative_donors": row[
                    "gain_difference_negative_donors"
                ],
                "gain_difference_zero_donors": row["gain_difference_zero_donors"],
                "gain_difference_nominal_sign_p": row[
                    "gain_difference_nominal_sign_p"
                ],
            }
        ]
    )


def island_tissue() -> pd.DataFrame:
    rows = pd.read_csv(ISLAND_ROOT / "tissue_comparison.csv")
    return pd.DataFrame(
        {
            "filter": "cpg_island",
            "tissue": rows["tissue"],
            "donors": rows["donors"],
            "paired_tokens": rows["paired_tokens"],
            "retained_tokens": rows["island_overlap_tokens"],
            "mean_unfiltered_weighted_f1_e10": rows[
                "mean_unfiltered_weighted_f1_e10"
            ],
            "mean_unfiltered_weighted_f1_e3": rows[
                "mean_unfiltered_weighted_f1_e3"
            ],
            "mean_unfiltered_delta_e3_minus_e10": rows[
                "mean_unfiltered_delta_e3_minus_e10"
            ],
            "mean_filtered_weighted_f1_e10": rows["mean_island_weighted_f1_e10"],
            "mean_filtered_weighted_f1_e3": rows["mean_island_weighted_f1_e3"],
            "mean_filtered_delta_e3_minus_e10": rows[
                "mean_island_delta_e3_minus_e10"
            ],
            "mean_filtered_minus_unfiltered_delta": rows[
                "mean_island_minus_unfiltered_delta"
            ],
        }
    )


def regular_overall() -> pd.DataFrame:
    other = equal_donor_rows(OTHER_ROOT / "overall_comparison.csv")
    union = equal_donor_rows(UNION_ROOT / "overall_comparison.csv")
    rows = pd.concat([island_overall(), other, union], ignore_index=True)
    return order_annotations(rows[OVERALL_COLUMNS])


def regular_tissue() -> pd.DataFrame:
    other = pd.read_csv(OTHER_ROOT / "tissue_comparison.csv")
    union = pd.read_csv(UNION_ROOT / "tissue_comparison.csv")
    rows = pd.concat([island_tissue(), other, union], ignore_index=True)
    return order_annotations(rows[TISSUE_COLUMNS])


def chr12_overall() -> pd.DataFrame:
    rows = equal_donor_rows(CHR12_ROOT / "overall_comparison.csv")
    return order_annotations(rows[OVERALL_COLUMNS])


def chr12_tissue() -> pd.DataFrame:
    rows = pd.read_csv(CHR12_ROOT / "tissue_comparison.csv")
    return order_annotations(rows[TISSUE_COLUMNS])


def order_annotations(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.loc[rows["filter"].isin(ANNOTATION_ORDER)].copy()
    output["filter"] = pd.Categorical(
        output["filter"], categories=ANNOTATION_ORDER, ordered=True
    )
    columns = ["filter", "tissue"] if "tissue" in output else ["filter"]
    return output.sort_values(columns).reset_index(drop=True)


def enrich(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    output["annotation"] = output["filter"].astype(str).map(DISPLAY)
    output["positive_donor_fraction"] = (
        output["gain_difference_positive_donors"] / output["donors"]
    )
    output["retained_fraction"] = output["retained_tokens"] / output["paired_tokens"]
    return output


def consistency_table(regular: pd.DataFrame, chr12: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "filter",
        "filtered_minus_unfiltered_delta",
        "gain_difference_positive_donors",
        "positive_donor_fraction",
    ]
    rows = regular[columns].merge(
        chr12[columns], on="filter", suffixes=("_regular", "_chr12")
    )
    rows["annotation"] = rows["filter"].astype(str).map(DISPLAY)
    rows["same_direction"] = np.sign(
        rows["filtered_minus_unfiltered_delta_regular"]
    ).eq(np.sign(rows["filtered_minus_unfiltered_delta_chr12"]))
    return order_annotations(rows)


def write_tables(
    output_dir: Path,
    regular: pd.DataFrame,
    chr12: pd.DataFrame,
    regular_tissues: pd.DataFrame,
    chr12_tissues: pd.DataFrame,
    consistency: pd.DataFrame,
) -> None:
    regular.to_csv(output_dir / "regular_test_overall_summary.csv", index=False)
    chr12.to_csv(output_dir / "chr12_overall_summary.csv", index=False)
    regular_tissues.to_csv(output_dir / "regular_test_tissue_summary.csv", index=False)
    chr12_tissues.to_csv(output_dir / "chr12_tissue_summary.csv", index=False)
    consistency.to_csv(output_dir / "cross_setting_comparison.csv", index=False)


def format_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def markdown_table(rows: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, separator]
    for _, row in rows.iterrows():
        values = []
        for column, _, formatter in columns:
            value = row[column]
            if formatter == "int":
                values.append(f"{int(value):,}")
            elif formatter == "f1":
                values.append(f"{float(value):.6f}")
            elif formatter == "pct":
                values.append(f"{100 * float(value):.1f}%")
            elif formatter == "p":
                values.append(format_p(float(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def overall_markdown(rows: pd.DataFrame) -> str:
    columns = [
        ("annotation", "Annotation", "text"),
        ("retained_tokens", "Tokens", "int"),
        ("filtered_weighted_f1_e10", "E10 F1", "f1"),
        ("filtered_weighted_f1_e3", "E3 F1", "f1"),
        ("filtered_delta_e3_minus_e10", "E3-E10", "f1"),
        ("filtered_minus_unfiltered_delta", "Gain shift", "f1"),
        ("positive_donor_fraction", "Donors positive", "pct"),
        ("gain_difference_nominal_sign_p", "Nominal p", "p"),
    ]
    return markdown_table(rows, columns)


def tissue_markdown(rows: pd.DataFrame) -> str:
    matrix = rows.pivot(
        index="filter",
        columns="tissue",
        values="mean_filtered_minus_unfiltered_delta",
    ).reindex(index=ANNOTATION_ORDER, columns=TISSUE_ORDER)
    table = matrix.reset_index()
    table["annotation"] = table["filter"].map(DISPLAY)
    columns = [("annotation", "Annotation", "text")]
    columns.extend((tissue, tissue, "f1") for tissue in TISSUE_ORDER)
    return markdown_table(table, columns)


def report_text(
    regular: pd.DataFrame,
    chr12: pd.DataFrame,
    regular_tissues: pd.DataFrame,
    chr12_tissues: pd.DataFrame,
    consistency: pd.DataFrame,
) -> str:
    pearson = consistency["filtered_minus_unfiltered_delta_regular"].corr(
        consistency["filtered_minus_unfiltered_delta_chr12"]
    )
    spearman = consistency["filtered_minus_unfiltered_delta_regular"].corr(
        consistency["filtered_minus_unfiltered_delta_chr12"], method="spearman"
    )
    same = int(consistency["same_direction"].sum())
    return f"""# Exploratory methylation-personalization annotation synthesis

## Scope

- Comparison: donor-specific leave-one-donor-out pretrained E10 versus personalized E3.
- Metric: K=3 class-support-weighted F1, calculated per donor.
- Primary descriptive effect: F1(E3) - F1(E10).
- Annotation contrast: filtered personalization gain minus the corresponding unfiltered gain.
- Biological unit for direction summaries: donor (n=31) across six tissue/cell-type strata.
- Regular TEST and chromosome 12 are analyzed separately. Chromosome 12 is an exposed exploratory sensitivity analysis, not an independent confirmation.

Every row-level record underlying these summaries is a documented 6-base prediction token that overlaps an annotation by at least 1 bp. It is not an independently predicted CpG cytosine. SCREEN cCRE labels denote candidate cis-regulatory elements and do not establish function in these samples.

## Main findings

The unfiltered average personalization effect was negative in both settings: -0.014057 weighted-F1 units in regular TEST and -0.006318 on chromosome 12. Annotation filtering identified narrower contexts in which this deficit was reduced or reversed.

The largest positive gain shifts in both settings occurred in **CpG islands** (regular +0.021543; chr12 +0.010315) and **cCRE PLS** (regular +0.018430; chr12 +0.012050). The shifts were positive for 30/31 regular-TEST donors for each annotation, and for 25/31 and 27/31 chr12 donors, respectively. In regular TEST, these were the only annotations with positive average filtered personalization gains themselves: CpG islands +0.007486 and cCRE PLS +0.004374.

**cCRE pELS** and **promoters** also had positive gain shifts in both settings. cCRE CA-H3K4me3 was directionally positive in both, although its donor pattern was less uniform. In contrast, **Open Sea** and **cCRE TF** had negative gain shifts in both settings.

Cross-setting agreement was incomplete: {same}/16 annotations had the same gain-shift sign. The descriptive Pearson correlation was {pearson:.3f} and the rank correlation was {spearman:.3f}. CpG shores, CpG shelves, cCRE dELS, cCRE CA, cCRE CA-CTCF, and cCRE CA-TF changed from nonpositive regular-TEST shifts to positive chr12 shifts; gene body changed in the opposite direction. These reversals argue against treating chr12 as straightforward replication.

Tissue summaries show heterogeneity. Heart-Cardiomyocyte had especially large CpG-island and cCRE-PLS gain shifts in both settings. Heart-Fibroblasts was the notable exception for several otherwise favorable annotations: cCRE PLS, cCRE pELS, and promoters were negative in this tissue in both settings, and the chr12 CpG-island shift was also negative. Tissue sample sizes are only 4-8 donors, so these are descriptive patterns rather than tissue-specific tests.

The broad regulatory union had only a small regular-TEST shift (+0.000724, 16/31 donors positive) but a larger chr12 shift (+0.004426, 26/31 positive). A broad union can dilute narrower positive and negative subclasses, and chr12 was already exposed.

## Regular TEST results

{overall_markdown(regular)}

Nominal p is the unadjusted two-sided donor sign-test p-value for the filtered-minus-unfiltered gain difference. It does not test token-level independence and is not corrected across overlapping annotations.

## Chromosome 12 results

{overall_markdown(chr12)}

Chromosome 12 was excluded from the audited task-training methylation labels, but its results were previously examined. These values are therefore exploratory and exposed.

## Regular TEST tissue gain shifts

{tissue_markdown(regular_tissues)}

## Chromosome 12 tissue gain shifts

{tissue_markdown(chr12_tissues)}

## Interpretation limits

- Absolute filtered F1 values are affected by annotation-specific class balance; the gain shift is the more relevant cross-annotation comparison.
- The millions of retained tokens are repeated genomic observations, not independent biological replicates.
- Nominal sign tests use 31 donor directions, ignore effect magnitude, and are unadjusted across 16 overlapping annotation definitions.
- A nominal result inside one annotation and a non-nominal result outside it do not establish effect modification.
- Annotation overlap does not demonstrate regulatory mechanism, causal activity, or individual-CpG behavior.
- Tissue patterns are descriptive because each stratum contains only 4-8 donors.
- Regular TEST and chr12 are not pooled.

## Sources

- {ISLAND_ROOT / "REPORT.md"}
- {OTHER_ROOT / "REPORT.md"}
- {UNION_ROOT / "REPORT.md"}
- {CHR12_ROOT / "REPORT.md"}
- Corresponding overall_comparison.csv, tissue_comparison.csv, and per_person_comparison.csv files in each directory.

The large row-level compressed datasets were not reread because no recalculation or new annotation-by-bin analysis was required.
"""


def text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(11, 8.5))
    axis.axis("off")
    axis.text(0.04, 0.96, title, fontsize=18, fontweight="bold", va="top")
    y = 0.88
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, width=115)
        axis.text(0.05, y, wrapped, fontsize=10.5, va="top", linespacing=1.35)
        y -= 0.045 * (wrapped.count("\n") + 2)
    pdf.savefig(figure)
    plt.close(figure)


def effect_page(pdf: PdfPages, rows: pd.DataFrame, title: str) -> None:
    selected = rows.sort_values("filtered_minus_unfiltered_delta")
    y = np.arange(len(selected))
    effects = selected["filtered_minus_unfiltered_delta"].to_numpy()
    colors = np.where(effects >= 0, "#2A9D8F", "#C44E52")
    figure, axis = plt.subplots(figsize=(11, 8.5))
    axis.axvline(0, color="#333333", linewidth=1)
    axis.barh(y, effects, color=colors, alpha=0.88)
    axis.set_yticks(y, selected["annotation"])
    axis.set_xlabel("Filtered gain minus unfiltered gain (weighted F1)")
    axis.set_title(title, loc="left", fontsize=15)
    axis.grid(axis="x", alpha=0.2)
    span = max(abs(effects.min()), abs(effects.max()))
    for index, row in enumerate(selected.itertuples(index=False)):
        x = row.filtered_minus_unfiltered_delta
        label = (
            f"{x:+.4f} | {int(row.gain_difference_positive_donors)}/"
            f"{int(row.donors)} donors positive"
        )
        axis.text(
            x + (0.015 * span if x >= 0 else -0.015 * span),
            index,
            label,
            ha="left" if x >= 0 else "right",
            va="center",
            fontsize=7.5,
        )
    axis.set_xlim(-1.45 * span, 1.45 * span)
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def overall_table_page(pdf: PdfPages, rows: pd.DataFrame, title: str) -> None:
    table_rows = []
    for row in rows.itertuples(index=False):
        table_rows.append(
            [
                row.annotation,
                f"{int(row.retained_tokens):,}",
                f"{row.filtered_weighted_f1_e10:.4f}",
                f"{row.filtered_weighted_f1_e3:.4f}",
                f"{row.filtered_delta_e3_minus_e10:+.4f}",
                f"{row.filtered_minus_unfiltered_delta:+.4f}",
                f"{int(row.gain_difference_positive_donors)}/{int(row.donors)}",
                format_p(row.gain_difference_nominal_sign_p),
            ]
        )
    figure, axis = plt.subplots(figsize=(11, 8.5))
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=15, pad=16)
    table = axis.table(
        cellText=table_rows,
        colLabels=[
            "Annotation",
            "Tokens",
            "E10 F1",
            "E3 F1",
            "E3-E10",
            "Gain shift",
            "Donors +",
            "Nominal p",
        ],
        loc="center",
        cellLoc="right",
        colLoc="right",
        colWidths=[0.19, 0.15, 0.09, 0.09, 0.09, 0.10, 0.09, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.55)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#DCE6F1")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F6F7")
        if column == 0:
            cell.set_text_props(ha="left")
    axis.text(
        0.0,
        0.02,
        "Nominal p: unadjusted two-sided donor sign test for the gain shift; exploratory only.",
        fontsize=8,
    )
    pdf.savefig(figure)
    plt.close(figure)


def consistency_page(pdf: PdfPages, rows: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    x = rows["filtered_minus_unfiltered_delta_regular"]
    y = rows["filtered_minus_unfiltered_delta_chr12"]
    limit = 1.15 * max(abs(x).max(), abs(y).max())
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].axvline(0, color="#777777", linewidth=0.8)
    axes[0].plot([-limit, limit], [-limit, limit], "--", color="#AAAAAA")
    axes[0].scatter(x, y, color="#3C78A8", s=38)
    for row in rows.itertuples(index=False):
        axes[0].annotate(
            row.annotation,
            (
                row.filtered_minus_unfiltered_delta_regular,
                row.filtered_minus_unfiltered_delta_chr12,
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    axes[0].set_xlim(-limit, limit)
    axes[0].set_ylim(-limit, limit)
    axes[0].set_xlabel("Regular TEST gain shift")
    axes[0].set_ylabel("Chr12 gain shift")
    axes[0].set_title("Effect-size consistency", loc="left")
    positions = np.arange(len(rows))
    width = 0.38
    axes[1].barh(
        positions - width / 2,
        100 * rows["positive_donor_fraction_regular"],
        height=width,
        label="Regular TEST",
        color="#4C78A8",
    )
    axes[1].barh(
        positions + width / 2,
        100 * rows["positive_donor_fraction_chr12"],
        height=width,
        label="Chr12",
        color="#F28E2B",
    )
    axes[1].axvline(50, color="#777777", linestyle=":", linewidth=1)
    axes[1].set_yticks(positions, rows["annotation"], fontsize=7)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Donors with positive gain shift (%)")
    axes[1].set_title("Donor directions", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Regular TEST and chromosome 12 are separate exploratory analyses",
        fontsize=15,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(figure)
    plt.close(figure)


def tissue_heatmap_page(pdf: PdfPages, rows: pd.DataFrame, title: str) -> None:
    matrix = rows.pivot(
        index="filter",
        columns="tissue",
        values="mean_filtered_minus_unfiltered_delta",
    ).reindex(index=ANNOTATION_ORDER, columns=TISSUE_ORDER)
    values = matrix.to_numpy()
    limit = max(abs(np.nanmin(values)), abs(np.nanmax(values)))
    figure, axis = plt.subplots(figsize=(11, 8.5))
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    axis.set_xticks(np.arange(len(TISSUE_ORDER)), TISSUE_ORDER, rotation=30, ha="right")
    axis.set_yticks(
        np.arange(len(ANNOTATION_ORDER)), [DISPLAY[x] for x in ANNOTATION_ORDER]
    )
    axis.set_title(title, loc="left", fontsize=15)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if abs(value) > 0.55 * limit else "black"
            axis.text(
                column,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
            )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
    colorbar.set_label("Filtered gain minus unfiltered gain")
    figure.tight_layout()
    pdf.savefig(figure)
    plt.close(figure)


def write_pdf(
    path: Path,
    regular: pd.DataFrame,
    chr12: pd.DataFrame,
    regular_tissues: pd.DataFrame,
    chr12_tissues: pd.DataFrame,
    consistency: pd.DataFrame,
) -> None:
    with PdfPages(path) as pdf:
        text_page(
            pdf,
            "Exploratory methylation-personalization annotation synthesis",
            [
                "E10 is the donor-specific leave-one-donor-out pretrained tissue model; E3 is the final personalized model. Positive E3-E10 values favor personalization.",
                "The primary cross-annotation quantity is the filtered personalization gain minus the corresponding unfiltered gain. All overall summaries average 31 donors equally across six tissue/cell-type strata.",
                "Regular TEST and chromosome 12 are reported separately. Chromosome 12 is an exposed sensitivity analysis because its results were previously examined.",
                "Tokens are 6-base prediction intervals overlapping an annotation by at least one base. They are not independent biological replicates or individually predicted CpG cytosines.",
                "SCREEN labels describe candidate cis-regulatory elements, not experimentally confirmed regulatory activity in these samples.",
            ],
        )
        effect_page(pdf, regular, "Regular TEST: annotation-specific personalization gain shifts")
        overall_table_page(pdf, regular, "Regular TEST: equal-donor weighted-F1 summary")
        effect_page(pdf, chr12, "Chromosome 12: exposed exploratory gain shifts")
        overall_table_page(pdf, chr12, "Chromosome 12: equal-donor weighted-F1 summary")
        consistency_page(pdf, consistency)
        tissue_heatmap_page(
            pdf, regular_tissues, "Regular TEST: tissue-level descriptive gain shifts"
        )
        tissue_heatmap_page(
            pdf, chr12_tissues, "Chromosome 12: tissue-level descriptive gain shifts"
        )
        text_page(
            pdf,
            "Interpretation and limitations",
            [
                "CpG islands and cCRE PLS had the largest positive gain shifts in both settings. cCRE pELS and promoters were also directionally positive in both. Open Sea and cCRE TF were directionally negative in both.",
                "Only 9 of 16 annotations retained the same gain-shift sign across settings. Several cCRE subclasses reversed, so chromosome 12 should not be described as independent replication.",
                "Tissue summaries are descriptive because each stratum contains only 4-8 donors. Heart-Fibroblasts differed from several otherwise favorable patterns.",
                "Nominal sign-test p-values use donor directions, are unadjusted across overlapping annotations, and ignore effect magnitude. They are exploratory and not confirmatory clustered genomic inference.",
                "Filtering changes class balance, so absolute filtered F1 values should not be compared as though annotations define matched prediction tasks. Annotation overlap does not establish regulation, mechanism, or individual-CpG behavior.",
            ],
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regular = enrich(regular_overall())
    chr12 = enrich(chr12_overall())
    regular_tissues = regular_tissue()
    chr12_tissues = chr12_tissue()
    consistency = consistency_table(regular, chr12)
    write_tables(
        args.output_dir,
        regular,
        chr12,
        regular_tissues,
        chr12_tissues,
        consistency,
    )
    markdown = report_text(
        regular,
        chr12,
        regular_tissues,
        chr12_tissues,
        consistency,
    )
    markdown_path = args.output_dir / "REPORT.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    pdf_path = args.output_dir / "annotation_personalization_exploratory_report.pdf"
    write_pdf(
        pdf_path,
        regular,
        chr12,
        regular_tissues,
        chr12_tissues,
        consistency,
    )
    print(markdown_path.resolve())
    print(pdf_path.resolve())


if __name__ == "__main__":
    main()
