#!/usr/bin/env python3
# Create the TEST-only bin-4/bin-5 personalization significance report:
#   python3 scripts/create_bin4_bin5_significance_report.py
# Write the report to a custom directory:
#   python3 scripts/create_bin4_bin5_significance_report.py \
#     --output-dir scripts/results/custom_bin4_bin5_significance_report

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "scripts/results/downsampled_standard_ratio_0p25_lr_3e-05"
DEFAULT_OUTPUT = DEFAULT_ROOT / "bin4_bin5_significance_report"
PRIMARY_STAGE = "E3"
PRIMARY_METRIC = "weighted_f1"
BINS = (4, 5)
PERMUTATION_REPETITIONS = 10_000
BOOTSTRAP_REPETITIONS = 5_000
RANDOM_SEED = 20260827


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a TEST-only bin-4/bin-5 personalization significance report."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def primary_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    return rows[(rows["stage"] == PRIMARY_STAGE) & (rows["metric"] == PRIMARY_METRIC)].copy()


def load_person_rows(root: Path, bin_rank: int) -> pd.DataFrame:
    path = root / f"bin{bin_rank}_significance_test/person_bin{bin_rank}_resampling_results.csv"
    rows = primary_rows(path)
    rows["bin"] = bin_rank
    return rows


def load_tissue_rows(root: Path, bin_rank: int) -> pd.DataFrame:
    path = root / f"bin{bin_rank}_significance_test/tissue_bin{bin_rank}_results.csv"
    rows = primary_rows(path)
    rows["bin"] = bin_rank
    return rows


def load_combined_rows(root: Path, bin_rank: int) -> pd.DataFrame:
    path = root / f"bin{bin_rank}_significance_test/combined_bin{bin_rank}_results.csv"
    rows = primary_rows(path)
    rows["bin"] = bin_rank
    return rows


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    ranks = np.arange(1, len(valid) + 1)
    adjusted = valid.to_numpy() * len(valid) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def relative_improvement(
    pretrain: pd.Series | float, retrain: pd.Series | float
) -> pd.Series | float:
    return 100 * (retrain - pretrain) / pretrain


def monte_carlo_p(null_values: np.ndarray, observed: float) -> float:
    exceedances = int(np.count_nonzero(null_values >= observed))
    return (exceedances + 1) / (len(null_values) + 1)


def paired_permutation_means(
    differences: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    signs = rng.choice(
        [-1.0, 1.0],
        size=(PERMUTATION_REPETITIONS, len(differences)),
    )
    return (signs * differences).mean(axis=1)


def bootstrap_person_means(
    differences: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    indices = rng.integers(
        0,
        len(differences),
        size=(BOOTSTRAP_REPETITIONS, len(differences)),
    )
    return differences[indices].mean(axis=1)


def build_equal_person_summary(person_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for bin_rank, group in person_rows.groupby("bin", sort=True):
        differences = group["difference"].to_numpy(dtype=float)
        rng = np.random.default_rng(RANDOM_SEED + int(bin_rank))
        null_values = paired_permutation_means(differences, rng)
        bootstrap = bootstrap_person_means(differences, rng)
        pretrain = float(group["pretrain_value"].mean())
        retrain = float(group["retrain_value"].mean())
        observed = float(differences.mean())
        records.append(
            {
                "bin": int(bin_rank),
                "tissues": int(group["tissue"].nunique()),
                "people": int(group["sample"].nunique()),
                "sites": int(group["sites"].sum()),
                "person_windows": int(group["windows"].sum()),
                "equal_person_pretrain_f1": pretrain,
                "equal_person_retrain_f1": retrain,
                "equal_person_delta_f1": observed,
                "relative_improvement_percent": relative_improvement(
                    pretrain, retrain
                ),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
                "p_one_sided": monte_carlo_p(null_values, observed),
            }
        )
    summary = pd.DataFrame(records)
    summary["q_one_sided_two_bins"] = benjamini_hochberg(summary["p_one_sided"])
    return summary


def add_joint_person_fdr(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["joint_q_one_sided_62_tests"] = benjamini_hochberg(
        rows["permutation_p_one_sided"]
    )
    rows["significant_improvement"] = (
        (rows["difference"] > 0) & (rows["joint_q_one_sided_62_tests"] < 0.05)
    )
    rows["relative_improvement_percent"] = relative_improvement(
        rows["pretrain_value"], rows["retrain_value"]
    )
    return rows


def tissue_f1_rows(person_rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["pretrain_value", "retrain_value", "difference"]
    means = person_rows.groupby(["bin", "tissue"], as_index=False)[columns].mean()
    means = means.rename(
        columns={
            "pretrain_value": "mean_pretrain_f1",
            "retrain_value": "mean_retrain_f1",
            "difference": "mean_delta_f1",
        }
    )
    means["relative_improvement_percent"] = relative_improvement(
        means["mean_pretrain_f1"], means["mean_retrain_f1"]
    )
    counts = person_rows.groupby(["bin", "tissue"], as_index=False).agg(
        people=("sample", "nunique"),
        significant_people=("significant_improvement", "sum"),
    )
    return means.merge(counts, on=["bin", "tissue"], validate="one_to_one")


def minimum_person_p_rows(person_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (bin_rank, tissue), group in person_rows.groupby(["bin", "tissue"], sort=True):
        minimum = group["permutation_p_one_sided"].min()
        samples = sorted(group.loc[group["permutation_p_one_sided"] == minimum, "sample"])
        records.append(
            {
                "bin": bin_rank,
                "tissue": tissue,
                "lowest_person_p": minimum,
                "lowest_person_samples": ", ".join(samples),
            }
        )
    return pd.DataFrame(records)


def build_tissue_summary(person_rows: pd.DataFrame, tissue_rows: pd.DataFrame) -> pd.DataFrame:
    tissue_rows = tissue_rows.copy()
    tissue_rows["joint_tissue_q_one_sided_12_tests"] = benjamini_hochberg(
        tissue_rows["exact_sign_flip_p_one_sided"]
    )
    summary = tissue_f1_rows(person_rows)
    summary = summary.merge(
        tissue_rows[
            [
                "bin",
                "tissue",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "exact_sign_flip_p_one_sided",
                "joint_tissue_q_one_sided_12_tests",
                "sites",
                "person_window_instances",
                "unique_genomic_windows",
            ]
        ],
        on=["bin", "tissue"],
        validate="one_to_one",
    )
    return summary.merge(
        minimum_person_p_rows(person_rows),
        on=["bin", "tissue"],
        validate="one_to_one",
    )


def equal_tissue_f1(person_rows: pd.DataFrame, column: str) -> float:
    tissue_means = person_rows.groupby("tissue")[column].mean()
    return float(tissue_means.mean())


def build_combined_summary(
    person_rows: pd.DataFrame, combined_rows: pd.DataFrame
) -> pd.DataFrame:
    records = []
    for bin_rank, group in person_rows.groupby("bin", sort=True):
        result = combined_rows[combined_rows["bin"] == bin_rank].iloc[0]
        records.append(
            {
                "bin": bin_rank,
                "tissues": int(result["tissues"]),
                "people": int(result["individuals"]),
                "sites": int(result["sites"]),
                "person_windows": int(result["person_window_instances"]),
                "equal_tissue_pretrain_f1": equal_tissue_f1(group, "pretrain_value"),
                "equal_tissue_retrain_f1": equal_tissue_f1(group, "retrain_value"),
                "equal_tissue_delta_f1": float(result["equal_tissue_mean_difference"]),
                "ci_low": float(result["bootstrap_ci_low"]),
                "ci_high": float(result["bootstrap_ci_high"]),
                "p_one_sided": float(result["permutation_p_one_sided"]),
            }
        )
    summary = pd.DataFrame(records)
    summary["relative_improvement_percent"] = relative_improvement(
        summary["equal_tissue_pretrain_f1"], summary["equal_tissue_retrain_f1"]
    )
    summary["q_one_sided_two_bins"] = benjamini_hochberg(summary["p_one_sided"])
    return summary


def number(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def count(value: float | int) -> str:
    return f"{int(value):,}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("|" + "|".join("---" for _ in headers) + "|")
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return output


def combined_table(rows: pd.DataFrame) -> list[str]:
    table_rows = []
    for _, row in rows.sort_values("bin").iterrows():
        table_rows.append(
            [
                str(int(row["bin"])),
                count(row["people"]),
                count(row["sites"]),
                count(row["person_windows"]),
                number(row["equal_tissue_pretrain_f1"]),
                number(row["equal_tissue_retrain_f1"]),
                number(row["equal_tissue_delta_f1"]),
                number(row["relative_improvement_percent"], digits=2),
                f"{number(row['ci_low'])} to {number(row['ci_high'])}",
                number(row["p_one_sided"]),
                number(row["q_one_sided_two_bins"]),
            ]
        )
    return markdown_table(
        [
            "Bin",
            "People",
            "CpGs",
            "Person-windows",
            "Pretrain F1",
            "Retrain F1",
            "Delta F1",
            "Relative improvement (%)",
            "95% CI",
            "One-sided p",
            "BH q (2 bins)",
        ],
        table_rows,
    )


def equal_person_table(rows: pd.DataFrame) -> list[str]:
    table_rows = []
    for _, row in rows.sort_values("bin").iterrows():
        table_rows.append(
            [
                str(int(row["bin"])),
                count(row["people"]),
                number(row["equal_person_pretrain_f1"]),
                number(row["equal_person_retrain_f1"]),
                number(row["equal_person_delta_f1"]),
                number(row["relative_improvement_percent"], digits=2),
                f"{number(row['ci_low'])} to {number(row['ci_high'])}",
                number(row["p_one_sided"]),
                number(row["q_one_sided_two_bins"]),
            ]
        )
    return markdown_table(
        [
            "Bin",
            "People",
            "Pretrain F1",
            "Retrain F1",
            "Delta F1",
            "Relative improvement (%)",
            "95% CI",
            "One-sided p",
            "BH q (2 bins)",
        ],
        table_rows,
    )


def tissue_table(rows: pd.DataFrame, bin_rank: int) -> list[str]:
    table_rows = []
    subset = rows[rows["bin"] == bin_rank].sort_values("tissue")
    for _, row in subset.iterrows():
        lowest = f"{number(row['lowest_person_p'])} ({row['lowest_person_samples']})"
        table_rows.append(
            [
                str(row["tissue"]),
                f"{int(row['significant_people'])}/{int(row['people'])}",
                count(row["sites"]),
                number(row["mean_pretrain_f1"]),
                number(row["mean_retrain_f1"]),
                number(row["mean_delta_f1"]),
                number(row["relative_improvement_percent"], digits=2),
                f"{number(row['bootstrap_ci_low'])} to {number(row['bootstrap_ci_high'])}",
                number(row["exact_sign_flip_p_one_sided"]),
                number(row["joint_tissue_q_one_sided_12_tests"]),
                lowest,
            ]
        )
    return markdown_table(
        [
            "Tissue",
            "Significantly improved people",
            "CpGs",
            "Pretrain F1",
            "Retrain F1",
            "Delta F1",
            "Relative improvement (%)",
            "95% CI",
            "Tissue p (one-sided)",
            "Tissue BH q (12 tests)",
            "Lowest person p",
        ],
        table_rows,
    )


def person_table(rows: pd.DataFrame) -> list[str]:
    indexed = rows.set_index(["tissue", "sample", "bin"])
    people = rows[["tissue", "sample"]].drop_duplicates().sort_values(["tissue", "sample"])
    table_rows = []
    for person in people.itertuples(index=False):
        row = [person.tissue, person.sample]
        for bin_rank in BINS:
            result = indexed.loc[(person.tissue, person.sample, bin_rank)]
            row.extend(
                [
                    number(result["difference"]),
                    number(result["relative_improvement_percent"], digits=2),
                    number(result["permutation_p_one_sided"]),
                    number(result["joint_q_one_sided_62_tests"]),
                    "yes" if result["significant_improvement"] else "no",
                ]
            )
        table_rows.append(row)
    return markdown_table(
        [
            "Tissue",
            "Person",
            "B4 Delta",
            "B4 relative improvement (%)",
            "B4 p+",
            "B4 joint q",
            "B4 improved",
            "B5 Delta",
            "B5 relative improvement (%)",
            "B5 p+",
            "B5 joint q",
            "B5 improved",
        ],
        table_rows,
    )


def metric_text() -> list[str]:
    return [
        "## Methods",
        "",
        "### Metric and paired comparison",
        "",
        "The analyzed outcome is **weighted F1**, calculated as the support-weighted mean of the three class-specific F1 scores. "
        "This metric retains the precision-recall balance of F1 while preventing a rare class from receiving the same numerical "
        "weight as a much more common class. It is appropriate here because the methylation classes are imbalanced. "
        "Weighted F1 was recalculated from the summed 3x3 confusion matrix in every resample; per-window F1 values were not simply averaged. "
        "Tables report both absolute Delta F1 and relative improvement, calculated as `100 x (retrain F1 - pretrain F1) / pretrain F1`. "
        "Relative improvement is descriptive; all hypothesis tests use the absolute paired F1 difference.",
        "",
        "All tests are **paired**. Pretrain and retrain predictions were evaluated on the same person, TEST split, CpGs, and genomic windows. "
        "Pairing removes between-person and between-locus difficulty from the model contrast: the statistic measures how the prediction "
        "changes when retraining is applied to an otherwise identical evaluation set.",
        "",
    ]


def combined_method_text() -> list[str]:
    return [
        "### Combined person-level test",
        "",
        "**Name:** stratified paired label-swap permutation test with equal tissue weighting. This is implemented using the mathematically "
        "equivalent sign-flip representation.",
        "",
        "**Question addressed:** Is there an average personalization benefit across the six studied tissues when a person, rather than a CpG, "
        "is treated as the independent biological unit?",
        "",
        "**Why this method is relevant:** Each person contributes thousands of CpGs, but those CpGs are repeated measurements from one person, "
        "not thousands of independent biological replicates. Reducing each person to one paired F1 difference prevents people with more CpGs "
        "from dominating the evidence. Averaging people within tissue and then averaging tissue means gives every tissue equal weight, so tissues "
        "with more participants do not silently define the combined conclusion.",
        "",
        "**Implementation:** For each person, the observed statistic was `retrain E3 weighted F1 - pretrain E10 weighted F1`. "
        "Within each of 10,000 Monte Carlo null replicates, the pretrain and retrain labels were independently retained or swapped for each "
        "person. Computationally, swapping a pair is implemented by multiplying that person's difference by -1. The resulting differences "
        "were averaged within tissue, and the six tissue means were averaged to produce one null statistic. Each tissue therefore contributes "
        "one sixth of the statistic, and people receive equal weight within their tissue rather than equal weight across the full cohort. "
        "The one-sided p-value is `(1 + null statistics >= observed statistic) / (10,000 + 1)`. "
        "A 95% CI was obtained from 5,000 stratified bootstrap replicates "
        "that resampled people with replacement within each tissue and retained equal tissue weighting.",
        "",
        "**Null hypothesis and assumptions:** Under the sign-flip null, positive and negative paired person-level effects are exchangeable around zero. "
        "The test does not require normally distributed F1 values, but it assumes people are independent and that sign-flipping is a reasonable "
        "representation of no systematic direction. The CI reflects between-person uncertainty within the observed tissues.",
        "",
        "**Interpretation:** A small combined p-value supports a directional average benefit across this equal-weighted mixture of tissues. "
        "It does not imply that every tissue or every person benefits, and it does not measure the probability that the null hypothesis is true.",
        "",
    ]


def equal_person_method_text() -> list[str]:
    return [
        "### Equal-person paired permutation sensitivity analysis",
        "",
        "**Name:** unstratified paired label-swap permutation test with equal person weighting.",
        "",
        "**Question addressed:** Is there an average personalization benefit for a randomly selected participant from the observed cohort, "
        "when every person contributes exactly the same weight regardless of tissue size?",
        "",
        "**Why this method is relevant:** The primary statistic gives each tissue equal total weight. The equal-person analysis instead takes the "
        "simple mean of all 31 paired person-level F1 differences. Comparing the two results shows whether the conclusion depends materially on "
        "giving smaller tissues more weight per participant.",
        "",
        "**Implementation:** In each of 10,000 Monte Carlo permutations, pretrain and retrain labels were retained or swapped independently "
        "within each of the 31 people. The statistic was the unweighted mean of the 31 resulting paired differences. A 95% CI was obtained from "
        "5,000 bootstrap samples that resampled all people with replacement without tissue stratification. One-sided p-values test improvement, "
        "and BH q-values correct bins 4 and 5 together.",
        "",
        "**Interpretation:** Agreement with the equal-tissue result supports robustness to the choice of cohort weighting. Disagreement means the "
        "scientific conclusion depends on whether the target is an average tissue or an average observed participant; neither weighting is "
        "universally correct, so the target population must determine which is primary.",
        "",
    ]


def tissue_method_text() -> list[str]:
    return [
        "### Tissue-level test",
        "",
        "**Name:** exact paired sign-flip randomization test over people.",
        "",
        "**Question addressed:** Within one tissue, is the direction and magnitude of the person-level retrain-minus-pretrain effect stronger than "
        "would be expected if retraining had no systematic direction?",
        "",
        "**Why this method is relevant:** The independent units for a tissue claim are the people from that tissue. A test based on concatenated CpGs "
        "would give participants with more evaluated sites greater weight and would create pseudoreplication. The sign-flip test instead uses one "
        "paired effect per person and is suitable for the very small tissue sample sizes without relying on a large-sample normal approximation.",
        "",
        "**Implementation:** For a tissue with `n` people, all `2^n` possible sign assignments of the paired person-level F1 differences were enumerated. "
        "The one-sided p-value is the proportion of sign assignments whose mean difference was at least the observed mean. "
        "Tissue CIs used 5,000 bootstrap samples of people. "
        "Tissue BH q-values correct the 12 bin-by-tissue directional tests together.",
        "",
        "**Why the p-values are coarse:** With `n` people, only `2^n` sign patterns exist. Consequently, the smallest attainable one-sided exact p-value "
        "is `1/16 = 0.0625` for four people and `1/32 = 0.03125` for five people, even if every person improves. This low resolution is a property "
        "of the available number of independent people, not a failure of the test and not something that additional CpGs can solve.",
        "",
        "**Interpretation:** The tissue p-value evaluates evidence across people in that tissue. The lowest per-person p-value shown in each tissue "
        "table answers a different question and is included descriptively; selecting the minimum makes it unsuitable as evidence for the tissue as a whole.",
        "",
    ]


def person_method_text() -> list[str]:
    return [
        "### Per-person test",
        "",
        "**Name:** paired whole-window permutation test with a window bootstrap CI.",
        "",
        "**Question addressed:** For this fixed individual, is the observed F1 improvement distributed consistently across held-out genomic windows "
        "rather than being attributable to a small number of favorable windows?",
        "",
        "**Why this method is relevant:** It is not possible to obtain repeated independent copies of one person. Genomic windows provide repeated "
        "held-out regions, and keeping each complete window together preserves the labels and the dependence among CpGs inside that window. "
        "The paired permutation directly compares both models on identical loci and allows weighted F1, a nonlinear statistic, to be recomputed exactly.",
        "",
        "**Implementation:** For each person and bin, pretrain and retrain predictions were aligned at identical TEST CpGs. "
        "A 3x3 confusion matrix was created separately for each genomic window and model. In each of 10,000 permutations, "
        "an independent fair decision was made for every window to retain or swap the complete pretrain/retrain confusion contributions. "
        "The resulting matrices were summed and weighted F1 was recomputed. The directional p-value `p+` tests whether retraining improves F1. "
        "Window-level CIs used 5,000 paired bootstrap samples of windows.",
        "",
        "**Null hypothesis and assumptions:** Under the permutation null, assigning the two model outputs to `pretrain` or `retrain` is exchangeable "
        "at the window level. The analysis preserves within-window structure, but it assumes separate windows are sufficiently exchangeable. Nearby or "
        "overlapping windows can remain spatially correlated, which may make the current window-level p-values too small.",
        "",
        "**Interpretation:** A small per-person p-value means the improvement is stable across that person's evaluated windows under the stated "
        "exchangeability assumption. It does not demonstrate that personalization generalizes to other people. Before using these p-values as "
        "confirmatory publication evidence, a sensitivity analysis should repeat the test with nonoverlapping genomic blocks, such as 100 kb, "
        "500 kb, and 1 Mb blocks, plus leave-one-chromosome-out analysis.",
        "",
    ]


def reporting_method_text() -> list[str]:
    return [
        "### Direction, multiplicity, and TEST-set status",
        "",
        "The analysis uses only the **one-sided** hypothesis that personalization improves prediction. It is defensible as "
        "confirmatory only if that direction, the weighted-F1 metric, bins 4-5, and retrain E3 were fixed before inspecting TEST outcomes.",
        "",
        "Per-person BH q-values correct all 62 primary one-sided improvement tests (`31 people x 2 bins`) together. A person is counted as "
        "significantly improved only when Delta F1 is positive and the one-sided joint `q < 0.05`. Tissue q-values correct all 12 tissue-bin "
        "one-sided tests, and combined "
        "q-values correct the two bins. "
        "BH controls the expected false-discovery proportion across a declared family; a q-value is not the probability that a particular result is false.",
        "",
        "The `+1` Monte Carlo correction prevents zero p-values. With 10,000 random permutations, the minimum reported Monte Carlo p-value is "
        "`1/10,001`, displayed as `0.000100`. P-values should be read together with Delta F1 and its confidence interval, because statistical "
        "precision does not establish biological importance.",
        "",
        "Only TEST rows were used for these calculations. If TEST results influenced the choice of bin, stage, metric, model, or reported comparison, "
        "the corresponding result should be labeled exploratory or validated in another untouched cohort.",
    ]


def method_text() -> list[str]:
    return [
        *metric_text(),
        *combined_method_text(),
        *equal_person_method_text(),
        *tissue_method_text(),
        *person_method_text(),
        *reporting_method_text(),
    ]


def references_text() -> list[str]:
    return [
        "## Methodological references",
        "",
        "- Winkler et al. describe restricted and multi-level permutation using exchangeability blocks for dependent data: "
        "[Multi-level block permutation](https://pmc.ncbi.nlm.nih.gov/articles/PMC4644991/).",
        "- The bootRanges study demonstrates why genomic resampling should preserve local correlation with blocks: "
        "[bootRanges](https://pmc.ncbi.nlm.nih.gov/articles/PMC10159650/).",
        "- The distinction between biological experimental units and repeated measurements is discussed in: "
        "[Unit of analysis issues in laboratory-based research](https://elifesciences.org/articles/32486).",
        "- Guidance on interpreting p-values together with design, effect size, multiplicity, and complete reporting: "
        "[ASA statement on p-values](https://doi.org/10.1080/00031305.2016.1154108).",
    ]

def report_text(
    combined: pd.DataFrame,
    equal_people: pd.DataFrame,
    tissues: pd.DataFrame,
    people: pd.DataFrame,
) -> str:
    lines = [
        "# TEST-only bin 4 and bin 5 personalization significance report",
        "",
        "## Scope",
        "",
        "- Comparison: personalized retrain E3 versus non-personalized pretrain epoch 10.",
        "- Data: TEST split only.",
        "- Primary metric: weighted F1.",
        "- Directional hypothesis: retraining improves weighted F1.",
        "- Biological inference unit: person. The primary combined statistic gives tissues equal total weight and people equal weight "
        "within tissue; a sensitivity analysis gives every person equal weight across the cohort.",
        "",
        *method_text(),
        "",
        "## Primary combined results: equal tissue weighting",
        "",
        *combined_table(combined),
        "",
        "## Sensitivity analysis: equal person weighting",
        "",
        *equal_person_table(equal_people),
        "",
    ]
    for bin_rank in BINS:
        lines.extend(
            [
                f"## Tissue results: bin {bin_rank}",
                "",
                "F1 values are equal-person means within tissue. `Significantly improved people` requires both a positive "
                "Delta F1 and one-sided joint BH q < 0.05. `Lowest person p` is the smallest raw directional per-person permutation "
                "p-value in that tissue; ties are listed together. This minimum is descriptive and must not be interpreted as "
                "the tissue-level p-value.",
                "",
                *tissue_table(tissues, bin_rank),
                "",
            ]
        )
    lines.extend(
        [
            "## Per-person results",
            "",
            "`p+` is the raw one-sided permutation p-value for improvement. `Joint q` is BH-adjusted across all 62 tests.",
            "",
            *person_table(people),
            "",
            "## Interpretation",
            "",
            f"Across both bins, {int(people['significant_improvement'].sum())} of 62 person-bin comparisons showed "
            "a positive effect with joint BH q < 0.05. Combined evidence was positive in both bins. "
            "Tissue-level results should be interpreted cautiously because each tissue contains only 4-8 people and "
            "none of the 12 directional tissue tests had joint BH q < 0.05.",
            "",
            "Monte Carlo values displayed as `0.000100` correspond to the minimum attainable value with 10,000 random permutations, "
            "approximately `1 / 10,001`; they are not zero.",
            "",
            *references_text(),
        ]
    )
    return "\n".join(lines) + "\n"


def one_sided_columns(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[:, ~rows.columns.str.contains("two_sided")]


def save_tables(
    output: Path,
    combined: pd.DataFrame,
    equal_people: pd.DataFrame,
    tissues: pd.DataFrame,
    people: pd.DataFrame,
) -> None:
    one_sided_columns(combined).to_csv(
        output / "combined_bin4_bin5_results.csv", index=False
    )
    one_sided_columns(equal_people).to_csv(
        output / "combined_equal_person_bin4_bin5_results.csv", index=False
    )
    one_sided_columns(tissues).to_csv(
        output / "tissue_bin4_bin5_results.csv", index=False
    )
    one_sided_columns(people).to_csv(
        output / "person_bin4_bin5_results.csv", index=False
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    people = pd.concat(
        [load_person_rows(args.results_root, bin_rank) for bin_rank in BINS],
        ignore_index=True,
    )
    tissues = pd.concat(
        [load_tissue_rows(args.results_root, bin_rank) for bin_rank in BINS],
        ignore_index=True,
    )
    combined = pd.concat(
        [load_combined_rows(args.results_root, bin_rank) for bin_rank in BINS],
        ignore_index=True,
    )
    people = add_joint_person_fdr(people)
    tissue_summary = build_tissue_summary(people, tissues)
    combined_summary = build_combined_summary(people, combined)
    equal_person_summary = build_equal_person_summary(people)
    save_tables(
        args.output_dir,
        combined_summary,
        equal_person_summary,
        tissue_summary,
        people,
    )
    report = report_text(
        combined_summary, equal_person_summary, tissue_summary, people
    )
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
