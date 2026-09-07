#!/usr/bin/env python3
# Select the best complete EVAL/TEST retraining epoch per participant using one
# EVAL bin's weighted F1, then summarize that fixed epoch on TEST:
# Selected participants using the default bin 4:
#   python3 scripts/select_best_eval_bin4_epochs.py \
#     --results-root scripts/results/downsampled_standard_ratio_0p25_lr_3e-05 \
#     --extra-results-root scripts/results/downsampled_breast_epochs_4_5_ratio_0p25_lr_3e-05 \
#     --participants Z000000V6 Z000000VG Z000000VL Z0000043E \
#     --output-dir scripts/results/downsampled_breast_epochs_4_5_ratio_0p25_lr_3e-05/eval_bin4_selection
# All discovered participants using bin 5:
#   python3 scripts/select_best_eval_bin4_epochs.py \
#     --results-root scripts/results/downsampled_standard_ratio_0p25_lr_3e-05 \
#     --extra-results-root scripts/results/downsampled_breast_epochs_4_5_ratio_0p25_lr_3e-05 \
#     --selection-bin 5 \
#     --output-dir scripts/results/downsampled_standard_ratio_0p25_lr_3e-05/eval_bin5_selected_epoch_all_tissues
# Omitting --participants applies the rule to every discovered participant and
# writes per-sample, overall-average, and tissue-average TEST tables.

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

from create_downsampled_standard_report import (
    infer_tissue,
    section_map,
    stage_from_path,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select each participant's best complete EVAL/TEST retraining epoch "
            "on one EVAL bin and report that fixed epoch on TEST."
        )
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--extra-results-root", action="append", type=Path, default=[]
    )
    parser.add_argument("--selection-bin", type=int, choices=range(1, 6), default=4)
    parser.add_argument("--participants", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_sample(path: Path, participant: str) -> dict:
    if not path.exists():
        return {}
    payload = ast.literal_eval(path.read_text(encoding="utf-8"))
    return payload.get(participant, {})


def discover_participants(roots: list[Path]) -> list[str]:
    return sorted(
        {
            child.name
            for root in roots
            if root.exists()
            for child in root.iterdir()
            if child.is_dir() and (child / "eval_eval_objects_dict.txt").exists()
        }
    )


def merge_results(roots: list[Path], participant: str, split: str) -> dict:
    merged = {}
    for root in roots:
        path = root / participant / f"{split}_eval_objects_dict.txt"
        merged.update(read_sample(path, participant))
    return merged


def predicted_metric(prediction_eval: dict, section: str) -> dict:
    sections = section_map(prediction_eval)
    raw_section = next(raw for raw, rank in sections.items() if rank == section)
    return summarize(prediction_eval[raw_section], "predicted_class")


def selection_stage_map(paths: dict) -> dict[str, str]:
    raw_stages = {stage_from_path(path) for path in paths}
    checkpoints = sorted(
        (stage for stage in raw_stages if stage.startswith("checkpoint-")),
        key=lambda stage: int(stage.split("-", 1)[1]),
    )
    mapping = {"pretrain": "pretrain"}
    for index, checkpoint in enumerate(checkpoints, start=1):
        mapping[checkpoint] = f"E{index}"
    for stage in raw_stages:
        if stage.startswith("E") and stage[1:].isdigit():
            mapping[stage] = stage
        elif stage not in mapping:
            mapping[stage] = stage
    return mapping


def source_priority(path: str) -> int:
    return int(Path(path).parent.name.startswith("epoch-"))


def test_path_for_eval(eval_path: str) -> str:
    name = Path(eval_path).name.replace("eval_", "test_", 1)
    return str(Path(eval_path).with_name(name))


def paired_test_result(eval_path: str, test_results: dict) -> tuple[str, dict] | None:
    test_path = test_path_for_eval(eval_path)
    if test_path not in test_results:
        return None
    return test_path, test_results[test_path]


def retraining_stage_rows(eval_results: dict, selection_bin: int) -> pd.DataFrame:
    stages = selection_stage_map(eval_results)
    rows = []
    for path, prediction_eval in eval_results.items():
        stage = stages[stage_from_path(path)]
        if stage == "pretrain":
            continue
        metric = predicted_metric(prediction_eval, f"bin_{selection_bin}")
        rows.append(
            {
                "epoch": stage,
                "eval_selection_weighted_f1": metric["weighted_f1"],
                "eval_selection_support": metric["support"],
                "eval_prediction_path": path,
                "source_priority": source_priority(path),
            }
        )
    if not rows:
        raise ValueError("No retraining epochs found in EVAL results")
    frame = pd.DataFrame(rows)
    return (
        frame.sort_values(
            ["epoch", "source_priority", "eval_prediction_path"],
            ascending=[True, False, False],
        )
        .drop_duplicates("epoch", keep="first")
        .reset_index(drop=True)
    )


def test_result_for_stage(test_results: dict, stage: str) -> tuple[str, dict] | None:
    matches = [
        (path, prediction_eval)
        for path, prediction_eval in test_results.items()
        if stage_from_path(path) == stage
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: (source_priority(item[0]), item[0]))


def select_participant(
    roots: list[Path], participant: str, selection_bin: int
) -> tuple[dict, list[dict]]:
    eval_results = merge_results(roots, participant, "eval")
    test_results = merge_results(roots, participant, "test")
    if not eval_results:
        raise ValueError(f"{participant}: no EVAL results found")
    candidates = retraining_stage_rows(eval_results, selection_bin)
    candidates["paired_test_available"] = candidates["eval_prediction_path"].map(
        lambda path: paired_test_result(path, test_results) is not None
    )
    incomplete_epochs = candidates.loc[
        ~candidates["paired_test_available"], "epoch"
    ].tolist()
    eligible = candidates.loc[candidates["paired_test_available"]].copy()
    if eligible.empty:
        raise ValueError(f"{participant}: no complete EVAL/TEST retraining pairs found")
    eligible["epoch_number"] = eligible["epoch"].str.removeprefix("E").astype(int)
    selected = eligible.sort_values(
        ["eval_selection_weighted_f1", "epoch_number"], ascending=[False, True]
    ).iloc[0]
    selected_epoch = selected["epoch"]
    selected_test = paired_test_result(selected["eval_prediction_path"], test_results)
    tissue = infer_tissue(eval_results)
    selection = {
        "participant": participant,
        "tissue": tissue,
        "selected_epoch": selected_epoch,
        "selection_bin": selection_bin,
        "eval_selection_weighted_f1": selected["eval_selection_weighted_f1"],
        "eval_selection_support": int(selected["eval_selection_support"]),
        "selected_test_available": selected_test is not None,
        "ignored_incomplete_epochs": ",".join(incomplete_epochs),
        "eval_prediction_path": selected["eval_prediction_path"],
        "test_prediction_path": selected_test[0] if selected_test else "",
    }
    if selected_test is None:
        return selection, []
    pretrain_test = test_result_for_stage(test_results, "pretrain")
    rows = selected_test_rows(participant, tissue, selected_epoch, selected_test)
    if pretrain_test is not None:
        rows.extend(selected_test_rows(participant, tissue, "pretrain", pretrain_test))
        rows.extend(
            selected_test_rows(
                participant, tissue, "atlas", pretrain_test, "mean_label"
            )
        )
    fixed_e3_rows = candidates.loc[candidates["epoch"].eq("E3")]
    fixed_e3_test = None
    if not fixed_e3_rows.empty:
        fixed_e3_test = paired_test_result(
            fixed_e3_rows.iloc[0]["eval_prediction_path"], test_results
        )
    if fixed_e3_test is not None:
        rows.extend(
            selected_test_rows(participant, tissue, "fixed_e3", fixed_e3_test)
        )
    return selection, rows


def selected_test_rows(
    participant: str,
    tissue: str,
    stage: str,
    test_result: tuple[str, dict],
    prediction_type: str = "predicted_class",
) -> list[dict]:
    prediction_path, prediction_eval = test_result
    sections = section_map(prediction_eval)
    rows = []
    for raw_section, section in sections.items():
        metric = summarize(prediction_eval[raw_section], prediction_type)
        rows.append(
            {
                "participant": participant,
                "tissue": tissue,
                "stage": stage,
                "section_rank": section,
                "support": metric["support"],
                "weighted_f1": metric["weighted_f1"],
                "prediction_path": prediction_path,
            }
        )
    return rows


def comparison_table(metric_rows: pd.DataFrame) -> pd.DataFrame:
    if metric_rows.empty:
        return pd.DataFrame()
    reference_stages = {"pretrain", "atlas", "fixed_e3"}
    selected = metric_rows.loc[~metric_rows["stage"].isin(reference_stages)].rename(
        columns={"stage": "selected_epoch", "weighted_f1": "selected_weighted_f1"}
    )
    columns = [
        "participant", "tissue", "selected_epoch", "section_rank", "support",
        "selected_weighted_f1", "prediction_path",
    ]
    result = selected[columns]
    for stage in ["pretrain", "atlas", "fixed_e3"]:
        reference = metric_rows.loc[metric_rows["stage"].eq(stage), [
            "participant", "section_rank", "weighted_f1"
        ]].rename(columns={"weighted_f1": f"{stage}_weighted_f1"})
        result = result.merge(
            reference,
            on=["participant", "section_rank"],
            how="left",
            validate="one_to_one",
        )
    result["selected_minus_pretrain"] = (
        result["selected_weighted_f1"] - result["pretrain_weighted_f1"]
    )
    result["selected_minus_fixed_e3"] = (
        result["selected_weighted_f1"] - result["fixed_e3_weighted_f1"]
    )
    result["selected_minus_atlas"] = (
        result["selected_weighted_f1"] - result["atlas_weighted_f1"]
    )
    return result


def average_table(comparisons: pd.DataFrame, total_participants: int) -> pd.DataFrame:
    if comparisons.empty:
        return pd.DataFrame()
    result = comparisons.groupby("section_rank", as_index=False).agg(
        participants=("participant", "nunique"),
        pretrain_weighted_f1=("pretrain_weighted_f1", "mean"),
        pretrain_weighted_f1_std=("pretrain_weighted_f1", "std"),
        atlas_weighted_f1=("atlas_weighted_f1", "mean"),
        atlas_weighted_f1_std=("atlas_weighted_f1", "std"),
        fixed_e3_weighted_f1=("fixed_e3_weighted_f1", "mean"),
        fixed_e3_weighted_f1_std=("fixed_e3_weighted_f1", "std"),
        selected_weighted_f1=("selected_weighted_f1", "mean"),
        selected_weighted_f1_std=("selected_weighted_f1", "std"),
        selected_minus_pretrain=("selected_minus_pretrain", "mean"),
        selected_minus_atlas=("selected_minus_atlas", "mean"),
        selected_minus_fixed_e3=("selected_minus_fixed_e3", "mean"),
        support=("support", "sum"),
    )
    result["total_participants"] = total_participants
    result["complete"] = result["participants"].eq(total_participants)
    order = {"all_results": 0, **{f"bin_{index}": index for index in range(1, 6)}}
    return result.sort_values("section_rank", key=lambda values: values.map(order))


def tissue_average_table(
    comparisons: pd.DataFrame, selections: pd.DataFrame
) -> pd.DataFrame:
    if comparisons.empty:
        return pd.DataFrame()
    totals = selections.groupby("tissue")["participant"].nunique()
    result = comparisons.groupby(["tissue", "section_rank"], as_index=False).agg(
        participants=("participant", "nunique"),
        pretrain_weighted_f1=("pretrain_weighted_f1", "mean"),
        pretrain_weighted_f1_std=("pretrain_weighted_f1", "std"),
        atlas_weighted_f1=("atlas_weighted_f1", "mean"),
        atlas_weighted_f1_std=("atlas_weighted_f1", "std"),
        fixed_e3_weighted_f1=("fixed_e3_weighted_f1", "mean"),
        fixed_e3_weighted_f1_std=("fixed_e3_weighted_f1", "std"),
        selected_weighted_f1=("selected_weighted_f1", "mean"),
        selected_weighted_f1_std=("selected_weighted_f1", "std"),
        selected_minus_pretrain=("selected_minus_pretrain", "mean"),
        selected_minus_atlas=("selected_minus_atlas", "mean"),
        selected_minus_fixed_e3=("selected_minus_fixed_e3", "mean"),
        support=("support", "sum"),
    )
    result["total_participants"] = result["tissue"].map(totals)
    result["complete"] = result["participants"].eq(result["total_participants"])
    order = {"all_results": 0, **{f"bin_{index}": index for index in range(1, 6)}}
    result["section_order"] = result["section_rank"].map(order)
    return result.sort_values(["tissue", "section_order"]).drop(
        columns="section_order"
    )


def write_outputs(output_dir: Path, selections: list[dict], rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_frame = pd.DataFrame(selections)
    metric_frame = pd.DataFrame(rows)
    comparisons = comparison_table(metric_frame)
    average = average_table(comparisons, len(selection_frame))
    tissue_average = tissue_average_table(comparisons, selection_frame)
    selection_frame.to_csv(output_dir / "selected_epochs.csv", index=False)
    metric_frame.to_csv(output_dir / "selected_test_metric_rows.csv", index=False)
    comparisons.to_csv(output_dir / "selected_test_comparisons.csv", index=False)
    average.to_csv(output_dir / "selected_test_average.csv", index=False)
    tissue_average.to_csv(output_dir / "selected_test_tissue_average.csv", index=False)


def main() -> None:
    args = parse_args()
    roots = [args.results_root, *args.extra_results_root]
    participants = args.participants or discover_participants(roots)
    selections = []
    rows = []
    for participant in participants:
        selection, participant_rows = select_participant(
            roots, participant, args.selection_bin
        )
        selections.append(selection)
        rows.extend(participant_rows)
    write_outputs(args.output_dir, selections, rows)
    selection_frame = pd.DataFrame(selections)
    print(selection_frame[
        ["participant", "selected_epoch", "eval_selection_weighted_f1", "selected_test_available"]
    ].to_string(index=False))
    missing = selection_frame.loc[
        ~selection_frame["selected_test_available"], "participant"
    ].tolist()
    if missing:
        print(f"Missing selected TEST predictions: {', '.join(missing)}")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
