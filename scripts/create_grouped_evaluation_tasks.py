#!/usr/bin/env python3
"""Create tissue/group task files for grouped prediction evaluation arrays."""

import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-base", type=Path, default=Path("/sci/labs/michall/roeizucker/token_cls_models2"))
    parser.add_argument("--comparison-dicts-base", type=Path, default=Path("/sci/labs/michall/roeizucker/comparison_dicts"))
    parser.add_argument("--output-root-base", type=Path, required=True)
    parser.add_argument("--mode", choices=["variability_free", "variability_bins"], required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--tissues", default=None, help="Comma-separated tissue names to include.")
    parser.add_argument("--groups", default=None, help="Comma-separated group names to include.")
    return parser.parse_args()


def wanted_values(raw_values):
    if raw_values is None:
        return None
    return {value.strip() for value in raw_values.split(",") if value.strip()}


def group_name(run_name):
    if re.search(r"^[^_]+_no_pretraining_retrain_", run_name):
        return "no_pretraining"
    match = re.search(r"^[^_]+_epoch-(\d+)-step-\d+_retrain_", run_name)
    if match:
        return f"epoch_{int(match.group(1))}_pretraining"
    match = re.search(r"^[^_]+_epoch-(\d+)-step-\d+_pretrain_prediction$", run_name)
    if match:
        return f"first_phase_epoch_{int(match.group(1))}"
    return None


def group_sort_key(group):
    if group == "no_pretraining":
        return (0, 0)
    if group.startswith("first_phase_epoch_"):
        return (1, int(group.split("_")[-1]))
    if group.startswith("epoch_"):
        return (2, int(group.split("_")[1]))
    return (9, group)


def result_exists(output_root_base, tissue, mode, group):
    return (output_root_base / tissue / mode / group / "eval_objects_dict.txt").exists()


def discover_tasks(args):
    wanted_tissues = wanted_values(args.tissues)
    wanted_groups = wanted_values(args.groups)
    tasks = set()
    for tissue_dir in sorted(args.results_base.glob("_token_cls_*_kmer")):
        tissue = tissue_dir.name.removeprefix("_token_cls_").removesuffix("_kmer")
        if wanted_tissues is not None and tissue not in wanted_tissues:
            continue
        if not (args.comparison_dicts_base / tissue).exists():
            continue
        for result_file in tissue_dir.glob("*/epoch-*/eval_predictions.csv.gitbackup"):
            group = group_name(result_file.parents[1].name)
            if group is None:
                continue
            if wanted_groups is not None and group not in wanted_groups:
                continue
            if not args.include_existing and result_exists(args.output_root_base, tissue, args.mode, group):
                continue
            tasks.add((tissue, group))
    return sorted(tasks, key=lambda task: (task[0], group_sort_key(task[1])))


def write_tasks(task_file, tasks):
    task_file.parent.mkdir(parents=True, exist_ok=True)
    with task_file.open("w") as handle:
        for tissue, group in tasks:
            handle.write(f"{tissue}\t{group}\n")


def main():
    args = parse_args()
    tasks = discover_tasks(args)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    write_tasks(args.task_file, tasks)
    print(args.task_file)
    print(len(tasks))


if __name__ == "__main__":
    main()

# Example, create all remaining variability-bin tasks:
# python3 scripts/create_grouped_evaluation_tasks.py \
#   --mode variability_bins \
#   --output-root-base /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/results/all_tissues_variability_bins \
#   --task-file /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/scripts/outputs/variability_bins_tasks.tsv
#
# Example, create only five test tasks:
# python3 scripts/create_grouped_evaluation_tasks.py \
#   --mode variability_bins \
#   --output-root-base /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/results/all_tissues_variability_bins \
#   --task-file /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/scripts/outputs/variability_bins_tasks_5.tsv \
#   --limit 5
#
# Example, create tasks for specific groups:
# python3 scripts/create_grouped_evaluation_tasks.py \
#   --mode variability_bins \
#   --output-root-base /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/results/all_tissues_variability_bins \
#   --task-file /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/scripts/outputs/variability_bins_epoch3_tasks.tsv \
#   --groups epoch_3_pretraining
