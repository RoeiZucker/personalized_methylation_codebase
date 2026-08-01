# Create a TSV task file for W&B downsample/learning-rate Slurm arrays.
# Example full grid:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_wandb_downsample_lr_tasks.py --output-task-file /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/wandb_downsample_lr_tasks.tsv --ratios 1.0 0.5 0.25 --learning-rates 1e-6 1e-5 5e-5
# Example smoke grid:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/scripts/create_wandb_downsample_lr_tasks.py --output-task-file /tmp/wandb_smoke_tasks.tsv --ratios 1.0 --learning-rates 1e-5 --dataset-root /tmp/downsampled_datasets

import argparse
import os


DEFAULT_DATASET_ROOT = (
    "/sci/labs/michall/roeizucker/roei_test/token_cls_datasets_downsampled/"
    "_token_cls_Liver-Hepatocytes_kmer"
)
DEFAULT_SAMPLE_PREFIX = "R3"


def parse_args():
    parser = argparse.ArgumentParser(description="Create W&B Slurm task rows for downsample/LR runs.")
    parser.add_argument("--output-task-file", required=True)
    parser.add_argument("--ratios", nargs="+", required=True)
    parser.add_argument("--learning-rates", nargs="+", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sample-prefix", default=DEFAULT_SAMPLE_PREFIX)
    return parser.parse_args()


def ratio_tag(ratio):
    return str(ratio).replace(".", "p")


def lr_tag(learning_rate):
    return str(learning_rate).replace("-", "m").replace(".", "p")


def dataset_path(dataset_root, ratio):
    ratio_name = ratio_tag(ratio)
    dataset_name = (
        "Z000000R3_retrain_token_cls_Liver-Hepatocytes_kmer_seq_5400_"
        f"datasets_train_ratio_{ratio_name}"
    )
    return os.path.join(dataset_root, dataset_name)


def task_line(args, ratio, learning_rate):
    ratio_name = ratio_tag(ratio)
    learning_rate_name = lr_tag(learning_rate)
    sample_name = f"{args.sample_prefix}-ratio{ratio_name}"
    run_id = f"r3-downsample-ratio{ratio_name}-lr{learning_rate_name}-bs14"
    return "\t".join([ratio, learning_rate, dataset_path(args.dataset_root, ratio), sample_name, run_id])


def create_task_lines(args):
    lines = []
    for ratio in args.ratios:
        for learning_rate in args.learning_rates:
            lines.append(task_line(args, ratio, learning_rate))
    return lines


def write_lines(path, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main():
    args = parse_args()
    lines = create_task_lines(args)
    write_lines(args.output_task_file, lines)
    print({"task_file": args.output_task_file, "tasks": len(lines)}, flush=True)


if __name__ == "__main__":
    main()
