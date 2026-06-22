import argparse
import os
import shlex
import sys
from pathlib import Path

import yaml


NO_PRETRAINING_CONFIG_NAME = "no_pretraining"
DEFAULT_OUTPUT_FILENAME = "eval_predictions.csv.gitbackup"
DEFAULT_PYTHON_COMMAND = "/home/users/roeizucker/tests/my_env/bin/python"
DEFAULT_SBATCH_PREFIX = (
    "sbatch --container-mounts=/shared:/shared --partition=compute-gpu "
    "--qos=owner_95 --gres=gpu:1 --time=95:10:00 "
    "--container-image=docker://nvcr.io/nvidia/pytorch:25.10-py3"
)


def _default_repo_root():
    current = Path(__file__).absolute()
    for candidate in current.parents:
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    return current.parents[1]


REPO_ROOT = _default_repo_root()
DEFAULT_RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run_pretrain_checkpoint_prediction.py"


def _load_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _yaml_files(path):
    if not os.path.isdir(path):
        return []
    return [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if name.endswith(".yaml")
    ]


def _sample_from_analysis_name(analysis_name, marker):
    if analysis_name and marker in analysis_name:
        return analysis_name.split(marker, 1)[0]
    return None


def _sample_from_yaml_path(yaml_path, marker):
    name = os.path.basename(yaml_path)
    if marker in name:
        return name.split(marker, 1)[0]
    return name.split("_", 1)[0]


def _pretrain_sample_name(yaml_path, cfg):
    analysis_name = cfg.get("task", {}).get("analysis_name", "")
    return (
        _sample_from_analysis_name(analysis_name, "_pretrain")
        or _sample_from_yaml_path(yaml_path, "_pretrain")
    )


def _no_pretraining_sample_name(yaml_path, cfg):
    analysis_name = cfg.get("task", {}).get("analysis_name", "")
    marker = f"_{NO_PRETRAINING_CONFIG_NAME}_retrain"
    return (
        _sample_from_analysis_name(analysis_name, marker)
        or _sample_from_yaml_path(yaml_path, marker)
    )


def _epoch_checkpoint_dirs(model_dir, checkpoint_prefix):
    if not model_dir or not os.path.isdir(model_dir):
        return []
    return [
        os.path.join(model_dir, name)
        for name in sorted(os.listdir(model_dir))
        if name.startswith(checkpoint_prefix) and os.path.isdir(os.path.join(model_dir, name))
    ]


def _build_no_pretraining_config_by_sample(project_dir):
    retrain_training_dir = os.path.join(project_dir, "retrain_training")
    configs = {}
    for yaml_path in _yaml_files(retrain_training_dir):
        cfg = _load_yaml(yaml_path)
        analysis_name = cfg.get("task", {}).get("analysis_name", "")
        if f"_{NO_PRETRAINING_CONFIG_NAME}_retrain" not in analysis_name:
            continue
        sample = _no_pretraining_sample_name(yaml_path, cfg)
        configs[sample] = yaml_path
    return configs


def _quote_command(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def _runner_command(python_command, runner_script, config_path, checkpoint_path, output_path, overwrite):
    parts = [
        python_command,
        runner_script,
        "--config",
        config_path,
        "--checkpoint-path",
        checkpoint_path,
        "--output-path",
        output_path,
    ]
    if overwrite:
        parts.append("--overwrite")
    return _quote_command(parts)


def _job_name(sample, checkpoint_name):
    value = f"prepred:{sample}:{checkpoint_name}"
    return value.replace(" ", "_").replace("/", "-")[:120]


def _sbatch_command(sbatch_prefix, sample, checkpoint_name, runner_cmd):
    return f'{sbatch_prefix} --job-name="{_job_name(sample, checkpoint_name)}" --wrap="{runner_cmd}"'


def create_pretrain_prediction_commands(
    project_dir,
    output_base_dir=None,
    runner_script=DEFAULT_RUNNER_SCRIPT,
    python_command=DEFAULT_PYTHON_COMMAND,
    checkpoint_prefix="epoch-",
    output_filename=DEFAULT_OUTPUT_FILENAME,
    overwrite=False,
    no_sbatch=False,
    sbatch_prefix=DEFAULT_SBATCH_PREFIX,
    limit=None,
):
    project_dir = os.path.abspath(project_dir)
    pretrain_training_dir = os.path.join(project_dir, "pretrain_training")
    no_pretraining_configs = _build_no_pretraining_config_by_sample(project_dir)
    printed = 0
    skipped = 0

    for pretrain_yaml_path in _yaml_files(pretrain_training_dir):
        pretrain_cfg = _load_yaml(pretrain_yaml_path)
        sample = _pretrain_sample_name(pretrain_yaml_path, pretrain_cfg)
        eval_config_path = no_pretraining_configs.get(sample)
        if eval_config_path is None:
            print(f"# skipping {sample}: no matching no-pretraining retrain config found")
            skipped += 1
            continue

        pretrain_output_dir = pretrain_cfg.get("paths", {}).get("output_dir")
        checkpoint_dirs = _epoch_checkpoint_dirs(pretrain_output_dir, checkpoint_prefix)
        if not checkpoint_dirs:
            print(f"# skipping {sample}: no {checkpoint_prefix} checkpoints found in {pretrain_output_dir}")
            skipped += 1
            continue

        model_project_dir = os.path.dirname(pretrain_output_dir)
        current_output_base_dir = output_base_dir or model_project_dir

        for checkpoint_path in checkpoint_dirs:
            checkpoint_name = os.path.basename(checkpoint_path)
            analysis_output_dir = os.path.join(
                current_output_base_dir,
                f"{sample}_{checkpoint_name}_pretrain_prediction",
            )
            output_path = os.path.join(analysis_output_dir, checkpoint_name, output_filename)

            if os.path.exists(output_path) and not overwrite:
                print(f"# skipping existing {output_path}")
                skipped += 1
                continue

            runner_cmd = _runner_command(
                python_command=python_command,
                runner_script=runner_script,
                config_path=eval_config_path,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                overwrite=overwrite,
            )
            if no_sbatch:
                print(runner_cmd)
            else:
                print(_sbatch_command(sbatch_prefix, sample, checkpoint_name, runner_cmd))
            printed += 1

            if limit is not None and printed >= limit:
                return {"printed": printed, "skipped": skipped}

    return {"printed": printed, "skipped": skipped}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print commands for predicting existing token-classification pretrain epoch checkpoints."
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        help="Generated project directory containing pretrain_training and retrain_training YAML dirs.",
    )
    parser.add_argument(
        "--output-base-dir",
        help="Optional base directory for prediction outputs. Defaults to the model project directory.",
    )
    parser.add_argument("--runner-script", default=str(DEFAULT_RUNNER_SCRIPT))
    parser.add_argument("--python-command", default=DEFAULT_PYTHON_COMMAND)
    parser.add_argument("--checkpoint-prefix", default="epoch-")
    parser.add_argument("--output-filename", default=DEFAULT_OUTPUT_FILENAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-sbatch", action="store_true", help="Print plain runner commands instead of sbatch commands.")
    parser.add_argument("--sbatch-prefix", default=DEFAULT_SBATCH_PREFIX)
    parser.add_argument("--limit", type=int, help="Optional maximum number of commands to print.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = create_pretrain_prediction_commands(
        project_dir=args.project_dir,
        output_base_dir=args.output_base_dir,
        runner_script=args.runner_script,
        python_command=args.python_command,
        checkpoint_prefix=args.checkpoint_prefix,
        output_filename=args.output_filename,
        overwrite=args.overwrite,
        no_sbatch=args.no_sbatch,
        sbatch_prefix=args.sbatch_prefix,
        limit=args.limit,
    )
    print(f"# printed {summary['printed']} commands, skipped {summary['skipped']}")


if __name__ == "__main__":
    main()
