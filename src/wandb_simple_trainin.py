# W&B sweep runner for existing YAML training configs.
# Example direct smoke run:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/src/wandb_sweep_training.py --base-config /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/wandb_classification_training.yaml --sample-name R3
# Example restart/resume run:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/src/wandb_sweep_training.py --base-config /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/wandb_classification_training.yaml --sample-name R3 --wandb-run-id R3-sweep-001 --resume
# Example explicit hyperparameter reconnect run:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/src/wandb_sweep_training.py --base-config /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/wandb_classification_full_training.yaml --sample-name Z000000R3 --wandb-run-id cjfqy9u7 --resume --learning-rate 1e-6 --batch-size 1
# Example explicit LoRA override run:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/src/wandb_sweep_training.py --base-config /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/wandb_classification_full_training.yaml --sample-name Z000000R3 --wandb-run-id lora-r4-alpha16 --resume --learning-rate 1e-5 --batch-size 5 --lora-r 4 --lora-alpha 16 --lora-dropout 0.1 --lora-target-modules query,value
# Example dry-run validation:
# python3 /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/src/wandb_sweep_training.py --base-config /cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code/configs/wandb_classification_training.yaml --sample-name R3 --mode disabled --wandb-run-id dry-run-r3 --resume --dry-run
# Example W&B sweep agent use:
# wandb agent ENTITY/PROJECT/SWEEP_ID

import argparse
import copy
import os

os.environ.setdefault("WANDB_LOG_MODEL", "false")
os.environ.setdefault("WANDB_WATCH", "false")

import re
import sys
from typing import Any, Callable, Dict

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.training import run_training
from src.utils.wandb_utils import prepare_wandb_environment


SweepApplier = Callable[[Dict[str, Any], Any], None]


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_learning_rate(cfg: Dict[str, Any], value: Any) -> None:
    cfg["train"]["learning_rate"] = float(value)


def set_batch_size(cfg: Dict[str, Any], value: Any) -> None:
    batch_size = int(value)
    cfg["train"]["per_device_train_batch_size"] = batch_size
    cfg["train"]["per_device_eval_batch_size"] = batch_size


def set_train_batch_size(cfg: Dict[str, Any], value: Any) -> None:
    cfg["train"]["per_device_train_batch_size"] = int(value)


def set_eval_batch_size(cfg: Dict[str, Any], value: Any) -> None:
    cfg["train"]["per_device_eval_batch_size"] = int(value)


def set_num_train_epochs(cfg: Dict[str, Any], value: Any) -> None:
    cfg["train"]["num_train_epochs"] = int(value)


def set_top_rows(cfg: Dict[str, Any], value: Any) -> None:
    cfg["task"]["top_rows"] = int(value)


def set_freeze_model(cfg: Dict[str, Any], value: Any) -> None:
    cfg["model"]["freeze_model"] = parse_bool(value)


def set_use_lora(cfg: Dict[str, Any], value: Any) -> None:
    cfg["model"]["use_lora"] = parse_bool(value)


def set_lora_value(cfg: Dict[str, Any], key: str, value: Any) -> None:
    cfg["model"].setdefault("lora", {})[key] = value


def set_lora_r(cfg: Dict[str, Any], value: Any) -> None:
    set_lora_value(cfg, "r", int(value))


def set_lora_alpha(cfg: Dict[str, Any], value: Any) -> None:
    set_lora_value(cfg, "lora_alpha", int(value))


def set_lora_dropout(cfg: Dict[str, Any], value: Any) -> None:
    set_lora_value(cfg, "lora_dropout", float(value))


def set_lora_target_modules(cfg: Dict[str, Any], value: Any) -> None:
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    set_lora_value(cfg, "target_modules", list(value))


def set_min_cpg_sites(cfg: Dict[str, Any], value: Any) -> None:
    cfg["train"]["min_number_of_cpg_sites"] = int(value)


def set_save_steps(cfg: Dict[str, Any], value: Any) -> None:
    cfg["train"]["save_stratagy"] = "steps"
    cfg["train"]["number_of_steps"] = int(value)


def set_save_total_limit(cfg: Dict[str, Any], value: Any) -> None:
    cfg["train"]["save_total_limit"] = int(value)


def set_max_steps(cfg: Dict[str, Any], value: Any) -> None:
    cfg["train"]["max_steps"] = int(value)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def get_sweep_appliers() -> Dict[str, SweepApplier]:
    return {
        "learning_rate": set_learning_rate,
        "batch_size": set_batch_size,
        "per_device_train_batch_size": set_train_batch_size,
        "per_device_eval_batch_size": set_eval_batch_size,
        "num_train_epochs": set_num_train_epochs,
        "top_rows": set_top_rows,
        "freeze_model": set_freeze_model,
        "use_lora": set_use_lora,
        "lora_r": set_lora_r,
        "lora_alpha": set_lora_alpha,
        "lora_dropout": set_lora_dropout,
        "lora_target_modules": set_lora_target_modules,
        "min_number_of_cpg_sites": set_min_cpg_sites,
        "save_steps": set_save_steps,
        "number_of_steps": set_save_steps,
        "save_total_limit": set_save_total_limit,
        "max_steps": set_max_steps,
    }


def get_cli_sweep_params(args: argparse.Namespace) -> Dict[str, Any]:
    params = {}
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.max_steps is not None:
        params["max_steps"] = args.max_steps
    if args.lora_r is not None:
        params["lora_r"] = args.lora_r
    if args.lora_alpha is not None:
        params["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        params["lora_dropout"] = args.lora_dropout
    if args.lora_target_modules is not None:
        params["lora_target_modules"] = args.lora_target_modules
    return params


def apply_sweep_params(cfg: Dict[str, Any], sweep_params: Dict[str, Any]) -> None:
    appliers = get_sweep_appliers()
    unused_params = []
    for name, value in sweep_params.items():
        applier = appliers.get(name)
        if applier is None:
            unused_params.append(name)
            continue
        applier(cfg, value)
    if unused_params:
        print("unused sweep params:", sorted(unused_params), flush=True)


def get_task_prefix(cfg: Dict[str, Any]) -> str:
    model_type = cfg.get("model", {}).get("model_type", "")
    if model_type == "classification_analysis":
        return "cls"
    if model_type == "regression_analysis":
        return "reg"
    return "task"


def get_run_name(sample_name: str, cfg: Dict[str, Any]) -> str:
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    task_prefix = get_task_prefix(cfg)
    lr = format_value(train_cfg.get("learning_rate"))
    batch_size = train_cfg.get("per_device_train_batch_size")
    freeze_tag = "frz" if model_cfg.get("freeze_model", False) else "full"
    lora_tag = "lora" if model_cfg.get("use_lora", False) else "base"
    return sanitize_name(f"{sample_name}-{task_prefix}-lr{lr}-bs{batch_size}-{freeze_tag}-{lora_tag}")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.0e}"
    return str(value)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return name.strip("-")


def set_unique_output_dir(cfg: Dict[str, Any], run_id: str) -> None:
    output_dir = cfg["paths"].get("output_dir")
    if not output_dir:
        return
    cfg["paths"]["output_dir"] = os.path.join(output_dir, run_id)


def set_resume_training(cfg: Dict[str, Any], should_resume: bool) -> None:
    if should_resume:
        cfg["train"]["continue_from_last"] = True


def get_requested_run_id(args: argparse.Namespace) -> str | None:
    return args.wandb_run_id or os.environ.get("WANDB_RUN_ID")


def get_resume_mode(should_resume: bool) -> str | None:
    if should_resume:
        return "allow"
    return None


def get_effective_run_id(run: Any, requested_run_id: str | None) -> str:
    return requested_run_id or run.id


def print_dry_run_summary(cfg: Dict[str, Any], run_name: str, run_id: str) -> None:
    print("dry run only; training was not started", flush=True)
    print("wandb run id:", run_id, flush=True)
    print("run name:", run_name, flush=True)
    print("output_dir:", cfg["paths"].get("output_dir"), flush=True)
    print("continue_from_last:", cfg["train"].get("continue_from_last"), flush=True)
    print("wandb tags:", cfg.get("wandb", {}).get("tags"), flush=True)


def prepare_initial_wandb_environment(
    wandb_config: Dict[str, Any],
    sample_name: str,
    project: str | None,
    entity: str | None,
    group: str | None,
    mode: str | None,
) -> None:
    initial_config = copy.deepcopy(wandb_config)
    initial_config["enabled"] = True
    if project is not None:
        initial_config["project"] = project
    if entity is not None:
        initial_config["entity"] = entity
    if group is not None:
        initial_config["group"] = group
    if mode is not None:
        initial_config["mode"] = mode
    metadata = {
        "project": initial_config.get("project"),
        "entity": initial_config.get("entity"),
        "name": initial_config.get("name"),
        "group": initial_config.get("group"),
        "mode": initial_config.get("mode"),
        "tags": unique_tags([sample_name, "sweep", *(initial_config.get("tags") or [])]),
        "dir": initial_config.get("dir"),
    }
    prepare_wandb_environment(initial_config, metadata)


def update_wandb_config(
    cfg: Dict[str, Any],
    run_name: str,
    sample_name: str,
    project: str | None,
    entity: str | None,
    group: str | None,
    mode: str | None,
) -> None:
    wandb_cfg = cfg.setdefault("wandb", {})
    base_tags = wandb_cfg.get("tags") or []
    wandb_cfg["enabled"] = True
    wandb_cfg["name"] = run_name
    wandb_cfg["project"] = project or wandb_cfg.get("project")
    wandb_cfg["entity"] = entity or wandb_cfg.get("entity")
    wandb_cfg["group"] = group or wandb_cfg.get("group") or f"{sample_name}-{get_task_prefix(cfg)}-sweep"
    wandb_cfg["mode"] = mode or wandb_cfg.get("mode") or "online"
    wandb_cfg["tags"] = unique_tags([sample_name, "sweep", *base_tags])


def unique_tags(tags: list[Any]) -> list[str]:
    unique = []
    for tag in tags:
        if tag is None:
            continue
        tag = str(tag)
        if tag not in unique:
            unique.append(tag)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one W&B sweep trial using an existing YAML training config.")
    parser.add_argument("--base-config", required=True, help="Path to the base training YAML config.")
    parser.add_argument("--sample-name", help="Short sample name to include in W&B run names and tags.")
    parser.add_argument("--project", default=None, help="Optional W&B project override.")
    parser.add_argument("--entity", default=None, help="Optional W&B entity override.")
    parser.add_argument("--group", default=None, help="Optional W&B group override.")
    parser.add_argument("--mode", default=None, choices=["online", "offline", "disabled"], help="Optional W&B mode override.")
    parser.add_argument("--wandb-run-id", default=None, help="Stable W&B run id to use when resuming a stopped trial.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Optional direct learning-rate override for explicit non-agent runs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional direct train/eval batch-size override for explicit non-agent runs.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional direct max-steps override for explicit non-agent runs.")
    parser.add_argument("--lora-r", type=int, default=None, help="Optional direct LoRA rank override.")
    parser.add_argument("--lora-alpha", type=int, default=None, help="Optional direct LoRA alpha override.")
    parser.add_argument("--lora-dropout", type=float, default=None, help="Optional direct LoRA dropout override.")
    parser.add_argument("--lora-target-modules", default=None, help="Comma-separated direct LoRA target module override.")
    parser.add_argument("--resume", action="store_true", help="Resume the W&B run and continue training from the last checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Build and print the final config values without starting training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required to run sweeps. Install wandb in the training environment.") from exc

    base_cfg = load_config(args.base_config)
    cfg = copy.deepcopy(base_cfg)
    # sweep_params = dict(wandb.config)
    # sweep_params.update(get_cli_sweep_params(args))
    # apply_sweep_params(cfg, sweep_params)
    # run_name = get_run_name(args.sample_name, cfg)
    # update_wandb_config(cfg, run_name, args.sample_name, args.project, args.entity, args.group, args.mode)
    # print("starting sweep trial", run_name, "with output_dir", cfg["paths"].get("output_dir"), flush=True)
    run_training(cfg)
    # s


if __name__ == "__main__":
    main()
