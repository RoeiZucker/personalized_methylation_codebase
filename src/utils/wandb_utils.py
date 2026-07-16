import os
from typing import Any, Dict, Iterable, List


def get_wandb_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("wandb") or {}


def is_wandb_enabled(wandb_config: Dict[str, Any] | None) -> bool:
    if not wandb_config:
        return False
    return bool(wandb_config.get("enabled", False))


def get_wandb_report_to(wandb_config: Dict[str, Any] | None):
    if is_wandb_enabled(wandb_config):
        return "wandb"
    return "none"


def build_wandb_metadata(
    wandb_config: Dict[str, Any] | None,
    task_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    wandb_config = wandb_config or {}
    task_type = task_config.get("task_type")
    model_name = model_config.get("model_name")
    model_type = model_config.get("model_type")

    return {
        "project": wandb_config.get("project"),
        "entity": wandb_config.get("entity"),
        "name": wandb_config.get("name") or task_config.get("analysis_name"),
        "group": wandb_config.get("group") or get_default_group(task_type, model_name),
        "tags": get_wandb_tags(wandb_config.get("tags"), task_type, model_type, model_config.get("use_lora", False)),
        "mode": wandb_config.get("mode"),
        "dir": wandb_config.get("dir"),
    }


def get_default_group(task_type: str | None, model_name: str | None) -> str | None:
    parts = [part for part in [task_type, model_name] if part]
    if not parts:
        return None
    return "_".join(parts)


def get_wandb_tags(tags: Iterable[str] | None, task_type: str | None, model_type: str | None, use_lora: bool) -> List[str]:
    default_tags = [task_type, model_type, "lora" if use_lora else "full_finetune"]
    return _unique_tags([*(tags or []), *default_tags])


def _unique_tags(tags: Iterable[str]) -> List[str]:
    unique = []
    for tag in tags:
        if tag is None:
            continue
        tag = str(tag)
        if tag not in unique:
            unique.append(tag)
    return unique


def prepare_wandb_environment(wandb_config: Dict[str, Any] | None, metadata: Dict[str, Any]) -> None:
    if not is_wandb_enabled(wandb_config):
        return
    _set_env("WANDB_PROJECT", metadata.get("project"))
    _set_env("WANDB_ENTITY", metadata.get("entity"))
    _set_env("WANDB_NAME", metadata.get("name"))
    _set_env("WANDB_RUN_GROUP", metadata.get("group"))
    _set_env("WANDB_MODE", metadata.get("mode"))
    _set_env("WANDB_TAGS", ",".join(metadata.get("tags") or []))
    _set_env("WANDB_DIR", metadata.get("dir"))
    _set_env("WANDB_LOG_MODEL", wandb_config.get("log_model", False))
    _set_env("WANDB_WATCH", wandb_config.get("watch", False))


def _set_env(name: str, value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, bool):
        value = str(value).lower()
    os.environ[name] = str(value)
