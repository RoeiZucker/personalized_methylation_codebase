import os

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForTokenClassification

from .tissue_prompt_model import SEQUENCE_TASK_LEVEL, TissuePromptModel


def _get_problem_type(model_type):
    return "regression" if model_type == "regression_analysis" else "single_label_classification"


def _get_task_type(task_level):
    return TaskType.SEQ_CLS if task_level == SEQUENCE_TASK_LEVEL else TaskType.TOKEN_CLS


def _build_tissue_prompt_model(
    base_model_name,
    num_labels,
    model_type,
    tissue_prompt_config,
    load_kwargs,
):
    task_level = tissue_prompt_config.get("task_level", "token")
    return TissuePromptModel.from_backbone_pretrained(
        base_model_name,
        num_labels=num_labels,
        task_level=task_level,
        num_tissue_types=tissue_prompt_config["num_tissue_types"],
        num_virtual_tokens=tissue_prompt_config["num_virtual_tokens"],
        tissue_embedding_dim=tissue_prompt_config.get("tissue_embedding_dim"),
        projector_hidden_dim=tissue_prompt_config.get("projector_hidden_dim"),
        problem_type=tissue_prompt_config.get("problem_type", _get_problem_type(model_type)),
        ignore_index=tissue_prompt_config.get("ignore_index", -100),
        token_labels_include_cls=tissue_prompt_config.get("token_labels_include_cls", True),
        trust_remote_code=tissue_prompt_config.get("trust_remote_code", True),
        load_kwargs=load_kwargs,
    )


def _maybe_freeze_base_model(model, freeze_model):
    if not freeze_model:
        return
    for param in model.base_model.parameters():
        param.requires_grad = False


def _wrap_with_lora(model, use_lora, task_level):
    if not use_lora:
        return model
    peft_config = LoraConfig(
        task_type=_get_task_type(task_level),
        inference_mode=False,
        r=1,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    return get_peft_model(model, peft_config)


def _has_tissue_prompt_checkpoint(model_path):
    from .tissue_prompt_model import TISSUE_PROMPT_CONFIG_NAME

    return model_path is not None and os.path.exists(os.path.join(model_path, TISSUE_PROMPT_CONFIG_NAME))


def get_base_model(
    use_lora: bool,
    freeze_model: bool,
    num_labels: int,
    base_model_name: str,
    model_type: str = "regression_analysis",
    tissue_prompt_config=None,
):
    non_lora_load_kwargs = {}
    task_level = (tissue_prompt_config or {}).get("task_level", "token")
    if tissue_prompt_config:
        if _has_tissue_prompt_checkpoint(base_model_name):
            model = TissuePromptModel.from_pretrained(base_model_name)
        else:
            model = _build_tissue_prompt_model(
                base_model_name,
                num_labels,
                model_type,
                tissue_prompt_config,
                load_kwargs=non_lora_load_kwargs,
            )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            device_map="auto",
            trust_remote_code=True,
            **non_lora_load_kwargs,
        )
    _maybe_freeze_base_model(model, freeze_model)
    return _wrap_with_lora(model, use_lora, task_level)


def get_fine_tuned_model(
    use_lora,
    num_labels,
    base_model_name,
    model_path,
    for_inference=False,
    freeze_model=False,
    model_type="regression_analysis",
    tissue_prompt_config=None,
):
    if use_lora:
        print("Loading lora model from:", base_model_name)
    else:
        print("Loading model from:", base_model_name)
    print("with path:", model_path)
    print("freeze_model is:", freeze_model)
    print("for_inference is", for_inference, flush=True)

    use_tissue_prompt_model = tissue_prompt_config is not None or _has_tissue_prompt_checkpoint(model_path)
    task_level = (tissue_prompt_config or {}).get("task_level", "token")

    if use_lora:
        lora_load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if use_tissue_prompt_model:
            if tissue_prompt_config is None and _has_tissue_prompt_checkpoint(model_path):
                model = TissuePromptModel.from_pretrained(model_path)
                _maybe_freeze_base_model(model, freeze_model)
                return model
            base_model = _build_tissue_prompt_model(
                base_model_name,
                num_labels,
                model_type,
                tissue_prompt_config,
                load_kwargs=lora_load_kwargs,
            )
        else:
            base_model = AutoModelForTokenClassification.from_pretrained(
                base_model_name,
                return_dict=True,
                num_labels=num_labels,
                trust_remote_code=True,
                **lora_load_kwargs,
            )
        _maybe_freeze_base_model(base_model, freeze_model)
        peft_config = LoraConfig(
            task_type=_get_task_type(task_level),
            inference_mode=False,
            r=1,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        print()
        return PeftModel.from_pretrained(base_model, model_path, is_trainable=True, config=peft_config)

    if use_tissue_prompt_model:
        model = TissuePromptModel.from_pretrained(model_path)
        _maybe_freeze_base_model(model, freeze_model)
        return model

    non_lora_load_kwargs = {
        "device_map": "auto",
        "ignore_mismatched_sizes": True,
        "num_labels": num_labels,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    model = AutoModelForTokenClassification.from_pretrained(model_path, **non_lora_load_kwargs)
    _maybe_freeze_base_model(model, freeze_model)
    return model
