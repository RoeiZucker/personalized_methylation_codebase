import torch
from transformers import AutoModelForTokenClassification
from peft import LoraConfig, TaskType, PeftModel, get_peft_model


DEFAULT_LORA_CONFIG = {
    "r": 1,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["query", "value"],
}


def normalize_target_modules(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def get_lora_config(lora_config=None):
    config = dict(DEFAULT_LORA_CONFIG)
    config.update(lora_config or {})
    return LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=int(config["r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=normalize_target_modules(config["target_modules"]),
    )


# TODO: can these two functiosn be merged?
def get_base_model(use_lora :bool,freeze_model  :bool,num_labels :int, base_model_name : str, lora_config=None):
    non_lora_load_kwargs = {
            # "ignore_mismatched_sizes": True,
        }
        # if for_inference:
        #     non_lora_load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        # 
    # TODO: uncomment this for next version
    # non_lora_load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForTokenClassification.from_pretrained(base_model_name,num_labels=num_labels, device_map="auto",trust_remote_code=True,**non_lora_load_kwargs)
    if freeze_model:
        for param in model.base_model.parameters():    # .base_model works for BERT/LLM
            param.requires_grad = False
    if use_lora:
        peft_config = get_lora_config(lora_config)
        model = get_peft_model(model, peft_config)
    return model

def get_fine_tuned_model(use_lora, num_labels, base_model_name, model_path, for_inference=False,freeze_model = False, lora_config=None):
    if use_lora:
        print("Loading lora model from:", base_model_name)
    else:
        print("Loading model from:", base_model_name)
    print("with path:", model_path)
    print("freeze_model is:",freeze_model)
    print("for_inference is",for_inference,flush=True)
    if use_lora:
        lora_load_kwargs = {}
        # if for_inference:
        #     lora_load_kwargs["device_map"] = "auto"
        #     lora_load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        lora_load_kwargs["device_map"] = "auto"
        lora_load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base_model = AutoModelForTokenClassification.from_pretrained(base_model_name,return_dict=True,num_labels=num_labels,
                                                                     trust_remote_code=True, **lora_load_kwargs)
        if freeze_model:
            for param in base_model.base_model.parameters():    # .base_model works for BERT/LLM
                param.requires_grad = False
        peft_config = get_lora_config(lora_config)

        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True,
                                          config=peft_config)
    else:
        non_lora_load_kwargs = {
            "device_map": "auto",
            "ignore_mismatched_sizes": True,
            "num_labels": num_labels,
        }
        # if for_inference:
        #     non_lora_load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        non_lora_load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForTokenClassification.from_pretrained(model_path, **non_lora_load_kwargs)
                                                                
    return model
