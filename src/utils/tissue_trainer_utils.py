import os
import pickle
import sys
import warnings

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.modeling_utils import unwrap_model
from transformers.trainer import TRAINING_ARGS_NAME

sys.path.insert(0, os.path.abspath("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code"))

from src.constants import REGRESSION_ANALYSIS_SYMBOL, SAVED_EPOCH_PREFIX
from src.utils.metrics_utils import (
    accuracy_score,
    compute_metrics,
    compute_metrics_regression,
    f1_score,
    mcc_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.utils.tissue_metrics_utils import (
    compute_sequence_metrics,
    compute_sequence_metrics_regression,
)


class EpochCheckpointCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        epoch_dir = os.path.join(args.output_dir, f"{SAVED_EPOCH_PREFIX}-{int(state.epoch)}-step-{state.global_step}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir, safe_serialization=getattr(args, "save_safetensors", True))


class SavePretrainedCompatMixin:
    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = unwrap_model(self.model)
        if hasattr(model_to_save, "save_pretrained"):
            if state_dict is None:
                state_dict = model_to_save.state_dict()
            model_to_save.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=getattr(self.args, "save_safetensors", True),
            )
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
            return

        return super()._save(output_dir=output_dir, state_dict=state_dict)


class ResumeCheckpointCompatMixin:
    def _load_rng_state(self, checkpoint):
        try:
            return super()._load_rng_state(checkpoint)
        except pickle.UnpicklingError as exc:
            if "Weights only load failed" not in str(exc):
                raise

            warnings.warn(
                "Retrying checkpoint RNG-state load with weights_only=False for PyTorch 2.6 compatibility. "
                "Use this only with trusted checkpoints.",
                RuntimeWarning,
            )
            original_torch_load = torch.load

            def _torch_load_compat(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return original_torch_load(*args, **kwargs)

            torch.load = _torch_load_compat
            try:
                return super()._load_rng_state(checkpoint)
            finally:
                torch.load = original_torch_load


class CompatibleTrainer(SavePretrainedCompatMixin, ResumeCheckpointCompatMixin, Trainer):
    pass


class TokenRegressionLossMixin:
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"].float()
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits.squeeze(-1)

        mask = labels != -100
        if not mask.any().item():
            loss = logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        valid_predictions = torch.sigmoid(logits[mask])
        valid_labels = labels[mask]
        loss = F.mse_loss(valid_predictions, valid_labels)

        return (loss, outputs) if return_outputs else loss


class TokenRegressionTrainer(TokenRegressionLossMixin, SavePretrainedCompatMixin, ResumeCheckpointCompatMixin, Trainer):
    pass


def get_trainer_type(model_type, task_level="token"):
    trainer_type = CompatibleTrainer
    if model_type == REGRESSION_ANALYSIS_SYMBOL and task_level == "token":
        trainer_type = TokenRegressionTrainer
    return trainer_type


def get_compute_func(model_type, task_level="token"):
    if task_level == "sequence":
        return compute_sequence_metrics_regression if model_type == REGRESSION_ANALYSIS_SYMBOL else compute_sequence_metrics

    compute_metrics_func = compute_metrics
    if model_type == REGRESSION_ANALYSIS_SYMBOL:
        compute_metrics_func = compute_metrics_regression
    return compute_metrics_func


def get_trainer(
    dataset,
    model,
    tokenizer,
    model_type,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=4,
    num_train_epochs=2,
    load_best_model_at_end=False,
    eval_dataset=None,
    output_dir="/home/users/roeizucker/tests/temp_output_panic",
    metric_for_best_model=None,
    eval_mode=True,
    save_stratagy="epoch",
    number_of_steps=-1,
    save_total_limit=-1,
    add_epoch_end_save_callback=False,
    task_level="token",
) -> Trainer:
    if eval_dataset is None:
        eval_dataset = dataset
    trainer_type = get_trainer_type(model_type, task_level=task_level)
    compute_metrics_func = get_compute_func(model_type, task_level=task_level)

    args = get_training_args(
        learning_rate,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        eval_accumulation_steps,
        num_train_epochs,
        load_best_model_at_end,
        output_dir,
        metric_for_best_model,
        eval_mode,
        save_stratagy,
        number_of_steps,
        save_total_limit,
    )

    trainer = trainer_type(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_func,
    )
    if add_epoch_end_save_callback:
        trainer.add_callback(EpochCheckpointCallback())
    return trainer


def get_training_args(
    learning_rate,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    eval_accumulation_steps,
    num_train_epochs,
    load_best_at_end,
    output_dir,
    metric_for_best_model,
    eval_mode,
    save_stratagy,
    number_of_steps,
    save_total_limit,
):
    evaluation_strategy = "no"
    if load_best_at_end:
        evaluation_strategy = "epoch"
    args = None
    extra_params = {}
    if save_stratagy == "steps":
        extra_params["save_steps"] = number_of_steps
        if save_total_limit > 0:
            extra_params["save_total_limit"] = save_total_limit
    if eval_mode:
        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy=evaluation_strategy,
            save_strategy="no",
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_accumulation_steps=eval_accumulation_steps,
            num_train_epochs=num_train_epochs,
            logging_steps=1000,
            logging_strategy="epoch",
            report_to="none",
            dataloader_num_workers=4,
            dataloader_prefetch_factor=3,
            bf16=True,
        )
    else:
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_stratagy,
            load_best_model_at_end=load_best_at_end,
            metric_for_best_model=metric_for_best_model,
            report_to="none",
            dataloader_num_workers=120,
            dataloader_prefetch_factor=90,
            bf16=True,
            **extra_params,
        )

    return args
