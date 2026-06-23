from transformers import AutoModelForTokenClassification

# model parts
def get_base_model(num_labels :int, base_model_name : str):

    # TODO: uncomment this for next version
    # non_lora_load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return AutoModelForTokenClassification.from_pretrained(base_model_name,num_labels=num_labels, device_map="auto",trust_remote_code=True)



import csv
import sys
import os
import pickle
import warnings
sys.path.insert(0, os.path.abspath("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code"))


from transformers import Trainer, TrainingArguments, TrainerCallback
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.metrics_utils import (
    compute_metrics,
)

from src.constants import (
    SAVED_EPOCH_PREFIX,
    INSTADEEP_KMER_SIZE,
)


def _softmax_np(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()

class EpochCheckpointCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]  # provided by Trainer
        epoch_dir = os.path.join(args.output_dir, f"{SAVED_EPOCH_PREFIX}-{int(state.epoch)}-step-{state.global_step}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Save model (works for regular and PEFT/LoRA models)
        # safe_serialization uses .safetensors when available
        model.save_pretrained(epoch_dir, safe_serialization=getattr(args, "save_safetensors", True))

class EpochPredictionCallback(TrainerCallback):
    def __init__(
        self,
        trainer,
        prediction_dataset,
        metadata_dataset,
        output_filename="eval_predictions.csv.gitbackup",
        default_bases_per_token=INSTADEEP_KMER_SIZE,
    ):
        self.trainer = trainer
        self.prediction_dataset = prediction_dataset
        self.metadata_dataset = metadata_dataset
        self.output_filename = output_filename
        self.default_bases_per_token = default_bases_per_token

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None or self.prediction_dataset is None or self.metadata_dataset is None:
            return

        model = kwargs["model"]
        was_training = model.training
        prediction_output = self.trainer.predict(self.prediction_dataset)
        if was_training:
            model.train()

        if not getattr(state, "is_world_process_zero", True):
            return

        predictions = self._normalize_predictions(prediction_output.predictions)
        if len(predictions) != len(self.metadata_dataset):
            raise ValueError(
                "Prediction rows do not match metadata rows: "
                f"{len(predictions)} predictions vs {len(self.metadata_dataset)} metadata examples."
            )

        epoch_dir = os.path.join(args.output_dir, f"{SAVED_EPOCH_PREFIX}-{int(state.epoch)}-step-{state.global_step}")
        os.makedirs(epoch_dir, exist_ok=True)
        output_path = os.path.join(epoch_dir, self.output_filename)
        self._write_predictions(output_path, predictions)

    def _normalize_predictions(self, predictions):
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        if predictions.ndim == 3 and predictions.shape[-1] == 1:
            predictions = predictions[..., 0]
        return predictions

    def _write_predictions(self, output_path, predictions):
        num_classes = int(predictions.shape[-1])

        fieldnames = [
            "window_id",
            "window_start",
            "window_end",
            "token_index",
            "token_position_in_window",
            "base_position_in_window",
            "genomic_position",
            "label",
            "prediction",
            "logit",
            "probability",
        ]
        fieldnames.append("predicted_class")
        for class_id in range(num_classes):
            fieldnames.append(f"logit_class_{class_id}")
        for class_id in range(num_classes):
            fieldnames.append(f"probability_class_{class_id}")

        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for example_index in range(len(self.metadata_dataset)):
                metadata_example = self.metadata_dataset[example_index]
                labels = np.asarray(metadata_example["labels"])
                prediction_values = np.asarray(predictions[example_index])

                expected_length = prediction_values.shape[0]
                if expected_length != labels.shape[0]:
                    raise ValueError(
                        "Prediction length does not match label length for example "
                        f"{example_index}: {expected_length} vs {labels.shape[0]}."
                    )

                window_id = metadata_example.get("window_id", f"window_{example_index}")
                window_start = metadata_example.get("start")
                window_end = metadata_example.get("end")
                bases_per_token = self._infer_bases_per_token(window_start, window_end, len(labels))

                for token_index in np.flatnonzero(labels != -100):
                    token_position_in_window = int(token_index) - 1
                    base_position_in_window = token_position_in_window * bases_per_token
                    genomic_position = None
                    if window_start is not None:
                        genomic_position = int(window_start) + base_position_in_window

                    row = {
                        "window_id": window_id,
                        "window_start": window_start,
                        "window_end": window_end,
                        "token_index": int(token_index),
                        "token_position_in_window": token_position_in_window,
                        "base_position_in_window": base_position_in_window,
                        "genomic_position": genomic_position,
                        "label": float(labels[token_index]),
                    }

                    logits = np.asarray(prediction_values[token_index], dtype=np.float64)
                    probabilities = _softmax_np(logits)
                    predicted_class = int(np.argmax(probabilities))
                    row.update(
                        {
                            "prediction": predicted_class,
                            "logit": float(logits[predicted_class]),
                            "probability": float(probabilities[predicted_class]),
                            "predicted_class": predicted_class,
                        }
                    )
                    for class_id in range(num_classes):
                        row[f"logit_class_{class_id}"] = float(logits[class_id])
                        row[f"probability_class_{class_id}"] = float(probabilities[class_id])
                    writer.writerow(row)

    def _infer_bases_per_token(self, window_start, window_end, label_count):
        if window_start is None or window_end is None or label_count <= 1:
            return self.default_bases_per_token

        total_bases = int(window_end) - int(window_start)
        total_tokens = label_count - 1
        if total_bases <= 0 or total_tokens <= 0 or total_bases % total_tokens != 0:
            return self.default_bases_per_token

        inferred_bases_per_token = total_bases // total_tokens
        if inferred_bases_per_token <= 0:
            return self.default_bases_per_token
        return int(inferred_bases_per_token)

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


class CompatibleTrainer(ResumeCheckpointCompatMixin, Trainer):
    pass


def get_trainer_type(model_type):
    trainer_type = CompatibleTrainer
    return trainer_type

def get_compute_func(model_type):
    compute_metrics_func = compute_metrics
    return compute_metrics_func




def get_trainer(dataset, model, tokenizer,model_type,learning_rate=1e-5,per_device_train_batch_size=1,per_device_eval_batch_size=1, 
                # eval_accumulation_steps=4,num_train_epochs=2,load_best_model_at_end=False,eval_dataset=None,output_dir = "/home/users/roeizucker/tests/temp_output_panic",
                eval_accumulation_steps=4,num_train_epochs=2,load_best_model_at_end=False,eval_dataset=None,output_dir = "/cs/usr/roeizucker/new_storage/junk",
                metric_for_best_model=None,eval_mode = True,save_stratagy="epoch",number_of_steps=-1,save_total_limit=-1,
                add_epoch_end_save_callback=False,add_epoch_end_prediction=True,prediction_metadata_dataset=None) -> Trainer:
    if eval_dataset is None:
        eval_dataset = dataset
    trainer_type = get_trainer_type(model_type)
    compute_metrics_func = get_compute_func(model_type)

    args = get_training_args(learning_rate, per_device_train_batch_size, per_device_eval_batch_size, eval_accumulation_steps, num_train_epochs, 
                             load_best_model_at_end, output_dir, metric_for_best_model, eval_mode,save_stratagy,number_of_steps,save_total_limit)
    

    trainer = trainer_type(
            model=model,
            args=args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_func
        )
    if add_epoch_end_save_callback:
        trainer.add_callback(EpochCheckpointCallback())
    if add_epoch_end_prediction:
        trainer.add_callback(
            EpochPredictionCallback(
                trainer=trainer,
                prediction_dataset=eval_dataset,
                metadata_dataset=prediction_metadata_dataset if prediction_metadata_dataset is not None else eval_dataset,
            )
        )
    return trainer

def get_training_args(learning_rate, per_device_train_batch_size, per_device_eval_batch_size, eval_accumulation_steps, num_train_epochs,
                       load_best_at_end, output_dir, metric_for_best_model, eval_mode,save_stratagy,number_of_steps,save_total_limit):
    # TODO: constants
    evaluation_strategy="no"
    if load_best_at_end:
        evaluation_strategy="epoch"
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
            learning_rate=learning_rate,            # 1e-5 if full fine-tune
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_accumulation_steps=eval_accumulation_steps,
            num_train_epochs=num_train_epochs,
            # TODO: return this to 1000
            logging_steps=10,
            eval_steps=20,
            logging_strategy="epoch",
            report_to="none",
            dataloader_num_workers=4,
            dataloader_prefetch_factor=3,
            bf16=True,
            weight_decay=0.15
        )
    else:
        args = TrainingArguments(
        output_dir= output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,            # 1e-5 if full fine-tune
        num_train_epochs=num_train_epochs,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_stratagy,
        load_best_model_at_end=load_best_at_end,
        metric_for_best_model=metric_for_best_model,
        report_to="none",
        # TODO: make sure these numbers are normal
        dataloader_num_workers=40,
        dataloader_prefetch_factor=30,
        # TODO Consider removing quantization to improve performance
        bf16=True,
        weight_decay=0.15,
        **extra_params,

        )
        
    return args
