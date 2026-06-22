import argparse
import os
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer


def _default_repo_root():
    current = Path(__file__).absolute()
    for candidate in current.parents:
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    return current.parents[1]


REPO_ROOT = _default_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training import prepare_dataset_for_training
from src.utils.dataset_utils import get_dataset_for_paths
from src.utils.label_transform_utils import apply_label_transform_to_dataset, get_task_label_transform
from src.utils.model_utils import get_fine_tuned_model
from src.utils.trainer_utils import EpochPredictionCallback, get_trainer


DEFAULT_OUTPUT_FILENAME = "eval_predictions.csv.gitbackup"


def _load_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _load_prediction_dataset(cfg):
    paths = cfg["paths"]
    model_type = cfg["model"]["model_type"]
    train_config = cfg["train"]
    top_rows = cfg.get("task", {}).get("top_rows", -1)
    load_dataset_to_memory = train_config.get("load_dataset_to_memory", False)
    label_transform = get_task_label_transform(cfg["task"])

    eval_dataset_path = paths.get("eval_dataset_path")
    if eval_dataset_path is None:
        raise ValueError("Missing paths.eval_dataset_path in prediction config")

    metadata_dataset = get_dataset_for_paths(eval_dataset_path, top_rows, load_dataset_to_memory)
    metadata_dataset = apply_label_transform_to_dataset(
        metadata_dataset,
        label_transform,
        verbose=True,
        dataset_name="pretrain checkpoint prediction dataset",
    )
    prediction_dataset = prepare_dataset_for_training(metadata_dataset, model_type=model_type)
    return prediction_dataset, metadata_dataset


def _create_prediction_trainer(cfg, checkpoint_path, prediction_dataset, output_dir):
    model_config = cfg["model"]
    train_config = cfg["train"]
    base_model_name = model_config["model_repo"] + "/" + model_config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = get_fine_tuned_model(
        model_config.get("use_lora", False),
        model_config.get("num_labels"),
        base_model_name,
        checkpoint_path,
        for_inference=True,
        freeze_model=model_config.get("freeze_model", False),
    )

    return get_trainer(
        dataset=prediction_dataset,
        model=model,
        tokenizer=tokenizer,
        model_type=model_config["model_type"],
        output_dir=output_dir,
        eval_mode=True,
        num_train_epochs=1,
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 1),
        learning_rate=float(train_config.get("learning_rate", 1e-6)),
        load_best_model_at_end=False,
        metric_for_best_model=train_config.get("metric_for_best_model"),
        add_epoch_end_save_callback=False,
        add_epoch_end_prediction=False,
        eval_dataset=prediction_dataset,
        save_stratagy="no",
        number_of_steps=-1,
        save_total_limit=-1,
    )


def run_prediction(config_path, checkpoint_path, output_path, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Prediction output exists and overwrite is false: {output_path}")

    cfg = _load_yaml(config_path)
    output_dir = os.path.dirname(os.path.dirname(output_path))
    prediction_dataset, metadata_dataset = _load_prediction_dataset(cfg)
    trainer = _create_prediction_trainer(cfg, checkpoint_path, prediction_dataset, output_dir)

    prediction_output = trainer.predict(trainer.eval_dataset)
    writer = EpochPredictionCallback(
        trainer=None,
        prediction_dataset=None,
        metadata_dataset=metadata_dataset,
        output_filename=os.path.basename(output_path) or DEFAULT_OUTPUT_FILENAME,
    )
    predictions = writer._normalize_predictions(prediction_output.predictions)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer._write_predictions(output_path, predictions)
    print(f"wrote predictions to {output_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one pretrain checkpoint on one eval dataset.")
    parser.add_argument("--config", required=True, help="Training YAML with paths.eval_dataset_path and model settings.")
    parser.add_argument("--checkpoint-path", required=True, help="Pretrain epoch checkpoint directory to load.")
    parser.add_argument("--output-path", required=True, help="CSV output path to write.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_prediction(
        config_path=args.config,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
