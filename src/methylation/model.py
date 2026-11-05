from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data import MAIN_CHROMS, MODEL


def load_datasets(paths):
    from datasets import concatenate_datasets, load_from_disk
    sets = [load_from_disk(str(path)) for path in paths]
    return sets[0] if len(sets) == 1 else concatenate_datasets(sets)


def model_columns(dataset):
    return dataset.remove_columns([c for c in dataset.column_names if c not in {"input_ids", "attention_mask", "labels"}])


def train(paths, output, stage, donor, model=None, batch_size=14, epochs=None, resume=None, workers=0):
    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)
    set_seed(42)
    dataset = load_datasets(paths)
    if "window_id" not in dataset.column_names:
        raise ValueError("Training data must contain window_id to verify chromosome membership")
    if not all(w.split(":")[0] in MAIN_CHROMS for w in dataset["window_id"]):
        raise ValueError("Training includes chromosomes outside 1–5 and 18–22")
    for path in paths:
        metadata = Path(path).parent / "study.json"
        if not metadata.exists():
            raise ValueError(f"Missing dataset metadata: {metadata}")
        source = json.loads(metadata.read_text())
        if (stage == "donor-excluded" and source["donor"] == donor) or (stage == "personalized" and source["donor"] != donor):
            raise ValueError("Dataset donor does not match the requested training stage")
        if stage == "donor-excluded" and source["mode"] != "reference":
            raise ValueError("Donor-excluded training requires full reference-donor datasets")
        if stage == "personalized" and (source["mode"] != "personalized" or Path(path).name != "TRAIN"):
            raise ValueError("Personalization requires the rebalanced target-donor TRAIN dataset")
    dataset = dataset.filter(lambda row: sum(x != -100 for x in row["labels"]) >= 10)
    if not len(dataset):
        raise ValueError("No windows satisfy the ten-site training filter")
    dataset = model_columns(dataset)
    if stage == "personalized" and model is None:
        raise ValueError("Personalization requires the donor-excluded epoch-10 model")
    output = Path(output)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"Nonempty output directory: {output}")
    model_path = model or MODEL
    tokenizer_path = model_path if (Path(model_path) / "tokenizer_config.json").exists() else MODEL
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    network = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=3, trust_remote_code=True)
    epochs = epochs or (12 if stage == "donor-excluded" else 3)
    learning_rate = 1e-6 if stage == "donor-excluded" else 3e-5

    class SaveEpoch(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                destination = Path(args.output_dir) / f"epoch-{int(state.epoch)}"
                kwargs["model"].save_pretrained(destination, safe_serialization=True)
                tokenizer.save_pretrained(destination)
            return control

    args = TrainingArguments(output_dir=str(output), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=1,
        learning_rate=learning_rate, weight_decay=.15, warmup_steps=0,
        lr_scheduler_type="linear", optim="adamw_torch", max_grad_norm=1.,
        bf16=torch.cuda.is_available(), seed=42, data_seed=42,
        dataloader_num_workers=workers, evaluation_strategy="no", save_strategy="epoch",
        logging_strategy="epoch", report_to="none", load_best_model_at_end=False)
    trainer = Trainer(model=network, args=args, train_dataset=dataset, tokenizer=tokenizer,
                      data_collator=DataCollatorForTokenClassification(tokenizer), callbacks=[SaveEpoch()])
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output / "final"))
    tokenizer.save_pretrained(output / "final")
    (output / "study.json").write_text(json.dumps({"donor": donor, "stage": stage, "initial_model": model_path,
        "epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size, "seed": 42}, indent=2) + "\n")


def predict(dataset_path, checkpoint, donor, tissue, output, batch_size=14):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
    dataset = load_datasets([dataset_path])
    tokenizer_path = checkpoint if (Path(checkpoint) / "tokenizer_config.json").exists() else MODEL
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    network = AutoModelForTokenClassification.from_pretrained(checkpoint, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device).eval()
    loader = DataLoader(model_columns(dataset), batch_size=batch_size, shuffle=False,
                        collate_fn=DataCollatorForTokenClassification(tokenizer))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    fields = ["donor", "tissue", "chrom", "window_id", "window_start", "window_end", "token_index",
              "genomic_position", "label", "predicted_class", "probability_class_0", "probability_class_1", "probability_class_2"]
    offset = 0
    with output.open("w", newline="", encoding="utf-8") as handle, torch.inference_mode():
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch in loader:
            probabilities = network(**{k: v.to(device) for k, v in batch.items() if k != "labels"}).logits.float().softmax(-1).cpu().numpy()
            for i, scores in enumerate(probabilities):
                example = dataset[offset + i]
                labels = np.asarray(example["labels"])
                if len(labels) != len(scores):
                    raise ValueError("Prediction and label lengths differ")
                for token in np.flatnonzero(labels != -100):
                    if token == 0:
                        raise ValueError("Special-token label must be masked")
                    writer.writerow({"donor":donor, "tissue":tissue, "chrom":example["window_id"].split(":")[0],
                        "window_id":example["window_id"], "window_start":example["start"], "window_end":example["end"],
                        "token_index":int(token), "genomic_position":int(example["start"]) + (int(token)-1)*6,
                        "label":int(labels[token]), "predicted_class":int(scores[token].argmax()),
                        **{f"probability_class_{j}":float(scores[token,j]) for j in range(3)}})
            offset += len(probabilities)
