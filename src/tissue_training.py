from typing import Any, Dict

import os

from datasets import Sequence, Value
from transformers import AutoTokenizer

try:
    from .constants import (
        CPG_RETRAINING_TASK_TYPE,
        CPG_TRAINING_TASK_TYPE,
        DEFAULT_RANDOM_STATE,
    )
    from .utils.dataset_utils import get_dataset_for_paths
    from .utils.label_transform_utils import apply_label_transform_to_dataset, get_task_label_transform
    from .utils.tissue_model_utils import get_base_model, get_fine_tuned_model
    from .utils.tissue_trainer_utils import get_trainer
except ImportError:
    from constants import (
        CPG_RETRAINING_TASK_TYPE,
        CPG_TRAINING_TASK_TYPE,
        DEFAULT_RANDOM_STATE,
    )
    from utils.dataset_utils import get_dataset_for_paths
    from utils.label_transform_utils import apply_label_transform_to_dataset, get_task_label_transform
    from utils.tissue_model_utils import get_base_model, get_fine_tuned_model
    from utils.tissue_trainer_utils import get_trainer


def run_tissue_training(cfg: Dict[str, Any] | str):
    task_type = cfg["task"]["task_type"]

    verbose = cfg.get("verbose", False)
    random_state = cfg.get("random_state", DEFAULT_RANDOM_STATE)
    testing = cfg["testing_params"]
    top_rows = cfg["task"].get("top_rows", -1)

    testing_mode = testing["test_mode"]
    model_config = cfg["model"]
    train_config = cfg["train"]
    paths = cfg["paths"]

    tissue_prompt_config = model_config.get("tissue_prompt")
    if tissue_prompt_config is None:
        raise ValueError("run_tissue_training was called without model.tissue_prompt in the config.")

    model_name = model_config["model_name"]
    model_repo = model_config["model_repo"]
    freeze_model = model_config.get("freeze_model", False)
    use_lora = model_config.get("use_lora", False)
    num_labels = model_config.get("num_labels")
    task_level = model_config.get("task_level", tissue_prompt_config.get("task_level", "token"))
    train_dataset_path = paths.get("train_dataset_path", None)
    eval_dataset_path = paths.get("eval_dataset_path", None)
    trained_model_path = paths.get("trained_model_path", None)

    model_type = model_config["model_type"]
    output_dir = paths["output_dir"]
    per_device_train_batch_size = train_config["per_device_train_batch_size"]
    per_device_eval_batch_size = train_config["per_device_eval_batch_size"]
    learning_rate = float(train_config["learning_rate"])
    num_train_epochs = train_config["num_train_epochs"]
    load_best_model_at_end = train_config["load_best_model_at_end"]
    metric_for_best_model = train_config["metric_for_best_model"]
    resample_rate = train_config.get("resample_rate", None)

    save_stratagy = train_config["save_stratagy"]
    number_of_steps = train_config.get("number_of_steps", -1)
    save_total_limit = train_config.get("save_total_limit", -1)
    load_dataset_to_memory = train_config.get("load_dataset_to_memory", False)
    min_number_of_cpg_sites = train_config.get("min_number_of_cpg_sites", -1)
    label_transform = get_task_label_transform(cfg["task"])

    add_epoch_end_save_callback = train_config.get("add_epoch_end_save_callback", False)
    continue_from_last = train_config.get("continue_from_last", False)
    lora_over_finetuned = train_config.get("lora_over_finetuned", False)

    if task_type == CPG_TRAINING_TASK_TYPE:
        cpg_training(
            model_name,
            model_repo,
            freeze_model,
            use_lora,
            num_labels,
            train_dataset_path,
            eval_dataset_path,
            model_type,
            output_dir,
            per_device_train_batch_size,
            per_device_eval_batch_size,
            learning_rate,
            num_train_epochs,
            load_best_model_at_end,
            metric_for_best_model,
            resample_rate,
            testing_mode,
            top_rows,
            continue_from_last,
            task_level,
            tissue_prompt_config,
            save_stratagy,
            number_of_steps,
            save_total_limit,
            add_epoch_end_save_callback,
            load_dataset_to_memory,
            min_number_of_cpg_sites,
            verbose,
            label_transform,
        )
    elif task_type == CPG_RETRAINING_TASK_TYPE:
        cpg_retraining(
            random_state,
            top_rows,
            continue_from_last,
            model_name,
            model_repo,
            use_lora,
            lora_over_finetuned,
            num_labels,
            train_dataset_path,
            eval_dataset_path,
            trained_model_path,
            model_type,
            output_dir,
            per_device_train_batch_size,
            per_device_eval_batch_size,
            learning_rate,
            num_train_epochs,
            load_best_model_at_end,
            metric_for_best_model,
            resample_rate,
            task_level,
            tissue_prompt_config,
            save_stratagy,
            number_of_steps,
            save_total_limit,
            add_epoch_end_save_callback,
            load_dataset_to_memory,
            min_number_of_cpg_sites,
            verbose,
            label_transform,
        )
    else:
        ValueError(f"Unknown task type: {task_type}")


def cpg_retraining(
    random_state,
    top_rows,
    continue_from_last,
    model_name,
    model_repo,
    use_lora,
    lora_over_finetuned,
    num_labels,
    train_dataset_path,
    eval_dataset_path,
    trained_model_path,
    model_type,
    output_dir,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    learning_rate,
    num_train_epochs,
    load_best_model_at_end,
    metric_for_best_model,
    resample_rate,
    task_level,
    tissue_prompt_config,
    save_stratagy,
    number_of_steps,
    save_total_limit,
    add_epoch_end_save_callback,
    load_dataset_to_memory,
    min_number_of_cpg_sites,
    verbose,
    label_transform="none",
):
    if verbose:
        print("started retraining for model at path", trained_model_path, flush=True)
    eval_dataset, train_dataset = get_datasets(
        train_dataset_path,
        eval_dataset_path,
        top_rows,
        load_dataset_to_memory,
        label_transform=label_transform,
        verbose=verbose,
    )
    train_dataset = filter_dataset_by_min_cpg_sites(
        train_dataset,
        min_number_of_cpg_sites,
        task_level=task_level,
        verbose=verbose,
        dataset_name="retrain train",
    )
    train_dataset = prepare_dataset_for_training(train_dataset, model_type=model_type, task_level=task_level)
    eval_dataset = prepare_dataset_for_training(eval_dataset, model_type=model_type, task_level=task_level)
    base_model_name = model_repo + "/" + model_name
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if trained_model_path is None:
        raise ValueError("missing trained model path for retraining")
    if resample_rate is not None and resample_rate < 1:
        shuffled_dataset = train_dataset.shuffle(seed=random_state)
        train_dataset = shuffled_dataset.select(range(int(len(shuffled_dataset) * resample_rate)))
    if use_lora and lora_over_finetuned:
        model = get_base_model(
            use_lora,
            True,
            num_labels,
            trained_model_path,
            model_type=model_type,
            tissue_prompt_config=tissue_prompt_config,
        )
    else:
        model = get_fine_tuned_model(
            use_lora,
            num_labels,
            base_model_name,
            trained_model_path,
            model_type=model_type,
            tissue_prompt_config=tissue_prompt_config,
        )
    trainer = get_trainer(
        dataset=train_dataset,
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        task_level=task_level,
        output_dir=output_dir,
        eval_mode=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        eval_dataset=eval_dataset,
        save_stratagy=save_stratagy,
        number_of_steps=number_of_steps,
        save_total_limit=save_total_limit,
        add_epoch_end_save_callback=add_epoch_end_save_callback,
    )

    if len(os.listdir(output_dir)) == 0:
        continue_from_last = False
    trainer.train(resume_from_checkpoint=continue_from_last)


def get_datasets(train_dataset_path, eval_dataset_path, top_rows, load_dataset_to_memory, label_transform="none", verbose=False):
    eval_dataset = None
    train_dataset = get_dataset_for_paths(train_dataset_path, top_rows, load_dataset_to_memory)
    train_dataset = apply_label_transform_to_dataset(
        train_dataset,
        label_transform,
        verbose=verbose,
        dataset_name="training dataset",
    )
    if eval_dataset_path is not None:
        eval_dataset = get_dataset_for_paths(eval_dataset_path, top_rows, load_dataset_to_memory)
        eval_dataset = apply_label_transform_to_dataset(
            eval_dataset,
            label_transform,
            verbose=verbose,
            dataset_name="evaluation dataset",
        )
    return eval_dataset, train_dataset


def filter_dataset_by_min_cpg_sites(dataset, min_number_of_cpg_sites, task_level="token", verbose=False, dataset_name="train"):
    if dataset is None or min_number_of_cpg_sites == -1 or task_level != "token":
        return dataset
    if min_number_of_cpg_sites < -1:
        raise ValueError("min_number_of_cpg_sites must be -1 or a non-negative integer")

    original_size = len(dataset)

    def keep_example(example):
        labels = example.get("labels")
        if labels is None:
            return True
        if isinstance(labels, (list, tuple)):
            return sum(label != -100 for label in labels) >= min_number_of_cpg_sites
        return min_number_of_cpg_sites <= 1 and labels != -100

    filtered_dataset = dataset.filter(keep_example)
    if len(filtered_dataset) == 0:
        raise ValueError(
            f"Filtering {dataset_name} dataset with min_number_of_cpg_sites={min_number_of_cpg_sites} "
            "removed all windows."
        )
    if verbose:
        print(
            f"filtered {dataset_name} dataset with min_number_of_cpg_sites={min_number_of_cpg_sites}: "
            f"kept {len(filtered_dataset)}/{original_size} windows",
            flush=True,
        )
    return filtered_dataset


def prepare_dataset_for_training(dataset, model_type="regression_analysis", task_level="token"):
    if dataset is None:
        return None
    if "tissue_ids" not in dataset.column_names:
        raise ValueError(
            "This config uses tissue prompts, but the dataset has no 'tissue_ids' column. "
            "Recreate the HF datasets after adding tissue-aware extraction."
        )

    drop_cols = [c for c in ["start", "end", "window_id"] if c in dataset.column_names]
    if drop_cols:
        dataset = dataset.remove_columns(drop_cols)

    new_features = dataset.features.copy()
    changed = False
    if "labels" in new_features:
        is_regression = model_type == "regression_analysis"
        if task_level == "token":
            wanted_dtype = "float32" if is_regression else "int64"
            curr = new_features["labels"]
            curr_dtype = getattr(getattr(curr, "feature", None), "dtype", None)
            if curr_dtype != wanted_dtype:
                new_features["labels"] = Sequence(Value(wanted_dtype))
                changed = True
        else:
            wanted_dtype = "float32" if is_regression else "int64"
            curr_dtype = getattr(new_features["labels"], "dtype", None)
            if curr_dtype != wanted_dtype:
                new_features["labels"] = Value(wanted_dtype)
                changed = True
    if "input_ids" in new_features:
        curr = new_features["input_ids"]
        curr_dtype = getattr(getattr(curr, "feature", None), "dtype", None)
        if curr_dtype != "int64":
            new_features["input_ids"] = Sequence(Value("int64"))
            changed = True
    if "attention_mask" in new_features:
        curr = new_features["attention_mask"]
        curr_dtype = getattr(getattr(curr, "feature", None), "dtype", None)
        if curr_dtype != "int64":
            new_features["attention_mask"] = Sequence(Value("int64"))
            changed = True
    curr_dtype = getattr(new_features["tissue_ids"], "dtype", None)
    if curr_dtype != "int64":
        new_features["tissue_ids"] = Value("int64")
        changed = True
    if changed:
        dataset = dataset.cast(new_features)
    return dataset


def cpg_training(
    model_name,
    model_repo,
    freeze_model,
    use_lora,
    num_labels,
    train_dataset_path,
    eval_dataset_path,
    model_type,
    output_dir,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    learning_rate,
    num_train_epochs,
    load_best_model_at_end,
    metric_for_best_model,
    resample_rate,
    testing_mode,
    top_rows,
    continue_from_last,
    task_level,
    tissue_prompt_config,
    save_stratagy,
    number_of_steps,
    save_total_limit,
    add_epoch_end_save_callback,
    load_dataset_to_memory,
    min_number_of_cpg_sites,
    verbose,
    label_transform="none",
):
    if verbose:
        print("started training for", model_name, flush=True)

    eval_dataset, train_dataset = get_datasets(
        train_dataset_path,
        eval_dataset_path,
        top_rows,
        load_dataset_to_memory,
        label_transform=label_transform,
        verbose=verbose,
    )
    train_dataset = filter_dataset_by_min_cpg_sites(
        train_dataset,
        min_number_of_cpg_sites,
        task_level=task_level,
        verbose=verbose,
        dataset_name="train",
    )
    train_dataset = prepare_dataset_for_training(train_dataset, model_type=model_type, task_level=task_level)
    eval_dataset = prepare_dataset_for_training(eval_dataset, model_type=model_type, task_level=task_level)

    base_model_name = model_repo + "/" + model_name
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=tissue_prompt_config.get("trust_remote_code", True),
    )
    model = get_base_model(
        use_lora,
        freeze_model,
        num_labels,
        base_model_name,
        model_type=model_type,
        tissue_prompt_config=tissue_prompt_config,
    )
    trainer = get_trainer(
        dataset=train_dataset,
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        task_level=task_level,
        output_dir=output_dir,
        eval_mode=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        eval_dataset=eval_dataset,
        save_stratagy=save_stratagy,
        number_of_steps=number_of_steps,
        save_total_limit=save_total_limit,
        add_epoch_end_save_callback=add_epoch_end_save_callback,
    )

    if len(os.listdir(output_dir)) == 0:
        continue_from_last = False
    trainer.train(resume_from_checkpoint=continue_from_last)
