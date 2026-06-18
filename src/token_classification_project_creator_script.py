import argparse
import copy
import math
import os
from pathlib import Path

import yaml

def _default_repo_root():
    override = os.environ.get("TOKEN_CLASSIFICATION_REPO_ROOT")
    if override:
        return Path(override)

    preferred_root = Path("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code")
    if preferred_root.exists():
        return preferred_root

    pwd = Path(os.environ.get("PWD", os.getcwd()))
    for candidate in (pwd / "Tom_Hope_Project" / "refactored_code", pwd):
        if (candidate / "scripts").is_dir() and (candidate / "src").is_dir():
            return candidate

    return Path(__file__).absolute().parents[1]


try:
    from .constants import (
        BLANK_LABEL_VALUE,
        CLASSIFICATION_ANALYSIS_SYMBOL,
        CPG_RETRAINING_TASK_TYPE,
        CPG_SEPERATING_SITES_TASK_NAME,
        CPG_TOKEN_CLASSIFICATION_EXTRACTION_TASK_NAME,
        CPG_TRAINING_TASK_TYPE,
        HG38_PATH,
        KMER_SAMPLE_TRAIN_TEST_FILTRATION,
        NO_PRETRAINING_CONFIG_NAME,
        QUANTILE_SEPERATION_TYPE,
        RANDOM_SAMPLE_TRAIN_TEST_FILTRATION,
        RAW_INPUT_NAME,
        STANDART_NO_OVERLAP_WINDOW_TYPE,
        STD_VARIABILITY_TYPE,
    )
except ImportError:
    from constants import (
        BLANK_LABEL_VALUE,
        CLASSIFICATION_ANALYSIS_SYMBOL,
        CPG_RETRAINING_TASK_TYPE,
        CPG_SEPERATING_SITES_TASK_NAME,
        CPG_TOKEN_CLASSIFICATION_EXTRACTION_TASK_NAME,
        CPG_TRAINING_TASK_TYPE,
        HG38_PATH,
        KMER_SAMPLE_TRAIN_TEST_FILTRATION,
        NO_PRETRAINING_CONFIG_NAME,
        QUANTILE_SEPERATION_TYPE,
        RANDOM_SAMPLE_TRAIN_TEST_FILTRATION,
        RAW_INPUT_NAME,
        STANDART_NO_OVERLAP_WINDOW_TYPE,
        STD_VARIABILITY_TYPE,
    )


REPO_ROOT = _default_repo_root()
DEFAULT_SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_EXTRACT_SCRIPT = DEFAULT_SCRIPTS_DIR / "run_data_extraction_params.sh"
DEFAULT_TRAIN_SCRIPT = DEFAULT_SCRIPTS_DIR / "run_training_params.sh"

DEFAULT_GENERATOR_CACHE_DIR = "/sci/archive/michall/roeizucker/huggingface_modles_cache/datasets/generator"
DEFAULT_MM10_PATH = "/sci/archive/michall/roeizucker/reference_genome/mm10.fa"
DEFAULT_TOKENIZER_NAME = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
DEFAULT_TOKEN_LABEL_BINNING = {"method": "fixed", "low": 0.2, "high": 0.8}
DEFAULT_TOKEN_LABEL_DOWNSAMPLING = None


def create_base_dictionary(assemblies=None, generator_cache_dir=DEFAULT_GENERATOR_CACHE_DIR, random_state=42, verbose=True):
    if assemblies is None:
        assemblies = {
            "HG38": HG38_PATH,
            "MM10": DEFAULT_MM10_PATH,
        }
    return {
        "paths": {
            "assemblies": assemblies,
            "generetor_cache_dir": generator_cache_dir,
        },
        "task": {},
        "testing_params": {
            "test_mode": False,
        },
        "random_state": random_state,
        "verbose": verbose,
    }


def _as_list(value, default=None):
    if value is None:
        value = default
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _ensure_dir(path, dry_run=False):
    if path is None:
        return
    if dry_run:
        print(f"mkdir -p {path}")
        return
    os.makedirs(path, exist_ok=True)


def _write_yaml(path, config, overwrite=False, dry_run=False):
    if os.path.exists(path) and not overwrite:
        return "skipped"
    if dry_run:
        print(f"would write {path}")
        return "created"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)
    return "created"


def _split_tokenizer_name(tokenizer_name):
    if "/" not in tokenizer_name:
        raise ValueError(f"Expected tokenizer name in 'repo/model' format, got: {tokenizer_name}")
    model_repo, model_name = tokenizer_name.split("/", 1)
    return model_repo, model_name


def create_token_classification_extraction_config(
    raw_data_path,
    dataset_base_dir,
    created_datasets_base_name,
    tokenizer_name,
    chromosomes,
    seq_size,
    test_size,
    train_test_seperation,
    token_label_binning,
    token_label_downsampling=None,
    use_variant_filtering=False,
    variant_filtering_upper_bound=-1,
    variant_filtering_lower_bound=-1,
    variant_file_path=None,
    replace_min1=True,
    override_dataset=False,
    random_state=42,
    verbose=True,
    assemblies=None,
    generator_cache_dir=DEFAULT_GENERATOR_CACHE_DIR,
):
    base_dict = create_base_dictionary(
        assemblies=assemblies,
        generator_cache_dir=generator_cache_dir,
        random_state=random_state,
        verbose=verbose,
    )
    paths = base_dict["paths"]
    task = base_dict["task"]

    paths["raw_data_path"] = raw_data_path
    paths["intermediate_train_data_path"] = os.path.join(
        dataset_base_dir,
        f"{created_datasets_base_name}_intermediate_train.csv",
    )
    paths["intermediate_test_data_path"] = os.path.join(
        dataset_base_dir,
        f"{created_datasets_base_name}_intermediate_test.csv",
    )
    paths["hf_dataset_train_path"] = os.path.join(dataset_base_dir, f"{created_datasets_base_name}_train")
    paths["hf_dataset_test_path"] = os.path.join(dataset_base_dir, f"{created_datasets_base_name}_test")

    task["type"] = CPG_TOKEN_CLASSIFICATION_EXTRACTION_TASK_NAME
    task["assembly"] = "HG38"
    task["value_column"] = "methyl_rate"
    task["input_mode"] = RAW_INPUT_NAME
    task["seq_size"] = seq_size
    task["test_size"] = test_size
    task["blank_label"] = BLANK_LABEL_VALUE
    task["use_fasta"] = True
    task["shuffle"] = True
    task["chromosomes"] = chromosomes
    task["tokenizer_name"] = tokenizer_name
    task["output_preprocessed_data"] = False
    task["output_intermediate_data"] = True
    task["output_hf_dataset"] = True
    task["window_type"] = STANDART_NO_OVERLAP_WINDOW_TYPE
    task["override_dataset"] = override_dataset
    task["train_test_seperation"] = train_test_seperation
    task["clear_generator_cache"] = True
    task["replace_min1"] = replace_min1
    task["token_label_binning"] = copy.deepcopy(token_label_binning)
    if token_label_downsampling is not None:
        task["token_label_downsampling"] = copy.deepcopy(token_label_downsampling)

    variant_filtering = {
        "use_variant_filtering": use_variant_filtering,
        "variant_filtering_upper_bound": variant_filtering_upper_bound,
        "variant_filtering_lower_bound": variant_filtering_lower_bound,
    }
    if use_variant_filtering:
        paths["variant_file_path"] = variant_file_path
    base_dict["variant_filtering"] = variant_filtering
    return base_dict



def create_variability_extraction_config(
    bigwig_files,
    current_bigwig_file,
    per_variant_data_path,
    chromosomes,
    seq_size,
    random_state=42,
    verbose=True,
    assemblies=None,
    generator_cache_dir=DEFAULT_GENERATOR_CACHE_DIR,
):
    base_dict = create_base_dictionary(
        assemblies=assemblies,
        generator_cache_dir=generator_cache_dir,
        random_state=random_state,
        verbose=verbose,
    )
    raw_data_paths = list(bigwig_files)
    raw_data_paths.remove(current_bigwig_file)

    paths = base_dict["paths"]
    task = base_dict["task"]
    paths["raw_data_paths"] = raw_data_paths
    paths["per_variant_data_path"] = per_variant_data_path

    task["type"] = CPG_SEPERATING_SITES_TASK_NAME
    task["chromosomes"] = chromosomes
    task["test_size"] = 0
    task["variability_type"] = STD_VARIABILITY_TYPE
    task["variant_seperation_type"] = QUANTILE_SEPERATION_TYPE
    task["variant_seperation_threshold"] = 0.09
    task["output_per_variant_data"] = True
    task["output_window_data"] = False
    task["seq_size"] = seq_size
    return base_dict

def create_token_classification_training_config(
    tokenizer_name,
    train_dataset_path,
    output_dir,
    analysis_name,
    num_train_epochs,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    learning_rate,
    metric_for_best_model="mcc",
    save_stratagy="steps",
    number_of_steps=10000,
    save_total_limit=2,
    load_best_model_at_end=False,
    use_lora=False,
    freeze_model=False,
    num_labels=3,
    task_type=CPG_TRAINING_TASK_TYPE,
    eval_dataset_path=None,
    trained_model_path=None,
    top_rows=-1,
    max_grad_norm=1,
    add_epoch_end_save_callback=True,
    add_epoch_end_prediction=True,
    save_at_end=False,
    continue_from_last=True,
    lora_over_finetuned=False,
    min_number_of_cpg_sites=-1,
    load_dataset_to_memory=True,
    label_transform="none",
    random_state=42,
    verbose=True,
    assemblies=None,
    generator_cache_dir=DEFAULT_GENERATOR_CACHE_DIR,
):
    model_repo, model_name = _split_tokenizer_name(tokenizer_name)
    base_dict = create_base_dictionary(
        assemblies=assemblies,
        generator_cache_dir=generator_cache_dir,
        random_state=random_state,
        verbose=verbose,
    )

    paths = base_dict["paths"]
    paths["train_dataset_path"] = train_dataset_path
    paths["output_dir"] = output_dir
    if eval_dataset_path is not None:
        paths["eval_dataset_path"] = eval_dataset_path
    if trained_model_path is not None:
        paths["trained_model_path"] = trained_model_path

    base_dict["task"] = {
        "task_type": task_type,
        "analysis_name": analysis_name,
        "top_rows": top_rows,
        "label_transform": label_transform,
    }
    base_dict["model"] = {
        "model_name": model_name,
        "model_repo": model_repo,
        "model_type": CLASSIFICATION_ANALYSIS_SYMBOL,
        "blank_label": BLANK_LABEL_VALUE,
        "use_lora": use_lora,
        "freeze_model": freeze_model,
        "num_labels": num_labels,
        "lora_over_finetuned": lora_over_finetuned,
    }
    base_dict["train"] = {
        "load_best_model_at_end": load_best_model_at_end,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "learning_rate": learning_rate,
        "metric_for_best_model": metric_for_best_model,
        "save_stratagy": save_stratagy,
        "lora_over_finetuned": lora_over_finetuned,
        "load_dataset_to_memory": load_dataset_to_memory,
        "min_number_of_cpg_sites": min_number_of_cpg_sites,
        "max_grad_norm": max_grad_norm,
        "number_of_steps": number_of_steps,
        "save_total_limit": save_total_limit,
        "add_epoch_end_save_callback": add_epoch_end_save_callback,
        "add_epoch_end_prediction": add_epoch_end_prediction,
        "save_at_end": save_at_end,
        "continue_from_last": continue_from_last,
    }
    return base_dict


def _effective_batch_size(batch_size, seq_size, seq_batch_size_overrides):
    if not seq_batch_size_overrides:
        return batch_size
    if seq_size in seq_batch_size_overrides:
        return seq_batch_size_overrides[seq_size]
    seq_size_key = str(seq_size)
    if seq_size_key in seq_batch_size_overrides:
        return seq_batch_size_overrides[seq_size_key]
    return batch_size


def _project_suffix(base_suffix, lr, batch_size, seq_size, test_size, include_test_size):
    suffix = f"{base_suffix}_lr_{lr}_bs_{batch_size}_seq_{seq_size}"
    if include_test_size:
        suffix += f"_testsize_{test_size}"
    return suffix


def create_token_classification_project_config(
    project_suffix,
    bigwig_files,
    names,
    created_configs_path,
    tokenizer_name=DEFAULT_TOKENIZER_NAME,
    dataset_base_dir=None,
    base_model_location=None,
    train_test_seperation=RANDOM_SAMPLE_TRAIN_TEST_FILTRATION,
    chromosomes=None,
    seq_size=5400,
    test_size=0.2,
    token_label_binning=None,
    token_label_downsampling=DEFAULT_TOKEN_LABEL_DOWNSAMPLING,
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epoch=5,
    num_pretrain_epoch=2,
    metric_for_best_model="mcc",
    save_stratagy="steps",
    number_of_steps=10000,
    save_total_limit=2,
    load_best_model_at_end=False,
    add_epoch_end_save_callback=True,
    save_at_end=False,
    continue_from_last=True,
    use_lora=False,
    freeze_model=False,
    num_labels=3,
    use_variant_filtering=False,
    pretraining_variant_filtering_upper_bound=-1,
    pretraining_variant_filtering_lower_bound=-1,
    retraining_variant_filtering_upper_bound=-1,
    retraining_variant_filtering_lower_bound=-1,
    variant_file_path=None,
    max_grad_norm=1,
    top_rows=-1,
    min_number_of_cpg_sites=-1,
    load_dataset_to_memory=True,
    override_dataset=False,
    replace_min1=True,
    datasets_suffix=None,
    label_transform="none",
    random_state=42,
    verbose=True,
    assemblies=None,
    generator_cache_dir=DEFAULT_GENERATOR_CACHE_DIR,
    overwrite=False,
    dry_run=False,
):
    if dataset_base_dir is None:
        raise ValueError("dataset_base_dir is required")
    if base_model_location is None:
        raise ValueError("base_model_location is required")
    if len(bigwig_files) != len(names):
        raise ValueError("bigwig_files and names must have the same length")
    if chromosomes is None:
        chromosomes = ["chr5"]
    if token_label_binning is None:
        token_label_binning = DEFAULT_TOKEN_LABEL_BINNING
    if datasets_suffix is None:
        datasets_suffix = project_suffix

    summary = {"created": 0, "skipped": 0, "project_dirs": [created_configs_path]}
    pretrain_extraction_path = os.path.join(created_configs_path, "pretrain_extraction")
    retrain_extraction_path = os.path.join(created_configs_path, "retrain_extraction")
    variability_extraction_path = os.path.join(created_configs_path, "variability_extraction")
    pretrain_training_path = os.path.join(created_configs_path, "pretrain_training")
    retrain_training_path = os.path.join(created_configs_path, "retrain_training")

    for config_dir in [
        created_configs_path,
        pretrain_extraction_path,
        retrain_extraction_path,
        variability_extraction_path,
        pretrain_training_path,
        retrain_training_path,
        dataset_base_dir,
        base_model_location,
    ]:
        _ensure_dir(config_dir, dry_run=dry_run)

    created_datasets_variability_base_name_suffix = "_per_varaint_variability" + datasets_suffix
    created_datasets_pretrain_base_name_suffix = "_pretrain" + datasets_suffix
    created_datasets_retrain_base_name_suffix = "_retrain" + datasets_suffix
    pretrain_name_suffix = "_pretrain" + project_suffix
    retrain_name_suffix = "_retrain" + project_suffix

    def record(status):
        summary[status] += 1

    for name, bigwig_file in zip(names, bigwig_files):
        per_variant_data_path = os.path.join(
            dataset_base_dir,
            f"{name}{created_datasets_variability_base_name_suffix}.csv",
        )
        variability_config = create_variability_extraction_config(
            bigwig_files=bigwig_files,
            current_bigwig_file=bigwig_file,
            per_variant_data_path=per_variant_data_path,
            chromosomes=chromosomes,
            seq_size=seq_size,
            random_state=random_state,
            verbose=verbose,
            assemblies=assemblies,
            generator_cache_dir=generator_cache_dir,
        )
        record(
            _write_yaml(
                os.path.join(
                    variability_extraction_path,
                    f"{name}{created_datasets_variability_base_name_suffix}.yaml",
                ),
                variability_config,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )

        extraction_variant_file_path = variant_file_path or per_variant_data_path
        pretrain_dataset_name = name + created_datasets_pretrain_base_name_suffix
        pretrain_config = create_token_classification_extraction_config(
            raw_data_path=bigwig_file,
            dataset_base_dir=dataset_base_dir,
            created_datasets_base_name=pretrain_dataset_name,
            tokenizer_name=tokenizer_name,
            chromosomes=chromosomes,
            seq_size=seq_size,
            test_size=0,
            train_test_seperation=train_test_seperation,
            token_label_binning=token_label_binning,
            token_label_downsampling=token_label_downsampling,
            use_variant_filtering=use_variant_filtering,
            variant_filtering_upper_bound=pretraining_variant_filtering_upper_bound,
            variant_filtering_lower_bound=pretraining_variant_filtering_lower_bound,
            variant_file_path=extraction_variant_file_path,
            replace_min1=replace_min1,
            override_dataset=override_dataset,
            random_state=random_state,
            verbose=verbose,
            assemblies=assemblies,
            generator_cache_dir=generator_cache_dir,
        )
        record(
            _write_yaml(
                os.path.join(pretrain_extraction_path, f"{name}_pretrain_data_extraction{project_suffix}.yaml"),
                pretrain_config,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )

        retrain_dataset_name = name + created_datasets_retrain_base_name_suffix
        retrain_config = create_token_classification_extraction_config(
            raw_data_path=bigwig_file,
            dataset_base_dir=dataset_base_dir,
            created_datasets_base_name=retrain_dataset_name,
            tokenizer_name=tokenizer_name,
            chromosomes=chromosomes,
            seq_size=seq_size,
            test_size=test_size,
            train_test_seperation=train_test_seperation,
            token_label_binning=token_label_binning,
            token_label_downsampling=token_label_downsampling,
            use_variant_filtering=use_variant_filtering,
            variant_filtering_upper_bound=retraining_variant_filtering_upper_bound,
            variant_filtering_lower_bound=retraining_variant_filtering_lower_bound,
            variant_file_path=extraction_variant_file_path,
            replace_min1=replace_min1,
            override_dataset=override_dataset,
            random_state=random_state,
            verbose=verbose,
            assemblies=assemblies,
            generator_cache_dir=generator_cache_dir,
        )
        record(
            _write_yaml(
                os.path.join(retrain_extraction_path, f"{name}_retrain_data_extraction{project_suffix}.yaml"),
                retrain_config,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )

    for index, curr_base_name in enumerate(names):
        pretrain_train_paths = [
            os.path.join(dataset_base_dir, f"{other_name}{created_datasets_pretrain_base_name_suffix}_train")
            for other_index, other_name in enumerate(names)
            if other_index != index
        ]
        if pretrain_train_paths:
            pretrain_analysis_name = curr_base_name + pretrain_name_suffix
            pretrain_output_dir = os.path.join(base_model_location, pretrain_analysis_name)
            _ensure_dir(pretrain_output_dir, dry_run=dry_run)
            pretrain_training_config = create_token_classification_training_config(
                tokenizer_name=tokenizer_name,
                train_dataset_path=pretrain_train_paths,
                output_dir=pretrain_output_dir,
                analysis_name=pretrain_analysis_name,
                task_type=CPG_TRAINING_TASK_TYPE,
                num_train_epochs=num_pretrain_epoch,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                learning_rate=learning_rate,
                metric_for_best_model=metric_for_best_model,
                save_stratagy=save_stratagy,
                number_of_steps=number_of_steps,
                save_total_limit=save_total_limit,
                load_best_model_at_end=load_best_model_at_end,
                use_lora=use_lora,
                freeze_model=freeze_model,
                num_labels=num_labels,
                top_rows=top_rows,
                max_grad_norm=max_grad_norm,
                add_epoch_end_save_callback=add_epoch_end_save_callback,
                add_epoch_end_prediction=False,
                save_at_end=save_at_end,
                continue_from_last=continue_from_last,
                min_number_of_cpg_sites=min_number_of_cpg_sites,
                load_dataset_to_memory=load_dataset_to_memory,
                label_transform=label_transform,
                random_state=random_state,
                verbose=verbose,
                assemblies=assemblies,
                generator_cache_dir=generator_cache_dir,
            )
            record(
                _write_yaml(
                    os.path.join(pretrain_training_path, f"{curr_base_name}_pretrain_training{project_suffix}.yaml"),
                    pretrain_training_config,
                    overwrite=overwrite,
                    dry_run=dry_run,
                )
            )

        retrain_dataset_base = curr_base_name + created_datasets_retrain_base_name_suffix
        retrain_train_path = [os.path.join(dataset_base_dir, f"{retrain_dataset_base}_train")]
        retrain_eval_path = os.path.join(dataset_base_dir, f"{retrain_dataset_base}_test")

        no_pretraining_suffix = f"_{NO_PRETRAINING_CONFIG_NAME}{retrain_name_suffix}"
        no_pretraining_analysis_name = curr_base_name + no_pretraining_suffix
        no_pretraining_output_dir = os.path.join(base_model_location, no_pretraining_analysis_name)
        _ensure_dir(no_pretraining_output_dir, dry_run=dry_run)
        no_pretraining_config = create_token_classification_training_config(
            tokenizer_name=tokenizer_name,
            train_dataset_path=retrain_train_path,
            eval_dataset_path=retrain_eval_path,
            output_dir=no_pretraining_output_dir,
            analysis_name=no_pretraining_analysis_name,
            task_type=CPG_TRAINING_TASK_TYPE,
            num_train_epochs=num_train_epoch,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            metric_for_best_model=metric_for_best_model,
            save_stratagy=save_stratagy,
            number_of_steps=number_of_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            use_lora=use_lora,
            freeze_model=freeze_model,
            num_labels=num_labels,
            top_rows=top_rows,
            max_grad_norm=max_grad_norm,
            add_epoch_end_save_callback=False,
            add_epoch_end_prediction=True,
            save_at_end=save_at_end,
            continue_from_last=continue_from_last,
            min_number_of_cpg_sites=min_number_of_cpg_sites,
            load_dataset_to_memory=load_dataset_to_memory,
            label_transform=label_transform,
            random_state=random_state,
            verbose=verbose,
            assemblies=assemblies,
            generator_cache_dir=generator_cache_dir,
        )
        record(
            _write_yaml(
                os.path.join(
                    retrain_training_path,
                    f"{curr_base_name}_{NO_PRETRAINING_CONFIG_NAME}_retrain_training{project_suffix}.yaml",
                ),
                no_pretraining_config,
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )

        base_models_path = os.path.join(base_model_location, curr_base_name + pretrain_name_suffix)
        if not os.path.isdir(base_models_path):
            continue
        for checkpoint_name in sorted(os.listdir(base_models_path)):
            if not checkpoint_name.startswith("epoch"):
                continue
            retrain_suffix = f"_{checkpoint_name}{retrain_name_suffix}"
            retrain_analysis_name = curr_base_name + retrain_suffix
            retrain_output_dir = os.path.join(base_model_location, retrain_analysis_name)
            _ensure_dir(retrain_output_dir, dry_run=dry_run)
            retrain_config = create_token_classification_training_config(
                tokenizer_name=tokenizer_name,
                train_dataset_path=retrain_train_path,
                eval_dataset_path=retrain_eval_path,
                output_dir=retrain_output_dir,
                analysis_name=retrain_analysis_name,
                task_type=CPG_RETRAINING_TASK_TYPE,
                trained_model_path=os.path.join(base_models_path, checkpoint_name),
                num_train_epochs=num_train_epoch,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                learning_rate=learning_rate,
                metric_for_best_model=metric_for_best_model,
                save_stratagy=save_stratagy,
                number_of_steps=number_of_steps,
                save_total_limit=save_total_limit,
                load_best_model_at_end=load_best_model_at_end,
                use_lora=use_lora,
                freeze_model=freeze_model,
                num_labels=num_labels,
                top_rows=top_rows,
                max_grad_norm=max_grad_norm,
                add_epoch_end_save_callback=False,
                add_epoch_end_prediction=True,
                save_at_end=save_at_end,
                continue_from_last=continue_from_last,
                min_number_of_cpg_sites=min_number_of_cpg_sites,
                load_dataset_to_memory=load_dataset_to_memory,
                label_transform=label_transform,
                random_state=random_state,
                verbose=verbose,
                assemblies=assemblies,
                generator_cache_dir=generator_cache_dir,
            )
            record(
                _write_yaml(
                    os.path.join(retrain_training_path, f"{curr_base_name}_{checkpoint_name}_retrain_training{project_suffix}.yaml"),
                    retrain_config,
                    overwrite=overwrite,
                    dry_run=dry_run,
                )
            )

    return summary


def build_master_project_config(
    name,
    grouped_files,
    base_file_path,
    base_config_path,
    dataset_base_dir,
    base_model_location,
    filtration_method,
    project_prefix="_token_cls_",
    tokenizer_name=DEFAULT_TOKENIZER_NAME,
    token_label_binning=None,
    token_label_downsampling=DEFAULT_TOKEN_LABEL_DOWNSAMPLING,
    chromosomes=None,
    learning_rates=None,
    per_device_train_batch_sizes=None,
    per_device_eval_batch_size=1,
    seq_sizes=None,
    test_sizes=None,
    num_train_epoch=5,
    num_pretrain_epoch=2,
    batch_size_overrides=None,
):
    if token_label_binning is None:
        token_label_binning = DEFAULT_TOKEN_LABEL_BINNING
    if chromosomes is None:
        chromosomes = ["chr5"]
    if learning_rates is None:
        learning_rates = [1e-6]
    if per_device_train_batch_sizes is None:
        per_device_train_batch_sizes = [1]
    if seq_sizes is None:
        seq_sizes = [5400]
    if test_sizes is None:
        test_sizes = [0.2]

    master_config = {
        "params": {
            "project_suffix": project_prefix + name,
            "bigwig_files": [os.path.join(base_file_path, file_name) for file_name in grouped_files],
            "names": [file_name.replace(".hg38.bigwig", "").split("-")[-1] for file_name in grouped_files],
            "created_configs_path": os.path.join(base_config_path, "_" + name),
            "tokenizer_name": tokenizer_name,
            "dataset_base_dir": dataset_base_dir,
            "base_model_location": base_model_location,
            "model_type": CLASSIFICATION_ANALYSIS_SYMBOL,
            "train_test_seperation": filtration_method,
            "chromosomes": chromosomes,
            "use_lora": False,
            "freeze_model": False,
            "num_labels": 3,
            "load_best_model_at_end": False,
            "num_train_epoch": num_train_epoch,
            "num_pretrain_epoch": num_pretrain_epoch,
            "per_device_train_batch_sizes": per_device_train_batch_sizes,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "learning_rates": learning_rates,
            "metric_for_best_model": "mcc",
            "save_stratagy": "steps",
            "number_of_steps": 10000,
            "seq_sizes": seq_sizes,
            "test_sizes": test_sizes,
            "save_total_limit": 2,
            "add_epoch_end_save_callback": True,
            "save_at_end": False,
            "continue_from_last": True,
            "use_variant_filtering": False,
            "override_dataset": False,
            "pretraining_variant_filtering_upper_bound": -1,
            "pretraining_variant_filtering_lower_bound": -1,
            "retraining_variant_filtering_upper_bound": -1,
            "retraining_variant_filtering_lower_bound": -1,
            "max_grad_norm": 1,
            "top_rows": -1,
            "min_number_of_cpg_sites": 10,
            "load_dataset_to_memory": True,
            "label_transform": "none",
            "token_label_binning": copy.deepcopy(token_label_binning),
            "seq_batch_size_overrides": batch_size_overrides or {},
        }
    }
    if token_label_downsampling is not None:
        master_config["params"]["token_label_downsampling"] = copy.deepcopy(token_label_downsampling)
    return master_config


def _safe_load_yaml(path):
    try:
        with open(path, "r") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def _yaml_files(config_dir):
    if not os.path.isdir(config_dir):
        return []
    return [
        os.path.join(config_dir, name)
        for name in sorted(os.listdir(config_dir))
        if name.endswith(".yaml")
    ]


def _dataset_dir_ready(path):
    if not path or not os.path.isdir(path):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    if not entries:
        return False
    markers = {"dataset_info.json", "state.json", "data", "dataset_dict.json"}
    if any(marker in entries for marker in markers):
        return True
    return any(name.endswith(".arrow") or name.endswith(".parquet") for name in entries)


def _extraction_done(cfg):
    task = cfg.get("task", {})
    paths = cfg.get("paths", {})
    if task.get("type") == CPG_SEPERATING_SITES_TASK_NAME:
        out_path = paths.get("per_variant_data_path")
        return bool(out_path and os.path.exists(out_path))

    train_path = paths.get("hf_dataset_train_path")
    test_path = paths.get("hf_dataset_test_path")
    test_size = float(task.get("test_size", 0.0) or 0.0)
    if not _dataset_dir_ready(train_path):
        return False
    if test_size == 0:
        return True
    return _dataset_dir_ready(test_path)


def _epoch_checkpoints(model_dir):
    if not model_dir or not os.path.isdir(model_dir):
        return []
    return [name for name in os.listdir(model_dir) if name.startswith("epoch-")]


def _training_status(cfg):
    out_dir = cfg.get("paths", {}).get("output_dir")
    expected_epochs = int(cfg.get("train", {}).get("num_train_epochs", 0) or 0)
    if expected_epochs <= 0:
        expected_epochs = 1
    if not out_dir or not os.path.isdir(out_dir):
        return "not_started"
    epoch_checkpoints = _epoch_checkpoints(out_dir)
    if not epoch_checkpoints:
        return "not_started"
    if len(epoch_checkpoints) < expected_epochs:
        return "started_not_finished"
    return "finished"


def build_job_name(project_prefix, stage, yaml_path):
    cfg = _safe_load_yaml(yaml_path)
    paths = cfg.get("paths", {})
    task = cfg.get("task", {})
    train = cfg.get("train", {})

    analysis = task.get("analysis_name")
    sample = None
    out_dir = paths.get("output_dir")
    if out_dir:
        sample = os.path.basename(out_dir).split("_")[0]
    if not sample and analysis:
        sample = analysis.split("_")[0]
    if not sample:
        sample = os.path.basename(yaml_path).split("_")[0]

    def _fmt_lr(value):
        if value is None:
            return "na"
        try:
            return f"{float(value):.0e}".replace("+0", "").replace("+", "")
        except Exception:
            return str(value)

    parts = [
        str(stage),
        str(sample),
        str(analysis) if analysis else "na",
        f"lr{_fmt_lr(train.get('learning_rate'))}",
        f"bs{train.get('per_device_train_batch_size', 'na')}",
        f"s{task.get('seq_size', 'na')}",
    ]
    return ":".join(parts).replace(" ", "_").replace("/", "-")[:120]


def _extract_command(extract_script, yaml_path):
    return f"{extract_script} {yaml_path}"


def _train_command(cfg, yaml_path, stage, project_prefix, train_script, sbatch_prefix=None):
    batch_size = int(cfg.get("train", {}).get("per_device_train_batch_size", 2) or 2)
    gpus = max(2, min(math.ceil(batch_size / 3), 8))
    if sbatch_prefix is None:
        sbatch_prefix = f"sbatch --mem=45g -c15 --gres=gg:g4:{gpus} --time=5-23 --killable --requeue"
    job_name = build_job_name(project_prefix, stage, yaml_path)
    return f'{sbatch_prefix} --job-name="{job_name}" --wrap="{train_script} {yaml_path}"'


def _print_extract_pending(yaml_paths, extract_script):
    printed = 0
    for yaml_path in yaml_paths:
        cfg = _safe_load_yaml(yaml_path)
        if _extraction_done(cfg):
            continue
        print(_extract_command(extract_script, yaml_path))
        printed += 1
    if printed == 0:
        print("# none pending")


def _print_train_pending(yaml_paths, stage, project_prefix, train_script, sbatch_prefix=None):
    not_started = []
    started_not_finished = []
    for yaml_path in yaml_paths:
        cfg = _safe_load_yaml(yaml_path)
        status = _training_status(cfg)
        if status == "finished":
            continue
        cmd = _train_command(cfg, yaml_path, stage, project_prefix, train_script, sbatch_prefix=sbatch_prefix)
        if status == "started_not_finished":
            started_not_finished.append(cmd)
        else:
            not_started.append(cmd)

    print("# training not started")
    if not_started:
        for cmd in not_started:
            print(cmd)
    else:
        print("# none pending")

    print("# training started but not finished")
    if started_not_finished:
        for cmd in started_not_finished:
            print(cmd)
    else:
        print("# none pending")


def print_commands_for_project_dirs(project_dirs, project_prefix, extract_script=DEFAULT_EXTRACT_SCRIPT, train_script=DEFAULT_TRAIN_SCRIPT, sbatch_prefix=None):
    for project_dir in project_dirs:
        print(f"# commands for {project_dir}")

        print("# run variability extraction:")
        _print_extract_pending(_yaml_files(os.path.join(project_dir, "variability_extraction")), extract_script)

        print("# run pretrain extraction:")
        _print_extract_pending(_yaml_files(os.path.join(project_dir, "pretrain_extraction")), extract_script)

        print("# run pretrain training:")
        _print_train_pending(
            _yaml_files(os.path.join(project_dir, "pretrain_training")),
            stage="pre",
            project_prefix=project_prefix,
            train_script=train_script,
            sbatch_prefix=sbatch_prefix,
        )

        print("# run retrain extraction:")
        _print_extract_pending(_yaml_files(os.path.join(project_dir, "retrain_extraction")), extract_script)

        print("# run retrain training:")
        _print_train_pending(
            _yaml_files(os.path.join(project_dir, "retrain_training")),
            stage="ret",
            project_prefix=project_prefix,
            train_script=train_script,
            sbatch_prefix=sbatch_prefix,
        )


def create_token_classification_project_from_master_config(master_config_path, overwrite=False, dry_run=False):
    with open(master_config_path, "r") as handle:
        config = yaml.safe_load(handle) or {}
    params = copy.deepcopy(config["params"])
    raw_params = copy.deepcopy(params)

    base_suffix = params["project_suffix"]
    configs_base_dir = params["created_configs_path"]
    dataset_project_dir = os.path.join(params["dataset_base_dir"], base_suffix)
    model_project_dir = os.path.join(params["base_model_location"], base_suffix)

    learning_rates = _as_list(params.pop("learning_rates", None), [params.get("learning_rate", 1e-6)])
    batch_sizes = _as_list(
        params.pop("per_device_train_batch_sizes", None),
        [params.get("per_device_train_batch_size", 1)],
    )
    seq_sizes = _as_list(params.pop("seq_sizes", None), [params.get("seq_size", 5400)])
    test_sizes = _as_list(params.pop("test_sizes", None), [params.get("test_size", 0.2)])
    seq_batch_size_overrides = params.pop("seq_batch_size_overrides", {})
    params.pop("model_type", None)

    _ensure_dir(configs_base_dir, dry_run=dry_run)
    _ensure_dir(dataset_project_dir, dry_run=dry_run)
    _ensure_dir(model_project_dir, dry_run=dry_run)

    include_test_size = len(test_sizes) > 1 or any(float(value) != 0.2 for value in test_sizes)
    project_dirs = []
    total_created = 0
    total_skipped = 0

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for seq_size in seq_sizes:
                for test_size in test_sizes:
                    effective_batch_size = _effective_batch_size(batch_size, seq_size, seq_batch_size_overrides)
                    curr_suffix = _project_suffix(base_suffix, learning_rate, batch_size, seq_size, test_size, include_test_size)
                    datasets_suffix = f"{base_suffix}_seq_{seq_size}_datasets"
                    curr_created_configs_path = os.path.join(configs_base_dir, curr_suffix)
                    curr_params = copy.deepcopy(params)
                    curr_params.update(
                        {
                            "project_suffix": curr_suffix,
                            "datasets_suffix": datasets_suffix,
                            "created_configs_path": curr_created_configs_path,
                            "dataset_base_dir": dataset_project_dir,
                            "base_model_location": model_project_dir,
                            "learning_rate": learning_rate,
                            "per_device_train_batch_size": effective_batch_size,
                            "per_device_eval_batch_size": effective_batch_size
                            if "per_device_eval_batch_size" not in raw_params
                            else raw_params["per_device_eval_batch_size"],
                            "seq_size": seq_size,
                            "test_size": test_size,
                        }
                    )
                    summary = create_token_classification_project_config(
                        overwrite=overwrite,
                        dry_run=dry_run,
                        **curr_params,
                    )
                    project_dirs.extend(summary["project_dirs"])
                    total_created += summary["created"]
                    total_skipped += summary["skipped"]

    return {
        "created": total_created,
        "skipped": total_skipped,
        "project_dirs": project_dirs,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Create token-classification extraction and training project configs.")
    parser.add_argument("master_config_path", nargs="?", help="Path to a token-classification master project YAML.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated YAML files.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created without writing files.")
    parser.add_argument("--no-print-commands", action="store_true", help="Do not print extraction/training commands.")
    parser.add_argument("--extract-script", default=str(DEFAULT_EXTRACT_SCRIPT), help="Data extraction runner script.")
    parser.add_argument("--train-script", default=str(DEFAULT_TRAIN_SCRIPT), help="Training runner script.")
    parser.add_argument("--sbatch-prefix", default="", help="Optional training sbatch prefix. Defaults to the old project-creator sbatch format.")
    return parser.parse_args()


def main():
    args = parse_args()
    master_config_path = args.master_config_path
    if not master_config_path:
        master_config_path = input("enter master token-classification project config path: ").strip()
    if not master_config_path.endswith(".yaml") or not os.path.exists(master_config_path):
        raise ValueError(f"invalid config path: {master_config_path}")

    summary = create_token_classification_project_from_master_config(
        master_config_path,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"created {summary['created']} configs, skipped {summary['skipped']} existing configs")
    if not args.no_print_commands:
        print_commands_for_project_dirs(
            summary["project_dirs"],
            project_prefix=_safe_load_yaml(master_config_path).get("params", {}).get("project_suffix", "token_cls"),
            extract_script=args.extract_script,
            train_script=args.train_script,
            sbatch_prefix=args.sbatch_prefix or None,
        )


if __name__ == "__main__":
    main()
