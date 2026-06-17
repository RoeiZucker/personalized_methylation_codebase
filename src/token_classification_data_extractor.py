import os

from datasets import Dataset
from transformers import AutoTokenizer

try:
    from .constants import (
        DATASET_BATCH_SIZE,
        INTERMEDIATE_INPUT_NAME,
        INSTADEEP_KMER_SIZE,
        KMER_SAMPLE_TRAIN_TEST_FILTRATION,
        PREPROCESSED_INPUT_NAME,
        RAW_INPUT_NAME,
    )
    from .data_extractor import (
        create_intermediate_data_files,
        load_chrom_dict,
        load_preprocessed_data,
        map_seperate_window_labels_wrapper,
        save_preprocessed_data,
    )
    from .utils.dataset_utils import dataset_generator_wrapper
    from .utils.token_label_binning_utils import (
        apply_token_label_binning_to_dataset,
        resolve_token_label_binning,
    )
    from .utils.token_label_downsampling_utils import apply_token_label_downsampling_to_dataset
    from .utils.variability_utils import load_variability_dict
except ImportError:
    from constants import (
        DATASET_BATCH_SIZE,
        INTERMEDIATE_INPUT_NAME,
        INSTADEEP_KMER_SIZE,
        KMER_SAMPLE_TRAIN_TEST_FILTRATION,
        PREPROCESSED_INPUT_NAME,
        RAW_INPUT_NAME,
    )
    from data_extractor import (
        create_intermediate_data_files,
        load_chrom_dict,
        load_preprocessed_data,
        map_seperate_window_labels_wrapper,
        save_preprocessed_data,
    )
    from utils.dataset_utils import dataset_generator_wrapper
    from utils.token_label_binning_utils import (
        apply_token_label_binning_to_dataset,
        resolve_token_label_binning,
    )
    from utils.token_label_downsampling_utils import apply_token_label_downsampling_to_dataset
    from utils.variability_utils import load_variability_dict


def _get_dataset_generator(task, dataframe_path, kmer_size, seq_size):
    tissue_id = task.get("tissue_id")
    if tissue_id is None:
        return dataset_generator_wrapper(dataframe_path, DATASET_BATCH_SIZE, kmer_size, seq_size)

    try:
        from .utils.tissue_dataset_utils import dataset_generator_wrapper as tissue_dataset_generator_wrapper
    except ImportError:
        from utils.tissue_dataset_utils import dataset_generator_wrapper as tissue_dataset_generator_wrapper

    return tissue_dataset_generator_wrapper(
        dataframe_path,
        DATASET_BATCH_SIZE,
        kmer_size,
        seq_size,
        tissue_id=tissue_id,
    )


def _load_token_dataset(task, dataframe_path, kmer_size, seq_size):
    return Dataset.from_generator(_get_dataset_generator(task, dataframe_path, kmer_size, seq_size))


def _tokenize_dataset(dataset, tokenizer):
    return dataset.map(lambda examples: tokenizer(examples["seq"])).remove_columns(["seq"])


def _save_dataset(dataset, save_path, verbose):
    if verbose:
        print("saving dataset to:", save_path, flush=True)
    dataset.save_to_disk(save_path, num_proc=1)


def _create_token_classification_hf_datasets(
    task,
    paths,
    tokenizer,
    train_test_seperation_type,
    blank_label,
    test_size,
    random_state,
    kmer_size,
    seq_size,
    train_only,
    verbose,
):
    if train_test_seperation_type == KMER_SAMPLE_TRAIN_TEST_FILTRATION:
        dataset = _load_token_dataset(task, paths["intermediate_train_data_path"], kmer_size, seq_size)
        new_dataset = dataset.map(map_seperate_window_labels_wrapper(blank_label, 1 - test_size, random_state))
        train_ds = new_dataset.remove_columns(["test_labels", "labels"]).rename_column("train_labels", "labels")
        test_ds = new_dataset.remove_columns(["train_labels", "labels"]).rename_column("test_labels", "labels")

        train_ds = train_ds.filter(lambda example: set(example["labels"]) != {blank_label})
        if test_size > 0:
            test_ds = test_ds.filter(lambda example: set(example["labels"]) != {blank_label})

        resolved_binning = resolve_token_label_binning(task, train_ds)
        train_ds = apply_token_label_binning_to_dataset(train_ds, resolved_binning)
        train_ds = apply_token_label_downsampling_to_dataset(
            train_ds,
            task,
            blank_label=resolved_binning.blank_label,
            seed=random_state,
            verbose=verbose,
        )
        if test_size > 0:
            test_ds = apply_token_label_binning_to_dataset(test_ds, resolved_binning)

        encoded_train = _tokenize_dataset(train_ds, tokenizer)
        _save_dataset(encoded_train, paths["hf_dataset_train_path"], verbose)
        if test_size > 0:
            encoded_test = _tokenize_dataset(test_ds, tokenizer)
            _save_dataset(encoded_test, paths["hf_dataset_test_path"], verbose)
        return resolved_binning

    train_ds = _load_token_dataset(task, paths["intermediate_train_data_path"], kmer_size, seq_size)
    resolved_binning = resolve_token_label_binning(task, train_ds)
    train_ds = apply_token_label_binning_to_dataset(train_ds, resolved_binning)
    train_ds = apply_token_label_downsampling_to_dataset(
        train_ds,
        task,
        blank_label=resolved_binning.blank_label,
        seed=random_state,
        verbose=verbose,
    )
    encoded_train = _tokenize_dataset(train_ds, tokenizer)
    _save_dataset(encoded_train, paths["hf_dataset_train_path"], verbose)

    if not train_only:
        test_ds = _load_token_dataset(task, paths["intermediate_test_data_path"], kmer_size, seq_size)
        test_ds = apply_token_label_binning_to_dataset(test_ds, resolved_binning)
        encoded_test = _tokenize_dataset(test_ds, tokenizer)
        _save_dataset(encoded_test, paths["hf_dataset_test_path"], verbose)
    return resolved_binning


def _maybe_create_intermediate_data(cfg):
    task = cfg["task"]
    paths = cfg["paths"]
    testing_params = cfg["testing_params"]
    verbose = cfg["verbose"]
    input_mode = task["input_mode"]
    output_intermediate_data = task["output_intermediate_data"]
    output_hf_dataset = task["output_hf_dataset"]

    if (
        not output_intermediate_data
        and not output_hf_dataset
        and not task.get("output_preprocessed_data", False)
    ):
        return
    if input_mode == INTERMEDIATE_INPUT_NAME:
        if output_intermediate_data:
            raise ValueError("tried to recreate exactly the same input, probably something went wrong")
        if verbose:
            print(
                f"Loading intermediate data from already created "
                f"{paths['intermediate_train_data_path']} and {paths['intermediate_test_data_path']}.",
                flush=True,
            )
        return

    preprocessed_df = None
    if input_mode in {RAW_INPUT_NAME, PREPROCESSED_INPUT_NAME}:
        preprocessed_df = load_preprocessed_data(
            input_mode,
            verbose,
            paths["raw_data_path"],
            task.get("chromosomes", None),
            paths.get("pre_processed_data_path", None),
            task.get("replace_min1", True),
        )
        raw_data_top_rows = testing_params.get("raw_data_top_rows", -1)
        if testing_params["test_mode"] and raw_data_top_rows > -1:
            preprocessed_df = preprocessed_df.head(raw_data_top_rows)
        if task.get("output_preprocessed_data", False):
            save_preprocessed_data(verbose, paths.get("pre_processed_data_path", None), preprocessed_df)

    if not output_intermediate_data:
        return

    variant_filtering = cfg.get("variant_filtering", dict())
    variability_dict = load_variability_dict(
        paths.get("variant_file_path", None),
        use_variant_filtering=variant_filtering.get("use_variant_filtering", False),
        verbose=verbose,
    )
    preprocessed_df["value"] = preprocessed_df[task["value_column"]]
    create_intermediate_data_files(
        task,
        paths,
        task.get("train_test_seperation", None),
        task.get("window_type", None),
        load_chrom_dict(cfg, task),
        task.get("test_size", 0.2),
        cfg.get("random_state", 42),
        task.get("shuffle", True),
        task["seq_size"],
        testing_params["test_mode"],
        task["blank_label"],
        verbose,
        paths.get("train_window_names_path", None),
        paths.get("test_window_names_path", None),
        preprocessed_df,
        variability_dict,
        variant_filtering.get("variant_filtering_upper_bound", -1),
        variant_filtering.get("variant_filtering_lower_bound", -1),
    )


def encode_cpg_token_classification_extraction(cfg):
    task = cfg["task"]
    paths = cfg["paths"]
    verbose = cfg.get("verbose", True)
    test_size = task.get("test_size", 0.2)
    train_only = test_size == 0

    if os.path.exists(paths["hf_dataset_train_path"]) and not task.get("override_dataset", False):
        raise ValueError("HF dataset exists and override is False or not supplied" + paths["hf_dataset_train_path"])

    _maybe_create_intermediate_data(cfg)
    if not task["output_hf_dataset"]:
        return

    if verbose:
        print("Creating token-classification Hugging Face dataset from intermediate data.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(task["tokenizer_name"])
    resolved_binning = _create_token_classification_hf_datasets(
        task=task,
        paths=paths,
        tokenizer=tokenizer,
        train_test_seperation_type=task.get("train_test_seperation", None),
        blank_label=task["blank_label"],
        test_size=test_size,
        random_state=cfg.get("random_state", 42),
        kmer_size=task.get("kmer_size", INSTADEEP_KMER_SIZE),
        seq_size=task["seq_size"],
        train_only=train_only,
        verbose=verbose,
    )
    if verbose:
        print(
            "resolved token label binning:",
            {
                "method": resolved_binning.method,
                "low": resolved_binning.low,
                "high": resolved_binning.high,
                "resolved_low": resolved_binning.resolved_low,
                "resolved_high": resolved_binning.resolved_high,
            },
            flush=True,
        )
