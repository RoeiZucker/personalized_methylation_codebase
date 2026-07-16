from datasets import Dataset

try:
    from .constants import DATASET_BATCH_SIZE, KMER_SAMPLE_TRAIN_TEST_FILTRATION
    from .utils.tissue_dataset_utils import dataset_generator_wrapper
except ImportError:
    from constants import DATASET_BATCH_SIZE, KMER_SAMPLE_TRAIN_TEST_FILTRATION
    from utils.tissue_dataset_utils import dataset_generator_wrapper


def save_tissue_huggingface_dataset(dataframe_path, save_path, tokenizer, kmer_size, seq_size, verbose, tissue_id):
    dataset = Dataset.from_generator(
        dataset_generator_wrapper(
            dataframe_path,
            DATASET_BATCH_SIZE,
            kmer_size,
            seq_size,
            tissue_id=tissue_id,
        )
    )
    encoded_dataset = dataset.map(lambda examples: tokenizer(examples["seq"])).remove_columns(["seq"])
    save_encoded_tissue_dataset(encoded_dataset, save_path, verbose)


def save_encoded_tissue_dataset(dataset, save_path, verbose):
    if verbose:
        print("saving dataset to:", save_path, flush=True)
    dataset.save_to_disk(save_path, num_proc=1)


def has_eval_dataset(eval_size):
    return eval_size is not None and eval_size > 0


def validate_eval_dataset_params(hf_dataset_eval_path, eval_size):
    if eval_size is None:
        return
    if eval_size < 0 or eval_size >= 1:
        raise ValueError("eval_size must be greater than or equal to 0 and smaller than 1")
    if has_eval_dataset(eval_size) and hf_dataset_eval_path is None:
        raise ValueError("eval_size was provided but hf_dataset_eval_path is missing")


def save_test_eval_tissue_datasets(test_ds, hf_dataset_test_path, hf_dataset_eval_path, eval_size, random_state, verbose):
    validate_eval_dataset_params(hf_dataset_eval_path, eval_size)
    if not has_eval_dataset(eval_size):
        save_encoded_tissue_dataset(test_ds, hf_dataset_test_path, verbose)
        return

    split = test_ds.train_test_split(test_size=eval_size, seed=random_state)
    save_encoded_tissue_dataset(split["train"], hf_dataset_test_path, verbose)
    save_encoded_tissue_dataset(split["test"], hf_dataset_eval_path, verbose)


def create_tissue_huggingface_datasets(
    *,
    intermediate_train_data_path,
    intermediate_test_data_path,
    hf_dataset_train_path,
    hf_dataset_test_path,
    hf_dataset_eval_path,
    tokenizer,
    train_test_seperation_type,
    blank_label,
    test_size,
    eval_size,
    random_state,
    kmer_size,
    seq_size,
    verbose,
    train_only,
    tissue_id,
    map_seperate_window_labels_wrapper,
):
    if train_test_seperation_type == KMER_SAMPLE_TRAIN_TEST_FILTRATION:
        dataset = Dataset.from_generator(
            dataset_generator_wrapper(
                intermediate_train_data_path,
                DATASET_BATCH_SIZE,
                kmer_size,
                seq_size,
                tissue_id=tissue_id,
            )
        )
        new_dataset = dataset.map(map_seperate_window_labels_wrapper(blank_label, 1 - test_size, random_state))
        train_ds = new_dataset.remove_columns(["test_labels", "labels"]).rename_column("train_labels", "labels")
        test_ds = new_dataset.remove_columns(["train_labels", "labels"]).rename_column("test_labels", "labels")

        train_ds = train_ds.filter(lambda example: set(example["labels"]) != {blank_label})
        if test_size > 0:
            test_ds = test_ds.filter(lambda example: set(example["labels"]) != {blank_label})
        encoded_train = train_ds.map(lambda examples: tokenizer(examples["seq"])).remove_columns(["seq"])
        encoded_test = test_ds.map(lambda examples: tokenizer(examples["seq"])).remove_columns(["seq"])

        if verbose:
            print("saving train dataset to:", hf_dataset_train_path, flush=True)
            print("saving test dataset to:", hf_dataset_test_path, flush=True)
            if has_eval_dataset(eval_size):
                print("saving eval dataset to:", hf_dataset_eval_path, flush=True)
        encoded_train.save_to_disk(hf_dataset_train_path, num_proc=1)
        if test_size > 0:
            save_test_eval_tissue_datasets(encoded_test, hf_dataset_test_path, hf_dataset_eval_path, eval_size, random_state, verbose)
        return

    save_tissue_huggingface_dataset(
        intermediate_train_data_path,
        hf_dataset_train_path,
        tokenizer,
        kmer_size,
        seq_size,
        verbose,
        tissue_id=tissue_id,
    )
    if not train_only:
        dataset = Dataset.from_generator(
            dataset_generator_wrapper(
                intermediate_test_data_path,
                DATASET_BATCH_SIZE,
                kmer_size,
                seq_size,
                tissue_id=tissue_id,
            )
        )
        encoded_dataset = dataset.map(lambda examples: tokenizer(examples["seq"])).remove_columns(["seq"])
        save_test_eval_tissue_datasets(
            encoded_dataset,
            hf_dataset_test_path,
            hf_dataset_eval_path,
            eval_size,
            random_state,
            verbose,
        )
