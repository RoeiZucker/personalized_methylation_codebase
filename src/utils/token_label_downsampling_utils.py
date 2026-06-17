from collections import Counter
import math

import numpy as np


TOKEN_LABEL_DOWNSAMPLING_CONFIG_KEY = "token_label_downsampling"
DEFAULT_BLANK_LABEL = -100
DEFAULT_NUM_LABELS = 3


def _labels_as_iterable(labels):
    if isinstance(labels, np.ndarray):
        return labels.tolist()
    if isinstance(labels, (list, tuple)):
        return labels
    return [labels]


def normalize_token_label_downsampling_config(
    task_config,
    blank_label=DEFAULT_BLANK_LABEL,
    seed=None,
):
    config = task_config.get(TOKEN_LABEL_DOWNSAMPLING_CONFIG_KEY)
    if config is None or config.get("enabled", True) is False:
        return None

    if "minority_to_majority_ratio" not in config:
        raise ValueError(
            "Missing required token label downsampling config value: "
            "minority_to_majority_ratio"
        )

    ratio = float(config["minority_to_majority_ratio"])
    if ratio <= 0 or ratio > 1:
        raise ValueError(
            "token_label_downsampling.minority_to_majority_ratio must be in "
            f"(0, 1], got {ratio}"
        )

    num_labels = int(config.get("num_labels", task_config.get("num_labels", DEFAULT_NUM_LABELS)))
    if num_labels <= 0:
        raise ValueError(f"token_label_downsampling num_labels must be positive, got {num_labels}")

    return {
        "minority_to_majority_ratio": ratio,
        "blank_label": int(config.get("blank_label", blank_label)),
        "num_labels": num_labels,
        "seed": int(config.get("seed", seed if seed is not None else 42)),
    }


def count_token_class_labels(dataset, label_column="labels", blank_label=DEFAULT_BLANK_LABEL, class_ids=None):
    counts = Counter()
    if class_ids is not None:
        counts.update({int(class_id): 0 for class_id in class_ids})

    for example in dataset:
        for label in _labels_as_iterable(example[label_column]):
            label_int = int(label)
            if label_int != int(blank_label):
                counts[label_int] += 1
    return counts


def _build_keep_occurrence_indices(counts, class_ids, ratio, seed):
    missing_classes = [class_id for class_id in class_ids if counts[class_id] == 0]
    if missing_classes:
        raise ValueError(
            "Cannot apply token label downsampling because these token classes "
            f"are missing from the training labels: {missing_classes}"
        )

    minority_count = min(counts[class_id] for class_id in class_ids)
    target_max_count = int(math.floor(minority_count / ratio))
    target_max_count = max(minority_count, target_max_count)
    rng = np.random.default_rng(seed)
    keep_occurrence_indices = {}

    for class_id in class_ids:
        class_count = counts[class_id]
        if class_count <= target_max_count:
            continue
        keep_indices = rng.choice(class_count, size=target_max_count, replace=False)
        keep_occurrence_indices[class_id] = set(int(index) for index in keep_indices)

    return keep_occurrence_indices, target_max_count


def _filter_all_blank_examples(dataset, label_column, blank_label):
    return dataset.filter(
        lambda example: any(int(label) != int(blank_label) for label in _labels_as_iterable(example[label_column])),
        load_from_cache_file=False,
    )


def apply_token_label_downsampling_to_dataset(
    dataset,
    task_config,
    label_column="labels",
    blank_label=DEFAULT_BLANK_LABEL,
    seed=None,
    verbose=False,
):
    config = normalize_token_label_downsampling_config(task_config, blank_label=blank_label, seed=seed)
    if dataset is None or config is None:
        return dataset

    class_ids = list(range(config["num_labels"]))
    effective_blank_label = config["blank_label"]
    counts_before = count_token_class_labels(
        dataset,
        label_column=label_column,
        blank_label=effective_blank_label,
        class_ids=class_ids,
    )
    keep_occurrence_indices, target_max_count = _build_keep_occurrence_indices(
        counts_before,
        class_ids,
        config["minority_to_majority_ratio"],
        config["seed"],
    )

    if not keep_occurrence_indices:
        if verbose:
            print(
                "token label downsampling skipped; all classes already satisfy ratio:",
                {
                    "counts": dict(counts_before),
                    "minority_to_majority_ratio": config["minority_to_majority_ratio"],
                },
                flush=True,
            )
        return dataset

    seen_counts = {class_id: 0 for class_id in class_ids}

    def _map_batch(batch):
        downsampled_labels = []
        for labels in batch[label_column]:
            new_labels = []
            for label in _labels_as_iterable(labels):
                label_int = int(label)
                if label_int == effective_blank_label:
                    new_labels.append(effective_blank_label)
                    continue
                if label_int not in seen_counts:
                    raise ValueError(f"Unexpected token class label during downsampling: {label_int}")

                occurrence_index = seen_counts[label_int]
                seen_counts[label_int] += 1
                keep_indices = keep_occurrence_indices.get(label_int)
                if keep_indices is None or occurrence_index in keep_indices:
                    new_labels.append(label_int)
                else:
                    new_labels.append(effective_blank_label)
            downsampled_labels.append(new_labels)
        return {label_column: downsampled_labels}

    dataset = dataset.map(_map_batch, batched=True, load_from_cache_file=False)
    dataset = _filter_all_blank_examples(dataset, label_column, effective_blank_label)

    if verbose:
        counts_after = count_token_class_labels(
            dataset,
            label_column=label_column,
            blank_label=effective_blank_label,
            class_ids=class_ids,
        )
        print(
            "token label downsampling:",
            {
                "minority_to_majority_ratio": config["minority_to_majority_ratio"],
                "target_max_count": target_max_count,
                "counts_before": dict(counts_before),
                "counts_after": dict(counts_after),
            },
            flush=True,
        )

    return dataset
