import math
import numpy as np

try:
    from scipy.special import expit
except ImportError:
    expit = None


SUPPORTED_LABEL_TRANSFORMS = {"none", "log1p"}


def normalize_label_transform(label_transform):
    if label_transform is None:
        return "none"
    normalized = str(label_transform).strip().lower()
    if normalized == "":
        return "none"
    if normalized not in SUPPORTED_LABEL_TRANSFORMS:
        raise ValueError(
            f"Unsupported label transform '{label_transform}'. "
            f"Supported values: {sorted(SUPPORTED_LABEL_TRANSFORMS)}"
        )
    return normalized


def get_task_label_transform(task_config):
    if task_config is None:
        return "none"
    return normalize_label_transform(task_config.get("label_transform", "none"))


def transform_label_values(values, label_transform, blank_label=-100):
    normalized = normalize_label_transform(label_transform)
    if normalized == "none":
        return list(values)

    transformed = []
    for value in values:
        if value == blank_label:
            transformed.append(value)
            continue
        if value <= -1:
            raise ValueError(
                f"log1p label transform requires label values greater than -1, got {value}"
            )
        transformed.append(math.log1p(value))
    return transformed


def apply_label_transform_to_dataset(
    dataset,
    label_transform,
    label_column="labels",
    blank_label=-100,
    verbose=False,
    dataset_name="dataset",
):
    normalized = normalize_label_transform(label_transform)
    if normalized == "none" or dataset is None:
        return dataset
    if label_column not in getattr(dataset, 'column_names', []):
        return dataset

    def _map_batch(batch):
        return {
            label_column: [
                transform_label_values(values, normalized, blank_label=blank_label)
                for values in batch[label_column]
            ]
        }

    if verbose:
        print(f"Applying {normalized} label transform to {dataset_name}.", flush=True)

    return dataset.map(_map_batch, batched=True)


def maybe_decode_regression_predictions(predictions, label_transform):
    normalized = normalize_label_transform(label_transform)
    if normalized != "log1p":
        return predictions
    if expit is None:
        raise ImportError("scipy is required to decode regression predictions for log1p labels")
    return expit(np.asarray(predictions))
