from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


TOKEN_LABEL_BINNING_CONFIG_KEY = "token_label_binning"
FIXED_BINNING_METHOD = "fixed"
QUANTILE_BINNING_METHOD = "quantile"

CLOSE_TO_ZERO_CLASS_ID = 0
IN_BETWEEN_CLASS_ID = 1
CLOSE_TO_ONE_CLASS_ID = 2


@dataclass(frozen=True)
class ResolvedTokenLabelBinning:
    method: str
    low: float
    high: float
    resolved_low: float
    resolved_high: float
    blank_label: int = -100


def _get_config_value(config, names, config_name):
    for name in names:
        if name in config:
            return config[name]
    raise ValueError(f"Missing required token label binning config value: {config_name}")


def normalize_token_label_binning_config(task_config):
    config = task_config.get(TOKEN_LABEL_BINNING_CONFIG_KEY, task_config)
    if config is None:
        raise ValueError(f"Missing task.{TOKEN_LABEL_BINNING_CONFIG_KEY} config")

    if config.get("enabled", True) is False:
        raise ValueError("Token label binning task was selected, but token_label_binning.enabled is false")

    method = str(config.get("method", FIXED_BINNING_METHOD)).strip().lower()
    if method not in {FIXED_BINNING_METHOD, QUANTILE_BINNING_METHOD}:
        raise ValueError(
            f"Unsupported token label binning method '{method}'. "
            f"Supported methods: {FIXED_BINNING_METHOD}, {QUANTILE_BINNING_METHOD}"
        )

    low = float(_get_config_value(config, ("low", "lower", "lower_threshold"), "low"))
    high = float(_get_config_value(config, ("high", "upper", "upper_threshold"), "high"))
    if low >= high:
        raise ValueError(f"Token label binning low must be smaller than high, got {low} >= {high}")

    blank_label = int(config.get("blank_label", -100))
    if method == QUANTILE_BINNING_METHOD and (low < 0 or high > 1):
        raise ValueError("Quantile token label binning requires 0 <= low < high <= 1")

    return {
        "method": method,
        "low": low,
        "high": high,
        "blank_label": blank_label,
    }


def iter_non_blank_token_label_values(dataset, label_column="labels", blank_label=-100):
    for example in dataset:
        labels = example[label_column]
        if not isinstance(labels, (list, tuple, np.ndarray)):
            labels = [labels]
        for label in labels:
            if float(label) != float(blank_label):
                yield float(label)


def resolve_token_label_binning(task_config, train_dataset=None, label_column="labels"):
    config = normalize_token_label_binning_config(task_config)
    method = config["method"]
    low = config["low"]
    high = config["high"]
    blank_label = config["blank_label"]

    if method == FIXED_BINNING_METHOD:
        resolved_low = low
        resolved_high = high
    else:
        if train_dataset is None:
            raise ValueError("Quantile token label binning requires a train dataset")
        values = np.array(
            list(iter_non_blank_token_label_values(train_dataset, label_column, blank_label)),
            dtype=np.float64,
        )
        if values.size == 0:
            raise ValueError("Cannot resolve token label quantiles because the train dataset has no non-blank labels")
        resolved_low = float(np.quantile(values, low))
        resolved_high = float(np.quantile(values, high))
        if resolved_low >= resolved_high:
            raise ValueError(
                "Resolved token label quantile thresholds are not distinct: "
                f"{resolved_low} >= {resolved_high}"
            )

    return ResolvedTokenLabelBinning(
        method=method,
        low=low,
        high=high,
        resolved_low=resolved_low,
        resolved_high=resolved_high,
        blank_label=blank_label,
    )


def bin_token_label_value(value, resolved_binning: ResolvedTokenLabelBinning):
    if float(value) == float(resolved_binning.blank_label):
        return resolved_binning.blank_label
    if float(value) <= resolved_binning.resolved_low:
        return CLOSE_TO_ZERO_CLASS_ID
    if float(value) >= resolved_binning.resolved_high:
        return CLOSE_TO_ONE_CLASS_ID
    return IN_BETWEEN_CLASS_ID


def bin_token_label_values(values, resolved_binning: ResolvedTokenLabelBinning):
    return [int(bin_token_label_value(value, resolved_binning)) for value in values]


def _cast_label_column_to_int64(dataset, label_column):
    try:
        from datasets import Sequence, Value
    except ImportError:
        return dataset

    if not hasattr(dataset, "features") or label_column not in dataset.features:
        return dataset

    features = dataset.features.copy()
    label_feature = features[label_column]
    if getattr(label_feature, "feature", None) is not None:
        current_dtype = getattr(label_feature.feature, "dtype", None)
        if current_dtype == "int64":
            return dataset
        features[label_column] = Sequence(Value("int64"))
    else:
        current_dtype = getattr(label_feature, "dtype", None)
        if current_dtype == "int64":
            return dataset
        features[label_column] = Value("int64")
    return dataset.cast(features)


def apply_token_label_binning_to_dataset(
    dataset,
    resolved_binning: Optional[ResolvedTokenLabelBinning],
    label_column="labels",
):
    if dataset is None or resolved_binning is None:
        return dataset

    def _map_batch(batch):
        return {
            label_column: [
                bin_token_label_values(labels, resolved_binning)
                for labels in batch[label_column]
            ]
        }

    dataset = dataset.map(_map_batch, batched=True)
    return _cast_label_column_to_int64(dataset, label_column)
