import numpy as np
from scipy.stats import pearsonr

from .metrics_utils import (
    apply_global_results,
    apply_results_for_class,
)


def compute_sequence_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    probs = np.exp(preds) / np.exp(preds).sum(axis=-1, keepdims=True)
    labels = np.array(p.label_ids).reshape(-1)
    flat_preds = np.argmax(probs, axis=-1).reshape(-1)
    num_classes = probs.shape[-1]
    flat_probs = [np.array(probs[:, class_id], dtype=np.float32) for class_id in range(num_classes)]

    results = {}
    apply_global_results(labels, flat_preds, results)
    for class_id in range(num_classes):
        apply_results_for_class(class_id, labels, flat_preds, flat_probs, results)
    return results


def compute_sequence_metrics_regression(eval_pred):
    preds, labels = eval_pred
    preds = np.array(preds).squeeze(-1)
    labels = np.array(labels).squeeze(-1)

    mse = np.mean((preds - labels) ** 2)
    mae = np.mean(np.abs(preds - labels))
    r, _ = pearsonr(preds, labels) if np.size(labels) > 1 else (np.nan, np.nan)
    return {"mse": mse, "mae": mae, "pearson_r": r}
