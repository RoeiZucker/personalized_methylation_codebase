from evaluate import load
import numpy as np
from scipy.stats import pearsonr

roc_auc_score = load("roc_auc")
f1_score = load("f1")
precision_score = load("precision")
recall_score = load("recall")
accuracy_score = load("accuracy")
# average_precision_score = load("average_precision")
mcc_score = load("matthews_correlation")
pearsonr_eval = load("pearsonr")
mse = load("mse")
mae = load("mae")


def _sigmoid_np(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))

def apply_results_for_class(class_id, flat_labels, flat_preds, flat_probs, results):
    y_true = (flat_labels == class_id).astype(np.int32)
    y_pred = (flat_preds == class_id).astype(np.int32)
    y_scores = np.array(flat_probs[class_id], dtype=np.float32)
    results[f"roc_auc_class_{class_id}"] = roc_auc_score.compute(
        prediction_scores=y_scores,
        references=y_true
    )["roc_auc"]
    results[f"f1_class_{class_id}"] = f1_score.compute(
        predictions=y_pred,
        references=y_true
    )["f1"]
    results[f"precision_class_{class_id}"] = precision_score.compute(
        predictions=y_pred,
        references=y_true
    )["precision"]
    results[f"recall_class_{class_id}"] = recall_score.compute(
        predictions=y_pred,
        references=y_true
    )["recall"]

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    # Convert logits to probabilities
    probs = np.exp(preds) / np.exp(preds).sum(axis=-1, keepdims=True)
    labels = np.array(p.label_ids)
    num_classes = probs.shape[-1]

    flat_labels, flat_preds, flat_probs = flatten_and_mask_results(labels, num_classes, probs)
    
    results = {}
    # Global metrics
    apply_global_results(flat_labels, flat_preds, results)

    # Per-class metrics
    for class_id in range(num_classes):
        apply_results_for_class(class_id, flat_labels, flat_preds, flat_probs, results)

    return results

def compute_metrics_regression(eval_pred):
    """
    eval_pred is a transformers.EvalPrediction:
        predictions – ndarray (N, seq_len, 1) or (N, seq_len)
        label_ids   – ndarray (N, seq_len)
    We  ➜  squeeze the last dim, apply sigmoid, then apply the –100 mask
    and report MSE, MAE, and Pearson-r over the remaining tokens.
    """
    preds, labels = eval_pred
    preds = _sigmoid_np(np.asarray(preds).squeeze(-1))
    labels = np.asarray(labels)

    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    if preds.size == 0:
        return {"mse": np.nan, "mae": np.nan, "pearson_r": np.nan}

    mse = np.mean((preds - labels) ** 2)
    mae = np.mean(np.abs(preds - labels))
    r, _ = pearsonr(preds, labels) if len(labels) > 1 else (np.nan, np.nan)

    return {"mse": mse, "mae": mae, "pearson_r": r}

# TODO: change name to something more descriptive

def flatten_and_mask_results(labels, num_classes, probs):
    # Flatten with mask
    flat_labels = []
    flat_probs = [[] for _ in range(num_classes)]
    flat_preds = []
    for prob_row, label_row in zip(probs, labels):
        for prob, label in zip(prob_row, label_row):
            if label != -100:
                flat_labels.append(label)
                flat_preds.append(np.argmax(prob))
                for i in range(num_classes):
                    flat_probs[i].append(prob[i])
    flat_labels = np.array(flat_labels)
    flat_preds = np.array(flat_preds)
    return flat_labels, flat_preds, flat_probs

# TODO think how to use generic score functions



def apply_global_results(flat_labels, flat_preds, results):
    results["accuracy"] = accuracy_score.compute(
        predictions=flat_preds,
        references=flat_labels
    )["accuracy"]
    results["mcc"] = mcc_score.compute(
        predictions=flat_preds,
        references=flat_labels
    )["matthews_correlation"]
