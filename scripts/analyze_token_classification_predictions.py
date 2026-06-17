#!/usr/bin/env python3
import argparse
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROBABILITY_COLUMN_RE = re.compile(r"^probability_class_(\d+)$")


def find_class_probability_columns(columns):
    class_columns = []
    for column in columns:
        match = PROBABILITY_COLUMN_RE.match(column)
        if match:
            class_columns.append((int(match.group(1)), column))
    return sorted(class_columns)


def load_prediction_dataframe(prediction_path):
    df = pd.read_csv(prediction_path)
    required_columns = {"label"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {prediction_path}: {sorted(missing)}")

    class_columns = find_class_probability_columns(df.columns)
    if not class_columns:
        raise ValueError(f"No probability_class_<id> columns found in {prediction_path}")

    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df = df[df["label"] != -100]
    df["label"] = df["label"].astype(int)

    for _, column in class_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column for _, column in class_columns])

    if "predicted_class" in df.columns:
        df["predicted_class"] = pd.to_numeric(df["predicted_class"], errors="coerce").astype("Int64")
    elif "prediction" in df.columns:
        df["predicted_class"] = pd.to_numeric(df["prediction"], errors="coerce").astype("Int64")
    else:
        probability_columns = [column for _, column in class_columns]
        class_ids = np.array([class_id for class_id, _ in class_columns])
        df["predicted_class"] = class_ids[np.argmax(df[probability_columns].to_numpy(), axis=1)]

    df = df.dropna(subset=["predicted_class"])
    df["predicted_class"] = df["predicted_class"].astype(int)
    return df, class_columns


def safe_auc(y_true_binary, y_score):
    if len(np.unique(y_true_binary)) < 2:
        return np.nan
    return roc_auc_score(y_true_binary, y_score)


def safe_average_precision(y_true_binary, y_score):
    if y_true_binary.sum() == 0:
        return np.nan
    return average_precision_score(y_true_binary, y_score)


def safe_divide(numerator, denominator, default=np.nan):
    if denominator == 0:
        return default
    return numerator / denominator


def compute_per_class_metrics(df, class_columns):
    rows = []
    labels = df["label"].to_numpy()
    predictions = df["predicted_class"].to_numpy()
    total = len(df)

    for class_id, probability_column in class_columns:
        y_true = (labels == class_id).astype(int)
        y_pred = (predictions == class_id).astype(int)
        y_score = df[probability_column].to_numpy()

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        precision = safe_divide(tp, tp + fp, default=0.0)
        recall = safe_divide(tp, tp + fn, default=0.0)
        specificity = safe_divide(tn, tn + fp)
        negative_predictive_value = safe_divide(tn, tn + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall, default=0.0)

        rows.append(
            {
                "class_id": class_id,
                "support": int(tp + fn),
                "predicted_count": int(tp + fp),
                "non_class_count": int(tn + fp),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "auc_ovr": safe_auc(y_true, y_score),
                "average_precision_ovr": safe_average_precision(y_true, y_score),
                "accuracy_ovr": safe_divide(tp + tn, total),
                "balanced_accuracy_ovr": np.nanmean([recall, specificity]),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "negative_predictive_value": negative_predictive_value,
                "false_positive_rate": safe_divide(fp, fp + tn),
                "false_negative_rate": safe_divide(fn, fn + tp),
                "prevalence": safe_divide(tp + fn, total),
                "predicted_prevalence": safe_divide(tp + fp, total),
                "mean_probability_for_true_class": float(df.loc[labels == class_id, probability_column].mean())
                if y_true.sum() > 0
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_overall_metrics(df, class_columns):
    labels = df["label"].to_numpy()
    predictions = df["predicted_class"].to_numpy()
    classes = [class_id for class_id, _ in class_columns]
    probability_columns = [column for _, column in class_columns]

    overall = {
        "n_rows": int(len(df)),
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision_score(labels, predictions, labels=classes, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predictions, labels=classes, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, labels=classes, average="macro", zero_division=0),
        "weighted_precision": precision_score(labels, predictions, labels=classes, average="weighted", zero_division=0),
        "weighted_recall": recall_score(labels, predictions, labels=classes, average="weighted", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, labels=classes, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
    }

    present_classes = sorted(set(labels))
    if len(present_classes) > 1:
        try:
            overall["macro_auc_ovr"] = roc_auc_score(
                labels,
                df[probability_columns].to_numpy(),
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            overall["macro_auc_ovr"] = np.nan
    else:
        overall["macro_auc_ovr"] = np.nan

    return overall


def print_analysis(prediction_path, df, class_columns, per_class_df, overall):
    print(f"prediction_path: {prediction_path}")
    print(f"rows_used: {overall['n_rows']}")
    print()
    print("overall:")
    for key in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "mcc",
        "macro_auc_ovr",
    ]:
        print(f"  {key}: {overall[key]:.6f}" if pd.notna(overall[key]) else f"  {key}: nan")

    print()
    print("label counts:")
    print(df["label"].value_counts().sort_index().to_string())

    print()
    print("confusion matrix rows=true cols=pred:")
    classes = [class_id for class_id, _ in class_columns]
    confusion = confusion_matrix(df["label"], df["predicted_class"], labels=classes)
    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{class_id}" for class_id in classes],
        columns=[f"pred_{class_id}" for class_id in classes],
    )
    print(confusion_df.to_string())

    print()
    print("per-class metrics:")
    print(per_class_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze token-classification prediction CSV files.")
    parser.add_argument("prediction_path", help="Path to eval_predictions.csv.gitbackup")
    parser.add_argument("--output-csv", help="Optional path for per-class metrics CSV.")
    parser.add_argument("--overall-output-csv", help="Optional path for one-row overall metrics CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    df, class_columns = load_prediction_dataframe(args.prediction_path)
    if df.empty:
        raise ValueError(f"No usable prediction rows found in {args.prediction_path}")

    per_class_df = compute_per_class_metrics(df, class_columns)
    overall = compute_overall_metrics(df, class_columns)
    print_analysis(args.prediction_path, df, class_columns, per_class_df, overall)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        per_class_df.to_csv(args.output_csv, index=False)
        print(f"wrote per-class metrics to {args.output_csv}")

    if args.overall_output_csv:
        os.makedirs(os.path.dirname(args.overall_output_csv) or ".", exist_ok=True)
        pd.DataFrame([overall]).to_csv(args.overall_output_csv, index=False)
        print(f"wrote overall metrics to {args.overall_output_csv}")


if __name__ == "__main__":
    main()
