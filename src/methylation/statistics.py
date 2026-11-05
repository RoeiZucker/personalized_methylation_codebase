from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def confusion(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.shape != predicted.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("Expected nonempty, aligned one-dimensional class arrays")
    if not np.isin(truth, [0, 1, 2]).all() or not np.isin(predicted, [0, 1, 2]).all():
        raise ValueError("Classes must be 0, 1 or 2")
    return np.bincount(3 * truth.astype(int) + predicted.astype(int), minlength=9).reshape(3, 3)


def f1_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=float)
    support = matrix.sum(axis=1)
    denominator = support + matrix.sum(axis=0)
    per_class = np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(3), where=denominator > 0)
    if support.sum() == 0:
        raise ValueError("Cannot score an empty confusion matrix")
    return float(np.average(per_class, weights=support)), per_class


def weighted_f1(truth, predicted):
    return f1_from_confusion(confusion(truth, predicted))[0]


def random_f1(truth):
    support = np.bincount(np.asarray(truth, dtype=int), minlength=3)
    if support.sum() == 0:
        raise ValueError("Cannot score empty labels")
    frequencies = support / support.sum()
    return float(frequencies @ frequencies), frequencies


def bootstrap(values, seed=20261014, resamples=20000):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite donor scores")
    indices = np.random.default_rng(seed).integers(0, len(values), (resamples, len(values)))
    lo, hi = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi), "donors": len(values)}


def sign_test(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite paired differences")
    positive, negative = int((values > 0).sum()), int((values < 0).sum())
    return {"positive": positive, "negative": negative, "ties": int((values == 0).sum()),
            "p_value": float(binomtest(positive, positive + negative).pvalue) if positive + negative else 1.0}


def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Expected p-values between zero and one")
    if len(values) == 0:
        return values.copy()
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty(len(values))
    result[order] = np.minimum(adjusted, 1)
    return result
