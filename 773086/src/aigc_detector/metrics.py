from __future__ import annotations

import numpy as np


def _auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for tied scores."""
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    ranks = np.empty(len(probabilities), dtype=np.float64)
    start = 0
    while start < len(probabilities):
        end = start + 1
        while end < len(probabilities) and sorted_probabilities[end] == sorted_probabilities[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    rank_sum = ranks[positives].sum()
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[order]
    sorted_probabilities = probabilities[order]
    true_positives = np.cumsum(sorted_labels == 1)
    # Evaluate precision/recall only after complete tied-score groups, matching the
    # non-interpolated AP definition used by common evaluation toolkits.
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_probabilities) != 0), len(labels) - 1]
    group_true_positives = true_positives[group_ends]
    precision = group_true_positives / (group_ends + 1)
    recall = group_true_positives / int((labels == 1).sum())
    recall_increase = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increase * precision))


def binary_metrics(labels: list[float], probabilities: list[float], threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    real_accuracy = float((y_pred[y_true == 0] == 0).mean()) if np.any(y_true == 0) else float("nan")
    fake_accuracy = float((y_pred[y_true == 1] == 1).mean()) if np.any(y_true == 1) else float("nan")
    result = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.nanmean([real_accuracy, fake_accuracy])),
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
    }
    if len(np.unique(y_true)) == 2:
        result["auroc"] = _auroc(y_true, y_prob)
        result["average_precision"] = _average_precision(y_true, y_prob)
    else:
        result["auroc"] = float("nan")
        result["average_precision"] = float("nan")
    return result
