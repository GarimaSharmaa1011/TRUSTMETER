"""Evaluation helpers used by training and reporting."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support": int(len(y_true)),
    }


def compute_ece(y_true, y_pred, confidence, n_bins=10):
    """Top-label ECE: weighted accuracy-confidence gap across equal-width bins."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    correct = y_true == y_pred
    ece = 0.0
    for index in range(n_bins):
        in_bin = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == n_bins - 1
            else confidence < edges[index + 1]
        )
        if np.any(in_bin):
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def compute_reliability_diagram(y_true, y_pred, confidence, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result = []
    correct = np.asarray(y_true) == np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)
    for index in range(n_bins):
        in_bin = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == n_bins - 1
            else confidence < edges[index + 1]
        )
        result.append({
            "bin_center": float((edges[index] + edges[index + 1]) / 2),
            "mean_confidence": float(confidence[in_bin].mean()) if np.any(in_bin) else None,
            "accuracy": float(correct[in_bin].mean()) if np.any(in_bin) else None,
            "count": int(in_bin.sum()),
        })
    return result


def compute_confidence_histogram(confidence, n_bins=20):
    counts, edges = np.histogram(confidence, bins=n_bins, range=(0.0, 1.0))
    return {
        "bin_centers": [float((edges[i] + edges[i + 1]) / 2) for i in range(n_bins)],
        "counts": [int(count) for count in counts],
    }


def compute_coverage_metrics(y_true, probabilities, thresholds):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    results = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        n_accepted = int(accepted.sum())
        n_rejected = int(len(y_true) - n_accepted)
        if n_accepted:
            metrics = compute_classification_metrics(y_true[accepted], predictions[accepted])
            accuracy = metrics["accuracy"]
            risk = 1.0 - accuracy
        else:
            metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None}
            accuracy = None
            risk = None
        results.append({
            "threshold": float(threshold),
            "coverage": float(n_accepted / len(y_true)),
            "abstention_rate": float(n_rejected / len(y_true)),
            "accuracy": accuracy,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "risk": risk,
            "n_predicted": n_accepted,
            "n_abstained": n_rejected,
        })
    return results
