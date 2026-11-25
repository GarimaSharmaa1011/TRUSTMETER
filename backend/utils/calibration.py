"""Platt-scaling calibration helpers for a pre-trained classifier."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class SigmoidCalibratedClassifier:
    """Keep the training estimator fixed while fitting a logistic map on validation scores."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X_validation, y_validation):
        probabilities = self.estimator.predict_proba(X_validation)[:, 1]
        logits = self._logits(probabilities).reshape(-1, 1)
        self.calibrator_ = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.calibrator_.fit(logits, y_validation)
        self.classes_ = self.calibrator_.classes_
        return self

    def predict_proba(self, X):
        probabilities = self.estimator.predict_proba(X)[:, 1]
        return self.calibrator_.predict_proba(self._logits(probabilities).reshape(-1, 1))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @staticmethod
    def _logits(probabilities):
        clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        return np.log(clipped / (1.0 - clipped))


def calibrate_model(base_estimator, X_val, y_val, method="sigmoid"):
    """Fit a sigmoid calibration map using only the held-out validation split."""
    if method != "sigmoid":
        raise ValueError("Only sigmoid (Platt scaling) calibration is supported.")
    return SigmoidCalibratedClassifier(base_estimator).fit(X_val, y_val)


def get_confidence(probabilities):
    """Return top-class confidence for a single probability vector."""
    return float(max(probabilities))


def apply_threshold(probabilities, threshold=0.7):
    """Return (prediction, confidence, abstain) for a probability vector."""
    confidence = get_confidence(probabilities)
    if confidence < threshold:
        return None, confidence, True
    return int(max(range(len(probabilities)), key=probabilities.__getitem__)), confidence, False
