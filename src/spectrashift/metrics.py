from __future__ import annotations

import numpy as np


def _validate_binary_inputs(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have identical shapes")
    if labels.ndim < 2:
        raise ValueError("inputs must include a class dimension")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("labels must be binary")
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return labels, probabilities


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    cumulative = np.cumsum(ranked)
    precision = cumulative / (np.arange(ranked.size) + 1)
    return float(np.sum(precision * ranked) / positives)


def multilabel_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    labels, probabilities = _validate_binary_inputs(labels, probabilities)
    predictions = probabilities >= threshold
    class_count = labels.shape[-1]
    per_class: list[dict[str, float | int]] = []
    for class_index in range(class_count):
        truth = labels[..., class_index].astype(bool)
        predicted = predictions[..., class_index]
        tp = int(np.sum(truth & predicted))
        fp = int(np.sum(~truth & predicted))
        fn = int(np.sum(truth & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        union = tp + fp + fn
        per_class.append(
            {
                "class_index": class_index,
                "support": int(np.sum(truth)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "iou": float(tp / union if union else 0.0),
                "average_precision": average_precision(truth, probabilities[..., class_index]),
            }
        )
    return {
        "threshold": threshold,
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "mean_iou": float(np.mean([row["iou"] for row in per_class])),
        "macro_average_precision": float(
            np.mean([row["average_precision"] for row in per_class])
        ),
        "per_class": per_class,
    }


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    labels, probabilities = _validate_binary_inputs(labels, probabilities)
    if bins < 2:
        raise ValueError("bins must be at least two")
    truth = labels.reshape(-1)
    confidence = probabilities.reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(mask):
            error += float(np.mean(mask)) * abs(float(np.mean(confidence[mask])) - float(np.mean(truth[mask])))
    return float(error)


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels, probabilities = _validate_binary_inputs(labels, probabilities)
    return float(np.mean((probabilities - labels) ** 2))


def selective_risk_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    points: int = 20,
) -> list[dict[str, float]]:
    labels, probabilities = _validate_binary_inputs(labels, probabilities)
    truth = labels.reshape(-1).astype(bool)
    probabilities = probabilities.reshape(-1)
    predictions = probabilities >= threshold
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    order = np.argsort(-confidence, kind="stable")
    correct = (predictions[order] == truth[order]).astype(np.float64)
    cumulative_accuracy = np.cumsum(correct) / (np.arange(correct.size) + 1)
    counts = np.unique(np.linspace(1, correct.size, min(points, correct.size), dtype=int))
    return [
        {
            "coverage": float(count / correct.size),
            "risk": float(1.0 - cumulative_accuracy[count - 1]),
            "confidence_floor": float(confidence[order[count - 1]]),
        }
        for count in counts
    ]

