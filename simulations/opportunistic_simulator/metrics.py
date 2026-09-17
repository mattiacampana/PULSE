"""Dependency-free classification metrics used for per-node reporting."""

from __future__ import annotations

import torch


def classification_metrics(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict[str, float]:
    accuracy = float((predictions == labels).float().mean())
    f1_scores = []
    for label in range(num_classes):
        tp = ((predictions == label) & (labels == label)).sum().item()
        fp = ((predictions == label) & (labels != label)).sum().item()
        fn = ((predictions != label) & (labels == label)).sum().item()
        denominator = 2 * tp + fp + fn
        f1_scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return {"accuracy": accuracy, "macro_f1": sum(f1_scores) / num_classes}
