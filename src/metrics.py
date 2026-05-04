import csv
import json
from pathlib import Path

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[label_to_idx[str(true)], label_to_idx[str(pred)]] += 1
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    per_class = {}
    f1_values = []
    for idx, label in enumerate(labels):
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        support = int(matrix[idx, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "support": total,
    }


def top_k_predictions(probabilities: np.ndarray, classes: np.ndarray, k: int = 3) -> list[list[dict]]:
    rows = []
    for row in probabilities:
        top_indices = np.argsort(row)[::-1][:k]
        rows.append(
            [
                {"class": str(classes[idx]), "confidence": float(row[idx])}
                for idx in top_indices
            ]
        )
    return rows


def write_confusion_matrix_csv(path: str | Path, matrix: np.ndarray, labels: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])
    return path


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
