import argparse
from pathlib import Path

import numpy as np

from src.config import (
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_FEATURES_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORT_PATH,
)
from src.metrics import classification_metrics, confusion_matrix, write_confusion_matrix_csv, write_json
from src.modeling import get_classes, load_model_bundle, predict_proba, view_features


def _evaluate_view(model, X: np.ndarray, y_true: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    classes = get_classes(model)
    probabilities = predict_proba(model, X)
    predictions = classes[np.argmax(probabilities, axis=1)]
    metrics = classification_metrics(y_true, predictions, labels=classes.astype(str).tolist())
    return metrics, predictions, classes


def _markdown_report(payload: dict, split: str, is_synthetic: bool) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"Split: `{split}`",
        f"Dataset type: `{'synthetic demo' if is_synthetic else 'real/custom'}`",
        "",
        "These metrics are computed from the local feature/model artifacts. Demo-data results are only a functional smoke test.",
        "",
        "| Model | Accuracy | Macro F1 |",
        "| --- | ---: | ---: |",
    ]
    for name in ("image", "text", "multimodal"):
        metrics = payload[name]
        lines.append(f"| {name} | {metrics['accuracy']:.3f} | {metrics['macro_f1']:.3f} |")
    lines.extend(["", "## Per-Class Multimodal Metrics", "", "| Class | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |"])
    for label, values in payload["multimodal"]["per_class"].items():
        lines.append(
            f"| {label} | {values['precision']:.3f} | {values['recall']:.3f} | {values['f1']:.3f} | {values['support']} |"
        )
    lines.append("")
    return "\n".join(lines)


def evaluate_classifier_bundle(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    confusion_matrix_path: str | Path = DEFAULT_CONFUSION_MATRIX_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    split: str = "test",
) -> dict:
    features = np.load(features_path, allow_pickle=True)
    bundle = load_model_bundle(model_path)
    mask = features["splits"] == split
    if not mask.any():
        raise ValueError(f"No examples found for split: {split}")
    y_true = features["labels"][mask]

    payload = {}
    predictions_by_view = {}
    classes_by_view = {}
    for view, model_key in [
        ("image", "image_model"),
        ("text", "text_model"),
        ("multimodal", "multimodal_model"),
    ]:
        metrics, predictions, classes = _evaluate_view(
            bundle[model_key],
            view_features(features, view)[mask],
            y_true,
        )
        payload[view] = metrics
        predictions_by_view[view] = predictions
        classes_by_view[view] = classes

    payload["metadata"] = {
        **bundle.get("metadata", {}),
        "evaluated_split": split,
        "n_examples": int(mask.sum()),
    }
    write_json(metrics_path, payload)

    labels = classes_by_view["multimodal"].astype(str).tolist()
    matrix = confusion_matrix(y_true, predictions_by_view["multimodal"], labels)
    write_confusion_matrix_csv(confusion_matrix_path, matrix, labels)

    report_text = _markdown_report(
        payload,
        split=split,
        is_synthetic=bool(features["is_synthetic"].all()),
    )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(report_text, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classifier baselines and multimodal model.")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH), help="Input feature bundle.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Input model bundle.")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH), help="Output metrics JSON.")
    parser.add_argument("--confusion-matrix", default=str(DEFAULT_CONFUSION_MATRIX_PATH), help="Output multimodal confusion matrix CSV.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Output Markdown report.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to evaluate.")
    args = parser.parse_args()

    payload = evaluate_classifier_bundle(
        features_path=args.features,
        model_path=args.model,
        metrics_path=args.metrics,
        confusion_matrix_path=args.confusion_matrix,
        report_path=args.report,
        split=args.split,
    )
    print(f"Evaluation written: {args.metrics}")
    print(f"Multimodal accuracy: {payload['multimodal']['accuracy']:.3f}")
    print(f"Multimodal macro F1: {payload['multimodal']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
