from pathlib import Path

from src.config import (
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_FEATURES_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REPORT_PATH,
    DEMO_METADATA_PATH,
)
from src.evaluate_classifier import evaluate_classifier_bundle
from src.extract_features import build_feature_bundle
from src.make_demo_data import ensure_demo_dataset
from src.train_classifier import train_classifier_bundle


def ensure_demo_artifacts(force: bool = False) -> dict[str, Path]:
    metadata = ensure_demo_dataset(DEMO_METADATA_PATH.parent, force=force)
    if force or not DEFAULT_FEATURES_PATH.exists():
        build_feature_bundle(metadata_path=metadata, output_path=DEFAULT_FEATURES_PATH, backend="demo")
    if force or not DEFAULT_MODEL_PATH.exists():
        train_classifier_bundle(DEFAULT_FEATURES_PATH, DEFAULT_MODEL_PATH)
    if force or not DEFAULT_METRICS_PATH.exists() or not DEFAULT_CONFUSION_MATRIX_PATH.exists():
        evaluate_classifier_bundle(
            features_path=DEFAULT_FEATURES_PATH,
            model_path=DEFAULT_MODEL_PATH,
            metrics_path=DEFAULT_METRICS_PATH,
            confusion_matrix_path=DEFAULT_CONFUSION_MATRIX_PATH,
            report_path=DEFAULT_REPORT_PATH,
        )
    return {
        "metadata": metadata,
        "features": DEFAULT_FEATURES_PATH,
        "model": DEFAULT_MODEL_PATH,
        "metrics": DEFAULT_METRICS_PATH,
        "confusion_matrix": DEFAULT_CONFUSION_MATRIX_PATH,
        "report": DEFAULT_REPORT_PATH,
    }
