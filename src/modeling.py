from pathlib import Path

import joblib
import numpy as np


class CentroidClassifier:
    """Small fallback classifier with calibrated-looking distance probabilities."""

    def __init__(self, temperature: float = 0.35):
        self.temperature = temperature
        self.classes_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.array(sorted(set(y.tolist())), dtype=object)
        self.centroids_ = np.vstack([X[y == label].mean(axis=0) for label in self.classes_])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.centroids_ is None:
            raise RuntimeError("Classifier has not been fit.")
        distances = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        logits = -distances / max(self.temperature, 1e-6)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


def train_single_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    prefer_sklearn: bool = True,
):
    if prefer_sklearn:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
            classifier.fit(X_train, y_train)
            return classifier, "sklearn-logistic-regression"
        except ImportError:
            pass

    classifier = CentroidClassifier()
    classifier.fit(X_train, y_train)
    return classifier, "numpy-centroid"


def get_classes(model) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.array(model.classes_, dtype=object)
    if hasattr(model, "named_steps"):
        for step in reversed(model.steps):
            estimator = step[1]
            if hasattr(estimator, "classes_"):
                return np.array(estimator.classes_, dtype=object)
    raise RuntimeError("Could not infer classifier classes.")


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    raise RuntimeError("Classifier does not expose predict_proba.")


def save_model_bundle(
    path: str | Path,
    image_model,
    text_model,
    multimodal_model,
    metadata: dict,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "image_model": image_model,
            "text_model": text_model,
            "multimodal_model": multimodal_model,
            "metadata": metadata,
        },
        path,
    )
    return path


def load_model_bundle(path: str | Path) -> dict:
    return joblib.load(path)


def view_features(features: dict | np.lib.npyio.NpzFile, view: str) -> np.ndarray:
    image = features["image_embeddings"]
    text = features["text_embeddings"]
    if view == "image":
        return image
    if view == "text":
        return text
    if view == "multimodal":
        return np.hstack([image, text])
    raise ValueError("view must be image, text, or multimodal")
