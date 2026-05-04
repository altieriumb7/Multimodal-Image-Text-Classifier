import argparse
from pathlib import Path

import numpy as np

from src.config import DEFAULT_FEATURES_PATH, DEFAULT_MODEL_PATH
from src.modeling import save_model_bundle, train_single_classifier, view_features


def train_classifier_bundle(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    output_path: str | Path = DEFAULT_MODEL_PATH,
    seed: int = 42,
    prefer_sklearn: bool = True,
) -> Path:
    features = np.load(features_path, allow_pickle=True)
    splits = features["splits"]
    y = features["labels"]
    train_mask = splits == "train"

    image_model, image_backend = train_single_classifier(
        view_features(features, "image")[train_mask],
        y[train_mask],
        seed=seed,
        prefer_sklearn=prefer_sklearn,
    )
    text_model, text_backend = train_single_classifier(
        view_features(features, "text")[train_mask],
        y[train_mask],
        seed=seed,
        prefer_sklearn=prefer_sklearn,
    )
    multimodal_model, multimodal_backend = train_single_classifier(
        view_features(features, "multimodal")[train_mask],
        y[train_mask],
        seed=seed,
        prefer_sklearn=prefer_sklearn,
    )

    metadata = {
        "features_path": str(Path(features_path).resolve()),
        "feature_backend": str(features["feature_backend"].item()),
        "feature_model": str(features["feature_model"].item()),
        "labels": sorted(set(y.tolist())),
        "is_synthetic_dataset": bool(features["is_synthetic"].all()),
        "classifiers": {
            "image": image_backend,
            "text": text_backend,
            "multimodal": multimodal_backend,
        },
    }
    return save_model_bundle(output_path, image_model, text_model, multimodal_model, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train image-only, text-only, and multimodal classifiers.")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES_PATH), help="Input feature bundle.")
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH), help="Output model bundle.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument("--no-sklearn", action="store_true", help="Use the numpy centroid fallback classifier.")
    args = parser.parse_args()

    output = train_classifier_bundle(
        features_path=args.features,
        output_path=args.output,
        seed=args.seed,
        prefer_sklearn=not args.no_sklearn,
    )
    print(f"Model bundle written: {output}")


if __name__ == "__main__":
    main()
