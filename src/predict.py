import argparse
import json
from pathlib import Path

import numpy as np

from src.config import DEFAULT_CLIP_MODEL, DEFAULT_MODEL_PATH
from src.features import get_feature_extractor
from src.metrics import top_k_predictions
from src.modeling import get_classes, load_model_bundle, predict_proba
from src.preprocessing import prepare_text


def _view_prediction(model, X: np.ndarray, top_k: int = 3) -> dict:
    classes = get_classes(model)
    probabilities = predict_proba(model, X)
    top = top_k_predictions(probabilities, classes, k=top_k)[0]
    return {
        "predicted_class": top[0]["class"],
        "confidence": top[0]["confidence"],
        "top_k": top,
    }


def predict_listing(
    image_path: str | Path,
    text: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    backend: str | None = None,
    clip_model: str = DEFAULT_CLIP_MODEL,
    allow_download: bool = False,
    top_k: int = 3,
) -> dict:
    bundle = load_model_bundle(model_path)
    metadata = bundle.get("metadata", {})
    feature_backend = backend or metadata.get("feature_backend", "auto")
    if feature_backend not in {"clip", "demo", "auto"}:
        feature_backend = "auto"

    extractor = get_feature_extractor(
        backend=feature_backend,
        model_name=metadata.get("feature_model", clip_model),
        allow_download=allow_download,
    )
    image_embeddings = extractor.encode_image_paths([image_path])
    text_embeddings = extractor.encode_texts([prepare_text(text)])
    multimodal_embeddings = np.hstack([image_embeddings, text_embeddings])

    image_result = _view_prediction(bundle["image_model"], image_embeddings, top_k=top_k)
    text_result = _view_prediction(bundle["text_model"], text_embeddings, top_k=top_k)
    multimodal_result = _view_prediction(bundle["multimodal_model"], multimodal_embeddings, top_k=top_k)

    return {
        "predicted_class": multimodal_result["predicted_class"],
        "confidence": multimodal_result["confidence"],
        "top_3": multimodal_result["top_k"],
        "comparison": {
            "image_only": image_result,
            "text_only": text_result,
            "multimodal": multimodal_result,
        },
        "feature_backend_used": getattr(extractor, "backend_name", feature_backend),
        "model_metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a product class from an image and product text.")
    parser.add_argument("--image", required=True, help="Path to product image.")
    parser.add_argument("--text", required=True, help="Product title or description.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to model bundle.")
    parser.add_argument("--backend", choices=["auto", "clip", "demo"], default=None, help="Override feature backend.")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="CLIP model name when using CLIP backend.")
    parser.add_argument("--allow-download", action="store_true", help="Allow CLIP model download.")
    args = parser.parse_args()

    result = predict_listing(
        image_path=args.image,
        text=args.text,
        model_path=args.model,
        backend=args.backend,
        clip_model=args.clip_model,
        allow_download=args.allow_download,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
