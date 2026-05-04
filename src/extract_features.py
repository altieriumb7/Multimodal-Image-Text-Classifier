import argparse
import json
from pathlib import Path

import numpy as np

from src.config import DEFAULT_CLIP_MODEL, DEFAULT_FEATURES_PATH, DEMO_METADATA_PATH, MODELS_DIR
from src.data import flatten_splits, load_dataset, stratified_split
from src.features import extract_feature_arrays, get_feature_extractor


def build_feature_bundle(
    metadata_path: str | Path = DEMO_METADATA_PATH,
    output_path: str | Path = DEFAULT_FEATURES_PATH,
    backend: str = "auto",
    clip_model: str = DEFAULT_CLIP_MODEL,
    allow_download: bool = False,
    seed: int = 42,
) -> Path:
    examples = load_dataset(metadata_path)
    splits = stratified_split(examples, seed=seed)
    ordered_examples, split_names = flatten_splits(splits)
    extractor = get_feature_extractor(backend=backend, model_name=clip_model, allow_download=allow_download)
    image_embeddings, text_embeddings = extract_feature_arrays(ordered_examples, extractor)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.array([example.id for example in ordered_examples], dtype=object),
        image_paths=np.array([str(example.image_path) for example in ordered_examples], dtype=object),
        texts=np.array([example.text for example in ordered_examples], dtype=object),
        labels=np.array([example.label for example in ordered_examples], dtype=object),
        splits=np.array(split_names, dtype=object),
        image_embeddings=image_embeddings.astype(np.float32),
        text_embeddings=text_embeddings.astype(np.float32),
        feature_backend=np.array(getattr(extractor, "backend_name", backend), dtype=object),
        feature_model=np.array(getattr(extractor, "model_name", clip_model), dtype=object),
        is_synthetic=np.array([example.is_synthetic for example in ordered_examples], dtype=bool),
    )

    metadata = {
        "feature_backend": getattr(extractor, "backend_name", backend),
        "feature_model": getattr(extractor, "model_name", clip_model),
        "dataset_rows": len(ordered_examples),
        "labels": sorted({example.label for example in ordered_examples}),
        "splits": {split: len(rows) for split, rows in splits.items()},
        "is_synthetic_dataset": all(example.is_synthetic for example in ordered_examples),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract image and text embeddings for product listings.")
    parser.add_argument("--metadata", default=str(DEMO_METADATA_PATH), help="CSV metadata path.")
    parser.add_argument("--output", default=str(DEFAULT_FEATURES_PATH), help="Output .npz feature bundle.")
    parser.add_argument("--backend", default="auto", choices=["auto", "clip", "demo"], help="Feature backend.")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="Hugging Face CLIP model name.")
    parser.add_argument("--allow-download", action="store_true", help="Allow transformers to download CLIP weights.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output = build_feature_bundle(
        metadata_path=args.metadata,
        output_path=args.output,
        backend=args.backend,
        clip_model=args.clip_model,
        allow_download=args.allow_download,
        seed=args.seed,
    )
    print(f"Feature bundle written: {output}")


if __name__ == "__main__":
    main()
