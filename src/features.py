import hashlib
from pathlib import Path

import numpy as np

from src.config import DEFAULT_CLIP_MODEL
from src.data import ListingExample
from src.preprocessing import image_to_array, load_image


def _l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, eps)


def _projection_matrix(input_dim: int, output_dim: int, salt: str) -> np.ndarray:
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False) % (2**32)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=0.0, scale=1.0 / np.sqrt(input_dim), size=(input_dim, output_dim))
    return matrix.astype(np.float32)


class DemoFeatureExtractor:
    """Deterministic offline feature extractor used when CLIP weights are unavailable."""

    backend_name = "demo"
    model_name = "deterministic-demo-embedder"

    def __init__(self, embedding_dim: int = 64, image_size: tuple[int, int] = (224, 224)):
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self._image_projection: np.ndarray | None = None
        self._text_projection: np.ndarray | None = None

    def encode_image_paths(self, image_paths: list[str | Path]) -> np.ndarray:
        raw_features = [self._image_raw_features(path) for path in image_paths]
        raw = np.vstack(raw_features).astype(np.float32)
        if self._image_projection is None:
            self._image_projection = _projection_matrix(raw.shape[1], self.embedding_dim, "demo-image")
        return _l2_normalize(raw @ self._image_projection)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        raw = np.vstack([self._text_raw_features(text) for text in texts]).astype(np.float32)
        if self._text_projection is None:
            self._text_projection = _projection_matrix(raw.shape[1], self.embedding_dim, "demo-text")
        return _l2_normalize(raw @ self._text_projection)

    def _image_raw_features(self, image_path: str | Path) -> np.ndarray:
        image = load_image(image_path, self.image_size)
        arr = image_to_array(image)
        channels = [arr[:, :, idx].ravel() for idx in range(3)]
        histograms = [np.histogram(channel, bins=16, range=(0.0, 1.0), density=True)[0] for channel in channels]
        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))
        q25 = np.quantile(arr, 0.25, axis=(0, 1))
        q75 = np.quantile(arr, 0.75, axis=(0, 1))
        edge_proxy = np.array(
            [
                np.abs(np.diff(arr, axis=0)).mean(),
                np.abs(np.diff(arr, axis=1)).mean(),
            ],
            dtype=np.float32,
        )
        return np.concatenate(histograms + [means, stds, q25, q75, edge_proxy]).astype(np.float32)

    def _text_raw_features(self, text: str) -> np.ndarray:
        buckets = np.zeros(256, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            for gram in _token_ngrams(token):
                idx = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % len(buckets)
                buckets[idx] += 1.0
        length_features = np.array(
            [
                len(tokens),
                sum(len(token) for token in tokens),
                len(set(tokens)),
                float(any(char.isdigit() for char in text)),
            ],
            dtype=np.float32,
        )
        if buckets.sum() > 0:
            buckets = buckets / buckets.sum()
        return np.concatenate([buckets, length_features]).astype(np.float32)


def _token_ngrams(token: str) -> list[str]:
    grams = [token]
    if len(token) >= 3:
        grams.extend(token[idx : idx + 3] for idx in range(len(token) - 2))
    if len(token) >= 4:
        grams.extend(token[idx : idx + 4] for idx in range(len(token) - 3))
    return grams


class CLIPFeatureExtractor:
    backend_name = "clip"

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: str | None = None,
        local_files_only: bool = True,
    ):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers to use the CLIP backend.") from exc

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.to(self.device)
        self.model.eval()

    def encode_image_paths(self, image_paths: list[str | Path]) -> np.ndarray:
        images = [load_image(path) for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
        features = features.detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(features)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            features = self.model.get_text_features(**inputs)
        features = features.detach().cpu().numpy().astype(np.float32)
        return _l2_normalize(features)


def get_feature_extractor(
    backend: str = "auto",
    model_name: str = DEFAULT_CLIP_MODEL,
    allow_download: bool = False,
):
    backend = backend.lower()
    if backend == "demo":
        return DemoFeatureExtractor()
    if backend not in {"auto", "clip"}:
        raise ValueError("backend must be one of: auto, clip, demo")

    try:
        return CLIPFeatureExtractor(
            model_name=model_name,
            local_files_only=not allow_download,
        )
    except Exception:
        if backend == "clip":
            raise
        return DemoFeatureExtractor()


def extract_feature_arrays(
    examples: list[ListingExample],
    extractor,
) -> tuple[np.ndarray, np.ndarray]:
    image_paths = [example.image_path for example in examples]
    texts = [example.text for example in examples]
    image_embeddings = extractor.encode_image_paths(image_paths)
    text_embeddings = extractor.encode_texts(texts)
    if image_embeddings.shape[0] != len(examples) or text_embeddings.shape[0] != len(examples):
        raise ValueError("Feature extractor returned a row count that does not match the dataset.")
    return image_embeddings, text_embeddings
