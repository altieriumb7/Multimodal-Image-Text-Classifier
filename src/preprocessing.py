import re
from pathlib import Path

import numpy as np
from PIL import Image


TEXT_PATTERN = re.compile(r"[^a-z0-9%+\-\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    text = (text or "").lower().strip()
    text = TEXT_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def prepare_text(title: str | None, description: str | None = None) -> str:
    joined = " ".join(part for part in [title or "", description or ""] if part)
    return clean_text(joined)


def load_image(path: str | Path, image_size: tuple[int, int] = (224, 224)) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize(image_size)


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
