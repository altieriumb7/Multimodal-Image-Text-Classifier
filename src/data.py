import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.config import DEMO_METADATA_PATH, RANDOM_SEED
from src.make_demo_data import ensure_demo_dataset
from src.preprocessing import prepare_text


@dataclass(frozen=True)
class ListingExample:
    id: str
    image_path: Path
    title: str
    description: str
    label: str
    is_synthetic: bool = False

    @property
    def text(self) -> str:
        return prepare_text(self.title, self.description)


REQUIRED_COLUMNS = {"id", "image_path", "title", "description", "label"}


def _bool_from_csv(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_dataset(
    metadata_path: str | Path | None = None,
    auto_generate_demo: bool = True,
) -> list[ListingExample]:
    metadata_path = Path(metadata_path or DEMO_METADATA_PATH)
    if auto_generate_demo and metadata_path == DEMO_METADATA_PATH and not metadata_path.exists():
        ensure_demo_dataset(metadata_path.parent)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset metadata is missing columns: {sorted(missing)}")

        examples = []
        for row in reader:
            raw_image_path = Path(row["image_path"])
            image_path = raw_image_path if raw_image_path.is_absolute() else metadata_path.parent / raw_image_path
            examples.append(
                ListingExample(
                    id=row["id"],
                    image_path=image_path,
                    title=row["title"],
                    description=row["description"],
                    label=row["label"],
                    is_synthetic=_bool_from_csv(row.get("is_synthetic")),
                )
            )

    if not examples:
        raise ValueError(f"No rows found in dataset metadata: {metadata_path}")
    return examples


def labels_from_examples(examples: Iterable[ListingExample]) -> list[str]:
    return sorted({example.label for example in examples})


def stratified_split(
    examples: list[ListingExample],
    seed: int = RANDOM_SEED,
    train_size: float = 0.6,
    val_size: float = 0.2,
) -> dict[str, list[ListingExample]]:
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be less than 1.")

    rng = random.Random(seed)
    by_label: dict[str, list[ListingExample]] = {}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)

    splits = {"train": [], "val": [], "test": []}
    for label_examples in by_label.values():
        group = list(label_examples)
        rng.shuffle(group)
        n = len(group)
        if n < 3:
            raise ValueError("Each class needs at least 3 examples for train/val/test splitting.")
        n_train = max(1, int(round(n * train_size)))
        n_val = max(1, int(round(n * val_size)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])

    for split_examples in splits.values():
        rng.shuffle(split_examples)
    return splits


def flatten_splits(splits: dict[str, list[ListingExample]]) -> tuple[list[ListingExample], list[str]]:
    examples: list[ListingExample] = []
    split_names: list[str] = []
    for split_name in ("train", "val", "test"):
        for example in splits[split_name]:
            examples.append(example)
            split_names.append(split_name)
    return examples, split_names
