import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.config import DEMO_DATA_DIR


DEMO_ROWS = [
    ("electronics_001", "electronics", "Wireless noise canceling headphones", "Bluetooth over-ear headphones with padded ear cups."),
    ("electronics_002", "electronics", "USB-C fast charger", "Compact wall adapter for phones, tablets, and laptops."),
    ("electronics_003", "electronics", "Portable smart speaker", "Voice assistant speaker with room-filling audio."),
    ("electronics_004", "electronics", "Fitness smartwatch", "Touchscreen watch with heart-rate and step tracking."),
    ("electronics_005", "electronics", "Mechanical keyboard", "Backlit keyboard with tactile switches for desk setups."),
    ("electronics_006", "electronics", "Action camera bundle", "Water resistant camera kit with mounts and spare battery."),
    ("apparel_001", "apparel", "Cotton running shirt", "Lightweight crew neck athletic shirt in breathable fabric."),
    ("apparel_002", "apparel", "Denim jacket", "Classic blue jacket with button front and chest pockets."),
    ("apparel_003", "apparel", "Leather wallet", "Slim bifold wallet with card slots and cash pocket."),
    ("apparel_004", "apparel", "Trail hiking shoes", "Durable outdoor shoes with grippy rubber soles."),
    ("apparel_005", "apparel", "Wool winter scarf", "Soft knit scarf for cold weather layering."),
    ("apparel_006", "apparel", "Canvas tote bag", "Reusable carry bag with reinforced handles."),
    ("grocery_001", "grocery", "Organic coffee beans", "Medium roast whole beans with cocoa and citrus notes."),
    ("grocery_002", "grocery", "Granola cereal box", "Crunchy oat cereal with almonds and honey clusters."),
    ("grocery_003", "grocery", "Extra virgin olive oil", "Cold pressed olive oil in a glass bottle."),
    ("grocery_004", "grocery", "Herbal tea sampler", "Assorted caffeine-free tea bags with fruit flavors."),
    ("grocery_005", "grocery", "Dark chocolate bar", "Single origin chocolate with 72 percent cocoa."),
    ("grocery_006", "grocery", "Pasta sauce jar", "Tomato basil sauce for weeknight dinners."),
    ("home_001", "home", "Ceramic coffee mug", "Dishwasher safe mug with a matte glaze finish."),
    ("home_002", "home", "Adjustable desk lamp", "LED lamp with dimming controls and flexible arm."),
    ("home_003", "home", "Cotton throw pillow", "Decorative pillow cover for sofa or bedroom styling."),
    ("home_004", "home", "Bamboo cutting board", "Kitchen prep board with juice groove."),
    ("home_005", "home", "Scented soy candle", "Glass jar candle with clean linen fragrance."),
    ("home_006", "home", "Storage basket set", "Woven baskets for organizing shelves and closets."),
    ("books_001", "books", "Paperback mystery novel", "Fast-paced detective fiction with a coastal setting."),
    ("books_002", "books", "Vegetarian cookbook", "Recipe collection for simple seasonal meals."),
    ("books_003", "books", "Machine learning handbook", "Practical guide to supervised models and evaluation."),
    ("books_004", "books", "Children picture book", "Illustrated bedtime story for early readers."),
    ("books_005", "books", "Travel guide Italy", "City walks, maps, and museum recommendations."),
    ("books_006", "books", "Hardcover art history", "Survey of modern painting, sculpture, and design."),
]

PALETTE = {
    "electronics": ((42, 97, 167), (234, 246, 255)),
    "apparel": ((184, 62, 82), (255, 241, 244)),
    "grocery": ((68, 145, 82), (245, 255, 239)),
    "home": ((166, 119, 53), (255, 248, 233)),
    "books": ((104, 72, 153), (247, 242, 255)),
}


def _font(size: int = 22):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_icon(draw: ImageDraw.ImageDraw, label: str, primary: tuple[int, int, int]) -> None:
    if label == "electronics":
        draw.rounded_rectangle((72, 70, 184, 142), radius=14, outline=primary, width=8)
        draw.ellipse((82, 132, 112, 162), fill=primary)
        draw.ellipse((144, 132, 174, 162), fill=primary)
        draw.line((128, 144, 128, 190), fill=primary, width=8)
    elif label == "apparel":
        draw.polygon([(90, 72), (116, 58), (140, 58), (166, 72), (184, 114), (158, 126), (150, 102), (150, 188), (106, 188), (106, 102), (98, 126), (72, 114)], fill=primary)
        draw.arc((112, 52, 144, 84), 0, 180, fill=(255, 255, 255), width=4)
    elif label == "grocery":
        draw.rounded_rectangle((96, 62, 160, 190), radius=12, fill=primary)
        draw.rectangle((108, 46, 148, 72), fill=primary)
        draw.rounded_rectangle((106, 102, 150, 148), radius=8, fill=(255, 255, 255))
    elif label == "home":
        draw.polygon([(84, 130), (128, 76), (172, 130)], fill=primary)
        draw.rectangle((120, 130, 136, 190), fill=primary)
        draw.rounded_rectangle((96, 184, 160, 198), radius=6, fill=primary)
    elif label == "books":
        draw.rectangle((70, 70, 116, 190), fill=primary)
        draw.rectangle((122, 58, 168, 190), fill=tuple(max(0, c - 35) for c in primary))
        draw.line((86, 88, 102, 88), fill=(255, 255, 255), width=3)
        draw.line((138, 78, 154, 78), fill=(255, 255, 255), width=3)


def _create_placeholder_image(path: Path, label: str, title: str) -> None:
    primary, background = PALETTE[label]
    image = Image.new("RGB", (256, 256), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, 238, 238), radius=18, outline=primary, width=4)
    _draw_icon(draw, label, primary)
    initials = "".join(word[0] for word in title.split()[:2]).upper()
    draw.text((24, 214), initials, fill=primary, font=_font(22))
    draw.text((178, 214), "DEMO", fill=primary, font=_font(14))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def ensure_demo_dataset(output_dir: Path | str = DEMO_DATA_DIR, force: bool = False) -> Path:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    metadata_path = output_dir / "listings.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists() and not force:
        return metadata_path

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "image_path", "title", "description", "label", "is_synthetic"],
        )
        writer.writeheader()
        for item_id, label, title, description in DEMO_ROWS:
            image_rel = f"images/{item_id}.png"
            image_path = output_dir / image_rel
            _create_placeholder_image(image_path, label, title)
            writer.writerow(
                {
                    "id": item_id,
                    "image_path": image_rel,
                    "title": title,
                    "description": description,
                    "label": label,
                    "is_synthetic": "true",
                }
            )
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the synthetic demo product dataset.")
    parser.add_argument("--output-dir", default=str(DEMO_DATA_DIR), help="Directory for demo CSV and images.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing demo files.")
    args = parser.parse_args()
    path = ensure_demo_dataset(args.output_dir, force=args.force)
    print(f"Demo dataset ready: {path}")


if __name__ == "__main__":
    main()
