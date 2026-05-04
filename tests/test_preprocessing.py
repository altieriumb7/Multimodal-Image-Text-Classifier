import unittest
from pathlib import Path

from src.make_demo_data import ensure_demo_dataset
from src.preprocessing import clean_text, load_image, prepare_text

TEST_TMP = Path(__file__).resolve().parents[1] / ".test_tmp"


class PreprocessingTest(unittest.TestCase):
    def test_text_preprocessing_normalizes_case_and_punctuation(self):
        text = prepare_text("USB-C Charger!", "Fast, compact + travel-ready.")
        self.assertEqual(text, "usb-c charger fast compact + travel-ready")
        self.assertEqual(clean_text(None), "")

    def test_image_preprocessing_returns_rgb_resized_image(self):
        TEST_TMP.mkdir(exist_ok=True)
        metadata = ensure_demo_dataset(TEST_TMP / "preprocess" / "demo", force=True)
        image_path = metadata.parent / "images" / "electronics_001.png"
        image = load_image(image_path, image_size=(64, 64))

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (64, 64))


if __name__ == "__main__":
    unittest.main()
