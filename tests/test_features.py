import unittest
from pathlib import Path

from src.data import load_dataset
from src.features import DemoFeatureExtractor, extract_feature_arrays
from src.make_demo_data import ensure_demo_dataset

TEST_TMP = Path(__file__).resolve().parents[1] / ".test_tmp"


class FeatureExtractionTest(unittest.TestCase):
    def test_demo_feature_shapes_match_dataset(self):
        TEST_TMP.mkdir(exist_ok=True)
        metadata = ensure_demo_dataset(TEST_TMP / "features" / "demo", force=True)
        examples = load_dataset(metadata, auto_generate_demo=False)[:4]
        extractor = DemoFeatureExtractor(embedding_dim=32)
        image_embeddings, text_embeddings = extract_feature_arrays(examples, extractor)

        self.assertEqual(image_embeddings.shape, (4, 32))
        self.assertEqual(text_embeddings.shape, (4, 32))


if __name__ == "__main__":
    unittest.main()
