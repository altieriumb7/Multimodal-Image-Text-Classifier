import unittest
from pathlib import Path

from src.data import load_dataset
from src.extract_features import build_feature_bundle
from src.make_demo_data import ensure_demo_dataset
from src.predict import predict_listing
from src.train_classifier import train_classifier_bundle

TEST_TMP = Path(__file__).resolve().parents[1] / ".test_tmp"


class PredictionTest(unittest.TestCase):
    def test_prediction_output_contains_required_fields(self):
        TEST_TMP.mkdir(exist_ok=True)
        root = TEST_TMP / "prediction"
        root.mkdir(parents=True, exist_ok=True)
        metadata = ensure_demo_dataset(root / "demo", force=True)
        features_path = root / "features.npz"
        model_path = root / "model.joblib"
        build_feature_bundle(metadata, features_path, backend="demo")
        train_classifier_bundle(features_path, model_path, prefer_sklearn=False)
        example = load_dataset(metadata, auto_generate_demo=False)[0]

        result = predict_listing(
            image_path=example.image_path,
            text=example.text,
            model_path=model_path,
            backend=None,
        )

        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertEqual(len(result["top_3"]), 3)
        self.assertIn("image_only", result["comparison"])
        self.assertIn("text_only", result["comparison"])
        self.assertIn("multimodal", result["comparison"])
        self.assertEqual(result["feature_backend_used"], "demo")


if __name__ == "__main__":
    unittest.main()
