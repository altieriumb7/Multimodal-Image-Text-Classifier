import unittest

import numpy as np

from src.metrics import classification_metrics, confusion_matrix, top_k_predictions


class MetricsTest(unittest.TestCase):
    def test_metric_computation(self):
        y_true = np.array(["a", "a", "b", "b"])
        y_pred = np.array(["a", "b", "b", "b"])
        labels = ["a", "b"]

        matrix = confusion_matrix(y_true, y_pred, labels)
        metrics = classification_metrics(y_true, y_pred, labels)

        self.assertEqual(matrix.tolist(), [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertIn("precision", metrics["per_class"]["a"])
        self.assertIn("recall", metrics["per_class"]["b"])

    def test_top_k_predictions(self):
        probabilities = np.array([[0.2, 0.7, 0.1]])
        classes = np.array(["a", "b", "c"])
        top = top_k_predictions(probabilities, classes, k=2)

        self.assertEqual(top[0][0]["class"], "b")
        self.assertEqual(len(top[0]), 2)


if __name__ == "__main__":
    unittest.main()
