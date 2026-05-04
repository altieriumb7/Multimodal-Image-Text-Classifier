import unittest
from pathlib import Path

from src.data import load_dataset, stratified_split
from src.make_demo_data import ensure_demo_dataset

TEST_TMP = Path(__file__).resolve().parents[1] / ".test_tmp"


class DatasetLoadingTest(unittest.TestCase):
    def test_demo_dataset_loads_with_expected_columns(self):
        TEST_TMP.mkdir(exist_ok=True)
        metadata = ensure_demo_dataset(TEST_TMP / "data_load" / "demo", force=True)
        examples = load_dataset(metadata, auto_generate_demo=False)

        self.assertEqual(len(examples), 30)
        self.assertGreaterEqual(len({example.label for example in examples}), 5)
        self.assertTrue(all(example.image_path.exists() for example in examples))
        self.assertTrue(all(example.is_synthetic for example in examples))

    def test_stratified_split_has_all_splits(self):
        TEST_TMP.mkdir(exist_ok=True)
        metadata = ensure_demo_dataset(TEST_TMP / "split" / "demo", force=True)
        examples = load_dataset(metadata, auto_generate_demo=False)
        splits = stratified_split(examples)

        self.assertEqual(set(splits), {"train", "val", "test"})
        self.assertTrue(all(splits[name] for name in splits))
        self.assertEqual(sum(len(rows) for rows in splits.values()), len(examples))


if __name__ == "__main__":
    unittest.main()
