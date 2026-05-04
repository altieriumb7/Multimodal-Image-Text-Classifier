# Multimodal Image + Text Product Classifier

Portfolio-quality, local-first classifier for product listings that combines an image and a title/description. The project includes preprocessing, feature extraction, baseline models, multimodal fusion, evaluation, a prediction CLI, tests, and a Streamlit demo.

## What It Builds

- **Image-only baseline** trained on image embeddings.
- **Text-only baseline** trained on text embeddings.
- **Multimodal classifier** trained on concatenated image + text embeddings.
- **Evaluation report** with accuracy, macro F1, per-class precision/recall/F1, and a confusion matrix.
- **Interactive demo** for upload + text prediction and baseline comparison.

The pipeline is CLIP-ready through Hugging Face `transformers`. If CLIP dependencies or weights are unavailable, the default demo path uses a deterministic offline feature extractor so the repository can still run locally and be tested. Any synthetic/demo usage is clearly marked.

## Project Structure

```text
data/
  README.md
  demo/                  # generated synthetic demo CSV and placeholder images
models/                  # generated feature/model artifacts
notebooks/
reports/                 # generated metrics, confusion matrix, report
src/
  data.py
  preprocessing.py
  features.py
  extract_features.py
  train_classifier.py
  evaluate_classifier.py
  predict.py
tests/
app.py
requirements.txt
README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Data

The repository defaults to a small synthetic product dataset generated locally:

```bash
python -m src.make_demo_data
```

The demo dataset contains placeholder images and labels for `electronics`, `apparel`, `grocery`, `home`, and `books`. It is intended for pipeline validation, not real-world performance claims.

To use real data, provide a CSV with:

```text
id,image_path,title,description,label
```

`image_path` can be absolute or relative to the CSV file.

## Reproducible Pipeline

Offline demo feature extraction:

```bash
python -m src.extract_features --backend demo
python -m src.train_classifier
python -m src.evaluate_classifier
```

Real CLIP feature extraction, if you have internet/model access:

```bash
python -m src.extract_features --backend clip --allow-download
python -m src.train_classifier
python -m src.evaluate_classifier
```

Prediction CLI:

```bash
python -m src.predict --image data/demo/images/electronics_001.png --text "wireless headphones with bluetooth"
```

## Demo

```bash
streamlit run app.py
```

The demo shows:

- uploaded image preview
- text input
- predicted class and confidence
- top-3 predicted classes
- image-only, text-only, and multimodal comparison
- confusion matrix
- example product listings
- preloaded examples

## Results

Run:

```bash
python -m src.evaluate_classifier
```

This writes:

- `reports/evaluation_metrics.json`
- `reports/confusion_matrix_multimodal.csv`
- `reports/evaluation_report.md`

Metrics are computed from the actual local artifacts. No fixed accuracy or operational impact claim is made in this README because results depend on the dataset and feature backend used. Demo-data metrics are a smoke test only.

## Testing

```bash
python -m unittest discover -s tests
```

or, after installing `pytest`:

```bash
pytest
```

## Limitations

- The bundled data is synthetic and small.
- The offline demo extractor is not CLIP; it exists to keep tests and demos runnable without downloads.
- Real CLIP embeddings require `torch`, `transformers`, and accessible model weights.
- The classifier is intentionally simple and local: logistic regression when `scikit-learn` is installed, otherwise a deterministic centroid fallback.
- No Pinterest-scale, production-scale, or manual-tagging reduction claims are made.
