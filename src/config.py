from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DEMO_DATA_DIR = DATA_DIR / "demo"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

DEMO_METADATA_PATH = DEMO_DATA_DIR / "listings.csv"
DEFAULT_FEATURES_PATH = MODELS_DIR / "features_demo.npz"
DEFAULT_MODEL_PATH = MODELS_DIR / "classifier_bundle.joblib"
DEFAULT_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
DEFAULT_CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix_multimodal.csv"
DEFAULT_REPORT_PATH = REPORTS_DIR / "evaluation_report.md"

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
RANDOM_SEED = 42
