import csv
import json
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from src.bootstrap import ensure_demo_artifacts
from src.config import DEFAULT_MODEL_PATH
from src.data import load_dataset
from src.predict import predict_listing
from src.runtime_settings import get_runtime_settings, resolve_backend_choice


st.set_page_config(page_title="Multimodal Product Classifier", layout="wide")


@st.cache_resource(show_spinner=False)
def bootstrap():
    return ensure_demo_artifacts(force=False)


@st.cache_data(show_spinner=False)
def load_examples(metadata_path: str):
    return load_dataset(metadata_path)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_confusion_matrix(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


artifacts = bootstrap()
examples = load_examples(str(artifacts["metadata"]))
metrics = read_json(artifacts["metrics"])
confusion_rows = read_confusion_matrix(artifacts["confusion_matrix"])

runtime = get_runtime_settings()
requested_backend = st.sidebar.selectbox(
    "Feature backend",
    options=["demo", "auto", "clip"],
    index=0,
    help="`demo` is fully offline. `clip` and `auto` may use live model loading.",
)
selected_backend = resolve_backend_choice(
    requested_backend=requested_backend,
    demo_mode=bool(runtime["demo_mode"]),
    allow_live_runs=bool(runtime["allow_live_runs"]),
)

st.sidebar.header("Live Test Settings")
session_token = st.sidebar.text_input(
    "Hugging Face Token (optional, session-only)",
    type="password",
    help="Only needed for gated/private model access in live CLIP mode. Never stored to disk.",
)
if session_token:
    st.session_state["hf_token_session"] = session_token.strip()

active_token = str(st.session_state.get("hf_token_session", "")).strip()
token_present = bool(active_token or runtime["hf_token_present"])
live_requested = selected_backend in {"auto", "clip"} and not bool(runtime["demo_mode"])
live_allowed = live_requested and bool(runtime["allow_live_runs"])
credits_ack = True
if live_requested:
    credits_ack = st.sidebar.checkbox(
        "I understand live runs may consume network/API resources.",
        value=False,
    )

st.sidebar.divider()
st.sidebar.subheader("Runtime Status")
st.sidebar.write(f"Mode: `{'demo' if runtime['demo_mode'] else 'live'}`")
st.sidebar.write(f"DEMO_MODE: `{runtime['demo_mode']}`")
st.sidebar.write(f"ALLOW_LIVE_RUNS: `{runtime['allow_live_runs']}`")
st.sidebar.write(f"HF token present: `{token_present}`")
st.sidebar.write(f"DEFAULT_CONFIG_PATH: `{runtime['default_config_path']}`")
st.sidebar.write(f"REPORTS_DIR: `{runtime['reports_dir']}`")
st.sidebar.write(f"Selected backend: `{selected_backend}`")
st.sidebar.write(f"Selected report: `{artifacts['metrics']}`")

if runtime["demo_mode"]:
    st.info(
        "Public demo mode: live model calls are disabled. "
        "This demo uses local sample artifacts. Clone the repo and disable DEMO_MODE to run live CLIP tests."
    )
elif live_requested and not runtime["allow_live_runs"]:
    st.warning("Live backend selected but ALLOW_LIVE_RUNS=false. Falling back to demo backend.")
elif live_allowed and not credits_ack:
    st.warning("Live mode requires confirmation because it may consume network/API resources.")

st.title("Multimodal Product Classifier")
st.caption("Local demo using product images plus title/description text. The bundled dataset is synthetic demo data.")

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    example_labels = [f"{item.label}: {item.title}" for item in examples[:12]]
    selected = st.selectbox("Example product listings", options=list(range(len(example_labels))), format_func=lambda idx: example_labels[idx])
    selected_example = examples[selected]

    uploaded = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg"])
    default_text = f"{selected_example.title}. {selected_example.description}"
    text = st.text_area("Product title or description", value=default_text, height=120)

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        preview_path = None
    else:
        image = Image.open(selected_example.image_path).convert("RGB")
        preview_path = selected_example.image_path

    st.image(image, caption="Image preview", use_container_width=True)

with right:
    st.button("Refresh prediction", type="primary", use_container_width=True)
    temporary_image_path: Path | None = None
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            temporary_image_path = Path(tmp.name)
            preview_path = temporary_image_path

    result = None
    try:
        if live_requested and not runtime["allow_live_runs"]:
            st.warning("Live runs are disabled by configuration. Set ALLOW_LIVE_RUNS=true to enable.")
        elif live_requested and not credits_ack:
            st.warning("Please confirm live-run resource usage in the sidebar.")
        else:
            try:
                result = predict_listing(
                    image_path=preview_path,
                    text=text,
                    model_path=DEFAULT_MODEL_PATH,
                    backend=selected_backend,
                    allow_download=live_allowed,
                    hf_token=active_token or None,
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
    finally:
        if temporary_image_path is not None and temporary_image_path.exists():
            try:
                temporary_image_path.unlink()
            except OSError:
                pass

    if result is not None:
        metric_left, metric_right = st.columns(2)
        metric_left.metric("Predicted class", result["predicted_class"])
        metric_right.metric("Confidence", format_percent(result["confidence"]))

        st.subheader("Top-3 multimodal predictions")
        st.table(
            [
                {"Rank": idx + 1, "Class": row["class"], "Confidence": format_percent(row["confidence"])}
                for idx, row in enumerate(result["top_3"])
            ]
        )

        st.subheader("Image-only vs text-only vs multimodal")
        st.table(
            [
                {
                    "Model": "Image only",
                    "Prediction": result["comparison"]["image_only"]["predicted_class"],
                    "Confidence": format_percent(result["comparison"]["image_only"]["confidence"]),
                },
                {
                    "Model": "Text only",
                    "Prediction": result["comparison"]["text_only"]["predicted_class"],
                    "Confidence": format_percent(result["comparison"]["text_only"]["confidence"]),
                },
                {
                    "Model": "Multimodal",
                    "Prediction": result["comparison"]["multimodal"]["predicted_class"],
                    "Confidence": format_percent(result["comparison"]["multimodal"]["confidence"]),
                },
            ]
        )

st.divider()

summary_col, matrix_col = st.columns(2, gap="large")
with summary_col:
    st.subheader("Evaluation summary")
    if metrics:
        st.table(
            [
                {
                    "Model": "Image only",
                    "Accuracy": f"{metrics['image']['accuracy']:.3f}",
                    "Macro F1": f"{metrics['image']['macro_f1']:.3f}",
                },
                {
                    "Model": "Text only",
                    "Accuracy": f"{metrics['text']['accuracy']:.3f}",
                    "Macro F1": f"{metrics['text']['macro_f1']:.3f}",
                },
                {
                    "Model": "Multimodal",
                    "Accuracy": f"{metrics['multimodal']['accuracy']:.3f}",
                    "Macro F1": f"{metrics['multimodal']['macro_f1']:.3f}",
                },
            ]
        )
        st.caption("Metrics shown here are computed from the local test split and should be interpreted in context of the synthetic demo dataset.")
    else:
        st.info("Run evaluation to generate metrics.")

with matrix_col:
    st.subheader("Multimodal confusion matrix")
    if confusion_rows:
        st.table(confusion_rows)
    else:
        st.info("No confusion matrix is available yet.")

st.subheader("Preloaded example product listings")
st.table(
    [
        {
            "ID": example.id,
            "Label": example.label,
            "Title": example.title,
            "Description": example.description,
        }
        for example in examples[:6]
    ]
)
