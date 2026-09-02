import json
from pathlib import Path
import streamlit as st
import plotly.io as pio

project_root = Path(__file__).parent.parent
metrics_path = project_root / "artifacts" / "model" / "metrics.json"
charts_path = project_root / "artifacts" / "charts"

st.set_page_config(
    page_title="Heart Abnormalities Classifier",
    layout="wide"
)

st.title("📊 Model Performance")

st.write(
    """
    Evaluation results for the final SVC model
    on the held-out test dataset.
    """
)

with open(metrics_path, "r") as f:
    metrics = json.load(f)

st.header("Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("F1 Score", f"{metrics['f1_score']:.3f}")
col2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
col3.metric("Precision", f"{metrics['precision']:.3f}")
col4.metric("Recall", f"{metrics['recall']:.3f}")
col5.metric("Roc-auc", f"{metrics['roc_auc_score']:.3f}")


st.divider()

try:
    fig = pio.read_json(str(charts_path / 'model_comparison.json'))
    st.plotly_chart(fig, use_container_width=True)
except FileNotFoundError:
    st.error("File not found")

st.subheader('interpretation')

st.write()

st.divider()


try:
    fig_cm = pio.read_json(str(charts_path / 'confusion_matrix.json'))
    st.plotly_chart(fig_cm, use_container_width=True)
except FileNotFoundError:
    st.error("File not found")

st.subheader('interpretation')

st.write()

st.divider()

try:
    fig_roc = pio.read_json(str(charts_path / 'roc_curve.json'))
    st.plotly_chart(fig_roc, use_container_width=True)
except FileNotFoundError:
    st.error("File not found")

st.subheader('interpretation')

st.write()

st.divider()

try:
    fig_pr = pio.read_json(str(charts_path / 'precision_recall_curve.json'))
    st.plotly_chart(fig_pr, use_container_width=True)
except FileNotFoundError:
    st.error("File not found")

st.subheader('interpretation')

st.write()

st.divider()
