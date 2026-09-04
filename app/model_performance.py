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

chart_col1, text_col1 = st.columns([2, 1], gap="large")

with chart_col1:
    try:
        fig = pio.read_json(str(charts_path / 'model_comparison.json'))
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.error("File not found")

with text_col1:
    st.subheader('Interpretation')
    st.write("The SVC model turned out to be the best option based on F1 score, it outperformed all other options, I've used F1 score as the standard for model performance because the classes are highly imbalanced and F1 score takes that into account.")

st.divider()

chart_col2, text_col2 = st.columns([2, 1], gap="large")

with chart_col2:
    try:
        fig_cm = pio.read_json(str(charts_path / 'confusion_matrix.json'))
        st.plotly_chart(fig_cm, use_container_width=True)
    except FileNotFoundError:
        st.error("File not found")

with text_col2:
    st.subheader('Interpretation')
    st.write("The Confusion Matrix successfully classified 35% of normal cases as normal which is worrying as it opted for the 'easy' approach where it classifies the case as abnormal when in doubt."
             " on a more positive note, it led to the model accurately classifying 89% of abnormal cases")

st.divider()

chart_col3, text_col3 = st.columns([2, 1], gap="large")

with chart_col3:
    try:
        fig_roc = pio.read_json(str(charts_path / 'roc_curve.json'))
        st.plotly_chart(fig_roc, use_container_width=True)
    except FileNotFoundError:
        st.error("File not found")

with text_col3:
    st.subheader('Interpretation')
    st.write("The ROC curve depicts that the model performs much better than random guessing, the AUC is in the 'good model' range, ")

st.divider()

chart_col4, text_col4 = st.columns([2, 1], gap="large")

with chart_col4:
    try:
        fig_pr = pio.read_json(str(charts_path / 'precision_recall_curve.json'))
        st.plotly_chart(fig_pr, use_container_width=True)
    except FileNotFoundError:
        st.error("File not found")

with text_col4:
    st.subheader('Interpretation')
    st.write("Your precision-recall interpretation goes here...")

st.divider()
