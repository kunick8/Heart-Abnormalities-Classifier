import streamlit as st
import json

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

with open("metrics.json", "r") as f:
    metrics = json.load(f)

st.header("Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("F1 Score", f"{metrics['f1_score']:.3f}")
col2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
col3.metric("Precision", f"{metrics['precision']:.3f}")
col4.metric("Recall", f"{metrics['recall']:.3f}")
col5.metric("Roc-auc", f"{metrics['roc_auc_score']:.3f}")


st.divider()

st.image(
    "artifacts/charts/model_comparison.png",
    caption="Model Comparison"
)

st.subheader('interpretation')

st.write()

st.divider()


st.image(
    "artifacts/charts/confusion_matrix.png",
    caption="Confusion Matrix"
)

st.subheader('interpretation')

st.write()

st.divider()

st.image(
    "artifacts/charts/roc_curve.png",
    caption="ROC Curve"
)

st.subheader('interpretation')

st.write()

st.divider()

st.image(
    "artifacts/charts/precision_recall_curve.png",
    caption="Precision-Recall Curve"
)

st.subheader('interpretation')

st.write()

st.divider()
