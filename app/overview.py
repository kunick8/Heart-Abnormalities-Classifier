import streamlit as st
import json
import plotly.io as pio
from pathlib import Path

project_root = Path(__file__).parent.parent
charts_dir = project_root / "artifacts" / "charts"
charts_dir.mkdir(parents=True, exist_ok=True)
path = charts_dir / "class_pie.json"


st.set_page_config(
    page_title="Heart Abnormalities Classifier",
    layout="wide"
)

st.title("Overview")

st.write(
    """
    This application uses a machine learning model
    to classify observations as normal or abnormal.
    
    The model was trained using the SPECTF dataset.
    """
)

st.divider()

st.write(
    '''
## How does it work?

Raw data
   ↓
Data preprocessing
   ↓
Feature scaling
   ↓
Feature selection
   ↓
SVC
   ↓
Normal / Abnormal

---

## Final model

SVC

## Data Split


'''
)
st.divider()

try:
    fig = pio.read_json(str(path))
    st.plotly_chart(fig, use_container_width=True)
except FileNotFoundError:
    st.error("File not found")

st.write(
    '''

## Model selection

Several machine learning algorithms were evaluated
using cross-validation and hyperparameter optimization
with Optuna.

The final model was selected based on mean CV F1 score.

---

## Technologies

Python
Scikit-learn
Optuna
MLflow
TensorFlow
Streamlit
    
    '''
)
