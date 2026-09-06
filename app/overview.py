from pathlib import Path

import plotly.io as pio
import streamlit as st

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
    
    The model was trained using the SPECTF dataset,
    it contains precomputed SPECT heart images.
    """
)

st.divider()

st.write(
    '''
## How does it work?

Raw data
   ->
Data preprocessing
   ->
Feature scaling
   ->
SMOTE resampling
   ->
Feature selection
   ->
Random Forest Classifier
   ->
Normal / Abnormal

---

## Final model

Random Forest Classifier

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

5 different Machine Learning models and an Artificial Neural Network were trained, optimized and tested to find the best performing one.

The final model was selected based on F1 score(macro) and accuracy metrics.

---

## Technologies

Python, 
Scikit-learn, 
Optuna, 
MLflow, 
TensorFlow, 
Streamlit, 
Plotly, 
XGBoost, 
Imblearn
    
    '''
)
