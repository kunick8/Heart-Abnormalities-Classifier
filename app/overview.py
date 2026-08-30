import streamlit as st

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

st.image(
    "artifacts/charts/class_pie.png",
    caption="Class Pie"
)

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
