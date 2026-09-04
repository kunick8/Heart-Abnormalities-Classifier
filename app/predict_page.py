import numpy as np
import streamlit as st
import sys
import os
from pathlib import Path


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

folder_b_path = os.path.join(parent_dir, 'src')
sys.path.append('src')

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.predict import predict

st.set_page_config(
    page_title="Heart Abnormalities Classifier",
    layout="wide"
)

st.title("Predict")
st.write(
    '''
    The original dataset's instances consist of 44 continuous values in 0-100 range,
    the features are in no way described, so the prediction page should be treated
    as a fun display of the model actually making a prediction
    '''
)

st.header('Make a prediction')
st.write(
    """
    Enter the feature values below.
    """
)
X = None
FEATURE_NAMES = []
for i in range(1,45):
    FEATURE_NAMES.append(f'feature_{i}')
with st.form("prediction_form"):

    st.write("Enter feature values")

    columns = st.columns(4)

    values = []

    for i, feature in enumerate(FEATURE_NAMES):
        with columns[i % 4]:
            value = st.number_input(
                feature,
                min_value=0,
                max_value=100,
                value=0,
                step=1,
                key=feature
            )
            values.append(value)

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True
    )

    if submitted:
        X = np.array(values).reshape(1, -1)
        prediction = predict(X)
st.divider()

if X is not None:
    st.write(
        '''
    ### Result

    The model classified this observation as:
        '''
    )
    st.write(f'##### {prediction}')
else:
    st.write('You must first submit data to make a prediction')

st.write(
    '''
This is a machine-learning classification result and
should not be interpreted as a medical diagnosis.'''
)