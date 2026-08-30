import numpy as np
import streamlit as st

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
    Enter the patient's feature values below.
    """
)

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
                value=0.0,
                format="%.4f",
                key=feature
            )
            values.append(value)

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True
    )

    if submitted:
        X = np.array(values).reshape(1, -1)

st.divider()

prediction = predict(X)

st.write(
    '''
### Result

The model classified this observation as:
    '''
)
st.write(f'Prediction: {prediction}')
st.write(
    '''
This is a machine-learning classification result and
should not be interpreted as a medical diagnosis.'''
)