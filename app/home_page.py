import streamlit as st

st.set_page_config(
    page_title="Heart Abnormalities Classifier",
    layout="wide"
)

st.title("Heart Abnormalities Classifier")

st.write(
    """
    Welcome to the Heart Abnormalities Classifier.

    This application uses a machine learning model
    to classify observations, created from processed 
    SPECT images, as normal or abnormal.

    Use the navigation menu to learn about the model,
    explore its performance, or make a prediction.
    """
)