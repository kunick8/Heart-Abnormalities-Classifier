import json
from pathlib import Path

import plotly.io as pio
import streamlit as st

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
col5.metric("F1 Score (Macro)", f"{metrics['f1_score_macro']:.3f}")

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
    st.write("The Random Forest model turned out to be the best option based on F1 score(macro), it outperformed all other options, I've used macro F1 score as the standard for model performance because the classes are highly imbalanced and the macro F1 score takes that into account.")

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
    st.write(
        '''
        The Model correctly classified 65% of normal cases and 91% of abnormal cases.
        #### Impact of False Positives:   
        The patient gets misdiagnosed with a potential heart abnormality, which may bring the patient a lot of stress.   
        #### Impact of False Negatives:
        The Patient is not diagnosed with an actual condition, the health of the patient may be in danger.
        
        #### Conclusion:   
        The balance is acceptable at this point but it would be beneficial to bring down the False Positive Rate
        '''
    )

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
    st.write('''The Model achieved a ROC-AUC of 0.86, meaning there is an 87% chance the model will correctly rank 
    a randomly chosen positive observation higher than a randomly chosen negative one. The curve shoots straight up at
    the start, that means the Model can correctly classify 45% of abnormal cases before returning first false alarms.   
    The curve then starts plateauing around the 82% TPR, in order to catch the remaining 18% of abnormal cases,
    the model would classify need to classify 76% of normal cases as abnormal.  
    To catch 82% of abnormal cases, the curve shows we must accept a 24% False Positive Rate, which means that 24% of 
    all normal cases will be classified as abnormal. Depending on the operational cost of reviewing false alarms, it
    could be beneficial to select a stricter threshold on the lower-left portion of the curve.
    
    
    '''
             )

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
    st.write('''
    Since the dataset consists of abnormal cases in 79%, a random model would have a horizontal baseline at a precision 
    of 0.79.
    The model achieved an average Precision of 0.96, demonstrating it performs significantly better than random 
    guessing at identifying the target class.   
    The curve remains relatively flat at 87% precision until we reach a recall of 88%. After this point, 
    attempting to capture the remaining 12% of positive cases causes precision to plummet rapidly.  
    To ensure a doctor doesn't worry about false positives, we should operate at the left side of the curve, accepting
    a lower Recall of 91% to maintain a high Precision of 90%. When the model flags an abnormal case the doctor should 
    trust that verdict. Such split would mean that 91% of cases classified as abnormal are correct but 9% of abnormal 
    cases are not classified correctly.
    
    ''')

st.divider()
