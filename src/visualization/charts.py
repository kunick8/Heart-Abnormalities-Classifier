import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


#cm refers to confusion matrix
def visualize_confusion_matrix(cm):
    labels = ['normal', 'abnormal']
    cm_normalized = np.round(cm / np.sum(cm, axis=1).reshape(-1, 1), 2)

    fig = px.imshow(
        cm_normalized,
        text_auto=".2f",
        color_continuous_scale="Greens",
        x=labels,
        y=labels,
        labels=dict(x="Predicted label", y="True label", color="Proportion"),
        title="Normalized Confusion Matrix"
    )

    return fig

def plot_model_comparison(score_per_model:dict):
    models = list(score_per_model.keys())
    scores = list(score_per_model.values())

    fig = px.bar(
        x=models,
        y=scores,
        labels={'x': 'Models', 'y': 'Mean CV F1'},
        title="Model Performance",
        color=scores,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(showlegend=False)
    return fig

#scores refers to model.decision_function(X_test)
def plot_roc_curve(y_test, scores):
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc_score = roc_auc_score(y_test, scores)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'Model ROC (AUC = {auc_score:.3f})',
        line=dict(width=3, color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        hovermode="x unified",
        xaxis=dict(range=[-0.01, 1.0]),
        yaxis=dict(range=[0.0, 1.05])
    )
    return fig

def plot_pr_curve(y_test, scores):
    precision, recall, _ = precision_recall_curve(y_test, scores)
    ap_score = average_precision_score(y_test, scores)
    baseline = np.mean(y_test)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'Model PR (AP = {ap_score:.3f})',
        line=dict(width=3, color='purple')
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[baseline, baseline],
        mode='lines',
        name=f'Random Classifier (Baseline = {baseline:.3f})',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall (True Positive Rate)",
        yaxis_title="Precision (Positive Predictive Value)",
        hovermode="x unified",
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.05])
    )

    return fig

def plot_class_pie(y_train, y_test):
    y = np.concatenate((y_train, y_test))
    total = len(y)
    positive = sum(y)
    negative = total - positive

    categories = ['abnormal', 'normal']
    values = [positive, negative]

    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=values,
        textinfo='label+percent',
        insidetextorientation='radial',
        marker=dict(colors=['#ff9999', '#99ff99'])
    )])

    fig.update_layout(
        title_text="Class Distribution (Train + Test)"
    )

    return fig

