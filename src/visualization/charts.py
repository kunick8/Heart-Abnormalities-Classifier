import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

#cm refers to confusion matrix
def visualize_confusion_matrix(cm):
    labels = ['normal', 'abnormal']
    cm_normalized = np.round(cm / np.sum(cm, axis=1).reshape(-1, 1), 2)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        cbar_kws={"label": "Proportion"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized Confusion Matrix")


    return fig

def plot_model_comparison(score_per_model:dict):
    models = []
    scores = []
    for model, score in score_per_model.items():
        models.append(model)
        scores.append(score)

    fig, ax = plt.subplots()

    ax.bar(models, scores)

    ax.set_ylabel("Mean CV F1")
    ax.set_title("Model Performance")

    return fig

#scores refers to model.decision_function(X_test)
def plot_roc_curve(y_test, scores):

    fpr, tpr, _ = roc_curve(
        y_test,
        scores
    )
    auc_score = roc_auc_score(
        y_test,
        scores
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        fpr,
        tpr,
        linewidth=2.5,
        label=f"Model ROC (AUC = {auc_score:.3f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier (AUC = 0.500)"
    )

    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return fig

def plot_pr_curve(y_test, scores):

    precision, recall, _ = precision_recall_curve(
        y_test,
        scores
    )

    ap_score = average_precision_score(
        y_test,
        scores
    )

    baseline = np.mean(y_test)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        recall,
        precision,
        linewidth=2.5,
        label=f"Model PR (AP = {ap_score:.3f})"
    )

    ax.plot(
        [0, 1],
        [baseline, baseline],
        linestyle="--",
        label=f"Random Classifier (Baseline = {baseline:.3f})"
    )

    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall (True Positive Rate)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return fig

def plot_class_pie(y_train, y_test):
    y = np.concatenate((y_train, y_test))
    total = len(y)
    positive = sum(y)
    negative = total - positive
    categories = ['normal', 'abnormal']
    values = [positive/total, negative/total]

    fig, ax = plt.subplots()

    ax.pie(values, labels=categories, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.set_title("Class Pie")
    ax.axis('equal')
    return fig

