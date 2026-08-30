import mlflow
import os
import json

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from src.visualization.charts import (
plot_roc_curve,
visualize_confusion_matrix,
plot_pr_curve,
plot_class_pie,
plot_model_comparison
)
from src.mlflow_config import get_best_score_per_model


from src.visualization.optuna_charts import (
plot_param_importances,
plot_optimization_history,
plot_parallel_coordinate
)


def evaluate_model(y_test, y_pred):
    metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc_score': roc_auc_score(y_test, y_pred)
    }

    with open("artifacts/model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def plot_charts(X_test, y_test, y_pred, model):

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        scores = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = visualize_confusion_matrix(cm)
    roc_fig = plot_roc_curve(y_test, scores)
    pr_fig = plot_pr_curve(y_test, scores)
    class_pie_fig = plot_class_pie(y_test, y_pred)
    score_per_model = get_best_score_per_model("http://localhost:5000",{
        'LogisticRegression_optimization',
        'RandomForestClassifier_optimization',
        'NaiveBayes_optimization',
        'XGBClassifier_optimization',
        'SVC_optimization'
        })
    model_comparison_fig = plot_model_comparison(score_per_model)


    return {
        'cm_fig': cm_fig,
        'roc_fig': roc_fig,
        'pr_fig': pr_fig,
        'class_pie_fig': class_pie_fig,
        'model_comparison_fig': model_comparison_fig,
    }




def save_and_log_charts(X_test, y_test, y_pred, model):
    results = plot_charts(X_test, y_test, y_pred, model)

    os.makedirs("artifacts/charts", exist_ok=True)

    charts = {
        "confusion_matrix": results["cm_fig"],
        "roc_curve": results["roc_fig"],
        "precision_recall_curve": results["pr_fig"],
        "class_pie": results["class_pie_fig"],
        'model_comparison': results["model_comparison_fig"],
    }

    for name, fig in charts.items():
        path = f"artifacts/charts/{name}.png"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        mlflow.log_artifact(path)

    return