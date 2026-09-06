import json
from pathlib import Path

import mlflow
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.visualization.charts import (
    plot_class_pie,
    plot_model_comparison,
    plot_pr_curve,
    plot_roc_curve,
    visualize_confusion_matrix,
)


def evaluate_model(y_test, y_pred):
    project_root = Path(__file__).parent.parent
    metrics_path = project_root / "artifacts" / "model" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'f1_score_macro': f1_score(y_test, y_pred, average='macro'),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def plot_charts(y_test, y_pred, scores):

    cm = confusion_matrix(y_test, y_pred)
    cm_fig = visualize_confusion_matrix(cm)
    roc_fig = plot_roc_curve(y_test, scores)
    pr_fig = plot_pr_curve(y_test, scores)
    class_pie_fig = plot_class_pie(y_test, y_pred)

    project_root = Path(__file__).parent.parent
    metrics_path = project_root / "artifacts" / "scores" / "f1_scores.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "r") as f:
        scores_per_model = json.load(f)
    model_comparison_fig = plot_model_comparison(scores_per_model)


    return {
        'cm_fig': cm_fig,
        'roc_fig': roc_fig,
        'pr_fig': pr_fig,
        'class_pie_fig': class_pie_fig,
        'model_comparison_fig': model_comparison_fig,
    }




def save_and_log_charts(y_test, y_pred, scores):
    results = plot_charts(y_test, y_pred, scores)

    project_root = Path(__file__).parent.parent
    charts_dir = project_root / "artifacts" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    charts = {
        "confusion_matrix": results["cm_fig"],
        "roc_curve": results["roc_fig"],
        "precision_recall_curve": results["pr_fig"],
        "class_pie": results["class_pie_fig"],
        'model_comparison': results["model_comparison_fig"],
    }

    for name, fig in charts.items():
        path = charts_dir /f"{name}.json"

        fig.write_json(str(path))

        mlflow.log_artifact(str(path), artifact_path="charts")

