import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from src.evaluation import evaluate_model, plot_charts, save_and_log_charts

from src import evaluation


@pytest.fixture
def fake_project_root(tmp_path, monkeypatch):
    fake_file = tmp_path / "project" / "src" / "evaluation.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.touch()
    monkeypatch.setattr(evaluation, "__file__", str(fake_file))
    return fake_file.parent.parent  # .../project


@pytest.fixture
def mock_chart_functions(monkeypatch):
    fake_fig = MagicMock()
    for name in (
            "visualize_confusion_matrix", "plot_roc_curve",
            "plot_pr_curve", "plot_class_pie", "plot_model_comparison",
    ):
        monkeypatch.setattr(evaluation, name, MagicMock(return_value=fake_fig))
    return fake_fig


def test_evaluate_model_returns_expected_keys(fake_project_root, y_true_pred):
    y_test, y_pred = y_true_pred
    metrics = evaluate_model(y_test, y_pred)

    assert set(metrics) == {
        "accuracy", "precision", "recall", "f1_score", "f1_score_macro",
    }


def test_evaluate_model_metrics_are_in_valid_range(fake_project_root, y_true_pred):
    y_test, y_pred = y_true_pred
    metrics = evaluate_model(y_test, y_pred)
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_evaluate_model_perfect_predictions_score_one(fake_project_root):
    y = np.array([0, 1, 0, 1, 1])
    metrics = evaluate_model(y, y)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_evaluate_model_writes_metrics_json(fake_project_root, y_true_pred):
    y_test, y_pred = y_true_pred
    evaluate_model(y_test, y_pred)

    metrics_path = fake_project_root / "artifacts" / "model" / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path) as f:
        saved = json.load(f)
    assert saved.keys() == {
        "accuracy", "precision", "recall", "f1_score", "f1_score_macro",
    }


def test_plot_charts_returns_all_five_figures(
        fake_project_root, mock_chart_functions, y_true_pred
):
    scores_path = fake_project_root / "artifacts" / "scores" / "f1_scores.json"
    scores_path.parent.mkdir(parents=True)
    scores_path.write_text(json.dumps({"rfc": 0.8, "svc": 0.75}))

    y_test, y_pred = y_true_pred
    scores = np.random.RandomState(0).rand(len(y_test))
    result = plot_charts(y_test, y_pred, scores)

    assert set(result) == {
        "cm_fig", "roc_fig", "pr_fig", "class_pie_fig", "model_comparison_fig",
    }
    evaluation.plot_model_comparison.assert_called_once_with(
        {"rfc": 0.8, "svc": 0.75}
    )


def test_plot_charts_raises_when_f1_scores_file_missing(
        fake_project_root, mock_chart_functions, y_true_pred
):
    y_test, y_pred = y_true_pred
    scores = np.random.RandomState(0).rand(len(y_test))

    with pytest.raises(FileNotFoundError):
        plot_charts(y_test, y_pred, scores)


def test_save_and_log_charts_writes_and_logs_one_artifact_per_chart(
        fake_project_root, mock_chart_functions, y_true_pred, monkeypatch
):
    scores_path = fake_project_root / "artifacts" / "scores" / "f1_scores.json"
    scores_path.parent.mkdir(parents=True)
    scores_path.write_text(json.dumps({"rfc": 0.8}))

    mock_log_artifact = MagicMock()
    monkeypatch.setattr(evaluation.mlflow, "log_artifact", mock_log_artifact)

    y_test, y_pred = y_true_pred
    scores = np.random.RandomState(0).rand(len(y_test))
    save_and_log_charts(y_test, y_pred, scores)

    assert mock_chart_functions.write_json.call_count == 5
    assert mock_log_artifact.call_count == 5
    for call in mock_log_artifact.call_args_list:
        assert call.kwargs.get("artifact_path") == "charts"


def test_save_and_log_charts_creates_charts_directory(
        fake_project_root, mock_chart_functions, y_true_pred, monkeypatch
):
    scores_path = fake_project_root / "artifacts" / "scores" / "f1_scores.json"
    scores_path.parent.mkdir(parents=True)
    scores_path.write_text(json.dumps({"rfc": 0.8}))
    monkeypatch.setattr(evaluation.mlflow, "log_artifact", MagicMock())

    y_test, y_pred = y_true_pred
    scores = np.random.RandomState(0).rand(len(y_test))
    save_and_log_charts(y_test, y_pred, scores)

    assert (fake_project_root / "artifacts" / "charts").is_dir()