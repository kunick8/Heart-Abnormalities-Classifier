from unittest.mock import MagicMock, patch

import pytest

from src.mlflow_config import (
    get_best_params,
    get_best_run,
    get_converted_params,
    log_ann_trial,
    log_trial,
)


@patch("src.mlflow_config.mlflow.log_param")
@patch("src.mlflow_config.mlflow.log_metric")
def test_log_trial(
    mock_log_metric,
    mock_log_param
):
    model_name = 'SVC'
    params = {
        "C": 1.0
    }
    cv_f1 = 0.92
    score_std = 0.3
    metrics = {'mean_cv_f1': cv_f1, 'mean_std_f1': score_std}

    log_trial(model_name, params, trial_number= 10, metrics=metrics)

    mock_log_param.assert_any_call(
        "model",
        "SVC"
    )

    mock_log_param.assert_any_call(
        "C",
        1.0
    )

    mock_log_param.assert_any_call(
        'trial_number',
        10
    )

    mock_log_metric.assert_any_call(
        "mean_cv_f1",
        0.92
    )
    mock_log_metric.assert_any_call(
        "mean_std_f1",
        0.3
    )

@patch("src.mlflow_config.mlflow.log_param")
@patch("src.mlflow_config.mlflow.log_metric")
def test_log_ann_trial(
    mock_log_metric,
    mock_log_param
):

    params = {
        'learning_rate': 0.01
    }
    metrics = {'val_accuracy': 0.9}


    log_ann_trial(params, trial_number=10, metrics=metrics)

    mock_log_param.assert_any_call(
        "trial_number",
        10
    )

    mock_log_param.assert_any_call(
        "learning_rate",
        0.01
    )

    mock_log_metric.assert_called_once_with(
        "val_accuracy",
        0.9
    )






@patch("src.mlflow_config.MlflowClient")
def test_get_best_run(mock_client):

    mock_client_instance = mock_client.return_value

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "123"

    mock_best_run = MagicMock()

    mock_client_instance.get_experiment_by_name.return_value = (
        mock_experiment
    )

    mock_client_instance.search_runs.return_value = [
        mock_best_run
    ]

    result = get_best_run(
        "http://localhost:5000",
        "SVC_optimization"
    )

    assert result == mock_best_run


    mock_client_instance.get_experiment_by_name.assert_called_once_with(
        "SVC_optimization"
    )

    mock_client_instance.search_runs.assert_called_once_with(
        experiment_ids=["123"],
        order_by=["metrics.mean_cv_f1 DESC"],
        max_results=1
    )

@patch("src.mlflow_config.MlflowClient")
def test_get_best_run_experiment_not_found(mock_client):

    mock_client_instance = mock_client.return_value

    mock_client_instance.get_experiment_by_name.return_value = None

    with pytest.raises(
        ValueError,
        match="Experiment 'SVC_optimization' not found"
    ):
        get_best_run(
            "http://localhost:5000",
            "SVC_optimization"
        )

@patch("src.mlflow_config.get_best_run")
def test_get_best_params(mock_get_best_run):

    mock_run = MagicMock()

    mock_run.data.params = {
        "model": "SVC",
        "C": "2.5",
        "gamma": "0.01"
    }

    mock_get_best_run.return_value = mock_run

    params = get_best_params(
        "http://localhost:5000",
        "SVC_optimization"
    )

    assert params == {
        "model": "SVC",
        "C": "2.5",
        "gamma": "0.01"
    }

    mock_get_best_run.assert_called_once_with(
        "http://localhost:5000",
        "SVC_optimization"
    )

@patch("src.mlflow_config.get_best_params")
def test_get_converted_params_svc(mock_get_best_params):

    mock_get_best_params.return_value = {
        "model": "SVC",
        "C": "2.5",
        "gamma": "0.01",
        "kernel": "rbf"
    }

    params = get_converted_params(
        "http://localhost:5000",
        "SVC_optimization"
    )


    assert isinstance(params["C"], float)
    assert params["C"] == 2.5

    assert isinstance(params["gamma"], float)
    assert params["gamma"] == 0.01

    assert params["kernel"] == "rbf"

@patch("src.mlflow_config.get_best_params")
def test_get_converted_params_logistic_regression(
    mock_get_best_params
):

    mock_get_best_params.return_value = {
        "model": "LogisticRegression",
        "max_iter": "1000",
        "C": "1.5",
        "random_state": "42"
    }

    params = get_converted_params(
        "http://localhost:5000",
        "LogisticRegression_optimization"
    )

    assert isinstance(params["max_iter"], int)
    assert isinstance(params["C"], float)
    assert isinstance(params["random_state"], int)

    assert params["max_iter"] == 1000
    assert params["C"] == 1.5
    assert params["random_state"] == 42

@patch("src.mlflow_config.get_best_params")
def test_get_converted_params_random_forest(
    mock_get_best_params
):

    mock_get_best_params.return_value = {
        "model": "RandomForestClassifier",
        "n_estimators": "200",
        "max_depth": "20",
        "min_samples_split": "4"
    }

    params = get_converted_params(
        "http://localhost:5000",
        "RandomForestClassifier_optimization"
    )

    assert isinstance(params["n_estimators"], int)
    assert isinstance(params["max_depth"], int)
    assert isinstance(params["min_samples_split"], int)

@patch("src.mlflow_config.get_best_params")
def test_get_converted_params_naive_bayes(
    mock_get_best_params
):

    mock_get_best_params.return_value = {
        "model": "NaiveBayes",
        "var_smoothing": "0.000001"
    }

    params = get_converted_params(
        "http://localhost:5000",
        "NaiveBayes_optimization"
    )

    assert isinstance(
        params["var_smoothing"],
        float
    )

    assert params["var_smoothing"] == 0.000001

@patch("src.mlflow_config.get_best_params")
def test_get_converted_params_xgb(
    mock_get_best_params
):

    mock_get_best_params.return_value = {
        "model": "XGBClassifier",
        "n_estimators": "200",
        "max_depth": "5",
        "min_child_weight": "2",
        "scale_pos_weight": "3.5",
        "learning_rate": "0.1"
    }

    params = get_converted_params(
        "http://localhost:5000",
        "XGBClassifier_optimization"
    )

    assert isinstance(params["n_estimators"], int)
    assert isinstance(params["max_depth"], int)
    assert isinstance(params["min_child_weight"], int)
    assert isinstance(params["scale_pos_weight"], float)
    assert isinstance(params["learning_rate"], float)

@patch("src.mlflow_config.MlflowClient")
def test_get_best_run_no_runs(mock_client):

    mock_client_instance = mock_client.return_value

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "123"

    mock_client_instance.get_experiment_by_name.return_value = (
        mock_experiment
    )

    mock_client_instance.search_runs.return_value = []

    with pytest.raises(
        ValueError,
        match="contains no runs"
    ):
        get_best_run(
            "http://localhost:5000",
            "SVC_optimization"
        )