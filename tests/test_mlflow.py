from unittest.mock import patch
import mlflow


@patch("mlflow.log_param")
def test_log_parameter(mock_log_param):

    mlflow.log_param(
        "model",
        "ANN"
    )

    mock_log_param.assert_called_once_with(
        "model",
        "ANN"
    )