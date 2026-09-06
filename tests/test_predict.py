from unittest.mock import MagicMock

import numpy as np
import pytest

from src import predict as predict_module
from src.predict import predict


@pytest.fixture
def mock_artifacts(monkeypatch):
    mock_model = MagicMock()
    mock_scaler = MagicMock()
    mock_selector = MagicMock()

    mock_scaler.transform.side_effect = lambda x: x
    mock_selector.transform.side_effect = lambda x: x

    def fake_load(path):
        if "rfc" in path:
            return mock_model
        if "scaler" in path:
            return mock_scaler
        if "selector" in path:
            return mock_selector
        raise FileNotFoundError(path)

    monkeypatch.setattr(predict_module.joblib, "load", MagicMock(side_effect=fake_load))
    return mock_model, mock_scaler, mock_selector


def test_loads_model_scaler_and_selector(mock_artifacts):
    mock_model, _, _ = mock_artifacts
    mock_model.predict.return_value = np.array([0])
    predict(np.array([[1.0, 2.0, 3.0]]))
    assert predict_module.joblib.load.call_count == 3


def test_applies_scaler_then_selector_before_predicting(mock_artifacts):
    mock_model, mock_scaler, mock_selector = mock_artifacts
    mock_model.predict.return_value = np.array([0])
    data = np.array([[1.0, 2.0, 3.0]])

    predict(data)

    mock_scaler.transform.assert_called_once()
    mock_selector.transform.assert_called_once()
    mock_model.predict.assert_called_once()


def test_class_zero_maps_to_normal(mock_artifacts):
    mock_model, _, _ = mock_artifacts
    mock_model.predict.return_value = np.array([0])
    assert predict(np.array([[1.0, 2.0, 3.0]])) == "Normal"


def test_nonzero_class_maps_to_abnormal(mock_artifacts):
    mock_model, _, _ = mock_artifacts
    mock_model.predict.return_value = np.array([1])
    assert predict(np.array([[1.0, 2.0, 3.0]])) == "Abnormal"


def test_missing_artifact_raises(monkeypatch):
    monkeypatch.setattr(
        predict_module.joblib,
        "load",
        MagicMock(side_effect=FileNotFoundError("artifacts/model/rfc.joblib")),
    )
    with pytest.raises(FileNotFoundError):
        predict(np.array([[1.0, 2.0, 3.0]]))
