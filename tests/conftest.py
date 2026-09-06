import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_dataset():
    return pd.DataFrame({
        "target": [0, 1, 0, 1],
        "feature_1": [10, 20, 30, 40],
        "feature_2": [1, 2, 3, 4],
    })


@pytest.fixture
def medium_dataset():
    return pd.DataFrame({
        "target": [0, 1] * 5,
        "feature_1": range(10),
        "feature_2": range(10, 20),
    })


@pytest.fixture
def large_dataset():
    return pd.DataFrame({
        "target": [0, 1] * 20,
        "feature_1": range(40),
        "feature_2": range(40, 80),
    })


@pytest.fixture
def imbalanced_dataset():
    rng = np.random.RandomState(42)
    n_samples = 60
    n_features = 6
    X = rng.normal(size=(n_samples, n_features))
    y = np.array([0] * 40 + [1] * 20)
    rng.shuffle(y)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df.insert(0, "target", y)
    return df


@pytest.fixture
def y_true_pred():
    y_test = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
    return y_test, y_pred