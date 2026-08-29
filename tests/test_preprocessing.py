from src.data.data_preprocessing import DataPreprocessing
import pandas as pd
import numpy as np
from src.data.data_cleaning import join_data
from sklearn.feature_selection import RFECV

def test_data_splitter():
    data = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "feature_1": [10, 20, 30, 40],
        "feature_2": [1, 2, 3, 4]
    })

    preprocessor = DataPreprocessing(data)
    X, y = preprocessor.data_splitter()
    assert X.shape == (4, 2)
    assert y.shape == (4,)

    assert np.array_equal(
        y,
        np.array([0,1,0,1])
    )

def test_train_test_split():
    data = pd.DataFrame({
        "target": [0, 1] * 5,
        'feature_1': range(10),
        'feature_2': range(10,20)
    })

    preprocessor = DataPreprocessing(data)
    X, y = preprocessor.data_splitter()

    X_train, X_test, y_train, y_test = preprocessor.training_test_split(X,y)

    assert len(X_train) == 7
    assert len(X_test) == 3

    assert len(y_train) == 7
    assert len(y_test) == 3

def test_feature_scaling():
    data = pd.DataFrame({
        "target": [0, 1] * 5,
        'feature_1': range(10),
        'feature_2': range(10, 20)
    })

    preprocessor = DataPreprocessing(data)
    X, y = preprocessor.data_splitter()

    X_train, X_test, y_train, y_test = preprocessor.training_test_split(X, y)

    X_train, X_test, scaler = preprocessor.feature_scaling(X_train, X_test)

    assert np.allclose(
        X_train.mean(axis=0),
        0,
        atol=1e-7
    )

    assert np.allclose(
        X_train.std(axis=0),
        1,
        atol=1e-7
    )

def test_feature_selection():
    data = pd.DataFrame({
        "target": [0, 1] * 5,
        'feature_1': range(10),
        'feature_2': range(10, 20)
    })

    preprocessor = DataPreprocessing(data)
    X, y = preprocessor.data_splitter()

    X_train, X_test, y_train, y_test = preprocessor.training_test_split(X, y)

    X_train, X_test, scaler = preprocessor.feature_scaling(X_train, X_test)

    X_train_sel, X_test_sel, selector = preprocessor.feature_selection(X_train, X_test, y_train)

    assert X_train_sel.shape[0] == X_train.shape[0]
    assert X_test_sel.shape[0] == X_test.shape[0]

    assert X_train_sel.shape[1] == X_test_sel.shape[1]

    assert X_train_sel.shape[1] <= X_train.shape[1]

    assert isinstance(selector, RFECV)



def test_data_cleaning():
    data = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "feature_1": [10, 20, 30, 40],
        "feature_2": [1, 2, 3, 4]
    })

    data2 = pd.DataFrame({
        "target": [0, 2, 0, 2],
        "feature_1": [20, 30, 40, 50],
        "feature_2": [2, 3, 4, 5]
    })

    data3 = join_data(data, data2)

    assert data3.shape == (8, 3)
