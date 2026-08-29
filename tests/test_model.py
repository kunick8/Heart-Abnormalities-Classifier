from src.model import Classifier, NeuralNetwork
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pytest
import tensorflow as tf
import numpy as np

def test_logistic_regression_creation():

    classifier = Classifier("LogisticRegression")

    params = {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000
    }

    model = classifier.create_model(params)

    assert isinstance(
        model,
        LogisticRegression
    )

def test_random_forest_creation():

    classifier = Classifier(
        "RandomForestClassifier"
    )

    params = {
        "n_estimators": 100,
        "max_depth": 10
    }

    model = classifier.create_model(params)

    assert isinstance(
        model,
        RandomForestClassifier
    )

def test_naive_bayes_creation():

    classifier = Classifier("NaiveBayes")

    model = classifier.create_model({})

    assert isinstance(
        model,
        GaussianNB
    )

def test_svc_creation():

    classifier = Classifier("SVC")

    params = {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale"
    }

    model = classifier.create_model(params)

    assert isinstance(
        model,
        SVC
    )

def test_xgb_creation():

    classifier = Classifier("XGBClassifier")

    params = {
        "n_estimators": 100,
        "max_depth": 3
    }

    model = classifier.create_model(params)

    assert isinstance(
        model,
        XGBClassifier
    )

def test_invalid_model_name():

    classifier = Classifier("SomethingInvalid")

    with pytest.raises(ValueError):
        classifier.create_model({})


def test_neural_network_creation():
    nn = NeuralNetwork(20)

    model = nn.get_model()

    assert isinstance(model, tf.keras.Sequential)

def test_add_dense_layer():

    nn = NeuralNetwork(32)

    params = {
        "units": 32,
        "activation": "relu"
    }

    nn.add_dense_layer(**params)

    model = nn.get_model()

    assert len(model.layers) == 1

    assert model.layers[0].units == 32
    assert model.layers[0].activation.__name__ == "relu"

def test_output_layer():
    nn = NeuralNetwork(32)

    params = {
        "units": 32,
        "activation": "relu"
    }

    nn.add_dense_layer(**params)
    nn.add_dense_layer(units=1, activation='sigmoid')

    model = nn.get_model()

    assert len(model.layers) == 2
    assert model.layers[-1].units == 1
    assert model.layers[-1].activation.__name__ == "sigmoid"

def test__neural_network_compile():
    nn = NeuralNetwork(32)

    params = {
        "units": 32,
        "activation": "relu"
    }

    nn.add_dense_layer(**params)
    nn.add_dense_layer(units=1, activation='sigmoid')

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )
    optimizer_params = {'optimizer': optimizer}
    nn.compile(optimizer_params)

    model = nn.get_model()

    assert model.optimizer is not None
    assert model.loss == 'binary_crossentropy'

def test_neural_network_fit():
    X = np.random.rand(20, 5)

    y = np.array(
        [0, 1] * 10
    )

    nn = NeuralNetwork(5)

    params = {
        "units": 5,
        "activation": "relu"
    }

    nn.add_dense_layer(**params)
    nn.add_dense_layer(units=1, activation='sigmoid')

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )
    optimizer_params = {'optimizer': optimizer}
    nn.compile(optimizer_params)

    history = nn.fit(
        X,
        y,
        validation_data=(X, y),
        epochs=1,
        batch_size=4
    )

    assert history is not None
    assert "loss" in history.history
    assert "accuracy" in history.history
