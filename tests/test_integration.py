import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from src.data.data_preprocessing import DataPreprocessing
from src.model import Classifier, NeuralNetwork


def test_preprocessing_and_model():

    data = pd.DataFrame({
        "target": [0, 1] * 20,
        "feature_1": range(40),
        "feature_2": range(40, 80)
    })

    preprocessing = DataPreprocessing(data)

    X_train, X_test, y_train, y_test, _, _ = preprocessing.get_preprocessed_data()

    classifier = Classifier(
        "LogisticRegression"
    )

    model = classifier.create_model({
        "C": 1.0,
        "solver": "liblinear",
        "penalty": "l2"
    })

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})



def test_preprocessing_and_ann():
    data = pd.DataFrame({
        "target": [0, 1] * 20,
        "feature_1": range(40),
        "feature_2": range(40, 80)
    })

    preprocessing = DataPreprocessing(data)

    X_train, X_test, y_train, y_test, _, _ = preprocessing.get_preprocessed_data()

    network = NeuralNetwork(X_train.shape[1])

    params = {
        "units": X_train.shape[1],
        "activation": "relu"
    }

    network.add_dense_layer(**params)

    network.add_dense_layer(units=1, activation='sigmoid')

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )
    optimizer_params = {'optimizer': optimizer}
    network.compile(optimizer_params)

    network.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=1,
        batch_size=4
    )
    model = network.get_model()
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test)
    assert y_pred.shape == (len(y_test), 1)
    assert np.all((y_pred >= 0) & (y_pred <= 1))