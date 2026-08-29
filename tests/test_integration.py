import pandas as pd
from src.model import Classifier, NeuralNetwork
from src.data.data_preprocessing import DataPreprocessing
import tensorflow as tf


def test_preprocessing_and_model():

    data = pd.DataFrame({
        "target": [0, 1] * 20,
        "feature_1": range(40),
        "feature_2": range(40, 80)
    })

    preprocessing = DataPreprocessing(data)

    X_train, X_test, y_train, y_test, scaler, selector = preprocessing.get_preprocessed_data()

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



def test_preprocessing_and_ann():
    data = pd.DataFrame({
        "target": [0, 1] * 20,
        "feature_1": range(40),
        "feature_2": range(40, 80)
    })

    preprocessing = DataPreprocessing(data)

    X_train, X_test, y_train, y_test, scaler, selector = preprocessing.get_preprocessed_data()

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

    history = network.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=1,
        batch_size=4
    )
    model = network.get_model()
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test)