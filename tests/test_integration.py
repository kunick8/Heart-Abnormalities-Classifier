import numpy as np
import tensorflow as tf

from src.data.data_preprocessing import (
    get_preprocessed_data,
    get_preprocessed_data_non_linear,
)
from src.model import Classifier, NeuralNetwork


def test_preprocessing_and_model(imbalanced_dataset):
    X_train, X_test, y_train, y_test, _, _, _ = get_preprocessed_data(
        imbalanced_dataset,
        smote_ratio = 1
    )

    classifier = Classifier("LogisticRegression")
    model = classifier.create_model({
        "C": 1.0,
        "solver": "liblinear",
        "penalty": "l2",
    })

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})


def test_preprocessing_and_ann(imbalanced_dataset):
    # get_preprocessed_data_non_linear is the pipeline meant for ANN/tree
    # models (see ann_feature_selection's docstring), so the ANN integration
    # test exercises that path rather than the RFECV-based one above.
    X_train, X_test, y_train, y_test, _, _, _ = (
        get_preprocessed_data_non_linear(imbalanced_dataset, smote_ratio = 1)
    )

    network = NeuralNetwork(X_train.shape[1])
    network.add_dense_layer(units=X_train.shape[1], activation="relu")
    network.add_dense_layer(units=1, activation="sigmoid")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    network.compile({"optimizer": optimizer})

    network.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=1,
        batch_size=4,
    )

    model = network.get_model()
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test)
    assert y_pred.shape == (len(y_test), 1)
    assert np.all((y_pred >= 0) & (y_pred <= 1))