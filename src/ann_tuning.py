import mlflow
import optuna
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.mlflow_config import log_ann_trial, setup_mlflow
from src.model import NeuralNetwork
from src.data.data_preprocessing import get_preprocessed_data_non_linear, feature_scaling, ann_feature_selection

def objective(trial, X_train, y_train):

    n_layers = trial.suggest_int("n_layers", 1, 5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["sgd", "adam"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    layer_params = {}
    for i in range(n_layers):
        if i == 0:
            num_hidden = X_train.shape[1]
        else:
            num_hidden = trial.suggest_int(f"{i}_layer_neurons", 6, 128)

        activation = trial.suggest_categorical(f"{i}_layer_activation", ["relu", "tanh"])
        layer_params[f"{i}_layer_activation"] = activation
        layer_params[f"{i}_layer_neurons"] = num_hidden

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_losses = []

    with mlflow.start_run(run_name=f"ANN_trial_{trial.number}"):

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):

            tf.keras.backend.clear_session()

            X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
            X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]


            X_fold_train, X_fold_val, _ = feature_scaling(X_fold_train, X_fold_val)
            X_fold_train, X_fold_val, _ = ann_feature_selection(X_fold_train, X_fold_val, y_fold_train)


            network = NeuralNetwork(input_dim=X_fold_train.shape[1])

            for i in range(n_layers):
                network.add_dense_layer(
                    units=layer_params[f"{i}_layer_neurons"],
                    activation=layer_params[f"{i}_layer_activation"]
                )

            network.add_dense_layer(units=1, activation='sigmoid')

            if optimizer_name == "sgd":
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            network.compile({'optimizer': optimizer})

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                )
            ]

            history = network.fit(
                X_fold_train,
                y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                batch_size=batch_size,
                callbacks=callbacks,
                epochs=100
            )

            best_val_accuracy = max(history.history['val_accuracy'])
            best_val_loss = min(history.history['val_loss'])

            fold_accuracies.append(best_val_accuracy)
            fold_losses.append(best_val_loss)

        avg_accuracy = np.mean(fold_accuracies)
        avg_loss = np.mean(fold_losses)

        params = {
            'n_layers': n_layers,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimizer': optimizer_name
        }
        params.update(layer_params)
        metrics = {'best_accuracy': avg_accuracy, 'best_loss': avg_loss}

        log_ann_trial(params=params, metrics=metrics, trial_number=trial.number)

    return avg_accuracy

def tune_ann(
    X_train,
    y_train,
    n_trials=100
):
    setup_mlflow(experiment_name="ANN_keras_optimization")
    study = optuna.create_study(direction="maximize", study_name="ANN_optimization")

    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    best_trial = study.best_trial

    with mlflow.start_run(
            run_name="ANN_BEST_TRIAL"
    ):
        mlflow.log_param(
            "model",
            "ANN"
        )
        metrics = {'best_val_accuracy': best_trial.value}

        log_ann_trial(params = best_trial.params, metrics = metrics, trial_number = best_trial.number)

        mlflow.set_tag(
            "run_type",
            "best_trial"
        )

    return study