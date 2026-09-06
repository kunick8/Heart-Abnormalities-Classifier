import os
import sys

import mlflow
import numpy as np
import optuna
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.data.data_preprocessing import ann_feature_selection, feature_scaling
from src.mlflow_functions import log_ann_trial, setup_mlflow
from src.model import NeuralNetwork


def objective(trial, X_train, y_train):

    n_layers = trial.suggest_int("n_layers", 1, 5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["sgd", "adam"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    smote_ratio = trial.suggest_float("smote_ratio", 0.3, 1.0)
    use_class_weight = trial.suggest_categorical("use_class_weight", [True, False])

    layer_params = {}
    for i in range(n_layers):

        num_hidden = trial.suggest_int(f"{i}_layer_neurons", 6, 128)
        activation = trial.suggest_categorical(f"{i}_layer_activation", ["relu", "tanh"])
        layer_params[f"{i}_layer_activation"] = activation
        layer_params[f"{i}_layer_neurons"] = num_hidden

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_f1scores = []
    fold_losses = []

    with mlflow.start_run(run_name=f"ANN_trial_{trial.number}"):

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):

            tf.keras.backend.clear_session()

            X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
            X_fold_val, y_fold_val = X_train[val_idx], y_train[val_idx]


            X_fold_train, X_fold_val, _ = feature_scaling(X_fold_train, X_fold_val)

            smote = SMOTE(random_state=42, sampling_strategy=smote_ratio)
            X_fold_train, y_fold_train = smote.fit_resample(X_fold_train, y_fold_train)

            X_fold_train, X_fold_val, _ = ann_feature_selection(X_fold_train, X_fold_val, y_fold_train)


            if use_class_weight:
                classes = np.unique(y_fold_train)
                weights = compute_class_weight(class_weight='balanced' ,classes = classes, y = y_fold_train)
                class_weight_dict = dict(zip(classes, weights))
            else:
                class_weight_dict = None

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
                epochs=100,
                class_weight=class_weight_dict
            )




            val_probs = network.predict(X_fold_val).ravel()
            val_preds = (val_probs >= 0.5).astype(int)
            fold_f1 = f1_score(y_fold_val, val_preds, average='macro')
            fold_f1scores.append(fold_f1)
            fold_losses.append(min(history.history['val_loss']))

        mean_f1_score = np.mean(fold_f1scores)
        mean_loss =np.mean(fold_losses)

        params = {
            'n_layers': n_layers,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'optimizer': optimizer_name,
            'smote_ratio': smote_ratio,
            'use_class_weight': use_class_weight,

        }
        params.update(layer_params)
        metrics = {'mean_f1_score': mean_f1_score, 'mean_loss': mean_loss}

        log_ann_trial(params=params, metrics=metrics, trial_number=trial.number)

    return mean_f1_score

def tune_ann(
    X_train,
    y_train,
    n_trials=100
):
    setup_mlflow(experiment_name="ANN_optimization")
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