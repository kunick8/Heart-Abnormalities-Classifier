import mlflow
import optuna
import tensorflow as tf
from optuna_integration import TFKerasPruningCallback

from src.mlflow_config import log_ann_trial, setup_mlflow
from src.model import NeuralNetwork

def objective(trial, X_train, X_val, y_train, y_val):

    tf.keras.backend.clear_session()


    with mlflow.start_run(
        run_name=f"ANN_trial_{trial.number}"
    ):

        n_layers = trial.suggest_int("n_layers", 1, 5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log = True)

        network = NeuralNetwork(input_dim = X_train.shape[1])

        layer_params = {}

        for i in range(n_layers):
            if i != 0:
                num_hidden = trial.suggest_int(f"{i}_layer_neurons", 6, 128)
                activation = trial.suggest_categorical(f"{i}_layer_activation", ["relu", "tanh"])
            else:
                num_hidden = X_train.shape[1]
                activation = trial.suggest_categorical(f"{i}_layer_activation", ["relu", "tanh",])
            layer_params[f"{i}_layer_activation"] = activation
            layer_params[f"{i}_layer_neurons"] = num_hidden

            network.add_dense_layer(units = num_hidden, activation=activation)


        network.add_dense_layer(units = 1, activation= 'sigmoid')

        optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam"])
        batch_size = trial.suggest_categorical(
            "batch_size",
            [16, 32, 64, 128]
        )
        params = {'n_layers': n_layers, 'learning_rate': learning_rate, 'batch_size': batch_size, 'optimizer': optimizer}
        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        compile_params = {'optimizer':optimizer}
        network.compile(compile_params)

        callbacks = [

            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            ),

            TFKerasPruningCallback(
                trial,
                "val_accuracy"
            )
        ]

        history = network.fit(
        X_train,
        y_train,
        validation_data=(X_val,y_val),
        batch_size=batch_size,
        callbacks=callbacks,
        epochs=100
        )
        best_val_accuracy = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        metrics ={'best_val_accuracy': best_val_accuracy, 'best_val_loss': best_val_loss}

        params.update(layer_params)

        log_ann_trial(params = params, metrics = metrics, trial_number=trial.number)


    return best_val_accuracy

def tune_ann(
    X_train,
    X_val,
    y_train,
    y_val,
    n_trials=100
):
    setup_mlflow(experiment_name="ANN_optimization_100epochs")
    study = optuna.create_study(direction="maximize", study_name="ANN_optimization")

    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), n_trials=n_trials)

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