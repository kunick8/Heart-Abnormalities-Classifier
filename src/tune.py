import mlflow
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from mlflow_config import log_trial, setup_mlflow
from model import Classifier


def objective(trial, X_train, y_train, model_name):
    if model_name == "LogisticRegression":
        params = {}
        solver = trial.suggest_categorical("solver", [ "lbfgs", "liblinear"])
        params['C'] = trial.suggest_float("C", 1e-3, 100, log=True)
        params['max_iter'] = 2000
        params['random_state'] = 42
        params['class_weight'] = 'balanced'
        params["solver"] = solver
        if solver == "liblinear":
            params['penalty'] = trial.suggest_categorical("penalty_liblinear", [ "l1", "l2"])
        else:
            params['penalty'] = trial.suggest_categorical("penalty_lbfgs", ['l2', None])

    elif model_name == "RandomForestClassifier":
        params = {"n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                  'max_depth': trial.suggest_int("max_depth", 10, 100),
                  'min_samples_split': trial.suggest_int("min_samples_split", 2, 5),
                  'criterion': trial.suggest_categorical("criterion",['gini','entropy']),
                  'class_weight': 'balanced',
                  'random_state': 42,
                  }

    elif model_name == "NaiveBayes":
        params = {'var_smoothing': trial.suggest_float("var_smoothing", 1e-9, 1e-3, log=True)}


    elif model_name == "XGBClassifier":
        params = {'learning_rate' : 0.1,
                  'n_estimators': trial.suggest_int("n_estimators", 100, 200),
                  'max_depth': trial.suggest_int("max_depth", 3, 7),
                  'min_child_weight': trial.suggest_int("min_child_weight", 1, 5),
                  'scale_pos_weight': (np.count_nonzero(y_train == 0))/(np.count_nonzero(y_train == 1)),
                  'random_state': 42,
                  }
    elif model_name == 'SVC':
        params = {'C': trial.suggest_float("C", 1e-3, 50, log=True),
                  'kernel': trial.suggest_categorical("kernel",['linear','rbf']),
                  'gamma': trial.suggest_float("gamma", 1e-3, 5, log=True),
                  'class_weight': 'balanced',
                  'random_state': 42,
                  }
    else:
        raise ValueError("Invalid model name")

    classifier = Classifier(model_name)
    model = classifier.create_model(params)

    if model_name in ["RandomForestClassifier", "XGBClassifier"]:
        selector_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        needs_scaling = False
    else:
        selector_estimator = LogisticRegression(random_state=42, max_iter=1000)
        needs_scaling = True


    threshold = trial.suggest_categorical("feature_selection_threshold", ["mean", "median", "1.25*mean"])
    selector = SelectFromModel(estimator=selector_estimator, threshold=threshold)

    steps = []
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))

    steps.append(("selector", selector))
    steps.append(("classifier", model))

    pipeline = Pipeline(steps)
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    f1_mean = scores.mean()
    metrics = {'mean_cv_f1': f1_mean, 'mean_std_f1': scores.std()}

    with mlflow.start_run(
            run_name=f"{model_name}_trial_{trial.number}"
    ):

        log_trial(model_name, params, trial.number, metrics)

    return f1_mean


def tune_model(X_train, y_train, model_name, n_trials=100):

    experiment_name = f'{model_name}_optimization'

    setup_mlflow(experiment_name)

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            model_name
        ),
        n_trials=n_trials
    )

    return study


