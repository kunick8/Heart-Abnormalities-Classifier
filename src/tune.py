import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model import Classifier

def objective(trial, X_train, y_train, model_name):
    if model_name == "LogisticRegression":
        params = {}
        solver = trial.suggest_categorical("solver", [ "lbfgs", "liblinear"])
        params['C'] = trial.suggest_float("C", 1e-3, 100, log=True)
        params['max_iter'] = 1000
        params['random_state'] = 42
        params['class_weight'] = 'balanced'
        params["solver"] = solver
        if solver == "liblinear":
            params['penalty'] = trial.suggest_categorical("penalty_liblinear", [ "l1", "l2"])
        else:
            params['penalty'] = trial.suggest_categorical("penalty_lbfgs", ['l2', "none"])

    elif model_name == "RandomForestClassifier":
        params = {"n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                  'max_depth': trial.suggest_int("max_depth", 10, 100),
                  'min_samples_split': trial.suggest_int("min_samples_split", 2, 5),
                  'criterion': trial.suggest_categorical("criterion",['gini','entropy']),
                  'class_weight': 'balanced',
                  'random_state': 42,
                  }

    elif model_name == "NaiveBayes":
        params = {'var_smoothing': trial.suggest_float("var_smoothing", 1e-9, 1e-3, log=True),
                  'random_state': 42,}


    elif model_name == "XGBClassifier":
        params = {'learning_rate' : 0.1,
                  'n_estimators': trial.suggest_int("n_estimators", 100, 200),
                  'max_depth': trial.suggest_int("max_depth", 3, 7),
                  'min_child_weight': trial.suggest_int("min_child_weight", 1, 5),
                  'scale_pos_weight': y_train.value_counts()[0]/y_train.value_counts()[1],
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
    model = classifier.create_model(**params)

    if model_name in [
        "LogisticRegression",
        "NaiveBayes",
        "SVC"
    ]:

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model)
        ])

    else:

        pipeline = Pipeline([
            ("classifier", model)
        ])

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

    return scores.mean()


def tune_model(X_train, y_train, model_name, n_trials=100):

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