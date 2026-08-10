import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('n_estimators', 2, 32)
    min_samples_leaf = trial.suggest_int('n_estimators', 1, 32)
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    score = cross_val_score(classifier, )

study = optuna.create_study(direction='maximize', )