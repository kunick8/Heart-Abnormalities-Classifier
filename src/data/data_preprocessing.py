import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler





def data_splitter(dataset):
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values
    return X, y

def training_test_split(X, y, test_size=0.3):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

def feature_scaling(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler

def feature_selection(X_train, X_test, y_train):
    selector = RFECV(
        estimator=LogisticRegression(
            random_state=42,
            max_iter=1000
        ),
        n_jobs=-1
    )

    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    return X_train, X_test, selector

def ann_feature_selection(X_train, X_test,  y_train, n_features = None):

    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)

    if n_features is None:
        selector = SelectFromModel(xgb_model, prefit=True)
    else:
        selector = SelectFromModel(xgb_model, prefit=True,
                                    threshold=-np.inf,
                                    max_features=n_features
                                    )

    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, selector


def get_preprocessed_data(dataset):
    X, y = data_splitter(dataset)
    X_train, X_test, y_train, y_test = training_test_split(X, y)
    X_train, X_test, scaler = feature_scaling(X_train, X_test)
    X_train, X_test, selector = feature_selection(X_train, X_test, y_train)
    return (X_train,
            X_test,
            y_train,
            y_test,
            scaler,
            selector)

def get_preprocessed_data_non_linear (dataset, n_features:int = None):
    X, y = data_splitter(dataset)
    X_train, X_test, y_train, y_test = training_test_split(X, y)
    X_train, X_test, scaler = feature_scaling(X_train, X_test)
    X_train, X_test, selector = ann_feature_selection(X_train, X_test, y_train, n_features)
    return (X_train,
            X_test,
            y_train,
            y_test,
            scaler,
            selector)



    #data for ml models optuna optimizations, it's without feature selection or feature scaling
    #to prevent data leakage(optuna pipelines use k-fold for optimization)
def get_optimization_data(dataset):
    X, y = data_splitter(dataset)
    X_train, X_test, y_train, y_test = training_test_split(X, y)
    return (X_train,
            X_test,
            y_train,
            y_test,)
