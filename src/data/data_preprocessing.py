import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessing:

    def __init__(self, dataset:pd.DataFrame = None, X_train = None, y_train = None, X_test = None, y_test = None):
        if dataset is not None:
            self.dataset = dataset
        if X_train:
            self.X_train = X_train
        if y_train:
            self.y_train = y_train
        if X_test:
            self.X_test = X_test
        if y_test:
            self.y_test = y_test


    def data_splitter(self):
        X = self.dataset.iloc[:, 1:].values
        y = self.dataset.iloc[:, 0].values
        return X, y

    def training_test_split(self, X, y, test_size=0.3):
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

    def feature_scaling(self, X_train, X_test):
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, scaler

    def feature_selection(self, X_train, X_test, y_train):
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

    def ann_feature_selection(self, X_train, X_test,  y_train):

        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)

        selector = SelectFromModel(xgb_model, prefit=True)

        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)

        return X_train_selected, X_test_selected, selector


    def get_preprocessed_data(self):
        X, y = self.data_splitter()
        X_train, X_test, y_train, y_test = self.training_test_split(X, y)
        X_train, X_test, scaler = self.feature_scaling(X_train, X_test)
        X_train, X_test, selector = self.feature_selection(X_train, X_test, y_train)
        return (X_train,
            X_test,
            y_train,
            y_test,
            scaler,
            selector)

    #data for ml models optuna optimizations, it's without feature selection or feature scaling
    #to prevent data leakage(optuna pipelines use k-fold for optimization)
    def get_optimization_data(self):
        X, y = self.data_splitter()
        X_train, X_test, y_train, y_test = self.training_test_split(X, y)
        return (X_train,
                X_test,
                y_train,
                y_test,)
