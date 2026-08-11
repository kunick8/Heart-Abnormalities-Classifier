import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression


class DataPreprocessing:

    def __init__(self, dataset:pd.DataFrame):
        self.dataset = dataset

    def data_splitter(self):
        X = self.dataset.iloc[:, 1:].values
        y = self.dataset.iloc[:, 0].values
        return X, y

    def training_test_split(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=0.2,
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

