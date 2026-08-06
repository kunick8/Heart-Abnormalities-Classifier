import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessing:

    def __init__(self, dataset:pd.DataFrame):
        self.dataset = dataset
        self.X, self.y = self.data_splitter()

    def data_splitter(self):
        X = self.dataset.iloc[:, 1:].values
        y = self.dataset.iloc[:, 0].values
        return X,y

    def feature_scailing(self, X_train: np.ndarray, X_test: np.ndarray):
        scaler = StandardScaler()
        scaler.fit_transform(X_train)
        scaler.transform(X_test)
        return X_train, X_test, scaler