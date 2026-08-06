from sklearn.ensemble import RandomForestRegressor

from data_preprocessing import DataPreprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


first_data_batch = pd.read_csv('dataset/SPECTF.test')
second_data_batch = pd.read_csv('dataset/SPECTF.train')
dataset = pd.DataFrame.join(first_data_batch, second_data_batch)


Data = DataPreprocessing(dataset)
X, y = Data.data_splitter()

#train test split + feature standarisation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test = Data.feature_scailing(X_train, X_test)

#feature_selection
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
rfe = RFECV(estimator=LogisticRegression(random_state=42, max_iter=1000), n_jobs=-1)
X_train = rfe.fit_transform(X_train, y_train)
X_test = rfe.transform(X_test)


regressor = LogisticRegression(random_state=42).fit(X_train, y_train)

