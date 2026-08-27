from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf

class Classifier:
    def __init__(self, model_name):
        self.model_name = model_name

    def create_model(self, params):
        if self.model_name == 'LogisticRegression':
            classifier = LogisticRegression(**params)

        elif self.model_name == 'RandomForestClassifier':
            classifier = RandomForestClassifier(**params)

        elif self.model_name == 'NaiveBayes':
            classifier = GaussianNB(**params)

        elif self.model_name == 'XGBClassifier':
            classifier = XGBClassifier(**params)

        elif self.model_name == 'SVC':
            classifier = SVC(**params)

        else:
            raise ValueError('Invalid model name')
        return classifier

    @staticmethod
    def create_best_model():
        return SVC(C = 16.16798771326778, kernel = 'rbf', gamma = 1.9741165665216929)


class NeuralNetwork:
    def __init__(self):
        self.model = tf.keras.models.Sequential()

    def add_dense_layer(self, units:int, activation:str):
        self.model.add(tf.keras.layers.Dense(units = units, activation=activation))

    def compile(self, params:dict):
        self.model.compile(optimizer = params['optimizer'], loss = 'binary_crossentropy', metrics = ['accuracy'])

    def fit(self, X_train, y_train, validation_data, epochs, batch_size, callbacks = None):
        return self.model.fit(X_train, y_train, validation_data = validation_data, epochs = epochs, batch_size = batch_size, callbacks = callbacks, verbose = 0)

    def get_model(self):
        return self.model