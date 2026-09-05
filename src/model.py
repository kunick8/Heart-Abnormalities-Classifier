import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier


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




class NeuralNetwork:
    def __init__(self, input_dim):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(input_dim,)))

    def add_dense_layer(self, units:int, activation:str):
        self.model.add(tf.keras.layers.Dense(units = units, activation=activation))

    def compile(self, params:dict):
        self.model.compile(optimizer = params['optimizer'], loss = 'binary_crossentropy', metrics = ['accuracy'])

    def fit(self, X_train, y_train, epochs, batch_size, validation_data= None, callbacks = None, class_weight = None):
        return self.model.fit(X_train, y_train, validation_data = validation_data, epochs = epochs,
                              batch_size = batch_size, callbacks = callbacks, verbose = 0, class_weight = class_weight)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_model(self):
        return self.model