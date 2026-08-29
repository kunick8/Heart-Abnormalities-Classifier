from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from mlflow import MlflowClient

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


    def create_best_model(self, tracking_uri):

        client = MlflowClient(tracking_uri=tracking_uri)
        experiment_name = f'{self.model_name}_optimization'
        experiment = client.get_experiment_by_name(experiment_name)

        best_run = client.search_runs(
            experiment_ids = [experiment.experiment_id],
            order_by = ['metrics.mean_cv_f1 DESC'],
            max_results=1
        )[0]
        params = best_run.data.params

        if params['model'] == 'LogisticRegression':
            params['max_iter'] = int(params['max_iter'])
            params['C'] = float(params['C'])
            params['random_state'] = int(params['random_state'])

        elif params['model'] == 'RandomForestClassifier':
            params["n_estimators"] = int(params['n_estimators'])
            params['max_depth'] = int(params['max_depth'])
            params['min_samples_split'] = int(params['min_samples_split'])

        elif params['model'] == 'NaiveBayes':
            params['var_smoothing'] = float(params['var_smoothing'])

        elif params['model'] == 'XGBClassifier':
            params['n_estimators'] = int(params['n_estimators'])
            params['max_depth'] = int(params['max_depth'])
            params['min_samples_split'] = float(params['min_samples_split'])
            params['scale_pos_weight'] = float(params['scale_pos_weight'])
            params['learning_rate'] = float(params['learning_rate'])

        elif params['model'] == 'SVC':
            params['C'] = float(params['C'])
            params['gamma'] = float(params['gamma'])
            params['random_state'] = int(params['random_state'])

        return self.create_model(self, params=params)


class NeuralNetwork:
    def __init__(self, input_dim):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(input_dim,)))

    def add_dense_layer(self, units:int, activation:str):
        self.model.add(tf.keras.layers.Dense(units = units, activation=activation))

    def compile(self, params:dict):
        self.model.compile(optimizer = params['optimizer'], loss = 'binary_crossentropy', metrics = ['accuracy'])

    def fit(self, X_train, y_train, validation_data, epochs, batch_size, callbacks = None):
        return self.model.fit(X_train, y_train, validation_data = validation_data, epochs = epochs, batch_size = batch_size, callbacks = callbacks, verbose = 0)

    def get_model(self):
        return self.model