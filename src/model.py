from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

class Classifier:
    def __init__(self, model_name):
        self.model_name = model_name

    def create_model(self, params):
        if self.model_name == 'LogisticRegression':
            classifier = LogisticRegression(**params)

        elif self.model_name == 'RandomForest':
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
