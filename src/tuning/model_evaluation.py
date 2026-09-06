import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.data.data_preprocessing import get_optimization_data
from src.tuning.tune import tune_model


def evaluate_models(models, dataset):


    X_train, _, y_train, _ = get_optimization_data(dataset)

    for model in models:
        tune_model(model_name=model, X_train = X_train, y_train = y_train)