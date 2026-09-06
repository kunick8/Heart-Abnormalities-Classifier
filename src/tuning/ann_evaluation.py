import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.data.data_preprocessing import get_optimization_data
from src.tuning.ann_tuning import tune_ann


def evaluate_ann(dataset):
    X_train, _, y_train, _ = get_optimization_data(dataset)


    tune_ann(X_train, y_train)