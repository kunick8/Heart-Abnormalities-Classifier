import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.tuning.ann_tuning import tune_ann
from src.data.data_preprocessing import get_optimization_data




def evaluate_ann(dataset):
    X_train, X_test, y_train, y_test = get_optimization_data(dataset)


    tune_ann(X_train, y_train)