from src.tuning.ann_tuning import tune_ann
from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.data.data_preprocessing import get_optimization_data

dataset1 = extract_data('../../data/raw/SPECTF.test')
dataset2 = extract_data('../../data/raw/SPECTF.train')
dataset = join_data(dataset1, dataset2)


def evaluate_ann(dataset):
    X_train, X_test, y_train, y_test, _, _ = get_optimization_data(dataset)


    tune_ann(X_train, y_train)