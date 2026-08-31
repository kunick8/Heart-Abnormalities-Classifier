from src.ann_tuning import tune_ann
from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.data.data_preprocessing import DataPreprocessing

dataset1 = extract_data('../data/raw/SPECTF.test')
dataset2 = extract_data('../data/raw/SPECTF.train')
dataset = join_data(dataset1, dataset2)

Preprocessor = DataPreprocessing(dataset)
X_train, _, X_val, y_train, _, y_val, _, _ = Preprocessor.get_ann_optimization_data()


tune_ann(X_train, X_val, y_train, y_val)