from src.ann_tuning import tune_ann
from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.data.data_preprocessing import DataPreprocessing

dataset1 = extract_data('../data/raw/SPECTF.test')
dataset2 = extract_data('../data/raw/SPECTF.train')
dataset = join_data(dataset1, dataset2)

preprocessor = DataPreprocessing(dataset)
X_train, X_test, y_train, y_test, _, _ = preprocessor.get_preprocessed_data()


tune_ann(X_train, y_train)