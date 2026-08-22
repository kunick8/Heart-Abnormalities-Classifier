from src.data.data_preprocessing import DataPreprocessing
from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.tune import tune_model

dataset1 = extract_data('../data/raw/SPECTF.test')
dataset2 = extract_data('../data/raw/SPECTF.train')
dataset = join_data(dataset1, dataset2)
models = ["LogisticRegression", "RandomForestClassifier", "NaiveBayes", "XGBClassifier", "SVC"]


Preprocessor = DataPreprocessing(dataset)
preprocessed_data = Preprocessor.get_preprocessed_not_scaled_data()


tune_model(model_name="SVC", X_train = preprocessed_data[0], y_train = preprocessed_data[2])