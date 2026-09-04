from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.data.data_preprocessing import get_optimization_data
from src.tune import tune_model

dataset1 = extract_data('../data/raw/SPECTF.test')
dataset2 = extract_data('../data/raw/SPECTF.train')
dataset = join_data(dataset1, dataset2)
models = ["LogisticRegression", "RandomForestClassifier", "NaiveBayes", "XGBClassifier", "SVC"]


X_train, _, y_train, _ = get_optimization_data(dataset)

for model in models:
    tune_model(model_name=model, X_train = X_train, y_train = y_train)