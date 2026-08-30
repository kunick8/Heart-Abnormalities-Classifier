import joblib

from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.data.data_preprocessing import DataPreprocessing
from src.evaluation_metrics import evaluate_model, save_and_log_charts
from src.mlflow_config import get_converted_params
from src.model import Classifier


dataset1 = extract_data('../data/raw/SPECTF.test')
dataset2 = extract_data('../data/raw/SPECTF.train')

dataset = join_data(dataset1, dataset2)


Preprocessor = DataPreprocessing(dataset)

X_train, X_test, y_train, y_test, scaler, selector = Preprocessor.get_preprocessed_data()

params = get_converted_params(tracking_uri="http://localhost:5000", experiment_name='SVC_optimization')
classifier = Classifier('SVC')

model = classifier.create_model(params)

model.fit(X_train, y_train)

joblib.dump(model, 'artifacts/model/svc.joblib')
joblib.dump(scaler, 'artifacts/model/scaler.joblib')
joblib.dump(selector, 'artifacts/model/selector.joblib')

y_pred = model.predict(X_test)

metrics = evaluate_model(y_test, y_pred)

save_and_log_charts(X_test, y_test, y_pred, model)

