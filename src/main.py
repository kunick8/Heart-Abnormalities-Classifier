import joblib
from pathlib import Path

from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.data.data_preprocessing import get_preprocessed_data
from src.evaluation_metrics import evaluate_model, save_and_log_charts
from src.mlflow_config import get_converted_params
from src.model import Classifier


current_dir = Path(__file__).parent
model_dir = current_dir.parent / "artifacts" / "model"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "svc.joblib"

dataset1 = extract_data('../data/raw/SPECTF.test')
dataset2 = extract_data('../data/raw/SPECTF.train')

dataset = join_data(dataset1, dataset2)




X_train, X_test, y_train, y_test, scaler, selector = get_preprocessed_data(dataset)

params = get_converted_params(tracking_uri="http://localhost:5000", experiment_name='SVC_optimization')
classifier = Classifier('SVC')

model = classifier.create_model(params)

model.fit(X_train, y_train)

joblib.dump(model, model_dir / "svc.joblib")
joblib.dump(scaler, model_dir / "scaler.joblib")
joblib.dump(selector, model_dir / "selector.joblib")

y_pred = model.predict(X_test)

metrics = evaluate_model(y_test, y_pred)

save_and_log_charts(X_test, y_test, y_pred, model)

