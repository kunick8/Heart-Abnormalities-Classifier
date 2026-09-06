import joblib
from pathlib import Path

from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.data.data_preprocessing import get_preprocessed_data_non_linear
from src.evaluation_metrics import evaluate_model, save_and_log_charts
from src.mlflow_functions import get_converted_params
from src.model import Classifier


current_dir = Path(__file__).parent
model_dir = current_dir.parent / "artifacts" / "model"
model_dir.mkdir(parents=True, exist_ok=True)

dataset1 = extract_data('../data/raw/SPECTF.test')
dataset2 = extract_data('../data/raw/SPECTF.train')

dataset = join_data(dataset1, dataset2)


params, smote_ratio = get_converted_params(tracking_uri="http://localhost:5000", experiment_name='RandomForestClassifier_optimization')
classifier = Classifier('RandomForestClassifier')

X_train, X_test, y_train, y_test, scaler, _, selector = get_preprocessed_data_non_linear(dataset, smote_ratio=smote_ratio)



model = classifier.create_model(params)

model.fit(X_train, y_train)

joblib.dump(model, model_dir / "rfc.joblib")
joblib.dump(scaler, model_dir / "scaler.joblib")
joblib.dump(selector, model_dir / "selector.joblib")

y_pred = model.predict(X_test)
scores = model.predict_proba(X_test)[:, 1]

metrics = evaluate_model(y_test, y_pred)

save_and_log_charts(y_test, y_pred, scores)

