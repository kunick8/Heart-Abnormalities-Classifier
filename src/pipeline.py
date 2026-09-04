import tensorflow as tf
import json
from pathlib import Path

from src.data import data_cleaning
from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.model_comparison import compare_trained_models
from src.tune import tune_model
from src.ann_tuning import tune_ann
from src.data.data_preprocessing import DataPreprocessing

project_root = Path(__file__).parent.parent
scores_dir = project_root / "artifacts" / "scores"
scores_dir.mkdir(parents=True, exist_ok=True)

dataset1 = extract_data('../data/raw/SPECTF.test')
dataset2 = extract_data('../data/raw/SPECTF.train')
dataset = join_data(dataset1, dataset2)

preprocessor = DataPreprocessing()
X_train, X_test, y_train, y_test, _, _ = preprocessor.get_preprocessed_data_non_linear()

f1_scores, accuracy_scores = compare_trained_models(dataset)



with open(scores_dir / "f1_scores.json", "w") as f:
    json.dump(f1_scores, f)

with open(scores_dir / "accuracy_scores.json", "w") as f:
    json.dump(accuracy_scores, f)


