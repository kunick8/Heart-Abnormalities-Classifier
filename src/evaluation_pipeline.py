import json
from pathlib import Path

from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.model_comparison import compare_trained_models
from src.tuning.model_evaluation import evaluate_models
from src.tuning.ann_evaluation import evaluate_ann


project_root = Path(__file__).parent.parent
scores_dir = project_root / "artifacts" / "scores"
scores_dir.mkdir(parents=True, exist_ok=True)



dataset1 = extract_data('../data/raw/SPECTF.test')
dataset2 = extract_data('../data/raw/SPECTF.train')
dataset = join_data(dataset1, dataset2)

models =["LogisticRegression", "RandomForestClassifier", "NaiveBayes", "XGBClassifier", "SVC"]

evaluate_models(models=models, dataset=dataset)
evaluate_ann(dataset)


f1_scores, accuracy_scores = compare_trained_models(dataset,models =models)



with open(scores_dir / "f1_scores.json", "w") as f:
    json.dump(f1_scores, f)

with open(scores_dir / "accuracy_scores.json", "w") as f:
    json.dump(accuracy_scores, f)

print(f1_scores)
print(accuracy_scores)
