import tensorflow as tf
import json
from pathlib import Path

from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.data.data_preprocessing import DataPreprocessing
from src.mlflow_config import  get_converted_params
from src.model import Classifier, NeuralNetwork
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

dataset1 = extract_data('../data/raw/SPECTF.test')
dataset2 = extract_data('../data/raw/SPECTF.train')
dataset = join_data(dataset1, dataset2)
models = ["LogisticRegression", "RandomForestClassifier", "NaiveBayes", "XGBClassifier", "SVC"]

preprocessor = DataPreprocessing(dataset)




f1_scores = {}
accuracy_scores ={}
for model in models:
    params = get_converted_params("http://localhost:5000", f'{model}_optimization', )
    classifier = Classifier(model)

    if model in ["RandomForestClassifier", "XGBClassifier"]:
        X_train, X_test, y_train, y_test, _, _ = preprocessor.get_preprocessed_data_non_linear()
    else:
        X_train, X_test, y_train, y_test, _, _ = preprocessor.get_preprocessed_data()


    predictor = classifier.create_model(params)
    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)
    f1_scores[model] = f1_score(y_test, y_pred)
    accuracy_scores[model] = accuracy_score(y_test, y_pred)



params = get_converted_params("http://localhost:5000",'ANN_keras_optimization')
X_train, X_test, y_train, y_test, _, _ = preprocessor.get_preprocessed_data_non_linear(params['0_layer_neurons'])
network = NeuralNetwork(X_train.shape[1])
for i in range(params['n_layers']):
    if i > 0:
        network.add_dense_layer(units=params[f'{i}_layer_neurons'], activation=params[f'{i}_layer_activation'])
    else:
        network.add_dense_layer(units=X_train.shape[1], activation=params[f'{i}_layer_activation'])

network.add_dense_layer(units = 1, activation= 'sigmoid')

optimizer = params['optimizer']
if optimizer == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])

else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

compile_params = {'optimizer': optimizer}
network.compile(compile_params)
network.fit(X_train, y_train, epochs=50, batch_size=params['batch_size'])
y_pred = network.predict(X_test)
for pred in y_pred:
    if pred >= 0.5:
        pred = 1
    else:
        pred = 0
y_pred = (y_pred >= 0.5).astype(int).ravel()

f1_scores["ANN"] = f1_score(y_test, y_pred)
accuracy_scores["ANN"] = accuracy_score(y_test, y_pred)


project_root = Path(__file__).parent.parent
scores_dir = project_root / "artifacts" / "scores"
scores_dir.mkdir(parents=True, exist_ok=True)

with open(scores_dir / "f1_scores.json", "w") as f:
    json.dump(f1_scores, f)

with open(scores_dir / "accuracy_scores.json", "w") as f:
    json.dump(f1_scores, f)

print(f1_scores)
print(accuracy_scores)





