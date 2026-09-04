import tensorflow as tf

from src.data.data_cleaning import join_data
from src.data.data_extractor import extract_data
from src.data.data_preprocessing import get_preprocessed_data_non_linear, get_preprocessed_data
from src.mlflow_config import  get_converted_params
from src.model import Classifier, NeuralNetwork
from sklearn.metrics import accuracy_score, f1_score

dataset1 = extract_data('../data/raw/SPECTF.test')
dataset2 = extract_data('../data/raw/SPECTF.train')
dataset = join_data(dataset1, dataset2)



def compare_trained_models(dataset, models:list):


    f1_scores = {}
    accuracy_scores ={}
    best_model = None
    best_f1_score = 0
    for model in models:
        params = get_converted_params("http://localhost:5000", f'{model}_optimization', )
        classifier = Classifier(model)

        if model in ["RandomForestClassifier", "XGBClassifier"]:
            X_train, X_test, y_train, y_test, _, _ = get_preprocessed_data_non_linear(dataset)
        else:
            X_train, X_test, y_train, y_test, _, _ = get_preprocessed_data(dataset)


        predictor = classifier.create_model(params)
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)

        model_f1_score = f1_score(y_test, y_pred)
        f1_scores[model] = model_f1_score
        accuracy_scores[model] = accuracy_score(y_test, y_pred)

        if model_f1_score > best_f1_score:
            best_f1_score = model_f1_score
            best_model = predictor


    params = get_converted_params("http://localhost:5000",'ANN_keras_optimization')
    X_train, X_test, y_train, y_test, _, _ = get_preprocessed_data_non_linear(dataset, params['0_layer_neurons'])
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

    y_pred = (y_pred >= 0.5).astype(int).ravel()

    ann_f1_score = f1_score(y_test, y_pred)
    f1_scores["ANN"] = ann_f1_score
    accuracy_scores["ANN"] = accuracy_score(y_test, y_pred)

    if ann_f1_score > best_f1_score:
        best_model = network

    return f1_scores, accuracy_scores, best_model




