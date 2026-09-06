import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.data.data_preprocessing import get_preprocessed_data_non_linear, get_preprocessed_data
from src.mlflow_functions import  get_converted_params
from src.model import Classifier, NeuralNetwork
from sklearn.metrics import accuracy_score, f1_score


def compare_trained_models(dataset, models:list):


    f1_scores = {}
    accuracy_scores ={}
    best_f1_score = 0
    for model in models:
        params, smote_ratio = get_converted_params("http://localhost:5000", f'{model}_optimization', )
        classifier = Classifier(model)

        if model in ["RandomForestClassifier", "XGBClassifier"]:
            X_train, X_test, y_train, y_test, _, _, _ = get_preprocessed_data_non_linear(dataset, smote_ratio = smote_ratio)
        else:
            X_train, X_test, y_train, y_test, _, _, _ = get_preprocessed_data(dataset, smote_ratio = smote_ratio)


        predictor = classifier.create_model(params)
        predictor.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)

        model_f1_score = f1_score(y_test, y_pred, average='macro')
        f1_scores[model] = model_f1_score
        accuracy_scores[model] = accuracy_score(y_test, y_pred)

        if model_f1_score > best_f1_score:
            best_f1_score = model_f1_score


    params, smote_ratio = get_converted_params("http://localhost:5000",'ANN_optimization')
    X_train, X_test, y_train, y_test, _, _, _ = get_preprocessed_data_non_linear(dataset, params['0_layer_neurons'], smote_ratio = smote_ratio)
    if params['use_class_weight']:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
    else:
        class_weight_dict = None


    network = NeuralNetwork(X_train.shape[1])

    for i in range(params['n_layers']):
        network.add_dense_layer(units=params[f'{i}_layer_neurons'], activation=params[f'{i}_layer_activation'])


    network.add_dense_layer(units = 1, activation= 'sigmoid')

    optimizer = params['optimizer']
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])

    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    compile_params = {'optimizer': optimizer}
    network.compile(compile_params)


    network.fit(X_train, y_train, epochs=50, batch_size=params['batch_size'], class_weight=class_weight_dict)
    y_pred = network.predict(X_test)

    y_pred = (y_pred >= 0.5).astype(int).ravel()

    ann_f1_score = f1_score(y_test, y_pred, average='macro')
    f1_scores["ANN"] = ann_f1_score
    accuracy_scores["ANN"] = accuracy_score(y_test, y_pred)


    return f1_scores, accuracy_scores




