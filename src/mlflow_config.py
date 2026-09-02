import mlflow
from mlflow import MlflowClient


def setup_mlflow(experiment_name):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

def log_trial(model_name, params, trial_number, metrics):
    mlflow.log_param('model', model_name)
    mlflow.log_param('trial_number', trial_number)

    for parameter, value in params.items():
        mlflow.log_param(parameter, value)
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)

def log_ann_trial( params, trial_number, metrics):
    mlflow.log_param('trial_number', trial_number)
    for parameter, value in params.items():
        mlflow.log_param(parameter, value)
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)

def get_best_run(tracking_uri, experiment_name):

    client = MlflowClient(
        tracking_uri=tracking_uri
    )

    experiment = client.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found"
        )

    if experiment_name == 'ANN_keras_optimization':
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.best_accuracy DESC"],
            max_results=1
        )
    else:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.mean_cv_f1 DESC"],
            max_results=1
        )

    if not runs:
        raise ValueError(f'Experiment "{experiment_name}" contains no runs')

    best_run = runs[0]

    return best_run

def get_best_params(
    tracking_uri,
    experiment_name
):

    best_run = get_best_run(
        tracking_uri,
        experiment_name
    )

    params = best_run.data.params

    return params

def get_converted_params(tracking_uri, experiment_name):

    params = get_best_params(tracking_uri, experiment_name)
    params = params.copy()

    if experiment_name == 'LogisticRegression_optimization':
        params['max_iter'] = int(params['max_iter'])
        params['C'] = float(params['C'])
        params['random_state'] = int(params['random_state'])


    elif experiment_name == 'RandomForestClassifier_optimization':
        params["n_estimators"] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['random_state'] = int(params['random_state'])

    elif experiment_name == 'NaiveBayes_optimization':
        params['var_smoothing'] = float(params['var_smoothing'])

    elif experiment_name == 'XGBClassifier_optimization':
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['scale_pos_weight'] = float(params['scale_pos_weight'])
        params['learning_rate'] = float(params['learning_rate'])
        params['random_state'] = int(params['random_state'])

    elif experiment_name == 'SVC_optimization':
        params['C'] = float(params['C'])
        params['gamma'] = float(params['gamma'])
        params['random_state'] = int(params['random_state'])

    elif experiment_name == 'ANN_keras_optimization':
        params['n_layers'] = int(params['n_layers'])
        params['learning_rate'] = float(params['learning_rate'])
        params['batch_size'] = int(params['batch_size'])
        for i in range(params['n_layers']):
            params[f'{i}_layer_neurons'] = int(params[f'{i}_layer_neurons'])

    else:
        raise ValueError(
            f"Experiment '{experiment_name}' not found"
        )





    params.pop('model', None)
    params.pop('trial_number', None)


    return params

def get_best_metrics(
    tracking_uri, experiment_name):

    best_run = get_best_run(
        tracking_uri,
        experiment_name
    )

    metrics = best_run.data.metrics

    return metrics


def get_best_score_per_model(tracking_uri, experiment_names:set):
    score_per_model = {}
    for experiment in experiment_names:
        metrics = get_best_metrics(tracking_uri, experiment)
        model= experiment[:-13]
        score_per_model[model] = float(metrics['mean_cv_f1'])
    return score_per_model
