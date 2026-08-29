import mlflow
from mlflow import MlflowClient


def setup_mlflow(experiment_name):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

def log_trial(model_name, params, trial_number, score, score_std):
    mlflow.log_param('model', model_name)
    mlflow.log_param('trial_number', trial_number)
    for parameter, value in params.items():
        mlflow.log_param(parameter, value)
    mlflow.log_metric('mean_cv_f1', score)
    mlflow.log_metric('mean_std_f1', score_std)

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

    if params['model'] == 'LogisticRegression':
        params['max_iter'] = int(params['max_iter'])
        params['C'] = float(params['C'])
        params['random_state'] = int(params['random_state'])

    elif params['model'] == 'RandomForestClassifier':
        params["n_estimators"] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])

    elif params['model'] == 'NaiveBayes':
        params['var_smoothing'] = float(params['var_smoothing'])

    elif params['model'] == 'XGBClassifier':
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['scale_pos_weight'] = float(params['scale_pos_weight'])
        params['learning_rate'] = float(params['learning_rate'])

    elif params['model'] == 'SVC':
        params['C'] = float(params['C'])
        params['gamma'] = float(params['gamma'])

    params.pop('model', None)

    return params