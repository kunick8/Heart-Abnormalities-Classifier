import mlflow

def setup_mlflow(experiment_name):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

def log_trial(model_name, params, trial_number, score, score_std):
    mlflow.log_param('model', model_name)
    mlflow.log_param('params', trial_number)
    for parameter, value in params.items():
        mlflow.log_param(parameter, value)
    mlflow.log_metric('mean_cv_f1', score)
    mlflow.log_metric('mean_std_f1', score_std)
