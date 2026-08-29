import optuna


def plot_optimization_history(study):

    return optuna.visualization.plot_optimization_history(
        study
    )


def plot_param_importances(study):

    return optuna.visualization.plot_param_importances(
        study
    )


def plot_parallel_coordinate(study):

    return optuna.visualization.plot_parallel_coordinate(
        study
    )