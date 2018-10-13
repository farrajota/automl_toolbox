"""
The cross_validation module contains wrapper methods for performing
cross-validation on Pandas DataFrames.
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from automl_toolbox.model_selection import xgboost
from automl_toolbox.model_selection import lightgbm
from automl_toolbox.exceptions import raise_invalid_task_error, raise_invalid_model_backend_error


def cross_validation_score(df, target, task='classification', backend='lightgbm',
                           params=None, cv=5, n_jobs=None, device_type='cpu',
                           verbose=0):
    """Evaluates a Gradient Boosting Tree model by cross-validation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    target : str/list
        Target feature name(s).
    task : str, optional (default='classification')
        Name of the task.
    backend : str, optional (default='lightgbm')
        Name of the model's backend.

    Returns
    -------
    scores : pd.DataFrame
        DataFrame with cross-validation scores
    """
    assert target
    assert target in df, f"Target does not exist in the DataFrame: '{target}'"

    # parse strings to a default name
    task = parse_task_name_string(task)
    backend = parse_backend_name_string(backend)

    # parse parameters
    if params is None:
        params = {}
    params['device_type'] = device_type

    # Setup data
    X = df.drop(columns=target)
    X = pd.get_dummies(X, drop_first=True)
    y = df[target]

    # setup model
    model = get_model(task, backend, params)
    scoring = get_cv_scoring(task)

    if verbose > 0:
        print('Starting cross-validation...')
    scores = cross_val_score(estimator=model,
                             X=X,
                             y=y,
                             scoring=scoring,
                             cv=cv,
                             n_jobs=n_jobs,
                             verbose=verbose)
    if verbose > 0:
        print('\nCross-validation complete!')

    if task == 'classification':
        if verbose > 0:
            print(f'\nAccuracy (%): {scores.mean() * 100:.5f} +- {scores.std() * 100:.5f}')
    elif task == 'regression':
        scores_mean = np.sqrt(-scores).mean()
        scores_std = np.sqrt(-scores).std()
        if verbose > 0:
            print(f'\nScore (rmse): {scores_mean:.2f} +- {scores_std:.2f}')
    else:
        raise_invalid_task_error(task)

    df_scores = pd.DataFrame(data={'score': scores})
    df_scores.index.name = 'cv'
    return df_scores


def parse_task_name_string(task):
    if task.lower() in ['classification', 'cls']:
        task_name = 'classification'
    elif task.lower() in ['regression', 'reg']:
        task_name = 'regression'
    else:
        raise_invalid_task_error(task)
    return task_name


def parse_backend_name_string(backend):
    if backend.lower() in ['xgboost', 'xgb']:
        backend_name = 'xgboost'
    elif backend.lower() in ['lightgbm', 'lgb']:
        backend_name = 'lightgbm'
    else:
        raise_invalid_model_backend_error(backend)
    return backend_name


def get_model(task: str, backend: str, params: dict):
    """Returns a model for a specific task and backend."""
    assert task
    assert backend
    assert params
    if backend == 'xgboost':
        model = xgboost.get_model_by_task(task, params)
    elif backend == 'lightgbm':
        model = lightgbm.get_model_by_task(task, params)
    else:
        raise_invalid_model_backend_error(backend)
    return model


def get_cv_scoring(task):
    if task == 'classification':
        scoring = 'roc_auc'
    elif task == 'regression':
        scoring = 'neg_mean_squared_error'
    else:
        raise_invalid_task_error(task)
    return scoring


def cross_validation_score_iter(df, target, task='classification', backend='xgboost',
                                n_rounds=300, nfold=5, stratified=False, shuffle=True,
                                early_stopping_rounds=15, seed=0, show_stdv=True,
                                device_type='cpu', verbose=0):
    """Cross-validates a model on input data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    target : str or list
        Target name or list of names.
    task : str, optional (default='classification')
        Name of the task.
    backend : str, optional (default='xgboost')
        Name of the backend.
    n_rounds : int, optional (default=300)
        Number of iterations.
    nfold : int, optional (default=5)
        Number of folds used in CV.
    stratified : bool, optional (default=False)
        Whether to perform stratified sampling.
    shuffle: bool, optional (default=True)
        Whether to shuffle before splitting data.
    early_stopping_rounds : int or None, optional (default=15)
        Activates early stopping.
        CV score needs to improve at least every ``early_stopping_rounds`` round(s)
        to continue.
        Requires at least one metric. If there's more than one, will check all of them.
        Last entry in evaluation history is the one from best iteration.
    seed : int, optional (default=0)
        Seed used to generate the folds (passed to numpy.random.seed).
    show_stdv : bool, optional (default=True)
        Whether to display the standard deviation in progress.
        Results are not affected by this parameter, and always contains std.
    verbose_eval : int, optional (default=0)
        Whether to display the progress.
        The progress will be displayed at every given ``verbose_eval`` boosting stage.

    Returns
    -------
    cv_scores : pandas.DataFrame
        CV Scores per iteration.
    """
    assert target
    assert target in df, f"Target does not exist in the DataFrame: '{target}'"

    # parse strings to a default name
    task = parse_task_name_string(task)
    backend = parse_backend_name_string(backend)

    # Setup data
    X = df.drop(columns=target)
    X = pd.get_dummies(X, drop_first=True)
    y = df[target].values

    # Get model / task parameters
    params, metrics, cross_val_iter_fn = get_cv_iter_params(task, backend)

    if verbose_eval > 0:
        print(f"Starting cross-validation (task = '{task}')...")

    if verbose > 0:
        verbose_eval = verbose
    else:
        verbose_eval = None
    cv_results = cross_val_iter_fn(
        data=X,
        labels=y,
        params=params,
        metrics=metrics,
        n_rounds=n_rounds,
        nfold=nfold,
        stratified=stratified,
        shuffle=shuffle,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        show_stdv=show_stdv,
        verbose_eval=verbose_eval
    )
    if verbose > 0:
        print('\nCross-validation complete!')
    return pd.DataFrame(cv_results)


def get_cv_iter_params(task, backend):
    """Returns the task's name, objective and metrics for evaluation for XGBoost models."""
    metrics = get_metrics(task)
    params = get_model_parameters(backend)
    params["objective"] = get_objective_by_task(task, backend)
    cross_val_iter_fn = get_cross_validation_iter_method(backend)
    return params, metrics, cross_val_iter_fn


def get_metrics(task):
    if task == 'classification':
        metrics = 'auc'
    elif task == 'regression':
        metrics = 'rmse'
    else:
        raise_invalid_task_error(task)
    return metrics


def get_objective_by_task(task, backend):
    if backend == 'xgboost':
        objective = xgboost.get_objective_by_task(task)
    elif backend == 'lightgbm':
        objective = lightgbm.get_objective_by_task(task)
    else:
        raise_invalid_model_backend_error(backend)
    return objective


def get_model_parameters(backend):
    """Returns a dictionary of model parameters.

    Warning: The list of parameters is vary basic at this point.
    A more detailed list of parameters will be se in a later stage
    """
    if backend == 'xgboost':
        parameters = xgboost.get_default_parameters()
    elif backend == 'lightgbm':
        parameters = lightgbm.get_default_parameters()
    else:
        raise_invalid_model_backend_error(backend)
    return parameters


def get_cross_validation_iter_method(backend):
    if backend == 'xgboost':
        fn = xgboost.cross_validation_iter
    elif backend == 'lightgbm':
        fn = lightgbm.cross_validation_iter
    else:
        raise_invalid_model_backend_error(backend)
    return fn
