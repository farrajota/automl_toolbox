"""
Model train, test and evaluate functions.
"""


import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import cross_val_score

from .exceptions import InvalidTaskError, ModelBackendError


def cross_validation(df, target, task='classification', backend='lightgbm', cv=5,
                     n_jobs=None, verbose=0):
    """Evaluates a Gradient Boosting Tree model on input data.

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

    # Setup data
    X = df.drop(columns=target)
    X = pd.get_dummies(X, drop_first=True)
    y = df[target]

    # setup model
    model, scoring = get_model(task, backend)

    print('Starting cross-validation...')
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose)
    print('\nCross-validation complete!')

    if task.lower() in ['classification', 'cls']:
        print(f'\nAccuracy (%): {scores.mean() * 100:.5f} +- {scores.std() * 100:.5f}')
    elif task.lower() in ['regression', 'reg']:
        scores_mean = np.sqrt(-scores).mean()
        scores_std = np.sqrt(-scores).std()
        print(f'\nScore (rmse): {scores_mean:.2f} +- {scores_std:.2f}')
    else:
        raise InvalidTaskError(f"Invalid task: '{task}'.")

    df_scores = pd.DataFrame(data={'score': scores})
    df_scores.index.name = 'cv'
    return df_scores


def get_model(task, backend):
    """Returns a model w.r.t. a task and backend."""
    assert task
    assert backend

    if task.lower() in ['classification', 'cls']:
        if backend.lower() in ['xgboost', 'xgb']:
            model = xgb.XGBClassifier()
        elif backend.lower() in ['lightgbm', 'lgb']:
            model = lgb.LGBMClassifier()
        else:
            raise ModelBackendError(f"Invalid backend: '{backend}'")
        scoring = 'roc_auc'
    elif task.lower() in ['regression', 'reg']:
        if backend.lower() in ['xgboost', 'xgb']:
            model = xgb.XGBRegressor()
        elif backend.lower() in ['lightgbm', 'lgb']:
            model = lgb.LGBMRegressor()
        else:
            raise ModelBackendError(f"Invalid backend: '{backend}'")
        scoring = 'neg_mean_squared_error'
    else:
        raise InvalidTaskError(f"Invalid task: '{task}'.")

    return model, scoring


def cross_validation_iter(df, target, task='classification', backend='xgboost',
                          n_rounds=300, nfold=5, stratified=False, shuffle=True,
                          early_stopping_rounds=15, seed=0, show_stdv=True,
                          verbose_eval=None):
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
    verbose_eval : bool, int, or None, optional (default=None)
        Whether to display the progress.
        If None, progress will be displayed when np.ndarray is returned.
        If True, progress will be displayed at every boosting stage.
        If int, progress will be displayed at every given ``verbose_eval`` boosting stage.

    Returns
    -------
    cv_scores : pandas.DataFrame
        CV Scores per iteration.
    """
    assert target
    assert target in df, f"Target does not exist in the DataFrame: '{target}'"

    # Setup data
    X = df.drop(columns=target)
    X = pd.get_dummies(X, drop_first=True)
    y = df[target].values

    # Get model / task parameters
    task_full_name, params, metrics, cross_val_iter_fn = get_cv_iter_params(df, target, task, backend)

    print(f"Starting cross-validation (task = '{task_full_name}')...")

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

    print('\nCross-validation complete!')

    return pd.DataFrame(cv_results)


def get_cv_iter_params(df, target, task, backend):
    """Returns the task's name, objective and metrics for evaluation for XGBoost models."""
    name, objective, metrics = get_task_params(df, target, task, backend)
    params = get_model_parameters()
    params["objective"] = objective
    cross_val_iter_fn = get_cross_validation_iter_method(backend)
    return name, params, metrics, cross_val_iter_fn


def get_task_params(df, target, task, backend):
    if task.lower() in ['classification', 'cls']:
        if backend.lower() in ['xgboost', 'xgb']:
            if df[target].nunique() == 2:
                objective = 'binary:logistic'
            else:
                objective = 'multi:softmax'
        elif backend.lower() in ['lightgbm', 'lgb']:
            if df[target].nunique() == 2:
                objective = 'binary'
            else:
                objective = 'multi'
        else:
            raise ModelBackendError(f"Invalid backend: '{backend}'.")
        metrics = 'auc'
        name = 'classification'
    elif task.lower() in ['regression', 'reg']:
        if backend.lower() in ['xgboost', 'xgb']:
            objective = 'reg:linear'
        elif backend.lower() in ['lightgbm', 'lgb']:
            objective = 'regression'
        else:
            raise ModelBackendError(f"Invalid backend: '{backend}'.")
        name = 'regression'
        metrics = 'rmse'
    else:
        raise InvalidTaskError(f"Invalid task: '{task}'.")
    return name, objective, metrics


def get_model_parameters():
    """Returns a dictionary of model parameters.

    Warning: The list of parameters is vary basic at this point.
    A more detailed list of parameters will be se in a later stage
    """
    # TODO : create a more detailed list of parameters
    return {
        'learning_rate': 0.03,
        'max_depth': 10,
        'tree_method': 'exact',
    }


def get_cross_validation_iter_method(backend):
    if backend.lower() in ['xgboost', 'xgb']:
        return cross_validation_iter_xgboost
    elif backend.lower() in ['lightgbm', 'lgb']:
        return cross_validation_iter_lightgbm
    else:
        raise ModelBackendError(f"Invalid backend: '{backend}'.")


def cross_validation_iter_xgboost(data, labels, params, metrics, n_rounds, nfold,
                                  stratified, shuffle, early_stopping_rounds, seed,
                                  show_stdv, verbose_eval):
    """Cross validates a model using the XGBoost backend."""
    dftrainXGB = xgb.DMatrix(data=data, label=labels, feature_names=list(data), silent=1, nthread=-1)
    cv_results = xgb.cv(
        params,
        dftrainXGB,
        num_boost_round=n_rounds,
        nfold=nfold,
        metrics=metrics,
        stratified=stratified,
        shuffle=shuffle,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        show_stdv=show_stdv,
        verbose_eval=verbose_eval
    )
    return cv_results


def cross_validation_iter_lightgbm(data, labels, params, metrics, n_rounds, nfold,
                                   stratified, shuffle, early_stopping_rounds, seed,
                                   show_stdv, verbose_eval):
    """Cross validates a model using the LightGBM backend."""
    dftrainLGB = lgb.Dataset(data=data, label=labels, feature_name=list(data))
    cv_results = lgb.cv(
        params,
        dftrainLGB,
        num_boost_round=n_rounds,
        nfold=nfold,
        metrics=metrics,
        stratified=stratified,
        shuffle=shuffle,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        show_stdv=show_stdv,
        verbose_eval=verbose_eval
    )
    return cv_results
