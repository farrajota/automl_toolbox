"""
Methods for cross-validating data using XGBoost models.
"""


import xgboost as xgb

from .exceptions import raise_invalid_task_error


def get_model_by_task(task, params):
    """Returns a model for a specific task."""
    assert task
    assert params
    if task.lower() in ['classification', 'cls']:
        model = xgb.XGBClassifier(**params)
    elif task.lower() in ['regression', 'reg']:
        model = xgb.XGBRegressor(**params)
    else:
        raise_invalid_task_error(task)
    return model, scoring


def get_objective_by_task(task):
    """Returns an objective and a set of metrics for a specific task."""
    if task.lower() in ['classification', 'cls']:
        if df[target].nunique() == 2:
            objective = 'binary:logistic'
        else:
            objective = 'multi:softmax'
        metrics = 'auc'
        name = 'classification'
    elif task.lower() in ['regression', 'reg']:
        objective = 'reg:linear'
        name = 'regression'
        metrics = 'rmse'
    else:
        raise_invalid_task_error(task)
    return name, objective, metrics


def cross_validation_iter(data, labels, params, metrics, n_rounds, nfold,
                          stratified, shuffle, early_stopping_rounds, seed,
                          show_stdv, verbose_eval=0):
    """Cross validates a model using XGBoost.

    Parameters
    ----------

    Returns
    -------
    """
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
