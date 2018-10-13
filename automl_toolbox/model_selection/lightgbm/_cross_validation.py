"""
Methods for cross-validating data using LightGBM models.
"""


import lightgbm as lgb

from automl_toolbox.exceptions import raise_invalid_task_error


def get_model_by_task(task, params):
    """Returns a model for a specific task."""
    assert task
    assert params
    if task == 'classification':
        model = lgb.LGBMClassifier(**params)
    elif task == 'regression':
        model = lgb.LGBMRegressor(**params)
    else:
        raise_invalid_task_error(task)
    return model


def get_objective_by_task(df, target, task):
    """Returns an objective and a set of metrics for a specific task."""
    if task == 'classification':
        if df[target].nunique() == 2:
            objective = 'binary'
        else:
            objective = 'multi'
    elif task == 'regression':
        objective = 'regression'
    else:
        raise_invalid_task_error(task)
    return objective


def cross_validation_iter(data, labels, params, metrics, n_rounds, nfold,
                          stratified, shuffle, early_stopping_rounds, seed,
                          show_stdv, verbose_eval=0):
    """Cross validates a model using LightGBM.

    Parameters
    ----------

    Returns
    -------
    """
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
