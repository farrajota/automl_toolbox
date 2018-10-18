"""
Utility methods.
"""


import pandas as pd
from .exceptions import (raise_clustering_not_implemented_error,
                         raise_invalid_task_error,
                         raise_invalid_model_backend_error)


def parse_task_name_string(task: str) -> str:
    task_l = task.lower()
    if task_l in ['classification', 'cls']:
        task_name = 'classification'
    elif task_l in ['regression', 'reg']:
        task_name = 'regression'
    elif task_l in ['clustering', 'cluster']:
        raise_clustering_not_implemented_error()
    else:
        raise_invalid_task_error(task)
    return task_name


def parse_backend_name_string(backend: str) -> str:
    if backend.lower() in ['xgboost', 'xgb']:
        backend_name = 'xgboost'
    elif backend.lower() in ['lightgbm', 'lgb']:
        backend_name = 'lightgbm'
    else:
        raise_invalid_model_backend_error(backend)
    return backend_name


def transform_data_target(df: pd.DataFrame, target: str):
    """Transforms the data format to be used by models.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    target : str
        Column name of the target.

    Returns
    -------
    pandas.DataFrame, pandas.Series
        Returns the features and the target transformed and splitted.
    """
    X = df.drop(columns=target)
    X = pd.get_dummies(X, drop_first=True)
    y = df[target]
    return X, y


def infer_task_type(df: pd.DataFrame, target: str) -> str:
    """Infers the type of the task by using the data of the target
    column of a Pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    target : str
        Column name of the target.

    Returns
    -------
    str
        Name of the task.
    """
    target_df = df[target]
    if target_df.nunique() / len(target_df) <= 0.5:
        return 'classification'
    else:
        return 'regression'
