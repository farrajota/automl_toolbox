"""
Utility methods.
"""


from .exceptions import raise_invalid_task_error, raise_invalid_model_backend_error


def parse_task_name_string(task: str) -> str:
    if task.lower() in ['classification', 'cls']:
        task_name = 'classification'
    elif task.lower() in ['regression', 'reg']:
        task_name = 'regression'
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
