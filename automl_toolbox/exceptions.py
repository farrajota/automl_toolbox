"""
Exceptions classes.
"""


class UndefinedMethodError(Exception):
    """Raised if a method is not defined."""
    pass


class ModelBackendError(Exception):
    """Raised if a backend is not defined."""
    pass


class InvalidTaskError(Exception):
    """Raised if a task is not valid."""
    pass


def raise_invalid_task_error(task: str) -> None:
    raise InvalidTaskError(f"Invalid task: '{task}'.")


def raise_invalid_model_backend_error(backend: str) -> None:
    raise ModelBackendError(f"Invalid backend: '{backend}'")
