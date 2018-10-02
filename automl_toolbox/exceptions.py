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
