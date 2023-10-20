"""Module with helpers for logging."""

from contextlib import contextmanager
from typing import Generator

from ._custom_logger import logger  # type: ignore [attr-defined]


@contextmanager
def exc_to_log(message: str = "", reraise: bool = True) -> Generator:
    """Log `Exception` raised inside this context manager and reraise.

    Args:
        message (str, optional): Log message. Defaults to "".
        reraise (bool, optional): Whether to reraise the exception. Defaults to True.

    Yields:
        Generator: Context manager.
    """
    try:
        yield
    except Exception:  # pylint: disable=broad-except
        logger.opt(depth=2).exception(message)
        logger.opt(depth=2).error("=== Exception logs above ===\n")
        if reraise:
            raise
    finally:
        pass
