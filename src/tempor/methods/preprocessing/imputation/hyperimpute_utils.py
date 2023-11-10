"""Utilities for `hyperimpute`."""

import contextlib
from typing import Generator


@contextlib.contextmanager
def monkeypatch_hyperimpute_logger() -> Generator:
    """In `hyperimpute`, at least as of version ``0.1.17``, the following call in `hyperimpute.logger` causes a
    conflict with `loguru` logger as used in TemporAI. To circumvent this problem, this context manager monkeypatches
    `loguru` ``logger.remove`` call with a no-op. To be used around `hyperimpute` imports.

    ```
    from loguru import logger
    ...
    logger.remove()
    ```

    """
    from loguru import logger

    original_remove = logger.remove

    def monkeypatched_remove() -> None:
        pass

    logger.remove = monkeypatched_remove  # type: ignore

    try:
        yield

    finally:
        logger.remove = original_remove  # type: ignore
