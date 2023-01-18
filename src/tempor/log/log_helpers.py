import sys
from contextlib import contextmanager

from ._custom_logger import logger


def _suppress_exception_output(exc_type, exc_value, exc_traceback):
    # See: https://stackoverflow.com/a/16993115
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return


@contextmanager
def exc_to_log(message: str = ""):
    try:
        yield
    except Exception:  # pylint: disable=broad-except
        sys.excepthook = _suppress_exception_output
        logger.bind(always_show=True).opt(depth=1).exception(message)
        raise
    finally:
        pass
