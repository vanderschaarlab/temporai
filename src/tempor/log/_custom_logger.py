"""Module with custom loguru logger setup for TemporAI."""

import logging
import os
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, Union

from loguru import logger  # pyright: ignore
from typing_extensions import Literal

import tempor.config as conf

# Make INFO level format not bold:
logger.level("INFO", color="")


# A formatter to automatically "drop" the log message on a new line if the message is already multiline. ---
_nl_replace = "$NL"
_log_like_base = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>" + _nl_replace + "{message}</level>\n{exception}"
)
_log_like_newline = _log_like_base.replace(_nl_replace, "\n")
_log_like_no_newline = _log_like_base.replace(_nl_replace, "")


def _log_like_formatter(record: Dict) -> str:
    return _log_like_newline if "\n" in record["message"] else _log_like_no_newline


_LogFormatType = Literal["print-like", "log-like"]

log_formats: Dict[_LogFormatType, Union[str, Callable]] = {
    "print-like": "<level>{message}</level>",
    "log-like": _log_like_formatter,
}
# --- --- ---


_this_module = sys.modules[__name__]  # Needed to update `_LOGGERS`, `_ADD_CONFIGS` inside `_configure_loggers()`.

_initial_config = conf.get_config()

_LOGGERS = []  # Store current logger IDs, so that they can be removed an re-created if needed.
_ADD_CONFIGS = dict()


# Dynamically add print method. ---
def _print_filter(record: Dict) -> bool:
    # A filter to enable special handling of logger.print() case.
    return record["extra"].get("print", False)


def _logger_print(message: str) -> None:
    logger.bind(print=True).opt(depth=1).info(message)


logger.print = _logger_print  # type: ignore
# --- --- ---


def _configure_loggers(config: conf.TemporConfig) -> None:
    # Reset _LOGGERS, _ADD_CONFIGS
    _this_module._LOGGERS = []  # type: ignore  # pylint: disable=protected-access
    _this_module._ADD_CONFIGS = dict()  # type: ignore  # pylint: disable=protected-access

    log_file = os.path.join(config.get_working_dir(), "logs", "tempor_{time:YYYY-MM-DD}.log")

    # Remove loguru default logger.
    logger.remove()

    # Common settings for console loggers.
    common_settings: Any = dict(
        sink=sys.stderr,
        # Take diagnose and backtrace settings from config:
        diagnose=config.logging.diagnose,
        backtrace=config.logging.backtrace,
    )

    # This logger will behave like normal `print()`. For `logger.print()` function only.
    _ADD_CONFIGS["print"] = dict(
        **common_settings,
        format=log_formats["print-like"],
        filter=_print_filter,
        level="INFO",
    )
    console_logger_print = logger.add(**_ADD_CONFIGS["print"])  # pyright: ignore

    # This is the main console logger.
    _ADD_CONFIGS["main"] = dict(
        **common_settings,
        format=log_formats["log-like"],
        filter=lambda record: not _print_filter(record),
        level=config.logging.level,
    )
    console_logger_main = logger.add(**_ADD_CONFIGS["main"])  # pyright: ignore

    _LOGGERS.extend([console_logger_print, console_logger_main])

    # File logger setup:
    if config.logging.file_log:
        _ADD_CONFIGS["file"] = dict(
            sink=log_file,
            format=log_formats["log-like"],
            rotation="10 MB",
            retention="1 day",
            level=logging.DEBUG,
            # Always have diagnose and backtrace enabled in the file logger.
            diagnose=True,
            backtrace=True,
        )
        file_logger = logger.add(**_ADD_CONFIGS["file"])  # pyright: ignore
        _LOGGERS.append(file_logger)


_configure_loggers(_initial_config)

conf.updated_on_configure.add(_configure_loggers)


if TYPE_CHECKING:  # pragma: no cover
    # This is to allow type checkers to recognize the custom added logger.print method.

    from loguru import Logger as _Logger

    class Logger(_Logger):
        """A `TYPE_CHECKING`-only subclass of `loguru.Logger` with added `print()` method."""

        def print(self, message: str) -> None:
            """A logger method that acts as both logger ``INFO`` message and Python ``print()``.

            Args:
                message (str): Message to print.
            """
            pass  # pylint: disable=unnecessary-pass

    logger: Logger  # type: ignore
