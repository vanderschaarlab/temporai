import logging
import os
import sys
from typing import Any, Callable, Dict, Literal, Union

from loguru import logger

import tempor.config as conf

_LogFormatType = Literal["print-like", "log-like"]

logger.level("INFO", color="")  # Make INFO level format not bold.


# A formatter to automatically "drop" the log message on a new line if the message is already multiline.
_nl_replace = "$NL"
_log_like_base = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>" + _nl_replace + "{message}</level>\n{exception}"
)
_log_like_newline = _log_like_base.replace(_nl_replace, "\n")
_log_like_no_newline = _log_like_base.replace(_nl_replace, "")


def _log_like_formatter(record) -> str:
    return _log_like_newline if "\n" in record["message"] else _log_like_no_newline


log_formats: Dict[_LogFormatType, Union[str, Callable]] = {
    "print-like": "<level>{message}</level>",
    "log-like": _log_like_formatter,
}
# ---

_initial_config = conf.get_config()

LOG_FILE = os.path.join(_initial_config.get_working_dir(), "logs", "tempor_{time:YYYY-MM-DD}.log")

_LOGGERS = []  # Store current logger IDs, so that they can be removed an re-created if needed.


# Dynamically add print method.
def _print_filter(record):
    # A filter to enable special handling of logger.print() case.
    return record["extra"].get("print", False)


def _logger_print(message: str):
    logger.bind(print=True).opt(depth=1).info(message)


logger.print = _logger_print  # type: ignore
# ---


def _configure_loggers(config: conf.TemporConfig):
    # Remove loguru default logger.
    logger.remove()

    # Console logger setup:
    console_log_format_map: Dict[conf.LoggingMode, Union[str, Callable]] = {
        conf.LoggingMode.LIBRARY: log_formats["print-like"],
        conf.LoggingMode.SCRIPT: log_formats["log-like"],
    }

    console_logger_common: Any = dict(
        sink=sys.stderr,
        diagnose=config.logging.diagnose,
        backtrace=config.logging.backtrace,
    )

    # This logger will behave like normal print(). For logger.print() function.
    console_logger_print = logger.add(
        **console_logger_common,
        format=log_formats["print-like"],
        filter=_print_filter,
        level="DEBUG",
    )

    # This is the main console logger.
    console_logger_main = logger.add(
        **console_logger_common,
        format=console_log_format_map[config.logging.mode],
        filter=lambda record: not _print_filter(record),
        level=config.logging.level,
    )

    _LOGGERS.extend([console_logger_print, console_logger_main])

    # File logger setup:
    if config.logging.file_log:
        file_logger = logger.add(
            LOG_FILE,
            format=log_formats["log-like"],
            diagnose=True,
            backtrace=True,
            rotation="10 MB",
            retention="1 day",
            level=logging.DEBUG,
        )
        _LOGGERS.append(file_logger)


_configure_loggers(_initial_config)

conf.updated_on_configure.add(_configure_loggers)
