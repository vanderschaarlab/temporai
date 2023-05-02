import contextlib
import glob
import logging
import os
import pathlib
import re
from typing import Any, List

from _pytest.logging import caplog  # pylint: disable=unused-import  # noqa: F401

import tempor
from tempor.log import logger

TEMPORAI_LOGURU_CONSOLE_LOGGERS = ("print", "main")

console_logs_part0 = r".*^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\s*?\|\s*?"
console_logs_part1 = r"\s*?.*?logger.*?\|\s*?"
re_flags = re.DOTALL | re.MULTILINE


@contextlib.contextmanager
def propagate_loguru(_caplog):
    """Since by default `pytest`'s ``caplog`` fixture doesn't capture `loguru` console logs, this context manager
    enables log propagation to resolve this issue.

    The code takes the log configurations stored in ``tempor.log._custom_logger._ADD_CONFIGS`` dictionary, which
    has the kwargs for ``logger.add(...)`` of each TemporAI console logger stored by its name key. The names are
    found in ``TEMPORAI_LOGURU_CONSOLE_LOGGERS`` - which needs to reflect the loguru console loggers registered in
    ``tempor.log._custom_logger``.

    Based on instructions from `loguru` docs:
    https://loguru.readthedocs.io/en/stable/resources/migration.html#replacing-caplog-fixture-from-pytest-library

    Args:
        _caplog: `pytest` ``caplog`` fixture.

    Yields:
        Modified `pytest` ``caplog``.
    """
    # NOTE: Need to import from tempor.log._custom_logger here (not at the top of file),
    # so that tempor.configure(config) re-configuration is taken into account.
    from tempor.log._custom_logger import _ADD_CONFIGS  # pyright: ignore

    class PropogateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    print("_ADD_CONFIGS\n", _ADD_CONFIGS)

    handler_ids = []
    for name in TEMPORAI_LOGURU_CONSOLE_LOGGERS:
        _ADD_CONFIGS[name]["sink"] = PropogateHandler()
        handler_ids.append(logger.add(**_ADD_CONFIGS[name]))

    yield _caplog
    for handler_id in handler_ids:
        logger.remove(handler_id)


def as_loguru_logs(records: List[Any]) -> str:
    """Takes in `pytest` ``caplog.records`` as they have been propagated to by the ``propagate_loguru`` helper and
    return these in the format that would appear in the console (including line breaks etc.). Colors and other fancy
    formatting is ignored here.
    """
    out = ""
    for r in records:
        message = r.getMessage()
        if message[-1] != "\n":
            message += "\n"
        out += message
    return out


def test_console_logging_at_trace(caplog):  # pylint: disable=redefined-outer-name  # noqa: F811

    config = tempor.get_config()
    config.logging.level = "TRACE"
    tempor.configure(config)

    with propagate_loguru(caplog) as cap_log:
        logger.trace("This is trace")
        logger.debug("This is debug")
        logger.info("This is info")
        logger.print("This one is printed")
        logger.warning("This is warning")
        logger.error("This is error")

        loguru_logs = as_loguru_logs(cap_log.records)

    assert re.match(console_logs_part0 + "TRACE" + console_logs_part1 + "This is trace\n", loguru_logs, re_flags)
    assert re.match(console_logs_part0 + "DEBUG" + console_logs_part1 + "This is debug\n", loguru_logs, re_flags)
    assert re.match(console_logs_part0 + "INFO" + console_logs_part1 + "This is info\n", loguru_logs, re_flags)
    assert re.match(r".*^This one is printed\n", loguru_logs, re_flags)
    assert re.match(console_logs_part0 + "WARNING" + console_logs_part1 + "This is warning\n", loguru_logs, re_flags)
    assert re.match(console_logs_part0 + "ERROR" + console_logs_part1 + "This is error\n", loguru_logs, re_flags)


def test_console_logging_at_info(caplog):  # pylint: disable=redefined-outer-name  # noqa: F811

    config = tempor.get_config()
    config.logging.level = "INFO"
    tempor.configure(config)

    with propagate_loguru(caplog) as cap_log:
        logger.trace("This is trace")
        logger.debug("This is debug")
        logger.info("This is info")
        logger.print("This one is printed")
        logger.warning("This is warning")
        logger.error("This is error")

        loguru_logs = as_loguru_logs(cap_log.records)

    assert not re.match(console_logs_part0 + "TRACE" + console_logs_part1 + "This is trace\n", loguru_logs, re_flags)
    assert not re.match(console_logs_part0 + "DEBUG" + console_logs_part1 + "This is debug\n", loguru_logs, re_flags)
    assert re.match(console_logs_part0 + "INFO" + console_logs_part1 + "This is info\n", loguru_logs, re_flags)
    assert re.match(r".*^This one is printed\n", loguru_logs, re_flags)
    assert re.match(console_logs_part0 + "WARNING" + console_logs_part1 + "This is warning\n", loguru_logs, re_flags)
    assert re.match(console_logs_part0 + "ERROR" + console_logs_part1 + "This is error\n", loguru_logs, re_flags)


def test_console_logging_at_error(caplog):  # pylint: disable=redefined-outer-name  # noqa: F811

    config = tempor.get_config()
    config.logging.level = "ERROR"
    tempor.configure(config)

    with propagate_loguru(caplog) as cap_log:
        logger.trace("This is trace")
        logger.debug("This is debug")
        logger.info("This is info")
        logger.print("This one is printed")
        logger.warning("This is warning")
        logger.error("This is error")

        loguru_logs = as_loguru_logs(cap_log.records)

    assert not re.match(console_logs_part0 + "TRACE" + console_logs_part1 + "This is trace\n", loguru_logs, re_flags)
    assert not re.match(console_logs_part0 + "DEBUG" + console_logs_part1 + "This is debug\n", loguru_logs, re_flags)
    assert not re.match(console_logs_part0 + "INFO" + console_logs_part1 + "This is info\n", loguru_logs, re_flags)
    assert re.match(r".*^This one is printed\n", loguru_logs, re_flags)  # NOTE: logger.print should still show up!
    assert not re.match(
        console_logs_part0 + "WARNING" + console_logs_part1 + "This is warning\n", loguru_logs, re_flags
    )
    assert re.match(console_logs_part0 + "ERROR" + console_logs_part1 + "This is error\n", loguru_logs, re_flags)


def test_file_log_enabled(tmp_path: pathlib.Path):
    config = tempor.get_config()
    config.working_directory = str(tmp_path)
    config.logging.level = "ERROR"  # NOTE: The file logging will ignore this and always log at "DEBUG".
    config.logging.file_log = True
    tempor.configure(config)

    logger.trace("This is trace")
    logger.debug("This is debug")
    logger.info("This is info")
    logger.print("This one is printed")
    logger.warning("This is warning")
    logger.error("This is error")

    log_dir = tmp_path / "logs"

    assert os.path.isdir(log_dir)

    log_files = list(glob.glob(os.path.join(log_dir, "*.log")))
    assert len(log_files) > 0

    with open(log_files[0], "r", encoding="utf8") as f:
        log_content = f.read()

    assert not re.match(console_logs_part0 + "TRACE" + console_logs_part1 + "This is trace\n", log_content, re_flags)
    assert re.match(console_logs_part0 + "DEBUG" + console_logs_part1 + "This is debug\n", log_content, re_flags)
    assert re.match(console_logs_part0 + "INFO" + console_logs_part1 + "This is info\n", log_content, re_flags)
    assert re.match(console_logs_part0 + "INFO" + console_logs_part1 + "This is info\n", log_content, re_flags)
    assert re.match(console_logs_part0 + "WARNING" + console_logs_part1 + "This is warning\n", log_content, re_flags)
    assert re.match(console_logs_part0 + "ERROR" + console_logs_part1 + "This is error\n", log_content, re_flags)
