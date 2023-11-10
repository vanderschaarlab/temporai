"""Package directory for TemporAI configuration."""

import dataclasses
import os
import pathlib
import sys
from typing import TYPE_CHECKING, Callable, Set, Union

import hydra.core.config_store
import omegaconf
from omegaconf import OmegaConf

import tempor

# NOTE: This config module is loaded before everything else, as other modules may use the config.

DEFAULT_CONFIG_DIR = "conf/tempor"
DEFAULT_CONFIG_FILE_NAME = "config"
DEFAULT_CONFIG_FILE_PATH = os.path.join(
    os.path.dirname(__file__),
    f"{DEFAULT_CONFIG_DIR}/{DEFAULT_CONFIG_FILE_NAME}.yaml",
)


# --- Config structure. ---
# Define the structure of the library config below.
# Use missing for everything and define the defaults in the config file: DEFAULT_CONFIG_FILE_PATH.


@dataclasses.dataclass
class LoggingConfig:
    """The configuration class for logging."""

    level: str = omegaconf.MISSING
    """Logging level. One of: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``."""
    diagnose: bool = omegaconf.MISSING
    """Whether to use `loguru`'s ``diagnose`` setting for exceptions."""
    backtrace: bool = omegaconf.MISSING
    """Whether to use `loguru`'s ``backtrace`` setting for exceptions."""
    file_log: bool = omegaconf.MISSING
    """Whether to log to a file."""


@dataclasses.dataclass
class TemporConfig:
    """The main configuration class for the TemporAI library."""

    logging: LoggingConfig
    """Logging configuration."""
    working_directory: str = omegaconf.MISSING
    """Working directory for the library. Can be set to a path string, ``"$PWD"``. or ``"~"``."""

    def get_working_dir(self) -> str:
        """Get the working directory, with ``"$PWD"`` and ``"~"`` expanded.

        Returns:
            str: The working directory string.
        """
        if self.working_directory.startswith("$PWD"):
            return self.working_directory.replace("$PWD", os.getcwd(), 1)
        elif self.working_directory.startswith("~"):
            return self.working_directory.replace("~", os.path.expanduser("~"), 1)
        else:
            return self.working_directory


# --- Config structure: end. ---

# Minimal observer-pattern like setup to trigger relevant changes when configure() is called.
# To "subscribe":
# from tempor.configuration import updated_on_configure
# updated_on_configure.add(my_method)
updated_on_configure: Set[Callable[[TemporConfig], None]] = set()

# Register dataclass with Hydra config store for Hydra type checking.
_cs = hydra.core.config_store.ConfigStore.instance()
_cs.store(name=tempor.import_name, node=TemporConfig)

# Initialize OmegaConf schema.
_tempor_config_schema = OmegaConf.structured(TemporConfig)


_this_module = sys.modules[__name__]  # Needed to directly set `config` on this module in the configure() function.


# pylint: disable=protected-access
def get_config() -> TemporConfig:
    """Get the current TemporAI configuration.

    Returns:
        TemporConfig: TemporAI configuration.
    """
    return _this_module._config


def _load(loaded_config: omegaconf.DictConfig) -> TemporConfig:
    if hasattr(_this_module, "_config"):
        merge_into = _this_module._config
    else:
        merge_into = _tempor_config_schema
    merged_validated_config = OmegaConf.merge(merge_into, loaded_config)
    config_as_dataclass = OmegaConf.to_container(
        merged_validated_config, structured_config_mode=omegaconf.SCMode.INSTANTIATE
    )
    if TYPE_CHECKING:  # pragma: no cover
        assert isinstance(config_as_dataclass, TemporConfig)  # nosec B101

    return config_as_dataclass


def load_yaml_file(path: Union[str, pathlib.Path]) -> TemporConfig:
    """Load a YAML file as a `TemporConfig` object.

    Args:
        path (Union[str, pathlib.Path]): The path to the YAML file.

    Returns:
        TemporConfig: TemporAI configuration.
    """
    loaded = OmegaConf.load(path)
    if TYPE_CHECKING:  # pragma: no cover
        assert isinstance(loaded, omegaconf.DictConfig)  # nosec B101
    return _load(loaded)


def load_dictconfig(config_node: omegaconf.DictConfig) -> TemporConfig:
    """Load an `omegaconf.DictConfig` as a `TemporConfig` object.

    Args:
        config_node (omegaconf.DictConfig): The configuration ``DictConfig``.

    Returns:
        TemporConfig: TemporAI configuration.
    """
    return _load(config_node)


# Load initial default configuration:
_config: TemporConfig = load_yaml_file(path=DEFAULT_CONFIG_FILE_PATH)


def configure(new_config: Union[TemporConfig, omegaconf.DictConfig, str, pathlib.Path]) -> TemporConfig:
    """Configure TemporAI with a new config.

    Args:
        new_config (Union[TemporConfig, omegaconf.DictConfig, str, pathlib.Path]):
            The new configuration. Can be a ``TemporConfig`` object, a ``DictConfig`` object, or a path to a YAML file.

    Returns:
        TemporConfig: TemporAI configuration.
    """
    if isinstance(new_config, (str, pathlib.Path)):
        _this_module._config = load_yaml_file(path=new_config)  # type: ignore
    elif isinstance(new_config, omegaconf.DictConfig):
        _this_module._config = load_dictconfig(config_node=new_config)  # type: ignore
    elif isinstance(new_config, TemporConfig):
        _this_module._config = new_config  # type: ignore
    else:
        raise TypeError(f"`new_config` of type {type(new_config)} not supported")

    for updater in updated_on_configure:
        updater(_this_module._config)

    return _this_module._config
