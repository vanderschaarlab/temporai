import dataclasses
import enum
import os
import pathlib
import sys
from typing import TYPE_CHECKING, Callable, Set, Union

import hydra.core.config_store
import omegaconf
from omegaconf import OmegaConf

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


class LoggingMode(enum.Enum):
    LIBRARY = enum.auto()
    SCRIPT = enum.auto()


@dataclasses.dataclass
class LoggingConfig:
    mode: LoggingMode = LoggingMode.LIBRARY
    level: str = omegaconf.MISSING
    diagnose: bool = omegaconf.MISSING
    backtrace: bool = omegaconf.MISSING
    file_log: bool = omegaconf.MISSING


@dataclasses.dataclass
class TemporConfig:
    logging: LoggingConfig
    working_directory: str = omegaconf.MISSING

    def get_working_dir(self):
        if self.working_directory == "$PWD":
            return os.getcwd()
        elif self.working_directory == "~":
            return os.path.expanduser("~")
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
_cs.store(name="tempor", node=TemporConfig)

# Initialize OmegaConf schema.
_tempor_config_schema = OmegaConf.structured(TemporConfig)


_this_module = sys.modules[__name__]  # Needed to directly set `config` on this module in the configure() function.


# pylint: disable=protected-access
def get_config() -> TemporConfig:
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
    loaded = OmegaConf.load(path)
    if TYPE_CHECKING:  # pragma: no cover
        assert isinstance(loaded, omegaconf.DictConfig)  # nosec B101
    return _load(loaded)


def load_dictconfig(config_node: omegaconf.DictConfig) -> TemporConfig:
    return _load(config_node)


# Load initial default configuration:
_config: TemporConfig = load_yaml_file(path=DEFAULT_CONFIG_FILE_PATH)


def configure(new_config: Union[TemporConfig, omegaconf.DictConfig, str, pathlib.Path]) -> TemporConfig:
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
