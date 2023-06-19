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
    """The class creates backbone for configuration data in the login process."""
    level: str = omegaconf.MISSING
    diagnose: bool = omegaconf.MISSING
    backtrace: bool = omegaconf.MISSING
    file_log: bool = omegaconf.MISSING


@dataclasses.dataclass
class TemporConfig:
    """The class contains data necessary for temporary configuration.
    It plays a key role during the execution of the configure_loggers function.
    """

    logging: LoggingConfig
    working_directory: str = omegaconf.MISSING

    def get_working_dir(self):
        """Return working directory regardless operating system."""
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
    """Return module configuration data wrapped in an dataclass object.
    This method is used to initialize configuration data during the login process.
    """
    return _this_module._config


def _load(loaded_config: omegaconf.DictConfig) -> TemporConfig:
    """Receive new configuration data, merge it and wrapp into dataclass object.
    The function also verifies whether the result of the operation is a TemporConfig object.

    Returns:
            config_as_dataclass: instance of dataclass with new configuration data.

    """

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
    """Function receives a file path, converts it to omegaconf.DictConfig type
    and then passes it to the _load function.
    """

    loaded = OmegaConf.load(path)
    if TYPE_CHECKING:  # pragma: no cover
        assert isinstance(loaded, omegaconf.DictConfig)  # nosec B101
    return _load(loaded)


def load_dictconfig(config_node: omegaconf.DictConfig) -> TemporConfig:
    """Send dictionary with configuration data to _load function."""
    return _load(config_node)


# Load initial default configuration:
_config: TemporConfig = load_yaml_file(path=DEFAULT_CONFIG_FILE_PATH)


def configure(new_config: Union[TemporConfig, omegaconf.DictConfig, str, pathlib.Path]) -> TemporConfig:
    """Main-line function which sets up new configuration regardless of the type of the allowed batch object.
     Then updates it.

    Returns:
        _this_module._config: instance of dataclass with new configuration data.

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
