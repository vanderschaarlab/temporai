# pylint: disable=redefined-outer-name, unused-argument

from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import tempor
from tempor.config import TemporConfig


@pytest.fixture
def mock_updated_on_configure(monkeypatch):
    # Mock the observer.
    monkeypatch.setattr(tempor.config, "updated_on_configure", set())


def test_updated_on_configure(monkeypatch):
    config = tempor.get_config()
    assert isinstance(config, TemporConfig)
    assert config.logging.level == "INFO"
    assert config == tempor.config.get_config()


def test_default_config(mock_updated_on_configure):
    config = tempor.get_config()
    assert isinstance(config, TemporConfig)
    assert config.logging.level == "INFO"
    assert config == tempor.config.get_config()


def test_change_config_file_log(mock_updated_on_configure):
    config = tempor.get_config()
    config.logging.file_log = False
    tempor.configure(config)
    assert isinstance(config, TemporConfig)
    assert config.logging.file_log is False
    assert config == tempor.config.get_config()


@pytest.mark.parametrize("str_path", [False, True])
def test_change_config_yaml_file(tmp_path_factory, str_path, mock_updated_on_configure):
    yaml_file_path = tmp_path_factory.mktemp("my_config") / "config.yaml"
    if str_path:
        yaml_file_path = str(yaml_file_path)
    with open(yaml_file_path, "w", encoding="utf8") as f:
        f.write("logging:\n  level: VALUE_SET_VIA_YAML_FILE")

    config = tempor.configure(yaml_file_path)

    assert isinstance(config, TemporConfig)
    assert config.logging.level == "VALUE_SET_VIA_YAML_FILE"
    assert config == tempor.get_config()


def test_change_config_dictconfig(mock_updated_on_configure):
    config = tempor.configure(OmegaConf.create({"logging": {"level": "VALUE_SET_VIA_DICTCONFIG"}}))
    assert isinstance(config, TemporConfig)
    assert config.logging.level == "VALUE_SET_VIA_DICTCONFIG"
    assert config == tempor.get_config()


def test_change_config_dictconfig_alternative_import(mock_updated_on_configure):
    from tempor import config

    config = config.configure(OmegaConf.create({"logging": {"level": "VALUE_SET_VIA_DICTCONFIG"}}))
    assert isinstance(config, TemporConfig)
    assert config.logging.level == "VALUE_SET_VIA_DICTCONFIG"
    assert config == tempor.get_config()


def test_change_config_via_hydra(tmp_path_factory, mock_updated_on_configure):
    user_config_dir = tmp_path_factory.mktemp("conf")
    user_config_name = "config"
    user_config_filepath = user_config_dir / f"{user_config_name}.yaml"
    with open(user_config_filepath, "w", encoding="utf8") as f:
        f.write("tempor:\n  logging:\n    level: VALUE_SET_VIA_HYDRA")

    # Test using hydra's compose API:
    # https://hydra.cc/docs/advanced/compose_api/
    with initialize_config_dir(version_base=None, config_dir=str(user_config_dir)):
        user_config = compose(config_name=user_config_name)
        lib_config = tempor.configure(user_config.tempor)
        assert isinstance(lib_config, TemporConfig)
        assert lib_config.logging.level == "VALUE_SET_VIA_HYDRA"
        assert lib_config == tempor.get_config()


def test_change_config_fails_wrong_type():
    with pytest.raises(TypeError, match=".*type.*not supported.*"):
        tempor.configure([])  # type: ignore


@pytest.mark.parametrize("wd_raw", ["~", "$PWD"])
def test_get_working_dir(wd_raw):
    c = TemporConfig(logging=Mock(), working_directory=wd_raw)
    wd = c.get_working_dir()
    assert wd_raw not in wd


def test_observer(monkeypatch):
    mock_func = Mock()
    monkeypatch.setattr(tempor.config, "updated_on_configure", {mock_func})
    config = tempor.get_config()
    tempor.configure(config)
    mock_func.assert_called_once()
