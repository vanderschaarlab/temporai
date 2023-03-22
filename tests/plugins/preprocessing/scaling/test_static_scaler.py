from typing import Any

import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.preprocessing.scaling import BaseScaler
from tempor.plugins.preprocessing.scaling.plugin_static_scaler import (
    StaticScalerScaler as plugin,
)
from tempor.utils.dataloaders import GoogleStocksDataLoader, SineDataLoader


def from_api() -> BaseScaler:
    return plugin_loader.get("preprocessing.scaling.static_scaler", random_state=123)


def from_module() -> BaseScaler:
    return plugin(random_state=123)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_static_scaler_plugin_sanity(test_plugin: BaseScaler) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "static_scaler"
    assert len(test_plugin.hyperparameter_space()) == 0


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
@pytest.mark.parametrize("dataloader", [GoogleStocksDataLoader(), SineDataLoader(static_scale=5)])
def test_static_scaler_plugin_fit(test_plugin: BaseScaler, dataloader: Any) -> None:
    dataset = dataloader.load()
    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_static_scaler_plugin_transform(test_plugin: BaseScaler) -> None:
    dataset = SineDataLoader(static_scale=100).load()
    assert dataset.static is not None  # nosec B101
    assert (dataset.static.numpy() > 50).any()

    output = test_plugin.fit(dataset).transform(dataset)

    print(output.static.numpy().min(), output.static.numpy().max())
    assert (output.static.numpy() < 50).all()


def test_hyperparam_sample():
    for repeat in range(100):  # pylint: disable=unused-variable
        args = plugin._cls.sample_hyperparameters()  # pylint: disable=no-member, protected-access
        plugin(**args)
