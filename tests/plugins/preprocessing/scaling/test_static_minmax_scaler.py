import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.preprocessing.scaling import BaseScaler
from tempor.plugins.preprocessing.scaling.plugin_static_minmax_scaler import (
    StaticMinMaxScalerScaler as plugin,
)
from tempor.utils.dataloaders import SineDataLoader


def from_api() -> BaseScaler:
    return plugin_loader.get("preprocessing.scaling.static_minmax_scaler", random_state=123)


def from_module() -> BaseScaler:
    return plugin(random_state=123)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_static_minmax_scaler_plugin_sanity(test_plugin: BaseScaler) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "static_minmax_scaler"
    assert len(test_plugin.hyperparameter_space()) == 0


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_static_minmax_scaler_plugin_fit(test_plugin: BaseScaler) -> None:
    dataset = SineDataLoader(static_scale=5).load()
    assert dataset.static is not None  # nosec B101

    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_static_minmax_scaler_plugin_transform(test_plugin: BaseScaler) -> None:
    dataset = SineDataLoader(static_scale=100).load()
    assert dataset.static is not None  # nosec B101
    assert (dataset.static.numpy() > 1.1).any()

    output = test_plugin.fit(dataset).transform(dataset)

    assert (output.static.numpy() < 1 + 1e-1).all()
    assert (output.static.numpy() >= 0).all()


def test_hyperparam_sample():
    for repeat in range(100):  # pylint: disable=unused-variable
        args = plugin._cls.sample_hyperparameters()  # pylint: disable=no-member, protected-access
        plugin(**args)
