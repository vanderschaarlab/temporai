import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.preprocessing.imputation import BaseImputer
from tempor.plugins.preprocessing.imputation.plugin_ffill import FFillImputer as plugin
from tempor.utils.datasets.sine import SineDataloader


def from_api() -> BaseImputer:
    return plugin_loader.get("preprocessing.imputation.ffill", random_state=123)


def from_module() -> BaseImputer:
    return plugin(random_state=123)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_ffill_plugin_sanity(test_plugin: BaseImputer) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "ffill"
    assert len(test_plugin.hyperparameter_space()) == 1


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_ffill_plugin_fit(test_plugin: BaseImputer) -> None:
    dataset = SineDataloader(with_missing=True).load()
    assert dataset.static.dataframe().isna().sum().sum() != 0

    test_plugin.fit(dataset)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_ffill_plugin_transform(test_plugin: BaseImputer) -> None:
    dataset = SineDataloader(with_missing=True).load()
    assert dataset.static.dataframe().isna().sum().sum() != 0

    output = test_plugin.fit(dataset).transform(dataset)

    assert output.static.dataframe().isna().sum().sum() == 0
    assert output.time_series.dataframe().isna().sum().sum() == 0


def test_hyperparam_sample():
    for repeat in range(100):
        args = plugin._cls.sample_hyperparameters()
        plugin(**args)
