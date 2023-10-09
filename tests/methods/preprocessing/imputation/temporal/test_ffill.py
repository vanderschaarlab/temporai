# pylint: disable=redefined-outer-name

from typing import Any, Callable, Dict

import pytest

from tempor.methods.preprocessing.imputation import BaseImputer
from tempor.methods.preprocessing.imputation.temporal.plugin_ffill import FFillImputer
from tempor.utils.serialization import load, save

INIT_KWARGS: Any = dict()
PLUGIN_FROM_OPTIONS = ["from_api", pytest.param("from_module", marks=pytest.mark.extra)]
TEST_ON_DATASETS = [
    "sine_data_missing_small",
    pytest.param("sine_data_missing_full", marks=pytest.mark.extra),
]


@pytest.fixture
def get_test_plugin(get_plugin: Callable):
    def func(plugin_from: str, base_kwargs: Dict):
        return get_plugin(
            plugin_from,
            fqn="preprocessing.imputation.temporal.ffill",
            cls=FFillImputer,
            kwargs=base_kwargs,
        )

    return func


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
def test_sanity(get_test_plugin: Callable, plugin_from: str) -> None:
    test_plugin = get_test_plugin(plugin_from, INIT_KWARGS)
    assert test_plugin is not None
    assert test_plugin.name == "ffill"
    assert len(test_plugin.hyperparameter_space()) == 0


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(plugin_from: str, data: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseImputer = get_test_plugin(plugin_from, INIT_KWARGS)
    dataset = get_dataset(data)
    assert dataset.static.dataframe().isna().sum().sum() != 0
    test_plugin.fit(dataset)


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize(
    "covariates_dataset",
    [
        False,
        pytest.param(True, marks=pytest.mark.extra),
    ],
)
def test_transform(
    plugin_from: str,
    data: str,
    covariates_dataset: bool,
    get_test_plugin: Callable,
    get_dataset: Callable,
    as_covariates_dataset: Callable,
) -> None:
    test_plugin: BaseImputer = get_test_plugin(plugin_from, INIT_KWARGS)
    dataset = get_dataset(data)

    if covariates_dataset:
        dataset = as_covariates_dataset(dataset)

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.transform(dataset)

    assert output.time_series.dataframe().isna().sum().sum() == 0
