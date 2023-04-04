from typing import TYPE_CHECKING

import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.preprocessing.imputation import BaseImputer
from tempor.plugins.preprocessing.imputation.temporal.plugin_ffill import (
    FFillImputer as plugin,
)
from tempor.utils.serialization import load, save

from ...helpers_preprocessing import as_covariates_dataset

train_kwargs = {"random_state": 123}

TEST_ON_DATASETS = ["sine_data_missing_small"]


def from_api() -> BaseImputer:
    return plugin_loader.get("preprocessing.imputation.temporal.ffill", **train_kwargs)


def from_module() -> BaseImputer:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: BaseImputer) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "ffill"
    assert len(test_plugin.hyperparameter_space()) == 1


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(test_plugin: BaseImputer, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)
    if TYPE_CHECKING:  # pragma: no cover
        assert dataset.static is not None  # nosec B101

    assert dataset.static.dataframe().isna().sum().sum() != 0

    test_plugin.fit(dataset)


@pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")  # Expected: problem with current serialization.
@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize(
    "covariates_dataset",
    [
        False,
        pytest.param(True, marks=pytest.mark.extra),
    ],
)
def test_transform(
    test_plugin: BaseImputer, covariates_dataset: bool, data: str, request: pytest.FixtureRequest
) -> None:
    dataset = request.getfixturevalue(data)

    if covariates_dataset:
        dataset = as_covariates_dataset(dataset)

    if TYPE_CHECKING:  # pragma: no cover
        assert dataset.static is not None  # nosec B101

    assert dataset.static.dataframe().isna().sum().sum() != 0

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.transform(dataset)

    assert output.static.dataframe().isna().sum().sum() == 0
    assert output.time_series.dataframe().isna().sum().sum() == 0
