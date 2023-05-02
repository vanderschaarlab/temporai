from typing import TYPE_CHECKING

import pytest
from typing_extensions import get_args

from tempor.plugins import plugin_loader
from tempor.plugins.preprocessing.imputation import BaseImputer, TabularImputerType
from tempor.plugins.preprocessing.imputation.static.plugin_static_tabular_imputer import StaticTabularImputer as plugin
from tempor.utils.serialization import load, save

from ...helpers_preprocessing import as_covariates_dataset

train_kwargs = {"random_state": 123}

TEST_ON_DATASETS = ["sine_data_missing_small"]


def from_api() -> BaseImputer:
    return plugin_loader.get("preprocessing.imputation.static.static_tabular_imputer", **train_kwargs)


def from_module() -> BaseImputer:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: BaseImputer) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "static_tabular_imputer"
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

    dataset = request.getfixturevalue(data)

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.transform(dataset)

    assert output.static.dataframe().isna().sum().sum() == 0


@pytest.mark.filterwarnings("ignore:.*nonzero.*0d.*:DeprecationWarning")  # Expected for EM imputer.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # Expected for EM imputer.
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("imputer", list(get_args(TabularImputerType)))
def test_imputer_types(imputer: TabularImputerType, data: str, request: pytest.FixtureRequest):
    dataset = request.getfixturevalue(data)
    assert dataset.static.dataframe().isna().sum().sum() != 0

    p = plugin(imputer=imputer, **train_kwargs)
    transformed = p.fit_transform(data=dataset)

    assert transformed.static.dataframe().isna().sum().sum() == 0  # pyright: ignore


def test_random_state():
    p = plugin(**train_kwargs)
    assert p.params.random_state == train_kwargs["random_state"]
    assert p.params.imputer_params["random_state"] == train_kwargs["random_state"]

    with pytest.raises(ValueError, match=".*[Dd]o not pass.*random_state.*"):
        p = plugin(imputer_params={"random_state": 12345})
