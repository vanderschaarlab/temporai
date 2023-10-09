# pylint: disable=redefined-outer-name

import copy
from typing import Any, Callable, Dict

import pytest
from typing_extensions import get_args

from tempor.methods.preprocessing.imputation import BaseImputer, TabularImputerType
from tempor.methods.preprocessing.imputation.temporal.plugin_ts_tabular_imputer import TemporalTabularImputer
from tempor.utils.serialization import load, save

INIT_KWARGS = {"random_state": 123}
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
            fqn="preprocessing.imputation.temporal.ts_tabular_imputer",
            cls=TemporalTabularImputer,
            kwargs=base_kwargs,
        )

    return func


@pytest.mark.filterwarnings("ignore:.*validate_arguments.*:DeprecationWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*conflict.*:UserWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
def test_sanity(get_test_plugin: Callable, plugin_from: str) -> None:
    test_plugin = get_test_plugin(plugin_from, INIT_KWARGS)
    assert test_plugin is not None
    assert test_plugin.name == "ts_tabular_imputer"
    assert len(test_plugin.hyperparameter_space()) == 1


@pytest.mark.filterwarnings("ignore:.*validate_arguments.*:DeprecationWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*conflict.*:UserWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(plugin_from: str, data: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseImputer = get_test_plugin(plugin_from, INIT_KWARGS)
    dataset = get_dataset(data)
    assert dataset.time_series.dataframe().isna().sum().sum() != 0
    test_plugin.fit(dataset)


@pytest.mark.filterwarnings("ignore:.*validate_arguments.*:DeprecationWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*conflict.*:UserWarning")  # Exp pydantic2 warns from HI.
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


@pytest.mark.filterwarnings("ignore:.*validate_arguments.*:DeprecationWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*conflict.*:UserWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*nonzero.*0d.*:DeprecationWarning")  # Expected for EM imputer.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # Expected for EM imputer.
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")  # May happen in some cases.
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("imputer", list(get_args(TabularImputerType)))
def test_imputer_types(data: str, imputer: TabularImputerType, get_test_plugin: Callable, get_dataset: Callable):
    dataset = get_dataset(data)
    assert dataset.time_series.dataframe().isna().sum().sum() != 0

    # To speed up the test:
    kwargs: Dict[str, Any] = copy.copy(INIT_KWARGS)
    if imputer == "hyperimpute":
        kwargs["imputer_params"] = {"n_inner_iter": 3}
    if imputer == "missforest":
        kwargs["imputer_params"] = {"max_iter": 10}
    if imputer == "sinkhorn":
        kwargs["imputer_params"] = {"n_epochs": 2}

    p: BaseImputer = get_test_plugin("from_module", dict(imputer=imputer, **kwargs))
    transformed = p.fit_transform(data=dataset)

    assert transformed.time_series.dataframe().isna().sum().sum() == 0  # pyright: ignore


@pytest.mark.filterwarnings("ignore:.*validate_arguments.*:DeprecationWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*conflict.*:UserWarning")  # Exp pydantic2 warns from HI.
def test_random_state(get_test_plugin: Callable):
    p: BaseImputer = get_test_plugin("from_module", INIT_KWARGS)
    assert p.params.random_state == INIT_KWARGS["random_state"]
    assert p.params.imputer_params["random_state"] == INIT_KWARGS["random_state"]

    with pytest.raises(ValueError, match=".*[Dd]o not pass.*random_state.*"):
        p = get_test_plugin("from_module", dict(imputer_params={"random_state": 12345}))
