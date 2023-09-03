# pylint: disable=redefined-outer-name

from typing import Callable, Dict

import pytest

from tempor.plugins.preprocessing.encoding import BaseEncoder
from tempor.plugins.preprocessing.encoding.static.plugin_static_onehot_encoder import StaticOneHotEncoder
from tempor.utils.serialization import load, save

INIT_KWARGS = {"features": ["categorical_feat_1", "categorical_feat_2"]}
PLUGIN_FROM_OPTIONS = ["from_api", pytest.param("from_module", marks=pytest.mark.extra)]
TEST_ON_DATASETS = [
    "dummy_data_with_categorical_features_small",
    pytest.param("dummy_data_with_categorical_features_full", marks=pytest.mark.extra),
]


@pytest.fixture
def get_test_plugin(get_plugin: Callable):
    def func(plugin_from: str, base_kwargs: Dict):
        return get_plugin(
            plugin_from,
            fqn="preprocessing.encoding.static.static_onehot_encoder",
            cls=StaticOneHotEncoder,
            kwargs=base_kwargs,
        )

    return func


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
def test_sanity(get_test_plugin: Callable, plugin_from: str) -> None:
    test_plugin = get_test_plugin(plugin_from, INIT_KWARGS)
    assert test_plugin is not None
    assert test_plugin.name == "static_onehot_encoder"
    assert len(test_plugin.hyperparameter_space()) == 3


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(plugin_from: str, data: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseEncoder = get_test_plugin(plugin_from, INIT_KWARGS)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)


# @pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")  # Expected: problem with current serialization.
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
    test_plugin: BaseEncoder = get_test_plugin(plugin_from, INIT_KWARGS)
    dataset = get_dataset(data)

    if covariates_dataset:
        dataset = as_covariates_dataset(dataset)

    assert dataset.static is not None  # nosec B101

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.transform(dataset)

    assert "categorical_feat_1" not in output.static.dataframe().columns.tolist()
    assert "categorical_feat_2" not in output.static.dataframe().columns.tolist()

    new_cols = [
        "categorical_feat_1_a",
        "categorical_feat_1_b",
        "categorical_feat_1_c",
        "categorical_feat_2_D",
        "categorical_feat_2_E",
    ]

    for new_col in new_cols:
        assert new_col in output.static.dataframe().columns.tolist()

    for new_col in new_cols:
        assert sorted(output.static.dataframe()[new_col].unique().tolist()) == [0.0, 1.0]
