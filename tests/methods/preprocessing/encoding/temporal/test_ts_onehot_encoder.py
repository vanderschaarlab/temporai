# pylint: disable=redefined-outer-name

from typing import Callable, Dict

import pytest

from tempor.methods.preprocessing.encoding import BaseEncoder
from tempor.methods.preprocessing.encoding.temporal.plugin_ts_onehot_encoder import TimeSeriesOneHotEncoder
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
            fqn="preprocessing.encoding.temporal.ts_onehot_encoder",
            cls=TimeSeriesOneHotEncoder,
            kwargs=base_kwargs,
        )

    return func


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
def test_sanity(get_test_plugin: Callable, plugin_from: str) -> None:
    test_plugin = get_test_plugin(plugin_from, INIT_KWARGS)
    assert test_plugin is not None
    assert test_plugin.name == "ts_onehot_encoder"
    assert len(test_plugin.hyperparameter_space()) == 3


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(plugin_from: str, data: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseEncoder = get_test_plugin(plugin_from, INIT_KWARGS)
    dataset = get_dataset(data)
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
    test_plugin: BaseEncoder = get_test_plugin(plugin_from, INIT_KWARGS)
    dataset = get_dataset(data)

    if covariates_dataset:
        dataset = as_covariates_dataset(dataset)

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.transform(dataset)

    assert "categorical_feat_1" not in output.time_series.dataframe().columns.tolist()
    assert "categorical_feat_2" not in output.time_series.dataframe().columns.tolist()

    new_cols = [
        "categorical_feat_1_p",
        "categorical_feat_1_q",
        "categorical_feat_1_r",
        "categorical_feat_2_S",
        "categorical_feat_2_T",
    ]

    for new_col in new_cols:
        assert new_col in output.time_series.dataframe().columns.tolist()

    for new_col in new_cols:
        assert sorted(output.time_series.dataframe()[new_col].unique().tolist()) == [0.0, 1.0]


def test_transform_all_features(get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseEncoder = get_test_plugin("from_api", {"features": None})
    dataset = get_dataset(TEST_ON_DATASETS[0])

    ts_df = dataset.time_series.dataframe()
    ts_df.drop(
        columns=[c for c in ts_df.columns.tolist() if c not in ["categorical_feat_1", "categorical_feat_2"]],
        inplace=True,
    )
    dataset.time_series = dataset.time_series.from_dataframe(ts_df)

    test_plugin.fit(dataset)
    output = test_plugin.transform(dataset)

    assert "categorical_feat_1" not in output.time_series.dataframe().columns.tolist()
    assert "categorical_feat_2" not in output.time_series.dataframe().columns.tolist()

    assert len(output.time_series.dataframe().columns) == 5

    new_cols = [
        "categorical_feat_1_p",
        "categorical_feat_1_q",
        "categorical_feat_1_r",
        "categorical_feat_2_S",
        "categorical_feat_2_T",
    ]

    for new_col in new_cols:
        assert new_col in output.time_series.dataframe().columns.tolist()

    for new_col in new_cols:
        assert sorted(output.time_series.dataframe()[new_col].unique().tolist()) == [0.0, 1.0]
