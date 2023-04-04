import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.prediction.temporal.regression import BaseTemporalRegressor
from tempor.plugins.prediction.temporal.regression.plugin_seq2seq_regressor import (
    Seq2seqRegressor as plugin,
)
from tempor.utils.serialization import load, save

train_kwargs = {"random_state": 123, "epochs": 5}

TEST_ON_DATASETS = ["sine_data_temporal_small"]


def from_api() -> BaseTemporalRegressor:
    return plugin_loader.get("prediction.temporal.regression.seq2seq_regressor", **train_kwargs)


def from_module() -> BaseTemporalRegressor:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: BaseTemporalRegressor) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "seq2seq_regressor"
    assert len(test_plugin.hyperparameter_space()) == 8


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(test_plugin: BaseTemporalRegressor, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)
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
def test_predict(test_plugin: BaseTemporalRegressor, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.predict(dataset, n_future_steps=10)

    assert output.numpy().shape == (len(dataset.time_series), 10, 5)
