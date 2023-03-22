import pytest

from tempor.data import dataset
from tempor.plugins import plugin_loader
from tempor.plugins.regression import BaseRegressor
from tempor.plugins.regression.plugin_seq2seq_regressor import (
    Seq2seqRegressor as plugin,
)
from tempor.utils.dataloaders.sine import SineDataLoader


def from_api() -> BaseRegressor:
    return plugin_loader.get("regression.seq2seq_regressor", random_state=123, epochs=10)


def from_module() -> BaseRegressor:
    return plugin(random_state=123, epochs=10)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_seq2seq_regressor_plugin_sanity(test_plugin: BaseRegressor) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "seq2seq_regressor"
    assert len(test_plugin.hyperparameter_space()) == 8


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_seq2seq_regressor_plugin_fit(test_plugin: BaseRegressor) -> None:
    raw_data = SineDataLoader().load()
    data = dataset.TemporalPredictionDataset(
        time_series=raw_data.time_series.dataframe(),
        static=raw_data.static.dataframe(),  # type: ignore
        targets=raw_data.time_series.dataframe().copy(),
    )

    test_plugin.fit(data)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_seq2seq_regressor_plugin_predict(test_plugin: BaseRegressor) -> None:
    temporal_dim = 11
    raw_data = SineDataLoader(temporal_dim=temporal_dim).load()
    data = dataset.TemporalPredictionDataset(
        time_series=raw_data.time_series.dataframe(),
        static=raw_data.static.dataframe(),  # type: ignore
        targets=raw_data.time_series.dataframe().copy(),
    )
    test_plugin.fit(data)
    output = test_plugin.predict(data, n_future_steps=10)

    assert output.numpy().shape == (len(raw_data.time_series), 10, temporal_dim)


def test_hyperparam_sample():
    for repeat in range(10):  # pylint: disable=unused-variable
        args = plugin._cls.sample_hyperparameters()  # pylint: disable=no-member, protected-access
        plugin(**args)
