import pandas as pd
import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.treatments import BaseTreatments
from tempor.plugins.treatments.plugin_synctwin_regressor import (
    SyncTwinTreatmentsRegressor as plugin,
)

train_kwargs = {
    "pretraining_iterations": 3,
    "matching_iterations": 3,
    "inference_iterations": 3,
}

TEST_ON_DATASETS = ["pkpd_data_small"]


def from_api() -> BaseTreatments:
    return plugin_loader.get("treatments.synctwin_regressor", **train_kwargs)


def from_module() -> BaseTreatments:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: BaseTreatments) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "synctwin_regressor"
    assert len(test_plugin.hyperparameter_space()) == 5


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(test_plugin: BaseTreatments, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)
    test_plugin.fit(dataset)


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_predict_counterfactuals(test_plugin: BaseTreatments, data: str, request: pytest.FixtureRequest) -> None:
    dataset = request.getfixturevalue(data)
    test_plugin.fit(dataset)

    output = test_plugin.predict_counterfactuals(dataset)

    assert len(output) == len(dataset)
    for o in output:
        assert isinstance(o, (list, str))
        if isinstance(o, list):
            assert len(o) == 1
            assert isinstance(o[0], pd.DataFrame)
        else:
            assert "SyncTwin implementation can currently only predict counterfactuals for treated samples" in o
