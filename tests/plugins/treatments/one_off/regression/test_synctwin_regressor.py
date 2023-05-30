# pylint: disable=redefined-outer-name

from typing import TYPE_CHECKING, Callable, Dict
from unittest.mock import Mock

import pandas as pd
import pytest

from tempor.plugins.treatments.one_off.regression.plugin_synctwin_regressor import SyncTwinTreatmentsRegressor

if TYPE_CHECKING:  # pragma: no cover
    from tempor.plugins.treatments.one_off import BaseOneOffTreatmentEffects

INIT_KWARGS = {
    "pretraining_iterations": 3,
    "matching_iterations": 3,
    "inference_iterations": 3,
}
PLUGIN_FROM_OPTIONS = ["from_api", pytest.param("from_module", marks=pytest.mark.extra)]
DEVICES = [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)]
TEST_ON_DATASETS = [
    "pkpd_data_small",
    pytest.param("pkpd_data_full", marks=pytest.mark.extra),
]


@pytest.fixture
def get_test_plugin(get_plugin: Callable):
    def func(plugin_from: str, base_kwargs: Dict, device: str):
        base_kwargs["device"] = device
        return get_plugin(
            plugin_from,
            fqn="treatments.one_off.regression.synctwin_regressor",
            cls=SyncTwinTreatmentsRegressor,
            kwargs=base_kwargs,
        )

    return func


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
def test_sanity(get_test_plugin: Callable, plugin_from: str) -> None:
    test_plugin = get_test_plugin(plugin_from, INIT_KWARGS, device="cpu")
    assert test_plugin is not None
    assert test_plugin.name == "synctwin_regressor"
    assert len(test_plugin.hyperparameter_space()) == 5


def test_fit_first_runtime_error(get_test_plugin: Callable, monkeypatch):
    from tempor.data.dataset import OneOffTreatmentEffectsDataset

    test_plugin = get_test_plugin("from_api", INIT_KWARGS, device="cpu")
    monkeypatch.setattr(test_plugin, "_fit", Mock())

    test_plugin.fit(Mock(OneOffTreatmentEffectsDataset))
    test_plugin.model = None

    with pytest.raises(RuntimeError, match=".*[Ff]it.*first.*"):
        test_plugin.predict_counterfactuals(Mock(OneOffTreatmentEffectsDataset))


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_fit(plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: "BaseOneOffTreatmentEffects" = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
@pytest.mark.parametrize("device", DEVICES)
def test_predict_counterfactuals(
    plugin_from: str, data: str, device: str, get_test_plugin: Callable, get_dataset: Callable
) -> None:
    test_plugin: "BaseOneOffTreatmentEffects" = get_test_plugin(plugin_from, INIT_KWARGS, device=device)
    dataset = get_dataset(data)
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
