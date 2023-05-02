import pytest

from tempor.plugins import plugin_loader
from tempor.plugins.preprocessing.scaling import BaseScaler
from tempor.plugins.preprocessing.scaling.static.plugin_static_minmax_scaler import StaticMinMaxScaler as plugin
from tempor.utils.serialization import load, save

from ...helpers_preprocessing import as_covariates_dataset

train_kwargs = {"random_state": 123}

TEST_ON_DATASETS = ["google_stocks_data_full", "sine_data_scaled_small"]
TEST_TRANSFORM_ON_DATASETS = ["sine_data_scaled_small"]


def from_api() -> BaseScaler:
    return plugin_loader.get("preprocessing.scaling.static.static_minmax_scaler", **train_kwargs)


def from_module() -> BaseScaler:
    return plugin(**train_kwargs)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_sanity(test_plugin: BaseScaler) -> None:
    assert test_plugin is not None
    assert test_plugin.name == "static_minmax_scaler"
    assert len(test_plugin.hyperparameter_space()) == 0


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        pytest.param(from_module(), marks=pytest.mark.extra),
    ],
)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(test_plugin: BaseScaler, data: str, request: pytest.FixtureRequest) -> None:
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
@pytest.mark.parametrize("data", TEST_TRANSFORM_ON_DATASETS)
@pytest.mark.parametrize(
    "covariates_dataset",
    [
        False,
        pytest.param(True, marks=pytest.mark.extra),
    ],
)
def test_transform(
    test_plugin: BaseScaler, covariates_dataset: bool, data: str, request: pytest.FixtureRequest
) -> None:
    dataset = request.getfixturevalue(data)

    if covariates_dataset:
        dataset = as_covariates_dataset(dataset)

    assert dataset.static is not None  # nosec B101
    assert (dataset.static.numpy() > 1.1).any()

    dump = save(test_plugin)
    reloaded = load(dump)

    reloaded.fit(dataset)

    dump = save(reloaded)
    reloaded = load(dump)

    output = reloaded.transform(dataset)

    assert (output.static.numpy() < 1 + 1e-1).all()
    assert (output.static.numpy() >= 0).all()
