import copy
from types import ModuleType
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np
import pytest

if TYPE_CHECKING:
    from tempor.data import dataset  # nosec B101

# --- Test utilities. ---


@pytest.fixture(scope="function")
def patch_module(monkeypatch, request):
    """A test utility to reload a module under test, patching certain module-level variables
    and optionally refresh pydantic.

    For reference:
        - https://github.com/streamlit/streamlit/issues/3218#issuecomment-1050647471
        - https://docs.pytest.org/en/6.2.x/fixture.html#adding-finalizers-directly

    Args:
        monkeypatch: pytest monkeypatch object.
        request: pytest request object.

    Returns:
        Callable: configuration function whose parameters should be set downstream.
    """

    def _patch_module(
        main_module: ModuleType,
        module_vars: List[
            Tuple[
                ModuleType,  # Module of variable.
                Any,  # Variable value.
                str,  # Name of variable.
                Any,  # Patch value.
            ]
        ],
        refresh_pydantic: bool = False,
    ):
        """Configuration function for patching a module.

        Args:
            main_module (ModuleType): The module which will be reloaded after patching.
            module_vars (List[Tuple[ModuleType, Any, str, Any]): A list of tuples specifying which module-level
            variables to patch. Each tuple is of form `(<module of variable>, <variable>, <name of variable>,
            <patch value>)`. The variables to patch may not be in `main_module` directly but rather in modules that
            `main_modules` imports.
            refresh_pydantic (bool, optional): Refresh pydantic (e.g. class validators to avoid duplicate registration
            of validators). Defaults to False.
        """
        undo = dict()
        for module, var, var_name, patch_value in module_vars:
            undo[var_name] = copy.deepcopy(var)
            monkeypatch.setattr(module, var_name, patch_value)

        if refresh_pydantic:
            import pydantic

            pydantic.class_validators._FUNCS.clear()  # pylint: disable=protected-access

        import importlib

        importlib.reload(main_module)

        def teardown():
            """Teardown: reset the module variables at the teardown of this fixture, so that patching doesn't affect
            other tests.
            """

            for module, _, var_name, _ in module_vars:
                setattr(module, var_name, undo[var_name])

            if refresh_pydantic:
                pydantic.class_validators._FUNCS.clear()  # pylint: disable=protected-access  # pyright: ignore

            importlib.reload(main_module)

        request.addfinalizer(teardown)

    return _patch_module


# --- Reusable utility functions. ---


@pytest.fixture
def get_event0_time_percentiles():
    def func(data: "dataset.TimeToEventAnalysisDataset", horizon_percentiles: List):
        event0_times = data.predictive.targets.split_as_two_dataframes()[0].to_numpy().reshape((-1,))
        return np.quantile(event0_times, horizon_percentiles).tolist()

    return func


# --- Reusable datasets. ---

# Sine data: full.
@pytest.fixture(scope="session")
def _sine_data_full():
    from tempor.utils import dataloaders

    return dataloaders.SineDataLoader(no=100, temporal_dim=5, random_state=42).load()


@pytest.fixture(scope="function")
def sine_data_full(_sine_data_full: "dataset.OneOffPredictionDataset") -> "dataset.OneOffPredictionDataset":
    # Give each test a copy, just in case.
    return copy.deepcopy(_sine_data_full)


# Sine data: full, missing.
@pytest.fixture(scope="session")
def _sine_data_missing_full():
    from tempor.utils import dataloaders

    return dataloaders.SineDataLoader(no=100, with_missing=True, temporal_dim=5, random_state=42).load()


@pytest.fixture(scope="function")
def sine_data_missing_full(
    _sine_data_missing_full: "dataset.OneOffPredictionDataset",
) -> "dataset.OneOffPredictionDataset":
    return copy.deepcopy(_sine_data_missing_full)


# Sine data: small.
@pytest.fixture(scope="session")
def _sine_data_small(_sine_data_full: "dataset.OneOffPredictionDataset") -> "dataset.OneOffPredictionDataset":
    data, _ = copy.deepcopy(_sine_data_full).train_test_split(
        train_size=6,
        stratify=_sine_data_full.predictive.targets.numpy(),
        random_state=42,
    )
    return data


@pytest.fixture(scope="function")
def sine_data_small(_sine_data_small: "dataset.OneOffPredictionDataset") -> "dataset.OneOffPredictionDataset":
    return copy.deepcopy(_sine_data_small)


# Sine data: small, missing.
@pytest.fixture(scope="session")
def _sine_data_missing_small(
    _sine_data_missing_full: "dataset.OneOffPredictionDataset",
) -> "dataset.OneOffPredictionDataset":
    data, _ = copy.deepcopy(_sine_data_missing_full).train_test_split(
        train_size=6,
        stratify=_sine_data_missing_full.predictive.targets.numpy(),
        random_state=42,
    )
    return data


@pytest.fixture(scope="function")
def sine_data_missing_small(
    _sine_data_missing_small: "dataset.OneOffPredictionDataset",
) -> "dataset.OneOffPredictionDataset":
    return copy.deepcopy(_sine_data_missing_small)


# Sine data: temporal, full.
@pytest.fixture(scope="session")
def _sine_data_temporal(_sine_data_full: "dataset.OneOffPredictionDataset") -> "dataset.TemporalPredictionDataset":
    from tempor.data import dataset

    raw_data, _ = copy.deepcopy(_sine_data_full).train_test_split(
        train_size=6,
        stratify=_sine_data_full.predictive.targets.numpy(),
        random_state=42,
    )
    data = dataset.TemporalPredictionDataset(
        time_series=raw_data.time_series.dataframe(),
        static=raw_data.static.dataframe(),  # type: ignore
        targets=raw_data.time_series.dataframe().copy(),
    )
    return data


# Sine data: temporal, small.
@pytest.fixture(scope="function")
def sine_data_temporal(_sine_data_temporal: "dataset.TemporalPredictionDataset") -> "dataset.TemporalPredictionDataset":
    return copy.deepcopy(_sine_data_temporal)


@pytest.fixture(scope="session")
def _sine_data_temporal_small(
    _sine_data_temporal: "dataset.TemporalPredictionDataset",
) -> "dataset.TemporalPredictionDataset":
    data = copy.deepcopy(_sine_data_temporal)[:6]
    return data


@pytest.fixture(scope="function")
def sine_data_temporal_small(
    _sine_data_temporal_small: "dataset.TemporalPredictionDataset",
) -> "dataset.TemporalPredictionDataset":
    return copy.deepcopy(_sine_data_temporal_small)


# Google stocks data: full.
@pytest.fixture(scope="session")
def _google_stocks_data_full() -> "dataset.OneOffPredictionDataset":
    from tempor.utils import dataloaders

    data = dataloaders.GoogleStocksDataLoader(seq_len=50).load()
    return data


@pytest.fixture(scope="function")
def google_stocks_data_full(
    _google_stocks_data_full: "dataset.OneOffPredictionDataset",
) -> "dataset.OneOffPredictionDataset":
    return copy.deepcopy(_google_stocks_data_full)


# Google stocks data: small.
@pytest.fixture(scope="session")
def _google_stocks_data_small(
    _google_stocks_data_full: "dataset.OneOffPredictionDataset",
) -> "dataset.OneOffPredictionDataset":
    data = copy.deepcopy(_google_stocks_data_full)[:6]
    return data


@pytest.fixture(scope="function")
def google_stocks_data_small(
    _google_stocks_data_small: "dataset.OneOffPredictionDataset",
) -> "dataset.OneOffPredictionDataset":
    return copy.deepcopy(_google_stocks_data_small)


# PBC data: full.
@pytest.fixture(scope="session")
def _pbc_data_full() -> "dataset.TimeToEventAnalysisDataset":
    from tempor.utils import dataloaders

    data = dataloaders.PBCDataLoader().load()
    return data


@pytest.fixture(scope="function")
def pbc_data_full(_pbc_data_full: "dataset.TimeToEventAnalysisDataset") -> "dataset.TimeToEventAnalysisDataset":
    return copy.deepcopy(_pbc_data_full)


# PBC data: small.
@pytest.fixture(scope="session")
def _pbc_data_small(_pbc_data_full: "dataset.TimeToEventAnalysisDataset") -> "dataset.TimeToEventAnalysisDataset":
    si = list(range(len(_pbc_data_full.time_series)))
    np.random.seed(42)
    np.random.shuffle(si)
    _pbc_data_full = copy.deepcopy(_pbc_data_full)[si]
    data = _pbc_data_full[:90]  # As small as feasible without causing convergence errors.
    return data


@pytest.fixture(scope="function")
def pbc_data_small(_pbc_data_small: "dataset.TimeToEventAnalysisDataset") -> "dataset.TimeToEventAnalysisDataset":
    return copy.deepcopy(_pbc_data_small)


# Clairvoyance dummy data: full.
@pytest.fixture(scope="session")
def _clv_data_full() -> "dataset.TemporalTreatmentEffectsDataset":
    from tempor.utils.dataloaders import DummyTemporalTreatmentEffectsDataLoader

    return DummyTemporalTreatmentEffectsDataLoader(
        n_samples=100,
        temporal_covariates_n_features=5,
        temporal_covariates_max_len=11,
        temporal_covariates_missing_prob=0.0,
        static_covariates_n_features=13,
        temporal_treatments_n_features=5,
        temporal_treatments_n_categories=2,
        temporal_targets_n_features=3,
        temporal_targets_n_categories=4,
        random_state=12345,
    ).load()


@pytest.fixture(scope="function")
def clv_data_full(
    _clv_data_full: "dataset.TemporalTreatmentEffectsDataset",
) -> "dataset.TemporalTreatmentEffectsDataset":
    return copy.deepcopy(_clv_data_full)


# Clairvoyance dummy data: small.
@pytest.fixture(scope="session")
def _clv_data_small(
    _clv_data_full: "dataset.TemporalTreatmentEffectsDataset",
) -> "dataset.TemporalTreatmentEffectsDataset":
    data = copy.deepcopy(_clv_data_full)[:10]
    return data


@pytest.fixture(scope="function")
def clv_data_small(
    _clv_data_small: "dataset.TemporalTreatmentEffectsDataset",
) -> "dataset.TemporalTreatmentEffectsDataset":
    return copy.deepcopy(_clv_data_small)


# PKPD data: full.
@pytest.fixture(scope="session")
def _pkpd_data_full() -> "dataset.OneOffTreatmentEffectsDataset":
    from tempor.utils.dataloaders import PKPDDataLoader

    return PKPDDataLoader(
        n_timesteps=30, time_index_treatment_event=25, n_control_samples=50, n_treated_samples=50, random_state=123
    ).load()


@pytest.fixture(scope="function")
def pkpd_data_full(_pkpd_data_full: "dataset.OneOffTreatmentEffectsDataset") -> "dataset.OneOffTreatmentEffectsDataset":
    return copy.deepcopy(_pkpd_data_full)


# PKPD data: small.
@pytest.fixture(scope="session")
def _pkpd_data_small() -> "dataset.OneOffTreatmentEffectsDataset":
    from tempor.utils.dataloaders import PKPDDataLoader

    return PKPDDataLoader(
        n_timesteps=6, time_index_treatment_event=3, n_control_samples=4, n_treated_samples=4, random_state=123
    ).load()


@pytest.fixture(scope="function")
def pkpd_data_small(
    _pkpd_data_small: "dataset.OneOffTreatmentEffectsDataset",
) -> "dataset.OneOffTreatmentEffectsDataset":
    return copy.deepcopy(_pkpd_data_small)
