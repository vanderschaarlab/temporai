import copy
from types import ModuleType
from typing import TYPE_CHECKING, Any, List, Tuple

if TYPE_CHECKING:
    from tempor.data.dataset import TimeToEventAnalysisDataset

import numpy as np
import pytest

# --- Reusable functions. ---


@pytest.fixture
def get_event0_time_percentiles():
    def func(dataset: "TimeToEventAnalysisDataset", horizon_percentiles: List):
        event0_times = dataset.predictive.targets.split_as_two_dataframes()[0].to_numpy().reshape((-1,))
        return np.quantile(event0_times, horizon_percentiles).tolist()

    return func


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
