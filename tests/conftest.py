import copy
from types import ModuleType
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pytest

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


# --- Reusable data objects. ---


@pytest.fixture
def df_static_cat_num_hasnan():
    # Has categorical and float columns, has null values.
    categories = ["A", "B", "C"]
    np.random.seed(12345)
    size = 10
    df = pd.DataFrame(
        {
            "cat_var_1": pd.Categorical(np.random.choice(categories, size=size)),
            "cat_var_2": pd.Categorical(np.random.choice(categories, size=size)),
            "num_var_1": np.random.uniform(0, 10, size=size),
            "num_var_2": np.random.uniform(20, 30, size=size),
        }
    )
    df.loc[0, "num_var_1"] = np.nan
    return df


@pytest.fixture
def df_time_series_num_nonan():
    df = pd.DataFrame(
        {
            "sample_idx": ["a", "a", "a", "a", "b", "b", "c"],
            "time_idx": [1, 2, 3, 4, 2, 4, 9],
            "f1": [11, 12, 13, 14, 21, 22, 31],
            "f2": [1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 3.1],
        }
    )
    df = df.set_index(keys=["sample_idx", "time_idx"])
    return df


@pytest.fixture
def df_event_num_nonan():
    df = pd.DataFrame(
        {
            "sample_idx": ["a", "b", "c"],
            "time_idx": [1, 2, 2],
            "f1": [True, False, True],
            "f2": [0, 0, 1],
        }
    )
    df = df.set_index(keys=["sample_idx", "time_idx"])
    return df
