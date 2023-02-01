import numpy as np
import pandas as pd
import pytest

# Contains common fixtures.


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
