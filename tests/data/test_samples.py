# pylint: disable=redefined-outer-name

import dataclasses
import re
from typing import List, Tuple
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import tempor.exc
from tempor.data import data_typing, samples

PAD = 999.0


@dataclasses.dataclass
class DFsTest:
    # Static:
    df_static_success: List[pd.DataFrame]
    df_static_fail: List[Tuple[pd.DataFrame, str]]  # (Failing DataFrame, str to match in error msg)
    # TimeSeries:
    df_time_series_success: List[pd.DataFrame]
    df_time_series_fail: List[Tuple[pd.DataFrame, str]]
    # Event:
    df_event_success: List[pd.DataFrame]
    df_event_fail: List[Tuple[pd.DataFrame, str]]


def set_up_dfs_test():

    # --- Static. ---

    categories = ["A", "B", "C"]
    np.random.seed(12345)
    size = 10
    df_s_success_nonan = pd.DataFrame(
        {
            "sample_idx": [f"sample_{x}" for x in range(1, size + 1)],
            "cat_feat_1": pd.Categorical(np.random.choice(categories, size=size)),
            "cat_feat_2": pd.Categorical(np.random.choice(categories, size=size)),
            "num_feat_1": np.random.uniform(0, 10, size=size),
            "num_feat_2": np.random.uniform(20, 30, size=size),
        }
    )
    df_s_success_nonan.set_index("sample_idx", drop=True, inplace=True)

    df_s_success_nan = df_s_success_nonan.copy()
    df_s_success_nan.loc["sample_0", "num_feat_1"] = np.nan

    df_s_fail_index_multiindex = df_s_success_nonan.copy()
    df_s_fail_index_multiindex["sample_index"] = list(df_s_fail_index_multiindex.index)
    df_s_fail_index_multiindex["some_other_index"] = [f"something_{x}" for x in range(100, size + 100)]
    df_s_fail_index_multiindex.set_index(keys=["sample_index", "some_other_index"], drop=True, inplace=True)

    df_s_fail_index_nonunique = df_s_success_nonan.copy()
    df_s_fail_index_nonunique = pd.concat([df_s_fail_index_nonunique] * 2)

    df_s_fail_index_dtype_float = df_s_success_nonan.copy()
    df_s_fail_index_dtype_float["sample_idx"] = list([x + 0.5 for x in range(len(df_s_success_nonan))])
    df_s_fail_index_dtype_float.set_index(keys=["sample_idx"], drop=True, inplace=True)

    df_s_fail_columns_multiindex = pd.DataFrame(
        data=df_s_success_nonan.to_numpy(),
        columns=pd.MultiIndex.from_tuples([("something", x) for x in df_s_success_nonan.columns]),
    )

    df_s_fail_values_dtype_string = df_s_success_nonan.copy()
    df_s_fail_values_dtype_string["cat_feat_1"] = [f"{x}" for x in df_s_success_nonan["cat_feat_1"].to_list()]

    # --- TimeSeries. ---

    df_t_success_nonan = pd.DataFrame(
        {
            "sample_idx": ["a", "a", "a", "a", "b", "b", "c"],
            "time_idx": [1, 2, 3, 4, 2, 4, 9],
            "feat_1": [11, 12, 13, 14, 21, 22, 31],
            "feat_2": [1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 3.1],
        }
    )
    df_t_success_nonan.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)

    df_t_success_nan = df_t_success_nonan.copy()
    df_t_success_nan.loc[("a", 3), "feat_2"] = np.nan

    df_t_fail_index_singleindex = df_t_success_nonan.copy()
    df_t_fail_index_singleindex["sample_idx"] = list(df_t_fail_index_singleindex.index)
    df_t_fail_index_singleindex.set_index(keys=["sample_idx"], drop=True, inplace=True)

    df_t_fail_index_nonunique = df_t_success_nonan.copy()
    df_t_fail_index_nonunique = pd.concat([df_t_fail_index_nonunique] * 2)

    df_t_fail_index_sample_idx_dtype_float = df_t_success_nonan.copy()
    df_t_fail_index_sample_idx_dtype_float["sample_idx"] = list([x + 0.5 for x in range(len(df_t_success_nonan))])
    df_t_fail_index_sample_idx_dtype_float["time_idx"] = list(df_t_success_nonan.index.get_level_values(1))
    df_t_fail_index_sample_idx_dtype_float.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)

    df_s_fail_index_time_idx_dtype_str = df_t_success_nonan.copy()
    df_s_fail_index_time_idx_dtype_str["sample_idx"] = list(
        df_s_fail_index_time_idx_dtype_str.index.get_level_values(0)
    )
    df_s_fail_index_time_idx_dtype_str["time_idx"] = list([f"t_{x}" for x in range(len(df_t_success_nonan))])
    df_s_fail_index_time_idx_dtype_str.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)

    df_t_fail_columns_multiindex = pd.DataFrame(
        data=df_t_success_nonan.to_numpy(),
        columns=pd.MultiIndex.from_tuples([("something", x) for x in df_t_success_nonan.columns]),
    )

    df_t_fail_values_dtype_string = df_t_success_nonan.copy()
    df_t_fail_values_dtype_string["feat_2"] = [f"{x}" for x in df_t_success_nonan["feat_2"].to_list()]

    # --- Event. ---

    df_e_success = pd.DataFrame(
        {
            "sample_idx": [f"sample_{x}" for x in range(1, 3 + 1)],
            "feat_1": [(5, True), (6, False), (3, True)],
            "feat_2": [(1, False), (8, False), (8, True)],
            "feat_3": [
                (pd.to_datetime("2000-01-02"), False),
                (pd.to_datetime("2000-01-03"), True),
                (pd.to_datetime("2000-01-01"), True),
            ],
        },
    )
    df_e_success.set_index("sample_idx", drop=True, inplace=True)

    df_e_fail_index_multiindex = df_e_success.copy()
    df_e_fail_index_multiindex["sample_index"] = list(df_e_success.index)
    df_e_fail_index_multiindex["some_other_index"] = [
        f"something_{x}" for x in range(100, len(df_e_fail_index_multiindex) + 100)
    ]
    df_e_fail_index_multiindex.set_index(keys=["sample_index", "some_other_index"], drop=True, inplace=True)

    df_e_fail_index_nonunique = df_e_success.copy()
    df_e_fail_index_nonunique = pd.concat([df_e_fail_index_nonunique] * 2)

    df_e_fail_index_dtype_float = df_e_success.copy()
    df_e_fail_index_dtype_float["sample_idx"] = list([x + 0.5 for x in range(len(df_e_success))])
    df_e_fail_index_dtype_float.set_index(keys=["sample_idx"], drop=True, inplace=True)

    df_e_fail_columns_multiindex = pd.DataFrame(
        data=df_e_success.to_numpy(),
        columns=pd.MultiIndex.from_tuples([("something", x) for x in df_e_success.columns]),
    )

    df_e_fail_values_non_len2_seq = df_e_success.copy()
    df_e_fail_values_non_len2_seq.loc[:, "feat_4"] = "abc"

    df_e_fail_values_dtype_unexp_event_value = df_e_success.copy()
    df_e_fail_values_dtype_unexp_event_value.loc["sample_2", "feat_2"] = (8, "abc")  # pyright: ignore

    df_e_fail_values_dtype_unexp_event_time = df_e_success.copy()
    df_e_fail_values_dtype_unexp_event_time.loc["sample_2", "feat_2"] = ("abc", False)  # pyright: ignore

    # --- Record all dataframes for testing. ---

    dfs_test = DFsTest(
        df_static_success=[
            df_s_success_nonan,
            df_s_success_nan,
        ],
        df_static_fail=[
            (df_s_fail_index_multiindex, r".*MultiIndex.*index.*not allowed.*"),
            (df_s_fail_index_nonunique, r".*sample_idx.*duplicate.*"),
            (df_s_fail_index_dtype_float, r".*DataFrame.*index.*dtype.*"),
            (df_s_fail_columns_multiindex, r".*MultiIndex.*columns.*not allowed.*"),
            (df_s_fail_values_dtype_string, r".*DataFrame.*dtype.*"),
        ],
        df_time_series_success=[
            df_t_success_nonan,
            df_t_success_nan,
        ],
        df_time_series_fail=[
            (df_t_fail_index_singleindex, r".*must.*MultiIndex.*2 levels.*"),
            (df_t_fail_index_nonunique, r".*sample_idx.*time_idx.*not unique.*"),
            (df_s_fail_index_time_idx_dtype_str, r".*DataFrame.*index.*dtype"),
            (df_t_fail_index_sample_idx_dtype_float, r".*DataFrame.*index.*dtype.*"),
            (df_t_fail_columns_multiindex, r".*MultiIndex.*columns.*not allowed.*"),
            (df_t_fail_values_dtype_string, r".*DataFrame.*dtype.*"),
        ],
        df_event_success=[
            df_e_success,
        ],
        df_event_fail=[
            (df_e_fail_index_multiindex, r".*MultiIndex.*index.*not allowed.*"),
            (df_e_fail_index_nonunique, r".*sample_idx.*duplicate.*"),
            (df_e_fail_index_dtype_float, r".*DataFrame.*index.*dtype.*"),
            (df_e_fail_columns_multiindex, r".*MultiIndex.*columns.*not allowed.*"),
            (df_e_fail_values_non_len2_seq, r".*sequence.*length 2.*"),
            (df_e_fail_values_dtype_unexp_event_value, r".*feat_2.*DataFrame.*dtype.*"),
            (df_e_fail_values_dtype_unexp_event_time, r".*feat_2_time.*DataFrame.*dtype.*"),
        ],
    )

    return dfs_test


dfs_test: DFsTest = set_up_dfs_test()


@pytest.fixture
def df_static() -> pd.DataFrame:
    return dfs_test.df_static_success[0]


@pytest.fixture
def df_time_series() -> pd.DataFrame:
    return dfs_test.df_time_series_success[0]


@pytest.fixture
def df_event() -> pd.DataFrame:
    return dfs_test.df_event_success[0]


@pytest.fixture
def array_static() -> np.ndarray:
    return np.random.uniform(0, 10, size=(4, 3))


@pytest.fixture
def array_event() -> np.ndarray:
    return dfs_test.df_event_success[0].to_numpy()


class TestStaticSamples:
    def test_modality(self, df_static: pd.DataFrame):
        s = samples.StaticSamples(data=df_static)
        assert s.modality == data_typing.DataModality.STATIC

    def test_num_samples(self, df_static: pd.DataFrame):
        s = samples.StaticSamples(data=df_static)
        assert s.num_samples == 10
        assert s.num_samples == len(s)

    def test_num_features(self, df_static: pd.DataFrame):
        s = samples.StaticSamples(data=df_static)
        assert s.num_features == 4

    def test_dataframe(self, df_static: pd.DataFrame):
        s = samples.StaticSamples(data=df_static)
        assert s.dataframe().equals(df_static)

    def test_numpy(self, df_static: pd.DataFrame):
        s = samples.StaticSamples(data=df_static)
        assert (s.numpy() == s._data.to_numpy()).all()  # pylint: disable=protected-access

    def test_from_dataframe(self, df_static: pd.DataFrame):
        s = samples.StaticSamples.from_dataframe(df_static)
        assert s.dataframe().equals(df_static)

    @pytest.mark.parametrize(
        "sample_index, feature_index, expected_sample_index, expected_feature_index",
        [
            (None, None, [0, 1, 2, 3], ["feat_0", "feat_1", "feat_2"]),
            (["s1", "s2", "s3", "s4"], ["f1", "f2", "f3"], ["s1", "s2", "s3", "s4"], ["f1", "f2", "f3"]),
        ],
    )
    def test_from_numpy(
        self, array_static: np.ndarray, sample_index, feature_index, expected_sample_index, expected_feature_index
    ):
        s = samples.StaticSamples.from_numpy(array_static, sample_index=sample_index, feature_index=feature_index)
        df = s._data  # pylint: disable=protected-access
        assert (df.to_numpy() == array_static).all()
        assert list(df.index) == expected_sample_index
        assert list(df.columns) == expected_feature_index

    def test_sample_index(self, df_static: pd.DataFrame):
        s = samples.StaticSamples.from_dataframe(df_static)
        assert s.sample_index() == [f"sample_{x}" for x in range(1, 10 + 1)]

    def test_repr(self, df_static: pd.DataFrame):
        s = samples.StaticSamples(data=df_static)
        assert "StaticSamples with data:" in str(s)
        assert str(s.dataframe()) in str(s)

    @pytest.mark.parametrize("df", dfs_test.df_static_success)
    def test_init_success(self, df: pd.DataFrame):
        samples.StaticSamples(data=df)
        assert df.index.name == "sample_idx"

    @pytest.mark.parametrize("df, match_exc", dfs_test.df_static_fail)
    def test_init_fail(self, df: pd.DataFrame, match_exc: str):
        with pytest.raises(tempor.exc.DataValidationException) as excinfo:
            samples.StaticSamples(data=df)
        assert re.search(match_exc, str(excinfo.getrepr()), re.S | re.IGNORECASE)

    def test_init_array_data_success(self, array_static: np.ndarray):
        samples.StaticSamples(data=array_static)

    def test_short_repr(self, df_static: pd.DataFrame):
        s = samples.StaticSamples.from_dataframe(df_static)
        assert s.short_repr() == "StaticSamples([10, 4])"

    @pytest.mark.parametrize(
        "key, expected_sample_index",
        [
            (0, ["sample_1"]),
            (9, ["sample_10"]),
            ([1, 5, 7], ["sample_2", "sample_6", "sample_8"]),
            (slice(3, 6), ["sample_4", "sample_5", "sample_6"]),
            (slice(7, None), ["sample_8", "sample_9", "sample_10"]),
        ],
    )
    def test_getitem(self, df_static: pd.DataFrame, key, expected_sample_index):
        s = samples.StaticSamples.from_dataframe(df_static)
        s = s[key]
        assert isinstance(s, samples.StaticSamples)
        assert s.sample_index() == expected_sample_index


class TestTimeSeriesSamples:
    def test_modality(self, df_time_series: pd.DataFrame):
        ss = samples.TimeSeriesSamples(data=df_time_series)
        assert ss.modality == data_typing.DataModality.TIME_SERIES

    def test_num_samples(self, df_time_series: pd.DataFrame):
        s = samples.TimeSeriesSamples(data=df_time_series)
        assert s.num_samples == 3
        assert s.num_samples == len(s)

    def test_num_features(self, df_time_series: pd.DataFrame):
        s = samples.TimeSeriesSamples(data=df_time_series)
        assert s.num_features == 2

    def test_dataframe(self, df_time_series: pd.DataFrame):
        s = samples.TimeSeriesSamples(data=df_time_series)
        assert s.dataframe().equals(df_time_series)

    def test_numpy(self, df_time_series: pd.DataFrame):
        s = samples.TimeSeriesSamples(data=df_time_series)

        pad = 999.0
        array = s.numpy(padding_indicator=pad)
        expected_array = np.transpose(
            np.asarray(
                [  # pyright: ignore
                    [
                        [11, 12, 13, 14],
                        [1.1, 1.2, 1.3, 1.4],
                    ],
                    [
                        [21, 22, pad, pad],
                        [2.1, 2.2, pad, pad],
                    ],
                    [
                        [31, pad, pad, pad],
                        [3.1, pad, pad, pad],
                    ],
                ]
            ),
            (0, 2, 1),
        )

        assert (array == expected_array).all()

    def test_from_dataframe(self, df_time_series: pd.DataFrame):
        s = samples.TimeSeriesSamples.from_dataframe(df_time_series)
        assert s.dataframe().equals(df_time_series)

    @pytest.mark.parametrize(
        "array_timeseries, padding_indicator, sample_index, time_indexes, feature_index, expected_df",
        [
            # Case: padded, indexes not provided.
            (
                # array_timeseries:
                np.transpose(
                    np.asarray(
                        [  # pyright: ignore
                            [
                                [11, 12, 13, 14, PAD],
                                [1.1, 1.2, 1.3, 1.4, PAD],
                            ],
                            [
                                [21, 22, PAD, PAD, PAD],
                                [2.1, 2.2, PAD, PAD, PAD],
                            ],
                            [
                                [31, PAD, PAD, PAD, PAD],
                                [3.1, PAD, PAD, PAD, PAD],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                # padding_indicator:
                PAD,
                # sample_index:
                None,
                # time_indexes:
                None,
                # feature_index:
                None,
                # expected_df:
                pd.DataFrame(
                    data={
                        "feat_0": [11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 31.0],
                        "feat_1": [1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 3.1],
                    },
                    index=pd.MultiIndex.from_tuples(
                        [
                            (0, 0),
                            (0, 1),
                            (0, 2),
                            (0, 3),
                            (1, 0),
                            (1, 1),
                            (2, 0),
                        ]
                    ),
                ),
            ),
            # Case: padded, indexes provided.
            (
                # array_timeseries:
                np.transpose(
                    np.asarray(
                        [  # pyright: ignore
                            [
                                [11, 12, PAD],
                                [1.1, 1.2, PAD],
                            ],
                            [
                                [21, PAD, PAD],
                                [2.1, PAD, PAD],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                # padding_indicator:
                PAD,
                # sample_index:
                ["sample_0", "sample_1"],
                # time_indexes:
                [[1.1, 1.2], [2.1]],
                # feature_index:
                ["f1", "f2"],
                # expected_df:
                pd.DataFrame(
                    data={
                        "f1": [11.0, 12.0, 21.0],
                        "f2": [1.1, 1.2, 2.1],
                    },
                    index=pd.MultiIndex.from_tuples(
                        [
                            ("sample_0", 1.1),
                            ("sample_0", 1.2),
                            ("sample_1", 2.1),
                        ]
                    ),
                ),
            ),
            # Case: no padding:
            (
                # array_timeseries:
                np.transpose(
                    np.asarray(
                        [  # pyright: ignore
                            [
                                [11, 12, 13],
                                [1.1, 1.2, 1.3],
                            ],
                            [
                                [21, 22, 23],
                                [2.1, 2.2, 2.3],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                # padding_indicator:
                None,
                # sample_index:
                None,
                # time_indexes:
                None,
                # feature_index:
                None,
                # expected_df:
                pd.DataFrame(
                    data={
                        "feat_0": [11.0, 12.0, 13.0, 21.0, 22.0, 23.0],
                        "feat_1": [1.1, 1.2, 1.3, 2.1, 2.2, 2.3],
                    },
                    index=pd.MultiIndex.from_tuples(
                        [
                            (0, 0),
                            (0, 1),
                            (0, 2),
                            (1, 0),
                            (1, 1),
                            (1, 2),
                        ]
                    ),
                ),
            ),
            # Case: datetime time index:
            (
                # array_timeseries:
                np.transpose(
                    np.asarray(
                        [  # pyright: ignore
                            [
                                [11, 12, 13],
                                [1.1, 1.2, 1.3],
                            ],
                            [
                                [21, PAD, PAD],
                                [2.1, PAD, PAD],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                # padding_indicator:
                PAD,
                # sample_index:
                None,
                # time_indexes:
                [
                    list(pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"])),
                    list(pd.to_datetime(["2002-02-02"])),
                ],
                # feature_index:
                None,
                # expected_df:
                pd.DataFrame(
                    data={
                        "feat_0": [11.0, 12.0, 13.0, 21.0],
                        "feat_1": [1.1, 1.2, 1.3, 2.1],
                    },
                    index=pd.MultiIndex.from_tuples(
                        [
                            (0, pd.Timestamp("2000-01-01")),
                            (0, pd.Timestamp("2000-01-02")),
                            (0, pd.Timestamp("2000-01-03")),
                            (1, pd.Timestamp("2002-02-02")),
                        ]
                    ),
                ),
            ),
        ],
    )
    def test_from_numpy(
        self,
        array_timeseries: np.ndarray,
        padding_indicator,
        sample_index,
        time_indexes,
        feature_index,
        expected_df: pd.DataFrame,
    ):
        s = samples.TimeSeriesSamples.from_numpy(
            array_timeseries,
            padding_indicator=padding_indicator,
            sample_index=sample_index,
            time_indexes=time_indexes,
            feature_index=feature_index,
        )
        df = s._data  # pylint: disable=protected-access
        assert expected_df.equals(df)

    def test_sample_index(self, df_time_series: pd.DataFrame):
        s = samples.TimeSeriesSamples.from_dataframe(df_time_series)
        assert s.sample_index() == ["a", "b", "c"]

    def test_time_indexes(self, df_time_series: pd.DataFrame):
        s = samples.TimeSeriesSamples.from_dataframe(df_time_series)
        assert s.time_indexes() == [[1, 2, 3, 4], [2, 4], [9]]

    def test_time_indexes_as_dict(self, df_time_series: pd.DataFrame):
        s = samples.TimeSeriesSamples.from_dataframe(df_time_series)
        assert s.time_indexes_as_dict() == {"a": [1, 2, 3, 4], "b": [2, 4], "c": [9]}

    def test_num_timesteps(self, df_time_series: pd.DataFrame):
        s = samples.TimeSeriesSamples.from_dataframe(df_time_series)
        assert s.num_timesteps() == [4, 2, 1]

    def test_num_timesteps_as_dict(self, df_time_series: pd.DataFrame):
        s = samples.TimeSeriesSamples.from_dataframe(df_time_series)
        assert s.num_timesteps_as_dict() == {"a": 4, "b": 2, "c": 1}

    @pytest.mark.parametrize(
        "samples, expected",
        [
            (samples.TimeSeriesSamples.from_dataframe(dfs_test.df_time_series_success[0]), False),
            (samples.TimeSeriesSamples.from_numpy(np.ones(shape=(3, 5, 2))), True),
        ],
    )
    def test_num_timesteps_equal(self, samples: samples.TimeSeriesSamples, expected: bool):
        assert samples.num_timesteps_equal() is expected

    def test_repr(self, df_time_series: pd.DataFrame):
        s = samples.TimeSeriesSamples.from_dataframe(df_time_series)
        assert "TimeSeriesSamples with data:" in str(s)
        assert str(s.dataframe()) in str(s)

    @pytest.mark.parametrize("df", dfs_test.df_time_series_success)
    def test_init_success(self, df: pd.DataFrame):
        samples.TimeSeriesSamples(data=df)

    @pytest.mark.parametrize("df, match_exc", dfs_test.df_time_series_fail)
    def test_init_fail(self, df: pd.DataFrame, match_exc: str):
        with pytest.raises(tempor.exc.DataValidationException) as excinfo:
            samples.TimeSeriesSamples(data=df)
        assert re.search(match_exc, str(excinfo.getrepr()), re.S | re.IGNORECASE)

    def test_short_repr(self, df_time_series: pd.DataFrame):
        s = samples.TimeSeriesSamples.from_dataframe(df_time_series)
        assert s.short_repr() == "TimeSeriesSamples([3, *, 2])"

    @pytest.mark.parametrize(
        "key, expected_sample_index",
        [
            (0, ["sample_1"]),
            (4, ["sample_5"]),
            ([0, 2, 4], ["sample_1", "sample_3", "sample_5"]),
            (slice(1, 3), ["sample_2", "sample_3"]),
            (slice(3, None), ["sample_4", "sample_5"]),
        ],
    )
    def test_getitem(self, key, expected_sample_index):
        sample_idxs = [["sample_1"] * 4, ["sample_2"] * 3, ["sample_3"] * 1, ["sample_4"] * 2, ["sample_5"] * 3]
        df = pd.DataFrame(
            {
                "sample_idx": [item for sublist in sample_idxs for item in sublist],  # Flatten.
                "time_idx": list(range(13)),
                "feat_1": [x * 0.5 for x in list(range(13))],
            }
        )
        df.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)

        s = samples.TimeSeriesSamples.from_dataframe(df)
        s = s[key]
        assert isinstance(s, samples.TimeSeriesSamples)
        assert s.sample_index() == expected_sample_index


class TestEventSamples:
    def test_modality(self, df_event: pd.DataFrame):
        s = samples.EventSamples(data=df_event)
        assert s.modality == data_typing.DataModality.EVENT

    def test_num_samples(self, df_event: pd.DataFrame):
        s = samples.EventSamples(data=df_event)
        assert s.num_samples == 3
        assert s.num_samples == len(s)

    def test_num_features(self, df_event: pd.DataFrame):
        s = samples.EventSamples(data=df_event)
        assert s.num_features == 3

    def test_dataframe(self, df_event: pd.DataFrame):
        s = samples.EventSamples(data=df_event)
        assert s.dataframe().equals(df_event)

    def test_numpy(self, df_event: pd.DataFrame):
        s = samples.EventSamples(data=df_event)
        assert (s.numpy() == s._data.to_numpy()).all()  # pylint: disable=protected-access

    def test_from_dataframe(self, df_event: pd.DataFrame):
        s = samples.EventSamples.from_dataframe(df_event)
        assert s.dataframe().equals(df_event)

    @pytest.mark.parametrize(
        "sample_index, feature_index, expected_sample_index, expected_feature_index",
        [
            (None, None, [0, 1, 2], ["feat_0", "feat_1", "feat_2"]),
            (["s1", "s2", "s3"], ["f1", "f2", "f3"], ["s1", "s2", "s3"], ["f1", "f2", "f3"]),
        ],
    )
    def test_from_numpy(
        self, array_event: np.ndarray, sample_index, feature_index, expected_sample_index, expected_feature_index
    ):
        s = samples.EventSamples.from_numpy(array_event, sample_index=sample_index, feature_index=feature_index)
        df = s._data  # pylint: disable=protected-access
        assert (df.to_numpy() == array_event).all()
        assert list(df.index) == expected_sample_index
        assert list(df.columns) == expected_feature_index

    def test_sample_index(self, df_event: pd.DataFrame):
        s = samples.EventSamples.from_dataframe(df_event)
        assert s.sample_index() == [f"sample_{x}" for x in range(1, 3 + 1)]

    def test_repr(self, df_event: pd.DataFrame):
        s = samples.EventSamples.from_dataframe(df_event)
        assert "EventSamples with data:" in str(s)
        assert str(s.dataframe()) in str(s)

    @pytest.mark.parametrize("df", dfs_test.df_event_success)
    def test_init_success(self, df: pd.DataFrame):
        samples.EventSamples(data=df)

    @pytest.mark.parametrize("df, match_exc", dfs_test.df_event_fail)
    def test_init_fail(self, df: pd.DataFrame, match_exc: str):
        with pytest.raises(tempor.exc.DataValidationException) as excinfo:
            samples.EventSamples(data=df)
        assert re.search(match_exc, str(excinfo.getrepr()), re.S | re.IGNORECASE)

    def test_init_array_data_success(self, array_event: np.ndarray):
        samples.EventSamples(data=array_event)

    def test_split(self, df_event: pd.DataFrame):
        s = samples.EventSamples(data=df_event)
        s_split = s.split(time_feature_suffix="_time")

        expected_df = pd.DataFrame(
            {
                "sample_idx": [f"sample_{x}" for x in range(1, 3 + 1)],
                "feat_1_time": [5, 6, 3],
                "feat_1": [True, False, True],
                "feat_2_time": [1, 8, 8],
                "feat_2": [False, False, True],
                "feat_3_time": [
                    pd.to_datetime("2000-01-02"),
                    pd.to_datetime("2000-01-03"),
                    pd.to_datetime("2000-01-01"),
                ],
                "feat_3": [False, True, True],
            },
        )
        expected_df.set_index("sample_idx", drop=True, inplace=True)

        assert s_split.equals(expected_df)

    def test_split_as_two_dataframes(self, df_event: pd.DataFrame):
        s = samples.EventSamples(data=df_event)
        s_split_df_event_time, s_split_df_event_value = s.split_as_two_dataframes(time_feature_suffix="_time")

        expected_df_event_time = pd.DataFrame(
            {
                "sample_idx": [f"sample_{x}" for x in range(1, 3 + 1)],
                "feat_1_time": [5, 6, 3],
                "feat_2_time": [1, 8, 8],
                "feat_3_time": [
                    pd.to_datetime("2000-01-02"),
                    pd.to_datetime("2000-01-03"),
                    pd.to_datetime("2000-01-01"),
                ],
            },
        )
        expected_df_event_time.set_index("sample_idx", drop=True, inplace=True)

        expected_df_event_value = pd.DataFrame(
            {
                "sample_idx": [f"sample_{x}" for x in range(1, 3 + 1)],
                "feat_1": [True, False, True],
                "feat_2": [False, False, True],
                "feat_3": [False, True, True],
            },
        )
        expected_df_event_value.set_index("sample_idx", drop=True, inplace=True)

        assert s_split_df_event_time.equals(expected_df_event_time)
        assert s_split_df_event_value.equals(expected_df_event_value)

    def test_split_fails_column_naming_conflict(self):
        df = pd.DataFrame({"feat_1_time": [(5, True), (6, False), (3, True)]})

        samples.EventSamples.validate = Mock()  # Skip validation.

        s = samples.EventSamples(data=df)

        with pytest.raises(ValueError, match=".*[Cc]olumn.*not contain.*_time.*"):
            s.split(time_feature_suffix="_time")

    def test_short_repr(self, df_event: pd.DataFrame):
        s = samples.EventSamples.from_dataframe(df_event)
        assert s.short_repr() == "EventSamples([3, 3])"

    @pytest.mark.parametrize(
        "key, expected_sample_index",
        [
            (0, ["sample_1"]),
            (9, ["sample_10"]),
            ([1, 5, 7], ["sample_2", "sample_6", "sample_8"]),
            (slice(3, 6), ["sample_4", "sample_5", "sample_6"]),
            (slice(7, None), ["sample_8", "sample_9", "sample_10"]),
        ],
    )
    def test_getitem(self, key, expected_sample_index):
        df = pd.DataFrame(
            {
                "sample_idx": [f"sample_{x}" for x in range(1, 10 + 1)],
                "feat_1": [(x, True) for x in range(1, 10 + 1)],
                "feat_2": [(x + 0.5, False) for x in range(1, 10 + 1)],
            },
        )
        df.set_index("sample_idx", drop=True, inplace=True)
        s = samples.EventSamples.from_dataframe(df)
        s = s[key]
        assert isinstance(s, samples.EventSamples)
        assert s.sample_index() == expected_sample_index
