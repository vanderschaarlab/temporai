# pyright: reportPrivateImportUsage=false
# pylint: disable=protected-access

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from tempor import plugin_loader
from tempor.data import data_typing, samples_experimental

# --- Plugin tests ---


def test_plugins_registered():
    dataformat_plugins = plugin_loader.list("dataformat")
    assert "static_samples_dask" in dataformat_plugins["static_samples"]
    assert "time_series_samples_dask" in dataformat_plugins["time_series_samples"]
    assert "event_samples_dask" in dataformat_plugins["event_samples"]


# --- Static samples tests ---

categories = ["A", "B", "C"]
np.random.seed(12345)
size = 10
df_s = pd.DataFrame(
    {
        "sample_idx": [f"sample_{x}" for x in range(1, size + 1)],
        "cat_feat_1": pd.Categorical(np.random.choice(categories, size=size)),
        "cat_feat_2": pd.Categorical(np.random.choice(categories, size=size)),
        "num_feat_1": np.random.uniform(0, 10, size=size),
        "num_feat_2": np.random.uniform(20, 30, size=size),
    }
)
df_s.set_index("sample_idx", drop=True, inplace=True)
ddf_s = dd.from_pandas(df_s, npartitions=2)


class TestStaticSamplesDask:
    def test_init_ddf(self):
        s = samples_experimental.StaticSamplesDask(ddf_s)  # type: ignore
        assert isinstance(s._data, dd.DataFrame)

    def test_init_df(self):
        s = samples_experimental.StaticSamplesDask(df_s, npartitions=2)
        assert isinstance(s._data, dd.DataFrame)

        s = samples_experimental.StaticSamplesDask(df_s, chunksize=2)
        assert isinstance(s._data, dd.DataFrame)

        s = samples_experimental.StaticSamplesDask(df_s)
        assert isinstance(s._data, dd.DataFrame)

    def test_init_array(self):
        with pytest.raises(NotImplementedError):
            samples_experimental.StaticSamplesDask(np.random.rand(10, 10))

    def test_from_dataframe(self):
        s = samples_experimental.StaticSamplesDask.from_dataframe(df_s, npartitions=2)
        assert isinstance(s._data, dd.DataFrame)

        s = samples_experimental.StaticSamplesDask.from_dataframe(df_s, chunksize=2)
        assert isinstance(s._data, dd.DataFrame)

        s = samples_experimental.StaticSamplesDask.from_dataframe(df_s)
        assert isinstance(s._data, dd.DataFrame)

    def test_from_numpy(self):
        with pytest.raises(NotImplementedError):
            samples_experimental.StaticSamplesDask.from_numpy(np.random.rand(10, 10))

    def test_numpy(self):
        s = samples_experimental.StaticSamplesDask(ddf_s)  # type: ignore
        arr = s.numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (size, 4)

    def test_dataframe(self):
        s = samples_experimental.StaticSamplesDask(ddf_s)  # type: ignore
        df = s.dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (size, 4)

    def test_sample_index(self):
        s = samples_experimental.StaticSamplesDask(ddf_s)  # type: ignore
        assert sorted(s.sample_index()) == sorted(list(df_s.index))

    def test_num_samples(self):
        s = samples_experimental.StaticSamplesDask(ddf_s)  # type: ignore
        assert s.num_samples == size

    def test_num_features(self):
        s = samples_experimental.StaticSamplesDask(ddf_s)  # type: ignore
        assert s.num_features == 4

    def test_short_repr(self):
        s = samples_experimental.StaticSamplesDask(ddf_s)  # type: ignore
        assert s.short_repr() == "StaticSamplesDask([10, 4])"

    def test_getitem(self):
        s = samples_experimental.StaticSamplesDask(ddf_s)  # type: ignore
        s0 = s[0]
        s1_2 = s[1:3]
        assert isinstance(s0, samples_experimental.StaticSamplesDask)  # pyright: ignore
        assert isinstance(s1_2, samples_experimental.StaticSamplesDask)  # pyright: ignore
        assert s0.num_samples == 1
        assert s1_2.num_samples == 2

    def test_len(self):
        s = samples_experimental.StaticSamplesDask(ddf_s)  # type: ignore
        assert len(s) == size

    def test_repr(self):
        s = samples_experimental.StaticSamplesDask(ddf_s)  # type: ignore
        assert "StaticSamplesDask" in repr(s)

    def test_repr_html(self):
        s = samples_experimental.StaticSamplesDask(ddf_s)  # type: ignore
        assert "StaticSamplesDask" in s._repr_html_()

    def test_modality(self):
        s = samples_experimental.StaticSamplesDask(ddf_s)  # type: ignore
        assert s.modality == data_typing.DataModality.STATIC


# --- Time series samples tests ---

df_t = pd.DataFrame(
    {
        "sample_idx": ["a", "a", "a", "a", "b", "b", "c"],
        "time_idx": [1, 2, 3, 4, 2, 4, 9],
        "feat_1": [11, 12, 13, 14, 21, 22, 31],
        "feat_2": [1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 3.1],
    }
)
df_t.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)
ddf_t = samples_experimental.multiindex_df_to_compatible_ddf(df_t, npartitions=2)


class TestTimeSeriesSamplesDask:
    def test_init_ddf(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert isinstance(t._data, dd.DataFrame)

    def test_init_df(self):
        t = samples_experimental.TimeSeriesSamplesDask(df_t, npartitions=2)
        assert isinstance(t._data, dd.DataFrame)

        t = samples_experimental.TimeSeriesSamplesDask(df_t, chunksize=2)
        assert isinstance(t._data, dd.DataFrame)

        t = samples_experimental.TimeSeriesSamplesDask(df_t)
        assert isinstance(t._data, dd.DataFrame)

    def test_init_array(self):
        with pytest.raises(NotImplementedError):
            samples_experimental.TimeSeriesSamplesDask(np.random.rand(10, 10, 10))

    def test_from_dataframe(self):
        t = samples_experimental.TimeSeriesSamplesDask.from_dataframe(df_t, npartitions=2)
        assert isinstance(t._data, dd.DataFrame)

        t = samples_experimental.TimeSeriesSamplesDask.from_dataframe(df_t, chunksize=2)
        assert isinstance(t._data, dd.DataFrame)

        t = samples_experimental.TimeSeriesSamplesDask.from_dataframe(df_t)
        assert isinstance(t._data, dd.DataFrame)

    def test_from_numpy(self):
        with pytest.raises(NotImplementedError):
            samples_experimental.TimeSeriesSamplesDask.from_numpy(np.random.rand(10, 10, 10))

    def test_numpy(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        arr = t.numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 4, 2)

    def test_dataframe(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        df = t.dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (7, 2)
        assert df.index.nlevels == 2

    def test_sample_index(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert sorted(t.sample_index()) == sorted(list(df_t.index.unique(level=0)))

    def test_time_indexes(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert t.time_indexes() == [[1, 2, 3, 4], [2, 4], [9]]

    def test_time_indexes_as_dict(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert t.time_indexes_as_dict() == {"a": [1, 2, 3, 4], "b": [2, 4], "c": [9]}

    def test_time_indexes_float(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        exp = [
            np.asarray([1, 2, 3, 4], dtype=float),
            np.asarray([2, 4], dtype=float),
            np.asarray([9], dtype=float),
        ]
        act = t.time_indexes_float()
        for a, e in zip(act, exp):
            assert (a == e).all()

    def test_num_timesteps(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert t.num_timesteps() == [4, 2, 1]

    def test_num_timesteps_as_dict(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert t.num_timesteps_as_dict() == {"a": 4, "b": 2, "c": 1}

    def test_num_timesteps_equal(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert not t.num_timesteps_equal()

    def test_list_of_dataframes(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        dfs = t.list_of_dataframes()
        assert isinstance(dfs, list)
        assert len(dfs) == 3
        assert isinstance(dfs[0], pd.DataFrame)
        assert dfs[0].shape == (4, 2)

    def test_num_samples(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert t.num_samples == 3

    def test_num_features(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert t.num_features == 2

    def test_short_repr(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert t.short_repr() == "TimeSeriesSamplesDask([3, *, 2])"

    def test_getitem(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        t0 = t[0]
        t1_2 = t[1:3]
        assert isinstance(t0, samples_experimental.TimeSeriesSamplesDask)  # pyright: ignore
        assert isinstance(t1_2, samples_experimental.TimeSeriesSamplesDask)  # pyright: ignore
        assert t0.num_samples == 1
        assert t1_2.num_samples == 2

    def test_len(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert len(t) == 3

    def test_repr(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert "TimeSeriesSamplesDask" in repr(t)

    def test_repr_html(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert "TimeSeriesSamplesDask" in t._repr_html_()

    def test_modality(self):
        t = samples_experimental.TimeSeriesSamplesDask(ddf_t)  # type: ignore
        assert t.modality == data_typing.DataModality.TIME_SERIES


# --- Event samples tests ---

df_e = pd.DataFrame(
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
df_e.set_index("sample_idx", drop=True, inplace=True)
ddf_e = dd.from_pandas(df_e, npartitions=2)


class TestEventSamplesDask:
    def test_init_ddf(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        assert isinstance(e._data, dd.DataFrame)

    def test_init_df(self):
        e = samples_experimental.EventSamplesDask(df_e, npartitions=2)
        assert isinstance(e._data, dd.DataFrame)

        e = samples_experimental.EventSamplesDask(df_e, chunksize=2)
        assert isinstance(e._data, dd.DataFrame)

        e = samples_experimental.EventSamplesDask(df_e)
        assert isinstance(e._data, dd.DataFrame)

    def test_init_array(self):
        with pytest.raises(NotImplementedError):
            samples_experimental.EventSamplesDask(np.random.rand(10, 10))

    def test_from_dataframe(self):
        e = samples_experimental.EventSamplesDask.from_dataframe(df_e, npartitions=2)
        assert isinstance(e._data, dd.DataFrame)

        e = samples_experimental.EventSamplesDask.from_dataframe(df_e, chunksize=2)
        assert isinstance(e._data, dd.DataFrame)

        e = samples_experimental.EventSamplesDask.from_dataframe(df_e)
        assert isinstance(e._data, dd.DataFrame)

    def test_from_numpy(self):
        with pytest.raises(NotImplementedError):
            samples_experimental.EventSamplesDask.from_numpy(np.random.rand(10, 10))

    def test_numpy(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        arr = e.numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 3)

    def test_dataframe(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        df = e.dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)

    def test_sample_index(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        assert sorted(e.sample_index()) == sorted(list(df_e.index))

    def test_num_samples(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        assert e.num_samples == 3

    def test_num_features(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        assert e.num_features == 3

    def test_short_repr(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        assert e.short_repr() == "EventSamplesDask([3, 3])"

    def test_getitem(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        e0 = e[0]
        e1_2 = e[1:3]
        assert isinstance(e0, samples_experimental.EventSamplesDask)  # pyright: ignore
        assert isinstance(e1_2, samples_experimental.EventSamplesDask)  # pyright: ignore
        assert e0.num_samples == 1
        assert e1_2.num_samples == 2

    def test_len(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        assert len(e) == 3

    def test_repr(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        assert "EventSamplesDask" in repr(e)

    def test_repr_html(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        assert "EventSamplesDask" in e._repr_html_()

    def test_modality(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        assert e.modality == data_typing.DataModality.EVENT

    def test_split_as_two_dataframes(self):
        e = samples_experimental.EventSamplesDask(ddf_e)  # type: ignore
        df1, df2 = e.split_as_two_dataframes()
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        assert df1.shape == (3, 3)
        assert df2.shape == (3, 3)
