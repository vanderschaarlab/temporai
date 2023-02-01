# pylint: disable=redefined-outer-name, unused-argument

import numpy as np
import pytest

import tempor.data.utils as utils


@pytest.fixture
def multiindex_timeseries_df(df_time_series_num_nonan):
    return df_time_series_num_nonan


class TestMultiindexTimeseriesDfToArray:
    @pytest.mark.parametrize("padding_indicator", [999.0, np.nan])
    def test_success(self, multiindex_timeseries_df, padding_indicator):
        pad = padding_indicator
        array = utils.multiindex_timeseries_df_to_array(multiindex_timeseries_df, padding_indicator=pad)
        assert isinstance(array, np.ndarray)
        assert array.shape == (3, 4, 2)
        assert (
            np.logical_xor(
                array
                == np.transpose(
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
                ),
                np.isnan(array),
            )
        ).all()

    @pytest.mark.parametrize("padding_indicator", [999.0, np.nan])
    def test_success_extend(self, multiindex_timeseries_df, padding_indicator):
        pad = padding_indicator
        array = utils.multiindex_timeseries_df_to_array(
            multiindex_timeseries_df, padding_indicator=pad, max_timesteps=5
        )
        assert isinstance(array, np.ndarray)
        assert array.shape == (3, 5, 2)
        assert (
            np.logical_xor(
                array
                == np.transpose(
                    np.asarray(
                        [
                            [
                                [11, 12, 13, 14, pad],
                                [1.1, 1.2, 1.3, 1.4, pad],
                            ],
                            [
                                [21, 22, pad, pad, pad],
                                [2.1, 2.2, pad, pad, pad],
                            ],
                            [
                                [31, pad, pad, pad, pad],
                                [3.1, pad, pad, pad, pad],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                np.isnan(array),
            )
        ).all()

    @pytest.mark.parametrize("padding_indicator", [999.0, np.nan])
    def test_success_shrink(self, multiindex_timeseries_df, padding_indicator):
        pad = padding_indicator
        array = utils.multiindex_timeseries_df_to_array(
            multiindex_timeseries_df, padding_indicator=pad, max_timesteps=3
        )
        assert isinstance(array, np.ndarray)
        assert array.shape == (3, 3, 2)
        assert (
            np.logical_xor(
                array
                == np.transpose(
                    np.asarray(
                        [  # pyright: ignore
                            [
                                [11, 12, 13],
                                [1.1, 1.2, 1.3],
                            ],
                            [
                                [21, 22, pad],
                                [2.1, 2.2, pad],
                            ],
                            [
                                [31, pad, pad],
                                [3.1, pad, pad],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                np.isnan(array),
            )
        ).all()

    @pytest.mark.parametrize("padding_indicator", [999.0, np.nan])
    def test_fails_padding_found(self, multiindex_timeseries_df, padding_indicator):
        multiindex_timeseries_df.loc[("a", 2), "f1"] = padding_indicator
        with pytest.raises(ValueError, match=".*padding.*"):
            utils.multiindex_timeseries_df_to_array(multiindex_timeseries_df, padding_indicator=padding_indicator)
