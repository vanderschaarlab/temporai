# pylint: disable=redefined-outer-name, unused-argument

from typing import Any, List, Tuple
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import tempor.data.utils as utils


@pytest.fixture
def multiindex_timeseries_df():
    df_t = pd.DataFrame(
        {
            "sample_idx": ["a", "a", "a", "a", "b", "b", "c"],
            "time_idx": [1, 2, 3, 4, 2, 4, 9],
            "feat_1": [11, 12, 13, 14, 21, 22, 31],
            "feat_2": [1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 3.1],
        }
    )
    df_t.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)
    return df_t


PAD = 999.0


class TestMultiindexTimeseriesDataframeToArray3d:
    @pytest.mark.parametrize("padding_indicator", [PAD, np.nan])
    def test_success(self, multiindex_timeseries_df, padding_indicator):
        pad = padding_indicator
        array = utils.multiindex_timeseries_dataframe_to_array3d(multiindex_timeseries_df, padding_indicator=pad)
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
        array = utils.multiindex_timeseries_dataframe_to_array3d(
            multiindex_timeseries_df, padding_indicator=pad, max_timesteps=5
        )
        assert isinstance(array, np.ndarray)
        assert array.shape == (3, 5, 2)

        # Check, element-wise, that `array` element is equal to the expected array element, or (in case padding is a
        # `nan`) that the `array` has a `nan` at the element location.
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
        array = utils.multiindex_timeseries_dataframe_to_array3d(
            multiindex_timeseries_df, padding_indicator=pad, max_timesteps=3
        )
        assert isinstance(array, np.ndarray)
        assert array.shape == (3, 3, 2)

        # Check, element-wise, that `array` element is equal to the expected array element, or (in case padding is a
        # `nan`) that the `array` has a `nan` at the element location.
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
            utils.multiindex_timeseries_dataframe_to_array3d(
                multiindex_timeseries_df, padding_indicator=padding_indicator
            )


class TestCheckBoolArray1dTruesConsecutive:
    @pytest.mark.parametrize(
        "array,match_exc",
        [
            (np.ones(shape=(2, 3)).astype(bool), ".*1d.*"),
            (np.asarray([1, 2, 3]), ".*bool.*"),
        ],
    )
    def test_input_validation_exception(self, array: np.ndarray, match_exc: str):
        with pytest.raises(ValueError, match=match_exc):
            utils.check_bool_array1d_trues_consecutive(array)

    @pytest.mark.parametrize(
        "array,expected",
        [
            (np.asarray([False, True, True, True, False]), True),
            (np.asarray([False, True, False]), True),
            (np.asarray([False, True, True, False, True, True, False]), False),
            (np.asarray([False, True, False, True, False]), False),
        ],
    )
    def test_base_case(self, array: np.ndarray, expected: bool):
        assert utils.check_bool_array1d_trues_consecutive(array) == expected

    @pytest.mark.parametrize(
        "array,expected",
        [
            (np.asarray([], dtype=bool), True),  # Empty array.
            (np.asarray([False, False, False]), True),  # No True.
            (np.asarray([True]), True),  # Single element, True.
            (np.asarray([False]), True),  # Single element, False.
        ],
    )
    def test_corner_case_arrays(self, array: np.ndarray, expected: bool):
        assert utils.check_bool_array1d_trues_consecutive(array) == expected

    @pytest.mark.parametrize(
        "array,expected",
        [
            (np.asarray([False, True, True, True, False]), False),
            (np.asarray([True, True, True, True, False]), True),
            (np.asarray([True, True, True, False, True, True, False]), False),
            (np.asarray([True]), True),
            (np.asarray([False]), True),  # Expected.
        ],
    )
    def test_with_at_beginning(self, array: np.ndarray, expected: bool):
        assert utils.check_bool_array1d_trues_consecutive(array, at_beginning=True) == expected

    @pytest.mark.parametrize(
        "array,expected",
        [
            (np.asarray([False, True, True, True, False]), False),
            (np.asarray([False, True, True, True, True]), True),
            (np.asarray([False, True, True, False, True, True, True]), False),
            (np.asarray([True]), True),
            (np.asarray([False]), True),  # Expected.
        ],
    )
    def test_with_at_end(self, array: np.ndarray, expected: bool):
        assert utils.check_bool_array1d_trues_consecutive(array, at_end=True) == expected

    @pytest.mark.parametrize(
        "array,expected",
        [
            (np.asarray([True, True, True]), True),
            (np.asarray([True, False, True]), False),
        ],
    )
    def test_with_at_beginning_and_end(self, array: np.ndarray, expected: bool):
        assert utils.check_bool_array1d_trues_consecutive(array, at_beginning=True, at_end=True) == expected


class TestCheckBoolArray2dIdenticalAlongDim1:
    @pytest.mark.parametrize(
        "array,match_exc",
        [
            (np.ones(shape=(2,)).astype(bool), ".*2d.*"),
            (np.ones(shape=(2, 3, 2)).astype(bool), ".*2d.*"),
            (np.asarray([[1, 9, 2], [2, 3, 3]]).T, ".*bool.*"),
        ],
    )
    def test_input_validation_exception(self, array: np.ndarray, match_exc: str):
        with pytest.raises(ValueError, match=match_exc):
            utils.check_bool_array2d_identical_along_dim1(array)

    @pytest.mark.parametrize(
        "array,expected",
        [
            (np.asarray([[True, True, False], [True, True, False]]).T, True),
            (np.asarray([[True, True, False], [True, False, False]]).T, False),
            (np.asarray([[True, True, False], [True, True, False], [True, True, False]]).T, True),
            (np.asarray([[True, True, False], [True, False, False], [True, False, True]]).T, False),
        ],
    )
    def test_base_case(self, array: np.ndarray, expected: bool):
        assert utils.check_bool_array2d_identical_along_dim1(array) == expected

    @pytest.mark.parametrize(
        "array,expected",
        [
            (np.asarray([[], []], dtype=bool), True),  # Empty array.
            (np.asarray([[False, True, False]]).T, True),  # Only one long along dim 1.
            (np.asarray([[False]]).T, True),  # Only one long along dim 1 & one element.
            (np.asarray([[False], [False]]).T, True),  # One element per, False.
            (np.asarray([[True], [True]]).T, True),  # One element per, False.
            (np.asarray([[True], [False]]).T, False),  # One element per, mixed.
        ],
    )
    def test_corner_case_arrays(self, array: np.ndarray, expected: bool):
        assert utils.check_bool_array2d_identical_along_dim1(array) == expected


class TestGetArray1dLengthUntilPadding:
    @pytest.mark.parametrize(
        "array,padding_indicator,match_exc",
        [
            (
                np.random.randint(0, 5, size=(2, 3, 2)),
                PAD,
                ".*1d.*",
            ),
            (
                np.asarray([1.5, 2.5, np.nan]),
                np.nan,
                ".*[Pp]adding.*nan.*",
            ),
        ],
    )
    def test_input_validation_exception(self, array: np.ndarray, padding_indicator: Any, match_exc: str):
        with pytest.raises(ValueError, match=match_exc):
            utils.get_array1d_length_until_padding(array, padding_indicator=padding_indicator)

    @pytest.mark.parametrize(
        "array,expected",
        [
            (np.asarray([1, 8, -3, PAD, PAD]), 3),
            (np.asarray([1, 8, -3, 9, PAD]), 4),
            (np.asarray([1, 8, -3]), 3),
        ],
    )
    def test_base_case(self, array: np.ndarray, expected: int):
        length = utils.get_array1d_length_until_padding(array, padding_indicator=PAD)
        assert length == expected

    @pytest.mark.parametrize(
        "array,expected",
        [
            (np.asarray([]), 0),  # Empty.
            (np.asarray([PAD, PAD, PAD]), 0),  # All padding.
            (np.asarray([1]), 1),  # Single element, not padding.
            (np.asarray([PAD]), 0),  # Single element, padding.
        ],
    )
    def test_corner_case_arrays(self, array: np.ndarray, expected: int):
        length = utils.get_array1d_length_until_padding(array, padding_indicator=PAD)
        assert length == expected


class TestValidateTimeseriesArray3d:
    @pytest.mark.parametrize(
        "array,padding_indicator",
        [
            (np.ones(shape=(3, 5, 2)), None),
            (np.ones(shape=(3, 5, 2)), PAD),
        ],
    )
    def test_passes(self, array: np.ndarray, padding_indicator):
        utils.validate_timeseries_array3d(array, padding_indicator=padding_indicator)

    @pytest.mark.parametrize(
        "array,padding_indicator",
        [
            (np.ones(shape=(3, 5, 0)), None),
            (np.ones(shape=(3, 5, 0)), PAD),
        ],
    )
    def test_fails_0_feature_dim(self, array: np.ndarray, padding_indicator):
        with pytest.raises(ValueError, match=".*feature dim.*"):
            utils.validate_timeseries_array3d(array, padding_indicator=padding_indicator)

    @pytest.mark.parametrize("padding_indicator", [PAD, None])
    @pytest.mark.parametrize("array", [np.ones(shape=(1,)), np.ones(shape=(3, 5)), np.ones(shape=(3, 5, 2, 8))])
    def test_fails_not_3dim(self, array: np.ndarray, padding_indicator):
        with pytest.raises(ValueError, match="3d"):
            utils.validate_timeseries_array3d(array, padding_indicator=padding_indicator)

    def test_fails_unsupported_padding(self):
        array = np.ones(shape=(3, 5, 2))
        padding_indicator = np.nan
        with pytest.raises(ValueError, match=".*[Pp]adding.*nan.*"):
            utils.validate_timeseries_array3d(array, padding_indicator=padding_indicator)


class TestGetSeqLengthsTimeseriesArray3d:
    @pytest.mark.parametrize(
        "array,expected",
        [
            (np.ones(shape=(3, 5, 2)), [5, 5, 5]),
            (np.ones(shape=(4, 2, 1)), [2, 2, 2, 2]),
            (np.ones(shape=(3, 0, 1)), [0, 0, 0]),  # Edge case: zero timesteps.
        ],
    )
    def test_padding_not_provided_case(self, array: np.ndarray, expected: List[int], monkeypatch):
        mock_validate_timeseries_array3d = Mock(side_effect=utils.validate_timeseries_array3d)
        monkeypatch.setattr(utils, "validate_timeseries_array3d", mock_validate_timeseries_array3d)
        # ^ To confirm input array validation function is always called.
        # Also call it as side effect, to make sure it doesn't raise exceptions in these cases.

        lengths = utils.get_seq_lengths_timeseries_array3d(array, padding_indicator=None)

        mock_validate_timeseries_array3d.assert_called_once()
        assert lengths == expected

    @pytest.mark.parametrize(
        "array,padding_indicator,expected",
        [
            # Typical case:
            (
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
                PAD,
                [4, 2, 1],
            ),
            # No padding case:
            (
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
                PAD,
                [3, 3],
            ),
            # All padding case:
            (
                np.transpose(
                    np.asarray(
                        [  # pyright: ignore
                            [
                                [PAD, PAD, PAD, PAD],
                                [PAD, PAD, PAD, PAD],
                            ],
                            [
                                [PAD, PAD, PAD, PAD],
                                [PAD, PAD, PAD, PAD],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                PAD,
                [0, 0],
            ),
            # Single sample case:
            (
                np.transpose(
                    np.asarray(
                        [
                            [  # pyright: ignore
                                [21, 22, PAD, PAD],
                                [2.1, 2.2, PAD, PAD],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                PAD,
                [2],
            ),
        ],
    )
    def test_padding_provided_case_success(
        self, array: np.ndarray, padding_indicator: Any, expected: List[int], monkeypatch
    ):
        mock_validate_timeseries_array3d = Mock()
        monkeypatch.setattr(utils, "validate_timeseries_array3d", mock_validate_timeseries_array3d)

        lengths = utils.get_seq_lengths_timeseries_array3d(array, padding_indicator=padding_indicator)

        mock_validate_timeseries_array3d.assert_called_once()
        assert lengths == expected

    @pytest.mark.parametrize(
        "array,padding_indicator,expected",
        [
            # No samples case:
            (
                np.ones(shape=(0, 5, 3)),
                PAD,
                [],
            ),
            # No timesteps case:
            (
                np.ones(shape=(4, 0, 3)),
                PAD,
                [0, 0, 0, 0],
            ),
            # NOTE: No features case will raise an exception, see TestValidateTimeseriesArray3d.
        ],
    )
    def test_padding_provided_case_success_edge_cases(
        self, array: np.ndarray, padding_indicator: Any, expected: List[int], monkeypatch
    ):
        mock_validate_timeseries_array3d = Mock()
        monkeypatch.setattr(utils, "validate_timeseries_array3d", mock_validate_timeseries_array3d)

        lengths = utils.get_seq_lengths_timeseries_array3d(array, padding_indicator=padding_indicator)

        mock_validate_timeseries_array3d.assert_called_once()
        assert lengths == expected

    @pytest.mark.parametrize(
        "array,padding_indicator,match_exc",
        [
            # Case: padding indicated differently between features.
            (
                np.transpose(
                    np.asarray(
                        [
                            [  # pyright: ignore
                                [11, 12, 13, PAD, PAD],
                                [1.1, 1.2, 1.3, 1.4, PAD],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                PAD,
                ".*indicate.*",
            ),
            # Case: padding not correctly formatted.
            (
                np.transpose(
                    np.asarray(
                        [
                            [  # pyright: ignore
                                [11, 12, PAD, 14, PAD],
                                [1.1, 1.2, PAD, 1.4, PAD],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                PAD,
                ".*consecutive.*end.*",
            ),
        ],
    )
    def test_padding_provided_case_fails_padding_not_defined_correctly(
        self, array: np.ndarray, padding_indicator: Any, match_exc: str
    ):
        with pytest.raises(ValueError, match=match_exc):
            utils.get_seq_lengths_timeseries_array3d(array, padding_indicator=padding_indicator)


class TestUnpadTimeseriesArray3d:
    @pytest.mark.parametrize(
        "array,padding_indicator,expected",
        [
            # Typical case:
            (
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
                PAD,
                [
                    np.asarray(
                        [
                            [11, 12, 13, 14],
                            [1.1, 1.2, 1.3, 1.4],
                        ]
                    ).T,
                    np.asarray(
                        [
                            [21, 22],
                            [2.1, 2.2],
                        ]
                    ).T,
                    np.asarray(
                        [
                            [31],
                            [3.1],
                        ]
                    ).T,
                ],
            ),
            # No padding case:
            (
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
                PAD,
                [
                    np.asarray(
                        [
                            [11, 12, 13],
                            [1.1, 1.2, 1.3],
                        ]
                    ).T,
                    np.asarray(
                        [
                            [21, 22, 23],
                            [2.1, 2.2, 2.3],
                        ]
                    ).T,
                ],
            ),
            # All padding case:
            (
                np.transpose(
                    np.asarray(
                        [  # pyright: ignore
                            [
                                [PAD, PAD, PAD, PAD],
                                [PAD, PAD, PAD, PAD],
                            ],
                            [
                                [PAD, PAD, PAD, PAD],
                                [PAD, PAD, PAD, PAD],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                PAD,
                [
                    np.ones(shape=(0, 2), dtype=float),
                    np.ones(shape=(0, 2), dtype=float),
                ],
            ),
            # Single sample case:
            (
                np.transpose(
                    np.asarray(
                        [
                            [  # pyright: ignore
                                [21, 22, PAD, PAD],
                                [2.1, 2.2, PAD, PAD],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                PAD,
                [
                    np.asarray(
                        [
                            [21, 22],
                            [2.1, 2.2],
                        ]
                    ).T,
                ],
            ),
        ],
    )
    def test_success(self, array: np.ndarray, padding_indicator: Any, expected: List[np.ndarray], monkeypatch):
        mock_validate_timeseries_array3d = Mock()
        monkeypatch.setattr(utils, "validate_timeseries_array3d", mock_validate_timeseries_array3d)

        unpadded = utils.unpad_timeseries_array3d(array, padding_indicator=padding_indicator)

        mock_validate_timeseries_array3d.assert_called()
        assert len(unpadded) == len(expected)
        for sample_i in range(len(expected)):
            assert (unpadded[sample_i] == expected[sample_i]).all()

    @pytest.mark.parametrize(
        "array,padding_indicator,expected",
        [
            # No samples case:
            (
                np.ones(shape=(0, 5, 3)),
                PAD,
                [],
            ),
            # No timesteps case:
            (
                np.ones(shape=(4, 0, 3)),
                PAD,
                [
                    np.ones(shape=(0, 3), dtype=float),
                    np.ones(shape=(0, 3), dtype=float),
                    np.ones(shape=(0, 3), dtype=float),
                    np.ones(shape=(0, 3), dtype=float),
                ],
            ),
            # NOTE: No features case will raise an exception, see TestValidateTimeseriesArray3d.
        ],
    )
    def test_success_edge_cases(
        self, array: np.ndarray, padding_indicator: Any, expected: List[np.ndarray], monkeypatch
    ):
        mock_validate_timeseries_array3d = Mock()
        monkeypatch.setattr(utils, "validate_timeseries_array3d", mock_validate_timeseries_array3d)

        unpadded = utils.unpad_timeseries_array3d(array, padding_indicator=padding_indicator)

        mock_validate_timeseries_array3d.assert_called()
        assert len(unpadded) == len(expected)
        for sample_i in range(len(expected)):
            assert (unpadded[sample_i] == expected[sample_i]).all()


class TestMakeSampleTimeIndexTuples:
    @pytest.mark.parametrize(
        "sample_index,time_indexes,expected",
        [
            # Typical case:
            (
                ["s1", "s2"],
                [[1, 2, 3], [1, 5, 9, 10]],
                [("s1", 1), ("s1", 2), ("s1", 3), ("s2", 1), ("s2", 5), ("s2", 9), ("s2", 10)],
            ),
            # Single time indexes case:
            (
                ["s1"],
                [[]],
                [],
            ),
            # Single item case:
            (
                ["s1"],
                [[1]],
                [("s1", 1)],
            ),
            # Empty lists case:
            (
                [],
                [],
                [],
            ),
        ],
    )
    def test_success(self, sample_index: List, time_indexes: List[List], expected: List[Tuple]):
        assert utils.make_sample_time_index_tuples(sample_index, time_indexes) == expected

    def test_fails_len_mismatch(self):
        sample_index = ["s1", "s2"]
        time_indexes = [[1, 2, 3]]
        with pytest.raises(ValueError, match=".*elements.*"):
            utils.make_sample_time_index_tuples(sample_index, time_indexes)  # pyright: ignore


class TestArray3dToMultiindexTimeseriesDataframe:
    @pytest.mark.parametrize(
        "array,sample_index,time_indexes,expected",
        [
            # Typical case:
            (
                # array:
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
                # sample_index:
                ["s1", "s2"],
                # time_indexes:
                [[1, 2, 3], [10, 11, 15]],
                # expected:
                pd.DataFrame(
                    {"feat_0": [11.0, 12.0, 13.0, 21.0, 22.0, 23.0], "feat_1": [1.1, 1.2, 1.3, 2.1, 2.2, 2.3]},
                    index=pd.MultiIndex.from_tuples(
                        [("s1", 1), ("s1", 2), ("s1", 3), ("s2", 10), ("s2", 11), ("s2", 15)]
                    ),
                ),
            ),
            # Single sample case:
            (
                # array:
                np.transpose(
                    np.asarray(
                        [
                            [  # pyright: ignore
                                [11, 12],
                                [1.1, 1.2],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                # sample_index:
                ["s1"],
                # time_indexes:
                [[1, 100]],
                # expected:
                pd.DataFrame(
                    {"feat_0": [11.0, 12.0], "feat_1": [1.1, 1.2]},
                    index=pd.MultiIndex.from_tuples([("s1", 1), ("s1", 100)]),
                ),
            ),
        ],
    )
    def test_success_no_padding(
        self,
        array: np.ndarray,
        sample_index: List,
        time_indexes: List[List],
        expected: pd.DataFrame,
        monkeypatch,
    ):
        mock_validate_timeseries_array3d = Mock()
        monkeypatch.setattr(utils, "validate_timeseries_array3d", mock_validate_timeseries_array3d)
        feature_index = [f"feat_{x}" for x in range(array.shape[2])]

        df = utils.array3d_to_multiindex_timeseries_dataframe(
            array,
            sample_index=sample_index,
            time_indexes=time_indexes,
            feature_index=feature_index,
            padding_indicator=None,
        )

        mock_validate_timeseries_array3d.assert_called()
        print(df)
        print(expected)
        assert df.equals(expected)

    @pytest.mark.parametrize(
        "array,sample_index,time_indexes,padding_indicator,expected",
        [
            # Typical case:
            (
                # array:
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
                        ]
                    ),
                    (0, 2, 1),
                ),
                # sample_index:
                ["s1", "s2"],
                # time_indexes:
                [[1, 2, 3, 4], [10, 20]],
                # padding_indicator:
                PAD,
                # expected:
                pd.DataFrame(
                    {"feat_0": [11.0, 12.0, 13.0, 14.0, 21.0, 22.0], "feat_1": [1.1, 1.2, 1.3, 1.4, 2.1, 2.2]},
                    index=pd.MultiIndex.from_tuples(
                        [("s1", 1), ("s1", 2), ("s1", 3), ("s1", 4), ("s2", 10), ("s2", 20)]
                    ),
                ),
            ),
            # Single sample case:
            (
                # array:
                np.transpose(
                    np.asarray(
                        [
                            [  # pyright: ignore
                                [21, 22, PAD, PAD, PAD],
                                [2.1, 2.2, PAD, PAD, PAD],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                # sample_index:
                ["s1"],
                # time_indexes:
                [[1, 2]],
                # padding_indicator:
                PAD,
                # expected:
                pd.DataFrame(
                    {"feat_0": [21.0, 22.0], "feat_1": [2.1, 2.2]},
                    index=pd.MultiIndex.from_tuples([("s1", 1), ("s1", 2)]),
                ),
            ),
            # All padding sample case:
            (
                # array:
                np.transpose(
                    np.asarray(
                        [  # pyright: ignore
                            [
                                [11, 12, PAD],
                                [1.1, 1.2, PAD],
                            ],
                            [
                                [PAD, PAD, PAD],
                                [PAD, PAD, PAD],
                            ],
                        ]
                    ),
                    (0, 2, 1),
                ),
                # sample_index:
                ["s1", "s2"],
                # time_indexes:
                [[1, 2], []],
                # padding_indicator:
                PAD,
                # expected:
                pd.DataFrame(
                    {"feat_0": [11.0, 12.0], "feat_1": [1.1, 1.2]},
                    index=pd.MultiIndex.from_tuples([("s1", 1), ("s1", 2)]),
                ),
            ),
        ],
    )
    def test_success_with_padding(
        self,
        array: np.ndarray,
        sample_index: List,
        time_indexes: List[List],
        padding_indicator: Any,
        expected: pd.DataFrame,
        monkeypatch,
    ):
        mock_validate_timeseries_array3d = Mock()
        monkeypatch.setattr(utils, "validate_timeseries_array3d", mock_validate_timeseries_array3d)
        feature_index = [f"feat_{x}" for x in range(array.shape[2])]

        df = utils.array3d_to_multiindex_timeseries_dataframe(
            array,
            sample_index=sample_index,
            time_indexes=time_indexes,
            feature_index=feature_index,
            padding_indicator=padding_indicator,
        )

        mock_validate_timeseries_array3d.assert_called()
        assert df.equals(expected)
