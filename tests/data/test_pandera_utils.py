import pandas as pd
import pandera as pa
import pytest

from tempor.data.pandera_utils import UnionDtype


class TestUnionDtype:
    @pytest.mark.parametrize(
        "union, df",
        [
            (
                UnionDtype[pa.Int, pa.String],  # type: ignore
                pd.DataFrame(
                    {
                        "a": [1, 2, 3, 4],
                        "b": ["aa", "bb", "cc", "dd"],
                    }
                ),
            ),
            (
                UnionDtype[pa.Int, pa.Float],  # type: ignore
                pd.DataFrame(
                    {
                        "a": [1, 2, 3, 4],
                        "b": [1.1, 2.2, 3.3, 4.4],
                    }
                ),
            ),
            (
                UnionDtype[pa.Category],  # type: ignore
                pd.DataFrame(
                    {
                        "a": pd.Series(["x", "y", "x"], dtype="category"),
                        "b": pd.Series(["a", "a", "b"], dtype="category"),
                    }
                ),
            ),
        ],
    )
    def test_validation_passes_dataframe(self, union, df):
        pa.DataFrameSchema(
            columns={
                # Using regex here to apply the same union dtype to all dataframe columns.
                ".*": pa.Column(
                    dtype=union,
                    coerce=False,
                    regex=True,
                )
            }
        ).validate(df)

    @pytest.mark.parametrize(
        "union, df",
        [
            (
                UnionDtype[pa.Int, pa.Float],  # type: ignore
                pd.DataFrame(
                    {
                        "a": [1, 2, 3, 4],
                        "b": ["aa", "bb", "cc", "dd"],
                    }
                ),
            ),
            (
                UnionDtype[pa.Category],  # type: ignore
                pd.DataFrame(
                    {
                        "a": [1, 2, 3, 4],
                        "b": [1.1, 2.2, 3.3, 4.4],
                    }
                ),
            ),
        ],
    )
    def test_validation_fails_dataframe(self, union, df):
        with pytest.raises(
            pa.errors.SchemaError,  # pyright: ignore
            match=".*" + union.name.replace("[", r"\[").replace("]", r"\]") + ".*",
        ):
            pa.DataFrameSchema(
                columns={
                    # Using regex here to apply the same union dtype to all dataframe columns.
                    ".*": pa.Column(
                        dtype=union,
                        coerce=False,
                        regex=True,
                    )
                }
            ).validate(df)

    @pytest.mark.parametrize(
        "union, series",
        [
            (
                UnionDtype[pa.Int, pa.String],  # type: ignore
                pd.Series(["aa", "bb", "cc"]),
            ),
            (
                UnionDtype[pa.Int, pa.String],  # type: ignore
                pd.Series([1, 2, 3]),
            ),
            (
                UnionDtype[pa.Int],  # type: ignore
                pd.Series([1, 2, 3]),
            ),
            (
                UnionDtype[pa.Float],  # type: ignore
                pd.Series([1.0, 2.0, 3.0]),
            ),
            (
                UnionDtype[pa.Category],  # type: ignore
                pd.Series(["a", "a", "b"], dtype="category"),
            ),
        ],
    )
    def test_validation_passes_series(self, union, series):
        pa.SeriesSchema(union, coerce=False).validate(series)

    @pytest.mark.parametrize(
        "union, series",
        [
            (
                UnionDtype[pa.Int],  # type: ignore
                pd.Series(["aa", "bb", "cc"]),
            ),
        ],
    )
    def test_validation_fails_series(self, union, series):
        with pytest.raises(
            pa.errors.SchemaError,  # pyright: ignore
            match=".*" + union.name.replace("[", r"\[").replace("]", r"\]") + ".*",
        ):
            pa.SeriesSchema(union, coerce=False).validate(series)

    def test_fails_types_not_specified(self):
        with pytest.raises(TypeError):
            pa.SeriesSchema(UnionDtype, coerce=False).validate(pd.Series([1, 2, 3]))

    def test_fails_coerce_not_supported(self):
        with pytest.raises((TypeError, pa.errors.SchemaError)):  # pyright: ignore
            pa.SeriesSchema(UnionDtype[pa.Int], coerce=True).validate(pd.Series(["1", "2", "3"]))

    @pytest.mark.parametrize(
        "union, dtype",
        [
            (UnionDtype[pa.Int, pa.String], pa.String()),  # type: ignore
            (UnionDtype[pa.Int, pa.String], pa.Int()),  # type: ignore
            (UnionDtype[pa.Category], pa.Category()),  # type: ignore
        ],
    )
    def test_validation_passes_dtype_directly(self, union, dtype):
        assert union().check(pandera_dtype=dtype, data_container=None) is True

    @pytest.mark.parametrize(
        "union, dtype",
        [
            (UnionDtype[pa.Int, pa.String], pa.Category()),  # type: ignore
            (UnionDtype[pa.Int, pa.Float], pa.String()),  # type: ignore
            (UnionDtype[pa.Category], pa.Int()),  # type: ignore
        ],
    )
    def test_validation_fails_dtype_directly(self, union, dtype):
        assert union().check(pandera_dtype=dtype, data_container=None) is False
