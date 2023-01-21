from typing import Tuple

import pandas as pd
import pandera as pa

import tempor.data as dat
import tempor.data.requirements as r
from tempor.data import DATA_SETTINGS as DS  # For brevity.
from tempor.log import logger

from . import _impl as impl
from . import pandera_utils as pu


class DataValidatorDF(impl.ValidatorImplementation):
    @property
    def schema(self):
        return self._validation_records["schema"]

    @schema.setter
    def schema(self, value):
        self._validation_records["schema"] = value

    @impl.RegisterValidation.register_method_for(r.ValueDTypes)
    def _(self, data: pd.DataFrame, req: r.DataRequirement) -> pd.DataFrame:
        assert isinstance(req, r.ValueDTypes)
        self.schema = pu.add_all_column_checks(
            self.schema,
            dtype=None,
            nullable=True,
            checks_list=[pu.checks.dynamic.values_are_one_of_dtypes(set(req.value_dtypes))],
        )
        return self.schema.validate(data)

    @impl.RegisterValidation.register_method_for(r.AllowMissing)
    def _(self, data: pd.DataFrame, req: r.DataRequirement) -> pd.DataFrame:
        assert isinstance(req, r.AllowMissing)
        self.schema = pu.add_all_column_checks(
            self.schema,
            dtype=None,
            nullable=req.allow_missing,
            checks_list=[],
        )
        return self.schema.validate(data)


class StaticDataValidator(DataValidatorDF):
    @property
    def data_category(self) -> dat.DataCategory:
        return dat.DataCategory.STATIC

    def root_validate(self, data: pd.DataFrame) -> pd.DataFrame:
        schema = pa.infer_schema(data)
        assert isinstance(schema, pa.DataFrameSchema)
        logger.debug(f"Inferred schema:\n{schema}")

        # DF-wide:
        schema = pu.add_df_wide_checks(
            schema,
            checks_list=[
                pu.checks.forbid_multiindex_index,
                pu.checks.forbid_multiindex_columns,
                pu.checks.dynamic.column_index_is_one_of_dtypes(
                    DS.feature_index_dtypes, nullable=DS.feature_index_nullable
                ),
            ],
        )
        schema.validate(data)

        # (Column) values:
        schema = pu.add_all_column_checks(
            schema,
            dtype=None,
            nullable=DS.values_nullable,
            checks_list=[pu.checks.dynamic.values_are_one_of_dtypes(DS.value_dtypes)],
        )
        data = schema.validate(data)

        # Index:
        schema, data = pu.set_up_index(
            schema,
            data,
            name=DS.sample_index_name,
            nullable=DS.sample_index_nullable,
            unique=DS.sample_index_unique,
            checks_list=[pu.checks.dynamic.index_is_one_of_dtypes(DS.sample_index_dtypes)],
        )
        data = schema.validate(data)
        assert isinstance(data, pd.DataFrame)

        logger.debug(f"Final schema:\n{schema}")

        self.schema = schema
        return data


class TimeSeriesDataValidator(DataValidatorDF):
    @property
    def data_category(self) -> dat.DataCategory:
        return dat.DataCategory.TIME_SERIES

    def root_validate(self, data: pd.DataFrame) -> pd.DataFrame:
        schema = pa.infer_schema(data)
        assert isinstance(schema, pa.DataFrameSchema)
        logger.debug(f"Inferred schema:\n{schema}")

        # DF-wide:
        schema = pu.add_df_wide_checks(
            schema,
            checks_list=[
                pu.checks.forbid_multiindex_columns,
                pu.checks.require_2level_multiindex_index,
                pu.checks.dynamic.column_index_is_one_of_dtypes(
                    DS.feature_index_dtypes, nullable=DS.feature_index_nullable
                ),
            ],
        )
        schema.validate(data)

        # (Column) values:
        schema = pu.add_all_column_checks(
            schema,
            dtype=None,
            nullable=DS.values_nullable,
            checks_list=[pu.checks.dynamic.values_are_one_of_dtypes(DS.value_dtypes)],
        )
        schema.validate(data)

        # Index:
        multiindex_unique_def: Tuple[str, ...] = tuple()
        if DS.sample_index_unique:
            multiindex_unique_def = (DS.sample_index_name,)
        if DS.sample_timestep_index_unique:
            multiindex_unique_def = (DS.sample_index_name, DS.time_index_name)
        schema, data = pu.set_up_2level_multiindex(
            schema,
            data,
            names=(DS.sample_index_name, DS.time_index_name),
            nullable=(DS.sample_index_nullable, DS.time_index_nullable),
            unique=multiindex_unique_def,
            checks_list=(
                [pu.checks.dynamic.index_is_one_of_dtypes(DS.feature_index_dtypes)],
                [pu.checks.dynamic.index_is_one_of_dtypes(DS.time_index_dtypes)],
            ),
        )
        data = schema.validate(data)
        assert isinstance(data, pd.DataFrame)

        logger.debug(f"Final schema:\n{schema}")

        self.schema = schema
        return data


class EventDataValidator(DataValidatorDF):
    @property
    def data_category(self) -> dat.DataCategory:
        return dat.DataCategory.EVENT

    def root_validate(self, data: pd.DataFrame) -> pd.DataFrame:
        schema = pa.infer_schema(data)
        assert isinstance(schema, pa.DataFrameSchema)
        logger.debug(f"Inferred schema:\n{schema}")

        # DF-wide:
        schema = pu.add_df_wide_checks(
            schema,
            checks_list=[
                pu.checks.forbid_multiindex_columns,
                pu.checks.require_2level_multiindex_index,
                pu.checks.require_2level_multiindex_one_to_one,  # NOTE this.
                pu.checks.dynamic.column_index_is_one_of_dtypes(
                    DS.feature_index_dtypes, nullable=DS.feature_index_nullable
                ),
            ],
        )
        schema.validate(data)

        # (Column) values:
        schema = pu.add_all_column_checks(
            schema,
            dtype=None,
            nullable=DS.values_nullable,
            checks_list=[pu.checks.dynamic.values_are_one_of_dtypes(DS.value_dtypes)],
        )
        schema.validate(data)

        # Index:
        multiindex_unique_def: Tuple[str, ...] = tuple()
        if DS.sample_index_unique:
            multiindex_unique_def = (DS.sample_index_name,)
        if DS.sample_timestep_index_unique:
            multiindex_unique_def = (DS.sample_index_name, DS.time_index_name)
        schema, data = pu.set_up_2level_multiindex(
            schema,
            data,
            names=(DS.sample_index_name, DS.time_index_name),
            nullable=(DS.sample_index_nullable, DS.time_index_nullable),
            unique=multiindex_unique_def,
            checks_list=(
                [pu.checks.dynamic.index_is_one_of_dtypes(DS.feature_index_dtypes)],
                [pu.checks.dynamic.index_is_one_of_dtypes(DS.time_index_dtypes)],
            ),
        )
        data = schema.validate(data)
        assert isinstance(data, pd.DataFrame)

        logger.debug(f"Final schema:\n{schema}")

        self.schema = schema
        return data
