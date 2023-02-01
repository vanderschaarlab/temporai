from typing import Tuple

import pandas as pd
import pandera as pa

import tempor.data._types as types
import tempor.data.container._requirements as dr
from tempor.data.settings import DATA_SETTINGS as DS  # For brevity.
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

    @impl.RegisterValidation.register_method_for(dr.ValueDTypes)
    def _(self, target: pd.DataFrame, req: dr.DataContainerRequirement) -> pd.DataFrame:
        assert isinstance(req, dr.ValueDTypes)
        self.schema = pu.add_all_column_checks(
            self.schema,
            dtype=None,
            nullable=True,
            checks_list=[pu.checks.dynamic.values_are_one_of_dtypes(set(req.definition))],
        )
        return self.schema.validate(target)

    @impl.RegisterValidation.register_method_for(dr.AllowMissing)
    def _(self, target: pd.DataFrame, req: dr.DataContainerRequirement) -> pd.DataFrame:
        assert isinstance(req, dr.AllowMissing)
        self.schema = pu.add_all_column_checks(
            self.schema,
            dtype=None,
            nullable=req.definition,
            checks_list=[],
        )
        return self.schema.validate(target)


class StaticDataValidator(DataValidatorDF):
    @property
    def data_category(self) -> types.DataCategory:
        return types.DataCategory.STATIC

    def root_validate(self, target: pd.DataFrame) -> pd.DataFrame:
        schema = pa.infer_schema(target)
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
        schema.validate(target)

        # (Column) values:
        schema = pu.add_all_column_checks(
            schema,
            dtype=None,
            nullable=DS.values_nullable,
            checks_list=[pu.checks.dynamic.values_are_one_of_dtypes(DS.value_dtypes)],
        )
        target = schema.validate(target)

        # Index:
        schema, target = pu.set_up_index(
            schema,
            target,
            name=DS.sample_index_name,
            nullable=DS.sample_index_nullable,
            unique=DS.sample_index_unique,
            checks_list=[pu.checks.dynamic.index_is_one_of_dtypes(DS.sample_index_dtypes)],
        )
        target = schema.validate(target)
        assert isinstance(target, pd.DataFrame)

        logger.debug(f"Final schema:\n{schema}")

        self.schema = schema
        return target


class TimeSeriesDataValidator(DataValidatorDF):
    @property
    def data_category(self) -> types.DataCategory:
        return types.DataCategory.TIME_SERIES

    def root_validate(self, target: pd.DataFrame) -> pd.DataFrame:
        schema = pa.infer_schema(target)
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
        schema.validate(target)

        # (Column) values:
        schema = pu.add_all_column_checks(
            schema,
            dtype=None,
            nullable=DS.values_nullable,
            checks_list=[pu.checks.dynamic.values_are_one_of_dtypes(DS.value_dtypes)],
        )
        schema.validate(target)

        # Index:
        multiindex_unique_def: Tuple[str, ...] = tuple()
        if DS.sample_index_unique:
            multiindex_unique_def = (DS.sample_index_name,)
        if DS.sample_timestep_index_unique:
            multiindex_unique_def = (DS.sample_index_name, DS.time_index_name)
        schema, target = pu.set_up_2level_multiindex(
            schema,
            target,
            names=(DS.sample_index_name, DS.time_index_name),
            nullable=(DS.sample_index_nullable, DS.time_index_nullable),
            unique=multiindex_unique_def,
            checks_list=(
                [pu.checks.dynamic.index_is_one_of_dtypes(DS.feature_index_dtypes)],
                [pu.checks.dynamic.index_is_one_of_dtypes(DS.time_index_dtypes)],
            ),
        )
        target = schema.validate(target)
        assert isinstance(target, pd.DataFrame)

        logger.debug(f"Final schema:\n{schema}")

        self.schema = schema
        return target


class EventDataValidator(DataValidatorDF):
    @property
    def data_category(self) -> types.DataCategory:
        return types.DataCategory.EVENT

    def root_validate(self, target: pd.DataFrame) -> pd.DataFrame:
        schema = pa.infer_schema(target)
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
        schema.validate(target)

        # (Column) values:
        schema = pu.add_all_column_checks(
            schema,
            dtype=None,
            nullable=DS.values_nullable,
            checks_list=[pu.checks.dynamic.values_are_one_of_dtypes(DS.value_dtypes)],
        )
        schema.validate(target)

        # Index:
        multiindex_unique_def: Tuple[str, ...] = tuple()
        if DS.sample_index_unique:
            multiindex_unique_def = (DS.sample_index_name,)
        if DS.sample_timestep_index_unique:
            multiindex_unique_def = (DS.sample_index_name, DS.time_index_name)
        schema, target = pu.set_up_2level_multiindex(
            schema,
            target,
            names=(DS.sample_index_name, DS.time_index_name),
            nullable=(DS.sample_index_nullable, DS.time_index_nullable),
            unique=multiindex_unique_def,
            checks_list=(
                [pu.checks.dynamic.index_is_one_of_dtypes(DS.feature_index_dtypes)],
                [pu.checks.dynamic.index_is_one_of_dtypes(DS.time_index_dtypes)],
            ),
        )
        target = schema.validate(target)
        assert isinstance(target, pd.DataFrame)

        logger.debug(f"Final schema:\n{schema}")

        self.schema = schema
        return target
