from typing import Dict, Sequence

import pandas as pd

import tempor.data._settings as settings
import tempor.data._supports_container as sc
import tempor.data._types as types
import tempor.data.container._requirements as dr

from . import impl, interface


class DataValidator(interface.DataValidatorInterface, sc.SupportsContainer[impl.ValidatorImplementation]):
    def _validate(
        self,
        target: types.DataContainer,
        *,
        requirements: Sequence[dr.DataContainerRequirement],
        container_flavor: types.ContainerFlavor,
        **kwargs,
    ) -> types.DataContainer:
        self.check_data_container_supported_types(target)
        if container_flavor is None:
            container_flavor = settings.DEFAULT_CONTAINER_FLAVOR[(self.data_category, type(target))]
        dispatched_to_validator = self.dispatch_to_implementation((type(target), container_flavor))
        return dispatched_to_validator._validate(  # pylint: disable=protected-access
            target, requirements=requirements, container_flavor=container_flavor
        )


class StaticDataValidator(DataValidator, sc.StaticSupportsContainer):
    def _register_implementations(self) -> Dict[sc.DataContainerDef, impl.ValidatorImplementation]:
        df_sample_x_feature_def = (pd.DataFrame, types.ContainerFlavor.DF_SAMPLE_X_FEATURE)
        df_sample_x_feature_impl = impl.df_validator.StaticDataValidator()
        return {
            df_sample_x_feature_def: df_sample_x_feature_impl,
        }


class TimeSeriesDataValidator(DataValidator, sc.TimeSeriesSupportsContainer):
    def _register_implementations(self) -> Dict[sc.DataContainerDef, impl.ValidatorImplementation]:
        df_sample_x_feature_def = (pd.DataFrame, types.ContainerFlavor.DF_SAMPLE_TIMESTEP_X_FEATURE)
        df_sample_x_feature_impl = impl.df_validator.TimeSeriesDataValidator()
        return {
            df_sample_x_feature_def: df_sample_x_feature_impl,
        }


class EventDataValidator(DataValidator, sc.EventSupportsContainer):
    def _register_implementations(self) -> Dict[sc.DataContainerDef, impl.ValidatorImplementation]:
        df_sample_x_time_event_def = (pd.DataFrame, types.ContainerFlavor.DF_SAMPLE_TIME_X_EVENT)
        df_sample_x_time_event_impl = impl.df_validator.EventDataValidator()
        return {
            df_sample_x_time_event_def: df_sample_x_time_event_impl,
        }
