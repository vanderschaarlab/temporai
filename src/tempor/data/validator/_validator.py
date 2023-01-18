from typing import Dict, List

import pandas as pd

import tempor.data as dat
import tempor.data.requirements as r

from . import impl, interface


class DataValidator(interface.DataValidatorInterface, dat.SupportsContainer[impl.ValidatorImplementation]):
    def validate(
        self, data: dat.DataContainer, requirements: List[r.DataRequirement], container_flavor: dat.ContainerFlavor
    ) -> dat.DataContainer:
        self.check_data_container_supported_types(data)
        return self.dispatch_to_implementation((type(data), container_flavor)).validate(
            data, requirements, container_flavor
        )


class StaticDataValidator(DataValidator, dat.StaticSupportsContainer):
    def _register_implementations(self) -> Dict[dat.DataContainerDef, impl.ValidatorImplementation]:
        df_sample_x_feature_def = (pd.DataFrame, dat.ContainerFlavor.DF_SAMPLE_X_FEATURE)
        df_sample_x_feature_impl = impl.df_validator.StaticDataValidator()
        return {
            df_sample_x_feature_def: df_sample_x_feature_impl,
        }


class TimeSeriesDataValidator(DataValidator, dat.TimeSeriesSupportsContainer):
    def _register_implementations(self) -> Dict[dat.DataContainerDef, impl.ValidatorImplementation]:
        df_sample_x_feature_def = (pd.DataFrame, dat.ContainerFlavor.DF_SAMPLE_TIMESTEP_X_FEATURE)
        df_sample_x_feature_impl = impl.df_validator.TimeSeriesDataValidator()
        return {
            df_sample_x_feature_def: df_sample_x_feature_impl,
        }


class EventDataValidator(DataValidator, dat.EventSupportsContainer):
    def _register_implementations(self) -> Dict[dat.DataContainerDef, impl.ValidatorImplementation]:
        df_sample_x_time_event_def = (pd.DataFrame, dat.ContainerFlavor.DF_SAMPLE_TIME_X_EVENT)
        df_sample_x_time_event_impl = impl.df_validator.EventDataValidator()
        return {
            df_sample_x_time_event_def: df_sample_x_time_event_impl,
        }
