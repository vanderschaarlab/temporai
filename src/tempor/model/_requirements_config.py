import abc
from typing import Dict, Optional, Sequence

import pydantic
import rich.pretty

import tempor.data.settings
import tempor.data.types
from tempor.data.bundle import requirements as br
from tempor.data.container import _requirements as dr

from . import _types as types

DataContainerRequirementsSet = Sequence[dr.DataContainerRequirement]
DataBundleRequirementsSet = Sequence[br.DataBundleRequirement]


class RequirementsSet(pydantic.BaseModel):
    data_container_requirements: DataContainerRequirementsSet
    data_bundle_requirements: DataBundleRequirementsSet

    class Config:
        arbitrary_types_allowed = True


class _ConfigBase:
    arbitrary_types_allowed = False
    extra = "forbid"
    allow_mutation = False


class _RequirementsConfigBase(pydantic.BaseModel, abc.ABC):
    def __rich_repr__(self):
        for k in self.__fields__.keys():
            yield k, getattr(self, k)

    def __repr__(self) -> str:
        return rich.pretty.pretty_repr(self)

    def __str__(self) -> str:
        return self.__repr__()

    class Config(_ConfigBase):
        pass

    @abc.abstractmethod
    def get_data_container_requirements(self) -> DataContainerRequirementsSet:  # pragma: no cover
        ...

    @abc.abstractmethod
    def get_data_bundle_requirements(self) -> DataBundleRequirementsSet:  # pragma: no cover
        ...

    def get_requirements(self) -> RequirementsSet:
        return RequirementsSet(
            data_container_requirements=self.get_data_container_requirements(),
            data_bundle_requirements=self.get_data_bundle_requirements(),
        )


class _DataContainerConfig(_RequirementsConfigBase):
    value_dtypes: Sequence[tempor.data.types.Dtype] = tuple(tempor.data.settings.DATA_SETTINGS.value_dtypes)
    allow_missing: bool = True

    @property
    @abc.abstractmethod
    def data_category(self) -> tempor.data.types.DataCategory:  # pragma: no cover
        ...

    def get_data_container_requirements(self) -> DataContainerRequirementsSet:
        return [dr.ValueDTypes(self.value_dtypes), dr.AllowMissing(self.allow_missing)]

    def get_data_bundle_requirements(self) -> DataBundleRequirementsSet:
        return []


class TimeSeriesDataContainerConfig(_DataContainerConfig):
    @property
    def data_category(self) -> tempor.data.types.DataCategory:
        return tempor.data.types.DataCategory.TIME_SERIES


class StaticDataContainerConfig(_DataContainerConfig):
    @property
    def data_category(self) -> tempor.data.types.DataCategory:
        return tempor.data.types.DataCategory.STATIC


class EventDataContainerConfig(_DataContainerConfig):
    @property
    def data_category(self) -> tempor.data.types.DataCategory:
        return tempor.data.types.DataCategory.EVENT


METHOD_CONFIG_DEFAULTS_DISPATCH = {
    TimeSeriesDataContainerConfig: ("Xt", "Yt", "At"),
    StaticDataContainerConfig: ("Xs", "Ys", "As"),
    EventDataContainerConfig: ("Xe", "Ye", "Ae"),
}


class _MethodConfig(_RequirementsConfigBase):
    data_present: Sequence[tempor.data.types.SamplesAttributes]
    Xt_config: TimeSeriesDataContainerConfig = TimeSeriesDataContainerConfig()
    Yt_config: Optional[TimeSeriesDataContainerConfig] = None
    At_config: Optional[TimeSeriesDataContainerConfig] = None
    Xs_config: Optional[StaticDataContainerConfig] = None
    Ys_config: Optional[StaticDataContainerConfig] = None
    As_config: Optional[StaticDataContainerConfig] = None
    Xe_config: Optional[EventDataContainerConfig] = None
    Ye_config: Optional[EventDataContainerConfig] = None
    Ae_config: Optional[EventDataContainerConfig] = None

    @pydantic.root_validator
    def self_defaults_configs_if_needed(cls, values: Dict):  # pylint: disable=no-self-argument
        for class_, attrs in METHOD_CONFIG_DEFAULTS_DISPATCH.items():
            for attr in attrs:
                if attr in values.get("data_present", []):
                    config_expected = f"{attr}_config"
                    if values.get(config_expected, None) is None:
                        values[config_expected] = class_()  # type: ignore
        return values

    @property
    @abc.abstractmethod
    def method_type(self) -> types.MethodTypes:  # pragma: no cover
        ...

    def get_data_container_requirements(self) -> DataContainerRequirementsSet:
        return []

    def get_data_bundle_requirements(self) -> DataBundleRequirementsSet:
        return [br.DataPresent(self.data_present)]


class FitConfig(_MethodConfig):
    data_present: Sequence[tempor.data.types.SamplesAttributes] = ["Xt"]

    @property
    def method_type(self) -> types.MethodTypes:
        return types.MethodTypes.FIT


class TransformConfig(_MethodConfig):
    @property
    def method_type(self) -> types.MethodTypes:
        return types.MethodTypes.TRANSFORM


class PredictConfig(_MethodConfig):
    @property
    def method_type(self) -> types.MethodTypes:
        return types.MethodTypes.PREDICT


class PredictCounterfactualConfig(_MethodConfig):
    @property
    def method_type(self) -> types.MethodTypes:
        return types.MethodTypes.PREDICT_COUNTERFACTUAL


class RequirementsConfig(_RequirementsConfigBase):
    fit_config: FitConfig = FitConfig()
    transform_config: Optional[TransformConfig] = None
    predict_config: Optional[PredictConfig] = None
    predict_counterfactual_config: Optional[PredictCounterfactualConfig] = None

    def get_data_container_requirements(self) -> DataContainerRequirementsSet:
        return []

    def get_data_bundle_requirements(self) -> DataBundleRequirementsSet:
        return []
