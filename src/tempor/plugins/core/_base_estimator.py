import abc
import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type

import omegaconf
import pydantic
import rich.pretty

import tempor.core.utils
import tempor.data.bundle
import tempor.data.container
import tempor.data.types
from tempor.data.bundle._bundle import (
    DataBundle as Dataset,  # TODO: To be dealt with later.
)
from tempor.log import logger

from .. import _requirements_config as rq
from . import _types as types
from ._params import Params
from ._plugin import Plugin


@dataclasses.dataclass
class EmptyParamsDefinition:
    pass


class BaseEstimator(Plugin, abc.ABC):
    PARAMS_DEFINITION: ClassVar[Type] = EmptyParamsDefinition
    CONFIG: ClassVar[Dict] = {  # TODO: Simplify / deal with this later.
        "fit_config": {
            "data_present": ["Xt"],
        },
    }
    _fitted: bool

    class _InitArgsValidator(pydantic.BaseModel):
        params: Optional[Dict[str, Any]]
        ParamsDefinitionClass: Type

        # Output:
        params_processed: Optional[omegaconf.DictConfig] = None

        @pydantic.root_validator
        def root_checks(cls, values: Dict):  # pylint: disable=no-self-argument
            params = values.get("params", dict())
            ParamsDefinitionClass = values.get("ParamsDefinitionClass")

            if TYPE_CHECKING:  # pragma: no cover
                assert ParamsDefinitionClass is not None

            try:
                defined_params = omegaconf.OmegaConf.structured(ParamsDefinitionClass(**params))
            except Exception as ex:
                name = tempor.core.utils.get_class_full_name(ex)
                sep = "\n" + "-" * (len(name) + 1) + "\n"
                raise ValueError(
                    "Model parameters could not be converted to OmegaConf Structured Config "
                    f"as defined by `{ParamsDefinitionClass.__name__}`, cause: {sep}{name}:\n{ex}{sep}"
                ) from ex

            values["params_processed"] = defined_params
            return values

        class Config:
            arbitrary_types_allowed = True

    def __init__(self, **params) -> None:
        Plugin.__init__(self)
        self._fitted = False
        self.config = rq.RequirementsConfig.parse_obj(self.CONFIG)
        args_validator = self._InitArgsValidator(params=params, ParamsDefinitionClass=self.PARAMS_DEFINITION)
        params_processed = args_validator.params_processed
        print(params_processed)
        if TYPE_CHECKING:  # pragma: no cover
            assert isinstance(params_processed, omegaconf.DictConfig)
        self.params = params_processed

    @property
    def is_fitted(self) -> bool:
        """Check if the model was trained"""
        return self._fitted

    def __rich_repr__(self):
        yield "name", self.name
        yield "category", self.category
        yield "params", omegaconf.OmegaConf.to_container(self.params)

    def __repr__(self) -> str:
        return rich.pretty.pretty_repr(self)

    def _validate_estimator_method_config(self, data: Dataset, estimator_method: types.EstimatorMethods):
        # TODO: Will simplify all this when dealing with Dataloader.

        logger.debug(f"Validating method config for method type {estimator_method} on {self.__class__.__name__}")
        config_name = f"{tempor.core.utils.get_enum_name(estimator_method)}_config"
        config = getattr(self.config, config_name, None)
        if config is None:
            raise ValueError(f"Expected '{config_name}' to be set in model {self.__class__.__name__} config")
        logger.trace("Validating data bundle requirements")
        tempor.data.bundle.requirements.DataBundleValidator().validate(
            data, requirements=config.get_data_bundle_requirements()
        )
        for t_container_name, ts_samples in data.get_time_series_containers.items():
            logger.trace(f"Validating data container config for {t_container_name}")
            container_config = getattr(config, f"{t_container_name}_config")
            tempor.data.container.requirements.TimeSeriesDataValidator().validate(
                ts_samples.data,
                requirements=container_config.get_data_container_requirements(),
                container_flavor=data.container_flavor_spec[t_container_name],  # type: ignore
            )
        for s_container_name, s_samples in data.get_static_containers.items():
            logger.trace(f"Validating data container config for {s_container_name}")
            container_config = getattr(config, f"{s_container_name}_config")
            tempor.data.container.requirements.StaticDataValidator().validate(
                s_samples.data,
                requirements=container_config.get_data_container_requirements(),
                container_flavor=data.container_flavor_spec[s_container_name],  # type: ignore
            )
        for e_container_name, e_samples in data.get_event_containers.items():
            logger.trace(f"Validating data container config for {e_container_name}")
            container_config = getattr(config, f"{e_container_name}_config")
            tempor.data.container.requirements.EventDataValidator().validate(
                e_samples.data,
                requirements=container_config.get_data_container_requirements(),
                container_flavor=data.container_flavor_spec[e_container_name],  # type: ignore
            )

    def fit(
        self,
        data: Dataset,
        *args,
        **kwargs,
    ) -> "BaseEstimator":
        logger.debug(f"Validating fit() config on {self.__class__.__name__}")
        self._validate_estimator_method_config(data, estimator_method=types.EstimatorMethods.FIT)

        logger.debug(f"Calling _fit() implementation on {self.__class__.__name__}")
        fitted_model = self._fit(data, *args, **kwargs)

        self._fitted = True
        return fitted_model

    @abc.abstractmethod
    def _fit(self, data: Dataset, *args, **kwargs) -> "BaseEstimator":  # pragma: no cover
        ...

    @staticmethod
    @abc.abstractmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # pragma: no cover
        """The hyperparameter search domain, used for tuning."""
        ...  # pylint: disable=unnecessary-ellipsis

    @classmethod
    def sample_hyperparameters(cls, trial: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Sample hyperparameters."""
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample(trial)

        return results
