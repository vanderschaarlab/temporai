import abc
import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, overload

import omegaconf
import pydantic
import rich.pretty

import tempor.core.utils
import tempor.data.bundle
import tempor.data.container
import tempor.data.types

# import tempor.data.container._validator
from tempor.core import pydantic_utils
from tempor.log import logger

from . import _requirements_config as rq
from . import _types as types


class MethodArgsValidator(pydantic.BaseModel):
    X: Optional[tempor.data.types.DataContainer]
    Y: Optional[tempor.data.types.DataContainer]
    A: Optional[tempor.data.types.DataContainer]
    Xt: Optional[tempor.data.types.DataContainer]
    Xs: Optional[tempor.data.types.DataContainer]
    Xe: Optional[tempor.data.types.DataContainer]
    Yt: Optional[tempor.data.types.DataContainer]
    Ys: Optional[tempor.data.types.DataContainer]
    Ye: Optional[tempor.data.types.DataContainer]
    At: Optional[tempor.data.types.DataContainer]
    As: Optional[tempor.data.types.DataContainer]
    Ae: Optional[tempor.data.types.DataContainer]

    # Output:
    params: Optional[omegaconf.DictConfig] = None

    @pydantic.root_validator(pre=True)
    def check_exclusive(cls, values):  # pylint: disable=no-self-argument
        incompatible_args_def = {"X": ("Xt", "Xs", "Xe"), "Y": ("Yt", "Ys", "Ye"), "A": ("At", "As", "Ae")}
        for arg, incompatible_args in incompatible_args_def.items():
            for incompatible_arg in incompatible_args:
                pydantic_utils.exclusive_args(
                    values,
                    arg1=arg,
                    arg2=incompatible_arg,
                )
        return values

    @pydantic.root_validator
    def set_values(cls, values: Dict):  # pylint: disable=no-self-argument
        X_provided = values.get("X", None)
        Y_provided = values.get("Y", None)
        A_provided = values.get("A", None)
        if X_provided is not None:
            values["Xt"] = X_provided
        if Y_provided is not None:
            values["Yt"] = Y_provided
        if A_provided is not None:
            values["At"] = A_provided
        return values

    @pydantic.root_validator
    def expect_at_least_one_covariate_container(cls, values: Dict):  # pylint: disable=no-self-argument
        Xt = values.get("Xt", None)
        Xs = values.get("Xs", None)
        Xe = values.get("Xe", None)
        if Xt is None and Xs is None and Xe is None:
            raise ValueError("None of the covariate containers (Xt, Xs, Xe) were provided, at least one is required")
        return values

    class Config:
        arbitrary_types_allowed = True


@dataclasses.dataclass
class EmptyParamsDefinition:
    pass


class TemporBaseModel(abc.ABC):
    PARAMS_DEFINITION: ClassVar[type] = EmptyParamsDefinition
    CONFIG: ClassVar[Dict] = {
        "fit_config": {
            "data_present": ["Xt"],
        },
    }
    _fit_called: bool

    class _InitArgsValidator(pydantic.BaseModel):
        params_as_dict: Optional[Dict[str, Any]]
        params_as_kwargs: Optional[Dict[str, Any]]
        ParamsDefinitionClass: type

        # Output:
        params: Optional[omegaconf.DictConfig] = None

        @pydantic.root_validator(pre=True)
        def check_exclusive(cls, values):  # pylint: disable=no-self-argument
            arg1_value = values.get("params_as_dict", None)
            arg2_value = values.get("params_as_kwargs", dict())
            if arg1_value is not None and arg2_value != dict():
                raise ValueError(
                    "Must provide either the `params` argument or parameters as individual keyword arguments "
                    "but not both"
                )
            return values

        @pydantic.root_validator
        def root_checks(cls, values: Dict):  # pylint: disable=no-self-argument
            params_as_dict = values.get("params_as_dict", None)
            params_as_kwargs = values.get("params_as_kwargs", None)
            params = params_as_dict if params_as_dict else params_as_kwargs

            ParamsDefinitionClass = values.get("ParamsDefinitionClass")

            if TYPE_CHECKING:  # pragma: no cover
                assert params is not None
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

            values["params"] = defined_params
            return values

        class Config:
            arbitrary_types_allowed = True

    class _FitArgsValidator(MethodArgsValidator):
        pass

    @overload
    def __init__(self, *, params: Dict[str, Any]) -> None:  # pragma: no cover
        ...

    @overload
    def __init__(self, **params_as_kwargs) -> None:  # pragma: no cover
        ...

    def __init__(self, *, params: Optional[Dict[str, Any]] = None, **params_as_kwargs) -> None:
        self._fit_called = False
        self.config = rq.RequirementsConfig.parse_obj(self.CONFIG)
        args_validator = self._InitArgsValidator(
            params_as_dict=params,
            params_as_kwargs=params_as_kwargs,
            ParamsDefinitionClass=self.PARAMS_DEFINITION,
        )
        params_processed = args_validator.params
        if TYPE_CHECKING:  # pragma: no cover
            assert isinstance(params_processed, omegaconf.DictConfig)
        self.params = params_processed
        super().__init__()

    def __rich_repr__(self):
        yield "params", omegaconf.OmegaConf.to_container(self.params)

    def __repr__(self) -> str:
        return rich.pretty.pretty_repr(self)

    def _validate_method_config(self, data: tempor.data.bundle.DataBundle, method_type: types.MethodTypes):
        logger.debug(f"Validating method config for method type {method_type} on {self.__class__.__name__}")
        config_name = f"{types.get_method_name(method_type)}_config"
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

    @overload
    def fit(
        self,
        X: Optional[tempor.data.types.DataContainer],
        Y: Optional[tempor.data.types.DataContainer] = None,
        A: Optional[tempor.data.types.DataContainer] = None,
        **kwargs,
    ) -> "TemporBaseModel":  # pragma: no cover
        ...

    @overload
    def fit(
        self,
        *,
        Xt: Optional[tempor.data.types.DataContainer] = None,
        Xs: Optional[tempor.data.types.DataContainer] = None,
        Xe: Optional[tempor.data.types.DataContainer] = None,
        Yt: Optional[tempor.data.types.DataContainer] = None,
        Ys: Optional[tempor.data.types.DataContainer] = None,
        Ye: Optional[tempor.data.types.DataContainer] = None,
        At: Optional[tempor.data.types.DataContainer] = None,
        As: Optional[tempor.data.types.DataContainer] = None,
        Ae: Optional[tempor.data.types.DataContainer] = None,
        **kwargs,
    ) -> "TemporBaseModel":  # pragma: no cover
        ...

    def fit(
        self,
        X: Optional[tempor.data.types.DataContainer] = None,  # Alias for Xt.
        Y: Optional[tempor.data.types.DataContainer] = None,  # Alias for Yt.
        A: Optional[tempor.data.types.DataContainer] = None,  # Alias for At.
        *,
        Xt: Optional[tempor.data.types.DataContainer] = None,
        Xs: Optional[tempor.data.types.DataContainer] = None,
        Xe: Optional[tempor.data.types.DataContainer] = None,
        Yt: Optional[tempor.data.types.DataContainer] = None,
        Ys: Optional[tempor.data.types.DataContainer] = None,
        Ye: Optional[tempor.data.types.DataContainer] = None,
        At: Optional[tempor.data.types.DataContainer] = None,
        As: Optional[tempor.data.types.DataContainer] = None,
        Ae: Optional[tempor.data.types.DataContainer] = None,
        # ---
        # NOTE: Since currently only support one container flavor per container type, this argument is not
        # shown in th public API:
        container_flavor_spec: Optional[tempor.data.types.ContainerFlavorSpec] = None,
        # ---
        **kwargs,
    ) -> "TemporBaseModel":
        logger.debug(f"Validating fit() arguments on {self.__class__.__name__}")
        args_validator = self._FitArgsValidator(
            X=X, Y=Y, A=A, Xt=Xt, Xs=Xs, Xe=Xe, Yt=Yt, Ys=Ys, Ye=Ye, At=At, As=As, Ae=Ae
        )
        if TYPE_CHECKING:  # pragma: no cover
            assert args_validator.Xt is not None
        logger.debug(f"Creating {tempor.data.bundle.DataBundle.__name__} in {self.__class__.__name__} fit() call")
        data = tempor.data.bundle.DataBundle.from_data_containers(
            Xt=args_validator.Xt,
            Xs=args_validator.Xs,
            Xe=args_validator.Xe,
            Yt=args_validator.Yt,
            Ys=args_validator.Ys,
            Ye=args_validator.Ye,
            At=args_validator.At,
            As=args_validator.As,
            Ae=args_validator.Ae,
            container_flavor_spec=container_flavor_spec,
        )
        self._validate_method_config(data, method_type=types.MethodTypes.FIT)
        fitted_model = self._fit(data, **kwargs)
        self._fit_called = True
        return fitted_model

    @abc.abstractmethod
    def _fit(self, data: tempor.data.bundle.DataBundle, **kwargs) -> "TemporBaseModel":  # pragma: no cover
        ...
