import abc
import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Generator, List, Optional, Type

import omegaconf
import optuna
import pydantic
import rich.pretty
from typing_extensions import Self

from tempor.core import pydantic_utils, utils
from tempor.core.plugins import Plugin
from tempor.data import dataset
from tempor.log import logger

from .params import Params


@dataclasses.dataclass
class EmptyParamsDefinition:
    pass


class BaseEstimator(Plugin, abc.ABC):
    ParamsDefinition: ClassVar[Type] = EmptyParamsDefinition
    _fitted: bool

    class _InitArgsValidator(pydantic.BaseModel):
        params: Optional[Dict[str, Any]] = None
        ParamsDefinitionClass: Type

        # Output:
        params_processed: Optional[omegaconf.DictConfig] = None

        @pydantic.model_validator(mode="before")
        def root_checks(cls, values: Dict) -> Dict:  # pylint: disable=no-self-argument
            params = values.get("params", dict())
            ParamsDefinitionClass = values.get("ParamsDefinitionClass")

            if TYPE_CHECKING:  # pragma: no cover
                assert ParamsDefinitionClass is not None  # nosec B101

            try:
                ParamsDefinitionClass = pydantic_utils.make_pydantic_dataclass(ParamsDefinitionClass)
                defined_params = omegaconf.OmegaConf.structured(dataclasses.asdict(ParamsDefinitionClass(**params)))
            except Exception as ex:
                name = utils.get_class_full_name(ex)
                sep = "\n" + "-" * (len(name) + 1) + "\n"
                raise ValueError(
                    f"Model parameters could not be validated as defined by `{ParamsDefinitionClass.__name__}`, "
                    f"cause: {sep}{name}:\n{ex}{sep}"
                ) from ex

            values["params_processed"] = defined_params
            return values

        model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **params: Any) -> None:
        Plugin.__init__(self)
        self._fitted = False
        args_validator = self._InitArgsValidator(params=params, ParamsDefinitionClass=self.ParamsDefinition)
        params_processed = args_validator.params_processed
        if TYPE_CHECKING:  # pragma: no cover
            assert isinstance(params_processed, omegaconf.DictConfig)  # nosec B101
        self.params = params_processed

    @property
    def is_fitted(self) -> bool:
        """Check if the model was trained"""
        return self._fitted

    def __rich_repr__(self) -> Generator:
        yield "name", self.name
        yield "category", self.category
        yield "plugin_type", self.plugin_type
        yield "params", omegaconf.OmegaConf.to_container(self.params)

    def __repr__(self) -> str:
        return rich.pretty.pretty_repr(self)

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def fit(
        self,
        data: dataset.BaseDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        if not data.fit_ready:
            raise ValueError(
                f"The dataset was not fit-ready, check that all necessary data components are present:\n{data}"
            )

        logger.debug(f"Calling _fit() implementation on {self.__class__.__name__}")
        fitted_model = self._fit(data, *args, **kwargs)

        self._fitted = True
        return fitted_model

    @abc.abstractmethod
    def _fit(self, data: dataset.BaseDataset, *args: Any, **kwargs: Any) -> Self:  # pragma: no cover
        ...

    @staticmethod
    @abc.abstractmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # pragma: no cover
        """The hyperparameter search domain, used for tuning."""
        ...  # pylint: disable=unnecessary-ellipsis

    @classmethod
    def sample_hyperparameters(
        cls, *args: Any, override: Optional[List[Params]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Sample hyperparameters. Hyperparameters will be sampled as defined in the ``hyperparameter_space`` static
        method, unless ``override`` is provided, in which case, they will be sampled from the ``override`` definition.

        Can provide variadics ``*args`` and ``**kwargs``, these will be passed on to the ``hyperparameter_space``
        method.

        Note:
            If using `optuna` as the hyperparameter optimizer, an additional argument, ``trial`` (`optuna.Trial`) must
            be passed either as an argument or keyword argument to this method, i.e.
            ``.sample_hyperparameters(trial, ...)`` or ``.sample_hyperparameters(..., trial=trial, ...)``.

        Args:
            override (Optional[List[Params]], optional):
                If this is not `None`, hyperparameters will be sampled from this list, rather than from those defined\
                    in the ``hyperparameter_space`` method. Defaults to `None`.

        Returns:
            Dict[str, Any]: _description_
        """
        # Pop the `trial` argument for optuna if such is found (if not, `trial` will be `None`).
        trial, args, kwargs = utils.get_from_args_or_kwargs(
            args, kwargs, argument_name="trial", argument_type=optuna.Trial, position_if_args=0
        )

        if override is None:
            param_space = cls.hyperparameter_space(*args, **kwargs)
        else:
            param_space = override

        results = dict()

        for hp in param_space:
            results[hp.name] = hp.sample(trial)

        return results
