import abc
import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type

import omegaconf
import pydantic
import rich.pretty
from typing_extensions import Self

import tempor.core.utils
from tempor.data import dataset
from tempor.log import logger

from ._params import Params
from ._plugin import Plugin


@dataclasses.dataclass
class EmptyParamsDefinition:
    pass


class BaseEstimator(Plugin, abc.ABC):
    ParamsDefinition: ClassVar[Type] = EmptyParamsDefinition
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
                assert ParamsDefinitionClass is not None  # nosec B101

            try:
                ParamsDefinitionClass = pydantic.dataclasses.dataclass(ParamsDefinitionClass)
                defined_params = omegaconf.OmegaConf.structured(dataclasses.asdict(ParamsDefinitionClass(**params)))
            except Exception as ex:
                name = tempor.core.utils.get_class_full_name(ex)
                sep = "\n" + "-" * (len(name) + 1) + "\n"
                raise ValueError(
                    f"Model parameters could not be validated as defined by `{ParamsDefinitionClass.__name__}`, "
                    f"cause: {sep}{name}:\n{ex}{sep}"
                ) from ex

            values["params_processed"] = defined_params
            return values

        class Config:
            arbitrary_types_allowed = True

    def __init__(self, **params) -> None:
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

    def __rich_repr__(self):
        yield "name", self.name
        yield "category", self.category
        yield "params", omegaconf.OmegaConf.to_container(self.params)

    def __repr__(self) -> str:
        return rich.pretty.pretty_repr(self)

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> Self:
        logger.debug(f"Calling _fit() implementation on {self.__class__.__name__}")
        fitted_model = self._fit(data, *args, **kwargs)

        self._fitted = True
        return fitted_model

    @abc.abstractmethod
    def _fit(self, data: dataset.Dataset, *args, **kwargs) -> Self:
        ...

    @staticmethod
    @abc.abstractmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # pragma: no cover
        """The hyperparameter search domain, used for tuning."""
        ...  # pylint: disable=unnecessary-ellipsis

    @classmethod
    def sample_hyperparameters(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Sample hyperparameters."""
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample()

        return results
