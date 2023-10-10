import dataclasses
from typing import Any, Dict, List

from typing_extensions import Self, get_args

from tempor.core import plugins
from tempor.data import dataset
from tempor.data.samples import StaticSamples
from tempor.methods.core._params import CategoricalParams, Params
from tempor.methods.preprocessing.imputation._base import BaseImputer, TabularImputerType

from ..hyperimpute_utils import monkeypatch_hyperimpute_logger

with monkeypatch_hyperimpute_logger():
    from hyperimpute.plugins.imputers import Imputers


@dataclasses.dataclass
class StaticTabularImputerParams:
    """Initialization parameters for :class:`StaticTabularImputer`."""

    imputer: TabularImputerType = "ice"
    """Which imputer to use for static covariate imputation."""
    random_state: int = 0
    """Random seed. Will be passed on to the underlying imputer."""
    imputer_params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    """Parameters to pass to the underlying imputer as a keyword arguments dictionary. Defaults to ``{}``."""


@plugins.register_plugin(name="static_tabular_imputer", category="preprocessing.imputation.static")
class StaticTabularImputer(BaseImputer):
    ParamsDefinition = StaticTabularImputerParams
    params: StaticTabularImputerParams  # type: ignore

    def __init__(self, **params) -> None:
        """Impute the static covariates using any tabular imputer from the `hyperimpute` library.

        Args:
            params:
                Parameters and defaults as defined in :class:`StaticTabularImputerParams`.

        Example:
            >>> from tempor.data.datasources import SineDataSource
            >>> from tempor import plugin_loader
            >>>
            >>> dataset = SineDataSource(with_missing = True).load()
            >>> assert dataset.static.dataframe().isna().sum().sum() != 0
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("preprocessing.imputation.static.static_tabular_imputer")
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            StaticTabularImputer(...)
            >>>
            >>> # Impute:
            >>> imputed = model.transform(dataset)
            >>> assert imputed.static.dataframe().isna().sum().sum() == 0
        """
        if "imputer_params" in params and "random_state" in params["imputer_params"]:
            raise ValueError(
                "Do not pass `random_state` as a key in `imputer_params`, pass it directly as `random_state`"
            )
        super().__init__(**params)
        self.params.imputer_params["random_state"] = self.params.random_state
        self.imputer = Imputers().get(self.params.imputer, **self.params.imputer_params)

    def _fit(self, data: dataset.BaseDataset, *args, **kwargs) -> Self:
        if data.static is not None:
            self.imputer.fit(data.static.dataframe())
        return self

    def _transform(self, data: dataset.BaseDataset, *args, **kwargs) -> dataset.BaseDataset:
        # Impute static data.
        if data.static is not None:
            static_data = data.static.dataframe()
            imputed_static_data = self.imputer.transform(static_data)
            imputed_static_data.columns = static_data.columns
            imputed_static_data.index = static_data.index
            data.static = StaticSamples.from_dataframe(imputed_static_data)
        return data

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        hs: List[Params] = [
            CategoricalParams(name="imputer", choices=list(get_args(TabularImputerType))),
        ]
        return hs
