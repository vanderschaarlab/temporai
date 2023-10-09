import dataclasses
from typing import Any, Dict, List

from typing_extensions import Self, get_args

import tempor.methods.core as plugins
from tempor.data import dataset
from tempor.data.samples import TimeSeriesSamples
from tempor.methods.core._params import CategoricalParams, Params
from tempor.methods.preprocessing.imputation._base import BaseImputer, TabularImputerType

from ..hyperimpute_utils import monkeypatch_hyperimpute_logger

with monkeypatch_hyperimpute_logger():
    from hyperimpute.plugins.imputers import Imputers


@dataclasses.dataclass
class TemporalTabularImputerParams:
    """Initialization parameters for :class:`TemporalTabularImputer`."""

    imputer: TabularImputerType = "ice"
    """Which imputer to use for temporal covariate imputation."""
    random_state: int = 0
    """Random seed. Will be passed on to the underlying imputer."""
    imputer_params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    """Parameters to pass to the underlying imputer as a keyword arguments dictionary. Defaults to ``{}``."""


@plugins.register_plugin(name="ts_tabular_imputer", category="preprocessing.imputation.temporal")
class TemporalTabularImputer(BaseImputer):
    ParamsDefinition = TemporalTabularImputerParams
    params: TemporalTabularImputerParams  # type: ignore

    def __init__(self, **params) -> None:
        """Impute the temporal covariates using any tabular imputer from the `hyperimpute` library.

        Note:
            The data will be represented as a multi-index `(sample_idx, time_idx)` dataframe of features, and the
            tabular imputer will be applied to this dataframe directly.

        Args:
            **params:
                Parameters and defaults as defined in :class:`TemporalTabularImputerParams`.

        Example:
            >>> from tempor.utils.dataloaders import SineDataLoader
            >>> from tempor.methods import plugin_loader
            >>>
            >>> dataset = SineDataLoader(with_missing = True).load()
            >>> assert dataset.time_series.dataframe().isna().sum().sum() != 0
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("preprocessing.imputation.temporal.ts_tabular_imputer")
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            TemporalTabularImputer(...)
            >>>
            >>> # Impute:
            >>> imputed = model.transform(dataset)
            >>> assert imputed.time_series.dataframe().isna().sum().sum() == 0
        """
        if "imputer_params" in params and "random_state" in params["imputer_params"]:
            raise ValueError(
                "Do not pass `random_state` as a key in `imputer_params`, pass it directly as `random_state`"
            )
        super().__init__(**params)
        self.params.imputer_params["random_state"] = self.params.random_state
        self.imputer = Imputers().get(self.params.imputer, **self.params.imputer_params)

    def _fit(self, data: dataset.BaseDataset, *args, **kwargs) -> Self:
        self.imputer.fit(data.time_series.dataframe())
        return self

    def _transform(self, data: dataset.BaseDataset, *args, **kwargs) -> dataset.BaseDataset:
        # Impute temporal data.
        ts_data = data.time_series.dataframe()
        imputed_ts_data = self.imputer.transform(ts_data)
        imputed_ts_data.columns = ts_data.columns
        imputed_ts_data.index = ts_data.index
        data.time_series = TimeSeriesSamples.from_dataframe(imputed_ts_data)
        return data

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        hs: List[Params] = [
            CategoricalParams(name="imputer", choices=list(get_args(TabularImputerType))),
        ]
        return hs
