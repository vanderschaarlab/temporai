from typing import Any, List

from typing_extensions import Self

import tempor.core.plugins as plugins
from tempor.data import dataset
from tempor.data.samples import TimeSeriesSamples
from tempor.methods.core._params import Params
from tempor.methods.preprocessing.imputation._base import BaseImputer


@plugins.register_plugin(name="bfill", category="preprocessing.imputation.temporal")
class BFillImputer(BaseImputer):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        """Backward-first Time-Series Imputation.

        Note:
            The data will be represented as a multi-index `(sample_idx, time_idx)` dataframe of features.
            Then ``bfill``, ``ffill`` and ``fillna(0.0)``` will be called in that order.

        Example:
            >>> from tempor.data.datasources import SineDataSource
            >>> from tempor import plugin_loader
            >>>
            >>> dataset = SineDataSource(with_missing = True).load()
            >>> assert dataset.time_series.dataframe().isna().sum().sum() != 0
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("preprocessing.imputation.temporal.bfill")
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            BFillImputer(...)
            >>>
            >>> # Impute:
            >>> imputed = model.transform(dataset)
            >>> assert imputed.time_series.dataframe().isna().sum().sum() == 0
        """
        super().__init__(**params)

    def _fit(self, data: dataset.BaseDataset, *args, **kwargs) -> Self:
        return self

    def _transform(self, data: dataset.BaseDataset, *args, **kwargs) -> dataset.BaseDataset:
        # Impute temporal data.
        sample_ts_index = data.time_series.sample_index()
        imputed_ts = data.time_series.dataframe()
        for idx in sample_ts_index:
            imputed_ts.loc[(idx, slice(None)), :] = imputed_ts.loc[(idx, slice(None)), :].bfill()  # pyright: ignore
            imputed_ts.loc[(idx, slice(None)), :] = imputed_ts.loc[(idx, slice(None)), :].ffill()  # pyright: ignore
            imputed_ts.loc[(idx, slice(None)), :] = imputed_ts.loc[(idx, slice(None)), :].fillna(0.0)  # pyright: ignore
        data.time_series = TimeSeriesSamples.from_dataframe(imputed_ts)
        return data

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        return []
