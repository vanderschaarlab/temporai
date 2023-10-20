from typing import Any, List

from typing_extensions import Self

from tempor.core import plugins
from tempor.data import dataset
from tempor.data.samples import TimeSeriesSamples
from tempor.methods.core.params import Params
from tempor.methods.preprocessing.imputation._base import BaseImputer


@plugins.register_plugin(name="ffill", category="preprocessing.imputation.temporal")
class FFillImputer(BaseImputer):
    def __init__(self, **params: Any) -> None:  # pylint: disable=useless-super-delegation
        """Forward-first Time-Series Imputation.

        Note:
            The data will be represented as a multi-index `(sample_idx, time_idx)` dataframe of features.
            Then ``ffill``, ``bfill`` and ``fillna(0.0)``` will be called in that order.

        Example:
            >>> from tempor import plugin_loader
            >>>
            >>> dataset = plugin_loader.get(
            ...     "prediction.one_off.sine",
            ...     plugin_type="datasource",
            ...     with_missing=True,
            ... ).load()
            >>> assert dataset.time_series.dataframe().isna().sum().sum() != 0
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("preprocessing.imputation.temporal.ffill")
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            FFillImputer(...)
            >>>
            >>> # Impute:
            >>> imputed = model.transform(dataset)
            >>> assert imputed.time_series.dataframe().isna().sum().sum() == 0
        """
        super().__init__(**params)

    def _fit(self, data: dataset.BaseDataset, *args: Any, **kwargs: Any) -> Self:
        return self

    def _transform(self, data: dataset.BaseDataset, *args: Any, **kwargs: Any) -> dataset.BaseDataset:
        # Impute temporal data.
        sample_ts_index = data.time_series.sample_index()
        imputed_ts = data.time_series.dataframe()
        for idx in sample_ts_index:
            imputed_ts.loc[(idx, slice(None)), :] = imputed_ts.loc[(idx, slice(None)), :].ffill()  # pyright: ignore
            imputed_ts.loc[(idx, slice(None)), :] = imputed_ts.loc[(idx, slice(None)), :].bfill()  # pyright: ignore
            imputed_ts.loc[(idx, slice(None)), :] = imputed_ts.loc[(idx, slice(None)), :].fillna(0.0)  # pyright: ignore
        data.time_series = TimeSeriesSamples.from_dataframe(imputed_ts)
        return data

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        return []
