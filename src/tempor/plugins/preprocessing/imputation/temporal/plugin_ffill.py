from typing import Any, List

from hyperimpute.plugins.imputers import Imputers as StaticImputers
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset
from tempor.data.samples import StaticSamples, TimeSeriesSamples
from tempor.plugins.core._params import CategoricalParams, Params
from tempor.plugins.preprocessing.imputation._base import BaseImputer


@plugins.register_plugin(name="ffill", category="preprocessing.imputation.temporal")
class FFillImputer(BaseImputer):
    """Forward-first Time-Series Imputation.

    Args:
        static_imputer (str, optional):
            Which imputer to use for the static data (if any). Defaults to ``"mean"``.
        random_state (int, optional):
            Random seed. Defaults to ``0``.

    Example:
        >>> from tempor.utils.dataloaders import SineDataLoader
        >>> from tempor.plugins import plugin_loader
        >>>
        >>> dataset = SineDataLoader(with_missing = True).load()
        >>> assert dataset.static.dataframe().isna().sum().sum() != 0
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
        >>> assert imputed.static.dataframe().isna().sum().sum() == 0
        >>> assert imputed.time_series.dataframe().isna().sum().sum() == 0
    """

    def __init__(
        self, static_imputer: str = "mean", random_state: int = 0, **params
    ) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)
        self.static_imputer = StaticImputers().get(static_imputer, random_state=random_state)
        self.random_state = random_state

    def _fit(self, data: dataset.BaseDataset, *args, **kwargs) -> Self:
        if data.static is not None:
            self.static_imputer.fit(data.static.dataframe())

        return self

    def _transform(self, data: dataset.BaseDataset, *args, **kwargs) -> dataset.BaseDataset:
        # impute static data
        if data.static is not None:
            static_data = data.static.dataframe()
            imputed_static_data = self.static_imputer.transform(static_data)
            imputed_static_data.columns = static_data.columns
            imputed_static_data.index = static_data.index

            data.static = StaticSamples.from_dataframe(imputed_static_data)

        # impute temporal data
        sample_ts_index = data.time_series.sample_index()
        imputed_ts = data.time_series.dataframe()
        for idx in sample_ts_index:
            imputed_ts.loc[(idx, slice(None)), :] = imputed_ts.loc[(idx, slice(None)), :].ffill()  # pyright: ignore
            imputed_ts.loc[(idx, slice(None)), :] = imputed_ts.loc[(idx, slice(None)), :].bfill()  # pyright: ignore
            imputed_ts.loc[(idx, slice(None)), :] = imputed_ts.loc[(idx, slice(None)), :].fillna(0.0)  # pyright: ignore

        data.time_series = TimeSeriesSamples.from_dataframe(imputed_ts)
        return data

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # pragma: no cover
        return [CategoricalParams(name="static_imputer", choices=["mean", "ice", "missforest"])]
