import dataclasses
from typing import Any, Dict, List

import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing_extensions import Self

from tempor.core import plugins
from tempor.data import dataset
from tempor.data.samples import StaticSamples
from tempor.methods.core.params import Params
from tempor.methods.preprocessing.scaling._base import BaseScaler


@dataclasses.dataclass
class StaticStandardScalerParams:
    """Initialization parameters for :class:`StaticStandardScaler`."""

    with_mean: bool = True
    """If True, center the data before scaling. See `sklearn.preprocessing.StandardScaler`."""
    with_std: bool = True
    """If True, scale the data to unit variance. See `sklearn.preprocessing.StandardScaler`."""


@plugins.register_plugin(name="static_standard_scaler", category="preprocessing.scaling.static")
class StaticStandardScaler(BaseScaler):
    ParamsDefinition = StaticStandardScalerParams
    params: StaticStandardScalerParams  # type: ignore

    def __init__(self, **params: Any) -> None:
        """Standard scaling for the static data.

        Standardize the static features by removing the mean and scaling to unit variance.

        Args:
            **params (Any):
                Parameters and defaults as defined in :class:`StaticStandardScalerParams`.

        Example:
            >>> from tempor import plugin_loader
            >>>
            >>> dataset = plugin_loader.get("prediction.one_off.sine", plugin_type="datasource").load()
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("preprocessing.scaling.static.static_standard_scaler")
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            StaticStandardScaler(...)
            >>>
            >>> # Scale:
            >>> scaled = model.transform(dataset)
        """
        super().__init__(**params)
        sklearn_params: Dict[str, Any] = dict(self.params)  # type: ignore
        self.model = StandardScaler(**sklearn_params)

    def _fit(
        self,
        data: dataset.BaseDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        if data.static is None:
            return self

        self.model.fit(data.static.dataframe())
        return self

    def _transform(self, data: dataset.BaseDataset, *args: Any, **kwargs: Any) -> dataset.BaseDataset:
        if data.static is None:
            return data

        static_data = data.static.dataframe()
        scaled = pd.DataFrame(self.model.transform(static_data))
        scaled.columns = static_data.columns
        scaled.index = static_data.index

        data.static = StaticSamples.from_dataframe(scaled)

        return data

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        return []
