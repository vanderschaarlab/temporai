"""MinMax scaling for the static data."""

import dataclasses
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Self

from tempor.core import plugins
from tempor.data import dataset
from tempor.data.samples import StaticSamples
from tempor.methods.core.params import CategoricalParams, Params
from tempor.methods.preprocessing.scaling._base import BaseScaler


@dataclasses.dataclass
class StaticMinMaxScalerParams:
    """Initialization parameters for :class:`StaticMinMaxScaler`."""

    feature_range: Tuple[int, int] = (0, 1)
    """Desired range of transformed data. See `sklearn.preprocessing.MinMaxScaler`."""
    clip: bool = False
    """Set to True to clip transformed values of held-out data to provided ``feature_range``.
    See `sklearn.preprocessing.MinMaxScaler`.
    """


@plugins.register_plugin(name="static_minmax_scaler", category="preprocessing.scaling.static")
class StaticMinMaxScaler(BaseScaler):
    ParamsDefinition = StaticMinMaxScalerParams
    params: StaticMinMaxScalerParams  # type: ignore

    def __init__(self, **params: Any) -> None:
        """MinMax scaling for the static data.

        Transform the static features by scaling each feature to a given range. This estimator scales and translates
        each feature individually such that it is in the given range on the training set, e.g. between zero and one.

        Args:
            **params (Any):
                Parameters and defaults as defined in :class:`StaticMinMaxScalerParams`.

        Example:
            >>> from tempor import plugin_loader
            >>>
            >>> dataset = plugin_loader.get("prediction.one_off.sine", plugin_type="datasource").load()
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("preprocessing.scaling.static.static_minmax_scaler")
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            StaticMinMaxScaler(...)
            >>>
            >>> # Scale:
            >>> scaled = model.transform(dataset)
        """
        super().__init__(**params)
        sklearn_params: Dict[str, Any] = dict(self.params)  # type: ignore
        sklearn_params["feature_range"] = tuple(sklearn_params["feature_range"])
        self.model = MinMaxScaler(**sklearn_params)

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
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # noqa: D102
        return [
            CategoricalParams("clip", [True, False]),
        ]
