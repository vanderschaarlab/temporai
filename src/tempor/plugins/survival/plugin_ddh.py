import dataclasses
from typing import List, Optional, cast

import numpy as np
from typing_extensions import Literal, Self

import tempor.exc
import tempor.plugins.core as plugins
from tempor.data import dataset, samples
from tempor.models.ddh import DynamicDeepHitModel
from tempor.plugins.survival import BaseSurvivalAnalysis

RnnModes = Literal[
    "GRU",
    "LSTM",
    "RNN",
    # "Transformer",
]
OutputModes = Literal[
    "MLP",
    "LSTM",
    "GRU",
    "RNN",
    # "Transformer",
    # "TCN",
    # "InceptionTime",
    # "InceptionTimePlus",
    # "ResCNN",
    # "XCM",
]


@dataclasses.dataclass
class DynamicDeepHitSurvivalAnalysisParams:
    # TODO: Docstring.
    n_iter: int = 1000
    batch_size: int = 100
    lr: float = 1e-3
    n_layers_hidden: int = 1
    n_units_hidden: int = 40
    split: int = 100
    rnn_mode: RnnModes = "GRU"
    alpha: float = 0.34
    beta: float = 0.27
    sigma: float = 0.21
    dropout: float = 0.06
    device: str = "cpu"
    patience: int = 20
    output_mode: OutputModes = "MLP"
    random_state: int = 0


@plugins.register_plugin(name="dynamic_deephit", category="survival")
class DynamicDeepHitSurvivalAnalysis(BaseSurvivalAnalysis):
    """Class docstring"""

    ParamsDefinition = DynamicDeepHitSurvivalAnalysisParams
    params: DynamicDeepHitSurvivalAnalysisParams  # type: ignore

    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        """:class:`DynamicDeepHit` survival analysis model.

        Args:
            **params:
                Parameters and defaults as defined in :class:`DynamicDeepHitSurvivalAnalysisParams`.
        """
        super().__init__(**params)

        self.model = DynamicDeepHitModel(
            split=self.params.split,
            n_layers_hidden=self.params.n_layers_hidden,
            n_units_hidden=self.params.n_units_hidden,
            rnn_mode=self.params.rnn_mode,
            alpha=self.params.alpha,
            beta=self.params.beta,
            sigma=self.params.sigma,
            dropout=self.params.dropout,
            patience=self.params.patience,
            lr=self.params.lr,
            batch_size=self.params.batch_size,
            n_iter=self.params.n_iter,
            output_mode=self.params.output_mode,
            device=self.params.device,
        )

    def _merge_data(
        self,
        static: Optional[np.ndarray],
        temporal: np.ndarray,
        observation_times: List[List],
    ) -> np.ndarray:
        if static is None:
            # TODO: Investigate - why add an all-zeros feature?
            static = np.zeros((len(temporal), 0))

        merged = []
        for idx, item in enumerate(temporal):  # pylint: disable=unused-variable
            local_static = static[idx].reshape(1, -1)
            local_static = np.repeat(local_static, len(temporal[idx]), axis=0)
            tst = np.concatenate(
                [
                    temporal[idx],
                    local_static,
                    np.asarray(observation_times[idx]).reshape(-1, 1),
                ],
                axis=1,
            )
            merged.append(tst)

        return np.array(merged, dtype=object)

    def _validate_data(self, data: dataset.TimeToEventAnalysisDataset) -> None:
        if data.predictive.targets.num_features > 1:
            raise tempor.exc.UnsupportedSetupException(
                f"{self.__class__.__name__} does not currently support more than one event feature, "
                f"but features found were: {data.predictive.targets.dataframe().columns}"
            )
        if not data.time_series.num_timesteps_equal():
            raise tempor.exc.UnsupportedSetupException(
                f"{self.__class__.__name__} currently requires all samples to have the same number of timesteps, "
                f"but found timesteps of varying lengths {np.unique(data.time_series.num_timesteps()).tolist()}"
            )
        if not isinstance(data.time_series.time_indexes()[0][0], (int, float)):
            raise tempor.exc.UnsupportedSetupException(
                f"{self.__class__.__name__} currently only supports `int` or `float` time indices, "
                f"but found time index of type {type(data.time_series.time_indexes()[0][0])}"
            )

    def _fit(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> Self:
        data = cast(dataset.TimeToEventAnalysisDataset, data)
        self._validate_data(data)

        static = data.static.numpy() if data.static is not None else None
        temporal = data.time_series.numpy(padding_indicator=-1)  # TODO: check, padding is messy.
        observation_times = data.time_series.time_indexes()

        event_times, event_values = (df.to_numpy() for df in data.predictive.targets.split_as_two_dataframes())

        print(static)
        print(temporal)
        print(observation_times)

        print(event_times)
        print(event_values)

        processed_data = self._merge_data(static, temporal, observation_times)

        self.model.fit(processed_data, event_times, event_values)

        # TODO: WIP
        raise NotImplementedError
        # return self

    def _predict(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        raise NotImplementedError

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return []
