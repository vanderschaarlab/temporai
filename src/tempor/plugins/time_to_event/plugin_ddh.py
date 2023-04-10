import dataclasses
from typing import TYPE_CHECKING, List, Optional, cast

import numpy as np
from typing_extensions import Self

import tempor.exc
import tempor.plugins.core as plugins
from tempor.data import data_typing, dataset, samples
from tempor.models import utils
from tempor.models.ddh import (
    DynamicDeepHitModel,
    OutputMode,
    RnnMode,
    output_modes,
    rnn_modes,
)
from tempor.plugins.core._params import CategoricalParams, FloatParams, IntegerParams
from tempor.plugins.time_to_event import BaseTimeToEventAnalysis


@dataclasses.dataclass
class DynamicDeepHitTimeToEventAnalysisParams:
    # TODO: Docstring.
    n_iter: int = 1000
    batch_size: int = 100
    lr: float = 1e-3
    n_layers_hidden: int = 1
    n_units_hidden: int = 40
    split: int = 100
    rnn_mode: RnnMode = "GRU"
    alpha: float = 0.34
    beta: float = 0.27
    sigma: float = 0.21
    dropout: float = 0.06
    device: str = "cpu"
    patience: int = 20
    output_mode: OutputMode = "MLP"
    random_state: int = 0


@plugins.register_plugin(name="dynamic_deephit", category="time_to_event")
class DynamicDeepHitTimeToEventAnalysis(BaseTimeToEventAnalysis):
    ParamsDefinition = DynamicDeepHitTimeToEventAnalysisParams
    params: DynamicDeepHitTimeToEventAnalysisParams  # type: ignore

    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        """Dynamic DeepHit survival analysis model.

        Note:
            Current implementation has the following limitations:
                - Only one output feature is supported (no competing risks).
                - Risk prediction for time points beyond the last event time in the dataset may throw errors.

        Args:
            **params:
                Parameters and defaults as defined in :class:`DynamicDeepHitTimeToEventAnalysisParams`.

        References:
            "Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis With Competing Risks Based on
            Longitudinal Data", Changhee Lee, Jinsung Yoon, Mihaela van der Schaar.
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
        temporal: List[np.ndarray],
        observation_times: List[np.ndarray],
    ) -> np.ndarray:
        if static is None:
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
        if data.predictive.targets is not None and data.predictive.targets.num_features > 1:
            raise tempor.exc.UnsupportedSetupException(
                f"{self.__class__.__name__} does not currently support more than one event feature, "
                f"but features found were: {data.predictive.targets.dataframe().columns}"
            )
        # TODO: This needs investigating - likely different length sequences aren't handled properly.
        # if not data.time_series.num_timesteps_equal():
        #     raise tempor.exc.UnsupportedSetupException(
        #         f"{self.__class__.__name__} currently requires all samples to have the same number of timesteps, "
        #         f"but found timesteps of varying lengths {np.unique(data.time_series.num_timesteps()).tolist()}"
        #     )

    def _convert_data(self, data: dataset.TimeToEventAnalysisDataset):
        if data.has_static:
            static = data.static.numpy() if data.static is not None else None
        else:
            static = np.zeros((data.time_series.num_samples, 0))
        temporal = [df.to_numpy() for df in data.time_series.list_of_dataframes()]
        observation_times = data.time_series.time_indexes_float()
        if data.predictive is not None and data.predictive.targets is not None:
            event_times, event_values = (
                df.to_numpy().reshape((-1,)) for df in data.predictive.targets.split_as_two_dataframes()
            )
        else:
            event_times, event_values = None, None
        return (static, temporal, observation_times, event_times, event_values)

    def _fit(
        self,
        data: dataset.BaseDataset,
        *args,
        **kwargs,
    ) -> Self:
        utils.enable_reproducibility(self.params.random_state)

        data = cast(dataset.TimeToEventAnalysisDataset, data)
        self._validate_data(data)
        (static, temporal, observation_times, event_times, event_values) = self._convert_data(data)
        processed_data = self._merge_data(static, temporal, observation_times)
        if TYPE_CHECKING:  # pragma: no cover
            assert event_times is not None and event_values is not None  # nosec B101

        self.model.fit(processed_data, event_times, event_values)

        return self

    def _predict(  # type: ignore[override]
        self,
        data: dataset.PredictiveDataset,
        horizons: data_typing.TimeIndex,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        # NOTE: kwargs will be passed to DynamicDeepHitModel.predict_risk().
        # E.g. `bs` batch size parameter can be provided this way.
        data = cast(dataset.TimeToEventAnalysisDataset, data)
        self._validate_data(data)
        (static, temporal, observation_times, _, _) = self._convert_data(data)
        processed_data = self._merge_data(static, temporal, observation_times)
        risk = self.model.predict_risk(processed_data, horizons, **kwargs)
        return samples.TimeSeriesSamples(
            risk.reshape((risk.shape[0], risk.shape[1], 1)),
            sample_index=data.time_series.sample_index(),
            time_indexes=[horizons] * data.time_series.num_samples,  # pyright: ignore
            feature_index=["risk_score"],
        )

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return [
            IntegerParams(name="n_units_hidden", low=10, high=100, step=10),
            IntegerParams(name="n_layers_hidden", low=1, high=4),
            CategoricalParams(name="batch_size", choices=[100, 200, 500]),
            CategoricalParams(name="lr", choices=[1e-2, 1e-3, 1e-4]),
            CategoricalParams(name="rnn_mode", choices=list(rnn_modes)),
            CategoricalParams(name="output_mode", choices=list(output_modes)),
            FloatParams(name="alpha", low=0.0, high=0.5),
            FloatParams(name="sigma", low=0.0, high=0.5),
            FloatParams(name="beta", low=0.0, high=0.5),
            FloatParams(name="dropout", low=0.0, high=0.2),
        ]
