import abc
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from typing_extensions import Self

import tempor.exc
from tempor.data import data_typing, dataset, samples
from tempor.methods.core._params import CategoricalParams, FloatParams, IntegerParams
from tempor.models import utils
from tempor.models.ddh import DynamicDeepHitModel, output_modes, rnn_modes


class OutputTimeToEventAnalysis:
    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> Self:  # pragma: no cover
        ...

    @abc.abstractmethod
    def predict_risk(self, X: pd.DataFrame, time_horizons: List) -> pd.DataFrame:  # pragma: no cover
        ...


class DDHEmbedding:
    def __init__(self, emb_model: DynamicDeepHitModel) -> None:
        """Survival analysis embedding creation for time-series with :class:`tempor.models.ddh.DynamicDeepHitModel`.

        Args:
            emb_model (DynamicDeepHitModel):
                :class:`tempor.models.ddh.DynamicDeepHitModel` to use for temporal feature embedding.
        """
        self.emb_model = emb_model

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

    def prepare_fit(
        self,
        data: dataset.BaseDataset,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        utils.enable_reproducibility(self.emb_model.random_state)

        data = cast(dataset.TimeToEventAnalysisDataset, data)
        self._validate_data(data)
        (static, temporal, observation_times, event_times, event_values) = self._convert_data(data)
        processed_data = self._merge_data(static, temporal, observation_times)
        if TYPE_CHECKING:  # pragma: no cover
            assert event_times is not None and event_values is not None  # nosec B101
        return processed_data, event_times, event_values

    def prepare_predict(
        self,
        data: dataset.PredictiveDataset,
        horizons: data_typing.TimeIndex,
        *args,  # pylint: disable=unused-argument
        **kwargs,
    ) -> np.ndarray:
        data = cast(dataset.TimeToEventAnalysisDataset, data)
        self._validate_data(data)
        (static, temporal, observation_times, _, _) = self._convert_data(data)
        processed_data = self._merge_data(static, temporal, observation_times)
        return processed_data

    @staticmethod
    def hyperparameter_space(*args, **kwargs):  # pylint: disable=unused-argument
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


class DDHEmbeddingTimeToEventAnalysis(DDHEmbedding):
    def __init__(
        self,
        output_model: OutputTimeToEventAnalysis,
        emb_model: DynamicDeepHitModel,
    ) -> None:
        """Survival analysis embedding creation for time-series with :class:`tempor.models.ddh.DynamicDeepHitModel`
        followed by ``output_model`` :class:`OutputTimeToEventAnalysis` survival analysis estimator.

        Args:
            output_model (OutputTimeToEventAnalysis):
                Output model to use for predicting risk.
            emb_model (DynamicDeepHitModel):
                :class:`tempor.models.ddh.DynamicDeepHitModel` to use for temporal feature embedding.
        """
        DDHEmbedding.__init__(self, emb_model=emb_model)
        self.output_model = output_model

    def fit(
        self,
        data: dataset.BaseDataset,
        *args,  # pylint: disable=unused-argument
        **kwargs,
    ) -> Self:
        processed_data, event_times, event_values = self.prepare_fit(data)

        self.emb_model.fit(processed_data, event_times, event_values)
        embeddings = self.emb_model.predict_emb(processed_data)
        self.output_model.fit(
            pd.DataFrame(embeddings),  # pyright: ignore
            pd.Series(event_times),
            pd.Series(event_values),
        )

        return self

    def predict(
        self,
        data: dataset.PredictiveDataset,
        horizons: data_typing.TimeIndex,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        # NOTE: kwargs will be passed to DynamicDeepHitModel.predict_emb().
        # E.g. `batch_size` batch size parameter can be provided this way.
        processed_data = self.prepare_predict(data, horizons, *args, **kwargs)

        embeddings = self.emb_model.predict_emb(processed_data)
        risk = self.output_model.predict_risk(
            pd.DataFrame(embeddings),  # pyright: ignore
            horizons,
        )
        risk = np.asarray(risk)

        return samples.TimeSeriesSamples(
            risk.reshape((risk.shape[0], risk.shape[1], 1)),
            sample_index=data.time_series.sample_index(),
            time_indexes=[horizons] * data.time_series.num_samples,  # pyright: ignore
            feature_index=["risk_score"],
        )

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return DDHEmbedding.hyperparameter_space(*args, **kwargs)
