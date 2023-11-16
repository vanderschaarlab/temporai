"""Temporal classification estimator based on Seq2Seq model."""

from typing import Any, List, cast

import clairvoyance2.data.dataformat as cl_dataformat
from clairvoyance2.prediction.seq2seq import Seq2SeqClassifier, TimeIndexHorizon
from typing_extensions import Self

from tempor.core import plugins
from tempor.data import dataset, samples
from tempor.data.clv2conv import _from_clv2_time_series, tempor_dataset_to_clairvoyance2_dataset
from tempor.methods.constants import Seq2seqParams
from tempor.methods.core.params import CategoricalParams, FloatParams, IntegerParams, Params
from tempor.methods.prediction.temporal.classification import BaseTemporalClassifier


@plugins.register_plugin(name="seq2seq_classifier", category="prediction.temporal.classification")
class Seq2seqClassifier(BaseTemporalClassifier):
    ParamsDefinition = Seq2seqParams
    params: Seq2seqParams  # type: ignore

    def __init__(
        self,
        **params: Any,
    ) -> None:
        """Seq2seq classifier.

        Args:
            **params (Any):
                Parameters and defaults as defined in :class:`Seq2seqParams`.

        Example:
            >>> import doctest; doctest.ELLIPSIS_MARKER = "[...]"  # Doctest config, ignore.
            >>>
            >>> from tempor.data import dataset
            >>> from tempor import plugin_loader
            >>>
            >>> raw_data = plugin_loader.get("prediction.one_off.sine", plugin_type="datasource", temporal_dim=5).load()
            >>> data = dataset.TemporalPredictionDataset(
            ...    time_series=raw_data.time_series.dataframe(),
            ...    static=raw_data.static.dataframe(),
            ...    targets=raw_data.time_series.dataframe(),
            ... )
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("prediction.temporal.classification.seq2seq_classifier", epochs=10)
            >>>
            >>> # Train:
            >>> model.fit(data)
            [...]
            >>>
            >>> # Predict:
            >>> assert model.predict(data, n_future_steps = 10).numpy().shape == (len(data), 10, 5)
            >>>
            >>> doctest.ELLIPSIS_MARKER = "..."  # Doctest config, ignore.
        """
        super().__init__(**params)
        self.model = Seq2SeqClassifier(
            params=self.params,  # pyright: ignore
        )

    def _fit(
        self,
        data: dataset.BaseDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)
        self.model.fit(cl_dataset)
        return self

    def _predict(
        self,
        data: dataset.PredictiveDataset,
        n_future_steps: int,
        *args: Any,
        time_delta: int = 1,
        **kwargs: Any,
    ) -> samples.TimeSeriesSamples:
        if self.model is None:
            raise RuntimeError("Fit the model first")
        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)

        horizons = TimeIndexHorizon.future_horizon_from_dataset(
            cl_dataset,
            forecast_n_future_steps=n_future_steps,
            time_delta=time_delta,
        )

        preds_cl = cast(cl_dataformat.TimeSeriesSamples, self.model.predict(cl_dataset, horizons))
        preds = _from_clv2_time_series(preds_cl.to_multi_index_dataframe())
        return samples.TimeSeriesSamples.from_dataframe(preds)

    def _predict_proba(
        self,
        data: dataset.PredictiveDataset,
        *args: Any,
        **kwargs: Any,
    ) -> samples.TimeSeriesSamples:
        raise NotImplementedError("Not currently supported")

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # noqa: D102
        return [
            IntegerParams(name="encoder_hidden_size", low=10, high=500),
            IntegerParams(name="encoder_num_layers", low=1, high=10),
            FloatParams(name="encoder_dropout", low=0, high=0.2),
            CategoricalParams(name="encoder_rnn_type", choices=["LSTM", "GRU", "RNN"]),
            IntegerParams(name="decoder_hidden_size", low=10, high=500),
            IntegerParams(name="decoder_num_layers", low=1, high=10),
            FloatParams(name="decoder_dropout", low=0, high=0.2),
            CategoricalParams(name="decoder_rnn_type", choices=["LSTM", "GRU", "RNN"]),
        ]
