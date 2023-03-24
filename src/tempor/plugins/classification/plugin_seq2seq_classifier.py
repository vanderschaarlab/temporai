import dataclasses
from typing import Any, Dict, List, Optional, cast

import clairvoyance2.data.dataformat as cl_dataformat
from clairvoyance2.data import DEFAULT_PADDING_INDICATOR
from clairvoyance2.prediction.seq2seq import Seq2SeqClassifier, TimeIndexHorizon
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset, samples
from tempor.data.clv2conv import (
    _from_clv2_time_series,
    tempor_dataset_to_clairvoyance2_dataset,
)
from tempor.plugins.classification import BaseClassifier
from tempor.plugins.core._params import CategoricalParams, FloatParams, IntegerParams


@dataclasses.dataclass
class seq2seqParams:
    # Encoder:
    encoder_rnn_type: str = "LSTM"
    encoder_hidden_size: int = 100
    encoder_num_layers: int = 1
    encoder_bias: bool = True
    encoder_dropout: float = 0.0
    encoder_bidirectional: bool = False
    encoder_nonlinearity: Optional[str] = None
    encoder_proj_size: Optional[int] = None
    # Decoder:
    decoder_rnn_type: str = "LSTM"
    decoder_hidden_size: int = 100
    decoder_num_layers: int = 1
    decoder_bias: bool = True
    decoder_dropout: float = 0.0
    decoder_bidirectional: bool = False
    decoder_nonlinearity: Optional[str] = None
    decoder_proj_size: Optional[int] = None
    # Adapter FF NN:
    adapter_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [50])
    adapter_out_activation: Optional[str] = "Tanh"
    # Predictor FF NN:
    predictor_hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [])
    predictor_out_activation: Optional[str] = None
    # Misc:
    max_len: Optional[int] = None
    optimizer_str: str = "Adam"
    optimizer_kwargs: Dict[str, Any] = dataclasses.field(default_factory=lambda: dict(lr=0.01, weight_decay=1e-5))
    batch_size: int = 32
    epochs: int = 100
    padding_indicator: float = DEFAULT_PADDING_INDICATOR


@plugins.register_plugin(name="seq2seq_classifier", category="classification")
class Seq2seqClassifier(BaseClassifier):
    ParamsDefinition = seq2seqParams
    params: seq2seqParams  # type: ignore

    def __init__(
        self,
        **params,
    ) -> None:
        """Seq2seq classifier.

        Example:
            >>> import doctest; doctest.ELLIPSIS_MARKER = "[...]"  # Doctest config, ignore.
            >>>
            >>> from tempor.utils.dataloaders.sine import SineDataLoader
            >>> from tempor.data import dataset
            >>> from tempor.plugins import plugin_loader
            >>>
            >>> raw_data = SineDataLoader(temporal_dim=5).load()
            >>> data = dataset.TemporalPredictionDataset(
            ...    time_series=raw_data.time_series.dataframe(),
            ...    static=raw_data.static.dataframe(),
            ...    targets=raw_data.time_series.dataframe()
            ... )
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("classification.seq2seq_classifier", epochs=10)
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
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> Self:
        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)
        self.model.fit(cl_dataset)
        return self

    def _predict(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        data: dataset.Dataset,
        n_future_steps: int,
        *args,
        time_delta: int = 1,
        **kwargs,
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

    def _predict_proba(  # type: ignore[override]
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        raise NotImplementedError("not supported")

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
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
