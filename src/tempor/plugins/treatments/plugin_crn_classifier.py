import dataclasses
from typing import Any, Dict, List, Optional, cast

import clairvoyance2.data.dataformat as cl_dataformat
import pandas as pd
from clairvoyance2.data import DEFAULT_PADDING_INDICATOR
from clairvoyance2.treatment_effects.crn import CRNClassifier, TimeIndexHorizon
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import dataset, samples
from tempor.data.clv2conv import (
    _from_clv2_time_series,
    tempor_dataset_to_clairvoyance2_dataset,
)
from tempor.plugins.core._params import CategoricalParams, FloatParams, IntegerParams
from tempor.plugins.treatments import BaseTreatments


@dataclasses.dataclass
class CRNParams:
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


@plugins.register_plugin(name="crn_classifier", category="treatments")
class CRNTreatmentsClassifier(BaseTreatments):
    ParamsDefinition = CRNParams
    params: CRNParams  # type: ignore

    def __init__(
        self,
        **params,
    ) -> None:
        """Counterfactual Recurrent Network treatment effects model for classification on the outcomes (targets).

        Paper:
            Estimating counterfactual treatment outcomes over time through adversarially balanced representations,
            Ioana Bica, Ahmed M. Alaa, James Jordon, Mihaela van der Schaar.

        Example:
            >>> from tempor.plugins import plugin_loader
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("treatments.crn_classifier", n_iter=50)
            >>>
            >>> # Train:
            >>> # model.fit(dataset)
            >>>
            >>> # Predict:
            >>> # assert model.predict(dataset, n_future_steps = 10).numpy().shape == (len(dataset), 10, 5)
        """
        super().__init__(**params)
        self.model: Optional[CRNClassifier] = None

    def _fit(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> Self:
        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)

        self.model = CRNClassifier(
            params=self.params,  # pyright: ignore
        )

        self.model.fit(cl_dataset)
        return self

    def _predict(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        data: dataset.Dataset,
        horizons: List[List[float]],
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        if self.model is None:
            raise RuntimeError("Fit the model first")
        if len(data) != len(horizons):
            raise ValueError("Invalid horizons length")

        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)

        cl_horizons_pd = []
        for horizon in horizons:
            cl_horizons_pd.append(pd.Index(horizon))
        cl_horizons = TimeIndexHorizon(time_index_sequence=cl_horizons_pd)

        preds_cl = cast(cl_dataformat.TimeSeriesSamples, self.model.predict(cl_dataset, cl_horizons))
        preds = _from_clv2_time_series(preds_cl.to_multi_index_dataframe())
        return samples.TimeSeriesSamples.from_dataframe(preds)

    def _predict_counterfactuals(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        data: dataset.Dataset,
        horizons: List[List[float]],
        treatment_scenarios: List[List[int]],
        *args,
        **kwargs,
    ) -> List:
        if self.model is None:
            raise RuntimeError("Fit the model first")
        if len(data) != len(horizons):
            raise ValueError("Invalid horizons length")
        if len(horizons) != len(treatment_scenarios):
            raise ValueError("Invalid treatment_scenarios length")

        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)
        cl_horizons_pd = []
        for horizon in horizons:
            cl_horizons_pd.append(pd.Index(horizon))
        cl_horizons = TimeIndexHorizon(time_index_sequence=cl_horizons_pd)

        counterfactuals = []
        for idx, sample_idx in enumerate(cl_dataset.sample_indices):
            treat_scenarios = treatment_scenarios[idx]
            horizon_counterfactuals_sample = cl_horizons.time_index_sequence[idx]

            # TODO: should enforce treat - treat_scenarios shapes here.
            c = self.model.predict_counterfactuals(
                cl_dataset,
                sample_index=sample_idx,
                treatment_scenarios=treat_scenarios,  # pyright: ignore
                horizon=TimeIndexHorizon(time_index_sequence=[horizon_counterfactuals_sample]),
                **kwargs,
            )
            counterfactuals.append(c)
        return counterfactuals

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
