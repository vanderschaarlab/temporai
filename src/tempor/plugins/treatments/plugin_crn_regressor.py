import dataclasses
from typing import Optional

from clairvoyance2.data import DEFAULT_PADDING_INDICATOR
from clairvoyance2.treatment_effects.crn import CRNRegressor, TimeIndexHorizon

import tempor.plugins.core as plugins
from tempor.data import dataset, samples
from tempor.data.clv2conv import (
    _from_clv2_time_series,
    tempor_dataset_to_clairvoyance2_dataset,
)
from tempor.plugins.core._params import FloatParams, IntegerParams
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
    adapter_out_activation: Optional[str] = "Tanh"
    # Predictor FF NN:
    predictor_out_activation: Optional[str] = None
    # Misc:
    max_len: Optional[int] = None
    optimizer_str: str = "Adam"
    batch_size: int = 32
    epochs: int = 100
    padding_indicator: float = DEFAULT_PADDING_INDICATOR


@plugins.register_plugin(name="crn_regressor", category="treatments")
class CRNTreatmentsRegressor(BaseTreatments):
    """
    Paper: Estimating counterfactual treatment outcomes over time through adversarially balanced representations, Ioana Bica, Ahmed M. Alaa, James Jordon, Mihaela van der Schaar
    """

    ParamsDefinition = CRNParams
    params: CRNParams  # type: ignore

    def __init__(
        self,
        **params,
    ) -> None:
        """.

        Example:
            >>> from tempor.plugins import plugin_loader
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("treatments.crn", n_iter=50)
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            CRNTreatmentsRegressor(...)
            >>>
            >>> # Predict:
            >>> assert model.predict(dataset, n_future_steps = 10).numpy().shape == (len(dataset), 10, 5)
        """
        super().__init__(**params)
        self.model = CRNRegressor(
            params=self.params,
        )

    def _fit(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> "CRNTreatmentsRegressor":  # pyright: ignore
        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)
        self.model.fit(cl_dataset)
        return self

    def _predict(  # type: ignore[override]
        self,
        data: dataset.Dataset,
        n_future_steps: Optional[int] = None,
        time_delta: int = 1,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        if self.model is None:
            raise RuntimeError("Fit the model first")
        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)

        horizons = None
        if n_future_steps is not None:
            horizons = TimeIndexHorizon.future_horizon_from_dataset(
                cl_dataset,
                forecast_n_future_steps=n_future_steps,
                time_delta=time_delta,
            )

        preds = _from_clv2_time_series(self.model.predict(cl_dataset, horizons).to_multi_index_dataframe())
        return samples.TimeSeriesSamples.from_dataframe(preds)

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return [
            IntegerParams(name="encoder_hidden_size", low=10, high=500),
            IntegerParams(name="encoder_num_layers", low=1, high=10),
            FloatParams(name="encoder_dropout", low=0, high=0.2),
            IntegerParams(name="decoder_hidden_size", low=10, high=500),
            IntegerParams(name="decoder_num_layers", low=1, high=10),
            FloatParams(name="decoder_dropout", low=0, high=0.2),
        ]
