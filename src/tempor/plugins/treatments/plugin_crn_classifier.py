import dataclasses
from typing import List, Optional

import numpy as np
from clairvoyance2.data import DEFAULT_PADDING_INDICATOR
from clairvoyance2.treatment_effects.crn import CRNClassifier, TimeIndexHorizon

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
    adapter_out_activation: Optional[str] = "Tanh"
    # Predictor FF NN:
    predictor_out_activation: Optional[str] = None
    # Misc:
    max_len: Optional[int] = None
    optimizer_str: str = "Adam"
    batch_size: int = 32
    epochs: int = 100
    padding_indicator: float = DEFAULT_PADDING_INDICATOR


@plugins.register_plugin(name="crn_classifier", category="treatments")
class CRNTreatmentsClassifier(BaseTreatments):
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
            >>> model = plugin_loader.get("treatments.crn_classifier", n_iter=50)
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            CRNTreatmentsClassifier(...)
            >>>
            >>> # Predict:
            >>> assert model.predict(dataset, n_future_steps = 10).numpy().shape == (len(dataset), 10, 5)
        """
        super().__init__(**params)
        self.model = CRNClassifier(
            params=self.params,
        )

    def _fit(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> "CRNTreatmentsClassifier":  # pyright: ignore
        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)
        self.model.fit(cl_dataset)
        return self

    def _predict(  # type: ignore[override]
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        if self.model is None:
            raise RuntimeError("Fit the model first")
        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)

        horizons = TimeIndexHorizon(
            time_index_sequence=[tc.time_index[len(tc.time_index) // 2 :] for tc in cl_dataset.temporal_covariates]
        )

        preds = _from_clv2_time_series(self.model.predict(cl_dataset, horizons).to_multi_index_dataframe())
        return samples.TimeSeriesSamples.from_dataframe(preds)

    def _predict_counterfactuals(  # type: ignore[override]
        self,
        data: dataset.Dataset,
        n_counterfactuals_per_sample: int = 2,
        *args,
        **kwargs,
    ) -> List:
        if self.model is None:
            raise RuntimeError("Fit the model first")

        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)

        horizon_counterfactuals = TimeIndexHorizon(
            time_index_sequence=[tc.time_index[len(tc.time_index) // 2 :] for tc in cl_dataset.temporal_covariates]
        )

        counterfactuals = []
        for idx, sample_idx in enumerate(cl_dataset.sample_indices):
            treat = cl_dataset.temporal_treatments[sample_idx].df.values
            horizon_counterfactuals_sample = horizon_counterfactuals.time_index_sequence[idx]
            treat_scenarios = []
            for treat_sc_idx in range(n_counterfactuals_per_sample):
                np.random.seed(12345 + treat_sc_idx)
                treat_sc = np.random.randint(
                    low=0, high=1 + 1, size=(len(horizon_counterfactuals_sample), treat.shape[1])
                )
                treat_scenarios.append(treat_sc)

            c = self.model.predict_counterfactuals(
                cl_dataset,
                sample_index=sample_idx,
                treatment_scenarios=treat_scenarios,
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