"""Counterfactual Recurrent Network treatment effects model for classification on the outcomes (targets)."""

from typing import Any, List, Optional, cast

import clairvoyance2.data.dataformat as cl_dataformat
import pandas as pd
from clairvoyance2.treatment_effects.crn import CRNClassifier, TimeIndexHorizon
from typing_extensions import Self

from tempor.core import plugins
from tempor.data import dataset, samples
from tempor.data.clv2conv import _from_clv2_time_series, tempor_dataset_to_clairvoyance2_dataset
from tempor.methods._const import Seq2seqParams
from tempor.methods.core.params import CategoricalParams, FloatParams, IntegerParams, Params
from tempor.methods.treatments.temporal._base import BaseTemporalTreatmentEffects


@plugins.register_plugin(name="crn_classifier", category="treatments.temporal.classification")
class CRNTreatmentsClassifier(BaseTemporalTreatmentEffects):
    ParamsDefinition = Seq2seqParams
    params: Seq2seqParams  # type: ignore

    def __init__(
        self,
        **params: Any,
    ) -> None:
        """Counterfactual Recurrent Network treatment effects model for classification on the outcomes (targets).

        Args:
            **params (Any): Parameters for the model.

        Example:
            >>> from tempor import plugin_loader
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("treatments.temporal.classification.crn_classifier", n_iter=50)
            >>>
            >>> # Train:
            >>> # model.fit(dataset)
            >>>
            >>> # Predict:
            >>> # assert model.predict(dataset, n_future_steps = 10).numpy().shape == (len(dataset), 10, 5)

        References:
            Estimating counterfactual treatment outcomes over time through adversarially balanced representations,
            Ioana Bica, Ahmed M. Alaa, James Jordon, Mihaela van der Schaar.
        """
        super().__init__(**params)
        self.model: Optional[CRNClassifier] = None

    def _fit(
        self,
        data: dataset.BaseDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)

        self.model = CRNClassifier(
            params=self.params,  # pyright: ignore
        )

        self.model.fit(cl_dataset)
        return self

    def _predict(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        data: dataset.PredictiveDataset,
        horizons: List[List[float]],
        *args: Any,
        **kwargs: Any,
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
        data: dataset.PredictiveDataset,
        horizons: List[List[float]],
        treatment_scenarios: List[List[int]],
        *args: Any,
        **kwargs: Any,
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

            # Export as DFs, rather than clairvoyance2 TimeSeries:
            c_dfs = []
            for c_ in c:
                c_df = c_.df
                c_df.index.name = "time_idx"
                c_dfs.append(c_df)

            counterfactuals.append(c)

        return counterfactuals

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
