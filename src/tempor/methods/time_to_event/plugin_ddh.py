"""Dynamic DeepHit survival analysis model."""

import dataclasses
from typing import Any, List

from typing_extensions import Self

from tempor.core import plugins
from tempor.data import data_typing, dataset, samples
from tempor.methods.core.params import Params
from tempor.methods.time_to_event import BaseTimeToEventAnalysis
from tempor.models.ddh import DynamicDeepHitModel, OutputMode, RnnMode

from .helper_embedding import DDHEmbedding


@dataclasses.dataclass
class DynamicDeepHitTimeToEventAnalysisParams:
    n_iter: int = 1000
    """Number of training epochs."""
    batch_size: int = 100
    """Training batch size."""
    lr: float = 1e-3
    """Training learning rate."""
    n_layers_hidden: int = 1
    """Number of hidden layers in the network."""
    n_units_hidden: int = 40
    """Number of units for each hidden layer."""
    split: int = 100
    """Number of discrete buckets."""
    rnn_mode: RnnMode = "GRU"
    """Internal temporal architecture, one of `RnnMode`."""
    alpha: float = 0.34
    """Weighting (0, 1) likelihood and rank loss (L2 in paper). 1 gives only likelihood, and 0 gives only rank loss."""
    beta: float = 0.27
    """Beta, see paper."""
    sigma: float = 0.21
    """From eta in rank loss (L2 in paper)."""
    dropout: float = 0.06
    """Network dropout value."""
    device: str = "cpu"
    """PyTorch Device."""
    val_size: float = 0.1
    """Early stopping: size of validation set."""
    patience: int = 20
    """Early stopping: training patience without any improvement."""
    output_mode: OutputMode = "MLP"
    """Output network, on of `OutputMode`."""
    random_state: int = 0
    """Random seed."""


@plugins.register_plugin(name="dynamic_deephit", category="time_to_event")
class DynamicDeepHitTimeToEventAnalysis(BaseTimeToEventAnalysis, DDHEmbedding):
    ParamsDefinition = DynamicDeepHitTimeToEventAnalysisParams
    params: DynamicDeepHitTimeToEventAnalysisParams  # type: ignore

    def __init__(self, **params: Any) -> None:
        """Dynamic DeepHit survival analysis model.

        Note:
            Current implementation has the following limitations:
                - Only one output feature is supported (no competing risks).
                - Risk prediction for time points beyond the last event time in the dataset may throw errors.

        Args:
            **params (Any):
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
            val_size=self.params.val_size,
            patience=self.params.patience,
            lr=self.params.lr,
            batch_size=self.params.batch_size,
            n_iter=self.params.n_iter,
            output_mode=self.params.output_mode,
            device=self.params.device,
        )
        DDHEmbedding.__init__(self, emb_model=self.model)

    def _fit(
        self,
        data: dataset.BaseDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        processed_data, event_times, event_values = self.prepare_fit(data)
        self.model.fit(processed_data, event_times, event_values)
        return self

    def _predict(
        self,
        data: dataset.PredictiveDataset,
        horizons: data_typing.TimeIndex,
        *args: Any,
        **kwargs: Any,
    ) -> samples.TimeSeriesSamples:
        # NOTE: kwargs will be passed to DynamicDeepHitModel.predict_risk().
        # E.g. `batch_size` batch size parameter can be provided this way.
        processed_data = self.prepare_predict(data, horizons, *args, **kwargs)
        risk = self.model.predict_risk(processed_data, horizons, **kwargs)
        return samples.TimeSeriesSamples(
            risk.reshape((risk.shape[0], risk.shape[1], 1)),
            sample_index=data.time_series.sample_index(),
            time_indexes=[horizons] * data.time_series.num_samples,  # pyright: ignore
            feature_index=["risk_score"],
        )

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # noqa: D102
        return DDHEmbedding.hyperparameter_space(*args, **kwargs)
