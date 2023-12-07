"""CoxPH survival analysis model with Dynamic DeepHit embeddings."""

import contextlib
import dataclasses
from typing import Any, Dict, Generator, List, Optional

import lifelines
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from packaging.version import Version
from typing_extensions import Self

from tempor.core import plugins
from tempor.data import data_typing, dataset, samples
from tempor.methods.core.params import FloatParams, Params
from tempor.methods.time_to_event import BaseTimeToEventAnalysis
from tempor.models.ddh import DynamicDeepHitModel, OutputMode, RnnMode

from .helper_embedding import DDHEmbeddingTimeToEventAnalysis, OutputTimeToEventAnalysis


@contextlib.contextmanager
def monkeypatch_lifelines_pd2_compatibility() -> Generator:
    """lifelines (before 0.27.6) is not compatible with pandas 2.0.0+, due to
    ``TypeError: describe() got an unexpected keyword argument 'datetime_is_numeric'`` thrown by pandas in e.g.
    ``CoxPHFitter.fit``. This monkeypatch fixes this compatibility issue, until the problem is addressed by
    `lifelines`.
    """

    def problem_versions() -> bool:  # pragma: no cover
        # lifelines is compatible with pandas 2 version 0.27.6 onwards,
        # so the workaround is needed for lifelines < 0.27.6.
        return Version(pd.__version__) >= Version("2.0.0rc0") and Version(lifelines.__version__) < Version("0.27.6")

    if problem_versions():  # pragma: no cover
        # Monkeypatch `pandas.DataFrame.describe`.

        original_pd_df_describe = pd.DataFrame.describe

        def monkeypatched_describe(*args: Any, **kwargs: Any) -> pd.DataFrame:
            # Remove the offending keyword argument (it is no longer needed to pass in).
            kwargs.pop("datetime_is_numeric", None)
            return original_pd_df_describe(*args, **kwargs)

        pd.DataFrame.describe = monkeypatched_describe

    try:
        yield

    finally:
        if problem_versions():  # pragma: no cover
            # Restore `pandas.DataFrame.describe`.
            pd.DataFrame.describe = original_pd_df_describe  # pyright: ignore


@dataclasses.dataclass
class CoxPHTimeToEventAnalysisParams:
    # Output model:
    coxph_alpha: float = 0.05
    """``alpha`` parameter for `lifelines.CoxPHFitter`."""
    coxph_penalizer: float = 0.0
    """``penalizer`` parameter for `lifelines.CoxPHFitter`."""
    # Embedding model:
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
    """Early stopping (embeddings training): size of validation set."""
    patience: int = 20
    """Early stopping (embeddings training): training patience without any improvement."""
    output_mode: OutputMode = "MLP"
    """Output network, on of `OutputMode`."""
    random_state: int = 0
    """Random seed."""


def drop_constant_columns(dataframe: pd.DataFrame) -> list:
    """Drops constant value columns of pandas dataframe."""
    result = []
    for column in dataframe.columns:
        if len(dataframe[column].unique()) == 1:
            result.append(column)
    return result


class CoxPHSurvivalAnalysis(OutputTimeToEventAnalysis):
    def __init__(
        self,
        alpha: float = 0.05,
        penalizer: float = 0,
        fit_options: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """CoxPHFitter wrapper.

        Args:
            alpha (float, optional):
                The level in the confidence intervals. Defaults to ``0.05``.
            penalizer (float, optional):
                Attach a penalty to the size of the coefficients during regression. Defaults to ``0``.
            fit_options (Optional[Dict], optional):
                Pass kwargs for the fitting algorithm. Defaults to ``{"step_size": 0.1}``.
            **kwargs (Any):
                Additional keyword arguments for `lifelines.CoxPHFitter`.
        """
        if fit_options is None:
            fit_options = {"step_size": 0.1}
        self.fit_options = fit_options
        self.model = CoxPHFitter(alpha=alpha, penalizer=penalizer, **kwargs)

    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> Self:  # noqa: D102
        self.constant_cols = drop_constant_columns(X)  # pylint: disable=attribute-defined-outside-init
        X = X.drop(columns=self.constant_cols)

        df = X.copy()
        df["event"] = Y
        df["time"] = T

        with monkeypatch_lifelines_pd2_compatibility():
            self.model.fit(df, "time", "event", fit_options=self.fit_options)

        return self

    def predict_risk(self, X: pd.DataFrame, time_horizons: List) -> pd.DataFrame:  # noqa: D102
        """Predict risk estimation."""

        X = X.drop(columns=self.constant_cols)

        chunks = int(len(X) / 1024) + 1

        preds_ = []
        for chunk in np.array_split(X, chunks):
            local_preds_ = np.zeros([len(chunk), len(time_horizons)])
            surv = self.model.predict_survival_function(chunk)
            surv_times = np.asarray(surv.index).astype(int)
            surv = np.asarray(surv.T)

            for t, eval_time in enumerate(time_horizons):
                tmp_time = np.where(eval_time <= surv_times)[0]
                if len(tmp_time) == 0:
                    local_preds_[:, t] = 1.0 - surv[:, 0]
                else:
                    local_preds_[:, t] = 1.0 - surv[:, tmp_time[0]]

            preds_.append(local_preds_)

        return pd.DataFrame(np.concatenate(preds_, axis=0), columns=time_horizons, index=X.index)


@plugins.register_plugin(name="ts_coxph", category="time_to_event")
class CoxPHTimeToEventAnalysis(BaseTimeToEventAnalysis):
    ParamsDefinition = CoxPHTimeToEventAnalysisParams
    params: CoxPHTimeToEventAnalysisParams  # type: ignore

    def __init__(self, **params: Any) -> None:
        """CoxPH survival analysis model.

        Args:
            **params (Any):
                Parameters and defaults as defined in :class:`CoxPHTimeToEventAnalysisParams`.
        """
        super().__init__(**params)

        output_model = CoxPHSurvivalAnalysis(
            alpha=self.params.coxph_alpha,
            penalizer=self.params.coxph_penalizer,
        )
        self.model = DDHEmbeddingTimeToEventAnalysis(
            output_model=output_model,
            emb_model=DynamicDeepHitModel(
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
            ),
        )

    def _fit(
        self,
        data: dataset.BaseDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        self.model.fit(data, *args, **kwargs)
        return self

    def _predict(
        self,
        data: dataset.PredictiveDataset,
        horizons: data_typing.TimeIndex,
        *args: Any,
        **kwargs: Any,
    ) -> samples.TimeSeriesSamplesBase:
        return self.model.predict(data, horizons, *args, **kwargs)

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:  # noqa: D102
        return [
            FloatParams(name="coxph_alpha", low=0.05, high=0.1),
            FloatParams(name="coxph_penalizer", low=0, high=0.2),
        ] + DDHEmbeddingTimeToEventAnalysis.hyperparameter_space()
