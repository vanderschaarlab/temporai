import dataclasses
from typing import Any, List

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from typing_extensions import Self

import tempor.plugins.core as plugins
from tempor.data import data_typing, dataset, samples
from tempor.models.ddh import OutputMode, RnnMode
from tempor.plugins.core._params import FloatParams
from tempor.plugins.time_to_event import BaseTimeToEventAnalysis

from .helper_embedding import EmbTimeToEventAnalysis, OutputTimeToEventAnalysis


@dataclasses.dataclass
class CoxPHTimeToEventAnalysisParams:
    # TODO: Docstring.
    # Output model
    coxph_alpha: float = 0.05
    coxph_penalizer: float = 0
    # Embedding model
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


def constant_columns(dataframe: pd.DataFrame) -> list:
    """
    Drops constant value columns of pandas dataframe.
    """
    result = []
    for column in dataframe.columns:
        if len(dataframe[column].unique()) == 1:
            result.append(column)
    return result


class CoxPHSurvivalAnalysis(OutputTimeToEventAnalysis):
    """CoxPHFitter wrapper
    Args:
        alpha: float
            the level in the confidence intervals.
        penalizer: float
            Attach a penalty to the size of the coefficients during regression.
        fit_options: dict
             pass kwargs for the fitting algorithm.
    """

    def __init__(
        self, alpha: float = 0.05, penalizer: float = 0, fit_options: dict = {"step_size": 0.1}, **kwargs: Any
    ) -> None:
        super().__init__()
        self.fit_options = fit_options
        self.model = CoxPHFitter(alpha=alpha, **kwargs)

    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "CoxPHSurvivalAnalysis":
        self.constant_cols = constant_columns(X)
        X = X.drop(columns=self.constant_cols)

        df = X.copy()
        df["event"] = Y
        df["time"] = T

        self.model.fit(df, "time", "event", fit_options=self.fit_options)

        return self

    def predict_risk(self, X: pd.DataFrame, time_horizons: List) -> pd.DataFrame:
        "Predict risk estimation"

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


# TODO: Docstring.
@plugins.register_plugin(name="ts_coxph", category="time_to_event")
class CoxPHTimeToEventAnalysis(BaseTimeToEventAnalysis):
    ParamsDefinition = CoxPHTimeToEventAnalysisParams
    params: CoxPHTimeToEventAnalysisParams  # type: ignore

    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        """CoxPH survival analysis model.

        Args:
            **params:
                Parameters and defaults as defined in :class:`CoxPHTimeToEventAnalysisParams`.
        """
        super().__init__(**params)

        output_model = CoxPHSurvivalAnalysis(
            alpha=self.params.coxph_alpha,
            penalizer=self.params.coxph_penalizer,
        )
        self.model = EmbTimeToEventAnalysis(
            output_model=output_model,
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

    def _fit(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> Self:
        self.model.fit(data, *args, **kwargs)
        return self

    def _predict(  # type: ignore[override]
        self,
        data: dataset.Dataset,
        horizons: data_typing.TimeIndex,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        return self.model.predict(data, horizons, *args, **kwargs)

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return [
            FloatParams(name="coxph_alpha", low=0.05, high=0.1),
            FloatParams(name="coxph_penalizer", low=0, high=0.2),
        ] + EmbTimeToEventAnalysis.hyperparameter_space()
