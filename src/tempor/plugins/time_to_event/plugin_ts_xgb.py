import dataclasses
from typing import Any, List

import numpy as np
import pandas as pd
from typing_extensions import Literal, Self
from xgbse import XGBSEDebiasedBCE, XGBSEKaplanNeighbors, XGBSEStackedWeibull
from xgbse.converters import convert_to_structured

import tempor.plugins.core as plugins
from tempor.data import data_typing, dataset, samples
from tempor.models.constants import DEVICE
from tempor.models.ddh import DynamicDeepHitModel, OutputMode, RnnMode
from tempor.plugins.core._params import CategoricalParams, FloatParams, IntegerParams
from tempor.plugins.time_to_event import BaseTimeToEventAnalysis

from .helper_embedding import DDHEmbeddingTimeToEventAnalysis, OutputTimeToEventAnalysis

XGBObjective = Literal["aft", "cox"]
XGBStrategy = Literal["weibull", "debiased_bce", "km"]


@dataclasses.dataclass
class XGBTimeToEventAnalysisParams:
    # TODO: Docstring.
    # Output model:
    xgb_n_estimators: int = 100
    """Respective parameter for `xgbse` ``XGBSE<Method>`` class initializer ``xgb_params``."""
    xgb_colsample_bynode: float = 1.0
    """Respective parameter for `xgbse` ``XGBSE<Method>`` class initializer ``xgb_params``."""
    xgb_colsample_bytree: float = 1.0
    """Respective parameter for `xgbse` ``XGBSE<Method>`` class initializer ``xgb_params``."""
    xgb_colsample_bylevel: float = 1.0
    """Respective parameter for `xgbse` ``XGBSE<Method>`` class initializer ``xgb_params``."""
    xgb_max_depth: int = 5
    """Respective parameter for `xgbse` ``XGBSE<Method>`` class initializer ``xgb_params``."""
    xgb_subsample: float = 0.5
    """Respective parameter for `xgbse` ``XGBSE<Method>`` class initializer ``xgb_params``."""
    xgb_learning_rate: float = 5e-2
    """Respective parameter for `xgbse` ``XGBSE<Method>`` class initializer ``xgb_params``."""
    xgb_min_child_weight: int = 50
    """Respective parameter for `xgbse` ``XGBSE<Method>`` class initializer ``xgb_params``."""
    xgb_tree_method: str = "hist"
    """Respective parameter for `xgbse` ``XGBSE<Method>`` class initializer ``xgb_params``."""
    xgb_booster: int = 0
    """Respective parameter for `xgbse` ``XGBSE<Method>`` class initializer ``xgb_params``."""
    xgb_objective: XGBObjective = "aft"
    """XGB Objective, one of `XGBObjective`."""
    xgb_strategy: XGBStrategy = "debiased_bce"
    """XGB Objective, one of `XGBStrategy`:
    weibull: `XGBSEStackedWeibull`, debiased_bce: `XGBSEDebiasedBCE`, km: `XGBSEKaplanNeighbors`.
    """
    xgb_bce_n_iter: int = 1000
    """Parameter for `xgbse` ``XGBSEDebiasedBCE`` initializer ``lr_params.max_iter``."""
    xgb_time_points: int = 100
    """Number of discrete time points to use."""
    xgb_reg_lambda: float = 1
    """Respective parameter for `xgbse` ``XGBSE<Method>`` class initializer ``xgb_params``."""
    xgb_reg_alpha: float = 0
    """Respective parameter for `xgbse` ``XGBSE<Method>`` class initializer ``xgb_params``."""
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


class XGBSurvivalAnalysis(OutputTimeToEventAnalysis):
    booster = ["gbtree", "gblinear", "dart"]

    def __init__(
        self,
        n_estimators: int = 100,
        colsample_bynode: float = 1,
        colsample_bylevel: float = 1,
        colsample_bytree: float = 1,
        max_depth: int = 5,
        subsample: float = 0.5,
        learning_rate: float = 5e-2,
        min_child_weight: int = 50,
        tree_method: str = "hist",
        booster: int = 0,
        random_state: int = 0,
        objective: XGBObjective = "aft",
        strategy: XGBStrategy = "debiased_bce",
        bce_n_iter: int = 1000,
        time_points: int = 100,
        reg_lambda: float = 1,
        reg_alpha: float = 0,
        device: Any = DEVICE,  # pylint: disable=unused-argument
        **kwargs: Any,
    ) -> None:
        super().__init__()

        surv_params = {}
        if objective == "aft":
            surv_params = {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": 1.0,
            }
        else:
            surv_params = {
                "objective": "survival:cox",
                "eval_metric": "cox-nloglik",
            }
        xgboost_params = {
            # survival
            **surv_params,
            **kwargs,
            # basic xgboost
            "n_estimators": n_estimators,
            "colsample_bynode": colsample_bynode,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bytree": colsample_bytree,
            "max_depth": max_depth,
            "subsample": subsample,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "verbosity": 0,
            "tree_method": tree_method,
            "booster": XGBSurvivalAnalysis.booster[booster],
            "random_state": random_state,
            "n_jobs": 2,
        }
        lr_params = {
            "C": 1e-3,
            "max_iter": bce_n_iter,
        }

        if strategy == "debiased_bce":
            self.model = XGBSEDebiasedBCE(xgboost_params, lr_params)
        elif strategy == "weibull":
            self.model = XGBSEStackedWeibull(xgboost_params)
        elif strategy == "km":
            self.model = XGBSEKaplanNeighbors(xgboost_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.time_points = time_points

    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series):
        y = convert_to_structured(T, Y)

        censored_times = T[Y == 0]
        obs_times = T[Y == 1]

        lower_bound = max(censored_times.min(), obs_times.min()) + 1
        if pd.isna(lower_bound):  # pragma: no cover
            lower_bound = T.min()
        upper_bound = T.max()

        time_bins = np.linspace(lower_bound, upper_bound, self.time_points, dtype=int)

        self.model.fit(X, y, time_bins=time_bins)
        return self

    def _find_nearest(self, array: np.ndarray, value: float) -> float:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def predict_risk(self, X: pd.DataFrame, time_horizons: List) -> pd.DataFrame:
        """Predict risk."""
        chunks = int(len(X) / 1024) + 1

        preds_ = []
        for chunk in np.array_split(X, chunks):
            local_preds_ = np.zeros([len(chunk), len(time_horizons)])
            surv = self.model.predict(chunk)
            surv = surv.loc[:, ~surv.columns.duplicated()]  # pyright: ignore
            time_bins = surv.columns
            for t, eval_time in enumerate(time_horizons):
                nearest = self._find_nearest(time_bins, eval_time)
                local_preds_[:, t] = np.asarray(1 - surv[nearest])
            preds_.append(local_preds_)
        return pd.DataFrame(np.concatenate(preds_, axis=0), columns=time_horizons, index=X.index)


@plugins.register_plugin(name="ts_xgb", category="time_to_event")
class XGBTimeToEventAnalysis(BaseTimeToEventAnalysis):
    ParamsDefinition = XGBTimeToEventAnalysisParams
    params: XGBTimeToEventAnalysisParams  # type: ignore

    def __init__(self, **params) -> None:
        """XGB survival analysis model.

        Args:
            **params:
                Parameters and defaults as defined in :class:`XGBTimeToEventAnalysisParams`.
        """
        super().__init__(**params)

        output_model = XGBSurvivalAnalysis(
            n_estimators=self.params.xgb_n_estimators,
            colsample_bynode=self.params.xgb_colsample_bynode,
            colsample_bytree=self.params.xgb_colsample_bytree,
            colsample_bylevel=self.params.xgb_colsample_bylevel,
            reg_alpha=self.params.xgb_reg_alpha,
            reg_lambda=self.params.xgb_reg_lambda,
            max_depth=self.params.xgb_max_depth,
            subsample=self.params.xgb_subsample,
            learning_rate=self.params.xgb_learning_rate,
            min_child_weight=self.params.xgb_min_child_weight,
            tree_method=self.params.xgb_tree_method,
            booster=self.params.xgb_booster,
            random_state=self.params.random_state,
            objective=self.params.xgb_objective,
            strategy=self.params.xgb_strategy,
            bce_n_iter=self.params.xgb_bce_n_iter,
            time_points=self.params.xgb_time_points,
            device=self.params.device,
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
        *args,
        **kwargs,
    ) -> Self:
        self.model.fit(data, *args, **kwargs)
        return self

    def _predict(  # type: ignore[override]
        self,
        data: dataset.PredictiveDataset,
        horizons: data_typing.TimeIndex,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        return self.model.predict(data, horizons, *args, **kwargs)

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return [
            IntegerParams(name="xgb_max_depth", low=2, high=6),
            IntegerParams(name="xgb_min_child_weight", low=0, high=50),
            CategoricalParams(name="xgb_objective", choices=["aft", "cox"]),
            CategoricalParams(name="xgb_strategy", choices=["weibull", "debiased_bce", "km"]),
            FloatParams(name="xgb_reg_lambda", low=1e-3, high=10.0),
            FloatParams(name="xgb_reg_alpha", low=1e-3, high=10.0),
            FloatParams(name="xgb_colsample_bytree", low=0.1, high=1),
            FloatParams(name="xgb_colsample_bynode", low=0.1, high=1),
            FloatParams(name="xgb_colsample_bylevel", low=0.1, high=1),
            FloatParams(name="xgb_subsample", low=0.1, high=0.9),
            FloatParams(name="xgb_learning_rate", low=1e-4, high=1e-2),
            IntegerParams(name="xgb_max_bin", low=256, high=512),
            IntegerParams(name="xgb_booster", low=0, high=len(XGBSurvivalAnalysis.booster) - 1),
        ] + DDHEmbeddingTimeToEventAnalysis.hyperparameter_space()
