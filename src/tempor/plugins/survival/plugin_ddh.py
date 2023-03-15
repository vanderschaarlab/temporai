import dataclasses

from typing_extensions import Literal

import tempor.plugins.core as plugins
from tempor.data import dataset, samples
from tempor.plugins.survival import BaseSurvivalAnalysis

RnnModes = Literal[
    "GRU",
    "LSTM",
    "RNN",
    # "Transformer",
]
OutputModes = Literal[
    "MLP",
    "LSTM",
    "GRU",
    "RNN",
    # "Transformer",
    # "TCN",
    # "InceptionTime",
    # "InceptionTimePlus",
    # "ResCNN",
    # "XCM",
]


@dataclasses.dataclass
class _Params:
    n_iter: int
    batch_size: int
    lr: float
    n_layers_hidden: int
    n_units_hidden: int
    split: int
    rnn_type: RnnModes
    alpha: float
    beta: float
    sigma: float
    random_state: int
    dropout: float
    device: str
    patience: int
    output_type: OutputModes


@plugins.register_plugin(name="dynamic_deephit", category="survival")
class DynamicDeepHitSurvivalAnalysis(BaseSurvivalAnalysis):
    ParamsDefinition = _Params

    def __init__(
        self,
        *,
        n_iter: int = 1000,
        batch_size: int = 100,
        lr: float = 1e-3,
        n_layers_hidden: int = 1,
        n_units_hidden: int = 40,
        split: int = 100,
        rnn_type: RnnModes = "GRU",
        alpha: float = 0.34,
        beta: float = 0.27,
        sigma: float = 0.21,
        random_state: int = 0,
        dropout: float = 0.06,
        device: str = "cpu",
        patience: int = 20,
        output_type: OutputModes = "MLP",
    ) -> None:
        super().__init__(
            n_iter=n_iter,
            batch_size=batch_size,
            lr=lr,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            split=split,
            rnn_type=rnn_type,
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            random_state=random_state,
            dropout=dropout,
            device=device,
            patience=patience,
            output_type=output_type,
        )
        print(self.params)
        # WIP...

    def fit(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> "DynamicDeepHitSurvivalAnalysis":  # pyright: ignore
        raise NotImplementedError

    def _fit(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> "DynamicDeepHitSurvivalAnalysis":  # pyright: ignore
        raise NotImplementedError

    def _predict(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        raise NotImplementedError

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return []
