import dataclasses
from typing import Optional

import numpy as np
from typing_extensions import Self

from tempor.core import plugins
from tempor.data import dataset, samples
from tempor.methods.core._params import CategoricalParams, FloatParams, IntegerParams
from tempor.methods.prediction.one_off.classification import BaseOneOffClassifier
from tempor.models import utils as model_utils
from tempor.models.constants import Nonlin, Samp
from tempor.models.ts_ode import Interpolation, NeuralODE


@dataclasses.dataclass
class CDEClassifierParams:
    """Initialization parameters for :class:`CDEClassifier`."""

    n_units_hidden: int = 100
    """Number of hidden units."""
    n_layers_hidden: int = 1
    """Number of hidden layers."""
    nonlin: Nonlin = "relu"
    """Activation for hidden layers. Available options: :obj:`~tempor.models.constants.Nonlin`."""
    dropout: float = 0
    """Dropout value."""

    # CDE specific:
    atol: float = 1e-2
    """Absolute tolerance for solution."""
    rtol: float = 1e-2
    """Relative tolerance for solution."""
    interpolation: Interpolation = "cubic"
    """``"cubic"`` or ``"linear"``."""

    # Training:
    lr: float = 1e-3
    """Learning rate for optimizer."""
    weight_decay: float = 1e-3
    """l2 (ridge) penalty for the weights."""
    n_iter: int = 1000
    """Maximum number of iterations."""
    batch_size: int = 500
    """Batch size."""
    n_iter_print: int = 100
    """Number of iterations after which to print updates and check the validation loss."""
    random_state: int = 0
    """Random_state used."""
    patience: int = 10
    """Number of iterations to wait before early stopping after decrease in validation loss."""
    clipping_value: int = 1
    """Gradients clipping value."""
    train_ratio: float = 0.8
    """Train/test split ratio."""
    device: Optional[str] = None
    """String representing PyTorch device. If `None`, `~tempor.models.constants.DEVICE`."""
    dataloader_sampler: Optional[Samp] = None
    """Custom data sampler for training."""


@plugins.register_plugin(name="cde_classifier", category="prediction.one_off.classification")
class CDEClassifier(BaseOneOffClassifier):
    ParamsDefinition = CDEClassifierParams
    params: CDEClassifierParams  # type: ignore

    def __init__(self, **params) -> None:
        """Neural Controlled Differential Equations for Irregular Time Series.

        Args:
            params:
                Parameters and defaults as defined in :class:`CDEClassifierParams`.

        Example:
            >>> from tempor.datasources import SineDataSource
            >>> from tempor import plugin_loader
            >>>
            >>> dataset = SineDataSource().load()
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("prediction.one_off.classification.cde_classifier", n_iter=50)
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            CDEClassifier(...)
            >>>
            >>> # Predict:
            >>> assert model.predict(dataset).numpy().shape == (len(dataset), 1)

        References:
            "Neural Controlled Differential Equations for Irregular Time Series", Patrick Kidger, James Morrill,
            James Foster, Terry Lyons.
        """
        super().__init__(**params)

        self.device = model_utils.get_device(self.params.device)
        self.dataloader_sampler = model_utils.get_sampler(self.params.dataloader_sampler)

        self.model: Optional[NeuralODE] = None

    def _fit(
        self,
        data: dataset.BaseDataset,
        *args,
        **kwargs,
    ) -> Self:
        static, temporal, observation_times, outcome = self._unpack_dataset(data)
        outcome = outcome.squeeze()

        n_classes = len(np.unique(outcome))

        self.model = NeuralODE(
            task_type="classification",
            n_static_units_in=static.shape[-1],
            n_temporal_units_in=temporal.shape[-1],
            output_shape=[n_classes],
            n_units_hidden=self.params.n_units_hidden,
            n_layers_hidden=self.params.n_layers_hidden,
            # CDE
            backend="cde",
            atol=self.params.atol,
            rtol=self.params.rtol,
            interpolation=self.params.interpolation,
            # training
            n_iter=self.params.n_iter,
            n_iter_print=self.params.n_iter_print,
            batch_size=self.params.batch_size,
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
            device=self.device,
            dataloader_sampler=self.dataloader_sampler,
            dropout=self.params.dropout,
            nonlin=self.params.nonlin,
            random_state=self.params.random_state,
            clipping_value=self.params.clipping_value,
            patience=self.params.patience,
            train_ratio=self.params.train_ratio,
        )

        self.model.fit(static, temporal, observation_times, outcome)
        return self

    def _predict(
        self,
        data: dataset.PredictiveDataset,
        *args,
        **kwargs,
    ) -> samples.StaticSamples:
        if self.model is None:
            raise RuntimeError("Fit the model first")
        static, temporal, observation_times, _ = self._unpack_dataset(data)

        preds = self.model.predict(static, temporal, observation_times)
        preds = preds.astype(float)

        preds = preds.reshape(-1, 1)
        return samples.StaticSamples.from_numpy(preds)

    def _predict_proba(
        self,
        data: dataset.PredictiveDataset,
        *args,
        **kwargs,
    ) -> samples.StaticSamples:
        if self.model is None:
            raise RuntimeError("Fit the model first")
        static, temporal, observation_times, _ = self._unpack_dataset(data)

        preds = self.model.predict_proba(static, temporal, observation_times)
        preds = preds.astype(float)

        return samples.StaticSamples.from_numpy(preds)

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return [
            IntegerParams(name="n_units_hidden", low=100, high=1000),
            IntegerParams(name="n_layers_hidden", low=1, high=5),
            FloatParams(name="atol", low=0, high=1),
            FloatParams(name="rtol", low=0, high=1),
            CategoricalParams(name="interpolation", choices=["cubic", "linear"]),
            CategoricalParams(name="batch_size", choices=[64, 128, 256, 512]),
            CategoricalParams(name="lr", choices=[1e-3, 1e-4, 2e-4]),
            FloatParams(name="dropout", low=0, high=0.2),
            CategoricalParams(name="nonlin", choices=["relu", "elu", "leaky_relu", "selu"]),
        ]
