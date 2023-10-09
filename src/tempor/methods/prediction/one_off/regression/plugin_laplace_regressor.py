import dataclasses
from typing import Optional

from typing_extensions import Self

import tempor.methods.core as plugins
from tempor.data import dataset, samples
from tempor.methods.core._params import CategoricalParams, FloatParams, IntegerParams
from tempor.methods.prediction.one_off.regression import BaseOneOffRegressor
from tempor.models import utils as model_utils
from tempor.models.constants import Nonlin, Samp
from tempor.models.ts_ode import ILTAlgorithm, NeuralODE


@dataclasses.dataclass
class LaplaceODERegressorParams:
    """Initialization parameters for :class:`LaplaceODERegressor`."""

    n_units_hidden: int = 100
    """Number of hidden units."""
    n_layers_hidden: int = 1
    """Number of hidden layers."""
    nonlin: Nonlin = "relu"
    """Activation for hidden layers. Available options: :obj:`~tempor.models.constants.Nonlin`."""
    dropout: float = 0
    """Dropout value."""

    # CDE specific:
    ilt_reconstruction_terms: int = 33
    """Number of ILT reconstruction terms, i.e. the number of complex :math:`s` points in
    ``laplace_rep_func`` to reconstruct a single time point."""
    ilt_algorithm: ILTAlgorithm = "fourier"
    """Inverse Laplace transform algorithm to use. Available are {``fourier``, ``dehoog``, ``cme``,
    ``fixed_tablot``, ``stehfest``}."""

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


@plugins.register_plugin(name="laplace_ode_regressor", category="prediction.one_off.regression")
class LaplaceODERegressor(BaseOneOffRegressor):
    ParamsDefinition = LaplaceODERegressorParams
    params: LaplaceODERegressorParams  # type: ignore

    def __init__(self, **params) -> None:
        """Inverse Laplace Transform (ILT) algorithms implemented in PyTorch.
        Backpropagation through differential equation (DE) solutions in the Laplace domain is supported using the
        Riemann stereographic projection for better global representation of the complex Laplace domain.

        Args:
            **params:
                Parameters and defaults as defined in :class:`LaplaceODERegressorParams`.

        Example:
            >>> from tempor.data.datasources import GoogleStocksDataSource
            >>> from tempor.methods import plugin_loader
            >>>
            >>> dataset = GoogleStocksDataSource().load()
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("prediction.one_off.regression.laplace_ode_regressor", n_iter=50)
            >>>
            >>> # Train:
            >>> model.fit(dataset)
            LaplaceODERegressor(...)
            >>>
            >>> # Predict:
            >>> assert model.predict(dataset).numpy().shape == (len(dataset), 1)

        References:
            "Neural Laplace: Learning diverse classes of differential equations in the Laplace domain",
            Holt, Samuel I and Qian, Zhaozhi and van der Schaar, Mihaela.
        """
        # TODO: Model currently fails to run with SineDataSource data. Investigate and resolve.

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

        self.model = NeuralODE(
            task_type="regression",
            n_static_units_in=static.shape[-1],
            n_temporal_units_in=temporal.shape[-1],
            output_shape=[1],
            n_units_hidden=self.params.n_units_hidden,
            n_layers_hidden=self.params.n_layers_hidden,
            # Laplace
            backend="laplace",
            ilt_algorithm=self.params.ilt_algorithm,
            ilt_reconstruction_terms=self.params.ilt_reconstruction_terms,
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

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return [
            IntegerParams(name="n_units_hidden", low=100, high=1000),
            IntegerParams(name="n_layers_hidden", low=1, high=5),
            CategoricalParams(name="ilt_algorithm", choices=["fourier", "dehoog", "cme", "fixed_tablot", "stehfest"]),
            CategoricalParams(name="batch_size", choices=[64, 128, 256, 512]),
            CategoricalParams(name="lr", choices=[1e-3, 1e-4, 2e-4]),
            FloatParams(name="dropout", low=0, high=0.2),
            CategoricalParams(name="nonlin", choices=["relu", "elu", "leaky_relu", "selu"]),
        ]
