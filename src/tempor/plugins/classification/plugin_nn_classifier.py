from typing import Any, Optional

import numpy as np
from torch.utils.data import sampler
from typing_extensions import get_args

import tempor.plugins.core as plugins
from tempor.data import dataset, samples
from tempor.models.constants import DEVICE
from tempor.models.ts_model import Nonlin, TimeSeriesModel, TSModelMode
from tempor.plugins.classification import BaseClassifier
from tempor.plugins.core._params import CategoricalParams, FloatParams, IntegerParams


@plugins.register_plugin(name="nn_classifier", category="classification")
class NeuralNetClassifier(BaseClassifier):
    def __init__(
        self,
        *,
        n_static_units_hidden: int = 100,
        n_static_layers_hidden: int = 2,
        n_temporal_units_hidden: int = 102,
        n_temporal_layers_hidden: int = 2,
        n_iter: int = 500,
        mode: TSModelMode = "RNN",
        n_iter_print: int = 10,
        batch_size: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        window_size: int = 1,
        device: Any = DEVICE,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        dropout: float = 0,
        nonlin: Nonlin = "relu",
        random_state: int = 0,
        clipping_value: int = 1,
        patience: int = 20,
        train_ratio: float = 0.8,
    ) -> None:
        """Neural-net classifier.

        Args:
            n_static_units_hidden (int, optional):
                Number of hidden units for the static features. Defaults to ``100``.
            n_static_layers_hidden (int, optional):
                Number of hidden layers for the static features. Defaults to ``2``.
            n_temporal_units_hidden (int, optional):
                Number of hidden units for the temporal features. Defaults to ``102``.
            n_temporal_layers_hidden (int, optional):
                Number of hidden layers for the temporal features. Defaults to ``2``.
            n_iter (int, optional):
                Number of epochs. Defaults to ``500``.
            mode (TSModelMode, optional):
                Core neural net architecture. Available options: :obj:`~tempor.models.ts_model.TSModelMode`.
                Defaults to ``"RNN"``.
            n_iter_print (int, optional):
                Number of epochs to print the loss. Defaults to ``10``.
            batch_size (int, optional):
                Batch size. Defaults to ``100``.
            lr (float, optional):
                Learning rate. Defaults to ``1e-3``.
            weight_decay (float, optional):
                l2 (ridge) penalty for the weights. Defaults to ``1e-3``.
            window_size (int, optional):
                How many hidden states to use for the outcome. Defaults to ``1``.
            device (Any, optional):
                PyTorch device to use. Defaults to `~tempor.models.constants.DEVICE`.
            dataloader_sampler (Optional[sampler.Sampler], optional):
                Custom data sampler for training. Defaults to `None`.
            dropout (float, optional):
                Dropout value. Defaults to ``0``.
            nonlin (Nonlin, optional):
                Activation for hidden layers. Available options: :obj:`~tempor.models.constants.Nonlin`.
                Defaults to ``"relu"``.
            random_state (int, optional):
                Random seed. Defaults to ``0``.
            clipping_value (int, optional):
                Gradients clipping value. Zero disables the feature. Defaults to ``1``.
            patience (int, optional):
                How many epoch * n_iter_print to wait without loss improvement. Defaults to ``20``.
            train_ratio (float, optional):
                Train/test split ratio. Defaults to ``0.8``.

            Example:
            >>> from tempor.utils.datasets.sine import SineDataloader
            >>> from tempor.plugins import plugin_loader
            >>>
            >>> dataset = SineDataloader().load()
            >>>
            >>> # load the model
            >>> model = plugin_loader.get("classification.nn_classifier", n_iter=50)
            >>>
            >>> # train
            >>> model.fit(dataset)
            >>>
            >>> # predict
            >>> assert model.predict(dataset).numpy().shape == (len(dataset), 1)
        """
        super().__init__()
        self.n_static_units_hidden = n_static_units_hidden
        self.n_static_layers_hidden = n_static_layers_hidden
        self.n_temporal_units_hidden = n_temporal_units_hidden
        self.n_temporal_layers_hidden = n_temporal_layers_hidden
        self.n_iter = n_iter
        self.mode: TSModelMode = mode
        self.n_iter_print = n_iter_print
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.window_size = window_size
        self.device = device
        self.dataloader_sampler = dataloader_sampler
        self.dropout = dropout
        self.nonlin: Nonlin = nonlin
        self.random_state = random_state
        self.clipping_value = clipping_value
        self.patience = patience
        self.train_ratio = train_ratio

        self.model: Optional[TimeSeriesModel] = None

    def _fit(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> "NeuralNetClassifier":  # pyright: ignore
        static, temporal, observation_times, outcome = self._unpack_dataset(data)
        outcome = outcome.squeeze()

        n_classes = len(np.unique(outcome))

        self.model = TimeSeriesModel(
            task_type="classification",
            n_static_units_in=static.shape[-1],
            n_temporal_units_in=temporal.shape[-1],
            n_temporal_window=temporal.shape[1],
            output_shape=[n_classes],
            n_static_units_hidden=self.n_static_units_hidden,
            n_static_layers_hidden=self.n_static_layers_hidden,
            n_temporal_units_hidden=self.n_temporal_units_hidden,
            n_temporal_layers_hidden=self.n_temporal_layers_hidden,
            n_iter=self.n_iter,
            mode=self.mode,
            n_iter_print=self.n_iter_print,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            window_size=self.window_size,
            device=self.device,
            dataloader_sampler=self.dataloader_sampler,
            dropout=self.dropout,
            nonlin=self.nonlin,
            random_state=self.random_state,
            clipping_value=self.clipping_value,
            patience=self.patience,
            train_ratio=self.train_ratio,
        )

        self.model.fit(static, temporal, observation_times, outcome)
        return self

    def _predict(
        self,
        data: dataset.Dataset,
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
        data: dataset.Dataset,
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
            IntegerParams(name="n_static_units_hidden", low=100, high=1000),
            IntegerParams(name="n_static_layers_hidden", low=1, high=5),
            IntegerParams(name="n_temporal_units_hidden", low=100, high=1000),
            IntegerParams(name="n_temporal_layers_hidden", low=1, high=5),
            CategoricalParams(name="mode", choices=list(get_args(TSModelMode))),
            CategoricalParams(name="batch_size", choices=[64, 128, 256, 512]),
            CategoricalParams(name="lr", choices=[1e-3, 1e-4, 2e-4]),
            FloatParams(name="dropout", low=0, high=0.2),
            CategoricalParams(name="nonlin", choices=["relu", "elu", "leaky_relu", "selu"]),
        ]
