"""Module with the base predictor class."""

import abc
from typing import Any

import pydantic

from tempor.core import pydantic_utils
from tempor.data import dataset
from tempor.log import logger

from . import _base_estimator as estimator


class BasePredictor(estimator.BaseEstimator):
    def __init__(self, **params: Any) -> None:  # pylint: disable=useless-super-delegation
        """Abstract base class for all predictors.

        Defines some core methods, primarily:
        - ``predict``: Predicts the target variable for the given data.
        - ``predict_proba``: Predicts the probability of the target variable for the given data.
        - ``predict_counterfactuals``: Predicts the counterfactuals for the given data.
        - The `_` versions of the above methods are the implementations of the above methods in the derived classes.
        """
        super().__init__(**params)

    def predict(
        self,
        data: dataset.PredictiveDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Predicts the target variable for the given data.

        Args:
            data (dataset.PredictiveDataset): The dataset to predict on.
            *args (Any): Additional positional arguments passed to the implementation (``_predict``).
            **kwargs (Any): Additional keyword arguments passed to the implementation (``_predict``).

        Returns:
            Any: The predictions.
        """
        if not self.is_fitted:
            raise ValueError("The model was not fitted, call `fit` first")
        if not data.predict_ready:
            raise ValueError(
                f"The dataset was not predict-ready, check that all necessary data components are present:\n{data}"
            )

        logger.debug(f"Calling _predict() implementation on {self.__class__.__name__}")
        prediction = self._predict(data, *args, **kwargs)

        return prediction

    def predict_proba(
        self,
        data: dataset.PredictiveDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Predicts the probability of the target variable for the given data.

        Args:
            data (dataset.PredictiveDataset): The dataset to predict on.
            *args (Any): Additional positional arguments passed to the implementation (``_predict_proba``).
            **kwargs (Any): Additional keyword arguments passed to the implementation (``_predict_proba``).

        Returns:
            Any: The predicted probabilities.
        """
        if not self.is_fitted:
            raise ValueError("The model was not fitted, call `fit` first")
        if not data.predict_ready:
            raise ValueError(
                f"The dataset was not predict-ready, check that all necessary data components are present:\n{data}"
            )

        logger.debug(f"Calling _predict_proba() implementation on {self.__class__.__name__}")
        prediction = self._predict_proba(data, *args, **kwargs)

        return prediction

    def predict_counterfactuals(
        self,
        data: dataset.PredictiveDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Predicts the counterfactuals for the given data.

        Args:
            data (dataset.PredictiveDataset): The dataset to predict on.
            *args (Any): Additional positional arguments passed to the implementation (``_predict_counterfactuals``).
            **kwargs (Any): Additional keyword arguments passed to the implementation (``_predict_counterfactuals``).

        Returns:
            Any: The predicted counterfactuals.
        """
        if not self.is_fitted:
            raise ValueError("The model was not fitted, call `fit` first")
        if not data.predict_ready:
            raise ValueError(
                f"The dataset was not predict-ready, check that all necessary data components are present:\n{data}"
            )

        logger.debug(f"Calling _predict_counterfactuals() implementation on {self.__class__.__name__}")
        prediction = self._predict_counterfactuals(data, *args, **kwargs)

        return prediction

    # TODO: Add similar methods for predict_{proba,counterfactuals}.
    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def fit_predict(
        self,
        data: dataset.PredictiveDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Fit the model to the data and then predict on the same data. Equivalent to calling ``fit`` and then
        ``predict``.

        Args:
            data (dataset.PredictiveDataset): The dataset to fit and predict on.
            *args (Any): Additional positional arguments passed to the implementations (``_fit`` and ``_predict``).
            **kwargs (Any): Additional keyword arguments passed to the implementations (``_fit`` and ``_predict``).

        Returns:
            Any: The predictions.
        """
        self.fit(data, *args, **kwargs)
        return self.predict(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict(self, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        """The implementation of ``predict``. Must be implemented by the child class.

        Args:
            data (dataset.PredictiveDataset): The dataset to predict on.
            *args (Any): Additional positional arguments received by the implementation.
            **kwargs (Any): Additional keyword arguments received by the implementation.

        Returns:
            Any: The predictions.
        """
        ...  # pylint: disable=unnecessary-ellipsis

    def _predict_proba(self, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any) -> Any:
        """The implementation of ``predict_proba``. Must be implemented by the child class.

        Args:
            data (dataset.PredictiveDataset): The dataset to predict on.
            *args (Any): Additional positional arguments received by the implementation.
            **kwargs (Any): Additional keyword arguments received by the implementation.

        Returns:
            Any: The predicted probabilities.
        """
        raise NotImplementedError("`predict_proba` is supported only for classification tasks")

    def _predict_counterfactuals(self, data: dataset.PredictiveDataset, *args: Any, **kwargs: Any) -> Any:
        """The implementation of ``predict_counterfactuals``. Must be implemented by the child class.

        Args:
            data (dataset.PredictiveDataset): The dataset to predict on.
            *args (Any): Additional positional arguments received by the implementation.
            **kwargs (Any): Additional keyword arguments received by the implementation.

        Returns:
            Any: The predicted counterfactuals.
        """
        raise NotImplementedError("`predict_counterfactuals` is supported only for treatments tasks")
