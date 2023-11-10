"""Module with the base transformer class."""

import abc
from typing import Any

import pydantic

from tempor.core import pydantic_utils
from tempor.data import dataset
from tempor.log import logger

from . import _base_estimator as estimator


class BaseTransformer(estimator.BaseEstimator):
    def __init__(self, **params: Any) -> None:  # pylint: disable=useless-super-delegation
        """Abstract base class for all transformers.

        Defines some core methods, primarily:
        - ``transform``: Transforms the given data.
        - ``_transform``: The implementation of the above method in the derived classes.
        """
        super().__init__(**params)

    def transform(
        self,
        data: dataset.BaseDataset,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Transforms the given data.

        Args:
            data (dataset.BaseDataset): The dataset to transform.
            *args (Any): Additional positional arguments passed to the implementation (``_transform``).
            **kwargs (Any): Additional keyword arguments passed to the implementation (``_transform``).

        Returns:
            Any: The transformed data.
        """
        logger.debug(f"Calling _transform() implementation on {self.__class__.__name__}")
        transformed_data = self._transform(data, *args, **kwargs)

        return transformed_data

    @pydantic_utils.validate_arguments(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def fit_transform(
        self,
        data: dataset.BaseDataset,
        *args: Any,
        **kwargs: Any,
    ) -> dataset.BaseDataset:
        """Fit the method to the data and transform it. Equivalent to calling ``fit`` and then ``transform``.

        Args:
            data (dataset.BaseDataset): The dataset to fit and transform.
            *args (Any): Additional arguments to pass to the ``_fit`` and ``_transform`` methods.
            **kwargs (Any): Additional keyword arguments to pass to the ``_fit`` and ``_transform`` methods.

        Returns:
            dataset.BaseDataset: The transformed dataset.
        """
        self.fit(data, *args, **kwargs)
        return self.transform(data, *args, **kwargs)

    @abc.abstractmethod
    def _transform(
        self, data: dataset.BaseDataset, *args: Any, **kwargs: Any
    ) -> dataset.BaseDataset:  # pragma: no cover
        """The implementation of ``transform``. Must be implemented by the child class.

        Args:
            data (dataset.BaseDataset): The dataset to transform.
            *args (Any): Additional positional arguments received by the implementation.
            **kwargs (Any): Additional keyword arguments received by the implementation.

        Returns:
            dataset.BaseDataset: The transformed dataset.
        """
        ...
