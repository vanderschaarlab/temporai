import abc
from typing import Any

from tempor.data import dataset
from tempor.log import logger

from . import _base_estimator as estimator


class BaseTransformer(estimator.BaseEstimator):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def transform(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> Any:
        logger.debug(f"Calling _transform() implementation on {self.__class__.__name__}")
        transformed_data = self._transform(data, *args, **kwargs)

        return transformed_data

    def fit_transform(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> Any:
        self.fit(data, *args, **kwargs)
        return self.transform(data, *args, **kwargs)

    @abc.abstractmethod
    def _transform(self, data: dataset.Dataset, *args, **kwargs) -> Any:  # pragma: no cover
        ...
