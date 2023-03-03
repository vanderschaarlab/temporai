import abc
from typing import Any

from tempor.data import dataset
from tempor.log import logger

from . import _base_estimator as estimator


class BasePredictor(estimator.BaseEstimator):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def predict(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> Any:  # TODO: Narrow down output formats later.
        logger.debug(f"Calling _predict() implementation on {self.__class__.__name__}")
        prediction = self._predict(data, *args, **kwargs)

        return prediction

    def fit_predict(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> Any:
        self.fit(data, *args, **kwargs)
        return self.predict(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict(self, data: dataset.Dataset, *args, **kwargs) -> Any:  # pragma: no cover
        ...
