import abc
from typing import Any

from tempor.data.bundle._bundle import DataBundle as Dataset
from tempor.log import logger

from . import _base_estimator as estimator
from . import _types as types


class BasePredictor(estimator.BaseEstimator):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def predict(
        self,
        data: Dataset,
        *args,
        **kwargs,
    ) -> Any:  # TODO: Narrow down output formats later.
        logger.debug(f"Validating predict() config on {self.__class__.__name__}")
        self._validate_estimator_method_config(data, estimator_method=types.EstimatorMethods.PREDICT)

        logger.debug(f"Calling _predict() implementation on {self.__class__.__name__}")
        prediction = self._predict(data, *args, **kwargs)

        return prediction

    def fit_predict(
        self,
        data: Dataset,
        *args,
        **kwargs,
    ) -> Any:
        self.fit(data, *args, **kwargs)
        return self.predict(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict(self, data: Dataset, *args, **kwargs) -> Any:  # pragma: no cover
        ...
