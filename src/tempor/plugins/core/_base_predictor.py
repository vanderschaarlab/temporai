import abc
from typing import Any

import pydantic

from tempor.data import dataset
from tempor.log import logger

from . import _base_estimator as estimator


class BasePredictor(estimator.BaseEstimator):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def predict(
        self,
        data: dataset.BaseDataset,
        *args,
        **kwargs,
    ) -> Any:  # TODO: Narrow down output formats later.
        logger.debug(f"Calling _predict() implementation on {self.__class__.__name__}")
        prediction = self._predict(data, *args, **kwargs)

        return prediction

    def predict_proba(
        self,
        data: dataset.BaseDataset,
        *args,
        **kwargs,
    ) -> Any:  # TODO: Narrow down output formats later.
        logger.debug(f"Calling _predict_proba() implementation on {self.__class__.__name__}")
        prediction = self._predict_proba(data, *args, **kwargs)

        return prediction

    def predict_counterfactuals(
        self,
        data: dataset.BaseDataset,
        *args,
        **kwargs,
    ) -> Any:  # TODO: Narrow down output formats later.
        logger.debug(f"Calling _predict_counterfactuals() implementation on {self.__class__.__name__}")
        prediction = self._predict_counterfactuals(data, *args, **kwargs)

        return prediction

    @pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit_predict(
        self,
        data: dataset.BaseDataset,
        *args,
        **kwargs,
    ) -> Any:
        self.fit(data, *args, **kwargs)
        return self.predict(data, *args, **kwargs)

    @abc.abstractmethod
    def _predict(self, data: dataset.BaseDataset, *args, **kwargs) -> Any:  # pragma: no cover
        ...

    def _predict_proba(self, data: dataset.BaseDataset, *args, **kwargs) -> Any:  # pragma: no cover
        raise NotImplementedError("`predict_proba` is supported only for classification tasks")

    def _predict_counterfactuals(self, data: dataset.BaseDataset, *args, **kwargs) -> Any:  # pragma: no cover
        raise NotImplementedError("`predict_proba` is supported only for treatments tasks")
