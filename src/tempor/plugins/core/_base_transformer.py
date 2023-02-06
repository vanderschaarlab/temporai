import abc

from tempor.data.bundle._bundle import DataBundle as Dataset
from tempor.log import logger

from . import _base_estimator as estimator
from . import _types as types


class BaseTransformer(estimator.BaseEstimator):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)

    def transform(
        self,
        data: Dataset,
        *args,
        **kwargs,
    ) -> Dataset:
        logger.debug(f"Validating transform() config on {self.__class__.__name__}")
        self._validate_estimator_method_config(data, estimator_method=types.EstimatorMethods.TRANSFORM)

        logger.debug(f"Calling _transform() implementation on {self.__class__.__name__}")
        transformed_data = self._transform(data, *args, **kwargs)

        return transformed_data

    def fit_transform(
        self,
        data: Dataset,
        *args,
        **kwargs,
    ) -> Dataset:
        self.fit(data, *args, **kwargs)
        return self.transform(data, *args, **kwargs)

    @abc.abstractmethod
    def _transform(self, data: Dataset, *args, **kwargs) -> Dataset:  # pragma: no cover
        ...
