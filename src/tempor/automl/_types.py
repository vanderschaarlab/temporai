from typing import Type, Union

from typing_extensions import Literal

from tempor.plugins.core import BasePredictor

from .pipeline_selector import PipelineSelector

OptimDirection = Literal["minimize", "maximize"]
"""Optimization direction for AutoML (with respect to a metric)"""

AutoMLCompatibleEstimator = Union[Type[BasePredictor], PipelineSelector]
"""Estimator types supported by AutoML. Either a predictor class,
or a special `PipelineSelector` object for pipeline search.
"""
