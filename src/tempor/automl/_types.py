"""Type definitions for AutoML."""

from typing import Type, Union

from tempor.methods.core import BasePredictor

from .pipeline_selector import PipelineSelector

AutoMLCompatibleEstimator = Union[Type[BasePredictor], PipelineSelector]
"""Estimator types supported by AutoML. Either a predictor class,
or a special `PipelineSelector` object for pipeline search.
"""
