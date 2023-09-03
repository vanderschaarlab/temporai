"""A module that contains methods for encoding data features, such as one-hot encoding.
"""

from . import static, temporal
from ._base import BaseEncoder

__all__ = [
    "BaseEncoder",
    "static",
    "temporal",
]
