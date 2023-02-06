import enum


class EstimatorMethods(enum.Enum):
    FIT = enum.auto()
    TRANSFORM = enum.auto()
    PREDICT = enum.auto()
    PREDICT_COUNTERFACTUAL = enum.auto()
