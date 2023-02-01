import enum


class MethodTypes(enum.Enum):
    FIT = enum.auto()
    TRANSFORM = enum.auto()
    PREDICT = enum.auto()
    PREDICT_COUNTERFACTUAL = enum.auto()


def get_method_name(method_type: MethodTypes):
    if method_type == MethodTypes.FIT:
        return "fit"
    elif method_type == MethodTypes.TRANSFORM:
        return "transform"
    elif method_type == MethodTypes.PREDICT:
        return "predict"
    elif method_type == MethodTypes.PREDICT_COUNTERFACTUAL:
        return "predict_counterfactual"
    else:
        raise ValueError(f"Method name for method type {method_type} not defined")
