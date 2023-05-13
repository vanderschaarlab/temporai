from typing_extensions import Literal

PredictiveTaskType = Literal[
    "prediction.one_off.classification",
    "prediction.one_off.regression",
    "prediction.temporal.classification",
    "prediction.temporal.regression",
    "time_to_event",
    "treatments.one_off.classification",
    "treatments.one_off.regression",
    "treatments.temporal.classification",
    "treatments.temporal.regression",
]
