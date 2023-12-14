from typing import List, Sequence, Union

import numpy as np
import pandas as pd
import torch

from ...interface import TCounterfactualPredictions
from ...interface.horizon import TimeIndexHorizon
from .. import TimeSeries

# TODO: Test


def to_counterfactual_predictions(
    list_counterfactual_predictions: Sequence[Union[np.ndarray, torch.Tensor]],
    data_historic_temporal_targets: TimeSeries,
    horizon: TimeIndexHorizon,
) -> TCounterfactualPredictions:
    assert len(horizon.time_index_sequence) == 1
    time_index = horizon.time_index_sequence[0]

    list_ts: List[TimeSeries] = []
    template_ts = data_historic_temporal_targets
    for arr in list_counterfactual_predictions:
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        assert isinstance(arr, np.ndarray)
        if arr.ndim not in (2, 3):
            raise ValueError(
                "Arrays for counterfactual predictions must be either 2D or 3D (with 0th dimension size 1)"
            )
        if arr.ndim == 3 and arr.shape[0] != 1:
            raise ValueError(
                "Arrays for counterfactual predictions must have 0th dimension size 1 if 3D (i.e. single sample)"
            )
        if arr.ndim == 3:
            arr = arr[0, :, :]
        arr = arr.astype(float)
        assert isinstance(template_ts, TimeSeries)
        list_ts.append(
            TimeSeries.new_like(
                like=template_ts,
                data=pd.DataFrame(data=arr, index=time_index, columns=template_ts.df.columns),
            )
        )
    return list_ts
