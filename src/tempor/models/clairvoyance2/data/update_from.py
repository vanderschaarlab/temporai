from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from ..utils import tensor_like as tl
from ..utils.common import df_eq_indicator
from ..utils.tensor_like import TTensorLike
from .constants import DEFAULT_PADDING_INDICATOR, T_TSIndexClass
from .internal_utils import check_index_regular, df_align_and_overwrite


def get_n_step_ahead_index(time_index: T_TSIndexClass, n_step: int) -> T_TSIndexClass:
    regular, diff = check_index_regular(time_index)
    if not regular:
        raise RuntimeError("Time index must have regular intervals to compute n step ahead index")
    if diff is None:
        raise RuntimeError("Time index must have at least two elements to compute n step ahead index")
    last_index = time_index[-1]
    new_indices = [last_index + i * diff for i in range(1, n_step + 1)]
    time_index = time_index.append(pd.Index(new_indices))
    return time_index[n_step:]


class UpdateFromArrayExtension:
    df: pd.DataFrame
    time_index: Any
    n_timesteps: Any

    def update_from_array(
        self,
        update_array: TTensorLike,
        time_index: T_TSIndexClass,
        padding_indicator: float = DEFAULT_PADDING_INDICATOR,
    ) -> None:
        if update_array.ndim != 2:
            raise ValueError(f"Expected a 2D array but {update_array.ndim} dimensions found")
        if len(list(time_index)) != update_array.shape[0]:
            raise ValueError(
                "Time index length must match the 0th dimension size of update array "
                f"but was {len(list(time_index))} and {update_array.shape[0]} respectively"
            )

        df_new = pd.DataFrame(data=update_array, index=time_index, columns=self.df.columns)

        selector = df_eq_indicator(df_new, padding_indicator)
        df_new[selector] = np.nan
        df_new.dropna(axis=0, how="any", inplace=True)

        self.df = df_align_and_overwrite(df_to_update=self.df, df_with_new_data=df_new)

    def update_from_array_n_step_ahead(
        self,
        update_array: TTensorLike,
        n_step: int,
        padding_indicator: float = DEFAULT_PADDING_INDICATOR,
    ) -> None:
        # Get n_step ahead index for current TimeSeries.df, e.g.
        # if TimeSeries.df.index is [1, 2, 3] and n_step=2, get [3, 4, 5].
        # The values in update_array are expected to correspond to the n_step ahead index.
        # In the above case, say update_array=[[33], [32], [34]] (1 feature), then the values are expected to
        # correspond to the index like so: (3 -> 33), (4 -> 32), (5 -> 34).
        # If array has more elements than the new index, the remaining values are ignored.
        if update_array.ndim != 2:
            raise ValueError(f"Expected a 2D array but {update_array.ndim} dimensions found")

        new_index = get_n_step_ahead_index(self.time_index, n_step=n_step)
        expected_n_timesteps = len(new_index)

        # Discard any past time indices which are not found in the new n-step-ahead index.
        first_new_index = list(new_index)[0]
        if first_new_index in self.time_index:
            self.df = self.df.loc[first_new_index:, :]
        else:
            self.df = self.df.iloc[:0, :]

        if update_array.shape[0] < expected_n_timesteps:
            raise ValueError(
                f"Expected at least {expected_n_timesteps} timesteps in update array but only "
                f"{update_array.shape[0]} found"
            )

        # NOTE: Any values for timesteps too far into the future will be ignored.
        update_array = update_array[:expected_n_timesteps, :]

        # Ensure no padding:
        if bool(tl.eq_indicator(update_array, padding_indicator).sum() > 0):
            raise ValueError(
                f"There should not be padding indicator values ({padding_indicator}) in update array "
                "for n step ahead update, but padding indicator(s) found"
            )

        self.update_from_array(update_array, time_index=new_index, padding_indicator=padding_indicator)


class UpdateFromSequenceOfArraysExtension:
    n_samples: Any
    sample_indices: Any

    def _check_n_samples_in_update_array_seq(self, update_array_sequence: Sequence[TTensorLike]):
        if self.n_samples != len(update_array_sequence):
            raise ValueError(
                "Expected number of samples to correspond to the len / 0th dim size of update array sequence but "
                f"was {self.n_samples} and {len(update_array_sequence)} correspondingly"
            )

    def update_from_sequence_of_arrays(
        self,
        update_array_sequence: Sequence[TTensorLike],
        time_index_sequence: Sequence[T_TSIndexClass],
        padding_indicator: float = DEFAULT_PADDING_INDICATOR,
    ) -> None:
        self._check_n_samples_in_update_array_seq(update_array_sequence)
        if self.n_samples != len(time_index_sequence):
            raise ValueError(
                "Expected number of samples to correspond to the len of time index sequence "
                f"was {self.n_samples} and {len(time_index_sequence)} correspondingly"
            )

        for sample_idx, sample_array, sample_time_index in zip(
            self.sample_indices, update_array_sequence, time_index_sequence
        ):
            if sample_array.ndim != 2:
                raise ValueError(
                    "Expected number of dimensions of update arrays to be 2 (for each sample), "
                    f"was {sample_array.ndim}"
                )

            # In case we have padding in the update array that goes beyond the time index length, trim sample array.
            if len(sample_time_index) < len(sample_array):
                selector = ~tl.any(tl.eq_indicator(sample_array, padding_indicator), axis=-1)
                sample_array = sample_array[selector, :]
            assert len(sample_time_index) >= len(sample_array)

            if isinstance(sample_array, torch.Tensor):
                sample_array = sample_array.detach().cpu().numpy()

            ts: UpdateFromArrayExtension = self[sample_idx]  # type: ignore  # pylint: disable=E
            ts.update_from_array(sample_array, sample_time_index, padding_indicator)

    def update_from_array_n_step_ahead(
        self,
        update_array_sequence: Sequence[TTensorLike],
        n_step: int,
        padding_indicator: float = DEFAULT_PADDING_INDICATOR,
    ) -> None:
        self._check_n_samples_in_update_array_seq(update_array_sequence)

        for sample_idx, sample_array in zip(self.sample_indices, update_array_sequence):
            if isinstance(sample_array, torch.Tensor):
                sample_array = sample_array.detach().cpu().numpy()

            ts: UpdateFromArrayExtension = self[sample_idx]  # type: ignore  # pylint: disable=E

            if not (tl.eq_indicator(sample_array[ts.n_timesteps :, :], padding_indicator)).all():
                raise ValueError(
                    "Found non-padding value(s) in the tail part of the update array which "
                    "was expected to be padding"
                )

            ts.update_from_array_n_step_ahead(sample_array, n_step=n_step, padding_indicator=padding_indicator)
