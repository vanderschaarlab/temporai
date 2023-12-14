from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from .constants import TIndexDiff


def all_items_are_of_types(series: pd.Series, of_types: Union[type, Tuple[type, ...]]) -> bool:
    return not series.apply(lambda x, t=of_types: not isinstance(x, t)).any()


def check_index_regular(index: pd.Index) -> Tuple[bool, Optional[TIndexDiff]]:
    idx_as_list = list(index)
    diffs = np.diff(idx_as_list)
    if len(diffs) == 0:
        return (True, None)
    else:
        is_regular = bool((diffs[0] == diffs).all())  # np.bool_ --> bool
        diff = diffs[0] if is_regular else None
        return is_regular, diff


def df_align_and_overwrite(df_to_update: pd.DataFrame, df_with_new_data: pd.DataFrame):
    df_to_update_aligned: pd.DataFrame
    df_with_new_data_aligned: pd.DataFrame
    df_to_update_aligned, df_with_new_data_aligned = df_to_update.align(df_with_new_data, join="outer", axis=0)
    df_to_update_aligned[~df_with_new_data_aligned.isnull()] = df_with_new_data_aligned[
        ~df_with_new_data_aligned.isnull()
    ]
    return df_to_update_aligned
