# mypy: ignore-errors

import numpy as np
import pandas as pd

from ...data import Dataset, EventSamples, TimeSeriesSamples
from . import _simple_pkpd

_SANITY_CHECK_ON = False
_sanity_check = dict()


def simple_pkpd_dataset(
    n_timesteps: int = 30,
    time_index_treatment_event: int = 25,
    n_control_samples: int = 200,
    n_treated_samples: int = 200,
    seed: int = 100,
):
    print(f"Generating simple PKPD dataset with random seed {seed}...")

    hidden_confounder: int = 0
    (
        x_full,
        t_full,
        mask_full,
        batch_ind_full,
        y_full,
        y_control,
        y_mask_full,
        y_full_all,
    ) = _simple_pkpd.generate(
        seed=seed,
        train_step=time_index_treatment_event,
        step=n_timesteps,
        control_sample=n_control_samples,
        treatment_sample=n_treated_samples,
        hidden_confounder=hidden_confounder,
    )

    if _SANITY_CHECK_ON:
        _sanity_check["x_full"] = x_full
        _sanity_check["t_full"] = t_full
        _sanity_check["mask_full"] = mask_full
        _sanity_check["batch_ind_full"] = batch_ind_full
        _sanity_check["y_full"] = y_full
        _sanity_check["y_control"] = y_control
        _sanity_check["y_mask_full"] = y_mask_full

    x_everything = np.concatenate([x_full, y_full_all], axis=0)
    assert (x_everything[:time_index_treatment_event, :, :] == x_full).all()
    assert (x_everything[time_index_treatment_event:, :, [2]] == y_full).all()
    assert (x_everything[time_index_treatment_event:, :n_control_samples, [2]] == y_control).all()

    sample_index = batch_ind_full.astype(int)
    tss = TimeSeriesSamples(
        data=[
            pd.DataFrame(data=x_everything[:, idx, :], columns=["k_in", "p", "y"], index=range(n_timesteps))
            for idx in sample_index
        ]
    )

    treat_event_feature = np.zeros(shape=(n_control_samples + n_treated_samples,), dtype=float)
    treat_event_feature[n_control_samples:] = 1.0
    df = pd.DataFrame(
        data={
            "si": sample_index,
            "ti": [time_index_treatment_event] * (n_control_samples + n_treated_samples),
            "a": treat_event_feature,
        }
    )
    es = EventSamples.from_df(data=df, column_sample_index="si", column_time_index="ti")

    return Dataset(temporal_covariates=tss[:, ["k_in", "p"]], temporal_targets=tss[:, ["y"]], event_treatments=es)
