import io
import os
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tempor.core import plugins
from tempor.data import dataset, utils
from tempor.datasources import datasource


# TODO: Docstring to explain the dataset.
@plugins.register_plugin(name="pbc", category="time_to_event", plugin_type="datasource")
class PBCDataSource(datasource.TimeToEventAnalysisDataSource):
    def __init__(self, **kwargs: Any) -> None:
        self.datafile_path = os.path.join(self.dataset_dir(), "pbc2.csv")
        super().__init__(**kwargs)

    @staticmethod
    def dataset_dir() -> str:
        return os.path.join(PBCDataSource.data_root_dir, "pbc/")

    @staticmethod
    def url() -> str:
        return (
            "https://raw.githubusercontent.com/autonlab/auton-survival/"
            "cf583e598ec9ab92fa5d510a0ca72d46dfe0706f/dsm/datasets/pbc2.csv"
        )

    def load(self, **kwargs: Any) -> dataset.TimeToEventAnalysisDataset:
        if os.path.exists(self.datafile_path):
            data = pd.read_csv(self.datafile_path)
        else:
            request = requests.get(self.url(), timeout=5).content
            data = pd.read_csv(io.StringIO(request.decode("utf-8")))
            data.to_csv(self.datafile_path, index=False)

        data["time"] = data["years"] - data["year"]
        data = data.sort_values(by=["id", "time"], ignore_index=True)
        data["histologic"] = data["histologic"].astype(str)
        dat_cat = data[["drug", "sex", "ascites", "hepatomegaly", "spiders", "edema", "histologic"]].copy()
        dat_num = data[["serBilir", "serChol", "albumin", "alkaline", "SGOT", "platelets", "prothrombin"]].copy()
        age = data["age"] + data["years"]

        for col in dat_cat.columns:
            dat_cat[col] = LabelEncoder().fit_transform(dat_cat[col])

        x = pd.concat([dat_cat, dat_num, pd.Series(age, name="age")], axis=1)

        time = data["time"]
        event = data["status2"]

        x = pd.DataFrame(
            SimpleImputer(missing_values=np.nan, strategy="mean").fit_transform(x),
            columns=x.columns,
        )
        scaled_cols = list(dat_num.columns) + ["age"]

        x_scaled = x.copy()
        x_scaled[scaled_cols] = pd.DataFrame(
            StandardScaler().fit_transform(x[scaled_cols]),
            columns=scaled_cols,
            index=data.index,
        )

        x_static, x_temporal, t, e = [], [], [], []
        time_indexes, event_feat_full = [], []

        temporal_cols = [
            "drug",
            "ascites",
            "hepatomegaly",
            "spiders",
            "edema",
            "histologic",
            "serBilir",
            "serChol",
            "albumin",
            "alkaline",
            "SGOT",
            "platelets",
            "prothrombin",
            "age",
        ]
        static_cols = ["sex"]
        sample_index = []

        for id_ in sorted(list(set(data["id"]))):
            sample_index.append(id_)
            patient = x_scaled[data["id"] == id_]

            patient_static = patient[static_cols]
            if not (patient_static.iloc[0] == patient_static).all().all():  # pragma: no cover
                # This is a sanity check.
                raise RuntimeError(
                    f"Found patient with static data that are not actually fixed:\nid: {id_}\n{patient_static}"
                )
            x_static.append(patient_static.values[0].tolist())

            patient_temporal = patient[temporal_cols]
            patient_temporal.index = time[data["id"] == id_].values

            x_temporal.append(patient_temporal)

            events = event[data["id"] == id_].values
            times = time[data["id"] == id_].values
            evt = np.amax(events)  # pyright: ignore
            if evt == 0:
                pos = np.max(np.where(events == evt))  # Last censored
            else:
                pos = np.min(np.where(events == evt))  # First event

            t.append(times[pos])
            e.append({0: False, 1: True}[evt])

            time_indexes.append(list(times))
            event_feat_full.append(events)

        df_static = pd.DataFrame(x_static, columns=static_cols, index=sample_index)
        df_time_series = utils.list_of_dataframes_to_multiindex_timeseries_dataframe(
            x_temporal,
            sample_index=sample_index,
            time_indexes=time_indexes,
            feature_index=temporal_cols,
        )
        df_event = utils.event_time_value_pairs_to_event_dataframe(
            [(t, e)], sample_index=sample_index, feature_index=["status"]
        )

        return dataset.TimeToEventAnalysisDataset(
            time_series=df_time_series,
            static=df_static,
            targets=df_event,
        )
