# mypy: ignore-errors

import math
import os
import tarfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import unlzw3

from ..data import Dataset, TimeSeriesSamples
from .dataset_retriever import DatasetRetriever

_DEBUG = False


class UCIDiabetesRetriever(DatasetRetriever):
    dataset_subdir = "uci_diabetes"
    dataset_files = [
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/Index", "Index"),
        ("https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/README", "README"),
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/diabetes-data.tar.Z",
            "diabetes-data.tar.Z",
        ),
    ]

    codes_map = {
        33: "regular_insulin_dose",
        34: "nph_insulin_dose",
        35: "ultralente_insulin_dose",
        48: "unspecified_blood_glucose_measurement",
        57: "unspecified_blood_glucose_measurement",
        58: "pre-breakfast_blood_glucose_measurement",
        59: "post-breakfast_blood_glucose_measurement",
        60: "pre-lunch_blood_glucose_measurement",
        61: "post-lunch_blood_glucose_measurement",
        62: "pre-supper_blood_glucose_measurement",
        63: "post-supper_blood_glucose_measurement",
        64: "pre-snack_blood_glucose_measurement",
        65: "hypoglycemic_symptoms",
        66: "typical_meal_ingestion",
        67: "more-than-usual_meal_ingestion",
        68: "less-than-usual_meal_ingestion",
        69: "typical_exercise_activity",
        70: "more-than-usual_exercise_activity",
        71: "less-than-usual_exercise_activity",
        72: "unspecified_special_event",
    }

    features = [
        "regular_insulin_dose",
        "nph_insulin_dose",
        "ultralente_insulin_dose",
        "unspecified_blood_glucose_measurement",
        "pre-breakfast_blood_glucose_measurement",
        "post-breakfast_blood_glucose_measurement",
        "pre-lunch_blood_glucose_measurement",
        "post-lunch_blood_glucose_measurement",
        "pre-supper_blood_glucose_measurement",
        "post-supper_blood_glucose_measurement",
        "pre-snack_blood_glucose_measurement",
        "hypoglycemic_symptoms",
        "typical_meal_ingestion",
        "more-than-usual_meal_ingestion",
        "less-than-usual_meal_ingestion",
        "typical_exercise_activity",
        "more-than-usual_exercise_activity",
        "less-than-usual_exercise_activity",
        "unspecified_special_event",
    ]

    _timedelta_for_make_regular = pd.Timedelta(5, "h")

    def __init__(
        self, data_home: Optional[str] = None, make_regular: bool = False, use_int_index: bool = False
    ) -> None:
        self.make_regular = make_regular
        self.use_int_index = use_int_index
        super().__init__(data_home)

    @property
    def dataset_extracted_dir(self):
        return os.path.join(self.dataset_dir, "Diabetes-Data")

    def extract(self) -> None:
        if not os.path.exists(os.path.join(self.dataset_extracted_dir, "data-70")):
            # ^ Check if already extracted previously (by proxy to file data-70).
            archive_file = os.path.join(self.dataset_dir, "diabetes-data.tar.Z")
            temp_file = os.path.join(self.dataset_dir, "temp.tar")
            uncompressed_data = unlzw3.unlzw(Path(archive_file))
            with open(temp_file, "wb") as f:
                f.write(uncompressed_data)

            with tarfile.open(temp_file) as tar:

                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    return prefix == abs_directory

                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            # pylint: disable-next=broad-exception-raised
                            raise Exception("Attempted Path Traversal in Tar File")
                    tar.extractall(path, members, numeric_owner=numeric_owner)

                safe_extract(tar, self.dataset_dir)  # NOTE: Will extract into a subdirectory "Diabetes-Data/"

            os.remove(temp_file)

    def process_individual_file(self, filepath: str) -> pd.DataFrame:
        if _DEBUG:
            print(f"Processing file: {filepath.split('/')[-1]}")

        # Read file.
        df = pd.read_csv(filepath, sep="\t", header=None, names=["date", "time", "code", "value"])

        # Convert to date-time
        df.dropna(subset=["date", "time"], inplace=True)
        # ^ Drop rows with of missing date or time, which do exist in dataset.
        df_dt = pd.DataFrame(
            pd.to_datetime(df["date"] + " " + df["time"], dayfirst=False, errors="coerce"), columns=["datetime"]
        )
        df_dt.dropna(subset=["datetime"], inplace=True)
        # ^ There are a few broken date/time items in the dataset,
        # so use "coerce" to get NaTs, then drop those, then use "inner" join below.
        df = pd.concat([df_dt, df[["code", "value"]]], axis=1, join="inner")
        df.set_index("datetime", drop=True, inplace=True)

        # Convert int codes to str feature names.
        df.drop(df[~df["code"].isin(self.codes_map.keys())].index, axis=0, inplace=True)
        # ^ Drop unknown codes, there are some.
        df["code"] = df["code"].map(self.codes_map)

        # Make values float
        df.loc[df["value"].isin(["0Hi", "0Lo"]), "value"] = np.nan
        # ^ NOTE: This is questionable, but unclear what 0Hi/0Lo means
        df.loc[df["value"].isin(["0''"]), "value"] = np.nan
        # ^ Drop broken value 0''.
        df["value"] = df["value"].astype(float)

        # Pivot to index by features format. # TODO: May extract this functionality on its own.
        # 1. Drop duplicate (index, column) cases (otherwise cannot sensibly pivot)
        df["copy_of_index"] = df.index
        df.drop_duplicates(subset=["copy_of_index", "code"], keep="last", inplace=True)
        df.drop(["copy_of_index"], axis=1)
        # 2. Actually pivot.
        df_pivoted = df.pivot(columns="code", values="value")
        # 3. Enforce that all dataset features are present in each sample.
        df_all_features = pd.DataFrame(data=np.nan, columns=self.features, index=df_pivoted.index)
        for f in self.features:
            if f in df_pivoted:
                df_all_features.loc[:, f] = df_pivoted[f]

        if _DEBUG:
            # Report missing-ness.
            total_elements = df_all_features.shape[0] * df_all_features.shape[1]
            null_elements = df_all_features.isnull().sum().sum()
            print("Non-null elements:", total_elements - null_elements)
            print("Missingness fraction:", null_elements / total_elements)

        # Sort index ascending.
        df_all_features.sort_index(inplace=True)

        # Case: make_regular=True.
        if self.make_regular:
            full_range = df_all_features.index[-1] - df_all_features.index[0]  # pyright: ignore
            n_periods = math.ceil(full_range / self._timedelta_for_make_regular)
            new_index = pd.date_range(
                df_all_features.index[0],  # pyright: ignore
                periods=n_periods + 1,
                freq=self._timedelta_for_make_regular,
            )
            assert new_index[-1] >= df_all_features.index[-1]  # pyright: ignore
            df_all_features = df_all_features.reindex(new_index, method="nearest")

        # Case use_int_index=True.
        if self.use_int_index:
            df_all_features.reset_index(drop=True, inplace=True)

        return df_all_features

    @staticmethod
    def _get_file_id_range():
        return range(1, 70 + 1)

    def _get_cache_file_name(self, file_id: int) -> str:
        return f"mr{self.make_regular}_uii{self.use_int_index}_{file_id}.pkl"

    def is_cached(self) -> bool:
        return all(
            [
                os.path.exists(os.path.join(self.dataset_cache_dir, self._get_cache_file_name(file_id)))
                for file_id in self._get_file_id_range()
            ]
        )

    def get_cache(self) -> Dataset:
        list_dfs: List[pd.DataFrame] = []
        for file_id in self._get_file_id_range():
            cache_path = os.path.join(self.dataset_cache_dir, self._get_cache_file_name(file_id))
            list_dfs.append(pd.read_pickle(cache_path))
        temporal_covariates = TimeSeriesSamples(data=list_dfs, sample_indices=None)
        return Dataset(temporal_covariates)

    def cache(self, data: Dataset) -> None:
        temporal_covariates = data.temporal_covariates
        for file_id, ts in zip(self._get_file_id_range(), temporal_covariates):
            cache_path = os.path.join(self.dataset_cache_dir, self._get_cache_file_name(file_id))
            ts.df.to_pickle(cache_path)

    def prepare(self) -> Dataset:
        self.extract()
        list_dfs: List[pd.DataFrame] = []
        for file_id in self._get_file_id_range():
            df = self.process_individual_file(os.path.join(self.dataset_extracted_dir, f"data-{file_id:02}"))
            list_dfs.append(df)
        temporal_covariates = TimeSeriesSamples(data=list_dfs, sample_indices=None)
        return Dataset(temporal_covariates)


def uci_diabetes(
    data_home: Optional[str] = None,
    refresh_cache: bool = False,
    redownload: bool = False,
    make_regular: bool = False,
    use_int_index: bool = False,
) -> Dataset:
    retriever = UCIDiabetesRetriever(data_home=data_home, make_regular=make_regular, use_int_index=use_int_index)
    tss = retriever.retrieve(refresh_cache=refresh_cache, redownload=redownload)
    return tss
