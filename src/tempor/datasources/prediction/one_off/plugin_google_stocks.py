import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

from tempor.core import plugins
from tempor.data import dataset
from tempor.datasources import datasource


# TODO: Docstring to explain the dataset.
@plugins.register_plugin(name="google_stocks", category="prediction.one_off", plugin_type="datasource")
class GoogleStocksDataSource(datasource.OneOffPredictionDataSource):
    def __init__(self, seq_len: int = 10, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.seq_len = seq_len
        self.df_path = Path(self.dataset_dir()) / "goog.csv"

    @staticmethod
    def url() -> str:
        return "https://raw.githubusercontent.com/PacktPublishing/Learning-Pandas-Second-Edition/master/data/goog.csv"

    @staticmethod
    def dataset_dir() -> str:
        return str(Path(GoogleStocksDataSource.data_root_dir) / "google_stocks")

    def load(self, **kwargs: Any) -> dataset.OneOffPredictionDataset:
        # Load Google Data
        if not self.df_path.exists():
            s = requests.get(self.url(), timeout=5).content
            df = pd.read_csv(io.StringIO(s.decode("utf-8")))

            df.to_csv(self.df_path, index=False)
        else:
            df = pd.read_csv(self.df_path)

        # Flip the data to make chronological data
        df = pd.DataFrame(df.values[::-1], columns=df.columns)
        T = pd.to_datetime(df["Date"]).astype(np.int64).astype(np.float64) / 10**9
        T = pd.Series(MinMaxScaler().fit_transform(T.values.reshape(-1, 1)).squeeze())  # pyright: ignore

        df = df.drop(columns=["Date"])

        df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
        # Build dataset
        dataX = []
        outcome = []

        # Cut data by sequence length
        sample_idxs = []
        for i in range(0, len(df) - self.seq_len - 1):
            df_seq = df.loc[i : i + self.seq_len - 1].copy()
            horizons = T.loc[i : i + self.seq_len - 1].copy()
            out = df["Open"].loc[i + self.seq_len].copy()

            df_seq["time_idx"] = horizons
            df_seq["sample_idx"] = str(i)

            dataX.append(df_seq)
            outcome.append(out)
            sample_idxs.append(str(i))

        time_series_df = pd.concat(dataX, ignore_index=True)
        time_series_df.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)

        outcome_df = pd.DataFrame(outcome)
        outcome_df.index = sample_idxs  # pyright: ignore
        outcome_df.columns = ["out"]

        time_series_df.sort_index(level=[0, 1], inplace=True)
        outcome_df.sort_index(inplace=True)

        return dataset.OneOffPredictionDataset(
            time_series=time_series_df,
            targets=outcome_df,
            sample_index="sample_idx",
            time_index="time_idx",
        )
