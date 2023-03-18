import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

from tempor.data.dataset import OneOffPredictionDataset

URL = "https://raw.githubusercontent.com/PacktPublishing/Learning-Pandas-Second-Edition/master/data/goog.csv"


class GoogleStocksDataloader:
    def __init__(self, seq_len: int = 10) -> None:
        self.seq_len = seq_len

        df_folder = Path(__file__).parent / "data"
        df_folder.mkdir(parents=True, exist_ok=True)
        df_path = "goog.csv"

        self.df_path = df_folder / df_path

    def load(
        self,
    ) -> OneOffPredictionDataset:
        # Load Google Data
        if not self.df_path.exists():
            s = requests.get(URL, timeout=5).content
            df = pd.read_csv(io.StringIO(s.decode("utf-8")))

            df.to_csv(self.df_path, index=False)
        else:
            df = pd.read_csv(self.df_path)

        # Flip the data to make chronological data
        df = pd.DataFrame(df.values[::-1], columns=df.columns)
        T = pd.to_datetime(df["Date"], infer_datetime_format=True).astype(np.int64).astype(np.float64) / 10**9
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

        return OneOffPredictionDataset(
            time_series=time_series_df,
            targets=outcome_df,
            sample_index="sample_idx",
            time_index="time_idx",
        )
