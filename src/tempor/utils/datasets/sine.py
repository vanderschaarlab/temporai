# third party
import numpy as np
import pandas as pd

from tempor.data.dataset import OneOffPredictionDataset


class SineDataloader:
    """Sine data generation.

    Args:

    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions

    Returns:
    - data: generated data

    """

    def __init__(
        self,
        no: int = 100,
        seq_len: int = 10,
        temporal_dim: int = 5,
        static_dim: int = 4,
        freq_scale: float = 1,
        with_missing: bool = False,
        miss_ratio: float = 0.1,
    ) -> None:
        self.no = no
        self.seq_len = seq_len
        self.temporal_dim = temporal_dim
        self.static_dim = static_dim
        self.freq_scale = freq_scale
        self.with_missing = with_missing
        self.miss_ratio = miss_ratio

    def load(
        self,
    ) -> OneOffPredictionDataset:
        # Initialize the output

        static_data = pd.DataFrame(np.random.rand(self.no, self.static_dim))
        static_data["sample_idx"] = [str(i) for i in range(self.no)]
        static_data.set_index(keys=["sample_idx"], drop=True, inplace=True)

        static_data.columns = static_data.columns.astype(str)
        if self.with_missing:
            for col in static_data.columns:
                static_data.loc[static_data.sample(frac=self.miss_ratio).index, col] = np.nan

        temporal_data = []

        outcome = pd.DataFrame(np.random.randint(0, 2, self.no))
        outcome["sample_idx"] = [str(i) for i in range(self.no)]
        outcome.columns = outcome.columns.astype(str)

        outcome.set_index(keys=["sample_idx"], drop=True, inplace=True)

        # Generate sine data

        for i in range(self.no):

            # Initialize each time-series
            local = list()

            # For each feature
            seq_len = self.seq_len

            for k in range(self.temporal_dim):

                # Randomly drawn frequency and phase
                freq = np.random.beta(2, 2)
                phase = np.random.normal()

                # Generate sine signal based on the drawn frequency and phase
                temp_data = [np.sin(self.freq_scale * freq * j + phase) for j in range(seq_len)]

                local.append(temp_data)

            # Align row/column
            # DataFrame with index - time, and columns - temporal features
            local_data = pd.DataFrame(np.transpose(np.asarray(local)))
            local_data.columns = local_data.columns.astype(str)

            if self.with_missing:
                for col in local_data.columns:
                    local_data.loc[local_data.sample(frac=self.miss_ratio).index, col] = np.nan

            # Stack the generated data
            local_data["sample_idx"] = str(i)
            local_data["time_idx"] = list(range(seq_len))
            temporal_data.append(local_data)

        time_series_df = pd.concat(temporal_data, ignore_index=True)
        time_series_df.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)

        return OneOffPredictionDataset(
            time_series=time_series_df,
            targets=outcome,
            static=static_data,
            sample_index="sample_idx",
            time_index="time_idx",
        )
