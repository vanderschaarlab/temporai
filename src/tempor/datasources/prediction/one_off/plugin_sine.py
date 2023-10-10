import numpy as np
import pandas as pd

from tempor.core import plugins
from tempor.data import dataset
from tempor.datasources import datasource


@plugins.register_plugin(name="sine", category="prediction.one_off", plugin_type="datasource")
class SineDataSource(datasource.OneOffPredictionDataSource):
    def __init__(
        self,
        no: int = 100,
        seq_len: int = 10,
        temporal_dim: int = 5,
        static_dim: int = 4,
        freq_scale: float = 1,
        with_missing: bool = False,
        miss_ratio: float = 0.1,
        static_scale: float = 1.0,
        ts_scale: float = 1.0,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        """Sine data generation.

        Args:
            no (int, optional):
                The number of samples. Defaults to ``100``.
            seq_len (int, optional):
                Sequence length of the time-series. Defaults to ``10``.
            temporal_dim (int, optional):
                Time-series feature dimensions. Defaults to ``5``.
            static_dim (int, optional):
                Static feature dimensions. Defaults to ``4``.
            freq_scale (float, optional):
                The frequency scaling multiplier for the signal (``sin(freq_scale * random_drawn_freq * x + phase)``).
                Defaults to ``1``.
            with_missing (bool, optional):
                Whether to generate missing data points (`np.nan`). Defaults to `False`.
            miss_ratio (float, optional):
                The ration of missing data points. Defaults to ``0.1``.
            static_scale (float, optional):
                The scaling factor to apply to the static data. Defaults to ``1.0``.
            ts_scale (float, optional):
                The scaling factor to apply to the time series data. Defaults to ``1.0``.
            random_state (int, optional):
                The random seed to set for `numpy.random.seed`. Defaults to ``42``.
        """
        super().__init__(**kwargs)

        self.no = no
        self.seq_len = seq_len
        self.temporal_dim = temporal_dim
        self.static_dim = static_dim
        self.freq_scale = freq_scale
        self.with_missing = with_missing
        self.miss_ratio = miss_ratio
        self.static_scale = static_scale
        self.ts_scale = ts_scale
        self.random_state = random_state

    @staticmethod
    def url() -> None:
        return None

    @staticmethod
    def dataset_dir() -> None:
        return None

    def load(self, **kwargs) -> dataset.OneOffPredictionDataset:
        # Initialize the output.
        np.random.seed(self.random_state)

        static_data = pd.DataFrame(np.random.rand(self.no, self.static_dim) * self.static_scale)
        static_data["sample_idx"] = [i for i in range(self.no)]
        static_data.set_index(keys=["sample_idx"], drop=True, inplace=True)

        static_data.columns = static_data.columns.astype(str)
        if self.with_missing:
            for col in static_data.columns:
                static_data.loc[static_data.sample(frac=self.miss_ratio).index, col] = np.nan

        temporal_data = []

        outcome = pd.DataFrame(np.random.randint(0, 2, self.no))
        outcome["sample_idx"] = [i for i in range(self.no)]
        outcome.columns = outcome.columns.astype(str)

        outcome.set_index(keys=["sample_idx"], drop=True, inplace=True)

        # Generate sine data.

        for i in range(self.no):
            # Initialize each time-series
            local = list()

            # For each feature
            seq_len = self.seq_len

            for k in range(self.temporal_dim):  # pylint: disable=unused-variable
                # Randomly drawn frequency and phase:
                freq = np.random.beta(2, 2)
                phase = np.random.normal()

                # Generate sine signal based on the drawn frequency and phase:
                temp_data = [np.sin(self.freq_scale * freq * j + phase) * self.ts_scale for j in range(seq_len)]

                local.append(temp_data)

            # Align row/column.
            # DataFrame with index - time, and columns - temporal features.
            local_data = pd.DataFrame(np.transpose(np.asarray(local)))
            local_data.columns = local_data.columns.astype(str)

            if self.with_missing:
                for col in local_data.columns:
                    local_data.loc[local_data.sample(frac=self.miss_ratio).index, col] = np.nan

            # Stack the generated data:
            local_data["sample_idx"] = i
            local_data["time_idx"] = list(range(seq_len))
            temporal_data.append(local_data)

        time_series_df = pd.concat(temporal_data, ignore_index=True)
        time_series_df.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)

        time_series_df.sort_index(level=[0, 1], inplace=True)
        static_data.sort_index(inplace=True)
        outcome.sort_index(inplace=True)

        return dataset.OneOffPredictionDataset(
            time_series=time_series_df,
            targets=outcome,
            static=static_data,
            sample_index="sample_idx",
            time_index="time_idx",
        )
