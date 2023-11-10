"""Module defining the UCI diabetes data source."""

import time
import traceback
import urllib.error
from pathlib import Path
from typing import Any, Tuple, cast

import requests
from clairvoyance2.datasets.uci import uci_diabetes
from clairvoyance2.preprocessing.convenience import TemporalTargetsExtractor

from tempor.core import plugins
from tempor.data import dataset
from tempor.data.clv2conv import clairvoyance2_dataset_to_tempor_dataset
from tempor.datasources import datasource
from tempor.log import logger


@plugins.register_plugin(name="uci_diabetes", category="prediction.temporal", plugin_type="datasource")
class UCIDiabetesDataSource(datasource.TemporalPredictionDataSource):
    def __init__(
        self,
        make_regular: bool = False,
        use_int_index: bool = True,
        targets: Tuple[str, ...] = ("hypoglycemic_symptoms",),
        **kwargs: Any,
    ) -> None:
        """UCI diabetes data source.

        See: https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes

        Args:
            make_regular (bool, optional):
                Whether to reindex the dataset to have regular timesteps. Defaults to `False`.
            use_int_index (bool, optional):
                Whether to use integer index. Defaults to `True`.
            targets (Tuple[str, ...], optional):
                The target feature(s). Defaults to ``("hypoglycemic_symptoms",)``.
            **kwargs (Any):
                Any additional keyword arguments will be passed to parent constructor.
        """
        super().__init__(**kwargs)
        self.make_regular = make_regular
        self.use_int_index = use_int_index
        self.targets = targets

    @staticmethod
    def url() -> str:  # noqa: D102
        return "https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes"

    @staticmethod
    def dataset_dir() -> str:  # noqa: D102
        return str(Path(UCIDiabetesDataSource.data_root_dir) / "uci_diabetes")

    def load(self, **kwargs: Any) -> dataset.TemporalPredictionDataset:  # noqa: D102
        download_retries = 3
        download_pause_sec = 5
        for retry in range(download_retries):
            # NOTE: Connection to archive.ics.uci.edu tends to be flaky, attempt download retries.
            # TODO: May wish to exclude this from tests / download files from a more stable location /
            # make tests resilient to internet connection failures.
            try:
                clv_dataset = uci_diabetes(
                    data_home=UCIDiabetesDataSource.data_root_dir,
                    refresh_cache=True,
                    redownload=False,
                    make_regular=self.make_regular,
                    use_int_index=self.use_int_index,
                )
            except (requests.exceptions.RequestException, urllib.error.URLError) as ex:  # pragma: no cover
                if retry + 1 == download_retries:
                    logger.error(f"Failed to download UCI diabetes dataset after {download_retries} retries.")
                    raise
                logger.debug(
                    f"Caught exception and will retry ({retry + 1}/{download_retries}): "
                    f"{ex}\n{traceback.format_exc()}"
                )
                time.sleep(download_pause_sec)
        clv_dataset = TemporalTargetsExtractor(params={"targets": self.targets}).fit_transform(
            clv_dataset  # pyright: ignore
        )
        data = clairvoyance2_dataset_to_tempor_dataset(clv_dataset)
        data = cast(dataset.TemporalPredictionDataset, data)
        return data
