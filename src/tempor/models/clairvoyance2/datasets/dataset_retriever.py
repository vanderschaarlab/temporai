# mypy: ignore-errors

import os
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, TypeVar

from ..data import Dataset
from .download import download_file

TUrl = TypeVar("TUrl", bound=str)
TDatasetFileDef = Tuple[TUrl, str]  # ("URL", "local_file_name")

DATASET_ROOT_DIR = os.path.join(os.path.expanduser("~"), ".clairvoyance/datasets/")


# TODO: Unit test.
class DatasetRetriever(ABC):
    dataset_subdir: str
    dataset_files: Optional[Sequence[TDatasetFileDef]]
    cache_subdir: str = "cache"

    @property
    def dataset_dir(self) -> str:
        return os.path.join(self.dataset_root_dir, self.dataset_subdir)

    @property
    def dataset_cache_dir(self) -> str:
        return os.path.join(self.dataset_dir, self.cache_subdir)

    def __init__(self, data_home: Optional[str] = None) -> None:
        if data_home is None:
            self.dataset_root_dir = DATASET_ROOT_DIR
        else:
            self.dataset_root_dir = data_home
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.dataset_cache_dir, exist_ok=True)

    def download_dataset(self) -> None:
        if self.dataset_files is not None:
            for dataset_file in self.dataset_files:
                url, file_name = dataset_file
                download_file(url, os.path.join(self.dataset_dir, file_name))

    @abstractmethod
    def is_cached(self) -> bool:
        # Check if the dataset has been cached.
        ...

    @abstractmethod
    def get_cache(self) -> Dataset:
        # Retrieve dataset from cache.
        ...

    @abstractmethod
    def cache(self, data: Dataset) -> None:
        # Cache the dataset for faster opening.
        ...

    @abstractmethod
    def prepare(self) -> Dataset:
        # Prepare the dataset and return it.
        ...

    def retrieve(self, refresh_cache: bool = False, redownload: bool = False) -> Dataset:
        # Download dataset files (if required).
        if self.dataset_files is not None:
            if (
                any([not os.path.exists(os.path.join(self.dataset_dir, f)) for _, f in self.dataset_files])
                or redownload
            ):
                self.download_dataset()
        # Prepare and retrieve dataset.
        if self.is_cached() and not refresh_cache:
            return self.get_cache()
        else:
            data = self.prepare()
            self.cache(data)
            return self.get_cache()
