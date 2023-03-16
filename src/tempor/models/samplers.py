from typing import Any, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data.sampler
from pydantic import validate_arguments
from sklearn.model_selection import train_test_split


class BaseSampler(torch.utils.data.sampler.Sampler):
    """DataSampler samples the conditional vector and corresponding data."""

    def get_dataset_conditionals(self) -> Optional[np.ndarray]:
        return None

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample_conditional(self, batch: int, **kwargs: Any) -> Optional[Tuple]:  # pylint: disable=unused-argument
        return None

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample_conditional_for_class(
        self, batch: int, c: int  # pylint: disable=unused-argument
    ) -> Optional[np.ndarray]:
        return None

    def conditional_dimension(self) -> int:
        """Return the total number of categories."""
        return 0

    def conditional_probs(self) -> Optional[np.ndarray]:
        """Return the total number of categories."""
        return None

    def train_test(self) -> Tuple:
        raise NotImplementedError()


class ImbalancedDatasetSampler(BaseSampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, labels: List, train_size: float = 0.8) -> None:
        super().__init__(None)

        # if indices is not provided, all elements in the dataset will be considered
        indices = list(range(len(labels)))
        self.train_idx, self.test_idx = train_test_split(indices, train_size=train_size)
        self.train_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.train_idx)}

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_train_samples = len(self.train_idx)

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = labels
        df.index = indices  # pyright: ignore

        df = df.loc[self.train_idx]

        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self) -> Generator:
        return (
            self.train_mapping[self.train_idx[i]]
            for i in torch.multinomial(self.weights, self.num_train_samples, replacement=True)
        )

    def __len__(self) -> int:
        return len(self.train_idx)

    def train_test(self) -> Tuple:
        return self.train_idx, self.test_idx
