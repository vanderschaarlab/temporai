from pathlib import Path
from typing import Tuple, cast

from clairvoyance2.datasets.uci import uci_diabetes
from clairvoyance2.preprocessing.convenience import TemporalTargetsExtractor

from tempor.data import dataloader, dataset
from tempor.data.clv2conv import clairvoyance2_dataset_to_tempor_dataset


# TODO: Docstring.
class UCIDiabetesDataLoader(dataloader.TemporalPredictionDataLoader):
    def __init__(
        self,
        make_regular: bool = False,
        use_int_index: bool = True,
        targets: Tuple[str, ...] = ("hypoglycemic_symptoms",),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.make_regular = make_regular
        self.use_int_index = use_int_index
        self.targets = targets

    @staticmethod
    def url() -> str:
        return "https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes"

    @staticmethod
    def dataset_dir() -> str:
        return str(Path(UCIDiabetesDataLoader.data_root_dir) / "uci_diabetes")

    def load(self, **kwargs) -> dataset.TemporalPredictionDataset:
        clv_dataset = uci_diabetes(
            data_home=UCIDiabetesDataLoader.data_root_dir,
            refresh_cache=True,
            redownload=False,
            make_regular=self.make_regular,
            use_int_index=self.use_int_index,
        )
        clv_dataset = TemporalTargetsExtractor(params={"targets": self.targets}).fit_transform(clv_dataset)
        data = clairvoyance2_dataset_to_tempor_dataset(clv_dataset)
        data = cast(dataset.TemporalPredictionDataset, data)
        return data
