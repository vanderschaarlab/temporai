from typing import cast

from clairvoyance2.datasets.simulated.simple_pkpd import simple_pkpd_dataset

from tempor.data import dataloader, dataset
from tempor.data.clv2conv import clairvoyance2_dataset_to_tempor_dataset


# TODO: Docstring.
class PKPDDataLoader(dataloader.OneOffTreatmentEffectsDataLoader):
    def __init__(
        self,
        n_timesteps: int = 10,
        time_index_treatment_event: int = 7,
        n_control_samples: int = 20,
        n_treated_samples: int = 20,
        random_state: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.n_timesteps = n_timesteps
        self.time_index_treatment_event = time_index_treatment_event
        self.n_control_samples = n_control_samples
        self.n_treated_samples = n_treated_samples
        self.random_state = random_state

    @staticmethod
    def url() -> None:
        return None

    @staticmethod
    def dataset_dir() -> None:
        return None

    def load(self, **kwargs) -> dataset.OneOffTreatmentEffectsDataset:
        clv_dataset = simple_pkpd_dataset(
            n_timesteps=self.n_timesteps,
            time_index_treatment_event=self.time_index_treatment_event,
            n_control_samples=self.n_control_samples,
            n_treated_samples=self.n_treated_samples,
            seed=self.random_state,
        )
        data = clairvoyance2_dataset_to_tempor_dataset(clv_dataset)
        data = cast(dataset.OneOffTreatmentEffectsDataset, data)
        return data
