"""Module for the PKPD data source plugin."""

from typing import Any, cast

from clairvoyance2.datasets.simulated.simple_pkpd import simple_pkpd_dataset

from tempor.core import plugins
from tempor.data import dataset
from tempor.data.clv2conv import clairvoyance2_dataset_to_tempor_dataset
from tempor.datasources import datasource


@plugins.register_plugin(name="pkpd", category="treatments.one_off", plugin_type="datasource")
class PKPDDataSource(datasource.OneOffTreatmentEffectsDataSource):
    def __init__(
        self,
        n_timesteps: int = 10,
        time_index_treatment_event: int = 7,
        n_control_samples: int = 20,
        n_treated_samples: int = 20,
        random_state: int = 100,
        **kwargs: Any,
    ) -> None:
        """PKPD data source for one-off treatment effects tasks.

        Adapted from: https://github.com/ZhaozhiQIAN/SyncTwin-NeurIPS-2021

        Args:
            n_timesteps (int, optional): Number of timesteps. Defaults to ``10``.
            time_index_treatment_event (int, optional): Time index of the treatment event. Defaults to ``7``.
            n_control_samples (int, optional): Number of control samples to generate. Defaults to ``20``.
            n_treated_samples (int, optional): Number of treated samples to generate. Defaults to ``20``.
            random_state (int, optional): Random state to use. Defaults to ``100``.
            **kwargs (Any): Any additional keyword arguments will be passed to parent constructor.

        Reference:
            Qian, Z., Zhang, Y., Bica, I., Wood, A., & van der Schaar, M. (2021). Synctwin: Treatment effect \
            estimation with longitudinal outcomes. Advances in Neural Information Processing Systems, 34, 3178-3190.
        """
        super().__init__(**kwargs)

        self.n_timesteps = n_timesteps
        self.time_index_treatment_event = time_index_treatment_event
        self.n_control_samples = n_control_samples
        self.n_treated_samples = n_treated_samples
        self.random_state = random_state

    @staticmethod
    def url() -> None:  # noqa: D102
        return None

    @staticmethod
    def dataset_dir() -> None:  # noqa: D102
        return None

    def load(self, **kwargs: Any) -> dataset.OneOffTreatmentEffectsDataset:  # noqa: D102
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
