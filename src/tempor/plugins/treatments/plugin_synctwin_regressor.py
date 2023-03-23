import dataclasses
from typing import List

import numpy as np
from clairvoyance2.treatment_effects.synctwin import SyncTwinRegressor, TimeIndexHorizon

import tempor.plugins.core as plugins
from tempor.data import dataset, samples
from tempor.data.clv2conv import tempor_dataset_to_clairvoyance2_dataset
from tempor.plugins.core._params import FloatParams, IntegerParams
from tempor.plugins.treatments import BaseTreatments


@dataclasses.dataclass
class SyncTwinParams:
    # Main hyperparameters:
    hidden_size: int = 20
    tau: float = 1.0
    lambda_prognostic: float = 1.0
    lambda_reconstruction: float = 1.0
    batch_size: int = 32
    pretraining_iterations: int = 5_000
    matching_iterations: int = 20_000
    inference_iterations: int = 20_000
    # Misc:
    use_validation_set_in_training: bool = True
    treatment_status_is_treated: int = 1


@plugins.register_plugin(name="synctwin_regressor", category="treatments")
class SyncTwinTreatmentsRegressor(BaseTreatments):
    """
    Paper: Estimating counterfactual treatment outcomes over time through adversarially balanced representations, Ioana Bica, Ahmed M. Alaa, James Jordon, Mihaela van der Schaar
    """

    ParamsDefinition = SyncTwinParams
    params: SyncTwinParams  # type: ignore

    def __init__(
        self,
        **params,
    ) -> None:
        """.

        Example:
            >>> from tempor.plugins import plugin_loader
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("treatments.synctwin_regressor", n_iter=50)
            >>>
            >>> # Train:
            >>> # model.fit(dataset)
            >>>
            >>> # Predict:
            >>> # assert model.predict(dataset, n_future_steps = 10).numpy().shape == (len(dataset), 10, 5)
        """
        super().__init__(**params)
        self.model = SyncTwinRegressor(
            params=self.params,
        )

    def _fit(
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> "SyncTwinTreatmentsRegressor":  # pyright: ignore
        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)
        self.model.fit(cl_dataset)
        return self

    def _predict(  # type: ignore[override]
        self,
        data: dataset.Dataset,
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        raise NotImplementedError()

    def _predict_counterfactuals(  # type: ignore[override]
        self,
        data: dataset.Dataset,
        n_counterfactuals_per_sample: int = 2,
        *args,
        **kwargs,
    ) -> List:
        if self.model is None:
            raise RuntimeError("Fit the model first")

        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)

        horizon_counterfactuals = TimeIndexHorizon(
            time_index_sequence=[tc.time_index[len(tc.time_index) // 2 :] for tc in cl_dataset.temporal_covariates]
        )

        counterfactuals = []
        for idx, sample_idx in enumerate(cl_dataset.sample_indices):
            treat = cl_dataset.temporal_treatments[sample_idx].df.values
            horizon_counterfactuals_sample = horizon_counterfactuals.time_index_sequence[idx]
            treat_scenarios = []
            for treat_sc_idx in range(n_counterfactuals_per_sample):
                np.random.seed(12345 + treat_sc_idx)
                treat_sc = np.random.randint(
                    low=0, high=1 + 1, size=(len(horizon_counterfactuals_sample), treat.shape[1])
                )
                treat_scenarios.append(treat_sc)

            c = self.model.predict_counterfactuals(
                cl_dataset,
                sample_index=sample_idx,
                treatment_scenarios=treat_scenarios,
                horizon=TimeIndexHorizon(time_index_sequence=[horizon_counterfactuals_sample]),
                **kwargs,
            )
            counterfactuals.append(c)
        return counterfactuals

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return [
            IntegerParams(name="hidden_size", low=10, high=500),
            FloatParams(name="tau", low=0.0, high=2.0),
            FloatParams(name="lambda_prognostic", low=0.0, high=2.0),
            FloatParams(name="lambda_reconstruction", low=0.0, high=2.0),
            IntegerParams(name="batch_size", low=10, high=500),
        ]
