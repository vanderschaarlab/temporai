import dataclasses
from typing import List

from clairvoyance2.treatment_effects.synctwin import SyncTwinRegressor

import tempor.plugins.core as plugins
from tempor.data import dataset, samples
from tempor.data.clv2conv import tempor_dataset_to_clairvoyance2_dataset
from tempor.plugins.core._params import FloatParams, IntegerParams
from tempor.plugins.treatments.one_off._base import BaseOneOffTreatmentEffects


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


@plugins.register_plugin(name="synctwin_regressor", category="treatments.one_off.regression")
class SyncTwinTreatmentsRegressor(BaseOneOffTreatmentEffects):
    ParamsDefinition = SyncTwinParams
    params: SyncTwinParams  # type: ignore

    def __init__(
        self,
        **params,
    ) -> None:
        """SyncTwin treatment effects estimation.

        Example:
            >>> from tempor.plugins import plugin_loader
            >>>
            >>> # Load the model:
            >>> model = plugin_loader.get("treatments.one_off.regression.synctwin_regressor", n_iter=50)
            >>>
            >>> # Train:
            >>> # model.fit(dataset)
            >>>
            >>> # Predict:
            >>> # assert model.predict(dataset, n_future_steps = 10).numpy().shape == (len(dataset), 10, 5)

        References:
            SyncTwin: Treatment Effect Estimation with Longitudinal Outcomes,
            Zhaozhi Qian, Yao Zhang, Ioana Bica, Angela Wood, Mihaela van der Schaar.
        """
        super().__init__(**params)
        self.model = SyncTwinRegressor(
            params=self.params,  # pyright: ignore
        )

    def _fit(
        self,
        data: dataset.PredictiveDataset,
        *args,
        **kwargs,
    ) -> "SyncTwinTreatmentsRegressor":  # pyright: ignore
        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)
        self.model.fit(cl_dataset)
        return self

    def _predict(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        data: dataset.PredictiveDataset,
        horizons: List[List[float]],
        *args,
        **kwargs,
    ) -> samples.TimeSeriesSamples:
        raise NotImplementedError(
            "SyncTwin implementation does not currently support `predict`, only `predict_counterfactuals`"
        )

    def _predict_counterfactuals(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        data: dataset.PredictiveDataset,
        # horizons: SyncTwin can only handle the same time horizon as targets in the data.
        # treatment_scenarios: SyncTwin can only handle the one alternative treatment case.
        *args,
        **kwargs,
    ) -> List:
        if self.model is None:
            raise RuntimeError("Fit the model first")

        cl_dataset = tempor_dataset_to_clairvoyance2_dataset(data)

        counterfactuals: List = []
        for idx, sample_idx in enumerate(cl_dataset.sample_indices):  # pylint: disable=unused-variable
            treatment_status = data[idx].predictive.treatments.dataframe().iloc[0, 0][1]  # type: ignore

            if treatment_status is True:
                treat_scenarios = self.model.get_possible_treatment_scenarios(sample_index=sample_idx, data=cl_dataset)
                horizon_counterfactuals_sample = self.model.get_possible_prediction_horizon(
                    sample_index=sample_idx, data=cl_dataset
                )

                # TODO: should enforce treat - treat_scenarios shapes here.
                c = self.model.predict_counterfactuals(
                    cl_dataset,
                    sample_index=sample_idx,
                    treatment_scenarios=treat_scenarios,
                    horizon=horizon_counterfactuals_sample,
                )

                # Export as DFs, rather than clairvoyance2 TimeSeries:
                c_dfs = []
                for c_ in c:
                    c_df = c_.df
                    c_df.index.name = "time_idx"
                    c_dfs.append(c_df)

                counterfactuals.append(c_dfs)

            else:
                counterfactuals.append(
                    "SyncTwin implementation can currently only predict counterfactuals for treated "
                    f"samples (event value = True), but this sample (sample_idx = {sample_idx}) was untreated"
                )

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
