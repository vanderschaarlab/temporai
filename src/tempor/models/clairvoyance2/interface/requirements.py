# mypy: ignore-errors

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, NoReturn, Optional, Sequence, Union

from ..data import Dataset, EventSamples, StaticSamples, TimeSeries, TimeSeriesSamples
from ..data.constants import T_NumericDtype_AsTuple, T_SamplesIndexDtype
from ..utils.common import python_type_from_np_pd_dtype
from .horizon import Horizon, HorizonOpts, NStepAheadHorizon

if TYPE_CHECKING:
    from .model import TTreatmentScenarios


class DataStructureOpts(Enum):
    TIME_SERIES = auto()
    STATIC = auto()
    EVENT = auto()


class DataValueOpts(Enum):
    ANY = auto()
    NUMERIC = auto()
    NUMERIC_CATEGORICAL = auto()
    NUMERIC_BINARY = auto()


@dataclass(frozen=True)
class DatasetRequirements:
    # Miscellaneous:
    requires_static_covariates_present: bool = False
    requires_no_missing_data: bool = False
    # Value types:
    static_covariates_value_type: DataValueOpts = DataValueOpts.ANY
    temporal_covariates_value_type: DataValueOpts = DataValueOpts.ANY
    temporal_targets_value_type: DataValueOpts = DataValueOpts.ANY
    temporal_treatments_value_type: DataValueOpts = DataValueOpts.ANY
    event_covariates_value_type: DataValueOpts = DataValueOpts.ANY
    event_targets_value_type: DataValueOpts = DataValueOpts.ANY
    event_treatments_value_type: DataValueOpts = DataValueOpts.ANY
    # Special temporal requirements:
    requires_all_temporal_data_samples_aligned: bool = False
    requires_all_temporal_data_regular: bool = False
    requires_all_temporal_data_index_numeric: bool = False
    requires_all_temporal_containers_shares_index: bool = True


@dataclass(frozen=True)
class PredictionRequirements:
    target_data_structure: DataStructureOpts = DataStructureOpts.TIME_SERIES
    horizon_type: HorizonOpts = HorizonOpts.N_STEP_AHEAD
    min_timesteps_target_when_fit: int = 1
    min_timesteps_target_when_predict: int = 1


@dataclass(frozen=True)
class TreatmentEffectsRequirements:
    # NOTE: target, horizon are expected to be the same as for PredictionRequirements.
    treatment_data_structure: DataStructureOpts = DataStructureOpts.TIME_SERIES
    min_timesteps_treatment_when_fit: int = 1
    min_timesteps_treatment_when_predict: int = 1
    min_timesteps_treatment_when_predict_counterfactual: int = 1


@dataclass(frozen=True)
class Requirements:
    dataset_requirements: DatasetRequirements = DatasetRequirements()
    prediction_requirements: Optional[PredictionRequirements] = None
    treatment_effects_requirements: Optional[TreatmentEffectsRequirements] = None


def raise_requirements_mismatch_error(requirement_name: str, explanation_text: str) -> NoReturn:
    raise RuntimeError(f"Requirements mismatch occurred. Requirement: '{requirement_name}'. {explanation_text}.")


def get_container_friendly_name(container_name: str) -> str:
    return container_name.replace("_", " ")


# NOTE: Needs more testing.
class RequirementsChecker:
    @staticmethod
    def _check_min_timesteps(min_timesteps: int, container: TimeSeriesSamples, container_name: str, preface: str):
        if min(container.n_timesteps_per_sample) < min_timesteps:
            raise_requirements_mismatch_error(
                f"{preface} {get_container_friendly_name(container_name)}",
                f"Requires at least {min_timesteps} but a sample with "
                f"{min(container.n_timesteps_per_sample)} timesteps was found",
            )

    @staticmethod
    def _check_data_value_type(
        requirement: DataValueOpts,
        container: Union[TimeSeriesSamples, StaticSamples, EventSamples],
        preface: str,
    ):
        if requirement in (
            DataValueOpts.NUMERIC,
            DataValueOpts.NUMERIC_CATEGORICAL,
            DataValueOpts.NUMERIC_BINARY,
        ):
            if not container.all_features_numeric:
                raise_requirements_mismatch_error(
                    f"{preface} `{requirement.name}`",
                    f"Incompatible data found. Preview:\n{container.df.head()}",
                )
        if requirement == DataValueOpts.NUMERIC_CATEGORICAL:
            if not container.all_features_categorical:
                raise_requirements_mismatch_error(
                    f"{preface} `{requirement.name}`",
                    f"Incompatible data found. Preview:\n{container.df.head()}",
                )
        if requirement == DataValueOpts.NUMERIC_BINARY:
            if not container.all_features_binary:
                raise_requirements_mismatch_error(
                    f"{preface} `{requirement.name}`",
                    f"Incompatible data found. Preview:\n{container.df.head()}",
                )

    @staticmethod
    def _check_data_requirements_predict(  # pylint: disable=unused-argument
        called_at_fit_time: bool,
        requirements: Requirements,
        data: Dataset,
        horizon: Optional[Horizon],
        **kwargs,
    ):
        if requirements.prediction_requirements is None:
            raise RuntimeError("Prediction requirements need to be set on a predictor model, but None found")

        if requirements.prediction_requirements.target_data_structure == DataStructureOpts.TIME_SERIES:
            if data.temporal_targets is None:
                raise_requirements_mismatch_error(
                    f"Prediction requirement: prediction target `{DataStructureOpts.TIME_SERIES}`",
                    "Dataset must contain temporal targets in this case but did not",
                )

            if called_at_fit_time:
                if requirements.prediction_requirements.min_timesteps_target_when_fit is not None:
                    RequirementsChecker._check_min_timesteps(
                        min_timesteps=requirements.prediction_requirements.min_timesteps_target_when_fit,
                        container=data.temporal_targets,
                        container_name="temporal_targets",
                        preface="Prediction requirement: minimum number of timesteps at fit-time,",
                    )
            else:
                RequirementsChecker._check_min_timesteps(
                    min_timesteps=requirements.prediction_requirements.min_timesteps_target_when_predict,
                    container=data.temporal_targets,
                    container_name="temporal_targets",
                    preface="Prediction requirement: minimum number of timesteps at predict-time,",
                )
                if requirements.treatment_effects_requirements is not None and data.temporal_treatments is not None:
                    RequirementsChecker._check_min_timesteps(
                        min_timesteps=requirements.treatment_effects_requirements.min_timesteps_treatment_when_predict,
                        container=data.temporal_treatments,
                        container_name="temporal_treatments",
                        preface="Treatment effects requirement: minimum number of timesteps at predict-time,",
                    )

            if horizon is not None:
                if requirements.prediction_requirements.horizon_type == HorizonOpts.N_STEP_AHEAD:
                    if not isinstance(horizon, NStepAheadHorizon):
                        raise_requirements_mismatch_error(
                            f"Prediction requirement: prediction horizon `{HorizonOpts.N_STEP_AHEAD}`",
                            f"A prediction horizon of type {NStepAheadHorizon} is expected, but found {type(horizon)}",
                        )
                    for container_name, container in data.temporal_data_containers.items():
                        len_ = max(container.n_timesteps_per_sample)
                        if horizon.n_step >= len_:
                            raise_requirements_mismatch_error(
                                f"Prediction requirement: prediction horizon `{HorizonOpts.N_STEP_AHEAD}`",
                                "N step ahead horizon must be < max timesteps in "
                                f"{get_container_friendly_name(container_name)}, but was "
                                f"{horizon.n_step} >= {len_}",
                            )

                # PredictionTargetType.TIME_SERIES > PredictionHorizonType.TIME_INDEX:
                if requirements.prediction_requirements.horizon_type == HorizonOpts.TIME_INDEX:
                    # TODO: Implement any data requirements.
                    pass

        elif requirements.prediction_requirements.target_data_structure == DataStructureOpts.EVENT:
            # TODO: Any requirements checks.
            pass

        elif requirements.prediction_requirements.target_data_structure == DataStructureOpts.STATIC:
            # TODO: Any requirements checks.
            pass

    @staticmethod
    def _check_data_requirements_predict_counterfactuals(  # pylint: disable=unused-argument
        called_at_fit_time: bool,
        requirements: Requirements,
        data: Dataset,
        sample_index: Optional[T_SamplesIndexDtype],
        treatment_scenarios: Optional["TTreatmentScenarios"],
        horizon: Optional[Horizon],
        **kwargs,
    ):
        if requirements.treatment_effects_requirements is None:
            raise RuntimeError(
                "Treatment effects requirements need to be set on a treatment effects model, but None found"
            )

        # DataStructure.TIME_SERIES:
        if requirements.treatment_effects_requirements.treatment_data_structure == DataStructureOpts.TIME_SERIES:
            if data.temporal_targets is None:
                raise_requirements_mismatch_error(
                    f"Treatment effects requirements: treatment type `{DataStructureOpts.TIME_SERIES}`",
                    "Dataset must contain temporal targets in this case but did not",
                )
            if data.temporal_treatments is None:
                raise_requirements_mismatch_error(
                    f"Treatment effects requirements: treatment type `{DataStructureOpts.TIME_SERIES}`",
                    "Dataset must contain temporal treatments in this case but did not",
                )

            if called_at_fit_time:
                RequirementsChecker._check_min_timesteps(
                    min_timesteps=requirements.treatment_effects_requirements.min_timesteps_treatment_when_fit,
                    container=data.temporal_treatments,
                    container_name="temporal_treatments",
                    preface="Treatment effects requirement: minimum number of timesteps at fit-time,",
                )
            else:
                RequirementsChecker._check_min_timesteps(
                    min_timesteps=requirements.treatment_effects_requirements.min_timesteps_treatment_when_predict_counterfactual,
                    container=data.temporal_targets,
                    container_name="temporal_targets",
                    preface="Treatment effects requirement: minimum number of timesteps at "
                    "predict-counterfactual-time,",
                )

        elif requirements.treatment_effects_requirements.treatment_data_structure == DataStructureOpts.EVENT:
            # TODO: Any requirements checks.
            pass

        elif requirements.treatment_effects_requirements.treatment_data_structure == DataStructureOpts.STATIC:
            # TODO: Any requirements checks.
            pass

        # TODO: The below is temporary. Interface is not settled and may change.
        if treatment_scenarios is not None:
            if requirements.treatment_effects_requirements.treatment_data_structure == DataStructureOpts.TIME_SERIES:
                assert isinstance(treatment_scenarios, Sequence)
                for ts in treatment_scenarios:
                    assert isinstance(ts, TimeSeries)
            if requirements.dataset_requirements.temporal_treatments_value_type == DataValueOpts.NUMERIC_BINARY:
                for ts in treatment_scenarios:
                    assert ts.all_features_numeric and ts.all_features_binary

    @staticmethod
    def check_data_requirements_general(called_at_fit_time: bool, requirements: Requirements, data: Dataset, **kwargs):
        # General data requirements.

        # Miscellaneous.
        if requirements.dataset_requirements.requires_static_covariates_present:
            if data.static_covariates is None:
                raise_requirements_mismatch_error(
                    "Dataset requirement: requires static samples", "Dataset did not have static samples"
                )
        if requirements.dataset_requirements.requires_no_missing_data:
            for container_name, container in data.all_data_containers.items():
                if container.has_missing:
                    raise_requirements_mismatch_error(
                        "Dataset requirement: requires no missing data",
                        f"Dataset {get_container_friendly_name(container_name)} had missing data",
                    )

        # Check data value types.
        if data.static_covariates is not None and requirements.dataset_requirements.static_covariates_value_type:
            RequirementsChecker._check_data_value_type(
                requirement=requirements.dataset_requirements.static_covariates_value_type,
                container=data.static_covariates,
                preface="Dataset requirement: static covariates data type",
            )
        if requirements.dataset_requirements.temporal_covariates_value_type:
            RequirementsChecker._check_data_value_type(
                requirement=requirements.dataset_requirements.temporal_covariates_value_type,
                container=data.temporal_covariates,
                preface="Dataset requirement: temporal covariates data type",
            )
        if data.temporal_targets is not None and requirements.dataset_requirements.temporal_targets_value_type:
            RequirementsChecker._check_data_value_type(
                requirement=requirements.dataset_requirements.temporal_targets_value_type,
                container=data.temporal_targets,
                preface="Dataset requirement: temporal target data type",
            )
        if data.temporal_treatments is not None and requirements.dataset_requirements.temporal_treatments_value_type:
            RequirementsChecker._check_data_value_type(
                requirement=requirements.dataset_requirements.temporal_treatments_value_type,
                container=data.temporal_treatments,
                preface="Dataset requirement: temporal treatment data type",
            )
        if data.event_covariates is not None and requirements.dataset_requirements.event_covariates_value_type:
            RequirementsChecker._check_data_value_type(
                requirement=requirements.dataset_requirements.event_covariates_value_type,
                container=data.event_covariates,
                preface="Dataset requirement: event covariates data type",
            )
        if data.event_targets is not None and requirements.dataset_requirements.event_targets_value_type:
            RequirementsChecker._check_data_value_type(
                requirement=requirements.dataset_requirements.event_targets_value_type,
                container=data.event_targets,
                preface="Dataset requirement: event target data type",
            )
        if data.event_treatments is not None and requirements.dataset_requirements.event_treatments_value_type:
            RequirementsChecker._check_data_value_type(
                requirement=requirements.dataset_requirements.event_treatments_value_type,
                container=data.event_treatments,
                preface="Dataset requirement: event treatment data type",
            )

        # Special temporal requirements.
        if requirements.dataset_requirements.requires_all_temporal_data_regular:
            for container_name, container in data.temporal_data_containers.items():
                is_regular, _ = container.is_regular()
                # TODO: Compare the diff. and ensure they are the same?
                if not is_regular:
                    raise_requirements_mismatch_error(
                        "Dataset requirement: requires regular timeseries",
                        f"Dataset {get_container_friendly_name(container_name)} did not have a regular time index",
                    )
        if requirements.dataset_requirements.requires_all_temporal_data_samples_aligned:
            for container_name, container in data.temporal_data_containers.items():
                if not container.all_samples_aligned:
                    raise_requirements_mismatch_error(
                        "Dataset requirement: requires aligned timeseries",
                        f"Dataset {get_container_friendly_name(container_name)} were not all aligned by their index",
                    )
        if requirements.dataset_requirements.requires_all_temporal_data_index_numeric:
            acceptable_types = T_NumericDtype_AsTuple
            for container_name, container in data.temporal_data_containers.items():
                if len(container) > 0:
                    ts = container[0]
                    if TYPE_CHECKING:
                        assert isinstance(ts, TimeSeries)
                    dtype = python_type_from_np_pd_dtype(ts.time_index.dtype)  # type: ignore
                    if dtype not in acceptable_types:
                        raise_requirements_mismatch_error(
                            "Dataset requirement: requires numeric timeseries index",
                            f"Dataset {get_container_friendly_name(container_name)} had index of dtype {dtype}",
                        )
        if requirements.dataset_requirements.requires_all_temporal_containers_shares_index:
            check_outcome, names = data.check_temporal_containers_have_same_time_index()
            if check_outcome is False:
                assert names is not None
                a_name, b_name = names
                raise_requirements_mismatch_error(
                    "Dataset requirement: requires all temporal containers have same time index (for each sample)",
                    f"The containers {a_name} and {b_name} did not have the same time index for all samples",
                )

        # Try to get additional kwargs if provided.
        horizon = kwargs.pop("horizon") if "horizon" in kwargs else None
        sample_index = kwargs.pop("sample_index") if "sample_index" in kwargs else None
        treatment_scenarios = kwargs.pop("treatment_scenarios") if "treatment_scenarios" in kwargs else None

        # Prediction-specific data requirements:
        if requirements.prediction_requirements is not None:
            RequirementsChecker._check_data_requirements_predict(
                called_at_fit_time=called_at_fit_time, requirements=requirements, data=data, horizon=horizon, **kwargs
            )

        # Treatment effects -specific data requirements:
        if requirements.treatment_effects_requirements is not None:
            # DataStructure.TIME_SERIES:
            RequirementsChecker._check_data_requirements_predict_counterfactuals(
                called_at_fit_time=called_at_fit_time,
                requirements=requirements,
                data=data,
                sample_index=sample_index,
                treatment_scenarios=treatment_scenarios,
                horizon=horizon,
                **kwargs,
            )

    @staticmethod
    def check_data_requirements_transform(requirements: Requirements, data: Dataset, **kwargs):
        # Currently no checks.
        pass

    @staticmethod
    def check_data_requirements_predict(
        requirements: Requirements, data: Dataset, horizon: Optional[Horizon], **kwargs
    ):
        # Currently no checks.
        if horizon is None:
            raise RuntimeError("Prediction model must receive a horizon object at predict-time")

        RequirementsChecker._check_data_requirements_predict(
            called_at_fit_time=False, requirements=requirements, data=data, horizon=horizon, **kwargs
        )

    @staticmethod
    def check_data_requirements_predict_counterfactuals(
        requirements: Requirements,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: "TTreatmentScenarios",
        horizon: Optional[Horizon],
        **kwargs,
    ):
        # Currently no checks.
        if horizon is None:
            raise RuntimeError("Treatment effects model must receive a horizon object at predict-time")

        RequirementsChecker._check_data_requirements_predict_counterfactuals(
            called_at_fit_time=False,
            requirements=requirements,
            data=data,
            sample_index=sample_index,
            treatment_scenarios=treatment_scenarios,
            horizon=horizon,
            **kwargs,
        )

    @staticmethod
    def check_predictor_model_requirements(predictor):
        requirements: Requirements = predictor.requirements
        if requirements.prediction_requirements is None:
            raise_requirements_mismatch_error(
                "Prediction requirements",
                f"Prediction model {predictor.__class__.__name__} must have prediction requirements defined, "
                "but found None",
            )

    @staticmethod
    def check_treatment_effects_model_requirements(treatment_effects_model):
        requirements: Requirements = treatment_effects_model.requirements
        if requirements.treatment_effects_requirements is None:
            raise_requirements_mismatch_error(
                "Treatment effects requirements",
                f"Treatment effects model {treatment_effects_model.__class__.__name__} must have treatment effects "
                "requirements defined, but found None",
            )
