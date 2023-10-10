"""Test pipeline end-t-end for different predictive categories"""

from typing import Callable, Dict, List

import pytest

from tempor.methods.pipeline import PipelineBase, pipeline
from tempor.methods.prediction.one_off.classification import BaseOneOffClassifier
from tempor.methods.prediction.one_off.regression import BaseOneOffRegressor
from tempor.methods.prediction.temporal.classification import BaseTemporalClassifier
from tempor.methods.prediction.temporal.regression import BaseTemporalRegressor
from tempor.methods.time_to_event import BaseTimeToEventAnalysis
from tempor.methods.treatments.one_off import BaseOneOffTreatmentEffects
from tempor.methods.treatments.temporal import BaseTemporalTreatmentEffects
from tempor.utils.serialization import load, save

# TODO: It would be useful to create automated tests where all existing plugins available in each category are
# used as the last step of a pipeline.


TEST_TRANSFORM_STEPS: Dict[str, List] = {
    "CASE_A": [
        "preprocessing.imputation.static.static_tabular_imputer",
        "preprocessing.imputation.temporal.ts_tabular_imputer",
        "preprocessing.nop.nop_transformer",
        "preprocessing.imputation.temporal.bfill",
        "preprocessing.scaling.static.static_minmax_scaler",
        "preprocessing.scaling.temporal.ts_minmax_scaler",
    ],
    "CASE_B": [
        "preprocessing.imputation.static.static_tabular_imputer",
        "preprocessing.imputation.temporal.bfill",
        "preprocessing.scaling.temporal.ts_minmax_scaler",
    ],
    "CASE_C": [
        "preprocessing.imputation.static.static_tabular_imputer",
        "preprocessing.imputation.temporal.ffill",
        "preprocessing.scaling.static.static_minmax_scaler",
        "preprocessing.scaling.temporal.ts_minmax_scaler",
    ],
    "CASE_D": [
        "preprocessing.imputation.static.static_tabular_imputer",
        "preprocessing.imputation.temporal.ffill",
    ],
    "CASE_E": [],  # No transformer steps case.
}


def init_pipeline_and_fit(plugins_str, data_missing, data_not_missing, serialize, init_params=None):
    if len(plugins_str) > 1:
        dataset = data_missing
    else:
        dataset = data_not_missing

    PipelineCls = pipeline(plugins_str)
    pipe = PipelineCls(init_params)

    if serialize:
        dump = save(pipe)
        pipe = load(dump)

    pipe.fit(dataset)

    if serialize:
        dump = save(pipe)
        pipe = load(dump)

    return dataset, pipe


# Category: prediction.one_off.classification:


@pytest.mark.filterwarnings("ignore:.*validate_arguments.*:DeprecationWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*conflict.*:UserWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            *TEST_TRANSFORM_STEPS["CASE_A"],
            "prediction.one_off.classification.nn_classifier",
        ],
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_B"],
                "prediction.one_off.classification.nn_classifier",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_C"],
                "prediction.one_off.classification.nn_classifier",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_D"],
                "prediction.one_off.classification.nn_classifier",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_E"],
                "prediction.one_off.classification.nn_classifier",
            ],
            marks=pytest.mark.extra,
        ),
    ],
)
@pytest.mark.parametrize(
    "serialize",
    [
        pytest.param(True, marks=pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")),
        # ^ Expected: problem with current serialization.
        False,
    ],
)
def test_end2end_prediction_oneoff_classification(
    plugins_str, serialize: bool, sine_data_small, sine_data_missing_small
) -> None:
    dataset, pipe = init_pipeline_and_fit(
        plugins_str=plugins_str,
        data_missing=sine_data_missing_small,
        data_not_missing=sine_data_small,
        serialize=serialize,
    )

    y_pred = pipe.predict(dataset)
    y_proba = pipe.predict_proba(dataset)

    assert isinstance(pipe, PipelineBase)
    assert isinstance(pipe, BaseOneOffClassifier)
    assert y_pred.dataframe().shape == (len(dataset.predictive.targets.dataframe()), 1)
    assert y_proba.numpy().shape == (len(dataset.time_series), 2)


# Category: prediction.one_off.regression:


@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            *TEST_TRANSFORM_STEPS["CASE_A"],
            "prediction.one_off.regression.nn_regressor",
        ],
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_B"],
                "prediction.one_off.regression.nn_regressor",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_C"],
                "prediction.one_off.regression.nn_regressor",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_D"],
                "prediction.one_off.regression.nn_regressor",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_E"],
                "prediction.one_off.regression.nn_regressor",
            ],
            marks=pytest.mark.extra,
        ),
    ],
)
@pytest.mark.parametrize(
    "serialize",
    [
        pytest.param(True, marks=pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")),
        # ^ Expected: problem with current serialization.
        False,
    ],
)
def test_end2end_prediction_oneoff_regression(
    plugins_str, serialize: bool, sine_data_small, sine_data_missing_small
) -> None:
    dataset, pipe = init_pipeline_and_fit(
        plugins_str=plugins_str,
        data_missing=sine_data_missing_small,
        data_not_missing=sine_data_small,
        serialize=serialize,
    )

    y_pred = pipe.predict(dataset)

    assert isinstance(pipe, PipelineBase)
    assert isinstance(pipe, BaseOneOffRegressor)
    assert y_pred.dataframe().shape == (len(dataset.predictive.targets.dataframe()), 1)


# Category: prediction.temporal.classification:


@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            *TEST_TRANSFORM_STEPS["CASE_A"],
            "prediction.temporal.classification.seq2seq_classifier",
        ],
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_B"],
                "prediction.temporal.classification.seq2seq_classifier",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_C"],
                "prediction.temporal.classification.seq2seq_classifier",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_D"],
                "prediction.temporal.classification.seq2seq_classifier",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_E"],
                "prediction.temporal.classification.seq2seq_classifier",
            ],
            marks=pytest.mark.extra,
        ),
    ],
)
@pytest.mark.parametrize(
    "serialize",
    [
        pytest.param(True, marks=pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")),
        # ^ Expected: problem with current serialization.
        False,
    ],
)
def test_end2end_prediction_temporal_classification(
    plugins_str,
    serialize: bool,
    sine_data_temporal_small,
) -> None:
    dataset, pipe = init_pipeline_and_fit(
        plugins_str=plugins_str,
        data_missing=sine_data_temporal_small,
        data_not_missing=sine_data_temporal_small,
        serialize=serialize,
        init_params={"seq2seq_classifier": {"epochs": 2}},
    )

    y_pred = pipe.predict(dataset, n_future_steps=10)

    assert isinstance(pipe, PipelineBase)
    assert isinstance(pipe, BaseTemporalClassifier)
    assert y_pred.numpy().shape == (dataset.predictive.targets.num_samples, 10, 5)


# Category: prediction.temporal.regression:


@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            *TEST_TRANSFORM_STEPS["CASE_A"],
            "prediction.temporal.regression.seq2seq_regressor",
        ],
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_B"],
                "prediction.temporal.regression.seq2seq_regressor",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_C"],
                "prediction.temporal.regression.seq2seq_regressor",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_D"],
                "prediction.temporal.regression.seq2seq_regressor",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_E"],
                "prediction.temporal.regression.seq2seq_regressor",
            ],
            marks=pytest.mark.extra,
        ),
    ],
)
@pytest.mark.parametrize(
    "serialize",
    [
        pytest.param(True, marks=pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")),
        # ^ Expected: problem with current serialization.
        False,
    ],
)
def test_end2end_prediction_temporal_regression(
    plugins_str,
    serialize: bool,
    sine_data_temporal_small,
) -> None:
    dataset, pipe = init_pipeline_and_fit(
        plugins_str=plugins_str,
        data_missing=sine_data_temporal_small,
        data_not_missing=sine_data_temporal_small,
        serialize=serialize,
        init_params={"seq2seq_regressor": {"epochs": 2}},
    )

    y_pred = pipe.predict(dataset, n_future_steps=10)

    assert isinstance(pipe, PipelineBase)
    assert isinstance(pipe, BaseTemporalRegressor)
    assert y_pred.numpy().shape == (dataset.predictive.targets.num_samples, 10, 5)


# Category: time_to_event:


@pytest.mark.filterwarnings("ignore:.*Validation.*small.*:RuntimeWarning")  # Expected for small test datasets with DDH.
@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            *TEST_TRANSFORM_STEPS["CASE_A"],
            "time_to_event.dynamic_deephit",
        ],
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_B"],
                "time_to_event.dynamic_deephit",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_C"],
                "time_to_event.dynamic_deephit",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_D"],
                "time_to_event.dynamic_deephit",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_E"],
                "time_to_event.dynamic_deephit",
            ],
            marks=pytest.mark.extra,
        ),
    ],
)
@pytest.mark.parametrize(
    "serialize",
    [
        pytest.param(True, marks=pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")),
        # ^ Expected: problem with current serialization.
        False,
    ],
)
def test_end2end_time_to_event(
    plugins_str,
    serialize: bool,
    pbc_data_small,
    get_event0_time_percentiles,
) -> None:
    dataset, pipe = init_pipeline_and_fit(
        plugins_str=plugins_str,
        data_missing=pbc_data_small,
        data_not_missing=pbc_data_small,
        serialize=serialize,
        init_params={"dynamic_deephit": {"n_iter": 2}},
    )

    horizons = get_event0_time_percentiles(dataset, [0.25, 0.5, 0.75])
    y_pred = pipe.predict(dataset, horizons=horizons)

    assert isinstance(pipe, PipelineBase)
    assert isinstance(pipe, BaseTimeToEventAnalysis)
    assert y_pred.numpy().shape == (len(dataset.time_series), len(horizons), 1)


# Category: treatments.one_off.classification:
# NOTE: None.


# Category: treatments.one_off.regression:


@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            *TEST_TRANSFORM_STEPS["CASE_A"],
            "treatments.one_off.regression.synctwin_regressor",
        ],
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_B"],
                "treatments.one_off.regression.synctwin_regressor",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_C"],
                "treatments.one_off.regression.synctwin_regressor",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_D"],
                "treatments.one_off.regression.synctwin_regressor",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_E"],
                "treatments.one_off.regression.synctwin_regressor",
            ],
            marks=pytest.mark.extra,
        ),
    ],
)
@pytest.mark.parametrize(
    "serialize",
    [
        pytest.param(True, marks=pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")),
        # ^ Expected: problem with current serialization.
        False,
    ],
)
def test_end2end_treatments_oneoff_regression(
    plugins_str,
    serialize: bool,
    pkpd_data_small,
) -> None:
    dataset, pipe = init_pipeline_and_fit(
        plugins_str=plugins_str,
        data_missing=pkpd_data_small,
        data_not_missing=pkpd_data_small,
        serialize=serialize,
        init_params={
            "synctwin_regressor": {
                "pretraining_iterations": 3,
                "matching_iterations": 3,
                "inference_iterations": 3,
            }
        },
    )

    output = pipe.predict_counterfactuals(dataset)

    assert isinstance(pipe, PipelineBase)
    assert isinstance(pipe, BaseOneOffTreatmentEffects)
    assert len(output) == len(dataset)


# Category: treatments.temporal.classification:


@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            *TEST_TRANSFORM_STEPS["CASE_A"],
            "treatments.temporal.classification.crn_classifier",
        ],
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_B"],
                "treatments.temporal.classification.crn_classifier",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_C"],
                "treatments.temporal.classification.crn_classifier",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_D"],
                "treatments.temporal.classification.crn_classifier",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_E"],
                "treatments.temporal.classification.crn_classifier",
            ],
            marks=pytest.mark.extra,
        ),
    ],
)
@pytest.mark.parametrize(
    "serialize",
    [
        pytest.param(True, marks=pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")),
        # ^ Expected: problem with current serialization.
        False,
    ],
)
def test_end2end_treatments_temporal_classification(
    plugins_str,
    serialize: bool,
    clv_data_small,
    simulate_treatments_scenarios: Callable,
) -> None:
    dataset, pipe = init_pipeline_and_fit(
        plugins_str=plugins_str,
        data_missing=clv_data_small,
        data_not_missing=clv_data_small,
        serialize=serialize,
        init_params={"crn_classifier": {"n_iter": 2}},
    )

    n_counterfactuals_per_sample = 2
    horizons, treatment_scenarios = simulate_treatments_scenarios(
        dataset, n_counterfactuals_per_sample=n_counterfactuals_per_sample
    )
    output = pipe.predict_counterfactuals(dataset, horizons=horizons, treatment_scenarios=treatment_scenarios)

    assert isinstance(pipe, PipelineBase)
    assert isinstance(pipe, BaseTemporalTreatmentEffects)
    assert len(output) == len(dataset)
    assert len(output[0]) == n_counterfactuals_per_sample


# Category: treatments.temporal.regression:


@pytest.mark.filterwarnings("ignore:.*validate_arguments.*:DeprecationWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*conflict.*:UserWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            *TEST_TRANSFORM_STEPS["CASE_A"],
            "treatments.temporal.regression.crn_regressor",
        ],
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_B"],
                "treatments.temporal.regression.crn_regressor",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_C"],
                "treatments.temporal.regression.crn_regressor",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_D"],
                "treatments.temporal.regression.crn_regressor",
            ],
            marks=pytest.mark.extra,
        ),
        pytest.param(
            [
                *TEST_TRANSFORM_STEPS["CASE_E"],
                "treatments.temporal.regression.crn_regressor",
            ],
            marks=pytest.mark.extra,
        ),
    ],
)
@pytest.mark.parametrize(
    "serialize",
    [
        pytest.param(True, marks=pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")),
        # ^ Expected: problem with current serialization.
        False,
    ],
)
def test_end2end_treatments_temporal_regression(
    plugins_str,
    serialize: bool,
    clv_data_small,
    simulate_treatments_scenarios: Callable,
) -> None:
    dataset, pipe = init_pipeline_and_fit(
        plugins_str=plugins_str,
        data_missing=clv_data_small,
        data_not_missing=clv_data_small,
        serialize=serialize,
        init_params={"crn_regressor": {"n_iter": 2}},
    )

    n_counterfactuals_per_sample = 2
    horizons, treatment_scenarios = simulate_treatments_scenarios(
        dataset, n_counterfactuals_per_sample=n_counterfactuals_per_sample
    )
    output = pipe.predict_counterfactuals(dataset, horizons=horizons, treatment_scenarios=treatment_scenarios)

    assert isinstance(pipe, PipelineBase)
    assert isinstance(pipe, BaseTemporalTreatmentEffects)
    assert len(output) == len(dataset)
    assert len(output[0]) == n_counterfactuals_per_sample
