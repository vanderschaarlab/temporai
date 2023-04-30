# pylint: disable=no-member

from typing import Any, List

import pytest

from tempor.plugins.pipeline import Pipeline, PipelineGroup, PipelineMeta
from tempor.utils.serialization import load, save

TEST_ON_DATASETS = ["sine_data_small"]


@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            "preprocessing.imputation.temporal.bfill",
            "preprocessing.scaling.static.static_minmax_scaler",
            "preprocessing.scaling.temporal.ts_minmax_scaler",
            "prediction.one_off.classification.nn_classifier",
        ],
        [
            "preprocessing.imputation.static.static_tabular_imputer",
            "preprocessing.imputation.temporal.ts_tabular_imputer",
            "prediction.one_off.regression.nn_regressor",
        ],
        [
            "prediction.one_off.classification.nn_classifier",
        ],
    ],
)
def test_sanity(plugins_str: List[Any]) -> None:
    dtype: PipelineMeta = Pipeline(plugins_str)
    plugins = PipelineGroup(plugins_str)

    assert dtype.name() == "->".join(p for p in plugins_str)

    args = {"features_count": 10}
    for act, pl in zip(dtype.hyperparameter_space(**args), plugins):
        assert len(dtype.hyperparameter_space(**args)[act]) == len(pl.hyperparameter_space(**args))


@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            "prediction.one_off.classification.nn_classifier",
            "preprocessing.imputation.temporal.bfill",
        ],
        [
            "preprocessing.imputation.temporal.bfill",
            "preprocessing.imputation.temporal.bfill",
            "preprocessing.imputation.temporal.bfill",
            "preprocessing.imputation.temporal.bfill",
        ],
        [
            "prediction.one_off.regression.nn_regressor",
            "prediction.one_off.regression.nn_regressor",
        ],
        [
            "prediction.one_off.regression.nn_regressor",
            "prediction.one_off.regression.nn_regressor",
            "preprocessing.imputation.temporal.bfill",
        ],
        [
            "prediction.one_off.regression.nn_regressor",
            "preprocessing.imputation.temporal.bfill",
            "preprocessing.imputation.temporal.bfill",
            "preprocessing.imputation.temporal.bfill",
            "preprocessing.scaling.temporal.ts_minmax_scaler",
        ],
        [],
    ],
)
def test_fails(plugins_str: List[Any]) -> None:
    with pytest.raises(RuntimeError):
        Pipeline(plugins_str)()


@pytest.mark.filterwarnings("ignore:RNN.*contiguous.*:UserWarning")  # Expected: problem with current serialization.
@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            "preprocessing.imputation.static.static_tabular_imputer",
            "preprocessing.imputation.temporal.ts_tabular_imputer",
            "preprocessing.nop.nop_transformer",
            "preprocessing.imputation.temporal.bfill",
            "preprocessing.scaling.static.static_minmax_scaler",
            "preprocessing.scaling.temporal.ts_minmax_scaler",
            "prediction.one_off.classification.nn_classifier",
        ],
        [
            "preprocessing.imputation.static.static_tabular_imputer",
            "preprocessing.imputation.temporal.bfill",
            "preprocessing.scaling.temporal.ts_minmax_scaler",
            "prediction.one_off.regression.nn_regressor",
        ],
        [
            "preprocessing.imputation.static.static_tabular_imputer",
            "preprocessing.imputation.temporal.ffill",
            "preprocessing.scaling.static.static_minmax_scaler",
            "preprocessing.scaling.temporal.ts_minmax_scaler",
            "prediction.one_off.regression.nn_regressor",
        ],
        [
            "preprocessing.imputation.static.static_tabular_imputer",
            "preprocessing.imputation.temporal.ffill",
            "prediction.one_off.regression.nn_regressor",
        ],
        [
            "prediction.one_off.classification.nn_classifier",
        ],
    ],
)
@pytest.mark.parametrize("serialize", [True, False])
def test_end2end(plugins_str, serialize: bool, sine_data_small, sine_data_missing_small) -> None:
    if len(plugins_str) > 1:
        dataset = sine_data_missing_small
    else:
        dataset = sine_data_small

    template: PipelineMeta = Pipeline(plugins_str)
    pipeline = template()

    if serialize:
        dump = save(pipeline)
        pipeline = load(dump)

    pipeline.fit(dataset)

    if serialize:
        dump = save(pipeline)
        pipeline = load(dump)

    y_pred = pipeline.predict(dataset)

    assert y_pred.dataframe().shape == (len(dataset.predictive.targets.dataframe()), 1)
