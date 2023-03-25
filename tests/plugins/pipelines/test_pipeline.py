# pylint: disable=no-member

from typing import Any, List

import pytest

from tempor.plugins.pipeline import Pipeline, PipelineGroup, PipelineMeta
from tempor.utils.dataloaders.sine import SineDataLoader
from tempor.utils.serialization import load, save


@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            "preprocessing.imputation.bfill",
            "preprocessing.scaling.static_minmax_scaler",
            "preprocessing.scaling.ts_minmax_scaler",
            "classification.nn_classifier",
        ],
        [
            "preprocessing.imputation.static_imputation",
            "regression.nn_regressor",
        ],
        [
            "classification.nn_classifier",
        ],
    ],
)
def test_pipeline_sanity(plugins_str: List[Any]) -> None:
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
            "classification.nn_classifier",
            "preprocessing.imputation.bfill",
        ],
        [
            "preprocessing.imputation.bfill",
            "preprocessing.imputation.bfill",
            "preprocessing.imputation.bfill",
            "preprocessing.imputation.bfill",
        ],
        [
            "regression.nn_regressor",
            "regression.nn_regressor",
        ],
        [
            "regression.nn_regressor",
            "regression.nn_regressor",
            "preprocessing.imputation.bfill",
        ],
        [
            "regression.nn_regressor",
            "preprocessing.imputation.bfill",
            "preprocessing.imputation.bfill",
            "preprocessing.imputation.bfill",
            "preprocessing.scaling.ts_minmax_scaler",
        ],
        [],
    ],
)
def test_pipeline_fails(plugins_str: List[Any]) -> None:
    with pytest.raises(RuntimeError):
        Pipeline(plugins_str)()


@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            "preprocessing.imputation.static_imputation",
            "preprocessing.imputation.nop_imputer",
            "preprocessing.imputation.bfill",
            "preprocessing.scaling.static_minmax_scaler",
            "preprocessing.scaling.ts_minmax_scaler",
            "classification.nn_classifier",
        ],
        [
            "preprocessing.imputation.bfill",
            "preprocessing.scaling.ts_minmax_scaler",
            "regression.nn_regressor",
        ],
        [
            "preprocessing.imputation.ffill",
            "preprocessing.scaling.static_minmax_scaler",
            "preprocessing.scaling.ts_minmax_scaler",
            "regression.nn_regressor",
        ],
        [
            "preprocessing.imputation.ffill",
            "regression.nn_regressor",
        ],
        [
            "classification.nn_classifier",
        ],
    ],
)
@pytest.mark.parametrize("serialize", [True, False])
def test_pipeline_end2end(plugins_str, serialize: bool) -> None:
    if len(plugins_str) > 1:
        dataset = SineDataLoader(with_missing=True).load()
    else:
        dataset = SineDataLoader(with_missing=False).load()

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
