# stdlib
from typing import Any, List

# third party
import pytest

from tempor.plugins.pipeline import Pipeline, PipelineGroup
from tempor.utils.datasets.sine import SineDataloader


@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            "preprocessing.imputation.bfill",
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
    dtype = Pipeline(plugins_str)
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
            "preprocessing.imputation.bfill",
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
            "classification.nn_classifier",
        ],
        [
            "preprocessing.imputation.bfill",
            "preprocessing.scaling.nop_scaler",
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
def test_pipeline_end2end(plugins_str) -> None:
    if len(plugins_str) > 1:
        dataset = SineDataloader(with_missing=True).load()
    else:
        dataset = SineDataloader(with_missing=False).load()

    template = Pipeline(plugins_str)
    pipeline = template()

    pipeline.fit(dataset)

    y_pred = pipeline.predict(dataset)

    assert y_pred.dataframe().shape == (len(dataset.predictive.targets.dataframe()), 1)
