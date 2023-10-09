# pylint: disable=no-member

import re
from typing import Any, List, Type

import pytest

from tempor.methods.pipeline import PipelineBase, pipeline, pipeline_classes


@pytest.mark.filterwarnings("ignore:.*validate_arguments.*:DeprecationWarning")  # Exp pydantic2 warns from HI.
@pytest.mark.filterwarnings("ignore:.*conflict.*:UserWarning")  # Exp pydantic2 warns from HI.
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
def test_sanity(plugins_str: List[Any]):
    PipelineCls: Type[PipelineBase] = pipeline(plugins_str)
    plugin_classes = pipeline_classes(plugins_str)

    assert issubclass(PipelineCls, PipelineBase)
    assert PipelineCls.pipeline_seq() == "->".join(p for p in plugins_str)
    assert list(PipelineCls.plugin_types) == list(plugin_classes)

    args = {"features_count": 10}
    for act, pl in zip(PipelineCls.hyperparameter_space(**args), plugin_classes):
        assert len(PipelineCls.hyperparameter_space(**args)[act]) == len(pl.hyperparameter_space(**args))

    pipe = PipelineCls()
    hps = pipe.sample_hyperparameters()

    assert len(pipe.stages) == len(plugin_classes)
    assert pipe.predictor_category == plugin_classes[-1].category
    assert len(pipe.params) == len(plugin_classes)
    for stage in pipe.stages:
        assert pipe.params[stage.name] == stage.params
    assert len(hps) == len(plugin_classes)
    for stage in pipe.stages:
        assert stage.name in hps


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
        ["invalid_fqn"],
        [],
    ],
)
def test_fails(plugins_str: List[Any]):
    with pytest.raises(RuntimeError):
        pipeline(plugins_str)()


def test_hyperparameter_space_for_step():
    plugins_str = [
        "preprocessing.imputation.temporal.bfill",
        "prediction.one_off.classification.nn_classifier",
    ]
    PipelineCls = pipeline(plugins_str)

    hps = PipelineCls.hyperparameter_space_for_step("nn_classifier")

    assert len(hps) > 0


def test_hyperparameter_space_for_step_fails():
    plugins_str = [
        "preprocessing.imputation.temporal.bfill",
        "prediction.one_off.classification.nn_classifier",
    ]
    PipelineCls = pipeline(plugins_str)

    with pytest.raises(ValueError, match="Invalid layer.*"):
        PipelineCls.hyperparameter_space_for_step("non_existent_step")


def test_repr():
    plugins_str = [
        "preprocessing.imputation.temporal.bfill",
        "preprocessing.scaling.static.static_minmax_scaler",
        "preprocessing.scaling.temporal.ts_minmax_scaler",
        "prediction.one_off.classification.nn_classifier",
    ]

    PipelineCls = pipeline(plugins_str)

    pipe = PipelineCls()
    repr_ = repr(pipe)

    seq = pipe.pipeline_seq()
    assert re.search(
        r"^Pipeline\(.*pipeline_seq='" + seq + r"'.*predictor_category='prediction.one_off.classification'"
        r".*params=.?\{.*'bfill':.*'static_minmax_scaler':.*'ts_minmax_scaler':.*'nn_classifier':.*\}.*\)",
        repr_,
        re.S,
    )
