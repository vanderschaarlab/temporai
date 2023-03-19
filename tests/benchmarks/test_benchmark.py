from tempor.benchmarks import (
    benchmark_models,
    classifier_supported_metrics,
    regression_supported_metrics,
)
from tempor.plugins import plugin_loader
from tempor.plugins.pipeline import Pipeline
from tempor.utils.datasets.google_stocks import GoogleStocksDataLoader
from tempor.utils.datasets.sine import SineDataLoader


def test_classifier_benchmark() -> None:
    testcases = [
        (
            "pipeline1",
            Pipeline(
                [
                    "preprocessing.imputation.ffill",
                    "classification.nn_classifier",
                ]
            )({"nn_classifier": {"n_iter": 10}}),
        ),
        (
            "plugin1",
            plugin_loader.get("classification.nn_classifier", n_iter=10),
        ),
    ]
    dataset = SineDataLoader().load()

    aggr_score, per_test_score = benchmark_models(
        task_type="classification", tests=testcases, data=dataset, n_splits=2, random_state=0
    )

    for testcase, _ in testcases:
        assert testcase in aggr_score.columns
        assert testcase in per_test_score

    for metric in classifier_supported_metrics:
        assert metric in aggr_score.index

        for testcase, _ in testcases:
            assert metric in per_test_score[testcase].index


def test_regressor_benchmark() -> None:
    testcases = [
        (
            "pipeline1",
            Pipeline(
                [
                    "preprocessing.imputation.ffill",
                    "regression.nn_regressor",
                ]
            )({"nn_regressor": {"n_iter": 10}}),
        ),
        (
            "plugin1",
            plugin_loader.get("regression.nn_regressor", n_iter=10),
        ),
    ]
    dataset = GoogleStocksDataLoader().load()

    aggr_score, per_test_score = benchmark_models(
        task_type="regression", tests=testcases, data=dataset, n_splits=2, random_state=0
    )

    for testcase, _ in testcases:
        assert testcase in aggr_score.columns
        assert testcase in per_test_score

    for metric in regression_supported_metrics:
        assert metric in aggr_score.index

        for testcase, _ in testcases:
            assert metric in per_test_score[testcase].index
