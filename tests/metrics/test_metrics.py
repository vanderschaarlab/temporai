import re
from typing import Any

import pytest

from tempor.metrics.metric import Metric, MetricDirection


class MyMetric(Metric):
    """My metric under test."""

    name = "my_metric"
    category = "my_category"
    plugin_type = "metric"

    @property
    def direction(self) -> MetricDirection:
        return "maximize"

    def _evaluate(self, actual: Any, predicted: Any, *args: Any, **kwargs: Any) -> Any:
        return 0.5


class TestMetric:
    def test_init_success(self):
        metric = MyMetric()
        assert metric.name == "my_metric"
        assert metric.direction == "maximize"

    def test_evaluate_success(self):
        metric = MyMetric()
        actual = [1.0, 2.0, 3.0]
        predicted = [1.0, 2.0, 3.0]
        score = metric.evaluate(actual=actual, predicted=predicted)
        assert score == 0.5

    def test_evaluate_via_call_success(self):
        metric = MyMetric()
        actual = [1.0, 2.0, 3.0]
        predicted = [1.0, 2.0, 3.0]
        score = metric(actual=actual, predicted=predicted)
        assert score == 0.5

    def test_validation_fails(self):
        metric = MyMetric()
        with pytest.raises(ValueError):
            metric.evaluate(actual=None, predicted=[1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            metric.evaluate(actual=[1.0, 2.0, 3.0], predicted=None)

    def test_repr(self):
        metric = MyMetric()
        assert re.match(r"MyMetric\(name='my_metric', description='My metric under test.'\)", repr(metric)) is not None
