# pylint: disable=protected-access

from unittest.mock import Mock

import numpy as np
import pytest

from tempor.benchmarks import metrics


def test_validation_check_y_survival():
    with pytest.raises(ValueError, match=".*y must be.*"):
        metrics.check_y_survival(y_or_event="wrong")  # type: ignore

    with pytest.raises(ValueError, match=".*boolean.*"):
        metrics.check_y_survival(
            y_or_event=np.asarray([[1, 3, 7], [1, 1, 1]], dtype={"names": ("a", "b"), "formats": ("int", "f8")})
        )

    with pytest.raises(ValueError, match=".*all.*censored.*"):
        metrics.check_y_survival(
            y_or_event=np.asarray(
                [[False, False, False], [0, 0, 0]], dtype={"names": ("a", "b"), "formats": ("bool", "f8")}
            ),
            allow_all_censored=False,
        )


def test_validation_check_estimate_1d():
    with pytest.raises(ValueError, match=".*Expected 1D.*"):
        metrics._check_estimate_1d([[1, 1], [2, 2]], test_time=100)


def test_validation_check_times():
    with pytest.raises(ValueError, match=".*within.*"):
        metrics._check_times(np.asarray([1000]), [1, 4])


def test_validation_check_estimate_2d():
    with pytest.raises(ValueError, match=".*estimate with.*"):
        metrics._check_estimate_2d(np.asarray([[1, 2], [1, 2]]), np.asarray([1, 3]), np.asarray([1]), Mock())


def test_compute_counts():
    out = metrics._compute_counts(np.asarray([True, True, False]), np.asarray([1, 8, 9]))
    assert len(out) == 4


def test_compute_counts_order_given():
    out = metrics._compute_counts(np.asarray([True, True, False]), np.asarray([1, 8, 9]), order=np.asarray([2, 1, 0]))
    assert len(out) == 4


def test_compute_counts_truncated_validation():
    with pytest.raises(ValueError, match=".*must be larger.*"):
        metrics._compute_counts_truncated(
            np.asarray([True, True, False]), np.asarray([11, 11, 11]), np.asarray([1, 1, 1])
        )


def test_kaplan_meier_estimator_reverse():
    uniq_times, y = metrics.kaplan_meier_estimator(
        event=np.asarray([True, True, False]),
        time_exit=np.asarray([1, 8, 9]),
        reverse=True,
    )
    assert len(uniq_times) == 3
    assert len(y) == 3


def test_kaplan_meier_estimator_time_enter():
    uniq_times, y = metrics.kaplan_meier_estimator(
        event=np.asarray([True, True, False]),
        time_exit=np.asarray([1, 8, 9]),
        time_enter=np.asarray([0, 1, 0]),
    )
    assert len(uniq_times) == 4
    assert len(y) == 4

    with pytest.raises(ValueError, match=".*left.*"):
        metrics.kaplan_meier_estimator(
            event=np.asarray([True, True, False]),
            time_exit=np.asarray([1, 8, 9]),
            time_enter=np.asarray([0, 1, 0]),
            reverse=True,
        )


def test_kaplan_meier_estimator_time_min():
    uniq_times, y = metrics.kaplan_meier_estimator(
        event=np.asarray([True, True, False]),
        time_exit=np.asarray([1, 8, 9]),
        time_min=0,
    )
    assert len(uniq_times) == 3
    assert len(y) == 3


def test_survival_function_estimator_fit():
    model = metrics.SurvivalFunctionEstimator()
    model.fit(
        y=np.asarray(
            list(zip([False, False, True, True], [1, 2, 11, 4])), dtype={"names": ("e", "t"), "formats": ("bool", "f8")}
        )
    )


def test_survival_function_estimator_predict_proba_invalid():
    model = metrics.SurvivalFunctionEstimator()
    model.fit(
        y=np.asarray(
            list(zip([False, False, True, True], [1, 2, 11, 4])), dtype={"names": ("e", "t"), "formats": ("bool", "f8")}
        )
    )
    model.prob_[-1] = 1
    with pytest.raises(ValueError, match=".*smaller than largest.*"):
        model.predict_proba(np.asarray([12, 13]))


def test_censoring_distribution_estimator_fit_all():
    model = metrics.CensoringDistributionEstimator()
    model.fit(
        y=np.asarray(
            list(zip([True, True, True, True], [1, 2, 11, 4])), dtype={"names": ("e", "t"), "formats": ("bool", "f8")}
        )
    )


def test_censoring_distribution_estimator_predict_ipcw_ghat_0(monkeypatch):
    model = metrics.CensoringDistributionEstimator()
    monkeypatch.setattr(model, "predict_proba", Mock(return_value=np.asarray([0, 0, 0, 0])))

    model.fit(
        y=np.asarray(
            list(zip([True, True, True, True], [1, 2, 11, 4])), dtype={"names": ("e", "t"), "formats": ("bool", "f8")}
        )
    )
    with pytest.raises(ValueError, match=".*is zero.*"):
        model.predict_ipcw(
            y=np.asarray(
                list(zip([True, True, True, True], [1, 2, 11, 4])),
                dtype={"names": ("e", "t"), "formats": ("bool", "f8")},
            )
        )


def test_estimate_concordance_index_invalid_comparable(monkeypatch):
    monkeypatch.setattr(metrics, "_get_comparable", Mock(return_value=([], [])))

    with pytest.raises(ValueError, match=".*pairs.*"):
        metrics._estimate_concordance_index(Mock(), Mock(), Mock(), Mock())


def test_create_structured_array_validation():
    with pytest.raises(ValueError, match=".*different.*"):
        metrics.create_structured_array([], [], name_event="a", name_time="a")


def test_create_structured_array_non_bool_event():
    y = metrics.create_structured_array([1, 1, 0, 1], [22, 34, 22, 134])
    assert len(y) == 4


def test_create_structured_array_non_bool_event_validation():
    with pytest.raises(ValueError, match=".*binary.*"):
        metrics.create_structured_array([1, 22, 1111, 44544], [22, 34, 22, 134])
    with pytest.raises(ValueError, match=".*0 and 1.*"):
        metrics.create_structured_array([3, 4, 3, 4], [22, 34, 22, 134])


def test_brier_score_reshape():
    out = metrics.brier_score(
        survival_train=np.asarray(
            list(zip([True, True, True, True], [1, 2, 11, 4])),
            dtype={"names": ("e", "t"), "formats": ("bool", "f8")},
        ),
        survival_test=np.asarray(
            list(zip([True, True], [1, 4])),
            dtype={"names": ("e", "t"), "formats": ("bool", "f8")},
        ),
        estimate=np.asarray([0.7, 0.9]),
        times=[3],
    )
    assert len(out) == 2


def test_concordance_index_ipcw_tau_none():
    out = metrics.concordance_index_ipcw(
        survival_train=np.asarray(
            list(zip([True, True, True, True], [1, 2, 11, 4])),
            dtype={"names": ("e", "t"), "formats": ("bool", "f8")},
        ),
        survival_test=np.asarray(
            list(zip([True, True], [1, 4])),
            dtype={"names": ("e", "t"), "formats": ("bool", "f8")},
        ),
        estimate=np.asarray([0.7, 0.9]),
        tau=None,
    )
    assert len(out) == 5
