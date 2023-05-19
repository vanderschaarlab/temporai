from unittest.mock import Mock

import numpy as np
import pytest

from tempor.benchmarks import utils


def test_get_y_pred_proba_hlpr_edge():
    mock = Mock()
    out = utils.get_y_pred_proba_hlpr(
        y_pred_proba=mock,
        nclasses=3,
    )
    assert out == mock

    mock = Mock(shape=[1])
    out = utils.get_y_pred_proba_hlpr(
        y_pred_proba=mock,
        nclasses=2,
    )
    assert out == mock

    mock = Mock(shape=[1, 4])
    out = utils.get_y_pred_proba_hlpr(
        y_pred_proba=mock,
        nclasses=2,
    )
    assert out == mock


def test_evaluate_auc_multiclass_nans():
    with pytest.raises(ValueError, match=".*nan.*"):
        utils.evaluate_auc_multiclass(y_test=np.asarray([1, 1, 1]), y_pred_proba=np.asarray([0.9, np.nan, 0.9]))


def test_evaluate_auc_multiclass():
    out = utils.evaluate_auc_multiclass(
        y_test=np.asarray([0, 2, 1]), y_pred_proba=np.asarray([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    )
    assert len(out) == 2
