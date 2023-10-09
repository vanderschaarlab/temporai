from unittest.mock import Mock

import pytest

from tempor.exc import UnsupportedSetupException
from tempor.methods.time_to_event.helper_embedding import DDHEmbedding, DynamicDeepHitModel


def test_merge_data_no_static(pbc_data_full):
    emb = DDHEmbedding(emb_model=DynamicDeepHitModel())
    data = pbc_data_full
    (_, temporal, observation_times, *_) = emb._convert_data(data)  # pylint: disable=protected-access
    emb._merge_data(None, temporal, observation_times)  # pylint: disable=protected-access


def test_convert_data_no_static_no_targets(pbc_data_full):
    emb = DDHEmbedding(emb_model=DynamicDeepHitModel())
    data = pbc_data_full
    data._validate = Mock()  # pylint: disable=protected-access
    data.static = None
    data.predictive.targets = None
    (static, _, _, event_times, event_values) = emb._convert_data(data)  # pylint: disable=protected-access
    assert event_times is None
    assert event_values is None
    assert static.shape[1] == 0  # type: ignore


def test_validate_data(pbc_data_full):
    emb = DDHEmbedding(emb_model=DynamicDeepHitModel())
    data = pbc_data_full

    data._validate = Mock()  # pylint: disable=protected-access
    data.predictive.targets = Mock(return_value=100, num_features=100)

    with pytest.raises(UnsupportedSetupException):
        emb._validate_data(data)  # pylint: disable=protected-access
