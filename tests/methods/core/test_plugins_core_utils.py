import pytest

import tempor.methods.core.utils as utils


@pytest.mark.parametrize(
    "initial_dict, key_path, final_dict",
    [
        (
            {"a": 1},
            "b",
            {"a": 1, "b": 99},
        ),
        (
            {"a": 1},
            "b.b_1",
            {"a": 1, "b": {"b_1": 99}},
        ),
        (
            {"a": 1, "b": {"b_1": 2}},
            "b.b_2",
            {"a": 1, "b": {"b_1": 2, "b_2": 99}},
        ),
        (
            {"a": 1, "b": {"b_1": 2, "b_2": 3}},
            "b.b_2",
            {"a": 1, "b": {"b_1": 2, "b_2": 99}},
        ),
    ],
)
def test_add_by_dot_path(initial_dict, key_path, final_dict):
    value_to_add = 99
    dictionary = utils.add_by_dot_path(initial_dict, key_path, value_to_add)
    assert dictionary == final_dict


@pytest.mark.parametrize(
    "dictionary, key_path",
    [
        (
            {"a": 99},
            "a",
        ),
        (
            {"a": 1, "b": {"b_1": 99}},
            "b.b_1",
        ),
    ],
)
def test_get_by_dot_path(dictionary, key_path):
    assert utils.get_by_dot_path(dictionary, key_path) == 99


@pytest.mark.parametrize(
    "initial_dict, key_path, final_dict",
    [
        (
            {"a": [1], "b": {"b_1": [2]}},
            "b.b_2",
            {"a": [1], "b": {"b_1": [2], "b_2": [99]}},
        ),
        (
            {"a": [1], "b": {"b_1": [2]}},
            "b.b_1",
            {"a": [1], "b": {"b_1": [2, 99]}},
        ),
    ],
)
def test_append_by_dot_path(initial_dict, key_path, final_dict):
    value_to_add = 99
    dictionary = utils.append_by_dot_path(initial_dict, key_path, value_to_add)
    assert dictionary == final_dict
