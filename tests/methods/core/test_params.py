from typing import Any, List
from unittest.mock import MagicMock, Mock

import pytest
from optuna.trial import Trial

from tempor.methods.core import _params as params


class TestParams:
    def test_basic_functionality(self):
        get_mock = MagicMock()
        optuna_sample = Mock()
        default_sample = Mock()

        class MyParams(params.Params):
            def get(self) -> List[Any]:
                return get_mock

            def _sample_optuna_trial(self, trial: Trial) -> Any:
                return optuna_sample

            def _sample_default(self) -> Any:
                return default_sample

        my_params = MyParams(name="my_params", bounds=(0, 100))

        assert my_params.name == "my_params"
        assert my_params.bounds == (0, 100)

        assert my_params.get() == get_mock
        assert my_params.sample() == default_sample
        assert my_params.sample(trial=Mock()) == optuna_sample

        assert "MyParams(" in str(my_params)
        assert "name=" in str(my_params)
        assert "bounds=" in str(my_params)

    def test_init_fails_reserved_name(self):
        class MyParams(params.Params):
            def get(self) -> List[Any]:
                return Mock()

            def _sample_optuna_trial(self, trial: Trial) -> Any:
                return Mock()

            def _sample_default(self) -> Any:
                return Mock()

        with pytest.raises(ValueError, match=".*special.*"):
            MyParams(name="trail", bounds=(0, 100))
        with pytest.raises(ValueError, match=".*special.*"):
            MyParams(name="override", bounds=(0, 100))

        for special in params.RESERVED_ARG_NAMES:
            with pytest.raises(ValueError, match=".*special.*"):
                MyParams(name=special, bounds=(0, 100))


class TestCategoricalParams:
    def test_basic_functionality(self):
        cat_params = params.CategoricalParams(name="my_cat", choices=["a", "b", "c"])

        # Init basics:

        assert cat_params.name == "my_cat"
        assert cat_params.choices == ["a", "b", "c"]
        assert cat_params.bounds == ("a", "c")

        assert cat_params.get() == ["my_cat", ["a", "b", "c"]]

        # Sampling:

        trial = Mock(Trial)
        mock_suggest_categorical = Mock()

        def suggest_categorical(name, choices):
            mock_suggest_categorical(name=name, choices=choices)
            return "optuna_suggestion"

        trial.suggest_categorical = suggest_categorical

        optuna_sample = cat_params.sample(trial=trial)

        mock_suggest_categorical.assert_called_once_with(name="my_cat", choices=["a", "b", "c"])
        assert optuna_sample == "optuna_suggestion"

        for _ in range(100):
            assert cat_params.sample() in ["a", "b", "c"]

        # Repr:
        assert "CategoricalParams(" in str(cat_params)
        assert "name=" in str(cat_params)
        assert "choices=" in str(cat_params)


class TestFloatParams:
    def test_basic_functionality(self):
        float_params = params.FloatParams(name="my_float", low=1.5, high=2.5)

        # Init basics:

        assert float_params.name == "my_float"
        assert float_params.bounds == (1.5, 2.5)
        assert float_params.low == 1.5
        assert float_params.high == 2.5

        assert float_params.get() == ["my_float", 1.5, 2.5]

        # Sampling:

        trial = Mock(Trial)
        mock_suggest_float = Mock()

        def suggest_float(name, low, high):
            mock_suggest_float(name=name, low=low, high=high)
            return "optuna_suggestion"

        trial.suggest_float = suggest_float

        optuna_sample = float_params.sample(trial=trial)

        mock_suggest_float.assert_called_once_with(name="my_float", low=1.5, high=2.5)
        assert optuna_sample == "optuna_suggestion"

        for _ in range(100):
            assert 1.5 <= float_params.sample() <= 2.5

        # Repr:
        assert "FloatParams(" in str(float_params)
        assert "name=" in str(float_params)
        assert "low=1.5" in str(float_params)
        assert "high=2.5" in str(float_params)


class TestIntegerParams:
    def test_basic_functionality(self):
        int_params = params.IntegerParams(name="my_int", low=1, high=5, step=2)

        # Init basics:

        assert int_params.name == "my_int"
        assert int_params.bounds == (1, 5)
        assert int_params.low == 1
        assert int_params.high == 5
        assert int_params.step == 2

        assert int_params.choices == [1, 3, 5]

        assert int_params.get() == ["my_int", 1, 5, 2]

        # Sampling:

        trial = Mock(Trial)
        mock_suggest_int = Mock()

        def suggest_int(name, low, high, step):
            mock_suggest_int(name=name, low=low, high=high, step=step)
            return "optuna_suggestion"

        trial.suggest_int = suggest_int

        optuna_sample = int_params.sample(trial=trial)

        mock_suggest_int.assert_called_once_with(name="my_int", low=1, high=5, step=2)
        assert optuna_sample == "optuna_suggestion"

        for _ in range(100):
            assert int_params.sample() in [1, 3, 5]

        # Repr:
        assert "IntegerParams(" in str(int_params)
        assert "name=" in str(int_params)
        assert "low=1" in str(int_params)
        assert "high=5" in str(int_params)
        assert "step=2" in str(int_params)
