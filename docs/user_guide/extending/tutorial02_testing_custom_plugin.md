# Extending TemporAI Tutorial 02: Testing a Custom Plugin
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/extending/tutorial02_testing_custom_plugin.ipynb)

This tutorial gives a brief overview of how to test your custom plugin.

For the basics of writing a custom plugin, see the *"Writing Custom Plugin" tutorial* first. This tutorial assumes you have already written a custom plugin.

*Skip the below cell if you are not on Google Colab / already have TemporAI installed:*


```python
%pip install temporai[dev]
```

**⚙️ Installation for testing**

You should install the `[dev]` TemporAI to run tests:
```bash
pip install temporai[dev]
```

Alternatively, the best way to install TemporAI for development is to clone the repo, checkout a branch, and install in editable mode:
```bash
git clone https://github.com/vanderschaarlab/temporai.git
cd temporai
git checkout -b my-branch-name
pip install -e .[dev]
```

📘 See also the [contribution guide](https://github.com/vanderschaarlab/temporai/blob/main/CONTRIBUTING.md).

## 1. Find and adapt suitable tests

The simplest way to run a set of tests on your custom plugin is to find and adapt existing tests for plugins of the same category.

Plugin categories in TemporAI can be found as below.


```python
from tempor.plugins import plugin_loader
from rich.pretty import pprint  # For prettifying the print output only.

plugin_categories = plugin_loader.list_categories()

print("Plugin categories:")
pprint(list(plugin_categories.keys()), indent_guides=False)

print("All plugins:")
pprint(plugin_loader.list(), indent_guides=True)
```

    Plugin categories:



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">[</span>
    <span style="color: #008000; text-decoration-color: #008000">'prediction.one_off.classification'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'prediction.one_off.regression'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'prediction.temporal.classification'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'prediction.temporal.regression'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'preprocessing.imputation.static'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'preprocessing.imputation.temporal'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'preprocessing.nop'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'preprocessing.scaling.static'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'preprocessing.scaling.temporal'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'time_to_event'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'treatments.one_off.regression'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'treatments.temporal.classification'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'treatments.temporal.regression'</span>
<span style="font-weight: bold">]</span>
</pre>



    All plugins:



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #008000; text-decoration-color: #008000">'prediction'</span>: <span style="font-weight: bold">{</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'one_off'</span>: <span style="font-weight: bold">{</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'classification'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'cde_classifier'</span>, <span style="color: #008000; text-decoration-color: #008000">'ode_classifier'</span>, <span style="color: #008000; text-decoration-color: #008000">'nn_classifier'</span>, <span style="color: #008000; text-decoration-color: #008000">'laplace_ode_classifier'</span><span style="font-weight: bold">]</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'regression'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'laplace_ode_regressor'</span>, <span style="color: #008000; text-decoration-color: #008000">'nn_regressor'</span>, <span style="color: #008000; text-decoration-color: #008000">'ode_regressor'</span>, <span style="color: #008000; text-decoration-color: #008000">'cde_regressor'</span><span style="font-weight: bold">]</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="font-weight: bold">}</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'temporal'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'classification'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'seq2seq_classifier'</span><span style="font-weight: bold">]</span>, <span style="color: #008000; text-decoration-color: #008000">'regression'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'seq2seq_regressor'</span><span style="font-weight: bold">]}</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="font-weight: bold">}</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #008000; text-decoration-color: #008000">'preprocessing'</span>: <span style="font-weight: bold">{</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'imputation'</span>: <span style="font-weight: bold">{</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'static'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'static_tabular_imputer'</span><span style="font-weight: bold">]</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'temporal'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'ffill'</span>, <span style="color: #008000; text-decoration-color: #008000">'ts_tabular_imputer'</span>, <span style="color: #008000; text-decoration-color: #008000">'bfill'</span><span style="font-weight: bold">]</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="font-weight: bold">}</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'nop'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'nop_transformer'</span><span style="font-weight: bold">]</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'scaling'</span>: <span style="font-weight: bold">{</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'static'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'static_minmax_scaler'</span>, <span style="color: #008000; text-decoration-color: #008000">'static_standard_scaler'</span><span style="font-weight: bold">]</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'temporal'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'ts_minmax_scaler'</span>, <span style="color: #008000; text-decoration-color: #008000">'ts_standard_scaler'</span><span style="font-weight: bold">]</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="font-weight: bold">}</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="font-weight: bold">}</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #008000; text-decoration-color: #008000">'time_to_event'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'ts_coxph'</span>, <span style="color: #008000; text-decoration-color: #008000">'ts_xgb'</span>, <span style="color: #008000; text-decoration-color: #008000">'dynamic_deephit'</span><span style="font-weight: bold">]</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #008000; text-decoration-color: #008000">'treatments'</span>: <span style="font-weight: bold">{</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'one_off'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'regression'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'synctwin_regressor'</span><span style="font-weight: bold">]}</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'temporal'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'classification'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'crn_classifier'</span><span style="font-weight: bold">]</span>, <span style="color: #008000; text-decoration-color: #008000">'regression'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'crn_regressor'</span><span style="font-weight: bold">]}</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="font-weight: bold">}</span>
<span style="font-weight: bold">}</span>
</pre>



Suitable tests can be found in TemporAI source code (`tests/plugins/...`), organized hierarchically by plugin category.

For example, tests for `prediction/one_off/classification` will be located under:
* [`tests/plugins/prediction/one_off/classification`](https://github.com/vanderschaarlab/temporai/tree/main/tests/plugins/prediction/one_off/classification).

Find an example test file in this category, and adapt the tests from it to your custom plugin. For instance, we could look at:
* [test file for `nn_classifier` plugin](https://github.com/vanderschaarlab/temporai/blob/main/tests/plugins/prediction/one_off/classification/test_nn_classifier.py).

The test suite differs by plugin category, but in general, the following points are worth noting.

* [`pytest`](https://docs.pytest.org/) is used for testing.
* [Parametrization](https://docs.pytest.org/en/7.3.x/how-to/parametrize.html) is used to run similar tests with different parameters.
* [Fixtures](https://docs.pytest.org/en/7.3.x/how-to/fixtures.html) are used for reusable test elements, e.g. datasets.
* Plugins are tested with two methods of importing them (`PLUGIN_FROM_OPTIONS`): `"from_api"` and `"from_module"`, this is handled by the helper fixture `get_test_plugin`.
* Typically, the set of test functions needed looks something like this:
    * `test_sanity`: the very basics, check loading of your plugin works.
    * `test_fit`: test your plugin's `fit` method.
    * `test_predict, test_transform, ...`: test the additional methods relevant to your plugin's category.
    * `test_serde`: test serialization and deserialization works.
* The below constants are set and reused as parameters in the test suite. Adapt these as necessary. The various dataset fixtures can be found in [`conftest.py`](https://github.com/vanderschaarlab/temporai/blob/main/tests/conftest.py).
    ```python
    INIT_KWARGS = {"random_state": 123, "n_iter": 5}  # Input parameters for the plugin.
    TEST_ON_DATASETS = ["sine_data_small"]  # A list of dataset fixtures to test the plugin on.
    ```

An example test file would looks something like below.

In this example, it is assumed that the plugin is of the `prediction/one_off/classification` category, and is named `"my_classifier"`.

```python
# test_my_classifier.py
# Example for illustration only - adapt to your own plugin as needed.

from typing import Callable, Dict

import pytest

from tempor.plugins.prediction.one_off.classification import BaseOneOffClassifier
from tempor.plugins.prediction.one_off.classification.plugin_my_classifier import MyClassifier
from tempor.utils.serialization import load, save

INIT_KWARGS = {"random_state": 123, "n_iter": 5}
PLUGIN_FROM_OPTIONS = ["from_api", pytest.param("from_module", marks=pytest.mark.extra)]
TEST_ON_DATASETS = ["sine_data_small"]


@pytest.fixture
def get_test_plugin(get_plugin: Callable):
    def func(plugin_from: str, base_kwargs: Dict):
        return get_plugin(
            plugin_from,
            fqn="prediction.one_off.classification.my_classifier",
            cls=MyClassifier,
            kwargs=base_kwargs,
        )

    return func


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
def test_sanity(get_test_plugin: Callable, plugin_from: str) -> None:
    test_plugin = get_test_plugin(plugin_from, INIT_KWARGS)
    assert test_plugin is not None
    assert test_plugin.name == "nn_classifier"
    assert test_plugin.fqn() == "prediction.one_off.classification.my_classifier"
    assert len(test_plugin.hyperparameter_space()) == 9


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_fit(plugin_from: str, data: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseOneOffClassifier = get_test_plugin(plugin_from, INIT_KWARGS)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)


@pytest.mark.parametrize("plugin_from", PLUGIN_FROM_OPTIONS)
@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_predict(
    plugin_from: str, data: str, no_targets: bool, get_test_plugin: Callable, get_dataset: Callable
) -> None:
    test_plugin: BaseOneOffClassifier = get_test_plugin(plugin_from, INIT_KWARGS)
    dataset = get_dataset(data)
    test_plugin.fit(dataset)
    output = test_plugin.predict(dataset)
    assert output.numpy().shape == (len(dataset.time_series), 1)


# Other categories of plugins would have more / different methods to test.


@pytest.mark.parametrize("data", TEST_ON_DATASETS)
def test_serde(data: str, get_test_plugin: Callable, get_dataset: Callable) -> None:
    test_plugin: BaseOneOffClassifier = get_test_plugin("from_api", INIT_KWARGS)
    dataset = get_dataset(data)

    dump = save(test_plugin)
    reloaded1 = load(dump)

    reloaded1.fit(dataset)

    dump = save(reloaded1)
    reloaded2 = load(dump)

    reloaded2.predict(dataset)

```


In order to run the tests, the test file (e.g. `test_my_classifier.py`) should be placed into the appropriate test directory, e.g. in this example under:

[`tests/plugins/prediction/one_off/classification`](https://github.com/vanderschaarlab/temporai/tree/main/tests/plugins/prediction/one_off/classification).

The tests can the be run like so:
```bash
pytest -x tests/plugins/prediction/one_off/classification/test_my_classifier.py
```

## 2. Plugin loader and all-plugin tests

Some additional tests that apply to your plugin are found in:
* [`tests/plugins/test_plugin_loader.py`](https://github.com/vanderschaarlab/temporai/blob/main/tests/plugins/test_plugin_loader.py): The plugin registry checks.
* [`tests/plugins/test_all_plugins.py`](https://github.com/vanderschaarlab/temporai/blob/main/tests/plugins/test_all_plugins.py): The common automatic basic tests for all plugins.

In `test_plugin_loader.py`, your custom plugin should be added to `test_tempor_plugin_loader_contents`, e.g.
```python
assert "my_classifier" in all_plugins["prediction"]["one_off"]["classification"]
```

`test_all_plugins.py` tests will run automatically, no changes here are needed.

To check that these two test files pass:
```bash
pytest -x tests/plugins/test_plugin_loader.py
pytest -x tests/plugins/test_all_plugins.py
```

## 3. Finally...

Now is a perfect time to contribute your awesome plugin to the open sources eco-system by submitting a PR to TemporAI!

Please follow the [contribution guide](https://github.com/vanderschaarlab/temporai/blob/main/CONTRIBUTING.md).

