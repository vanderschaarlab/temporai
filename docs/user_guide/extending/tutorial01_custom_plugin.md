# Extending TemporAI Tutorial 01: Writing a Custom Plugin
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/extending/tutorial01_custom_plugin.ipynb)

This tutorial shows how to extend TemporAI by wring a custom plugin.



## Writing a Custom `Plugin` 101

In order to write a custom plugin for TemporAI, you need to do the following:
1. Inherit from the appropriate **base class** for the type of plugin you are writing.
2. Implement the **methods** the plugin needs.
3. **Register** the plugin with TemporAI.

We will go through an example in this tutorial.

### 1. Inherit from the appropriate **base class** for the type of plugin you are writing.

You need to find which type (category) of plugin you are writing.

A summary of different plugin categories is available in the 
[README](https://github.com/vanderschaarlab/temporai/blob/main/README.md#-methods).

You can also view all the different plugin categories as so:


```python
from tempor.plugins import plugin_loader

plugin_categories = plugin_loader.list_categories()

list(plugin_categories.keys())
```




    ['prediction.one_off.classification',
     'prediction.one_off.regression',
     'prediction.temporal.classification',
     'prediction.temporal.regression',
     'preprocessing.imputation.static',
     'preprocessing.imputation.temporal',
     'preprocessing.nop',
     'preprocessing.scaling.static',
     'preprocessing.scaling.temporal',
     'time_to_event',
     'treatments.one_off.regression',
     'treatments.temporal.classification',
     'treatments.temporal.regression']



Remember you can also see the existing plugins and how they correspond to different categories, as follows:


```python
all_plugins = plugin_loader.list()

from rich.pretty import pprint  # For prettifying the print output only.

pprint(all_plugins, indent_guides=True)
```


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



Let's say you would like to write a plugin of category `"prediction.one_off.classification"`.

You can find which base class you need to inherit from as follows.


```python
plugin_categories = plugin_loader.list_categories()

print("Base classes for all categories:")
pprint(plugin_categories, indent_guides=False)

print("Base class you need:")
print(plugin_categories["prediction.one_off.classification"])
```

    Base classes for all categories:



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'prediction.one_off.classification'</span>: <span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">class</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.prediction.one_off.classification.BaseOneOffClassifier'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #008000; text-decoration-color: #008000">'prediction.one_off.regression'</span><span style="color: #000000; text-decoration-color: #000000">: &lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.prediction.one_off.regression.BaseOneOffRegressor'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #008000; text-decoration-color: #008000">'prediction.temporal.classification'</span><span style="color: #000000; text-decoration-color: #000000">: &lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.prediction.temporal.classification.BaseTemporalClassifier'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #008000; text-decoration-color: #008000">'prediction.temporal.regression'</span><span style="color: #000000; text-decoration-color: #000000">: &lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.prediction.temporal.regression.BaseTemporalRegressor'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #008000; text-decoration-color: #008000">'preprocessing.imputation.static'</span><span style="color: #000000; text-decoration-color: #000000">: &lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.preprocessing.imputation._base.BaseImputer'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #008000; text-decoration-color: #008000">'preprocessing.imputation.temporal'</span><span style="color: #000000; text-decoration-color: #000000">: &lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.preprocessing.imputation._base.BaseImputer'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #008000; text-decoration-color: #008000">'preprocessing.nop'</span><span style="color: #000000; text-decoration-color: #000000">: &lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.core._base_transformer.BaseTransformer'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #008000; text-decoration-color: #008000">'preprocessing.scaling.static'</span><span style="color: #000000; text-decoration-color: #000000">: &lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.preprocessing.scaling._base.BaseScaler'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #008000; text-decoration-color: #008000">'preprocessing.scaling.temporal'</span><span style="color: #000000; text-decoration-color: #000000">: &lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.preprocessing.scaling._base.BaseScaler'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #008000; text-decoration-color: #008000">'time_to_event'</span><span style="color: #000000; text-decoration-color: #000000">: &lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.time_to_event.BaseTimeToEventAnalysis'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #008000; text-decoration-color: #008000">'treatments.one_off.regression'</span><span style="color: #000000; text-decoration-color: #000000">: &lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.treatments.one_off._base.BaseOneOffTreatmentEffects'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #008000; text-decoration-color: #008000">'treatments.temporal.classification'</span><span style="color: #000000; text-decoration-color: #000000">: &lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.treatments.temporal._base.BaseTemporalTreatmentEffects'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #000000; text-decoration-color: #000000">    </span><span style="color: #008000; text-decoration-color: #008000">'treatments.temporal.regression'</span><span style="color: #000000; text-decoration-color: #000000">: &lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.treatments.temporal._base.BaseTemporalTreatmentEffects'</span><span style="font-weight: bold">&gt;</span>
<span style="font-weight: bold">}</span>
</pre>



    Base class you need:
    <class 'tempor.plugins.prediction.one_off.classification.BaseOneOffClassifier'>


You can then find the class in the TemporAI source code, to see its method signatures etc.

### 2. Implement the **methods** the plugin needs.

Different category plugins have different methods that need to be implemented, but the key methods are:
* `_fit()` where you provide your implementation of the fitting (training).
* `_predict()` where you provide your implementation of the prediction (inference).
* `_transform()` where you provide your implementation of data transformation (for preprocessing plugins).

Classification-related plugins also have `_predict_proba()` and treatment effects plugins have `_predict_counterfactuals()`.

Note that these methods have a preceding underscore `_`, and are different from the corresponding "public" methods
without the underscore (e.g `fit()`). When extending, you need to implement the `_<...>` method,
and the corresponding "public" method in TemporAI is what the user of your plugin will call.
The "public" methods also do various necessary validation and other checks behind the scenes.

If you haven't implemented some required method for the plugin, Python will notify you by raising an exception when you
attempt to instantiate your plugin (see [Python `abc`](https://docs.python.org/3/library/abc.html)).


In our example case, you will need to implement the following methods for `BaseOneOffClassifier`:

```python
from tempor.plugins.prediction.one_off.classification import BaseOneOffClassifier

class MyPlugin(BaseOneOffClassifier):
    # The initializer:
    def __init__(self, **params) -> None:
        ...

    # The _fit implementation.
    def _fit(self, data: dataset.BaseDataset, *args, **kwargs):
        ...

    def _predict(self, data: dataset.PredictiveDataset, *args, **kwargs) -> samples.StaticSamples:
        ...

    def _predict_proba(self, data: dataset.PredictiveDataset, *args, **kwargs) -> samples.StaticSamples:
        ...
    
    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        # This method is not currently used in TemporAI (it will be used once AutoML component is implemented).
        # For now, you may just return an empty list.
        ...
``` 

### 3. **Register** the plugin with TemporAI.

Registering your plugin with TemporAI is very simple, you need to use the `register_plugin` decorator,
as shown in the example below.

You will need to specify the `name` of your plugin and its `category` in the decorator.

```python
from tempor.plugins.core import register_plugin

@register_plugin(name="my_plugin", category="prediction.one_off.classification")
class MyPlugin(BaseOneOffClassifier):
    ...
```

### Note on `__init__` parameters (arguments)

You will also need to define the input parameters (arguments) that will be passed into your plugin's `__init__` in the
following way:

```python
import dataclasses

# 1. Write dataclass with your __init__ parameters:
@dataclasses.dataclass
class MyPluginParams:
    # Specify the parameter, data type and default value as below:
    lr: float = 0.001
    batch_size: int = 100

class MyPlugin(BaseOneOffClassifier):
    # 2. Set the `ParamsDefinition` class variable in your plugin to this dataclass.
    ParamsDefinition = MyPluginParams
    
    def __init__(self, **params) -> None:
        # 3. Call the parent __init__ as so.
        super().__init__(**params)

        # 4. You will now be able to access these in your class like so:
        print(self.params.lr)
        print(self.params.batch_size) 


# 5. The user will then be able to specify the arguments as necessary when initializing your plugin:
model = MyPlugin(batch_size=22)
```


### Putting it all together

Now putting this together in an example of a one-off classifier plugin that always returns `1`s.


```python
import dataclasses

import numpy as np

from tempor.plugins.core import register_plugin
from tempor.data import dataset, samples
from tempor.plugins.prediction.one_off.classification import BaseOneOffClassifier


@dataclasses.dataclass
class MyClassifierParams:
    some_parameter: int = 1
    other_parameter: float = 0.5


@register_plugin(name="my_classifier", category="prediction.one_off.classification")
class MyClassifierClassifier(BaseOneOffClassifier):
    ParamsDefinition = MyClassifierParams

    def __init__(self, **param) -> None:
        super().__init__(**param)

    def _fit(self, data: dataset.BaseDataset, *args, **kwargs):
        """Does nothing."""
        return self  # Fit method needs to return `self`.

    def _predict(self, data: dataset.PredictiveDataset, *args, **kwargs) -> samples.StaticSamples:
        """Always returns 1"""

        assert data.predictive.targets is not None
        preds = np.ones_like(data.predictive.targets.numpy())

        return samples.StaticSamples.from_numpy(preds, dtype=int)

    def _predict_proba(self, data: dataset.PredictiveDataset, *args, **kwargs) -> samples.StaticSamples:
        """Always returns 1.0"""

        assert data.predictive.targets is not None
        preds = np.ones_like(data.predictive.targets.numpy())

        return samples.StaticSamples.from_numpy(preds, dtype=float)

    @staticmethod
    def hyperparameter_space(*args, **kwargs):
        return []
```

We now see our plugin in TemporAI:


```python
from tempor.plugins import plugin_loader

all_plugins = plugin_loader.list()

pprint(all_plugins, indent_guides=True)

my_classifier_found = "my_classifier" in all_plugins["prediction"]["one_off"]["classification"]
print(f"`my_classifier` plugin found in the category 'prediction.one_off.classification': {my_classifier_found}")
assert my_classifier_found
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #008000; text-decoration-color: #008000">'prediction'</span>: <span style="font-weight: bold">{</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'one_off'</span>: <span style="font-weight: bold">{</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'classification'</span>: <span style="font-weight: bold">[</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'cde_classifier'</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'ode_classifier'</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'nn_classifier'</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'laplace_ode_classifier'</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'my_classifier'</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="font-weight: bold">]</span>,
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



    `my_classifier` plugin found in the category 'prediction.one_off.classification': True


The plugin can be used as normal.


```python
# Get the plugin.

my_classifier = plugin_loader.get("prediction.one_off.classification.my_classifier")

print(my_classifier)
```

    MyClassifierClassifier(
        name='my_classifier',
        category='prediction.one_off.classification',
        params={'some_parameter': 1, 'other_parameter': 0.5}
    )



```python
# Fit and predict on some data.

from tempor.utils.dataloaders import SineDataLoader

dataset = SineDataLoader(random_state=42).load()

my_classifier.fit(dataset)

print("Prediction:")
my_classifier.predict(dataset)
```

    Prediction:





<p><span style="font-family: monospace;">StaticSamples</span> with data:</p><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feat_0</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>1</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 1 columns</p>
</div>



