# User Guide Tutorial 01: Plugins
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/usage/tutorial01_plugins.ipynb)

This tutorial shows how to load TemporAI estimators (a.k.a. models), which are `Plugin`s.



## Loading a `Plugin`

All estimators (a.k.a. models) in TemporAI are implemented as `Plugin`s, for ease of extending the library.

Each `Plugin` has two plugin-specific attributes: its `name` and `category`.

You can load a plugin in two ways:
* From python module (file),
* From API.

From its python module (file):


```python
# Example of loading the prediction.one_off.classification.nn_classifier plugin from the module:

from tempor.plugins.prediction.one_off.classification.plugin_nn_classifier import NeuralNetClassifier

nn_classifier = NeuralNetClassifier(n_iter=100)

print(f"Plugin class:\n{NeuralNetClassifier}\n")
print(f"Plugin instance:\n{nn_classifier}")
```

    Plugin class:
    <class 'tempor.plugins.prediction.one_off.classification.plugin_nn_classifier.NeuralNetClassifier'>
    
    Plugin instance:
    NeuralNetClassifier(
        name='nn_classifier',
        category='prediction.one_off.classification',
        params={
            'n_static_units_hidden': 100,
            'n_static_layers_hidden': 2,
            'n_temporal_units_hidden': 102,
            'n_temporal_layers_hidden': 2,
            'n_iter': 100,
            'mode': 'RNN',
            'n_iter_print': 10,
            'batch_size': 100,
            'lr': 0.001,
            'weight_decay': 0.001,
            'window_size': 1,
            'device': None,
            'dataloader_sampler': None,
            'dropout': 0.0,
            'nonlin': 'relu',
            'random_state': 0,
            'clipping_value': 1,
            'patience': 20,
            'train_ratio': 0.8
        }
    )


Or from the plugin API, as below.

Note the `tempor.plugins.plugin_loader` object - this allows loading plugins by API.


```python
from tempor.plugins import plugin_loader

# ^ Import the `plugin_loader`.

nn_classifier_cls = plugin_loader.get_class("prediction.one_off.classification.nn_classifier")
# ^ Get the plugin class from API by the fully-qualified plugin name.
#   The fully-qualified plugin name is "<PLUGIN CATEGORY>.<PLUGIN NAME>".

nn_classifier = nn_classifier_cls(n_iter=100)

print(f"Plugin class:\n{nn_classifier_cls}\n")
print(f"Plugin instance:\n{nn_classifier}")
```

    Plugin class:
    <class 'tempor.plugins.prediction.one_off.classification.plugin_nn_classifier.NeuralNetClassifier'>
    
    Plugin instance:
    NeuralNetClassifier(
        name='nn_classifier',
        category='prediction.one_off.classification',
        params={
            'n_static_units_hidden': 100,
            'n_static_layers_hidden': 2,
            'n_temporal_units_hidden': 102,
            'n_temporal_layers_hidden': 2,
            'n_iter': 100,
            'mode': 'RNN',
            'n_iter_print': 10,
            'batch_size': 100,
            'lr': 0.001,
            'weight_decay': 0.001,
            'window_size': 1,
            'device': None,
            'dataloader_sampler': None,
            'dropout': 0.0,
            'nonlin': 'relu',
            'random_state': 0,
            'clipping_value': 1,
            'patience': 20,
            'train_ratio': 0.8
        }
    )


You can also get the plugin instance directly (rather than the class) from the API, as follows: 

## Listing all available `Plugin`s

You can list all `Plugin`s currently available in TemporAI as follows:


```python
from tempor.plugins import plugin_loader

# Use plugin_loader.list():
all_plugins = plugin_loader.list()

# Displaying using pretty print here for clarity:
from rich.pretty import pprint

pprint(all_plugins, indent_guides=False)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'prediction'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'one_off'</span>: <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'classification'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'cde_classifier'</span>, <span style="color: #008000; text-decoration-color: #008000">'ode_classifier'</span>, <span style="color: #008000; text-decoration-color: #008000">'nn_classifier'</span>, <span style="color: #008000; text-decoration-color: #008000">'laplace_ode_classifier'</span><span style="font-weight: bold">]</span>,
            <span style="color: #008000; text-decoration-color: #008000">'regression'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'laplace_ode_regressor'</span>, <span style="color: #008000; text-decoration-color: #008000">'nn_regressor'</span>, <span style="color: #008000; text-decoration-color: #008000">'ode_regressor'</span>, <span style="color: #008000; text-decoration-color: #008000">'cde_regressor'</span><span style="font-weight: bold">]</span>
        <span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'temporal'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'classification'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'seq2seq_classifier'</span><span style="font-weight: bold">]</span>, <span style="color: #008000; text-decoration-color: #008000">'regression'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'seq2seq_regressor'</span><span style="font-weight: bold">]}</span>
    <span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'preprocessing'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'imputation'</span>: <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'static'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'static_tabular_imputer'</span><span style="font-weight: bold">]</span>,
            <span style="color: #008000; text-decoration-color: #008000">'temporal'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'ffill'</span>, <span style="color: #008000; text-decoration-color: #008000">'ts_tabular_imputer'</span>, <span style="color: #008000; text-decoration-color: #008000">'bfill'</span><span style="font-weight: bold">]</span>
        <span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'nop'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'nop_transformer'</span><span style="font-weight: bold">]</span>,
        <span style="color: #008000; text-decoration-color: #008000">'scaling'</span>: <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'static'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'static_minmax_scaler'</span>, <span style="color: #008000; text-decoration-color: #008000">'static_standard_scaler'</span><span style="font-weight: bold">]</span>,
            <span style="color: #008000; text-decoration-color: #008000">'temporal'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'ts_minmax_scaler'</span>, <span style="color: #008000; text-decoration-color: #008000">'ts_standard_scaler'</span><span style="font-weight: bold">]</span>
        <span style="font-weight: bold">}</span>
    <span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'time_to_event'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'ts_coxph'</span>, <span style="color: #008000; text-decoration-color: #008000">'ts_xgb'</span>, <span style="color: #008000; text-decoration-color: #008000">'dynamic_deephit'</span><span style="font-weight: bold">]</span>,
    <span style="color: #008000; text-decoration-color: #008000">'treatments'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'one_off'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'regression'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'synctwin_regressor'</span><span style="font-weight: bold">]}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'temporal'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'classification'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'crn_classifier'</span><span style="font-weight: bold">]</span>, <span style="color: #008000; text-decoration-color: #008000">'regression'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'crn_regressor'</span><span style="font-weight: bold">]}</span>
    <span style="font-weight: bold">}</span>
<span style="font-weight: bold">}</span>
</pre>



Note that plugin categories are hierarchical (nested).

To quickly view the plugin fully qualified name (`fqn`) for all plugins, use `plugin_loader.list_fqns()`.

These are the names to use in `plugin_loader.{get,get_class}` calls.


```python
plugin_loader.list_fqns()
```




    ['prediction.one_off.classification.cde_classifier',
     'prediction.one_off.classification.ode_classifier',
     'prediction.one_off.classification.nn_classifier',
     'prediction.one_off.classification.laplace_ode_classifier',
     'prediction.one_off.regression.laplace_ode_regressor',
     'prediction.one_off.regression.nn_regressor',
     'prediction.one_off.regression.ode_regressor',
     'prediction.one_off.regression.cde_regressor',
     'prediction.temporal.classification.seq2seq_classifier',
     'prediction.temporal.regression.seq2seq_regressor',
     'preprocessing.imputation.static.static_tabular_imputer',
     'preprocessing.imputation.temporal.ffill',
     'preprocessing.imputation.temporal.ts_tabular_imputer',
     'preprocessing.imputation.temporal.bfill',
     'preprocessing.nop.nop_transformer',
     'preprocessing.scaling.static.static_minmax_scaler',
     'preprocessing.scaling.static.static_standard_scaler',
     'preprocessing.scaling.temporal.ts_minmax_scaler',
     'preprocessing.scaling.temporal.ts_standard_scaler',
     'time_to_event.ts_coxph',
     'time_to_event.ts_xgb',
     'time_to_event.dynamic_deephit',
     'treatments.one_off.regression.synctwin_regressor',
     'treatments.temporal.classification.crn_classifier',
     'treatments.temporal.regression.crn_regressor']



