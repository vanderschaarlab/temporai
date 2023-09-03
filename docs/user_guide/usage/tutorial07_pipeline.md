# User Guide Tutorial 07: Pipeline
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/usage/tutorial07_pipeline.ipynb)

This tutorial shows how to use TemporAI `Pipeline`s.



## TemporAI `Pipeline`

A TemporAI `Pipeline` allows you to combine multiple plugins into one;
inspired by be [scikit-learn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

* All but the final plugin in the pipeline need to be data transformers (the `preprocessing` plugin category),
* The final one must be a predictive plugin (any of the `prediction`, `time_to_event`, `treatments` plugin categories).

When fitting, all the stages will be fitted, and the data will be sequentially transformed by all the preprocessing
steps before fitting the final predictive method plugin.

When predicting, the data will be again transformed by the preprocessing steps, and prediction carried out using the
final predictive method plugin.

**Note:**

All pipelines follow `PipelineBase` interface, see API reference for details.

## Example

Below is an example of a pipeline ending with `prediction.one_off.nn_classifier`.

Initializing the `Pipeline` follows the following steps.
1. Use the `pipeline()` function to create a *pipeline class* from a list of strings denoting its steps.
1. Instantiate the pipeline class. The initialization arguments to each component plugin can be passed as a dictionary at this step.
1. Use the pipeline like any other TemporAI estimator (call `.fit(...)`, `.predict(...)` and so on).


```python
from rich.pretty import pprint  # For fancy printing only.
```


```python
from tempor.plugins.pipeline import pipeline

# 1. Create a pipeline class based on your desired definition of the pipeline.
PipelineClass = pipeline(
    # Provide plugin names for the pipeline, in order.
    [
        # Preprocessing (data transformer) plugins:
        "preprocessing.imputation.temporal.bfill",
        "preprocessing.imputation.static.static_tabular_imputer",
        "preprocessing.imputation.temporal.ts_tabular_imputer",
        "preprocessing.scaling.temporal.ts_minmax_scaler",
        # Prediction plugin:
        "prediction.one_off.classification.nn_classifier",
    ],
)
print("Pipeline class:")
print(PipelineClass)

print("\nPipeline base classes (note `PipelineBase`):")
pprint(PipelineClass.mro())

pipe = PipelineClass(
    # You can provide initialization arguments to each plugin comprising the pipeline as a dictionary, as follows:
    {
        "static_imputer": {"static_imputer": "ice", "random_state": 42},
        "nn_classifier": {"n_iter": 100},
    }
)

print("Pipeline instance:")
pprint(pipe)
```

    Pipeline class:
    <class 'tempor.plugins.pipeline.pipeline.<locals>.Pipeline'>
    
    Pipeline base classes (note `PipelineBase`):



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">[</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">class</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.pipeline.pipeline.&lt;locals&gt;.Pipeline'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #000000; text-decoration-color: #000000">&lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.pipeline.PipelineBase'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #000000; text-decoration-color: #000000">&lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.prediction.one_off.classification.BaseOneOffClassifier'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #000000; text-decoration-color: #000000">&lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.core._base_predictor.BasePredictor'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #000000; text-decoration-color: #000000">&lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.core._base_estimator.BaseEstimator'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #000000; text-decoration-color: #000000">&lt;class </span><span style="color: #008000; text-decoration-color: #008000">'tempor.plugins.core._plugin.Plugin'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #000000; text-decoration-color: #000000">&lt;class </span><span style="color: #008000; text-decoration-color: #008000">'abc.ABC'</span><span style="color: #000000; text-decoration-color: #000000">&gt;,</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #000000; text-decoration-color: #000000">&lt;class </span><span style="color: #008000; text-decoration-color: #008000">'object'</span><span style="font-weight: bold">&gt;</span>
<span style="font-weight: bold">]</span>
</pre>



    Pipeline instance:



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Pipeline</span><span style="font-weight: bold">(</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #808000; text-decoration-color: #808000">pipeline_seq</span>=<span style="color: #008000; text-decoration-color: #008000">'preprocessing.imputation.temporal.bfill-&gt;preprocessing.imputation.static.static_tabular_imputer-&gt;preprocessing.imputation.temporal.ts_tabular_imputer-&gt;preprocessing.scaling.temporal.ts_minmax_scaler-&gt;prediction.one_off.classification.nn_classifier'</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #808000; text-decoration-color: #808000">predictor_category</span>=<span style="color: #008000; text-decoration-color: #008000">'prediction.one_off.classification'</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="color: #808000; text-decoration-color: #808000">params</span>=<span style="font-weight: bold">{</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'bfill'</span>: <span style="font-weight: bold">{}</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'static_tabular_imputer'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'imputer'</span>: <span style="color: #008000; text-decoration-color: #008000">'ice'</span>, <span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>, <span style="color: #008000; text-decoration-color: #008000">'imputer_params'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">}}</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'ts_tabular_imputer'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'imputer'</span>: <span style="color: #008000; text-decoration-color: #008000">'ice'</span>, <span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>, <span style="color: #008000; text-decoration-color: #008000">'imputer_params'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">}}</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'ts_minmax_scaler'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'feature_range'</span>: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span><span style="font-weight: bold">]</span>, <span style="color: #008000; text-decoration-color: #008000">'clip'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">}</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="color: #008000; text-decoration-color: #008000">'nn_classifier'</span>: <span style="font-weight: bold">{</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'n_static_units_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'n_static_layers_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'n_temporal_units_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">102</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'n_temporal_layers_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'n_iter'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'mode'</span>: <span style="color: #008000; text-decoration-color: #008000">'RNN'</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'n_iter_print'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'batch_size'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'lr'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'weight_decay'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'window_size'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'device'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'dataloader_sampler'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'dropout'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'nonlin'</span>: <span style="color: #008000; text-decoration-color: #008000">'relu'</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'clipping_value'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'patience'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">20</span>,
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   │   </span><span style="color: #008000; text-decoration-color: #008000">'train_ratio'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.8</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   │   </span><span style="font-weight: bold">}</span>
<span style="color: #7fbf7f; text-decoration-color: #7fbf7f">│   </span><span style="font-weight: bold">}</span>
<span style="font-weight: bold">)</span>
</pre>



Using the `Pipeline`:


```python
from tempor.utils.dataloaders import SineDataLoader

dataset = SineDataLoader(random_state=42).load()

# Fit:
pipe.fit(dataset)

# Predict:
pipe.predict(dataset)  # This will transform the data ant then predict.
```




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
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 1 columns</p>
</div>



