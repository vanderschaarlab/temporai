# User Guide Tutorial 06: Treatment Effects
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/usage/tutorial06_treatments.ipynb)

This tutorial shows how to use TemporAI `treatments` plugins.



## All `treatments` plugins

> ⚠️ The `treatments` API is preliminary and likely to change.

In the treatment effects estimation task, the goal is to predict a counterfactual outcome given an alternative treatment.

To see all the relevant plugins:


```python
from tempor.plugins import plugin_loader
from rich.pretty import pprint

all_treatments_plugins = plugin_loader.list()["treatments"]

pprint(all_treatments_plugins, indent_guides=False)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'one_off'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'regression'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'synctwin_regressor'</span><span style="font-weight: bold">]}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'temporal'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'classification'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'crn_classifier'</span><span style="font-weight: bold">]</span>, <span style="color: #008000; text-decoration-color: #008000">'regression'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'crn_regressor'</span><span style="font-weight: bold">]}</span>
<span style="font-weight: bold">}</span>
</pre>



## Using a temporal treatment effects plugin.

In this setting, the treatments are time series, and the outcomes are also time series.


```python
from tempor.utils.dataloaders import DummyTemporalTreatmentEffectsDataLoader
from tempor.plugins import plugin_loader

dataset = DummyTemporalTreatmentEffectsDataLoader(
    random_state=42,
    temporal_covariates_missing_prob=0.0,
    temporal_treatments_n_features=1,
    temporal_treatments_n_categories=2,
).load()
print(dataset)

model = plugin_loader.get("treatments.temporal.regression.crn_regressor", epochs=20)
print(model)
```

    TemporalTreatmentEffectsDataset(
        time_series=TimeSeriesSamples([100, *, 5]),
        static=StaticSamples([100, 3]),
        predictive=TemporalTreatmentEffectsTaskData(
            targets=TimeSeriesSamples([100, *, 2]),
            treatments=TimeSeriesSamples([100, *, 1])
        )
    )
    CRNTreatmentsRegressor(
        name='crn_regressor',
        category='treatments.temporal.regression',
        params={
            'encoder_rnn_type': 'LSTM',
            'encoder_hidden_size': 100,
            'encoder_num_layers': 1,
            'encoder_bias': True,
            'encoder_dropout': 0.0,
            'encoder_bidirectional': False,
            'encoder_nonlinearity': None,
            'encoder_proj_size': None,
            'decoder_rnn_type': 'LSTM',
            'decoder_hidden_size': 100,
            'decoder_num_layers': 1,
            'decoder_bias': True,
            'decoder_dropout': 0.0,
            'decoder_bidirectional': False,
            'decoder_nonlinearity': None,
            'decoder_proj_size': None,
            'adapter_hidden_dims': [50],
            'adapter_out_activation': 'Tanh',
            'predictor_hidden_dims': [],
            'predictor_out_activation': None,
            'max_len': None,
            'optimizer_str': 'Adam',
            'optimizer_kwargs': {'lr': 0.01, 'weight_decay': 1e-05},
            'batch_size': 32,
            'epochs': 20,
            'padding_indicator': -999.0
        }
    )



```python
# Targets:
dataset.predictive.targets
```




<p><span style="font-family: monospace;">TimeSeriesSamples</span> with data:</p><div>
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
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">0</th>
      <th>0</th>
      <td>-3.110475</td>
      <td>-3.566948</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.528495</td>
      <td>-0.653673</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.275307</td>
      <td>-0.695371</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.844060</td>
      <td>3.469371</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.420301</td>
      <td>5.147500</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">99</th>
      <th>7</th>
      <td>5.994185</td>
      <td>6.225290</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10.913662</td>
      <td>5.346697</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.558824</td>
      <td>7.585175</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10.194430</td>
      <td>5.795619</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13.774189</td>
      <td>8.457336</td>
    </tr>
  </tbody>
</table>
<p>1573 rows × 2 columns</p>
</div>




```python
# Treatments:
dataset.predictive.treatments
```




<p><span style="font-family: monospace;">TimeSeriesSamples</span> with data:</p><div>
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
      <th></th>
      <th>0</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">0</th>
      <th>0</th>
      <td>0</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">99</th>
      <th>7</th>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1573 rows × 1 columns</p>
</div>




```python
# Train.
model.fit(dataset);
```

    Preparing data for decoder training...
    Preparing data for decoder training DONE.
    === Training stage: 1. Train encoder ===
    Epoch: 0, Prediction Loss: 76.050, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 76.050
    Epoch: 1, Prediction Loss: 37.508, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 37.508
    Epoch: 2, Prediction Loss: 21.953, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 21.953
    Epoch: 3, Prediction Loss: 18.895, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 18.895
    Epoch: 4, Prediction Loss: 19.568, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 19.568
    Epoch: 5, Prediction Loss: 16.534, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 16.534
    Epoch: 6, Prediction Loss: 12.039, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 12.039
    Epoch: 7, Prediction Loss: 10.009, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 10.009
    Epoch: 8, Prediction Loss: 8.666, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 8.666
    Epoch: 9, Prediction Loss: 6.788, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 6.788
    Epoch: 10, Prediction Loss: 5.674, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 5.674
    Epoch: 11, Prediction Loss: 5.016, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 5.016
    Epoch: 12, Prediction Loss: 4.605, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 4.605
    Epoch: 13, Prediction Loss: 4.349, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 4.349
    Epoch: 14, Prediction Loss: 4.096, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 4.096
    Epoch: 15, Prediction Loss: 4.008, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 4.008
    Epoch: 16, Prediction Loss: 4.159, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 4.159
    Epoch: 17, Prediction Loss: 3.980, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.980
    Epoch: 18, Prediction Loss: 3.904, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.904
    Epoch: 19, Prediction Loss: 3.995, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.995
    === Training stage: 2. Train decoder ===
    Epoch: 0, Prediction Loss: 34.805, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 34.805
    Epoch: 1, Prediction Loss: 13.675, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 13.675
    Epoch: 2, Prediction Loss: 7.021, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 7.021
    Epoch: 3, Prediction Loss: 4.086, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 4.086
    Epoch: 4, Prediction Loss: 3.847, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.847
    Epoch: 5, Prediction Loss: 3.795, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.795
    Epoch: 6, Prediction Loss: 3.762, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.762
    Epoch: 7, Prediction Loss: 3.800, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.800
    Epoch: 8, Prediction Loss: 3.792, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.792
    Epoch: 9, Prediction Loss: 3.703, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.703
    Epoch: 10, Prediction Loss: 3.683, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.683
    Epoch: 11, Prediction Loss: 3.734, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.734
    Epoch: 12, Prediction Loss: 3.690, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.690
    Epoch: 13, Prediction Loss: 3.728, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.728
    Epoch: 14, Prediction Loss: 3.694, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.694
    Epoch: 15, Prediction Loss: 3.647, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.647
    Epoch: 16, Prediction Loss: 3.639, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.639
    Epoch: 17, Prediction Loss: 3.612, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.612
    Epoch: 18, Prediction Loss: 3.633, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.633
    Epoch: 19, Prediction Loss: 3.604, Lambda: 1.000, Treatment BR Loss: 0.000, Loss: 3.604



```python
# Predict counterfactuals:

import numpy as np

dataset = dataset[:5]

# Define horizons for each sample.
horizons = [tc.time_indexes()[0][len(tc.time_indexes()[0]) // 2 :] for tc in dataset.time_series]
print("Horizons for sample 0:\n", horizons[0], end="\n\n")

# Define treatment scenarios for each sample.
treatment_scenarios = [[np.asarray([1] * len(h)), np.asarray([0] * len(h))] for h in horizons]
print("Alternative treatment scenarios for sample 0:\n", treatment_scenarios[0], end="\n\n")

# Call predict_counterfactuals.
counterfactuals = model.predict_counterfactuals(dataset, horizons=horizons, treatment_scenarios=treatment_scenarios)
print("Counterfactual outcomes for sample 0, given the alternative treatment scenarios:\n")
for idx, c in enumerate(counterfactuals[0]):
    print(f"Treatment scenario {idx}, {treatment_scenarios[0][idx]}")
    print(c, end="\n\n")
```

    Horizons for sample 0:
     [5, 6, 7, 8, 9, 10]
    
    Alternative treatment scenarios for sample 0:
     [array([1, 1, 1, 1, 1, 1]), array([0, 0, 0, 0, 0, 0])]
    
    Counterfactual outcomes for sample 0, given the alternative treatment scenarios:
    
    Treatment scenario 0, [1 1 1 1 1 1]
    TimeSeries() with data:
                     0         1
    time_idx                    
    5         6.339921  5.118720
    6         6.379971  5.159422
    7         6.380514  5.159956
    8         6.380521  5.159964
    9         6.380521  5.159964
    10        6.380521  5.159964
    
    Treatment scenario 1, [0 0 0 0 0 0]
    TimeSeries() with data:
                     0         1
    time_idx                    
    5         6.694273  5.142348
    6         6.732183  5.177425
    7         6.732780  5.178019
    8         6.732789  5.178029
    9         6.732790  5.178029
    10        6.732790  5.178029
    


