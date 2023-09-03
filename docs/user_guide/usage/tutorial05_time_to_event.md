# User Guide Tutorial 05: Time-to-event Analysis
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/usage/tutorial05_time_to_event.ipynb)

This tutorial shows how to use TemporAI `time_to_event` plugins.



## All `time_to_event` plugins

Time-to-event analysis, in the context of TemporAI, refers to models that estimate risk over time for each sample.
This may also be referred to as survival analysis.

To see all the relevant plugins:


```python
from tempor.plugins import plugin_loader

plugin_loader.list()["time_to_event"]
```




    ['ts_coxph', 'ts_xgb', 'dynamic_deephit']



## Using a time-to-event analysis plugin.


```python
from tempor.utils.dataloaders import PBCDataLoader
from tempor.plugins import plugin_loader

dataset = PBCDataLoader(random_state=42).load()
print(dataset)

model = plugin_loader.get("time_to_event.dynamic_deephit", n_iter=50)
print(model)
```

    TimeToEventAnalysisDataset(
        time_series=TimeSeriesSamples([312, *, 14]),
        static=StaticSamples([312, 1]),
        predictive=TimeToEventAnalysisTaskData(targets=EventSamples([312, 1]))
    )
    DynamicDeepHitTimeToEventAnalysis(
        name='dynamic_deephit',
        category='time_to_event',
        params={
            'n_iter': 50,
            'batch_size': 100,
            'lr': 0.001,
            'n_layers_hidden': 1,
            'n_units_hidden': 40,
            'split': 100,
            'rnn_mode': 'GRU',
            'alpha': 0.34,
            'beta': 0.27,
            'sigma': 0.21,
            'dropout': 0.06,
            'device': 'cpu',
            'patience': 20,
            'output_mode': 'MLP',
            'random_state': 0
        }
    )



```python
# Targets:
dataset.predictive.targets
```




<p><span style="font-family: monospace;">EventSamples</span> with data:</p><div>
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
      <th>status</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>(0.569488555470374, True)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(14.1523381885883, False)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(0.7365020260650499, True)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(0.27653050049282957, True)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(4.12057824991786, False)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>308</th>
      <td>(4.98850071186069, False)</td>
    </tr>
    <tr>
      <th>309</th>
      <td>(4.55317051801555, False)</td>
    </tr>
    <tr>
      <th>310</th>
      <td>(4.4025846019056, False)</td>
    </tr>
    <tr>
      <th>311</th>
      <td>(4.12879202716022, False)</td>
    </tr>
    <tr>
      <th>312</th>
      <td>(3.98915781404008, False)</td>
    </tr>
  </tbody>
</table>
<p>312 rows × 1 columns</p>
</div>




```python
# Train.
model.fit(dataset)
```




    DynamicDeepHitTimeToEventAnalysis(
        name='dynamic_deephit',
        category='time_to_event',
        params={
            'n_iter': 50,
            'batch_size': 100,
            'lr': 0.001,
            'n_layers_hidden': 1,
            'n_units_hidden': 40,
            'split': 100,
            'rnn_mode': 'GRU',
            'alpha': 0.34,
            'beta': 0.27,
            'sigma': 0.21,
            'dropout': 0.06,
            'device': 'cpu',
            'patience': 20,
            'output_mode': 'MLP',
            'random_state': 0
        }
    )




```python
# Predict:

model.predict(dataset, horizons=[0.25, 0.50, 0.75])
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
      <th>risk_score</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>0.25</th>
      <td>0.422643</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>0.626620</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.717206</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>0.25</th>
      <td>0.221314</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>0.288492</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">311</th>
      <th>0.50</th>
      <td>0.039302</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.048346</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">312</th>
      <th>0.25</th>
      <td>0.248833</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>0.390042</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>0.444533</td>
    </tr>
  </tbody>
</table>
<p>936 rows × 1 columns</p>
</div>



### Note:

The current Dynamic DeepHit implementation has the following limitations:
- Only one output feature is supported (no competing risks).
- Risk prediction for time points beyond the last event time in the dataset may throw errors.

