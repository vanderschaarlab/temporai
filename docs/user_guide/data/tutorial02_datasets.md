# Data Tutorial 02: Datasets
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/data/tutorial02_datasets.ipynb)

This tutorial shows different TemporAI `Dataset`s.



## Prepare some example data


```python
import pandas as pd
import numpy as np

# Some time series data:
time_series_df = pd.DataFrame(
    {
        "sample_idx": ["sample_0", "sample_0", "sample_0", "sample_0", "sample_1", "sample_1", "sample_2"],
        "time_idx": [1, 2, 3, 4, 2, 4, 9],
        "t_feat_0": [11, 12, 13, 14, 21, 22, 31],
        "t_feat_1": [1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 3.1],
        "t_feat_2": [10, 20, 30, 40, 11, 21, 111],
    }
)
time_series_df.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)

# Some static data:
static_df = pd.DataFrame(
    {
        "s_feat_0": [100, 200, 300],
        "s_feat_1": [-1.1, -1.2, -1.3],
        "s_feat_2": [0, 1, 0],
    },
    index=["sample_0", "sample_1", "sample_2"],
)

event_df = pd.DataFrame(
    {
        "e_feat_0": [(10, True), (12, False), (13, True)],
        "e_feat_1": [(10, False), (10, False), (11, True)],
    },
    index=["sample_0", "sample_1", "sample_2"],
)
```

Preview the dataframes below.


```python
time_series_df
```




<div>
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
      <th>t_feat_0</th>
      <th>t_feat_1</th>
      <th>t_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">sample_0</th>
      <th>1</th>
      <td>11</td>
      <td>1.1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>1.2</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>1.3</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>1.4</td>
      <td>40</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>21</td>
      <td>2.1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>2.2</td>
      <td>21</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>31</td>
      <td>3.1</td>
      <td>111</td>
    </tr>
  </tbody>
</table>
</div>




```python
static_df
```




<div>
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
      <th>s_feat_0</th>
      <th>s_feat_1</th>
      <th>s_feat_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_0</th>
      <td>100</td>
      <td>-1.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sample_1</th>
      <td>200</td>
      <td>-1.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>300</td>
      <td>-1.3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
event_df
```




<div>
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
      <th>e_feat_0</th>
      <th>e_feat_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_0</th>
      <td>(10, True)</td>
      <td>(10, False)</td>
    </tr>
    <tr>
      <th>sample_1</th>
      <td>(12, False)</td>
      <td>(10, False)</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>(13, True)</td>
      <td>(11, True)</td>
    </tr>
  </tbody>
</table>
</div>



## `CovariatesDataset`

A `CovariatesDataset` contains time series and optionally static covariates only, without any predictive data
(targets or treatments).

It can be used with `preprocessing` transformations.


```python
from tempor.data import dataset
```


```python
# Initialize a CovariatesDataset:
data = dataset.CovariatesDataset(
    time_series=time_series_df,
    static=static_df,  # Optional, can be `None`.
)

data
```




    CovariatesDataset(
        time_series=TimeSeriesSamples([3, *, 3]),
        static=StaticSamples([3, 3])
    )




```python
data.time_series
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
      <th>t_feat_0</th>
      <th>t_feat_1</th>
      <th>t_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">sample_0</th>
      <th>1</th>
      <td>11</td>
      <td>1.1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>1.2</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>1.3</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>1.4</td>
      <td>40</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>21</td>
      <td>2.1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>2.2</td>
      <td>21</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>31</td>
      <td>3.1</td>
      <td>111</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.static
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
      <th>s_feat_0</th>
      <th>s_feat_1</th>
      <th>s_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_0</th>
      <td>100</td>
      <td>-1.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sample_1</th>
      <td>200</td>
      <td>-1.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>300</td>
      <td>-1.3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## `OneOffPredictionDataset`

A `OneOffPredictionDataset` contains time series and optionally static covariates.

It also needs `StaticSamples` prediction *targets* for estimators to be able to `fit` on this dataset.

It can be used with `prediction.one_off` estimators. The task is to predict some one-off value for each sample.


```python
# Initialize a OneOffPredictionDataset:
data = dataset.OneOffPredictionDataset(
    time_series=time_series_df,
    static=static_df.loc[:, :"s_feat_1"],  # Optional, can be `None`.
    targets=static_df.loc[:, ["s_feat_2"]],  # Optional, can be `None` at inference time.
)

data
```




    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([3, *, 3]),
        static=StaticSamples([3, 2]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([3, 1]))
    )




```python
data.time_series
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
      <th>t_feat_0</th>
      <th>t_feat_1</th>
      <th>t_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">sample_0</th>
      <th>1</th>
      <td>11</td>
      <td>1.1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>1.2</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>1.3</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>1.4</td>
      <td>40</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>21</td>
      <td>2.1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>2.2</td>
      <td>21</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>31</td>
      <td>3.1</td>
      <td>111</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.static
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
      <th>s_feat_0</th>
      <th>s_feat_1</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_0</th>
      <td>100</td>
      <td>-1.1</td>
    </tr>
    <tr>
      <th>sample_1</th>
      <td>200</td>
      <td>-1.2</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>300</td>
      <td>-1.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.predictive.targets
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
      <th>s_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>sample_1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## `TemporalPredictionDataset`

A `TemporalPredictionDataset` contains time series and optionally static covariates.

It also needs `TimeSeriesSamples` prediction *targets* for estimators to be able to `fit` on this dataset.

It can be used with `prediction.temporal` estimators. The task is to predict some time series for each sample.


```python
# Initialize a TemporalPredictionDataset:
data = dataset.TemporalPredictionDataset(
    time_series=time_series_df.loc[:, :"t_feat_1"],
    static=static_df,  # Optional, can be `None`.
    targets=time_series_df.loc[:, ["t_feat_2"]],  # Optional, can be `None` at inference time.
)

data
```




    TemporalPredictionDataset(
        time_series=TimeSeriesSamples([3, *, 2]),
        static=StaticSamples([3, 3]),
        predictive=TemporalPredictionTaskData(
            targets=TimeSeriesSamples([3, *, 1])
        )
    )




```python
data.time_series
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
      <th>t_feat_0</th>
      <th>t_feat_1</th>
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
      <th rowspan="4" valign="top">sample_0</th>
      <th>1</th>
      <td>11</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>21</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>31</td>
      <td>3.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.static
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
      <th>s_feat_0</th>
      <th>s_feat_1</th>
      <th>s_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_0</th>
      <td>100</td>
      <td>-1.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sample_1</th>
      <td>200</td>
      <td>-1.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>300</td>
      <td>-1.3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.predictive.targets
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
      <th>t_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">sample_0</th>
      <th>1</th>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>111</td>
    </tr>
  </tbody>
</table>
</div>



## `TimeToEventAnalysisDataset`

A `TimeToEventAnalysisDataset` contains time series and optionally static covariates.

It also needs `EventSamples` prediction *targets* for estimators to be able to `fit` on this dataset.

It can be used with `time_to_event` estimators. The task is to predict risk scores for each sample.


```python
# Initialize a TimeToEventAnalysisDataset:
data = dataset.TimeToEventAnalysisDataset(
    time_series=time_series_df,
    static=static_df,  # Optional, can be `None`.
    targets=event_df,  # Optional, can be `None` at inference time.
)

data
```




    TimeToEventAnalysisDataset(
        time_series=TimeSeriesSamples([3, *, 3]),
        static=StaticSamples([3, 3]),
        predictive=TimeToEventAnalysisTaskData(targets=EventSamples([3, 2]))
    )




```python
data.time_series
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
      <th>t_feat_0</th>
      <th>t_feat_1</th>
      <th>t_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">sample_0</th>
      <th>1</th>
      <td>11</td>
      <td>1.1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>1.2</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>1.3</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>1.4</td>
      <td>40</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>21</td>
      <td>2.1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>2.2</td>
      <td>21</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>31</td>
      <td>3.1</td>
      <td>111</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.static
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
      <th>s_feat_0</th>
      <th>s_feat_1</th>
      <th>s_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_0</th>
      <td>100</td>
      <td>-1.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sample_1</th>
      <td>200</td>
      <td>-1.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>300</td>
      <td>-1.3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.predictive.targets
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
      <th>e_feat_0</th>
      <th>e_feat_1</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_0</th>
      <td>(10, True)</td>
      <td>(10, False)</td>
    </tr>
    <tr>
      <th>sample_1</th>
      <td>(12, False)</td>
      <td>(10, False)</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>(13, True)</td>
      <td>(11, True)</td>
    </tr>
  </tbody>
</table>
</div>



## `OneOffTreatmentEffectsDataset`

A `OneOffTreatmentEffectsDataset` contains time series and optionally static covariates.

It also needs `TimeSeriesSamples` prediction *targets* and `EventSamples` treatments
for estimators to be able to `fit` on this dataset.

It can be used with `treatments.one_off` estimators.
The task is to predict a time series counterfactual outcome based on a one-off treatment event.


```python
# Initialize a TimeToEventAnalysisDataset:
data = dataset.OneOffTreatmentEffectsDataset(
    time_series=time_series_df.loc[:, :"t_feat_1"],
    static=static_df,  # Optional, can be `None`.
    targets=time_series_df.loc[:, ["t_feat_2"]],  # Optional, can be `None` at inference time.
    treatments=event_df.loc[:, ["e_feat_0"]],
)

data
```




    OneOffTreatmentEffectsDataset(
        time_series=TimeSeriesSamples([3, *, 2]),
        static=StaticSamples([3, 3]),
        predictive=OneOffTreatmentEffectsTaskData(
            targets=TimeSeriesSamples([3, *, 1]),
            treatments=EventSamples([3, 1])
        )
    )




```python
data.time_series
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
      <th>t_feat_0</th>
      <th>t_feat_1</th>
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
      <th rowspan="4" valign="top">sample_0</th>
      <th>1</th>
      <td>11</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>21</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>31</td>
      <td>3.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.static
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
      <th>s_feat_0</th>
      <th>s_feat_1</th>
      <th>s_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_0</th>
      <td>100</td>
      <td>-1.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sample_1</th>
      <td>200</td>
      <td>-1.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>300</td>
      <td>-1.3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.predictive.targets
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
      <th>t_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">sample_0</th>
      <th>1</th>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>111</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.predictive.treatments
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
      <th>e_feat_0</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_0</th>
      <td>(10, True)</td>
    </tr>
    <tr>
      <th>sample_1</th>
      <td>(12, False)</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>(13, True)</td>
    </tr>
  </tbody>
</table>
</div>



## `TemporalTreatmentEffectsDataset`

A `TemporalTreatmentEffectsDataset` contains time series and optionally static covariates.

It also needs `TimeSeriesSamples` prediction *targets* and `TimeSeriesSamples` treatments
for estimators to be able to `fit` on this dataset.

It can be used with `treatments.temporal` estimators.
The task is to predict a time series counterfactual outcome based on a time series treatment.


```python
# Initialize a TimeToEventAnalysisDataset:
data = dataset.TemporalTreatmentEffectsDataset(
    time_series=time_series_df.loc[:, :"t_feat_0"],
    static=static_df,  # Optional, can be `None`.
    targets=time_series_df.loc[:, ["t_feat_1"]],  # Optional, can be `None` at inference time.
    treatments=time_series_df.loc[:, ["t_feat_2"]],
)

data
```




    TemporalTreatmentEffectsDataset(
        time_series=TimeSeriesSamples([3, *, 1]),
        static=StaticSamples([3, 3]),
        predictive=TemporalTreatmentEffectsTaskData(
            targets=TimeSeriesSamples([3, *, 1]),
            treatments=TimeSeriesSamples([3, *, 1])
        )
    )




```python
data.time_series
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
      <th>t_feat_0</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">sample_0</th>
      <th>1</th>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.static
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
      <th>s_feat_0</th>
      <th>s_feat_1</th>
      <th>s_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_0</th>
      <td>100</td>
      <td>-1.1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sample_1</th>
      <td>200</td>
      <td>-1.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>300</td>
      <td>-1.3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.predictive.targets
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
      <th>t_feat_1</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">sample_0</th>
      <th>1</th>
      <td>1.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>2.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.2</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>3.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.predictive.treatments
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
      <th>t_feat_2</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">sample_0</th>
      <th>1</th>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>111</td>
    </tr>
  </tbody>
</table>
</div>



