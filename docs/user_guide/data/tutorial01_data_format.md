# Data Tutorial 01: Data Format
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/data/tutorial01_data_format.ipynb)

This tutorial shows a minimal example of the data format for TemporAI.



## Time series data

Time series data contains data samples (e.g. patients), with features that unfold sequentially over some number of timesteps.

Time series data should take form of a `pandas.DataFrame`, with the following specifics:
* The index should be a 2-level multiindex, where level `0` index represents sampled IDs, and level `1` represents the timesteps for each sample.
* The sample index can be comprised of either `int`s or `str`s (homogenous, not a mix of these).
* The time index (timesteps) may be `int`, `float` or `pandas.Timestep`-compatible format (homogenous, not a mix of these).
* The columns of the dataframe represent the features, column names must be `str`.
* Column (feature) values currently supported are: `bool`, `int`, `float`, or `pandas.Categorical` (homogenous per column).

Other points to note:
* Sample IDs must be unique.
* (Sample ID, timestep) combination must be unique (a sample cannot have more than one of the same timestep).
* Null values such as `numpy.nan` are allowed and represent missing values.


```python
import pandas as pd
import numpy as np

from IPython.display import display
```


```python
# Create a time series dataframe.

time_series_df = pd.DataFrame(
    {
        "sample_idx": ["sample_0", "sample_0", "sample_0", "sample_0", "sample_1", "sample_1", "sample_2"],
        "time_idx": [1, 2, 3, 4, 2, 4, 9],
        "t_feat_0": [11, 12, 13, 14, 21, 22, 31],
        "t_feat_1": [1.1, 1.2, 1.3, np.nan, 2.1, 2.2, 3.1],
        "t_feat_2": ["a", "b", "b", "c", "a", "a", "c"],
    }
)

# Set the 2-level index:
time_series_df.set_index(keys=["sample_idx", "time_idx"], drop=True, inplace=True)

# "feat_2" needs to be set to a categorical, as `str` format is not supported.
time_series_df["t_feat_2"] = pd.Categorical(time_series_df["t_feat_2"])

# Preview the dataframe:
time_series_df.info()
time_series_df
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 7 entries, ('sample_0', 1) to ('sample_2', 9)
    Data columns (total 3 columns):
     #   Column    Non-Null Count  Dtype   
    ---  ------    --------------  -----   
     0   t_feat_0  7 non-null      int64   
     1   t_feat_1  6 non-null      float64 
     2   t_feat_2  7 non-null      category
    dtypes: category(1), float64(1), int64(1)
    memory usage: 725.0+ bytes





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
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>1.2</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>1.3</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>NaN</td>
      <td>c</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>21</td>
      <td>2.1</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>2.2</td>
      <td>a</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>31</td>
      <td>3.1</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>



## Static data

Static data contains data samples (e.g. patients), features that are not associated with a particular time.

Static data should take form of a `pandas.DataFrame`, with the following specifics:
* The index represents sample IDs and is a (single level) index that can be comprised of `int`s or `str`s (homogenous, not a mix of these).
* The columns of the dataframe represent the features, column names must be `str`.
* Column (feature) values currently supported are: `bool`, `int`, `float`, or `pandas.Categorical` (homogenous per column).

Other points to note:
* Sample IDs must be unique.
* Null values such as `numpy.nan` are allowed and represent missing values.


```python
# Create a static data dataframe.

static_df = pd.DataFrame(
    {
        "s_feat_0": [100, 200, 300],
        "s_feat_1": [-1.1, np.nan, -1.3],
    },
    index=["sample_0", "sample_1", "sample_2"],
)

# Preview the dataframe:
static_df.info()
static_df
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 3 entries, sample_0 to sample_2
    Data columns (total 2 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   s_feat_0  3 non-null      int64  
     1   s_feat_1  2 non-null      float64
    dtypes: float64(1), int64(1)
    memory usage: 72.0+ bytes





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
      <td>NaN</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>300</td>
      <td>-1.3</td>
    </tr>
  </tbody>
</table>
</div>



## Event data

Event data contains data samples (e.g. patients), with features that represent occurrence of an event at a certain time.
If the event did not occur, it is "censored".

Event data should take form of a `pandas.DataFrame`, with the following specifics:
* The index represents sample IDs and is a (single level) index that can be comprised of `int`s or `str`s (homogenous, not a mix of these).
* The columns of the dataframe represent the features, column names must be `str`.
* Column (feature) values must be of the form: `Tuple[<timestep>, bool]`.
    * The first element `<timestep>` may be `int`, `float` or `pandas.Timestep`-compatible format (homogenous per column).
    * The second element `bool` indicates whether the event occurred at this time (`True`) or the event feature is censored (`False`).
    * In case of censoring, the timestep should indicate the last time information about the sample was available. 

Other points to note:
* Sample IDs must be unique.
* Null values such as `numpy.nan` are not allowed allowed - indicate an event as censored (did not occur) instead.


```python
# Create an event dataframe.

event_df = pd.DataFrame(
    {
        "e_feat_0": [(10, True), (12, False), (13, True)],
        "e_feat_1": [(10, False), (10, False), (11, True)],
    },
    index=["sample_0", "sample_1", "sample_2"],
)

# Preview the dataframe:
event_df.info()
event_df
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 3 entries, sample_0 to sample_2
    Data columns (total 2 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   e_feat_0  3 non-null      object
     1   e_feat_1  3 non-null      object
    dtypes: object(2)
    memory usage: 72.0+ bytes





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



The data can also be initialised from a 2D `numpy` array (static, event) or a 3D `numpy` array (time series).

`TODO: more info`

## Dataset

The collection of data that represents the task at hand is a `Dataset`.

A `Dataset` contains:
* Time series data (covariates),
* Static data (covariates), optional,
* Predictive data, which depends on the *task*.
    * This may contain *targets* and *treatments*.

For example, for the *time-to-event analysis task* we create a dataset as follows.


```python
from tempor.data.dataset import TimeToEventAnalysisDataset

# Create a dataset of time-to-event analysis task:
data = TimeToEventAnalysisDataset(
    time_series=time_series_df,
    static=static_df,
    targets=event_df,
)

# Preview dataset:
data
```




    TimeToEventAnalysisDataset(
        time_series=TimeSeriesSamples([3, *, 3]),
        static=StaticSamples([3, 2]),
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
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>1.2</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>1.3</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>NaN</td>
      <td>c</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>21</td>
      <td>2.1</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>2.2</td>
      <td>a</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>31</td>
      <td>3.1</td>
      <td>c</td>
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
      <td>NaN</td>
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



## Useful methods

The data (`{TimeSeries,Event,Static}Samples` classes) provide a number of useful methods, some examples below.

### Examples for `TimeSeriesSamples`


```python
time_series = data.time_series
```


```python
# Return time series data as a dataframe:

time_series.dataframe()
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
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>1.2</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>1.3</td>
      <td>b</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14</td>
      <td>NaN</td>
      <td>c</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">sample_1</th>
      <th>2</th>
      <td>21</td>
      <td>2.1</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>2.2</td>
      <td>a</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <th>9</th>
      <td>31</td>
      <td>3.1</td>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Return time series data as a numpy array:

time_series.numpy(padding_indicator=-999.0)
```




    array([[[11, 1.1, 'a'],
            [12, 1.2, 'b'],
            [13, 1.3, 'b'],
            [14, nan, 'c']],
    
           [[21, 2.1, 'a'],
            [22, 2.2, 'a'],
            [-999.0, -999.0, -999.0],
            [-999.0, -999.0, -999.0]],
    
           [[31, 3.1, 'c'],
            [-999.0, -999.0, -999.0],
            [-999.0, -999.0, -999.0],
            [-999.0, -999.0, -999.0]]], dtype=object)




```python
# Return the time series data as a list of dataframes:

time_series.list_of_dataframes()
```




    [                     t_feat_0  t_feat_1 t_feat_2
     sample_idx time_idx                             
     sample_0   1               11       1.1        a
                2               12       1.2        b
                3               13       1.3        b
                4               14       NaN        c,
                          t_feat_0  t_feat_1 t_feat_2
     sample_idx time_idx                             
     sample_1   2               21       2.1        a
                4               22       2.2        a,
                          t_feat_0  t_feat_1 t_feat_2
     sample_idx time_idx                             
     sample_2   9               31       3.1        c]




```python
# Show number of features and samples:

print("num_features:", time_series.num_features)
print("num_samples:", time_series.num_samples)
```

    num_features: 3
    num_samples: 3



```python
# Show number of samples for each sample:

print("timesteps per sample:", time_series.num_timesteps())
```

    timesteps per sample: [4, 2, 1]


### Examples for `{Static,Event}Samples`


```python
assert data.static is not None
assert data.predictive.targets is not None

static = data.static
event = data.predictive.targets
```


```python
# Return the static data as a dataframe, numpy array:

display(static.dataframe())
display(static.numpy())
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>300</td>
      <td>-1.3</td>
    </tr>
  </tbody>
</table>
</div>



    array([[100. ,  -1.1],
           [200. ,   nan],
           [300. ,  -1.3]])



```python
# Return the event data as a dataframe, numpy array:

display(event.dataframe())
display(event.numpy())
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



    array([[(10, True), (10, False)],
           [(12, False), (10, False)],
           [(13, True), (11, True)]], dtype=object)


