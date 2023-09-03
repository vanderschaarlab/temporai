# User Guide Tutorial 03: Preprocessing › Scaling
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/usage/tutorial03_scaling.ipynb)

This tutorial shows how to use TemporAI `preprocessing.scaling` plugins.



## All `preprocessing.scaling` plugins

To see all the relevant plugins:


```python
from tempor.plugins import plugin_loader

plugin_loader.list()["preprocessing"]["scaling"]
```




    {'static': ['static_minmax_scaler', 'static_standard_scaler'],
     'temporal': ['ts_minmax_scaler', 'ts_standard_scaler']}



## Using a static data scaling plugin


```python
from tempor.utils.dataloaders import SineDataLoader
from tempor.plugins import plugin_loader

dataset = SineDataLoader(static_scale=5.0, random_state=42).load()
print(dataset)

model = plugin_loader.get("preprocessing.scaling.static.static_minmax_scaler", static_imputer="mean")
print(model)
```

    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([100, *, 5]),
        static=StaticSamples([100, 4]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([100, 1]))
    )
    StaticMinMaxScaler(
        name='static_minmax_scaler',
        category='preprocessing.scaling.static',
        params={}
    )



```python
# Note the scale of static features.

from IPython.display import display

print("Min, max values per feature:")
display(dataset.static.dataframe().describe().T.loc[:, ["min", "max"]])  # type: ignore

dataset.static
```

    Min, max values per feature:



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
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.025308</td>
      <td>4.818100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.045985</td>
      <td>4.950269</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.102922</td>
      <td>4.952526</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.082939</td>
      <td>4.858910</td>
    </tr>
  </tbody>
</table>
</div>





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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.872701</td>
      <td>4.753572</td>
      <td>3.659970</td>
      <td>2.993292</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.780093</td>
      <td>0.779973</td>
      <td>0.290418</td>
      <td>4.330881</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.005575</td>
      <td>3.540363</td>
      <td>0.102922</td>
      <td>4.849549</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.162213</td>
      <td>1.061696</td>
      <td>0.909125</td>
      <td>0.917023</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.521211</td>
      <td>2.623782</td>
      <td>2.159725</td>
      <td>1.456146</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.590824</td>
      <td>3.483686</td>
      <td>3.144714</td>
      <td>4.387360</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3.675355</td>
      <td>4.017405</td>
      <td>1.410173</td>
      <td>0.887198</td>
    </tr>
    <tr>
      <th>97</th>
      <td>3.753074</td>
      <td>4.034174</td>
      <td>4.952526</td>
      <td>2.063088</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1.860090</td>
      <td>3.882065</td>
      <td>1.704018</td>
      <td>4.653787</td>
    </tr>
    <tr>
      <th>99</th>
      <td>4.292064</td>
      <td>2.144970</td>
      <td>3.754355</td>
      <td>3.772714</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>




```python
# Note the new scale of static features.

dataset = model.fit_transform(dataset)  # Or call fit() then transform().

print("Min, max values per feature:")
display(dataset.static.dataframe().describe().T.loc[:, ["min", "max"]])  # type: ignore

dataset.static
```

    Min, max values per feature:



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
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>





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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.385452</td>
      <td>0.959893</td>
      <td>0.733472</td>
      <td>0.609374</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.157483</td>
      <td>0.149662</td>
      <td>0.038662</td>
      <td>0.889440</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.621823</td>
      <td>0.712515</td>
      <td>0.000000</td>
      <td>0.998040</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.863151</td>
      <td>0.207107</td>
      <td>0.166241</td>
      <td>0.174642</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.312115</td>
      <td>0.525621</td>
      <td>0.424118</td>
      <td>0.287524</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.117993</td>
      <td>0.700959</td>
      <td>0.627225</td>
      <td>0.901266</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.761570</td>
      <td>0.809786</td>
      <td>0.269558</td>
      <td>0.168397</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.777786</td>
      <td>0.813205</td>
      <td>1.000000</td>
      <td>0.414607</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.382821</td>
      <td>0.782190</td>
      <td>0.330150</td>
      <td>0.957051</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.890244</td>
      <td>0.427990</td>
      <td>0.752934</td>
      <td>0.772571</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>



## Using a temporal data scaling plugin


```python
from tempor.utils.dataloaders import SineDataLoader
from tempor.plugins import plugin_loader

dataset = SineDataLoader(ts_scale=5.0, random_state=42).load()
print(dataset)

model = plugin_loader.get("preprocessing.scaling.temporal.ts_standard_scaler")
print(model)
```

    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([100, *, 5]),
        static=StaticSamples([100, 4]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([100, 1]))
    )
    TimeSeriesStandardScaler(
        name='ts_standard_scaler',
        category='preprocessing.scaling.temporal',
        params={}
    )



```python
# Note the scale of time series features.

from IPython.display import display

print("Min, max values per feature:")
display(dataset.time_series.dataframe().describe().T.loc[:, ["min", "max"]])

dataset.time_series
```

    Min, max values per feature:



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
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-4.999519</td>
      <td>4.999999</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-4.999982</td>
      <td>4.999999</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-4.999923</td>
      <td>4.999992</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-4.999979</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-4.999970</td>
      <td>4.999928</td>
    </tr>
  </tbody>
</table>
</div>





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
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">0</th>
      <th>0</th>
      <td>-0.095075</td>
      <td>-0.240884</td>
      <td>-0.542729</td>
      <td>2.209324</td>
      <td>0.122539</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.500152</td>
      <td>1.822750</td>
      <td>2.882952</td>
      <td>4.450264</td>
      <td>2.673609</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.939520</td>
      <td>3.567547</td>
      <td>4.864963</td>
      <td>4.930896</td>
      <td>4.464728</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.073484</td>
      <td>4.688307</td>
      <td>4.410788</td>
      <td>3.461105</td>
      <td>4.986786</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.784229</td>
      <td>4.988986</td>
      <td>1.747861</td>
      <td>0.622269</td>
      <td>4.091392</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">99</th>
      <th>5</th>
      <td>4.835604</td>
      <td>0.634449</td>
      <td>4.634897</td>
      <td>4.910111</td>
      <td>4.815565</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.532665</td>
      <td>2.066645</td>
      <td>2.845170</td>
      <td>3.281070</td>
      <td>4.946122</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.263739</td>
      <td>3.315606</td>
      <td>0.121903</td>
      <td>0.253339</td>
      <td>4.998853</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-1.350749</td>
      <td>4.270597</td>
      <td>-2.641363</td>
      <td>-2.882388</td>
      <td>4.972930</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-3.595883</td>
      <td>4.846944</td>
      <td>-4.537960</td>
      <td>-4.789380</td>
      <td>4.868760</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 5 columns</p>
</div>




```python
# Note the new scale of time series features.

dataset = model.fit_transform(dataset)  # Or call fit() then transform().

print("Min, max values per feature:")
display(dataset.time_series.dataframe().describe().T.loc[:, ["min", "max"]])

dataset.time_series
```

    Min, max values per feature:



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
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.711349</td>
      <td>1.200819</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.724449</td>
      <td>1.239101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.734762</td>
      <td>1.230568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.592516</td>
      <td>1.277314</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.728804</td>
      <td>1.177170</td>
    </tr>
  </tbody>
</table>
</div>





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
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">0</th>
      <th>0</th>
      <td>-0.283024</td>
      <td>-0.314064</td>
      <td>-0.413046</td>
      <td>0.476436</td>
      <td>-0.240201</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.181555</td>
      <td>0.297505</td>
      <td>0.602791</td>
      <td>1.119549</td>
      <td>0.501140</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.600744</td>
      <td>0.814586</td>
      <td>1.190527</td>
      <td>1.257482</td>
      <td>1.021640</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.930989</td>
      <td>1.146729</td>
      <td>1.055848</td>
      <td>0.835676</td>
      <td>1.173350</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.137980</td>
      <td>1.235837</td>
      <td>0.266196</td>
      <td>0.020977</td>
      <td>0.913149</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">99</th>
      <th>5</th>
      <td>1.152942</td>
      <td>-0.054654</td>
      <td>1.122305</td>
      <td>1.251517</td>
      <td>1.123594</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.773486</td>
      <td>0.369785</td>
      <td>0.591587</td>
      <td>0.784009</td>
      <td>1.161533</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.112705</td>
      <td>0.739922</td>
      <td>-0.215959</td>
      <td>-0.084900</td>
      <td>1.176857</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.648715</td>
      <td>1.022938</td>
      <td>-1.035365</td>
      <td>-0.984802</td>
      <td>1.169324</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-1.302567</td>
      <td>1.193742</td>
      <td>-1.597773</td>
      <td>-1.532077</td>
      <td>1.139052</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 5 columns</p>
</div>



