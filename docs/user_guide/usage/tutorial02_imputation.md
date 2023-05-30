# User Guide Tutorial 02: Preprocessing › Imputation
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/user_guide/tutorial02_imputation.ipynb)

This tutorial shows how to use TemporAI `preprocessing.imputation` plugins.



## All `preprocessing.imputation` plugins

To see all the relevant plugins:


```python
from tempor.plugins import plugin_loader

plugin_loader.list()["preprocessing"]["imputation"]
```




    {'static': ['static_tabular_imputer'],
     'temporal': ['ffill', 'ts_tabular_imputer', 'bfill']}



## Using a static data imputation plugin


```python
from tempor.utils.dataloaders import SineDataLoader
from tempor.plugins import plugin_loader

dataset = SineDataLoader(with_missing=True, random_state=42).load()
print(dataset)

model = plugin_loader.get("preprocessing.imputation.static.static_tabular_imputer", static_imputer="mean")
print(model)
```

    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([100, *, 5]),
        static=StaticSamples([100, 4]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([100, 1]))
    )
    StaticTabularImputer(
        name='static_tabular_imputer',
        category='preprocessing.imputation.static',
        params={
            'imputer': 'ice',
            'random_state': 0,
            'imputer_params': {'random_state': 0}
        }
    )



```python
# Note missingness in static data.

print("Missing value count:", dataset.static.dataframe().isnull().sum().sum())  # type: ignore

dataset.static
```

    Missing value count: 40





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
      <td>0.374540</td>
      <td>0.950714</td>
      <td>0.731994</td>
      <td>0.598658</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.156019</td>
      <td>0.155995</td>
      <td>0.058084</td>
      <td>0.866176</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.601115</td>
      <td>0.708073</td>
      <td>0.020584</td>
      <td>0.969910</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.832443</td>
      <td>NaN</td>
      <td>0.181825</td>
      <td>0.183405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.304242</td>
      <td>0.524756</td>
      <td>0.431945</td>
      <td>0.291229</td>
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
      <td>NaN</td>
      <td>0.696737</td>
      <td>0.628943</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.735071</td>
      <td>0.803481</td>
      <td>0.282035</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.750615</td>
      <td>0.806835</td>
      <td>0.990505</td>
      <td>0.412618</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.372018</td>
      <td>0.776413</td>
      <td>0.340804</td>
      <td>0.930757</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.858413</td>
      <td>0.428994</td>
      <td>0.750871</td>
      <td>0.754543</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>




```python
# Note no more missingness in static data.

dataset = model.fit_transform(dataset)  # Or call fit() then transform().

print("Missing value count:", dataset.static.dataframe().isnull().sum().sum())  # type: ignore

dataset.static
```

    Missing value count: 0





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
      <td>0.374540</td>
      <td>0.950714</td>
      <td>0.731994</td>
      <td>0.598658</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.156019</td>
      <td>0.155995</td>
      <td>0.058084</td>
      <td>0.866176</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.601115</td>
      <td>0.708073</td>
      <td>0.020584</td>
      <td>0.969910</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.832443</td>
      <td>0.450438</td>
      <td>0.181825</td>
      <td>0.183405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.304242</td>
      <td>0.524756</td>
      <td>0.431945</td>
      <td>0.291229</td>
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
      <td>0.498806</td>
      <td>0.696737</td>
      <td>0.628943</td>
      <td>0.509994</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.735071</td>
      <td>0.803481</td>
      <td>0.282035</td>
      <td>0.503886</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.750615</td>
      <td>0.806835</td>
      <td>0.990505</td>
      <td>0.412618</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.372018</td>
      <td>0.776413</td>
      <td>0.340804</td>
      <td>0.930757</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.858413</td>
      <td>0.428994</td>
      <td>0.750871</td>
      <td>0.754543</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>



## Using a temporal data imputation plugin


```python
from tempor.utils.dataloaders import SineDataLoader
from tempor.plugins import plugin_loader

dataset = SineDataLoader(with_missing=True, random_state=42).load()
print(dataset)

model = plugin_loader.get("preprocessing.imputation.temporal.bfill")
print(model)
```

    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([100, *, 5]),
        static=StaticSamples([100, 4]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([100, 1]))
    )
    BFillImputer(
        name='bfill',
        category='preprocessing.imputation.temporal',
        params={}
    )



```python
# Note missingness in temporal data.

print("Missing value count:", dataset.time_series.dataframe().isnull().sum().sum())

dataset.time_series
```

    Missing value count: 500





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
      <td>-0.955338</td>
      <td>0.016053</td>
      <td>-0.995752</td>
      <td>0.948138</td>
      <td>0.738158</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.896718</td>
      <td>0.717189</td>
      <td>-0.497625</td>
      <td>0.962001</td>
      <td>0.968258</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.346466</td>
      <td>0.999920</td>
      <td>0.423104</td>
      <td>0.639780</td>
      <td>0.972469</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.393737</td>
      <td>0.699299</td>
      <td>0.984517</td>
      <td>0.094046</td>
      <td>0.749807</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.918072</td>
      <td>-0.009290</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>0.904284</td>
      <td>-0.939985</td>
      <td>0.994099</td>
      <td>-0.984349</td>
      <td>0.688521</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.990911</td>
      <td>-0.518593</td>
      <td>0.908681</td>
      <td>-0.801263</td>
      <td>0.813486</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.757745</td>
      <td>0.131791</td>
      <td>NaN</td>
      <td>-0.110629</td>
      <td>0.908965</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>0.723981</td>
      <td>0.476023</td>
      <td>0.650082</td>
      <td>0.971498</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.288052</td>
      <td>0.996486</td>
      <td>0.173255</td>
      <td>0.999008</td>
      <td>0.998817</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 5 columns</p>
</div>




```python
# Note no more missingness in temporal data.

dataset = model.fit_transform(dataset)  # Or call fit() then transform().

print("Missing value count:", dataset.time_series.dataframe().isnull().sum().sum())

dataset.time_series
```

    Missing value count: 0





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
      <td>-0.955338</td>
      <td>0.016053</td>
      <td>-0.995752</td>
      <td>0.948138</td>
      <td>0.738158</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.896718</td>
      <td>0.717189</td>
      <td>-0.497625</td>
      <td>0.962001</td>
      <td>0.968258</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.346466</td>
      <td>0.999920</td>
      <td>0.423104</td>
      <td>0.639780</td>
      <td>0.972469</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.393737</td>
      <td>0.699299</td>
      <td>0.984517</td>
      <td>0.094046</td>
      <td>0.749807</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.918072</td>
      <td>-0.009290</td>
      <td>-0.167662</td>
      <td>-0.893854</td>
      <td>-0.127538</td>
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
      <td>0.904284</td>
      <td>-0.939985</td>
      <td>0.994099</td>
      <td>-0.984349</td>
      <td>0.688521</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.990911</td>
      <td>-0.518593</td>
      <td>0.908681</td>
      <td>-0.801263</td>
      <td>0.813486</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.757745</td>
      <td>0.131791</td>
      <td>0.476023</td>
      <td>-0.110629</td>
      <td>0.908965</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.288052</td>
      <td>0.723981</td>
      <td>0.476023</td>
      <td>0.650082</td>
      <td>0.971498</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.288052</td>
      <td>0.996486</td>
      <td>0.173255</td>
      <td>0.999008</td>
      <td>0.998817</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 5 columns</p>
</div>



