# Data Tutorial 03: Data loaders
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/data/tutorial03_dataloaders.ipynb)

This tutorial shows TemporAI `Dataloader`s.



## `Dataloader` class

A TemporAI `Dataloader` implements a `load()` method which returns a TemporAI dataset.

`Dataloader`s are useful to load in some custom datasets, having done the necessary preprocessing,
perhaps user-configured.

Below is an example of `SineDataLoader`.


```python
from tempor.utils.dataloaders import SineDataLoader

# The DataLoader class:
SineDataLoader
```




    tempor.utils.dataloaders.sine.SineDataLoader



The constructor of the `Dataloader` can take various keyword arguments - this is where the user may customize the data
preprocessing etc.


```python
# Initialize.

sine_dataloader = SineDataLoader(
    no=80,  # Here, number of samples.
    seq_len=5,  # Here, time series sequence length.
    # ...
)

sine_dataloader
```




    <tempor.utils.dataloaders.sine.SineDataLoader at 0x7f8155017fa0>




```python
# Load the Dataset:
data = sine_dataloader.load()

print(type(data))

data
```

    <class 'tempor.data.dataset.OneOffPredictionDataset'>





    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([80, *, 5]),
        static=StaticSamples([80, 4]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([80, 1]))
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
      <td>-0.151203</td>
      <td>0.206110</td>
      <td>0.783078</td>
      <td>0.768667</td>
      <td>0.957344</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.679518</td>
      <td>0.785370</td>
      <td>0.913243</td>
      <td>0.999923</td>
      <td>0.973799</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.997174</td>
      <td>0.999603</td>
      <td>0.985913</td>
      <td>0.784278</td>
      <td>0.730349</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.561921</td>
      <td>0.749235</td>
      <td>0.996514</td>
      <td>0.218111</td>
      <td>0.291970</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.297606</td>
      <td>0.150635</td>
      <td>0.944377</td>
      <td>-0.445537</td>
      <td>-0.224335</td>
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
      <th rowspan="5" valign="top">79</th>
      <th>0</th>
      <td>0.999730</td>
      <td>0.101680</td>
      <td>-0.976039</td>
      <td>-0.999547</td>
      <td>-0.715265</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.803590</td>
      <td>0.577241</td>
      <td>-0.696389</td>
      <td>-0.897416</td>
      <td>-0.312411</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.269220</td>
      <td>0.903914</td>
      <td>-0.188132</td>
      <td>-0.586595</td>
      <td>0.160840</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.378464</td>
      <td>0.997443</td>
      <td>0.381883</td>
      <td>-0.139366</td>
      <td>0.597849</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.866853</td>
      <td>0.833703</td>
      <td>0.826536</td>
      <td>0.340273</td>
      <td>0.900138</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 5 columns</p>
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
      <td>0.212339</td>
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
      <th>75</th>
      <td>0.051682</td>
      <td>0.531355</td>
      <td>0.540635</td>
      <td>0.637430</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.726091</td>
      <td>0.975852</td>
      <td>0.516300</td>
      <td>0.322956</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.795186</td>
      <td>0.270832</td>
      <td>0.438971</td>
      <td>0.078456</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.025351</td>
      <td>0.962648</td>
      <td>0.835980</td>
      <td>0.695974</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.408953</td>
      <td>0.173294</td>
      <td>0.156437</td>
      <td>0.250243</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 4 columns</p>
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
      <th>0</th>
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
      <td>0</td>
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
      <th>75</th>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>1</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 1 columns</p>
</div>



## Provided `Dataloader`s

TemporAI comes with a number of dataloaders, see below.


```python
# Display information about each dataloader's default loaded dataset.

from tempor.utils.dataloaders import all_dataloaders

from IPython.display import display

for dataloader_cls in all_dataloaders:
    print(f"\n{'-' * 80}\n")

    print(f"{dataloader_cls.__name__} loads the following dataset:\n")
    data = dataloader_cls().load()
    print(data)

    print("This contains:", end="\n\n")

    print("time_series:")
    display(data.time_series)
    if data.static is not None:
        print("static:")
        display(data.static)
    if data.predictive.targets is not None:
        print("predictive.targets:")
        display(data.predictive.targets)
    if data.predictive.treatments is not None:
        print("predictive.treatments:")
        display(data.predictive.treatments)
```

    
    --------------------------------------------------------------------------------
    
    DummyTemporalPredictionDataLoader loads the following dataset:
    
    TemporalPredictionDataset(
        time_series=TimeSeriesSamples([100, *, 5]),
        static=StaticSamples([100, 3]),
        predictive=TemporalPredictionTaskData(
            targets=TimeSeriesSamples([100, *, 2])
        )
    )
    This contains:
    
    time_series:



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
      <td>NaN</td>
      <td>0.893763</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.047522</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.257931</td>
      <td>2.172271</td>
      <td>2.226089</td>
      <td>2.360713</td>
      <td>1.981578</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.247657</td>
      <td>0.853397</td>
      <td>2.525946</td>
      <td>3.213647</td>
      <td>2.897191</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.396456</td>
      <td>5.386071</td>
      <td>3.721545</td>
      <td>2.503248</td>
      <td>3.517212</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.387812</td>
      <td>3.365264</td>
      <td>5.612532</td>
      <td>5.573375</td>
      <td>4.767746</td>
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
      <th>12</th>
      <td>12.654769</td>
      <td>14.810888</td>
      <td>12.914859</td>
      <td>NaN</td>
      <td>12.818675</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13.418815</td>
      <td>12.135655</td>
      <td>12.481295</td>
      <td>13.336797</td>
      <td>13.696168</td>
    </tr>
    <tr>
      <th>14</th>
      <td>13.785503</td>
      <td>14.431228</td>
      <td>15.193174</td>
      <td>17.551818</td>
      <td>14.464249</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15.344934</td>
      <td>15.916966</td>
      <td>14.368132</td>
      <td>15.965113</td>
      <td>15.419334</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16.033907</td>
      <td>15.162631</td>
      <td>17.338485</td>
      <td>17.007235</td>
      <td>17.034645</td>
    </tr>
  </tbody>
</table>
<p>1547 rows × 5 columns</p>
</div>


    static:



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
      <th>0</th>
      <td>0.753423</td>
      <td>3.239284</td>
      <td>0.995587</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.829240</td>
      <td>3.175298</td>
      <td>0.770566</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.674581</td>
      <td>3.229741</td>
      <td>1.302317</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.584040</td>
      <td>3.234011</td>
      <td>1.594861</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.501552</td>
      <td>3.211027</td>
      <td>0.639503</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.680235</td>
      <td>3.287749</td>
      <td>0.705369</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.788814</td>
      <td>3.313229</td>
      <td>1.318394</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.589116</td>
      <td>3.268607</td>
      <td>1.646737</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.551060</td>
      <td>3.268599</td>
      <td>0.998024</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.716501</td>
      <td>3.254501</td>
      <td>1.047537</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>


    predictive.targets:



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
      <td>-1.433570</td>
      <td>0.714861</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.600733</td>
      <td>2.744446</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.622874</td>
      <td>1.816995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.879785</td>
      <td>4.981217</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.477957</td>
      <td>5.932101</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">99</th>
      <th>12</th>
      <td>10.736462</td>
      <td>13.415872</td>
    </tr>
    <tr>
      <th>13</th>
      <td>11.617465</td>
      <td>15.103293</td>
    </tr>
    <tr>
      <th>14</th>
      <td>12.858327</td>
      <td>16.105966</td>
    </tr>
    <tr>
      <th>15</th>
      <td>13.652358</td>
      <td>16.148926</td>
    </tr>
    <tr>
      <th>16</th>
      <td>14.442286</td>
      <td>17.567963</td>
    </tr>
  </tbody>
</table>
<p>1547 rows × 2 columns</p>
</div>


    
    --------------------------------------------------------------------------------
    
    DummyTemporalTreatmentEffectsDataLoader loads the following dataset:
    
    TemporalTreatmentEffectsDataset(
        time_series=TimeSeriesSamples([100, *, 5]),
        static=StaticSamples([100, 3]),
        predictive=TemporalTreatmentEffectsTaskData(
            targets=TimeSeriesSamples([100, *, 2]),
            treatments=TimeSeriesSamples([100, *, 2])
        )
    )
    This contains:
    
    time_series:



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
      <td>NaN</td>
      <td>0.893763</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.047522</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.257931</td>
      <td>2.172271</td>
      <td>2.226089</td>
      <td>2.360713</td>
      <td>1.981578</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.247657</td>
      <td>0.853397</td>
      <td>2.525946</td>
      <td>3.213647</td>
      <td>2.897191</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.396456</td>
      <td>5.386071</td>
      <td>3.721545</td>
      <td>2.503248</td>
      <td>3.517212</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.387812</td>
      <td>3.365264</td>
      <td>5.612532</td>
      <td>5.573375</td>
      <td>4.767746</td>
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
      <th>12</th>
      <td>12.654769</td>
      <td>14.810888</td>
      <td>12.914859</td>
      <td>NaN</td>
      <td>12.818675</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13.418815</td>
      <td>12.135655</td>
      <td>12.481295</td>
      <td>13.336797</td>
      <td>13.696168</td>
    </tr>
    <tr>
      <th>14</th>
      <td>13.785503</td>
      <td>14.431228</td>
      <td>15.193174</td>
      <td>17.551818</td>
      <td>14.464249</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15.344934</td>
      <td>15.916966</td>
      <td>14.368132</td>
      <td>15.965113</td>
      <td>15.419334</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16.033907</td>
      <td>15.162631</td>
      <td>17.338485</td>
      <td>17.007235</td>
      <td>17.034645</td>
    </tr>
  </tbody>
</table>
<p>1547 rows × 5 columns</p>
</div>


    static:



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
      <th>0</th>
      <td>0.753423</td>
      <td>3.239284</td>
      <td>0.995587</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.829240</td>
      <td>3.175298</td>
      <td>0.770566</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.674581</td>
      <td>3.229741</td>
      <td>1.302317</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.584040</td>
      <td>3.234011</td>
      <td>1.594861</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.501552</td>
      <td>3.211027</td>
      <td>0.639503</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.680235</td>
      <td>3.287749</td>
      <td>0.705369</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.788814</td>
      <td>3.313229</td>
      <td>1.318394</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.589116</td>
      <td>3.268607</td>
      <td>1.646737</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.551060</td>
      <td>3.268599</td>
      <td>0.998024</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.716501</td>
      <td>3.254501</td>
      <td>1.047537</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>


    predictive.targets:



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
      <td>-1.433570</td>
      <td>0.714861</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.600733</td>
      <td>2.744446</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.622874</td>
      <td>1.816995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.879785</td>
      <td>4.981217</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.477957</td>
      <td>5.932101</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">99</th>
      <th>12</th>
      <td>10.736462</td>
      <td>13.415872</td>
    </tr>
    <tr>
      <th>13</th>
      <td>11.617465</td>
      <td>15.103293</td>
    </tr>
    <tr>
      <th>14</th>
      <td>12.858327</td>
      <td>16.105966</td>
    </tr>
    <tr>
      <th>15</th>
      <td>13.652358</td>
      <td>16.148926</td>
    </tr>
    <tr>
      <th>16</th>
      <td>14.442286</td>
      <td>17.567963</td>
    </tr>
  </tbody>
</table>
<p>1547 rows × 2 columns</p>
</div>


    predictive.treatments:



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
      <td>-1.433570</td>
      <td>0.714861</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.600733</td>
      <td>2.744446</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.622874</td>
      <td>1.816995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.879785</td>
      <td>4.981217</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.477957</td>
      <td>5.932101</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">99</th>
      <th>12</th>
      <td>10.736462</td>
      <td>13.415872</td>
    </tr>
    <tr>
      <th>13</th>
      <td>11.617465</td>
      <td>15.103293</td>
    </tr>
    <tr>
      <th>14</th>
      <td>12.858327</td>
      <td>16.105966</td>
    </tr>
    <tr>
      <th>15</th>
      <td>13.652358</td>
      <td>16.148926</td>
    </tr>
    <tr>
      <th>16</th>
      <td>14.442286</td>
      <td>17.567963</td>
    </tr>
  </tbody>
</table>
<p>1547 rows × 2 columns</p>
</div>


    
    --------------------------------------------------------------------------------
    
    GoogleStocksDataLoader loads the following dataset:
    
    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([50, *, 5]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([50, 1]))
    )
    This contains:
    
    time_series:



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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
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
      <th>0.875000</th>
      <td>0.661264</td>
      <td>0.652789</td>
      <td>0.677836</td>
      <td>0.696887</td>
      <td>0.185147</td>
    </tr>
    <tr>
      <th>0.886364</th>
      <td>0.667446</td>
      <td>0.716935</td>
      <td>0.731552</td>
      <td>0.748318</td>
      <td>0.150912</td>
    </tr>
    <tr>
      <th>0.897727</th>
      <td>0.751374</td>
      <td>0.784055</td>
      <td>0.800261</td>
      <td>0.791407</td>
      <td>0.140203</td>
    </tr>
    <tr>
      <th>0.909091</th>
      <td>0.785577</td>
      <td>0.838572</td>
      <td>0.831813</td>
      <td>0.832628</td>
      <td>0.244291</td>
    </tr>
    <tr>
      <th>0.920455</th>
      <td>0.885578</td>
      <td>0.879778</td>
      <td>0.900782</td>
      <td>0.889539</td>
      <td>0.413625</td>
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
      <th rowspan="5" valign="top">9</th>
      <th>0.806818</th>
      <td>0.642857</td>
      <td>0.647974</td>
      <td>0.649153</td>
      <td>0.639975</td>
      <td>0.625178</td>
    </tr>
    <tr>
      <th>0.818182</th>
      <td>0.687362</td>
      <td>0.757221</td>
      <td>0.741200</td>
      <td>0.789788</td>
      <td>0.333141</td>
    </tr>
    <tr>
      <th>0.829545</th>
      <td>0.756044</td>
      <td>0.732512</td>
      <td>0.772230</td>
      <td>0.732379</td>
      <td>0.120629</td>
    </tr>
    <tr>
      <th>0.840909</th>
      <td>0.710852</td>
      <td>0.687907</td>
      <td>0.721525</td>
      <td>0.713076</td>
      <td>0.101900</td>
    </tr>
    <tr>
      <th>0.875000</th>
      <td>0.661264</td>
      <td>0.652789</td>
      <td>0.677836</td>
      <td>0.696887</td>
      <td>0.185147</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 5 columns</p>
</div>


    predictive.targets:



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
      <th>out</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.710852</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.756044</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.564835</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.557005</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.552061</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.510852</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.451786</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.421704</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.387225</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.345879</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.286951</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.332143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.687362</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.205906</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.286676</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.247939</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.492445</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.767858</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.810440</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.697940</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.597390</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.390659</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.385989</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.642857</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.361401</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.370879</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.388325</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.393819</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.389149</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.359753</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.399038</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.378984</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.225962</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.099863</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.628297</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.131181</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.054121</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0.062088</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0.204533</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.163049</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.166072</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.186126</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.233929</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.246566</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.671978</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.704808</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.684753</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.684753</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.607281</td>
    </tr>
  </tbody>
</table>
</div>


    
    --------------------------------------------------------------------------------
    
    PBCDataLoader loads the following dataset:
    
    TimeToEventAnalysisDataset(
        time_series=TimeSeriesSamples([312, *, 14]),
        static=StaticSamples([312, 1]),
        predictive=TimeToEventAnalysisTaskData(targets=EventSamples([312, 1]))
    )
    This contains:
    
    time_series:



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
      <th>drug</th>
      <th>ascites</th>
      <th>hepatomegaly</th>
      <th>spiders</th>
      <th>edema</th>
      <th>histologic</th>
      <th>serBilir</th>
      <th>serChol</th>
      <th>albumin</th>
      <th>alkaline</th>
      <th>SGOT</th>
      <th>platelets</th>
      <th>prothrombin</th>
      <th>age</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th>time_idx</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">1</th>
      <th>0.569489</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.281890</td>
      <td>0.000000</td>
      <td>-0.894575</td>
      <td>0.195532</td>
      <td>-1.485263</td>
      <td>-0.529101</td>
      <td>0.136768</td>
      <td>0.248058</td>
    </tr>
    <tr>
      <th>1.095170</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.015877</td>
      <td>-0.469461</td>
      <td>-1.570646</td>
      <td>0.285613</td>
      <td>0.195488</td>
      <td>-0.456022</td>
      <td>0.813132</td>
      <td>0.248058</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2</th>
      <th>5.319790</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.172710</td>
      <td>-0.658914</td>
      <td>-1.431455</td>
      <td>-0.605844</td>
      <td>-0.442126</td>
      <td>-1.395605</td>
      <td>0.339677</td>
      <td>1.292856</td>
    </tr>
    <tr>
      <th>6.261636</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>-0.013468</td>
      <td>-0.603657</td>
      <td>-1.172958</td>
      <td>-0.512364</td>
      <td>-0.046806</td>
      <td>-1.259888</td>
      <td>0.339677</td>
      <td>1.292856</td>
    </tr>
    <tr>
      <th>7.266455</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.098239</td>
      <td>0.000000</td>
      <td>-1.312149</td>
      <td>-0.443529</td>
      <td>0.293680</td>
      <td>-1.364286</td>
      <td>0.339677</td>
      <td>1.292856</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">312</th>
      <th>1.045888</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.672865</td>
      <td>3.319599</td>
      <td>0.059878</td>
      <td>1.385274</td>
      <td>0.986129</td>
      <td>-1.103291</td>
      <td>1.624769</td>
      <td>-1.962482</td>
    </tr>
    <tr>
      <th>1.867265</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.350998</td>
      <td>2.901224</td>
      <td>-0.099197</td>
      <td>0.916176</td>
      <td>0.641817</td>
      <td>-0.998892</td>
      <td>1.354223</td>
      <td>-1.962482</td>
    </tr>
    <tr>
      <th>2.921367</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.694010</td>
      <td>-0.066873</td>
      <td>0.338261</td>
      <td>0.327254</td>
      <td>0.552551</td>
      <td>-0.894494</td>
      <td>0.474950</td>
      <td>-1.962482</td>
    </tr>
    <tr>
      <th>3.425145</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.340271</td>
      <td>0.000000</td>
      <td>-0.377580</td>
      <td>0.251620</td>
      <td>0.016956</td>
      <td>-0.466462</td>
      <td>-0.066141</td>
      <td>-1.962482</td>
    </tr>
    <tr>
      <th>3.989158</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.507832</td>
      <td>2.017110</td>
      <td>0.795603</td>
      <td>0.622990</td>
      <td>0.169983</td>
      <td>-0.351624</td>
      <td>-0.133778</td>
      <td>-1.962482</td>
    </tr>
  </tbody>
</table>
<p>1945 rows × 14 columns</p>
</div>


    static:



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
      <th>sex</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
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
      <th>5</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>308</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>309</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>310</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>311</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>312</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>312 rows × 1 columns</p>
</div>


    predictive.targets:



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


    
    --------------------------------------------------------------------------------
    
    PKPDDataLoader loads the following dataset:
    
    Generating simple PKPD dataset with random seed 100...
    OneOffTreatmentEffectsDataset(
        time_series=TimeSeriesSamples([40, *, 2]),
        predictive=OneOffTreatmentEffectsTaskData(
            targets=TimeSeriesSamples([40, *, 1]),
            treatments=EventSamples([40, 1])
        )
    )
    This contains:
    
    time_series:



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
      <th>k_in</th>
      <th>p</th>
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
      <td>-0.781441</td>
      <td>-0.245827</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.001889</td>
      <td>-0.541524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.070862</td>
      <td>-0.589326</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.425115</td>
      <td>-1.065485</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.841006</td>
      <td>-1.542429</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">39</th>
      <th>5</th>
      <td>0.959902</td>
      <td>-0.690057</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.683426</td>
      <td>-0.128967</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.233045</td>
      <td>0.637906</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.645018</td>
      <td>1.056963</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.333051</td>
      <td>1.048721</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 2 columns</p>
</div>


    predictive.targets:



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
      <th>y</th>
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
      <td>-0.197049</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.020346</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.281120</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.483934</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.947253</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">39</th>
      <th>5</th>
      <td>-1.418583</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-1.495843</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.193632</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.850845</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.431988</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 1 columns</p>
</div>


    predictive.treatments:



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
      <th>a</th>
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>(7, False)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>22</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>25</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>26</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>27</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>28</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>29</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>30</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>31</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>32</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>33</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>34</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>35</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>36</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>37</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>(7, True)</td>
    </tr>
    <tr>
      <th>39</th>
      <td>(7, True)</td>
    </tr>
  </tbody>
</table>
</div>


    
    --------------------------------------------------------------------------------
    
    SineDataLoader loads the following dataset:
    
    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([100, *, 5]),
        static=StaticSamples([100, 4]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([100, 1]))
    )
    This contains:
    
    time_series:



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
      <td>-0.019015</td>
      <td>-0.048177</td>
      <td>-0.108546</td>
      <td>0.441865</td>
      <td>0.024508</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.300030</td>
      <td>0.364550</td>
      <td>0.576590</td>
      <td>0.890053</td>
      <td>0.534722</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.587904</td>
      <td>0.713509</td>
      <td>0.972993</td>
      <td>0.986179</td>
      <td>0.892946</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.814697</td>
      <td>0.937661</td>
      <td>0.882158</td>
      <td>0.692221</td>
      <td>0.997357</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.956846</td>
      <td>0.997797</td>
      <td>0.349572</td>
      <td>0.124454</td>
      <td>0.818278</td>
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
      <td>0.967121</td>
      <td>0.126890</td>
      <td>0.926979</td>
      <td>0.982022</td>
      <td>0.963113</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.706533</td>
      <td>0.413329</td>
      <td>0.569034</td>
      <td>0.656214</td>
      <td>0.989224</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.252748</td>
      <td>0.663121</td>
      <td>0.024381</td>
      <td>0.050668</td>
      <td>0.999771</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.270150</td>
      <td>0.854119</td>
      <td>-0.528273</td>
      <td>-0.576478</td>
      <td>0.994586</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.719177</td>
      <td>0.969389</td>
      <td>-0.907592</td>
      <td>-0.957876</td>
      <td>0.973752</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 5 columns</p>
</div>


    static:



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
      <td>0.212339</td>
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
      <td>0.118165</td>
      <td>0.696737</td>
      <td>0.628943</td>
      <td>0.877472</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.735071</td>
      <td>0.803481</td>
      <td>0.282035</td>
      <td>0.177440</td>
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


    predictive.targets:



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
    </tr>
    <tr>
      <th>sample_idx</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 1 columns</p>
</div>


