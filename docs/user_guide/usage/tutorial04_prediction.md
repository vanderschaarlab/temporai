# User Guide Tutorial 04: Prediction
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/usage/tutorial04_prediction.ipynb)

This tutorial shows how to use TemporAI `prediction` plugins.



## All `prediction` plugins

To see all the relevant plugins:


```python
from tempor.plugins import plugin_loader
from rich.pretty import pprint

all_prediction_plugins = plugin_loader.list()["prediction"]

pprint(all_prediction_plugins, indent_guides=False)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'one_off'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'classification'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'cde_classifier'</span>, <span style="color: #008000; text-decoration-color: #008000">'ode_classifier'</span>, <span style="color: #008000; text-decoration-color: #008000">'nn_classifier'</span>, <span style="color: #008000; text-decoration-color: #008000">'laplace_ode_classifier'</span><span style="font-weight: bold">]</span>,
        <span style="color: #008000; text-decoration-color: #008000">'regression'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'laplace_ode_regressor'</span>, <span style="color: #008000; text-decoration-color: #008000">'nn_regressor'</span>, <span style="color: #008000; text-decoration-color: #008000">'ode_regressor'</span>, <span style="color: #008000; text-decoration-color: #008000">'cde_regressor'</span><span style="font-weight: bold">]</span>
    <span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'temporal'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'classification'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'seq2seq_classifier'</span><span style="font-weight: bold">]</span>, <span style="color: #008000; text-decoration-color: #008000">'regression'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'seq2seq_regressor'</span><span style="font-weight: bold">]}</span>
<span style="font-weight: bold">}</span>
</pre>



## Using a one-off prediction plugin

One-off prediction is the task of predicting a single value for each sample (may be classification or regression).


```python
from tempor.utils.dataloaders import SineDataLoader
from tempor.plugins import plugin_loader

dataset = SineDataLoader(random_state=42).load()
print(dataset)

model = plugin_loader.get("prediction.one_off.classification.nn_classifier", n_iter=50)
print(model)
```

    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([100, *, 5]),
        static=StaticSamples([100, 4]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([100, 1]))
    )
    NeuralNetClassifier(
        name='nn_classifier',
        category='prediction.one_off.classification',
        params={
            'n_static_units_hidden': 100,
            'n_static_layers_hidden': 2,
            'n_temporal_units_hidden': 102,
            'n_temporal_layers_hidden': 2,
            'n_iter': 50,
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



```python
# Targets:
dataset.predictive.targets
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




```python
# Train.
model.fit(dataset)
```




    NeuralNetClassifier(
        name='nn_classifier',
        category='prediction.one_off.classification',
        params={
            'n_static_units_hidden': 100,
            'n_static_layers_hidden': 2,
            'n_temporal_units_hidden': 102,
            'n_temporal_layers_hidden': 2,
            'n_iter': 50,
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




```python
# Predict:

model.predict(dataset)
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
      <td>0.0</td>
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
      <td>0.0</td>
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
      <td>0.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 1 columns</p>
</div>



## Using a temporal prediction plugin

Temporal prediction is the task of predicting a time series for each sample (may be classification or regression).


```python
from tempor.utils.dataloaders import DummyTemporalPredictionDataLoader
from tempor.plugins import plugin_loader

dataset = DummyTemporalPredictionDataLoader(random_state=42, temporal_covariates_missing_prob=0.0).load()
print(dataset)

model = plugin_loader.get("prediction.temporal.regression.seq2seq_regressor", epochs=10)
print(model)
```

    TemporalPredictionDataset(
        time_series=TimeSeriesSamples([100, *, 5]),
        static=StaticSamples([100, 3]),
        predictive=TemporalPredictionTaskData(
            targets=TimeSeriesSamples([100, *, 2])
        )
    )
    Seq2seqRegressor(
        name='seq2seq_regressor',
        category='prediction.temporal.regression',
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
            'epochs': 10,
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
# Train.
model.fit(dataset);
```

    Preparing data for decoder training...
    Preparing data for decoder training DONE.
    === Training stage: 1. Train encoder ===
    Epoch: 0, Loss: 68.64124755859375
    Epoch: 1, Loss: 35.37634651184082
    Epoch: 2, Loss: 21.269977149963378
    Epoch: 3, Loss: 17.418054847717286
    Epoch: 4, Loss: 13.851956520080567
    Epoch: 5, Loss: 8.335217247009277
    Epoch: 6, Loss: 6.385376853942871
    Epoch: 7, Loss: 5.112505340576172
    Epoch: 8, Loss: 4.690582332611084
    Epoch: 9, Loss: 4.158801422119141
    === Training stage: 2. Train decoder ===
    Epoch: 0, Loss: 29.110631188485222
    Epoch: 1, Loss: 4.417614046603097
    Epoch: 2, Loss: 3.908697671333463
    Epoch: 3, Loss: 3.8485680392058788
    Epoch: 4, Loss: 3.8276885692068126
    Epoch: 5, Loss: 3.793576312405029
    Epoch: 6, Loss: 3.749186389072364
    Epoch: 7, Loss: 3.787092407949102
    Epoch: 8, Loss: 3.7378030509586164
    Epoch: 9, Loss: 3.717736062595655



```python
# Predict:

model.predict(dataset, n_future_steps=10)
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
      <th>11</th>
      <td>9.828533</td>
      <td>8.405809</td>
    </tr>
    <tr>
      <th>12</th>
      <td>9.831676</td>
      <td>8.414574</td>
    </tr>
    <tr>
      <th>13</th>
      <td>9.813161</td>
      <td>8.395016</td>
    </tr>
    <tr>
      <th>14</th>
      <td>9.798813</td>
      <td>8.378744</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9.772892</td>
      <td>8.352385</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">99</th>
      <th>17</th>
      <td>11.214181</td>
      <td>9.754721</td>
    </tr>
    <tr>
      <th>18</th>
      <td>11.275115</td>
      <td>9.821613</td>
    </tr>
    <tr>
      <th>19</th>
      <td>11.298031</td>
      <td>9.848743</td>
    </tr>
    <tr>
      <th>20</th>
      <td>11.280746</td>
      <td>9.832634</td>
    </tr>
    <tr>
      <th>21</th>
      <td>11.256371</td>
      <td>9.801851</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 2 columns</p>
</div>



