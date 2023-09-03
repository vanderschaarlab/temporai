# User Guide Tutorial 09: Benchmarks
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/usage/tutorial08_benchmarks.ipynb)

TemporAI provides some useful benchmarking tools in `tempor.benchmarks`, these are demonstrated here.



## Using `tempor.benchmarks.benchmark_models`

The `tempor.benchmarks.benchmark_models` function provides a quick way to benchmark a number of models (plugins) for
a particular task.

It takes a list of models (these may also be a `Pipeline`) and a dataset, and performs cross-validation to
get the mean and standard deviation of the various metrics.

It returns a tuple `(results_readable, results)` as below.


```python
from tempor.benchmarks import benchmark_models
from tempor.utils.dataloaders import SineDataLoader
from tempor.plugins import plugin_loader

from IPython.display import display

dataset = SineDataLoader(random_state=42, no=25).load()

results_readable, results = benchmark_models(
    task_type="prediction.one_off.classification",
    tests=[
        ("model_1", plugin_loader.get("prediction.one_off.classification.nn_classifier", n_iter=10)),
        ("model_2", plugin_loader.get("prediction.one_off.classification.ode_classifier", n_iter=100)),
    ],
    data=dataset,
    n_splits=3,
)

print("Results in easily-readable format:")
display(results_readable)

print("Full results:\n")
for model, value in results.items():
    print(f"{model}:")
    display(value)
```

    2023-05-14 19:18:40 | INFO     | tempor.benchmarks.benchmark:benchmark_models:91 | Test case: model_1
    2023-05-14 19:18:42 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.68637615442276, validation loss: 0.6680578589439392
    2023-05-14 19:18:43 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6887814402580261, validation loss: 0.6939960718154907
    2023-05-14 19:18:43 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6906945109367371, validation loss: 0.6966598033905029
    2023-05-14 19:18:43 | INFO     | tempor.benchmarks.benchmark:benchmark_models:91 | Test case: model_2
    2023-05-14 19:18:48 | INFO     | tempor.models.ts_ode:_train:608 | Epoch:99| train loss: 0.6465951204299927, validation loss: 0.5632617473602295
    2023-05-14 19:18:52 | INFO     | tempor.models.ts_ode:_train:608 | Epoch:99| train loss: 0.913261890411377, validation loss: 0.8132617473602295
    2023-05-14 19:18:57 | INFO     | tempor.models.ts_ode:_train:608 | Epoch:99| train loss: 0.7132617831230164, validation loss: 0.8132617473602295


    Results in easily-readable format:



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
      <th>model_1</th>
      <th>model_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aucroc</th>
      <td>0.311 +/- 0.083</td>
      <td>0.583 +/- 0.312</td>
    </tr>
    <tr>
      <th>aucprc</th>
      <td>0.385 +/- 0.028</td>
      <td>0.588 +/- 0.292</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.602 +/- 0.033</td>
      <td>0.519 +/- 0.105</td>
    </tr>
    <tr>
      <th>f1_score_micro</th>
      <td>0.602 +/- 0.033</td>
      <td>0.519 +/- 0.105</td>
    </tr>
    <tr>
      <th>f1_score_macro</th>
      <td>0.375 +/- 0.013</td>
      <td>0.338 +/- 0.048</td>
    </tr>
    <tr>
      <th>f1_score_weighted</th>
      <td>0.453 +/- 0.04</td>
      <td>0.361 +/- 0.116</td>
    </tr>
    <tr>
      <th>kappa</th>
      <td>0.0 +/- 0.0</td>
      <td>0.0 +/- 0.0</td>
    </tr>
    <tr>
      <th>kappa_quadratic</th>
      <td>0.0 +/- 0.0</td>
      <td>0.0 +/- 0.0</td>
    </tr>
    <tr>
      <th>precision_micro</th>
      <td>0.602 +/- 0.033</td>
      <td>0.519 +/- 0.105</td>
    </tr>
    <tr>
      <th>precision_macro</th>
      <td>0.301 +/- 0.016</td>
      <td>0.259 +/- 0.053</td>
    </tr>
    <tr>
      <th>precision_weighted</th>
      <td>0.363 +/- 0.039</td>
      <td>0.28 +/- 0.104</td>
    </tr>
    <tr>
      <th>recall_micro</th>
      <td>0.602 +/- 0.033</td>
      <td>0.519 +/- 0.105</td>
    </tr>
    <tr>
      <th>recall_macro</th>
      <td>0.5 +/- 0.0</td>
      <td>0.5 +/- 0.0</td>
    </tr>
    <tr>
      <th>recall_weighted</th>
      <td>0.602 +/- 0.033</td>
      <td>0.519 +/- 0.105</td>
    </tr>
    <tr>
      <th>mcc</th>
      <td>0.0 +/- 0.0</td>
      <td>0.0 +/- 0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Full results:
    
    model_1:



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
      <th>mean</th>
      <th>stddev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aucroc</th>
      <td>0.311111</td>
      <td>0.083148</td>
    </tr>
    <tr>
      <th>aucprc</th>
      <td>0.385185</td>
      <td>0.028154</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.601852</td>
      <td>0.032736</td>
    </tr>
    <tr>
      <th>f1_score_micro</th>
      <td>0.601852</td>
      <td>0.032736</td>
    </tr>
    <tr>
      <th>f1_score_macro</th>
      <td>0.375458</td>
      <td>0.012951</td>
    </tr>
    <tr>
      <th>f1_score_weighted</th>
      <td>0.452788</td>
      <td>0.039572</td>
    </tr>
    <tr>
      <th>kappa</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>kappa_quadratic</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>precision_micro</th>
      <td>0.601852</td>
      <td>0.032736</td>
    </tr>
    <tr>
      <th>precision_macro</th>
      <td>0.300926</td>
      <td>0.016368</td>
    </tr>
    <tr>
      <th>precision_weighted</th>
      <td>0.363297</td>
      <td>0.038647</td>
    </tr>
    <tr>
      <th>recall_micro</th>
      <td>0.601852</td>
      <td>0.032736</td>
    </tr>
    <tr>
      <th>recall_macro</th>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>recall_weighted</th>
      <td>0.601852</td>
      <td>0.032736</td>
    </tr>
    <tr>
      <th>mcc</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


    model_2:



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
      <th>mean</th>
      <th>stddev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aucroc</th>
      <td>0.583333</td>
      <td>0.311805</td>
    </tr>
    <tr>
      <th>aucprc</th>
      <td>0.587731</td>
      <td>0.291568</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.518519</td>
      <td>0.105369</td>
    </tr>
    <tr>
      <th>f1_score_micro</th>
      <td>0.518519</td>
      <td>0.105369</td>
    </tr>
    <tr>
      <th>f1_score_macro</th>
      <td>0.338162</td>
      <td>0.047609</td>
    </tr>
    <tr>
      <th>f1_score_weighted</th>
      <td>0.360713</td>
      <td>0.115623</td>
    </tr>
    <tr>
      <th>kappa</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>kappa_quadratic</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>precision_micro</th>
      <td>0.518519</td>
      <td>0.105369</td>
    </tr>
    <tr>
      <th>precision_macro</th>
      <td>0.259259</td>
      <td>0.052684</td>
    </tr>
    <tr>
      <th>precision_weighted</th>
      <td>0.279964</td>
      <td>0.104057</td>
    </tr>
    <tr>
      <th>recall_micro</th>
      <td>0.518519</td>
      <td>0.105369</td>
    </tr>
    <tr>
      <th>recall_macro</th>
      <td>0.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>recall_weighted</th>
      <td>0.518519</td>
      <td>0.105369</td>
    </tr>
    <tr>
      <th>mcc</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


## Supported tasks

> ⚠️ Not all task types are supported by `benchmark_models` yet.

Supported tasks (for each `task_type` argument):
* `task_type="prediction.one_off.classification"`.
* `task_type="prediction.one_off.regression"`.
* `task_type="time_to_event"`.


