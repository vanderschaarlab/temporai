<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/temporai.svg?branch=main)](https://cirrus-ci.com/github/<USER>/temporai)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/temporai/main.svg)](https://coveralls.io/r/<USER>/temporai)
[![PyPI-Server](https://img.shields.io/pypi/v/temporai.svg)](https://pypi.org/project/temporai/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/temporai.svg)](https://anaconda.org/conda-forge/temporai)
[![Monthly Downloads](https://pepy.tech/badge/temporai/month)](https://pepy.tech/project/temporai)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/temporai)
-->

<!-- exclude_docs -->
<!-- [![ReadTheDocs](https://readthedocs.org/projects/temporai/badge/?version=latest)][docs] -->
[![Tests](https://github.com/vanderschaarlab/temporai/actions/workflows/test.yml/badge.svg)](https://github.com/vanderschaarlab/temporai/actions/workflows/test.yml)
[![](https://pepy.tech/badge/temporai)](https://pypi.org/project/temporai/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License](https://img.shields.io/github/license/vanderschaarlab/temporai)](./LICENSE.txt)
[![about](https://img.shields.io/badge/about-The%20van%20der%20Schaar%20Lab-blue)](https://www.vanderschaar-lab.com/)
[![slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://vanderschaarlab.slack.com)

<!-- exclude_docs_end -->

# <img src="docs/assets/TemporAI_Logo_Icon.png" height=25> TemporAI

> **⚠️ Status**: **Work in progress, please come back at a later date.** We are migrating from a [previous iteration of the project](https://github.com/vanderschaarlab/clairvoyance).

*TemporAI* is a Machine Learning-centric time-series library for medicine.  The tasks that are currently of focus in TemporAI are: time-series prediction, time-to-event (a.k.a. survival) analysis with time-series data, and counterfactual inference (i.e. \[individualized\] treatment effects).  The library also aims to provide the user with understanding of their data, model, and problem, through e.g. integration with interpretability methods.

Key concepts:

<div align="center">

<!-- exclude_docs -->
<img src="docs/assets/Conceptual.png" width="600" alt="key concepts">
<!-- exclude_docs_end -->
<!-- include_docs
<img src="docs/assets/Conceptual.png" width="750" alt="key concepts">
include_docs_end -->

</div>

## 🚀 Installation

```bash
$ pip install temporai
```
or from source, using
```bash
$ pip install .
```

## 💥 Sample Usage
* List the available plugins
```python
from tempor.plugins import plugin_loader

print(plugin_loader.list())
```

* Use an imputer
```python
from tempor.utils.dataloaders import SineDataLoader
from tempor.plugins import plugin_loader

dataset = SineDataLoader(with_missing=True).load()
assert dataset.static.dataframe().isna().sum().sum() != 0
assert dataset.time_series.dataframe().isna().sum().sum() != 0

# load the model
model = plugin_loader.get("preprocessing.imputation.temporal.bfill")

# train
model.fit(dataset)

# impute
imputed = model.transform(dataset)
assert imputed.static.dataframe().isna().sum().sum() == 0
assert imputed.time_series.dataframe().isna().sum().sum() == 0
```

* Use a classifier
```python
from tempor.utils.dataloaders import SineDataLoader
from tempor.plugins import plugin_loader

dataset = SineDataLoader().load()

# load the model
model = plugin_loader.get("prediction.one_off.classification.nn_classifier", n_iter=50)

# train
model.fit(dataset)

# predict
assert model.predict(dataset).numpy().shape == (len(dataset), 1)
```

* Use a regressor
```python
from tempor.utils.dataloaders import SineDataLoader
from tempor.plugins import plugin_loader

dataset = SineDataLoader().load()

# load the model
model = plugin_loader.get("prediction.one_off.regression.nn_regressor", n_iter=50)

# train
model.fit(dataset)

# predict
assert model.predict(dataset).numpy().shape == (len(dataset), 1)
```

* Benchmark models
Classification task
```python
from tempor.benchmarks import (
    benchmark_models,
)
from tempor.plugins import plugin_loader
from tempor.plugins.pipeline import Pipeline
from tempor.utils.dataloaders import (
    SineDataLoader,
)

testcases = [
    (
        "pipeline1",
        Pipeline(
            [
                "preprocessing.scaling.static.static_minmax_scaler",
                "prediction.one_off.classification.nn_classifier",
            ]
        )({"nn_classifier": {"n_iter": 10}}),
    ),
    (
        "plugin1",
        plugin_loader.get("prediction.one_off.classification.nn_classifier", n_iter=10),
    ),
]
dataset = SineDataLoader().load()

aggr_score, per_test_score = benchmark_models(
    task_type="classification",
    tests=testcases,
    data=dataset,
    n_splits=2,
    random_state=0,
)

print(aggr_score)
```

* Serialization
```python
from tempor.utils.serialization import load, save
from tempor.plugins import plugin_loader

# load the model
model = plugin_loader.get("prediction.one_off.classification.nn_classifier", n_iter=50)

buff = save(model)  # save model to bytes
reloaded = load(buff)  # reload model

# save_to_file. load_from_file also available in the serialization module
```

## 🔑 Methods



### Prediction

#### One-off
Prediction where targets are static.

* Classification (category: `prediction.one_off.classification`)

| Name | Description| Reference |
| --- | --- | --- |
| `nn_classifier` | Neural-net based classifier. Supports multiple recurrent models, like RNN, LSTM, Transformer etc.  | --- |
| `ode_classifier` | Classifier based on ordinary differential equation (ODE) solvers.  | --- |
| `cde_classifier` | Classifier based Neural Controlled Differential Equations for Irregular Time Series.  | [Paper](https://arxiv.org/abs/2005.08926) |
| `laplace_ode_classifier` | Classifier based Inverse Laplace Transform (ILT) algorithms implemented in PyTorch.  | [Paper](https://arxiv.org/abs/2206.04843) |

* Regression (category: `prediction.one_off.regression`)

| Name | Description| Reference |
| --- | --- | --- |
| `nn_regressor` | Neural-net based regressor. Supports multiple recurrent models, like RNN, LSTM, Transformer etc.  | --- |
| `ode_regressor` | Regressor based on ordinary differential equation (ODE) solvers.  | --- |
| `cde_regressor` | Regressor based Neural Controlled Differential Equations for Irregular Time Series.  | [Paper](https://arxiv.org/abs/2005.08926)
| `laplace_ode_regressor` | Regressor based Inverse Laplace Transform (ILT) algorithms implemented in PyTorch.  | [Paper](https://arxiv.org/abs/2206.04843) |

#### Temporal
Prediction where targets are temporal (time series).

* Classification (category: `prediction.temporal.classification`)

| Name | Description| Reference |
| --- | --- | --- |
| `seq2seq_classifier` | Seq2Seq prediction, classification | --- |

* Regression (category: `prediction.temporal.regression`)

| Name | Description| Reference |
| --- | --- | --- |
| `seq2seq_regressor` | Seq2Seq prediction, regression | --- |



### Time-to-Event

Risk estimation given event data (category: `time_to_event`)

| Name | Description| Reference |
| --- | --- | --- |
| `dynamic_deephit` | Dynamic-DeepHit incorporates the available longitudinal data comprising various repeated measurements (rather than only the last available measurements) in order to issue dynamically updated survival predictions | [Paper](https://pubmed.ncbi.nlm.nih.gov/30951460/) |
| `ts_coxph` | Create embeddings from the time series and use a CoxPH model for predicting the survival function| --- |
| `ts_xgb` | Create embeddings from the time series and use a SurvivalXGBoost model for predicting the survival function| --- |



### Treatment effects

#### One-off
Treatment effects estimation where treatments are a one-off event.

<!--
* Classification on the outcomes (category: `treatments.one_off.classification`)
-->

* Regression on the outcomes (category: `treatments.one_off.regression`)

| Name | Description| Reference |
| --- | --- | --- |
| `synctwin_regressor` | SyncTwin is a treatment effect estimation method tailored for observational studies with longitudinal data, applied to the LIP setting: Longitudinal, Irregular and Point treatment.  | [Paper](https://proceedings.neurips.cc/paper/2021/hash/19485224d128528da1602ca47383f078-Abstract.html) |

#### Temporal
Treatment effects estimation where treatments are temporal (time series).

* Classification on the outcomes (category: `treatments.temporal.classification`)

| Name | Description| Reference |
| --- | --- | --- |
| `crn_classifier` | The Counterfactual Recurrent Network (CRN), a sequence-to-sequence model that leverages the available patient observational data to estimate treatment effects over time. | [Paper](https://arxiv.org/abs/2002.04083) |

* Regression on the outcomes (category: `treatments.temporal.regression`)

| Name | Description| Reference |
| --- | --- | --- |
| `crn_regressor` | The Counterfactual Recurrent Network (CRN), a sequence-to-sequence model that leverages the available patient observational data to estimate treatment effects over time. | [Paper](https://arxiv.org/abs/2002.04083) |



### Preprocessing

#### Imputation

* Static data (category: `preprocessing.imputation.static`)

| Name | Description| Reference |
| --- | --- | --- |
| `static_imputation` | Use HyperImpute to impute both the static and temporal data | [Paper](https://arxiv.org/abs/2206.07769) |

* Temporal data (category: `preprocessing.imputation.temporal`)

| Name | Description| Reference |
| --- | --- | --- |
| `ffill` | Propagate last valid observation forward to next valid  | --- |
| `bfill` | Use next valid observation to fill gap | --- |

### Scaling

* Static data (category: `preprocessing.scaling.static`)

| Name | Description| Reference |
| --- | --- | --- |
| `static_standard_scaler` | Scale the static features using a StandardScaler | --- |
| `static_minmax_scaler` | Scale the static features using a MinMaxScaler | --- |

* Temporal data (category: `preprocessing.scaling.temporal`)

| Name | Description| Reference |
| --- | --- | --- |
| `ts_standard_scaler` | Scale the temporal features using a StandardScaler | --- |
| `ts_minmax_scaler` | Scale the temporal features using a MinMaxScaler | --- |



## TemporAI Pipeline
The diagram below illustrates the structure of a *TemporAI* pipeline:

<img src="docs/assets/Pipeline.png" alt="pipeline diagram">

<!--
See [User Guide][docs/user_guide] for tutorials/examples.
-->

<!-- exclude_docs -->
<!--
## Documentation

📖 See the project documentation [here](https://temporai.readthedocs.io/en/latest/).
-->
<!-- exclude_docs_end -->



<!--- Reusable --->
  [van der Schaar Lab]:    https://www.vanderschaar-lab.com/
  [docs]:                  https://temporai.readthedocs.io/en/latest/
<!-- exclude_docs -->
  [docs/user_guide]:       https://temporai.readthedocs.io/en/latest/user_guide/index.html
<!-- exclude_docs_end -->

## 🔨 Tests

Install the testing dependencies using
```bash
pip install .[dev]
```
The tests can be executed using
```bash
pytest -vsx
```

## Citing

If you use this code, please cite the associated paper:
```
TODO
```
