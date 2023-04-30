<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/temporai.svg?branch=main)](https://cirrus-ci.com/github/<USER>/temporai)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/temporai/main.svg)](https://coveralls.io/r/<USER>/temporai)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/temporai.svg)](https://anaconda.org/conda-forge/temporai)
[![Monthly Downloads](https://pepy.tech/badge/temporai/month)](https://pepy.tech/project/temporai)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/temporai)
-->

<!-- exclude_docs -->
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/user_guide/tutorial04_prediction.ipynb)
[![Documentation Status](https://readthedocs.org/projects/temporai/badge/?version=latest)](https://temporai.readthedocs.io/en/latest/?badge=latest)

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyPI-Server](https://img.shields.io/pypi/v/temporai?color=blue)](https://pypi.org/project/temporai/)
[![](https://pepy.tech/badge/temporai)](https://pypi.org/project/temporai/)
[![Tests](https://github.com/vanderschaarlab/temporai/actions/workflows/test.yml/badge.svg)](https://github.com/vanderschaarlab/temporai/actions/workflows/test.yml)
[![Tests](https://github.com/vanderschaarlab/temporai/actions/workflows/test_full.yml/badge.svg)](https://github.com/vanderschaarlab/temporai/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/vanderschaarlab/temporai/LICENSE.txt)

[![arXiv](https://img.shields.io/badge/arXiv-2301.12260-b31b1b.svg)](https://arxiv.org/abs/2301.12260)
[![about](https://img.shields.io/badge/about-The%20van%20der%20Schaar%20Lab-blue)](https://www.vanderschaar-lab.com/)
[![slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://join.slack.com/t/vanderschaarlab/shared_invite/zt-1u2rmhw06-sHS5nQDMN3Ka2Zer6sAU6Q)
<!-- exclude_docs_end -->

# <img src='https://raw.githubusercontent.com/vanderschaarlab/temporai/main/docs/assets/TemporAI_Logo_Icon.png' height=25> TemporAI

> **⚗️ Status:** This project is still in *alpha*, and the API may change without warning.  

*TemporAI* is a Machine Learning-centric time-series library for medicine.  The tasks that are currently of focus in TemporAI are: time-series prediction, time-to-event (a.k.a. survival) analysis with time-series data, and counterfactual inference (i.e. \[individualized\] treatment effects).

In future versions, the library also aims to provide the user with understanding of their data, model, and problem, through e.g. integration with interpretability methods.

Key concepts:

<div align="center">

<!-- exclude_docs -->
<img src='https://raw.githubusercontent.com/vanderschaarlab/temporai/main/docs/assets/Conceptual.png' width="650" alt="key concepts">
<!-- exclude_docs_end -->
<!-- include_docs
<img src='https://raw.githubusercontent.com/vanderschaarlab/temporai/main/docs/assets/Conceptual.png' width="750" alt="key concepts">
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
static_data_n_missing = dataset.static.dataframe().isna().sum().sum()
temporal_data_n_missing = dataset.time_series.dataframe().isna().sum().sum()

print(static_data_n_missing, temporal_data_n_missing)
assert static_data_n_missing > 0
assert temporal_data_n_missing > 0

# Load the model:
model = plugin_loader.get("preprocessing.imputation.temporal.bfill")

# Train:
model.fit(dataset)

# Impute:
imputed = model.transform(dataset)
static_data_n_missing = imputed.static.dataframe().isna().sum().sum()
temporal_data_n_missing = imputed.time_series.dataframe().isna().sum().sum()

print(static_data_n_missing, temporal_data_n_missing)
assert static_data_n_missing == 0
assert temporal_data_n_missing == 0
```

* Use a classifier
```python
from tempor.utils.dataloaders import SineDataLoader
from tempor.plugins import plugin_loader

dataset = SineDataLoader().load()

# Load the model:
model = plugin_loader.get("prediction.one_off.classification.nn_classifier", n_iter=50)

# Train:
model.fit(dataset)

# Predict:
prediction = model.predict(dataset)
```

* Use a regressor
```python
from tempor.utils.dataloaders import SineDataLoader
from tempor.plugins import plugin_loader

dataset = SineDataLoader().load()

# Load the model:
model = plugin_loader.get("prediction.one_off.regression.nn_regressor", n_iter=50)

# Train:
model.fit(dataset)

# Predict:
prediction = model.predict(dataset)
```

* Benchmark models
Classification task
```python
from tempor.benchmarks import benchmark_models
from tempor.plugins import plugin_loader
from tempor.plugins.pipeline import Pipeline
from tempor.utils.dataloaders import SineDataLoader

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

# Load the model:
model = plugin_loader.get("prediction.one_off.classification.nn_classifier", n_iter=50)

buff = save(model)  # Save model to bytes.
reloaded = load(buff)  # Reload model.

# `save_to_file`, `load_from_file` also available in the serialization module.
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
| `static_tabular_imputer` | Use any method from [HyperImpute](https://github.com/vanderschaarlab/hyperimpute) (HyperImpute, Mean, Median, Most-frequent, MissForest, ICE, MICE, SoftImpute, EM, Sinkhorn, GAIN, MIRACLE, MIWAE) to impute the static data | [Paper](https://arxiv.org/abs/2206.07769) |

* Temporal data (category: `preprocessing.imputation.temporal`)

| Name | Description| Reference |
| --- | --- | --- |
| `ffill` | Propagate last valid observation forward to next valid  | --- |
| `bfill` | Use next valid observation to fill gap | --- |
| `ts_tabular_imputer` | Use any method from [HyperImpute](https://github.com/vanderschaarlab/hyperimpute) (HyperImpute, Mean, Median, Most-frequent, MissForest, ICE, MICE, SoftImpute, EM, Sinkhorn, GAIN, MIRACLE, MIWAE) to impute the time series data | [Paper](https://arxiv.org/abs/2206.07769) |


#### Scaling

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



## Tutorials

### Data

- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/data/tutorial01_data_format.ipynb) - [Data Format](https://github.com/vanderschaarlab/temporai/tutorials/data/tutorial01_data_format.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/data/tutorial02_datasets.ipynb) - [Datasets](https://github.com/vanderschaarlab/temporai/tutorials/data/tutorial02_datasets.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/data/tutorial03_dataloaders.ipynb) - [Data Loaders](https://github.com/vanderschaarlab/temporai/tutorials/data/tutorial03_dataloaders.ipynb)

### User Guide
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/user_guide/tutorial01_plugins.ipynb) - [Plugins](https://github.com/vanderschaarlab/temporai/tutorials/user_guide/tutorial01_plugins.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/user_guide/tutorial02_imputation.ipynb) - [Imputation](https://github.com/vanderschaarlab/temporai/tutorials/user_guide/tutorial02_imputation.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/user_guide/tutorial03_scaling.ipynb) - [Scaling](https://github.com/vanderschaarlab/temporai/tutorials/user_guide/tutorial03_scaling.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/user_guide/tutorial04_prediction.ipynb) - [Prediction](https://github.com/vanderschaarlab/temporai/tutorials/user_guide/tutorial04_prediction.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/user_guide/tutorial05_time_to_event.ipynb) - [Time-to-event Analysis](https://github.com/vanderschaarlab/temporai/tutorials/user_guide/tutorial05_time_to_event.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/user_guide/tutorial06_treatments.ipynb) - [Treatment Effects](https://github.com/vanderschaarlab/temporai/tutorials/user_guide/tutorial06_treatments.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/user_guide/tutorial07_pipeline.ipynb) - [Pipeline](https://github.com/vanderschaarlab/temporai/tutorials/user_guide/tutorial07_pipeline.ipynb)
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/user_guide/tutorial08_benchmarks.ipynb) - [Benchmarks](https://github.com/vanderschaarlab/temporai/tutorials/user_guide/tutorial08_benchmarks.ipynb)

### Extending TemporAI
- [![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/extending/tutorial01_custom_plugin.ipynb) - [Writing a Custom Plugin](https://github.com/vanderschaarlab/temporai/tutorials/extending/tutorial01_custom_plugin.ipynb)



<!-- exclude_docs -->
## 📘 Documentation

See the project documentation [here](https://temporai.readthedocs.io/en/latest/).
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
@article{saveliev2023temporai,
  title={TemporAI: Facilitating Machine Learning Innovation in Time Domain Tasks for Medicine},
  author={Saveliev, Evgeny S and van der Schaar, Mihaela},
  journal={arXiv preprint arXiv:2301.12260},
  year={2023}
}
```
