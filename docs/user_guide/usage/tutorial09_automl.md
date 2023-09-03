# User Guide Tutorial 09: AutoML
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/usage/tutorial09_automl.ipynb)

TemporAI provides AutoML tools for finding the best model for your use case in `tempor.automl`, these are demonstrated here.



## AutoML in TemporAI Overview

TemporAI provides two AutoML approaches ("seekers") under the `tempor.automl.seekers` module.

1. `MethodSeeker`: Search the hyperparameter space of a particular predictive method.
2. `PipelineSeeker`: Search the hyperparameter space of a pipeline like `preprocessing steps -> predictive step`.

The optimization strategies are facilitated by [`optuna`](https://optuna.readthedocs.io/) and the currently supported strategies are:
* [Bayesian, specifically Tree-structured Parzen estimator](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) (`"bayesian"`),
* [Random](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html) (`"random"`),
* [CMA-ES](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html) (`"cmaes"`),
* [QMC](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.QMCSampler.html) (`"qmc"`),
* [Grid](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GridSampler.html) (`"grid"`).

## Using `MethodSeeker`

Use a `MethodSeeker` to search for best algorithm and hyperparameters parameters for a particular task.
No preprocessing (data transformation) steps are carried out in this approach, so preprocess the data using
`tempor.plugins.preprocessing` first, as needed.

A `MethodSeeker` can be initialized as follows.


```python
from tempor.automl.seeker import MethodSeeker
from tempor.utils.dataloaders import SineDataLoader

# Load your dataset.
dataset = SineDataLoader().load()

seeker = MethodSeeker(
    # Name your AutoML study:
    study_name="my_automl_study",
    # Select the type of task:
    task_type="prediction.one_off.classification",
    # Choose which predictive methods to use in the search:
    estimator_names=[
        "cde_classifier",
        "ode_classifier",
        "nn_classifier",
    ],
    # Choose a metric. Metric maximization/minimization will be determined automatically.
    metric="aucroc",
    # Pass in your dataset.
    dataset=dataset,
    # How many best models to return:
    return_top_k=3,
    # Number of AutoML iterations:
    num_iter=100,
    # Type of AutoML tuner to use:
    tuner_type="bayesian",
    # You can also provide some other options like early stopping patience, number of cross-validation folds etc.
)
```

    2023-05-14 21:48:44 | INFO     | tempor.automl.seeker:_set_up_tuners:354 | Setting up estimators and tuners for study my_automl_study.
    2023-05-14 21:48:44 | INFO     | tempor.automl.seeker:_init_estimator:563 | Creating estimator cde_classifier.
    2023-05-14 21:48:44 | INFO     | tempor.automl.seeker:_init_estimator:563 | Creating estimator ode_classifier.
    2023-05-14 21:48:44 | INFO     | tempor.automl.seeker:_init_estimator:563 | Creating estimator nn_classifier.


You can then run the AutoML search as below.

The below example also shows how you can provide a custom hyperparameter space (*override* the default hyperparameter
space for a model).


```python
from tempor.automl.params import IntegerParams, CategoricalParams

# Provide a custom hyperparameter space to search for each type of model.
# NOTE: For the sake of speed of this example, we limit epochs to 2.
hp_space = {
    "cde_classifier": [
        IntegerParams(name="n_iter", low=2, high=2),
        IntegerParams(name="n_temporal_units_hidden", low=5, high=20),
        CategoricalParams(name="lr", choices=[1e-2, 1e-3, 1e-4]),
    ],
    "ode_classifier": [
        IntegerParams(name="n_iter", low=2, high=2),
        IntegerParams(name="n_units_hidden", low=5, high=20),
        CategoricalParams(name="lr", choices=[1e-2, 1e-3, 1e-4]),
    ],
    "nn_classifier": [
        IntegerParams(name="n_iter", low=2, high=2),
        IntegerParams(name="n_units_hidden", low=5, high=20),
        CategoricalParams(name="lr", choices=[1e-2, 1e-3, 1e-4]),
    ],
}

# Initialize a `MethodSeeker` and provide `override_hp_space`.
seeker = MethodSeeker(
    study_name="my_automl_study",
    task_type="prediction.one_off.classification",
    estimator_names=[
        "cde_classifier",
        "ode_classifier",
        "nn_classifier",
    ],
    metric="aucroc",
    dataset=dataset,
    return_top_k=3,
    num_iter=3,  # For the sake of speed of this example, only 3 AutoML iterations.
    tuner_type="bayesian",
    # Override hyperparameter space:
    override_hp_space=hp_space,
)
```

    2023-05-14 21:48:44 | INFO     | tempor.automl.seeker:_set_up_tuners:354 | Setting up estimators and tuners for study my_automl_study.
    2023-05-14 21:48:44 | INFO     | tempor.automl.seeker:_init_estimator:563 | Creating estimator cde_classifier.
    2023-05-14 21:48:44 | INFO     | tempor.automl.seeker:_init_estimator:563 | Creating estimator ode_classifier.
    2023-05-14 21:48:44 | INFO     | tempor.automl.seeker:_init_estimator:563 | Creating estimator nn_classifier.



```python
# Execute the search.

best_methods, best_scores = seeker.search()
```

    2023-05-14 21:48:44 | INFO     | tempor.automl.seeker:search:413 | Running  search for estimator 'cde_classifier' 1/3.
    2023-05-14 21:48:44 | INFO     | tempor.automl.tuner:tune:205 | Baseline score computation skipped
    2023-05-14 21:48:44 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from CDEClassifier:
    {'n_iter': 2, 'n_temporal_units_hidden': 13, 'lr': 0.01}
    2023-05-14 21:48:49 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from CDEClassifier:
    {'n_iter': 2, 'n_temporal_units_hidden': 11, 'lr': 0.0001}
    2023-05-14 21:48:53 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from CDEClassifier:
    {'n_iter': 2, 'n_temporal_units_hidden': 20, 'lr': 0.001}
    2023-05-14 21:48:56 | INFO     | tempor.automl.seeker:search:413 | Running  search for estimator 'ode_classifier' 2/3.
    2023-05-14 21:48:56 | INFO     | tempor.automl.tuner:tune:205 | Baseline score computation skipped
    2023-05-14 21:48:56 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from ODEClassifier:
    {'n_iter': 2, 'n_units_hidden': 13, 'lr': 0.01}
    2023-05-14 21:48:59 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from ODEClassifier:
    {'n_iter': 2, 'n_units_hidden': 11, 'lr': 0.0001}
    2023-05-14 21:49:02 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from ODEClassifier:
    {'n_iter': 2, 'n_units_hidden': 20, 'lr': 0.001}
    2023-05-14 21:49:05 | INFO     | tempor.automl.seeker:search:413 | Running  search for estimator 'nn_classifier' 3/3.
    2023-05-14 21:49:05 | INFO     | tempor.automl.tuner:tune:205 | Baseline score computation skipped
    2023-05-14 21:49:05 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from NeuralNetClassifier:
    {'n_iter': 2, 'n_units_hidden': 13, 'lr': 0.01}
    2023-05-14 21:49:06 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6919505596160889, validation loss: 0.6882655620574951
    2023-05-14 21:49:07 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.691156268119812, validation loss: 0.6865323185920715
    2023-05-14 21:49:07 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.692247211933136, validation loss: 0.6861071586608887
    2023-05-14 21:49:08 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6928325891494751, validation loss: 0.6825112104415894
    2023-05-14 21:49:08 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6919053792953491, validation loss: 0.6819757223129272
    2023-05-14 21:49:08 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from NeuralNetClassifier:
    {'n_iter': 2, 'n_units_hidden': 11, 'lr': 0.0001}
    2023-05-14 21:49:09 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6919505596160889, validation loss: 0.691270649433136
    2023-05-14 21:49:09 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.691156268119812, validation loss: 0.6910805106163025
    2023-05-14 21:49:10 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.692247211933136, validation loss: 0.6900344491004944
    2023-05-14 21:49:10 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6928325891494751, validation loss: 0.6906693577766418
    2023-05-14 21:49:11 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6919053792953491, validation loss: 0.6910874843597412
    2023-05-14 21:49:11 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from NeuralNetClassifier:
    {'n_iter': 2, 'n_units_hidden': 20, 'lr': 0.001}
    2023-05-14 21:49:11 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6919505596160889, validation loss: 0.6896670460700989
    2023-05-14 21:49:12 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.691156268119812, validation loss: 0.6886401772499084
    2023-05-14 21:49:12 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.692247211933136, validation loss: 0.6884845495223999
    2023-05-14 21:49:13 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6928325891494751, validation loss: 0.6887120604515076
    2023-05-14 21:49:13 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6919053792953491, validation loss: 0.6893669962882996
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:search:447 | 
    Evaluation for cde_classifier scores:
    [0.47310606060606064, 0.48396464646464643, 0.4840277777777778].
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:search:447 | 
    Evaluation for ode_classifier scores:
    [0.47992424242424236, 0.48409090909090907, 0.484090909090909].
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:search:447 | 
    Evaluation for nn_classifier scores:
    [0.37064393939393936, 0.5370580808080809, 0.4579545454545454].
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:search:448 | 
    All estimator definitions searched:
    ['cde_classifier', 'ode_classifier', 'nn_classifier']
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:search:449 | 
    Best scores for each estimator searched:
    [0.4840277777777778, 0.48409090909090907, 0.5370580808080809]
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:search:450 | 
    Best hyperparameters for each estimator searched:
    [{'n_iter': 2, 'n_temporal_units_hidden': 20, 'lr': 0.001}, {'n_iter': 2, 'n_units_hidden': 11, 'lr': 0.0001}, {'n_iter': 2, 'n_units_hidden': 11, 'lr': 0.0001}]
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:_create_estimator_with_hps:571 | 
    Selected score 0.5370580808080809 for nn_classifier with hyperparameters:
    {'n_iter': 2, 'n_units_hidden': 11, 'lr': 0.0001}
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:_create_estimator_with_hps:571 | 
    Selected score 0.48409090909090907 for ode_classifier with hyperparameters:
    {'n_iter': 2, 'n_units_hidden': 11, 'lr': 0.0001}
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:_create_estimator_with_hps:571 | 
    Selected score 0.4840277777777778 for cde_classifier with hyperparameters:
    {'n_iter': 2, 'n_temporal_units_hidden': 20, 'lr': 0.001}



```python
# The best methods are returned, and can be used by calling .predict() and so on.

import rich.pretty  # For pretty printing only.

for method in best_methods:
    rich.pretty.pprint(method, indent_guides=False)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">NeuralNetClassifier</span><span style="font-weight: bold">(</span>
    <span style="color: #808000; text-decoration-color: #808000">name</span>=<span style="color: #008000; text-decoration-color: #008000">'nn_classifier'</span>,
    <span style="color: #808000; text-decoration-color: #808000">category</span>=<span style="color: #008000; text-decoration-color: #008000">'prediction.one_off.classification'</span>,
    <span style="color: #808000; text-decoration-color: #808000">params</span>=<span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'n_static_units_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
        <span style="color: #008000; text-decoration-color: #008000">'n_static_layers_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
        <span style="color: #008000; text-decoration-color: #008000">'n_temporal_units_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">102</span>,
        <span style="color: #008000; text-decoration-color: #008000">'n_temporal_layers_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
        <span style="color: #008000; text-decoration-color: #008000">'n_iter'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
        <span style="color: #008000; text-decoration-color: #008000">'mode'</span>: <span style="color: #008000; text-decoration-color: #008000">'RNN'</span>,
        <span style="color: #008000; text-decoration-color: #008000">'n_iter_print'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>,
        <span style="color: #008000; text-decoration-color: #008000">'batch_size'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
        <span style="color: #008000; text-decoration-color: #008000">'lr'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0001</span>,
        <span style="color: #008000; text-decoration-color: #008000">'weight_decay'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001</span>,
        <span style="color: #008000; text-decoration-color: #008000">'window_size'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
        <span style="color: #008000; text-decoration-color: #008000">'device'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
        <span style="color: #008000; text-decoration-color: #008000">'dataloader_sampler'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
        <span style="color: #008000; text-decoration-color: #008000">'dropout'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0</span>,
        <span style="color: #008000; text-decoration-color: #008000">'nonlin'</span>: <span style="color: #008000; text-decoration-color: #008000">'relu'</span>,
        <span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>,
        <span style="color: #008000; text-decoration-color: #008000">'clipping_value'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
        <span style="color: #008000; text-decoration-color: #008000">'patience'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">20</span>,
        <span style="color: #008000; text-decoration-color: #008000">'train_ratio'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.8</span>
    <span style="font-weight: bold">}</span>
<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">ODEClassifier</span><span style="font-weight: bold">(</span>
    <span style="color: #808000; text-decoration-color: #808000">name</span>=<span style="color: #008000; text-decoration-color: #008000">'ode_classifier'</span>,
    <span style="color: #808000; text-decoration-color: #808000">category</span>=<span style="color: #008000; text-decoration-color: #008000">'prediction.one_off.classification'</span>,
    <span style="color: #808000; text-decoration-color: #808000">params</span>=<span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'n_units_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>,
        <span style="color: #008000; text-decoration-color: #008000">'n_layers_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
        <span style="color: #008000; text-decoration-color: #008000">'nonlin'</span>: <span style="color: #008000; text-decoration-color: #008000">'relu'</span>,
        <span style="color: #008000; text-decoration-color: #008000">'dropout'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0</span>,
        <span style="color: #008000; text-decoration-color: #008000">'atol'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.01</span>,
        <span style="color: #008000; text-decoration-color: #008000">'rtol'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.01</span>,
        <span style="color: #008000; text-decoration-color: #008000">'interpolation'</span>: <span style="color: #008000; text-decoration-color: #008000">'cubic'</span>,
        <span style="color: #008000; text-decoration-color: #008000">'lr'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0001</span>,
        <span style="color: #008000; text-decoration-color: #008000">'weight_decay'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001</span>,
        <span style="color: #008000; text-decoration-color: #008000">'n_iter'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
        <span style="color: #008000; text-decoration-color: #008000">'batch_size'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">500</span>,
        <span style="color: #008000; text-decoration-color: #008000">'n_iter_print'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
        <span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>,
        <span style="color: #008000; text-decoration-color: #008000">'patience'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>,
        <span style="color: #008000; text-decoration-color: #008000">'clipping_value'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
        <span style="color: #008000; text-decoration-color: #008000">'train_ratio'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.8</span>,
        <span style="color: #008000; text-decoration-color: #008000">'device'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
        <span style="color: #008000; text-decoration-color: #008000">'dataloader_sampler'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>
    <span style="font-weight: bold">}</span>
<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">CDEClassifier</span><span style="font-weight: bold">(</span>
    <span style="color: #808000; text-decoration-color: #808000">name</span>=<span style="color: #008000; text-decoration-color: #008000">'cde_classifier'</span>,
    <span style="color: #808000; text-decoration-color: #808000">category</span>=<span style="color: #008000; text-decoration-color: #008000">'prediction.one_off.classification'</span>,
    <span style="color: #808000; text-decoration-color: #808000">params</span>=<span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'n_units_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
        <span style="color: #008000; text-decoration-color: #008000">'n_layers_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
        <span style="color: #008000; text-decoration-color: #008000">'nonlin'</span>: <span style="color: #008000; text-decoration-color: #008000">'relu'</span>,
        <span style="color: #008000; text-decoration-color: #008000">'dropout'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0</span>,
        <span style="color: #008000; text-decoration-color: #008000">'atol'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.01</span>,
        <span style="color: #008000; text-decoration-color: #008000">'rtol'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.01</span>,
        <span style="color: #008000; text-decoration-color: #008000">'interpolation'</span>: <span style="color: #008000; text-decoration-color: #008000">'cubic'</span>,
        <span style="color: #008000; text-decoration-color: #008000">'lr'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001</span>,
        <span style="color: #008000; text-decoration-color: #008000">'weight_decay'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001</span>,
        <span style="color: #008000; text-decoration-color: #008000">'n_iter'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
        <span style="color: #008000; text-decoration-color: #008000">'batch_size'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">500</span>,
        <span style="color: #008000; text-decoration-color: #008000">'n_iter_print'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
        <span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>,
        <span style="color: #008000; text-decoration-color: #008000">'patience'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>,
        <span style="color: #008000; text-decoration-color: #008000">'clipping_value'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
        <span style="color: #008000; text-decoration-color: #008000">'train_ratio'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.8</span>,
        <span style="color: #008000; text-decoration-color: #008000">'device'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
        <span style="color: #008000; text-decoration-color: #008000">'dataloader_sampler'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>
    <span style="font-weight: bold">}</span>
<span style="font-weight: bold">)</span>
</pre>



## Using `PipelineSeeker`

Use a `PipelineSeeker` to search for best *pipeline* (`preprocessing steps -> prediction step`) for a particular task.

This seeker will create pipelines comprised of:
- A static imputer (if at lease one candidate in ``static_imputers`` provided),
- A static scaler (if at lease one candidate in ``static_scalers`` provided),
- A temporal imputer (if at lease one candidate in ``temporal_imputers`` provided),
- A temporal scaler (if at lease one candidate in ``temporal_scalers`` provided),
- The final predictor, from the ``estimator_names`` options.

The imputer/scaler candidates will be sampled as a categorical hyperparameter. The hyperparameter spaces of these,
and of the final predictor, will be sampled.

A `PipelineSeeker` uses a very similar interface to `MethodSeeker`, and can be initialized as follows.


```python
from tempor.automl.seeker import PipelineSeeker

seeker = PipelineSeeker(
    study_name="my_automl_study",
    task_type="prediction.one_off.classification",
    # The estimators here will be the final step of the pipeline:
    estimator_names=[
        "cde_classifier",
        "ode_classifier",
        "nn_classifier",
    ],
    metric="aucroc",
    dataset=dataset,
    return_top_k=3,
    num_iter=100,
    tuner_type="bayesian",
    # The following arguments specify the candidates of the different preprocessing steps, e.g.:
    static_imputers=["static_tabular_imputer"],
    static_scalers=[],
    temporal_imputers=["ffill", "bfill"],
    temporal_scalers=["ts_minmax_scaler"],
)
```

    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:_set_up_tuners:354 | Setting up estimators and tuners for study my_automl_study.
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:_init_estimator:733 | Creating estimator <Pipeline with cde_classifier>.
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:_init_estimator:733 | Creating estimator <Pipeline with ode_classifier>.
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:_init_estimator:733 | Creating estimator <Pipeline with nn_classifier>.


By default, the following preprocessing candidates will be used, if you do not specify the argument:


```python
from tempor.automl.seeker import (
    DEFAULT_STATIC_IMPUTERS,
    DEFAULT_STATIC_SCALERS,
    DEFAULT_TEMPORAL_IMPUTERS,
    DEFAULT_TEMPORAL_SCALERS,
)

print("Static imputer candidates:", DEFAULT_STATIC_IMPUTERS)
print("Static scaler candidates:", DEFAULT_STATIC_SCALERS)
print("Temporal imputer candidates:", DEFAULT_TEMPORAL_IMPUTERS)
print("Temporal scaler candidates:", DEFAULT_TEMPORAL_SCALERS)
```

    Static imputer candidates: ['static_tabular_imputer']
    Static scaler candidates: ['static_minmax_scaler', 'static_standard_scaler']
    Temporal imputer candidates: ['ffill', 'ts_tabular_imputer', 'bfill']
    Temporal scaler candidates: ['ts_minmax_scaler', 'ts_standard_scaler']


You can execute the search as follows.


```python
from tempor.automl.params import IntegerParams, CategoricalParams

# Provide a custom hyperparameter space to search for each type of model.
# These can be provided for the final (predictive) step of the pipeline.
# Default hyperparameter space will be sampled for the preprocessing steps.
# NOTE: For the sake of speed of this example, we limit epochs to 2.
hp_space = {
    "cde_classifier": [
        IntegerParams(name="n_iter", low=2, high=2),
        IntegerParams(name="n_temporal_units_hidden", low=5, high=20),
        CategoricalParams(name="lr", choices=[1e-2, 1e-3, 1e-4]),
    ],
    "ode_classifier": [
        IntegerParams(name="n_iter", low=2, high=2),
        IntegerParams(name="n_units_hidden", low=5, high=20),
        CategoricalParams(name="lr", choices=[1e-2, 1e-3, 1e-4]),
    ],
    "nn_classifier": [
        IntegerParams(name="n_iter", low=2, high=2),
        IntegerParams(name="n_units_hidden", low=5, high=20),
        CategoricalParams(name="lr", choices=[1e-2, 1e-3, 1e-4]),
    ],
}

# Initialize a `PipelineSeeker` and provide `override_hp_space`.
seeker = PipelineSeeker(
    study_name="my_automl_study",
    task_type="prediction.one_off.classification",
    estimator_names=[
        "cde_classifier",
        "ode_classifier",
        "nn_classifier",
    ],
    metric="aucroc",
    dataset=dataset,
    return_top_k=3,
    num_iter=3,  # For the sake of speed of this example, only 3 AutoML iterations.
    tuner_type="bayesian",
    # Override hyperparameter space:
    override_hp_space=hp_space,
    # Specify preprocessing candidates:
    static_imputers=["static_tabular_imputer"],
    static_scalers=["static_minmax_scaler", "static_standard_scaler"],
    temporal_imputers=[],
    temporal_scalers=["ts_minmax_scaler", "ts_standard_scaler"],
)
```

    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:_set_up_tuners:354 | Setting up estimators and tuners for study my_automl_study.
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:_init_estimator:733 | Creating estimator <Pipeline with cde_classifier>.
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:_init_estimator:733 | Creating estimator <Pipeline with ode_classifier>.
    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:_init_estimator:733 | Creating estimator <Pipeline with nn_classifier>.



```python
best_pipelines, best_scores = seeker.search()
```

    2023-05-14 21:49:14 | INFO     | tempor.automl.seeker:search:413 | Running  search for estimator '<Pipeline with cde_classifier>' 1/3.
    2023-05-14 21:49:14 | INFO     | tempor.automl.tuner:tune:205 | Baseline score computation skipped
    2023-05-14 21:49:14 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from preprocessing.imputation.static.static_tabular_imputer->preprocessing.scaling.static.static_minmax_scaler->preprocessing.scaling.temporal.ts_standard_scaler->prediction.one_off.classification.cde_classifier:
    {'plugin_params': {'static_tabular_imputer': {'imputer': 'softimpute'}, 'static_minmax_scaler': {'clip': True}, 'ts_standard_scaler': {}, 'cde_classifier': {'n_iter': 2, 'n_temporal_units_hidden': 17, 'lr': 0.001}}}
    2023-05-14 21:49:30 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from preprocessing.imputation.static.static_tabular_imputer->preprocessing.scaling.static.static_minmax_scaler->preprocessing.scaling.temporal.ts_minmax_scaler->prediction.one_off.classification.cde_classifier:
    {'plugin_params': {'static_tabular_imputer': {'imputer': 'mean'}, 'static_minmax_scaler': {'clip': False}, 'ts_minmax_scaler': {'clip': False}, 'cde_classifier': {'n_iter': 2, 'n_temporal_units_hidden': 14, 'lr': 0.001}}}
    2023-05-14 21:49:36 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from preprocessing.imputation.static.static_tabular_imputer->preprocessing.scaling.static.static_standard_scaler->preprocessing.scaling.temporal.ts_standard_scaler->prediction.one_off.classification.cde_classifier:
    {'plugin_params': {'static_tabular_imputer': {'imputer': 'most_frequent'}, 'static_standard_scaler': {}, 'ts_standard_scaler': {}, 'cde_classifier': {'n_iter': 2, 'n_temporal_units_hidden': 6, 'lr': 0.0001}}}
    2023-05-14 21:49:42 | INFO     | tempor.automl.seeker:search:413 | Running  search for estimator '<Pipeline with ode_classifier>' 2/3.
    2023-05-14 21:49:42 | INFO     | tempor.automl.tuner:tune:205 | Baseline score computation skipped
    2023-05-14 21:49:42 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from preprocessing.imputation.static.static_tabular_imputer->preprocessing.scaling.static.static_minmax_scaler->preprocessing.scaling.temporal.ts_standard_scaler->prediction.one_off.classification.ode_classifier:
    {'plugin_params': {'static_tabular_imputer': {'imputer': 'softimpute'}, 'static_minmax_scaler': {'clip': True}, 'ts_standard_scaler': {}, 'ode_classifier': {'n_iter': 2, 'n_units_hidden': 17, 'lr': 0.001}}}
    2023-05-14 21:50:00 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from preprocessing.imputation.static.static_tabular_imputer->preprocessing.scaling.static.static_minmax_scaler->preprocessing.scaling.temporal.ts_minmax_scaler->prediction.one_off.classification.ode_classifier:
    {'plugin_params': {'static_tabular_imputer': {'imputer': 'mean'}, 'static_minmax_scaler': {'clip': False}, 'ts_minmax_scaler': {'clip': False}, 'ode_classifier': {'n_iter': 2, 'n_units_hidden': 14, 'lr': 0.001}}}
    2023-05-14 21:50:06 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from preprocessing.imputation.static.static_tabular_imputer->preprocessing.scaling.static.static_standard_scaler->preprocessing.scaling.temporal.ts_standard_scaler->prediction.one_off.classification.ode_classifier:
    {'plugin_params': {'static_tabular_imputer': {'imputer': 'most_frequent'}, 'static_standard_scaler': {}, 'ts_standard_scaler': {}, 'ode_classifier': {'n_iter': 2, 'n_units_hidden': 6, 'lr': 0.0001}}}
    2023-05-14 21:50:11 | INFO     | tempor.automl.seeker:search:413 | Running  search for estimator '<Pipeline with nn_classifier>' 3/3.
    2023-05-14 21:50:11 | INFO     | tempor.automl.tuner:tune:205 | Baseline score computation skipped
    2023-05-14 21:50:11 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from preprocessing.imputation.static.static_tabular_imputer->preprocessing.scaling.static.static_minmax_scaler->preprocessing.scaling.temporal.ts_standard_scaler->prediction.one_off.classification.nn_classifier:
    {'plugin_params': {'static_tabular_imputer': {'imputer': 'softimpute'}, 'static_minmax_scaler': {'clip': True}, 'ts_standard_scaler': {}, 'nn_classifier': {'n_iter': 2, 'n_units_hidden': 17, 'lr': 0.001}}}
    2023-05-14 21:50:14 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6918608546257019, validation loss: 0.6910383105278015
    2023-05-14 21:50:17 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6909429430961609, validation loss: 0.6872987151145935
    2023-05-14 21:50:21 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.692150354385376, validation loss: 0.6879752278327942
    2023-05-14 21:50:24 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6926363706588745, validation loss: 0.6881052851676941
    2023-05-14 21:50:27 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6915659308433533, validation loss: 0.6888220310211182
    2023-05-14 21:50:28 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from preprocessing.imputation.static.static_tabular_imputer->preprocessing.scaling.static.static_minmax_scaler->preprocessing.scaling.temporal.ts_minmax_scaler->prediction.one_off.classification.nn_classifier:
    {'plugin_params': {'static_tabular_imputer': {'imputer': 'mean'}, 'static_minmax_scaler': {'clip': False}, 'ts_minmax_scaler': {'clip': False}, 'nn_classifier': {'n_iter': 2, 'n_units_hidden': 14, 'lr': 0.001}}}
    2023-05-14 21:50:28 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6920319199562073, validation loss: 0.6892786622047424
    2023-05-14 21:50:29 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6913070678710938, validation loss: 0.6893746852874756
    2023-05-14 21:50:30 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6924894452095032, validation loss: 0.6886722445487976
    2023-05-14 21:50:31 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6930248141288757, validation loss: 0.6894596815109253
    2023-05-14 21:50:32 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6921922564506531, validation loss: 0.6893032789230347
    2023-05-14 21:50:33 | INFO     | tempor.automl.tuner:objective:227 | 
    Hyperparameters sampled from preprocessing.imputation.static.static_tabular_imputer->preprocessing.scaling.static.static_standard_scaler->preprocessing.scaling.temporal.ts_standard_scaler->prediction.one_off.classification.nn_classifier:
    {'plugin_params': {'static_tabular_imputer': {'imputer': 'most_frequent'}, 'static_standard_scaler': {}, 'ts_standard_scaler': {}, 'nn_classifier': {'n_iter': 2, 'n_units_hidden': 6, 'lr': 0.0001}}}
    2023-05-14 21:50:33 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6914154291152954, validation loss: 0.6911771297454834
    2023-05-14 21:50:34 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.690178394317627, validation loss: 0.6934390664100647
    2023-05-14 21:50:35 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6933540105819702, validation loss: 0.6896734833717346
    2023-05-14 21:50:36 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.694333553314209, validation loss: 0.690070390701294
    2023-05-14 21:50:37 | INFO     | tempor.models.ts_model:_train:379 | Epoch:0| train loss: 0.6921719312667847, validation loss: 0.6909489631652832
    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:search:447 | 
    Evaluation for <Pipeline with cde_classifier> scores:
    [0.4716540404040404, 0.47575757575757577, 0.4657828282828283].
    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:search:447 | 
    Evaluation for <Pipeline with ode_classifier> scores:
    [0.47575757575757577, 0.4779040404040404, 0.4779671717171716].
    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:search:447 | 
    Evaluation for <Pipeline with nn_classifier> scores:
    [0.48478535353535346, 0.4361742424242424, 0.5071338383838383].
    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:search:448 | 
    All estimator definitions searched:
    ['<Pipeline with cde_classifier>', '<Pipeline with ode_classifier>', '<Pipeline with nn_classifier>']
    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:search:449 | 
    Best scores for each estimator searched:
    [0.47575757575757577, 0.4779671717171716, 0.5071338383838383]
    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:search:450 | 
    Best hyperparameters for each estimator searched:
    [{'<candidates>(preprocessing.imputation.static)': 'static_tabular_imputer', '[static_tabular_imputer](imputer)': 'mean', '<candidates>(preprocessing.scaling.static)': 'static_minmax_scaler', '[static_minmax_scaler](clip)': False, '<candidates>(preprocessing.scaling.temporal)': 'ts_minmax_scaler', '[ts_minmax_scaler](clip)': False, '[cde_classifier](n_iter)': 2, '[cde_classifier](n_temporal_units_hidden)': 14, '[cde_classifier](lr)': 0.001}, {'<candidates>(preprocessing.imputation.static)': 'static_tabular_imputer', '[static_tabular_imputer](imputer)': 'most_frequent', '<candidates>(preprocessing.scaling.static)': 'static_standard_scaler', '[static_minmax_scaler](clip)': False, '<candidates>(preprocessing.scaling.temporal)': 'ts_standard_scaler', '[ts_minmax_scaler](clip)': False, '[ode_classifier](n_iter)': 2, '[ode_classifier](n_units_hidden)': 6, '[ode_classifier](lr)': 0.0001}, {'<candidates>(preprocessing.imputation.static)': 'static_tabular_imputer', '[static_tabular_imputer](imputer)': 'most_frequent', '<candidates>(preprocessing.scaling.static)': 'static_standard_scaler', '[static_minmax_scaler](clip)': False, '<candidates>(preprocessing.scaling.temporal)': 'ts_standard_scaler', '[ts_minmax_scaler](clip)': False, '[nn_classifier](n_iter)': 2, '[nn_classifier](n_units_hidden)': 6, '[nn_classifier](lr)': 0.0001}]
    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:_create_estimator_with_hps:748 | 
    Selected score 0.5071338383838383 for <Pipeline with nn_classifier> with hyperparameters:
    {'<candidates>(preprocessing.imputation.static)': 'static_tabular_imputer', '[static_tabular_imputer](imputer)': 'most_frequent', '<candidates>(preprocessing.scaling.static)': 'static_standard_scaler', '[static_minmax_scaler](clip)': False, '<candidates>(preprocessing.scaling.temporal)': 'ts_standard_scaler', '[ts_minmax_scaler](clip)': False, '[nn_classifier](n_iter)': 2, '[nn_classifier](n_units_hidden)': 6, '[nn_classifier](lr)': 0.0001}
    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:_create_estimator_with_hps:748 | 
    Selected score 0.4779671717171716 for <Pipeline with ode_classifier> with hyperparameters:
    {'<candidates>(preprocessing.imputation.static)': 'static_tabular_imputer', '[static_tabular_imputer](imputer)': 'most_frequent', '<candidates>(preprocessing.scaling.static)': 'static_standard_scaler', '[static_minmax_scaler](clip)': False, '<candidates>(preprocessing.scaling.temporal)': 'ts_standard_scaler', '[ts_minmax_scaler](clip)': False, '[ode_classifier](n_iter)': 2, '[ode_classifier](n_units_hidden)': 6, '[ode_classifier](lr)': 0.0001}
    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:_create_estimator_with_hps:748 | 
    Selected score 0.47575757575757577 for <Pipeline with cde_classifier> with hyperparameters:
    {'<candidates>(preprocessing.imputation.static)': 'static_tabular_imputer', '[static_tabular_imputer](imputer)': 'mean', '<candidates>(preprocessing.scaling.static)': 'static_minmax_scaler', '[static_minmax_scaler](clip)': False, '<candidates>(preprocessing.scaling.temporal)': 'ts_minmax_scaler', '[ts_minmax_scaler](clip)': False, '[cde_classifier](n_iter)': 2, '[cde_classifier](n_temporal_units_hidden)': 14, '[cde_classifier](lr)': 0.001}



```python
# The best performing pipelines are returned, and can be used by calling .predict() and so on.

for method in best_pipelines:
    rich.pretty.pprint(method, indent_guides=False)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Pipeline</span><span style="font-weight: bold">(</span>
    <span style="color: #808000; text-decoration-color: #808000">pipeline_seq</span>=<span style="color: #008000; text-decoration-color: #008000">'preprocessing.imputation.static.static_tabular_imputer-&gt;preprocessing.scaling.static.static_standard_scaler-&gt;preprocessing.scaling.temporal.ts_standard_scaler-&gt;prediction.one_off.classification.nn_classifier'</span>,
    <span style="color: #808000; text-decoration-color: #808000">predictor_category</span>=<span style="color: #008000; text-decoration-color: #008000">'prediction.one_off.classification'</span>,
    <span style="color: #808000; text-decoration-color: #808000">params</span>=<span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'static_tabular_imputer'</span>: <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'imputer'</span>: <span style="color: #008000; text-decoration-color: #008000">'most_frequent'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>,
            <span style="color: #008000; text-decoration-color: #008000">'imputer_params'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">}</span>
        <span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'static_standard_scaler'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'with_mean'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>, <span style="color: #008000; text-decoration-color: #008000">'with_std'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span><span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'ts_standard_scaler'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'with_mean'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>, <span style="color: #008000; text-decoration-color: #008000">'with_std'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span><span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'nn_classifier'</span>: <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'n_static_units_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
            <span style="color: #008000; text-decoration-color: #008000">'n_static_layers_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
            <span style="color: #008000; text-decoration-color: #008000">'n_temporal_units_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">102</span>,
            <span style="color: #008000; text-decoration-color: #008000">'n_temporal_layers_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
            <span style="color: #008000; text-decoration-color: #008000">'n_iter'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
            <span style="color: #008000; text-decoration-color: #008000">'mode'</span>: <span style="color: #008000; text-decoration-color: #008000">'RNN'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'n_iter_print'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>,
            <span style="color: #008000; text-decoration-color: #008000">'batch_size'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
            <span style="color: #008000; text-decoration-color: #008000">'lr'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0001</span>,
            <span style="color: #008000; text-decoration-color: #008000">'weight_decay'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001</span>,
            <span style="color: #008000; text-decoration-color: #008000">'window_size'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
            <span style="color: #008000; text-decoration-color: #008000">'device'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
            <span style="color: #008000; text-decoration-color: #008000">'dataloader_sampler'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
            <span style="color: #008000; text-decoration-color: #008000">'dropout'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0</span>,
            <span style="color: #008000; text-decoration-color: #008000">'nonlin'</span>: <span style="color: #008000; text-decoration-color: #008000">'relu'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>,
            <span style="color: #008000; text-decoration-color: #008000">'clipping_value'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
            <span style="color: #008000; text-decoration-color: #008000">'patience'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">20</span>,
            <span style="color: #008000; text-decoration-color: #008000">'train_ratio'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.8</span>
        <span style="font-weight: bold">}</span>
    <span style="font-weight: bold">}</span>
<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Pipeline</span><span style="font-weight: bold">(</span>
    <span style="color: #808000; text-decoration-color: #808000">pipeline_seq</span>=<span style="color: #008000; text-decoration-color: #008000">'preprocessing.imputation.static.static_tabular_imputer-&gt;preprocessing.scaling.static.static_standard_scaler-&gt;preprocessing.scaling.temporal.ts_standard_scaler-&gt;prediction.one_off.classification.ode_classifier'</span>,
    <span style="color: #808000; text-decoration-color: #808000">predictor_category</span>=<span style="color: #008000; text-decoration-color: #008000">'prediction.one_off.classification'</span>,
    <span style="color: #808000; text-decoration-color: #808000">params</span>=<span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'static_tabular_imputer'</span>: <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'imputer'</span>: <span style="color: #008000; text-decoration-color: #008000">'most_frequent'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>,
            <span style="color: #008000; text-decoration-color: #008000">'imputer_params'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">}</span>
        <span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'static_standard_scaler'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'with_mean'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>, <span style="color: #008000; text-decoration-color: #008000">'with_std'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span><span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'ts_standard_scaler'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'with_mean'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>, <span style="color: #008000; text-decoration-color: #008000">'with_std'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span><span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'ode_classifier'</span>: <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'n_units_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>,
            <span style="color: #008000; text-decoration-color: #008000">'n_layers_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
            <span style="color: #008000; text-decoration-color: #008000">'nonlin'</span>: <span style="color: #008000; text-decoration-color: #008000">'relu'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'dropout'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0</span>,
            <span style="color: #008000; text-decoration-color: #008000">'atol'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.01</span>,
            <span style="color: #008000; text-decoration-color: #008000">'rtol'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.01</span>,
            <span style="color: #008000; text-decoration-color: #008000">'interpolation'</span>: <span style="color: #008000; text-decoration-color: #008000">'cubic'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'lr'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0001</span>,
            <span style="color: #008000; text-decoration-color: #008000">'weight_decay'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001</span>,
            <span style="color: #008000; text-decoration-color: #008000">'n_iter'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
            <span style="color: #008000; text-decoration-color: #008000">'batch_size'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">500</span>,
            <span style="color: #008000; text-decoration-color: #008000">'n_iter_print'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
            <span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>,
            <span style="color: #008000; text-decoration-color: #008000">'patience'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>,
            <span style="color: #008000; text-decoration-color: #008000">'clipping_value'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
            <span style="color: #008000; text-decoration-color: #008000">'train_ratio'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.8</span>,
            <span style="color: #008000; text-decoration-color: #008000">'device'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
            <span style="color: #008000; text-decoration-color: #008000">'dataloader_sampler'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>
        <span style="font-weight: bold">}</span>
    <span style="font-weight: bold">}</span>
<span style="font-weight: bold">)</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Pipeline</span><span style="font-weight: bold">(</span>
    <span style="color: #808000; text-decoration-color: #808000">pipeline_seq</span>=<span style="color: #008000; text-decoration-color: #008000">'preprocessing.imputation.static.static_tabular_imputer-&gt;preprocessing.scaling.static.static_minmax_scaler-&gt;preprocessing.scaling.temporal.ts_minmax_scaler-&gt;prediction.one_off.classification.cde_classifier'</span>,
    <span style="color: #808000; text-decoration-color: #808000">predictor_category</span>=<span style="color: #008000; text-decoration-color: #008000">'prediction.one_off.classification'</span>,
    <span style="color: #808000; text-decoration-color: #808000">params</span>=<span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'static_tabular_imputer'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'imputer'</span>: <span style="color: #008000; text-decoration-color: #008000">'mean'</span>, <span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>, <span style="color: #008000; text-decoration-color: #008000">'imputer_params'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="font-weight: bold">}}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'static_minmax_scaler'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'feature_range'</span>: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span><span style="font-weight: bold">]</span>, <span style="color: #008000; text-decoration-color: #008000">'clip'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'ts_minmax_scaler'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'feature_range'</span>: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span><span style="font-weight: bold">]</span>, <span style="color: #008000; text-decoration-color: #008000">'clip'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'cde_classifier'</span>: <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'n_units_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
            <span style="color: #008000; text-decoration-color: #008000">'n_layers_hidden'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
            <span style="color: #008000; text-decoration-color: #008000">'nonlin'</span>: <span style="color: #008000; text-decoration-color: #008000">'relu'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'dropout'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0</span>,
            <span style="color: #008000; text-decoration-color: #008000">'atol'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.01</span>,
            <span style="color: #008000; text-decoration-color: #008000">'rtol'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.01</span>,
            <span style="color: #008000; text-decoration-color: #008000">'interpolation'</span>: <span style="color: #008000; text-decoration-color: #008000">'cubic'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'lr'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001</span>,
            <span style="color: #008000; text-decoration-color: #008000">'weight_decay'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001</span>,
            <span style="color: #008000; text-decoration-color: #008000">'n_iter'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
            <span style="color: #008000; text-decoration-color: #008000">'batch_size'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">500</span>,
            <span style="color: #008000; text-decoration-color: #008000">'n_iter_print'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span>,
            <span style="color: #008000; text-decoration-color: #008000">'random_state'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>,
            <span style="color: #008000; text-decoration-color: #008000">'patience'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>,
            <span style="color: #008000; text-decoration-color: #008000">'clipping_value'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
            <span style="color: #008000; text-decoration-color: #008000">'train_ratio'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.8</span>,
            <span style="color: #008000; text-decoration-color: #008000">'device'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
            <span style="color: #008000; text-decoration-color: #008000">'dataloader_sampler'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>
        <span style="font-weight: bold">}</span>
    <span style="font-weight: bold">}</span>
<span style="font-weight: bold">)</span>
</pre>



## Advanced customization

You may further customize the AutoML tuning behavior by specifying the sampler an pruner, if desired.

See the below example.


```python
# 1. Import a Tuner:
from tempor.automl.tuner import OptunaTuner

# 2. Customize this as needed:
import optuna

custom_tuner = OptunaTuner(
    study_name="my_automl_study",
    direction="maximize",
    # Customized sampler:
    study_sampler=optuna.samplers.TPESampler(seed=12345, n_startup_trials=3),
    # Customized pruner:
    study_pruner=optuna.pruners.MedianPruner(interval_steps=2),
    # Using default optuna storage object here, but may a provide custom one, e.g. redis.
    study_storage=None,
)

# 3. Pass the Tuner to the {Method/Pipeline}Seeker:
seeker = MethodSeeker(
    study_name="my_automl_study",
    task_type="prediction.one_off.classification",
    estimator_names=[
        "cde_classifier",
        "ode_classifier",
        "nn_classifier",
    ],
    metric="aucroc",
    dataset=dataset,
    # Like so:
    custom_tuner=custom_tuner,
)

# 4. Execute search:
# results = seeker.search() ...
```

    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:_set_up_tuners:354 | Setting up estimators and tuners for study my_automl_study.
    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:_init_estimator:563 | Creating estimator cde_classifier.
    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:_init_estimator:563 | Creating estimator ode_classifier.
    2023-05-14 21:50:38 | INFO     | tempor.automl.seeker:_init_estimator:563 | Creating estimator nn_classifier.


## Supported tasks

> ⚠️ The tasks for which benchmarking is supported are supported by AutoML. See the benchmarking tutorial.


