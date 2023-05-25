# Data Tutorial 04: Data splitting
[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanderschaarlab/temporai/blob/main/tutorials/data/tutorial04_data_splitting.ipynb)

This tutorial shows how a `Dataset` can be split or KFold-ed.



## 1. Splitting data


```python
# Get a dataset.

from tempor.utils.dataloaders import SineDataLoader

sine_dataset = SineDataLoader().load()

sine_dataset
```




    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([100, *, 5]),
        static=StaticSamples([100, 4]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([100, 1]))
    )



The method `train_test_split` (same API as [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)) can be used to split the dataset into train and test sets.


```python
sine_dataset_train, sine_dataset_test = sine_dataset.train_test_split(test_size=0.2)

print("Training set:")
print(sine_dataset_train)
print("Test set:")
print(sine_dataset_test)
```

    Training set:
    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([80, *, 5]),
        static=StaticSamples([80, 4]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([80, 1]))
    )
    Test set:
    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([20, *, 5]),
        static=StaticSamples([20, 4]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([20, 1]))
    )


## 2. Using `KFold` and other `sklearn` splitter classes

Any of the [sklearn splitter classes](https://scikit-learn.org/stable/modules/classes.html#splitter-classes) can be used with `Dataset` `split` method.

The following example illustrates this with the [`KFold` splitter](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold):


```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, random_state=123, shuffle=True)

splits = list(sine_dataset.split(splitter=kfold))

print(len(splits))
```

    5



```python
train_set_0, test_set_0 = splits[0]

print("0th split training set:")
print(sine_dataset_train)
print("0th split test set:")
print(sine_dataset_test)
```

    0th split training set:
    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([80, *, 5]),
        static=StaticSamples([80, 4]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([80, 1]))
    )
    0th split test set:
    OneOffPredictionDataset(
        time_series=TimeSeriesSamples([20, *, 5]),
        static=StaticSamples([20, 4]),
        predictive=OneOffPredictionTaskData(targets=StaticSamples([20, 1]))
    )


