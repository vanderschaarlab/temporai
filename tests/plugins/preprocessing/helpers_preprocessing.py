import copy

from tempor.data import dataset


def as_covariates_dataset(_sine_data_full: dataset.PredictiveDataset) -> dataset.CovariatesDataset:
    data = copy.deepcopy(_sine_data_full)
    data = dataset.CovariatesDataset(
        time_series=data.time_series.dataframe(),
        static=data.static.dataframe() if data.static is not None else None,
    )
    return data
