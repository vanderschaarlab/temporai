from tempor.data import dataset


def as_covariates_dataset(ds: dataset.PredictiveDataset) -> dataset.CovariatesDataset:
    data = dataset.CovariatesDataset(
        time_series=ds.time_series.dataframe(),
        static=ds.static.dataframe() if ds.static is not None else None,
    )
    return data
