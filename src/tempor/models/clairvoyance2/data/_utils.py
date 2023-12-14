from typing import TypeVar

from .dataformat import TimeSeries, TimeSeriesSamples

THasTimeIndex = TypeVar("THasTimeIndex", TimeSeries, TimeSeriesSamples)


def time_index_equal(a: THasTimeIndex, b: THasTimeIndex) -> bool:
    if isinstance(a, TimeSeries):
        return _time_index_equal__time_series(a, b)
    elif isinstance(a, TimeSeriesSamples):
        return _time_index_equal__time_series_samples(a, b)
    else:
        raise TypeError(f"Unexpected type encountered: {type(a)}")


def _time_index_equal__time_series(a: TimeSeries, b: TimeSeries) -> bool:
    if len(a.time_index) != len(b.time_index):
        return False
    else:
        return (a.time_index == b.time_index).all()


def _time_index_equal__time_series_samples(a: TimeSeriesSamples, b: TimeSeriesSamples) -> bool:
    if a.sample_indices != b.sample_indices:
        return False
    return all(_time_index_equal__time_series(a_, b_) for a_, b_ in zip(a, b))
