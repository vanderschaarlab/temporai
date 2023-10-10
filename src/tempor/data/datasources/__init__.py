from .datasource import (
    DataSource,
    OneOffPredictionDataSource,
    OneOffTreatmentEffectsDataSource,
    TemporalPredictionDataSource,
    TemporalTreatmentEffectsDataSource,
    TimeToEventAnalysisDataSource,
)
from .dummy import DummyTemporalPredictionDataSource, DummyTemporalTreatmentEffectsDataSource
from .google_stocks import GoogleStocksDataSource
from .pbc import PBCDataSource
from .pkpd import PKPDDataSource
from .sine import SineDataSource
from .uci_diabetes import UCIDiabetesDataSource

base_classes = [
    DataSource,
    OneOffPredictionDataSource,
    OneOffTreatmentEffectsDataSource,
    TemporalPredictionDataSource,
    TemporalTreatmentEffectsDataSource,
    TimeToEventAnalysisDataSource,
]

all_datasources = [
    DummyTemporalPredictionDataSource,
    DummyTemporalTreatmentEffectsDataSource,
    GoogleStocksDataSource,
    PBCDataSource,
    PKPDDataSource,
    SineDataSource,
    UCIDiabetesDataSource,
]

__all__ = [x.__name__ for x in base_classes] + [x.__name__ for x in all_datasources]  # pyright: ignore
