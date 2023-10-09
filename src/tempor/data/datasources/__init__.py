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

all_datasources = [
    # Base classes:
    DataSource,
    OneOffPredictionDataSource,
    OneOffTreatmentEffectsDataSource,
    TemporalPredictionDataSource,
    TemporalTreatmentEffectsDataSource,
    TimeToEventAnalysisDataSource,
    # Concrete classes:
    DummyTemporalPredictionDataSource,
    DummyTemporalTreatmentEffectsDataSource,
    GoogleStocksDataSource,
    PBCDataSource,
    PKPDDataSource,
    SineDataSource,
    UCIDiabetesDataSource,
]

__all__ = [x.__name__ for x in all_datasources]  # pyright: ignore
