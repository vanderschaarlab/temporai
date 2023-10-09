from .dummy import DummyTemporalPredictionDataSource, DummyTemporalTreatmentEffectsDataSource
from .google_stocks import GoogleStocksDataSource
from .pbc import PBCDataSource
from .pkpd import PKPDDataSource
from .sine import SineDataSource
from .uci_diabetes import UCIDiabetesDataSource

all_dataloaders = [
    DummyTemporalPredictionDataSource,
    DummyTemporalTreatmentEffectsDataSource,
    GoogleStocksDataSource,
    PBCDataSource,
    PKPDDataSource,
    SineDataSource,
    UCIDiabetesDataSource,
]

__all__ = [x.__name__ for x in all_dataloaders]  # pyright: ignore
