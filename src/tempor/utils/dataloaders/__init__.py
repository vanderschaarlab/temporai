from .dummy import DummyTemporalPredictionDataLoader, DummyTemporalTreatmentEffectsDataLoader
from .google_stocks import GoogleStocksDataLoader
from .pbc import PBCDataLoader
from .pkpd import PKPDDataLoader
from .sine import SineDataLoader

all_dataloaders = [
    DummyTemporalPredictionDataLoader,
    DummyTemporalTreatmentEffectsDataLoader,
    GoogleStocksDataLoader,
    PBCDataLoader,
    PKPDDataLoader,
    SineDataLoader,
]

__all__ = [x.__name__ for x in all_dataloaders]  # pyright: ignore
