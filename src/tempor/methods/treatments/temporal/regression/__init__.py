import tempor.methods.core as plugins
from tempor.methods.treatments.temporal._base import BaseTemporalTreatmentEffects

plugins.register_plugin_category("treatments.temporal.regression", BaseTemporalTreatmentEffects)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
]
