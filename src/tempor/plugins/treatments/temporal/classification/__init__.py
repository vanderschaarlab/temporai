import tempor.plugins.core as plugins
from tempor.plugins.treatments.temporal._base import BaseTemporalTreatmentEffects

plugins.register_plugin_category("treatments.temporal.classification", BaseTemporalTreatmentEffects)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
]
