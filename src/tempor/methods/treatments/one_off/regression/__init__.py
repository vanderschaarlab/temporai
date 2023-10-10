from tempor.core import plugins
from tempor.methods.treatments.one_off._base import BaseOneOffTreatmentEffects

plugins.register_plugin_category("treatments.one_off.regression", BaseOneOffTreatmentEffects)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
]
