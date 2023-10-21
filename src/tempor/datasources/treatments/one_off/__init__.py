"""Package for one-off treatment effects data sources."""

from tempor.core import plugins
from tempor.datasources import datasource

plugins.register_plugin_category(
    "treatments.one_off", datasource.OneOffTreatmentEffectsDataSource, plugin_type="datasource"
)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
]
