import tempor.plugins.core as plugins


class BaseImputer(plugins.BaseTransformer):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)


plugins.register_plugin_category("preprocessing.imputation", BaseImputer)

plugins.importing.import_plugins(__file__)

__all__ = [  # pyright: ignore
    *plugins.importing.gather_modules_names(__file__),
    "BaseImputer",
]
