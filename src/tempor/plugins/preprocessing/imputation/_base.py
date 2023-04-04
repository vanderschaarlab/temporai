import tempor.plugins.core as plugins


class BaseImputer(plugins.BaseTransformer):
    def __init__(self, **params) -> None:  # pylint: disable=useless-super-delegation
        super().__init__(**params)
