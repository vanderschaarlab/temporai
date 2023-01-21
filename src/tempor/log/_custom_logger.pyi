from loguru import Logger as _Logger

# flake8: noqa

# In order to give type checkers guidance on the dynamically added methods:
class Logger(_Logger):
    def print(self, message: str) -> None:
        pass

logger: Logger
