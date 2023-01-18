from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    import_name = "tempor"
    dist_name = "temporai"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# Import the config type and the configure function:
from .config import TemporConfig, configure, get_config

__all__ = [
    "get_config",
    "configure",
    "TemporConfig",
]
