import sys

if sys.version_info[:2] >= (3, 8):  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version
else:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

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
