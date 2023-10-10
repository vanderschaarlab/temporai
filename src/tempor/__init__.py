import sys
import warnings

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

# Import the config type and the configure function:
# Global warnings suppression:
warnings.filterwarnings("ignore", message=".*validate_arguments.*", category=DeprecationWarning)

# NOTE:
# Currently, migrating to Pydantic 2.0's validate_call decorator is not possible due to several incompatibilities, e.g.:
# - cloudpickle fails to pickle objects: TypeError: cannot pickle 'EncodedFile' object.
# - PipelineMeta setup is incompatible, triggers: TypeError: can't apply this __setattr__ to ABCMeta object.
# For the time being, reasonable to stick to validate_arguments and silence Pydantic 2.0's deprecation warning.


import tempor.data.datasources  # noqa: E402 F401
import tempor.methods  # noqa: E402 F401

# Prepare the plugin loader:
from tempor.core import plugins  # noqa: E402

# ^ Importing of necessary package directories is necessary to trigger the registration of plugins.

plugin_loader = plugins.PluginLoader()


__all__ = [
    "get_config",
    "configure",
    "TemporConfig",
]
