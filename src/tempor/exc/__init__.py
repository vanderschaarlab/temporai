"""Package directory with custom exceptions for TemporAI."""

# pylint: disable=unnecessary-pass


class DataValidationException(ValueError):
    """Exception raised when TemporAI-specific data format validation fails."""

    pass


class UnsupportedSetupException(RuntimeError):
    """Raise this exception when an TemporAI estimator (model) encounters an unsupported situation,
    e.g. incompatible data format.
    """

    pass
