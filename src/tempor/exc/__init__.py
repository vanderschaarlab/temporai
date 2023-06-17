class DataValidationException(ValueError):
    """Raise this exception when the user provides invalid data values  for training."""
    pass


class UnsupportedSetupException(RuntimeError):
    """Raise this exception when an TemporAI estimator (model) encounters an unsupported situation,
    e.g. incompatible data format.
    """

    pass  # pylint: disable=unnecessary-pass
