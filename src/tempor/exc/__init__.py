class DataValidationException(ValueError):
    pass


class UnsupportedSetupException(RuntimeError):
    """Raise this exception when an TemporAI estimator (model) encounters an unsupported situation,
    e.g. incompatible data format.
    """

    pass  # pylint: disable=unnecessary-pass
