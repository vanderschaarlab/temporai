"""Serialization utilities for models."""

from pathlib import Path
from typing import Any, Union

import cloudpickle


def save(model: Any) -> bytes:
    """Save a model to bytes using `cloudpickle`. Companion to `~tempor.utils.serialization.load`.

    Args:
        model (Any): Model object to be saved.

    Returns:
        bytes: Model serialized to bytes.
    """
    return cloudpickle.dumps(model)


def load(buff: bytes) -> Any:
    """Load a model from bytes using `cloudpickle`. Companion to `~tempor.utils.serialization.save`.

    Args:
        buff (bytes): Model as serialized to bytes (buffer), e.g. loaded from file.

    Returns:
        Any: The model object.
    """
    return cloudpickle.loads(buff)


def save_to_file(path: Union[str, Path], model: Any) -> None:
    """Save a model to file using `cloudpickle`. Companion to `~tempor.utils.serialization.load_from_file`.

    Args:
        path (Union[str, Path]): Path to save the model to.
        model (Any): The model object to be saved.
    """
    path = Path(path)
    ppath = path.absolute().parent

    if not ppath.exists():
        ppath.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        cloudpickle.dump(model, f)


def load_from_file(path: Union[str, Path]) -> Any:
    """Load a model from file using `cloudpickle`. Companion to `~tempor.utils.serialization.save_to_file`.

    Args:
        path (Union[str, Path]): The file path to load the model from.

    Returns:
        Any: The loaded model object.
    """
    with open(path, "rb") as f:
        return cloudpickle.load(f)
