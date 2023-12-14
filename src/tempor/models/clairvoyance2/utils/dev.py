# mypy: ignore-errors

import inspect
from typing import Callable, NoReturn

NEEDED = None
"""In case a value is set to `None` as a placeholder, or temporarily, or for any other reason,
differentiate with `NEEDED`. For code clarity only.
"""

NOT_IMPLEMENTED_MESSAGE = "Feature not yet implemented: $FEATURE"


def raise_not_implemented(feature: str) -> NoReturn:
    raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE.replace("$FEATURE", feature))


def function_is_notimplemented_stub(function: Callable) -> bool:
    # Somewhat of a kludge.
    source = inspect.getsource(function)
    lines = [line.strip() for line in source.split("\n") if len(line) > 0]
    if len(lines) > 0 and "raise NotImplementedError" in lines[-1]:
        return True
    else:
        return False
