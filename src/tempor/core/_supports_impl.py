from abc import ABC, abstractmethod
from typing import Dict, Generic, Tuple, TypeVar

SupportsT = TypeVar("SupportsT")
ImplementationT = TypeVar("ImplementationT")


class SupportsImplementations(Generic[SupportsT, ImplementationT], ABC):
    def __init__(self) -> None:
        super().__init__()
        self._implementations: Dict[SupportsT, ImplementationT] = self._register_implementations()
        registered = set(self._implementations.keys())
        supported = set(self.supports_implementations_for)
        if registered != supported:
            raise TypeError(
                f"{self.__class__.__name__}: Expected implementations to be registered for the following supported "
                f"item(s) {list(supported)} but found registered item(s) {list(registered)}"
            )

    def dispatch_to_implementation(self, key: SupportsT) -> ImplementationT:
        if key not in self._implementations:
            raise TypeError(
                f"Implementation for `{key}` has not been registered on `{self.__class__}`, "
                f"only implementations the following have been registered {list(self._implementations.keys())}"
            )
        return self._implementations[key]

    @property
    @abstractmethod
    def supports_implementations_for(self) -> Tuple[SupportsT, ...]:  # pragma: no cover
        ...

    @abstractmethod
    def _register_implementations(self) -> Dict[SupportsT, ImplementationT]:  # pragma: no cover
        ...

    def __str__(self) -> str:
        # For clarity.
        return f"{self.__class__.__name__}(_implementations={self._implementations})"
