import abc
import enum
from typing import Any, ClassVar, Sequence, get_type_hints

import pydantic
import rich.pretty


class RequirementCategory(enum.Enum):
    DATA_CONTAINER = enum.auto()
    DATA_BUNDLE = enum.auto()
    MODEL = enum.auto()


class Requirement(abc.ABC):
    definition: Any

    name: ClassVar[str] = None  # type: ignore
    validator_methods: ClassVar[Sequence[Any]] = tuple()

    def __init__(self, definition) -> None:
        validator = pydantic.create_model(
            f"{self.__class__.__name__}Validator",
            __validators__={clsmtd.__func__.__name__: clsmtd for clsmtd in self.validator_methods},  # type: ignore
            name=(str, ...),
            definition=(get_type_hints(self)["definition"], ...),
        )
        validator(name=self.name, definition=definition)
        self.definition = definition

    @property
    @abc.abstractmethod
    def requirement_category(self) -> RequirementCategory:  # pragma: no cover
        ...

    def __rich_repr__(self):
        yield "definition", self.definition

    def __repr__(self) -> str:
        return rich.pretty.pretty_repr(self)


class RequirementValidator(abc.ABC):
    @property
    @abc.abstractmethod
    def supported_requirement_category(self) -> RequirementCategory:  # pragma: no cover
        ...

    def validate(self, target: Any, *, requirements: Sequence[Requirement], **kwargs):  # pragma: no cover
        for r in requirements:
            if r.requirement_category != self.supported_requirement_category:
                raise ValueError(
                    f"{self.__class__.__name__} does not support requirement category {r.requirement_category}"
                )
        self._validate(target=target, requirements=requirements, **kwargs)

    @abc.abstractmethod
    def _validate(self, *args, **kwargs):  # pragma: no cover
        ...
