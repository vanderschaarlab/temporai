from typing import Any, Callable, Dict

from tempor.log import logger


class RegisterMethodDecorator:
    # Inspired by: https://stackoverflow.com/a/54316392

    _registering_key: Any = None
    _registering_dict: Dict = {}

    owner_class: type = None  # type: ignore
    registration_dict_attribute_name: str = None  # type: ignore
    key_type: type = None  # type: ignore
    method_category_name: str = None  # type: ignore

    class _Decorator:
        # The following attributes are set by RegisterMethodDecorator.register() so that this class has access to them
        # in the __set_name__() and __init__() methods.
        registering_key: Any = None
        registering_dict: Dict = {}
        owner_class: type = None  # type: ignore
        registration_dict_attribute_name: str = None  # type: ignore
        key_type: type = None  # type: ignore
        method_category_name: str = None  # type: ignore

        def __init__(self, method: Callable):
            # Called when each decorator is initialized.

            try:
                logger.trace(
                    f"Recording {self.method_category_name} method {method} for {self.key_type} "
                    f"{self.registering_key}"
                )
                if self.registering_key in self.registering_dict:
                    raise TypeError(
                        f"{self.method_category_name.capitalize()} method to handle {self.key_type} "
                        f"{self.registering_dict} registered multiple times"
                    )
                self.registering_dict[self.registering_key] = method

            except Exception as ex:
                self.registering_dict.clear()
                raise ex

        def __set_name__(self, owner, _):
            # Called only once per `ValidationImplementation` derived class. `owner` contains the class.

            try:
                logger.trace(f"Registering {self.method_category_name} methods for class {owner}")

                if not issubclass(owner, self.owner_class):
                    raise TypeError(f"Decorator expected to be used in a subclass of {self.owner_class.__name__}")

                # Copy registration_dict_attribute from parent class if parent class is a owner_class type.
                # Do not copy if a particular method has been defined (overridden) on the class.
                relevant_bases = [
                    b
                    for b in owner.mro()
                    if issubclass(b, self.owner_class)
                    and b != owner
                    and hasattr(b, self.registration_dict_attribute_name)
                ]
                if relevant_bases:
                    base = relevant_bases[0]
                    setattr(
                        owner,
                        self.registration_dict_attribute_name,
                        {
                            k: v
                            for k, v in getattr(base, self.registration_dict_attribute_name).items()
                            if k not in self.registering_dict
                        },
                    )
                else:
                    setattr(owner, self.registration_dict_attribute_name, dict())

                for type_, method in self.registering_dict.items():
                    logger.trace(f"Registering validation method {method}")
                    reg_dict = getattr(owner, self.registration_dict_attribute_name)
                    reg_dict[type_] = method

            finally:
                self.registering_dict.clear()

    @classmethod
    def register_method_for(cls, registering_key):
        cls._registering_key = registering_key

        if cls.owner_class is None:
            raise TypeError("Must set `owner_class`")
        if cls.registration_dict_attribute_name is None:
            raise TypeError("Must set `registration_dict_attribute_name`")
        if cls.key_type is None:
            raise TypeError("Must set `key_type`")
        if cls.method_category_name is None:
            raise TypeError("Must set `method_category_name`")

        cls._Decorator.registering_key = cls._registering_key
        cls._Decorator.registering_dict = cls._registering_dict
        cls._Decorator.owner_class = cls.owner_class
        cls._Decorator.registration_dict_attribute_name = cls.registration_dict_attribute_name
        cls._Decorator.key_type = cls.key_type
        cls._Decorator.method_category_name = cls.method_category_name

        return cls._Decorator
