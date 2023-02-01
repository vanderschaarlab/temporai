import abc
from typing import TYPE_CHECKING, Optional

import tempor.data.bundle
import tempor.data.types
from tempor.log import logger

from . import _base_model as bm
from . import _types as types


class TemporTransformerModel(bm.TemporBaseModel):
    class _TransformArgsValidator(bm.MethodArgsValidator):  # pylint: disable=protected-access
        pass

    def transform(
        self,
        X: Optional[tempor.data.types.DataContainer] = None,  # Alias for Xt.
        Y: Optional[tempor.data.types.DataContainer] = None,  # Alias for Yt.
        A: Optional[tempor.data.types.DataContainer] = None,  # Alias for At.
        *,
        Xt: Optional[tempor.data.types.DataContainer] = None,
        Xs: Optional[tempor.data.types.DataContainer] = None,
        Xe: Optional[tempor.data.types.DataContainer] = None,
        Yt: Optional[tempor.data.types.DataContainer] = None,
        Ys: Optional[tempor.data.types.DataContainer] = None,
        Ye: Optional[tempor.data.types.DataContainer] = None,
        At: Optional[tempor.data.types.DataContainer] = None,
        As: Optional[tempor.data.types.DataContainer] = None,
        Ae: Optional[tempor.data.types.DataContainer] = None,
        # ---
        # NOTE: Since currently only support one container flavor per container type, this argument is not
        # shown in th public API:
        container_flavor_spec: Optional[tempor.data.types.ContainerFlavorSpec] = None,
        # ---
        **kwargs,
    ) -> tempor.data.bundle.DataBundle:
        logger.debug(f"Validating transform() arguments on {self.__class__.__name__}")
        args_validator = self._TransformArgsValidator(
            X=X, Y=Y, A=A, Xt=Xt, Xs=Xs, Xe=Xe, Yt=Yt, Ys=Ys, Ye=Ye, At=At, As=As, Ae=Ae
        )
        if TYPE_CHECKING:  # pragma: no cover
            assert args_validator.Xt is not None
        logger.debug(f"Creating {tempor.data.bundle.DataBundle.__name__} in {self.__class__.__name__} transform() call")
        data = tempor.data.bundle.DataBundle.from_data_containers(
            Xt=args_validator.Xt,
            Xs=args_validator.Xs,
            Xe=args_validator.Xe,
            Yt=args_validator.Yt,
            Ys=args_validator.Ys,
            Ye=args_validator.Ye,
            At=args_validator.At,
            As=args_validator.As,
            Ae=args_validator.Ae,
            container_flavor_spec=container_flavor_spec,
        )
        self._validate_method_config(data, method_type=types.MethodTypes.TRANSFORM)
        transformed_data = self._transform(data, **kwargs)
        return transformed_data

    @abc.abstractmethod
    def _transform(
        self, data: tempor.data.bundle.DataBundle, **kwargs
    ) -> tempor.data.bundle.DataBundle:  # pragma: no cover
        ...
