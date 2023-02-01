from typing import Dict, Optional

import pydantic

import tempor.data._settings as settings
import tempor.data._types as types
import tempor.data.samples as s


class _DataBundleConfig:
    arbitrary_types_allowed = True
    validate_assignment = True
    extra = "forbid"


class _ContainerFlavorSpecValidator(pydantic.BaseModel):
    spec: types.ContainerFlavorSpec


@pydantic.dataclasses.dataclass(config=_DataBundleConfig)
class DataBundle:
    Xt: s.TimeSeriesSamples
    Xs: Optional[s.StaticSamples] = None
    Xe: Optional[s.EventSamples] = None
    Yt: Optional[s.TimeSeriesSamples] = None
    Ys: Optional[s.StaticSamples] = None
    Ye: Optional[s.EventSamples] = None
    At: Optional[s.TimeSeriesSamples] = None
    As: Optional[s.StaticSamples] = None
    Ae: Optional[s.EventSamples] = None

    container_flavor_spec: types.ContainerFlavorSpec = pydantic.Field(default=settings.DEFAULT_CONTAINER_FLAVOR_SPEC)

    @property
    def get_time_series_containers(self) -> Dict[str, s.TimeSeriesSamples]:
        return {
            s_: getattr(self, s_, None)  # type: ignore
            for s_ in ("Xt", "Yt", "At")
            if getattr(self, s_, None) is not None
        }

    @property
    def get_static_containers(self) -> Dict[str, s.StaticSamples]:
        return {
            s_: getattr(self, s_, None)  # type: ignore
            for s_ in ("Xs", "Ys", "As")
            if getattr(self, s_, None) is not None
        }

    @property
    def get_event_containers(self) -> Dict[str, s.EventSamples]:
        return {
            s_: getattr(self, s_, None)  # type: ignore
            for s_ in ("Xe", "Ye", "Ae")
            if getattr(self, s_, None) is not None
        }

    @staticmethod
    def from_data_containers(
        *,
        Xt: types.DataContainer,
        Xs: Optional[types.DataContainer] = None,
        Xe: Optional[types.DataContainer] = None,
        Yt: Optional[types.DataContainer] = None,
        Ys: Optional[types.DataContainer] = None,
        Ye: Optional[types.DataContainer] = None,
        At: Optional[types.DataContainer] = None,
        As: Optional[types.DataContainer] = None,
        Ae: Optional[types.DataContainer] = None,
        container_flavor_spec: Optional[types.ContainerFlavorSpec] = None,
    ):
        if container_flavor_spec is None:
            container_flavor_spec = settings.DEFAULT_CONTAINER_FLAVOR_SPEC
        container_flavor_spec = _ContainerFlavorSpecValidator(spec=container_flavor_spec).spec

        bundle = DataBundle(
            Xt=_make_data_bundle_sample(  # pyright: ignore
                Xt,
                s.TimeSeriesSamples,
                container_flavor=container_flavor_spec["Xt"],
            ),
            Xs=_make_data_bundle_sample(
                Xs,
                s.StaticSamples,
                container_flavor=container_flavor_spec["Xs"],
            ),
            Xe=_make_data_bundle_sample(
                Xe,
                s.EventSamples,
                container_flavor=container_flavor_spec["Xe"],
            ),
            Yt=_make_data_bundle_sample(
                Yt,
                s.TimeSeriesSamples,
                container_flavor=container_flavor_spec["Yt"],
            ),
            Ys=_make_data_bundle_sample(
                Ys,
                s.StaticSamples,
                container_flavor=container_flavor_spec["Ys"],
            ),
            Ye=_make_data_bundle_sample(
                Ye,
                s.EventSamples,
                container_flavor=container_flavor_spec["Ye"],
            ),
            At=_make_data_bundle_sample(
                At,
                s.TimeSeriesSamples,
                container_flavor=container_flavor_spec["At"],
            ),
            As=_make_data_bundle_sample(
                As,
                s.StaticSamples,
                container_flavor=container_flavor_spec["As"],
            ),
            Ae=_make_data_bundle_sample(
                Ae,
                s.EventSamples,
                container_flavor=container_flavor_spec["Ae"],
            ),
        )
        return bundle


def _make_data_bundle_sample(
    data_container: Optional[types.DataContainer], SamplesClass: type, container_flavor: Optional[types.ContainerFlavor]
):
    if data_container is None:
        return None
    else:
        return SamplesClass(data=data_container, container_flavor=container_flavor)
