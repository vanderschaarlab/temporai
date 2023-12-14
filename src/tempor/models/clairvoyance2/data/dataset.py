# mypy: ignore-errors

import copy
from collections.abc import Sequence as SequenceABC
from typing import Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

from ..utils.dev import raise_not_implemented
from ._utils import time_index_equal
from .constants import T_SamplesIndexDtype
from .dataformat import (
    EventSamples,
    StaticSamples,
    T_ContainerInitializable,
    T_SampleIndex_Compatible,
    T_SampleIndexClass,
    T_SamplesIndexDtype_AsTuple,
    T_TSS_ContainerInitializable,
    TimeSeriesSamples,
    TMissingIndicator,
)
from .dataformat_base import Copyable, SupportsNewLike

T_TimeSeriesSamplesInitializable = Union[TimeSeriesSamples, Sequence[T_TSS_ContainerInitializable]]
T_StaticSamplesInitializable = Union[StaticSamples, T_ContainerInitializable]
T_EventSamplesInitializable = Union[EventSamples]  # pyright: ignore


# TODO: Should probably implement a number of mixins that TimeSeriesSamples etc. have.
class Dataset(Copyable, SupportsNewLike, SequenceABC):
    temporal_covariates: TimeSeriesSamples
    static_covariates: Optional[StaticSamples] = None
    event_covariates: Optional[EventSamples] = None
    temporal_targets: Optional[TimeSeriesSamples] = None
    temporal_treatments: Optional[TimeSeriesSamples] = None
    event_targets: Optional[EventSamples] = None
    event_treatments: Optional[EventSamples] = None

    _container_names = [
        "temporal_covariates",
        "static_covariates",
        "event_covariates",
        "temporal_targets",
        "temporal_treatments",
        "event_targets",
        "event_treatments",
    ]
    # NOTE: ^ Keep the above in the same order as these are initialized in the constructor.
    _container_order = {
        container_name: container_order for container_order, container_name in enumerate(_container_names)
    }

    def _init_time_series_samples(self, data: T_TimeSeriesSamplesInitializable, **kwargs) -> TimeSeriesSamples:
        if isinstance(data, TimeSeriesSamples):
            return data
        else:
            return TimeSeriesSamples(data, **kwargs)

    def _init_static_samples(self, data: T_StaticSamplesInitializable, **kwargs) -> StaticSamples:
        if isinstance(data, StaticSamples):
            return data
        else:
            return StaticSamples(data, **kwargs)

    def _init_event_samples(self, data: T_EventSamplesInitializable, **kwargs) -> EventSamples:
        if isinstance(data, EventSamples):
            return data
        else:
            return EventSamples(data, **kwargs)

    def __init__(
        self,
        temporal_covariates: T_TimeSeriesSamplesInitializable,
        static_covariates: Optional[T_StaticSamplesInitializable] = None,
        event_covariates: Optional[T_EventSamplesInitializable] = None,
        temporal_targets: Optional[T_TimeSeriesSamplesInitializable] = None,
        temporal_treatments: Optional[T_TimeSeriesSamplesInitializable] = None,
        event_targets: Optional[T_EventSamplesInitializable] = None,
        event_treatments: Optional[T_EventSamplesInitializable] = None,
        # NOTE: The below arguments are only applicable when *initializing* the above containers from DF, array etc.,
        # not applicable when passing as TimeSeriesSamples/StaticSamples objects.
        # TODO: Maybe this is not user-friendly, could redo such that it will always set these.
        sample_indices: Optional[T_SampleIndex_Compatible] = None,
        missing_indicator: TMissingIndicator = np.nan,
    ) -> None:
        self.temporal_covariates = self._init_time_series_samples(
            temporal_covariates,
            sample_indices=sample_indices,
            missing_indicator=missing_indicator,
        )
        if static_covariates is not None:
            self.static_covariates = self._init_static_samples(
                static_covariates,
                sample_indices=sample_indices,
                missing_indicator=missing_indicator,
            )
        if event_covariates is not None:
            self.event_covariates = self._init_event_samples(data=event_covariates)
        if temporal_targets is not None:
            self.temporal_targets = self._init_time_series_samples(
                temporal_targets,
                sample_indices=sample_indices,
                missing_indicator=missing_indicator,
            )
        if temporal_treatments is not None:
            self.temporal_treatments = self._init_time_series_samples(
                temporal_treatments,
                sample_indices=sample_indices,
                missing_indicator=missing_indicator,
            )
        if event_targets is not None:
            self.event_targets = self._init_event_samples(data=event_targets)
        if event_treatments is not None:
            self.event_treatments = self._init_event_samples(data=event_treatments)

        self._refresh_container_dicts()

        self.validate()

    def _sort_container_dict(self, container_dict):
        return {k: container_dict[k] for k in sorted(container_dict, key=lambda x: self._container_order[x])}

    def _refresh_container_dicts(self) -> None:
        self._static_data_containers: Dict[str, Optional[StaticSamples]] = {
            "static_covariates": self.static_covariates,
        }
        self._temporal_data_containers: Dict[str, Optional[TimeSeriesSamples]] = {
            "temporal_covariates": self.temporal_covariates,
            "temporal_targets": self.temporal_targets,
            "temporal_treatments": self.temporal_treatments,
        }
        self._event_data_containers: Dict[str, Optional[EventSamples]] = {
            "event_covariates": self.event_covariates,
            "event_targets": self.event_targets,
            "event_treatments": self.event_treatments,
        }

        self._all_data_containers: Dict[str, Union[StaticSamples, TimeSeriesSamples, EventSamples, None]] = copy.copy(
            self._static_data_containers  # type: ignore  # Invariance issue, ignore.
        )
        self._all_data_containers.update(self._temporal_data_containers)
        self._all_data_containers.update(self._event_data_containers)

        self._static_data_containers = self._sort_container_dict(self._static_data_containers)
        self._temporal_data_containers = self._sort_container_dict(self._temporal_data_containers)
        self._event_data_containers = self._sort_container_dict(self._event_data_containers)
        self._all_data_containers = self._sort_container_dict(self._all_data_containers)

    @staticmethod
    def _time_series_samples_repr(time_series_samples: TimeSeriesSamples) -> str:
        name = time_series_samples.__class__.__name__
        if time_series_samples.all_samples_same_n_timesteps:
            n_timesteps = str(time_series_samples.n_timesteps_per_sample[0])
        else:
            n_timesteps = "*"
        shape = f"[{time_series_samples.n_samples},{n_timesteps},{len(time_series_samples.features)}]"
        return f"{name}({shape})"

    @staticmethod
    def _static_samples_repr(static_samples: StaticSamples) -> str:
        name = static_samples.__class__.__name__
        shape = f"[{static_samples.n_samples},{len(static_samples.features)}]"
        return f"{name}({shape})"

    @staticmethod
    def _event_samples_repr(event_samples: EventSamples) -> str:
        name = event_samples.__class__.__name__
        shape = f"[{event_samples.n_samples},{len(event_samples.features)}]"
        return f"{name}({shape})"

    def __repr__(self) -> str:
        tab = "    "
        sep = f"\n{tab}"

        attributes_repr = ""
        for container_name, container in self.all_data_containers.items():
            if isinstance(container, StaticSamples):
                repr_helper: Callable = self._static_samples_repr
            elif isinstance(container, TimeSeriesSamples):
                repr_helper = self._time_series_samples_repr
            else:
                repr_helper = self._event_samples_repr
            attributes_repr += f"{sep}{container_name}={repr_helper(container)},"  # type: ignore

        return f"{self.__class__.__name__}({attributes_repr}\n)"

    @staticmethod
    def _sample_indices_eq(
        sample_indices_a: Sequence[T_SamplesIndexDtype], sample_indices_b: Sequence[T_SamplesIndexDtype]
    ) -> bool:
        if len(sample_indices_a) != len(sample_indices_b):
            return False
        else:
            return all(a == b for a, b in zip(sample_indices_a, sample_indices_b))

    @property
    def n_samples(self) -> int:
        return self.temporal_covariates.n_samples

    @property
    def sample_index(self) -> T_SampleIndexClass:  # type: ignore
        return self.temporal_covariates.sample_index

    @property
    def sample_indices(self) -> Sequence[T_SamplesIndexDtype]:
        return self.temporal_covariates.sample_indices

    @property
    def static_data_containers(self) -> Dict[str, StaticSamples]:
        self._refresh_container_dicts()
        return {k: v for k, v in self._static_data_containers.items() if v is not None}

    @property
    def temporal_data_containers(self) -> Dict[str, TimeSeriesSamples]:
        self._refresh_container_dicts()
        return {k: v for k, v in self._temporal_data_containers.items() if v is not None}

    @property
    def event_data_containers(self) -> Dict[str, EventSamples]:
        self._refresh_container_dicts()
        return {k: v for k, v in self._event_data_containers.items() if v is not None}

    @property
    def all_data_containers(self) -> Dict[str, Union[StaticSamples, TimeSeriesSamples, EventSamples]]:
        self._refresh_container_dicts()
        return {k: v for k, v in self._all_data_containers.items() if v is not None}

    def validate(self) -> None:
        # Validate number of samples.
        # NOTE: Strictly this isn't necessary as checked below, but here for convenience.
        n_samples_expected = self.temporal_covariates.n_samples
        for container_name, container in self.all_data_containers.items():
            if container.n_samples != n_samples_expected:
                raise ValueError(
                    f"Expected {n_samples_expected} samples " f"but found {container.n_samples} in `{container_name}`"
                )
        # Validate that the sample indices match across all containers.
        sample_indices_expected = self.temporal_covariates.sample_indices
        for container_name, container in self.all_data_containers.items():
            if not self._sample_indices_eq(container.sample_indices, sample_indices_expected):
                raise ValueError(
                    f"The sample indices in `{container_name}` didn't match those in `temporal_covariates`"
                )

    def check_temporal_containers_have_same_time_index(self) -> Tuple[bool, Optional[Tuple[str, str]]]:
        list_container_name_container = list(self.temporal_data_containers.items())
        pairs = list(zip(list_container_name_container, list_container_name_container[1:]))
        for (a_name, a), (b_name, b) in pairs:
            if not time_index_equal(a, b):
                return False, (a_name, b_name)
        return True, None

    @staticmethod
    def new_like(like: "Dataset", **kwargs) -> "Dataset":
        kwargs = SupportsNewLike.process_kwargs(
            kwargs,
            dict(
                temporal_covariates=like.temporal_covariates,
                static_covariates=like.static_covariates,
                event_covariates=like.event_covariates,
                temporal_targets=like.temporal_targets,
                temporal_treatments=like.temporal_treatments,
                event_targets=like.event_targets,
                event_treatments=like.event_treatments,
                sample_indices=like.temporal_covariates.sample_indices,
                missing_indicator=like.temporal_covariates.missing_indicator,
            ),
        )
        return Dataset(**kwargs)  # type: ignore  # Mypy complains about kwargs but it's fine.

    @staticmethod
    def new_empty_like(like: "Dataset", **kwargs) -> "Dataset":
        raise_not_implemented("Dataset.new_empty_like method")

    # --- Sequence Interface ---

    def __getitem__(self, key) -> "Dataset":
        if not isinstance(key, (slice)) and not hasattr(key, "__len__"):
            if not isinstance(key, T_SamplesIndexDtype_AsTuple):
                raise KeyError(f"Expected `key` to be one of types {T_SamplesIndexDtype_AsTuple} but {type(key)} found")
            key = (key,)
        kwargs = dict()
        for container_name, container in self.all_data_containers.items():
            kwargs[container_name] = container[key]
        return self.new_like(like=self, **kwargs)

    def __len__(self) -> int:
        return len(self.temporal_covariates)

    def __iter__(self) -> Iterator["Dataset"]:
        for idx in self.sample_indices:
            yield self[idx]

    def __contains__(self, value) -> bool:
        return value in self.sample_indices

    def __reversed__(self) -> Iterator["Dataset"]:
        for idx in reversed(self.sample_indices):
            yield self[idx]

    def index(self, value, start=0, stop=None):
        raise NotImplementedError

    def count(self, value):
        raise NotImplementedError

    # --- Sequence Interface (End) ---
