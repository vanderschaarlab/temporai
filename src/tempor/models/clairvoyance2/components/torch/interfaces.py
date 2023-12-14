"""
Useful reusable interfaces for PyTorch models.
"""
# mypy: ignore-errors

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...data import Dataset
from ...data.constants import T_SamplesIndexDtype
from ...interface import Horizon, SavableModelMixin, TTreatmentScenarios

TPreparedData = Union[torch.Tensor, DataLoader]


class OrganizedModule(nn.Module, ABC):
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float
        nn.Module.__init__(self)

    @abstractmethod
    def _init_submodules(self) -> None:
        ...

    @abstractmethod
    def _init_optimizers(self) -> None:
        ...

    @abstractmethod
    def _init_inferred_params(self, data: Dataset, **kwargs) -> None:
        ...

    @abstractmethod
    def _prep_data_for_fit(self, data: Dataset, **kwargs) -> Tuple[TPreparedData, ...]:
        ...

    @abstractmethod
    def _prep_submodules_for_fit(self) -> None:
        ...

    def prep_fit(self, data: Dataset, **kwargs) -> Tuple[TPreparedData, ...]:
        self._init_inferred_params(data, **kwargs)
        prepared_data = self._prep_data_for_fit(data=data, **kwargs)
        self._init_submodules()
        self._init_optimizers()
        self._prep_submodules_for_fit()
        return prepared_data

    def set_attributes_from_kwargs(self, check_unknown_kwargs: bool = True, **kwargs):
        if "device" in kwargs:
            device = kwargs.pop("device")
            if isinstance(device, str):
                device = torch.device(device)
            assert isinstance(device, torch.device)
            self.device = device
        if "dtype" in kwargs:
            dtype = kwargs.pop("dtype")
            assert isinstance(dtype, torch.dtype)
            self.dtype = dtype
        if check_unknown_kwargs and len(kwargs) > 0:
            raise ValueError(f"Unknown kwarg(s) passed: {kwargs}")


class OrganizedPredictorModuleMixin(ABC):
    @abstractmethod
    def _prep_data_for_predict(self, data: Dataset, horizon: Optional[Horizon], **kwargs) -> Tuple[TPreparedData, ...]:
        ...

    @abstractmethod
    def _prep_submodules_for_predict(self) -> None:
        ...

    def prep_predict(self, data: Dataset, horizon: Optional[Horizon], **kwargs) -> Tuple[TPreparedData, ...]:
        prepared_data = self._prep_data_for_predict(data=data, horizon=horizon, **kwargs)
        self._prep_submodules_for_predict()
        return prepared_data


class OrganizedTreatmentEffectsModuleMixin(ABC):
    @abstractmethod
    def _prep_data_for_predict_counterfactuals(
        self,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: TTreatmentScenarios,
        horizon: Optional[Horizon],
        **kwargs,
    ) -> Tuple[TPreparedData, ...]:
        ...

    @abstractmethod
    def _prep_submodules_for_predict_counterfactuals(self) -> None:
        ...

    def prep_predict_counterfactuals(
        self,
        data: Dataset,
        sample_index: T_SamplesIndexDtype,
        treatment_scenarios: TTreatmentScenarios,
        horizon: Optional[Horizon],
        **kwargs,
    ) -> Tuple[TPreparedData, ...]:
        prepared_data = self._prep_data_for_predict_counterfactuals(
            data=data, sample_index=sample_index, treatment_scenarios=treatment_scenarios, horizon=horizon, **kwargs
        )
        self._prep_submodules_for_predict_counterfactuals()
        return prepared_data


class CustomizableLossMixin(ABC):
    def __init__(self, loss_fn) -> None:
        assert isinstance(loss_fn, nn.Module)
        self.loss_fn: nn.Module = loss_fn

    @abstractmethod
    def process_output_for_loss(self, output: torch.Tensor, **kwargs) -> torch.Tensor:
        ...


class SavableTorchModelMixin(SavableModelMixin):
    state_dict: Callable
    load_state_dict: Callable
    _init_submodules: Callable[[], None]

    def save(self, path: str) -> None:
        super().save(path)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str):
        # Load `params` and `inferred params`:
        loaded = super().load(path)

        # Run _init_submodules() if our model provides this method.
        has_init_submodules_method = False
        try:
            _ = loaded._init_submodules  # pylint: disable=protected-access
            has_init_submodules_method = True
        except AttributeError:
            pass
        if has_init_submodules_method:
            loaded._init_submodules()  # pylint: disable=protected-access

        # Finally, load the state dict.
        loaded.load_state_dict(torch.load(path))
        return loaded
