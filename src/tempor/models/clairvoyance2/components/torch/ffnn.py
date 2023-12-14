# mypy: ignore-errors

from typing import Optional, OrderedDict, Sequence

import torch
import torch.nn as nn

from .common import init_activation


class FeedForwardNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = tuple(),
        out_activation: Optional[str] = "ReLU",
        hidden_activations: Optional[str] = "ReLU",
    ) -> None:
        super().__init__()

        list_dims = [in_dim] + list(hidden_dims) + [out_dim]
        dim_pairs = list(zip(list_dims[:-1], list_dims[1:]))
        layer_counter = 0
        ordered_dict_components: OrderedDict[str, nn.Module] = OrderedDict()

        for in_feat, out_feat in dim_pairs[:-1]:
            ordered_dict_components[f"linear_{layer_counter}"] = nn.Linear(in_feat, out_feat)
            if hidden_activations is not None:
                ordered_dict_components[f"activation_{layer_counter}"] = init_activation(hidden_activations)
            layer_counter += 1

        in_feat, out_feat = dim_pairs[-1]
        ordered_dict_components[f"linear_{layer_counter}"] = nn.Linear(in_feat, out_feat)
        if out_activation is not None:
            ordered_dict_components[f"activation_{layer_counter}"] = init_activation(out_activation)

        self.seq = nn.Sequential(ordered_dict_components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
