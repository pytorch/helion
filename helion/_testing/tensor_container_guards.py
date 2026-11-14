from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import Kernel


@dataclass
class TensorField:
    weight: torch.Tensor


class TensorTuple(NamedTuple):
    weight: torch.Tensor


@helion.kernel(static_shapes=False, autotune_effort="none")
def field_bias(x: torch.Tensor, holder: TensorField | TensorTuple) -> torch.Tensor:
    bias = holder.weight
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + bias[tile]
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def mapping_bias(x: torch.Tensor, holder: dict[str, torch.Tensor]) -> torch.Tensor:
    bias = holder["weight"]
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + bias[tile]
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def sequence_bias(
    x: torch.Tensor, holder: list[torch.Tensor] | tuple[torch.Tensor, ...]
) -> torch.Tensor:
    bias = holder[0]
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + bias[tile]
    return out


@helion.kernel(static_shapes=False, autotune_effort="none")
def nested_bias(x: torch.Tensor, holder: dict[str, list[TensorTuple]]) -> torch.Tensor:
    bias = holder["weights"][0].weight
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + bias[tile]
    return out


def container_bias_case(
    kind: str, x: torch.Tensor, bias: torch.Tensor
) -> tuple[Kernel[torch.Tensor], tuple[object, ...]]:
    if kind == "dataclass":
        return field_bias, (x, TensorField(bias))
    if kind == "namedtuple":
        return field_bias, (x, TensorTuple(bias))
    if kind == "dict":
        return mapping_bias, (x, {"weight": bias})
    if kind == "list":
        return sequence_bias, (x, [bias])
    if kind == "tuple":
        return sequence_bias, (x, (bias,))
    assert kind == "nested"
    return nested_bias, (x, {"weights": [TensorTuple(bias)]})


@helion.kernel(static_shapes=False, autotune_effort="none")
def field_increment(
    holder: TensorField | TensorTuple, direct: hl.constexpr, device: torch.device
) -> torch.Tensor:
    values = holder.weight
    for tile in hl.tile(values.size()):
        if direct:
            holder.weight[tile] = holder.weight[tile] + 2
        else:
            values[tile] = values[tile] + 2
    return holder.weight
