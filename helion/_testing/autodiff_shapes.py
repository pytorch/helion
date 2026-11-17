"""Shared backward-graph fixtures for aliased singleton and broadcast shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.codecache import PyCodeCache
from torch.fx.experimental.proxy_tensor import make_fx

from helion.experimental.autodiff import FXToHelionConverter
from helion.experimental.autodiff import InputMapping
from helion.experimental.autodiff import differentiate_graph

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


# These coupled source boundaries are not a Cartesian knob grid: mask rank,
# equal-sized axes, reduced/nonlinear tangents, scalar/broadcast controls, and
# independent/shared nonlinear branches, local replication counts and both
# differentiated sides of a broadcast product exercise different shape proofs.
ALIAS_SHAPE_CASES = (
    "expanded_mask",
    "matrix_mask",
    "mask_sum",
    "mask_mean",
    "nonlinear_mask_mean",
    "scalar_mask",
    "broadcast_mask",
    "independent_outputs",
    "equal_axis_mask",
    "shared_nonlinear_outputs",
    "shared_scalar_gate",
    "reduced_expanded_mask",
    "counted_expanded_mask",
    "broadcast_product",
    "nonlinear_broadcast_product",
    "sum_then_expand",
    "sum_then_count",
    "unsqueeze_expand_mask",
    "unsqueeze_counted_mask",
    "sliced_unsqueeze_mask",
    "sliced_unsqueeze_count",
    "singleton_row_expand",
)

TRITON_ONLY_ALIAS_SHAPE_CASES = ("literal_singleton_expand",)


def alias_shape_backward(
    forward: Callable[..., tuple[torch.Tensor, ...]],
    inputs: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
) -> tuple[torch.fx.Graph, FXToHelionConverter, ModuleType, str]:
    """Convert a fixture once, retaining both graphs for CPU shape assertions."""
    graph = differentiate_graph(make_fx(forward)(*inputs).graph, inputs)
    before = str(graph)
    converter = FXToHelionConverter(
        graph,
        [
            InputMapping(f"arg{i}", f"input_{i}", value)
            for i, value in enumerate(inputs)
        ],
        inputs,
        tuple(tuple(output.shape) for output in outputs),
    )
    source = converter.convert()
    module = PyCodeCache.load(source)
    assert str(graph) == before
    return graph, converter, module, source


def alias_shape_case(
    case: str, device: str | torch.device
) -> tuple[
    Callable[..., tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...], list[int]
]:
    assert case in (*ALIAS_SHAPE_CASES, *TRITON_ONLY_ALIAS_SHAPE_CASES)
    if case == "singleton_row_expand":

        def singleton_row_expand(
            value: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor]:
            counts = mask.expand(2, 16).sum(0)
            return (torch.where(counts > 1, value, 0),)

        return (
            singleton_row_expand,
            (
                torch.randn(1, 16, device=device),
                (torch.arange(16, device=device) % 3 != 0).unsqueeze(0),
            ),
            [8],
        )
    if case in ("sliced_unsqueeze_mask", "sliced_unsqueeze_count"):

        def sliced_unsqueeze(
            value: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor]:
            counts = mask.unsqueeze(-1).expand(16, 2).sum(-1)
            threshold = 1 if case == "sliced_unsqueeze_count" else 0
            return (torch.where(counts > threshold, value, 0),)

        return (
            sliced_unsqueeze,
            (
                torch.linspace(-2, 2, 128, device=device).reshape(8, 16),
                torch.arange(16, device=device) % 3 != 0,
            ),
            [4],
        )
    if case in ("unsqueeze_expand_mask", "unsqueeze_counted_mask"):

        def unsqueeze_expand(value: torch.Tensor) -> tuple[torch.Tensor]:
            counts = (value > 0).unsqueeze(-1).expand(8, 16, 2).sum(-1)
            threshold = 1 if case == "unsqueeze_counted_mask" else 0
            return (torch.where(counts > threshold, value, 0),)

        return (
            unsqueeze_expand,
            (torch.linspace(-2, 2, 128, device=device).reshape(8, 16),),
            [4],
        )
    if case == "literal_singleton_expand":

        def singleton_expand(
            value: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor]:
            counts = mask.expand(2, 2).sum((0, 1), keepdim=True)
            return (torch.where(counts > 1, value, 0),)

        return (
            singleton_expand,
            (
                torch.randn(1, 16, device=device),
                torch.ones(1, 1, dtype=torch.bool, device=device),
            ),
            [1],
        )
    if case in ("sum_then_expand", "sum_then_count"):

        def reduced_then_expand(value: torch.Tensor) -> tuple[torch.Tensor]:
            repeated = value.sum(1, keepdim=True).expand(8, 2)
            if case == "sum_then_expand":
                gate = repeated.amax(1, keepdim=True) > 0
            else:
                gate = repeated.sum(1, keepdim=True) > 1
            return (torch.where(gate, value, 0),)

        return (
            reduced_then_expand,
            (torch.linspace(-0.05, 0.08, 128, device=device).reshape(8, 16),),
            [4],
        )
    if case in ("broadcast_product", "nonlinear_broadcast_product"):

        def broadcast_product(
            value: torch.Tensor, other: torch.Tensor
        ) -> tuple[torch.Tensor]:
            if case == "nonlinear_broadcast_product":
                value = value.sin()
            return (value.expand(8, 2) * other,)

        return (
            broadcast_product,
            (
                torch.linspace(-2, 2, 8, device=device).reshape(8, 1),
                torch.linspace(-3, 5, 16, device=device).reshape(8, 2),
            ),
            [4],
        )
    if case in ("reduced_expanded_mask", "counted_expanded_mask"):

        def reduced_mask(value: torch.Tensor) -> tuple[torch.Tensor]:
            viewed = value.view(8, 1)
            # >0 detects missing reduction axes; >1 also detects lost replication.
            threshold = int(case == "counted_expanded_mask")
            mask = (viewed > 0).expand(8, 2).view(8, 2, 1).sum(1) > threshold
            return (torch.where(mask.squeeze(-1), viewed.sin().squeeze(-1), 0),)

        return reduced_mask, (torch.linspace(-2, 2, 8, device=device),), [4]
    if case in (
        "independent_outputs",
        "shared_nonlinear_outputs",
        "shared_scalar_gate",
    ):

        def independent_outputs(
            value: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            gate = ~mask if case == "shared_scalar_gate" else mask
            viewed = value.view(8, 1)
            nonlinear = viewed.sin()
            if case == "shared_scalar_gate":
                nonlinear = torch.where(gate, nonlinear, 0)
            outputs = (
                torch.where(gate.expand(8, 1).view(8), value, 0),
                nonlinear.squeeze(-1),
            )
            if case == "shared_nonlinear_outputs":
                return (*outputs, viewed.cos().squeeze(-1))
            return outputs

        mask = torch.tensor(case != "shared_scalar_gate", device=device)
        return (
            independent_outputs,
            (torch.randn(8, device=device), mask),
            [4],
        )

    if case == "broadcast_mask":

        def broadcast_masked(
            value: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor]:
            return (torch.where(mask, value.view(8, 16, 1).sin().squeeze(-1), 0),)

        return (
            broadcast_masked,
            (
                torch.randn(8, 16, device=device),
                (torch.arange(16, device=device) % 3 != 0).unsqueeze(0),
            ),
            [4, 8],
        )

    if case == "scalar_mask":

        def scalar_masked(
            value: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor]:
            return (torch.where(mask, value.view(-1, 1).sin().squeeze(-1), 0),)

        return (
            scalar_masked,
            (torch.randn(8, device=device), torch.tensor(True, device=device)),
            [4],
        )

    width = 8 if case == "equal_axis_mask" else 16

    def masked(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        if mask.ndim == 1:
            mask = mask.unsqueeze(1)
        values = value.view(8, width, 1)
        if case == "nonlinear_mask_mean":
            values = values.sin()
        result = torch.where(mask.expand(8, width).view(8, width, 1), values, 0).view(
            8, width
        )
        if case == "mask_sum":
            result = result.sum(-1)
        elif case in ("mask_mean", "nonlinear_mask_mean"):
            result = result.mean(-1)
        return (result,)

    mask = torch.arange(8, device=device) % 2 == 0
    if case in ("matrix_mask", "equal_axis_mask"):
        mask = mask.unsqueeze(1)
    return masked, (torch.randn(8, width, device=device), mask), [4, 8]
