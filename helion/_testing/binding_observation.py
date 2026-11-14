"""CPU observations of compiled bindings without invoking a device launcher."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..runtime.kernel import BoundKernel


def observe_compiled_binding(bound: BoundKernel, *args: object) -> BoundKernel:
    # Keep real graph construction and generated-host syntax checks while
    # replacing only native execution for eager, direct and prepared calls.
    ast.parse(bound.to_code())
    bound._run = lambda *unused: bound
    return bound


def scalar_additions(bound: BoundKernel) -> list[object]:
    host_function = bound.host_function
    assert host_function is not None
    return [
        node.args[1]
        for graph in host_function.device_ir.graphs
        for node in graph.graph.nodes
        if node.target is torch.ops.aten.add.Tensor
    ]


def assert_scalar_addition(bound: BoundKernel, expected: float) -> None:
    assert scalar_additions(bound) == [expected]
    ast.parse(bound.to_code())
