"""Hand-built FX graphs for the flash row epilogue's program matcher
(``test_cute_flash_row_epilogue_program.py`` and
``test_cute_flash_row_epilogue.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from .._compiler.cute import flash_row_epilogue
from ..language import memory_ops

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.fx.node import Argument

    from .._compiler.cute.flash_row_epilogue import FlashRowEpilogueProgram


def row_value(
    graph: torch.fx.Graph,
    target: str | Callable[..., object],
    args: tuple[Argument, ...],
    dtype: torch.dtype,
    shape: tuple[int, ...] = (2, 4, 8),
) -> torch.fx.Node:
    """A node of a hand-built row program graph (a placeholder for a string
    ``target``) with a meta value of ``dtype`` and ``shape``."""
    if isinstance(target, str):
        node = graph.placeholder(target)
    else:
        node = graph.call_function(target, args)
    node.meta["val"] = torch.empty(
        shape,
        dtype=dtype,
        device="meta",  # @ignore-device-lint
    )
    return node


def match_row_program(
    graph: torch.fx.Graph,
    o: torch.fx.Node,
    value: torch.fx.Node,
    out_dtype: torch.dtype = torch.float32,
) -> FlashRowEpilogueProgram | None:
    """The program storing ``value`` to an output of ``out_dtype``, with ``o``
    the normalized accumulator."""
    out = row_value(graph, "out", (), out_dtype)
    store = graph.call_function(memory_ops.store, (out, [], value))
    return flash_row_epilogue.match_flash_row_epilogue(
        store,
        head_dim=8,
        is_o_norm=lambda node: node is o,
        is_row_load=lambda node: None,
    )


def row_accumulator(graph: torch.fx.Graph) -> torch.fx.Node:
    """The normalized accumulator ``acc / 2.0`` in fp32."""
    acc = row_value(graph, "acc", (), torch.float32)
    return row_value(graph, torch.ops.aten.div.Tensor, (acc, 2.0), torch.float32)


def program_consts(program: FlashRowEpilogueProgram) -> list[float]:
    """The values of the program's constants."""
    return [float(cast("float", op.attrs[0])) for op in program.ops if op.op == "const"]


def casts_of(program: FlashRowEpilogueProgram, op_name: str) -> list[object]:
    """The dtypes of the casts of ``op_name`` ops."""
    return [
        op.attrs[0]
        for op in program.ops
        if op.op == "cast" and program.op(op.inputs[0]).op == op_name
    ]
