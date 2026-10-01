from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_ir import GraphInfo
    from .chained_tcgen05 import _VectorLeaf


def has_inplace_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """Typed discovery only; exact addresses/ownership are proved in codegen."""
    from . import chained_matmul as chain

    root = chain._root_graph(graphs)
    if root is None:
        return False
    dots = {node for node in root.graph.nodes if node.target is chain.dot}
    for dot in dots:
        for operand in dot.args[:2]:
            operand = cast("Node", operand)
            ancestors = chain._ancestors(operand)
            dtype = operand.meta["val"].dtype
            if (
                dtype not in (torch.bfloat16, torch.float16)
                or chain._direct_operand(operand)
                or ancestors & dots
            ):
                continue
            for node in ancestors:
                if node.target is not memory_ops.load or node.meta["val"].ndim != 2:
                    continue
                source = cast("Node", node.args[0])
                if source.target is not _tracing_ops._host_tensor:
                    continue
                value = source.meta["val"]
                if value.dtype == dtype and any(
                    stride == 1 for stride in value.stride()
                ):
                    return True
    return False


def sw128_ownership(shape: tuple[int, int], inner: int) -> bool:
    """Prove the existing 16-bit SW128 layout and eight-value TV ownership.

    In oriented (row, col) coordinates both K/MN constructors are
    S<3,4,3> applied to col%64 + 64*row + (col//64)*64*height.
    Whole 64-column panels have a multiple-of-512 extent. The XOR only
    permutes bits4..6 within each 1024-cell slab, leaving every aligned
    eight-cell group intact. The TV layout bijects all rows/columns.
    This predicate is only used with _stage's existing _layout constructor;
    it does not admit arbitrary user-provided shared layouts.
    """
    if inner not in (0, 1):
        return False
    width, height = shape[inner], shape[1 - inner]
    if width < 64 or width % 64 or height <= 0 or height % 8:
        return False
    columns = width // 8
    if columns > 128 or columns & (columns - 1):
        return False
    return height % (128 // columns) == 0


@dataclass
class PointwiseInplace:
    """Per-codegen state; the disabled path preserves every emitted byte."""

    enabled: bool
    activated: bool = False

    def select(
        self,
        leaves: Sequence[_VectorLeaf],
        shape: tuple[int, int],
        inner: int,
        dtype: torch.dtype,
    ) -> int | None:
        if (
            not self.enabled
            or dtype not in (torch.bfloat16, torch.float16)
            or not sw128_ownership(shape, inner)
        ):
            return None
        width = shape[inner]
        for index, leaf in enumerate(leaves):
            if leaf.dtype == dtype and leaf.outer_stride >= width:
                self.activated = True
                return index
        return None

    def validate(self) -> None:
        if self.enabled and not self.activated:
            raise exc.BackendUnsupported(
                "cute", "inplace async requires an admitted same-dtype vector operand"
            )


def raw_preload(
    tag: str,
    raw: str,
    target: str,
    rows: int,
    columns: int,
    dtype: str,
    leaf: int,
) -> list[str]:
    return [
        f"{raw}_target = {target}",
        f"{raw}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), {dtype}, num_bits_per_copy=128), cute.make_layout(({rows}, {columns}), stride=({columns}, 1)), cute.make_layout((1, 8)))",
        f"{raw}_thread = {raw}_copy.get_slice(chain_thread)",
        f"cute.copy({raw}_copy, {raw}_thread.partition_S({tag}_leaf_{leaf}_source), {raw}_thread.partition_D({raw}_target))",
        "cute.arch.cp_async_commit_group()",
        "cute.arch.cp_async_wait_group(0)",
        "cute.arch.sync_threads()",
        f"{raw}_partition = {tag}_thread.partition_S({raw}_target)",
    ]
