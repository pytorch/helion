"""Find complete vector reductions whose free iota needs an owning coordinate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch
from torch._inductor.ir import Reduction

from ...language import _tracing_ops
from ..device_ir import RootGraphInfo
from ..inductor_lowering import ReductionLowering

if TYPE_CHECKING:
    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST


def free_iota_reductions(
    env: CompileEnvironment, graphs: list[GraphInfo]
) -> dict[Node, tuple[Node, ...]]:
    """Prove a full rank-one static iota flows into a root vector reduction.

    Scalar-grid indices and host tensor shapes are not element provenance.
    Follow only same-graph full-vector values, including tensor load indices;
    symbolic/tiled vectors already carry their own block identity. Captures and
    multi-axis contractions retain their existing ownership proofs.
    """
    result = {}
    for info in graphs:
        if not isinstance(info, RootGraphInfo):
            continue
        for node in info.graph.nodes:
            lowering = node.meta.get("lowering")
            if not isinstance(lowering, ReductionLowering):
                continue
            reduction = lowering.buffer.data
            assert isinstance(reduction, Reduction)
            if reduction.ranges or len(reduction.reduction_ranges) != 1:
                continue
            extent = reduction.reduction_ranges[0]
            if (
                not isinstance(extent, (int, sympy.Integer))
                or extent <= 1
                or not env.block_sizes[lowering.block_index].reduction
            ):
                continue
            pending = list(node.all_input_nodes)
            seen = set()
            iotas = []
            while pending:
                value = pending.pop()
                if value in seen or value.graph is not info.graph:
                    continue
                seen.add(value)
                tensor = value.meta.get("val")
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.ndim != 1
                    or not isinstance(tensor.shape[0], int)
                    or tensor.shape[0] != extent
                    or value.target is _tracing_ops._host_tensor
                ):
                    continue
                if value.target is torch.ops.prims.iota.default:
                    iotas.append(value)
                else:
                    pending.extend(value.all_input_nodes)
            if iotas:
                result[node] = tuple(iotas)
    return result


def owned_iota_reduction_axes(cg: GenerateAST) -> frozenset[int]:
    """Retain native reductions when the producer uses that active full axis.

    Use the ordinary iota resolver itself, not coincidental selected extents.
    A rolled root producer cannot see its reduction loop's coordinate; it must
    be evaluated by the complete-fragment owner before rolling instead.
    """
    from ..compile_environment import CompileEnvironment
    from .cute_reshape import _resolve_dim_block_id

    env = CompileEnvironment.current()
    result = set()
    for node, iotas in free_iota_reductions(
        env, cg.host_function.device_ir.graphs
    ).items():
        lowering = node.meta["lowering"]
        assert isinstance(lowering, ReductionLowering)
        axis = lowering.block_index
        if all(
            _resolve_dim_block_id(cg, iota.meta["val"], 0) == axis for iota in iotas
        ):
            result.add(axis)
    return frozenset(result)
