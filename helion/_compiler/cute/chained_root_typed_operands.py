"""Typed contraction images with explicit root-stage read/publication lifetimes.

Only pure FP32-to-FP16/BF16 cast and rank-two permutation paths are candidates.
The original expression, masks and native operand layout are checked again at
emission. FP32 seed/export/other reads remain in the ordinary result workspace.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from .chained_result_transport import is_m64_chain
from .chained_workspace import _reuse_requests
from .chained_workspace import plan_contraction_workspace
from .warp_specialized_plan import SharedBufferRequest
from .warp_specialized_plan import SharedMemoryLayoutPlan

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chained_fragment_epilogue import FragmentExpression
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _ScanInput


@dataclass(frozen=True)
class TypedOperandUse:
    stage: int
    operand_index: int
    operand: Node
    source_stage: int
    dtype: torch.dtype
    transpose: bool
    path: tuple[Node, ...]

    @property
    def physical_transpose(self) -> bool:
        return self.transpose != (self.operand_index == 1)


@dataclass(frozen=True)
class TypedOperandImage:
    source_stage: int
    dtype: torch.dtype
    name: str
    shape: tuple[int, int]
    uses: tuple[TypedOperandUse, ...]


@dataclass(frozen=True)
class RootTypedOperands:
    images: tuple[TypedOperandImage, ...]
    operands: SharedMemoryLayoutPlan
    fp32: SharedMemoryLayoutPlan

    def use(
        self, stage: int, operand_index: int
    ) -> tuple[TypedOperandImage, TypedOperandUse] | None:
        return next(
            (
                (image, use)
                for image in self.images
                for use in image.uses
                if (use.stage, use.operand_index) == (stage, operand_index)
            ),
            None,
        )

    def pointer(self, stage: int, role: str, dtype: str) -> str:
        region = self.operands.region(f"chain_{stage}_{role}_staging")
        return f"cute.recast_ptr(chain_operand_workspace + {region.byte_offset}, dtype={dtype})"


def _cast_path(
    dots: tuple[Node, ...], stage: int, index: int
) -> TypedOperandUse | None:
    operand = dots[stage].args[index]
    if not isinstance(operand, Node):
        return None
    node, path, transpose, dtype = operand, [], False, None
    while node not in dots:
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or node.kwargs:
            return None
        if node.target is torch.ops.prims.convert_element_type.default:
            if len(node.args) != 2 or dtype is not None:
                return None
            dtype = node.args[1]
            if dtype not in (torch.float16, torch.bfloat16):
                return None
        elif node.target is _tracing_ops._new_var:
            if len(node.args) != 1:
                return None
        elif (
            node.target is torch.ops.aten.t.default
            and len(node.args) == 1
            or (
                node.target is torch.ops.aten.permute.default
                and len(node.args) == 2
                and node.args[1] in ((1, 0), [1, 0])
            )
        ):
            transpose = not transpose
        elif node.target is torch.ops.aten.transpose.int and len(node.args) == 3:
            left, right = node.args[1:]
            if not (
                type(left) is int
                and type(right) is int
                and -2 <= left < 2
                and -2 <= right < 2
                and left % 2 != right % 2
            ):
                return None
            transpose = not transpose
        else:
            return None
        path.append(node)
        source = node.args[0]
        if not isinstance(source, Node):
            return None
        node = source
    source_stage = dots.index(node)
    if (
        source_stage >= stage
        or dtype is None
        or node.meta["val"].dtype != torch.float32
        or operand.meta["val"].dtype != dtype
    ):
        return None
    return TypedOperandUse(
        stage,
        index,
        operand,
        source_stage,
        cast("torch.dtype", dtype),
        transpose,
        tuple(path),
    )


def has_root_typed_operand_candidate(graphs: Sequence[GraphInfo]) -> bool:
    from ...language.matmul_ops import dot
    from ..device_ir import ForLoopGraphInfo

    if any(isinstance(graph, ForLoopGraphInfo) for graph in graphs):
        return False
    dots = tuple(
        node for graph in graphs for node in graph.graph.nodes if node.target is dot
    )
    return any(
        _cast_path(dots, stage, index) is not None
        for stage in range(len(dots))
        for index in (0, 1)
    )


def plan_root_typed_operands(plan: ChainedMatmulPlan) -> RootTypedOperands | None:
    """Plan ordinary M64 roots without changing any contraction or cast.

    The same two-event stage model as the FP32 arena is used: all operand/MMA
    reads finish at event2s before results publish at event2s+1. A typed image
    can therefore reuse its final consumer's transient operand allocation only
    after the emitter's completion wait, load fence and full-CTA join.
    """
    from .chained_matmul import _shape
    from .chained_tcgen05 import _prefetch_final_b

    if (
        not is_m64_chain(plan)
        or len(plan.dots) < 2
        or plan.threads != 128
        or plan.contraction_groups is not None
        or plan.k_schedule is not None
        or plan.direct_output
        or _prefetch_final_b(plan)
    ):
        return None
    uses = tuple(
        use
        for stage in range(len(plan.dots))
        for index in (0, 1)
        if (use := _cast_path(plan.dots, stage, index)) is not None
        and _shape(use.operand)
        == (
            plan.shapes[use.source_stage][:2][::-1]
            if use.transpose
            else plan.shapes[use.source_stage][:2]
        )
    )
    if not uses:
        return None
    if plan.pointwise_cache is not None:
        cached = {entry.node for entry in plan.pointwise_cache.entries}
        if any(not cached.isdisjoint(use.path) for use in uses):
            # A retained operand must still bind the producer's original FP32
            # fragment, not an independently published cache along its cast
            # path. Other cache recipes may read that producer: the FP32
            # workspace accounts those reads separately from elided operands.
            return None
    keys = tuple(dict.fromkeys((use.source_stage, use.dtype) for use in uses))
    images = tuple(
        TypedOperandImage(
            source_stage,
            dtype,
            f"chain_typed_{image_index}",
            plan.shapes[source_stage][:2],
            tuple(
                use
                for use in uses
                if (use.source_stage, use.dtype) == (source_stage, dtype)
            ),
        )
        for image_index, (source_stage, dtype) in enumerate(keys)
    )
    skipped = frozenset((use.stage, use.operand_index) for use in uses)
    requests = [
        SharedBufferRequest(
            image.name,
            2 * image.shape[0] * image.shape[1],
            128,
            2 * image.source_stage + 1,
            2 * max(use.stage for use in image.uses) + 1,
        )
        for image in images
    ]
    requests.extend(
        SharedBufferRequest(
            f"chain_{stage}_{role}_staging",
            2 * rows * k,
            128,
            2 * stage,
            2 * stage + 1,
        )
        for stage, (m, n, k) in enumerate(plan.shapes)
        for index, role, rows in ((0, "a", m), (1, "b", n))
        if (stage, index) not in skipped
    )
    return RootTypedOperands(
        images,
        _reuse_requests(tuple(requests)),
        plan_contraction_workspace(
            plan, include_final=False, skip_operand_reads=skipped, omit_unused=True
        ),
    )


def _signature(expression: FragmentExpression) -> tuple[str, tuple[str, ...]]:
    """Compare exact original pure expressions after harmless SSA expansion."""
    definitions: dict[str, ast.expr] = {}

    class Inline(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            return definitions.get(node.id, node)

    inline = Inline()
    for line in expression.expression.lines:
        statement = ast.parse(line).body
        if (
            len(statement) != 1
            or not isinstance(statement[0], ast.Assign)
            or len(statement[0].targets) != 1
            or not isinstance(statement[0].targets[0], ast.Name)
        ):
            from .chained_matmul import _UnsupportedChain

            raise _UnsupportedChain("typed root image is not a pure SSA expression")
        assignment = statement[0]
        definitions[cast("ast.Name", assignment.targets[0]).id] = cast(
            "ast.expr", inline.visit(assignment.value)
        )
    return (
        ast.dump(inline.visit(ast.parse(expression.value, mode="eval").body)),
        tuple(
            sorted(
                ast.dump(inline.visit(ast.parse(bound, mode="eval").body))
                for bound in expression.domain
            )
        ),
    )


def validate_root_typed_operands(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    bindings: Mapping[tuple[int, str], int],
    boundaries: dict[Node, str],
    scans: list[_ScanInput],
) -> None:
    """Late proof before allocating: exact masks, SSA casts and native views."""
    from .chained_fragment_epilogue import bind_fragment_expression
    from .chained_matmul import _UnsupportedChain

    retained = plan.root_typed_operands
    assert retained is not None
    if plan_root_typed_operands(plan) != retained:
        raise _UnsupportedChain("typed root operand plan changed")
    for image in retained.images:
        signature = None
        source = plan.dots[image.source_stage]
        for use in image.uses:
            role = "a" if use.operand_index == 0 else "b"
            if bindings[use.stage, role] != (0 if use.physical_transpose else 1):
                raise _UnsupportedChain("typed root operand native major mode differs")
            coords = ("typed_row", "typed_column")
            expression = bind_fragment_expression(
                cg,
                plan,
                boundaries,
                scans,
                source,
                use.operand,
                coords,
                coords[::-1] if use.transpose else coords,
                "typed_fragment",
                "cutlass.Float16"
                if image.dtype == torch.float16
                else "cutlass.BFloat16",
            )
            current = _signature(expression)
            if signature is not None and signature != current:
                raise _UnsupportedChain("typed root operand masks or casts differ")
            signature = current


def allocate_root_typed_operands(plan: ChainedMatmulPlan) -> list[str]:
    from .chained_tcgen05 import _layout

    retained = plan.root_typed_operands
    assert retained is not None
    lines = [
        f"chain_operand_workspace = cute.arch.alloc_smem(cutlass.Uint8, {retained.operands.allocated_bytes}, alignment=128)"
    ]
    for image in retained.images:
        dtype = (
            "cutlass.Float16" if image.dtype == torch.float16 else "cutlass.BFloat16"
        )
        offset = retained.operands.region(image.name).byte_offset
        lines.extend(
            [
                f"{image.name}_ptr = cute.recast_ptr(chain_operand_workspace + {offset}, dtype={dtype})",
                *_layout(image.name, image.shape, 1, dtype),
            ]
        )
    return lines


def operand_view(
    image: TypedOperandImage, use: TypedOperandUse, prefix: str
) -> list[str]:
    if use.physical_transpose:
        return [
            f"{prefix} = cute.make_tensor({image.name}.iterator, cute.select({image.name}.layout, mode=[1, 0]))"
        ]
    return [f"{prefix} = {image.name}"]


def publish_root_typed_operands(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    stage: int,
    boundaries: dict[Node, str],
    scans: list[_ScanInput],
) -> list[str]:
    from .chained_fragment_epilogue import bind_fragment_expression
    from .chained_matmul import _indent

    retained = plan.root_typed_operands
    assert retained is not None
    lines = []
    prefix = f"chain_{stage}"
    for image in retained.images:
        if image.source_stage != stage:
            continue
        use = image.uses[0]
        name = image.name
        dtype = (
            "cutlass.Float16" if image.dtype == torch.float16 else "cutlass.BFloat16"
        )
        coords = (f"{name}_row", f"{name}_column")
        expression = bind_fragment_expression(
            cg,
            plan,
            boundaries,
            scans,
            plan.dots[stage],
            use.operand,
            coords,
            coords[::-1] if use.transpose else coords,
            f"{prefix}_values[{name}_index]",
            dtype,
        )
        lines.extend(
            [
                f"{name}_values = cute.make_rmem_tensor({prefix}_values.shape, {dtype})",
                f"for {name}_index in cutlass.range_constexpr(cute.size({prefix}_values)):",
                f"    {coords[0]}, {coords[1]} = {prefix}_coords[{name}_index]",
                _indent(expression.expression.lines),
                f"    {name}_values[{name}_index] = {expression.masked_value()}",
                f"{name}_target = {prefix}_thread.partition_D({prefix}_slice.partition_C({name}))",
                f"cute.autovec_copy({name}_values, {name}_target)",
            ]
        )
    return lines
