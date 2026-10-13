"""Bounded raw-leaf rings for a proved ordinary warp-owned lexical carry.

Only transport moves. Each transformed operand still evaluates its original
pointwise expression at the exact raw fragment coordinates and precision.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import replace
import math
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_loop_tmem_bridges import _coordinate_path
from .chained_loop_tmem_bridges import _exchange_axes
from .chained_root_warp_stage import WarpOperandEmission
from .chained_root_warp_stage import WarpOperandLayout
from .chained_warp_loop_carry import _revision
from .chained_warp_loop_carry import _value_ancestors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan


def pending_groups(depth: int, leaves: int, remaining: int) -> int:
    """Groups belonging to future iterations, after the current slot is ready."""
    if depth < 1 or leaves < 1 or depth * leaves > 8 or remaining < 0:
        raise ValueError("invalid raw prefetch group capacity")
    return min(depth - 1, remaining) * leaves


@dataclass(frozen=True)
class RawOperand:
    operand: Node
    leaf: Node
    role: str
    shape: tuple[int, int]
    swapped: bool
    direct: bool
    padding: int

    @property
    def pitch(self) -> int:
        return self.shape[1] + (self.padding if self.direct else 0)

    @property
    def slot_elements(self) -> int:
        return (self.shape[0] * self.pitch + 63) // 64 * 64

    def slot_pointer(self, slot: str) -> str:
        # Both admitted dtypes occupy two bytes. The 128-byte ring base plus
        # a multiple of 64 elements is aligned independently of the slot value.
        # Preserve that proof after SDK dynamic pointer arithmetic loses it.
        return f"(chain_raw_{self.role}_ring + {slot} * {self.slot_elements}).align(16)"

    @property
    def inner(self) -> int:
        return int((self.role == "b") == self.swapped)


@dataclass(frozen=True)
class WarpLoopPrefetchPlan:
    depth: int
    operands: tuple[RawOperand, ...]
    revision: tuple[object, ...]

    @property
    def shared_bytes(self) -> int:
        return sum(item.slot_elements * 2 * self.depth for item in self.operands)

    @property
    def direct_roles(self) -> frozenset[str]:
        return frozenset(item.role for item in self.operands if item.direct)

    def check(self, plan: ChainedMatmulPlan) -> None:
        if plan.warp_loop_carry is None or self.revision != _revision(plan):
            raise chain._UnsupportedChain("raw prefetch ownership changed")
        plan.warp_loop_carry.check(plan)
        pending_groups(self.depth, len(self.operands), 0)


def plan_warp_loop_prefetch(
    plan: ChainedMatmulPlan, depth: int, *, padding: int = 8
) -> WarpLoopPrefetchPlan | None:
    loop = plan.loop
    if (
        depth != 3
        or loop is None
        or plan.warp_loop_carry is None
        or plan.warp_result_transport is not None
        or plan.warp_workspace is not None
        or plan.completed_image_bytes
        or plan.root_vector_cache is not None
        or plan.row_registers is not None
        or plan.row_parallel is not None
    ):
        return None
    plan.warp_loop_carry.check(plan)
    ancestors = {node: _value_ancestors(node) for node in loop.region.nodes}
    writes = {
        cast("Node", node.args[0]).meta["val"].untyped_storage()._cdata
        for graph in (loop.root.graph, loop.body.graph)
        for node in graph.nodes
        if node.target is memory_ops.store
    }
    operands = []
    for role, operand in zip(("a", "b"), plan.dots[0].args[:2], strict=True):
        assert isinstance(operand, Node)
        leaves = [
            node
            for node in ancestors[operand]
            if node.target is memory_ops.load
            and isinstance(node.meta.get("val"), torch.Tensor)
            and node.meta["val"].ndim == 2
        ]
        if len(leaves) != 1:
            return None
        leaf = leaves[0]
        source = leaf.args[0]
        shape = chain._shape(leaf)
        if (
            not isinstance(source, Node)
            or source.target is not _tracing_ops._host_tensor
            or source.meta["val"].untyped_storage()._cdata in writes
            or leaf.meta["val"].dtype not in (torch.float16, torch.bfloat16)
            or leaf.meta["val"].dtype != operand.meta["val"].dtype
            or len(shape) != 2
            or min(shape) <= 0
            or shape[1] % 8
            or shape[1] > plan.threads * 8
            or math.prod(shape) % (plan.threads * 8)
            or min(plan.threads, shape[1] // 8).bit_count() != 1
            or shape[0] % (plan.threads // min(plan.threads, shape[1] // 8))
            or any(item.leaf is leaf for item in operands)
        ):
            return None
        coordinate = ("raw_row", "raw_column")
        paths = [
            _coordinate_path(operand, coordinate, leaf, expected, ancestors)
            for expected in (coordinate, (coordinate[1], coordinate[0]))
        ]
        selected = next((index for index, path in enumerate(paths) if path), None)
        if selected is None:
            return None
        path = paths[selected]
        assert path is not None
        direct = all(
            node is leaf or node.target is _tracing_ops._new_var or _exchange_axes(node)
            for node in path
        )
        operands.append(
            RawOperand(
                operand,
                leaf,
                role,
                cast("tuple[int, int]", shape),
                bool(selected),
                direct,
                padding,
            )
        )
    if depth * len(operands) > 8:
        return None
    return WarpLoopPrefetchPlan(depth, tuple(operands), _revision(plan))


def _future_index(lines: Sequence[str]) -> tuple[str, ...]:
    """Rename only the owned loop-index loads; preserve all other source bytes."""
    return _copy_scope(lines)


def _copy_scope(lines: Sequence[str], phase: str | None = None) -> tuple[str, ...]:
    """Keep copy-local values out of other phases' SDK control-flow carries."""
    source = "\n".join(lines)
    nodes = tuple(ast.walk(ast.parse(source)))
    names = {node.id for node in nodes if isinstance(node, ast.Name)}
    renames = {"chain_loop_index": "chain_prefetch_index"} if phase is None else {}
    if phase is not None:
        local_names = {
            node.id
            for node in nodes
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        scoped = {name: f"chain_prefetch_{phase}_{name}" for name in local_names}
        if names.intersection(scoped.values()):
            raise chain._UnsupportedChain("raw copy phase variable collision")
        renames.update(scoped)
    encoded = source.encode("utf-8")
    offsets = [0]
    for line in encoded.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    replacements = []
    for node in nodes:
        if isinstance(node, ast.Name):
            if node.id == "chain_loop_index" and not isinstance(node.ctx, ast.Load):
                raise chain._UnsupportedChain("raw copy writes lexical index")
            if node.id not in renames:
                continue
            assert node.end_lineno is not None and node.end_col_offset is not None
            replacements.append(
                (
                    offsets[node.lineno - 1] + node.col_offset,
                    offsets[node.end_lineno - 1] + node.end_col_offset,
                    renames[node.id].encode("utf-8"),
                )
            )
    for start, end, replacement in sorted(replacements, reverse=True):
        encoded = encoded[:start] + replacement + encoded[end:]
    return tuple(encoded.decode("utf-8").splitlines())


class WarpLoopPrefetchEmission:
    def __init__(self, cg: GenerateAST, plan: ChainedMatmulPlan) -> None:
        assert plan.warp_loop_prefetch is not None
        self.cg = cg
        self.plan = plan
        self.selected = plan.warp_loop_prefetch
        self.selected.check(plan)
        assert plan.loop is not None
        self.step = cg.device_function.resolved_block_size(plan.loop.block_id)
        if self.step != 1:
            raise chain._UnsupportedChain("raw prefetch requires scalar lexical steps")
        self.copies: tuple[tuple[str, ...], ...] | None = None
        self.emitted = False

    def _raw_copy(self, item: RawOperand) -> tuple[str, ...]:
        prefix, role = f"chain_raw_{item.role}", "a"
        rows, columns = item.shape
        index = f"{prefix}_load"
        coordinates = (f"{index} // {columns}", f"{index} % {columns}")
        expression = chain._Expression(self.cg, self.plan, {})
        expression.coordinate_names.add(index)
        value = expression.value(item.leaf, coordinates)
        domain = chain._operand_domain(self.cg, item.leaf, coordinates, self.plan)
        dtype = CompileEnvironment.current().backend.dtype_str(
            item.leaf.meta["val"].dtype
        )
        fallback = [
            f"for {prefix}_step in cutlass.range_constexpr({rows * columns // (self.plan.threads * 8)}):",
            f"    for {prefix}_lane in cutlass.range_constexpr(8):",
            f"        {index} = chain_thread * 8 + {prefix}_step * {self.plan.threads * 8} + {prefix}_lane",
            chain._indent(expression.lines, 8),
            f"        {prefix}_a[{', '.join(coordinates)}] = {chain._masked_operand(value, dtype, domain)}",
        ]
        copy = chain._async_copy(
            self.cg,
            self.plan,
            item.leaf,
            prefix,
            role,
            item.shape,
            (item.pitch, 1),
            1,
            dtype,
            fallback,
        )
        if copy is None:
            raise chain._UnsupportedChain("raw prefetch leaf lacks affine copy proof")
        return _future_index(
            [
                f"{prefix}_a_ptr = {item.slot_pointer('chain_prefetch_slot')}",
                f"{prefix}_a = cute.make_tensor({prefix}_a_ptr, cute.make_layout({item.shape!r}, stride=({item.pitch}, 1)))",
                *copy,
                "cute.arch.cp_async_commit_group()",
            ]
        )

    def setup(self) -> list[str]:
        self.selected.check(self.plan)
        if self.copies is not None:
            raise chain._UnsupportedChain("raw prefetch setup repeated")
        dtype = CompileEnvironment.current().backend.dtype_str(self.plan.dtype)
        self.copies = tuple(self._raw_copy(item) for item in self.selected.operands)
        copies = _copy_scope(
            [
                "chain_prefetch_index = chain_loop_begin + chain_prefetch_warm",
                "chain_prefetch_slot = chain_prefetch_warm",
                *(line for program in self.copies for line in program),
            ],
            "warm",
        )
        return [
            *(
                f"chain_raw_{item.role}_ring = cute.arch.alloc_smem({dtype}, {item.slot_elements * self.selected.depth}, alignment=128)"
                for item in self.selected.operands
            ),
            "chain_prefetch_trip_count = chain_loop_end - chain_loop_begin if chain_loop_end > chain_loop_begin else 0",
            f"for chain_prefetch_warm in cutlass.range_constexpr({self.selected.depth}):",
            "    if chain_prefetch_warm < chain_prefetch_trip_count:",
            chain._indent(copies, 8),
        ]

    def operands(self) -> WarpOperandEmission:
        self.selected.check(self.plan)
        if self.copies is None or self.emitted:
            raise chain._UnsupportedChain("raw prefetch operand lifetime changed")
        self.emitted = True
        depth = self.selected.depth
        leaves = len(self.selected.operands)
        lines = ["chain_prefetch_ordinal = chain_loop_index - chain_loop_begin"]
        for remaining in reversed(range(1, depth)):
            lines.extend(
                [
                    f"{'if' if remaining == depth - 1 else 'elif'} chain_prefetch_ordinal + {remaining} < chain_prefetch_trip_count:",
                    f"    cute.arch.cp_async_wait_group({pending_groups(depth, leaves, remaining)})",
                ]
            )
        lines.extend(
            [
                "else:",
                "    cute.arch.cp_async_wait_group(0)",
                "cute.arch.sync_threads()",
                f"chain_prefetch_slot = chain_prefetch_ordinal % {depth}",
            ]
        )
        layouts, roles = [], []
        rows, columns, reduction = self.plan.shapes[0]
        for item in self.selected.operands:
            prefix = f"chain_raw_{item.role}"
            shape = (rows if item.role == "a" else columns, reduction)
            inner = item.inner
            stride = (
                (shape[1] + item.padding, 1) if inner else (1, shape[0] + item.padding)
            )
            layout = WarpOperandLayout(
                item.operand,
                item.role,
                shape,
                stride,
                inner,
                self.plan.operand_dtype(0),
            )
            layouts.append(layout)
            dtype = CompileEnvironment.current().backend.dtype_str(layout.dtype)
            role_lines = [
                f"{prefix}_current_ptr = {item.slot_pointer('chain_prefetch_slot')}",
                f"{prefix}_current = cute.make_tensor({prefix}_current_ptr, cute.make_layout({item.shape!r}, stride=({item.pitch}, 1)))",
                f"chain_0_{item.role}_ptr = "
                + (
                    f"{prefix}_current_ptr"
                    if item.direct
                    else f"chain_{item.role}_workspace"
                ),
                f"chain_0_{item.role} = cute.make_tensor(chain_0_{item.role}_ptr, cute.make_layout({shape!r}, stride={stride!r}))",
            ]
            if not item.direct:
                role_lines.extend(self._weighted(item, layout, dtype))
            lines.extend(role_lines)
            roles.append(tuple(role_lines))
        result = WarpOperandEmission(
            tuple(lines),
            tuple((x.role, x.inner) for x in layouts),
            False,
            self.copies,
            tuple(layouts),
            None,
            role_lines=tuple(roles),
        )
        return replace(result, _selection=result.facts())

    def _weighted(
        self, item: RawOperand, layout: WarpOperandLayout, dtype: str
    ) -> list[str]:
        prefix = f"chain_raw_{item.role}"
        rows, columns = item.shape
        thread_columns = min(self.plan.threads, columns // 8)
        coordinates = (
            f"{prefix}_packet // {columns // 8}",
            f"{prefix}_packet % {columns // 8} * 8 + {prefix}_lane",
        )
        operand_coords = coordinates[::-1] if item.swapped else coordinates
        expression = chain._Expression(self.cg, self.plan, {})
        expression.coordinate_names.update((f"{prefix}_packet", f"{prefix}_lane"))
        expression.fragments[item.leaf] = (
            coordinates,
            f"{prefix}_raw_registers[{prefix}_lane]",
        )
        value = expression.value(item.operand, operand_coords)
        domain = chain._operand_domain(self.cg, item.operand, operand_coords, self.plan)
        return [
            f"{prefix}_weighted_target = cute.make_tensor(chain_0_{item.role}_ptr, cute.make_layout({item.shape!r}, stride=({columns + item.padding}, 1)))",
            f"{prefix}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {dtype}, num_bits_per_copy=128), cute.make_layout(({self.plan.threads // thread_columns}, {thread_columns}), stride=({thread_columns}, 1)), cute.make_layout((1, 8)))",
            f"{prefix}_thread = {prefix}_copy.get_slice(chain_thread)",
            f"{prefix}_source = {prefix}_thread.partition_S({prefix}_current)",
            f"{prefix}_destination = {prefix}_thread.partition_D({prefix}_weighted_target)",
            f"{prefix}_raw_registers = cute.make_rmem_tensor_like({prefix}_source[None, 0, 0])",
            f"{prefix}_weighted_registers = cute.make_rmem_tensor_like({prefix}_destination[None, 0, 0])",
            f"for {prefix}_step in cutlass.range_constexpr({rows * columns // (self.plan.threads * 8)}):",
            f"    {prefix}_packet = chain_thread + {prefix}_step * {self.plan.threads}",
            f"    cute.copy({prefix}_copy, {prefix}_source[None, {prefix}_step, 0], {prefix}_raw_registers)",
            f"    for {prefix}_lane in cutlass.range_constexpr(8):",
            chain._indent(expression.lines, 8),
            f"        {prefix}_weighted_registers[{prefix}_lane] = {chain._masked_operand(value, dtype, domain)}",
            f"    cute.copy({prefix}_copy, {prefix}_weighted_registers, {prefix}_destination[None, {prefix}_step, 0])",
        ]

    def refill(self, joined: Sequence[str]) -> list[str]:
        self.selected.check(self.plan)
        if (
            not self.emitted
            or self.copies is None
            or not joined
            or joined[-1] != "cute.arch.sync_threads()"
        ):
            raise chain._UnsupportedChain(
                "raw slot refill lacks full reader retirement"
            )
        return [
            f"if chain_prefetch_ordinal + {self.selected.depth} < chain_prefetch_trip_count:",
            chain._indent(
                _copy_scope(
                    [
                        f"chain_prefetch_index = chain_loop_index + {self.selected.depth}",
                        *(line for program in self.copies for line in program),
                    ],
                    "refill",
                )
            ),
        ]

    def drain(self) -> list[str]:
        if not self.emitted:
            raise chain._UnsupportedChain("raw prefetch drained before body")
        return ["cute.arch.cp_async_wait_group(0)"]
