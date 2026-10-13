"""Append coupled row packets for dependent persistent reductions.

This is seed discovery, not a replacement for vector, alias, mask or launch
admission. Existing configurations and their lowering remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING

import sympy
import torch

from ...exc import InvalidConfig
from ...language._tracing_ops import _get_symnode
from ...language.memory_ops import load
from ...language.memory_ops import store
from ...language.tile_ops import tile_id
from .cute import _seq_config_list
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from .registry import CompilerHeuristicSpecializationFact


@dataclass(frozen=True)
class RowPacketPlan:
    row: int
    reduction: int
    extent: int
    loop: int | None
    traversal: tuple[int, ...]
    feature_width: int
    row_width: int


@dataclass(frozen=True)
class _Axes:
    direct: frozenset[int] = frozenset()
    indirect: frozenset[int] = frozenset()


def _merge(values: list[_Axes | None]) -> _Axes | None:
    if any(value is None for value in values):
        return None
    direct: set[int] = set()
    indirect: set[int] = set()
    for value in values:
        assert value is not None
        direct.update(value.direct)
        indirect.update(value.indirect)
    return _Axes(frozenset(direct), frozenset(indirect))


def _symbol(value: object) -> object:
    return value._sympy_() if isinstance(value, torch.SymInt) else value


def _structural_axes(
    env: CompileEnvironment, device_ir: DeviceIR
) -> tuple[int, int, int | None, tuple[int, ...]] | None:
    spec = env.config_spec
    grid = spec.kernel_grid_fact
    if (
        spec.matmul_facts
        or spec.reduction_loops
        or len(spec.block_sizes) != 1
        or len(spec.block_sizes[0].block_ids) != 1
        or grid is None
        or len(grid.roots) != 1
        or len(device_ir.graphs) != 1
        or "flat" not in spec.allowed_pid_types
        or env.settings.autotune_force_persistent
    ):
        return None
    root = grid.roots[0]
    if device_ir.graphs[0].graph_id != root.root_graph_id:
        return None
    row = spec.block_sizes[0].block_id
    reduction_blocks = [b for b in env.block_sizes if b.reduction]
    if len(reduction_blocks) != 1 or row not in root.block_ids:
        return None
    reduction = reduction_blocks[0]
    if (
        not isinstance(reduction.numel, sympy.Integer)
        or reduction.numel <= 0
        or any(env.is_jagged_tile(b) for b in (*root.block_ids, reduction.block_id))
    ):
        return None
    loops = [
        item for item in spec.loop_orders if tuple(item.block_ids) == root.block_ids
    ]
    if len(loops) != 1 and not (not loops and len(root.block_ids) == 1):
        return None
    required = (
        (spec.num_threads, row),
        (spec.num_threads, reduction.block_id),
        (spec.cute_vector_widths, row),
        (spec.cute_vector_widths, reduction.block_id),
        (spec.cute_lane_layouts, row),
        (spec.cute_lane_layouts, reduction.block_id),
        (spec.cute_reduction_reloads, reduction.block_id),
    )
    if any(block not in sequence.valid_block_ids() for sequence, block in required):
        return None
    return row, reduction.block_id, loops[0].block_id if loops else None, root.block_ids


def row_packet_plan(
    env: CompileEnvironment, device_ir: DeviceIR
) -> RowPacketPlan | None:
    structural = _structural_axes(env, device_ir)
    if structural is None:
        return None
    row, reduction, loop, roots = structural
    extent = int(env.block_sizes[reduction].numel)
    row_symbol = _symbol(env.block_sizes[row].var)
    root_symbols = {_symbol(env.block_sizes[b].var): b for b in roots}
    memo: dict[torch.fx.Node, _Axes | None] = {}

    def axes(value: object) -> _Axes | None:
        if type(value) is int:
            return _Axes()
        if not isinstance(value, torch.fx.Node):
            return None
        if value in memo:
            return memo[value]
        memo[value] = None
        result: _Axes | None = None
        val = value.meta.get("val")
        if value.target is _get_symnode and isinstance(val, torch.SymInt):
            block = root_symbols.get(_symbol(val))
            if block is not None:
                result = _Axes(frozenset((block,)))
        elif value.target is tile_id and len(value.args) == 1:
            result = axes(value.args[0])
        elif (
            value.target is torch.ops.prims.iota.default
            and value.args == (extent,)
            and value.kwargs.get("start") == 0
            and value.kwargs.get("step") == 1
            and value.kwargs.get("dtype") in (torch.int32, torch.int64)
            and isinstance(val, torch.Tensor)
            and tuple(val.shape) == (extent,)
        ):
            result = _Axes(frozenset((reduction,)))
        elif (
            value.target is torch.ops.prims.convert_element_type.default
            and len(value.args) == 2
            and value.args[1] in (torch.int32, torch.int64)
            and isinstance(val, torch.Tensor)
            and val.ndim == 0
        ):
            result = axes(value.args[0])
        elif value.target in (
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
        ):
            if len(value.args) == 2 and any(type(arg) is int for arg in value.args):
                result = _merge([axes(arg) for arg in value.args])
        elif (
            value.target is load
            and len(value.args) >= 2
            and isinstance(value.args[1], (tuple, list))
            and isinstance(val, torch.Tensor)
            and val.ndim == 0
            and val.dtype in (torch.int32, torch.int64)
        ):
            parents = _merge([axes(arg) for arg in value.args[1]])
            if parents is not None:
                result = _Axes(indirect=parents.direct | parents.indirect)
        memo[value] = result
        return result

    graph = device_ir.graphs[0].graph
    # Exact typed result dimensions distinguish a row reduction from a scalar
    # normalization over the same persistent feature axis.
    row_sums: set[torch.fx.Node] = set()
    lineage: dict[torch.fx.Node, frozenset[torch.fx.Node]] = {}
    dependent = False
    for node in graph.nodes:
        ancestors = frozenset().union(*(lineage[n] for n in node.all_input_nodes))
        is_row_sum = False
        if node.target is torch.ops.aten.sum.dim_IntList and len(node.args) >= 2:
            source = node.args[0]
            value = (
                source.meta.get("val") if isinstance(source, torch.fx.Node) else None
            )
            result = node.meta.get("val")
            is_row_sum = (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.float32
                and value.ndim == 2
                and _symbol(value.shape[0]) == row_symbol
                and type(value.shape[1]) is int
                and value.shape[1] == extent
                and node.args[1] in ([-1], [1], (-1,), (1,))
                and isinstance(result, torch.Tensor)
                and result.dtype == torch.float32
                and result.ndim == 1
                and _symbol(result.shape[0]) == row_symbol
            )
        if is_row_sum:
            dependent |= bool(ancestors & row_sums)
            row_sums.add(node)
        lineage[node] = ancestors | ({node} if is_row_sum else set())
    if not dependent:
        return None

    context = frozenset(roots) - {row}
    anchor_contexts: set[_Axes] = set()
    broadcast_axes: set[int] = set()
    feature_widths: list[int] = []
    row_widths: list[int] = []
    for node in graph.nodes:
        if node.target not in (load, store):
            continue
        if len(node.args) < 2 or not isinstance(node.args[0], torch.fx.Node):
            return None
        tensor = node.args[0].meta.get("val")
        indices = node.args[1]
        if not isinstance(tensor, torch.Tensor) or not isinstance(
            indices, (list, tuple)
        ):
            return None
        if len(indices) != tensor.ndim:
            return None
        index_axes = [axes(index) for index in indices]
        all_axes = _merge(index_axes)
        if all_axes is None:
            return None
        if (all_axes.direct | all_axes.indirect) - {*roots, reduction}:
            return None
        if row in all_axes.indirect or reduction in all_axes.indirect:
            return None
        for index, item in zip(indices, index_axes, strict=True):
            assert item is not None
            if row in item.direct:
                if (
                    not isinstance(index, torch.fx.Node)
                    or index.target is not _get_symnode
                    or _symbol(index.meta.get("val")) != row_symbol
                    or item != _Axes(frozenset((row,)))
                ):
                    return None
            if reduction in item.direct:
                if (
                    not isinstance(index, torch.fx.Node)
                    or index.target is not torch.ops.prims.iota.default
                    or item != _Axes(frozenset((reduction,)))
                ):
                    return None
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            continue
        item_width = 16 // tensor.element_size()
        for position, item in enumerate(index_axes):
            assert item is not None
            stride = tensor.stride(position)
            if item.direct == {reduction} and not item.indirect:
                if type(stride) is not int or stride != 1:
                    return None
                width = item_width
                # Stride divisibility only: traced tensors can normalize view
                # offsets. Actual base/view alignment and vector/mask legality
                # remain the original backend's authority.
                offsets = [
                    tensor.stride(i) for i in range(tensor.ndim) if i != position
                ]
                while width > 1 and any(
                    type(x) is not int or x % width for x in offsets
                ):
                    width //= 2
                feature_widths.append(width)
            if (
                item.direct == {row}
                and not item.indirect
                and type(stride) is int
                and stride == 1
            ):
                width = item_width
                offsets = [
                    tensor.stride(i) for i in range(tensor.ndim) if i != position
                ]
                while width > 1 and any(
                    type(x) is not int or x % width for x in offsets
                ):
                    width //= 2
                row_widths.append(width)
        if row in all_axes.direct and reduction in all_axes.direct:
            # No combined/indirect row or feature subscript is a carrier.
            if sum(item == _Axes(frozenset((row,))) for item in index_axes) != 1:
                return None
            if sum(item == _Axes(frozenset((reduction,))) for item in index_axes) != 1:
                return None
            direct = all_axes.direct - {row, reduction}
            indirect = all_axes.indirect
            if direct & indirect or direct | indirect != context:
                return None
            anchor_contexts.add(_Axes(direct, indirect))
        elif (
            node.target is load and row not in all_axes.direct and not all_axes.indirect
        ):
            broadcast_axes.update((all_axes.direct - {reduction}) & context)
    if len(anchor_contexts) != 1 or not feature_widths or not row_widths:
        return None
    (anchor,) = anchor_contexts
    if not anchor.direct <= broadcast_axes:
        return None
    traversal = (
        *(b for b in roots if b in anchor.direct),
        row,
        *(b for b in roots if b in anchor.indirect),
    )
    if len(traversal) != len(roots) or set(traversal) != set(roots):
        return None
    return RowPacketPlan(
        row,
        reduction,
        extent,
        loop,
        tuple(roots.index(b) for b in traversal),
        min(feature_widths),
        min(row_widths),
    )


class CuteRowPacketHeuristic(AutotunerHeuristic):
    name = "cute_dependent_row_packets"
    backend = "cute"
    CACHE_SPECIALIZATION_FACTS = frozenset({"input_tensor_metadata"})

    @classmethod
    def register_facts(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> frozenset[CompilerHeuristicSpecializationFact]:
        # Strides/alignment may turn a declined layout into an admitted one.
        # Register before those checks, including when optional seeds are off.
        return (
            cls.CACHE_SPECIALIZATION_FACTS
            if _structural_axes(env, device_ir)
            else frozenset()
        )

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        # This family is appended only after the entire preexisting prefix.
        return False


def append_row_packet_seeds(
    env: CompileEnvironment, device_ir: DeviceIR, seeds: list[Config]
) -> list[Config]:
    plan = row_packet_plan(env, device_ir)
    if (
        plan is None
        or "input_tensor_metadata" not in env.compiler_fact_specialization_facts
    ):
        return seeds
    spec = env.config_spec
    default = spec.default_config().config
    # SIMT omits an explicit pid_type search coordinate and defaults to flat.
    # Do not inject inactive fields or override a different launch policy.
    if default.get("pid_type", "flat") != "flat":
        return seeds
    appended = list(seeds)
    warp_size = 32  # NVIDIA warp width; this helper is registered only for CuTe.
    for reduction_threads in (warp_size // 2, warp_size):
        row_threads = 2 * warp_size // reduction_threads
        per_thread, remainder = divmod(plan.extent, reduction_threads)
        if remainder or per_thread < 1:
            continue
        width = min(plan.feature_width, per_thread)
        while width > 1 and (width & (width - 1) or per_thread % width):
            width -= 1
        for packet_rows in (2, 4):
            if packet_rows > plan.row_width:
                continue
            proposed = default | {
                "block_sizes": _seq_config_list(
                    spec.block_sizes, {plan.row: row_threads * packet_rows}
                ),
                "num_threads": _seq_config_list(
                    spec.num_threads,
                    {plan.row: row_threads, plan.reduction: reduction_threads},
                ),
                "cute_vector_widths": _seq_config_list(
                    spec.cute_vector_widths,
                    {plan.row: packet_rows, plan.reduction: width},
                ),
                "cute_lane_layouts": _seq_config_list(
                    spec.cute_lane_layouts,
                    {plan.row: "blocked", plan.reduction: "blocked"},
                ),
                "cute_reduction_reloads": _seq_config_list(
                    spec.cute_reduction_reloads, {plan.reduction: "register"}
                ),
            }
            if plan.loop is not None:
                proposed["loop_orders"] = _seq_config_list(
                    spec.loop_orders, {plan.loop: list(plan.traversal)}
                )
            if any(
                proposed.get(name) != value
                for name, value in env.settings.autotune_config_overrides.items()
            ):
                continue
            try:
                seed = spec.normalized_config(proposed)
            except InvalidConfig:
                continue
            if seed not in appended:
                appended.append(seed)
    if len(appended) != len(seeds):
        if CuteRowPacketHeuristic.name not in spec.autotuner_heuristics:
            spec.autotuner_heuristics.append(CuteRowPacketHeuristic.name)
        return appended
    return seeds
