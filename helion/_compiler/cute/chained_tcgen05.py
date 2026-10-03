"""Resident one-CTA TCgen05 schedule for a general contraction DAG.

Pointwise semantics and indexing come from the common chain interpreter. Only
an exclusive, coordinate-preserving edge into operand A uses packed TMEM;
other live results retain an FP32 shared boundary. The default performs no graph
reassociation; initialized-accumulator reuse is an explicit FP32 reassociation
option. M128 uses the full datapath family; an independent M64 result uses
the explicitly proven sparse 16x256 load and optional direct output transport.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_aux_cache import make_early_auxiliary_cache
from .chained_aux_cache import make_late_auxiliary_cache
from .chained_aux_cache import uses_early_cache
from .chained_cache_layout import PointwiseCacheLayouts
from .chained_collectives import allocate_collectives
from .chained_collectives import collective_bindings
from .chained_collectives import emit_collectives_before
from .chained_collectives import needs_materialized_final
from .chained_collectives import uses_general_collectives
from .chained_collectives import workspace_bytes as collective_workspace_bytes
from .chained_k_issue import emit_k_half_issues
from .chained_leaf_pipeline import LeafPipeline
from .chained_leaf_pipeline import finish_wrapper
from .chained_leaf_pipeline import plan_leaf_pipeline
from .chained_leaf_pipeline import produce_half
from .chained_pointwise_cache import PointwiseReadCache
from .chained_pointwise_inplace import PointwiseInplace
from .chained_pointwise_inplace import raw_preload
from .chained_pointwise_inplace import sw128_ownership
from .chained_pointwise_residency import allocate_pointwise_cache
from .chained_pointwise_residency import emit_pointwise_cache_before
from .chained_pointwise_unroll import PointwiseUnroll
from .chained_result_transport import direct_store
from .chained_result_transport import is_m64_plan
from .chained_result_transport import load_operation
from .chained_scan_export import codegen_scan_exports
from .chained_scratch_layout import ScratchLayouts
from .chained_seed_tiles import SeedTiling
from .chained_tmem_transport import emit_packed_tmem_fragment
from .chained_tmem_transport import emit_tmem_operand_view
from .chained_vector_stage import VectorStageOperand
from .chained_vector_stage import VectorStaging
from .chained_vector_stage import emit_vector_stage_group
from .fx_matcher import _GeneratedCodeTemplate
from .tcgen05_config import CuteTcgen05Config

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_root_snapshot import RootSnapshot
    from .chained_startup import StartupInput
    from .chained_tmem_drains import DrainTiling
    from .prepared_state_body import StateEffect
    from .prepared_tcgen_binding import PreparedTmemEdge


def _prefetch_final_b(plan: ChainedMatmulPlan) -> bool:
    operand = cast("Node", plan.dots[-1].args[1])
    return (
        len(plan.dots) > 1
        and chain._direct_operand(operand)
        and not (chain._ancestors(operand) & {*plan.dots, *plan.scans})
        and cast("Node", plan.store.args[0]).meta["val"].dtype == plan.operand_dtype(-1)
        and max(n * k for _, n, k in plan.shapes) >= math.prod(plan.shapes[-1][:2])
    )


def supported_plan(plan: ChainedMatmulPlan, *, startup: bool = False) -> bool:
    """Conservative physical bounds, separate from the common semantic proof."""
    if plan.threads != 128 or any(size % block for _, size, block in plan.axes):
        return False
    m64 = is_m64_plan(plan)
    if m64 and any(node.args[2] is not None for node in plan.dots):
        # The accumulator store below uses the full M128 datapath. M64 needs
        # its own proven sparse store map before accepting an explicit seed.
        return False
    if startup and (
        plan.initialized_accumulator is not None
        or plan.late_rhs_reuse is not None
        or plan.k_schedule is not None
    ):
        return False
    if plan.direct_output and not m64:
        return False
    if not m64 and any(
        m != 128 or n % 32 or not 32 <= n <= 256 or k % 16 or k <= 0
        for m, n, k in plan.shapes
    ):
        return False
    final_shape = chain._shape(cast("Node", plan.store.args[2]))
    if final_shape != plan.shapes[-1][:2]:
        return False
    device = cast("Node", plan.dots[0].args[0]).meta["val"].device
    return _shared_memory_bytes(
        plan, startup=startup
    ) <= CuteTcgen05Config.per_cta_smem_capacity_bytes(device)


def _shared_memory_bytes(plan: ChainedMatmulPlan, *, startup: bool = False) -> int:
    """Conservative aligned baseline footprint, excluding optional early caches."""
    final_shape = chain._shape(cast("Node", plan.store.args[2]))
    general_collectives = uses_general_collectives(plan)
    # Each swizzled operand has a power-of-two contiguous atom, tiled exactly
    # by the admitted dimensions. Its cosize is the logical element count.
    allocations = [
        plan.late_rhs_reuse.a_bytes
        if plan.late_rhs_reuse is not None
        else 2 * max(m * k for m, _, k in plan.shapes),
        plan.late_rhs_reuse.b_bytes
        if plan.late_rhs_reuse is not None
        else 2 * max(n * k for _, n, k in plan.shapes),
        (
            0
            if plan.late_rhs_reuse is not None or plan.direct_output
            else 2 * plan.shapes[-1][1] * plan.shapes[-1][2]
            if _prefetch_final_b(plan)
            else math.prod(final_shape)
            * cast("Node", plan.store.args[0]).meta["val"].element_size()
        ),
        *(
            4 * m * n
            for index, (m, n, _) in enumerate(plan.shapes[:-1])
            if plan.initialized_accumulator is None or index != 0
        ),
        *(
            4 * (chain._shape(scan)[0] + 4)
            for scan in plan.scans
            if not general_collectives
        ),
        *(
            chain._shape(scan)[0] * leaf.meta["val"].element_size()
            for scan in plan.scans
            if not general_collectives
            for leaf in chain._scan_cache_candidates(scan, plan.scans, plan.store)
        ),
        8 * len(plan.dots),
        4,
    ]
    return (
        sum((size + 127) // 128 * 128 for size in allocations)
        + 128 * startup
        + (collective_workspace_bytes(plan) if general_collectives else 0)
        + (plan.pointwise_cache.shared_bytes if plan.pointwise_cache is not None else 0)
        + (
            4 * math.prod(plan.shapes[-1][:2])
            if general_collectives and needs_materialized_final(plan)
            else 0
        )
    )


def _layout(prefix: str, shape: tuple[int, int], inner: int, dtype: str) -> list[str]:
    width = shape[inner]
    bits = 32 if dtype == "cutlass.Float32" else 16
    width_bytes = width * bits // 8
    atom_bytes = min(128, width_bytes & -width_bytes)
    mode = "K" if inner == 1 else "MN"
    order = (0, 1) if inner == 1 else (1, 0)
    return [
        f"{prefix}_layout = cute.tile_to_shape(tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.{mode}_SW{atom_bytes}, {dtype}), {shape!r}, order={order!r})",
        f"{prefix} = cute.make_tensor(cute.recast_ptr({prefix}_ptr, {prefix}_layout.inner, dtype={dtype}), {prefix}_layout.outer)",
    ]


def _scalar_stage(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    stage: int,
    role: str,
    inner: int,
    dtype: str,
    k_half: str | None = None,
) -> tuple[tuple[int, int], list[str], tuple[int, int], str]:
    """Original scalar operand, including its masks and address arithmetic."""
    prefix = f"chain_{stage}"
    m, n, k = plan.shapes[stage]
    shape = (m, k) if role == "a" else (n, k)
    operand = cast("Node", plan.dots[stage].args[0 if role == "a" else 1])
    index = f"{prefix}_{role}_load"
    x, y = (
        (f"{index} // {shape[1]}", f"{index} % {shape[1]}")
        if inner == 1
        else (f"{index} % {shape[0]}", f"{index} // {shape[0]}")
    )
    coords = (x, y) if role == "a" else (y, x)
    expression = chain._Expression(cg, plan, boundaries)
    expression.scan_inputs = scans
    expression.coordinate_names.add(index)
    value = expression.value(operand, coords)
    domain = chain._operand_domain(cg, operand, coords, plan)
    fallback = [
        f"for {prefix}_{role}_step in cutlass.range({math.prod(shape) // (256 if k_half is not None else 128)}, unroll=1):",
        (
            f"    {index} = (chain_thread + {prefix}_{role}_step * 128) // 64 * 128 + {k_half} * 64 + (chain_thread + {prefix}_{role}_step * 128) % 64"
            if k_half is not None
            else f"    {index} = chain_thread + {prefix}_{role}_step * 128"
        ),
        chain._indent(expression.lines),
        f"    {prefix}_{role}[{x}, {y}] = {chain._masked_operand(value, dtype, domain)}",
    ]
    stride = (shape[1], 1) if inner == 1 else (1, shape[0])
    target = (
        f"{prefix}_{role}"
        if inner == 1
        else f"cute.make_tensor({prefix}_{role}.iterator, cute.select({prefix}_{role}.layout, mode=[1, 0]))"
    )
    return shape, fallback, stride, target


def direct_stage(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    stage: int,
    role: str,
    dtype: str,
) -> list[str] | None:
    """Prove the original K-major cp.async path with its exact scalar fallback.

    Failure does not authorize substituting a different transfer schedule.
    Only direct host leaves are eligible; no materialized boundary is assumed.
    """
    operand = cast("Node", plan.dots[stage].args[0 if role == "a" else 1])
    if not chain._direct_operand(operand):
        return None
    shape, fallback, stride, target = _scalar_stage(
        cg, plan, {}, [], stage, role, 1, dtype
    )
    return chain._async_copy(
        cg,
        plan,
        operand,
        f"chain_{stage}",
        role,
        shape,
        stride,
        1,
        dtype,
        fallback,
        target,
    )


def _stage(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    stage: int,
    role: str,
    inner: int,
    dtype: str,
    pointwise_unroll: PointwiseUnroll,
    pointwise_cache: PointwiseReadCache,
    pointwise_inplace: PointwiseInplace,
    prestaged_leaf: Node | None = None,
    k_half: str | None = None,
    leaf_pipeline: LeafPipeline | None = None,
) -> list[str]:
    prefix = f"chain_{stage}"
    operand = cast("Node", plan.dots[stage].args[0 if role == "a" else 1])
    shape, fallback, stride, target = _scalar_stage(
        cg, plan, boundaries, scans, stage, role, inner, dtype, k_half
    )
    if k_half is not None:
        if role != "a" or inner != 1 or shape != (128, 128):
            raise chain._UnsupportedChain("K64 producer requires K-major M128 K128 A")
        if leaf_pipeline is not None:
            return produce_half(
                cg,
                plan,
                boundaries,
                scans,
                leaf_pipeline,
                operand,
                dtype,
                target,
                fallback,
                pointwise_cache,
                pointwise_unroll,
            )
        result = _pointwise_stage(
            cg,
            plan,
            boundaries,
            scans,
            operand,
            prefix,
            role,
            shape,
            inner,
            dtype,
            target,
            fallback,
            pointwise_unroll,
            pointwise_cache,
            pointwise_inplace,
            k_half=k_half,
        )
        if result is None:
            raise chain._UnsupportedChain("K64 producer requires proved vector leaves")
        return result
    result = chain._async_copy(
        cg,
        plan,
        operand,
        prefix,
        role,
        shape,
        stride,
        inner,
        dtype,
        fallback,
        target,
    ) or _pointwise_stage(
        cg,
        plan,
        boundaries,
        scans,
        operand,
        prefix,
        role,
        shape,
        inner,
        dtype,
        target,
        fallback,
        pointwise_unroll,
        pointwise_cache,
        pointwise_inplace,
        prestaged_leaf=prestaged_leaf,
    )
    if result is not None:
        return result
    if (
        inner == 1
        and prestaged_leaf is None
        and cg.device_function.config.config.get("cute_chained_pointwise_vectorize")
    ):
        from .chained_tcgen_stage import StageGeometry
        from .chained_vector_stage import emit_vector_stage

        # Masked or capture-dependent leaves need only a per-thread inner
        # vector proof, not an affine map for the entire operand tile. The same
        # producer component handles root DAGs and serial contraction loops.
        result = emit_vector_stage(
            cg,
            plan,
            boundaries,
            operand,
            StageGeometry(plan.shapes[stage], transpose=False),
            role=role,
            shape=shape,
            offset=0,
            tag=f"{prefix}_{role}_vector",
            target=target,
        )
        if result is not None:
            return result
    return fallback


@dataclasses.dataclass(frozen=True)
class _VectorLeaf:
    node: Node
    coordinates: tuple[str, ...]
    tensor: str
    offset: str
    outer_stride: int
    bounds: tuple[str, ...]
    dtype: torch.dtype


def _vector_leaf(
    expression: chain._Expression,
    node: Node,
    coordinates: tuple[str, ...],
    indices: list[str],
    shape: tuple[int, int],
    names: tuple[str, str],
) -> _VectorLeaf | None:
    """Prove a dense, fully bounded vector load, preserving runtime strides."""
    if len(node.args) > 2 and node.args[2] is not None:
        # The scalar expression owns the source mask and its `other` value.
        # A bulk copy is not legal merely because the allocation is in bounds.
        return None
    source = cast("Node", node.args[0])
    fake = source.meta["val"]
    if fake.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return None
    symbols = {
        name: sympy.Symbol(name, integer=True)
        for name in (*names, *expression.origins.values())
    }
    row, col = (symbols[name] for name in names)
    bases, coefficients, bounds = [], [], []
    try:
        for index, extent in zip(indices, fake.shape, strict=True):
            value = sympy.expand(
                chain._copy_index(index, expression.definitions, symbols)
            )
            base = value.subs({row: 0, col: 0})
            slopes = (sympy.diff(value, row), sympy.diff(value, col))
            if any(not isinstance(slope, sympy.Integer) for slope in slopes):
                return None
            pair = (int(slopes[0]), int(slopes[1]))
            if (
                sympy.expand(
                    value - base - sympy.Mul(pair[0], row) - sympy.Mul(pair[1], col)
                )
                != 0
            ):
                return None
            bases.append(base)
            coefficients.append(pair)
            low = sum(min(0, s * (e - 1)) for s, e in zip(pair, shape, strict=True))
            high = sum(max(0, s * (e - 1)) for s, e in zip(pair, shape, strict=True))
            bounds.extend(
                (
                    f"0 <= ({chain._copy_code(base + low)})",
                    f"({chain._copy_code(base + high)}) < {extent}",
                )
            )
        strides = tuple(fake.stride())
        physical = tuple(
            sum(
                pair[axis] * stride
                for pair, stride in zip(coefficients, strides, strict=True)
            )
            for axis in (0, 1)
        )
        if physical[1] != 1 or physical[0] < 0 or physical[0] % 8:
            return None
        offset = chain._copy_code(
            sympy.Add(
                *(base * stride for base, stride in zip(bases, strides, strict=True))
            )
        )
    except chain._UnsupportedChain:
        return None
    tensor = expression.tensor_name(source)
    bounds.extend(
        f"{tensor}.layout.stride[{axis}] == {stride}"
        for axis, stride in enumerate(strides)
    )
    return _VectorLeaf(
        node, coordinates, tensor, offset, physical[0], tuple(bounds), fake.dtype
    )


def _pointwise_stage(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    operand: Node,
    prefix: str,
    role: str,
    shape: tuple[int, int],
    inner: int,
    dtype: str,
    target: str,
    fallback: list[str],
    pointwise_unroll: PointwiseUnroll,
    pointwise_cache: PointwiseReadCache,
    pointwise_inplace: PointwiseInplace,
    k_half: str | None = None,
    prestaged_leaf: Node | None = None,
) -> list[str] | None:
    """Vectorize dense leaves, then evaluate the unchanged pointwise graph.

    Each source iteration uses eight values per leaf. Broadcast subexpressions are
    evaluated once per vector; arbitrary remaining loads retain scalar masks.
    The final logical-domain mask is applied after all pointwise operations.
    """
    if not cg.device_function.config.config.get("cute_chained_pointwise_vectorize"):
        return None
    if chain._direct_operand(operand) or chain._ancestors(operand) & set(plan.dots):
        return None
    width, height = shape[inner], shape[1 - inner]
    columns = (64 if k_half is not None else width) // 8
    if (
        width % 8
        or columns < 1
        or columns > 128
        or columns & (columns - 1)
        or height % (128 // columns)
    ):
        return None
    rows = 128 // columns
    tag = f"{prefix}_{role}_pointwise"
    names = (f"{tag}_row", f"{tag}_col")
    stored = names if inner == 1 else names[::-1]
    coords = stored if role == "a" else stored[::-1]
    probe = chain._Expression(cg, plan, boundaries)
    probe.scan_inputs = scans
    probe.coordinate_names.update(names)
    probe.value(operand, coords)
    leaves = [
        leaf
        for node, coordinates, indices, loaded in probe.loaded_inputs
        if not uses_early_cache(probe, loaded)
        if (
            leaf := _vector_leaf(
                probe, node, coordinates, indices, (height, width), names
            )
        )
        is not None
    ]
    if not leaves:
        return None
    reusable = pointwise_cache.vector_reads(probe, names, height // rows)
    cached_vectors = {
        index
        for index, leaf in enumerate(leaves)
        if leaf.outer_stride == 0 and (leaf.node, leaf.coordinates) in reusable
    }
    if cached_vectors:
        pointwise_cache.activated = True
    operand_dtype = operand.meta["val"].dtype
    raw_leaf = pointwise_inplace.select(leaves, shape, inner, operand_dtype)
    if prestaged_leaf is not None:
        matches = [i for i, leaf in enumerate(leaves) if leaf.node is prestaged_leaf]
        if (
            len(matches) != 1
            or sum(node is prestaged_leaf for node, *_ in probe.loaded_inputs) != 1
            or not sw128_ownership(shape, inner)
            or leaves[matches[0]].dtype != operand_dtype
            or (raw_leaf is not None and raw_leaf != matches[0])
        ):
            raise chain._UnsupportedChain(
                "pre-staged leaf requires a unique same-dtype bijective vector map"
            )
        raw_leaf = matches[0]
    if k_half is not None and raw_leaf is not None:
        raise chain._UnsupportedChain("K64 final-A raw preload is not half-scoped")
    raw = f"{prefix}_{role}_raw"
    expression = chain._Expression(cg, plan, boundaries)
    expression.bind_scan_reads = pointwise_cache.enabled
    expression.scan_inputs = scans
    expression.coordinate_names.update(names)
    element = f"{tag}_element"
    for index, leaf in enumerate(leaves):
        expression.memo[leaf.node, leaf.coordinates] = (
            f"{tag}_leaf_{index}_values[{element}]"
        )
    value = expression.value(operand, coords)
    domain = chain._operand_domain(cg, operand, coords, plan)
    cached_reads = pointwise_cache.prepare(
        expression,
        names,
        element,
        tag,
        columns,
        height // rows,
        column_base=f"{k_half} * 64" if k_half is not None else None,
    )
    invariant, varying = _hoist(expression, {names[1], element})
    pointer_lines, bounds, descriptors, loads, vector_preloads = [], [], [], [], []
    for index, leaf in enumerate(leaves):
        name = f"{tag}_leaf_{index}"
        leaf_dtype = CompileEnvironment.current().backend.dtype_str(leaf.dtype)
        copy, thread = f"{tag}_copy", f"{tag}_thread"
        if leaf.dtype != operand_dtype:
            # Preserve the source precision through the pointwise expression.
            # The same TV layout owns eight logical values for either dtype;
            # FP32 uses two 128-bit copies instead of a BF16 narrowing load.
            copy, thread = f"{name}_copy", f"{name}_thread"
            descriptors.extend(
                (
                    f"{copy} = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {leaf_dtype}, num_bits_per_copy=128), cute.make_layout(({rows}, {columns}), stride=({columns}, 1)), cute.make_layout((1, 8)))",
                    f"{thread} = {copy}.get_slice(chain_thread)",
                )
            )
        pointer_lines.append(
            f"{name}_pointer = {leaf.tensor}.iterator + ({leaf.offset})"
        )
        bounds.extend((*leaf.bounds, f"{name}_pointer.toint() % 16 == 0"))
        descriptors.extend(
            (
                f"{name}_source = cute.make_tensor({name}_pointer.align(16), cute.make_layout(({height}, {width}), stride=({leaf.outer_stride}, 1)))",
                f"{name}_partition = {thread}.partition_S("
                + (
                    f"cute.local_tile({name}_source, (128, 64), (0, {k_half}))"
                    if k_half is not None
                    else f"{name}_source"
                )
                + ")",
                f"{name}_values = cute.make_rmem_tensor({name}_partition[None, 0, 0].shape, {leaf_dtype})",
            )
        )
        # These vectors have the same typed address and complete load mask for
        # every row. Keep all original pointer/stride/bounds guards and the
        # untouched scalar fallback; only move their existing copy outside the
        # row loop, still inside this guarded branch and current K half.
        (vector_preloads if index in cached_vectors else loads).append(
            f"cute.copy({copy}, {raw if index == raw_leaf else name}_partition[None, {'0' if index in cached_vectors else f'{tag}_step'}, 0], {name}_values)"
        )
    fast = [
        f"{tag}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {dtype}, num_bits_per_copy=128), cute.make_layout(({rows}, {columns}), stride=({columns}, 1)), cute.make_layout((1, 8)))",
        f"{tag}_thread = {tag}_copy.get_slice(chain_thread)",
        f"{tag}_target = {tag}_thread.partition_D("
        + (
            f"cute.local_tile({target}, (128, 64), (0, {k_half}))"
            if k_half is not None
            else target
        )
        + ")",
        f"{tag}_values = cute.make_rmem_tensor({tag}_target[None, 0, 0].shape, {dtype})",
        *descriptors,
        *cached_reads,
        *vector_preloads,
        *(
            [
                f"{raw}_target = {target}",
                f"{raw}_partition = {tag}_thread.partition_S({raw}_target)",
            ]
            if prestaged_leaf is not None
            else raw_preload(tag, raw, target, rows, columns, dtype, raw_leaf)
            if raw_leaf is not None
            else []
        ),
        f"for {tag}_step in cutlass.range({height // rows}, unroll={pointwise_unroll.loop_factor(height // rows)}):",
        f"    {names[0]} = chain_thread // {columns} + {tag}_step * {rows}",
        chain._indent(loads),
        chain._indent(invariant),
        f"    for {element} in cutlass.range_constexpr(8):",
        f"        {names[1]} = "
        + (f"{k_half} * 64 + " if k_half is not None else "")
        + f"chain_thread % {columns} * 8 + {element}",
        chain._indent(varying, 8),
        f"        {tag}_values[{element}] = {chain._masked_operand(value, dtype, domain)}",
        f"    cute.copy({tag}_copy, {tag}_values, {tag}_target[None, {tag}_step, 0])",
    ]
    return [
        *pointer_lines,
        f"if {' & '.join(f'({bound})' for bound in dict.fromkeys(bounds))}:",
        chain._indent(fast),
        "else:",
        chain._indent(fallback),
    ]


def _finish_startup_operand(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    transfer: StartupInput,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    stage: int,
    inner: int,
    dtype: str,
    pointwise_unroll: PointwiseUnroll,
    pointwise_cache: PointwiseReadCache,
    pointwise_inplace: PointwiseInplace,
) -> list[str]:
    """Original typed startup conversion, shared by legacy and stage actions."""
    from .chained_startup import finish_lines

    if chain._direct_operand(transfer.operand):
        return []
    # M64 reuses the existing vector producer after TMA completion. Keep the
    # original M128 scalar finish and all vector guards/fallbacks unchanged.
    finish = (
        _stage(
            cg,
            plan,
            boundaries,
            scans,
            stage,
            transfer.role,
            inner,
            dtype,
            pointwise_unroll,
            pointwise_cache,
            pointwise_inplace,
            prestaged_leaf=transfer.leaf,
        )
        if is_m64_plan(plan)
        and cg.device_function.config.config.get("cute_chained_pointwise_vectorize")
        else finish_lines(cg, plan, transfer, boundaries, scans, dtype)
    )
    return [
        "cute.arch.mbarrier_wait(chain_start_bar, 0)",
        "cute.arch.sync_threads()",
        *finish,
    ]


def _load_result_views(
    prefix: str,
    shape: tuple[int, int],
    *,
    execution: ChainedExecution | None = None,
) -> list[str]:
    if execution is not None and execution.threads < 128:
        raise ValueError("TMEM loads require at least 128 execution participants")
    thread = "chain_thread" if execution is None else execution.thread
    return [
        f"{prefix}_copy = tcgen05.make_tmem_copy(cute.make_copy_atom({load_operation(shape)}, cutlass.Float32), {prefix}_acc)",
        f"{prefix}_thread = {prefix}_copy.get_slice({thread})",
        f"{prefix}_source = {prefix}_thread.partition_S({prefix}_acc)",
        f"{prefix}_identity = {prefix}_slice.partition_C(cute.make_identity_tensor({shape!r}))",
        f"{prefix}_coords = {prefix}_thread.partition_D({prefix}_identity)",
        f"{prefix}_values = cute.make_rmem_tensor({prefix}_coords.shape, cutlass.Float32)",
    ]


def _load_result(
    prefix: str,
    shape: tuple[int, int],
    *,
    execution: ChainedExecution | None = None,
    prepared_edge: PreparedTmemEdge | None = None,
) -> list[str]:
    lines = _load_result_views(prefix, shape, execution=execution)
    if prepared_edge is not None:
        prepared_edge.capture_load_setup(prefix, lines)
    lines.extend(
        [
            f"cute.copy({prefix}_copy, {prefix}_source, {prefix}_values)",
            "cute.arch.fence_view_async_tmem_load()",
        ]
        if prepared_edge is None
        else prepared_edge.read(prefix)
    )
    return lines


def _materialize_result(
    prefix: str,
    shape: tuple[int, int],
    scratch: ScratchLayouts,
    *,
    drain_tiling: DrainTiling | None = None,
) -> list[str]:
    m, n = shape
    if drain_tiling is not None:
        from .chained_execution import ChainedExecution
        from .chained_tmem_drains import FP32DrainPanels
        from .chained_tmem_drains import PartitionedPublication

        return [
            f"{prefix}_c = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, {m * n}, alignment=128), {scratch.layout(f'{prefix}_c', shape)})",
            *drain_tiling.emit(
                FP32DrainPanels(shape),
                f"{prefix}_drain",
                prefix,
                PartitionedPublication(f"{prefix}_c"),
                execution=ChainedExecution(128),
            ),
        ]
    return [
        f"{prefix}_c = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, {m * n}, alignment=128), {scratch.layout(f'{prefix}_c', shape)})",
        f"{prefix}_shared_target = {prefix}_thread.partition_D({prefix}_slice.partition_C({prefix}_c))",
        f"cute.autovec_copy({prefix}_values, {prefix}_shared_target)",
    ]


def _explicit_accumulator(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    stage: int,
    *,
    seed_tiling: SeedTiling | None = None,
) -> list[str]:
    """Stage the source dot's FP32 accumulator without reassociating arithmetic."""
    accumulator = plan.dots[stage].args[2]
    if accumulator is None:
        return []
    prefix = f"chain_{stage}"
    seed = f"{prefix}_seed"
    shape = plan.shapes[stage][:2]
    coords = (f"{seed}_row", f"{seed}_col")
    expression = chain._Expression(cg, plan, boundaries)
    expression.scan_inputs = scans
    expression.coordinate_names.update(coords)
    value = expression.value(cast("Node", accumulator), coords)
    if seed_tiling is not None and seed_tiling.max_columns:
        lines: list[str] = []
        for panel in seed_tiling.panels(shape, 0, shape[1]):
            lines.extend(
                [
                    *panel.store_views(seed, prefix, "chain_thread"),
                    f"{seed}_values = cute.make_rmem_tensor({seed}_coords.shape, cutlass.Float32)",
                    f"for {seed}_index in cutlass.range_constexpr(cute.size({seed}_values)):",
                    f"    {coords[0]}, {coords[1]} = {seed}_coords[{seed}_index]",
                    chain._indent(expression.lines),
                    f"    {seed}_values[{seed}_index] = cutlass.Float32({value})",
                    f"cute.copy({seed}_copy, {seed}_values, {seed}_target)",
                ]
            )
        return [
            *lines,
            "cute.arch.fence_view_async_tmem_store()",
            "cute.arch.sync_threads()",
        ]
    return [
        f"{seed}_copy = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32)), cutlass.Float32), {prefix}_acc)",
        f"{seed}_thread = {seed}_copy.get_slice(chain_thread)",
        f"{seed}_target = {seed}_thread.partition_D({prefix}_acc)",
        f"{seed}_coords = {seed}_thread.partition_S({prefix}_slice.partition_C(cute.make_identity_tensor({shape!r})))",
        f"{seed}_values = cute.make_rmem_tensor({seed}_coords.shape, cutlass.Float32)",
        f"for {seed}_index in cutlass.range_constexpr(cute.size({seed}_values)):",
        f"    {coords[0]}, {coords[1]} = {seed}_coords[{seed}_index]",
        chain._indent(expression.lines),
        f"    {seed}_values[{seed}_index] = cutlass.Float32({value})",
        f"cute.copy({seed}_copy, {seed}_values, {seed}_target)",
        "cute.arch.fence_view_async_tmem_store()",
        "cute.arch.sync_threads()",
    ]


def _bridge(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    stage: int,
    dtype: str,
    *,
    snapshot: RootSnapshot | None = None,
) -> list[str]:
    from .chained_fragment_epilogue import bind_fragment_expression

    previous, prefix = f"chain_{stage - 1}", f"chain_{stage}_bridge"
    shape = plan.shapes[stage - 1][:2]
    producer, consumer = plan.dots[stage - 1 : stage + 1]
    operand = cast("Node", consumer.args[0])
    coords = (f"{prefix}_row", f"{prefix}_col")
    epilogue = bind_fragment_expression(
        cg,
        plan,
        boundaries,
        scans,
        producer,
        operand,
        coords,
        coords,
        f"{previous}_values[{prefix}_index]",
        dtype,
    )
    expression = epilogue.expression
    if snapshot is not None:
        return snapshot.render(
            expression,
            coords,
            prefix,
            previous,
            dtype,
            epilogue.masked_value(),
        )
    return emit_packed_tmem_fragment(
        prefix,
        previous,
        shape,
        dtype,
        coords,
        expression.lines,
        epilogue.masked_value(),
    )


def _shared_leaf(
    expression: chain._Expression,
    source: Node,
    indices: list[str],
    staged: list[chain._StagedInput],
    coords: tuple[str, str],
    shape: tuple[int, int],
) -> str | None:
    """Match the complete source address map, not just its innermost stride."""
    symbols = {
        name: sympy.Symbol(name, integer=True)
        for name in (*expression.origins.values(), *coords)
    }
    fixed_origins = {
        symbols[expression.origins[axis]]: 0
        for axis, extent, block in expression.plan.axes
        if extent == block
    }
    try:
        query = tuple(
            sympy.expand(
                chain._copy_index(index, expression.definitions, symbols)
            ).subs(fixed_origins)
            for index in indices
        )
        for cached in staged:
            if cached.source is not source:
                continue
            domain_coords = tuple(str(coordinate) for coordinate in cached.coordinates)
            if chain._operand_domain(
                expression.cg,
                cached.operand,
                domain_coords if cached.role == "a" else domain_coords[::-1],
                expression.plan,
            ):
                # A padded contraction tile can contain zeros where the final
                # epilogue still has valid source data. Whole-fragment reuse
                # needs full logical coverage, not merely matching addresses.
                continue
            for transpose in (False, True):
                if cached.shape != (shape[::-1] if transpose else shape):
                    continue
                ordered = coords[::-1] if transpose else coords
                substitution = dict(
                    zip(cached.coordinates, (symbols[c] for c in ordered), strict=True)
                )
                if all(
                    sympy.simplify(
                        actual - original.subs(substitution).subs(fixed_origins)
                    )
                    == 0
                    for actual, original in zip(query, cached.indices, strict=True)
                ):
                    return (
                        f"cute.make_tensor({cached.shared}.iterator, cute.select({cached.shared}.layout, mode=[1, 0]))"
                        if transpose
                        else cached.shared
                    )
    except chain._UnsupportedChain:
        pass
    return None


def _hoist(
    expression: chain._Expression, varying: set[str]
) -> tuple[list[str], list[str]]:
    """Move coordinate-invariant pure assignments out of the register loop."""
    outer, inner = [], []
    for statement in expression.statements:
        if statement.target is not None and not (
            expression.expand_dependencies(statement.inputs) & varying
        ):
            outer.append(statement.code)
        else:
            inner.append(statement.code)
    return outer, inner


def _store(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    expression: chain._Expression,
    coords: tuple[str, str],
    shape: tuple[int, int],
    dtype: str,
) -> list[str]:
    """Cooperative output copy with per-call affine, stride and alignment proof."""
    output = cast("Node", plan.store.args[0])
    name = expression.tensor_name(output)
    fake = output.meta["val"]
    indices = expression.indices(plan.store, coords)
    m, n = shape
    fallback_expression = chain._Expression(cg, plan, {})
    scalar_coords = (f"chain_store // {n}", f"chain_store % {n}")
    scalar_indices = fallback_expression.indices(plan.store, scalar_coords)
    bounds = [
        f"0 <= ({index}) < {extent}"
        for index, extent in zip(scalar_indices, fake.shape, strict=True)
    ]
    fallback = [
        f"for chain_store_step in cutlass.range({m * n // 128}, unroll=1):",
        "    chain_store = chain_thread + chain_store_step * 128",
        chain._indent(fallback_expression.lines),
        f"    if {' & '.join(f'({bound})' for bound in bounds)}:",
        f"        {name}[{', '.join(scalar_indices)}] = chain_output[{', '.join(scalar_coords)}]",
    ]
    symbols = {
        value: sympy.Symbol(value, integer=True)
        for value in (*expression.origins.values(), *coords)
    }
    row, col = (symbols[value] for value in coords)
    bases, coefficients, guards = [], [], []
    try:
        for index, extent in zip(indices, fake.shape, strict=True):
            value = sympy.expand(
                chain._copy_index(index, expression.definitions, symbols)
            )
            base = value.subs({row: 0, col: 0})
            delta = (sympy.diff(value, row), sympy.diff(value, col))
            if any(not isinstance(item, sympy.Integer) for item in delta):
                return fallback
            pair = (int(delta[0]), int(delta[1]))
            if (
                sympy.expand(
                    sympy.Add(
                        value, -base, sympy.Mul(-pair[0], row), sympy.Mul(-pair[1], col)
                    )
                )
                != 0
            ):
                return fallback
            bases.append(base)
            coefficients.append(pair)
            lo = sum(min(0, a * (s - 1)) for a, s in zip(pair, shape, strict=True))
            hi = sum(max(0, a * (s - 1)) for a, s in zip(pair, shape, strict=True))
            guards.extend(
                [
                    f"0 <= {chain._copy_code(base + lo)}",
                    f"{chain._copy_code(base + hi)} < {extent}",
                ]
            )
        strides = tuple(fake.stride())
        physical = tuple(
            sum(
                pair[axis] * stride
                for pair, stride in zip(coefficients, strides, strict=True)
            )
            for axis in (0, 1)
        )
        vector = 16 // fake.element_size()
        if physical[1] != 1 or physical[0] <= 0 or physical[0] % vector:
            return fallback
        offset = chain._copy_code(
            sympy.Add(
                *(base * stride for base, stride in zip(bases, strides, strict=True))
            )
        )
        guards.extend(
            f"{name}.layout.stride[{axis}] == {stride}"
            for axis, stride in enumerate(strides)
        )
    except chain._UnsupportedChain:
        return fallback
    guards.append("chain_store_pointer.toint() % 16 == 0")
    groups = n // vector
    columns = min(128, groups & -groups)
    return [
        f"chain_store_pointer = {name}.iterator + ({offset})",
        f"if {' & '.join(f'({guard})' for guard in guards)}:",
        f"    chain_store_target = cute.make_tensor(chain_store_pointer.align(16), cute.make_layout({shape!r}, stride=({physical[0]}, 1)))",
        f"    chain_store_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {dtype}, num_bits_per_copy=128), cute.make_layout(({128 // columns}, {columns}), stride=({columns}, 1)), cute.make_layout((1, {vector})))",
        "    chain_store_thread = chain_store_copy.get_slice(chain_thread)",
        "    cute.copy(chain_store_copy, chain_store_thread.partition_S(chain_output), chain_store_thread.partition_D(chain_store_target))",
        "else:",
        chain._indent(fallback),
    ]


def _epilogue(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    staged: list[chain._StagedInput],
) -> list[str]:
    stage = len(plan.dots) - 1
    prefix = f"chain_{stage}"
    shape = plan.shapes[-1][:2]
    coords = ("chain_epi_row", "chain_epi_col")
    index = "chain_epi_index"
    node = cast("Node", plan.store.args[2])
    expression = chain._Expression(cg, plan, boundaries)
    expression.scan_inputs = scans
    expression.coordinate_names.update(coords)
    fragment = (
        plan.dots[-1]
        if plan.initialized_accumulator is None
        else plan.initialized_accumulator.join
    )
    expression.fragments[fragment] = (coords, f"{prefix}_values[{index}]")
    expression.value(node, coords)
    reused: dict[tuple[Node, tuple[str, ...]], str] = {}
    lines: list[str] = []
    for leaf, leaf_coords, indices, _ in expression.loaded_inputs:
        if (
            plan.late_rhs_reuse is not None
            and _shared_leaf(
                expression,
                cast("Node", leaf.args[0]),
                indices,
                [item for item in staged if item.role == "a"],
                coords,
                shape,
            )
            is not None
        ):
            # The final collective copy writes A. A staged-A reader in
            # another warp need not have completed without a new barrier.
            raise chain._UnsupportedChain(
                "late RHS output arena has an epilogue reader"
            )
        shared = _shared_leaf(
            expression, cast("Node", leaf.args[0]), indices, staged, coords, shape
        )
        if shared is None:
            continue
        tag = f"chain_epi_input_{len(reused)}"
        dtype = CompileEnvironment.current().backend.dtype_str(leaf.meta["val"].dtype)
        lines.extend(
            [
                f"{tag}_shared = {shared}",
                f"{tag}_source = {prefix}_thread.partition_D({prefix}_slice.partition_C({tag}_shared))",
                f"{tag}_values = cute.make_rmem_tensor({prefix}_coords.shape, {dtype})",
                f"cute.autovec_copy({tag}_source, {tag}_values)",
            ]
        )
        reused[leaf, leaf_coords] = f"{tag}_values[{index}]"
    expression = chain._Expression(cg, plan, boundaries)
    expression.scan_inputs = scans
    expression.coordinate_names.update(coords)
    expression.fragments[fragment] = (coords, f"{prefix}_values[{index}]")
    expression.memo.update(reused)
    value = expression.value(node, coords)
    outer, inner = _hoist(expression, {*coords, index})
    dtype = CompileEnvironment.current().backend.dtype_str(
        cast("Node", plan.store.args[0]).meta["val"].dtype
    )
    lines.extend(
        [
            *outer,
            f"chain_epi_values = cute.make_rmem_tensor({prefix}_coords.shape, {dtype})",
            f"for {index} in cutlass.range_constexpr(cute.size({prefix}_values)):",
            f"    {coords[0]}, {coords[1]} = {prefix}_coords[{index}]",
            chain._indent(inner),
            f"    chain_epi_values[{index}] = {dtype}({value})",
            *(
                [
                    "cute.arch.sync_threads()",
                    *direct_store(cg, plan, expression, coords, prefix),
                ]
                if plan.direct_output
                else [
                    f"chain_epi_target = {prefix}_thread.partition_D({prefix}_slice.partition_C(chain_output))",
                    "cute.autovec_copy(chain_epi_values, chain_epi_target)",
                    "cute.arch.sync_threads()",
                    *_store(cg, plan, expression, coords, shape, dtype),
                ]
            ),
        ]
    )
    return lines


def _issue_k_halves(
    prefix: str,
    stage: int,
    mode: str,
    producer: list[str],
    *,
    retire_each_half: bool = False,
) -> list[str]:
    return emit_k_half_issues(
        prefix, stage, mode, producer, retire_each_half=retire_each_half
    )


def _last_read_epilogue(
    body: list[str], epilogue: list[str], final_stage: int
) -> list[str]:
    """Release only after a complete register snapshot and all-reader rendezvous.

    The resident emitter owns these storage names. Track aliases of its TMEM
    pointer, but not pure layout/copy metadata or newly allocated registers.
    Fail closed if the following epilogue still references any TMEM storage.
    """
    prefix = f"chain_{final_stage}"
    if body[-3:] != [
        f"cute.copy({prefix}_copy, {prefix}_source, {prefix}_values)",
        "cute.arch.fence_view_async_tmem_load()",
        "cute.arch.sync_threads()",
    ]:
        raise chain._UnsupportedChain(
            "last_read requires the final TMEM load fence and CTA rendezvous"
        )
    aliases = {"chain_tptr"}
    metadata_factories = {
        "cute.make_rmem_tensor",
        "cute.make_identity_tensor",
        "cute.make_copy_atom",
        "cute.make_tiled_copy_tv",
        "tcgen05.make_tmem_copy",
    }

    def depends(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in aliases
        if isinstance(node, ast.Attribute) and node.attr in (
            "shape",
            "layout",
            "dtype",
        ):
            return False
        if isinstance(node, ast.Call) and ast.unparse(node.func) in metadata_factories:
            return False
        return any(depends(child) for child in ast.iter_child_nodes(node))

    statements = ast.parse("\n".join(body))
    while True:
        previous = set(aliases)
        for node in ast.walk(statements):
            if isinstance(node, ast.Assign) and depends(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases.add(target.id)
                    else:
                        raise chain._UnsupportedChain(
                            "last_read has an unproven TMEM alias target"
                        )
            elif isinstance(
                node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)
            ) and depends(node):
                raise chain._UnsupportedChain(
                    "last_read has an unproven TMEM alias statement"
                )
        if aliases == previous:
            break
    for node in ast.walk(ast.parse("\n".join(epilogue))):
        if (
            (
                isinstance(node, ast.Name)
                and node.id in aliases | {"chain_allocator", "tcgen05"}
            )
            or (isinstance(node, ast.Attribute) and "tmem" in node.attr)
            or (isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.gemm")
        ):
            raise chain._UnsupportedChain("last_read epilogue has a later TMEM use")
    return ["chain_allocator.free(chain_tptr)", *epilogue]


def codegen_chained_tcgen05(cg: GenerateAST, plan: ChainedMatmulPlan) -> bool:
    """Legacy root scheduling entry; unported stage policies remain unchanged."""
    return _codegen_root_sequence(cg, plan, shared_stage_actions=False)


def codegen_shared_root_sequence(
    cg: GenerateAST, plan: ChainedMatmulPlan, *, snapshot_tile_columns: int = 0
) -> bool:
    """Reuse root ordering while executing accepted physical stages in emit_stage."""
    return _codegen_root_sequence(
        cg, plan, shared_stage_actions=True, snapshot_tile_columns=snapshot_tile_columns
    )


def _codegen_root_sequence(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    *,
    shared_stage_actions: bool,
    snapshot_tile_columns: int = 0,
) -> bool:
    from .native_matmul_metadata import body_attempt

    with body_attempt(cg):
        return _emit_root_sequence(
            cg,
            plan,
            shared_stage_actions=shared_stage_actions,
            snapshot_tile_columns=snapshot_tile_columns,
        )


def _emit_root_sequence(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    *,
    shared_stage_actions: bool,
    snapshot_tile_columns: int = 0,
) -> bool:
    from .chained_execution import ChainedExecution
    from .chained_tcgen_stage import emit_full_k_issue
    from .chained_tmem_drains import DrainTiling

    df = cg.device_function
    drain_tiling = DrainTiling(
        cast("int", df.config.config.get("cute_chained_drain_tile_columns", 0))
    )
    vector_staging = VectorStaging(
        cast("bool", df.config.config.get("cute_chained_pointwise_vectorize", False)),
        group_enabled=cast(
            "bool", df.config.config.get("cute_chained_vector_group", False)
        ),
    )
    seed_tiling = SeedTiling(
        cast("int", df.config.config.get("cute_chained_seed_tile_columns", 0))
    )
    cache_layouts = PointwiseCacheLayouts(
        cast("str", df.config.config.get("cute_chained_pointwise_cache_layout", "auto"))
    )
    scratch = ScratchLayouts.for_plan(
        cast("str", df.config.config.get("cute_chained_scratch_layout", "row_major")),
        plan,
    )
    pointwise_unroll = PointwiseUnroll(
        cast("int", df.config.config.get("cute_chained_pointwise_unroll", 1))
    )
    pointwise_cache = PointwiseReadCache(
        cast("bool", df.config.config.get("cute_chained_pointwise_read_cache", False))
    )
    pointwise_inplace = PointwiseInplace(
        cast(
            "bool", df.config.config.get("cute_chained_pointwise_inplace_async", False)
        )
    )
    early_release = df.config.config.get("cute_chained_tmem_early_release", False)
    dtype = CompileEnvironment.current().backend.dtype_str(plan.dtype)
    index_dtype = CompileEnvironment.current().backend.dtype_str(
        CompileEnvironment.current().index_dtype
    )
    lines = [
        "from cutlass.cute.nvgpu import tcgen05",
        "from cutlass.utils import blackwell_helpers as chain_sm100",
        "from cutlass.utils import TmemAllocator",
        "import cutlass.pipeline as chain_pipeline",
        "chain_thread = cutlass.Int32(cute.arch.thread_idx()[0])",
        "chain_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())",
        "chain_pid = cutlass.Int32(cute.arch.block_idx()[0])",
    ]
    suffix = 1
    for axis, extent, block in reversed(plan.axes):
        count = extent // block
        lines.append(
            f"chain_origin_{axis} = {index_dtype}(chain_pid // {suffix} % {count}) * {block}"
        )
        suffix *= count
    general_collectives = uses_general_collectives(plan)
    boundaries: dict[Node, str] = (
        collective_bindings(plan) if general_collectives else {}
    )
    scans: list[chain._ScanInput] = []
    try:
        scan_lines = (
            []
            if general_collectives
            else chain._codegen_scans(cg, plan, boundaries, scans, True)
        )
        bridges = {
            stage: bridge
            for stage, bridge in chain._register_bridges(
                cg, plan, boundaries, dtype, scans
            ).items()
            if bridge.role == "a"
        }
        if (
            df.config.config.get("cute_chained_fragment_epilogues") is True
            and not bridges
        ):
            raise chain._UnsupportedChain(
                "fragment epilogues lack an admitted original result image"
            )
        independent_inputs = None
        if shared_stage_actions:
            from .chained_root_stage import independent_root_inputs
            from .chained_root_stage import supports_independent_root

            if supports_independent_root(cg, plan):
                independent_inputs = independent_root_inputs(cg, plan, boundaries)
        inner_axes = {}
        if independent_inputs is not None:
            inner_axes = dict(
                zip(((0, "a"), (0, "b")), independent_inputs.axes[0], strict=True)
            )
        else:
            for stage, node in enumerate(plan.dots):
                for role, operand in zip(("a", "b"), node.args[:2], strict=True):
                    inner = chain._operand_inner_axis(
                        cg,
                        plan,
                        {
                            **boundaries,
                            **{dot: f"chain_{i}_c" for i, dot in enumerate(plan.dots)},
                        },
                        cast("Node", operand),
                    )
                    inner_axes[stage, role] = inner if role == "a" else 1 - inner
        pair_inputs = None
        if shared_stage_actions:
            from .chained_root_stage import capture_root_pair_inputs

            pair_inputs = capture_root_pair_inputs(
                cg, plan, boundaries, inner_axes, bridges
            )
        if plan.k_schedule is not None and inner_axes[plan.k_schedule.stage, "a"] != 1:
            raise chain._UnsupportedChain("K64 scheduling requires K-major final A")
        startup_inputs = []
        startup_selection = None
        startup_issued = False
        if df.config.config.get("cute_chained_startup_transfer", "legacy") == "tma":
            from .chained_startup import plan_startup

            if (
                plan.initialized_accumulator is not None
                or plan.late_rhs_reuse is not None
                or plan.k_schedule is not None
            ):
                raise chain._UnsupportedChain(
                    "startup TMA requires uninitialized M128 full scheduling or an independent M64 dot"
                )
            if not supported_plan(plan, startup=True):
                raise chain._UnsupportedChain(
                    "startup TMA barrier exceeds shared capacity"
                )
            startup_inputs = plan_startup(cg, plan, boundaries, scans, inner_axes)
            if independent_inputs is not None:
                from .chained_root_stage import capture_root_startup

                startup_selection = capture_root_startup(
                    cg, plan, startup_inputs, independent_inputs.axes
                )
        leaf_pipeline = plan_leaf_pipeline(
            cg, plan, boundaries, scans, inner_axes[len(plan.dots) - 1, "a"]
        )
        paired_selection = None
        if shared_stage_actions and leaf_pipeline is not None:
            from .chained_root_stage import capture_paired_leaf
            from .chained_root_stage import supports_initialized_root

            if supports_initialized_root(cg, plan):
                paired_selection = capture_paired_leaf(
                    cg, plan, boundaries, scans, inner_axes, leaf_pipeline
                )
        leaf_extra_bytes = 128 if leaf_pipeline is not None else 0
        if leaf_pipeline is not None:
            device = cast("Node", plan.dots[0].args[0]).meta["val"].device
            if _shared_memory_bytes(
                plan, startup=bool(startup_inputs)
            ) + leaf_extra_bytes > CuteTcgen05Config.per_cta_smem_capacity_bytes(
                device
            ):
                raise chain._UnsupportedChain(
                    "paired leaf transaction barrier exceeds shared capacity"
                )
        seed_lines: list[str] = []
        initialized_result = None
        if plan.initialized_accumulator is not None:
            from .chained_initialized_accumulator import codegen_seed

            if bridges:
                raise chain._UnsupportedChain("initialized accumulator with bridge")
            state_effects: list[StateEffect] = []
            seed_lines = codegen_seed(
                cg,
                plan,
                boundaries,
                scans,
                inner_axes,
                seed_tiling=seed_tiling,
                effects=state_effects,
            )
            if shared_stage_actions:
                from .chained_root_stage import capture_initialized_result
                from .chained_root_stage import supports_initialized_root

                if supports_initialized_root(cg, plan):
                    initialized_result = capture_initialized_result(
                        cg,
                        plan,
                        boundaries,
                        scans,
                        inner_axes,
                        seed_tiling,
                        seed_lines,
                        paired=paired_selection,
                        state_effects=tuple(state_effects),
                    )
        a_size = max(m * k for m, _, k in plan.shapes)
        b_size = max(n * k for _, n, k in plan.shapes)
        if plan.late_rhs_reuse is not None:
            a_size = plan.late_rhs_reuse.a_bytes // 2
            b_size = plan.late_rhs_reuse.b_bytes // 2
        max_columns = max(n for _, n, _ in plan.shapes)
        tmem_columns = max(
            32, 2 ** ((max_columns * (2 if bridges else 1) - 1).bit_length())
        )
        output_dtype = CompileEnvironment.current().backend.dtype_str(
            cast("Node", plan.store.args[0]).meta["val"].dtype
        )
        output_shape = plan.shapes[-1][:2]
        prefetch_b = _prefetch_final_b(plan)
        output_storage = "chain_b_workspace" if prefetch_b else "chain_output_ptr"
        if plan.late_rhs_reuse is not None:
            output_storage = "chain_a_workspace"
        final_stage = len(plan.dots) - 1
        lines.extend(
            [
                f"chain_a_workspace = cute.arch.alloc_smem({dtype}, {a_size}, alignment=128)",
                f"chain_b_workspace = cute.arch.alloc_smem({dtype}, {b_size}, alignment=128)",
                *(
                    []
                    if plan.direct_output
                    else [
                        (
                            "chain_output_ptr = chain_a_workspace"
                            if plan.late_rhs_reuse is not None
                            else "chain_output_ptr = chain_b_workspace"
                            if prefetch_b
                            else f"chain_output_ptr = cute.arch.alloc_smem({output_dtype}, {math.prod(output_shape)}, alignment=128)"
                        ),
                        *_layout("chain_output", output_shape, 1, output_dtype),
                    ]
                ),
                f"chain_bars = cute.arch.alloc_smem(cutlass.Int64, {len(plan.dots)}, alignment=16)",
                *(
                    [
                        "chain_leaf_bar = cute.arch.alloc_smem(cutlass.Int64, 1, alignment=16)"
                    ]
                    if leaf_pipeline is not None
                    else []
                ),
                "chain_holding = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=4)",
                "if chain_thread == 0:",
                *(
                    f"    cute.arch.mbarrier_init(chain_bars + {stage}, 1)"
                    for stage in range(len(plan.dots))
                ),
                *(
                    ["    cute.arch.mbarrier_init(chain_leaf_bar, 1)"]
                    if leaf_pipeline is not None
                    else []
                ),
                "cute.arch.mbarrier_init_fence()",
                "chain_allocation_barrier = chain_pipeline.NamedBarrier(barrier_id=1, num_threads=128)",
                "chain_allocator = TmemAllocator(chain_holding, barrier_for_retrieve=chain_allocation_barrier)",
                f"chain_allocator.allocate({tmem_columns})",
            ]
        )
        if general_collectives:
            lines.extend(allocate_collectives(plan, scratch))
        if plan.pointwise_cache is not None:
            lines.extend(
                allocate_pointwise_cache(plan.pointwise_cache, scratch, cache_layouts)
            )
        if early_release:
            # Allocation and release use the same fully active allocator warp.
            # No later allocation occurs; wait/retrieve still publish the pointer.
            lines.append("chain_allocator.relinquish_alloc_permit()")
        deferred_rhs: list[str] = []
        seeded_inputs = None
        prefetch_position = len(lines)
        deferred_position = 0
        prefetch_lines: list[str] = []
        if prefetch_b:
            _, columns, reduction = plan.shapes[-1]
            tag = f"chain_{final_stage}_b"
            final_dtype = CompileEnvironment.current().backend.dtype_str(
                plan.operand_dtype(final_stage)
            )
            prefetch_lines = [
                (
                    f"{tag}_ptr = chain_b_workspace"
                    if plan.late_rhs_reuse is not None
                    else f"{tag}_ptr = cute.arch.alloc_smem({final_dtype}, {columns * reduction}, alignment=128)"
                ),
                *_layout(
                    tag, (columns, reduction), inner_axes[final_stage, "b"], final_dtype
                ),
                *_stage(
                    cg,
                    plan,
                    boundaries,
                    [],  # Scan cache storage is not live during prefetch.
                    final_stage,
                    "b",
                    inner_axes[final_stage, "b"],
                    final_dtype,
                    pointwise_unroll,
                    pointwise_cache,
                    pointwise_inplace,
                ),
            ]
            if plan.late_rhs_reuse is not None:
                deferred_rhs = prefetch_lines
                if initialized_result is not None:
                    from .chained_root_stage import capture_seeded_inputs

                    seeded_inputs = capture_seeded_inputs(
                        initialized_result, plan, deferred_rhs
                    )
            else:
                lines.extend(prefetch_lines)
                if initialized_result is not None:
                    from .chained_root_stage import capture_seeded_inputs

                    seeded_inputs = capture_seeded_inputs(
                        initialized_result, plan, prefetch_lines, mode="upfront"
                    )
        elif initialized_result is not None:
            from .chained_root_stage import capture_seeded_inputs

            seeded_inputs = capture_seeded_inputs(
                initialized_result, plan, [], mode="local"
            )
        early_scan = any(
            chain._ancestors(cast("Node", operand)) & set(plan.scans)
            for operand in plan.dots[0].args[:2]
        ) or (
            plan.pointwise_cache is not None
            and any(
                entry.first_stage == 0 and set(entry.dependencies) & set(plan.scans)
                for entry in plan.pointwise_cache.entries
            )
        )
        if startup_inputs:
            from .chained_startup import issue_lines

            for transfer in startup_inputs:
                tag = f"chain_0_{transfer.role}"
                lines.extend(
                    [
                        f"{tag}_ptr = chain_{transfer.role}_workspace",
                        *_layout(tag, transfer.shape, transfer.inner, dtype),
                    ]
                )
            lines.extend(issue_lines(startup_inputs))
            startup_issued = True
        if early_scan:
            lines.extend(scan_lines)
        early_cached: list[chain._ScanInput] = []
        if df.config.config.get("cute_chained_auxiliary_cache"):
            device = cast("Node", plan.dots[0].args[0]).meta["val"].device
            cache_plan = (
                dataclasses.replace(plan, late_rhs_reuse=None, direct_output=False)
                if plan.late_rhs_reuse is not None or plan.direct_output
                else plan
            )
            cache_lines, early_cached = make_early_auxiliary_cache(
                cg,
                plan,
                scans,
                CuteTcgen05Config.per_cta_smem_capacity_bytes(device)
                - leaf_extra_bytes
                - max(
                    _shared_memory_bytes(plan, startup=bool(startup_inputs)),
                    _shared_memory_bytes(cache_plan, startup=bool(startup_inputs)),
                ),
            )
            lines.extend(cache_lines)
            scans = [*scans, *early_cached]
        # Initiate the first independent operand copies before the scan prelude.
        root_sequence = None
        if shared_stage_actions:
            from .chained_root_stage import plan_root_stage_sequence

            root_sequence = plan_root_stage_sequence(
                cg,
                plan,
                bridges,
                inner_axes,
                scans,
                scan_lines,
                prefetched=prefetch_b and bool(prefetch_lines),
                pointwise_unroll=pointwise_unroll,
                pointwise_cache=pointwise_cache,
                pointwise_inplace=pointwise_inplace,
                independent=independent_inputs,
                pair_inputs=pair_inputs,
                pair_staging=vector_staging
                if pair_inputs is not None and vector_staging.group_enabled
                else None,
                boundaries=boundaries,
                early_scan=early_scan
                if independent_inputs is not None or initialized_result is not None
                else False,
                early_cached=tuple(early_cached)
                if independent_inputs is not None or initialized_result is not None
                else (),
                startup=startup_selection,
                startup_issued=startup_issued,
                initialized=initialized_result,
                seeded_inputs=seeded_inputs,
                snapshot_tile_columns=snapshot_tile_columns,
            )
            if (
                startup_selection is not None
                or initialized_result is not None
                or snapshot_tile_columns
                or pair_inputs is not None
            ) and root_sequence is None:
                raise chain._UnsupportedChain(
                    "accepted root inputs or result changed before stage selection"
                )
        staged: list[chain._StagedInput] = []
        from .chained_body_program import emit_body_program
        from .chained_body_program import plan_root_action_body
        from .chained_body_program import plan_root_body

        action_body = plan_root_action_body(
            cg, plan, root_sequence, boundaries, deferred_rhs=deferred_rhs
        )
        if (
            action_body is not None
            and action_body.continuation is None
            and action_body.sequence.initialized is not None
            and plan.initialized_accumulator is not None
            and plan.region is not None
            and len(action_body.actions) == 2
        ):
            # Keep the original sequence/input admission and public initialized
            # opt-out. Only its proven pair selects the common graph/state body.
            action_body.check()
            action_body = plan_root_action_body(
                cg,
                plan,
                root_sequence,
                boundaries,
                deferred_rhs=deferred_rhs,
                prepared_continuation=True,
            )
            if action_body is None:
                raise chain._UnsupportedChain("accepted initialized root changed")
        body = (
            plan_root_body(
                cg,
                plan,
                scratch,
                inner_axes=inner_axes,
                bridges=bridges,
                early_release=cast("bool", early_release),
                unroll=pointwise_unroll,
                cache=pointwise_cache,
                inplace=pointwise_inplace,
            )
            if root_sequence is None
            else None
        )
        body_start = len(lines)
        if action_body is not None:
            lines.extend(
                emit_body_program(
                    cg,
                    plan,
                    None,
                    action_body.execution,
                    None,
                    None,
                    scratch,
                    root_actions=action_body,
                    drain_tiling=drain_tiling,
                )
            )
            staged.extend(action_body.sequence.staged)
        elif body is not None:
            lines.extend(
                emit_body_program(
                    cg,
                    plan,
                    None,
                    body.execution,
                    None,
                    None,
                    scratch,
                    root=body,
                    drain_tiling=drain_tiling,
                )
            )
            boundaries.update(body.boundaries)
            staged.extend(body.staged)
        else:
            for stage, (node, (m, n, k)) in enumerate(
                zip(plan.dots, plan.shapes, strict=True)
            ):
                if root_sequence is not None and root_sequence.handles(stage):
                    from .chained_tcgen_stage import emit_stage

                    action = root_sequence.action(stage)
                    lines.extend(
                        emit_stage(
                            cg,
                            plan,
                            boundaries,
                            stage,
                            root_sequence.geometries[stage],
                            "0",
                            root_actions=action,
                            pending_allocation=cast("bool", early_release)
                            if stage == 0
                            else None,
                            terminal_fragment=stage == final_stage,
                            tmem_input=action.tmem_input,
                            tmem_accumulator=action.accumulator,
                        )
                    )
                    if stage == final_stage:
                        staged.extend(root_sequence.staged)
                    if stage == 0:
                        deferred_position = len(lines)
                        lines.extend(deferred_rhs)
                        root_sequence.enqueue_rhs(deferred_rhs)
                    if stage != 0 or plan.initialized_accumulator is None:
                        boundaries[node] = f"chain_{stage}_c"
                    continue
                if general_collectives:
                    lines.extend(emit_collectives_before(cg, plan, boundaries, stage))
                if plan.pointwise_cache is not None:
                    lines.extend(
                        emit_pointwise_cache_before(
                            cg, plan, boundaries, plan.pointwise_cache, stage
                        )
                    )
                prefix = f"chain_{stage}"
                dtype = CompileEnvironment.current().backend.dtype_str(
                    plan.operand_dtype(stage)
                )
                split_k = plan.k_schedule is not None and stage == plan.k_schedule.stage
                half_producer: list[str] = []
                major_a = "K" if stage in bridges or inner_axes[stage, "a"] else "MN"
                major_b = "K" if inner_axes[stage, "b"] else "MN"
                source = "TMEM" if stage in bridges else "SMEM"
                lines.append(
                    f"{prefix}_mma = chain_sm100.make_trivial_tiled_mma({dtype}, {dtype}, cute.nvgpu.OperandMajorMode.{major_a}, cute.nvgpu.OperandMajorMode.{major_b}, cutlass.Float32, tcgen05.CtaGroup.ONE, ({m}, {n}), tcgen05.OperandSource.{source})"
                )
                grouped_producer = None
                if (
                    vector_staging.group_enabled
                    and m == n
                    and inner_axes[stage, "a"] == inner_axes[stage, "b"] == 1
                    and stage not in bridges
                    and not split_k
                    and not startup_inputs
                    and not (prefetch_b and stage == final_stage)
                    and not scans
                    and not early_cached
                    and not pointwise_cache.enabled
                    and not pointwise_inplace.enabled
                ):
                    from .chained_tcgen_stage import StageGeometry

                    # Ordinary root and loop stages share the same coordinate-owned
                    # producer. Async/prefetched and legacy scan inputs retain their
                    # existing schedules; no transfer is moved across a wait here.
                    geometry = StageGeometry((m, n, k), transpose=False)
                    operands = tuple(
                        VectorStageOperand(
                            cast("Node", operand),
                            geometry,
                            role,
                            (m, k),
                            f"{prefix}_{role}",
                        )
                        for role, operand in zip(("a", "b"), node.args[:2], strict=True)
                    )
                    grouped_producer = emit_vector_stage_group(
                        cg,
                        plan,
                        boundaries,
                        operands,
                        vector_staging,
                        tag=f"{prefix}_vector_group",
                        producer_unroll=pointwise_unroll,
                    )
                    if grouped_producer is not None:
                        for role in ("a", "b"):
                            lines.extend(
                                [
                                    f"{prefix}_{role}_ptr = chain_{role}_workspace",
                                    *_layout(f"{prefix}_{role}", (m, k), 1, dtype),
                                ]
                            )
                        lines.extend(grouped_producer)
                for role, shape in (("a", (m, k)), ("b", (n, k))):
                    if role == "a" and stage in bridges:
                        continue
                    transfer = next(
                        (
                            item
                            for item in startup_inputs
                            if stage == 0 and item.role == role
                        ),
                        None,
                    )
                    if transfer is not None:
                        lines.extend(
                            _finish_startup_operand(
                                cg,
                                plan,
                                transfer,
                                boundaries,
                                scans if early_scan else early_cached,
                                stage,
                                inner_axes[stage, role],
                                dtype,
                                pointwise_unroll,
                                pointwise_cache,
                                pointwise_inplace,
                            )
                        )
                    elif grouped_producer is None and not (
                        prefetch_b and stage == final_stage and role == "b"
                    ):
                        producer = _stage(
                            cg,
                            plan,
                            boundaries,
                            scans if stage > 0 or early_scan else early_cached,
                            stage,
                            role,
                            inner_axes[stage, role],
                            dtype,
                            pointwise_unroll,
                            pointwise_cache,
                            pointwise_inplace,
                            k_half="chain_k_half" if split_k and role == "a" else None,
                            leaf_pipeline=leaf_pipeline
                            if split_k and role == "a"
                            else None,
                        )
                        if split_k and role == "a":
                            half_producer = producer
                        lines.extend(
                            [
                                f"{prefix}_{role}_ptr = chain_{role}_workspace",
                                *_layout(
                                    f"{prefix}_{role}",
                                    shape,
                                    inner_axes[stage, role],
                                    dtype,
                                ),
                                *([] if split_k and role == "a" else producer),
                            ]
                        )
                    if stage == len(plan.dots) - 1:
                        cached = chain._stage_input(
                            cg,
                            plan,
                            cast("Node", node.args[0 if role == "a" else 1]),
                            f"{prefix}_{role}",
                            role,
                            shape,
                        )
                        if cached is not None:
                            staged.append(cached)
                if not split_k:
                    lines.append("cute.arch.cp_async_commit_group()")
                if stage == 0 and not early_scan:
                    lines.extend(scan_lines)
                if stage == 0 and startup_inputs:
                    lines.extend(
                        [
                            "cute.arch.mbarrier_wait(chain_start_bar, 0)",
                            "cute.arch.sync_threads()",
                        ]
                    )
                lines.extend(
                    []
                    if split_k
                    else [
                        "cute.arch.cp_async_wait_group(0)",
                        "cute.arch.fence_view_async_shared()",
                        "cute.arch.sync_threads()",
                    ]
                )
                if stage == 0:
                    lines.extend(
                        [
                            "chain_allocator.wait_for_alloc()",
                            "chain_tptr = chain_allocator.retrieve_ptr(cutlass.Float32)",
                        ]
                    )
                    if not early_release:
                        lines.append("chain_allocator.relinquish_alloc_permit()")
                if stage in bridges:
                    if (
                        stage == final_stage
                        and output_storage != "chain_a_workspace"
                        and df.config.config.get("cute_chained_auxiliary_cache")
                    ):
                        # The previous MMA has completed and all threads passed
                        # the barrier above. Final A is TMEM, so no later SMEM-A
                        # consumer can observe this arena. Output is disjoint.
                        cache_lines, cached = make_late_auxiliary_cache(
                            cg,
                            plan,
                            scans,
                            arena="chain_a_workspace",
                            arena_bytes=2
                            * max(
                                rows * reduction for rows, _, reduction in plan.shapes
                            ),
                        )
                        lines.extend(cache_lines)
                        # Publish only after the emitted cache-fill barrier.
                        scans = [*scans, *cached]
                    lines.extend(_bridge(cg, plan, boundaries, scans, stage, dtype))
                offset = max_columns if stage in bridges else 0
                lines.extend(
                    [
                        f"{prefix}_layout = {prefix}_mma.make_fragment_C({prefix}_mma.partition_shape_C(({m}, {n}))).layout",
                        f"{prefix}_acc = cute.make_tensor(chain_tptr + {offset}, {prefix}_layout)",
                        f"{prefix}_slice = {prefix}_mma.get_slice(0)",
                    ]
                )
                lines.extend(
                    _explicit_accumulator(
                        cg, plan, boundaries, scans, stage, seed_tiling=seed_tiling
                    )
                )
                if stage in bridges:
                    lines.extend(emit_tmem_operand_view(prefix, (m, n, k), dtype))
                    a_slice = f"{prefix}_ra[None, None, {prefix}_kk, 0]"
                else:
                    lines.append(
                        f"{prefix}_ra = {prefix}_mma.make_fragment_A({prefix}_slice.partition_A({prefix}_a))"
                    )
                    a_slice = f"{prefix}_ra[None, None, {prefix}_kk]"
                if split_k:
                    assert plan.k_schedule is not None
                    lines.extend(
                        _issue_k_halves(
                            prefix,
                            stage,
                            plan.k_schedule.mode,
                            half_producer,
                            retire_each_half=leaf_pipeline is not None,
                        )
                    )
                lines.extend(
                    _load_result(prefix, (m, n))
                    if split_k
                    else [
                        *emit_full_k_issue(
                            prefix,
                            stage,
                            "0",
                            a_slice,
                            node.args[2] is not None
                            or (
                                stage == 1 and plan.initialized_accumulator is not None
                            ),
                            ChainedExecution(128),
                        ),
                        *(
                            [
                                "cute.arch.fence_view_async_shared()",
                                "cute.arch.sync_threads()",
                                *leaf_pipeline.issue("0", "0"),
                            ]
                            if stage == 0 and leaf_pipeline is not None
                            else []
                        ),
                        *(
                            []
                            if stage == 0
                            and plan.initialized_accumulator is not None
                            and seed_tiling.max_columns
                            else _load_result(prefix, (m, n))
                        ),
                    ]
                )
                if stage == 0 and plan.initialized_accumulator is not None:
                    lines.extend(seed_lines)
                elif stage < len(plan.dots) - 1 and stage + 1 not in bridges:
                    lines.extend(_materialize_result(prefix, (m, n), scratch))
                lines.append("cute.arch.sync_threads()")
                if stage == 0:
                    # First B is dead, and the FP32 seed store fence plus CTA
                    # boundary has published it before any final-RHS overwrite.
                    deferred_position = len(lines)
                    lines.extend(deferred_rhs)
                if stage != 0 or plan.initialized_accumulator is None:
                    boundaries[node] = f"{prefix}_c"
        terminal_collectives = (
            emit_collectives_before(cg, plan, boundaries, len(plan.dots))
            if general_collectives
            else []
        )
        epilogue = [
            *(
                [
                    *_materialize_result(
                        f"chain_{final_stage}", plan.shapes[-1][:2], scratch
                    ),
                    "cute.arch.sync_threads()",
                ]
                if general_collectives and needs_materialized_final(plan)
                else []
            ),
            *terminal_collectives,
            *_epilogue(cg, plan, boundaries, scans, staged),
            *codegen_scan_exports(cg, plan, boundaries, scans),
        ]
        if leaf_pipeline is not None:
            finish_wrapper(cg, plan, leaf_pipeline)
        if df.config.config.get("cute_chained_tmem_free", "legacy") == "last_read":
            lines.extend(_last_read_epilogue(lines, epilogue, final_stage))
        else:
            lines.extend(epilogue)
            lines.extend(
                ["cute.arch.sync_threads()", "chain_allocator.free(chain_tptr)"]
            )
        pointwise_unroll.validate()
        pointwise_cache.validate()
        pointwise_inplace.validate()
        scratch.validate()
        cache_layouts.validate()
        seed_tiling.validate()
        drain_tiling.check(df.config.config.get("cute_chained_drain_tile_columns", 0))
        drain_tiling.validate()
        vector_staging.validate_group()
        if action_body is not None and plan.late_rhs_reuse is not None:
            deferred_position = action_body.deferred_position(
                lines, body_start, deferred_rhs
            )
    except chain._UnsupportedChain as error:
        from ... import exc

        raise exc.BackendUnsupported(
            "cute", f"unsupported TCgen05 chain expression: {error}"
        ) from error
    df.preamble = []
    template = _GeneratedCodeTemplate("chain", tuple(plan.tensor_aliases), df.new_var)
    if plan.late_rhs_reuse is not None:
        # Allocate local names in the legacy order. The common renderer names
        # by first occurrence, so relocation otherwise renumbers unrelated
        # arithmetic even though all expression construction is unchanged.
        template.render(
            "\n".join(
                lines[:prefetch_position]
                + deferred_rhs
                + lines[prefetch_position:deferred_position]
                + lines[deferred_position + len(deferred_rhs) :]
            )
        )
    body = ast.parse(template.render("\n".join(lines))).body
    aliases = ast.parse(
        "\n".join(f"{alias} = {name}" for name, alias in plan.tensor_aliases.items())
    ).body
    df.body = [*aliases, *body]
    from .native_matmul_metadata import commit_body

    commit_body(cg, plan, lines)
    return True
