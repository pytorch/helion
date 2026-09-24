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
from .chained_aux_cache import make_late_auxiliary_cache
from .chained_early_aux_cache import make_early_auxiliary_cache
from .chained_early_aux_cache import uses_early_cache
from .chained_pointwise_cache import PointwiseReadCache
from .chained_pointwise_inplace import PointwiseInplace
from .chained_pointwise_inplace import raw_preload
from .chained_pointwise_inplace import sw128_ownership
from .chained_pointwise_unroll import PointwiseUnroll
from .chained_result_transport import direct_store
from .chained_result_transport import is_m64_plan
from .chained_result_transport import load_operation
from .chained_scan_export import codegen_scan_exports
from .fx_matcher import _GeneratedCodeTemplate
from .tcgen05_config import CuteTcgen05Config

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan


def _prefetch_final_b(plan: ChainedMatmulPlan) -> bool:
    operand = cast("Node", plan.dots[-1].args[1])
    return (
        len(plan.dots) > 1
        and chain._direct_operand(operand)
        and not (chain._ancestors(operand) & {*plan.dots, *plan.scans})
        and cast("Node", plan.store.args[0]).meta["val"].dtype == plan.dtype
        and max(n * k for _, n, k in plan.shapes) >= math.prod(plan.shapes[-1][:2])
    )


def supported_plan(plan: ChainedMatmulPlan, *, startup: bool = False) -> bool:
    """Conservative physical bounds, separate from the common semantic proof."""
    if plan.threads != 128 or any(size % block for _, size, block in plan.axes):
        return False
    m64 = is_m64_plan(plan)
    if startup and (
        plan.initialized_accumulator is not None or plan.late_rhs_reuse is not None
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
        *(4 * (chain._shape(scan)[0] + 4) for scan in plan.scans),
        *(
            chain._shape(scan)[0] * leaf.meta["val"].element_size()
            for scan in plan.scans
            for leaf in chain._scan_cache_candidates(scan, plan.scans, plan.store)
        ),
        8 * len(plan.dots),
        4,
    ]
    return sum((size + 127) // 128 * 128 for size in allocations) + 128 * startup


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
) -> list[str]:
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
        f"for {prefix}_{role}_step in cutlass.range({math.prod(shape) // (128)}, unroll=1):",
        (f"    {index} = chain_thread + {prefix}_{role}_step * 128"),
        chain._indent(expression.lines),
        f"    {prefix}_{role}[{x}, {y}] = {chain._masked_operand(value, dtype, domain)}",
    ]
    stride = (shape[1], 1) if inner == 1 else (1, shape[0])
    target = (
        f"{prefix}_{role}"
        if inner == 1
        else f"cute.make_tensor({prefix}_{role}.iterator, cute.select({prefix}_{role}.layout, mode=[1, 0]))"
    )
    return (
        chain._async_copy(
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
        )
        or _pointwise_stage(
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
        or fallback
    )


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
    source = cast("Node", node.args[0])
    fake = source.meta["val"]
    if fake.dtype not in (expression.plan.dtype, torch.float32):
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
    columns = (width) // 8
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
    raw_leaf = pointwise_inplace.select(leaves, shape, inner, plan.dtype)
    if prestaged_leaf is not None:
        matches = [i for i, leaf in enumerate(leaves) if leaf.node is prestaged_leaf]
        if (
            len(matches) != 1
            or sum(node is prestaged_leaf for node, *_ in probe.loaded_inputs) != 1
            or not sw128_ownership(shape, inner)
            or leaves[matches[0]].dtype != plan.dtype
            or (raw_leaf is not None and raw_leaf != matches[0])
        ):
            raise chain._UnsupportedChain(
                "pre-staged leaf requires a unique same-dtype bijective vector map"
            )
        raw_leaf = matches[0]
    raw = f"{prefix}_{role}_raw"
    expression = chain._Expression(cg, plan, boundaries)
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
        column_base=None,
    )
    invariant, varying = _hoist(expression, {names[1], element})
    pointer_lines, bounds, descriptors, loads, vector_preloads = [], [], [], [], []
    for index, leaf in enumerate(leaves):
        name = f"{tag}_leaf_{index}"
        leaf_dtype = CompileEnvironment.current().backend.dtype_str(leaf.dtype)
        copy, thread = f"{tag}_copy", f"{tag}_thread"
        if leaf.dtype != plan.dtype:
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
                f"{name}_partition = {thread}.partition_S({name}_source)",
                f"{name}_values = cute.make_rmem_tensor({name}_partition[None, 0, 0].shape, {leaf_dtype})",
            )
        )
        # These vectors have the same typed address and complete load mask for
        # every row. Keep all original pointer/stride/bounds guards and the
        # untouched scalar fallback; only move their existing copy outside the
        # row loop, still inside this guarded branch.
        (vector_preloads if index in cached_vectors else loads).append(
            f"cute.copy({copy}, {raw if index == raw_leaf else name}_partition[None, {'0' if index in cached_vectors else f'{tag}_step'}, 0], {name}_values)"
        )
    fast = [
        f"{tag}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {dtype}, num_bits_per_copy=128), cute.make_layout(({rows}, {columns}), stride=({columns}, 1)), cute.make_layout((1, 8)))",
        f"{tag}_thread = {tag}_copy.get_slice(chain_thread)",
        f"{tag}_target = {tag}_thread.partition_D({target})",
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
        f"        {names[1]} = chain_thread % {columns} * 8 + {element}",
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


def _load_result(prefix: str, shape: tuple[int, int]) -> list[str]:
    return [
        f"{prefix}_copy = tcgen05.make_tmem_copy(cute.make_copy_atom({load_operation(shape)}, cutlass.Float32), {prefix}_acc)",
        f"{prefix}_thread = {prefix}_copy.get_slice(chain_thread)",
        f"{prefix}_source = {prefix}_thread.partition_S({prefix}_acc)",
        f"{prefix}_identity = {prefix}_slice.partition_C(cute.make_identity_tensor({shape!r}))",
        f"{prefix}_coords = {prefix}_thread.partition_D({prefix}_identity)",
        f"{prefix}_values = cute.make_rmem_tensor({prefix}_coords.shape, cutlass.Float32)",
        f"cute.copy({prefix}_copy, {prefix}_source, {prefix}_values)",
        "cute.arch.fence_view_async_tmem_load()",
    ]


def _bridge(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    stage: int,
    dtype: str,
) -> list[str]:
    previous, prefix = f"chain_{stage - 1}", f"chain_{stage}_bridge"
    shape = plan.shapes[stage - 1][:2]
    producer, consumer = plan.dots[stage - 1 : stage + 1]
    operand = cast("Node", consumer.args[0])
    coords = (f"{prefix}_row", f"{prefix}_col")
    expression = chain._Expression(cg, plan, boundaries)
    expression.scan_inputs = scans
    expression.coordinate_names.update(coords)
    expression.fragments[producer] = (coords, f"{previous}_values[{prefix}_index]")
    value = expression.value(operand, coords)
    domain = chain._operand_domain(cg, operand, coords, plan)
    packed_shape = (shape[0], shape[1] // 2)
    store_repetition = min(32, packed_shape[1] & -packed_shape[1])
    return [
        f"{prefix}_layout = cute.composition({previous}_acc.layout, cute.make_layout({packed_shape!r}))",
        f"{prefix}_coords_layout = cute.composition({previous}_identity.layout, cute.make_layout({packed_shape!r}))",
        f"{prefix}_target = cute.make_tensor(chain_tptr, {prefix}_layout)",
        f"{prefix}_coords_tensor = cute.make_tensor({previous}_identity.iterator, {prefix}_coords_layout)",
        f"{prefix}_copy = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition({store_repetition})), cutlass.Float32), {prefix}_target)",
        f"{prefix}_thread = {prefix}_copy.get_slice(chain_thread)",
        f"{prefix}_destination = {prefix}_thread.partition_D({prefix}_target)",
        f"{prefix}_coords = {prefix}_thread.partition_S({prefix}_coords_tensor)",
        f"{prefix}_packed = cute.make_rmem_tensor({prefix}_coords.shape, cutlass.Float32)",
        f"{prefix}_values = cute.make_tensor(cute.recast_ptr({prefix}_packed.iterator, dtype={dtype}), {previous}_values.layout)",
        f"for {prefix}_index in cutlass.range_constexpr(cute.size({previous}_values)):",
        f"    {coords[0]}, {coords[1]} = {previous}_coords[{prefix}_index]",
        chain._indent(expression.lines),
        f"    {prefix}_values[{prefix}_index] = {chain._masked_operand(value, dtype, domain)}",
        "cute.arch.sync_threads()",
        f"cute.copy({prefix}_copy, {prefix}_packed, {prefix}_destination)",
        "cute.arch.fence_view_async_tmem_store()",
        "cute.arch.sync_threads()",
    ]


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
    for text in expression.lines:
        statement = ast.parse(text).body[0]
        if isinstance(statement, ast.Assign) and not (
            expression.dependencies(ast.unparse(statement.value)) & varying
        ):
            outer.append(text)
        else:
            inner.append(text)
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
    df = cg.device_function
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
    boundaries: dict[Node, str] = {}
    scans: list[chain._ScanInput] = []
    try:
        scan_lines = chain._codegen_scans(cg, plan, boundaries, scans, True)
        bridges = {
            stage: bridge
            for stage, bridge in chain._register_bridges(
                cg, plan, boundaries, dtype, scans
            ).items()
            if bridge.role == "a"
        }
        inner_axes = {}
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
        seed_lines: list[str] = []
        if plan.initialized_accumulator is not None:
            from .chained_initialized_accumulator import codegen_seed

            if bridges:
                raise chain._UnsupportedChain("initialized accumulator with bridge")
            seed_lines = codegen_seed(cg, plan, boundaries, scans, inner_axes)
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
                "chain_holding = cute.arch.alloc_smem(cutlass.Int32, 1, alignment=4)",
                "if chain_thread == 0:",
                *(
                    f"    cute.arch.mbarrier_init(chain_bars + {stage}, 1)"
                    for stage in range(len(plan.dots))
                ),
                "cute.arch.mbarrier_init_fence()",
                "chain_allocation_barrier = chain_pipeline.NamedBarrier(barrier_id=1, num_threads=128)",
                "chain_allocator = TmemAllocator(chain_holding, barrier_for_retrieve=chain_allocation_barrier)",
                f"chain_allocator.allocate({tmem_columns})",
            ]
        )
        if early_release:
            # Allocation and release use the same fully active allocator warp.
            # No later allocation occurs; wait/retrieve still publish the pointer.
            lines.append("chain_allocator.relinquish_alloc_permit()")
        deferred_rhs: list[str] = []
        prefetch_position = len(lines)
        deferred_position = 0
        if prefetch_b:
            _, columns, reduction = plan.shapes[-1]
            tag = f"chain_{final_stage}_b"
            prefetch_lines = [
                (
                    f"{tag}_ptr = chain_b_workspace"
                    if plan.late_rhs_reuse is not None
                    else f"{tag}_ptr = cute.arch.alloc_smem({dtype}, {columns * reduction}, alignment=128)"
                ),
                *_layout(
                    tag, (columns, reduction), inner_axes[final_stage, "b"], dtype
                ),
                *_stage(
                    cg,
                    plan,
                    boundaries,
                    [],  # Scan cache storage is not live during prefetch.
                    final_stage,
                    "b",
                    inner_axes[final_stage, "b"],
                    dtype,
                    pointwise_unroll,
                    pointwise_cache,
                    pointwise_inplace,
                ),
            ]
            if plan.late_rhs_reuse is not None:
                deferred_rhs = prefetch_lines
            else:
                lines.extend(prefetch_lines)
        early_scan = any(
            chain._ancestors(cast("Node", operand)) & set(plan.scans)
            for operand in plan.dots[0].args[:2]
        )
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
                - max(
                    _shared_memory_bytes(plan),
                    _shared_memory_bytes(cache_plan),
                ),
            )
            lines.extend(cache_lines)
            scans = [*scans, *early_cached]
        # Initiate the first independent operand copies before the scan prelude.
        staged: list[chain._StagedInput] = []
        for stage, (node, (m, n, k)) in enumerate(
            zip(plan.dots, plan.shapes, strict=True)
        ):
            prefix = f"chain_{stage}"
            major_a = "K" if stage in bridges or inner_axes[stage, "a"] else "MN"
            major_b = "K" if inner_axes[stage, "b"] else "MN"
            source = "TMEM" if stage in bridges else "SMEM"
            lines.append(
                f"{prefix}_mma = chain_sm100.make_trivial_tiled_mma({dtype}, {dtype}, cute.nvgpu.OperandMajorMode.{major_a}, cute.nvgpu.OperandMajorMode.{major_b}, cutlass.Float32, tcgen05.CtaGroup.ONE, ({m}, {n}), tcgen05.OperandSource.{source})"
            )
            for role, shape in (("a", (m, k)), ("b", (n, k))):
                if role == "a" and stage in bridges:
                    continue
                if not (prefetch_b and stage == final_stage and role == "b"):
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
                    )
                    lines.extend(
                        [
                            f"{prefix}_{role}_ptr = chain_{role}_workspace",
                            *_layout(
                                f"{prefix}_{role}",
                                shape,
                                inner_axes[stage, role],
                                dtype,
                            ),
                            *(producer),
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
            lines.append("cute.arch.cp_async_commit_group()")
            if stage == 0 and not early_scan:
                lines.extend(scan_lines)
            lines.extend(
                [
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
                        * max(rows * reduction for rows, _, reduction in plan.shapes),
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
            if stage in bridges:
                lines.extend(
                    [
                        f"{prefix}_weight_layout = chain_sm100.make_smem_layout_a({prefix}_mma, ({m}, {n}, {k}), {dtype}, 1)",
                        # TMEM make_fragment_A is a layout factory: it returns
                        # a zero-based operand even when passed a tensor with a
                        # nonzero allocation base. Rebase the resulting BF16/
                        # FP16 iterator in its own element units explicitly.
                        f"{prefix}_ra = {prefix}_mma.make_fragment_A({prefix}_weight_layout.outer)",
                        f"{prefix}_ra = cute.make_tensor({prefix}_ra.iterator + (cutlass.Float32.width // {dtype}.width) * chain_tptr.toint(), {prefix}_ra.layout)",
                    ]
                )
                a_slice = f"{prefix}_ra[None, None, {prefix}_kk, 0]"
            else:
                lines.append(
                    f"{prefix}_ra = {prefix}_mma.make_fragment_A({prefix}_slice.partition_A({prefix}_a))"
                )
                a_slice = f"{prefix}_ra[None, None, {prefix}_kk]"
            lines.extend(
                [
                    f"{prefix}_rb = {prefix}_mma.make_fragment_B({prefix}_slice.partition_B({prefix}_b))",
                    "if chain_warp == 0:",
                    f"    {prefix}_mma.set(tcgen05.Field.ACCUMULATE, {stage == 1 and plan.initialized_accumulator is not None})",
                    f"    for {prefix}_kk in cutlass.range_constexpr(cute.size({prefix}_ra, mode=[2])):",
                    f"        cute.gemm({prefix}_mma, {prefix}_acc, {a_slice}, {prefix}_rb[None, None, {prefix}_kk], {prefix}_acc)",
                    f"        {prefix}_mma.set(tcgen05.Field.ACCUMULATE, True)",
                    "    with cute.arch.elect_one():",
                    f"        tcgen05.commit(chain_bars + {stage})",
                    f"cute.arch.mbarrier_wait(chain_bars + {stage}, 0)",
                    *_load_result(prefix, (m, n)),
                ]
            )
            if stage == 0 and plan.initialized_accumulator is not None:
                lines.extend(seed_lines)
            elif stage < len(plan.dots) - 1 and stage + 1 not in bridges:
                lines.extend(
                    [
                        f"{prefix}_c = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, {m * n}, alignment=128), cute.make_layout(({m}, {n}), stride=({n}, 1)))",
                        f"{prefix}_shared_target = {prefix}_thread.partition_D({prefix}_slice.partition_C({prefix}_c))",
                        f"cute.autovec_copy({prefix}_values, {prefix}_shared_target)",
                    ]
                )
            lines.append("cute.arch.sync_threads()")
            if stage == 0:
                # First B is dead, and the FP32 seed store fence plus CTA
                # boundary has published it before any final-RHS overwrite.
                deferred_position = len(lines)
                lines.extend(deferred_rhs)
            if stage != 0 or plan.initialized_accumulator is None:
                boundaries[node] = f"{prefix}_c"
        epilogue = [
            *_epilogue(cg, plan, boundaries, scans, staged),
            *codegen_scan_exports(cg, plan, boundaries, scans),
        ]
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
    cg.cute_uses_matmul = True
    return True
