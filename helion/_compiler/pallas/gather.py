"""Pallas indirect-gather/scatter planning and fallback lowering.

Floating ``table[idx]`` normally emits ``one_hot(idx, V) @ table``; int32
accesses emit a boolean one-hot select and reduction. An eligible scheduler may
bind the access to explicit DMA scratch instead. This module only describes the
static transfer group and consumes the binding: scratch, semaphores, and DMA
lifetime remain owned by the active schedule.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..inductor_lowering import CodegenState
    from .plan_tiling import IndexingPattern


# Fail early on oversized tables instead of a generic Mosaic OOM.
# Replace with real VMEM budget accounting once available.
_GATHER_VMEM_THRESHOLD_BYTES: int = 16 << 20  # 16 MiB


@dataclass(frozen=True)
class GatherPlan:
    indirect_pos: int
    none_dims: tuple[int, ...]
    jnp_dtype: str
    table_ndim: int
    index_ndim: int
    emit_select: bool
    use_highest_precision: bool
    dma_group: DmaGroupCandidate | None
    resident_block_bytes: int | None


@dataclass(frozen=True)
class ScatterPlan:
    indirect_pos: int
    jnp_dtype: str
    target_ndim: int
    index_ndim: int
    dma_group: DmaGroupCandidate | None


@dataclass(frozen=True)
class DmaGroupCandidate:
    """A static group of equal-shaped transfers through one indirect axis.

    ``index_node`` is a normal Helion load from resident address metadata.  Its
    one non-scalar tiled dimension supplies ``group_count`` scalar HBM member
    addresses for the leading axis of a contiguous tensor.  Selecting one
    member leaves an untiled, full-or-fixed-slice ``member_shape``; the scheduler
    packs members contiguously in scratch.

    Dynamic live counts and per-member extents deliberately remain outside the
    first implementation.  The group/member representation can add them
    without changing loop scheduling or access-site binding.
    """

    index_node: torch.fx.Node
    index_block_id: int
    group_count: int
    member_shape: tuple[int, ...]


def build_gather_plan(
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    indirect_positions: list[int],
    patterns: list[IndexingPattern],
    config: Config,
    has_extra_mask: bool,
) -> GatherPlan:
    """Validate the gather site and return its plan. Runs during plan_tiling."""
    from ..compile_environment import CompileEnvironment
    from .plan_tiling import resident_block_elements

    if has_extra_mask:
        raise NotImplementedError(
            "Pallas gather: extra_mask is not supported for tensor indices"
        )
    if len(indirect_positions) > 1:
        raise NotImplementedError(
            "Pallas gather: multiple indirect dims are not supported"
        )
    indirect_pos = indirect_positions[0]
    emit_select = not tensor.dtype.is_floating_point
    if emit_select and indirect_pos != 0:
        raise NotImplementedError(
            "Pallas gather: integer table gather on non-zero dim is not yet supported"
        )
    if emit_select and tensor.dtype != torch.int32:
        raise NotImplementedError(
            f"Pallas gather: integer table gather only supports torch.int32, "
            f"got {tensor.dtype}"
        )

    dma_group = _dma_group_candidate(tensor, subscript, indirect_pos, patterns, config)
    elements = resident_block_elements(tensor, patterns, config)
    resident_block_bytes = (
        elements * tensor.dtype.itemsize if elements is not None else None
    )
    if dma_group is None:
        _check_resident_block_size(resident_block_bytes)

    # MXU truncates fp32 to bf16 without HIGHEST. For bf16/fp16 the truncation is a no-op.
    use_highest = tensor.dtype not in (torch.bfloat16, torch.float16)

    none_dims = tuple(i for i, idx in enumerate(subscript) if idx is None)
    jnp_dtype = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    idx_element = subscript[indirect_pos]
    index_ndim = idx_element.meta["val"].ndim  # type: ignore[union-attr]

    return GatherPlan(
        indirect_pos=indirect_pos,
        none_dims=none_dims,
        jnp_dtype=jnp_dtype,
        table_ndim=tensor.ndim,
        index_ndim=index_ndim,
        emit_select=emit_select,
        use_highest_precision=use_highest,
        dma_group=dma_group,
        resident_block_bytes=resident_block_bytes,
    )


def _check_resident_block_size(table_bytes: int | None) -> None:
    if table_bytes is None or table_bytes <= _GATHER_VMEM_THRESHOLD_BYTES:
        return
    raise NotImplementedError(
        f"Pallas gather: resident block is {table_bytes} bytes, exceeds "
        f"the {_GATHER_VMEM_THRESHOLD_BYTES} byte VMEM threshold. The "
        "current codegen requires the full gather axis in VMEM; reduce "
        "V, tile the broadcast dims, or use a half-precision dtype."
    )


def _concrete_size(size: int | torch.SymInt, config: Config) -> int | None:
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    concrete = env.try_concretize_symint(size)
    if isinstance(concrete, int):
        return concrete
    block_id = env.get_block_id(size)
    if block_id is None:
        return None
    block_size = env.block_sizes[block_id].from_config(config)
    return block_size if isinstance(block_size, int) else None


def _member_shape(
    tensor: torch.Tensor,
    patterns: list[IndexingPattern],
    indirect_pos: int,
    config: Config,
) -> tuple[int, ...] | None:
    """Return one indirect member's static rectangular transfer shape."""
    from ..compile_environment import CompileEnvironment
    from .plan_tiling import ArbitrarySlicePattern
    from .plan_tiling import NonePattern
    from helion._compiler.backend import PallasBackend

    env = CompileEnvironment.current()
    backend = env.backend
    assert isinstance(backend, PallasBackend)
    shape: list[int] = []
    tensor_dim = 0
    for pattern in patterns:
        if isinstance(pattern, NonePattern):
            return None
        dim_size = _concrete_size(tensor.shape[tensor_dim], config)
        if dim_size is None:
            return None
        if tensor_dim == indirect_pos:
            tensor_dim += 1
            continue
        if isinstance(pattern, ArbitrarySlicePattern):
            selected = pattern.slice
            if selected.step not in (None, 1):
                return None
            start = 0 if selected.start is None else selected.start
            stop = dim_size if selected.stop is None else selected.stop
            if not isinstance(start, int) or not isinstance(stop, int):
                return None
            if not 0 <= start <= stop <= dim_size:
                return None
            dim_from_end = tensor.ndim - tensor_dim - 1
            required_alignment = backend._get_pallas_required_alignment(
                dim_from_end,
                tensor.ndim,
                tensor.dtype.itemsize * 8,
            )
            if start % required_alignment != 0:
                return None
            dim_size = stop - start
        else:
            # Scalar trailing indices squeeze dimensions and arbitrary tensor
            # indices require a second indirect axis.  Neither is a rectangular
            # member transfer in this first implementation.
            return None
        shape.append(dim_size)
        tensor_dim += 1
    return tuple(shape)


def _dma_group_candidate(
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    indirect_pos: int,
    patterns: list[IndexingPattern],
    config: Config,
) -> DmaGroupCandidate | None:
    """Recognize a static leading-axis DMA group backed by resident metadata."""
    from ...language import memory_ops
    from .plan_tiling import ArbitraryIndexPattern
    from .plan_tiling import TileBeginWithOffsetPattern
    from .plan_tiling import TilePattern

    if indirect_pos != 0 or not tensor.dtype.is_floating_point:
        return None
    if tensor.ndim < 3 or not tensor.is_contiguous():
        return None
    # BlockSpecs normally apply TilePatterns before the body sees a Ref. Manual
    # DMA uses the raw HBM Ref and would need absolute, tail-safe pl.ds() slices.
    # Keep data members untiled; the metadata TilePattern below still defines
    # how many indirect rows form the group.
    if any(isinstance(pattern, TilePattern) for pattern in patterns):
        return None
    index_node = subscript[indirect_pos]
    if (
        not isinstance(index_node, torch.fx.Node)
        or index_node.op != "call_function"
        or index_node.target is not memory_ops.load
    ):
        return None
    index = index_node.meta.get("val")
    if (
        not isinstance(index, torch.Tensor)
        or index.ndim != 1
        or index.dtype != torch.int32
        or (len(index_node.args) > 2 and index_node.args[2] is not None)
    ):
        return None

    index_patterns = index_node.meta.get("indexing_patterns")
    if not isinstance(index_patterns, list):
        return None
    group_patterns = [
        pattern for pattern in index_patterns if isinstance(pattern, TilePattern)
    ]
    if len(group_patterns) != 1:
        return None
    if any(
        not isinstance(
            pattern,
            (
                ArbitraryIndexPattern,
                TileBeginWithOffsetPattern,
                TilePattern,
            ),
        )
        for pattern in index_patterns
    ):
        return None
    group_pattern = group_patterns[0]
    if any(
        isinstance(pattern, TileBeginWithOffsetPattern)
        and pattern.block_id == group_pattern.block_id
        for pattern in index_patterns
    ):
        return None
    group_count = _concrete_size(index.shape[0], config)
    member_shape = _member_shape(tensor, patterns, indirect_pos, config)
    source_rows = _concrete_size(tensor.shape[indirect_pos], config)
    if (
        group_count is None
        or group_count <= 0
        or source_rows is None
        or group_count > source_rows
        or not member_shape
    ):
        return None
    if any(size <= 0 for size in member_shape) or member_shape[-1] % 128 != 0:
        return None
    return DmaGroupCandidate(
        index_node=index_node,
        index_block_id=group_pattern.block_id,
        group_count=group_count,
        member_shape=member_shape,
    )


def build_scatter_plan(
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    indirect_positions: list[int],
    patterns: list[IndexingPattern],
    config: Config,
    has_extra_mask: bool,
) -> ScatterPlan:
    """Validate a Pallas scatter site and return its one-hot plan."""
    from ..compile_environment import CompileEnvironment

    if not tensor.dtype.is_floating_point:
        raise NotImplementedError(
            f"Pallas scatter: only floating-point output dtypes are supported, "
            f"got {tensor.dtype}"
        )
    if has_extra_mask:
        raise NotImplementedError(
            "Pallas scatter: extra_mask is not supported for tensor indices"
        )
    if len(indirect_positions) > 1:
        raise NotImplementedError(
            "Pallas scatter: multiple indirect dims are not supported"
        )
    indirect_pos = indirect_positions[0]
    if indirect_pos != 0:
        raise NotImplementedError("Pallas scatter: only indirect dim 0 is supported")
    idx_element = subscript[indirect_pos]
    index_ndim = idx_element.meta["val"].ndim  # type: ignore[union-attr]
    if index_ndim != 1:
        raise NotImplementedError(
            "Pallas scatter: only rank-1 tensor indices are supported"
        )
    jnp_dtype = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    dma_group = _dma_group_candidate(
        tensor,
        subscript,
        indirect_pos,
        patterns,
        config,
    )
    return ScatterPlan(
        indirect_pos=indirect_pos,
        jnp_dtype=jnp_dtype,
        target_ndim=tensor.ndim,
        index_ndim=index_ndim,
        dma_group=dma_group,
    )


def dma_group_transfer_statements(
    state: CodegenState,
    *,
    direction: str,
    group_count: int,
    index_name: str,
    member_hbm: str,
    aggregate_hbm: str,
    scratch_ref: str,
    sem_ref: str,
    methods: tuple[str, ...],
) -> list[ast.stmt]:
    """Emit starts and/or an aggregate wait for one indirect DMA group."""
    from .codegen import async_copy_statements

    result: list[ast.stmt] = []
    if "start" in methods:
        lane_name = state.device_function.new_var("_dma_member")
        member_hbm = member_hbm.replace("{index}", f"{index_name}[{lane_name}]")
        member_vmem = f"{scratch_ref}.at[{lane_name}]"
        source, destination = (
            (member_vmem, member_hbm)
            if direction == "store"
            else (member_hbm, member_vmem)
        )
        loop = statement_from_string(
            f"for {lane_name} in range({group_count}):\n    pass"
        )
        assert isinstance(loop, ast.For)
        loop.body = async_copy_statements(
            state,
            source,
            destination,
            sem_ref,
            ("start",),
            "_scatter_copy" if direction == "store" else "_gather_copy",
        )
        result.append(loop)
    if "wait" in methods:
        source, destination = (
            (scratch_ref, aggregate_hbm)
            if direction == "store"
            else (aggregate_hbm, scratch_ref)
        )
        result.extend(
            async_copy_statements(
                state,
                source,
                destination,
                sem_ref,
                ("wait",),
                "_scatter_wait" if direction == "store" else "_gather_wait",
            )
        )
    return result


def emit_grid_dma_group(
    state: CodegenState,
    plan: GatherPlan | ScatterPlan,
    name: str,
    direction: str,
) -> None:
    """Emit the initial immediate-wait policy for a root-grid DMA binding."""
    from . import codegen as pallas_codegen

    binding = pallas_codegen.grid_memory_op_dma_binding(state)
    if binding is None:
        return
    group = plan.dma_group
    assert group is not None
    scratch_ref, sem_ref = binding
    ast_subscripts = state.ast_args[1]
    assert isinstance(ast_subscripts, list)
    ast_idx = ast_subscripts[plan.indirect_pos]
    assert isinstance(ast_idx, ast.AST)
    index_name = state.codegen.lift(ast_idx, dce=False, prefix="index").id
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))
    parts, _ = pallas_codegen.index_parts(state, subscript, tensor)
    member_parts = [*parts]
    member_parts[plan.indirect_pos] = "{index}"
    aggregate_parts = [*parts]
    aggregate_parts[plan.indirect_pos] = f"pl.ds(0, {group.group_count})"
    member_hbm = f"{name}.at[{', '.join(member_parts)}]"
    aggregate_hbm = f"{name}.at[{', '.join(aggregate_parts)}]"
    for statement in dma_group_transfer_statements(
        state,
        direction=direction,
        group_count=group.group_count,
        index_name=index_name,
        member_hbm=member_hbm,
        aggregate_hbm=aggregate_hbm,
        scratch_ref=scratch_ref,
        sem_ref=sem_ref,
        methods=("start", "wait"),
    ):
        state.codegen.add_statement(statement)


def emit_gather(
    state: CodegenState,
    plan: GatherPlan,
    name: str,
) -> ast.AST:
    """Emit ``one_hot(idx, V) @ table``.

    MXU accumulates in fp32 via ``preferred_element_type``. fp32 tables need
    HIGHEST and fp32 one_hot; bf16/fp16 stay in the table dtype (MXU truncation
    is a no-op and we avoid a VMEM upcast).

    Contracting dim is ``jnp.ndim(idx)``: one_hot adds one trailing axis.
    """
    from . import codegen as pallas_codegen

    dma_ref = pallas_codegen.memory_op_dma_scratch(state)
    if dma_ref is not None:
        emit_grid_dma_group(state, plan, name, "load")
        return expr_from_string(f"{dma_ref}[...]")
    _check_resident_block_size(plan.resident_block_bytes)

    ast_subscripts = state.ast_args[1]
    assert isinstance(ast_subscripts, list)
    ast_idx = ast_subscripts[plan.indirect_pos]
    assert isinstance(ast_idx, ast.AST)
    idx_name = state.codegen.lift(ast_idx, dce=False, prefix="index").id
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))

    parts, _ = pallas_codegen.index_parts(state, subscript, tensor)
    base_index = ", ".join(parts)
    table_expr = f"{name}[{base_index}]"

    if plan.emit_select:
        mask_expr = (
            f"jax.nn.one_hot({idx_name}[...], {name}.shape[0], dtype={plan.jnp_dtype})"
        )
        for _ in range(plan.table_ndim - 1):
            mask_expr = f"jnp.expand_dims({mask_expr}, axis=-1)"
        result = expr_from_string(
            f"jnp.sum({table_expr} * {mask_expr}, "
            f"axis=jnp.ndim({idx_name}[...])"
            f").astype({plan.jnp_dtype})"
        )
        for dim in plan.none_dims:
            result = expr_from_string(
                f"jnp.expand_dims({{result}}, axis={dim})", result=result
            )
        return result

    if plan.use_highest_precision:
        oh_dtype = "jnp.float32"
        table_dot_expr = f"{table_expr}.astype(jnp.float32)"
        precision_arg = "precision=jax.lax.Precision.HIGHEST, "
    else:
        oh_dtype = plan.jnp_dtype
        table_dot_expr = table_expr
        precision_arg = ""

    p = plan.indirect_pos
    result = expr_from_string(
        "jax.lax.dot_general("
        f"jax.nn.one_hot({idx_name}[...], {name}.shape[{p}], dtype={oh_dtype}), "
        f"{table_dot_expr}, "
        f"(((jnp.ndim({idx_name}[...]),), ({p},)), ((), ())), "
        "preferred_element_type=jnp.float32, "
        f"{precision_arg}"
        f").astype({plan.jnp_dtype})"
    )
    if p > 0:
        n = plan.index_ndim
        src = tuple(range(n, n + p))
        dst = tuple(range(p))
        result = expr_from_string(
            f"jnp.moveaxis({{result}}, {src}, {dst})", result=result
        )
    for dim in plan.none_dims:
        result = expr_from_string(
            f"jnp.expand_dims({{result}}, axis={dim})", result=result
        )
    return result


def _scatter_one_hot_name(
    state: CodegenState,
    plan: ScatterPlan,
    name: str,
) -> str:
    ast_subscripts = state.ast_args[1]
    assert isinstance(ast_subscripts, list)
    ast_idx = ast_subscripts[plan.indirect_pos]
    assert isinstance(ast_idx, ast.AST)
    idx_name = state.codegen.lift(ast_idx, dce=False, prefix="index").id
    # TODO(tcombes): investigate making the metadata into dtype,
    # currently hitting Mosaic issues with bf16 mask.
    return (
        f"jax.nn.one_hot({idx_name}[...], {name}.shape[{plan.indirect_pos}], "
        "dtype=jnp.float32)"
    )


def emit_scatter_store(
    state: CodegenState,
    plan: ScatterPlan,
    name: str,
    base_index: str,
    value: ast.AST,
) -> ast.AST | None:
    """Emit one Pallas program's tensor-indexed store block.

    ``base_index`` is the target block with the indirect dimension replaced by
    ``:``. For ``target[idx, cols] = value`` this is ``target[:, cols]``.

    The lowering builds:
      - ``oh``: one-hot source-lane-to-target-row map, shape ``[M, V]``.
      - ``same_output_row``/``is_lane_j_after_i``/``is_last_writer``: local duplicate
        detection. If two lanes in this program target the same row, only the
        last source lane is kept.
      - ``row_to_lane``: target-row-to-source-lane map, shape ``[V, M]``.
      - ``updates``: projected values for every row in ``base_index``.
      - ``mask``: rows touched by this program.

    The final expression is ``where(mask, updates, old_target_block)`` so
    untouched rows keep their previous value. Duplicate handling is local to
    this Pallas program; duplicate writes from different programs have the same
    unspecified winner semantics as regular parallel stores in other backends.
    """
    from . import codegen as pallas_codegen

    dma_ref = pallas_codegen.memory_op_dma_scratch(state)
    if dma_ref is not None:
        # This intentionally follows torch.index_put_(accumulate=False): writes
        # to duplicate tensor indices have undefined ordering. Avoid a runtime
        # uniqueness check on the scatter hot path.
        state.codegen.add_statement(
            statement_from_string(f"{dma_ref}[...] = {{value}}", value=value)
        )
        emit_grid_dma_group(state, plan, name, "store")
        return None

    oh = _scatter_one_hot_name(state, plan, name)
    m = f"jnp.shape({oh})[0]"
    eye = f"jnp.eye({m}, dtype=jnp.float32)"
    is_lane_j_after_i = f"jnp.triu(jnp.ones(({m}, {m}), dtype=jnp.float32), k=1)"
    same_output_row = (
        f"jax.lax.dot_general({oh}, jnp.swapaxes({oh}, 0, 1), (((1,), (0,)), ((), ())))"
    )
    is_last_writer = f"(jnp.sum(({same_output_row}) * ({is_lane_j_after_i}), axis=1) == 0).astype(jnp.float32)"
    row_to_lane = (
        f"jax.lax.dot_general(jnp.swapaxes({oh}, 0, 1), "
        f"({eye}) * jnp.expand_dims({is_last_writer}, axis=0), "
        "(((1,), (0,)), ((), ())))"
    )
    updates = expr_from_string(
        "jax.lax.dot_general("
        f"{row_to_lane}, "
        "{value}.astype(jnp.float32), "
        "(((1,), (0,)), "
        "((), ())), "
        "preferred_element_type=jnp.float32, "
        "precision=jax.lax.Precision.HIGHEST"
        f").astype({plan.jnp_dtype})",
        value=value,
    )
    mask_expr = (
        "jax.lax.dot_general("
        f"{row_to_lane}, "
        "jnp.ones_like({value}, dtype=jnp.float32), "
        "(((1,), (0,)), ((), ()))"
        ") > 0"
    )
    return expr_from_string(
        f"jnp.where({mask_expr}, {{updates}}, {name}[{base_index}])",
        updates=updates,
        value=value,
    )
