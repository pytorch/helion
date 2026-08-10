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
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..inductor_lowering import CodegenState
    from .plan_tiling import IndexingPattern


# Fail early on oversized tables instead of a generic Mosaic OOM.
# Replace with real VMEM budget accounting once available.
_GATHER_VMEM_THRESHOLD_BYTES: int = 16 << 20  # 16 MiB
DMA_GROUP_CAPABLE_META = "pallas_dma_group_capable"


def one_hot_full_tensor_fits_vmem(tensor: torch.Tensor) -> bool:
    """Whether an untiled one-hot gather stays below the VMEM guard."""
    try:
        resident_bytes = int(tensor.numel()) * tensor.dtype.itemsize
    except (TypeError, ValueError):
        return True
    return resident_bytes <= _GATHER_VMEM_THRESHOLD_BYTES


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
    resident_block_bytes: int | None


@dataclass(frozen=True)
class DmaGroupCandidate:
    """A static group of equal-shaped transfers through one indirect axis.

    ``index_node`` is a normal Helion load from resident address metadata.  Its
    one non-scalar tiled dimension supplies ``group_count`` scalar HBM member
    addresses for the leading axis of a contiguous tensor.  Selecting one
    member leaves an untiled, full-or-fixed-slice transfer; the scheduler packs
    the traced result shape contiguously in scratch.

    Dynamic live counts and per-member extents deliberately remain outside the
    first implementation.  The group/member representation can add them
    without changing loop scheduling or access-site binding.
    """

    index_node: torch.fx.Node
    index_block_id: int
    group_count: int
    transfer_shape: tuple[int, ...]


def build_gather_plan(
    node: torch.fx.Node,
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

    dma_group = _dma_group_candidate(
        node, tensor, subscript, indirect_pos, patterns, config
    )
    if (
        config.get("pallas_indirect_access_mode", "one_hot") == "dma"
        and node.meta.get(DMA_GROUP_CAPABLE_META, False)
        and dma_group is None
    ):
        from ...exc import InvalidConfig

        raise InvalidConfig(
            "pallas_indirect_access_mode='dma' is not legal for this "
            "block-size configuration"
        )
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


def _metadata_dma_block_id(index_node: torch.fx.Node) -> int | None:
    """Recognize metadata indexing accepted by the indirect DMA scheduler."""
    from ..compile_environment import CompileEnvironment
    from .plan_tiling import ArbitraryIndexPattern
    from .plan_tiling import TileBeginWithOffsetPattern
    from .plan_tiling import TilePattern
    from .plan_tiling import _detect_indexing_pattern

    tensor_node = index_node.args[0] if index_node.args else None
    metadata = (
        tensor_node.meta.get("val") if isinstance(tensor_node, torch.fx.Node) else None
    )
    subscript = index_node.args[1] if len(index_node.args) > 1 else None
    if not isinstance(metadata, torch.Tensor) or not isinstance(
        subscript, (list, tuple)
    ):
        return None

    env = CompileEnvironment.current()
    patterns = []
    for position, item in enumerate(subscript):
        if item is None or position >= metadata.ndim:
            return None
        patterns.append(
            _detect_indexing_pattern(
                item,
                metadata,
                position,
                index_node,
                position,
                env,
            )
        )
    group_patterns = [
        pattern for pattern in patterns if isinstance(pattern, TilePattern)
    ]
    if len(group_patterns) != 1 or any(
        not isinstance(
            pattern,
            (ArbitraryIndexPattern, TileBeginWithOffsetPattern, TilePattern),
        )
        for pattern in patterns
    ):
        return None
    group_block_id = group_patterns[0].block_id
    if any(
        isinstance(pattern, TileBeginWithOffsetPattern)
        and pattern.block_id == group_block_id
        for pattern in patterns
    ):
        return None
    return group_block_id


def dma_group_structural_block_id(
    node: torch.fx.Node,
    tensor: torch.Tensor,
    subscript: Sequence[object],
    indirect_pos: int,
) -> int | None:
    """Return the indirect block id when this site can structurally use DMA."""
    from ...language import memory_ops
    from ..compile_environment import CompileEnvironment
    from helion._compiler.backend import PallasBackend

    if node.target not in (memory_ops.load, memory_ops.store):
        return None
    mask_position = 2 if node.target is memory_ops.load else 3
    if len(node.args) > mask_position and node.args[mask_position] is not None:
        return None
    if (
        indirect_pos != 0
        or tensor.ndim < 3
        or not tensor.dtype.is_floating_point
        or not tensor.is_contiguous()
    ):
        return None
    index_node = subscript[indirect_pos]
    if not isinstance(index_node, torch.fx.Node):
        return None
    index = index_node.meta.get("val")
    if (
        index_node.target is not memory_ops.load
        or not isinstance(index, torch.Tensor)
        or index.ndim != 1
        or index.dtype != torch.int32
        or (len(index_node.args) > 2 and index_node.args[2] is not None)
    ):
        return None

    env = CompileEnvironment.current()
    if not env.settings.static_shapes:
        return None
    block_id = env.resolve_block_id(index.shape[0])
    if block_id is None or _metadata_dma_block_id(index_node) != block_id:
        return None
    backend = env.backend
    assert isinstance(backend, PallasBackend)
    if len(subscript) > tensor.ndim:
        return None
    selected_extents: list[int] = []
    for tensor_dim in range(1, tensor.ndim):
        item = subscript[tensor_dim] if tensor_dim < len(subscript) else slice(None)
        value = item.meta.get("val") if isinstance(item, torch.fx.Node) else item
        if not isinstance(value, slice) or value.step not in (None, 1):
            return None
        dim_size = env.try_concretize_symint(tensor.shape[tensor_dim])
        start = 0 if value.start is None else value.start
        stop = dim_size if value.stop is None else value.stop
        if (
            not isinstance(dim_size, int)
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or not 0 <= start <= stop <= dim_size
        ):
            return None
        alignment = backend._get_pallas_required_alignment(
            tensor.ndim - tensor_dim - 1,
            tensor.ndim,
            tensor.dtype.itemsize * 8,
        )
        if start % alignment != 0:
            return None
        selected_extents.append(stop - start)
    if any(extent <= 0 for extent in selected_extents):
        return None

    value = node.meta.get("val")
    if node.target is memory_ops.store:
        value_arg = node.args[2]
        value = (
            value_arg.meta.get("val")
            if isinstance(value_arg, torch.fx.Node)
            else value_arg
        )
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != tensor.ndim
        or env.resolve_block_id(value.shape[0]) != block_id
        or any(
            env.try_concretize_symint(value.shape[dim]) != extent
            for dim, extent in enumerate(selected_extents, start=1)
        )
    ):
        return None
    last_dim = env.try_concretize_symint(value.shape[-1])
    if not isinstance(last_dim, int) or last_dim <= 0 or last_dim % 128 != 0:
        return None
    return block_id


def _dma_group_candidate(
    node: torch.fx.Node,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    indirect_pos: int,
    patterns: list[IndexingPattern],
    config: Config,
) -> DmaGroupCandidate | None:
    """Recognize a static leading-axis DMA group backed by resident metadata."""
    from ...language import memory_ops
    from .plan_tiling import TilePattern

    if dma_group_structural_block_id(node, tensor, subscript, indirect_pos) is None:
        return None
    # BlockSpecs normally apply TilePatterns before the body sees a Ref. Manual
    # DMA uses the raw HBM Ref and would need absolute, tail-safe pl.ds() slices.
    # Keep data members untiled; the metadata TilePattern below still defines
    # how many indirect rows form the group.
    if any(isinstance(pattern, TilePattern) for pattern in patterns):
        return None
    index_node = subscript[indirect_pos]
    assert isinstance(index_node, torch.fx.Node)
    index = index_node.meta.get("val")
    assert isinstance(index, torch.Tensor)

    group_block_id = _metadata_dma_block_id(index_node)
    assert group_block_id is not None
    group_count = _concrete_size(index.shape[0], config)
    value = node.meta.get("val")
    if node.target is memory_ops.store:
        value_arg = node.args[2]
        value = (
            value_arg.meta.get("val")
            if isinstance(value_arg, torch.fx.Node)
            else value_arg
        )
    if not isinstance(value, torch.Tensor):
        return None
    transfer_shape = tuple(_concrete_size(size, config) for size in value.shape)
    if (
        group_count is None
        or group_count <= 0
        or None in transfer_shape
        or not transfer_shape
        or transfer_shape[0] != group_count
    ):
        return None
    concrete_shape = tuple(size for size in transfer_shape if size is not None)
    if any(size <= 0 for size in concrete_shape) or concrete_shape[-1] % 128 != 0:
        return None

    return DmaGroupCandidate(
        index_node=index_node,
        index_block_id=group_block_id,
        group_count=group_count,
        transfer_shape=concrete_shape,
    )


def build_scatter_plan(
    node: torch.fx.Node,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    indirect_positions: list[int],
    patterns: list[IndexingPattern],
    config: Config,
    has_extra_mask: bool,
) -> ScatterPlan:
    """Validate a Pallas scatter site and return its one-hot plan."""
    from ..compile_environment import CompileEnvironment
    from .plan_tiling import resident_block_elements

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
        node,
        tensor,
        subscript,
        indirect_pos,
        patterns,
        config,
    )
    if (
        config.get("pallas_indirect_access_mode", "one_hot") == "dma"
        and node.meta.get(DMA_GROUP_CAPABLE_META, False)
        and dma_group is None
    ):
        from ...exc import InvalidConfig

        raise InvalidConfig(
            "pallas_indirect_access_mode='dma' is not legal for this "
            "block-size configuration"
        )
    elements = resident_block_elements(tensor, patterns, config)
    resident_block_bytes = (
        elements * tensor.dtype.itemsize if elements is not None else None
    )
    if dma_group is None:
        _check_resident_block_size(resident_block_bytes)
    return ScatterPlan(
        indirect_pos=indirect_pos,
        jnp_dtype=jnp_dtype,
        target_ndim=tensor.ndim,
        index_ndim=index_ndim,
        dma_group=dma_group,
        resident_block_bytes=resident_block_bytes,
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
    if (
        state.fx_node is not None
        and state.fx_node.meta.get(DMA_GROUP_CAPABLE_META, False)
        and state.config.get("pallas_indirect_access_mode", "one_hot") == "dma"
    ):
        from ...exc import InvalidConfig

        raise InvalidConfig(
            "pallas_indirect_access_mode='dma' was not admitted by the active schedule"
        )
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
    if (
        state.fx_node is not None
        and state.fx_node.meta.get(DMA_GROUP_CAPABLE_META, False)
        and state.config.get("pallas_indirect_access_mode", "one_hot") == "dma"
    ):
        from ...exc import InvalidConfig

        raise InvalidConfig(
            "pallas_indirect_access_mode='dma' was not admitted by the active schedule"
        )
    _check_resident_block_size(plan.resident_block_bytes)

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
