"""Triton-backend codegen for ops defined in ``helion.language.memory_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``memory_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
import itertools
import logging
import math
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import torch

from ..._compat import supports_tensor_descriptor
from ...language import _decorators
from ...language.memory_ops import PREFETCH_SLOT_META
from ...language.memory_ops import _maybe_materialize_tile_index_load
from ...language.memory_ops import load
from ...language.memory_ops import prefetch
from ...language.memory_ops import prefetch_is_bulk
from ...language.memory_ops import store
from ..ast_extension import statement_from_string
from ..compile_environment import CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE
from ..compile_environment import CompileEnvironment
from ..cross_loop_codegen import TaggedPeerFallback
from ..cross_loop_codegen import cta_threads
from ..device_function import SymbolArgument
from ..host_function import HostFunction
from ..indexing_strategy import SubscriptIndexing
from ..indexing_strategy import _has_active_codegen_block
from ..indexing_strategy import _resolve_codegen_block_id
from ..indexing_strategy import is_block_size_index
from .distributed_ops import inband_load_codegen
from .distributed_ops import inband_store_codegen
from .distributed_ops import tagged_access
from .distributed_ops import tagged_store_codegen
from .packed_mxfp8 import flat_output_descriptor_store
from .prefetch import TENSOR_MAP_ACQUIRE_ASM
from .prefetch import prefetch_asm
from .prefetch import tensor_prefetch_asm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_function import DeviceFunction
    from ..inductor_lowering import CodegenState
    from .gluon_ops import SharedValue

log = logging.getLogger(__name__)


# Map short config names to full Triton API names for eviction policies
_EVICTION_POLICY_MAP = {
    "": None,
    "first": "evict_first",
    "last": "evict_last",
}


@_decorators.codegen(store, "triton")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    value = state.ast_arg(2)
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, torch.Tensor):
        device_fn = state.device_function
        fx_node = state.fx_node
        assert fx_node is not None
        epilogue_subtile_group_id = fx_node.meta.get("epilogue_subtile_group_id")
        if epilogue_subtile_group_id is None:
            indexing_idx = device_fn.allocate_store_index()
        elif fx_node.meta.get("epilogue_subtile_primary_output", False):
            indexing_idx = device_fn.allocate_store_index()
            device_fn.epilogue_subtile_store_indices[epilogue_subtile_group_id] = (
                indexing_idx
            )
        else:
            indexing_idx = device_fn.epilogue_subtile_store_indices[
                epilogue_subtile_group_id
            ]
        strategy = device_fn.get_indexing_strategy(indexing_idx)
        cache_modifier = None
        if state.codegen.on_device:
            modifier_idx = device_fn.device_store_cache_modifier_index
            device_fn.device_store_cache_modifier_index += 1
            modifiers = state.config.store_cache_modifiers
            if modifier_idx < len(modifiers) and modifiers[modifier_idx]:
                cache_modifier = ast.Constant(value=modifiers[modifier_idx])

        strategy_store = strategy.codegen_store
        codegen_store = tagged_store_codegen(
            state, inband_store_codegen(state, strategy_store)
        )
        if (
            codegen_store is strategy_store
            and state.codegen.store_transform is None
            and cache_modifier is None
            and epilogue_subtile_group_id is None
            and (flat_store := flat_output_descriptor_store(state, tensor, value))
            is not None
        ):
            return flat_store
        if state.codegen.store_transform is not None:
            return state.codegen.store_transform(
                state,
                tensor,
                [*subscript],
                value,
                extra_mask,
                cache_modifier,
                codegen_store,
            )

        return codegen_store(
            state, tensor, [*subscript], value, extra_mask, cache_modifier
        )
    if isinstance(tensor, tuple):
        from ..indexing_strategy import StackIndexingStrategy

        # Fusion is not supported for stack stores (multi-tensor device pointers);
        # fall through to the unfused path regardless of store_transform.
        device_fn = state.device_function
        device_fn.allocate_store_index()
        cache_modifier = None
        if state.codegen.on_device:
            modifier_idx = device_fn.device_store_cache_modifier_index
            device_fn.device_store_cache_modifier_index += 1
            modifiers = state.config.store_cache_modifiers
            if modifier_idx < len(modifiers) and modifiers[modifier_idx]:
                cache_modifier = ast.Constant(value=modifiers[modifier_idx])
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        _tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_store(
            state, tensor, dev_ptrs_ast, [*subscript], value, extra_mask, cache_modifier
        )
    raise NotImplementedError(f"Cannot store to type: {type(tensor)}")


@_decorators.codegen(load, "triton")
def _(state: CodegenState) -> ast.AST | SharedValue:
    # A store or an atomic to this tensor earlier in the same loop body followed
    # by this load is a cross-thread read-after-write on global memory; emit a
    # barrier so the write is visible before the read (see
    # mark_intra_loop_raw_barriers).
    from ..loop_dependency_checker import emit_intra_loop_raw_barrier

    emit_intra_loop_raw_barrier(state)

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    eviction_policy = state.ast_args[3] if len(state.ast_args) > 3 else None

    device_fn = state.device_function

    # If no explicit eviction_policy and we're in device code, use tunable.
    # Only such loads own a load_eviction_policies slot (see
    # _load_needs_eviction_tunable).
    if eviction_policy is None and state.codegen.on_device:
        load_idx = device_fn.device_load_index
        device_fn.device_load_index += 1
        policies = state.config.load_eviction_policies
        if isinstance(policies, str):
            policy_value = policies
        elif load_idx < len(policies):
            policy_value = policies[load_idx]
        else:
            policy_value = None
        if policy_value is not None:
            eviction_policy = _EVICTION_POLICY_MAP.get(policy_value, policy_value)

    if eviction_policy is not None:
        assert isinstance(eviction_policy, str)
        eviction_policy = ast.Constant(value=eviction_policy)

    cache_modifier = None
    if state.codegen.on_device:
        modifier_idx = device_fn.device_load_cache_modifier_index
        device_fn.device_load_cache_modifier_index += 1
        modifiers = state.config.load_cache_modifiers
        if modifier_idx < len(modifiers) and modifiers[modifier_idx]:
            cache_modifier = ast.Constant(value=modifiers[modifier_idx])

    if isinstance(tensor, torch.Tensor):
        tile_index_result = _maybe_materialize_tile_index_load(state, tensor, subscript)
        if tile_index_result is not None:
            return tile_index_result

        # Use the shared memory op index for indexing strategy
        indexing_idx = device_fn.device_memory_op_index
        device_fn.device_memory_op_index += 1
        strategy = device_fn.get_indexing_strategy(indexing_idx)

        codegen_load = inband_load_codegen(state, strategy.codegen_load)
        # A ring slot replaces a plain load (not an in-band poll or a fusion).
        if (
            device_fn.ring is not None
            and codegen_load == strategy.codegen_load
            and state.codegen.load_transform is None
            and (
                ring_slot := device_fn.ring.codegen_load(
                    state, tensor, [*subscript], extra_mask, strategy
                )
            )
        ):
            return ring_slot
        if (access := tagged_access(state)) is not None:
            # Only a ring producer reads a tagged allocation (its mirror).
            raise TaggedPeerFallback(
                f"T6: {access.tensor_name}: a consumer load that is not a ring operand"
            )
        if state.codegen.load_transform is not None:
            return state.codegen.load_transform(
                state,
                tensor,
                [*subscript],
                extra_mask,
                eviction_policy,
                cache_modifier,
                codegen_load,
            )

        value = codegen_load(
            state, tensor, [*subscript], extra_mask, eviction_policy, cache_modifier
        )
        if CompileEnvironment.current().backend_name == "gluon" and state.fx_node:
            from .gluon_ops import plan_load  # Gluon needs Triton >= 3.8: gluon only

            plan_load(device_fn, state.fx_node, value)
        return value
    if isinstance(tensor, tuple):
        from ..indexing_strategy import StackIndexingStrategy

        # Fusion is not supported for stack loads (multi-tensor device pointers);
        # fall through to the unfused path regardless of load_transform.
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        return StackIndexingStrategy.codegen_load(
            state,
            tensor,
            dev_ptrs_ast,
            [*subscript],
            extra_mask,
            eviction_policy,
            cache_modifier,
        )
    raise NotImplementedError(f"Unsupported tensor type: {type(tensor)}")


class PrefetchBox(NamedTuple):
    """A tensor-form ``hl.prefetch``: the TMA box, the boxes per dim covering
    the subscript's extents, and the first box's coordinates (innermost dim
    first)."""

    box: list[int]
    counts: list[int]
    coords: list[str]


def tensor_prefetch_possible(
    tensor: torch.Tensor, index: Sequence[object], has_extra_mask: bool
) -> bool:
    """Whether ``hl.prefetch(tensor, index)`` can take the tensor form under
    some config (``prefetch_indexing`` "tensor_descriptor";
    ``Backend.prefetch_box_possible``):
    a bulk prefetch (``prefetch_is_bulk``: sm_90+, no extra mask, a stride-1
    last dim indexed by a full slice or a tile) with tensor descriptors and the
    native CUDA caching allocator (the maps' memory pool,
    ``tensor_maps.prefetch_map``: cudaMallocAsync has no private pools), of a
    tensor whose sizes and strides are static and within the TMA encoder's
    limits (16-byte strides), indexed by scalars (ints, symbols or loaded
    scalars), tiles and full slices, with a base the bound kernel can key
    (``CompileEnvironment.tensor_descriptor_alignment_source``: an input, or
    a view at its start) and, at a full last-dim slice, a row of boxes
    ``_tensor_prefetch_box`` accepts whatever the config.  The config decides
    the box."""
    env = CompileEnvironment.current()
    if (
        not prefetch_is_bulk(tensor, index, has_extra_mask)
        or not supports_tensor_descriptor()
        or torch.cuda.memory.get_allocator_backend() != "native"
        or not 1 <= tensor.ndim <= 5
        or tensor.element_size() not in _TMA_ELEMENT_SIZES
    ):
        return False
    for s in index:
        if isinstance(s, slice):
            if s != slice(None):
                return False
        elif not is_block_size_index(s) and not (
            isinstance(s, (int, torch.SymInt))
            or (isinstance(s, torch.Tensor) and s.ndim == 0)
        ):
            return False
    # Strides every call has (a dynamic-shape kernel's traced stride may be
    # only the first call's): the map is encoded per call from the real ones.
    sizes = [env.static_kernel_tensor_extent(n) for n in tensor.shape]
    strides = [env.known_stride(tensor, d) for d in range(tensor.ndim)]
    if any(n is None or n == 0 or n >= _TMA_MAX_DIM for n in sizes):
        return False
    size = tensor.element_size()
    if not all(
        st is not None and (st * size) % 16 == 0 and st * size < _TMA_MAX_STRIDE
        for st in strides[:-1]
    ):
        return False
    if isinstance(index[-1], slice):
        # The boxes a full last-dim slice takes whatever the config.
        row = min(cast("int", sizes[-1]), CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE)
        if (row * size) % 16 or -(-cast("int", sizes[-1]) // row) > _MAX_BOXES:
            return False
    return env.tensor_descriptor_alignment_source(tensor) is not None


def _tensor_prefetch_box(
    state: CodegenState, tensor: torch.Tensor, subscript: list[object]
) -> PrefetchBox | None:
    """The TMA boxes of a tensor-form ``hl.prefetch`` (``tensor_prefetch_possible``
    holds): the tile the subscript describes under the config.  A scalar index
    is extent 1 at its value; a tile its block size (at most the dim) at the
    tile's offset; a full slice its whole extent at 0, or a rolled reduction's
    chunk at the chunk's offset.  An extent longer than a box dim, or a box over the
    encoder's byte limit (the outermost dim halved until it fits), is covered
    by consecutive boxes, at most ``_MAX_BOXES``; None past that, for box rows
    not a multiple of 16 bytes, or for a tile or chunk with no offset of its
    own here (flattened, or its loop not active)."""
    env = CompileEnvironment.current()
    config = state.device_function.config
    sizes = [cast("int", env.static_kernel_tensor_extent(n)) for n in tensor.shape]
    extents: list[int] = []
    offsets: dict[int, str] = {}
    for d, s in enumerate(subscript):
        if isinstance(s, slice):
            if d == tensor.ndim - 1 or sizes[d] == 1:
                # The bulk form's row segment, or a unit dim (a slice of
                # extent 1 allocates no reduction dim): no rdim.
                extents.append(sizes[d])
                continue
            rdim = env.allocate_reduction_dimension(tensor.size(d))
            chunk = rdim.from_config(config)
            if not isinstance(chunk, int):
                return None
            extents.append(min(chunk, sizes[d]))
            if chunk < sizes[d]:
                if (offset := _active_offset(state, rdim.block_id)) is None:
                    return None
                offsets[d] = offset
        elif is_block_size_index(s):
            assert isinstance(s, torch.SymInt)
            block_id = env.get_block_id(s)
            assert block_id is not None
            info = env.block_sizes[block_id]
            block = info.from_config(config)
            if not isinstance(block, int) or info.is_flattened(config):
                return None
            if (offset := _active_offset(state, block_id)) is None:
                return None
            extents.append(min(block, sizes[d]))
            offsets[d] = offset
        else:
            extents.append(1)
    box = [min(n, CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE) for n in extents]
    counts = [-(-n // b) for n, b in zip(extents, box, strict=True)]
    size = tensor.element_size()
    while math.prod(box) * size > _MAX_BOX_BYTES:
        dim = next((d for d, n in enumerate(box[:-1]) if n > 1), None)
        if dim is None:
            return None
        box[dim] = -(-box[dim] // 2)
        counts[dim] = -(-extents[dim] // box[dim])
    if (box[-1] * size) % 16 or math.prod(counts) > _MAX_BOXES:
        return None
    # The scalar indices' values; the tiles' and chunks' offsets; 0 elsewhere.
    per_dim = SubscriptIndexing.compute_per_dim_indexing(
        state,
        tensor,
        [s if _is_scalar_index(s) else 0 for s in subscript],
    )
    coords = [
        f"tl.cast({offsets[d]}, tl.int32)"
        if d in offsets
        else f"tl.cast({per_dim.dim_index_exprs[d]}, tl.int32)"
        if _is_scalar_index(subscript[d])
        else "0"
        for d in reversed(range(tensor.ndim))
    ]
    return PrefetchBox(box, counts, coords)


def _active_offset(state: CodegenState, block_id: int) -> str | None:
    """The offset of the current tile (or reduction chunk) of ``block_id``,
    None outside its loop."""
    block_id = _resolve_codegen_block_id(state, block_id)
    if not _has_active_codegen_block(state, block_id):
        return None
    return state.codegen.offset_var(block_id)


def _is_scalar_index(index: object) -> bool:
    return not isinstance(index, slice) and not is_block_size_index(index)


# One prefetch issues at most this many boxes (a box spans at most
# CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE elements per dim; more keep the
# pointer form).
_MAX_BOXES = 16
# cuTensorMapEncodeTiled rejects a box larger than the device's shared memory
# (B200: past 228 KB); this stays below every sm_90+ part's.
_MAX_BOX_BYTES = 96 * 1024
# cuTensorMapEncodeTiled: element sizes of its integer types and byte strides it
# accepts.  Dims stay below 2**31: the prefetch's box coordinates are signed
# 32-bit (the encoder itself accepts up to 2**32).
_TMA_ELEMENT_SIZES = (1, 2, 4, 8)
_TMA_MAX_DIM = 1 << 31
_TMA_MAX_STRIDE = 1 << 40


def _prefetch_row_segment(
    state: CodegenState, tensor: torch.Tensor, last: object
) -> tuple[str, str] | None:
    """(first element, element count) of the row segment a bulk prefetch's
    last index covers, or None when its tile has no block size variable."""
    size = state.device_function.tensor_size(tensor, tensor.ndim - 1).name
    if isinstance(last, slice):
        return "0", size
    assert isinstance(last, torch.SymInt)
    block_id = CompileEnvironment.current().get_block_id(last)
    assert block_id is not None
    block_id = _resolve_codegen_block_id(state, block_id)
    block = state.device_function.block_size_var(block_id)
    if block is None:
        return None
    start = state.codegen.offset_var(block_id)
    if state.codegen.mask_var(block_id) is None:
        return start, block
    return start, f"tl.minimum({block}, {size} - {start})"


def _prefetch_map_arg(
    device_function: DeviceFunction, fake_value: torch.Tensor, box: list[int]
) -> str:
    """The address of a tensor map (TMA descriptor) in global memory of
    ``fake_value`` with box ``box``, for a tensor-form L2 prefetch: a kernel
    argument (``tensor_maps.prefetch_map`` builds the map) acquired once in the
    preamble."""
    key = (fake_value, tuple(box))
    cache = device_function.prefetch_map_args
    if key not in cache:
        origin = HostFunction.current().tensor_to_origin[fake_value]
        name = device_function.new_var(origin.suggest_var_name() + "_pf_map")
        # A host local holds the map until the launch is submitted.
        held = device_function.host_constexpr_def(
            name, f"helion_tensor_maps.prefetch_map({origin.host_str()}, {box!r})"
        )
        device_function.arguments.append(SymbolArgument(name, f"{held}.address"))
        acquired = device_function.new_var(name + "_acquired")
        device_function.preamble.append(
            statement_from_string(
                f"{acquired} = tl.inline_asm_elementwise("
                f"{TENSOR_MAP_ACQUIRE_ASM!r}, '=l,l', [{name}], "
                "dtype=tl.int64, is_pure=True, pack=1)"
            )
        )
        cache[key] = acquired
    return cache[key]


@_decorators.codegen(prefetch, "triton")
def _(state: CodegenState) -> None:
    env = CompileEnvironment.current()
    capability = env.ptx_capability
    if capability is None:
        return None  # both prefetch instructions are PTX
    device_fn = state.device_function
    assert state.fx_node is not None
    form = device_fn.prefetch_indexing(state.fx_node.meta[PREFETCH_SLOT_META])
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    tensor_form = None
    if form == "tensor_descriptor" and env.backend.prefetch_box_possible(
        tensor, [*subscript], extra_mask is not None
    ):
        # The bound kernel keys the base's alignment: a misaligned one (or
        # one no snapshot covers) keeps the pointer form.
        if env.prefetch_base_is_aligned(tensor):
            tensor_form = _tensor_prefetch_box(state, tensor, [*subscript])
        if tensor_form is None:
            log.debug("hl.prefetch: no TMA box for this call, pointer form")
    if tensor_form is not None:
        # TMA box prefetches from the tensor map: one per box covering the
        # subscript's extents.
        box, counts, coords = tensor_form
        tensor_map = _prefetch_map_arg(device_fn, tensor, box)
        # One warp of the launch's (``cta_threads``: Triton raises num_warps
        # for a warp-specialized loop) issues each box.
        asm = tensor_prefetch_asm(tensor.ndim, cta_threads(device_fn) // 32)
        constraints = "=r,l" + ",r" * tensor.ndim
        log.debug(
            "hl.prefetch(%s): tensor form, box %s x %d",
            tensor_map,
            box,
            math.prod(counts),
        )
        # ``coords`` is innermost dim first.
        for chunk in itertools.product(*(range(c) for c in reversed(counts))):
            moved = [
                c if k == 0 else f"({c}) + {k * b}"
                for c, k, b in zip(coords, chunk, reversed(box), strict=True)
            ]
            state.add_statement(
                statement_from_string(
                    f"tl.inline_asm_elementwise({asm!r}, {constraints!r}, "
                    f"[{tensor_map}, {', '.join(moved)}], "
                    "dtype=tl.int32, is_pure=False, pack=1)"
                )
            )
        return None
    name = state.device_function.tensor_arg(tensor).name
    segment = None
    if prefetch_is_bulk(tensor, [*subscript], extra_mask is not None):
        segment = _prefetch_row_segment(state, tensor, subscript[-1])
    log.debug(
        "hl.prefetch(%s): %s",
        name,
        "bulk row segments" if segment is not None else "per-element lines",
    )
    if segment is not None:
        # One bulk prefetch per element of the leading indices.
        start, count = segment
        index = [*subscript[:-1], 0]
        indexing = SubscriptIndexing.create(state, tensor, index)
        nbytes = f"({count}) * {tensor.element_size()}"
        operand = f"tl.where({{mask}}, {nbytes}, 0)" if indexing.has_mask() else nbytes
        pointer = f"{name} + ({{offset}} + {start})"
    else:
        index = [*subscript]
        indexing = SubscriptIndexing.create(state, tensor, index, extra_mask)
        operand = "{mask}" if indexing.has_mask() else "1"
        pointer = f"{name} + {{offset}}"
    scalar = extra_mask is None and not SubscriptIndexing.compute_shape(
        tensor, index, state
    )
    asm = prefetch_asm(
        bulk=segment is not None, scalar=scalar, elect=capability >= (9, 0)
    )
    state.add_statement(
        statement_from_string(
            f"tl.inline_asm_elementwise({asm!r}, '=r,l,r', "
            f"[{pointer}, tl.cast({operand}, tl.int32)], "
            "dtype=tl.int32, is_pure=False, pack=1)",
            offset=indexing.index_expr,
            mask=indexing.mask_expr,
        )
    )
    return None
