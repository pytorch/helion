"""CuTe-backend codegen for ops defined in ``helion.language.memory_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``memory_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import cast

import sympy
import torch
from torch._dynamo.source import LocalSource
from torch._subclasses import FakeTensor
from torch.fx.node import map_arg

from ... import exc
from ...language import _decorators
from ...language.atomic_ops import ATOMIC_OPS
from ...language.memory_ops import _CUTE_CACHE_LOAD_HELPERS
from ...language.memory_ops import _CUTE_VECTOR_DTYPES
from ...language.memory_ops import _CUTE_VECTOR_MAX_BYTES
from ...language.memory_ops import _CUTE_VECTOR_UNROLL_CARRIER
from ...language.memory_ops import _CUTE_VECTOR_UNROLL_DTYPES
from ...language.memory_ops import CuteTileVecStoreSite
from ...language.memory_ops import _codegen_cute_store_reshape_lane_loops
from ...language.memory_ops import _codegen_cute_store_tcgen05_tile
from ...language.memory_ops import _cute_access_regions
from ...language.memory_ops import _cute_active_index_var
from ...language.memory_ops import _cute_active_mask_var
from ...language.memory_ops import _cute_combined_mask
from ...language.memory_ops import _cute_index_exprs
from ...language.memory_ops import _cute_index_tuple
from ...language.memory_ops import _cute_is_unroll_dtype
from ...language.memory_ops import _cute_lane_axis_pos
from ...language.memory_ops import _cute_register_reduction_unroll_vec_store
from ...language.memory_ops import _cute_register_tile_unroll_vec_hoist
from ...language.memory_ops import _cute_register_tile_unroll_vec_store
from ...language.memory_ops import _cute_scalar_load_expr
from ...language.memory_ops import _cute_scalar_pointer_expr
from ...language.memory_ops import _cute_tag_access_regions
from ...language.memory_ops import _cute_tensor_dim_size_expr
from ...language.memory_ops import _cute_unique_graph_block_id
from ...language.memory_ops import _cute_unroll_vec_load_expr
from ...language.memory_ops import _matching_block_ids
from ...language.memory_ops import _maybe_codegen_cute_packed_affine_lhs_load
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import ReadWrites
from ..compile_environment import CompileEnvironment
from ..compile_environment import RuntimeInputSpecialization
from ..compile_environment import _replay_tensor_input_source
from ..compile_environment import _to_sympy
from ..indexing_strategy import _get_tile_with_offset_info
from .cute_epilogue import _ZERO_ARG_TARGETS
from .cute_epilogue import analyze_tcgen05_unary_epilogue_chain
from .cute_fx_walk import reach_tcgen05_matmul_anchors
from .cute_reshape import check_memory_mask_rebound
from .cute_reshape import codegen_cute_store_rebound_value
from .cute_reshape import describe_rebound_block_dims
from .cute_reshape import run_deferred_rebound_checks
from .cute_reshape import store_rebound_dims
from .cute_reshape import tcgen05_rebound_store_error
from .indexing import CUTE_SCALAR_LOAD_SITE_META
from .indexing import CuteScalarLoadSite
from .indexing import is_cute_direct_iota_index
from .indexing import is_cute_unit_stride_iota_index
from .indexing import match_cute_shifted_tile_index

if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Sequence

    from torch._guards import Source

    from ..device_function import DeviceFunction
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState
    from ..reduction_strategy import LoopedReductionStrategy
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import DeviceLoopOrGridState
    from ..tile_strategy import PerThreadFlattenedTileStrategy
    from ..tile_strategy import PerThreadNDTileStrategy

    CuteLaneTileStrategy = PerThreadNDTileStrategy | PerThreadFlattenedTileStrategy

log = logging.getLogger(__name__)


def _persistent_vec_alignment_signature(values: Sequence[object]) -> Hashable:
    """Cache-key facts needed by persistent vector alignment/extent checks."""
    if (
        len(values) != 1
        or not isinstance(values[0], torch.Tensor)
        or isinstance(values[0], FakeTensor)
    ):
        return None
    tensor = values[0]
    element_size = tensor.element_size()
    max_vector_elements = max(_CUTE_VECTOR_MAX_BYTES // element_size, 1)
    return (
        int(tensor.data_ptr()) % _CUTE_VECTOR_MAX_BYTES,
        tuple(int(size) % max_vector_elements for size in tensor.shape),
        tuple(
            (
                int(stride) == 1,
                (int(stride) * element_size) % _CUTE_VECTOR_MAX_BYTES,
            )
            for stride in tensor.stride()
        ),
    )


_PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY = "cute_persistent_vec_alignment_matrix_v2"


def _persistent_vec_alignment_matrix_signature(
    values: Sequence[object],
) -> Hashable:
    return tuple(_persistent_vec_alignment_signature((value,)) for value in values)


def register_persistent_vec_alignment_specializations(
    env: CompileEnvironment,
) -> None:
    """Specialize vector-packet candidates on runtime pointer alignment.

    Dynamic FakeTensors intentionally carry symbolic storage offsets.  The
    branch-local vectorizer therefore consults the real input tensor during
    codegen, and the tile-packet admission reads the bound residues.  Register
    the address/stride residue of every input whose dtype has a packet
    lowering (explicit vectors, unroll carriers and byte packets) in the bound
    kernel cache key before any config is compiled, so a later unaligned view
    can never reuse code emitted for an aligned tensor.
    """
    sources = tuple(
        source
        for tensor in env.input_sources
        if (tensor.dtype in _CUTE_VECTOR_DTYPES or _cute_is_unroll_dtype(tensor.dtype))
        and (source := env.tensor_input_source(tensor)) is not None
    )
    if not sources:
        return
    env.register_runtime_input_specialization(
        _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY,
        RuntimeInputSpecialization(
            sources=sources,
            classifier_identity=(
                "byte_alignment_matrix_v2",
                _CUTE_VECTOR_MAX_BYTES,
                tuple(map(repr, sources)),
            ),
            classifier=_persistent_vec_alignment_matrix_signature,
            reusable_tensor_properties=frozenset(("data_ptr",)),
        ),
    )


def runtime_tensor_has_specialized_alignment(
    env: CompileEnvironment,
    tensor: torch.Tensor,
    required_alignment: int,
) -> bool:
    """Return a cache-key-backed runtime base-pointer alignment proof."""
    if required_alignment <= 0 or _CUTE_VECTOR_MAX_BYTES % required_alignment:
        return False
    runtime_tensor = env.runtime_value_for_tensor(tensor)
    if not isinstance(runtime_tensor, torch.Tensor) or isinstance(
        runtime_tensor, FakeTensor
    ):
        return False
    source = env.tensor_input_source(tensor)
    specialization = env.runtime_input_specializations.get(
        _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
    )
    if source is None or specialization is None or source not in specialization.sources:
        return False
    runtime_values = tuple(
        _replay_tensor_input_source(item, env.runtime_arg_values_by_name)
        for item in specialization.sources
    )
    facts = specialization.classifier(runtime_values)
    return (
        env.runtime_input_specialization_matches_bound(
            _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY,
            facts,
        )
        and int(runtime_tensor.data_ptr()) % required_alignment == 0
    )


def tensor_has_specialized_tma_alignment(
    env: CompileEnvironment,
    tensor: torch.Tensor,
) -> bool:
    """Read cache-key-backed base/outer-stride alignment for a TensorMap.

    Use the immutable binding facts: code generation may run after the weakly
    held example inputs have expired. The dispatch cache checks these same
    facts before reusing this specialization with later tensors. Physical
    alignment is unchanged by logical axis permutations or group indexing.
    """
    source = env.tensor_input_source(tensor)
    specialization = env.runtime_input_specializations.get(
        _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
    )
    facts = env.bound_runtime_input_specialization_results.get(
        _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
    )
    if (
        source is None
        or specialization is None
        or source not in specialization.sources
        or facts is None
    ):
        return False
    signatures = cast(
        "tuple[tuple[int, tuple[int, ...], tuple[tuple[bool, int], ...]] | None, ...]",
        facts,
    )
    signature = signatures[specialization.sources.index(source)]
    if signature is None:
        return False
    base_residue, _size_residues, stride_facts = signature
    return (
        base_residue == 0
        and sum(unit_stride for unit_stride, _residue in stride_facts) == 1
        and all(unit_stride or residue == 0 for unit_stride, residue in stride_facts)
    )


def _bound_vec_alignment_signature(
    env: CompileEnvironment, source: Source | None
) -> tuple[int, tuple[int, ...], tuple[tuple[bool, int], ...]] | None:
    """The bound alignment signature of an input source, if it was specialized.

    ``_persistent_vec_alignment_signature`` records ``(data_ptr % 16,
    size % max_vector_elements per dim, (stride == 1, stride_bytes % 16) per
    dim)``; the readers below each derive one cache-key-backed fact from it
    without retaining the example inputs.
    """
    specialization = env.runtime_input_specializations.get(
        _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
    )
    facts = env.bound_runtime_input_specialization_results.get(
        _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
    )
    if (
        source is None
        or specialization is None
        or facts is None
        or source not in specialization.sources
    ):
        return None
    signatures = cast(
        "tuple[tuple[int, tuple[int, ...], tuple[tuple[bool, int], ...]] | None, ...]",
        facts,
    )
    return signatures[specialization.sources.index(source)]


def tensor_has_specialized_base_alignment(
    env: CompileEnvironment, tensor: torch.Tensor, alignment: int
) -> bool:
    """Read a cache-key-backed pointer residue without retaining example inputs."""
    if alignment <= 0 or _CUTE_VECTOR_MAX_BYTES % alignment:
        return False
    signature = _bound_vec_alignment_signature(env, env.tensor_input_source(tensor))
    return signature is not None and signature[0] % alignment == 0


def tensor_has_specialized_dim_multiple(
    env: CompileEnvironment, tensor: torch.Tensor, dim: int, multiple: int
) -> bool:
    """Read a cache-key-backed proof that ``tensor.shape[dim] % multiple == 0``.

    The signature records every input size modulo the widest vector of the
    tensor's dtype (8 bf16/fp16 lanes, 4 fp32 lanes), so any vector width up
    to that maximum is decidable.  The bound kernel is keyed on that residue:
    a later call whose size has a different residue binds separately and
    never reuses this code.
    """
    if multiple <= 0 or dim < 0 or dim >= tensor.ndim:
        return False
    max_vector_elements = max(_CUTE_VECTOR_MAX_BYTES // tensor.dtype.itemsize, 1)
    if max_vector_elements % multiple:
        return False
    signature = _bound_vec_alignment_signature(env, env.tensor_input_source(tensor))
    return (
        signature is not None
        and dim < len(signature[1])
        and signature[1][dim] % multiple == 0
    )


def tensor_has_specialized_stride_multiple(
    env: CompileEnvironment, tensor: torch.Tensor, dim: int, multiple: int
) -> bool:
    """Read a cache-key-backed proof that ``tensor.stride(dim) % multiple == 0``.

    The signature records each stride in bytes modulo the widest vector
    transaction, so any vector width of the tensor's dtype is decidable.
    """
    if multiple <= 0 or dim < 0 or dim >= tensor.ndim:
        return False
    vector_bytes = multiple * tensor.dtype.itemsize
    if _CUTE_VECTOR_MAX_BYTES % vector_bytes:
        return False
    signature = _bound_vec_alignment_signature(env, env.tensor_input_source(tensor))
    return (
        signature is not None
        and dim < len(signature[2])
        and signature[2][dim][1] % vector_bytes == 0
    )


def cute_tensor_base_is_aligned(
    env: CompileEnvironment, tensor: torch.Tensor, alignment: int
) -> bool:
    """Whether ``tensor``'s base pointer is a multiple of ``alignment`` bytes.

    The base is proven by provenance, as for tensor descriptors: an input, or
    a statically exact view that exactly one input owns, reads that input's
    bound pointer residue plus the view's static byte offset, and a fresh
    wrapper allocation is aligned to the allocator granularity less its
    static storage offset.  Anything else (a view of two inputs' shared
    storage, say) is refused rather than trusted.
    """
    if alignment <= 0 or _CUTE_VECTOR_MAX_BYTES % alignment:
        return False
    owner = env.tensor_alignment_owner(tensor)
    if owner is not None:
        source, offset_bytes = owner
        signature = _bound_vec_alignment_signature(env, source)
        return signature is not None and (signature[0] + offset_bytes) % alignment == 0
    if env.tensor_storage_is_compiler_allocated(tensor):
        offset = tensor.storage_offset()
        return (
            isinstance(offset, int)
            and (offset * tensor.dtype.itemsize) % alignment == 0
        )
    return False


def cute_reduction_vector_layout_aligned(
    env: CompileEnvironment, tensor: torch.Tensor, lane_dim: int, vec_width: int
) -> bool:
    """Whether every V-wide chunk along ``lane_dim`` is naturally aligned.

    A chunk base is a multiple of V along the lane dim, so the packet is
    aligned exactly when the tensor's base is (``cute_tensor_base_is_aligned``)
    and every other stride is a multiple of V elements: a 4100-wide bf16 row
    is only 8-byte aligned, and an LDG.128 at the start of its second row
    faults.  Static strides are checked directly; a symbolic stride is proven
    either as a multiple of V through the size residues (the contiguous
    ``stride == extent`` case) or through the bound stride residue of an
    input tensor.  A static size-1 dim is exempt: it never contributes to an
    address, and PyTorch gives it an arbitrary stride (``x.view(s, 1)`` and
    ``torch.empty([n, 1])`` both have strides ``(1, 1)``).
    """
    if vec_width <= 1:
        return True
    if not cute_tensor_base_is_aligned(env, tensor, vec_width * tensor.dtype.itemsize):
        return False
    for dim in range(tensor.ndim):
        if dim == lane_dim:
            continue
        size = tensor.shape[dim]
        if isinstance(size, int) and size == 1:
            continue
        stride = tensor.stride(dim)
        if isinstance(stride, int):
            if stride % vec_width:
                return False
            continue
        if cute_known_multiple(env, stride, vec_width):
            continue
        if not tensor_has_specialized_stride_multiple(env, tensor, dim, vec_width):
            return False
    return True


def cute_known_multiple(env: CompileEnvironment, extent: object, multiple: int) -> bool:
    """``extent % multiple == 0`` for a static or cache-key-specialized extent.

    Static extents use the shape environment.  A symbolic extent is matched
    against the symbolic sizes of the kernel's input tensors and proven from
    their bound residues (``tensor_has_specialized_dim_multiple``); the
    dispatch cache key already separates inputs whose residues differ.
    """
    if multiple <= 0:
        return False
    if multiple == 1:
        return True
    if isinstance(extent, torch.SymInt):
        extent = extent._sympy_()
    if isinstance(extent, (int, sympy.Integer)):
        return int(extent) % multiple == 0
    if not isinstance(extent, sympy.Expr):
        return False
    target = env.shape_env.replace(extent)
    for tensor in env.input_sources:
        for dim, size in enumerate(tensor.shape):
            if not isinstance(size, torch.SymInt):
                continue
            if env.shape_env.replace(size._sympy_()) != target:
                continue
            if tensor_has_specialized_dim_multiple(env, tensor, dim, multiple):
                return True
    return False


def _tensor_storage_disjoint_matrix_signature(
    values: Sequence[object],
) -> Hashable:
    """Classify every pair while reading each tensor's storage only once."""
    spans: list[tuple[torch.device, int, int] | None] = []
    for value in values:
        if not isinstance(value, torch.Tensor) or isinstance(value, FakeTensor):
            spans.append(None)
            continue
        storage = value.untyped_storage()
        start = int(storage.data_ptr())
        spans.append((value.device, start, start + storage.nbytes()))

    result: list[bool] = []
    for left, right in itertools.combinations(spans, 2):
        if left is None or right is None:
            result.append(False)
        elif left[1] == left[2] or right[1] == right[2] or left[0] != right[0]:
            result.append(True)
        else:
            result.append(left[2] <= right[1] or right[2] <= left[1])
    return tuple(result)


def _tensor_alias_sources(env: CompileEnvironment) -> tuple[Source, ...]:
    sources: list[Source] = []
    # Enumerate direct host tensor arguments by their argument names rather
    # than only asking FakeTensor -> Source.  FakeTensorConverter deliberately
    # reuses one FakeTensor when the same real tensor is passed in two argument
    # slots.  That makes the reverse mapping ambiguous, but the two explicit
    # argument Sources remain distinct and must both participate in the
    # cache-specialized alias matrix.
    from ..host_function import HostFunction

    for name, value in HostFunction.current().params.arguments.items():
        if isinstance(value, torch.Tensor):
            sources.append(LocalSource(name, is_input=True))
    for tensor in env.input_sources:
        source = env.tensor_input_source(tensor)
        if source is not None and source not in sources:
            sources.append(source)
    return tuple(sources)


_TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY = "cute_tensor_storage_disjoint_matrix_v1"


def register_cute_tensor_alias_specializations(env: CompileEnvironment) -> None:
    """Cache-key runtime alias facts used by CuTe memory reordering passes.

    Different kernel argument names are not a non-aliasing proof: callers may
    pass the same tensor or overlapping views.  Recording the storage-overlap
    predicate in the bound-kernel key lets codegen use a positive runtime fact
    without reusing that code for a later aliasing launch.
    """
    sources = _tensor_alias_sources(env)
    if len(sources) < 2:
        return
    env.register_runtime_input_specialization(
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY,
        RuntimeInputSpecialization(
            sources=sources,
            classifier_identity=(
                "storage_span_disjoint_matrix_v1",
                tuple(map(repr, sources)),
            ),
            classifier=_tensor_storage_disjoint_matrix_signature,
            reusable_tensor_properties=frozenset(("storage_span",)),
        ),
    )


def stores_into_input_storage(
    graphs: Sequence[GraphInfo], env: CompileEnvironment
) -> bool:
    """Whether a device store or atomic targets storage owned by a kernel input.

    A store into a fresh host allocation cannot alias an input, so grid
    kernels that only write such outputs need no storage-overlap dispatch
    guards.  Writing into an external tensor (an ``out`` argument or a view
    of one) does: vector memory passes move input loads across that store
    only under a cache-specialized disjointness proof.
    """
    input_storages = {id(tensor.untyped_storage()) for tensor in env.input_sources}
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if node.op != "call_function" or (
                node.target is not store and node.target not in ATOMIC_OPS
            ):
                continue
            target = node.args[0]
            if not isinstance(target, torch.fx.Node):
                continue
            value = target.meta.get("val")
            if (
                isinstance(value, torch.Tensor)
                and id(value.untyped_storage()) in input_storages
            ):
                return True
    return False


def runtime_tensors_are_proven_disjoint(
    env: CompileEnvironment,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    allow_unbound: bool = False,
) -> bool:
    """Return a cache-specialized positive runtime storage-disjointness fact."""
    left_source = env.tensor_input_source(left)
    right_source = env.tensor_input_source(right)
    if left_source is None:
        left_source = env.tensor_storage_input_source(left)
    if right_source is None:
        right_source = env.tensor_storage_input_source(right)
    if left_source is None or right_source is None:
        return False
    return runtime_tensor_sources_are_proven_disjoint(
        env,
        left_source,
        right_source,
        allow_unbound=allow_unbound,
    )


def runtime_tensor_sources_are_proven_disjoint(
    env: CompileEnvironment,
    left_source: Source,
    right_source: Source,
    *,
    allow_unbound: bool = False,
) -> bool:
    """Return a cache-specialized storage fact for explicit input Sources.

    Search registration runs during binding, before its immutable facts are
    recorded. It may opt into live facts from the registered classifier.
    Codegen must retain the default requirement for a bound fact.
    """
    if left_source == right_source:
        return False
    specialization = env.runtime_input_specializations.get(
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
    )
    sources = _tensor_alias_sources(env)
    if (
        specialization is None
        or specialization.sources != sources
        or specialization.classifier_identity
        != ("storage_span_disjoint_matrix_v1", tuple(map(repr, sources)))
        or specialization.reusable_tensor_properties != frozenset(("storage_span",))
    ):
        return False
    # This immutable matrix was recorded from the bound kernel's dispatch key.
    # Recompilation may outlive its construction tensors, which are held only
    # weakly. Missing live arguments do not invalidate a cache-specialized fact.
    facts = env.bound_runtime_input_specialization_results.get(
        _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
    )
    runtime_values = tuple(
        _replay_tensor_input_source(source, env.runtime_arg_values_by_name)
        for source in sources
    )
    live_facts = specialization.classifier(runtime_values)
    assert isinstance(live_facts, tuple)
    if (
        allow_unbound
        and _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
        not in env.bound_runtime_input_specialization_results
    ):
        facts = live_facts
    if (
        not isinstance(facts, tuple)
        or len(facts) != len(sources) * (len(sources) - 1) // 2
        or any(type(fact) is not bool for fact in facts)
    ):
        return False
    live = tuple(
        isinstance(value, torch.Tensor) and not isinstance(value, FakeTensor)
        for value in runtime_values
    )
    # Keep rejecting contradictory live arguments, including a partial weak
    # fallback where only some construction tensors are still alive.
    if any(
        left_live and right_live and live_fact != fact
        for (left_live, right_live), live_fact, fact in zip(
            itertools.combinations(live, 2), live_facts, facts, strict=True
        )
    ):
        return False
    wanted = frozenset((left_source, right_source))
    for pair, fact in zip(itertools.combinations(sources, 2), facts, strict=True):
        if frozenset(pair) == wanted:
            return fact is True
    return False


def _log_cute_layout(state: CodegenState, op_name: str) -> None:
    """Log the CuTe layout annotation for the current node, if any.

    This is used during CuTe load/store codegen to make layout info
    visible for debugging and future codegen integration.
    """
    layout = state.cute_layout
    if layout is None:
        return
    node_name = state.fx_node.name if state.fx_node else "?"
    log.debug(
        "cute %s %s: layout tag=%s thread=%s value=%s",
        op_name,
        node_name,
        layout.tag.value,
        layout.thread_shape,
        layout.value_shape,
    )


def _maybe_codegen_cute_packed_rhs_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
) -> ast.AST | None:
    from .indexing import match_cute_duplicate_stack_reshape_rhs

    fx_node = state.fx_node
    if fx_node is None or len(subscript) not in (2, 3) or len(fx_node.users) != 1:
        return None

    user = next(iter(fx_node.users))
    if user.op != "call_function" or user.target is not torch.ops.aten.stack.default:
        return None
    stack_users = list(user.users)
    if len(stack_users) != 1 or not isinstance(stack_users[0], torch.fx.Node):
        return None
    rhs_node = stack_users[0]
    packed_rhs = match_cute_duplicate_stack_reshape_rhs(rhs_node)
    if packed_rhs != (
        fx_node,
        len(user.args[0]) if isinstance(user.args[0], (list, tuple)) else 0,
    ):
        return None

    packed_block_id = _cute_unique_graph_block_id(state)
    if packed_block_id is None:
        return None
    packed_index = _cute_active_index_var(state, packed_block_id)
    if packed_index is None:
        return None

    leading_subscript = [*subscript[:-2]]
    col_index_exprs = _cute_index_exprs(
        state,
        [subscript[-1]],
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if len(col_index_exprs) != 1:
        return None
    (col_index,) = col_index_exprs
    leading_index_exprs = _cute_index_exprs(
        state,
        leading_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if len(leading_index_exprs) != len(leading_subscript):
        return None
    tensor_name = state.device_function.tensor_arg(tensor).name
    load_index_expr = ", ".join([*leading_index_exprs, packed_index, col_index])
    load_expr: ast.AST = expr_from_string(f"{tensor_name}[{load_index_expr}]")
    mask_terms: list[str] = []
    col_mask = _cute_combined_mask(
        state,
        [*leading_subscript, subscript[-1]],
        extra_mask,
        tensor=tensor,
    )
    if col_mask is not None:
        mask_terms.append(col_mask)
    if packed_mask := _cute_active_mask_var(state, packed_block_id):
        mask_terms.append(f"({packed_mask})")
    if not mask_terms:
        return load_expr
    zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    return expr_from_string(
        f"({{value}} if {' and '.join(mask_terms)} else {zero}(0))",
        value=load_expr,
    )


def _cute_scalar_storage_dtype(dtype: torch.dtype) -> str:
    if dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
        return "cutlass.Uint8"
    return CompileEnvironment.current().backend.dtype_str(dtype)


_PERSISTENT_BRANCH_VEC_LOAD = "_helion_persistent_branch_vec_load"
_PERSISTENT_BRANCH_VEC_STORE = "_helion_persistent_branch_vec_store"


def _persistent_branch_vec_load_marker(
    block_id: int,
    vec_width: int,
    dtype: torch.dtype,
    eviction_suffix: str,
    pointer: str,
    scalar_value: ast.expr,
) -> ast.expr:
    """Preserve a proven vec candidate until its branch-local loop exists."""
    pointer_expr = expr_from_string(pointer)
    assert isinstance(pointer_expr, ast.expr)
    return ast.fix_missing_locations(
        ast.Call(
            func=ast.Name(id=_PERSISTENT_BRANCH_VEC_LOAD, ctx=ast.Load()),
            args=[
                ast.Constant(value=block_id),
                ast.Constant(value=vec_width),
                ast.Constant(value=_cute_scalar_storage_dtype(dtype)),
                ast.Constant(value=eviction_suffix),
                pointer_expr,
                scalar_value,
            ],
            keywords=[],
        )
    )


def _persistent_branch_vec_store_marker(
    block_id: int,
    vec_width: int,
    dtype: torch.dtype,
    pointer: str,
    value: ast.expr,
    mask_expr: str | None,
) -> ast.stmt:
    """Emit a removable marker for a branch-local exact-fragment store."""
    pointer_expr = expr_from_string(pointer)
    assert isinstance(pointer_expr, ast.expr)
    mask = expr_from_string(mask_expr) if mask_expr is not None else ast.Constant(None)
    assert isinstance(mask, ast.expr)
    return ast.fix_missing_locations(
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id=_PERSISTENT_BRANCH_VEC_STORE, ctx=ast.Load()),
                args=[
                    ast.Constant(value=block_id),
                    ast.Constant(value=vec_width),
                    ast.Constant(value=_cute_scalar_storage_dtype(dtype)),
                    pointer_expr,
                    value,
                    mask,
                ],
                keywords=[],
            )
        )
    )


def _cute_scalar_store_expr(
    tensor_name: str, index_exprs: list[str], value: str
) -> str:
    if "None" in index_exprs:
        return f"{tensor_name}.__setitem__({_cute_index_tuple(index_exprs)}, {value})"
    return f"{_cute_scalar_pointer_expr(tensor_name, index_exprs)}.store({value})"


_CUTE_EVICTION_POLICY_MAP = {
    "": "",
    "first": "evict_first",
    "last": "evict_last",
}


def _cute_vector_load_expr(
    tensor_name: str,
    index_exprs: list[str],
    dtype: torch.dtype,
    *,
    vec_width: int,
    eviction_suffix: str = "",
) -> str:
    elem_str, _ = _CUTE_VECTOR_DTYPES[dtype]
    ptr = _cute_scalar_pointer_expr(tensor_name, index_exprs)
    if eviction_suffix in _CUTE_CACHE_LOAD_HELPERS:
        # Explicit-vec ("vec" mode) loads return FLOAT vectors, not the
        # carrier form the L2 helper produces; drop the hint here.
        eviction_suffix = ""
    return (
        f"cute.arch.load({ptr}, "
        f"ir.VectorType.get([{vec_width}], {elem_str}.mlir_type)"
        f"{eviction_suffix})"
    )


def _cute_vector_store_expr(
    tensor_name: str,
    index_exprs: list[str],
    value: str,
    dtype: torch.dtype,
    *,
    vec_width: int,
) -> str:
    elem_str, _ = _CUTE_VECTOR_DTYPES[dtype]
    ptr = _cute_scalar_pointer_expr(tensor_name, index_exprs)
    return (
        f"cute.arch.store({ptr}, {value}, "
        f"ir.VectorType.get([{vec_width}], {elem_str}.mlir_type))"
    )


def _cute_register_unroll_vec_hoist(
    state: CodegenState,
    strategy: object,  # Looped/PersistentReductionStrategy at runtime
    tensor: torch.Tensor,
    tensor_name: str,
    index_exprs: list[str],
    vec_width: int,
    mask_expr: str | None = None,
    eviction_suffix: str = "",
) -> str | None:
    """Register an integer-carrier vec load to be hoisted above the constexpr V-loop
    in the active lane body and return the per-element extract expression.

    The hoist runs once per outer-lane iter; the constexpr V-loop's body
    receives ``hoist_var[vi].bitcast(dtype)`` (a scalar) so the existing
    cast/mul/accumulate pipeline keeps working unchanged.
    """
    elem_dtype = _CUTE_VECTOR_UNROLL_DTYPES[tensor.dtype]
    carrier = _CUTE_VECTOR_UNROLL_CARRIER[tensor.dtype]
    base_index_var = getattr(strategy, "_cute_lane_base_index_var", None)
    lane_body = getattr(strategy, "_cute_lane_body", None)
    assert isinstance(base_index_var, str)
    assert isinstance(lane_body, list)
    from ..reduction_strategy import PersistentReductionStrategy

    if isinstance(strategy, PersistentReductionStrategy):
        # Persistent memory order is only complete after lane splitting.  The
        # caller leaves an exact-fragment marker for the late pass instead of
        # installing a wrapper-global hoist that could cross a later write.
        return None
    from ..reduction_strategy import LoopedReductionStrategy

    assert isinstance(strategy, LoopedReductionStrategy)
    # Looped reductions expose their reduction index directly in the last
    # position, so replacing that component is sufficient.
    base_exprs = list(index_exprs)
    base_exprs[-1] = base_index_var
    base_ptr_expr = _cute_scalar_pointer_expr(tensor_name, base_exprs)
    # A mask predicates the whole packet.  Outer terms such as the row mask
    # (proven uniform across the chunk by the caller) apply whether or not
    # the roll itself is masked: a partially filled row tile must not read
    # rows past the tensor even when its extent divides the block.  A masked
    # roll adds the chunk term spelled on the chunk base (the chunk decides
    # every lane when ``numel % V == 0``, otherwise its last lane must be in
    # bounds); the compare is inlined rather than read through the mask
    # variable so a later software pipelining pass that rebases the packet
    # address rebases its guard too.  A masked-off thread reads the tensor's
    # first element instead; the per-element mask gate discards those bytes.
    guard_terms: list[str] = []
    non_lane = (
        _cute_vector_load_non_lane_mask(mask_expr, strategy._mask_var)
        if mask_expr is not None
        else None
    )
    if non_lane is not None:
        guard_terms.append(non_lane)
    if strategy._mask_var is not None:
        guard_terms.append(strategy.cute_chunk_in_bounds_expr(state))
    guard = " and ".join(f"({term})" for term in guard_terms) or None
    cache_key = (tensor_name, base_ptr_expr, guard)
    cache = getattr(strategy, "_cute_lane_vec_loads", None)
    if cache is None:
        cache = {}
        # pyrefly: ignore [missing-attribute]
        strategy._cute_lane_vec_loads = cache
    # Locate the constexpr V-loop: prefer the node recorded by
    # codegen_device_loop (vec-store flushes may sit after it, so it is not
    # guaranteed to be the last lane_body entry).
    constexpr_loop = getattr(strategy, "_cute_lane_vloop", None)
    if constexpr_loop is None or constexpr_loop not in lane_body:
        constexpr_loop = lane_body[-1]
    if cache_key not in cache:
        hoist_var = state.device_function.new_var(
            f"_unroll_vec_{len(cache)}", dce=False
        )
        cache[cache_key] = (hoist_var, tensor.dtype)
        load_ptr_expr = base_ptr_expr
        if guard is not None:
            anchor_ptr_expr = _cute_scalar_pointer_expr(
                tensor_name, ["0"] * len(index_exprs)
            )
            load_ptr_expr = f"({base_ptr_expr} if {guard} else {anchor_ptr_expr})"
        hoist_stmt = statement_from_string(
            f"{hoist_var} = "
            f"{_cute_unroll_vec_load_expr(load_ptr_expr, tensor.dtype, vec_width, eviction_suffix)}"
        )
        # Insert the hoist just BEFORE the constexpr V-loop.
        lane_body.insert(lane_body.index(constexpr_loop), hoist_stmt)
    else:
        hoist_var, _ = cache[cache_key]
    assert isinstance(constexpr_loop, ast.For)
    assert isinstance(constexpr_loop.target, ast.Name)
    vec_lane_var = constexpr_loop.target.id
    extract = f"{carrier}({hoist_var}[{vec_lane_var}]).bitcast({elem_dtype})"
    chunk_full_var = strategy._cute_reduction_chunk_full_var
    if strategy._mask_var is not None and chunk_full_var is not None:
        # The straddling tail chunk holds anchor bytes: its in-bounds lanes
        # re-read their element individually (at most one chunk per row).
        scalar_load = _cute_scalar_load_expr(
            tensor_name, index_exprs, tensor.dtype, eviction_suffix=eviction_suffix
        )
        return f"({extract} if {chunk_full_var} else {scalar_load})"
    return extract


def _persistent_assignment_definitions(state: CodegenState) -> dict[str, ast.expr]:
    """Collect visible single-name definitions used by persistent lane code."""
    definitions: dict[str, ast.expr] = {}
    grid = state.codegen.current_grid_state
    statement_lists = [*state.codegen.statements_stack]
    if grid is not None:
        statement_lists.append(grid.lane_setup_statements)
    for statements in statement_lists:
        for stmt in statements:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                definitions[stmt.targets[0].id] = stmt.value
    return definitions


def _persistent_expr_depends_on_inner_lane(
    state: CodegenState,
    strategy: object,
    expression: str,
) -> bool:
    """Whether an expression transitively reads this persistent V-lane."""
    block_index = getattr(strategy, "block_index", None)
    if not isinstance(block_index, int):
        return False
    lane_var = getattr(strategy, "_synthetic_cute_lane_var", None)
    inner_names = {
        name
        for name in (cast("Any", strategy).index_var(block_index), lane_var)
        if isinstance(name, str)
    }
    try:
        expression_node = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return False
    definitions = _persistent_assignment_definitions(state)
    pending = list(ReadWrites.from_ast(expression_node).reads)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in inner_names:
            return True
        if name in seen:
            continue
        seen.add(name)
        definition = definitions.get(name)
        if definition is not None:
            pending.extend(ReadWrites.from_ast(definition).reads)
    return False


def _persistent_vec_base_index_exprs(
    state: CodegenState,
    strategy: object,
    index_exprs: list[str],
    *,
    preserve_lane_independent_aliases: bool = False,
) -> list[str] | None:
    """Rebase affine persistent-lane indices at the vector chunk start.

    Packed tensors commonly index a K slice as ``head * K + k_offset``.  The
    scalar index arrives here through generated aliases, while the vector
    transaction is emitted before the constexpr lane loop that defines those
    aliases.  Inline pure assignment chains and replace the reduction index by
    its chunk base.  A chain containing a memory load remains scalar.
    """
    base_index_var = getattr(strategy, "_cute_lane_base_index_var", None)
    block_index = getattr(strategy, "block_index", None)
    if not isinstance(base_index_var, str) or not isinstance(block_index, int):
        return None
    reduction_index_var = cast("Any", strategy).index_var(block_index)
    lane_var = getattr(strategy, "_synthetic_cute_lane_var", None)
    definitions = _persistent_assignment_definitions(state)

    class _InlineIndexAliases(ast.NodeTransformer):
        def __init__(self) -> None:
            super().__init__()
            self.expanding: set[str] = set()
            self.rebased = False

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if not isinstance(node.ctx, ast.Load):
                return node
            if node.id == reduction_index_var:
                self.rebased = True
                return ast.copy_location(
                    ast.Name(id=base_index_var, ctx=ast.Load()), node
                )
            if isinstance(lane_var, str) and node.id == lane_var:
                self.rebased = True
                return ast.copy_location(ast.Constant(value=0), node)
            value = definitions.get(node.id)
            if value is None or node.id in self.expanding:
                return node
            if preserve_lane_independent_aliases and not (
                _persistent_expr_depends_on_inner_lane(state, strategy, node.id)
            ):
                # A late branch-local hoist can retain a scalar/load-derived
                # outer index as an opaque value.  Expand only aliases that
                # carry the reduction lane so the affine proof below still
                # rejects reversed, strided, or otherwise non-unit access.
                return node
            # Stop at a symbol already defined in the hoist's enclosing
            # scope.  Expanding outer grid aliases such as ``tile_offset_*``
            # or host scalar arguments can expose host-only expressions (for
            # example ``tensor.size(0)``) inside the CuTe device function.
            if _persistent_vec_scope_safe(state, strategy, node.id):
                return node
            self.expanding.add(node.id)
            replacement = self.visit(ast.parse(ast.unparse(value), mode="eval").body)
            self.expanding.remove(node.id)
            return ast.copy_location(replacement, node)

    def unit_affine_in_reduction_index(node: ast.AST) -> tuple[bool, bool]:
        """Return ``(depends_on_reduction_index, is_unit_affine)``."""
        if isinstance(node, ast.Name):
            return node.id == base_index_var, True
        if isinstance(node, ast.BinOp):
            left_depends, left_valid = unit_affine_in_reduction_index(node.left)
            right_depends, right_valid = unit_affine_in_reduction_index(node.right)
            depends = left_depends or right_depends
            if not depends:
                return False, left_valid and right_valid
            if not left_valid or not right_valid or left_depends and right_depends:
                return True, False
            if isinstance(node.op, ast.Add):
                return True, True
            if isinstance(node.op, ast.Sub) and left_depends:
                return True, True
            return True, False
        if isinstance(node, ast.Call):
            dependencies = [unit_affine_in_reduction_index(arg) for arg in node.args]
            depends = any(item[0] for item in dependencies)
            if not depends:
                return False, all(item[1] for item in dependencies)
            is_cutlass_cast = (
                len(node.args) == 1
                and not node.keywords
                and ast.unparse(node.func).startswith("cutlass.")
            )
            return True, is_cutlass_cast and all(item[1] for item in dependencies)
        dependencies = [
            unit_affine_in_reduction_index(child)
            for child in ast.iter_child_nodes(node)
        ]
        depends = any(item[0] for item in dependencies)
        return depends, not depends and all(item[1] for item in dependencies)

    rewritten: list[str] = []
    rebased = False
    for expression in index_exprs:
        try:
            parsed = ast.parse(expression, mode="eval").body
        except SyntaxError:
            return None
        rewriter = _InlineIndexAliases()
        parsed = rewriter.visit(parsed)
        rebased = rebased or rewriter.rebased
        depends, is_unit_affine = unit_affine_in_reduction_index(parsed)
        if depends and not is_unit_affine:
            return None
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("load", "store")
            for node in ast.walk(parsed)
        ):
            return None
        rewritten.append(ast.unparse(parsed))
    return rewritten if rebased else None


def _persistent_vec_is_exact_aligned(
    state: CodegenState,
    strategy: object,
    index_exprs: list[str],
    tensor: torch.Tensor,
    vec_width: int,
) -> bool:
    """Whether the lane tensor dimension starts at a complete aligned fragment.

    The late branch-local pass cannot scalarize a partially valid vector.  In
    particular, ``cols + 1`` leaves seven valid values in the final fragment;
    guarding the transaction as a whole would either read one element OOB or
    drop those seven stores.  Restrict late vectorization to the reduction
    coordinate plus an offset provably divisible by the vector width.  The
    late pass separately turns the scalar bounds into a whole-fragment guard.
    """
    base_var = getattr(strategy, "_cute_lane_base_index_var", None)
    if not isinstance(base_var, str):
        return False
    rebased = _persistent_vec_base_index_exprs(
        state,
        strategy,
        index_exprs,
        preserve_lane_independent_aliases=True,
    )
    if rebased is None:
        return False

    definitions = _persistent_assignment_definitions(state)
    resolving: set[str] = set()

    def is_aligned_offset(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value % vec_width == 0
        if isinstance(node, ast.Name):
            definition = definitions.get(node.id)
            if definition is None or node.id in resolving:
                return False
            resolving.add(node.id)
            result = is_aligned_offset(definition)
            resolving.remove(node.id)
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return is_aligned_offset(node.operand)
        if (
            isinstance(node, ast.Call)
            and len(node.args) == 1
            and not node.keywords
            and ast.unparse(node.func).startswith("cutlass.")
        ):
            return is_aligned_offset(node.args[0])
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return is_aligned_offset(node.left) and is_aligned_offset(node.right)
            if isinstance(node.op, ast.Mult):
                return is_aligned_offset(node.left) or is_aligned_offset(node.right)
        return False

    class _RemoveBase(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if isinstance(node.ctx, ast.Load) and node.id == base_var:
                return ast.copy_location(ast.Constant(value=0), node)
            return node

    dependent: list[tuple[int, ast.expr]] = []
    for dim, expression in enumerate(rebased):
        try:
            parsed = ast.parse(expression, mode="eval").body
        except SyntaxError:
            return False
        if base_var in ReadWrites.from_ast(parsed).reads:
            dependent.append((dim, parsed))
    if len(dependent) != 1:
        return False
    offset = _RemoveBase().visit(
        ast.parse(ast.unparse(dependent[0][1]), mode="eval").body
    )
    if not isinstance(offset, ast.expr) or not is_aligned_offset(offset):
        return False

    env = CompileEnvironment.current()
    runtime_tensor = env.runtime_value_for_tensor(tensor)
    if isinstance(runtime_tensor, torch.Tensor) and not isinstance(
        runtime_tensor, FakeTensor
    ):
        strides = tuple(int(stride) for stride in runtime_tensor.stride())
        sizes = tuple(int(size) for size in runtime_tensor.shape)
        element_size = runtime_tensor.element_size()
        required_alignment = vec_width * element_size
        if not runtime_tensor_has_specialized_alignment(
            env, tensor, required_alignment
        ):
            return False
    else:
        # Kernel-owned allocations are suitably aligned by the backend.  A
        # nonzero or symbolic view offset cannot establish that guarantee.
        if env.tensor_input_source(tensor) is not None:
            return False
        storage_offset = tensor.storage_offset()
        if not isinstance(storage_offset, int) or storage_offset != 0:
            return False
        strides = tensor.stride()
        sizes = tensor.shape
        element_size = tensor.element_size()
        required_alignment = vec_width * element_size

    if len(strides) != len(rebased):
        return False
    lane_dim = dependent[0][0]
    lane_size = sizes[lane_dim]
    if not isinstance(lane_size, int) or lane_size % vec_width:
        return False
    for dim, stride in enumerate(strides):
        if not isinstance(stride, int):
            return False
        if dim == lane_dim:
            if stride != 1:
                return False
        elif stride * element_size % required_alignment:
            return False
    return True


def _persistent_vec_scope_safe(
    state: CodegenState,
    strategy: object,
    expression: str,
) -> bool:
    """Whether ``expression`` is available outside a persistent V-loop.

    Persistent reductions wrap a complete root graph rather than a dedicated
    device-loop graph.  A vec transaction is spliced before their constexpr
    V-loop, so it may only reference kernel/global values or coordinates from
    an *outer* grid lane.  Definitions emitted in the root body itself, and
    setup definitions that depend on this reduction's synthetic lane, are not
    available there.
    """
    try:
        expr = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return False
    reads = set(ReadWrites.from_ast(expr).reads)

    statement_stack = getattr(state.codegen, "statements_stack", None)
    if isinstance(statement_stack, list) and statement_stack:
        # The persistent wrapper is appended to ``hoist_parent_statements``.
        # Existing definitions in that exact list dominate it; every list
        # pushed after that point is a nested branch/loop body that will sit
        # below the wrapper.  Inspect all of those nested parent scopes, not
        # merely the current innermost list.
        grid = state.codegen.current_grid_state
        hoist_parent = getattr(grid, "hoist_parent_statements", None)
        try:
            hoist_parent_index = next(
                index
                for index, statements in enumerate(statement_stack)
                if statements is hoist_parent
            )
        except StopIteration:
            hoist_parent_index = -1
        locally_written = {
            name
            for statements in statement_stack[hoist_parent_index + 1 :]
            if isinstance(statements, list)
            for stmt in statements
            for name in ReadWrites.from_ast(stmt).writes
        }
        if reads & locally_written:
            return False

    grid = state.codegen.current_grid_state
    if grid is None:
        return False
    lane_var = getattr(strategy, "_synthetic_cute_lane_var", None)
    base_var = getattr(strategy, "_cute_lane_base_index_var", None)

    # A persistent wrapper is inserted according to ``lane_loops`` nesting.
    # Targets belonging to a deeper lane loop do not exist at its hoist site.
    # Earlier lane targets are outer scopes and remain available (for example
    # the row lane in a row-by-K reduction).
    persistent_depth = next(
        (
            depth
            for depth, (candidate, _extent) in enumerate(grid.lane_loops)
            if candidate == lane_var
        ),
        None,
    )
    if persistent_depth is None:
        return False

    # Resolve the exact lane nesting depth of every setup definition.  Merely
    # checking a setup statement's direct reads is insufficient: a seemingly
    # harmless alias can transitively read an inner lane (``b = a + 1`` where
    # ``a = inner_lane + 7``) and would then be referenced by the hoist before
    # either definition exists.  Unknown/fallback setup belongs to the
    # innermost scope, matching ``DeviceGridState.wrap_body`` conservatively.
    lane_scope_names: list[set[str]] = []
    for candidate, _extent in grid.lane_loops:
        names = {candidate}
        wrapper = grid.vec_lane_wrappers.get(candidate)
        if wrapper is not None:
            names.update((wrapper.vec_lane_var, wrapper.base_index_var))
        lane_scope_names.append(names)
    setup_name_depths: dict[str, int] = {}
    innermost_depth = len(grid.lane_loops) - 1
    for stmt in grid.lane_setup_statements:
        rw = ReadWrites.from_ast(stmt)
        stmt_reads = set(rw.reads)
        dependency_depths = [
            depth for depth, names in enumerate(lane_scope_names) if stmt_reads & names
        ]
        dependency_depths.extend(
            setup_name_depths[name] for name in stmt_reads if name in setup_name_depths
        )
        definition_depth = (
            max(dependency_depths) if dependency_depths else innermost_depth
        )
        for name in rw.writes:
            setup_name_depths[name] = definition_depth

    for depth, names in enumerate(lane_scope_names):
        if depth < persistent_depth:
            continue
        unavailable_names = set(names)
        if depth == persistent_depth and isinstance(base_var, str):
            # The wrapper defines its chunk base immediately before the
            # hoisted transaction; only its constexpr lane target is local to
            # the loop body.
            unavailable_names.discard(base_var)
        if reads & unavailable_names:
            return False
    if any(
        name in setup_name_depths and setup_name_depths[name] >= persistent_depth
        for name in reads
    ):
        return False

    # Device-loop induction variables are definitions owned by an enclosing
    # ``for`` node, rather than assignments in ``statements_stack``.  Treat
    # them as local: the persistent wrapper may be emitted outside that loop
    # when grid/lane bodies are reconstructed later.
    seen_loop_states: set[int] = set()
    for loop_states in state.codegen.active_device_loops.values():
        for loop_state in loop_states:
            if id(loop_state) in seen_loop_states:
                continue
            seen_loop_states.add(id(loop_state))
            for_node = getattr(loop_state, "for_node", None)
            if isinstance(for_node, ast.For):
                loop_writes = set(ReadWrites.from_ast(for_node.target).writes)
                if reads & loop_writes:
                    return False
    return True


def _cute_lane_strategy(state: CodegenState, block_id: int) -> object | None:
    """Return the live lane strategy, including persistent reductions.

    A persistent reduction contributes its synthetic lane wrapper to the
    current grid but is not itself pushed into ``active_device_loops``.  Look
    it up through the tile dispatcher when the active loop belongs only to the
    enclosing free-tile strategy.
    """
    loops = state.codegen.active_device_loops.get(block_id)
    if loops:
        return loops[-1].strategy
    env = CompileEnvironment.current()
    if env.block_sizes[block_id].reduction:
        from ..reduction_strategy import PersistentReductionStrategy

        strategy = state.device_function.tile_strategy.get_reduction_strategy(block_id)
        if isinstance(strategy, PersistentReductionStrategy):
            return strategy
    return None


def _cute_stack_tensor_offset_expr(
    state: CodegenState,
    tensor_like: torch.Tensor,
    subscript: list[object],
    ast_subscript: list[object] | tuple[object, ...],
) -> str:
    env = CompileEnvironment.current()
    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor_like,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if "None" in index_exprs:
        raise exc.BackendUnsupported("cute", "inactive stack tensor load dimension")
    index_dtype = env.index_type()
    terms = []
    for dim, index in enumerate(index_exprs):
        stride = tensor_like.stride(dim)
        stride_expr = (
            str(stride) if isinstance(stride, int) else state.sympy_expr(stride)
        )
        terms.append(f"({index_dtype}({index}) * {index_dtype}({stride_expr}))")
    return " + ".join(terms) if terms else "0"


def _cute_stack_tensor_mask_expr(
    state: CodegenState,
    tensor_like: torch.Tensor,
    dev_ptrs: torch.Tensor,
    subscript: list[object],
    extra_mask: ast.AST | None,
    *,
    for_store: bool = False,
) -> str | None:
    terms = []
    tensor_mask = _cute_combined_mask(
        state,
        subscript,
        extra_mask,
        tensor=tensor_like,
        include_tensor_index_masks=False,
        for_store=for_store,
    )
    if tensor_mask is not None:
        terms.append(tensor_mask)
    stack_mask = _cute_combined_mask(
        state,
        [slice(None)] * dev_ptrs.ndim,
        None,
        tensor=dev_ptrs,
    )
    if stack_mask is not None and stack_mask not in terms:
        terms.append(stack_mask)
    if not terms:
        return None
    return " and ".join(f"({term})" for term in terms)


def _cute_stack_tensor_pointer_expr(
    target_dtype: str,
    dev_ptrs_ast: ast.AST,
    offset_expr: str,
) -> ast.AST:
    return expr_from_string(
        f"(cute.make_ptr({target_dtype}, cutlass.Int64({{base}}), "
        f"cute.AddressSpace.gmem) + ({offset_expr}))",
        base=dev_ptrs_ast,
    )


def _codegen_cute_store_stack_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: tuple[object, ...] | list[object],
    ast_subscript: tuple[object, ...] | list[object],
    value: ast.AST,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    if value_node.op != "call_function" or value_node.target is not load:
        return None
    stack_arg = value_node.args[0]
    if not isinstance(stack_arg, tuple) or len(stack_arg) != 2:
        return None
    ptr_node = stack_arg[1]
    if (
        not isinstance(ptr_node, torch.fx.Node)
        or ptr_node.op != "call_function"
        or ptr_node.target is not load
        or len(ptr_node.args) < 2
    ):
        return None
    dev_ptrs = (
        ptr_node.args[0].meta.get("val")
        if isinstance(ptr_node.args[0], torch.fx.Node)
        else None
    )
    ptr_subscript = ptr_node.args[1]
    if not isinstance(dev_ptrs, torch.Tensor) or not isinstance(
        ptr_subscript, (list, tuple)
    ):
        return None
    tensor_like_node = stack_arg[0]
    tensor_like = (
        tensor_like_node.meta.get("val")
        if isinstance(tensor_like_node, torch.fx.Node)
        else tensor_like_node
    )
    if not isinstance(tensor_like, torch.Tensor):
        return None

    if (
        dev_ptrs.ndim == 2
        and len(ptr_subscript) == 2
        and all(isinstance(idx, slice) and idx == slice(None) for idx in ptr_subscript)
        and len(subscript) >= 3
        and isinstance(subscript[0], slice)
        and subscript[0] == slice(None)
        and isinstance(subscript[1], slice)
        and subscript[1] == slice(None)
    ):
        stack_value_subscript = value_node.args[1]
        if not isinstance(stack_value_subscript, (list, tuple)):
            return None
        stack_value_subscript_proxy = map_arg(
            stack_value_subscript, lambda arg: arg.meta["val"]
        )
        stack_value_subscript_ast = map_arg(
            stack_value_subscript, lambda arg: state.env[arg]
        )
        tensor_offset_expr = _cute_stack_tensor_offset_expr(
            state,
            tensor_like,
            [*stack_value_subscript_proxy],
            [*stack_value_subscript_ast],
        )
        target_index_exprs = _cute_index_exprs(
            state,
            [*subscript],
            ast_subscript,
            tensor=tensor,
            inactive_singleton_slice_expr="0",
        )
        if len(target_index_exprs) != tensor.ndim:
            return None
        first_stack_index = target_index_exprs[0]
        target_tail = target_index_exprs[2:]
        loop_var = state.device_function.new_var("stack_dim", dce=True)
        env = CompileEnvironment.current()
        index_dtype = env.index_type()
        dev_ptrs_name = state.device_function.tensor_arg(dev_ptrs).name
        tensor_name = state.device_function.tensor_arg(tensor).name
        target_dtype = env.backend.dtype_str(tensor.dtype)
        dev_ptr_offset = (
            f"{index_dtype}({first_stack_index}) * "
            f"{index_dtype}({dev_ptrs.stride(0)}) + "
            f"{index_dtype}({loop_var}) * {index_dtype}({dev_ptrs.stride(1)})"
        )
        stack_ptr_expr = (
            f"(cute.make_ptr({target_dtype}, "
            f"cutlass.Int64(({dev_ptrs_name}.iterator + {dev_ptr_offset}).load()), "
            f"cute.AddressSpace.gmem) + ({tensor_offset_expr}))"
        )
        target_indices = [first_stack_index, loop_var, *target_tail]
        store_expr = _cute_scalar_store_expr(
            tensor_name,
            target_indices,
            f"({stack_ptr_expr}).load()",
        )
        mask_expr = _cute_combined_mask(
            state, [*subscript], extra_mask, tensor=tensor, for_store=True
        )
        if mask_expr is None:
            body = f"    {store_expr}"
        else:
            body = f"    if {mask_expr}:\n        {store_expr}"
        state.add_statement(
            statement_from_string(
                f"for {loop_var} in range({dev_ptrs.size(1)}):\n{body}"
            )
        )
        return ast.Constant(value=None)

    ptr_subscript_proxy = map_arg(ptr_subscript, lambda arg: arg.meta["val"])
    ptr_subscript_ast = map_arg(ptr_subscript, lambda arg: state.env[arg])
    ptr_index_exprs = _cute_index_exprs(
        state,
        [*ptr_subscript_proxy],
        [*ptr_subscript_ast],
        tensor=dev_ptrs,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if "None" in ptr_index_exprs:
        return None

    target_index_exprs = _cute_index_exprs(
        state,
        [*subscript],
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    ptr_pos = 0
    rewritten_index_exprs = []
    for idx, index_expr in zip(subscript, target_index_exprs, strict=True):
        if isinstance(idx, slice) and idx == slice(None):
            replacement = (
                ptr_index_exprs[ptr_pos] if ptr_pos < len(ptr_index_exprs) else None
            )
            ptr_pos += 1
            rewritten_index_exprs.append(
                replacement if replacement is not None else index_expr
            )
        else:
            if ptr_pos < len(ptr_subscript_proxy) and not (
                isinstance(ptr_subscript_proxy[ptr_pos], slice)
                and ptr_subscript_proxy[ptr_pos] == slice(None)
            ):
                ptr_pos += 1
            rewritten_index_exprs.append(index_expr)

    tensor_name = state.device_function.tensor_arg(tensor).name
    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(tensor.dtype)
    value = expr_from_string(
        backend.ast_to_dtype_expr("{value}", target_dtype),
        value=value,
    )
    assert isinstance(value, ast.expr)
    store_expr = expr_from_string(
        _cute_scalar_store_expr(tensor_name, rewritten_index_exprs, "{value}"),
        value=value,
    )
    mask_expr = _cute_combined_mask(
        state, [*subscript], extra_mask, tensor=tensor, for_store=True
    )
    if mask_expr is None:
        return store_expr
    mask_ast = expr_from_string(mask_expr)
    assert isinstance(mask_ast, ast.expr)
    assert isinstance(store_expr, ast.expr)
    state.add_statement(
        ast.fix_missing_locations(
            ast.If(
                test=mask_ast,
                body=[ast.Expr(value=store_expr)],
                orelse=[],
            )
        )
    )
    return ast.Constant(value=None)


def _cute_affine_range_block_id(state: CodegenState, affine: object) -> int | None:
    from .indexing import CuteAffineRangeIndex

    if not isinstance(affine, CuteAffineRangeIndex):
        return None
    env = CompileEnvironment.current()
    base_meta = getattr(affine.base, "meta", {})
    base_val = base_meta.get("val") if isinstance(base_meta, dict) else None
    block_id = env.resolve_block_id(base_val) if base_val is not None else None
    if block_id is None:
        codegen = base_meta.get("codegen") if isinstance(base_meta, dict) else None
        if isinstance(codegen, ast.Name) and codegen.id.startswith("_BLOCK_SIZE_"):
            with contextlib.suppress(ValueError):
                block_id = int(codegen.id.removeprefix("_BLOCK_SIZE_"))
    if block_id is None:
        return None
    if state.fx_node is not None:
        return env.resolve_codegen_block_id(
            block_id, state.codegen, state.fx_node.graph
        )
    return block_id


def _cute_affine_range_expr(
    state: CodegenState,
    affine: object,
    lane_var: str,
    *,
    dtype: torch.dtype | None = None,
) -> str | None:
    from .indexing import CuteAffineRangeIndex

    if not isinstance(affine, CuteAffineRangeIndex):
        return None
    if affine.step != 1 or affine.factor <= 0:
        return None
    block_id = _cute_affine_range_block_id(state, affine)
    if block_id is None:
        return None
    index_var = _cute_active_index_var(state, block_id)
    if index_var is None:
        return None
    expr = f"({affine.factor}) * ({index_var}) + cutlass.Int32({lane_var})"
    if dtype is not None:
        expr = f"{CompileEnvironment.current().backend.dtype_str(dtype)}({expr})"
    return expr


def _codegen_cute_affine_range_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    value: object,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None = None,
) -> ast.AST | None:
    from ..ast_extension import create
    from .indexing import CuteAffineRangeIndex

    affine_positions = [
        (pos, idx)
        for pos, idx in enumerate(ast_subscript)
        if isinstance(idx, CuteAffineRangeIndex)
    ]
    if len(affine_positions) != 1 or len(subscript) != 1 or extra_mask is not None:
        return None
    _pos, affine = affine_positions[0]
    block_id = _cute_affine_range_block_id(state, affine)
    if block_id is None:
        return None

    lane_var = state.device_function.new_var("affine_lane", dce=True)
    index_expr = _cute_affine_range_expr(
        state, affine, lane_var, dtype=CompileEnvironment.current().index_dtype
    )
    if index_expr is None:
        return None
    backend = CompileEnvironment.current().backend
    if (
        value_node is not None
        and value_node.op == "call_function"
        and value_node.target is load
    ):
        source_tensor_node = value_node.args[0]
        if not isinstance(source_tensor_node, torch.fx.Node):
            return None
        source_tensor = source_tensor_node.meta.get("val")
        if not isinstance(source_tensor, torch.Tensor):
            return None
        source_subscript = value_node.args[1]
        if (
            not isinstance(source_subscript, (list, tuple))
            or len(source_subscript) != 1
        ):
            return None
        source_subscript_args = tuple(cast("Any", source_subscript))
        ast_source_subscript = list(
            map_arg(source_subscript_args, lambda arg: state.env[arg])
        )
        (source_affine,) = ast_source_subscript
        if not isinstance(source_affine, CuteAffineRangeIndex):
            return None
        if source_affine.factor != affine.factor:
            return None
        source_index_expr = _cute_affine_range_expr(
            state,
            source_affine,
            lane_var,
            dtype=CompileEnvironment.current().index_dtype,
        )
        if source_index_expr is None:
            return None
        source_name = state.device_function.tensor_arg(source_tensor).name
        value_expr = f"{source_name}[{source_index_expr}]"
        if source_tensor.dtype is torch.bool:
            value_expr = f"({value_expr} != cutlass.Uint8(0))"
    elif isinstance(value, CuteAffineRangeIndex):
        value_expr = _cute_affine_range_expr(state, value, lane_var, dtype=value.dtype)
        if value_expr is None:
            return None
    elif isinstance(value, ast.AST):
        value_expr = ast.unparse(value)
    elif isinstance(value, (int, float, bool)):
        value_expr = repr(value)
    else:
        return None

    target_dtype = backend.dtype_str(tensor.dtype)
    value_expr = backend.ast_to_dtype_expr(value_expr, target_dtype)
    tensor_name = state.device_function.tensor_arg(tensor).name
    store_expr = (
        f"{tensor_name}.__setitem__({_cute_index_tuple([index_expr])}, {value_expr})"
    )
    mask_var = _cute_active_mask_var(state, block_id)
    if mask_var is not None:
        store_expr = f"{store_expr} if {mask_var} else None"

    return create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({affine.factor})"),
        body=[create(ast.Expr, value=expr_from_string(store_expr))],
        orelse=[],
        type_comment=None,
    )


def _codegen_cute_affine_reshape_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None,
) -> ast.AST | None:
    """Lower a 2-D affine-row store fed by a reshape/stack chain.

    Handles ``out[(begin*K):(begin*K + block*K), tile_n] = reshaped`` where the
    leading index is a ``CuteAffineRangeIndex`` (factor ``K``) over the m-tile,
    the trailing index is the n-tile, and the value is a row-major shape chain
    (e.g. ``stack([a, b], dim=1).reshape(block*K, block_n)``).

    Each m-tile thread owns row ``m_local`` of the source; the reshaped tensor
    has ``K`` rows per source row, so the thread loops ``s in range(K)`` and
    writes the value resolved at flat index ``(K*m_local + s)*block_n + n_local``
    to output row ``K*m_global + s``, column ``n_global``.
    """
    from ..ast_extension import create
    from ..generate_ast import GenerateAST
    from .cute_reshape import _get_block_local_coord
    from .cute_reshape import resolve_cute_shape_chain_value_at
    from .indexing import CuteAffineRangeIndex
    from .indexing import is_cute_shape_chain_target

    if (
        tensor.ndim != 2
        or len(subscript) != 2
        or len(ast_subscript) != 2
        or extra_mask is not None
        or value_node is None
        or not isinstance(state.codegen, GenerateAST)
    ):
        return None
    affine = ast_subscript[0]
    if not isinstance(affine, CuteAffineRangeIndex):
        return None
    if affine.step != 1 or affine.factor <= 0:
        return None
    n_index = subscript[1]
    if not isinstance(n_index, torch.SymInt):
        return None
    env = CompileEnvironment.current()
    block_id_n = env.get_block_id(n_index)
    if block_id_n is None:
        return None
    block_id_m = _cute_affine_range_block_id(state, affine)
    if block_id_m is None:
        return None

    if value_node.op != "call_function" or not is_cute_shape_chain_target(
        value_node.target
    ):
        return None
    value_val = value_node.meta.get("val")
    if not isinstance(value_val, torch.Tensor) or value_val.ndim != 2:
        return None

    m_global = _cute_active_index_var(state, block_id_m)
    n_global = _cute_active_index_var(state, block_id_n)
    if m_global is None or n_global is None:
        return None
    m_local = _get_block_local_coord(state.codegen, block_id_m)
    n_local = _get_block_local_coord(state.codegen, block_id_n)
    if m_local is None or n_local is None:
        return None
    block_n = state.device_function.resolved_block_size(block_id_n)
    if not isinstance(block_n, int):
        return None

    factor = affine.factor
    lane_var = state.device_function.new_var("affine_lane", dce=True)
    row_local = f"cutlass.Int32({factor}) * ({m_local}) + cutlass.Int32({lane_var})"
    flat_index = (
        f"(({row_local}) * cutlass.Int32({block_n})) + ({n_local})"
        if block_n != 1
        else f"({row_local}) + ({n_local})"
    )
    value_ast = resolve_cute_shape_chain_value_at(state, value_node, flat_index)
    if value_ast is None:
        return None

    backend = env.backend
    index_dtype = backend.dtype_str(env.index_dtype)
    target_dtype = backend.dtype_str(tensor.dtype)
    value_expr = backend.ast_to_dtype_expr(ast.unparse(value_ast), target_dtype)

    # Bind the resolved (possibly select-based) value to a variable so the CuTe
    # DSL sees the stack `ifexp` as its own assignment rather than nested inside
    # the `.store(...)` call / masked store ternary.
    value_var = state.device_function.new_var("affine_value", dce=True)

    row_index = (
        f"{index_dtype}(cutlass.Int32({factor}) * ({m_global}) "
        f"+ cutlass.Int32({lane_var}))"
    )
    col_index = f"{index_dtype}({n_global})"
    tensor_name = state.device_function.tensor_arg(tensor).name
    store_expr = _cute_scalar_store_expr(tensor_name, [row_index, col_index], value_var)

    store_stmt: ast.stmt = create(ast.Expr, value=expr_from_string(store_expr))
    mask_parts = [
        mask
        for mask in (
            _cute_active_mask_var(state, block_id_m),
            _cute_active_mask_var(state, block_id_n),
        )
        if mask is not None
    ]
    if mask_parts:
        # Use a guard statement (not a ternary) so the CuTe DSL accepts the
        # device-value mask condition.
        mask_ast = expr_from_string(" and ".join(mask_parts))
        assert isinstance(mask_ast, ast.expr)
        store_stmt = ast.fix_missing_locations(
            ast.If(test=mask_ast, body=[store_stmt], orelse=[])
        )

    return create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({factor})"),
        body=[
            statement_from_string(f"{value_var} = {value_expr}"),
            store_stmt,
        ],
        orelse=[],
        type_comment=None,
    )


def _is_cute_affine_range_load_for_store(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
) -> bool:
    from .indexing import CuteAffineRangeIndex
    from .indexing import match_cute_affine_range_iota

    def compatible_store_user(user: torch.fx.Node) -> bool:
        if (
            user.op != "call_function"
            or user.target is not store
            or len(user.args) < 4
            or user.args[2] is not state.fx_node
            or user.args[3] is not None
        ):
            return False
        store_subscript = user.args[1]
        return (
            isinstance(store_subscript, (list, tuple))
            and len(store_subscript) == 1
            and isinstance(store_subscript[0], torch.fx.Node)
            and match_cute_affine_range_iota(store_subscript[0]) is not None
        )

    return (
        state.fx_node is not None
        and len(state.fx_node.users) > 0
        and all(compatible_store_user(user) for user in state.fx_node.users)
        and len(subscript) == 1
        and len(ast_subscript) == 1
        and isinstance(ast_subscript[0], CuteAffineRangeIndex)
    )


def _cute_positive_1d_slice_bounds(
    tensor: torch.Tensor, index: object
) -> tuple[int, int, int, int] | None:
    if not isinstance(index, slice) or index == slice(None):
        return None
    with contextlib.suppress(TypeError):
        dim_size = int(tensor.shape[0])
        start, stop, step = index.indices(dim_size)
        if step <= 0:
            return None
        length = max(0, (stop - start + step - 1) // step)
        return start, stop, step, length
    return None


def _is_cute_strided_slice_load_for_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
) -> bool:
    def compatible_store_user(user: torch.fx.Node) -> bool:
        if (
            user.op != "call_function"
            or user.target is not store
            or len(user.args) < 4
            or user.args[2] is not state.fx_node
            or user.args[3] is not None
        ):
            return False
        target_node = user.args[0]
        if not isinstance(target_node, torch.fx.Node):
            return False
        target_tensor = target_node.meta.get("val")
        if not isinstance(target_tensor, torch.Tensor) or target_tensor.ndim != 1:
            return False
        store_subscript = user.args[1]
        return (
            isinstance(store_subscript, (list, tuple))
            and len(store_subscript) == 1
            and _cute_positive_1d_slice_bounds(target_tensor, store_subscript[0])
            is not None
        )

    return (
        state.fx_node is not None
        and len(state.fx_node.users) > 0
        and all(compatible_store_user(user) for user in state.fx_node.users)
        and tensor.ndim == 1
        and len(subscript) == 1
        and _cute_positive_1d_slice_bounds(tensor, subscript[0]) is not None
    )


def _codegen_cute_strided_slice_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    value: object,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None = None,
) -> ast.AST | None:
    from ..ast_extension import create

    if tensor.ndim != 1 or len(subscript) != 1 or extra_mask is not None:
        return None
    target_bounds = _cute_positive_1d_slice_bounds(tensor, subscript[0])
    if target_bounds is None:
        return None
    target_start, _target_stop, target_step, target_length = target_bounds

    env = CompileEnvironment.current()
    backend = env.backend
    index_dtype = backend.dtype_str(env.index_dtype)
    loop_var = state.device_function.new_var("slice_idx", dce=True)
    target_index = f"{index_dtype}({target_start} + {loop_var} * {target_step})"

    if (
        value_node is not None
        and value_node.op == "call_function"
        and value_node.target is load
    ):
        source_tensor_node = value_node.args[0]
        if not isinstance(source_tensor_node, torch.fx.Node):
            return None
        source_tensor = source_tensor_node.meta.get("val")
        if not isinstance(source_tensor, torch.Tensor) or source_tensor.ndim != 1:
            return None
        source_subscript = value_node.args[1]
        if (
            not isinstance(source_subscript, (list, tuple))
            or len(source_subscript) != 1
        ):
            return None
        source_bounds = _cute_positive_1d_slice_bounds(
            source_tensor, source_subscript[0]
        )
        if source_bounds is None:
            return None
        source_start, _source_stop, source_step, source_length = source_bounds
        if source_length != target_length:
            return None
        source_index = f"{index_dtype}({source_start} + {loop_var} * {source_step})"
        source_name = state.device_function.tensor_arg(source_tensor).name
        value_expr = f"{source_name}[{source_index}]"
        if source_tensor.dtype is torch.bool:
            value_expr = f"({value_expr} != cutlass.Uint8(0))"
    elif isinstance(value, ast.AST):
        value_expr = ast.unparse(value)
    elif isinstance(value, (int, float, bool)):
        value_expr = repr(value)
    else:
        return None

    target_name = state.device_function.tensor_arg(tensor).name
    target_dtype = backend.dtype_str(tensor.dtype)
    value_expr = backend.ast_to_dtype_expr(value_expr, target_dtype)
    store_expr = f"{target_name}.__setitem__(({target_index},), {value_expr})"
    return create(
        ast.For,
        target=create(ast.Name, id=loop_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({target_length})"),
        body=[create(ast.Expr, value=expr_from_string(store_expr))],
        orelse=[],
        type_comment=None,
    )


def _codegen_cute_store_loaded_index_trailing_slices(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    from ..ast_extension import create

    if value_node.target is not load or len(value_node.args) < 2:
        return None
    source_tensor_node = value_node.args[0]
    if not isinstance(source_tensor_node, torch.fx.Node):
        return None
    source_tensor = source_tensor_node.meta.get("val")
    if not isinstance(source_tensor, torch.Tensor):
        return None
    source_subscript = value_node.args[1]
    if not isinstance(source_subscript, (list, tuple)) or not source_subscript:
        return None
    indexer = source_subscript[0]
    if not isinstance(indexer, torch.fx.Node):
        return None
    indexer_value = indexer.meta.get("val")
    if not isinstance(indexer_value, torch.Tensor) or indexer_value.ndim == 0:
        return None
    source_subscript_args = tuple(cast("Any", source_subscript))
    trailing_source = list(source_subscript_args[1:])
    if not trailing_source or not all(idx == slice(None) for idx in trailing_source):
        return None
    if len(subscript) != indexer_value.ndim + len(trailing_source):
        return None
    trailing_store = subscript[indexer_value.ndim :]
    if not all(idx == slice(None) for idx in trailing_store):
        return None

    ast_source_subscript = list(
        map_arg(source_subscript_args, lambda arg: state.env[arg])
    )
    index_exprs = _cute_index_exprs(
        state,
        [indexer_value],
        [ast_source_subscript[0]],
        tensor=source_tensor,
        inactive_singleton_slice_expr="0",
    )
    if len(index_exprs) != 1:
        return None

    prefix_subscript = [*subscript[: indexer_value.ndim]]
    prefix_ast_subscript = [*ast_subscript[: indexer_value.ndim]]
    target_prefix = _cute_index_exprs(
        state,
        prefix_subscript,
        prefix_ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    if len(target_prefix) != indexer_value.ndim:
        return None

    env = CompileEnvironment.current()
    index_dtype = env.backend.dtype_str(env.index_dtype)
    source_loop_vars = [
        state.device_function.new_var("slice_idx", dce=True) for _ in trailing_source
    ]
    source_indices = [
        index_exprs[0],
        *[f"{index_dtype}({var})" for var in source_loop_vars],
    ]
    target_indices = [
        *target_prefix,
        *[f"{index_dtype}({var})" for var in source_loop_vars],
    ]
    if len(source_indices) != source_tensor.ndim or len(target_indices) != tensor.ndim:
        return None

    source_name = state.device_function.tensor_arg(source_tensor).name
    target_name = state.device_function.tensor_arg(tensor).name
    source_dtype = env.backend.dtype_str(source_tensor.dtype)
    target_dtype = env.backend.dtype_str(tensor.dtype)
    source_mask = _cute_combined_mask(
        state,
        [indexer_value],
        None,
        tensor=source_tensor,
    )
    target_mask = _cute_combined_mask(
        state,
        prefix_subscript,
        extra_mask,
        tensor=tensor,
        for_store=True,
    )
    masks = [mask for mask in (source_mask, target_mask) if mask is not None]
    mask_expr = " and ".join(f"({mask})" for mask in masks) if masks else None
    load_expr = f"{source_name}[{', '.join(source_indices)}]"
    if mask_expr is not None:
        load_expr = f"({load_expr} if {mask_expr} else {source_dtype}(0))"
    store_expr = (
        f"{target_name}.__setitem__({_cute_index_tuple(target_indices)}, "
        f"{env.backend.ast_to_dtype_expr(load_expr, target_dtype)})"
    )
    if mask_expr is not None:
        store_expr = f"{store_expr} if {mask_expr} else None"

    tensor_dim = 0
    for idx in prefix_subscript:
        block_id = None
        if isinstance(idx, torch.SymInt):
            block_id = env.get_block_id(idx)
        elif idx == slice(None) and tensor_dim < tensor.ndim:
            block_id = next(
                (
                    candidate
                    for candidate in _matching_block_ids(env, tensor.shape[tensor_dim])
                    if candidate in state.codegen.active_device_loops
                ),
                None,
            )
        tensor_dim += 1
        if block_id is None:
            continue
        axis = None
        grid_state = state.codegen.current_grid_state
        if grid_state is not None:
            axis = grid_state.block_thread_axes.get(block_id)
        if axis is None:
            loops = state.codegen.active_device_loops.get(block_id)
            if loops:
                axis = loops[-1].block_thread_axes.get(block_id)
        if axis is None or not (0 <= axis < 3):
            continue
        block_size = state.device_function.resolved_block_size(block_id)
        if not isinstance(block_size, int):
            continue
        state.codegen.max_thread_block_dims[axis] = max(
            state.codegen.max_thread_block_dims[axis],
            block_size,
        )
        state.codegen.referenced_thread_block_dims[axis] = max(
            state.codegen.referenced_thread_block_dims[axis],
            block_size,
        )

    stmt: ast.stmt = create(ast.Expr, value=expr_from_string(store_expr))
    for loop_var, source_pos in reversed(
        [*zip(source_loop_vars, range(1, len(source_subscript)), strict=True)]
    ):
        extent = _cute_tensor_dim_size_expr(state, source_tensor, source_pos)
        stmt = create(
            ast.For,
            target=create(ast.Name, id=loop_var, ctx=ast.Store()),
            iter=expr_from_string(f"range({extent})"),
            body=[stmt],
            orelse=[],
            type_comment=None,
        )
    state.add_statement(stmt)
    return ast.Constant(value=None)


def _cute_expand_broadcast_dim(value_node: torch.fx.Node) -> int | None:
    """Return the dim an ``aten.expand`` broadcasts (input size 1 -> >1).

    Returns ``None`` unless ``value_node`` is an ``aten.expand`` whose value has
    exactly one broadcast dimension — i.e. the expanded value carries a stride-0
    mode at exactly one position whose pre-expand extent was 1. This is the
    signal that the stored value replicates one source element across that dim.
    """
    if value_node.target is not torch.ops.aten.expand.default:
        return None
    input_arg = value_node.args[0]
    if not isinstance(input_arg, torch.fx.Node):
        return None
    out_val = value_node.meta.get("val")
    in_val = input_arg.meta.get("val")
    if not isinstance(out_val, torch.Tensor) or not isinstance(in_val, torch.Tensor):
        return None
    if out_val.ndim != in_val.ndim:
        return None
    env = CompileEnvironment.current()
    broadcast_dims = [
        dim
        for dim in range(out_val.ndim)
        if env.known_equal(in_val.shape[dim], 1)
        and not env.known_equal(out_val.shape[dim], 1)
        and out_val.stride(dim) == 0
    ]
    if len(broadcast_dims) != 1:
        return None
    return broadcast_dims[0]


def _cute_block_tile_begin_expr(state: CodegenState, block_id: int) -> str | None:
    """Return the *per-block* tile start for a tile mapped onto a thread axis.

    In the CuTe SIMT model a tile dimension is spread across a thread axis, so
    the strategy's ``index_var`` is the per-*thread* global index
    (``pid * block + thread_idx[axis]``). Subtracting the thread-local coordinate
    yields the per-*block* tile base (``pid * block``), shared by every thread in
    the tile — the correct anchor for a broadcast lane loop. Returns ``None`` when
    the block id has no active thread axis in this scope.
    """
    from .cute_reshape import _grid_local_coord_expr
    from .cute_reshape import _per_thread_nd_tile_offset

    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return None
    loop_state = loops[-1]
    thread_axis = loop_state.block_thread_axes.get(block_id)
    global_index = loop_state.strategy.index_var(block_id)
    if thread_axis is None or global_index is None:
        return None
    tile_offset = _per_thread_nd_tile_offset(loop_state.strategy, block_id)
    if tile_offset is not None:
        return tile_offset
    local_coord = _grid_local_coord_expr(state.codegen, block_id, thread_axis)
    return state.codegen.lift(
        expr_from_string(f"({global_index}) - ({local_coord})"),
        dce=True,
        prefix="tile_begin",
    ).id


def _cute_unsqueeze_expand_load_source(
    value_node: torch.fx.Node, broadcast_dim: int
) -> torch.fx.Node | None:
    """Return the ``hl.load`` feeding ``expand(val[..., None, ...])``.

    Walks ``value_node`` (an ``aten.expand``) back through a single
    unsqueeze-style subscript op (``val[:, None, :]`` inserting the broadcast dim)
    to the originating ``hl.load``. Returns ``None`` unless the chain is exactly
    that shape, so the caller falls back to the load-agnostic path.
    """
    from ...language.view_ops import subscript as subscript_op

    inner = value_node.args[0]
    if not isinstance(inner, torch.fx.Node):
        return None
    if inner.op == "call_function" and inner.target is subscript_op:
        index_arg = inner.args[1] if len(inner.args) > 1 else None
        if not isinstance(index_arg, (list, tuple)):
            return None
        # Exactly one ``None`` (the inserted broadcast dim) at ``broadcast_dim``.
        index_arg_entries = tuple(cast("Any", index_arg))
        none_positions = [
            pos for pos, entry in enumerate(index_arg_entries) if entry is None
        ]
        if none_positions != [broadcast_dim]:
            return None
        load_node = inner.args[0]
    else:
        load_node = inner
    if (
        isinstance(load_node, torch.fx.Node)
        and load_node.op == "call_function"
        and load_node.target is load
        and len(load_node.args) >= 2
    ):
        return load_node
    return None


def _codegen_cute_store_expand_broadcast_tile(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    value: ast.AST,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    """Lower a store whose value is broadcast across a reused tile dimension.

    Handles the pattern::

        val = hl.load(src, [tile, hl.arange(k)])  # (block, k)
        val_3d = val[:, None, :].expand(block, block, k)  # stride-0 middle dim
        hl.store(out, [idx[tile], tile.index, hl.arange(k)], val_3d)

    Here ``tile`` appears twice in the store index — once as a tensor indexer
    (``idx[tile]``) and once as the bare tile index (``tile.index``) — while the
    value is broadcast (stride 0) along the second (``tile.index``) position. The
    generic SIMT store lowers both positions onto ``tile``'s single thread axis,
    so each thread only writes the ``a == b`` diagonal of the ``(block, block)``
    block. Instead emit a sequential lane loop over the broadcast position so a
    thread holding ``val[a]`` writes the full ``out[idx[a], begin+b, :]`` row for
    every ``b`` in the tile, filling the block. ``val`` is broadcast, so every
    lane reads the same per-thread register.

    Returns ``None`` (a strict no-op) unless every gate matches, so existing
    kernels are byte-for-byte unchanged.
    """
    env = CompileEnvironment.current()
    broadcast_dim = _cute_expand_broadcast_dim(value_node)
    if broadcast_dim is None:
        return None
    if broadcast_dim >= len(subscript):
        return None
    broadcast_idx = subscript[broadcast_dim]
    # The broadcast position must be a bare tile index (a SymInt block id), and
    # that same block id must be reused by another (tensor) index position — the
    # collision the generic path mis-handles.
    if not isinstance(broadcast_idx, torch.SymInt):
        return None
    broadcast_block_id = env.get_block_id(broadcast_idx)
    if broadcast_block_id is None:
        return None
    block_size = state.device_function.resolved_block_size(broadcast_block_id)
    if not isinstance(block_size, int) or block_size <= 1:
        return None
    reused = False
    for pos, idx in enumerate(subscript):
        if pos == broadcast_dim:
            continue
        if isinstance(idx, torch.Tensor):
            for dim_size in idx.shape:
                if broadcast_block_id in _matching_block_ids(env, dim_size):
                    reused = True
                    break
        if reused:
            break
    if not reused:
        return None

    # Walk the value chain ``expand -> unsqueeze(None) -> load`` to recover the
    # source load. The stored value is a per-thread register holding ``val[a, c]``
    # whose coordinates live on the *load*'s thread axes; the store's own free
    # ``hl.arange`` index entries are distinct nodes that the synthetic-axis
    # machinery assigns to *different* axes. Reusing the load's coordinate for
    # those non-broadcast positions keeps the register and the store address on
    # the same thread axis (otherwise thread ``(a, c_load, c_store)`` would write
    # ``out[..., c_store] = val[a, c_load]`` for ``c_load != c_store``).
    load_node = _cute_unsqueeze_expand_load_source(value_node, broadcast_dim)
    load_coords: list[str] | None = None
    load_subscript_proxy: tuple[object, ...] | None = None
    if load_node is not None:
        load_tensor_node = load_node.args[0]
        load_subscript = load_node.args[1]
        if isinstance(load_tensor_node, torch.fx.Node) and isinstance(
            load_subscript, (list, tuple)
        ):
            load_tensor = load_tensor_node.meta.get("val")
            if isinstance(load_tensor, torch.Tensor):
                load_subscript_args = tuple(cast("Any", load_subscript))
                load_subscript_proxy = tuple(
                    map_arg(load_subscript_args, lambda arg: arg.meta["val"])
                )
                load_subscript_ast = map_arg(
                    load_subscript_args, lambda arg: state.env[arg]
                )
                load_coords = _cute_index_exprs(
                    state,
                    [*load_subscript_proxy],
                    [*load_subscript_ast],
                    tensor=load_tensor,
                    inactive_singleton_slice_expr="0",
                )
                if len(load_coords) != load_tensor.ndim:
                    load_coords = None
                    load_subscript_proxy = None

    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    if len(index_exprs) != tensor.ndim or "None" in index_exprs:
        return None

    # Re-align each non-broadcast free-``hl.arange`` store position onto the
    # load's matching coordinate. Value dim ``d`` maps to load dim ``d`` before
    # the unsqueezed broadcast dim and ``d - 1`` after it. Only positions where
    # *both* the store and the matching load entry are free ``hl.arange`` index
    # tensors are remapped — a tensor *indexer* (``idx[tile]``) keeps its own
    # coordinate.
    if load_coords is not None and load_subscript_proxy is not None:
        for pos, idx in enumerate(subscript):
            if pos == broadcast_dim or not isinstance(idx, torch.Tensor):
                continue
            load_dim = pos if pos < broadcast_dim else pos - 1
            if not (0 <= load_dim < len(load_coords)):
                continue
            if isinstance(load_subscript_proxy[load_dim], torch.Tensor):
                index_exprs[pos] = load_coords[load_dim]

    # Replace the broadcast position's coordinate (currently the reused tile's
    # per-thread global index) with ``block_begin + lane`` so the lane loop sweeps
    # the full tile block, identically for every thread in the tile. ``block_begin``
    # is the *per-block* tile start (``global_index - local_coord``); in the CuTe
    # SIMT model the tile is mapped onto a thread axis, so the bare offset var
    # still carries the per-thread ``thread_idx`` lane and must be stripped.
    block_begin = _cute_block_tile_begin_expr(state, broadcast_block_id)
    if block_begin is None:
        return None
    lane_var = state.device_function.new_var("bcast_lane", dce=True)
    index_dtype = env.index_type()
    broadcast_coord = f"({block_begin}) + {index_dtype}({lane_var})"
    index_exprs[broadcast_dim] = broadcast_coord

    backend = env.backend
    target_dtype = backend.dtype_str(tensor.dtype)
    tensor_name = state.device_function.tensor_arg(tensor).name
    value = expr_from_string(
        backend.ast_to_dtype_expr("{value}", target_dtype),
        value=value,
    )
    store_expr = expr_from_string(
        _cute_scalar_store_expr(tensor_name, index_exprs, "{value}"),
        value=value,
    )

    # Base mask excludes the broadcast position (its bound is enforced by the lane
    # bound below); other positions keep their tile/tensor masks.
    base_subscript = [
        slice(None) if pos == broadcast_dim else idx
        for pos, idx in enumerate(subscript)
    ]
    mask_expr = _cute_combined_mask(
        state, base_subscript, extra_mask, tensor=tensor, for_store=True
    )
    dim_size = _cute_tensor_dim_size_expr(state, tensor, broadcast_dim)
    lane_bound = f"({broadcast_coord}) < {dim_size}"
    mask_expr = lane_bound if mask_expr is None else f"({mask_expr}) and {lane_bound}"

    from ..ast_extension import create

    mask_ast = expr_from_string(mask_expr)
    assert isinstance(mask_ast, ast.expr)
    assert isinstance(store_expr, ast.expr)
    body_stmt: ast.stmt = ast.fix_missing_locations(
        ast.If(
            test=mask_ast,
            body=[ast.Expr(value=store_expr)],
            orelse=[],
        )
    )
    loop_stmt = create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({block_size})"),
        body=[body_stmt],
        orelse=[],
        type_comment=None,
    )
    state.add_statement(loop_stmt)
    return ast.Constant(value=None)


def _pure_epilogue_ancestors(value_node: torch.fx.Node) -> tuple[torch.fx.Node, ...]:
    """Pure pointwise definitions that may become dead after an epilogue splice.

    Stop at memory operations and graph/loop boundaries. Only making these
    assignments DCE candidates, rather than deleting the FX slice, preserves
    scalar definitions shared with an unfused use.
    """
    pending = [value_node]
    visited: set[torch.fx.Node] = set()
    result = []
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        if node.graph is not value_node.graph or node.op != "call_function":
            continue
        target = node.target
        if isinstance(target, torch._ops.OpOverload):
            if (
                torch.Tag.pointwise not in target.tags
                or torch.Tag.nondeterministic_seeded in target.tags
                or target._schema.is_mutable
                or node.is_impure()
            ):
                continue
        elif target not in _ZERO_ARG_TARGETS:
            continue
        result.append(node)
        pending.extend(node.all_input_nodes)
    return tuple(result)


def _try_splice_tcgen05_unary_epilogue(
    state: CodegenState,
    tensor: object,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None,
) -> ast.AST | None:
    """Splice attempt for ``out[tile] = chain(acc)[.to(x.dtype)]``.

    Returns the splice-completion sentinel (``ast.Constant(value=None)``)
    on a successful splice (the caller should return it directly), and
    ``None`` if the splice did not fire — the caller should continue to
    the loud-failure backstop or the SIMT fallback.

    Splice is attempted only when the kernel has a tcgen05-registered
    matmul fx_node (``cute_state.matmul_fx_nodes`` non-empty), the
    store value has a backing FX node, the store target is a 2-D
    ``torch.Tensor``, and the chain analyzer accepts the value chain
    (returning ``(chain, anchor)`` for a non-empty chain rooted at
    a tcgen05 matmul). Chains the whitelist rejects (broadcast aux
    loads, reductions, kwarg-bearing binaries, etc.) leave the
    analyzer returning ``None`` and the splice does not fire — the
    loud-failure backstop then catches them.
    """
    cute_state = state.device_function.cute_state
    if not cute_state.matmul_fx_nodes:
        return None
    if value_node is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return None
    analyzed = analyze_tcgen05_unary_epilogue_chain(
        state, value_node, output_global_shape=tuple(tensor.shape)
    )
    if analyzed is None:
        return None
    chain, anchor = analyzed
    if not chain.steps:
        return None
    anchor_result_var = cute_state.matmul_fx_node_result_vars.get(anchor)
    if anchor_result_var is None:
        return None
    rewritten_stmt = _codegen_cute_store_tcgen05_tile(
        state,
        tensor,
        subscript,
        ast_subscript,
        extra_mask,
        anchor_result_var,
        epilogue_chain=chain,
    )
    if rewritten_stmt is None:
        return None
    # The splice computes these pure operations from TMEM. Their earlier
    # scalar lowering may use the accumulator's placeholder value, whose
    # dtype intentionally does not track intermediate casts. Remove it only
    # when no actual emitted consumer remains; shared CSE values stay live.
    state.codegen.allow_dead_assignments_owned_by_nodes(
        _pure_epilogue_ancestors(value_node)
    )
    stmts = rewritten_stmt if isinstance(rewritten_stmt, list) else [rewritten_stmt]
    for stmt in stmts:
        state.add_statement(stmt)
    return ast.Constant(value=None)


def _try_codegen_tcgen05_fragment_epilogue(
    state: CodegenState,
    tensor: object,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
) -> ast.AST | None:
    plan = state.device_function.cute_state.tcgen05_fragment_epilogue_plan_for_store(
        state.fx_node
    )
    if plan is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", "planned tcgen05 store is not a tensor")
    from ..inductor_lowering import is_deferred_tcgen05_fragment_epilogue

    value_node = state.fx_node.args[2] if state.fx_node is not None else None
    if value_node is not plan.value_node or not is_deferred_tcgen05_fragment_epilogue(
        state.ast_args[2]
    ):
        raise exc.BackendUnsupported(
            "cute",
            "committed tcgen05 thread-local epilogue received the wrong store value",
        )
    result_var = state.device_function.cute_state.matmul_fx_node_result_vars.get(
        plan.anchor
    )
    if result_var is None:
        raise exc.BackendUnsupported(
            "cute", "tcgen05 thread-local epilogue store ran before its MMA anchor"
        )
    rewritten = _codegen_cute_store_tcgen05_tile(
        state,
        tensor,
        subscript,
        ast_subscript,
        extra_mask,
        result_var,
        fragment_epilogue=plan,
    )
    if rewritten is None:
        raise exc.BackendUnsupported(
            "cute", "committed tcgen05 thread-local epilogue could not render its store"
        )
    for statement in rewritten if isinstance(rewritten, list) else [rewritten]:
        state.add_statement(statement)
    return ast.Constant(value=None)


def _try_splice_tcgen05_grouped_tail_epilogue(
    state: CodegenState,
    tensor: object,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None,
) -> ast.AST | None:
    """Splice grouped preserve-output M/N tail stores into tcgen05."""
    cute_state = state.device_function.cute_state
    if not cute_state.matmul_fx_nodes:
        return None
    if value_node is None or state.fx_node is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return None
    grouped_tail = cute_state.grouped_tail_proof_for_store(state.fx_node)
    if grouped_tail is None:
        return None
    if grouped_tail.store_mask is not None:
        if extra_mask is None or state.fx_node.args[3] is not grouped_tail.store_mask:
            return None
        # The proved grouped scheduler/store extent owns this exact M/N mask.
        # Its producer nodes are removed after the collective store is emitted.
        extra_mask = None
    anchor_result_var = cute_state.matmul_fx_node_result_vars.get(grouped_tail.anchor)
    if anchor_result_var is None:
        return None
    rewritten_stmt = _codegen_cute_store_tcgen05_tile(
        state,
        tensor,
        subscript,
        ast_subscript,
        extra_mask,
        anchor_result_var,
        grouped_tail_epilogue=grouped_tail,
    )
    if rewritten_stmt is None:
        return None
    state.codegen.remove_statements_owned_by_nodes(grouped_tail.producer_nodes)
    stmts = rewritten_stmt if isinstance(rewritten_stmt, list) else [rewritten_stmt]
    for stmt in stmts:
        state.add_statement(stmt)
    return ast.Constant(value=None)


@_decorators.codegen(store, "cute")
def _(state: CodegenState) -> ast.AST:
    def drain_deferred_rebound_checks() -> None:
        # The pointwise checks deferred to the epilogue classifier, for this
        # chain's ancestors only: another chain's store keeps the first word.
        run_deferred_rebound_checks(
            state.codegen,
            _pure_epilogue_ancestors(value_node)
            if isinstance(value_node, torch.fx.Node)
            else (),
        )

    def finish_tcgen05_store() -> None:
        # A tcgen05 store path accepted the chain: a re-bound value has no
        # per-thread element to exchange, and the pointwise checks deferred
        # to the classifier run now.
        if rebound_tcgen05:
            raise tcgen05_rebound_store_error(state, rebound)
        drain_deferred_rebound_checks()

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    raw_value = state.ast_args[2]
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))
    value_node = None
    if state.fx_node is not None and len(state.fx_node.args) > 2:
        maybe_value_node = state.fx_node.args[2]
        if isinstance(maybe_value_node, torch.fx.Node):
            value_node = maybe_value_node

    # Before any store path: a subscript that binds a value dim to another
    # block id needs the exchanged value, whichever path stores it.  On a
    # tcgen05 epilogue chain there is no per-thread value to exchange (the
    # value may still be the deferred fragment-epilogue marker here): the
    # chain is refused where a tcgen05 store path would accept it, and the
    # classifier's own diagnostic wins where it rejects the chain.
    rebound: list[tuple[int, int, int]] = []
    rebound_tcgen05 = False
    exchanged: ast.AST | None = None
    if isinstance(tensor, torch.Tensor):
        rebound = store_rebound_dims(state, tensor, subscript)
    elif isinstance(tensor, tuple):
        # A stack tensor store writes ``value`` through the pointer table's
        # dims and ``tensor_like[subscript]``; it has no exchange, so a
        # re-bound value is refused.
        tensor_like, dev_ptrs = tensor
        rebound = store_rebound_dims(
            state, tensor_like, subscript, leading_sizes=dev_ptrs.shape
        )
        if rebound:
            raise exc.BackendUnsupported(
                "cute",
                f"stack tensor store re-binds {describe_rebound_block_dims(rebound)}; "
                "the value must be held by the thread that owns the destination lane",
            )
    if rebound:
        rebound_tcgen05 = bool(
            value_node is not None
            and state.device_function.cute_state.matmul_fx_nodes
            and reach_tcgen05_matmul_anchors(state, value_node)
        )
        if not rebound_tcgen05:
            assert isinstance(tensor, torch.Tensor)
            exchanged = codegen_cute_store_rebound_value(
                state, tensor, subscript, state.ast_arg(2), rebound
            )
            raw_value = exchanged
    if (
        planned := _try_codegen_tcgen05_fragment_epilogue(
            state, tensor, subscript, ast_subscript, extra_mask
        )
    ) is not None:
        finish_tcgen05_store()
        return planned

    if isinstance(tensor, torch.Tensor):
        # An exchanged value must not be rebuilt from its FX node: the paths
        # below see no node for it and store ``raw_value`` (= the exchange).
        store_value_node = value_node if exchanged is None else None
        affine_range_store = _codegen_cute_affine_range_store(
            state,
            tensor,
            subscript,
            ast_subscript,
            raw_value,
            extra_mask,
            store_value_node,
        )
        if affine_range_store is not None:
            state.add_statement(affine_range_store)
            return ast.Constant(value=None)
        # The paths below rebuild the value from its FX node; an exchanged
        # value must take the generic store, which writes ``exchanged``.
        affine_reshape_store = (
            None
            if exchanged is not None
            else _codegen_cute_affine_reshape_store(
                state,
                tensor,
                subscript,
                ast_subscript,
                extra_mask,
                value_node,
            )
        )
        if affine_reshape_store is not None:
            state.add_statement(affine_reshape_store)
            return ast.Constant(value=None)
        strided_slice_store = _codegen_cute_strided_slice_store(
            state,
            tensor,
            subscript,
            raw_value,
            extra_mask,
            store_value_node,
        )
        if strided_slice_store is not None:
            state.add_statement(strided_slice_store)
            return ast.Constant(value=None)

    value = exchanged if exchanged is not None else state.ast_arg(2)

    if value_node is not None and exchanged is None:
        if value_node.op == "call_function":
            if isinstance(tensor, torch.Tensor):
                rewritten_stmt = _codegen_cute_store_stack_load(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    value,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
                rewritten_stmt = _codegen_cute_store_loaded_index_trailing_slices(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
                rewritten_stmt = _codegen_cute_store_expand_broadcast_tile(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    value,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
                rewritten_stmt = _codegen_cute_store_reshape_lane_loops(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    value,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt

    if isinstance(tensor, tuple):
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        _tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        assert isinstance(dev_ptrs_ast, ast.AST)
        tensor_like, dev_ptrs = tensor
        offset_expr = _cute_stack_tensor_offset_expr(
            state,
            tensor_like,
            [*subscript],
            ast_subscript,
        )
        backend = CompileEnvironment.current().backend
        target_dtype = backend.dtype_str(tensor_like.dtype)
        value = expr_from_string(
            backend.ast_to_dtype_expr("{value}", target_dtype),
            value=value,
        )
        ptr_expr = _cute_stack_tensor_pointer_expr(
            target_dtype, dev_ptrs_ast, offset_expr
        )
        store_expr = expr_from_string(
            "({ptr}).store({value})", ptr=ptr_expr, value=value
        )
        mask_expr = _cute_stack_tensor_mask_expr(
            state,
            tensor_like,
            dev_ptrs,
            [*subscript],
            extra_mask,
            for_store=True,
        )
        if mask_expr is None:
            return store_expr
        mask_ast = expr_from_string(mask_expr)
        assert isinstance(mask_ast, ast.expr)
        assert isinstance(store_expr, ast.expr)
        state.add_statement(
            ast.fix_missing_locations(
                ast.If(
                    test=mask_ast,
                    body=[ast.Expr(value=store_expr)],
                    orelse=[],
                )
            )
        )
        return ast.Constant(value=None)
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", f"store target type: {type(tensor)}")

    _log_cute_layout(state, "store")

    if isinstance(value, ast.Name):
        rewritten_stmt = _codegen_cute_store_tcgen05_tile(
            state,
            tensor,
            subscript,
            ast_subscript,
            extra_mask,
            value.id,
        )
        if rewritten_stmt is not None:
            stmts = (
                rewritten_stmt if isinstance(rewritten_stmt, list) else [rewritten_stmt]
            )
            finish_tcgen05_store()
            for stmt in stmts:
                state.add_statement(stmt)
            return ast.Constant(value=None)

    # Try to splice a whitelisted chain epilogue
    # (`out[tile] = chain(acc)[.to(x.dtype)]`) into the role-local
    # tcgen05 epilogue's per-thread T2R loop. Implementation in
    # ``_try_splice_tcgen05_unary_epilogue``. Chains the whitelist
    # rejects (broadcast aux loads, reductions, etc.) leave the
    # splice off and fall through to the loud-failure backstop
    # below.
    spliced = _try_splice_tcgen05_unary_epilogue(
        state, tensor, subscript, ast_subscript, extra_mask, value_node
    )
    if spliced is not None:
        finish_tcgen05_store()
        return spliced
    spliced = _try_splice_tcgen05_grouped_tail_epilogue(
        state, tensor, subscript, ast_subscript, extra_mask, value_node
    )
    if spliced is not None:
        finish_tcgen05_store()
        return spliced

    # Loud-failure backstop for fused-epilogue stores that follow a
    # tcgen05 matmul. The tcgen05 grid-emission path (in `program_id.py`)
    # does not bind the per-block-id `indices_<n>` / `mask_<n>` variable
    # names that the SIMT-fallback store path expects, so falling through
    # here would emit a kernel that crashes inside the cute DSL with
    # `name 'mask_0' is not defined`. Detect the pattern here — any
    # store value whose FX user chain transitively reaches a
    # tcgen05-registered matmul fx node — and raise a structured error
    # so the caller sees the actionable message instead of a cute-DSL
    # crash. Fixing this requires either (a) extending the tcgen05 grid
    # to emit per-block-id index/mask vars, or (b) per-subtile lambda
    # emission in `_codegen_cute_store_tcgen05_tile`.
    if (
        state.device_function.cute_state.matmul_fx_nodes
        and value_node is not None
        and reach_tcgen05_matmul_anchors(state, value_node)
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 MMA path does not yet emit per-block-id indices "
            "and masks for non-whitelisted fused epilogues that follow "
            "the MMA. The store target's value chain depends on a "
            "tcgen05 matmul result through ops the chain analyzer "
            "rejects (e.g. aux tensors with a 3-D underlying shape "
            "and a static collapse like `aux3d[tile_m, tile_n, 0]`, "
            "loads whose index expression is not exactly the "
            "carrier tile-id symbol, non-scalar binary ops, "
            "`aten.add.Tensor` with `alpha=k`, or an intermediate "
            "`.to(d_inter)` cast where `d_inter` differs from the "
            "store-target dtype). Identity stores "
            "(`out[tile] = acc.to(x.dtype)`), whitelisted unary chains "
            "(relu/tanh/exp/log/sqrt/abs/neg + scalar add/sub/mul/div "
            "on the accumulator carrier), exact-shape 2-D "
            "auxiliary-tensor binary ops (`acc + residual[tile_m, "
            "tile_n]`), rank-1 trailing-axis (rowvec) broadcast "
            "aux loads (`acc + bias[tile_n]`), and tile-uniform scalars "
            "(`alpha * acc` with a captured Python float, "
            "`acc * scale[()]`) all work via the "
            "fused-epilogue splice path. The leading-axis rank-1 "
            "form (`acc + bias[tile_m]`) is rejected because a bare "
            "rank-1 RHS aligns to the trailing axis under PyTorch "
            "broadcasting; an explicit colvec broadcast must be "
            "written with `bias[tile_m][:, None]` / "
            "`.unsqueeze(-1)`.",
        )

    drain_deferred_rebound_checks()

    tensor_name = state.device_function.tensor_arg(tensor).name
    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(tensor.dtype)
    value = expr_from_string(
        backend.ast_to_dtype_expr("{value}", target_dtype),
        value=value,
    )
    assert isinstance(value, ast.expr)
    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    # Regions come from the subscript as written: a re-addressed full-slice
    # dim (below) stays inside the slice's extent, so they remain an
    # over-approximation of the elements this store touches.
    regions = _cute_access_regions(state, subscript, tensor)
    mask_subscript: list[object] | tuple[object, ...] = subscript
    if state.fx_node is not None and len(state.fx_node.args) > 2:
        mask_subscript = _apply_cute_value_coord_meta(
            state, tensor, subscript, index_exprs, state.fx_node.args[2]
        )
    value_readdressed = mask_subscript != list(subscript)
    topk_lane_expr: object | None = None
    topk_k: object | None = None
    if state.fx_node is not None and len(state.fx_node.args) > 2:
        value_node = state.fx_node.args[2]
        if (
            isinstance(value_node, torch.fx.Node)
            and value_node.target is operator.getitem
            and isinstance(value_node.args[0], torch.fx.Node)
            and value_node.args[0].target is torch.ops.aten.topk.default
        ):
            topk_lane_expr = value_node.args[0].meta.get("cute_topk_lane_expr")
            topk_k = value_node.args[0].meta.get("cute_topk_k")
    if isinstance(topk_lane_expr, str) and isinstance(topk_k, int):
        index_exprs[-1] = topk_lane_expr
    store_uses_pointer = "None" not in index_exprs
    mask_expr = _cute_combined_mask(
        state, mask_subscript, extra_mask, tensor=tensor, for_store=True
    )
    branch_vec_store_candidate: tuple[int, int] | None = None

    # Vectorized store: when this store's stride-1 axis is a vec-partitioned
    # lane loop (same predicates as the vec-load hoist, so the per-element
    # mask is provably uniform across the V lanes), collect the per-lane
    # values and emit one ST.64/ST.128 after the constexpr V-loop instead of
    # V scalar 2-byte stores.
    if (
        store_uses_pointer
        and topk_lane_expr is None
        and not value_readdressed
        and extra_mask is None
        and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        vec_ctx = _cute_vector_load_ctx(state, tensor, subscript, index_exprs, None)
        if vec_ctx is not None and vec_ctx[2] == "tile_unroll":
            from ..tile_strategy import BlockSizeTileStrategy

            _vec_width, vec_block_id, _mode = vec_ctx
            strategy = _cute_lane_strategy(state, vec_block_id)
            assert isinstance(strategy, BlockSizeTileStrategy)
            from .signed_bitfield import packed_store_value
            from .signed_bitfield import signed_byte_site

            lane_axis_pos = _cute_lane_axis_pos(strategy, vec_block_id, index_exprs)
            # A grid-owned flush is emitted when the root body is wrapped, after
            # the deferred hoist has recorded its signed-byte packet. Resolve the
            # packed value then, but prove scope with this store's own site.
            store_site = signed_byte_site(state)
            scalar_store = statement_from_string(
                _cute_scalar_store_expr(tensor_name, index_exprs, "{value}"),
                value=value,
            )
            _cute_tag_access_regions(scalar_store, tensor_name, regions)
            if mask_expr is not None:
                scalar_store = statement_from_string(
                    f"if {mask_expr}:\n    {{store}}", store=scalar_store
                )

            def emit_tile_store() -> ast.AST | None:
                packed_values = packed_store_value(
                    state,
                    strategy,
                    vec_block_id,
                    _vec_width,
                    tensor,
                    index_exprs,
                    mask_expr,
                    site=store_site,
                )
                return _cute_register_tile_unroll_vec_store(
                    state,
                    strategy,
                    vec_block_id,
                    tensor_name,
                    index_exprs,
                    ast.unparse(value),
                    mask_expr,
                    tensor.dtype,
                    scalar_stmt=scalar_store,
                    lane_axis_pos=lane_axis_pos,
                    packed_values=packed_values,
                )

            if _cute_defer_grid_vector_op(
                state, strategy, vec_block_id, scalar_store, emit_tile_store
            ):
                state.add_statement(scalar_store)
                return ast.Constant(value=None)
            append_stmt = emit_tile_store()
            if append_stmt is not None:
                state.add_statement(append_stmt)
                return ast.Constant(value=None)
        elif vec_ctx is not None and vec_ctx[2] == "unroll":
            from ..reduction_strategy import LoopedReductionStrategy
            from ..reduction_strategy import PersistentReductionStrategy

            _vec_width, vec_block_id, _mode = vec_ctx
            red_strategy = _cute_lane_strategy(state, vec_block_id)
            append_stmt = None
            if isinstance(
                red_strategy,
                (LoopedReductionStrategy, PersistentReductionStrategy),
            ):
                can_vectorize = True
                if isinstance(red_strategy, PersistentReductionStrategy):
                    # Delay persistent stores until the complete lane body is
                    # visible. The late pass keeps interleaved stores in place
                    # when a later potentially-aliasing load needs ordering.
                    can_vectorize = False
                    if _persistent_vec_is_exact_aligned(
                        state,
                        red_strategy,
                        index_exprs,
                        tensor,
                        _vec_width,
                    ):
                        branch_vec_store_candidate = (vec_block_id, _vec_width)
                tail_store: ast.stmt | None = None
                flush_mask = mask_expr
                if can_vectorize:
                    assert isinstance(red_strategy, LoopedReductionStrategy)
                    mask_var = red_strategy._mask_var
                    if mask_var is not None and not _cute_looped_vec_mask_is_uniform(
                        red_strategy, mask_expr, mask_var
                    ):
                        # A per-element outer term cannot predicate the
                        # whole flush; keep the scalar stores.
                        can_vectorize = False
                    elif (
                        mask_var is not None
                        and red_strategy._cute_reduction_chunk_full_var is not None
                    ):
                        # Unknown ``numel % V``: full chunks flush one vector,
                        # the straddling tail chunk stores its in-bounds
                        # elements individually.
                        chunk_full = red_strategy._cute_reduction_chunk_full_var
                        non_lane = (
                            _cute_vector_load_non_lane_mask(mask_expr, mask_var)
                            if mask_expr is not None
                            else None
                        )
                        flush_mask = (
                            f"({non_lane}) and ({chunk_full})"
                            if non_lane is not None
                            else chunk_full
                        )
                        tail_predicate = (
                            f"not {chunk_full} and ({mask_expr})"
                            if mask_expr is not None
                            else f"not {chunk_full}"
                        )
                        tail_store = statement_from_string(
                            f"if {tail_predicate}:\n    {{store}}",
                            store=statement_from_string(
                                _cute_scalar_store_expr(
                                    tensor_name, index_exprs, "{value}"
                                ),
                                value=value,
                            ),
                        )
                if can_vectorize:
                    assert isinstance(red_strategy, LoopedReductionStrategy)
                    append_stmt = _cute_register_reduction_unroll_vec_store(
                        state,
                        red_strategy,
                        tensor_name,
                        index_exprs,
                        ast.unparse(value),
                        flush_mask,
                        tensor.dtype,
                    )
                if append_stmt is not None:
                    state.add_statement(append_stmt)
                    if tail_store is not None:
                        state.add_statement(tail_store)
                    return ast.Constant(value=None)

    if branch_vec_store_candidate is not None:
        vec_block_id, vec_width = branch_vec_store_candidate
        state.add_statement(
            _persistent_branch_vec_store_marker(
                vec_block_id,
                vec_width,
                tensor.dtype,
                _cute_scalar_pointer_expr(tensor_name, index_exprs),
                value,
                mask_expr,
            )
        )
        return ast.Constant(value=None)

    store_expr = _cute_scalar_store_expr(tensor_name, index_exprs, "{value}")
    assign_expr = expr_from_string(store_expr, value=value)
    _cute_tag_access_regions(assign_expr, tensor_name, regions)
    if isinstance(topk_lane_expr, str) and isinstance(topk_k, int):
        topk_mask = f"({topk_lane_expr}) < {topk_k}"
        mask_expr = topk_mask if mask_expr is None else f"({mask_expr}) and {topk_mask}"
    if mask_expr is None:
        return assign_expr
    if store_uses_pointer:
        mask_ast = expr_from_string(mask_expr)
        assert isinstance(mask_ast, ast.expr)
        assert isinstance(assign_expr, ast.expr)
        state.add_statement(
            ast.fix_missing_locations(
                ast.If(
                    test=mask_ast,
                    body=[ast.Expr(value=assign_expr)],
                    orelse=[],
                )
            )
        )
        return ast.Constant(value=None)
    return expr_from_string(
        f"({store_expr} if {mask_expr} else None)",
        value=value,
    )


def _cute_load_feeds_sort_or_scan(load_node: object) -> bool:
    """Return True if ``load_node`` feeds a sort/topk/_associative_scan.

    Direct users (sort/topk and the scalar ``_associative_scan`` path) are
    matched immediately.  For a tuple ``_associative_scan`` the index stream is
    typically a ``load`` that flows through a chain of dtype-cast / shape ops
    (e.g. ``indices[tile].float().unsqueeze(1).expand_as(vals)``) before
    reaching the scan.  To recover a scalar load for that stream we follow the
    forward chain through those pass-through ops.
    """
    from torch.fx.node import Node

    from .indexing import is_cute_shape_chain_target

    if not isinstance(load_node, Node):
        return False

    passthrough_targets = (torch.ops.prims.convert_element_type.default,)
    seen: set[Node] = set()
    stack: list[Node] = [load_node]
    while stack:
        node = stack.pop()
        for user in node.users:
            if not isinstance(user, Node):
                continue
            target = user.target
            if (
                target in (torch.ops.aten.sort.default, torch.ops.aten.topk.default)
                or getattr(target, "__name__", None) == "_associative_scan"
            ):
                return True
            if (
                is_cute_shape_chain_target(target) or target in passthrough_targets
            ) and user not in seen:
                seen.add(user)
                stack.append(user)
    return False


def _cute_flat_multi_cover_ok(
    env: CompileEnvironment,
    strategy: object,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
) -> bool:
    """True when a flat base pointer equals the per-dim indexed pointer for
    every element of a flattened multi-dim tile: the tensor is row-major
    contiguous, its dims match the iteration dims in order, and the
    strategy's div/mod decomposition follows row-major order."""
    block_ids = list(strategy.block_ids)  # pyrefly: ignore
    # reorder[0] (the ``offsets % n`` fastest-varying dim) must be the LAST
    # block: the identity loop_order reverses into exactly that.
    if list(getattr(strategy, "loop_order", [])) != list(range(len(block_ids))):
        return False
    sub_blocks: list[int | None] = []
    for idx in subscript:
        if idx is None:
            continue
        sub_blocks.append(
            env.get_block_id(idx) if isinstance(idx, torch.SymInt) else None
        )
    if sub_blocks != block_ids:
        return False
    if tensor.ndim != len(block_ids):
        return False
    expected_stride = 1
    for d in reversed(range(tensor.ndim)):
        stride_d = tensor.stride(d)
        size_d = tensor.shape[d]
        if not isinstance(stride_d, int) or not isinstance(size_d, int):
            return False
        if stride_d != expected_stride:
            return False
        try:
            block_numel = int(env.block_sizes[block_ids[d]].numel)
        except (TypeError, ValueError):
            return False
        if size_d != block_numel:
            return False
        expected_stride *= size_d
    return True


def _cute_tile_unroll_scope_safe(
    state: CodegenState,
    strategy: object,
    block_id: int,
    index_exprs: list[str],
    lane_axis_pos: int,
) -> bool:
    """Require one memory operation per V lane, with a dominating address.

    Tile-vector loads are inserted before the V-loop and stores after it.
    A nested loop/branch changes execution count or strands its indices below
    that boundary. Keep such sites scalar instead of moving their operations.
    """
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import DeviceLoopState

    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return False
    owner = loops[-1]
    stack = state.codegen.statements_stack
    body = stack[-1]
    vloop = getattr(strategy, "_cute_lane_vloop_by_block", {}).get(block_id)
    if not isinstance(vloop, ast.For):
        return False
    if isinstance(owner, DeviceGridState):
        if len(stack) < 2 or stack[-2] is not owner.hoist_parent_statements:
            return False
        if not owner.lane_loops:
            return False
        if not any(
            wrapper.vloop is vloop for wrapper in owner.vec_lane_wrappers.values()
        ):
            return False
    elif isinstance(owner, DeviceLoopState):
        if body is not owner.inner_statements or vloop.body is not body:
            return False
    else:
        return False
    # The vectorized axis is replaced by its already-defined lane base.
    # Other coordinates must not depend on definitions inside this V-loop.
    reads = {
        name
        for axis, expression in enumerate(index_exprs)
        if axis != lane_axis_pos
        for name in ReadWrites.from_ast(ast.parse(expression, mode="eval")).reads
    }
    local_writes = {name for stmt in body for name in ReadWrites.from_ast(stmt).writes}
    return not (reads & local_writes)


def _cute_defer_grid_vector_op(
    state: CodegenState,
    strategy: object,
    block_id: int,
    scalar: ast.AST,
    emit: Callable[[], ast.AST | None],
) -> bool:
    from ..tile_strategy import DeviceGridState

    owner = state.codegen.active_device_loops[block_id][-1]
    if not isinstance(owner, DeviceGridState):
        return False
    vloop = getattr(strategy, "_cute_lane_vloop_by_block", {})[block_id]
    owner.deferred_vector_ops.append((vloop, scalar, emit))
    return True


def _cute_vector_load_mask_is_lane_only(
    mask_expr: str | None, lane_mask: str | None
) -> bool:
    """The hoist's anchor pointer only protects its own vectorized axis.

    Other masks can invalidate an outer row or gather address even when the
    contiguous vector lies wholly within its column extent. Keep those loads
    behind their original scalar predicate; masking their extracted values
    after an unconditional vector load does not protect memory accesses.
    """
    if mask_expr is None:
        return True

    def lane_only(node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id == lane_mask
        if isinstance(node, ast.Constant):
            return node.value is True
        return (
            isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.And)
            and all(lane_only(value) for value in node.values)
        )

    return lane_only(ast.parse(mask_expr, mode="eval").body)


def _cute_reduction_masked_vec_ok(strategy: object, tensor: torch.Tensor) -> bool:
    """Whether a masked rolled-reduction lattice may still emit vector packets.

    Only the ``unroll`` protocol of ``LoopedReductionStrategy`` knows how to
    predicate a packet on its chunk (``codegen_device_loop`` placed either a
    chunk-level mask or the whole-chunk predicates in the lane body).  The
    persistent strategy keeps its own late branch-local vectorizer.
    """
    from ..reduction_strategy import LoopedReductionStrategy

    if not isinstance(strategy, LoopedReductionStrategy):
        return False
    if strategy._cute_reduction_vec_mode != "unroll":
        return False
    if tensor.dtype not in _CUTE_VECTOR_UNROLL_DTYPES:
        return False
    if strategy._cute_lane_base_index_var is None:
        return False
    return (
        strategy._cute_reduction_chunk_uniform_mask
        or strategy._cute_reduction_chunk_full_var is not None
    )


def _cute_looped_vec_mask_is_uniform(
    strategy: LoopedReductionStrategy, mask_expr: str | None, lane_mask: str | None
) -> bool:
    """Whether the non-lane mask terms hold one value for the whole V-chunk.

    An outer row mask is defined by the grid prefix and is uniform across the
    chunk, so it may predicate the packet pointer and the vector flush.  A
    term reading a value defined inside the lane body (a per-element gather
    bound) is not, and such a site keeps its scalar accesses.
    """
    if mask_expr is None:
        return True
    non_lane = _cute_vector_load_non_lane_mask(mask_expr, lane_mask)
    if non_lane is None:
        return True
    lane_body = strategy._cute_lane_body
    if lane_body is None:
        return False
    written: set[str] = set()
    for stmt in lane_body:
        written.update(ReadWrites.from_ast(stmt).writes)
    reads = set(ReadWrites.from_ast(expr_from_string(non_lane)).reads)
    return not (reads & written)


def _cute_vector_load_non_lane_mask(
    mask_expr: str, lane_mask: str | None
) -> str | None:
    """Conjunction of the mask terms other than the lane mask (None if none).

    A surviving term (an outer tile mask, the ``index < size`` bound of a
    gathered coordinate) still predicates a whole V-wide packet once
    ``_cute_tile_unroll_uniform_definitions`` proves it reads only values
    available above the constexpr V-loop.
    """
    terms: list[str] = []

    def collect(node: ast.expr) -> None:
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            for value in node.values:
                collect(value)
        elif not (
            (isinstance(node, ast.Name) and node.id == lane_mask)
            or (isinstance(node, ast.Constant) and node.value is True)
        ):
            terms.append(ast.unparse(node))

    collect(ast.parse(mask_expr, mode="eval").body)
    if not terms:
        return None
    return " and ".join(f"({term})" for term in terms)


@dataclasses.dataclass(frozen=True)
class _CuteLaneIndexFacts:
    """How the vectorized block's per-element index relates to its lane base.

    The lane setup defines ``index_var = base_var + cutlass.Int32(vec_lane_var)``
    for the constexpr lane ``vec_lane_var`` in ``[0, vec_width)``.  ``aligned``
    records that ``base_var`` is provably a multiple of ``vec_width`` (the tile
    begins at a multiple of V and the block, the per-thread span and the lane
    stride are multiples of V), so a predicate on ``index_var`` whose truth
    cannot change inside an aligned V-wide span can be evaluated once at the
    base.
    """

    index_var: str
    base_var: str
    vec_lane_var: str
    vec_width: int
    aligned: bool


@dataclasses.dataclass(frozen=True)
class CuteLaneRelocation:
    """Body definitions a packet hoist reads above a lane loop's V-loop.

    ``statements`` are V-loop body statements (a gathered row index, a mask
    term) whose values the hoisted packet load in ``lane_body`` needs before
    the constexpr V-loop ``vloop``.  A device loop applies the relocation at
    once.  A grid body is placed around its lane loops later by
    ``DeviceGridState.wrap_body``, which leaves the definitions in the body
    when its placement emits them before the loop and applies the relocation
    otherwise.
    """

    lane_body: list[ast.AST]
    vloop: ast.For
    statements: list[ast.AST]

    def apply(self, body: list[ast.AST]) -> None:
        """Move the statements still in ``body`` above their first reader."""
        moved = [
            statement
            for statement in self.statements
            if any(statement is candidate for candidate in body)
        ]
        if not moved:
            return
        for statement in moved:
            body.remove(statement)
        written = {
            name
            for statement in moved
            for name in ReadWrites.from_ast(statement).writes
        }
        position = len(self.lane_body)
        for index, statement in enumerate(self.lane_body):
            if (
                statement is self.vloop
                or set(ReadWrites.from_ast(statement).reads) & written
            ):
                position = index
                break
        self.lane_body[position:position] = moved


@dataclasses.dataclass(frozen=True)
class _CuteVloopScope:
    """The statement list a tile-vector site lowers into.

    ``unavailable`` holds the names that only exist per V lane (the constexpr
    lane variable and, for a grid, the lane setup ``wrap_body`` will place
    inside this V-loop).  ``deferred`` is the grid's deferred vector-op list,
    whose scalar placeholders must not be relocated.  ``lane_body`` is the
    outer lane body holding ``vloop``; ``lane`` describes the per-element
    lane index when its setup has the recognized form.  ``relocations`` is
    the grid's pending-relocation list (``None`` for a device loop, whose
    relocations apply at once).
    """

    body: list[ast.AST]
    unavailable: frozenset[str]
    deferred: list[tuple[ast.For, ast.AST, Callable[[], ast.AST | None]]]
    lane_body: list[ast.AST]
    vloop: ast.For
    lane: _CuteLaneIndexFacts | None = None
    relocations: list[CuteLaneRelocation] | None = None


@dataclasses.dataclass(frozen=True)
class _CuteTileUnrollHoist:
    """What a tile-vector load needs above the V-loop.

    ``moved`` are the V-loop body statements to relocate above the loop,
    ``uniform_mask`` the packet guard re-expressed at the lane base and
    ``lane_base_expr`` the lane-axis coordinate at the lane base (``None``
    when the coordinate is the block's own index and the base var applies).
    """

    moved: list[ast.AST]
    uniform_mask: str | None
    lane_base_expr: str | None


def _cute_grid_lane_setup_names_from_depth(
    grid: DeviceGridState, depth: int
) -> set[str]:
    """Lane-setup names ``DeviceGridState.wrap_body`` defines at ``depth`` or deeper.

    Replays the placement rule of ``wrap_body``: a setup statement lives in the
    deepest lane scope whose lane / base / constexpr-lane variable it reads
    (transitively through earlier setup), and in the innermost scope when it
    reads none.  Definitions at the vectorized lane's own depth land inside
    its V-loop, so a packet hoisted above that loop cannot read them.
    """
    lane_scope_names: list[set[str]] = []
    for lane_var, _extent in grid.lane_loops:
        names = {lane_var}
        wrapper = grid.vec_lane_wrappers.get(lane_var)
        if wrapper is not None:
            names.update((wrapper.vec_lane_var, wrapper.base_index_var))
        lane_scope_names.append(names)
    innermost = len(grid.lane_loops) - 1
    setup_depths: dict[str, int] = {}
    result: set[str] = set()
    for statement in grid.lane_setup_statements:
        rw = ReadWrites.from_ast(statement)
        reads = set(rw.reads)
        depths = [
            index for index, names in enumerate(lane_scope_names) if reads & names
        ]
        depths.extend(setup_depths[name] for name in reads if name in setup_depths)
        statement_depth = max(depths) if depths else innermost
        for name in rw.writes:
            setup_depths[name] = statement_depth
            if statement_depth >= depth:
                result.add(name)
    return result


def _cute_tile_unroll_vloop_scope(
    state: CodegenState, strategy: object, block_id: int
) -> _CuteVloopScope | None:
    """Locate the V-loop body of ``block_id``'s lane loop at the current site.

    Same structural requirements as ``_cute_tile_unroll_scope_safe``: the
    current statement list is the V-loop body (device loop) or the grid body
    that ``DeviceGridState.wrap_body`` later plugs into the pre-built V-loop.
    """
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import DeviceLoopState
    from ..tile_strategy import PerThreadFlattenedTileStrategy
    from ..tile_strategy import PerThreadNDTileStrategy

    if not isinstance(
        strategy, (PerThreadNDTileStrategy, PerThreadFlattenedTileStrategy)
    ):
        return None
    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return None
    owner = loops[-1]
    stack = state.codegen.statements_stack
    body = stack[-1]
    vloop = strategy._cute_lane_vloop_by_block.get(block_id)
    vec_lane_var = strategy._cute_vec_lane_var_by_block.get(block_id)
    if vloop is None or vec_lane_var is None:
        return None
    unavailable = {vec_lane_var}
    deferred: list[tuple[ast.For, ast.AST, Callable[[], ast.AST | None]]] = []
    relocations: list[CuteLaneRelocation] | None = None
    if isinstance(owner, DeviceGridState):
        if len(stack) < 2 or stack[-2] is not owner.hoist_parent_statements:
            return None
        depth = None
        for index, (lane_var, _extent) in enumerate(owner.lane_loops):
            wrapper = owner.vec_lane_wrappers.get(lane_var)
            if wrapper is not None and wrapper.vloop is vloop:
                depth = index
        if depth is None:
            return None
        unavailable |= _cute_grid_lane_setup_names_from_depth(owner, depth)
        deferred = owner.deferred_vector_ops
        relocations = owner.pending_relocations
    elif isinstance(owner, DeviceLoopState):
        if body is not owner.inner_statements or vloop.body is not body:
            return None
    else:
        return None
    setup = owner.lane_setup_statements if isinstance(owner, DeviceGridState) else body
    return _CuteVloopScope(
        body,
        frozenset(unavailable),
        deferred,
        strategy._cute_lane_body_by_block[block_id],
        vloop,
        _cute_lane_index_facts(state, strategy, block_id, owner, setup),
        relocations,
    )


def _cute_lane_index_facts(
    state: CodegenState,
    strategy: CuteLaneTileStrategy,
    block_id: int,
    owner: DeviceLoopOrGridState,
    setup: list[ast.AST],
) -> _CuteLaneIndexFacts | None:
    """Recognize ``index = base + cutlass.Int32(vec_lane)`` in the lane setup."""
    from ..tile_strategy import _plain_assignment_name

    index_var = _cute_active_index_var(state, block_id)
    base_var = strategy._cute_lane_base_index_var_by_block.get(block_id)
    vec_lane_var = strategy._cute_vec_lane_var_by_block.get(block_id)
    vec_width = strategy._cute_lane_vec_width_by_block.get(block_id, 1)
    if index_var is None or base_var is None or vec_lane_var is None or vec_width <= 1:
        return None
    definitions = [
        statement
        for statement in setup
        if _plain_assignment_name(statement) == index_var
    ]
    if len(definitions) != 1:
        return None
    definition = definitions[0]
    assert isinstance(definition, ast.Assign)
    if ast.unparse(definition.value) != f"{base_var} + cutlass.Int32({vec_lane_var})":
        return None
    return _CuteLaneIndexFacts(
        index_var,
        base_var,
        vec_lane_var,
        vec_width,
        _cute_lane_base_is_vec_aligned(strategy, block_id, owner, vec_width),
    )


def _cute_lane_base_is_vec_aligned(
    strategy: CuteLaneTileStrategy,
    block_id: int,
    owner: DeviceLoopOrGridState,
    vec_width: int,
) -> bool:
    """Whether every per-thread lane base of ``block_id`` is a multiple of V.

    The base is ``tile_offset + thread * EPT + lane * V`` (or the strided
    ``tile_offset + (lane * NT + thread) * V``): a multiple of V when the tile
    begins at a multiple of V (zero for ``hl.tile(size)``), the block and any
    cluster slice are multiples of V and the per-thread span is too.  The
    flattened strategy derives its per-dim indices from a flat base and is
    not covered.
    """
    from ..tile_strategy import PerThreadNDTileStrategy

    env = CompileEnvironment.current()
    if not isinstance(strategy, PerThreadNDTileStrategy):
        return False
    if block_id not in strategy.block_ids:
        return False
    block_size = strategy.block_size
    if isinstance(block_size, (list, tuple)):
        block_size = block_size[strategy.block_ids.index(block_id)]
    static_block = strategy._configured_block_size_int(block_size)
    if static_block is None or static_block % vec_width != 0:
        return False
    if strategy._elements_per_thread_for_block(block_id) % vec_width != 0:
        return False
    cluster_n = strategy._cute_cluster_by_block.get(block_id, 1)
    if cluster_n > 1 and (static_block // cluster_n) % vec_width != 0:
        return False
    info = owner.block_id_to_info.get(block_id)
    if info is None:
        return False
    if info.begin_expr is None:
        # A grid without an explicit begin starts at zero; a lifted begin
        # without a symbolic value is data dependent.
        return info.begin_var_name is None
    return info.begin_expr == 0 or env.known_multiple(info.begin_expr, vec_width)


def _cute_names_read(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _cute_clone_expr(node: ast.AST) -> ast.expr:
    return ast.parse(ast.unparse(node), mode="eval").body


def _cute_single_definitions(
    body: list[ast.AST], *, exclude: str | None = None
) -> dict[str, ast.expr]:
    """Values the V-loop body assigns exactly once by a pure, load-free statement.

    Inlining a value duplicates its computation into the packet guard that
    runs above the V-loop.  A load or an atomic result there would escape the
    checks a relocated statement gets (purity, position before the first
    effect, tensors the body writes), so such names stay names: the guard
    then reads them and the relocation pass proves or rejects their
    definitions.  ``exclude`` keeps the per-element lane index a name as well
    (a device loop defines it in the body) so the uniformity proof sees it.
    """
    from ..tile_strategy import _is_proven_relocatable_assignment
    from ..tile_strategy import _plain_assignment_name

    definitions: dict[str, ast.expr] = {}
    opaque: set[str] = set()
    for statement in body:
        plain = _plain_assignment_name(statement)
        for name in ReadWrites.from_ast(statement).writes:
            if (
                name in definitions
                or name in opaque
                or plain != name
                or name == exclude
                or not _is_proven_relocatable_assignment(statement, allow_load=False)
            ):
                opaque.add(name)
                definitions.pop(name, None)
            else:
                assert isinstance(statement, ast.Assign)
                definitions[name] = statement.value
    return definitions


class _CuteInlineDefinitions(ast.NodeTransformer):
    """Inline pure single-assignment body definitions into a cloned expression."""

    def __init__(self, definitions: dict[str, ast.expr]) -> None:
        super().__init__()
        self.definitions = definitions
        self.expanding: set[str] = set()

    def visit_Name(self, node: ast.Name) -> ast.AST:
        value = self.definitions.get(node.id)
        if (
            not isinstance(node.ctx, ast.Load)
            or value is None
            or node.id in self.expanding
        ):
            return node
        self.expanding.add(node.id)
        result = self.visit(_cute_clone_expr(value))
        self.expanding.discard(node.id)
        return result


class _CuteRenameLoad(ast.NodeTransformer):
    def __init__(self, old: str, new: str) -> None:
        super().__init__()
        self.old = old
        self.new = new

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id == self.old:
            return ast.Name(id=self.new, ctx=ast.Load())
        return node


def _cute_inline_definitions(
    node: ast.AST, definitions: dict[str, ast.expr]
) -> ast.expr:
    result = _CuteInlineDefinitions(definitions).visit(_cute_clone_expr(node))
    assert isinstance(result, ast.expr)
    return result


def _cute_at_lane_base(node: ast.expr, lane: _CuteLaneIndexFacts) -> str:
    """``node`` with the per-element index replaced by the lane base."""
    result = _CuteRenameLoad(lane.index_var, lane.base_var).visit(
        _cute_clone_expr(node)
    )
    return ast.unparse(result)


def _cute_qualified_call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _cute_qualified_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix is not None else None
    return None


_CUTE_LANE_INT_CASTS = frozenset({"cutlass.Int32", "cutlass.Int64"})
_CUTE_LANE_COMPARE_CALLS: dict[str, type[ast.cmpop]] = {
    "operator.lt": ast.Lt,
    "operator.le": ast.LtE,
    "operator.gt": ast.Gt,
    "operator.ge": ast.GtE,
    "operator.eq": ast.Eq,
    "operator.ne": ast.NotEq,
}
_CUTE_LANE_MIRRORED_COMPARE: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.Gt,
    ast.Gt: ast.Lt,
    ast.LtE: ast.GtE,
    ast.GtE: ast.LtE,
    ast.Eq: ast.Eq,
    ast.NotEq: ast.NotEq,
}


def _cute_lane_affine_form(node: ast.expr, index_var: str) -> tuple[int, int] | None:
    """``(coefficient, offset)`` of an integer expression in ``index_var``.

    Recognizes the index itself, integer literals, negation, addition and
    subtraction (as operators or ``operator.add`` / ``operator.sub`` calls)
    and integer casts.  ``None`` for anything else, including a coefficient
    outside ``{0, 1}``.
    """

    def combine(
        left: tuple[int, int] | None, right: tuple[int, int] | None, sign: int
    ) -> tuple[int, int] | None:
        if left is None or right is None:
            return None
        coefficient = left[0] + sign * right[0]
        if coefficient not in (0, 1):
            return None
        return coefficient, left[1] + sign * right[1]

    if isinstance(node, ast.Name):
        return (1, 0) if node.id == index_var else None
    if isinstance(node, ast.Constant):
        return (0, node.value) if type(node.value) is int else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return combine(
            (0, 0),
            _cute_lane_affine_form(node.operand, index_var),
            1 if isinstance(node.op, ast.UAdd) else -1,
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        return combine(
            _cute_lane_affine_form(node.left, index_var),
            _cute_lane_affine_form(node.right, index_var),
            1 if isinstance(node.op, ast.Add) else -1,
        )
    if isinstance(node, ast.Call) and not node.keywords:
        name = _cute_qualified_call_name(node.func)
        if name in _CUTE_LANE_INT_CASTS and len(node.args) == 1:
            return _cute_lane_affine_form(node.args[0], index_var)
        if name in ("operator.add", "operator.sub") and len(node.args) == 2:
            return combine(
                _cute_lane_affine_form(node.args[0], index_var),
                _cute_lane_affine_form(node.args[1], index_var),
                1 if name == "operator.add" else -1,
            )
        if name == "operator.neg" and len(node.args) == 1:
            return combine((0, 0), _cute_lane_affine_form(node.args[0], index_var), -1)
    return None


def _cute_lane_uniform_compare(
    left: ast.expr,
    op: type[ast.cmpop],
    right: ast.expr,
    lane: _CuteLaneIndexFacts,
) -> bool:
    """Whether ``left op right`` has one value for all lanes of the packet.

    With the base a multiple of V, ``index + k < C`` holds for every lane of
    the span exactly when it holds at the base if ``C - k`` is a multiple of V
    (the span cannot straddle such a bound); ``<=`` and ``>`` need ``C + 1 - k``
    to be one.  Equality is never uniform.  Comparisons that do not read the
    index are uniform trivially.
    """
    reads = _cute_names_read(left) | _cute_names_read(right)
    if lane.index_var not in reads:
        return True
    left_form = _cute_lane_affine_form(left, lane.index_var)
    right_form = _cute_lane_affine_form(right, lane.index_var)
    if left_form is None or right_form is None:
        return False
    if left_form[0] == right_form[0]:
        # ``index + a  op  index + b``: a constant comparison.
        return True
    if left_form[0] == 0:
        left_form, right_form = right_form, left_form
        op = _CUTE_LANE_MIRRORED_COMPARE[op]
    shift = left_form[1]
    bound = right_form[1]
    if op in (ast.Lt, ast.GtE):
        return (bound - shift) % lane.vec_width == 0
    if op in (ast.LtE, ast.Gt):
        return (bound + 1 - shift) % lane.vec_width == 0
    return False


def _cute_lane_uniform_predicate(node: ast.expr, lane: _CuteLaneIndexFacts) -> bool:
    """Whether a boolean expression has one value for all lanes of the packet.

    Boolean combinations (``and`` / ``or``, the ``&`` / ``|`` / ``^`` that
    ``torch.logical_and`` and the tensor operators lower to, negations) are
    uniform when every operand is; comparisons are decided by
    ``_cute_lane_uniform_compare``.
    """
    if lane.index_var not in _cute_names_read(node):
        return True
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        return all(_cute_lane_uniform_predicate(value, lane) for value in node.values)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)
    ):
        return _cute_lane_uniform_predicate(
            node.left, lane
        ) and _cute_lane_uniform_predicate(node.right, lane)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert)):
        return _cute_lane_uniform_predicate(node.operand, lane)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return _cute_lane_uniform_compare(
            node.left, type(node.ops[0]), node.comparators[0], lane
        )
    if isinstance(node, ast.Call) and not node.keywords:
        name = _cute_qualified_call_name(node.func)
        if (
            name in ("operator.and_", "operator.or_", "operator.xor")
            and len(node.args) == 2
        ):
            return all(_cute_lane_uniform_predicate(arg, lane) for arg in node.args)
        if (
            name in ("operator.not_", "operator.invert", "cutlass.Boolean")
            and len(node.args) == 1
        ):
            return _cute_lane_uniform_predicate(node.args[0], lane)
        if name in _CUTE_LANE_COMPARE_CALLS and len(node.args) == 2:
            return _cute_lane_uniform_compare(
                node.args[0], _CUTE_LANE_COMPARE_CALLS[name], node.args[1], lane
            )
    return False


def _cute_and_terms(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return [term for value in node.values for term in _cute_and_terms(value)]
    return [node]


def _cute_lane_uniform_mask(scope: _CuteVloopScope, uniform_mask: str) -> str | None:
    """Re-express ``uniform_mask`` without reading the per-element index.

    Terms that do not read the index are kept verbatim (the relocation pass
    then proves their definitions can move above the V-loop).  A term that
    does read it is inlined through the body's pure single-assignment
    definitions (``_cute_single_definitions``; a loaded flag or an atomic
    result stays a name the relocation pass must prove) and must be a boolean
    combination of comparisons proven uniform across the aligned V-wide span
    (an ``extra_mask`` such as ``tile.index < n`` with ``n % V == 0``, or
    ``(flag > 0) & (tile.index < n)``); it is then evaluated at the lane
    base.  ``None`` when a term varies per lane, keeping the site on scalar
    loads.
    """
    lane = scope.lane
    definitions = _cute_single_definitions(
        scope.body, exclude=None if lane is None else lane.index_var
    )
    terms: list[str] = []
    for term in _cute_and_terms(ast.parse(uniform_mask, mode="eval").body):
        expanded = _cute_inline_definitions(term, definitions)
        if lane is None or lane.index_var not in _cute_names_read(expanded):
            terms.append(ast.unparse(term))
            continue
        if not lane.aligned or not _cute_lane_uniform_predicate(expanded, lane):
            return None
        terms.append(_cute_at_lane_base(expanded, lane))
    return " and ".join(f"({term})" for term in terms)


def _cute_lane_base_coordinate(scope: _CuteVloopScope, coordinate: str) -> str | None:
    """The lane-axis coordinate of a packet evaluated at its lane base.

    The block's own index maps to the base var.  ``index + k`` (``tile.index -
    n1`` addressing the second half of a concatenation) maps to ``base + k``
    when ``k`` is a multiple of V, so the shifted packet stays aligned and
    contiguous.  ``None`` keeps the site scalar.
    """
    lane = scope.lane
    if lane is None:
        return None
    if coordinate == lane.index_var:
        return lane.base_var
    if not lane.aligned:
        return None
    expanded = _cute_inline_definitions(
        ast.parse(coordinate, mode="eval").body,
        _cute_single_definitions(scope.body, exclude=lane.index_var),
    )
    form = _cute_lane_affine_form(expanded, lane.index_var)
    if form is None or form[0] != 1 or form[1] % lane.vec_width != 0:
        return None
    return _cute_at_lane_base(expanded, lane)


def _cute_tensor_access_names(node: ast.AST) -> set[str]:
    """Generated tensor names addressed (``t.iterator`` / ``t[...]``) in ``node``."""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            if child.attr in ("iterator", "__setitem__") and isinstance(
                child.value, ast.Name
            ):
                names.add(child.value.id)
        elif isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name):
            names.add(child.value.id)
    return names


# Calls without memory effects that ``_is_proven_relocatable_call`` does not
# vouch for: loop ranges, and the layout arithmetic of an atomic's pointer.
_CUTE_EFFECT_FREE_CALLS = frozenset(
    {
        "range",
        "cutlass.range",
        "cutlass.range_constexpr",
        "cutlass.range_dynamic",
        "cute.crd2idx",
    }
)


def _cute_write_call_roots(call: ast.Call) -> set[str]:
    """Generated tensor names addressed by a recognized memory-write call.

    Pointer-method writes (``(ptr).store(v)``, ``t.__setitem__(...)``,
    ``(ptr).atomic_add(v)``) carry their pointer in the callee receiver;
    ``cute.arch.atomic_*`` and the helper stores/atomics take it as the first
    argument.  An empty result means the pointer does not name its tensor.
    """
    func = call.func
    if isinstance(func, ast.Attribute) and ast.unparse(func.value) != "cute.arch":
        pointer: ast.AST | None = func.value
    else:
        pointer = call.args[0] if call.args else None
    if pointer is None:
        return set()
    roots = _cute_tensor_access_names(pointer)
    if isinstance(pointer, ast.Name):
        roots.add(pointer.id)
    return roots


def _cute_statement_written_tensors(
    statement: ast.AST, sites: Sequence[CuteTileVecStoreSite]
) -> set[str] | None:
    """Generated tensor names ``statement`` may write; None for unknown effects.

    A collected vector store (``sites``) writes the tensor its flush
    addresses: its site statement, and the scalar store a grid keeps in its
    place until the wrap, are decided by identity first because a packed site
    is a plain name assignment that would otherwise pass as pure.  A proven
    pure assignment writes nothing.  Otherwise every call in the
    statement must be proven pure, effect-free (a loop range, layout
    arithmetic), or a recognized memory write whose pointer names its tensor,
    and every subscript assignment must name its target; a barrier, a helper
    of unknown purity, or an unfamiliar statement kind leaves the effects
    unknown.  The cross-thread reduction helpers synchronize the CTA like a
    barrier and are left unknown too.
    """
    from ..tile_strategy import _is_proven_relocatable_assignment
    from ..tile_strategy import _is_proven_relocatable_call
    from ..tile_strategy import _memory_write_calls
    from ..tile_strategy import _qualified_name

    for site in sites:
        if statement is site.body_stmt or statement is site.scalar_stmt:
            return {site.tensor_name}
    if _is_proven_relocatable_assignment(statement, allow_load=True):
        return set()
    write_calls = {id(call) for call in _memory_write_calls(statement)}
    written: set[str] = set()
    for node in ast.walk(statement):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    continue
                roots = _cute_tensor_access_names(target)
                if not roots:
                    return None
                written |= roots
        elif isinstance(node, ast.stmt):
            if not isinstance(node, (ast.Expr, ast.If, ast.For, ast.Pass)):
                return None
        elif isinstance(node, ast.Call):
            if id(node) in write_calls:
                roots = _cute_write_call_roots(node)
                if not roots:
                    return None
                written |= roots
            elif not (
                _is_proven_relocatable_call(node, allow_load=True)
                or _qualified_name(node.func) in _CUTE_EFFECT_FREE_CALLS
            ):
                return None
    return written


def _cute_tile_unroll_preceding_statements(
    body: list[ast.AST], placeholder: ast.AST | None
) -> list[ast.AST] | None:
    """Statements of a V-loop body that run before a tile-vector load site.

    A device-loop site is lowered while its body is being built, so every
    statement present precedes it.  A grid site is emitted once the root body
    is wrapped, so the statements before the one holding its scalar
    ``placeholder`` precede it; None when the placeholder is not in the body.
    """
    if placeholder is None:
        return list(body)
    for index, statement in enumerate(body):
        if any(node is placeholder for node in ast.walk(statement)):
            return body[:index]
    return None


def _cute_tile_unroll_hoist_allowed(
    state: CodegenState,
    strategy: CuteLaneTileStrategy,
    block_id: int,
    preceding: list[ast.AST],
    tensor_name: str,
) -> bool:
    """Whether a packet of ``tensor_name`` may be loaded above the V-loop.

    The hoisted packet reads before every statement of the body, so each
    preceding statement must be proven not to write ``tensor_name``: it is
    pure, or writes only tensors proven disjoint from it.  A statement of
    unknown effect (a barrier, a helper call), a write to the same tensor, or a
    write to a tensor that may alias it keeps the site on its scalar load.
    """
    sites = strategy._cute_lane_vec_stores_by_block.get(block_id, [])
    disjoint: set[frozenset[str]] | None = None
    for statement in preceding:
        written = _cute_statement_written_tensors(statement, sites)
        if written is None:
            return False
        for name in written:
            if name == tensor_name:
                return False
            if disjoint is None:
                disjoint = state.device_function.proven_disjoint_tensor_pairs()
            if frozenset((name, tensor_name)) not in disjoint:
                return False
    return True


def demote_reordered_tile_vec_stores(
    strategy: CuteLaneTileStrategy,
    device_function: DeviceFunction,
    block_id: int,
    body: list[ast.AST] | list[ast.stmt],
    *,
    inside: Sequence[ast.AST] | None = None,
) -> list[CuteTileVecStoreSite]:
    """Restore scalar stores whose deferred flush would run past a later access.

    A tile-vector store of ``block_id`` runs after the V-loop instead of in its
    lane, so a later statement of its lane loop (nested statements included)
    that names the stored tensor, or a tensor not proven disjoint from it,
    would observe memory before the store.  ``inside`` lists the statements of
    ``body`` that run inside the loop, in order, when the caller has placed
    the others around it (``distribute_lane_loops`` orders those against the
    flush); by default every statement of ``body`` does.  Only the other
    collected stores, which flush in source order, are exempt.  Later sites
    are decided first, so a demoted one counts as an access for the sites
    before it.  Returns the demoted sites.
    """
    from ..device_function import TensorArg

    sites = strategy._cute_lane_vec_stores_by_block.get(block_id)
    if not sites:
        return []
    lane_body = strategy._cute_lane_body_by_block[block_id]
    scope = body if inside is None else inside
    tensor_names = {
        arg.name for arg in device_function.arguments if isinstance(arg, TensorArg)
    }
    disjoint = device_function.proven_disjoint_tensor_pairs()
    demoted: list[CuteTileVecStoreSite] = []
    for site in reversed(list(sites)):
        index = next(
            (i for i, statement in enumerate(scope) if statement is site.body_stmt),
            None,
        )
        if index is None:
            continue
        exempt = {id(other.body_stmt) for other in sites if other is not site}
        accessed = {
            node.id
            for statement in scope[index + 1 :]
            if id(statement) not in exempt
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and node.id in tensor_names
        }
        observers = sorted(
            name
            for name in accessed
            if name == site.tensor_name
            or frozenset((name, site.tensor_name)) not in disjoint
        )
        if not observers:
            continue
        log.debug(
            "deferred store of %s restored to its scalar form: %s accessed later",
            site.tensor_name,
            ", ".join(observers),
        )
        position = next(
            i for i, statement in enumerate(body) if statement is site.body_stmt
        )
        body[position] = site.scalar_stmt
        if site.init_stmt is not None:
            lane_body.remove(site.init_stmt)
        lane_body.remove(site.flush_stmt)
        sites.remove(site)
        demoted.append(site)
    return demoted


def _cute_tile_unroll_uniform_definitions(
    scope: _CuteVloopScope,
    index_exprs: list[str],
    lane_axis_pos: int,
    uniform_mask: str | None,
) -> _CuteTileUnrollHoist | None:
    """What must happen above the V-loop for a tile-vector hoist.

    The hoisted packet's base pointer uses every coordinate other than the
    lane axis and its pointer guard uses ``uniform_mask``; both must be
    computable before the constexpr V-loop.  Names defined outside the body
    already are.  A name the body defines qualifies when its statement is a
    plain assignment of proven pure operations (loads included), does not have
    to cross a statement with effects (a store, a barrier, a branch), reads no
    tensor the body writes, and reads only names that qualify in turn.  Mask
    terms and a lane coordinate that read the per-element index are instead
    re-expressed at the lane base when they are provably uniform across the
    aligned packet (see ``_cute_lane_uniform_mask``).  Returns the statements
    to relocate in source order (possibly none) with the rewritten guard and
    coordinate, or ``None`` when a coordinate or mask term varies per V lane
    or depends on something that cannot move -- the site then keeps its
    scalar load.
    """
    from ..tile_strategy import _is_proven_relocatable_assignment
    from ..tile_strategy import _memory_write_calls

    lane_base_expr: str | None = None
    if scope.lane is not None and index_exprs[lane_axis_pos] != scope.lane.index_var:
        lane_base_expr = _cute_lane_base_coordinate(scope, index_exprs[lane_axis_pos])
        if lane_base_expr is None:
            return None
    if uniform_mask is not None:
        uniform_mask = _cute_lane_uniform_mask(scope, uniform_mask)
        if uniform_mask is None:
            return None
    body = scope.body
    roots: set[str] = set()
    for axis, expression in enumerate(index_exprs):
        if axis != lane_axis_pos:
            roots |= set(ReadWrites.from_ast(ast.parse(expression, mode="eval")).reads)
    for expression in (uniform_mask, lane_base_expr):
        if expression is not None:
            roots |= set(ReadWrites.from_ast(ast.parse(expression, mode="eval")).reads)
    definitions: dict[str, int] = {}
    repeated: set[str] = set()
    first_effect = len(body)
    for index, statement in enumerate(body):
        for name in ReadWrites.from_ast(statement).writes:
            if name in definitions or name in repeated:
                repeated.add(name)
                definitions.pop(name, None)
            else:
                definitions[name] = index
        if first_effect == len(body) and not _is_proven_relocatable_assignment(
            statement, allow_load=True
        ):
            first_effect = index
    placeholders = {id(scalar) for _vloop, scalar, _emit in scope.deferred}
    selected: set[int] = set()
    pending = list(roots)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in scope.unavailable or name in repeated:
            return None
        index = definitions.get(name)
        if index is None:
            # Defined above the V-loop (kernel argument, outer tile
            # coordinate, enclosing lane) -- already available to the hoist.
            continue
        statement = body[index]
        if (
            not _is_proven_relocatable_assignment(statement, allow_load=True)
            or index > first_effect
            or any(id(node) in placeholders for node in ast.walk(statement))
        ):
            return None
        selected.add(index)
        pending.extend(ReadWrites.from_ast(statement).reads)
    moved = [body[index] for index in sorted(selected)]
    if not moved:
        return _CuteTileUnrollHoist(moved, uniform_mask, lane_base_expr)
    # A relocated load runs once before every V lane instead of after the
    # lanes preceding it; refuse when the body writes a tensor it reads.
    loaded = set().union(*(_cute_tensor_access_names(stmt) for stmt in moved))
    for statement in body:
        for call in _memory_write_calls(statement):
            if loaded & _cute_tensor_access_names(call):
                return None
    return _CuteTileUnrollHoist(moved, uniform_mask, lane_base_expr)


def _cute_relocate_above_vloop(scope: _CuteVloopScope, moved: list[ast.AST]) -> None:
    """Move ``moved`` from the V-loop body to just before the V-loop.

    A device loop's body is final, so the statements move now.  A grid body
    is placed around its lane loops later by ``DeviceGridState.wrap_body``,
    which either emits the definitions before the loop or applies the
    recorded relocation (see ``CuteLaneRelocation``).
    """
    if not moved:
        return
    relocation = CuteLaneRelocation(scope.lane_body, scope.vloop, list(moved))
    if scope.relocations is not None:
        scope.relocations.append(relocation)
    else:
        relocation.apply(scope.body)


def _cute_tile_packet_is_aligned(
    env: CompileEnvironment,
    tensor: torch.Tensor,
    lane_axis: int,
    vec_width: int,
    *,
    flat: bool,
) -> bool:
    """Prove a tile packet's addresses from the bound pointer/stride residues.

    The hoist reads ``vec_width`` contiguous elements at a lane base that is
    a multiple of V, so the packet is naturally aligned exactly when the
    tensor base is aligned to the packet and every other stride is a multiple
    of V elements (``cute_reduction_vector_layout_aligned``); a flat packet
    addresses a contiguous tensor (``_cute_flat_multi_cover_ok``) and needs
    only the base.  A 132-wide bf16 row or a view four elements into its
    storage faults the 16-byte access, so an unprovable site stays on scalar
    loads.  The lane axis must be the unit-stride dim, and byte packets ride
    a packed integer carrier of at most eight bytes.
    """
    vector_bytes = vec_width * tensor.dtype.itemsize
    if vec_width < 2 or _CUTE_VECTOR_MAX_BYTES % vector_bytes:
        return False
    if tensor.dtype.itemsize == 1 and vec_width not in (2, 4, 8):
        return False
    if flat:
        return cute_tensor_base_is_aligned(env, tensor, vector_bytes)
    lane_stride = tensor.stride(lane_axis)
    if not isinstance(lane_stride, int) or lane_stride != 1:
        return False
    return cute_reduction_vector_layout_aligned(env, tensor, lane_axis, vec_width)


def _cute_tile_axis_block_id(idx: torch.SymInt) -> int | None:
    """Block id when ``idx`` is a tile's own index (its block-size symbol).

    ``CompileEnvironment.get_block_id`` also resolves tile edges (``tile.begin``
    / ``tile.end`` / ``tile.id``) and ``hl.grid`` indices to their block, but
    those are uniform scalar coordinates, not a lane axis: ``w[tile.begin]``
    must load one element, never a per-lane vector.
    """
    from ..host_function import HostFunction
    from ..variable_origin import BlockSizeOrigin

    expr = idx._sympy_()
    if not isinstance(expr, sympy.Symbol):
        return None
    origin_info = HostFunction.current().expr_to_origin.get(expr)
    if origin_info is None or not isinstance(origin_info.origin, BlockSizeOrigin):
        return None
    return origin_info.origin.block_id


def _cute_vector_load_ctx(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    index_exprs: list[str],
    extra_mask: ast.AST | None,
    *,
    allow_gathered_index: bool = False,
) -> tuple[int, int, str] | None:
    """Return (vec_width, lane_block_id, mode) when a vec load may be emitted.

    ``mode`` is one of ``"vec"`` (explicit ``cute.arch.load(..., V)``) or
    ``"unroll"`` (per-element scalar bitcast inside a constexpr V-loop).
    Returns None when any predicate for a 128-bit gmem load fails, in which
    case the caller falls back to ``_cute_scalar_load_expr``.

    ``allow_gathered_index`` (load sites only) admits a ``"tile_unroll"``
    site whose coordinate on a non-lane dim is a gathered tensor value
    (``weight[idx[tile_b], tile_e]``) or is defined inside the V-loop body,
    provided ``_cute_tile_unroll_uniform_definitions`` can relocate those
    definitions above the V-loop, and a site whose lane coordinate is the
    tile index shifted by a multiple of V (``y[tile0, tile1.index - n1]``).

    An ``extra_mask`` does not disqualify a site here: the load lowering
    keeps the packet only when every mask term other than the lane mask is
    uniform across the V lanes (``_cute_vector_load_mask_is_lane_only`` /
    ``_cute_lane_uniform_mask``), and falls back to scalar loads otherwise.
    """
    from ..reduction_strategy import LoopedReductionStrategy
    from ..reduction_strategy import PersistentReductionStrategy

    env = CompileEnvironment.current()
    if env.backend.name != "cute":
        return None
    if "None" in index_exprs:
        return None
    if tensor.dtype not in _CUTE_VECTOR_DTYPES and not _cute_is_unroll_dtype(
        tensor.dtype
    ):
        return None
    # Only enable the vec path when the load's result eventually feeds a
    # reduction op.  The consume-sweep mixes the loaded vector with scalar
    # values (e.g. the post-reduction inverse-RMS), and broadcasting
    # scalar->vec is not supported by the CuTe DSL today.  When the load's
    # immediate user is a dtype cast (``to(torch.float32)``), the
    # ``"unroll"`` mode further down keeps the strategy on a per-element
    # scalar pipeline and the explicit-vec path is skipped — the explicit
    # ``cute.arch.load(ptr, ir.VectorType.get([V], dtype.mlir_type))`` form
    # would otherwise crash inside the CuTe DSL when subscripting bf16/fp16
    # vectors.
    fx_node = state.fx_node
    if fx_node is None:
        return None
    visited: set[torch.fx.Node] = set()
    pending = list(fx_node.users.keys())
    feeds_reduction = False
    while pending:
        user = pending.pop()
        if user in visited:
            continue
        visited.add(user)
        target_name = getattr(user.target, "__name__", "") or ""
        target_qualname = getattr(user.target, "_qualname", "") or ""
        if (
            "reduction" in target_name
            or "_inductor_lowering_extra" in target_name
            or "reduction" in target_qualname
        ):
            feeds_reduction = True
            break
        pending.extend(user.users.keys())
    # Note: ``feeds_reduction`` is required ONLY for the ``vec`` mode below;
    # the ``unroll`` mode also applies to the consume sweep where the load
    # result feeds an elementwise pipeline (no reduction).
    # The lane/vec axis must be a tensor dim that is stride-1 so that
    # consecutive lane iters fetch consecutive bytes.  For a row-major lhs
    # the reduction axis is the LAST subscript position; for a column-major
    # rhs (e.g. the K-major ``y`` of a tcgen05 fp8 matmul) it is the FIRST.
    # ``_cute_lane_axis_pos`` records the index_exprs position of that
    # stride-1 lane axis so the hoist substitutes the per-lane base there
    # (not blindly at ``[-1]``).
    # Find the stride-1 dim WITHOUT forcing specialization of a symbolic
    # stride: a contiguous dim has a concrete ``int`` stride of 1, so only
    # accept plain ints here.  Calling ``int()`` on a ``SymInt`` stride would
    # bake the (otherwise-dynamic) size into the kernel — see the
    # ``test_mark_static`` regression where ``int(stride(0))`` specialized
    # ``n``.
    stride1_tensor_dim: int | None = None
    for d in range(tensor.ndim):
        s = tensor.stride(d)
        if isinstance(s, int) and s == 1:
            stride1_tensor_dim = d
            break
    if stride1_tensor_dim is None:
        return None
    # Locate the non-None subscript carrying an active lane block.  Slices
    # resolve to the matching tensor-dim block via the strategy that's
    # currently active for that block.  Prefer the block sitting on the
    # stride-1 tensor dim (the true lane axis), and record its index_exprs
    # position.
    inner_block_id: int | None = None
    lane_axis_pos: int | None = None
    lane_on_stride1 = False
    gathered_index = False
    # Scalar shifts of a ``tile.index + k`` lane coordinate; the packet is
    # admitted only when their sum keeps it aligned to the vector width.
    lane_shift_terms: tuple[int | torch.SymInt, ...] | None = None
    expr_pos = -1
    tensor_dim = 0
    for subscript_pos, idx in enumerate(subscript):
        if idx is None:
            continue
        expr_pos += 1
        if isinstance(idx, torch.SymInt):
            bid = _cute_tile_axis_block_id(idx)
            if bid is not None and _cute_lane_strategy(state, bid) is not None:
                if tensor_dim == stride1_tensor_dim or inner_block_id is None:
                    inner_block_id = bid
                    lane_axis_pos = expr_pos
                    lane_on_stride1 = tensor_dim == stride1_tensor_dim
        elif isinstance(idx, torch.Tensor) and idx.ndim == 1:
            # ``hl.arange(K)`` reaches indexing lowering as a 1-D FakeTensor,
            # not as the SymInt that identifies K. Resolve its static extent
            # back to the persistent reduction block.
            bid = env.resolve_block_id(idx.numel())
            candidate = _cute_lane_strategy(state, bid) if bid is not None else None
            # Matching the index tensor's extent identifies its lane axis,
            # but does not prove contiguous addressing.  A gather such as
            # x[tile.index // 64] has exactly the same extent as tile.index.
            # Hoisting it at the raw lane base changes the address and can
            # read beyond x.  Require a direct lane index unless the persistent
            # reduction's affine rebasing preserves a unit-stride offset.
            raw_subscript = (
                state.fx_node.args[1]
                if state.fx_node is not None and len(state.fx_node.args) > 1
                else None
            )
            raw_index = (
                raw_subscript[subscript_pos]
                if isinstance(raw_subscript, (list, tuple))
                and subscript_pos < len(raw_subscript)
                else None
            )
            valid_iota = (
                is_cute_unit_stride_iota_index(raw_index)
                if isinstance(candidate, PersistentReductionStrategy)
                else is_cute_direct_iota_index(raw_index)
            )
            shifted_tile_index = (
                match_cute_shifted_tile_index(raw_index)
                if not valid_iota
                and allow_gathered_index
                and tensor_dim == stride1_tensor_dim
                else None
            )
            if shifted_tile_index is not None:
                # ``tile.index + k`` on the lane axis addresses a contiguous
                # span shifted by ``k`` elements (the second half of a
                # concatenation); the tile strategy below admits it when
                # ``k`` keeps the packet aligned and the load lowering places
                # the packet at ``lane_base + k``.  The block comes from the
                # ``tile_index`` node itself, not from the extent match.
                tile_index_node, lane_shift_terms = shifted_tile_index
                tile_symbol = tile_index_node.args[0]
                if isinstance(tile_symbol, torch.fx.Node):
                    tile_symbol = tile_symbol.meta.get("val")
                shift_bid = (
                    env.get_block_id(tile_symbol)
                    if isinstance(tile_symbol, (int, torch.SymInt))
                    else None
                )
                if shift_bid is None or _cute_lane_strategy(state, shift_bid) is None:
                    return None
                inner_block_id = shift_bid
                lane_axis_pos = expr_pos
                lane_on_stride1 = True
            elif not valid_iota:
                # A gathered coordinate on a dim other than the stride-1 lane
                # axis (``weight[idx[tile_b], tile_e]``) selects one row per
                # thread and is constant across the V contiguous lanes, so a
                # hoisted packet can carry it in its base pointer as-is.  Load
                # sites accept it here; whether its value (and its bounds
                # mask) is available above the V-loop is decided by
                # ``_cute_tile_unroll_uniform_definitions``.  A gather along
                # the lane axis itself is not contiguous and stays scalar.
                if not allow_gathered_index or tensor_dim == stride1_tensor_dim:
                    return None
                gathered_index = True
            elif bid is not None and candidate is not None:
                if tensor_dim == stride1_tensor_dim or inner_block_id is None:
                    inner_block_id = bid
                    lane_axis_pos = expr_pos
                    lane_on_stride1 = tensor_dim == stride1_tensor_dim
        elif isinstance(idx, slice) and idx == slice(None):
            if tensor_dim < tensor.ndim:
                # Same extent matching as the mask builder: ``known_equal``
                # on the block's size handles static and symbolic extents
                # alike (a symbolic reduction extent has a sympy ``numel``
                # that no ``int()`` coercion could compare).
                matches: list[int] = [
                    cand_bid
                    for cand_bid in _matching_block_ids(env, tensor.shape[tensor_dim])
                    if _cute_lane_strategy(state, cand_bid) is not None
                ]
                if matches:
                    # Matching by extent alone is ambiguous when a
                    # non-reduction tile dim happens to have the same
                    # extent (e.g. a square MxN input): a full-slice
                    # subscript is a reduction axis whenever a reduction
                    # block of that extent exists, so prefer it.
                    cand = next(
                        (bid for bid in matches if env.block_sizes[bid].reduction),
                        matches[0],
                    )
                    if tensor_dim == stride1_tensor_dim or inner_block_id is None:
                        inner_block_id = cand
                        lane_axis_pos = expr_pos
                        lane_on_stride1 = tensor_dim == stride1_tensor_dim
        tensor_dim += 1
    if inner_block_id is None or lane_axis_pos is None:
        return None
    strategy = _cute_lane_strategy(state, inner_block_id)
    if isinstance(
        strategy,
        (LoopedReductionStrategy, PersistentReductionStrategy),
    ):
        # Reduction hoists place the row coordinate as-is; only the tile lane
        # protocol below knows how to guard a gathered row, shift a lane
        # coordinate or prove an ``extra_mask`` uniform across the packet, so
        # such sites keep their scalar loads here (the branch-local vectorizer
        # would otherwise wrap a masked load the lane-reduction restore proof
        # cannot re-derive).
        if (
            not lane_on_stride1
            or gathered_index
            or lane_shift_terms is not None
            or extra_mask is not None
        ):
            return None
        vec_width = getattr(strategy, "_cute_reduction_vec_width", 1)
        if vec_width <= 1:
            return None
        if strategy._mask_var is not None and not _cute_reduction_masked_vec_ok(
            strategy, tensor
        ):
            return None
        if isinstance(
            strategy, LoopedReductionStrategy
        ) and not cute_reduction_vector_layout_aligned(
            env, tensor, stride1_tensor_dim, vec_width
        ):
            # The rolled hoist addresses each packet from the row start; a
            # V-misaligned base or row stride would fault the vector access.
            return None
        if strategy._cute_reduction_lane_extent <= 0:
            return None
        mode = getattr(strategy, "_cute_reduction_vec_mode", "unroll")
        if mode == "vec":
            if not feeds_reduction:
                return None
            if tensor.dtype not in _CUTE_VECTOR_DTYPES:
                return None
            return vec_width, inner_block_id, "vec"
        if mode == "unroll":
            if tensor.dtype not in _CUTE_VECTOR_UNROLL_DTYPES:
                return None
            # Cap at one LDG.128 per hoist (fp32 V=8 would need 32 bytes);
            # oversized configs stay on the (correct) scalar fallback.
            if vec_width * tensor.dtype.itemsize > 16:
                return None
            # Need a lane base index var + a constexpr V-loop var; both
            # are set up by the strategy's codegen_device_loop.
            if (
                getattr(strategy, "_cute_lane_base_index_var", None) is None
                or getattr(strategy, "_cute_lane_body", None) is None
            ):
                return None
            return vec_width, inner_block_id, "unroll"
        return None
    # CuTe N-D tile strategy with lane loops: vec is set up per-block in
    # ``PerThreadNDTileStrategy.__init__`` when the autotuner picks
    # ``cute_vector_widths[block_id]`` > 1 and EPT is divisible by V.  Mode
    # is forced to ``"unroll"`` (per-element bitcast) for fp16/bf16 since
    # subscripting a bf16/fp16 vector in the CuTe DSL is unsafe; fp32
    # could in principle use ``"vec"`` but the per-element pipeline runs
    # most of the consume-sweep code after a cast, so unroll is the
    # robust choice.
    from ..tile_strategy import BlockSizeTileStrategy

    if isinstance(strategy, BlockSizeTileStrategy):
        # The hoisted load reads V CONTIGUOUS elements from the per-thread
        # base, so the lane axis must actually be the tensor's stride-1
        # dim.  A lane block accepted via the ``inner_block_id is None``
        # fallback (e.g. ``x[tile_m, 0]`` where dim 1 is contiguous) would
        # vectorize along the wrong dim and read garbage.
        if not lane_on_stride1:
            return None
        # Epilogue subtiling stages stores through smem with sync_threads
        # inside the per-element pipeline; the vec hoist/flush protocol
        # silently corrupts that form — stay scalar (matches pre-vec
        # behavior, which is correct under subtiling).
        subtile = state.config.config.get("epilogue_subtile")
        if isinstance(subtile, int) and subtile > 1:
            return None
        vec_by_block = getattr(strategy, "_cute_lane_vec_width_by_block", None)
        if not isinstance(vec_by_block, dict):
            return None
        vec_width = vec_by_block.get(inner_block_id, 1)
        if vec_width <= 1:
            return None
        if not _cute_is_unroll_dtype(tensor.dtype):
            return None
        # Cap at one LDG.128 per hoist: wider than 16 bytes per thread
        # exceeds the widest gmem access and is not supported.
        if vec_width * tensor.dtype.itemsize > 16:
            return None
        if lane_shift_terms is not None and (
            tensor.dtype is torch.int8
            or getattr(strategy, "_cute_flat_multi", False)
            or not all(
                env.known_multiple(_to_sympy(term), vec_width)
                for term in lane_shift_terms
            )
        ):
            # The shifted packet is 16-byte aligned only when the shift is a
            # multiple of V; signed-byte packets and flat bases place the
            # canonical lane base and cannot carry a shift.
            return None
        base_var_by_block = getattr(
            strategy, "_cute_lane_base_index_var_by_block", None
        )
        lane_body_by_block = getattr(strategy, "_cute_lane_body_by_block", None)
        vec_lane_var_by_block = getattr(strategy, "_cute_vec_lane_var_by_block", None)
        if (
            not isinstance(base_var_by_block, dict)
            or not isinstance(lane_body_by_block, dict)
            or not isinstance(vec_lane_var_by_block, dict)
            or inner_block_id not in base_var_by_block
            or inner_block_id not in lane_body_by_block
            or inner_block_id not in vec_lane_var_by_block
        ):
            return None
        if lane_shift_terms is not None or not _cute_tile_unroll_scope_safe(
            state, strategy, inner_block_id, index_exprs, lane_axis_pos
        ):
            # The non-lane coordinates read definitions of this V-loop body
            # (a gathered row index loaded in the same body, a per-thread row
            # coordinate of a device loop), or the lane coordinate is shifted
            # and must be re-expressed at the lane base.  A load site may
            # still vectorize when those definitions can be relocated above
            # the V-loop; the load codegen performs the relocation when it
            # emits the packet.
            scope = (
                _cute_tile_unroll_vloop_scope(state, strategy, inner_block_id)
                if allow_gathered_index
                else None
            )
            if (
                scope is None
                or _cute_tile_unroll_uniform_definitions(
                    scope, index_exprs, lane_axis_pos, None
                )
                is None
            ):
                return None
        if tensor.dtype is torch.int8 and index_exprs[
            lane_axis_pos
        ] != _cute_active_index_var(state, inner_block_id):
            # The hoist substitutes the canonical lane base. Do not discard
            # an affine/gather offset for a signed input.
            return None
        flat_multi = bool(getattr(strategy, "_cute_flat_multi", False))
        if flat_multi:
            # Flattened multi-dim tile: the hoist emits FLAT base pointers
            # (``t.iterator + lane_base``), which is only sound when the
            # tensor is contiguous and covers the whole iteration space in
            # iteration order (then a V-chunk that straddles a row boundary
            # is still memory-contiguous).  Broadcast operands (bias[N])
            # fail the cover check and stay on per-element scalar loads.
            if not _cute_flat_multi_cover_ok(env, strategy, tensor, subscript):
                return None
            numel = functools.reduce(  # pyrefly: ignore [incompatible-overload-residual]
                operator.mul,
                [
                    env.block_sizes[bid].numel
                    for bid in strategy.block_ids  # pyrefly: ignore
                ],
            )
            if not env.known_multiple(numel, vec_width):
                return None
        else:
            # When the per-thread vec base could straddle the tensor edge
            # (e.g. ``numel`` not a multiple of V), the masked-tail iter
            # could load garbage in some lanes.  Gate the per-element mask
            # path correctly by requiring ``numel % V == 0`` so partial-vec
            # straddles are impossible.
            numel = env.block_sizes[inner_block_id].numel
            if not env.known_multiple(numel, vec_width):
                return None
        if not _cute_tile_packet_is_aligned(
            env, tensor, stride1_tensor_dim, vec_width, flat=flat_multi
        ):
            # The lane base is a multiple of V, but the packet also needs an
            # aligned tensor base and non-lane strides; prove both from the
            # bound residues or stay scalar.
            return None
        # Record the index_exprs position of the stride-1 lane axis so the
        # hoist substitutes the per-lane base there.  Row-major lhs loads
        # use the last position; a column-major rhs (K-major ``y``) uses
        # position 0.
        pos_by_block = getattr(strategy, "_cute_lane_axis_pos_by_block", None)
        if not isinstance(pos_by_block, dict):
            pos_by_block = {}
            # pyrefly: ignore [missing-attribute]
            strategy._cute_lane_axis_pos_by_block = pos_by_block
        pos_by_block[inner_block_id] = lane_axis_pos
        return vec_width, inner_block_id, "tile_unroll"
    return None


def _cute_record_sinkable_scalar_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: Sequence[object],
    index_exprs: Sequence[str],
    tensor_name: str,
    eviction_suffix: str,
) -> None:
    """Record a scalar tile load the vector-loop sinking pass may widen.

    The ordinary tile hoist needs the load directly in the grid body so the
    packet can move above the constexpr V-loop.  A load nested in a serial or
    lane loop inside that V-loop keeps its scalar form here; when it still
    addresses the vectorized grid axis at the tensor's stride-1 dim with the
    plain per-element index, that axis is a zero-origin tile whose extent is
    a multiple of V (every V-wide chunk is V-aligned and lies entirely inside
    or outside the extent), the tensor's base and every other stride are
    proven aligned for the packet and the packet fits one 16-byte
    transaction, ``cute/sink_vector_loops.py`` can load the whole chunk once
    per row after interchanging the V-loop into those loops.  The base and
    stride proof is the tile hoist's (``cute_reduction_vector_layout_aligned``),
    so a misaligned input view the hoist left on scalar loads is not widened
    here either.  The fact is keyed by the scalar pointer expression the
    pass will find in the AST.
    """
    from ..tile_strategy import DeviceGridState

    env = CompileEnvironment.current()
    wrappers = state.device_function.cute_state.vloop_sink_wrappers
    if (
        env.backend.name != "cute"
        or not wrappers
        or not isinstance(state.codegen.current_grid_state, DeviceGridState)
        or tensor.dtype not in _CUTE_VECTOR_UNROLL_DTYPES
        or "None" in index_exprs
    ):
        return
    # Consecutive V lanes must fetch consecutive bytes, so the vectorized
    # axis has to index the tensor's stride-1 dim (a plain ``int`` stride of
    # 1: a symbolic stride is not specialized here).
    stride1_dim = next(
        (
            dim
            for dim in range(tensor.ndim)
            if isinstance(tensor.stride(dim), int) and tensor.stride(dim) == 1
        ),
        None,
    )
    if stride1_dim is None:
        return
    tensor_dim = 0
    lane_axis_pos: int | None = None
    block_id: int | None = None
    expr_pos = -1
    for idx in subscript:
        if idx is None:
            continue
        expr_pos += 1
        if tensor_dim == stride1_dim:
            if not isinstance(idx, torch.SymInt):
                return
            block_id = env.get_block_id(idx)
            lane_axis_pos = expr_pos
        tensor_dim += 1
    if block_id is None or lane_axis_pos is None:
        return
    wrapper = next(
        (fact for fact in wrappers.values() if fact.block_id == block_id), None
    )
    if (
        wrapper is None
        or not wrapper.uniform_vector_mask
        or wrapper.vec_width * tensor.dtype.itemsize > _CUTE_VECTOR_MAX_BYTES
        or index_exprs[lane_axis_pos] != wrapper.index_var
        or not cute_reduction_vector_layout_aligned(
            env, tensor, stride1_dim, wrapper.vec_width
        )
    ):
        return
    from .device_state import CuteVloopLoadFact

    # Key by the normalized source text the pass sees after ``ast.unparse``.
    key = ast.unparse(
        ast.parse(
            _cute_scalar_pointer_expr(tensor_name, list(index_exprs)), mode="eval"
        )
    )
    state.device_function.cute_state.vloop_sink_loads[key] = CuteVloopLoadFact(
        vec_lane_var=wrapper.vec_lane_var,
        tensor_name=tensor_name,
        vec_width=wrapper.vec_width,
        dtype=tensor.dtype,
        eviction_suffix=eviction_suffix,
    )


def _cute_resolved_load_mask(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: Sequence[object],
    index_exprs: Sequence[str],
    extra_mask: ast.AST | None,
) -> str | None:
    """Mask full slices with the same axis that supplies their address.

    Equal logical extents can belong to different active axes. The address
    resolver prefers a unique reduction axis for a full slice; choosing the
    first equal extent again while masking can instead capture an unrelated
    output-row mask. In collective staging that row coordinate is unavailable.
    Preserve the resolved block identity through the ordinary mask builder.
    """
    env = CompileEnvironment.current()
    mask_subscript = list(subscript)
    tensor_dim = 0
    for pos, index in enumerate(subscript):
        if index is None:
            continue
        if isinstance(index, slice) and index == slice(None):
            candidates = [
                block_id
                for block_id in _matching_block_ids(env, tensor.shape[tensor_dim])
                if _cute_active_index_var(state, block_id) == index_exprs[tensor_dim]
            ]
            if candidates:
                # Operand remapping can give multiple block IDs the same
                # address. They are interchangeable only with identical masks.
                if (
                    len(
                        {
                            _cute_active_mask_var(state, block_id)
                            for block_id in candidates
                        }
                    )
                    != 1
                ):
                    raise exc.BackendUnsupported(
                        "cute", "full-slice load mask has ambiguous coordinate bounds"
                    )
                mask_subscript[pos] = env.block_sizes[candidates[0]].var
        tensor_dim += 1
    return _cute_combined_mask(
        state,
        mask_subscript,
        extra_mask,
        tensor=tensor,
        include_tensor_index_masks=False,
    )


def _cute_load_output_dims(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: Sequence[object],
    index_exprs: Sequence[str],
) -> tuple[tuple[int, int | None] | None, ...] | None:
    """Map each load-output dim to ``(tensor_dim, block_id)``.

    Mirrors ``SubscriptIndexing.compute_shape``: ``None`` adds a unit axis,
    ints and scalar SymInts drop a dim, tiles and slices keep one. Returns
    ``None`` for tensor (gather) indices.
    """
    env = CompileEnvironment.current()
    result: list[tuple[int, int | None] | None] = []
    tensor_dim = 0
    for pos, idx in enumerate(subscript):
        if idx is None:
            result.append(None)
            continue
        if isinstance(idx, torch.Tensor) or tensor_dim >= tensor.ndim:
            return None
        block_id: int | None = None
        keeps_dim = False
        tile_info = _get_tile_with_offset_info(idx, state.fx_node, pos)
        if tile_info is not None and tile_info.block_size is not None:
            keeps_dim = True
            block_id = tile_info.block_id
        elif isinstance(idx, torch.SymInt):
            block_id = env.get_block_id(idx)
            keeps_dim = block_id is not None
        elif isinstance(idx, slice):
            keeps_dim = True
            for candidate in _matching_block_ids(env, tensor.shape[tensor_dim]):
                if _cute_active_index_var(state, candidate) == index_exprs[tensor_dim]:
                    block_id = candidate
                    break
        elif not isinstance(idx, int):
            return None
        if env.known_equal(tensor.shape[tensor_dim], 1):
            # ``_cute_index_exprs`` addresses size-1 tensor dims with a literal
            # ``0``; there is no block coordinate to swap there.
            block_id = None
        if keeps_dim:
            result.append((tensor_dim, block_id))
        tensor_dim += 1
    return tuple(result)


def _record_cute_scalar_load_site(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: Sequence[object],
    tensor_name: str,
    index_exprs: Sequence[str],
    mask_expr: str | None,
    has_extra_mask: bool,
    eviction_suffix: str,
) -> None:
    """Remember the scalar address so ``hl.split`` can re-read this tile."""
    assert state.fx_node is not None
    value = state.fx_node.meta.get("val")
    if "None" in index_exprs or not isinstance(value, torch.Tensor):
        return
    output_dims = _cute_load_output_dims(state, tensor, subscript, index_exprs)
    if output_dims is None or len(output_dims) != value.ndim:
        return
    state.fx_node.meta[CUTE_SCALAR_LOAD_SITE_META] = CuteScalarLoadSite(
        tensor_name=tensor_name,
        index_exprs=tuple(index_exprs),
        mask_expr=mask_expr,
        has_extra_mask=has_extra_mask,
        eviction_suffix=eviction_suffix,
        output_dims=output_dims,
    )


def _apply_cute_value_coord_meta(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: Sequence[object],
    index_exprs: list[str],
    value_node: object,
) -> list[object]:
    """Address store dims by the stored value's subtile coordinates.

    Dims created by a split view (``x[tile, :].view(..., 2, half)`` feeding
    ``hl.split``) are not owned by a block; their per-thread coordinate lives
    in ``CUTE_DIM_LOCAL_COORD_META``. A full-slice store of such a value would
    otherwise be addressed by an unrelated reduction dim of the same size, so
    substitute the value's coordinate. Returns the subscript to build the
    mask from: re-addressed dims are always in bounds and drop their block
    mask.
    """
    from .cute_reshape import CUTE_DIM_LOCAL_COORD_META
    from .cute_reshape import _get_block_local_coord
    from .cute_reshape import _subtile_coord_expr

    mask_subscript = list(subscript)
    if not isinstance(value_node, torch.fx.Node):
        return mask_subscript
    meta = value_node.meta.get(CUTE_DIM_LOCAL_COORD_META)
    value = value_node.meta.get("val")
    if (
        not isinstance(meta, (list, tuple))
        or not isinstance(value, torch.Tensor)
        or len(meta) != value.ndim
        or not any(isinstance(info, dict) for info in meta)
    ):
        return mask_subscript
    output_dims = _cute_load_output_dims(state, tensor, subscript, index_exprs)
    if output_dims is None or len(output_dims) != value.ndim:
        return mask_subscript
    from ..generate_ast import GenerateAST

    cg = state.codegen
    assert isinstance(cg, GenerateAST)
    positions = [pos for pos, idx in enumerate(subscript) if idx is not None]
    for info, mapping in zip(meta, output_dims, strict=True):
        if not isinstance(info, dict) or mapping is None:
            continue
        tensor_dim, block_id = mapping
        idx = subscript[positions[tensor_dim]]
        if not (isinstance(idx, slice) and idx == slice(None)):
            continue
        coord = _subtile_coord_expr(cg, info)
        if coord is None:
            continue
        own_coord = (
            _get_block_local_coord(cg, block_id) if block_id is not None else None
        )
        if own_coord is None:
            index_exprs[tensor_dim] = coord
        else:
            # Keep the slice's tile base, swap in the value's coordinate.
            index_exprs[tensor_dim] = (
                f"({index_exprs[tensor_dim]}) - ({own_coord}) + ({coord})"
            )
        mask_subscript[positions[tensor_dim]] = 0
    return mask_subscript


def cute_reindexed_scalar_load_expr(
    cg: GenerateAST,
    load_node: torch.fx.Node,
    flat_index: str,
) -> ast.AST | None:
    """Re-read ``load_node``'s tile at row-major ``flat_index`` of its tile shape.

    ``hl.split`` uses this to fetch both pair elements of a loaded tile with
    fresh scalar loads (the register-layout equivalent of ``tl.split``) rather
    than exchanging the whole tile through shared memory. Only dims addressed
    by a block index are re-coordinated. The original mask is kept (a thread
    whose own element is out of bounds has its outputs masked too) and, on a
    dim whose tile does not provably span the tensor, the partner's own bound
    check is added so an element past a partial edge tile reads as 0, exactly
    as ``tl.split`` sees the zero-filled tail of a masked tile load.  A load
    with ``extra_mask`` is not re-read: its mask is a per-thread value that
    cannot be re-evaluated at the partner's coordinates, so the partner would
    read raw memory where ``tl.split`` sees a zero.
    """
    from .cute_reshape import _coords_from_flat_index
    from .cute_reshape import _get_block_local_coord
    from .cute_reshape import _get_tile_shape

    site = load_node.meta.get(CUTE_SCALAR_LOAD_SITE_META)
    value = load_node.meta.get("val")
    tensor_node = load_node.args[0]
    tensor = (
        tensor_node.meta.get("val") if isinstance(tensor_node, torch.fx.Node) else None
    )
    if (
        not isinstance(site, CuteScalarLoadSite)
        or site.has_extra_mask
        or not isinstance(value, torch.Tensor)
        or not isinstance(tensor, torch.Tensor)
    ):
        return None
    env = CompileEnvironment.current()
    shape = _get_tile_shape(value, env, cg.device_function.config)
    flat_var = cg.lift(expr_from_string(flat_index), dce=True, prefix="split_index")
    coords = _coords_from_flat_index(flat_var.id, shape)
    index_exprs = list(site.index_exprs)
    mask_terms = [] if site.mask_expr is None else [site.mask_expr]
    for dim, mapping in enumerate(site.output_dims):
        if shape[dim] == 1 or mapping is None:
            continue
        tensor_dim, block_id = mapping
        if block_id is None:
            return None
        own_coord = _get_block_local_coord(cg, block_id)
        if own_coord is None:
            return None
        # The emitted index is ``tile base + this thread's local coordinate``;
        # keep the base and substitute the requested coordinate.
        partner = f"({index_exprs[tensor_dim]}) - ({own_coord}) + ({coords[dim]})"
        index_exprs[tensor_dim] = partner
        dim_size = tensor.size(tensor_dim)
        if not env.known_equal(dim_size, shape[dim]):
            size_expr = (
                str(dim_size)
                if isinstance(dim_size, int)
                else cg.device_function.sympy_expr(dim_size._sympy_())
            )
            mask_terms.append(f"(({partner}) < {size_expr})")
    load_expr = _cute_scalar_load_expr(
        site.tensor_name,
        index_exprs,
        value.dtype,
        eviction_suffix=site.eviction_suffix,
    )
    if value.dtype is torch.bool:
        load_expr = f"({load_expr} != cutlass.Uint8(0))"
        zero = "cutlass.Boolean"
    else:
        zero = _cute_scalar_storage_dtype(value.dtype)
    if not mask_terms:
        return expr_from_string(load_expr)
    mask_expr = " and ".join(mask_terms)
    return expr_from_string(f"({load_expr} if {mask_expr} else {zero}(0))")


@_decorators.codegen(load, "cute")
def _(state: CodegenState) -> object:
    # A store to this tensor earlier in the same loop body followed by this
    # load is a cross-thread read-after-write on global memory; emit a CTA
    # barrier so the store is visible before the read.  Marked loads sit in
    # uniform control flow (mark_intra_loop_raw_barriers skips divergent
    # branches for cute; masks are ternary expressions and loop trip counts
    # are uniform), so the convergent barrier is legal here.
    from ..loop_dependency_checker import INTRA_LOOP_RAW_BARRIER_META

    if state.fx_node is not None and state.fx_node.meta.get(
        INTRA_LOOP_RAW_BARRIER_META
    ):
        state.add_statement(statement_from_string("cute.arch.sync_threads()"))

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))
    if isinstance(tensor, torch.Tensor):
        check_memory_mask_rebound(
            state, tensor, subscript, state.proxy_arg(2), what="load"
        )
    elif isinstance(tensor, tuple):
        # A stack tensor load ANDs the mask through _cute_stack_tensor_mask_expr
        # over the pointer table's dims and tensor_like[subscript].
        stack_tensor_like, stack_dev_ptrs = tensor
        check_memory_mask_rebound(
            state,
            stack_tensor_like,
            subscript,
            state.proxy_arg(2),
            what="stack tensor load",
            leading_sizes=stack_dev_ptrs.shape,
        )

    if isinstance(tensor, tuple):
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        assert isinstance(dev_ptrs_ast, ast.AST)
        tensor_like, dev_ptrs = tensor
        offset_expr = _cute_stack_tensor_offset_expr(
            state,
            tensor_like,
            [*subscript],
            ast_subscript,
        )
        backend = CompileEnvironment.current().backend
        target_dtype = backend.dtype_str(tensor_like.dtype)
        ptr_expr = _cute_stack_tensor_pointer_expr(
            target_dtype, dev_ptrs_ast, offset_expr
        )
        load_expr = f"({ast.unparse(ptr_expr)}).load()"
        mask_expr = _cute_stack_tensor_mask_expr(
            state,
            tensor_like,
            dev_ptrs,
            [*subscript],
            extra_mask,
        )
        if tensor_like.dtype is torch.bool:
            load_expr = f"({load_expr} != cutlass.Uint8(0))"
            if mask_expr is None:
                return expr_from_string(load_expr)
            return expr_from_string(
                f"({load_expr} if {mask_expr} else cutlass.Boolean(0))"
            )
        if mask_expr is None:
            return expr_from_string(load_expr)
        return expr_from_string(f"({load_expr} if {mask_expr} else {target_dtype}(0))")
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", f"load tensor type: {type(tensor)}")

    _log_cute_layout(state, "load")

    from ...language import tile_index

    tensor_node = state.fx_node.args[0] if state.fx_node is not None else None
    if (
        isinstance(tensor_node, torch.fx.Node)
        and tensor_node.op == "call_function"
        and tensor_node.target == tile_index
    ):
        env = CompileEnvironment.current()
        block_id = env.get_block_id(tensor.size(0))
        if block_id is None:
            raise exc.BackendUnsupported("cute", "tile_index load block id")
        index_var = _cute_active_index_var(state, block_id)
        if index_var is None:
            raise exc.BackendUnsupported("cute", "inactive tile_index load")
        for idx in subscript:
            if idx is None or idx == slice(None):
                continue
            raise exc.BackendUnsupported(
                "cute", f"tile_index load index type: {type(idx)}"
            )
        return expr_from_string(index_var)

    cute_state = state.device_function.cute_state
    if cute_state.suppress_root_lane_loops or (
        state.fx_node is not None
        and cute_state.is_collective_handled_load(state.fx_node.name)
    ):
        zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
        return expr_from_string(f"{zero}(0)")

    packed_affine_lhs = _maybe_codegen_cute_packed_affine_lhs_load(
        state, tensor, subscript, extra_mask
    )
    if packed_affine_lhs is not None:
        return packed_affine_lhs

    packed_rhs_load = _maybe_codegen_cute_packed_rhs_load(
        state, tensor, subscript, extra_mask
    )
    if packed_rhs_load is not None:
        return packed_rhs_load

    if _is_cute_affine_range_load_for_store(state, subscript, ast_subscript):
        zero = _cute_scalar_storage_dtype(tensor.dtype)
        return expr_from_string(f"{zero}(0)")
    if _is_cute_strided_slice_load_for_store(state, tensor, subscript):
        zero = _cute_scalar_storage_dtype(tensor.dtype)
        return expr_from_string(f"{zero}(0)")

    tensor_name = state.device_function.tensor_arg(tensor).name
    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    regions = _cute_access_regions(state, subscript, tensor)
    mask_expr = _cute_resolved_load_mask(
        state, tensor, subscript, index_exprs, extra_mask
    )
    # Autotunable per-load-site cache hint, applied to the ``cute.arch.load``
    # forms (vectorized, and scalar when a hint is set).  "first"/"last" are
    # L1 eviction priorities; "streaming" is the ``ld.global.cs`` cache
    # operator (evict-first at both L1 and L2 — single-use streaming reads
    # stop displacing useful L2 lines).  Same site-order indexing scheme as
    # the Triton backend.
    eviction_suffix = ""
    if state.codegen.on_device:
        device_fn = state.device_function
        load_idx = device_fn.device_load_index
        device_fn.device_load_index += 1
        policies = state.config.load_eviction_policies
        if load_idx < len(policies):
            policy = policies[load_idx]
            if policy == "streaming":
                eviction_suffix = ", cop='cs'"
            elif policy in ("l2_last", "l1_l2_first", "l1_l2_last"):
                # Inline-PTX L2 hints apply only to aligned 16-byte packets.
                eviction_suffix = f"__{policy}__"
            elif mapped := _CUTE_EVICTION_POLICY_MAP.get(policy, ""):
                eviction_suffix = f", level1_eviction_priority={mapped!r}"
    if state.fx_node is not None:
        _record_cute_scalar_load_site(
            state,
            tensor,
            subscript,
            tensor_name,
            index_exprs,
            mask_expr,
            extra_mask is not None,
            eviction_suffix,
        )
    load_expr: str | None = None
    load_placeholders: dict[str, ast.AST] = {}
    branch_vec_candidate: tuple[int, int] | None = None
    vec_ctx = _cute_vector_load_ctx(
        state, tensor, subscript, index_exprs, extra_mask, allow_gathered_index=True
    )
    # Mask terms other than the lane mask that predicate the whole packet of a
    # ``tile_unroll`` site (see below); None when the mask is lane-only.
    uniform_mask: str | None = None
    if vec_ctx is not None and not _cute_vector_load_mask_is_lane_only(
        mask_expr, _cute_active_mask_var(state, vec_ctx[1])
    ):
        from ..reduction_strategy import LoopedReductionStrategy
        from ..reduction_strategy import PersistentReductionStrategy

        vec_width, vec_block_id, vec_mode = vec_ctx
        strategy = _cute_lane_strategy(state, vec_block_id)
        if vec_mode == "tile_unroll":
            # An outer tile mask or the bound of a gathered coordinate is
            # uniform across the V lanes when it only reads values available
            # above the constexpr V-loop.  Such a site keeps its packet: the
            # terms select a safe anchor pointer for the whole load (see
            # ``_cute_register_tile_unroll_vec_hoist``) while the per-element
            # gate below still zeroes the masked values.  Anything else stays
            # behind its scalar predicate.
            assert mask_expr is not None
            uniform_mask = _cute_vector_load_non_lane_mask(
                mask_expr, _cute_active_mask_var(state, vec_block_id)
            )
            scope = _cute_tile_unroll_vloop_scope(state, strategy, vec_block_id)
            if (
                uniform_mask is None
                or scope is None
                or _cute_tile_unroll_uniform_definitions(
                    scope,
                    index_exprs,
                    _cute_lane_axis_pos(strategy, vec_block_id, index_exprs),
                    uniform_mask,
                )
                is None
            ):
                uniform_mask = None
                vec_ctx = None
        elif (
            vec_mode == "unroll"
            and isinstance(strategy, LoopedReductionStrategy)
            and _cute_looped_vec_mask_is_uniform(
                strategy, mask_expr, _cute_active_mask_var(state, vec_block_id)
            )
        ):
            # The outer terms (a row mask) are defined above the lane body, so
            # the rolled hoist folds them into its packet guard.
            pass
        else:
            if (
                vec_mode == "unroll"
                and isinstance(strategy, PersistentReductionStrategy)
                and _persistent_vec_is_exact_aligned(
                    state, strategy, index_exprs, tensor, vec_width
                )
            ):
                # A resolved tensor-index mask can protect an outer row/gather
                # as well as the lane. Keep it on the scalar marker: the late
                # local pass proves whole-fragment validity and a safe inactive
                # pointer before placing the vector transaction inside its
                # legal scope.
                branch_vec_candidate = (vec_block_id, vec_width)
            vec_ctx = None
    if vec_ctx is not None:
        vec_width, vec_block_id, vec_mode = vec_ctx
        from ..reduction_strategy import LoopedReductionStrategy
        from ..reduction_strategy import PersistentReductionStrategy

        strategy = _cute_lane_strategy(state, vec_block_id)
        if vec_mode == "vec":
            load_expr = _cute_vector_load_expr(
                tensor_name,
                index_exprs,
                tensor.dtype,
                vec_width=vec_width,
                eviction_suffix=eviction_suffix,
            )
            # The mask is deferred to the post-fold scalar in
            # codegen_reduction.  The vec load itself is unconditional; the
            # mask is recorded on the active LoopedReductionStrategy and
            # applied around the folded sum.
            if isinstance(strategy, LoopedReductionStrategy):
                strategy._cute_emitted_vec_load = True
                if mask_expr is not None:
                    strategy._cute_pending_vec_masks.append(mask_expr)
            mask_expr = None
        elif vec_mode == "unroll":
            # Register (or reuse) a hoisted U16 vec load for this (tensor,
            # base_index) pair, then return ``hoist_var[vi].bitcast(dtype)``
            # so the existing scalar pipeline sees a scalar of the original
            # dtype.
            assert isinstance(
                strategy,
                (LoopedReductionStrategy, PersistentReductionStrategy),
            )
            load_expr = _cute_register_unroll_vec_hoist(
                state,
                strategy,
                tensor,
                tensor_name,
                index_exprs,
                vec_width,
                mask_expr=mask_expr,
                eviction_suffix=eviction_suffix,
            )
            # A persistent vec wrapper cannot hoist an address component
            # produced by its own root body. ``None`` keeps that site scalar
            # until the late branch-local vectorizer can place it legally.
            if (
                load_expr is None
                and isinstance(strategy, PersistentReductionStrategy)
                and _persistent_vec_is_exact_aligned(
                    state, strategy, index_exprs, tensor, vec_width
                )
            ):
                branch_vec_candidate = (vec_block_id, vec_width)
        else:
            assert vec_mode == "tile_unroll"
            # Same hoist protocol as ``LoopedReductionStrategy``'s
            # ``unroll`` mode but for ``PerThreadNDTileStrategy`` lane loops.
            from ..tile_strategy import DeviceGridState
            from ..tile_strategy import PerThreadFlattenedTileStrategy
            from ..tile_strategy import PerThreadNDTileStrategy

            assert isinstance(
                strategy, (PerThreadNDTileStrategy, PerThreadFlattenedTileStrategy)
            )
            lane_axis_pos = _cute_lane_axis_pos(strategy, vec_block_id, index_exprs)
            # The hoist may be emitted when the root body is wrapped; record a
            # signed-byte packet against the load's original lowering site.
            load_site = None
            if tensor.dtype is torch.int8:
                from .signed_bitfield import signed_byte_site

                load_site = signed_byte_site(state)
            # Coordinates and mask terms this packet needs but the V-loop body
            # defines (a gathered row index and its bound) are relocated above
            # the V-loop when the packet is emitted, so they run once per
            # packet and the hoist can read them; a shifted lane coordinate
            # and mask terms on the per-element index are re-expressed at the
            # lane base.  Sites whose coordinates are already available and
            # whose mask is lane-only need no relocation and keep their
            # existing lowering untouched.
            needs_hoist_plan = (
                uniform_mask is not None
                or index_exprs[lane_axis_pos]
                != _cute_active_index_var(state, vec_block_id)
                or not _cute_tile_unroll_scope_safe(
                    state, strategy, vec_block_id, index_exprs, lane_axis_pos
                )
            )
            # The memory-effect gate in ``emit_tile_load`` reads the V-loop
            # body whether or not the packet needs a relocation plan.
            vloop_scope = _cute_tile_unroll_vloop_scope(state, strategy, vec_block_id)
            relocation_scope = vloop_scope if needs_hoist_plan else None
            # A grid emits the packet once its root body is wrapped; the
            # scalar placeholder then marks where this load sits in that body.
            deferred_site = isinstance(
                state.codegen.active_device_loops[vec_block_id][-1], DeviceGridState
            )
            scalar_load = expr_from_string(
                _cute_scalar_load_expr(
                    tensor_name,
                    index_exprs,
                    tensor.dtype,
                    eviction_suffix=eviction_suffix,
                )
            )

            def emit_tile_load() -> ast.AST | None:
                # The packet reads before every statement of the V-loop body,
                # so a store, atomic or barrier that precedes this load in the
                # body must be proven not to touch this tensor.
                preceding = (
                    _cute_tile_unroll_preceding_statements(
                        vloop_scope.body, scalar_load if deferred_site else None
                    )
                    if vloop_scope is not None
                    else None
                )
                if preceding is None or not _cute_tile_unroll_hoist_allowed(
                    state, strategy, vec_block_id, preceding, tensor_name
                ):
                    return None
                packet_mask = uniform_mask
                lane_base_expr = None
                if needs_hoist_plan:
                    if relocation_scope is None:
                        return None
                    plan = _cute_tile_unroll_uniform_definitions(
                        relocation_scope, index_exprs, lane_axis_pos, uniform_mask
                    )
                    if plan is None:
                        # The body changed since admission; keep the scalar load.
                        return None
                    _cute_relocate_above_vloop(relocation_scope, plan.moved)
                    packet_mask = plan.uniform_mask
                    lane_base_expr = plan.lane_base_expr
                return expr_from_string(
                    _cute_register_tile_unroll_vec_hoist(
                        state,
                        strategy,
                        vec_block_id,
                        tensor,
                        tensor_name,
                        index_exprs,
                        vec_width,
                        eviction_suffix=eviction_suffix,
                        lane_axis_pos=lane_axis_pos,
                        mask_expr=mask_expr,
                        signed_byte_site=load_site,
                        uniform_mask=packet_mask,
                        lane_base_expr=lane_base_expr,
                    )
                )

            if _cute_defer_grid_vector_op(
                state, strategy, vec_block_id, scalar_load, emit_tile_load
            ):
                load_placeholders["tile_vector_load"] = scalar_load
                load_expr = "{tile_vector_load}"
            else:
                vector_load = emit_tile_load()
                if vector_load is not None:
                    load_expr = ast.unparse(vector_load)
    if load_expr is None:
        load_expr = _cute_scalar_load_expr(
            tensor_name,
            index_exprs,
            tensor.dtype,
            eviction_suffix=eviction_suffix,
        )
        _cute_record_sinkable_scalar_load(
            state, tensor, subscript, index_exprs, tensor_name, eviction_suffix
        )
    if tensor.dtype is torch.bool:
        load_expr = f"({load_expr} != cutlass.Uint8(0))"
        if mask_expr is None:
            return expr_from_string(load_expr, **load_placeholders)
        return expr_from_string(
            f"({load_expr} if {mask_expr} else cutlass.Boolean(0))",
            **load_placeholders,
        )
    if state.fx_node is not None and _cute_load_feeds_sort_or_scan(state.fx_node):
        from .indexing import CuteSortableLoad

        tensor_dim = 0
        sort_index_pos = -1
        for idx in subscript:
            if idx is None:
                continue
            if tensor_dim == tensor.ndim - 1:
                sort_index_pos = tensor_dim
                break
            tensor_dim += 1
        if sort_index_pos < 0:
            raise exc.BackendUnsupported("cute", "sort/topk input rank")
        sortable_load = CuteSortableLoad(
            expr=expr_from_string(
                load_expr
                if mask_expr is None
                else f"({load_expr} if {mask_expr} else {_cute_scalar_storage_dtype(tensor.dtype)}(0))",
                **load_placeholders,
            ),
            tensor_name=tensor_name,
            index_exprs=tuple(index_exprs),
            sort_index_pos=sort_index_pos,
            mask_expr=mask_expr,
            dtype=tensor.dtype,
        )
        state.fx_node.meta["cute_sortable_load"] = sortable_load
        return sortable_load.expr
    if mask_expr is None:
        result = expr_from_string(load_expr, **load_placeholders)
        assert isinstance(result, ast.expr)
        _cute_tag_access_regions(result, tensor_name, regions)
        if branch_vec_candidate is not None:
            vec_block_id, vec_width = branch_vec_candidate
            return _persistent_branch_vec_load_marker(
                vec_block_id,
                vec_width,
                tensor.dtype,
                eviction_suffix,
                _cute_scalar_pointer_expr(tensor_name, index_exprs),
                result,
            )
        return result
    zero = _cute_scalar_storage_dtype(tensor.dtype)
    result = expr_from_string(
        f"({load_expr} if {mask_expr} else {zero}(0))", **load_placeholders
    )
    assert isinstance(result, ast.expr)
    _cute_tag_access_regions(result, tensor_name, regions)
    if branch_vec_candidate is not None:
        vec_block_id, vec_width = branch_vec_candidate
        return _persistent_branch_vec_load_marker(
            vec_block_id,
            vec_width,
            tensor.dtype,
            eviction_suffix,
            _cute_scalar_pointer_expr(tensor_name, index_exprs),
            result,
        )
    return result
