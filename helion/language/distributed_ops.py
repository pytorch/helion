"""Distributed / RDMA primitives for Helion kernels.

Exposes an explicitly-async push-based RDMA API::

    op = hl.start_async_remote_copy(src, src_index, device_id)
    # ... compute overlapping the copy ...
    op.wait()

which pushes ``src[src_index]`` (local) into ``src[src_index]`` on the peer
identified by ``device_id`` (LOGICAL / NVSHMEM-style flat PE id).  This
symmetric, same-slot form is what ring all-gather kernels use (see
``examples/distributed/all_gather.py``).

For fused comms+compute kernels (reduce-scatter / all-to-all) the push must be
*asymmetric* -- a local staging row written into an arbitrary slot of a
*different* peer buffer.  That is expressed with the optional ``dst`` /
``dst_index`` arguments::

    op = hl.start_async_remote_copy(
        src, src_index, device_id, dst=out_buf, dst_index=[write_pos]
    )
    # pushes  src[src_index]  ->  dst[dst_index]  on peer device_id

``device_id``, ``src_index`` and ``dst_index`` may all be *runtime* (data-
dependent) scalars, so a loop can scatter each row to a different peer/slot.

Completion semantics: ``op.wait()`` guarantees only that the *locally-initiated*
push has drained on this rank.  It does **not** establish that *inbound* peer
writes into a local ``dst`` buffer have landed -- for a reduce-scatter /
all-to-all where a rank both sends and receives rows, a subsequent device- or
host-wide barrier (e.g. ``dist.barrier()``) is required before reading ``dst``
cross-rank.  A compiler-managed receive-side drain (counting-semaphore + one
wait) is future work; see the ``examples/distributed`` kernels for the current
barrier-based finalize.

Only the Pallas backend is wired up today.  ``start_async_remote_copy`` lowers
to::

    _op = pltpu.make_async_remote_copy(
        src.at[src_index],
        dst.at[dst_index],
        send_sem,
        recv_sem,
        device_id=device_id,
        device_id_type=pl.DeviceIdType.LOGICAL,
    )
    _op.start()

and the paired ``op.wait()`` lowers to ``_op.wait()`` where ``_op`` is the same
variable name emitted by the ``start`` op.  Send/recv DMA semaphores are
allocated by the compiler as Pallas scratch buffers via
:meth:`DeviceFunction.register_dma_semaphore`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .. import exc
from .._compiler.ast_extension import expr_from_string
from .._compiler.ast_extension import statement_from_string
from . import _decorators
from ._decorators import args_to_proxies

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState
    from .._compiler.type_info import Origin
    from .._compiler.type_info import TypeInfo


__all__ = [
    "AsyncCopyDescriptor",
    "start_async_remote_copy",
    "wait_async_remote_copy",
    "wait_send_async_remote_copy",
]


class AsyncCopyDescriptor:
    """Handle returned by :func:`start_async_remote_copy`.

    Call ``.wait()`` on the descriptor at the point in the kernel body
    where the copy's completion is needed.  The compiler pairs each
    ``wait`` call with the ``pltpu.make_async_remote_copy`` op emitted
    by the corresponding ``start_async_remote_copy``.
    """

    # Set by ``start_async_remote_copy``'s ``_to_device_ir`` handler at
    # trace time.  Points to the FX proxy for the ``start`` call, which
    # ``wait_async_remote_copy``'s handler feeds as the first arg of
    # its own FX call node so codegen can pair them up.
    _proxy: object | None = None

    def wait(self) -> None:
        # At kernel-trace time the tracer sees a real AsyncCopyDescriptor
        # instance (returned by ``start_async_remote_copy``'s
        # ``_to_device_ir`` handler) and dispatches this method to
        # ``wait_async_remote_copy(self)``, which adds a call_function
        # node to the FX graph.  Outside a kernel this is an error.
        return wait_async_remote_copy(self)

    def wait_send(self) -> None:
        """Wait for THIS rank's *outgoing* push to drain (the local send buffer
        is free / the data has been delivered to the peer), rather than for an
        *incoming* push into a local buffer.

        Use this for a reduce-scatter / all-to-all direct-write where each rank
        sends a data-dependent number of rows: every ``start`` is paired with
        exactly one ``wait_send`` (balanced), so it never deadlocks on an uneven
        receive count the way ``wait()`` (receive-side) would.  Cross-rank
        visibility of the delivered rows is then established by a subsequent
        device/host barrier (see the module docstring).
        """
        return wait_send_async_remote_copy(self)


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True, tiles_as_sizes=True)
def start_async_remote_copy(
    src: torch.Tensor,
    src_index: list[object],
    device_id: int,
    dst: torch.Tensor | None = None,
    dst_index: list[object] | None = None,
) -> AsyncCopyDescriptor:
    """Start an async push of ``src[src_index]`` to peer ``device_id``.

    By default (``dst is None``) the destination is the *same* tensor and
    *same* slot on the peer -- the symmetric form used by ring all-gather.
    Pass ``dst`` / ``dst_index`` to push into a *different* peer buffer or a
    *different* slot (the reduce-scatter / all-to-all form).

    ``src`` and ``dst`` must share a dtype and a matching per-slot element count
    (the copy is a raw memcpy, no conversion); this is validated at trace time.

    Args:
        src: The (host-side) symmetric source tensor whose slot is copied.
            Every rank must pass a tensor of the same shape and dtype at the
            same argument position; the caller is responsible for that.
        src_index: List selecting the source slot (``src[src_index]``).
        device_id: Flat integer PE id (0 <= device_id < world_size).  May be a
            runtime scalar (e.g. a per-row peer id) or a compile-time constant.
        dst: Optional destination tensor on the peer.  Defaults to ``src``.
        dst_index: Optional destination slot (``dst[dst_index]`` on the peer).
            Defaults to ``src_index`` (same slot).

    Returns:
        An :class:`AsyncCopyDescriptor` whose ``wait()`` method must be
        called before the copy's effects are observed.  ``wait()`` covers
        *send* completion only; see the module docstring for the receive-side
        (cross-rank ``dst`` visibility) barrier requirement.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(start_async_remote_copy)
def _(
    src: torch.Tensor,
    src_index: list[object],
    device_id: int,
    dst: torch.Tensor | None = None,
    dst_index: list[object] | None = None,
) -> tuple[torch.Tensor, list[object], int, torch.Tensor, list[object]]:
    from .tile_proxy import Tile

    src_index = Tile._prepare_index(src_index)
    src_index = Tile._tiles_to_sizes_for_index(src_index)
    # Resolve the symmetric defaults here so every downstream stage
    # (type-prop / fake / device-ir / codegen) sees concrete src+dst refs.
    if dst is None:
        dst = src
    if dst_index is None:
        dst_index = src_index
    else:
        dst_index = Tile._prepare_index(dst_index)
        dst_index = Tile._tiles_to_sizes_for_index(dst_index)
    # Validate the src<->dst contract once, here, so BOTH backends fail
    # identically at trace time (the custom register_to_device_ir below bypasses
    # register_fake, so this is the single choke point that always runs).  The
    # copy is a raw memcpy: dtypes must match and the selected slots must hold
    # the same number of elements.
    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
        _validate_copy_contract(src, src_index, dst, dst_index)
    return (src, src_index, device_id, dst, dst_index)


@_decorators.type_propagation(start_async_remote_copy)
def _(*args: TypeInfo, origin: Origin, **kwargs: TypeInfo) -> TypeInfo:
    # Runs in the type-propagation pass over the *source* AST, so it mirrors the
    # user's call: ``dst`` / ``dst_index`` may be passed positionally or by
    # keyword, and the symmetric form omits them entirely -- all before
    # ``prepare_args`` resolves the defaults.  The result type is independent of
    # the arguments, so accept any positional/keyword arity.
    from .._compiler.type_info import AsyncCopyDescriptorType

    return AsyncCopyDescriptorType(origin=origin, element_types={})


def _slot_numel(shape: object, ndim_indexed: int) -> int:
    """Number of elements in ``tensor[index]`` where ``index`` selects the
    leading ``ndim_indexed`` dims -- i.e. the product of the trailing dims.
    Non-integer (symbolic) trailing dims make the count indeterminate, so
    return ``-1`` to skip the equality check rather than guess.
    """
    numel = 1
    for d in list(shape)[ndim_indexed:]:
        if not isinstance(d, int):
            return -1
        numel *= d
    return numel


def _validate_copy_contract(
    src: torch.Tensor,
    src_index: list[object],
    dst: torch.Tensor,
    dst_index: list[object],
) -> None:
    """Raise if src/dst can't be a valid raw remote memcpy (dtype + element
    count).  Shared by prepare_args so both backends fail identically."""
    if src.dtype != dst.dtype:
        raise exc.TypeInferenceError(
            "start_async_remote_copy: src and dst must share a dtype "
            f"(got src={src.dtype}, dst={dst.dtype})"
        )
    src_slot = _slot_numel(src.shape, len(src_index))
    dst_slot = _slot_numel(dst.shape, len(dst_index))
    if src_slot >= 0 and dst_slot >= 0 and src_slot != dst_slot:
        raise exc.TypeInferenceError(
            "start_async_remote_copy: src[src_index] and dst[dst_index] must "
            f"have the same element count (got {src_slot} vs {dst_slot}); "
            f"src.shape={tuple(src.shape)} src_index={len(src_index)}d, "
            f"dst.shape={tuple(dst.shape)} dst_index={len(dst_index)}d"
        )


@_decorators.register_fake(start_async_remote_copy)
def _(
    src: torch.Tensor,
    src_index: list[object],
    device_id: int,
    dst: torch.Tensor | None = None,
    dst_index: list[object] | None = None,
) -> AsyncCopyDescriptor:
    return AsyncCopyDescriptor()


@_decorators.register_to_device_ir(start_async_remote_copy)
def _(
    tracer: object,
    src: torch.Tensor,
    src_index: list[object],
    device_id: int,
    dst: torch.Tensor,
    dst_index: list[object],
) -> AsyncCopyDescriptor:
    """Trace ``start_async_remote_copy`` into an FX call node manually.

    Bypasses the default proxy-args flow so the returned
    ``AsyncCopyDescriptor`` doesn't need to be a torch-recognised type
    for ``create_arg``.  The FX proxy for the ``start`` call is stashed
    on the descriptor so the paired ``wait`` can retrieve it.
    """

    proxy_out = tracer.create_proxy(  # type: ignore[attr-defined]
        "call_function",
        start_async_remote_copy,
        *args_to_proxies(tracer, (src, src_index, device_id, dst, dst_index), {}),  # type: ignore[arg-type]
    )
    descriptor = AsyncCopyDescriptor()
    descriptor._proxy = proxy_out
    # Helion's codegen expects every FX node to carry a fake ``val``
    # entry in its meta dict; the default proxy path does this via
    # ``proxy_tensor.track_tensor_tree`` but we bypassed that above.
    proxy_out.node.meta["val"] = descriptor
    return descriptor


def _codegen_at_expr(index_ast: object, prefix: str) -> tuple[str, dict[str, object]]:
    """Build a ``.at[...]`` subscript string + placeholder map from an index
    list.  Each element is either an ``int`` literal (emitted inline) or an
    already-lowered AST node (passed through as a placeholder).  ``prefix``
    namespaces the placeholders so src and dst indices never collide.
    """
    assert isinstance(index_ast, (list, tuple)), index_ast
    placeholders: dict[str, object] = {}
    parts: list[str] = []
    for i, elt in enumerate(index_ast):
        name = f"{prefix}{i}"
        if isinstance(elt, int):
            placeholders[name] = expr_from_string(repr(elt))
        else:
            placeholders[name] = elt
        parts.append(f"{{{name}}}")
    return ", ".join(parts), placeholders


@_decorators.codegen(start_async_remote_copy, "pallas")
def _(state: CodegenState) -> object:
    """Emit ``_op = pltpu.make_async_remote_copy(...); _op.start()``.

    Emits two *independent* refs (``src.at[index]`` -> ``dst.at[dst_index]``)
    so the copy can be asymmetric.  ``device_id`` may be a runtime scalar.
    Stashes the ``_op`` variable name on the FX node's meta dict so the paired
    ``wait_async_remote_copy`` codegen can look it up.
    """
    src = state.proxy_arg(0)
    dst = state.proxy_arg(3)
    assert isinstance(src, torch.Tensor)
    assert isinstance(dst, torch.Tensor)

    device_fn = state.device_function
    src_name = device_fn.tensor_arg(src).name
    dst_name = device_fn.tensor_arg(dst).name

    send_sem = device_fn.register_dma_semaphore(name_hint="send_sem")
    recv_sem = device_fn.register_dma_semaphore(name_hint="recv_sem")
    op_var = device_fn.new_var("_remote_copy", dce=False)

    # Stash the op variable name on the FX node so wait can find it.
    assert state.fx_node is not None
    state.fx_node.meta["_pallas_async_copy_op"] = op_var

    src_at, src_ph = _codegen_at_expr(state.ast_args[1], "_sidx")
    dst_at, dst_ph = _codegen_at_expr(state.ast_args[4], "_didx")

    device_id_ast = state.ast_args[2]
    if isinstance(device_id_ast, int):
        device_id_ast = expr_from_string(repr(device_id_ast))

    state.codegen.add_statement(
        statement_from_string(
            f"{op_var} = pltpu.make_async_remote_copy("
            f"{src_name}.at[{src_at}], {dst_name}.at[{dst_at}], "
            f"{send_sem}, {recv_sem}, "
            f"device_id={{device_id}}, "
            f"device_id_type=pl.DeviceIdType.LOGICAL)",
            device_id=device_id_ast,
            **src_ph,
            **dst_ph,
        )
    )
    state.codegen.add_statement(statement_from_string(f"{op_var}.start()"))
    return expr_from_string(op_var)


# ---------------------------------------------------------------------------
# wait


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def wait_async_remote_copy(descriptor: AsyncCopyDescriptor) -> None:
    """Wait for a previously-started async remote copy to complete.

    Normally invoked as ``descriptor.wait()`` — the descriptor's
    ``wait`` method dispatches here so both syntaxes are equivalent.
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(wait_async_remote_copy)
def _(
    descriptor: TypeInfo,
    *,
    origin: Origin,
) -> TypeInfo:
    from .._compiler.type_info import AsyncCopyDescriptorType
    from .._compiler.type_info import NoType

    if not isinstance(descriptor, AsyncCopyDescriptorType):
        raise exc.TypeInferenceError(
            "wait_async_remote_copy expects an AsyncCopyDescriptor "
            "(returned by hl.start_async_remote_copy)"
        )
    return NoType(origin=origin)


@_decorators.register_fake(wait_async_remote_copy)
def _(descriptor: AsyncCopyDescriptor) -> None:
    return None


@_decorators.register_to_device_ir(wait_async_remote_copy)
def _(tracer: object, descriptor: AsyncCopyDescriptor) -> None:
    """Trace ``wait_async_remote_copy`` manually, feeding the paired
    ``start`` call's FX proxy as this call's first argument so codegen
    can look up the op var by walking to the source node.
    """
    if not isinstance(descriptor, AsyncCopyDescriptor):
        raise exc.TypeInferenceError(
            "wait_async_remote_copy expects an AsyncCopyDescriptor "
            "(returned by hl.start_async_remote_copy)"
        )
    assert descriptor._proxy is not None, (
        "AsyncCopyDescriptor has no source FX proxy — this is a compiler bug"
    )
    tracer.create_proxy(  # type: ignore[attr-defined]
        "call_function",
        wait_async_remote_copy,
        (descriptor._proxy,),
        {},
    )
    return None


@_decorators.codegen(wait_async_remote_copy, "pallas")
def _(state: CodegenState) -> object:
    """Emit ``<op_var>.wait()`` by looking up the op var stashed by
    the paired ``start_async_remote_copy`` on its FX node.
    """
    assert state.fx_node is not None
    descriptor_arg = state.fx_node.args[0]
    assert isinstance(descriptor_arg, torch.fx.Node), (
        "wait_async_remote_copy argument must be an FX node returned by "
        "start_async_remote_copy"
    )
    op_var = descriptor_arg.meta.get("_pallas_async_copy_op")
    if op_var is None:
        raise exc.InternalError(
            RuntimeError(
                "wait_async_remote_copy could not find the op variable name "
                "on the descriptor's source node — check that codegen for "
                "start_async_remote_copy ran first."
            )
        )
    state.codegen.add_statement(statement_from_string(f"{op_var}.wait()"))
    return expr_from_string("None")


# ---------------------------------------------------------------------------
# wait_send — drain the OUTGOING push (send-side), not the incoming receive.


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def wait_send_async_remote_copy(descriptor: AsyncCopyDescriptor) -> None:
    """Wait for a previously-started remote copy's *send* side to drain.

    Normally invoked as ``descriptor.wait_send()``.  See
    :meth:`AsyncCopyDescriptor.wait_send` for when to prefer it over ``wait()``.
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(wait_send_async_remote_copy)
def _(descriptor: TypeInfo, *, origin: Origin) -> TypeInfo:
    from .._compiler.type_info import AsyncCopyDescriptorType
    from .._compiler.type_info import NoType

    if not isinstance(descriptor, AsyncCopyDescriptorType):
        raise exc.TypeInferenceError(
            "wait_send_async_remote_copy expects an AsyncCopyDescriptor "
            "(returned by hl.start_async_remote_copy)"
        )
    return NoType(origin=origin)


@_decorators.register_fake(wait_send_async_remote_copy)
def _(descriptor: AsyncCopyDescriptor) -> None:
    return None


@_decorators.register_to_device_ir(wait_send_async_remote_copy)
def _(tracer: object, descriptor: AsyncCopyDescriptor) -> None:
    if not isinstance(descriptor, AsyncCopyDescriptor):
        raise exc.TypeInferenceError(
            "wait_send_async_remote_copy expects an AsyncCopyDescriptor "
            "(returned by hl.start_async_remote_copy)"
        )
    assert descriptor._proxy is not None, (
        "AsyncCopyDescriptor has no source FX proxy — this is a compiler bug"
    )
    tracer.create_proxy(  # type: ignore[attr-defined]
        "call_function",
        wait_send_async_remote_copy,
        (descriptor._proxy,),
        {},
    )
    return None


@_decorators.codegen(wait_send_async_remote_copy, "pallas")
def _(state: CodegenState) -> object:
    """Emit ``<op_var>.wait_send()`` (drain the local send) using the op var
    stashed by the paired ``start_async_remote_copy``."""
    assert state.fx_node is not None
    descriptor_arg = state.fx_node.args[0]
    assert isinstance(descriptor_arg, torch.fx.Node), (
        "wait_send_async_remote_copy argument must be an FX node returned by "
        "start_async_remote_copy"
    )
    op_var = descriptor_arg.meta.get("_pallas_async_copy_op")
    if op_var is None:
        raise exc.InternalError(
            RuntimeError(
                "wait_send_async_remote_copy could not find the op variable name "
                "on the descriptor's source node."
            )
        )
    state.codegen.add_statement(statement_from_string(f"{op_var}.wait_send()"))
    return expr_from_string("None")


# ---------------------------------------------------------------------------
# GPU (Triton) lowering -> NVSHMEM point-to-point put-with-signal.
#
# The same push-based, async, LOGICAL-PE-addressed model maps 1:1 onto the
# NVSHMEM Triton device API (torch.distributed._symmetric_memory._nvshmem_triton):
#   start_async_remote_copy(src[i] -> dst[wp] @ pe)
#       -> nvshmem_putmem_signal_block(dst_row_ptr, src_row_ptr, nbytes,
#                                      sig, 1, NVSHMEM_SIGNAL_ADD, pe)
#   wait()  -> nvshmem_signal_wait_until(sig, NVSHMEM_CMP_GE, 1)
# device_id (pe), and the src/dst row indices may all be runtime scalars, exactly
# as on the Pallas side.
#
# Integration NOTE (for a runnable GPU path, beyond this codegen):
#   * the generated kernel must be decorated ``@requires_nvshmem`` so Triton links
#     ``libnvshmem_device`` (extern_libs) -- wire this into
#     TritonBackend.function_decorator_for_args / the launcher (see the map in
#     device_function.codegen_function_def).
#   * ``sig`` must be a compiler-allocated *symmetric* uint64 signal buffer (the
#     GPU analog of the Pallas DMA semaphore).  This is emitted below as a named
#     var so the paired wait can reference it; allocating it symmetrically is the
#     GPU counterpart of register_dma_semaphore.
# The lowering + generated code are meant to be inspected (no NVSHMEM runtime is
# required to emit or read them).


def _triton_slot_ptr(
    tensor: torch.Tensor, index_ast: object, base_name: str, tag: str
) -> tuple[str, dict[str, object], int]:
    """Build an nvshmem element pointer for ``tensor[index]``.

    The offset is ``base + sum_k index[k] * stride[k]`` using the tensor's real
    strides, so a non-contiguous leading dim (e.g. the column-major
    ``.t().contiguous().t()`` fp8 layout) still addresses the correct bytes, and
    an N-dim index is accepted -- mirroring the Pallas ``.at[...]`` path.  Each
    index element is either an int literal (inlined) or an already-lowered AST
    node (passed through as a placeholder namespaced by ``tag``).

    NVSHMEM putmem is a *flat* memcpy, so the selected slice ``tensor[index]``
    must be contiguous in memory (the common row-select of a row-major tensor).
    Returns ``(ptr_expr_str, placeholders, slot_numel)``.
    """
    assert (
        isinstance(index_ast, (list, tuple)) and 1 <= len(index_ast) <= tensor.ndim
    ), f"index must select 1..{tensor.ndim} leading dims (got {index_ast!r})"
    strides = tensor.stride()
    placeholders: dict[str, object] = {}
    terms: list[str] = []
    for i, elt in enumerate(index_ast):
        name = f"{tag}{i}"
        if isinstance(elt, int):
            idx_str = repr(elt)
        else:
            placeholders[name] = elt
            idx_str = f"{{{name}}}"
        terms.append(f"({idx_str}) * {int(strides[i])}")
    slot_numel = 1
    for d in tensor.shape[len(index_ast) :]:
        slot_numel *= int(d)
    return f"{base_name} + {' + '.join(terms)}", placeholders, slot_numel


@_decorators.codegen(start_async_remote_copy, "triton")
def _(state: CodegenState) -> object:
    src = state.proxy_arg(0)
    dst = state.proxy_arg(3)
    assert isinstance(src, torch.Tensor)
    assert isinstance(dst, torch.Tensor)
    device_fn = state.device_function
    src_name = device_fn.tensor_arg(src).name
    dst_name = device_fn.tensor_arg(dst).name

    src_ptr, src_ph, src_slot = _triton_slot_ptr(
        src, state.ast_args[1], src_name, "_src_idx"
    )
    dst_ptr, dst_ph, _ = _triton_slot_ptr(dst, state.ast_args[4], dst_name, "_dst_idx")
    nbytes = src_slot * src.element_size()

    device_id_ast = state.ast_args[2]
    if isinstance(device_id_ast, int):
        device_id_ast = expr_from_string(repr(device_id_ast))

    # Symmetric signal buffer (GPU analog of the Pallas DMA semaphore); the
    # paired wait references it via the FX node meta.
    sig = device_fn.new_var("_nvshmem_sig", dce=False)
    assert state.fx_node is not None
    state.fx_node.meta["_nvshmem_wait_sig"] = sig

    state.codegen.add_statement(
        statement_from_string(
            f"nvshmem_putmem_signal_block("
            f"{dst_ptr}, {src_ptr}, {nbytes}, "
            f"{sig}, 1, NVSHMEM_SIGNAL_ADD, {{device_id}})",
            device_id=device_id_ast,
            **src_ph,
            **dst_ph,
        )
    )
    return expr_from_string("None")


@_decorators.codegen(wait_async_remote_copy, "triton")
def _(state: CodegenState) -> object:
    assert state.fx_node is not None
    descriptor_arg = state.fx_node.args[0]
    assert isinstance(descriptor_arg, torch.fx.Node)
    sig = descriptor_arg.meta.get("_nvshmem_wait_sig")
    if sig is None:
        raise exc.InternalError(
            RuntimeError(
                "wait_async_remote_copy (triton) could not find the NVSHMEM "
                "signal var on the start node."
            )
        )
    state.codegen.add_statement(
        statement_from_string(f"nvshmem_signal_wait_until({sig}, NVSHMEM_CMP_GE, 1)")
    )
    return expr_from_string("None")


@_decorators.codegen(wait_send_async_remote_copy, "triton")
def _(state: CodegenState) -> object:
    """Drain the local send on GPU with ``nvshmem_quiet`` (all outstanding puts
    from this PE have completed) -- the NVSHMEM analog of ``.wait_send()``.  No
    signal is consulted: this waits on the OUTGOING side, not an incoming push.
    """
    state.codegen.add_statement(statement_from_string("nvshmem_quiet()"))
    return expr_from_string("None")
