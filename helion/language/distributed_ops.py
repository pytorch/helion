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
all-to-all where a rank both sends and receives rows, either a subsequent
device-/host-wide barrier (e.g. ``dist.barrier()``) or the compiler-managed
counting receive drain ``hl.wait_async_remote_recv(dst, count)`` establishes
cross-rank visibility of the inbound writes before reading ``dst``.

Both the Pallas (TPU) and Triton/NVSHMEM (GPU) backends are wired up.  On Pallas,
``start_async_remote_copy`` lowers to::

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
from . import _decorators
from ._decorators import args_to_proxies

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .._compiler.type_info import Origin
    from .._compiler.type_info import TypeInfo


__all__ = [
    "AsyncCopyDescriptor",
    "start_async_remote_copy",
    "sync_barrier",
    "wait_async_remote_copy",
    "wait_async_remote_recv",
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
    device_id: int | torch.Tensor,
    dst: torch.Tensor | None = None,
    dst_index: list[object] | None = None,
    block_rows: int = 1,
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
    device_id: int | torch.Tensor,
    dst: torch.Tensor | None = None,
    dst_index: list[object] | None = None,
    block_rows: int = 1,
) -> tuple[
    torch.Tensor, list[object], int | torch.Tensor, torch.Tensor, list[object], int
]:
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
        _validate_copy_contract(src, src_index, dst, dst_index, block_rows)
    return (src, src_index, device_id, dst, dst_index, block_rows)


@_decorators.type_propagation(start_async_remote_copy)
def _(*args: TypeInfo, origin: Origin, **kwargs: TypeInfo) -> TypeInfo:
    # Runs in the type-propagation pass over the *source* AST, so it mirrors the
    # user's call: ``dst`` / ``dst_index`` may be passed positionally or by
    # keyword, and the symmetric form omits them entirely -- all before
    # ``prepare_args`` resolves the defaults.  The result type is independent of
    # the arguments, so accept any positional/keyword arity.
    from .._compiler.type_info import AsyncCopyDescriptorType

    return AsyncCopyDescriptorType(origin=origin, element_types={})


def _slot_numel(shape: Iterable[object], ndim_indexed: int) -> int:
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
    block_rows: int = 1,
) -> None:
    """Raise if src/dst can't be a valid raw remote memcpy (dtype + element
    count).  Shared by prepare_args so both backends fail identically.

    ``block_rows > 1`` copies a contiguous block of ``block_rows`` rows starting
    at the *last* index element (on its dim), so that dim contributes
    ``block_rows`` to the element count instead of collapsing to a single row.
    """
    if src.dtype != dst.dtype:
        raise exc.TypeInferenceError(
            "start_async_remote_copy: src and dst must share a dtype "
            f"(got src={src.dtype}, dst={dst.dtype})"
        )
    src_slot = _slot_numel(src.shape, len(src_index))
    dst_slot = _slot_numel(dst.shape, len(dst_index))
    if src_slot >= 0:
        src_slot *= block_rows
    if dst_slot >= 0:
        dst_slot *= block_rows
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
    device_id: int | torch.Tensor,
    dst: torch.Tensor | None = None,
    dst_index: list[object] | None = None,
    block_rows: int = 1,
) -> AsyncCopyDescriptor:
    return AsyncCopyDescriptor()


@_decorators.register_to_device_ir(start_async_remote_copy)
def _(
    tracer: object,
    src: torch.Tensor,
    src_index: list[object],
    device_id: int | torch.Tensor,
    dst: torch.Tensor,
    dst_index: list[object],
    block_rows: int = 1,
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
        *args_to_proxies(
            tracer, (src, src_index, device_id, dst, dst_index, block_rows), {}
        ),  # type: ignore[arg-type]
    )
    descriptor = AsyncCopyDescriptor()
    descriptor._proxy = proxy_out
    # Helion's codegen expects every FX node to carry a fake ``val``
    # entry in its meta dict; the default proxy path does this via
    # ``proxy_tensor.track_tensor_tree`` but we bypassed that above.
    proxy_out.node.meta["val"] = descriptor
    return descriptor


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


# ---------------------------------------------------------------------------
# wait_recv — drain the RECEIVE-side counting semaphore at kernel exit.
#
# In a fused reduce-scatter each rank both sends its FFN rows to peers and
# receives peer rows into a local buffer.  ``wait_send`` drains only the
# outgoing side; every *inbound* push increments a receive semaphore that
# Mosaic requires to be zero at kernel exit.  This op consumes exactly
# ``count`` of those increments in a single wait -- the receive-side
# counterpart of ``wait_send`` that makes the direct-write scatter self-
# contained (no external barrier needed for the semaphore itself).


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def wait_async_remote_recv(dst: torch.Tensor, count: int) -> None:
    """Drain the receive-side DMA semaphore ``count`` times before kernel exit.

    A fused reduce-scatter direct-writes peer FFN rows into the local ``dst``
    buffer via :func:`start_async_remote_copy`; each landed row increments a
    compiler-allocated *receive* semaphore by one.  Mosaic requires every
    semaphore to read zero at kernel exit, so the receiving rank must consume
    exactly ``count`` increments -- the total number of rows peers (and this
    rank's own local self-pushes) wrote into ``dst``.

    Lowers to a single self-referential wait::

        pltpu.make_async_copy(dst[0:count], dst[0:count], recv_sem).wait()

    which blocks until ``count`` row-sized increments have accumulated on
    ``recv_sem`` (no data moves -- src and dst are the same slice).  Because the
    wait is *counted*, it also acts as the receive-side completion barrier: the
    kernel cannot exit until every expected inbound row has landed, so a
    ``dst`` read after the kernel is coherent without a separate device barrier.

    Args:
        dst: The local symmetric buffer peers pushed into (same tensor passed as
            ``dst`` to the paired ``start_async_remote_copy`` calls).  Only its
            leading-dim slice ``dst[0:count]`` is used, as a wait placeholder.
        count: Total number of inbound rows to drain.  For static routing this
            is known at compile time (each rank compiles its own kernel), e.g.
            ``dst.shape[0]`` when every row of ``dst`` is written exactly once.

    Must be reached *after* the ``start_async_remote_copy`` calls whose
    ``recv_sem`` it drains (they allocate the semaphore); it uses the receive
    semaphore stashed on the device function by the ``start`` codegen.
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(wait_async_remote_recv)
def _(dst: TypeInfo, count: TypeInfo, *, origin: Origin) -> TypeInfo:
    from .._compiler.type_info import NoType

    return NoType(origin=origin)


@_decorators.register_fake(wait_async_remote_recv)
def _(dst: torch.Tensor, count: int) -> None:
    return None


@_decorators.register_to_device_ir(wait_async_remote_recv)
def _(tracer: object, dst: torch.Tensor, count: int) -> None:
    tracer.create_proxy(  # type: ignore[attr-defined]
        "call_function",
        wait_async_remote_recv,
        *args_to_proxies(tracer, (dst, count), {}),  # type: ignore[arg-type]
    )
    return None


# ---------------------------------------------------------------------------
# sync_barrier — an in-kernel cross-chip barrier.
#
# The collective-completion counterpart to the async copy ops: a fused
# comms+compute kernel (e.g. an expert-parallel MoE reduce-scatter) must know
# that every peer has reached a given point before it drains its receive side.
# This is the Pallas ``get_barrier_semaphore`` idiom -- signal every peer, then
# wait for ``world_size`` signals -- lifted into a Helion primitive so a
# self-contained collective kernel needs no host-side ``dist.barrier``.


@has_side_effect
@_decorators.api(is_device_only=True)
def sync_barrier(world_size: int) -> None:
    """Barrier across all ``world_size`` chips from inside the kernel.

    Every chip signals a barrier semaphore on every chip (itself included) and
    then waits for ``world_size`` signals, so no chip proceeds until all have
    arrived.  Use it around a fused reduce-scatter / all-to-all: once at entry
    (so no peer starts writing into a local buffer before this chip is ready)
    and once before the receive-drain (so every peer's sends have been issued).

    Args:
        world_size: The number of participating chips (compile-time constant).
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(sync_barrier)
def _(world_size: TypeInfo, *, origin: Origin) -> TypeInfo:
    from .._compiler.type_info import NoType

    return NoType(origin=origin)


@_decorators.register_fake(sync_barrier)
def _(world_size: int) -> None:
    return None


@_decorators.register_to_device_ir(sync_barrier)
def _(tracer: object, world_size: int) -> None:
    tracer.create_proxy(  # type: ignore[attr-defined]
        "call_function",
        sync_barrier,
        *args_to_proxies(tracer, (world_size,), {}),  # type: ignore[arg-type]
    )
    return None
