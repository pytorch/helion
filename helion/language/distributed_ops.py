"""Device-side communication primitives.

The API intentionally describes only a one-sided push.  That is the common
operation supported by TPU Pallas remote DMA and GPU NVSHMEM, and it is enough
to build collectives and direct-write compute/communication pipelines.

Launching a distributed Pallas kernel is a host-side concern.  The generated
local-shard function must execute under ``jax.shard_map``.  This is true for
both a single-process JAX program and a multi-process TorchTPU program (where
``torch_tpu._internal.pallas.jax_op`` exports the shard-mapped function).

The Triton host must allocate the destination as symmetric memory.  Helion's
launcher rendezvouses that allocation and reserves completion slots in its
signal pad.  The local source tensor need not be symmetric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx import has_side_effect

from .. import exc
from . import _decorators
from ._decorators import args_to_proxies

if TYPE_CHECKING:
    from .._compiler.type_info import Origin
    from .._compiler.type_info import TypeInfo


__all__ = [
    "AsyncCopyDescriptor",
    "make_async_remote_copy",
    "remote_barrier",
]


_REMOTE_COPY_DESCRIPTOR_ID_META = "_helion_remote_copy_descriptor_id"
_REMOTE_COPY_DESCRIPTOR_OPS_META = "_helion_remote_copy_descriptor_ops"


class AsyncCopyDescriptor:
    """Handle for one asynchronous remote push.

    ``wait()`` drains the outgoing send and waits for the matching incoming
    transfer.  ``wait_send()`` only makes the local source reusable, while
    ``wait_recv()`` only makes the local destination readable.

    Receive waits assume a balanced communication phase: every participant
    must issue the matching operation in the same descriptor slot.  Routed
    operations with uneven receive counts should drain sends with
    ``wait_send()`` and use an explicit counting protocol or group barrier
    before reading their destinations.
    """

    _proxy: object | None = None
    _descriptor_id: int | None = None

    def start(self) -> None:
        return start_async_remote_copy_descriptor(self)

    def wait(self) -> None:
        return wait_async_remote_copy(self)

    def wait_send(self) -> None:
        return wait_send_async_remote_copy(self)

    def wait_recv(self) -> None:
        return wait_recv_async_remote_copy(self)


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def remote_barrier(
    device_ids: int | torch.Tensor | list[int | torch.Tensor],
) -> None:
    """Synchronize ranks connected by the listed logical peer IDs.

    ``device_ids`` may be one logical peer, a one-dimensional tensor of peers,
    or a Python list. Every rank must name each peer with which it communicates.
    Pallas statically unrolls one-dimensional peer tensors and uses a two-phase
    neighbor barrier so a fast rank cannot enter the next invocation on a
    reused collective ID while a peer is still finishing the prior one.
    Triton/NVSHMEM currently strengthens this to a world barrier.

    Pallas derives a stable best-effort collective ID from the kernel definition.
    Kernels whose runtime communication groups are incompatible despite sharing
    a definition must override it with distinct
    ``Config(pallas_collective_id=...)`` values. Repeated invocations of one
    kernel may reuse its ID because this barrier is two-phase.
    """
    raise exc.NotInsideKernel


@_decorators.type_propagation(remote_barrier)
def _(*args: object, origin: Origin, **kwargs: object) -> TypeInfo:
    from .._compiler.type_info import NoType

    return NoType(origin=origin)


@_decorators.register_fake(remote_barrier)
def _(device_ids: int | torch.Tensor | list[int | torch.Tensor]) -> None:
    return None


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def make_async_remote_copy(
    src: torch.Tensor,
    src_index: list[object],
    device_id: int | torch.Tensor,
    dst: torch.Tensor | None = None,
    dst_index: list[object] | None = None,
) -> AsyncCopyDescriptor:
    """Create a descriptor for a contiguous push to ``device_id``.

    Call ``start()`` on the returned descriptor to issue the transfer.  Keeping
    construction separate from issue lets software pipelines drain a descriptor
    slot from an earlier loop iteration before reusing it for the next transfer.

    ``device_id`` is a flat logical rank in the active communication mesh.  It
    may be a compile-time integer or a runtime scalar.  By default the copy is
    symmetric: ``src[src_index]`` is pushed into the same tensor and index on
    the peer.  Supplying ``dst`` and ``dst_index`` enables direct writes into a
    different peer buffer or slot.

    Indices select leading dimensions; the remaining suffix is copied as one
    contiguous region.  Source and destination regions must have identical
    dtypes and element counts.

    Completion storage is backend-managed. Pallas allocates DMA semaphores in
    compiler scratch. Triton/NVSHMEM reserves compiler-assigned slots in the
    symmetric destination allocation's signal pad; ``dst`` must therefore be a
    symmetric-memory tensor on GPU.
    """
    raise exc.NotInsideKernel


def _prepare_remote_copy_args(
    src: torch.Tensor,
    src_index: list[object],
    device_id: int | torch.Tensor,
    dst: torch.Tensor | None = None,
    dst_index: list[object] | None = None,
) -> tuple[
    torch.Tensor,
    list[object],
    int | torch.Tensor,
    torch.Tensor,
    list[object],
]:
    from .tile_proxy import Tile

    src_index = Tile._prepare_index(src_index)
    src_index = Tile._tiles_to_sizes_for_index(src_index)
    if dst is None:
        dst = src
    if dst_index is None:
        dst_index = src_index
    else:
        dst_index = Tile._prepare_index(dst_index)
        dst_index = Tile._tiles_to_sizes_for_index(dst_index)
    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor):
        _validate_copy_contract(src, src_index, dst, dst_index)
    return src, src_index, device_id, dst, dst_index


_decorators.prepare_args(make_async_remote_copy)(_prepare_remote_copy_args)


def _remote_copy_type(*args: TypeInfo, origin: Origin, **kwargs: TypeInfo) -> TypeInfo:
    from .._compiler.type_info import AsyncCopyDescriptorType

    return AsyncCopyDescriptorType(origin=origin, element_types={})


_decorators.type_propagation(make_async_remote_copy)(_remote_copy_type)


def _suffix_numel(shape: torch.Size, indexed_dims: int) -> int | None:
    if indexed_dims > len(shape):
        return None
    numel = 1
    for dim in shape[indexed_dims:]:
        if not isinstance(dim, int):
            return None
        numel *= dim
    return numel


def _suffix_is_contiguous(tensor: torch.Tensor, indexed_dims: int) -> bool:
    expected_stride = 1
    for size, stride in zip(
        reversed(tensor.shape[indexed_dims:]),
        reversed(tensor.stride()[indexed_dims:]),
        strict=True,
    ):
        if size != 1 and stride != expected_stride:
            return False
        expected_stride *= size
    return True


def _validate_copy_contract(
    src: torch.Tensor,
    src_index: list[object],
    dst: torch.Tensor,
    dst_index: list[object],
) -> None:
    if len(src_index) > src.ndim or len(dst_index) > dst.ndim:
        raise exc.TypeInferenceError(
            "make_async_remote_copy: an index cannot exceed its tensor rank"
        )
    if any(isinstance(index, slice) for index in (*src_index, *dst_index)):
        raise exc.TypeInferenceError(
            "make_async_remote_copy: indices must select leading dimensions; "
            "slice indexing is not supported"
        )
    if src.dtype != dst.dtype:
        raise exc.TypeInferenceError(
            "make_async_remote_copy: src and dst must share a dtype "
            f"(got src={src.dtype}, dst={dst.dtype})"
        )
    src_numel = _suffix_numel(src.shape, len(src_index))
    dst_numel = _suffix_numel(dst.shape, len(dst_index))
    if src_numel is not None and dst_numel is not None and src_numel != dst_numel:
        raise exc.TypeInferenceError(
            "make_async_remote_copy: selected source and destination regions "
            f"must have the same element count (got {src_numel} and {dst_numel})"
        )
    if not _suffix_is_contiguous(src, len(src_index)):
        raise exc.TypeInferenceError(
            "make_async_remote_copy: the selected source region must be contiguous"
        )
    if not _suffix_is_contiguous(dst, len(dst_index)):
        raise exc.TypeInferenceError(
            "make_async_remote_copy: the selected destination region must be contiguous"
        )


def _remote_copy_fake(
    src: torch.Tensor,
    src_index: list[object],
    device_id: int | torch.Tensor,
    dst: torch.Tensor,
    dst_index: list[object],
) -> AsyncCopyDescriptor:
    return AsyncCopyDescriptor()


_decorators.register_fake(make_async_remote_copy)(_remote_copy_fake)


def _make_descriptor_proxy(
    tracer: object,
    target: object,
    src: torch.Tensor,
    src_index: list[object],
    device_id: int | torch.Tensor,
    dst: torch.Tensor,
    dst_index: list[object],
) -> AsyncCopyDescriptor:
    proxy_out = tracer.create_proxy(  # type: ignore[attr-defined]
        "call_function",
        target,
        *args_to_proxies(
            tracer,  # pyrefly: ignore[bad-argument-type]
            (src, src_index, device_id, dst, dst_index),
            {},
        ),
    )
    descriptor = AsyncCopyDescriptor()
    descriptor._proxy = proxy_out
    descriptor._descriptor_id = id(proxy_out.node)
    proxy_out.node.meta["val"] = descriptor
    proxy_out.node.meta[_REMOTE_COPY_DESCRIPTOR_ID_META] = descriptor._descriptor_id
    return descriptor


@_decorators.register_to_device_ir(make_async_remote_copy)
def _(
    tracer: object,
    src: torch.Tensor,
    src_index: list[object],
    device_id: int | torch.Tensor,
    dst: torch.Tensor,
    dst_index: list[object],
) -> AsyncCopyDescriptor:
    return _make_descriptor_proxy(
        tracer,
        make_async_remote_copy,
        src,
        src_index,
        device_id,
        dst,
        dst_index,
    )


def _register_descriptor_op(
    tracer: object,
    descriptor: AsyncCopyDescriptor,
    op: object,
) -> None:
    if not isinstance(descriptor, AsyncCopyDescriptor):
        raise exc.TypeInferenceError(
            "remote-copy lifecycle operation expects the descriptor returned by "
            "hl.make_async_remote_copy"
        )
    if (
        not isinstance(descriptor._proxy, torch.fx.Proxy)
        or descriptor._descriptor_id is None
    ):
        raise exc.TypeInferenceError(
            "remote-copy descriptor is not associated with a copy operation"
        )
    descriptor._proxy.node.meta.setdefault(_REMOTE_COPY_DESCRIPTOR_OPS_META, set()).add(
        op
    )
    tracer.create_proxy(  # type: ignore[attr-defined]
        "call_function", op, (descriptor._descriptor_id,), {}
    )


def _descriptor_op_type(descriptor: TypeInfo, origin: Origin) -> TypeInfo:
    from .._compiler.type_info import AsyncCopyDescriptorType
    from .._compiler.type_info import NoType

    if not isinstance(descriptor, AsyncCopyDescriptorType):
        raise exc.TypeInferenceError(
            "remote-copy lifecycle operation expects the descriptor returned by "
            "hl.make_async_remote_copy"
        )
    return NoType(origin=origin)


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def start_async_remote_copy_descriptor(descriptor: AsyncCopyDescriptor) -> None:
    """Issue a transfer previously created by ``make_async_remote_copy``."""
    raise exc.NotInsideKernel


@_decorators.type_propagation(start_async_remote_copy_descriptor)
def _(descriptor: TypeInfo, *, origin: Origin) -> TypeInfo:
    return _descriptor_op_type(descriptor, origin)


@_decorators.register_fake(start_async_remote_copy_descriptor)
def _(descriptor: AsyncCopyDescriptor) -> None:
    return None


@_decorators.register_to_device_ir(start_async_remote_copy_descriptor)
def _(tracer: object, descriptor: AsyncCopyDescriptor) -> None:
    _register_descriptor_op(tracer, descriptor, start_async_remote_copy_descriptor)


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def wait_async_remote_copy(descriptor: AsyncCopyDescriptor) -> None:
    """Wait for both outgoing-send and matching incoming-copy completion."""
    raise exc.NotInsideKernel


@_decorators.type_propagation(wait_async_remote_copy)
def _(descriptor: TypeInfo, *, origin: Origin) -> TypeInfo:
    return _descriptor_op_type(descriptor, origin)


@_decorators.register_fake(wait_async_remote_copy)
def _(descriptor: AsyncCopyDescriptor) -> None:
    return None


@_decorators.register_to_device_ir(wait_async_remote_copy)
def _(tracer: object, descriptor: AsyncCopyDescriptor) -> None:
    _register_descriptor_op(tracer, descriptor, wait_async_remote_copy)


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def wait_send_async_remote_copy(descriptor: AsyncCopyDescriptor) -> None:
    """Wait only until this device's outgoing source can be reused."""
    raise exc.NotInsideKernel


@_decorators.type_propagation(wait_send_async_remote_copy)
def _(descriptor: TypeInfo, *, origin: Origin) -> TypeInfo:
    return _descriptor_op_type(descriptor, origin)


@_decorators.register_fake(wait_send_async_remote_copy)
def _(descriptor: AsyncCopyDescriptor) -> None:
    return None


@_decorators.register_to_device_ir(wait_send_async_remote_copy)
def _(tracer: object, descriptor: AsyncCopyDescriptor) -> None:
    _register_descriptor_op(tracer, descriptor, wait_send_async_remote_copy)


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def wait_recv_async_remote_copy(descriptor: AsyncCopyDescriptor) -> None:
    """Wait only for the matching incoming copy into this device."""
    raise exc.NotInsideKernel


@_decorators.type_propagation(wait_recv_async_remote_copy)
def _(descriptor: TypeInfo, *, origin: Origin) -> TypeInfo:
    return _descriptor_op_type(descriptor, origin)


@_decorators.register_fake(wait_recv_async_remote_copy)
def _(descriptor: AsyncCopyDescriptor) -> None:
    return None


@_decorators.register_to_device_ir(wait_recv_async_remote_copy)
def _(tracer: object, descriptor: AsyncCopyDescriptor) -> None:
    _register_descriptor_op(tracer, descriptor, wait_recv_async_remote_copy)
