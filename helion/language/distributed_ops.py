"""Device-side communication primitives.

The API intentionally describes only a one-sided push.  That is the common
operation supported by TPU Pallas remote DMA and GPU NVSHMEM, and it is enough
to build collectives and direct-write compute/communication pipelines.

Launching a distributed Pallas kernel is a host-side concern.  The generated
local-shard function must execute under ``jax.shard_map``.  This is true for
both a single-process JAX program and a multi-process TorchTPU program (where
``torch_tpu._internal.pallas.jax_op`` exports the shard-mapped function).

The Triton host must allocate and rendezvous destination and signal tensors as
symmetric memory before launch.  Helion deliberately leaves that host-side
coordination to PyTorch: allocating it inside the device API would assume one
particular process model.  The local source tensor need not be symmetric.
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
    "start_async_remote_copy",
    "wait_async_remote_copy",
    "wait_recv_async_remote_copy",
    "wait_send_async_remote_copy",
]


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

    def wait(self) -> None:
        return wait_async_remote_copy(self)

    def wait_send(self) -> None:
        return wait_send_async_remote_copy(self)

    def wait_recv(self) -> None:
        return wait_recv_async_remote_copy(self)


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def start_async_remote_copy(
    src: torch.Tensor,
    src_index: list[object],
    device_id: int | torch.Tensor,
    dst: torch.Tensor | None = None,
    dst_index: list[object] | None = None,
    signal: torch.Tensor | None = None,
    signal_index: list[object] | None = None,
) -> AsyncCopyDescriptor:
    """Start a contiguous push from this device to ``device_id``.

    ``device_id`` is a flat logical rank in the active communication mesh.  It
    may be a compile-time integer or a runtime scalar.  By default the copy is
    symmetric: ``src[src_index]`` is pushed into the same tensor and index on
    the peer.  Supplying ``dst`` and ``dst_index`` enables direct writes into a
    different peer buffer or slot.

    Indices select leading dimensions; the remaining suffix is copied as one
    contiguous region.  Source and destination regions must have identical
    dtypes and element counts.

    Pallas allocates completion semaphores in compiler-managed scratch, so TPU
    callers may omit ``signal``.  Triton/NVSHMEM requires ``dst`` to be a
    rendezvoused symmetric allocation.  Callers that use ``wait()`` or
    ``wait_recv()`` must also provide a symmetric int64 ``signal`` tensor and a
    ``signal_index`` selecting one scalar slot per concurrent copy.  The slots
    must be zero before launch and cannot be reused until every participating
    rank has completed the communication phase.  Send-only ``wait_send()``
    copies do not need a signal.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(start_async_remote_copy)
def _(
    src: torch.Tensor,
    src_index: list[object],
    device_id: int | torch.Tensor,
    dst: torch.Tensor | None = None,
    dst_index: list[object] | None = None,
    signal: torch.Tensor | None = None,
    signal_index: list[object] | None = None,
) -> tuple[
    torch.Tensor,
    list[object],
    int | torch.Tensor,
    torch.Tensor,
    list[object],
    torch.Tensor | None,
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
    if signal is None:
        if signal_index is not None:
            raise exc.TypeInferenceError(
                "start_async_remote_copy: signal_index requires a signal tensor"
            )
        signal_index = []
    else:
        if signal_index is None:
            signal_index = []
        else:
            signal_index = Tile._prepare_index(signal_index)
            signal_index = Tile._tiles_to_sizes_for_index(signal_index)
        _validate_signal_contract(signal, signal_index)
    return src, src_index, device_id, dst, dst_index, signal, signal_index


@_decorators.type_propagation(start_async_remote_copy)
def _(*args: TypeInfo, origin: Origin, **kwargs: TypeInfo) -> TypeInfo:
    from .._compiler.type_info import AsyncCopyDescriptorType

    return AsyncCopyDescriptorType(origin=origin, element_types={})


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
            "start_async_remote_copy: an index cannot exceed its tensor rank"
        )
    if any(isinstance(index, slice) for index in (*src_index, *dst_index)):
        raise exc.TypeInferenceError(
            "start_async_remote_copy: indices must select leading dimensions; "
            "slice indexing is not supported"
        )
    if src.dtype != dst.dtype:
        raise exc.TypeInferenceError(
            "start_async_remote_copy: src and dst must share a dtype "
            f"(got src={src.dtype}, dst={dst.dtype})"
        )
    src_numel = _suffix_numel(src.shape, len(src_index))
    dst_numel = _suffix_numel(dst.shape, len(dst_index))
    if src_numel is not None and dst_numel is not None and src_numel != dst_numel:
        raise exc.TypeInferenceError(
            "start_async_remote_copy: selected source and destination regions "
            f"must have the same element count (got {src_numel} and {dst_numel})"
        )
    if not _suffix_is_contiguous(src, len(src_index)):
        raise exc.TypeInferenceError(
            "start_async_remote_copy: the selected source region must be contiguous"
        )
    if not _suffix_is_contiguous(dst, len(dst_index)):
        raise exc.TypeInferenceError(
            "start_async_remote_copy: the selected destination region must be contiguous"
        )


def _validate_signal_contract(
    signal: torch.Tensor,
    signal_index: list[object],
) -> None:
    if signal.dtype not in (torch.int64, torch.uint64):
        raise exc.TypeInferenceError(
            "start_async_remote_copy: signal must have dtype int64 or uint64 "
            f"(got {signal.dtype})"
        )
    if len(signal_index) > signal.ndim:
        raise exc.TypeInferenceError(
            "start_async_remote_copy: signal_index cannot exceed the signal rank"
        )
    if any(isinstance(index, slice) for index in signal_index):
        raise exc.TypeInferenceError(
            "start_async_remote_copy: signal_index must select leading dimensions"
        )
    signal_numel = _suffix_numel(signal.shape, len(signal_index))
    if signal_numel is not None and signal_numel != 1:
        raise exc.TypeInferenceError(
            "start_async_remote_copy: signal_index must select exactly one scalar "
            f"slot (selected {signal_numel} elements)"
        )
    if not _suffix_is_contiguous(signal, len(signal_index)):
        raise exc.TypeInferenceError(
            "start_async_remote_copy: the selected signal slot must be contiguous"
        )


@_decorators.register_fake(start_async_remote_copy)
def _(
    src: torch.Tensor,
    src_index: list[object],
    device_id: int | torch.Tensor,
    dst: torch.Tensor,
    dst_index: list[object],
    signal: torch.Tensor | None,
    signal_index: list[object],
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
    signal: torch.Tensor | None,
    signal_index: list[object],
) -> AsyncCopyDescriptor:
    proxy_out = tracer.create_proxy(  # type: ignore[attr-defined]
        "call_function",
        start_async_remote_copy,
        *args_to_proxies(
            tracer,  # pyrefly: ignore[bad-argument-type]
            (src, src_index, device_id, dst, dst_index, signal, signal_index),
            {},
        ),
    )
    descriptor = AsyncCopyDescriptor()
    descriptor._proxy = proxy_out
    proxy_out.node.meta["val"] = descriptor
    return descriptor


def _register_wait_op(
    tracer: object,
    descriptor: AsyncCopyDescriptor,
    op: object,
) -> None:
    if not isinstance(descriptor, AsyncCopyDescriptor):
        raise exc.TypeInferenceError(
            "remote-copy wait expects the descriptor returned by "
            "hl.start_async_remote_copy"
        )
    if descriptor._proxy is None:
        raise exc.TypeInferenceError(
            "remote-copy descriptor is not associated with a start operation"
        )
    tracer.create_proxy(  # type: ignore[attr-defined]
        "call_function", op, (descriptor._proxy,), {}
    )


def _wait_type(descriptor: TypeInfo, origin: Origin) -> TypeInfo:
    from .._compiler.type_info import AsyncCopyDescriptorType
    from .._compiler.type_info import NoType

    if not isinstance(descriptor, AsyncCopyDescriptorType):
        raise exc.TypeInferenceError(
            "remote-copy wait expects the descriptor returned by "
            "hl.start_async_remote_copy"
        )
    return NoType(origin=origin)


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def wait_async_remote_copy(descriptor: AsyncCopyDescriptor) -> None:
    """Wait for both outgoing-send and matching incoming-copy completion."""
    raise exc.NotInsideKernel


@_decorators.type_propagation(wait_async_remote_copy)
def _(descriptor: TypeInfo, *, origin: Origin) -> TypeInfo:
    return _wait_type(descriptor, origin)


@_decorators.register_fake(wait_async_remote_copy)
def _(descriptor: AsyncCopyDescriptor) -> None:
    return None


@_decorators.register_to_device_ir(wait_async_remote_copy)
def _(tracer: object, descriptor: AsyncCopyDescriptor) -> None:
    _register_wait_op(tracer, descriptor, wait_async_remote_copy)


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def wait_send_async_remote_copy(descriptor: AsyncCopyDescriptor) -> None:
    """Wait only until this device's outgoing source can be reused."""
    raise exc.NotInsideKernel


@_decorators.type_propagation(wait_send_async_remote_copy)
def _(descriptor: TypeInfo, *, origin: Origin) -> TypeInfo:
    return _wait_type(descriptor, origin)


@_decorators.register_fake(wait_send_async_remote_copy)
def _(descriptor: AsyncCopyDescriptor) -> None:
    return None


@_decorators.register_to_device_ir(wait_send_async_remote_copy)
def _(tracer: object, descriptor: AsyncCopyDescriptor) -> None:
    _register_wait_op(tracer, descriptor, wait_send_async_remote_copy)


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True)
def wait_recv_async_remote_copy(descriptor: AsyncCopyDescriptor) -> None:
    """Wait only for the matching incoming copy into this device."""
    raise exc.NotInsideKernel


@_decorators.type_propagation(wait_recv_async_remote_copy)
def _(descriptor: TypeInfo, *, origin: Origin) -> TypeInfo:
    return _wait_type(descriptor, origin)


@_decorators.register_fake(wait_recv_async_remote_copy)
def _(descriptor: AsyncCopyDescriptor) -> None:
    return None


@_decorators.register_to_device_ir(wait_recv_async_remote_copy)
def _(tracer: object, descriptor: AsyncCopyDescriptor) -> None:
    _register_wait_op(tracer, descriptor, wait_recv_async_remote_copy)
