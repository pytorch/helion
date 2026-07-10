"""Distributed / RDMA primitives for Helion kernels.

v1 exposes an explicitly-async push-based RDMA API::

    ring_step = hl.start_async_remote_copy(tensor, index, device_id)
    # ... compute overlapping the copy ...
    ring_step.wait()

which pushes ``tensor[index]`` (local) into ``tensor[index]`` on the peer
identified by ``device_id`` — the shape of copy used by ring all-gather
kernels.  ``device_id`` is a flat integer PE id (LOGICAL addressing),
which matches how NVSHMEM peers are addressed; portable across TPU and
GPU backends.  See ``examples/distributed/all_gather.py``.

Only the Pallas backend is wired up in v1.  ``start_async_remote_copy``
lowers to::

    _op = pltpu.make_async_remote_copy(
        tensor.at[index],
        tensor.at[index],
        send_sem,
        recv_sem,
        device_id=device_id,
        device_id_type=pl.DeviceIdType.LOGICAL,
    )
    _op.start()

and the paired ``ring_step.wait()`` lowers to ``_op.wait()`` where
``_op`` is the same variable name emitted by the ``start`` op.

Send/recv DMA semaphores are allocated by the compiler as Pallas
scratch buffers via :meth:`DeviceFunction.register_dma_semaphore`.
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
    from .._compiler.type_info import TypeInfo
    from .._compiler.variable_origin import Origin


__all__ = [
    "AsyncCopyDescriptor",
    "start_async_remote_copy",
    "wait_async_remote_copy",
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


@has_side_effect
@_decorators.api(is_device_only=True, allow_host_tensor=True, tiles_as_sizes=True)
def start_async_remote_copy(
    tensor: torch.Tensor,
    index: list[object],
    device_id: int,
) -> AsyncCopyDescriptor:
    """Start an async push of ``tensor[index]`` to peer ``device_id``.

    Both the source (local) and destination (peer) refs address the
    same slot of the same symmetric tensor.

    Args:
        tensor: The (host-side) symmetric tensor whose slot is being
            copied.  Every rank must pass a tensor of the same shape
            and dtype at the same argument position; the caller is
            responsible for that invariant.
        index: List selecting the slot to copy — the same subscript is
            used on both the local and peer sides.
        device_id: Flat integer PE id (0 <= device_id < world_size).
            Usually derived from ``dist.get_rank()`` +/- 1 on the host.

    Returns:
        An :class:`AsyncCopyDescriptor` whose ``wait()`` method must be
        called before the copy's effects are observed.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(start_async_remote_copy)
def _(
    tensor: torch.Tensor,
    index: list[object],
    device_id: int,
) -> tuple[torch.Tensor, list[object], int]:
    from .tile_proxy import Tile

    index = Tile._prepare_index(index)
    index = Tile._tiles_to_sizes_for_index(index)
    return (tensor, index, device_id)


@_decorators.type_propagation(start_async_remote_copy)
def _(
    tensor: TypeInfo,
    index: TypeInfo,
    device_id: TypeInfo,
    *,
    origin: Origin,
) -> TypeInfo:
    from .._compiler.type_info import AsyncCopyDescriptorType

    return AsyncCopyDescriptorType(origin=origin, element_types={})


@_decorators.register_fake(start_async_remote_copy)
def _(
    tensor: torch.Tensor,
    index: list[object],
    device_id: int,
) -> AsyncCopyDescriptor:
    return AsyncCopyDescriptor()


@_decorators.register_to_device_ir(start_async_remote_copy)
def _(
    tracer: object,
    tensor: torch.Tensor,
    index: list[object],
    device_id: int,
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
        *args_to_proxies(tracer, (tensor, index, device_id), {}),  # type: ignore[arg-type]
    )
    descriptor = AsyncCopyDescriptor()
    descriptor._proxy = proxy_out
    # Helion's codegen expects every FX node to carry a fake ``val``
    # entry in its meta dict; the default proxy path does this via
    # ``proxy_tensor.track_tensor_tree`` but we bypassed that above.
    proxy_out.node.meta["val"] = descriptor
    return descriptor


@_decorators.codegen(start_async_remote_copy, "pallas")
def _(state: CodegenState) -> object:
    """Emit ``_op = pltpu.make_async_remote_copy(...); _op.start()``.

    Stashes the ``_op`` variable name on the FX node's meta dict so
    the paired ``wait_async_remote_copy`` codegen can look it up.
    """
    tensor = state.proxy_arg(0)
    assert isinstance(tensor, torch.Tensor)

    device_fn = state.device_function
    ref_name = device_fn.tensor_arg(tensor).name

    send_sem = device_fn.register_dma_semaphore(name_hint="send_sem")
    recv_sem = device_fn.register_dma_semaphore(name_hint="recv_sem")
    op_var = device_fn.new_var("_remote_copy", dce=False)

    # Stash the op variable name on the FX node so wait can find it.
    assert state.fx_node is not None
    state.fx_node.meta["_pallas_async_copy_op"] = op_var

    # Build the .at[i0, i1, ...] expression using placeholders so both
    # int literals and AST nodes flow through cleanly.
    index_ast = state.ast_args[1]
    assert isinstance(index_ast, (list, tuple))
    index_placeholders: dict[str, object] = {}
    index_parts: list[str] = []
    for i, elt in enumerate(index_ast):
        name = f"_idx{i}"
        if isinstance(elt, int):
            index_placeholders[name] = expr_from_string(repr(elt))
        else:
            index_placeholders[name] = elt
        index_parts.append(f"{{{name}}}")
    at_expr = ", ".join(index_parts)

    device_id_ast = state.ast_args[2]
    if isinstance(device_id_ast, int):
        device_id_ast = expr_from_string(repr(device_id_ast))

    state.codegen.add_statement(
        statement_from_string(
            f"{op_var} = pltpu.make_async_remote_copy("
            f"{ref_name}.at[{at_expr}], {ref_name}.at[{at_expr}], "
            f"{send_sem}, {recv_sem}, "
            f"device_id={{device_id}}, "
            f"device_id_type=pl.DeviceIdType.LOGICAL)",
            device_id=device_id_ast,
            **index_placeholders,  # type: ignore[arg-type]
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
