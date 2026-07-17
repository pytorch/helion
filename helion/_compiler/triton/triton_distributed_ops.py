"""Triton-backend codegen for ops defined in ``helion.language.distributed_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; it is listed in ``helion/_compiler/triton/_codegen_modules.py``
so the backend registry imports it with the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _decorators
from ...language.distributed_ops import start_async_remote_copy
from ...language.distributed_ops import wait_async_remote_copy
from ...language.distributed_ops import wait_async_remote_recv
from ...language.distributed_ops import wait_send_async_remote_copy
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


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
) -> tuple[str, dict[str, ast.AST], int]:
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
    placeholders: dict[str, ast.AST] = {}
    terms: list[str] = []
    for i, elt in enumerate(index_ast):
        name = f"{tag}{i}"
        if isinstance(elt, int):
            idx_str = repr(elt)
        else:
            assert isinstance(elt, ast.AST), elt
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
    assert isinstance(device_id_ast, ast.AST)

    # Symmetric signal buffer (GPU analog of the Pallas DMA semaphore); the
    # paired wait references it via the FX node meta.
    sig = device_fn.new_var("_nvshmem_sig", dce=False)
    assert state.fx_node is not None
    state.fx_node.meta["_nvshmem_wait_sig"] = sig
    # Also stash on the device function so a later counting receive drain
    # (``wait_async_remote_recv``) can wait on the same symmetric signal that
    # inbound pushes SIGNAL_ADD -- the GPU counterpart of the Pallas recv_sem stash.
    device_fn._distributed_recv_sig = sig  # type: ignore[attr-defined]

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


@_decorators.codegen(wait_async_remote_recv, "triton")
def _(state: CodegenState) -> object:
    """Counting receive drain -> block until the symmetric receive signal has
    accumulated ``count`` inbound pushes::

        nvshmem_signal_wait_until(sig, NVSHMEM_CMP_GE, count)

    The NVSHMEM counterpart of the Pallas recv_sem count-drain: same
    signal-based, counted-completion model as ``wait_async_remote_copy`` (which
    waits for ``1``), scaled to the total inbound row count.  Uses the signal
    stashed on the device function by ``start_async_remote_copy``'s codegen.
    """
    device_fn = state.device_function
    sig = getattr(device_fn, "_distributed_recv_sig", None)
    if sig is None:
        raise exc.InternalError(
            RuntimeError(
                "wait_async_remote_recv (triton) found no NVSHMEM receive signal "
                "on the device function — a start_async_remote_copy pushing into "
                "a local buffer must be codegen'd before the drain."
            )
        )
    count_ast = state.ast_args[1]
    if isinstance(count_ast, int):
        count_ast = expr_from_string(repr(count_ast))
    assert isinstance(count_ast, ast.AST)
    state.codegen.add_statement(
        statement_from_string(
            f"nvshmem_signal_wait_until({sig}, NVSHMEM_CMP_GE, {{count}})",
            count=count_ast,
        )
    )
    return expr_from_string("None")
