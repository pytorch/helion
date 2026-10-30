"""Spyre codegen for ``helion.language.memory_ops``.

Plan 2: memory nodes produce **no OpSpec**. SuperDSC has no load/store op -- an
``OpSpec`` reads and writes buffers directly through
``TensorArg.device_coordinates`` -- so a ``load`` only *fixes an address
expression*, and a ``store`` only tells the producing OpSpec where to write.

Both therefore return an ``Operand`` (a buffer handle), which is legal because
the per-node lowering seam passes non-``ast.AST`` results straight through
(inductor_lowering.py:1657-1662).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _decorators
from ...language.memory_ops import load
from ...language.memory_ops import store

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState
    from .backend import SpyreBackend
    from .emitter import Operand


def host_operand(state: CodegenState, tensor: torch.Tensor) -> Operand:
    """The buffer for a kernel-argument tensor, registering it on first use.

    Argument order follows Helion's own ``tensor_arg`` table, so the emitted
    ``arg_index`` matches the launcher's positional order.
    """
    from ..compile_environment import CompileEnvironment

    backend = cast("SpyreBackend", CompileEnvironment.current().backend)
    name = state.device_function.tensor_arg(tensor).name
    return backend.emitter.add_arg(name, tensor)


@_decorators.codegen(load, "spyre")
def _(state: CodegenState) -> object:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))

    indirect = _indirect_dim(subscript)
    if indirect is not None:
        return _indirect_operand(state, tensor, subscript, indirect)

    _reject_narrowing(subscript)
    return host_operand(state, tensor)


def _indirect_dim(subscript: object) -> int | None:
    """Host dim indexed by a *tensor*, i.e. a gather."""
    assert isinstance(subscript, (list, tuple))
    for dim, index in enumerate(subscript):
        if isinstance(index, torch.Tensor):
            return dim
    return None


def _indirect_operand(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: object,
    indirect_dim: int,
) -> Operand:
    """A gathered view of ``tensor``, indexed at runtime by another buffer.

    Spyre expresses this as ``IndirectAccess(name)`` in the value tensor's
    ``device_coordinates``, paired by name with an index ``TensorArg`` carrying
    that ``name`` (op_spec.py's ``IndirectAccess`` docstring;
    codegen/superdsc.py:1277-1279 does the pairing). The index operand must
    therefore travel with the value operand, which is what ``IndirectOperand``
    records for the consuming OpSpec to emit.
    """
    from .emitter import IndirectOperand
    from .emitter import Operand as _Operand

    # The index is a *loaded value*, not a kernel argument: Device IR shows
    # load(table, [load(idx, ...), slice]) -- so its buffer comes from the
    # already-codegen'd subscript argument, which the "spyre" load handler
    # returned as an Operand.
    index_value = state.ast_args[1][indirect_dim]
    if not isinstance(index_value, _Operand):
        raise exc.BackendUnsupported(
            "spyre",
            f"indirect index is {type(index_value).__name__}, not a buffer",
        )
    value = host_operand(state, tensor)
    return IndirectOperand(value=value, index=index_value, indirect_dim=indirect_dim)


@_decorators.codegen(store, "spyre")
def _(state: CodegenState) -> None:
    """Redirect the producing OpSpec's output to this tensor's buffer.

    Plan 2: a store "makes ``v``'s producing OpSpec write ``out`` directly".
    The value node already emitted its OpSpec, so rather than emit a copy we
    rewrite that OpSpec's output ``TensorArg`` in place to address ``out``.
    """
    from ..compile_environment import CompileEnvironment

    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(subscript, (list, tuple))
    _reject_narrowing(subscript)

    backend = cast("SpyreBackend", CompileEnvironment.current().backend)
    emitter = backend.emitter
    dest = host_operand(state, tensor)

    value = state.ast_args[2] if len(state.ast_args) > 2 else None
    from .emitter import IndirectOperand
    from .emitter import Operand as _Operand

    if isinstance(value, IndirectOperand):
        # A gather feeding a store is a copy through the indirect index: the
        # reference emits exactly this as an `identity` whose value arg carries
        # IndirectAccess (golden/ref_gather.py, op 2).
        emitter.emit_gather(value, dest)
        _mark_device_body(state)
        return
    if not isinstance(value, _Operand):
        raise exc.BackendUnsupported(
            "spyre", f"store of a non-buffer value: {type(value).__name__}"
        )

    # Rank first: a rank mismatch makes the stick dims incomparable, and the
    # relayout path below would fail with an opaque IndexError out of
    # ``stickify`` instead of naming the cause. A Spyre reduction result is
    # keepdim on the device (one live element per stick), so a rank-dropped
    # destination is a genuinely different layout -- see
    # ``repros/reduction_rank_dropped.py``, where the reference's own specs
    # give 139.43 max error that way.
    if len(value.sizes) != len(dest.sizes):
        raise exc.BackendUnsupported(
            "spyre",
            f"storing {value.name!r} (rank {len(value.sizes)}) into"
            f" {dest.name!r} (rank {len(dest.sizes)}): a Spyre reduction output"
            " keeps the reduced dim, so use `keepdim=True` and an output tensor"
            f" that keeps it ({value.sizes} rather than {dest.sizes})",
        )

    if value.stick_dim != dest.stick_dim:
        # Closing relayout, the mirror of M3's opening one. The compute follows
        # the *inputs'* stick axis, but the destination buffer need not share it
        # -- notably ``torch.empty_like`` of a transposed Spyre tensor returns a
        # default trailing-dim buffer, so a fully transposed kernel computes on
        # dim 0 and must store to a dim-1 buffer. Writing the computed layout
        # into that buffer is silently wrong (~99% error, nothing raises).
        import sympy

        symbols = [sympy.Symbol(f"c{i}") for i in range(len(dest.sizes))]
        # ``ReStickifyOpHBM`` reads and writes HBM, so the source must already
        # be HBM-resident. It is: when the destination-binding pass saw this
        # store it gave the value ``hbm_pool`` residency *at allocation time*
        # for exactly this reason. ``restage_to_pool`` used to arrange that here
        # by rewriting the producing spec, and it is gone -- the decision is now
        # made before the value is ever emitted.
        if "lx" in value.allocation:
            raise exc.BackendUnsupported(
                "spyre",
                f"relayout of {value.name!r} from an lx buffer: a closing"
                " ReStickifyOpHBM needs an HBM source, which the"
                " destination-binding pass assigns when it binds the store."
                " Reaching here means the pass declined the value but the"
                " store still needs a relayout",
            )
        emitter.emit_restickify_into(value, dest, symbols)
        _mark_device_body(state)
        return

    if value is dest:
        # **The destination-binding pass bound this.** The producing OpSpec
        # wrote ``dest`` directly, so there is nothing to redirect: the store is
        # a no-op beyond keeping the device body non-empty. This is the only
        # correct path now, and it is the point of the whole exercise --
        # emission is a pure function of the annotated graph.
        _mark_device_body(state)
        return

    # No fallback remains. ``redirect_output`` used to patch the already-emitted
    # spec tree from here, and it is gone (see ``spyre/graph_passes.py``), so a
    # value reaching this point was *declined* by the pass and there is nothing
    # left to write its destination.
    #
    # **Raising is required, not merely tidier.** While the rewrite existed a
    # silent decline was safe, because the fallback picked the work up; with it
    # deleted the same silent decline would emit a kernel whose final value is
    # never written to the output buffer -- the consumer reads whatever the
    # previous launch left in that buffer. That is not hypothetical: it is the
    # defect ``graph_passes_example.py::second_user`` gates, where the declined
    # multi-user value made ``o2`` come back 100% wrong alone and 0.188%
    # "correct" after another kernel happened to leave the right bytes behind.
    # A loud refusal is the only answer that cannot be mistaken for a result.
    raise exc.BackendUnsupported(
        "spyre",
        f"storing {value.name!r} into {dest.name!r}: the destination-binding"
        " pass declined this value, so no OpSpec writes the destination."
        " The usual cause is a stored value with another user, which needs its"
        " own buffer plus an explicit copy; binding it to the destination would"
        " make the other user read the output buffer",
    )
    # Unconditional: a store whose subscript needs no index arithmetic leaves
    # the device body empty and DCE then rejects the loop. A reduction is the
    # case that exposed it (both the rank-dropped and the keepdim spelling),
    # but the marker is inert whenever the body is already non-empty.
    _mark_device_body(state)


def _reject_narrowing(subscript: object) -> None:
    """Fail loudly on indexing this backend does not model yet.

    A Spyre tile is an address range, so a narrowing subscript would change
    ``device_coordinates`` rather than select registers. Silently ignoring it
    is wrong numerics, so plan 2's "never a silent fallback" applies.
    """
    assert isinstance(subscript, (list, tuple))
    for index in subscript:
        if isinstance(index, slice):
            if index != slice(None):
                raise exc.BackendUnsupported("spyre", f"narrowing slice {index!r}")
        elif not isinstance(index, (torch.SymInt, int)):
            raise exc.BackendUnsupported(
                "spyre", f"subscript of type {type(index).__name__}"
            )


def _mark_device_body(state: CodegenState) -> None:
    """Keep a vestigial device body non-empty so DCE does not reject the loop.

    ``generate_ast.py:1936-1937`` raises ``EmptyDeviceLoopAfterDCE`` when both
    the preamble and body are empty after elimination. On this backend the body
    is never executed -- the OpSpec tree is the kernel -- but a store whose
    subscript needs no index arithmetic (a gather, where the index is a buffer)
    leaves nothing behind, so the loop looks dead. The statement documents
    itself in the generated source.
    """
    from ..ast_extension import statement_from_string

    state.add_statement(
        statement_from_string("pass  # Spyre: the OpSpec tree is the kernel")
    )
