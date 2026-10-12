"""Spyre codegen for ops defined in ``helion.language._tracing_ops``.

Plan 3: loop *nesting* already comes for free, because ``_for_loop``'s
``"common"`` handler just recurses into the subgraph
(language/_tracing_ops.py:131-134). This module overrides it for ``"spyre"``
only to wrap that recursion in a ``LoopSpec``, which is the Pallas precedent
(pallas/tracing_ops.py:234).

The carry ops (``_new_var``, ``_phi``) are overridden because their ``"common"``
handlers assert an ``ast.Name`` result. On this backend a loop carry is a
*buffer* (plan 6: ``LoopSpec`` has no ``iter_args``), so both are identities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ... import exc
from ...language import _decorators
from ...language._tracing_ops import _for_loop
from ...language._tracing_ops import _new_var
from ...language._tracing_ops import _phi

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState
    from .backend import SpyreBackend
    from .emitter import SpyreEmitter


@_decorators.codegen(_new_var, "spyre")
def _(state: CodegenState) -> object:
    """Entering a loop with a carried buffer is a no-op.

    The ``"common"`` handler copies the value into a fresh device variable.
    Here the carry *is* the accumulator buffer and body OpSpecs read and write
    that same buffer, so the incoming placeholder passes straight through.
    """
    return state.ast_args[0]


@_decorators.codegen(_phi, "spyre")
def _(state: CodegenState) -> object:
    """Joining a carry is a no-op: both sides denote the same buffer.

    The ``"common"`` handler merges two device *variable names*. With the carry
    in a buffer the loop body already accumulated in place, so the phi has no
    content.
    """
    return state.ast_args[1]


@_decorators.codegen(_for_loop, "spyre")
def _(state: CodegenState) -> object:
    """A sequential device loop becomes a ``LoopSpec`` around its body.

    Plan 4.1: an inner ``for tile_k in hl.tile(k)`` is *sequential*, so it maps
    to coarse tiling (``LoopSpec.count``), never to work division. Conflating
    the two silently yields a kernel that is correct but single-core, or one
    that races -- so the trip count is derived here and asserted by the golden.

    Any accumulator allocated by ``hl.zeros`` is zero-filled *before* the loop
    (plan 2: SuperDSC has no ``iter_args``), matching the reference's leading
    ``coarse_tile_fill_buf0`` identity.

    Returns the body's carried values, as the ``"common"`` handler does, so the
    ``getitem``/``_phi`` pair reading the loop result still resolves.
    """
    import sympy

    from ..compile_environment import CompileEnvironment

    backend = cast("SpyreBackend", CompileEnvironment.current().backend)
    emitter = backend.emitter

    graph_info = state.get_graph(state.proxy_arg(0))
    count, block_ids, block_size = _loop_tiling(state, graph_info)

    _emit_fills(emitter)

    # One minted symbol per level, shared by tiled_symbols and every
    # device_tile_advance_expr at that level (spyre_kernel.py:552-568, :743):
    # the shared identity is what lets SDSC recover each level's stride.
    level_symbol = sympy.Symbol(f"_tile_adv_helion_lvl{emitter.loop_depth}")

    emitter.open_loop(level_symbol, count, block_ids, block_size)
    ok = False
    try:
        results = graph_info.codegen(state)
        ok = True
    finally:
        # Only wrap the body when the walk succeeded: closing an empty body
        # raises BackendUnsupported, which would mask the real error.
        if ok:
            emitter.close_loop(count)
        else:
            emitter.abandon_loop()
    return results


def _loop_tiling(
    state: CodegenState, graph_info: object
) -> tuple[int, list[int], int]:
    """``(trip count, tiled block ids, per-iteration extent)`` of a device loop."""
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    block_ids = getattr(graph_info, "block_ids", None)
    if not block_ids:
        raise exc.BackendUnsupported(
            "spyre", "device loop with no block id to derive a trip count from"
        )
    if len(block_ids) != 1:
        raise exc.BackendUnsupported(
            "spyre",
            f"device loop over {len(block_ids)} dims; only 1-D is modelled",
        )
    info = env.block_sizes[block_ids[0]]
    extent = int(info.size_hint())
    value = info.from_config(state.device_function.config)
    if value is None:
        raise exc.BackendUnsupported("spyre", "device loop with no configured block")
    block_size = int(value)
    if block_size <= 0 or extent % block_size != 0:
        raise exc.BackendUnsupported(
            "spyre",
            f"loop block size {block_size} does not divide extent {extent};"
            " a Spyre coarse tile count must be exact"
            " (wsr/coarse_tile.py:1384-1395)",
        )
    return extent // block_size, list(block_ids), block_size


def _emit_fills(emitter: SpyreEmitter) -> None:
    """Zero-fill every pending accumulator, one OpSpec each.

    ``identity`` from a scalar constant is how the reference fills its
    accumulator (golden/ref_matmul_ktiled.py op 1/4: a ``[1, 1, 64]`` source at
    constant coordinates writing the accumulator's full iteration space).
    """
    for operand, value in emitter.take_pending_fills():
        if value != 0:
            raise exc.BackendUnsupported(
                "spyre", f"accumulator fill value {value!r}; only 0 is modelled"
            )
        emitter.emit_zero_fill(operand)
