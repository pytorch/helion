"""Spyre codegen for compute nodes: one aten op over one tile -> one ``OpSpec``.

Plan 0's thesis: an ``OpSpec`` is one Spyre op over one iteration space and a
Device IR compute node is one aten op over one tile, so ``fx node -> OpSpec`` is
a genuine leaf mapping.

The iteration space is where Helion's authority is exercised. Plan 4.1: a GRID
``hl.tile([m, n])`` is *parallel*, so each dim contributes
``iteration_space[sym] = (extent, split)`` where the split is the core count --
and ``split = extent // block_size`` makes Helion's ``Config.block_sizes`` the
work division directly. A 128x256 kernel with block_sizes [16, 64] reproduces the
captured reference's ``{c0: (128, 8), c1: (256, 4)}``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from ... import exc
from ..aten_lowering import Lowering
from .optable import is_accumulating_matmul
from .optable import is_matmul
from .optable import is_reduction
from .optable import spyre_op_name

if TYPE_CHECKING:
    import sympy
    from torch.fx.node import Node

    from ..aten_lowering import LoweringContext
    from .backend import SpyreBackend
    from .emitter import Operand
    from .emitter import SpyreEmitter


def _spyre_backend() -> SpyreBackend:
    """The active backend, narrowed to ``SpyreBackend``.

    ``CompileEnvironment.backend`` is typed as the base ``Backend``, so every
    Spyre-specific attribute (``emitter``, ``kernel_stick_dim``,
    ``input_stick_dim``) is invisible to a type checker through it -- pyrefly
    reports ``Object of class 'Backend' has no attribute 'emitter'`` at each
    site. Four other modules in this package already worked around that with
    an inline ``cast("SpyreBackend", ...)``; this is the same narrowing in one
    place, so the cast is justified once rather than repeated.

    The cast is sound by construction: these lowerings are only installed by
    ``SpyreBackend.pre_inductor_lowering``, so reaching them means the Spyre
    backend is the active one.
    """
    from typing import cast

    from ..compile_environment import CompileEnvironment

    return cast("SpyreBackend", CompileEnvironment.current().backend)


def _iteration_symbols(n: int) -> list[sympy.Symbol]:
    """``c0, c1, ...`` -- torch-spyre's own iteration-symbol naming."""
    import sympy

    return [sympy.Symbol(f"c{i}") for i in range(n)]


def _split_from_block(extent: int, block_size: int) -> int:
    if block_size <= 0 or extent % block_size != 0:
        raise exc.BackendUnsupported(
            "spyre",
            f"block size {block_size} does not divide extent {extent};"
            " Spyre work division must be exact",
        )
    return extent // block_size


def tile_space(val: torch.Tensor) -> tuple[list[int], list[int]]:
    """``(extents, splits)`` for a tile-shaped value.

    A compute node's ``meta["val"]`` is *tile*-shaped: its sizes are the block
    symbols (u0, u1), not the tensor extents. Both the full extent and the core
    count live on the ``BlockSizeInfo`` behind each symbol, so resolve the block
    id per dim rather than reading the fake tensor's own sizes.

    A dim tiled by an enclosing *sequential* loop reports its per-iteration
    extent instead of its full one: inside the reference's K-tiled LoopSpec
    ``c2 = 128`` with K=512, and the rest of K is reached through
    ``device_tile_advance_expr``.
    """
    from ..compile_environment import CompileEnvironment
    from ..device_function import DeviceFunction

    env = CompileEnvironment.current()
    device_function = DeviceFunction.current()
    emitter = _spyre_backend().emitter

    extents: list[int] = []
    splits: list[int] = []
    for size in val.size():
        block_id = env.resolve_block_id(size)
        if block_id is None:
            # An untiled dim: its extent is concrete and it is not split.
            extents.append(int(env.size_hint(size)))
            splits.append(1)
            continue

        tiled = emitter.tiled_block_size(block_id)
        if tiled is not None:
            # Sequentially tiled: one coarse-tile step per iteration, and a
            # coarse tile is never also split across cores.
            extents.append(tiled)
            splits.append(1)
            continue

        info = env.block_sizes[block_id]
        extent = int(info.size_hint())
        value = info.from_config(device_function.config)
        extents.append(extent)
        splits.append(1 if value is None else _split_from_block(extent, int(value)))
    return extents, splits


@dataclasses.dataclass
class SpyreOpLowering(Lowering):
    """Per-node lowering emitting one ``OpSpec`` and returning its buffer."""

    backend: SpyreBackend
    target: object

    def codegen(self, ctx: LoweringContext, node: Node) -> object:
        op = spyre_op_name(self.target)
        if op is None:
            raise exc.BackendUnsupported("spyre", f"aten op {self.target}")
        if is_accumulating_matmul(self.target):
            return _emit_addmm(self.backend, ctx, node)
        if is_matmul(self.target):
            return _emit_matmul(self.backend, ctx, node, op)
        if is_reduction(self.target):
            return _emit_reduction(self.backend, ctx, node, op)
        return _emit_pointwise(self.backend, ctx, node, op)


def _operand_inputs(ctx: LoweringContext, node: Node) -> list[Operand]:
    """The buffers feeding this node, from the interpreter env.

    Non-buffer inputs (python scalars) are rejected rather than folded: Spyre
    takes a constant through ``op_info``, which this milestone does not model.
    """
    return [_as_operand(ctx, arg, node) for arg in node.all_input_nodes]


def _emit_pointwise(
    backend: SpyreBackend, ctx: LoweringContext, node: Node, op: str
) -> Operand:
    """One pointwise ``OpSpec`` over the GRID iteration space."""
    emitter = backend.emitter
    inputs = _operand_inputs(ctx, node)
    if not inputs:
        raise exc.BackendUnsupported("spyre", f"{node.target} with no buffer inputs")

    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    sizes, splits = tile_space(val)
    symbols = _iteration_symbols(len(sizes))

    # Every operand of a pointwise op must stick on the same axis, and the
    # *consumer's* axis is the one that wins: the output is written in the
    # layout its own users expect, so a disagreeing input is relaid onto that
    # axis by an explicit ReStickifyOpHBM (plan 4.2/M3), never by
    # reinterpreting the buffer. Picking a majority axis instead would relayout
    # the correct operands and leave the output inconsistent -- which
    # validates, then fails in the backend scheduler with
    # "There must be at least one valid candidate".
    #
    # The target axis is the one the *kernel's own output* will be stored on, so
    # the final store needs no relayout. Assuming the trailing dim instead is
    # wrong whenever the caller's output tensor is laid out otherwise: with all
    # operands transposed (dim_order=[1,0]) the chain still computed on dim 1
    # and wrote a dim-1-sticked result into a dim-0-sticked buffer, returning
    # ~99% wrong values with no error anywhere.
    stick_dim = _output_stick_dim(len(sizes))
    if any(o.stick_dim != stick_dim for o in inputs):
        inputs = [
            o
            if o.stick_dim == stick_dim
            else emitter.emit_restickify(o, stick_dim, symbols)
            for o in inputs
        ]

    # Accumulating into a loop carry writes the carry itself, in place. A fresh
    # output buffer would leave the carry never updated, so each iteration would
    # recompute ``initial_carry + this_tile`` and only the last one would
    # survive. ``_emit_addmm`` does this by hand for M2b's combine (arg0 and
    # arg2 both the accumulator); a rolled reduction reaches the same combine
    # through an ordinary pointwise ``add``, so the rule belongs here.
    carry = _carried_input(ctx, node)
    if carry is not None and carry.sizes == sizes:
        out = carry
    else:
        out = _bound_or_intermediate(emitter, node, sizes, val.dtype, stick_dim)

    iteration_space = _iteration_space(symbols, sizes, splits, stick_dim, val.dtype)
    args = [o.tensor_arg(symbols, is_input=True) for o in inputs]
    args.append(out.tensor_arg(symbols, is_input=False))

    emitter.emit(
        emitter.op_spec(
            op,
            is_reduction=False,
            iteration_space=iteration_space,
            args=args,
        )
    )
    return out


def _bound_or_intermediate(
    emitter: SpyreEmitter,
    node: Node,
    sizes: list[int],
    dtype: torch.dtype,
    stick_dim: int,
) -> Operand:
    """This node's output buffer: the bound destination, or a fresh intermediate.

    The whole point of the destination-binding graph pass. When the pass has
    annotated this node, the output operand **is** the kernel's output buffer,
    so the op writes its destination directly and the store has nothing left to
    patch -- emission becomes append-only for this value.

    Two annotated cases:

    * **stick axes agree** -- bind straight to the destination.
    * **stick axes disagree** -- the destination cannot be written directly,
      because a ``ReStickifyOpHBM`` must read HBM. The *decision* is still made
      early: give the value ``hbm_pool`` residency now, so the closing relayout
      reads it where it lies. That is what retires ``restage_to_pool``, which
      exists only because this was discovered too late, with the value already
      in ``lx`` and its producing spec needing a rewrite.

    Unannotated, or a shape the destination does not match, declines to the
    existing ``lx`` intermediate and the store's rewrite path.
    """
    from .graph_passes import BINDING
    from .graph_passes import bound_destination

    bound = bound_destination(node, sizes, stick_dim, dtype)
    if bound is None:
        BINDING["fallback"] += 1
        return emitter.add_intermediate(
            node.name, sizes, dtype, stick_dim, residency="lx"
        )
    if not bound.relayout:
        BINDING["bound"] += 1
        emitter.bind(node.name, bound.operand)
        return bound.operand

    BINDING["relayout"] += 1
    return emitter.add_intermediate(
        node.name, sizes, dtype, stick_dim, residency="hbm_pool"
    )


def _carried_input(ctx: LoweringContext, node: Node) -> Operand | None:
    """The loop-carry buffer among this node's inputs, if one of them is it.

    A carry crosses the loop boundary as an ``ast.Name`` (tracing_ops._new_var),
    so it is recognised by that placeholder rather than by buffer identity.
    """
    import ast as _ast

    emitter = _spyre_backend().emitter
    for arg in node.all_input_nodes:
        value = ctx.env.get(arg)
        if isinstance(value, _ast.Name):
            carried = emitter.carry(value.id)
            if carried is not None:
                return carried
    return None


def _reduced_dims(node: Node, rank: int) -> list[int]:
    """Host dims this reduction collapses, normalized and sorted.

    ``sum.default`` (no dim argument) reduces everything; ``sum.dim_IntList``
    and ``amax``/``amin`` carry an explicit dim list in ``args[1]``.
    """
    if len(node.args) < 2 or node.args[1] is None:
        return list(range(rank))
    dims = node.args[1]
    if isinstance(dims, int):
        dims = [dims]
    if not isinstance(dims, (list, tuple)):
        raise exc.BackendUnsupported("spyre", f"reduction dims {dims!r}")
    return sorted((int(d) % rank) for d in dims)


def _emit_reduction(
    backend: SpyreBackend, ctx: LoweringContext, node: Node, op: str
) -> Operand:
    """One reduction ``OpSpec`` over the stick axis, unrolled (no ``LoopSpec``).

    The reduced symbol **stays** in the iteration space at its full extent --
    the reduction is internal to the op, as it is for ``batchmatmul``'s K -- and
    disappears only from the *output's* coordinates. Both captured references
    (golden/ref_row_sum.py, golden/ref_row_amax.py) show exactly that: input
    ``[4, 64, 64]`` at ``[floor(c1/64), c0, Mod(c1, 64)]``, output
    ``[1, 64, 64]`` at ``[0, c0, 0]``.

    Helion's node shape follows ``keepdim``, but Spyre's output is *always*
    keepdim on the device: the reduced host dim survives as an extent-1 axis
    indexed by the constant 0, holding one live element per stick. Registering
    the output with keepdim sizes makes ``stickify`` produce the reference's
    layout with no reduction-specific layout code -- and a dense rank-dropped
    destination is a genuinely different layout, which ``redirect_output``
    rejects rather than silently mis-addresses.

    Only a stick-axis reduction is emitted. ``_check_stick_all_same``
    (op_spec_validation.py:675) requires every arg to share one stick symbol,
    and the output's stick coordinate is the constant 0 whose stick symbol is
    the reduced one -- so reducing a *non*-stick dim would leave the output
    sticked on a different symbol than the input and is rejected. torch-spyre's
    own rule is the mirror image for the normalization ops
    (``_check_stick_norm``, :727: the stick symbol *must* be the reduced one).
    """
    emitter = backend.emitter
    inputs = _operand_inputs(ctx, node)
    if len(inputs) != 1:
        raise exc.BackendUnsupported(
            "spyre", f"{op} with {len(inputs)} operands, expected 1"
        )
    source = inputs[0]

    in_node = node.all_input_nodes[0]
    in_val = in_node.meta["val"]
    assert isinstance(in_val, torch.Tensor)
    sizes, splits = tile_space(in_val)
    rank = len(sizes)
    symbols = _iteration_symbols(rank)

    reduced = _reduced_dims(node, rank)
    if len(reduced) != 1:
        raise exc.BackendUnsupported(
            "spyre", f"{op} over {len(reduced)} dims at once (dims {reduced})"
        )
    axis = reduced[0]

    # The input's own stick axis is the one the reduction must run along, and
    # the operand may disagree with it (a transposed input). Relayout rather
    # than reduce along the wrong axis: that is bug #6's failure mode.
    if source.stick_dim != axis:
        source = emitter.emit_restickify(source, axis, symbols)

    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    out_sizes = [1 if d == axis else extent for d, extent in enumerate(sizes)]
    out = _bound_or_intermediate(emitter, node, out_sizes, val.dtype, axis)

    # The reduced dim must not be split across cores: each core would produce a
    # partial with no combine step to merge them. The references keep it at
    # split 1 even with 32 cores available on the other axis.
    splits = list(splits)
    splits[axis] = 1
    iteration_space = _iteration_space(symbols, sizes, splits, axis, val.dtype)

    # Rolled into a sequential loop over the reduced dim, ``tile_space`` reports
    # one coarse tile, so the streamed input must advance by a tile per
    # iteration or every iteration re-reads the first one. That was measured:
    # without the advance a 4-iteration rolled sum returned 4x the first tile
    # (predicted max|4*t0 - full| = 31.87, observed error 31.25).
    #
    # The advance goes on a *read-copy* into a tile-sized staging buffer, not on
    # the reduction's own operand: the operand spans the whole reduced extent,
    # so an advance on it addresses a buffer whose shape disagrees with the
    # iteration space. golden/ref_matmul_ktiled.py shows the same split for the
    # K-tiled matmul (full-extent advancing read -> one-tile pool window).
    block_id = _block_id_of(in_val, axis)
    if block_id is not None and emitter.tiled_block_size(block_id) is not None:
        source = _stage_reduced_tile(
            emitter, node, source, sizes, symbols, axis, block_id
        )

    op_key = f"reduce_{node.name}"
    tiling = emitter.tiling_for(op_key)
    tiled_symbols, trip_counts = tiling if tiling is not None else ([], {})

    # The output's reduced axis is extent 1, so ``tensor_arg`` addresses it at
    # coordinate 0 on its own -- giving the reference's [0, c0, 0].
    args = [
        source.tensor_arg(symbols, is_input=True),
        out.tensor_arg(symbols, is_input=False),
    ]
    emitter.emit(
        emitter.op_spec(
            op,
            is_reduction=True,
            iteration_space=iteration_space,
            args=args,
            tiled_symbols=tiled_symbols,
            tiled_symbol_trip_counts=trip_counts,
        )
    )
    return out


def _stage_reduced_tile(
    emitter: SpyreEmitter,
    node: Node,
    operand: Operand,
    sizes: list[int],
    symbols: list[sympy.Expr],
    axis: int,
    block_id: int,
) -> Operand:
    """Read-copy one coarse tile of the reduced dim into a staging buffer.

    The reduction mirror of ``_stage_advancing_operand``. The source arg spans
    the **full** reduced extent and carries the per-iteration advance; the
    destination is tile-shaped, so the reduction op reads a buffer matching its
    own iteration space.
    """
    import dataclasses as _dc

    from torch_spyre._inductor.op_spec import TensorArg

    from ..compile_environment import CompileEnvironment
    from . import layout as _layout

    tile = sizes[axis]
    full_extent = int(CompileEnvironment.current().block_sizes[block_id].size_hint())
    if full_extent == tile:
        return operand

    op_key = f"read_copy_{node.name}"
    level = emitter.level_symbol_for(op_key)
    full_sizes = [full_extent if d == axis else e for d, e in enumerate(sizes)]
    full = _dc.replace(operand, sizes=full_sizes)
    device_size, coords = layout_stickify(full, symbols)
    advance = emitter.tile_advance_for(
        device_size, coords, symbols[axis], tile, level_symbol=level
    )
    if advance is None:
        return operand

    staged = emitter.add_intermediate(
        f"{node.name}_rtile",
        list(sizes),
        operand.dtype,
        axis,
        residency="hbm_pool",
    )
    src_arg = TensorArg(
        is_input=True,
        arg_index=operand.arg_index,
        device_dtype=_layout.device_dtype(operand.dtype),
        device_size=device_size,
        device_coordinates=coords,
        allocation=dict(operand.allocation),
        device_tile_advance_expr=advance,
    )
    tiling = emitter.tiling_for(op_key)
    tiled_symbols, trip_counts = tiling if tiling is not None else ([], {})
    emitter.emit(
        emitter.op_spec(
            "identity",
            is_reduction=False,
            iteration_space=_iteration_space(symbols, sizes, [1] * len(sizes)),
            args=[src_arg, staged.tensor_arg(symbols, is_input=False)],
            tiled_symbols=tiled_symbols,
            tiled_symbol_trip_counts=trip_counts,
        )
    )
    return staged


def _block_id_of(val: torch.Tensor, dim: int) -> int | None:
    """Block id behind host dim ``dim`` of a tile-shaped value, if tiled."""
    from ..compile_environment import CompileEnvironment

    sizes = list(val.size())
    if dim >= len(sizes):
        return None
    return CompileEnvironment.current().resolve_block_id(sizes[dim])


@dataclasses.dataclass
class MatmulShape:
    """Extents of a 2-D matmul, separating iteration from layout.

    ``k`` is the *iteration* extent of the reduction dim -- one coarse tile
    inside a K-tiled loop -- while ``x_sizes``/``y_sizes`` are the operands'
    full device extents. The reference keeps exactly this split: ``c2 = 128``
    with ``y`` still laid out ``[1, 512, 64]``, the rest of K being reached
    through ``device_tile_advance_expr``.
    """

    m: int
    n: int
    k: int
    out_splits: list[int]
    x_sizes: list[int]
    y_sizes: list[int]


#: The DF16 stick axis each matmul operand must carry, by role. This is the
#: rule at propagate_layouts.py:1365-1368 expressed as the *host dim* each 2-D
#: operand sticks on, and it is the single place the literal lives:
#: ``_matmul_arg``, ``_k_advance`` and ``_stage_advancing_operand`` all took a
#: bare ``stick_dim=1`` before, which made the assumption true by construction
#: rather than by agreement with the operand.
#:
#:   x[M, K] sticks on K = its dim 1 (the reduction var, absent from the output)
#:   y[K, N] sticks on N = its dim 1 (the generated var, present in the output)
#:   out[M, N] sticks on N = its dim 1
#:
#: All three are 1 for a 2-D matmul. They are named rather than inlined because
#: they are three *different* axes that happen to share an index, and a reader
#: who sees one ``1`` has no way to tell which rule it came from.
_MATMUL_STICK_DIM = {"x": 1, "y": 1, "out": 1}


def _reconcile_matmul_stick_axes(
    emitter: SpyreEmitter,
    shape: MatmulShape,
    *,
    x: Operand,
    y: Operand,
) -> tuple[Operand, Operand]:
    """Relayout any matmul operand whose recorded stick axis is not the DF16 one.

    **This must run before staging and before any ``TensorArg`` is built.**
    ``_matmul_arg`` reshapes an operand with ``stick_dim=_MATMUL_STICK_DIM[...]``
    and ``_stage_advancing_operand`` does the same, so both *assert* the rule
    rather than check it. Those assertions are only sound once a disagreeing
    operand has already been relaid out, which is what this does -- hence the
    ordering invariant ``reconcile -> stage -> build args``, tested by
    ``repros/matmul_stick_axis.py``.

    Without this, a dim-0-sticked operand was **silently reinterpreted**: the
    emitter addressed it as if it sticked on dim 1 and returned a plausible,
    wrong answer with nothing raised anywhere (measured 19.5% max error on a
    128x256 @ 256x128 fp16 matmul with x staged ``dim_order=[1, 0]``; G1, the
    backend compiler and the launch all accepted it). That is the review's
    finding (a), and it is the same class as bug #6 -- which
    ``_emit_pointwise`` (:163-169) and ``_emit_reduction`` (:293-294) already
    guard by comparing and emitting ``ReStickifyOpHBM`` on a mismatch. This is
    that guard for the matmul quadrant, which was simply never written.

    The relayout is done at the operand's **full** extents, not its registered
    ones: a buffer registered from a tile-shaped load carries the tile's
    extents, while the matmul addresses the full ones (the same reason
    ``_matmul_arg`` takes ``sizes`` explicitly). Relaying out at the tile shape
    would produce a correctly-sticked buffer of the wrong size.

    The DF16 rule itself is **not** re-decided here. This only detects
    disagreement; where the sticks must go stays where it was.
    """
    c0, c1, c2 = _iteration_symbols(3)

    def reconcile(
        operand: Operand, role: str, sizes: list[int], symbols: list[sympy.Expr]
    ) -> Operand:
        want = _MATMUL_STICK_DIM[role]
        if operand.stick_dim == want:
            return operand
        import dataclasses as _dc

        from .emitter import Operand as _Operand

        assert isinstance(operand, _Operand)
        # Restickify at the full extents the matmul will address. The source
        # keeps its own (wrong-for-this-op) axis, which is what makes the
        # emitted ReStickifyOpHBM non-degenerate -- emit_restickify rejects a
        # relayout onto the axis the operand already has.
        at_full = _dc.replace(operand, sizes=list(sizes))
        return emitter.emit_restickify(at_full, want, symbols)

    return (
        reconcile(x, "x", shape.x_sizes, [c0, c2]),
        reconcile(y, "y", shape.y_sizes, [c2, c1]),
    )


def _matmul_shape(node: Node, x_node: Node, y_node: Node) -> MatmulShape:
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    if val.dim() != 2:
        raise exc.BackendUnsupported("spyre", "only 2-D matmul is modelled")

    x_iter, _ = tile_space(x_node.meta["val"])
    y_iter, _ = tile_space(y_node.meta["val"])
    out_sizes, out_splits = tile_space(val)
    if len(x_iter) != 2 or len(y_iter) != 2:
        raise exc.BackendUnsupported("spyre", "only 2-D matmul is modelled")
    if y_iter[0] != x_iter[1]:
        raise exc.BackendUnsupported(
            "spyre", f"matmul shape mismatch: {x_iter} @ {y_iter}"
        )

    # device_size stays FULL for an operand that advances. The reference is
    # explicit about the two roles: its y reads full K
    # (device_size=[1,512,64]) *with* floor(8192*L), and 8192 = k_tile *
    # elems_per_stick is a stride within that full buffer -- so the advance and
    # the full extent belong together. Only a one-tile window read from a pool
    # shrinks device_size, and that window carries no advance at all
    # (golden/ref_matmul_ktiled.py: the bmm's x arg is [2,64,64], hbm_pool, no
    # advance). Helion reads straight from HBM, so it takes the advancing role.
    x_full = _full_sizes(x_node)
    y_full = _full_sizes(y_node)

    return MatmulShape(
        m=out_sizes[0],
        n=out_sizes[1],
        k=x_iter[1],
        out_splits=out_splits,
        x_sizes=x_full,
        y_sizes=y_full,
    )


def _full_sizes(node: Node) -> list[int]:
    """An operand's full device extents, ignoring sequential tiling."""
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    sizes: list[int] = []
    for size in val.size():
        block_id = env.resolve_block_id(size)
        if block_id is None:
            sizes.append(int(env.size_hint(size)))
        else:
            sizes.append(int(env.block_sizes[block_id].size_hint()))
    return sizes


def _matmul_op_spec(
    emitter: SpyreEmitter,
    op: str,
    *,
    x: Operand,
    y: Operand,
    out: Operand,
    shape: MatmulShape,
    op_key: str = "op",
    x_advances: bool = True,
) -> object:
    """Build the ``batchmatmul`` OpSpec under the DF16 stick rule.

    Plan 4.2, verbatim from propagate_layouts.py:1365-1368 -- the rule the
    captured reference confirms is *per-operand distinct*:

        Input1 (x): stick on reduction_var (loop var absent from output)
        Input2 (y): stick on generated_var (present in output, absent from x)
        Output:     stick on generated_var

    With ``c0=M``, ``c1=N``, ``c2=K``: x[M,K] sticks on K (its dim 1), y[K,N]
    sticks on N (its dim 1), out[M,N] sticks on N (its dim 1).
    """
    c0, c1, c2 = _iteration_symbols(3)
    # Plan 4.1's hard constraint: at most one reduction dim may be split across
    # cores (work_division.py:328-331), and a K-split builds a PSUM chain the
    # emitter does not model -- so K is always "not split".
    iteration_space = {
        c0: (_int(shape.m), shape.out_splits[0]),
        c1: (_int(shape.n), shape.out_splits[1]),
        c2: (_int(shape.k), 1),
    }

    # Both inputs are indexed by the reduction symbol c2, so inside a K-tiled
    # loop BOTH must advance -- each by its own amount, since they stick on
    # different axes. Omitting x's advance makes it re-read K-tile 0 forever,
    # which a loose numeric tolerance will not catch.
    # This op's own level symbol: sharing one symbol across the read-copy,
    # matmul and combine ops collapses their advances (spyre_kernel.py:588-610).
    level = emitter.level_symbol_for(op_key)

    # A staged x is already one K-tile wide and does not advance; an unstaged
    # x spans full K and must advance with the loop.
    x_sizes = shape.x_sizes if x_advances else [shape.x_sizes[0], shape.k]
    x_arg = _matmul_arg(
        x,
        [c0, c2],
        sizes=x_sizes,
        stick_dim=_MATMUL_STICK_DIM["x"],
        is_input=True,
        tile_advance=(
            _k_advance(
                emitter,
                x,
                shape.x_sizes,
                [c0, c2],
                _MATMUL_STICK_DIM["x"],
                shape.k,
                level,
            )
            if x_advances
            else None
        ),
    )
    y_arg = _matmul_arg(
        y,
        [c2, c1],
        sizes=shape.y_sizes,
        stick_dim=_MATMUL_STICK_DIM["y"],
        is_input=True,
        tile_advance=_k_advance(
            emitter,
            y,
            shape.y_sizes,
            [c2, c1],
            _MATMUL_STICK_DIM["y"],
            shape.k,
            level,
        ),
    )
    out_arg = _matmul_arg(
        out,
        [c0, c1],
        sizes=[shape.m, shape.n],
        stick_dim=_MATMUL_STICK_DIM["out"],
        is_input=False,
    )

    tiling = emitter.tiling_for(op_key)
    tiled_symbols, trip_counts = tiling if tiling is not None else ([], {})
    return emitter.op_spec(
        op,
        is_reduction=True,
        iteration_space=iteration_space,
        args=[x_arg, y_arg, out_arg],
        tiled_symbols=tiled_symbols,
        tiled_symbol_trip_counts=trip_counts,
    )


def _k_advance(
    emitter: SpyreEmitter,
    operand: Operand,
    sizes: list[int],
    symbols: list[sympy.Expr],
    stick_dim: int,
    k_tile: int,
    level_symbol: sympy.Expr | None = None,
) -> sympy.Expr | None:
    """This operand's per-step advance along the tiled reduction dim.

    Built from the operand's realized layout, so it cannot disagree with the
    ``device_coordinates`` the same operand is emitted with.
    """
    import dataclasses as _dc

    from . import layout
    from .emitter import Operand as _Operand

    assert isinstance(operand, _Operand)
    reshaped = _dc.replace(operand, sizes=list(sizes), stick_dim=stick_dim)
    device_size, coords = layout.stickify(
        reshaped.sizes, symbols, reshaped.stick_dim, reshaped.dtype
    )
    return emitter.tile_advance_for(
        device_size,
        coords,
        _reduction_symbol(symbols),
        k_tile,
        level_symbol=level_symbol,
    )


def _reduction_symbol(symbols: list[sympy.Expr]) -> sympy.Expr:
    """``c2``, the matmul reduction symbol, in an operand's symbol list."""
    for sym in symbols:
        if getattr(sym, "name", "") == "c2":
            return sym
    raise exc.BackendUnsupported("spyre", "matmul operand has no reduction symbol")


def _emit_matmul(
    backend: SpyreBackend, ctx: LoweringContext, node: Node, op: str
) -> Operand:
    """One ``batchmatmul`` OpSpec; K is an iteration-space dim, not a loop."""
    emitter = backend.emitter
    inputs = _operand_inputs(ctx, node)
    if len(inputs) != 2:
        raise exc.BackendUnsupported(
            "spyre", f"matmul with {len(inputs)} operands, expected 2"
        )
    x, y = inputs
    x_node, y_node = node.all_input_nodes[0], node.all_input_nodes[1]

    shape = _matmul_shape(node, x_node, y_node)
    # Compare each operand's *recorded* stick axis against the DF16 rule before
    # anything is addressed under that rule. See
    # _reconcile_matmul_stick_axes for why this ordering is the whole fix.
    x, y = _reconcile_matmul_stick_axes(emitter, shape, x=x, y=y)
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    out = _bound_or_intermediate(
        emitter, node, [shape.m, shape.n], val.dtype, _MATMUL_STICK_DIM["out"]
    )

    emitter.emit(_matmul_op_spec(emitter, op, x=x, y=y, out=out, shape=shape))
    return out


def _stage_advancing_operand(
    emitter: SpyreEmitter,
    operand: Operand,
    shape: MatmulShape,
    node: Node,
    *,
    dtype: torch.dtype,
) -> Operand:
    """Copy one K-tile of ``operand`` into a pool buffer, advancing per step.

    The staging op is an ``identity`` whose source keeps the operand's full
    device_size and carries the tile advance, and whose destination is a
    one-tile ``hbm_pool`` window with no advance -- the shape
    golden/ref_matmul_ktiled.py emits as ``coarse_tile_read_copy_0_arg0_1_0``.
    """
    import dataclasses as _dc

    c0, _, c2 = _iteration_symbols(3)
    k_tile = shape.k

    op_key = f"read_copy_{node.name}"
    level = emitter.level_symbol_for(op_key)
    # ``stick_dim`` here asserts the DF16 rule rather than checking it; that is
    # sound only because _reconcile_matmul_stick_axes has already relaid out any
    # operand that disagreed (the reconcile -> stage -> build-args invariant).
    source = _dc.replace(
        operand, sizes=list(shape.x_sizes), stick_dim=_MATMUL_STICK_DIM["x"]
    )
    device_size, coords = layout_stickify(source, [c0, c2])
    advance = emitter.tile_advance_for(
        device_size, coords, c2, k_tile, level_symbol=level
    )
    if advance is None:
        # No open coarse-tile level: the operand does not move, so the matmul
        # can read it directly and staging would be pure overhead.
        return operand

    staged = emitter.add_intermediate(
        f"{node.name}_ktile",
        [shape.x_sizes[0], k_tile],
        dtype,
        1,
        residency="hbm_pool",
    )

    from torch_spyre._inductor.op_spec import TensorArg

    from . import layout as _layout

    src_arg = TensorArg(
        is_input=True,
        arg_index=operand.arg_index,
        device_dtype=_layout.device_dtype(dtype),
        device_size=device_size,
        device_coordinates=coords,
        allocation=dict(operand.allocation),
        device_tile_advance_expr=advance,
    )
    staging_tiling = emitter.tiling_for(op_key)
    tiled_symbols, trip_counts = (
        staging_tiling if staging_tiling is not None else ([], {})
    )
    emitter.emit(
        emitter.op_spec(
            "identity",
            is_reduction=False,
            iteration_space=_iteration_space(
                [c0, c2], [shape.x_sizes[0], k_tile], [1, 1]
            ),
            args=[src_arg, staged.tensor_arg([c0, c2], is_input=False)],
            tiled_symbols=tiled_symbols,
            tiled_symbol_trip_counts=trip_counts,
        )
    )
    return staged


def layout_stickify(
    operand: Operand, symbols: list[sympy.Expr]
) -> tuple[list[int], list[sympy.Expr]]:
    from . import layout as _layout

    return _layout.stickify(operand.sizes, symbols, operand.stick_dim, operand.dtype)


def _emit_addmm(backend: SpyreBackend, ctx: LoweringContext, node: Node) -> Operand:
    """``addmm(acc, x, y)`` -> a ``batchmatmul`` partial plus an ``add``.

    Two OpSpecs, because Spyre's matmul writes its own output buffer rather
    than accumulating into a given one. This is exactly the structure
    torch-spyre's coarse tiler emits inside a K-tiled LoopSpec: a
    ``batchmatmul`` into an lx partial, then an ``add`` combining that partial
    into the accumulator in place (golden/ref_matmul_ktiled.py, ops 3 and 4).

    Emitting the ``add`` with the accumulator as both an input and the output
    is what makes the carry work without ``iter_args`` (plan 6).
    """
    emitter = backend.emitter
    acc_node, x_node, y_node = node.all_input_nodes[:3]
    acc = _as_operand(ctx, acc_node, node)
    x = _as_operand(ctx, x_node, node)
    y = _as_operand(ctx, y_node, node)

    shape = _matmul_shape(node, x_node, y_node)
    # Before staging, not after: _stage_advancing_operand reshapes x with
    # ``stick_dim=_MATMUL_STICK_DIM["x"]``, so a disagreeing operand reaching it
    # unreconciled re-introduces the silent reinterpretation through the staging
    # path even if the TensorArg construction below is correct. This is the
    # ordering invariant _reconcile_matmul_stick_axes documents.
    x, y = _reconcile_matmul_stick_axes(emitter, shape, x=x, y=y)
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)

    # Stage the K-advancing operand into a one-tile pool buffer first.
    # golden/ref_matmul_ktiled.py shows the two roles cannot be combined on one
    # TensorArg: the advancing read keeps its FULL device_size and carries
    # floor(8192*L) (the read-copy identity, arg0 [8,64,64] hbm + advance ->
    # arg1 [2,64,64] hbm_pool, no advance), and the matmul then reads that
    # one-tile pool window with no advance of its own. Emitting the matmul
    # directly against advancing full-K HBM instead makes every iteration read
    # the same final tile -- a per-stick probe showed only K[384:512] read, and
    # twice over.
    x_staged = _stage_advancing_operand(emitter, x, shape, node, dtype=val.dtype)
    mm_key = f"mm_{node.name}"
    combine_key = f"combine_{node.name}"

    partial = emitter.add_intermediate(
        f"{node.name}_partial", [shape.m, shape.n], val.dtype, 1, residency="lx"
    )
    emitter.emit(
        _matmul_op_spec(
            emitter,
            "batchmatmul",
            x=x_staged,
            y=y,
            out=partial,
            shape=shape,
            op_key=mm_key,
            x_advances=False,
        )
    )

    # Combine into the accumulator: acc = acc + partial, in place.
    c0, c1 = _iteration_symbols(2)
    combine_space = _iteration_space([c0, c1], [shape.m, shape.n], [1, 1])
    combine_tiling = emitter.tiling_for(combine_key)
    tiled_symbols, trip_counts = (
        combine_tiling if combine_tiling is not None else ([], {})
    )
    emitter.emit(
        emitter.op_spec(
            "add",
            is_reduction=False,
            iteration_space=combine_space,
            args=[
                acc.tensor_arg([c0, c1], is_input=True),
                partial.tensor_arg([c0, c1], is_input=True),
                acc.tensor_arg([c0, c1], is_input=False),
            ],
            tiled_symbols=tiled_symbols,
            tiled_symbol_trip_counts=trip_counts,
        )
    )
    return acc


def _as_operand(ctx: LoweringContext, arg: Node, node: Node) -> Operand:
    """The buffer for an input node, resolving a loop-carry placeholder.

    A carried accumulator crosses the loop boundary as an ``ast.Name``
    (see tracing_ops._new_var), so inside the body its env value is that name
    rather than the buffer.
    """
    import ast as _ast

    from .emitter import Operand as _Operand

    value = ctx.env.get(arg)
    if isinstance(value, _Operand):
        return value

    if isinstance(value, _ast.Name):
        emitter = _spyre_backend().emitter
        carried = emitter.carry(value.id)
        if carried is not None:
            return carried

    raise exc.BackendUnsupported(
        "spyre",
        f"{node.target} input {arg} is {type(value).__name__}, not a buffer",
    )


def _int(value: int) -> sympy.Expr:
    import sympy

    return sympy.Integer(value)


def _matmul_arg(
    operand: Operand,
    symbols: list[sympy.Expr],
    *,
    sizes: list[int],
    stick_dim: int,
    is_input: bool,
    tile_advance: sympy.Expr | None = None,
) -> object:
    """A matmul operand's ``TensorArg`` under its own stick rule.

    The operand's own extents are passed in rather than read off ``operand``:
    a buffer registered from a tile-shaped load carries the tile's extents,
    while the matmul needs the full ones.
    """
    import dataclasses as _dc

    from .emitter import Operand as _Operand

    assert isinstance(operand, _Operand)
    reshaped = _dc.replace(operand, sizes=list(sizes), stick_dim=stick_dim)
    return reshaped.tensor_arg(symbols, is_input=is_input, tile_advance=tile_advance)


def _output_stick_dim(rank: int) -> int:
    """Stick axis of the kernel's output tensor, from the one authority.

    Thin wrapper over ``SpyreBackend.kernel_stick_dim`` so this module, the
    stick-split bound in ``_iteration_space`` and
    ``autotune_config_is_viable`` cannot drift apart about which axis carries
    the sticks -- they used to, and the viability check guarded the trailing
    axis for a dim-0-sticked kernel (review finding (d)).
    """

    return _spyre_backend().kernel_stick_dim(rank)


def _iteration_space(
    symbols: list[sympy.Symbol],
    sizes: list[int],
    splits: list[int],
    stick_dim: int | None = None,
    dtype: torch.dtype | None = None,
) -> dict[sympy.Symbol, tuple[sympy.Expr, int]]:
    """``{sym: (extent, cores)}`` -- the OpSpec field that *is* work division.

    The **stick axis** may be divided only down to whole sticks. The rule at
    codegen/opspec_utils.py:515-516 ("a stick is the unit of transfer, so
    splitting it across cores would split a stick") is usually read as a
    statement about the trailing axis, but it binds whichever axis actually
    carries the sticks: a dim-0-sticked op over 128 elements (2 sticks at fp16)
    accepts splits 1 and 2 and is rejected by the backend scheduler at 3, 4 and
    8 with "Expect valid lower and upper bound parameters".

    **This used to clamp, and now it raises.** The clamp was justified as "a
    lower split is always correct", which is true numerically and wrong
    structurally: the autotuner then *benchmarked a work division it did not
    choose*. Measured on a dim-0-sticked 128x256 fp16 kernel, configs asking
    for c0 splits of 2, 4 and 8 all emitted a split of **2**, while
    ``autotune_config_is_viable`` returned ``True`` for all three -- so the
    search space held three configs that were one kernel, and the autotuner
    spent real launches distinguishing them. That is §1's governing invariant
    ("never solved behind their back") bent in the one place the autotuner is
    meant to be in charge, and it is the review's finding (d) / §9 item 3.

    The bound now lives in ``autotune_config_is_viable``, which reads the same
    axis through ``SpyreBackend.kernel_stick_dim``, so the search space
    excludes these configs instead of codegen silently rewriting them. Reaching
    here with an over-split stick axis therefore means the two disagreed, which
    is a bug in the backend rather than a bad config -- so it fails loudly
    rather than papering over itself.
    """
    import sympy

    if stick_dim is not None and dtype is not None and stick_dim < len(sizes):
        from . import layout as _layout

        sticks = max(1, -(-sizes[stick_dim] // _layout.elems_per_stick(dtype)))
        if splits[stick_dim] > sticks:
            raise exc.BackendUnsupported(
                "spyre",
                f"work division splits the stick axis (dim {stick_dim},"
                f" extent {sizes[stick_dim]} = {sticks} stick(s)) into"
                f" {splits[stick_dim]} parts. autotune_config_is_viable should"
                " have screened this config; reaching codegen means the"
                " viability check and this bound disagree about which axis"
                " carries the sticks",
            )

    return {
        sym: (sympy.Integer(size), split)
        for sym, size, split in zip(symbols, sizes, splits, strict=True)
    }
