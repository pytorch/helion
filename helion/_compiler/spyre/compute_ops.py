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
from typing import cast

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
    emitter = env.backend.emitter

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
        splits.append(
            1 if value is None else _split_from_block(extent, int(value))
        )
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
            raise exc.BackendUnsupported(
                "spyre", f"reduction {self.target} (plan M4, not yet implemented)"
            )
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

    # The output sticks where its inputs do; a mismatch is a relayout (M3), not
    # something to paper over.
    stick_dims = {o.stick_dim for o in inputs}
    if len(stick_dims) != 1:
        raise exc.BackendUnsupported(
            "spyre",
            f"operands disagree on stick axis {sorted(stick_dims)};"
            " needs a ReStickifyOpHBM (plan M3)",
        )
    stick_dim = stick_dims.pop()

    out = emitter.add_intermediate(
        node.name, sizes, val.dtype, stick_dim, residency="lx"
    )

    iteration_space = _iteration_space(symbols, sizes, splits)
    args = [o.tensor_arg(symbols, is_input=True) for o in inputs]
    args.append(out.tensor_arg(symbols, is_input=False))

    emitter.emit(
        emitter.op_spec(
            op,
            is_reduction=False,
            iteration_space=iteration_space,
            args=args,
            output=out,
            output_symbols=symbols,
        )
    )
    return out


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

    # Layout extents come from the operands' own buffers, which hold the full
    # tensor shape regardless of how the iteration space is tiled.
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
    emitter: object,
    op: str,
    *,
    x: Operand,
    y: Operand,
    out: Operand,
    shape: MatmulShape,
    tiled: tuple[list[list[sympy.Symbol]], dict[sympy.Symbol, int]] | None = None,
    k_advance: sympy.Expr | None = None,
) -> object:
    """Build the ``batchmatmul`` OpSpec under the DF16 stick rule.

    Plan 4.2, verbatim from propagate_layouts.py:1351-1354 -- the rule the
    captured reference confirms is *per-operand distinct*:

        Input1 (x): stick on reduction_var (loop var absent from output)
        Input2 (y): stick on generated_var (present in output, absent from x)
        Output:     stick on generated_var

    With ``c0=M``, ``c1=N``, ``c2=K``: x[M,K] sticks on K (its dim 1), y[K,N]
    sticks on N (its dim 1), out[M,N] sticks on N (its dim 1).
    """
    c0, c1, c2 = _iteration_symbols(3)
    # Plan 4.1's hard constraint: at most one reduction dim may be split across
    # cores (work_division.py:317-322), and a K-split builds a PSUM chain the
    # emitter does not model -- so K is always "not split".
    iteration_space = {
        c0: (_int(shape.m), shape.out_splits[0]),
        c1: (_int(shape.n), shape.out_splits[1]),
        c2: (_int(shape.k), 1),
    }

    x_arg = _matmul_arg(
        x, [c0, c2], sizes=shape.x_sizes, stick_dim=1, is_input=True
    )
    y_arg = _matmul_arg(
        y,
        [c2, c1],
        sizes=shape.y_sizes,
        stick_dim=1,
        is_input=True,
        tile_advance=k_advance,
    )
    out_arg = _matmul_arg(
        out, [c0, c1], sizes=[shape.m, shape.n], stick_dim=1, is_input=False
    )

    tiled_symbols, trip_counts = tiled if tiled is not None else ([], {})
    return emitter.op_spec(
        op,
        is_reduction=True,
        iteration_space=iteration_space,
        args=[x_arg, y_arg, out_arg],
        tiled_symbols=tiled_symbols,
        tiled_symbol_trip_counts=trip_counts,
        output=out,
        output_symbols=[c0, c1],
    )


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
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    out = emitter.add_intermediate(
        node.name, [shape.m, shape.n], val.dtype, 1, residency="lx"
    )

    emitter.emit(_matmul_op_spec(emitter, op, x=x, y=y, out=out, shape=shape))
    return out


def _emit_addmm(
    backend: SpyreBackend, ctx: LoweringContext, node: Node
) -> Operand:
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
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)

    tiled = emitter.current_tiling()
    partial = emitter.add_intermediate(
        f"{node.name}_partial", [shape.m, shape.n], val.dtype, 1, residency="lx"
    )
    emitter.emit(
        _matmul_op_spec(
            emitter,
            "batchmatmul",
            x=x,
            y=y,
            out=partial,
            shape=shape,
            tiled=tiled,
            k_advance=emitter.current_k_advance(shape.k, val.dtype),
        )
    )

    # Combine into the accumulator: acc = acc + partial, in place.
    c0, c1 = _iteration_symbols(2)
    combine_space = _iteration_space([c0, c1], [shape.m, shape.n], [1, 1])
    tiled_symbols, trip_counts = tiled if tiled is not None else ([], {})
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
            output=acc,
            output_symbols=[c0, c1],
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

    from ..compile_environment import CompileEnvironment
    from .emitter import Operand as _Operand

    value = ctx.env.get(arg)
    if isinstance(value, _Operand):
        return value

    if isinstance(value, _ast.Name):
        emitter = CompileEnvironment.current().backend.emitter
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
    return reshaped.tensor_arg(
        symbols, is_input=is_input, tile_advance=tile_advance
    )


def _iteration_space(
    symbols: list[sympy.Symbol], sizes: list[int], splits: list[int]
) -> dict[sympy.Symbol, tuple[sympy.Expr, int]]:
    """``{sym: (extent, cores)}`` -- the OpSpec field that *is* work division."""
    import sympy

    return {
        sym: (sympy.Integer(size), split)
        for sym, size, split in zip(symbols, sizes, splits, strict=True)
    }
