"""``SpyreEmitter``: the spec-tree accumulator behind the per-node handlers.

Plan 2: the node->emission map is two-kinded. Memory nodes bind ``TensorArg``
descriptors into emitter state and produce no ``OpSpec``; compute nodes produce
an ``OpSpec`` consuming those descriptors; structure nodes open and close a
``LoopSpec``.

The emitter owns exactly the fields plan 0 lists as "the field that *is* the
decision": iteration-space work division, core mapping, stick layout, residency,
and the loop nesting. torch-spyre only serializes them.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import Any
from collections.abc import Sequence

from ... import exc
from . import layout

if TYPE_CHECKING:
    import sympy
    import torch


@dataclasses.dataclass
class Operand:
    """One Spyre buffer, as Helion knows it before an OpSpec needs it.

    Plan 2's invariant: Helion's tile is a value in registers, Spyre's is an
    address range in a buffer, so every intermediate needs a real buffer. This
    is that buffer, independent of which OpSpec reads or writes it.
    """

    #: Host-visible logical shape.
    sizes: list[int]
    dtype: torch.dtype
    #: Host dim laid out along sticks (plan 4.2, per-operand).
    stick_dim: int
    #: ``{"hbm": i}`` / ``{"lx": 0}`` / ``{"hbm_pool": 0}``; plan 4.4.
    allocation: dict[str, Any]
    #: Kernel-arg position, or -1 for an intermediate (the lab's ``_pool_op``
    #: shows arg_index=-1 is what makes a buffer genuinely non-argument).
    arg_index: int
    name: str | None = None

    def tensor_arg(
        self,
        symbols: list[sympy.Expr],
        *,
        is_input: bool,
        tile_advance: sympy.Expr | None = None,
    ) -> object:
        """Realize this buffer as a ``TensorArg`` under a given indexing.

        ``symbols[i]`` indexes host dim ``i``. The same buffer yields different
        ``device_coordinates`` in different OpSpecs because the iteration
        symbols differ, which is exactly why layout is applied here and not
        cached on the operand.
        """
        from torch_spyre._inductor.op_spec import TensorArg

        if tile_advance is not None and "lx" in self.allocation:
            # scratchpad/allocator.py:628 -- LX addresses cannot be expressed as
            # affine.apply symbols, so an LX buffer may not advance per tile.
            raise exc.BackendUnsupported(
                "spyre", "an lx buffer cannot carry a device_tile_advance_expr"
            )

        device_size, device_coordinates = layout.stickify(
            self.sizes, symbols, self.stick_dim, self.dtype
        )
        return TensorArg(
            is_input=is_input,
            arg_index=self.arg_index,
            device_dtype=layout.device_dtype(self.dtype),
            device_size=device_size,
            device_coordinates=device_coordinates,
            allocation=dict(self.allocation),
            device_tile_advance_expr=tile_advance,
        )


class SpyreEmitter:
    """Accumulates the ``OpSpec``/``LoopSpec`` tree for one kernel.

    Reset per compile from ``SpyreBackend.pre_codegen``, mirroring how FlyDSL
    resets ``_tensor_use_buffer`` there (flydsl/backend.py:494-556).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        #: The spec tree. ``_stack[-1]`` is the list currently being appended
        #: to, so opening a LoopSpec just pushes its body.
        self._specs: list[object] = []
        self._stack: list[list[object]] = [self._specs]
        #: fx node name -> the buffer holding that value.
        self.operands: dict[str, Operand] = {}
        #: Kernel args in launch order; index is the ``arg_index``.
        self.args: list[Operand] = []
        self._n_lx = 0
        self._n_pool = 0
        self.pool_size = 0
        #: id(Operand) -> (OpSpec, index of its output arg), for the store rule.
        self._producers: dict[int, tuple[Any, int]] = {}
        #: id(Operand) -> the symbols its producing OpSpec indexed it with.
        self._output_symbols: dict[int, list[Any]] = {}
        #: Accumulator buffers awaiting a fill OpSpec, and their fill values.
        #: Plan 2: the fill is emitted *before* the enclosing LoopSpec, which
        #: only the _for_loop handler can locate.
        self.pending_fills: dict[int, tuple[Operand, float]] = {}
        #: Open coarse-tile levels, outermost first: (level symbol, trip count).
        self._levels: list[tuple[Any, int]] = []
        #: Scalar constants the launcher must append: (arg_index, value, dtype).
        self.const_args: list[tuple[int, float, Any]] = []
        #: Loop-carried buffers, by the placeholder name they cross on.
        self._carries: dict[str, Operand] = {}
        #: Per open level: {block_id: per-iteration tile extent}.
        self._tiled_blocks: list[dict[int, int]] = []

    def mark_accumulator(self, operand: Operand, value: float) -> None:
        self.pending_fills[id(operand)] = (operand, value)

    def take_pending_fills(self) -> list[tuple[Operand, float]]:
        fills = list(self.pending_fills.values())
        self.pending_fills.clear()
        return fills

    # -- buffers ----------------------------------------------------------

    def add_arg(
        self, name: str, tensor: torch.Tensor, *, stick_dim: int | None = None
    ) -> Operand:
        """Register a graph input/output as an ``hbm`` kernel arg.

        Plan 4.4: inputs/outputs are ``hbm`` with ``arg_index`` = position.
        Plan 5.3: the stick axis is *read* off the tensor, not imposed.
        """
        if name in self.operands:
            return self.operands[name]
        index = len(self.args)
        operand = Operand(
            sizes=list(tensor.size()),
            dtype=tensor.dtype,
            stick_dim=(
                layout.stick_dim_of(tensor) if stick_dim is None else stick_dim
            ),
            allocation={"hbm": index},
            arg_index=index,
            name=name,
        )
        self.args.append(operand)
        self.operands[name] = operand
        return operand

    def add_intermediate(
        self,
        name: str,
        sizes: list[int],
        dtype: torch.dtype,
        stick_dim: int,
        *,
        residency: str = "lx",
    ) -> Operand:
        """Register an intermediate buffer.

        ``residency="lx"`` matches what torch-spyre's own pipeline does for a
        fused pointwise chain (golden/ref_fused_add_mul.py: the ``(x+y)``
        intermediate is ``allocation={'lx': 0}``, ``arg_index=-1``, pool_size 0).
        ``hbm_pool`` is plan 4.4's alternative and needs ``pool_size`` set.
        """
        if residency == "lx":
            allocation: dict[str, Any] = {"lx": self._n_lx}
            self._n_lx += 1
        elif residency == "hbm_pool":
            allocation = {"hbm_pool": self._n_pool}
            self._n_pool += 1
            n = 1
            for s in sizes:
                n *= s
            self.pool_size += n * dtype.itemsize
        else:
            raise exc.BackendUnsupported("spyre", f"residency: {residency}")

        operand = Operand(
            sizes=list(sizes),
            dtype=dtype,
            stick_dim=stick_dim,
            allocation=allocation,
            arg_index=-1,
            name=name,
        )
        self.operands[name] = operand
        return operand

    def bind(self, name: str, operand: Operand) -> None:
        """Alias an fx node onto an existing buffer (a store, or a view)."""
        self.operands[name] = operand

    def get(self, name: str) -> Operand:
        operand = self.operands.get(name)
        if operand is None:
            raise exc.BackendUnsupported("spyre", f"no buffer bound for {name!r}")
        return operand

    # -- spec tree --------------------------------------------------------

    def emit(self, spec: object) -> None:
        self._stack[-1].append(spec)

    def add_constant_arg(self, value: float, dtype: torch.dtype) -> Operand:
        """A scalar constant as a kernel arg, materialized by the launcher.

        Spyre has no immediate operands: the reference's zero-fill reads a
        0-dim scalar *kernel argument* (golden/ref_matmul_ktiled.py arg0, a
        ``()`` fp16 tensor laid out ``[1, 64]``). Helion's host code has no such
        value to pass, so the emitter records the constant and the launcher
        appends it -- which keeps the host wrapper unchanged and the constant
        visible in the spec.
        """
        index = len(self.args)
        operand = Operand(
            sizes=[1, 1],
            dtype=dtype,
            stick_dim=1,
            allocation={"hbm": index},
            arg_index=index,
            name=f"_spyre_const{index}",
        )
        self.args.append(operand)
        self.const_args.append((index, value, dtype))
        return operand

    def emit_zero_fill(self, operand: Operand) -> None:
        """Zero an accumulator with an ``identity`` from a scalar constant.

        Matches the reference's leading fill: a 1-element source at constant
        coordinates broadcast over the accumulator's full iteration space.
        """
        import sympy

        source = self.add_constant_arg(0.0, operand.dtype)
        symbols = [sympy.Symbol(f"c{i}") for i in range(len(operand.sizes))]
        zero = [sympy.Integer(0)] * (len(operand.sizes) + 1)

        from torch_spyre._inductor.op_spec import TensorArg

        from . import layout

        source_arg = TensorArg(
            is_input=True,
            arg_index=source.arg_index,
            device_dtype=layout.device_dtype(operand.dtype),
            device_size=[1, 1, layout.elems_per_stick(operand.dtype)],
            device_coordinates=zero,
            allocation=dict(source.allocation),
        )
        self.emit(
            self.op_spec(
                "identity",
                is_reduction=False,
                iteration_space={
                    sym: (sympy.Integer(size), 1)
                    for sym, size in zip(symbols, operand.sizes, strict=True)
                },
                args=[source_arg, operand.tensor_arg(symbols, is_input=False)],
            )
        )

    def open_loop(
        self,
        level_symbol: sympy.Symbol | None = None,
        trip: int = 0,
        block_ids: Sequence[int] = (),
        block_size: int = 0,
    ) -> list[object]:
        """Open a coarse-tile level; body OpSpecs become its ``LoopSpec`` body.

        ``level_symbol`` is this level's tile-advance symbol. torch-spyre mints
        one symbol per level and reuses the *same object* in both
        ``tiled_symbols`` and every ``device_tile_advance_expr``
        (spyre_kernel.py:552-568, :743) -- that shared identity is what lets
        SDSC codegen recover each level's stride via ``coeff``.

        ``block_ids`` are the dims this loop tiles sequentially. Inside the
        body their iteration extent is one *tile* (``block_size``), not the full
        dim: the reference's K-tiled matmul has ``c2 = 128`` with K=512, and
        reaches the rest of K through ``device_tile_advance_expr``.
        """
        body: list[object] = []
        self._stack.append(body)
        self._levels.append((level_symbol, trip))
        self._tiled_blocks.append({int(b): block_size for b in block_ids})
        return body

    def tiled_block_size(self, block_id: int) -> int | None:
        """Per-iteration extent of a sequentially tiled dim, if it is one."""
        for level in reversed(self._tiled_blocks):
            if block_id in level:
                return level[block_id]
        return None

    def close_loop(self, count: int) -> None:
        """Close the innermost loop, wrapping its body in a ``LoopSpec``."""
        import sympy

        from torch_spyre._inductor.op_spec import LoopSpec

        body = self._stack.pop()
        self._levels.pop()
        self._tiled_blocks.pop()
        if not body:
            # op_spec_validation.py:213-233 rejects an empty body; fail here
            # with a Helion-level message instead.
            raise exc.BackendUnsupported("spyre", "loop with no emitted ops")
        self.emit(LoopSpec(count=sympy.Integer(count), body=body))

    def bind_carry(self, operand: Operand) -> str:
        """Name a buffer so it can cross a loop boundary as an AST placeholder.

        ``ForLoopGraphInfo.codegen`` asserts captured loop args are ``ast.AST``
        (device_ir.py:382), so a carried buffer travels as a ``Name`` and is
        recovered here by that name.
        """
        name = f"_spyre_carry{len(self._carries)}"
        self._carries[name] = operand
        return name

    def carry(self, name: str) -> Operand | None:
        """The buffer behind a carry placeholder name.

        Helion renames a captured loop arg when it crosses the boundary
        (``_spyre_carry0`` -> ``_spyre_carry0_copy``), so match on the stable
        prefix. Carry names are minted uniquely, so the prefix is unambiguous.
        """
        operand = self._carries.get(name)
        if operand is not None:
            return operand
        for key, value in self._carries.items():
            if name.startswith(key):
                return value
        return None

    def abandon_loop(self) -> None:
        """Drop an open level after a failed body walk, preserving the error."""
        self._stack.pop()
        self._levels.pop()
        self._tiled_blocks.pop()

    def current_tiling(
        self,
    ) -> tuple[list[list[Any]], dict[Any, int]] | None:
        """``(tiled_symbols, tiled_symbol_trip_counts)`` for the open levels.

        ``tiled_symbols`` is innermost-first and must have one entry per
        enclosing level even when empty (op_spec.py:323-346).
        """
        if not self._levels:
            return None
        symbols: list[list[Any]] = []
        trips: dict[Any, int] = {}
        for symbol, trip in reversed(self._levels):
            if symbol is None:
                symbols.append([])
                continue
            symbols.append([symbol])
            trips[symbol] = trip
        return symbols, trips

    def current_k_advance(
        self, k_tile: int, dtype: torch.dtype
    ) -> sympy.Expr | None:
        """Per-iteration device-element advance of a K-tiled operand.

        One coarse step moves the read window by a whole K-tile, which in
        device elements is ``k_tile * elems_per_stick``: the reference's
        ``floor(8192*_tile_adv_op0_lvl0)`` is 128 * 64 for a 128-element K-tile
        at fp16. ``spyre_kernel.py:732-768`` does the same reprojection from
        the tiling expression into device-element space.
        """
        import sympy

        from . import layout

        if not self._levels:
            return None
        symbol, _ = self._levels[-1]
        if symbol is None:
            return None
        return sympy.floor(k_tile * layout.elems_per_stick(dtype) * symbol)

    @property
    def loop_depth(self) -> int:
        return len(self._stack) - 1

    def redirect_output(self, value: Operand, dest: Operand) -> None:
        """Make the OpSpec that produced ``value`` write ``dest`` instead.

        Plan 2's store rule. ``value`` is an intermediate buffer whose producing
        OpSpec is the most recent one writing it; rewriting that OpSpec's output
        arg avoids emitting a copy. The intermediate then has no users, so its
        ``lx``/``hbm_pool`` slot is released by not having been needed.
        """
        producer = self._producers.get(id(value))
        if producer is None:
            raise exc.BackendUnsupported(
                "spyre", f"store of a value with no producing OpSpec: {value.name}"
            )
        op_spec, arg_pos = producer
        old = op_spec.args[arg_pos]
        op_spec.args[arg_pos] = dest.tensor_arg(
            self._output_symbols[id(value)],
            is_input=False,
            tile_advance=old.device_tile_advance_expr,
        )
        if "lx" in value.allocation:
            self._n_lx -= 1
        elif "hbm_pool" in value.allocation:
            self._n_pool -= 1
            n = 1
            for s in value.sizes:
                n *= s
            self.pool_size -= n * value.dtype.itemsize
        self.operands.pop(value.name, None) if value.name else None

    def specs(self) -> list[object]:
        if self.loop_depth != 0:
            raise exc.BackendUnsupported("spyre", "unclosed loop at end of kernel")
        return self._specs

    # -- OpSpec construction ---------------------------------------------

    def op_spec(
        self,
        op: str,
        *,
        is_reduction: bool,
        iteration_space: dict[sympy.Symbol, tuple[sympy.Expr, int]],
        args: list[object],
        op_info: dict[str, Any] | None = None,
        tiled_symbols: list[list[sympy.Symbol]] | None = None,
        tiled_symbol_trip_counts: dict[sympy.Symbol, int] | None = None,
        output: Operand | None = None,
        output_symbols: list[sympy.Expr] | None = None,
    ) -> object:
        """Build an ``OpSpec``, deriving the core mapping from the given space.

        ``core_id_to_work_slice`` is derived with torch-spyre's own
        ``derive_operation_mapping``, never hand-rolled: superdsc.py:2059-2060
        hard-errors if it is missing, and codegen "serializes this mapping but
        does not choose or reconstruct it".

        ``output``/``output_symbols`` record which buffer this op writes, so a
        later ``store`` can redirect it (plan 2).
        """
        from torch_spyre._inductor.core_mapping import derive_operation_mapping
        from torch_spyre._inductor.op_spec import OpSpec

        spec = OpSpec(
            op=op,
            is_reduction=is_reduction,
            iteration_space=iteration_space,
            core_id_to_work_slice=derive_operation_mapping(iteration_space),
            op_info=op_info if op_info is not None else {},
            args=list(args),
            tiled_symbols=tiled_symbols if tiled_symbols is not None else [],
            tiled_symbol_trip_counts=(
                tiled_symbol_trip_counts
                if tiled_symbol_trip_counts is not None
                else {}
            ),
        )
        if output is not None:
            assert output_symbols is not None
            self._producers[id(output)] = (spec, len(args) - 1)
            self._output_symbols[id(output)] = list(output_symbols)
        return spec
