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


#: Concurrent LX intermediates to request before spilling to the HBM pool.
#: Empirical, not nominal: 3 concurrent 128x256 fp16 LX buffers fail at launch
#: with a Compute CB hardware error, 2 work.
_MAX_LX_BUFFERS = 2


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
        spec_name: str | None = None,
        indirect_dim: int | None = None,
        indirect_from: str | None = None,
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
        if indirect_dim is not None:
            # Replace the coordinate for one host dim with a runtime index read
            # from another TensorArg: IndirectAccess(name) resolves by the
            # *spec_name* of the index arg (op_spec.py's IndirectAccess
            # docstring; the pairing is done in codegen/superdsc.py:1216-1217).
            from torch_spyre._inductor.op_spec import IndirectAccess

            assert indirect_from is not None
            position = self._device_axis_of(indirect_dim)
            device_coordinates = list(device_coordinates)
            device_coordinates[position] = IndirectAccess(indirect_from)
        return TensorArg(
            is_input=is_input,
            arg_index=self.arg_index,
            device_dtype=layout.device_dtype(self.dtype),
            device_size=device_size,
            device_coordinates=device_coordinates,
            allocation=dict(self.allocation),
            device_tile_advance_expr=tile_advance,
            name=spec_name,
        )

    def _device_axis_of(self, host_dim: int) -> int:
        """Device-coordinate position carrying a host dim's whole index.

        ``stickify`` emits ``[floor(stick/eps), *others, Mod(stick, eps)]``, so
        a non-stick host dim appears once in the middle and the stick dim is
        split across the first and last positions. Only a non-stick dim can
        carry an indirect index, since an indirect read is not stick-divisible.
        """
        if host_dim == self.stick_dim:
            raise exc.BackendUnsupported(
                "spyre",
                f"indirect access on the stick axis (host dim {host_dim})",
            )
        others = [i for i in range(len(self.sizes)) if i != self.stick_dim]
        return 1 + others.index(host_dim)


@dataclasses.dataclass
class IndirectOperand:
    """A gathered operand: a value buffer plus the index buffer addressing it.

    Spyre pairs the two by name, so they cannot be separated: the value arg's
    coordinate is ``IndirectAccess(<index arg name>)`` and the index arg carries
    that ``name``. Both must appear in the same OpSpec.
    """

    value: Operand
    index: Operand
    indirect_dim: int

    #: Name the index arg is published under, referenced by IndirectAccess.
    spec_name: str = "_spyre_idx"

    @property
    def sizes(self) -> list[int]:
        """Logical shape of the gathered result."""
        return [
            self.index.sizes[0] if d == self.indirect_dim else extent
            for d, extent in enumerate(self.value.sizes)
        ]

    @property
    def dtype(self) -> torch.dtype:
        return self.value.dtype

    @property
    def stick_dim(self) -> int:
        return self.value.stick_dim

    def args(self, symbols: list[sympy.Expr]) -> list[object]:
        """``[index arg, value arg]`` for the consuming OpSpec.

        Mirrors the captured reference's gather, whose op reads the index tensor
        first and then the value tensor whose coordinate is
        ``IndirectAccess('buf1')`` (golden/ref_gather.py).
        """
        index_symbols = [symbols[self.indirect_dim]]
        return [
            self.index.tensor_arg(index_symbols, is_input=True, spec_name=self.spec_name),
            self.value.tensor_arg(
                symbols,
                is_input=True,
                indirect_dim=self.indirect_dim,
                indirect_from=self.spec_name,
            ),
        ]


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
        #: (op_key, level) -> that op's minted tile-advance symbol.
        self._op_level_symbols: dict[tuple[str, int], Any] = {}

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
        if stick_dim is None:
            # Prefer the axis recorded from the *real* tensor before
            # fakification; a fake tensor carries no device layout.
            from ..compile_environment import CompileEnvironment

            backend = CompileEnvironment.current().backend
            recorded = backend.input_stick_dim(tensor)
            stick_dim = (
                recorded if recorded is not None else layout.stick_dim_of(tensor)
            )
        operand = Operand(
            sizes=list(tensor.size()),
            dtype=tensor.dtype,
            stick_dim=stick_dim,
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

        LX is a hard-capped resource and asking for too much of it is not
        diagnosed anywhere in the stack: a chain needing 3 concurrent LX
        buffers compiles and validates, then fails at launch with
        ``RAS::RUNTIMESCHEDULER::ComputeHardwareError`` ("Compute CB hardware
        error detected"). Measured on this device, 2 LX intermediates of
        128x256 fp16 work and 3 do not, well under the nominal budget of
        ``_LX_TRACKER_CAPACITY_BYTES * dxp_lx_frac_avail`` (~406 KB, ~6 such
        buffers; scratchpad/allocator.py:169, :1524). So the request is capped
        here and anything beyond it spills to ``hbm_pool``, which is the
        residency plan 4.4 proposed in the first place.
        """
        if residency == "lx" and self._n_lx >= _MAX_LX_BUFFERS:
            residency = "hbm_pool"
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

    def emit_restickify(
        self, source: Operand, stick_dim: int, symbols: list[sympy.Expr]
    ) -> Operand:
        """Relayout ``source`` onto a new stick axis, as a visible node.

        Plan 4.2/M3: where an operand's given layout conflicts with what an op
        needs, the backend must emit an explicit ``ReStickifyOpHBM`` rather than
        reinterpret the buffer -- silently reinterpreting is wrong numerics, not
        a crash. ``RESTICKIFY_OP`` is ``constants.py:21``.

        ``op_spec_validation._check_stick_restickify`` (`:687`) requires the
        input and output to have *different* stick symbols, which is exactly the
        condition the caller detected, so a no-op relayout is rejected here
        instead of producing a spec the validator will refuse.
        """
        from torch_spyre._inductor.constants import RESTICKIFY_OP

        if source.stick_dim == stick_dim:
            raise exc.BackendUnsupported(
                "spyre",
                f"restickify from stick axis {stick_dim} to itself is a no-op",
            )

        # The destination must live in HBM, as the op name says: an identical
        # spec writing an ``lx`` destination silently produces wrong data
        # (measured max|d| ~1.0 on a [0,1) input, versus fp16 rounding error
        # when the destination is HBM). Nothing validates this -- G1 and the
        # backend compiler both accept the lx form.
        relaid = self.add_intermediate(
            f"{source.name or 'buf'}_restick",
            list(source.sizes),
            source.dtype,
            stick_dim,
            residency="hbm_pool",
        )
        self.emit(
            self.op_spec(
                RESTICKIFY_OP,
                is_reduction=False,
                iteration_space=self._full_space(source.sizes, symbols),
                args=[
                    source.tensor_arg(symbols, is_input=True),
                    relaid.tensor_arg(symbols, is_input=False),
                ],
            )
        )
        return relaid

    def _full_space(
        self, sizes: list[int], symbols: list[sympy.Expr]
    ) -> dict[Any, tuple[Any, int]]:
        import sympy

        return {
            sym: (sympy.Integer(size), 1)
            for sym, size in zip(symbols, sizes, strict=True)
        }

    def emit_gather(self, gathered: IndirectOperand, dest: Operand) -> None:
        """Stage the gathered table, then copy through the runtime index.

        Two ops, as the reference does (golden/ref_gather.py):

        1. an ``identity`` relayouting the table so the **gathered dim is the
           outermost device axis** -- ``device_size=[256, 2, 64]`` rather than
           the default ``[2, 256, 64]``. ``IndirectAccess`` occupies device
           position 0, so a table whose gathered dim sits on a middle axis
           reads the wrong rows (measured: 98.8% error, a plausible-looking but
           wrong result).
        2. an ``identity`` reading the index arg and the
           IndirectAccess-coordinated staged table, writing ``dest``.
        """
        import sympy

        from torch_spyre._inductor.op_spec import IndirectAccess, TensorArg

        from . import layout

        value = gathered.value
        gdim = gathered.indirect_dim
        dtype = value.dtype
        eps = layout.elems_per_stick(dtype)
        device_dtype = layout.device_dtype(dtype)

        # Staged layout: [gathered extent, sticks of the rest..., eps].
        others = [i for i in range(len(value.sizes)) if i != gdim]
        if len(others) != 1:
            raise exc.BackendUnsupported(
                "spyre", "indirect access is modelled for 2-D operands only"
            )
        rest = value.sizes[others[0]]
        staged_size = [value.sizes[gdim], -(-rest // eps), eps]

        c_g, c_r = sympy.Symbol("c0"), sympy.Symbol("c1")
        source_size, source_coords = layout.stickify(
            value.sizes, [c_g, c_r], value.stick_dim, dtype
        )
        staged_coords = [c_g, sympy.floor(c_r / eps), sympy.Mod(c_r, eps)]

        staged = self.add_intermediate(
            f"{value.name or 'buf'}_gather_src",
            list(value.sizes),
            dtype,
            value.stick_dim,
            residency="hbm_pool",
        )
        self.emit(
            self.op_spec(
                "identity",
                is_reduction=False,
                iteration_space=self._full_space(value.sizes, [c_g, c_r]),
                args=[
                    TensorArg(
                        is_input=True,
                        arg_index=value.arg_index,
                        device_dtype=device_dtype,
                        device_size=source_size,
                        device_coordinates=source_coords,
                        allocation=dict(value.allocation),
                    ),
                    TensorArg(
                        is_input=False,
                        arg_index=-1,
                        device_dtype=device_dtype,
                        device_size=staged_size,
                        device_coordinates=staged_coords,
                        allocation=dict(staged.allocation),
                    ),
                ],
            )
        )

        out_sizes = gathered.sizes
        o_g, o_r = sympy.Symbol("c0"), sympy.Symbol("c1")
        index_arg = gathered.index.tensor_arg(
            [o_g], is_input=True, spec_name=gathered.spec_name
        )
        value_arg = TensorArg(
            is_input=True,
            arg_index=-1,
            device_dtype=device_dtype,
            device_size=staged_size,
            device_coordinates=[
                IndirectAccess(gathered.spec_name),
                sympy.floor(o_r / eps),
                sympy.Mod(o_r, eps),
            ],
            allocation=dict(staged.allocation),
        )
        self.emit(
            self.op_spec(
                "identity",
                is_reduction=False,
                iteration_space=self._full_space(out_sizes, [o_g, o_r]),
                args=[
                    index_arg,
                    value_arg,
                    dest.tensor_arg([o_g, o_r], is_input=False),
                ],
            )
        )

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

    @staticmethod
    def device_element_offset(
        device_size: list[int], coords: list[sympy.Expr]
    ) -> sympy.Expr:
        """Linear device-element offset of a coordinate tuple.

        ``device_size`` is row-major, so dim ``i``'s stride is the product of
        the dims inside it -- the same projection SDSC codegen uses to turn
        coordinates into addresses.
        """
        strides: list[int] = []
        acc = 1
        for dim in reversed(device_size):
            strides.insert(0, acc)
            acc *= dim
        total = 0
        for coord, stride in zip(coords, strides, strict=True):
            total = total + coord * stride
        return total

    def level_symbol_for(self, op_key: str) -> sympy.Symbol | None:
        """The innermost level's tile-advance symbol for one op.

        Minted per (op, level), never shared between ops: spyre_kernel.py:552-569
        mints it that way specifically to fix a "same-real-symbol-tiled-at-2+-
        levels collapse bug", and sharing one symbol across the read-copy,
        matmul and combine ops makes their advances collapse so every iteration
        reads the same tile. ``integer=True`` so Mod()/floor() of a shifted
        coordinate resolve in tile_advance_for.
        """
        import sympy

        if not self._levels:
            return None
        key = (op_key, len(self._levels) - 1)
        symbol = self._op_level_symbols.get(key)
        if symbol is None:
            symbol = sympy.Symbol(
                f"_tile_adv_{op_key}_lvl{len(self._levels) - 1}", integer=True
            )
            self._op_level_symbols[key] = symbol
        return symbol

    def tiling_for(
        self, op_key: str
    ) -> tuple[list[list[Any]], dict[Any, int]] | None:
        """``(tiled_symbols, trip_counts)`` for one op, with its own symbols."""
        if not self._levels:
            return None
        symbols: list[list[Any]] = []
        trips: dict[Any, int] = {}
        for depth in range(len(self._levels) - 1, -1, -1):
            _, trip = self._levels[depth]
            key = (op_key, depth)
            symbol = self._op_level_symbols.get(key)
            if symbol is None:
                import sympy

                symbol = sympy.Symbol(f"_tile_adv_{op_key}_lvl{depth}", integer=True)
                self._op_level_symbols[key] = symbol
            symbols.append([symbol])
            trips[symbol] = trip
        return symbols, trips

    def tile_advance_for(
        self,
        device_size: list[int],
        coords: list[sympy.Expr],
        tiled_symbol: sympy.Symbol,
        tile_extent: int,
        *,
        level_symbol: sympy.Symbol | None = None,
    ) -> sympy.Expr | None:
        """Per-step device-element advance of one operand along a tiled dim.

        Derived from the operand's *own* coordinates rather than a per-op
        formula: shift the tiled iteration symbol by one coarse tile, subtract,
        and evaluate at the window origin. That is the same reprojection
        ``spyre_kernel._general_tile_advance`` performs
        (spyre_kernel.py:732-768), and it is why a matmul's two inputs get
        different advances when they stick on different axes.

        Returns ``None`` when the operand does not move, which is correct for a
        loop-invariant buffer such as an accumulator.
        """
        import sympy

        if not self._levels:
            return None
        if level_symbol is None:
            level_symbol = self._levels[-1][0]
        if level_symbol is None:
            return None

        offset = self.device_element_offset(device_size, coords)
        shifted = offset.subs(tiled_symbol, tiled_symbol + tile_extent * level_symbol)
        delta = sympy.simplify(shifted - offset)
        # The advance is a per-step constant, so evaluate it at the window
        # origin; every other free coordinate cancels or is in-tile.
        origin = {s: 0 for s in delta.free_symbols if s is not level_symbol}
        delta = sympy.simplify(delta.subs(origin))
        if delta == 0:
            return None
        return delta

    @property
    def level_symbol(self) -> sympy.Symbol | None:
        """The innermost open level's tile-advance symbol."""
        if not self._levels:
            return None
        return self._levels[-1][0]

    @property
    def loop_depth(self) -> int:
        return len(self._stack) - 1

    def redirect_output(self, value: Operand, dest: Operand) -> None:
        """Rebind every reference to ``value`` onto ``dest``.

        Plan 2's store rule. Rewriting only the producing OpSpec's *output* arg
        is not enough for a loop carry: the combine op reads the accumulator as
        an input too, so leaving those reads on the old ``lx`` buffer makes each
        iteration overwrite instead of accumulate. A per-stick probe caught
        exactly that -- only the last K-tile survived in the output.

        The reference accumulates in place in HBM (golden/ref_matmul_ktiled.py's
        combine ``add`` has arg0 and arg2 both ``hbm: 3``), so the accumulator
        must *be* the store destination rather than a buffer copied out at the
        end.
        """
        producer = self._producers.get(id(value))
        if producer is None:
            raise exc.BackendUnsupported(
                "spyre", f"store of a value with no producing OpSpec: {value.name}"
            )

        symbols = self._output_symbols[id(value)]
        old_alloc = dict(value.allocation)
        old_index = value.arg_index

        def rebind(arg: object) -> object:
            if arg.allocation != old_alloc or arg.arg_index != old_index:
                return arg
            return dest.tensor_arg(
                symbols,
                is_input=arg.is_input,
                tile_advance=arg.device_tile_advance_expr,
            )

        for spec in self._iter_op_specs(self._specs):
            spec.args = [rebind(a) for a in spec.args]

        if "lx" in value.allocation:
            self._n_lx -= 1
        elif "hbm_pool" in value.allocation:
            self._n_pool -= 1
            n = 1
            for s in value.sizes:
                n *= s
            self.pool_size -= n * value.dtype.itemsize
        if value.name:
            self.operands.pop(value.name, None)

    @staticmethod
    def _iter_op_specs(specs: list[object]) -> list[object]:
        """Every OpSpec in a spec tree, descending into LoopSpec bodies."""
        out: list[object] = []
        for spec in specs:
            body = getattr(spec, "body", None)
            if body is not None:
                out.extend(SpyreEmitter._iter_op_specs(body))
            elif hasattr(spec, "args"):
                out.append(spec)
        return out

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
