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
import itertools
from typing import TYPE_CHECKING
from typing import Any

from ... import exc
from . import layout

if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy
    import torch
    from torch_spyre._inductor.op_spec import OpSpec
    from torch_spyre._inductor.op_spec import TensorArg


_BUFFER_SEQ = itertools.count()


def _new_buffer_id() -> int:
    """A fresh buffer identity. Monotonic, so it is never reused."""
    return next(_BUFFER_SEQ)


#: ``id(TensorArg) -> Operand.buffer_id``: which buffer each emitted arg
#: came from.
#:
#: This is the **sound rewrite key** review finding (b) asked for.
#: ``redirect_output`` used to find the args to rewrite by *matching* on
#: ``(arg.allocation, arg.arg_index)``, which is not a key: the free path hands
#: ``{"lx": 0}`` straight back, so the next ``add_intermediate`` mints a second
#: ``Operand`` carrying the identical pair, and a later ``redirect_output``
#: rewrites args belonging to the wrong buffer. It worked only because stores
#: happen to arrive last-in-first-out -- an invariant neither enforced nor
#: documented.
#:
#: The provenance is known at exactly one place, ``Operand.tensor_arg``, so it
#: is recorded there. A module-level table rather than a field on ``TensorArg``
#: because that is a torch-spyre dataclass the validator and SDSC codegen both
#: read, and adding to it would be a change to their type; a table keyed on
#: ``id`` rather than on the arg itself because ``TensorArg`` is ``eq=True``,
#: so structurally identical args from *different* buffers compare equal and
#: would collapse.
#:
#: Lifetime: cleared by :meth:`SpyreEmitter.reset`, which runs per kernel, so
#: this cannot grow without bound. The ids are only ever compared against ids
#: recorded in the same emission pass, so id reuse after a gc cannot alias
#: across kernels.
_ARG_ORIGIN: dict[int, int] = {}


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
    #: **Stable identity of the underlying buffer**, independent of this
    #: ``Operand`` object. Minted once per real buffer by ``_new_buffer_id``.
    #:
    #: ``id(self)`` cannot serve as the buffer's identity, and finding that out
    #: is most of T5. The emitter reshapes operands constantly with
    #: ``dataclasses.replace`` -- ``_matmul_arg``, ``_k_advance``,
    #: ``_stage_advancing_operand``, ``_stage_reduced_tile`` and
    #: ``_reconcile_matmul_stick_axes`` all do, because a buffer registered
    #: from a tile-shaped load carries the tile's extents while the op needs
    #: the full ones. Each ``replace`` produces a *new object for the same
    #: buffer*, so an id-based index silently misses every arg emitted through
    #: one. Measured: keying on ``id(self)`` made the matmul's own output
    #: invisible to ``redirect_output`` and the GEMM driver returned **100%**
    #: error.
    #:
    #: ``dataclasses.replace`` copies this field like any other, which is
    #: exactly the property needed: a reshaped view of a buffer keeps the
    #: buffer's identity while changing its shape.
    buffer_id: int = dataclasses.field(default_factory=_new_buffer_id)

    def tensor_arg(
        self,
        symbols: list[sympy.Expr],
        *,
        is_input: bool,
        tile_advance: sympy.Expr | None = None,
        spec_name: str | None = None,
        indirect_dim: int | None = None,
        indirect_from: str | None = None,
    ) -> TensorArg:
        """Realize this buffer as a ``TensorArg`` under a given indexing.

        ``symbols[i]`` indexes host dim ``i``. The same buffer yields different
        ``device_coordinates`` in different OpSpecs because the iteration
        symbols differ, which is exactly why layout is applied here and not
        cached on the operand.

        An **extent-1 host dim is addressed at coordinate 0**, whatever symbol
        the consuming op iterates it with. That is what makes a reduced buffer
        broadcast correctly back against the full-extent input: the reference's
        softmax ``sub`` reads its ``[1, 64, 64]`` operand at ``[0, c0, 0]``
        while reading the full operand at ``[floor(c1/64), c0, Mod(c1, 64)]``
        (golden/ref_softmax.py op 3). Indexing the reduced buffer with the full
        coordinates instead validates, compiles, launches -- and is 100% wrong.
        """
        from torch_spyre._inductor.op_spec import TensorArg

        if tile_advance is not None and "lx" in self.allocation:
            # scratchpad/allocator.py:1094-1103 -- LX addresses cannot be
            # expressed as affine.apply symbols, so an LX buffer may not advance
            # per tile. torch-spyre's own residency gate returns the exclusion
            # reason "tiled (advancing)" for exactly this case.
            raise exc.BackendUnsupported(
                "spyre", "an lx buffer cannot carry a device_tile_advance_expr"
            )

        import sympy

        indexing = [
            sympy.Integer(0) if extent == 1 else sym
            for extent, sym in zip(self.sizes, symbols, strict=True)
        ]
        device_size, device_coordinates = layout.stickify(
            self.sizes, indexing, self.stick_dim, self.dtype
        )
        if indirect_dim is not None:
            # Replace the coordinate for one host dim with a runtime index read
            # from another TensorArg: IndirectAccess(name) resolves by the
            # *spec_name* of the index arg (op_spec.py's IndirectAccess
            # docstring; the pairing is done in codegen/superdsc.py:1277-1279).
            from torch_spyre._inductor.op_spec import IndirectAccess

            assert indirect_from is not None
            position = self._device_axis_of(indirect_dim)
            device_coordinates = list(device_coordinates)
            device_coordinates[position] = IndirectAccess(indirect_from)
        arg = TensorArg(
            is_input=is_input,
            arg_index=self.arg_index,
            device_dtype=layout.device_dtype(self.dtype),
            device_size=device_size,
            device_coordinates=device_coordinates,
            allocation=dict(self.allocation),
            device_tile_advance_expr=tile_advance,
            name=spec_name,
        )
        # **Record which Operand this arg came from.** ``TensorArg`` is a
        # torch-spyre dataclass with no operand-identity field, and
        # ``(allocation, arg_index)`` was the *only* link from an emitted arg
        # back to its buffer -- which is exactly why ``redirect_output``'s
        # rewrite key was unsound (review finding (b)). The provenance is known
        # here and nowhere else, so it is recorded here.
        #
        # The side table is keyed on ``id(arg)`` rather than stored on the arg
        # because ``TensorArg`` is ``eq=True``: two structurally identical args
        # compare equal, so a dict keyed by the arg *value* would collapse
        # them, and adding a field would change a type the validator and SDSC
        # codegen both read. See ``_ARG_ORIGIN``'s own comment.
        _ARG_ORIGIN[id(arg)] = self.buffer_id
        return arg

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
            self.index.tensor_arg(
                index_symbols, is_input=True, spec_name=self.spec_name
            ),
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

    @staticmethod
    def _lx_budget() -> int:
        """Concurrent LX intermediates the active config asks for.

        ``spyre_lx_buffers`` is a real autotuner knob (see
        ``SpyreBackend.backend_tunables``), so residency is searched rather than
        hard-coded. Falls back to ``_MAX_LX_BUFFERS`` when no config is active
        -- e.g. a direct emit outside a bound kernel.

        This is a **count** of buffers and remains a knob because residency is
        a performance choice. It is no longer a *correctness* ceiling: that was
        bug #15, where the count stood in for an addressing rule. The real
        hardware limit is the byte budget, enforced by :meth:`_lx_fits`.
        """
        from .backend import _MAX_LX_BUFFERS

        requested = SpyreEmitter._config_int("spyre_lx_buffers", _MAX_LX_BUFFERS)
        return max(0, requested)

    def _lx_fits(self, span: int) -> bool:
        """Does another ``span``-byte LX buffer fit, by count *and* by bytes?

        Two independent limits, and conflating them was bug #15:

        * ``_lx_budget()`` is the autotuner's residency *preference* -- how many
          intermediates it wants in LX rather than the pool;
        * ``_LX_CAPACITY_BYTES`` is the hardware's. LX is a real scratchpad with
          a real size, and overrunning it is not diagnosed anywhere in the
          stack -- it fails at launch with
          ``RAS::RUNTIMESCHEDULER::ComputeHardwareError``, the same opaque
          error that made bug #8 look like a count limit for so long.
        """
        from .backend import _LX_CAPACITY_BYTES

        if self._n_lx >= self._lx_budget():
            return False
        stride = layout.STICK_BYTES
        return self.lx_size + -(-span // stride) * stride <= _LX_CAPACITY_BYTES

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
        #: Next free **byte address** in LX. Like ``pool_size`` this only ever
        #: grows within a kernel: an LX address handed back and reused would
        #: alias a buffer another ``Operand`` still refers to (the same hazard
        #: ``redirect_output`` documents for the pool).
        self.lx_size = 0
        #: Next free **byte offset** in the HBM pool, and -- because it only
        #: ever grows -- also the pool's total size. A true high-water mark:
        #: see :meth:`redirect_output` for why it must not be decremented.
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
        # The arg-provenance table is per kernel, like everything above: an id
        # recorded for one kernel's args must never be matched against
        # another's, and clearing here is what guarantees it.
        _ARG_ORIGIN.clear()

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

    @staticmethod
    def _device_span_bytes(sizes: list[int], stick_dim: int, dtype: torch.dtype) -> int:
        """Bytes this buffer actually occupies on the device.

        Derived by calling ``layout.stickify`` rather than by re-deriving the
        padding rule, so the pool allocator and the emitted ``device_size`` can
        never disagree about a buffer's footprint -- the same reason
        ``_k_advance`` builds its advance from the operand's realized layout.
        """
        import sympy

        symbols = [sympy.Symbol(f"_span{i}") for i in range(len(sizes))]
        device_size, _ = layout.stickify(sizes, symbols, stick_dim, dtype)
        elems = 1
        for extent in device_size:
            elems *= int(extent)
        return elems * dtype.itemsize

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

        **The ``lx`` value is a byte ADDRESS, not a slot index** -- bug #15, and
        it is bug #5's mechanism in the LX path. ``{"lx": self._n_lx}`` handed
        out 0, 1, 2, ... as if LX were a slot array; SDSC reads the value as an
        address, so every buffer after the first landed inside the first one.

        ~~Measured on this device, 2 LX intermediates of 128x256 fp16 work and
        3 do not~~ -- that was true but misdiagnosed (T7.3). Re-measured
        2026-10-28: the constraint is ``k % 128 in {0, 1}``, i.e. an address
        quantised to ``layout.STICK_BYTES``. ``{"lx": 2}`` failed because 2
        *bytes* is a misaligned address, not because a third slot is missing --
        and at properly aligned, non-overlapping addresses **6 concurrent LX
        buffers run**. ``repros/lx_allocation_is_an_address.py`` has the full
        table.

        So LX is allocated exactly like the pool: a monotonically increasing
        stick-aligned offset, sized by the buffer's *device* footprint (the
        same host-vs-device distinction as bug #13). The budget that remains is
        a real byte budget -- ``_LX_TRACKER_CAPACITY_BYTES * dxp_lx_frac_avail``
        (~406 KB; scratchpad/allocator.py:209, :2081-2088) -- and anything
        beyond it spills to ``hbm_pool``, which is the residency plan 4.4
        proposed in the first place.
        """
        span = self._device_span_bytes(sizes, stick_dim, dtype)
        stride = layout.STICK_BYTES
        if residency == "lx" and not self._lx_fits(span):
            residency = "hbm_pool"
        if residency == "lx":
            allocation: dict[str, Any] = {"lx": self.lx_size}
            self._n_lx += 1
            self.lx_size += -(-span // stride) * stride
        elif residency == "hbm_pool":
            # The pool value is a **byte offset**, not a slot index: SDSC codegen
            # reads it straight into ``start_addr``
            # (codegen/superdsc.py:1683-1689). Handing out 0,1,2,3 overlaps every
            # buffer at nearly the same address -- which validates, compiles and
            # runs, and silently returns ~95% wrong results. Allocate a real
            # offset instead, stick-aligned so no buffer starts mid-stick.
            #
            # The span is the **device** footprint, not the host one. Those
            # differ whenever the stick-axis extent is not a whole number of
            # sticks: ``stickify`` pads that axis up to ``elems_per_stick``, so
            # an 8x4 fp16 buffer is 64 host bytes but ``device_size=[1, 8, 64]``
            # = 1024 device bytes, and a 128x100 one is 25600 against 32768.
            # Sizing the pool by the host span therefore under-allocates by up
            # to a stick per non-stick-axis element, and the *next* buffer's
            # offset lands inside the previous buffer's real extent -- bug #5's
            # overlap, reached by a different route. Measured on a two-store
            # 8x4 kernel: host-span accounting gave offsets 0/192/320 for
            # buffers 1024 device bytes wide (every pair overlapping), against
            # 0/1024/2048 here.
            span = self._device_span_bytes(sizes, stick_dim, dtype)
            allocation = {"hbm_pool": self.pool_size}
            self._n_pool += 1
            stride = layout.STICK_BYTES
            self.pool_size += -(-span // stride) * stride
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

        ``op_spec_validation._check_stick_restickify`` (`:705`) requires the
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
        from torch_spyre._inductor.op_spec import IndirectAccess
        from torch_spyre._inductor.op_spec import TensorArg

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

    def restage_to_pool(self, value: Operand) -> Operand:
        """Move ``value`` out of ``lx`` into the HBM pool, in place.

        ``ReStickifyOpHBM`` operates on HBM, so a value produced into ``lx``
        cannot be relayouted directly. Rather than emit a copy, rebind the
        producing OpSpec's output onto a pool buffer of the same layout --
        the same rewrite ``redirect_output`` performs for a store.
        """
        if "lx" not in value.allocation:
            return value
        staged = self.add_intermediate(
            f"{value.name or 'buf'}_staged",
            list(value.sizes),
            value.dtype,
            value.stick_dim,
            residency="hbm_pool",
        )
        self.redirect_output(value, staged)
        return staged

    def emit_restickify_into(
        self, source: Operand, dest: Operand, symbols: list[sympy.Expr]
    ) -> None:
        """Relayout ``source`` directly into an existing buffer ``dest``.

        The closing counterpart to :meth:`emit_restickify`, which allocates a
        fresh intermediate. Here the destination already exists (it is the
        kernel's output), so the relayout writes it in place.

        ``dest`` must be HBM-resident: an otherwise identical relayout writing
        an ``lx`` destination passes validation, compiles, launches and returns
        wrong data (measured max|d| ~1.0 on a [0,1) input versus fp16 rounding
        error for HBM), so a non-HBM destination fails loudly here.
        """
        from torch_spyre._inductor.constants import RESTICKIFY_OP

        if source.stick_dim == dest.stick_dim:
            raise exc.BackendUnsupported(
                "spyre", "closing restickify between identical stick axes"
            )
        if "lx" in dest.allocation:
            raise exc.BackendUnsupported(
                "spyre", "ReStickifyOpHBM cannot write an lx destination"
            )
        # A relayout needs a real work division: with every dim unsplit the
        # backend scheduler rejects the program with "Expect valid lower and
        # upper bound parameters". The captured reference splits its leading
        # non-stick dim (golden/ref_restickify.py: c0 over 2 cores), so split
        # the leading dim that is not the destination's stick axis.
        #
        # **That dim is the *source's* stick axis**, and the whole-sticks rule
        # therefore binds it -- measured, not assumed: at a 2-stick source axis
        # (128 fp16) splits 1 and 2 compile and run while 4 and 8 fail in the
        # backend scheduler with "There must be at least one valid candidate.";
        # at a 4-stick axis (256 fp16) splits 2 and 4 run and 8 fails. So the
        # bound is the source's stick count, exactly as for a compute op's
        # stick axis. The frozen ``(2, 4, 1)`` never hit this because 2 is
        # legal for every source wider than one stick.
        splits = [1] * len(dest.sizes)
        for dim in range(len(dest.sizes)):
            if dim == dest.stick_dim:
                continue
            splits[dim] = self._relayout_split(
                dest.sizes[dim],
                max_sticks=self._stick_count(source.sizes[dim], source.dtype),
            )
            break

        import sympy

        iteration_space = {
            sym: (sympy.Integer(size), split)
            for sym, size, split in zip(symbols, dest.sizes, splits, strict=True)
        }
        self.emit(
            self.op_spec(
                RESTICKIFY_OP,
                is_reduction=False,
                iteration_space=iteration_space,
                args=[
                    source.tensor_arg(symbols, is_input=True),
                    dest.tensor_arg(symbols, is_input=False),
                ],
            )
        )

    @staticmethod
    def _stick_count(extent: int, dtype: torch.dtype) -> int:
        """Whole sticks an extent of ``extent`` elements occupies."""
        return max(1, -(-extent // layout.elems_per_stick(dtype)))

    @staticmethod
    def _relayout_split(extent: int, *, max_sticks: int) -> int:
        """Cores a relayout splits its leading non-stick dim over.

        **This was the frozen literal ``(2, 4, 1)``** -- "the reference uses a
        small split", first of the three that divides the extent. That made a
        work-division decision neither user-written nor autotuned, which is the
        governing invariant (§1: work division is Helion's to choose *and to be
        asked about*) bent in the way the assessment names as "the clearest
        next hardening target" (summary deviation 3). It is also what bug #5
        teaches: a decision hard-coded hides bugs in the path it never takes,
        and #5 was found only once a frozen constant became a knob.

        It now follows ``spyre_relayout_split`` (see
        ``SpyreBackend.backend_tunables``), defaulting to 2 so an unspecified
        config still gets the old behaviour.

        ``max_sticks`` is the hard ceiling: the split dim is the *source's*
        stick axis, so splitting it past its stick count splits a stick. The
        caller measures it from the source operand. Exceeding it is not a
        performance choice -- the backend scheduler rejects the program with
        "There must be at least one valid candidate." -- so it is clamped here
        rather than offered.

        A requested split that does not divide the extent cannot be honoured
        either (Spyre work division is exact), so the largest divisor at or
        below the ceiling is used. Clamping rather than raising is right for
        this one: unlike the stick-axis bound moved into
        ``autotune_config_is_viable`` in T3, the relayout is emitted *inside*
        the store path for a dim no config names, so there is no candidate the
        autotuner could have screened instead. The knob is a preference over a
        legal range; the range itself is hardware.
        """
        from .backend import _sencores

        requested = SpyreEmitter._config_int("spyre_relayout_split", 2)
        budget = min(requested, _sencores(), max_sticks)
        for candidate in range(max(1, budget), 0, -1):
            if extent % candidate == 0:
                return candidate
        return 1

    @staticmethod
    def _config_int(key: str, default: int) -> int:
        """One backend tunable's value from the active config, or ``default``.

        The ``DeviceFunction.current()`` lookup raises ``NoCurrentFunction``
        when there is no bound kernel -- a direct emit, or a call from the
        autotuner's search loop -- and that is the *only* condition this
        swallows. A bare ``except Exception`` here used to return the default
        for any failure at all, which is precisely the pattern §5b records
        already costing a bug: a defensive ``try/except`` in
        ``autotune_config_is_viable`` "silently accepted every config". For
        ``spyre_lx_buffers`` that default is ``_MAX_LX_BUFFERS`` -- *the one
        value the knob exists to stop the backend assuming*.
        """
        from ..device_function import DeviceFunction
        from ..host_function import NoCurrentFunction

        try:
            current = DeviceFunction.current()
        except NoCurrentFunction:
            return default
        value = current.config.config.get(key)
        return default if value is None else int(value)

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
        (spyre_kernel.py:588-610, :1072) -- that shared identity is what lets
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
            # op_spec_validation.py:220-239 rejects an empty body; fail here
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

        Minted per (op, level), never shared between ops: spyre_kernel.py:588-610
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

    def tiling_for(self, op_key: str) -> tuple[list[list[Any]], dict[Any, int]] | None:
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
        (spyre_kernel.py:666-735), and it is why a matmul's two inputs get
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

        **The rewrite is keyed on operand identity, not on a value match**
        (review finding (b)). It used to select args by
        ``arg.allocation == old_alloc and arg.arg_index == old_index``, which is
        not a key at all: the free path below hands ``{"lx": 0}`` straight back,
        so the next ``add_intermediate`` mints a *second* ``Operand`` carrying
        the identical pair, and a later ``redirect_output`` then rewrites args
        belonging to the wrong buffer. That it worked relied on stores arriving
        last-in-first-out -- an invariant neither enforced nor documented
        anywhere, which is precisely the kind of load-bearing accident this
        stack punishes.

        ``_ARG_ORIGIN`` records the producing ``Operand`` for every emitted arg
        at the one place that knows it (``Operand.tensor_arg``), so this walks
        a real index. Note what that does *not* change: the retroactive rewrite
        itself stays. SuperDSC has no store op, so a store must reach back
        (opspec plan §2 sanctions this); removing it would mean deferring
        OpSpec construction until the store is seen, which is a larger redesign
        and is not justified by anything observed.
        """
        producer = self._producers.get(id(value))
        if producer is None:
            raise exc.BackendUnsupported(
                "spyre", f"store of a value with no producing OpSpec: {value.name}"
            )

        symbols = self._output_symbols[id(value)]
        origin = value.buffer_id

        def rebind(arg: object) -> object:
            if _ARG_ORIGIN.get(id(arg)) != origin:
                return arg
            replacement = dest.tensor_arg(
                symbols,
                is_input=arg.is_input,
                tile_advance=arg.device_tile_advance_expr,
            )
            # Preserve a spec_name: it is how an IndirectAccess coordinate in
            # another arg resolves to this one (superdsc.py:1277-1279), so
            # dropping it on a rewrite would break a gather whose staged table
            # is later redirected.
            if getattr(arg, "name", None) is not None:
                replacement.name = arg.name
            return replacement

        for spec in self._iter_op_specs(self._specs):
            spec.args = [rebind(a) for a in spec.args]

        if "lx" in value.allocation:
            # The *count* is returned so a later intermediate can still prefer
            # LX, but ``lx_size`` is deliberately **not** rewound -- for the
            # same reason ``pool_size`` is not (see below). An LX address
            # handed back and reused would alias a buffer that another
            # ``Operand`` still carries in its ``allocation``, which is review
            # finding (b)'s hazard; and since bug #15 these are real addresses,
            # the aliasing would be a wrong answer rather than a harmless
            # index collision.
            self._n_lx -= 1
        elif "hbm_pool" in value.allocation:
            self._n_pool -= 1
            # ``pool_size`` is deliberately **not** decremented. It was, and
            # the arithmetic was asymmetric: allocation rounds up to a whole
            # stick (``self.pool_size += -(-span // stride) * stride``) while
            # the free subtracted the raw byte span. Any redirect therefore
            # left ``pool_size`` *below* the true high-water mark, and the next
            # ``allocation = {"hbm_pool": self.pool_size}`` handed out an offset
            # that was no longer stick-aligned -- the exact condition
            # ``layout.STICK_BYTES`` exists to prevent.
            #
            # Measured: a 200-byte span allocates 256 bytes (2 sticks) at
            # offset 0, the free leaves ``pool_size`` at **56**, and the next
            # buffer starts at byte 56 -- mid-stick, with 56 % 128 == 56.
            # Nothing in the stack catches that. §3 rates G1's evidential
            # weight as "nearly none" for this, and bug #5 is the precedent:
            # overlapping ``hbm_pool`` buffers validated, bundled, compiled,
            # launched, and returned ~95% wrong output.
            #
            # Rounding the free symmetrically would fix the alignment but keep
            # a subtler hazard: the slot is handed back while
            # ``redirect_output`` has only *rewritten* the args that referred
            # to it, so a later allocation can reuse an offset another
            # ``Operand`` still carries in ``allocation`` (review finding (b),
            # T5). Not reclaiming at all removes both at the cost of some dead
            # pool bytes, which is the cheap direction: the pool is sized once
            # per kernel and nothing counts pool slots.
        if value.name:
            self.operands.pop(value.name, None)

    @staticmethod
    def _iter_op_specs(specs: Sequence[object]) -> list[OpSpec]:
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
        args: Sequence[TensorArg],
        op_info: dict[str, Any] | None = None,
        tiled_symbols: list[list[sympy.Symbol]] | None = None,
        tiled_symbol_trip_counts: dict[sympy.Symbol, int] | None = None,
        output: Operand | None = None,
        output_symbols: list[sympy.Expr] | None = None,
    ) -> OpSpec:
        """Build an ``OpSpec``, deriving the core mapping from the given space.

        ``core_id_to_work_slice`` is derived with torch-spyre's own
        ``derive_operation_mapping``, never hand-rolled: superdsc.py:2131-2132
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
                tiled_symbol_trip_counts if tiled_symbol_trip_counts is not None else {}
            ),
        )
        if output is not None:
            assert output_symbols is not None
            self._producers[id(output)] = (spec, len(args) - 1)
            self._output_symbols[id(output)] = list(output_symbols)
        return spec
