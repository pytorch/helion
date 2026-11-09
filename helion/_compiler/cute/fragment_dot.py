"""Contractions of a computed fragment: staged ``hl.dot`` and the tf32 ``mma.sync`` path.

``FragmentDotEmitter`` lowers a dot over staged operands either as a per-element
serial contraction or, under ``cute_fp32_dot``, as a tf32 ``mma.sync`` phase whose
lanes may write the result straight to the host store (``DirectStoreSink``).  It
builds on the memory emitter for that direct store."""

from __future__ import annotations

import ast
import math
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import memory_ops
from ..ast_extension import statement_from_string
from ..inductor_lowering import PointwiseLowering
from ..inductor_lowering import ReductionLowering
from ..matmul_utils import _compute_out_dtype
from ..matmul_utils import _needs_f32_accumulator
from ..runtime_tensor_guards import runtime_tensors_are_proven_disjoint
from .fragment_context import DirectStoreSink
from .fragment_context import Fragment
from .fragment_context import HostTensor
from .fragment_memory import FragmentMemoryEmitter

# ``mma.sync`` helpers of the tf32 fragment contraction (``cute_fp32_dot``).
_FRAGMENT_MMA_RUNTIME = "_helion_fragment_mma"


_FRAGMENT_MMA_PREAMBLE = (
    f"import helion._compiler.cute._mma_dag_runtime as {_FRAGMENT_MMA_RUNTIME}\n"
)


def _mma_operand_conflict_free(layout: tuple[int, int], role: str) -> bool:
    """Whether one ``mma.sync.m16n8k8`` operand word read from a staged tile
    of *layout* ``(row bytes, element bytes)`` is bank-conflict-free (lanes
    that share a bank share the word).  Lane ``(g, c) = (lane // 4, lane %
    4)`` reads the A word at ``(g, c)`` (the other words shift rows or k by
    a multiple of 4) and the B word at ``(c, g)``."""
    row_bytes, itemsize = layout
    if row_bytes % 4:
        return False
    banks: dict[int, set[int]] = {}
    for lane in range(32):
        g, c = lane // 4, lane % 4
        row, col = (g, c) if role == "a" else (c, g)
        word = (row * row_bytes + col * itemsize) // 4
        banks.setdefault(word % 32, set()).add(word)
    return all(len(words) == 1 for words in banks.values())


def _mma_row_pad(extent: int, unit: int) -> int:
    """Elements appended to a staged mma operand row so the row stride is an
    odd multiple of *unit* words modulo the 32 shared-memory banks.

    A lane fragment of ``mma.sync.m16n8k8`` reads 8 rows x 4 consecutive
    columns (A, ``unit=4``) or 4 rows x 8 consecutive columns (B, ``unit=8``):
    with such a stride the 32 addresses of one instruction fall in 32 distinct
    banks, where a stride that is a multiple of 32 serializes them 8-fold.
    """
    return min((residue - extent) % 32 for residue in range(unit, 32, 2 * unit))


def _consumed_by_direct_store(_coords: tuple[str, ...]) -> str:
    raise AssertionError("a dot result stored by its lanes has no other reader")


class FragmentDotEmitter(FragmentMemoryEmitter):
    """``hl.dot`` over staged fragments, including the tf32 mma.sync path."""

    def materialize_padded(self, value: Fragment, pad: int) -> Fragment:
        """Copy *value* into a fresh shared buffer whose rows are *pad*
        elements longer than the tile's.  The returned fragment keeps the
        tile's shape and reads the padded rows; it lives only as a contraction
        operand (``emit_tf32_mma``), never as a result."""
        shape = value.shape
        padded_shape = (*shape[:-1], shape[-1] + pad)
        self.held.append(value)
        padded = self.allocate(Fragment(padded_shape, value.dtype, lambda _: "0"))
        view = Fragment(
            shape, value.dtype, padded.read, True, (padded,), padded.storage
        )
        self.copy(value, view)
        self.held.pop()
        return view

    def dot(self, node: Node, values: dict[Node, object]) -> Fragment:
        lhs, rhs = (values[cast("Node", x)] for x in node.args[:2])
        assert isinstance(lhs, Fragment) and isinstance(rhs, Fragment)
        acc = values[node.args[2]] if isinstance(node.args[2], Node) else None
        assert acc is None or isinstance(acc, Fragment)
        out = cast("torch.Tensor", node.meta["val"])
        shape = self.shape(out.shape)
        out_dtype = cast("torch.dtype | None", node.args[3]) or _compute_out_dtype(
            lhs.dtype, rhs.dtype, acc.dtype if acc is not None else None
        )
        # A narrow float accumulator still sums in fp32 and rounds once, as
        # the other hl.dot lowerings do.
        compute_dtype = (
            torch.float32
            if _needs_f32_accumulator(lhs.dtype, rhs.dtype)
            or (out_dtype.is_floating_point and out_dtype.itemsize < 4)
            else out_dtype
        )
        left_fake = cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])
        mma_contraction = self.tf32_mma_contraction(
            node, lhs, rhs, acc, shape, left_fake, compute_dtype
        )
        # Operand recipes must finish before the contraction begins: the same
        # input can have different coordinates in its two operand roles.
        if mma_contraction is None:
            lhs = self.materialize(lhs)
            self.held.append(lhs)
            rhs = self.materialize(rhs)
            self.held.append(rhs)
        else:
            # An operand every lane fragment reads exactly once (whole tiles,
            # one tile along the other output axis) whose recipe rests on
            # staged data is read in place by the mma lanes: no copy, no
            # barrier, no second buffer; its leaves' padded rows keep the
            # reads conflict-free.  Otherwise rows are padded to an odd
            # multiple of 4 (A) / 8 (B) banks so the lane fragments of one
            # mma read 32 distinct banks.
            whole = (
                shape[-2] % 16 == 0
                and shape[-1] % 8 == 0
                and mma_contraction % 8 == 0
                and mma_contraction == lhs.shape[-1]
            )

            # A recipe that reads no memory at all (an ``hl.arange`` mask, a
            # constant) is evaluated in place however often it is re-read: a
            # compare or a literal per operand word costs less than a staging
            # phase, its barrier and its buffer.  So is a bare cast / view of
            # one staged tile whose rows already spread the lane reads over
            # the banks: the padded copy would only repeat the tile.
            def in_place(operand: Fragment, role: str, read_once: bool) -> bool:
                if self.pure_recipe(operand):
                    return True
                if not self.staged_recipe(operand):
                    return False
                if whole and read_once:
                    return True
                return (
                    operand.cheap
                    and operand.physical_row is not None
                    and _mma_operand_conflict_free(operand.physical_row, role)
                )

            if not in_place(lhs, "a", shape[-1] <= 8):
                lhs = self.materialize_padded(lhs, _mma_row_pad(lhs.shape[-1], 4))
            self.held.append(lhs)
            if not in_place(rhs, "b", shape[-2] <= 16):
                rhs = self.materialize_padded(rhs, _mma_row_pad(rhs.shape[-1], 8))
            self.held.append(rhs)
        if mma_contraction is not None:
            result = self.emit_tf32_mma(
                lhs,
                rhs,
                acc,
                shape,
                mma_contraction,
                out_dtype,
                sink=self.direct_store_sink(node, values, shape),
            )
            self.held.pop()
            self.held.pop()
            return result
        result = self.allocate(Fragment(shape, out_dtype, lambda _: "0"))

        contraction = self.logical_axis_extent(left_fake, -1, lhs.shape[-1])

        def contract(coords: tuple[str, ...]) -> None:
            total = self.df.new_var("fragment_dot")
            k = self.df.new_var("fragment_k")
            initial = acc.broadcast(coords) if acc is not None else "0"
            self.emit(f"{total} = {self.cast(initial, compute_dtype)}")
            # All operands are in shared memory, so this serial per-output
            # contraction has ordinary IEEE FP32 multiplication and addition.
            left = lhs.broadcast((*coords[:-2], coords[-2], k))
            right = rhs.broadcast((*coords[:-2], k, coords[-1]))
            self.emit(
                f"for {k} in range({contraction}):\n    {total} = {total} + {self.cast(left, compute_dtype)} * {self.cast(right, compute_dtype)}"
            )
            self.emit(f"{result.read(coords)} = {self.cast(total, out_dtype)}")

        self.elements(shape, contract)
        self.held.pop()
        self.held.pop()
        return result

    @staticmethod
    def pure_recipe(value: Fragment) -> bool:
        """Whether *value*'s recipe reads no memory: every leaf is a pure
        coordinate function (``hl.arange``, a tile index, a constant) or a
        scalar, so a lane evaluates it in a few ALU instructions and it needs
        no staging phase, no barrier and no buffer."""
        pending = [value]
        seen: set[int] = set()
        while pending:
            fragment = pending.pop()
            if id(fragment) in seen:
                continue
            seen.add(id(fragment))
            if fragment.resident or fragment.snapshot is not None:
                return False
            if not fragment.dependencies:
                if fragment.shape and not fragment.pure:
                    return False
                continue
            pending.extend(fragment.dependencies)
        return True

    @staticmethod
    def staged_recipe(value: Fragment) -> bool:
        """Whether every leaf of *value*'s recipe is resident (a staged
        buffer or a scalar) or pure (an ``hl.arange`` row mask, a tile
        index), so any lane may evaluate it at any coordinate."""
        pending = [value]
        seen: set[int] = set()
        while pending:
            fragment = pending.pop()
            if id(fragment) in seen:
                continue
            seen.add(id(fragment))
            if fragment.resident:
                if fragment.snapshot is not None:
                    return False
                continue
            if not fragment.dependencies and fragment.shape and not fragment.pure:
                return False
            pending.extend(fragment.dependencies)
        return True

    def static_contraction(self, fake: torch.Tensor, capacity: int) -> int | None:
        """The contraction length when it is a compile-time constant.

        ``logical_axis_extent`` intersects the staged capacity with the live
        bound of a tiled axis: a per-CTA boundary (``offsets``) is dynamic and
        declines, a static reduction extent clips the capacity.
        """
        bid = self.env.resolve_block_id(fake.shape[-1])
        if bid is None:
            return capacity
        if bid in self.offsets:
            return None
        if self.env.block_sizes[bid].reduction:
            extent = self.static_extent(self.env.block_sizes[bid].numel)
            return None if extent is None else min(capacity, extent)
        return capacity

    def tf32_mma_contraction(
        self,
        node: Node,
        lhs: Fragment,
        rhs: Fragment,
        acc: Fragment | None,
        shape: tuple[int, ...],
        left_fake: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> int | None:
        """The static K of a dot ``cute_fp32_dot="tf32_mma"`` lowers through
        ``mma.sync``; None keeps the serial FMA contraction (knob off, half or
        wide operands, a dynamic contraction bound, a one-warp-short CTA)."""
        from .autotune.heuristics.cute_fp32_dot import fragment_tf32_dot_supported

        if self.df.config.get("cute_fp32_dot", "simt") != "tf32_mma":
            return None
        if (
            compute_dtype != torch.float32
            or self.threads % 32 != 0
            or not fragment_tf32_dot_supported(node, self.env)
            or (acc is not None and acc.dtype != torch.float32)
            or len(lhs.shape) < 2
            or len(rhs.shape) < 2
            or len(shape) < 2
        ):
            return None
        contraction = self.static_contraction(left_fake, lhs.shape[-1])
        if contraction is None or contraction <= 0:
            return None
        return contraction

    def host_storages_disjoint(self, left: torch.Tensor, right: torch.Tensor) -> bool:
        """Whether two host tensors are proved to view disjoint storage: a
        different fake storage and either one freshly allocated by a wrapper
        factory on this call (an in-kernel ``torch.empty``), or the bound
        dispatch key's alias matrix.  Two inputs with different fake storages
        alone prove nothing: a caller may pass overlapping views."""
        if left.untyped_storage() == right.untyped_storage():
            return False
        if self.env.tensor_storage_is_compiler_allocated(
            left
        ) or self.env.tensor_storage_is_compiler_allocated(right):
            return True
        return runtime_tensors_are_proven_disjoint(self.env, left, right)

    def direct_store_in_order(self, node: Node, user: Node) -> bool:
        """Whether the lanes may write ``user``'s window at ``node``'s position.

        The direct store moves the global write from the store node up to the
        dot, so every node between them in graph order must be free of host
        memory effects on the window: lazy recipes, views, coordinate
        functions, scalars, pointwise and reduction evaluation over
        fragments, another dot, a scan or a local allocation pass; a host
        load, store or atomic passes only when its tensor is proved disjoint
        from the window; a nested loop or branch, inline assembly, a gather
        or selection and any other opaque lowering keep the staged tile."""
        target_node = user.args[0]
        assert isinstance(target_node, Node)
        window = target_node.meta["val"]
        assert isinstance(window, torch.Tensor)
        current = node.next
        while current is not user:
            if current.op != "call_function":
                return False
            target = current.target
            if (
                target in (memory_ops.load, memory_ops.store)
                or target in atomic_ops.ATOMIC_OPS
            ):
                other = current.args[0]
                if (
                    not isinstance(other, Node)
                    or other.target is not _tracing_ops._host_tensor
                ):
                    return False
                fake = other.meta["val"]
                if not isinstance(
                    fake, torch.Tensor
                ) or not self.host_storages_disjoint(window, fake):
                    return False
            elif target not in self._HOST_MEMORY_FREE_TARGETS and not isinstance(
                current.meta.get("lowering"), (PointwiseLowering, ReductionLowering)
            ):
                return False
            current = current.next
        return True

    def direct_store_sink(
        self, node: Node, values: dict[Node, object], shape: tuple[int, ...]
    ) -> DirectStoreSink | None:
        """The host store a tf32 mma may write from its lanes: the dot's only
        user, storing the dot itself (no cast or view between them) into a
        host tensor window of the dot's shape, in the same graph, with every
        other input of the store already lowered and no host memory effect
        on the window between the dot and the store in graph order
        (``direct_store_in_order``).  Anything else -- another reader, an
        extra mask, a region or producer-cache walk, a warp-result or
        register-owned chain, a recorder -- keeps the staged result."""
        if (
            len(node.users) != 1
            or self.active_region is not None
            or node in self.cache_nodes
            or self.warp_producer_recorder is not None
        ):
            return None
        (user,) = node.users
        if (
            user.op != "call_function"
            or user.target is not memory_ops.store
            or user.graph is not node.graph
            or user.args[2] is not node
            or user.args[3] is not None
            or user in self.direct_stores
            or user in self.warp_result_nodes
            or user in self.local_register_nodes
            or not self.direct_store_in_order(node, user)
        ):
            return None
        for source in user.all_input_nodes:
            if source is node or source in values:
                continue
            # The store's host tensor (and a size symbol) may follow the dot
            # in graph order; both lower to a value without emitting anything.
            if source.target is _tracing_ops._host_tensor:
                fake = source.meta["val"]
                assert isinstance(fake, torch.Tensor)
                values[source] = HostTensor(
                    self.df.tensor_arg(
                        fake, prefer_name=cast("str", source.args[0])
                    ).name,
                    fake,
                )
            elif source.target in (
                _tracing_ops._get_symnode,
                torch.ops.aten.sym_size.int,
            ):
                values[source] = source.meta.get("val")
            else:
                return None
        if not isinstance(values.get(cast("Node", user.args[0])), HostTensor):
            return None
        sink = DirectStoreSink(user)
        self.memory(user, values, True, direct=sink)
        if sink.address is None or sink.shape != shape:
            return None
        return sink

    def emit_direct_lane_stores(
        self,
        sink: DirectStoreSink,
        batch: tuple[str, ...],
        slots: tuple[tuple[str, str], ...],
        accs: list[str],
        rows: int,
        cols: int,
        out_dtype: torch.dtype,
    ) -> None:
        """Write one lane's four ``mma.sync`` results through the sink's
        window: the two adjacent columns of a row leave as one aligned vector
        store when the window is a proved packet walk, else as two scalar
        stores.  Each value takes the dot's result dtype and then the
        tensor's, exactly as the staged tile and its store did."""
        assert sink.address is not None and sink.dtype is not None
        itemsize = sink.dtype.itemsize
        scalar = self.dtype(sink.dtype)
        paired = sink.lanes is not None and sink.lanes % 2 == 0 and cols % 2 == 0

        def value(name: str) -> str:
            return self.cast(self.cast(name, out_dtype), sink.dtype)

        for first in (0, 2):
            (r, c0), (_, c1) = slots[first], slots[first + 1]
            if not paired:
                for name, c in zip(accs[first : first + 2], (c0, c1), strict=True):
                    pointer, mask = sink.address((*batch, r, c))
                    guard = self.predicate(
                        [mask, *self.in_bounds(r, rows, 16), *self.in_bounds(c, cols)]
                    )
                    self.emit(f"if {guard}:\n    {pointer}.store({value(name)})")
                continue
            # ``c0`` is even and the packet proof makes every even column of
            # the window ``2 * itemsize``-aligned; ``cols`` is even, so the
            # pair is in bounds together.
            pointer, mask = sink.address((*batch, r, c0))
            guard = self.predicate(
                [mask, *self.in_bounds(r, rows, 16), *self.in_bounds(c0, cols)]
            )
            packet = self.df.new_var("fragment_mma_packet")
            self.emit(
                f"{packet} = cute.make_rmem_tensor(cute.make_layout((2,)), {scalar})"
            )
            self.emit(f"{packet}[0] = {value(accs[first])}")
            self.emit(f"{packet}[1] = {value(accs[first + 1])}")
            atom = (
                "cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), "
                f"{scalar}, num_bits_per_copy={16 * itemsize})"
            )
            self.emit(
                f"if {guard}:\n"
                f"    cute.copy({atom}, {packet}, cute.make_tensor({pointer}.align({2 * itemsize}), cute.make_layout((2,))))"
            )

    def emit_tf32_mma(
        self,
        lhs: Fragment,
        rhs: Fragment,
        acc: Fragment | None,
        shape: tuple[int, ...],
        contraction: int,
        out_dtype: torch.dtype,
        sink: DirectStoreSink | None = None,
    ) -> Fragment:
        """Contract through ``mma.sync.aligned.m16n8k8`` tf32 (``cute_fp32_dot``).

        Each warp owns whole 16 x 8 output tiles of the (batched) product and
        walks K in steps of 8: lane ``(g, c) = (lane // 4, lane % 4)`` reads
        A rows ``g``/``g + 8`` at k ``c``/``c + 4`` and B column ``g`` at the
        same k straight from the staged fp32 operand buffers (six smem reads
        per 1024 MACs instead of two per MAC; the padded rows keep the 32
        reads of one instruction in 32 banks), feeds their bit patterns to the
        tensor core, which truncates them to tf32, and writes its four fp32
        results ``(g, 2c)``, ``(g, 2c + 1)``, ``(g + 8, 2c)``, ``(g + 8, 2c + 1)``.
        Partial tiles (M % 16, N % 8, K % 8) read zeros past the operand and
        skip the writes past the result, so every qualifying dot of the root
        takes the mode.

        ``cute_fragment_split_k``: when the dot has fewer tiles than the CTA
        has warps, the idle warps take a share of K: ``splits`` warps walk
        disjoint step ranges of one tile (the accumulator joins split 0) and
        write fp32 partials; the result reads their sum.  Returns the result
        fragment (a staged tile, or the partial-summing view).

        With a *sink* (the dot's only reader is a host store of this shape)
        and no K split, the lanes write their results straight into the
        host window: no staged tile, no barrier before the store walk and no
        barrier after it.  The store node then lowers to nothing and the
        returned fragment must never be read.
        """
        rows, cols = shape[-2], shape[-1]
        batch_shape = shape[:-2]
        tiles_m = -(-rows // 16)
        tiles_n = -(-cols // 8)
        tiles = math.prod(batch_shape) * tiles_m * tiles_n
        warps = self.threads // 32
        steps = -(-contraction // 8)
        splits = 1
        if self.split_k and tiles < warps and steps > 1:
            splits = max(
                d for d in range(1, min(warps // tiles, steps) + 1) if steps % d == 0
            )
        direct = sink is not None and splits == 1
        # The lanes read staged operands: batched loads must be published.
        self.flush_pending_barrier()
        self.emit_fragment_mma_preamble()
        rt = _FRAGMENT_MMA_RUNTIME
        if splits > 1:
            partials = self.allocate(
                Fragment((splits, *shape), torch.float32, lambda _: "0")
            )
            result = partials
        elif direct:
            result = Fragment(shape, out_dtype, _consumed_by_direct_store)
        else:
            result = self.allocate(Fragment(shape, out_dtype, lambda _: "0"))
        # ``cute_fragment_mma_rows``: a warp owns whole 16-row tile rows of
        # the product and evaluates each row's A words once per K step for
        # all of the row's column tiles (the words do not depend on the
        # column).  The same mma sequence per output tile: bitwise identical
        # to the tile walk.  A K split keeps the (tile, split) unit walk.
        row_walk = self.mma_rows and splits == 1
        tile = self.df.new_var("fragment_mma_tile")
        # Without a split the warp loop walks tiles directly (the established
        # render); with one it walks (tile, split) units.
        unit = self.df.new_var("fragment_mma_unit") if splits > 1 else tile
        split = self.df.new_var("fragment_mma_split") if splits > 1 else ""
        lane = self.df.new_var("fragment_mma_lane")
        group = self.df.new_var("fragment_mma_group")
        quad = self.df.new_var("fragment_mma_quad")
        row = self.df.new_var("fragment_mma_row")
        col = self.df.new_var("fragment_mma_col")
        self.emit(f"{lane} = {self.thread} % 32")
        self.emit(f"{group} = {lane} // 4")
        self.emit(f"{quad} = {lane} % 4")
        f32 = self.dtype(torch.float32)
        row8 = f"{row} + 8"
        # Result slots: (row, 2 quad), (row, 2 quad + 1) and the same of
        # row + 8; a slot's column is relative to the tile's first column.
        slots = (
            (row, f"2 * {quad}"),
            (row, f"2 * {quad} + 1"),
            (row8, f"2 * {quad}"),
            (row8, f"2 * {quad} + 1"),
        )

        def init_accumulators(
            accs: list[str], batch: tuple[str, ...], col_base: str
        ) -> None:
            for name, (r, c) in zip(accs, slots, strict=True):
                c_abs = f"{col_base} + {c}"
                if acc is None:
                    self.emit(f"{name} = {f32}(0)")
                    continue
                self.guarded_assign(
                    name,
                    self.cast(acc.broadcast((*batch, r, c_abs)), torch.float32),
                    f"{f32}(0)",
                    self.in_bounds(r, rows, 16)
                    + self.in_bounds(c_abs, cols)
                    + ([f"{split} == 0"] if splits > 1 else []),
                )

        def operand(
            fragment: Fragment,
            coords: tuple[str, str],
            bounds: list[str],
            batch: tuple[str, ...],
        ) -> str:
            name = self.df.new_var("fragment_mma_operand")
            self.guarded_assign(
                name,
                self.cast(fragment.broadcast((*batch, *coords)), torch.float32),
                f"{f32}(0)",
                bounds,
            )
            return f"{rt}.f32_bits({name})"

        def a_words(k0: str, k1: str, batch: tuple[str, ...]) -> list[str]:
            return [
                operand(
                    lhs,
                    (r, k),
                    self.in_bounds(r, rows, 16) + self.in_bounds(k, contraction),
                    batch,
                )
                for r, k in ((row, k0), (row8, k0), (row, k1), (row8, k1))
            ]

        def b_words(k0: str, k1: str, batch: tuple[str, ...]) -> list[str]:
            return [
                operand(
                    rhs,
                    (k, col),
                    self.in_bounds(k, contraction) + self.in_bounds(col, cols),
                    batch,
                )
                for k in (k0, k1)
            ]

        def emit_stores(accs: list[str], batch: tuple[str, ...], col_base: str) -> None:
            if direct:
                assert sink is not None
                self.emit_direct_lane_stores(
                    sink,
                    batch,
                    tuple((r, f"{col_base} + {c}") for r, c in slots),
                    accs,
                    rows,
                    cols,
                    out_dtype,
                )
                return
            for name, (r, c) in zip(accs, slots, strict=True):
                c_abs = f"{col_base} + {c}"
                if splits > 1:
                    store = f"{result.read((split, *batch, r, c_abs))} = {name}"
                else:
                    store = f"{result.read((*batch, r, c_abs))} = {self.cast(name, out_dtype)}"
                bounds = self.in_bounds(r, rows, 16) + self.in_bounds(c_abs, cols)
                if bounds:
                    store = f"if {' and '.join(bounds)}:\n    {store}"
                self.emit(store)

        if row_walk:
            tile_rows = math.prod(batch_shape) * tiles_m
            tile_row = self.df.new_var("fragment_mma_tile_row")
            col_tile = self.df.new_var("fragment_mma_col_tile")
            loop = statement_from_string(
                f"for {tile_row} in range({self.thread} // 32, {tile_rows}, {warps}):\n    pass"
            )
            assert isinstance(loop, ast.For)
            loop.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                batch = self.coordinate_locals(
                    self.coordinates(f"({tile_row}) // {tiles_m}", batch_shape)
                )
                self.emit(f"{row} = ({tile_row}) % {tiles_m} * 16 + {group}")
                ks = [
                    (f"{index * 8} + {quad}", f"{index * 8 + 4} + {quad}")
                    for index in range(steps)
                ]
                hoisted = [a_words(k0, k1, batch) for k0, k1 in ks]
                n_loop = statement_from_string(
                    f"for {col_tile} in range({tiles_n}):\n    pass"
                )
                assert isinstance(n_loop, ast.For)
                n_loop.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", n_loop.body)):
                    col_base = f"({col_tile}) * 8"
                    self.emit(f"{col} = {col_base} + {group}")
                    accs = [self.df.new_var("fragment_mma_c") for _ in slots]
                    init_accumulators(accs, batch, col_base)
                    for (k0, k1), words in zip(ks, hoisted, strict=True):
                        words = [*words, *b_words(k0, k1, batch)]
                        self.emit(
                            f"{', '.join(accs)} = {rt}.mma_tf32({', '.join([*words, *accs])})"
                        )
                    emit_stores(accs, batch, col_base)
                self.cg.add_statement(n_loop)
            self.cg.add_statement(loop)
        else:
            loop = statement_from_string(
                f"for {unit} in range({self.thread} // 32, {tiles * splits}, {warps}):\n    pass"
            )
            assert isinstance(loop, ast.For)
            loop.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                if splits > 1:
                    self.emit(f"{tile} = {unit} // {splits}")
                    self.emit(f"{split} = {unit} % {splits}")
                batch = self.coordinate_locals(
                    self.coordinates(f"({tile}) // {tiles_m * tiles_n}", batch_shape)
                )
                self.emit(f"{row} = ({tile}) // {tiles_n} % {tiles_m} * 16 + {group}")
                col_base = f"({tile}) % {tiles_n} * 8"
                self.emit(f"{col} = {col_base} + {group}")
                accs = [self.df.new_var("fragment_mma_c") for _ in slots]
                init_accumulators(accs, batch, col_base)
                step = self.df.new_var("fragment_mma_step")
                k_loop = statement_from_string(
                    f"for {step} in cutlass.range_constexpr({steps // splits}):\n    pass"
                )
                assert isinstance(k_loop, ast.For)
                k_loop.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", k_loop.body)):
                    base_step = (
                        f"({split} * {steps // splits} + {step})"
                        if splits > 1
                        else step
                    )
                    k0 = f"{base_step} * 8 + {quad}"
                    k1 = f"{base_step} * 8 + {quad} + 4"
                    words = [*a_words(k0, k1, batch), *b_words(k0, k1, batch)]
                    self.emit(
                        f"{', '.join(accs)} = {rt}.mma_tf32({', '.join([*words, *accs])})"
                    )
                self.cg.add_statement(k_loop)
                emit_stores(accs, batch, col_base)
            self.cg.add_statement(loop)
        # The lanes read staged operands a later phase may recycle.
        self.synchronize()
        if direct:
            assert sink is not None
            self.direct_stores.add(sink.node)
        if splits == 1:
            return result
        partials = result

        def summed(coords: tuple[str, ...]) -> str:
            terms = " + ".join(
                partials.read((str(index), *coords)) for index in range(splits)
            )
            return self.cast(f"({terms})", out_dtype)

        return Fragment(
            shape,
            out_dtype,
            summed,
            True,
            (partials,),
            partials.storage,
        )

    @staticmethod
    def in_bounds(index: str, extent: int, tile: int = 8) -> list[str]:
        """The predicate keeping *index* inside an axis the mma tiles may
        overhang (16 rows, 8 columns, 8 k); empty when *tile* divides *extent*."""
        return [f"({index}) < {extent}"] if extent % tile else []

    def guarded_assign(
        self, name: str, value: str, fallback: str, bounds: list[str]
    ) -> None:
        if not bounds:
            self.emit(f"{name} = {value}")
            return
        self.emit(f"{name} = {fallback}")
        self.emit(f"if {' and '.join(bounds)}:\n    {name} = {value}")

    def emit_fragment_mma_preamble(self) -> None:
        """Record that the ``mma.sync`` helpers' import is needed; the root's
        commit emits it once per generated module (``emit_mma_preamble``)."""
        self.wants_mma_preamble = True

    def emit_mma_preamble(self) -> None:
        """Import the ``mma.sync`` helpers once per generated module."""
        if not self.wants_mma_preamble or getattr(
            self.cg, "_helion_fragment_mma_module_emitted", False
        ):
            return
        self.cg._helion_fragment_mma_module_emitted = True  # type: ignore[attr-defined]
        for stmt in ast.parse(_FRAGMENT_MMA_PREAMBLE).body:
            self.cg.module_statements.append(stmt)
