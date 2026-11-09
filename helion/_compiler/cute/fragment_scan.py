"""Scans and cooperative reductions of a computed fragment.

``FragmentScanEmitter`` lowers ``hl.associative_scan`` (serial, warp-shuffle
and cooperative forms), the CTA reductions a scan completes, and the
completed-scan reductions shared with the pointwise lowering."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.ir import Reduction

from ... import exc
from ..ast_extension import statement_from_string
from .fragment_context import Fragment
from .fragment_context import FragmentContext
from .fragment_context import _fragment_logical_shape
from .register_snapshots import SnapshotOwner

if TYPE_CHECKING:
    import ast
    from collections.abc import Callable

    from torch.fx import Node


class FragmentScanEmitter(FragmentContext):
    """Associative scans and CTA reductions over staged fragments."""

    def scan_axis_extent(self, node: Node, source: Fragment, dim: int) -> str:
        """Intersect physical/tile bounds with the scan producer's proved domain."""
        from .gather_domains import loop_domain_facts

        fake = cast("torch.Tensor", node.meta["val"])
        capacity = source.shape[dim]
        extent = self.logical_axis_extent(fake, dim, capacity)
        logical = _fragment_logical_shape(
            self.env,
            cast("Node", node.args[1]),
            scalar_indexed_loads=True,
            tensor_indexed_loads=True,
            proven_domains=loop_domain_facts(self.env, self.graphs).shapes,
        )
        if logical is not None and len(logical) == len(source.shape):
            bound = logical[dim]
            if not self.env.known_nonnegative(sympy.Add(bound, -capacity)):
                extent = f"min({extent}, {self.sym(bound)})"
        return extent

    def serial_scan(self, node: Node, values: dict[Node, object]) -> Fragment:
        source = values[cast("Node", node.args[1])]
        assert isinstance(source, Fragment)
        source = self.materialize(source)
        self.held.append(source)
        result = self.allocate(source)
        dim = cast("int", node.args[2]) % len(source.shape)
        reverse = bool(node.args[3])
        extent = self.scan_axis_extent(node, source, dim)

        def line(coords: tuple[str, ...]) -> None:
            total = self.df.new_var("fragment_scan")
            index = self.df.new_var("fragment_scan_index")
            initialized = self.df.new_var("fragment_scan_initialized")
            self.emit(f"{total} = {self.cast('0', source.dtype)}")
            self.emit(f"{initialized} = False")
            position = f"{source.shape[dim] - 1} - {index}" if reverse else index
            location = (*coords[:dim], position, *coords[dim:])
            value = source.read(location)
            self.emit(
                f"for {index} in range({source.shape[dim]}):\n"
                f"    if ({position}) < ({extent}):\n"
                f"        if {initialized}:\n"
                f"            {total} = {self.cast(f'{total} + {value}', source.dtype)}\n"
                f"        else:\n"
                f"            {total} = {value}\n"
                f"        {initialized} = True\n"
                f"        {result.read(location)} = {total}\n"
                f"    else:\n"
                f"        {result.read(location)} = {self.cast('0', source.dtype)}"
            )

        self.elements((*source.shape[:dim], *source.shape[dim + 1 :]), line)
        self.held.pop()
        return result

    def scan(self, node: Node, values: dict[Node, object]) -> Fragment:
        from .autotune.heuristics.cute_fragment_warp_scan import active_warp_scan_roots
        from .autotune.heuristics.cute_fragment_warp_scan import warp_scan_supported

        source = values[cast("Node", node.args[1])]
        assert isinstance(source, Fragment)
        dim = cast("int", node.args[2]) % len(source.shape)
        root = self.cg.current_root_graph_info
        assert root is not None
        if (
            self.df.config.get("cute_fragment_warp_scan", False)
            and root.graph_id in active_warp_scan_roots(self.env, self.df.config)
            and warp_scan_supported(source.dtype, source.shape[dim])
        ):
            return self.warp_scan(node, values)
        mode = self.df.config.get("cute_fragment_scan", "serial")
        if mode == "serial":
            return self.serial_scan(node, values)
        if mode == "cooperative":
            return self.cooperative_scan(node, values)
        raise exc.InvalidConfig(f"unsupported computed fragment scan: {mode!r}")

    def warp_scan(self, node: Node, values: dict[Node, object]) -> Fragment:
        """Two CTA phases, each with full-warp, typed prefix exchanges.

        A warp owns one row chunk. The second phase scans at most 32 chunk
        totals, then adds the exclusive chunk carry to its own partials.
        No input is overwritten and no collective is guarded by a lane test.
        """
        from .autotune.heuristics.cute_fragment_warp_scan import warp_scan_supported

        source = values[cast("Node", node.args[1])]
        assert isinstance(source, Fragment)
        dim = cast("int", node.args[2]) % len(source.shape)
        capacity = source.shape[dim]
        if not warp_scan_supported(source.dtype, capacity):
            raise exc.InvalidConfig(
                "warp-prefix scan requires a positive Int32/64 axis or Float32/64 axis capacity <= 1024"
            )
        assert self.threads % 32 == 0
        reverse = bool(node.args[3])
        extent = self.scan_axis_extent(node, source, dim)
        return self.warp_scan_fragment(
            source, dim, extent, reverse, export=node in self.scan_plans
        )

    def warp_scan_fragment(
        self,
        source: Fragment,
        dim: int,
        extent: str,
        reverse: bool,
        *,
        export: bool = False,
    ) -> Fragment:
        """Recursively scan chunk totals using associative integer addition.

        Every chunk writes a total, including zero for logical padding. These
        totals have their own complete logical domain. Their recursive scan is
        forward even when the original positions are visited in reverse order.
        Small scans retain the original two-phase floating-point tree verbatim.
        """
        statements = self.cg.statements_stack[-1]
        start = len(statements)
        capacity = source.shape[dim]
        assert capacity <= 1024 or source.dtype in (torch.int32, torch.int64)
        row_shape = (*source.shape[:dim], *source.shape[dim + 1 :])
        chunks = (capacity + 31) // 32
        work_shape = (*row_shape, chunks)
        zero = self.cast("0", source.dtype)
        register = self.df.new_var("fragment_scan_export") if export else ""
        slot = self.df.new_var("fragment_scan_slot") if export else ""
        if export:
            slots = (capacity + self.threads - 1) // self.threads
            self.emit(
                f"{register} = cute.make_rmem_tensor(({slots},), {self.dtype(source.dtype)})"
            )
            self.emit(f"{register}.fill({zero})")

        def scan_elements(body: Callable[[tuple[str, ...]], None]) -> None:
            if not export:
                self.elements(work_shape, body, threads_per_element=32)
                return
            loop = cast(
                "ast.For",
                statement_from_string(
                    f"for {slot} in cutlass.range_constexpr({slots}):\n    pass"
                ),
            )
            loop.body.clear()
            chunk = self.df.new_var("fragment_scan_chunk")
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                self.emit(
                    f"{chunk} = {self.thread} // 32 + {slot} * {self.threads // 32}"
                )
                guard = cast(
                    "ast.If", statement_from_string(f"if {chunk} < {chunks}:\n    pass")
                )
                guard.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", guard.body)):
                    body((*("0" for _ in row_shape), chunk))
                self.cg.add_statement(guard)
            self.cg.add_statement(loop)
            self.synchronize()

        self.held.append(source)
        result = self.allocate(source)
        self.held.append(result)
        totals = self.allocate(Fragment(work_shape, source.dtype, lambda _: zero))
        self.held.append(totals)

        def prefix(value: str, lane: str) -> None:
            for distance in (1, 2, 4, 8, 16):
                peer = self.df.new_var("fragment_scan_peer")
                self.emit(
                    f"{peer} = cute.arch.shuffle_sync_up({value}, offset={distance}, "
                    "mask=0xffffffff, mask_and_clamp=0)"
                )
                self.emit(
                    f"if {lane} >= {distance}:\n"
                    f"    {value} = {self.cast(f'{peer} + {value}', source.dtype)}"
                )

        def positions(coords: tuple[str, ...]) -> tuple[str, str, tuple[str, ...]]:
            lane = self.df.new_var("fragment_scan_lane")
            rank = self.df.new_var("fragment_scan_rank")
            self.emit(f"{lane} = {self.thread} % 32")
            self.emit(f"{rank} = ({coords[-1]}) * 32 + {lane}")
            position = f"({extent}) - 1 - {rank}" if reverse else rank
            row = coords[:-1]
            return lane, rank, (*row[:dim], position, *row[dim:])

        def partials(coords: tuple[str, ...]) -> None:
            lane, rank, location = positions(coords)
            value = self.df.new_var("fragment_scan_value")
            self.emit(f"{value} = {zero}")
            # A lazy source read can emit statements. Keep all such statements
            # under the logical-bound guard, including reverse-tail addresses.
            branch = cast(
                "ast.If", statement_from_string(f"if {rank} < ({extent}):\n    pass")
            )
            branch.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                self.emit(f"{value} = {self.cast(source.read(location), source.dtype)}")
            branch.orelse = [statement_from_string(f"{value} = {zero}")]
            self.cg.add_statement(branch)
            prefix(value, lane)
            if export:
                self.emit(f"{register}[{slot}] = {value}")
            row = coords[:-1]
            padding = (*row[:dim], rank, *row[dim:])
            self.emit(
                f"if {rank} < ({extent}):\n"
                f"    {result.read(location)} = {value}\n"
                f"elif {rank} < {capacity}:\n"
                f"    {result.read(padding)} = {zero}"
            )
            last = self.df.new_var("fragment_scan_last")
            total = self.df.new_var("fragment_scan_total")
            self.emit(f"{last} = min(31, max(0, ({extent}) - ({coords[-1]}) * 32 - 1))")
            self.emit(
                f"{total} = cute.arch.shuffle_sync({value}, {last}, "
                "mask=0xffffffff, mask_and_clamp=31)"
            )
            self.emit(
                f"if {lane} == 0:\n"
                f"    {totals.read(coords)} = {total} if ({coords[-1]}) * 32 < ({extent}) else {zero}"
            )

        scan_elements(partials)

        if chunks > 32:
            prefixes = self.warp_scan_fragment(
                totals, len(work_shape) - 1, str(chunks), False
            )
            self.held.append(prefixes)

            def wide_carries(coords: tuple[str, ...]) -> None:
                position = coords[dim]
                rank = f"({extent}) - 1 - ({position})" if reverse else position
                chunk = f"(({rank}) // 32)"
                row = (*coords[:dim], *coords[dim + 1 :])
                branch = cast(
                    "ast.If",
                    statement_from_string(
                        f"if ({position}) < ({extent}) and {chunk} > 0:\n    pass"
                    ),
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    carry = prefixes.read((*row, f"{chunk} - 1"))
                    partial = result.read(coords)
                    self.emit(
                        f"{partial} = {self.cast(f'{carry} + {partial}', source.dtype)}"
                    )
                self.cg.add_statement(branch)

            self.elements(source.shape, wide_carries)
            self.held.pop()
            self.held.pop()
            self.held.pop()
            self.held.pop()
            return result

        def carries(coords: tuple[str, ...]) -> None:
            lane, rank, location = positions(coords)
            value = self.df.new_var("fragment_scan_chunks")
            self.emit(
                f"{value} = {totals.read((*coords[:-1], lane))} if {lane} < {chunks} else {zero}"
            )
            prefix(value, lane)
            carry = self.df.new_var("fragment_scan_carry")
            self.emit(
                f"{carry} = cute.arch.shuffle_sync({value}, max(0, ({coords[-1]}) - 1), "
                "mask=0xffffffff, mask_and_clamp=31)"
            )
            # Chunk zero retains its partial bit-for-bit: adding an artificial
            # zero would change negative zero and is unnecessary.
            partial = f"{register}[{slot}]" if export else result.read(location)
            self.emit(
                f"if {rank} < ({extent}) and ({coords[-1]}) > 0:\n"
                f"    {result.read(location)} = {self.cast(f'{carry} + {partial}', source.dtype)}"
            )
            if export:
                self.emit(
                    f"if {rank} < ({extent}) and ({coords[-1]}) > 0:\n"
                    f"    {register}[{slot}] = {self.cast(f'{carry} + {partial}', source.dtype)}"
                )

        scan_elements(carries)
        self.held.pop()
        self.held.pop()
        self.held.pop()
        if self.warp_producer_recorder is not None:
            self.warp_producer_recorder.scan(statements[start:], source, result, totals)
        if export:

            def read(coords: tuple[str, ...]) -> str:
                if self.snapshot_owner is None:
                    return result.read(coords)
                self.snapshot_owner.prove(self.flatten(coords, source.shape))
                return f"{register}[{self.snapshot_owner.slot}]"

            return dataclasses.replace(result, element=read, dependencies=(result,))
        return result

    def cta_reduction(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        reduction_shape: tuple[int, ...],
        compute_dtype: torch.dtype,
        kind: str,
        element: Callable,
        combine: Callable,
        dependencies: tuple[Fragment, ...],
    ) -> Fragment:
        """Cooperate across the CTA for each complete output coordinate.

        Warp leaders publish partials, then a complete first warp combines
        them. Every thread participates in the publication and retirement
        barriers, including threads with an empty input stripe.
        """
        assert self.threads % 32 == 0
        self.held.append(dependencies)
        result = self.allocate(Fragment(shape, dtype, lambda _: "0"))
        self.held.append(result)
        warps = self.threads // 32
        scratch = self.allocate(Fragment((warps,), compute_dtype, lambda _: "0"))
        self.held.append(scratch)
        output = self.df.new_var("fragment_cta_output")
        index = self.df.new_var("fragment_cta_index")
        total = self.df.new_var("fragment_cta_reduce")
        peer = self.df.new_var("fragment_cta_peer")
        identity = self.cast(
            self.scalar(Reduction.default_accumulator(kind, compute_dtype)),
            compute_dtype,
        )
        loop = cast(
            "ast.For",
            statement_from_string(
                f"for {output} in range({math.prod(shape)}):\n    pass"
            ),
        )
        loop.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
            coords = self.coordinates(output, shape)
            self.emit(f"{total} = {identity}")
            unroll = self.df.config.get("cute_fragment_reduction") == "cta_unroll"
            slot = self.df.new_var("fragment_cta_slot") if unroll else index
            count = math.prod(reduction_shape)
            iteration = (
                f"cutlass.range_constexpr({(count + self.threads - 1) // self.threads})"
                if unroll
                else f"range({self.thread}, {count}, {self.threads})"
            )
            inputs = cast(
                "ast.For",
                statement_from_string(f"for {slot} in {iteration}:\n    pass"),
            )
            inputs.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", inputs.body)):
                if unroll:
                    self.emit(f"{index} = {self.thread} + {slot} * {self.threads}")
                    guard = cast(
                        "ast.If",
                        statement_from_string(f"if {index} < {count}:\n    pass"),
                    )
                    guard.body.clear()
                    with (
                        self.cg.set_statements(cast("list[ast.AST]", guard.body)),
                        self.snapshot_scope(index, slot, count),
                    ):
                        value = element(
                            coords, self.coordinates(index, reduction_shape)
                        )
                        self.emit(
                            f"{total} = {self.cast(combine(total, value), compute_dtype)}"
                        )
                    self.cg.add_statement(guard)
                else:
                    value = element(coords, self.coordinates(index, reduction_shape))
                    self.emit(
                        f"{total} = {self.cast(combine(total, value), compute_dtype)}"
                    )
            self.cg.add_statement(inputs)
            for distance in (16, 8, 4, 2, 1):
                self.emit(
                    f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                )
                self.emit(f"{total} = {self.cast(combine(total, peer), compute_dtype)}")
            self.emit(
                f"if {self.thread} % 32 == 0:\n    {scratch.read((f'{self.thread} // 32',))} = {total}"
            )
            self.synchronize()
            guard = cast(
                "ast.If", statement_from_string(f"if {self.thread} < 32:\n    pass")
            )
            guard.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", guard.body)):
                self.emit(f"{total} = {identity}")
                self.emit(
                    f"if {self.thread} < {warps}:\n    {total} = {scratch.read((self.thread,))}"
                )
                for distance in (16, 8, 4, 2, 1):
                    self.emit(
                        f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                    )
                    self.emit(
                        f"{total} = {self.cast(combine(total, peer), compute_dtype)}"
                    )
                self.emit(
                    f"if {self.thread} == 0:\n    {result.read(coords)} = {self.cast(total, dtype)}"
                )
            self.cg.add_statement(guard)
            self.synchronize()
        self.cg.add_statement(loop)
        self.held.pop()
        self.held.pop()
        self.held.pop()
        return result

    def completed_scan_reductions(self, shape: tuple[int, ...]) -> None:
        size = math.prod(shape)
        pending = self.scan_pending
        dtype = pending[0][2]
        assert all(item[2] == dtype for item in pending)
        totals = [self.df.new_var("fragment_scan_reduce") for _ in pending]
        identities = [
            self.cast(self.scalar(Reduction.default_accumulator(item[3], dtype)), dtype)
            for item in pending
        ]

        def combine(kind: str, left: str, right: str) -> str:
            return self.cast(
                self.env.backend.reduction_combine_expr(kind, left, right, dtype), dtype
            )

        for total, identity in zip(totals, identities, strict=True):
            self.emit(f"{total} = {identity}")
        slot = self.df.new_var("fragment_scan_consumer_slot")
        index = self.df.new_var("fragment_scan_consumer_index")
        loop = cast(
            "ast.For",
            statement_from_string(
                f"for {slot} in cutlass.range_constexpr({(size + self.threads - 1) // self.threads}):\n    pass"
            ),
        )
        loop.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
            self.emit(f"{index} = {self.thread} + {slot} * {self.threads}")
            guard = cast(
                "ast.If", statement_from_string(f"if {index} < {size}:\n    pass")
            )
            guard.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", guard.body)):
                assert self.snapshot_owner is None
                self.snapshot_owner = SnapshotOwner(index, slot, size)
                try:
                    for (result, element, _, kind), total in zip(
                        pending, totals, strict=True
                    ):
                        value = element(
                            tuple("0" for _ in result.shape),
                            self.coordinates(index, shape),
                        )
                        self.emit(f"{total} = {combine(kind, total, value)}")
                finally:
                    self.snapshot_owner = None
            self.cg.add_statement(guard)
        self.cg.add_statement(loop)
        peer = self.df.new_var("fragment_scan_reduce_peer")
        for distance in (16, 8, 4, 2, 1):
            for (_, _, _, kind), total in zip(pending, totals, strict=True):
                self.emit(
                    f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                )
                self.emit(f"{total} = {combine(kind, total, peer)}")
        warps = self.threads // 32
        scratch = self.allocate(Fragment((warps * len(pending),), dtype, lambda _: "0"))
        self.held.append(scratch)
        for i, total in enumerate(totals):
            self.emit(
                f"if {self.thread} % 32 == 0:\n    {scratch.read((f'{i * warps} + {self.thread} // 32',))} = {total}"
            )
        self.synchronize()
        guard = cast(
            "ast.If", statement_from_string(f"if {self.thread} < 32:\n    pass")
        )
        guard.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", guard.body)):
            for i, ((result, _, _, kind), total, identity) in enumerate(
                zip(pending, totals, identities, strict=True)
            ):
                self.emit(f"{total} = {identity}")
                self.emit(
                    f"if {self.thread} < {warps}:\n    {total} = {scratch.read((f'{i * warps} + {self.thread}',))}"
                )
                for distance in (16, 8, 4, 2, 1):
                    self.emit(
                        f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                    )
                    self.emit(f"{total} = {combine(kind, total, peer)}")
                self.emit(
                    f"if {self.thread} == 0:\n    {result.read(tuple('0' for _ in result.shape))} = {self.cast(total, result.dtype)}"
                )
        self.cg.add_statement(guard)
        self.synchronize()
        self.held.pop()

    def cooperative_scan(self, node: Node, values: dict[Node, object]) -> Fragment:
        source = values[cast("Node", node.args[1])]
        assert isinstance(source, Fragment)
        self.held.append(source)
        current = self.allocate(source)
        self.held.append(current)
        other = self.allocate(source)
        self.held.append(other)
        dim = cast("int", node.args[2]) % len(source.shape)
        reverse = bool(node.args[3])
        extent = self.scan_axis_extent(node, source, dim)

        def initialize(coords: tuple[str, ...]) -> None:
            self.emit(
                f"if ({coords[dim]}) < ({extent}):\n"
                f"    {current.read(coords)} = {self.cast(source.read(coords), source.dtype)}\n"
                f"else:\n"
                f"    {current.read(coords)} = {self.cast('0', source.dtype)}"
            )

        statements = self.cg.statements_stack[-1]
        start = len(statements)
        self.elements(source.shape, initialize)
        if self.warp_producer_recorder is not None:
            self.warp_producer_recorder.publication(statements[start:], current)
        # Every stage reads only the preceding stage. A CTA barrier follows
        # each complete tile write, including when a thread owns several
        # elements. Never overwrite the input: it may have other consumers.
        for stage in range((source.shape[dim] - 1).bit_length()):
            distance = 1 << stage

            def combine(
                coords: tuple[str, ...],
                distance: int = distance,
                current: Fragment = current,
                other: Fragment = other,
            ) -> None:
                position = coords[dim]
                neighbor = (
                    f"({position}) + {distance}"
                    if reverse
                    else f"({position}) - {distance}"
                )
                in_range = (
                    f"({neighbor}) < ({extent})" if reverse else f"({neighbor}) >= 0"
                )
                location = (*coords[:dim], neighbor, *coords[dim + 1 :])
                left = current.read(location)
                right = current.read(coords)
                self.emit(
                    f"if ({position}) < ({extent}) and ({in_range}):\n"
                    f"    {other.read(coords)} = {self.cast(f'{left} + {right}', source.dtype)}\n"
                    f"else:\n"
                    f"    {other.read(coords)} = {right}"
                )

            start = len(statements)
            self.elements(source.shape, combine)
            if self.warp_producer_recorder is not None:
                self.warp_producer_recorder.publication(statements[start:], other)
            current, other = other, current
        self.held.pop()
        self.held.pop()
        self.held.pop()
        return current
