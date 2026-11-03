"""Top-k / sort selection inside a computed fragment.

``FragmentSelectionEmitter`` owns the fragment-resident selection forms: the
per-thread partial top-k merge, the traced selection network
(``cute_fragment_topk_network``, lowered through ``selection_lowering``), the
CTA-wide int64 key sort (``cute_fragment_cta_topk``), the gather of selected
values and the proven bounded gather."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ..ast_extension import statement_from_string
from .fragment_context import Fragment
from .fragment_context import FragmentContext
from .fragment_context import _static_selection_extent
from .selection.ordered_selection import ordered_selection_args
from .selection.ordered_selection import selection_values_only

if TYPE_CHECKING:
    import ast

    from torch.fx import Node


class FragmentSelectionEmitter(FragmentContext):
    """Selection (top-k, sort, gather) over staged fragments."""

    def topk(self, node: Node, values: dict[Node, object]) -> tuple[Fragment, Fragment]:
        """Complete last-axis sorting/selection in a uniform fragment owner.

        Each worker owns one logical leading coordinate and all its selected
        slots. Integer ranks preserve subnormals, zeros, NaNs and stable index
        ties without moving comparisons across shared-memory lifetimes.
        """
        arguments = ordered_selection_args(node)
        assert arguments is not None
        source_node, k, dim, largest, _ = arguments
        source = values[source_node]
        assert isinstance(source, Fragment)
        fake = cast("torch.Tensor", source_node.meta["val"])
        n = _static_selection_extent(self.env, source_node)
        if k is None:
            k = n
        bid = self.env.resolve_block_id(fake.shape[-1])
        # The configured capacity can include padding, but a rolling selection
        # would choose independent subsets instead of one complete top-k.
        if (
            n is None
            or k is None
            or dim not in (-1, fake.ndim - 1)
            or bid in self.offsets
            or source.shape[-1] < n
            or not 0 < k <= n
        ):
            raise exc.InvalidConfig(
                "fragment topk requires a complete static last axis"
            )
        from .autotune.heuristics.cute_fragment_topk_network import (
            network_topk_supported,
        )

        values_only64 = source.dtype == torch.int64 and selection_values_only(node)
        use_network = self.df.config.get(
            "cute_fragment_topk_network", False
        ) and network_topk_supported(source.dtype, n, k, values_only=values_only64)
        source = self.materialize(source)
        self.held.append(source)
        out_shape = (*source.shape[:-1], k)
        selected = self.allocate(Fragment(out_shape, source.dtype, lambda _: "0"))
        self.held.append(selected)
        # Keep tuple metadata for the proved-dead index result without reserving
        # shared storage. Every observed-index/default path keeps its old owner.
        if use_network and values_only64:
            indices = Fragment(out_shape, torch.int64, lambda _: "0", True)
        else:
            indices = self.allocate(Fragment(out_shape, torch.int64, lambda _: "0"))
        self.held.append(indices)
        float_rank = source.dtype in (torch.float16, torch.bfloat16, torch.float32)
        rank_type = "cutlass.Int32" if float_rank else self.dtype(source.dtype)
        better = ">" if largest else "<"
        worse = "<" if largest else ">"

        def select(outer: tuple[str, ...]) -> None:
            slot = self.df.new_var("fragment_topk_slot")
            col = self.df.new_var("fragment_topk_column")
            last_rank = self.df.new_var("fragment_topk_previous_rank")
            last_col = self.df.new_var("fragment_topk_previous_column")
            best_rank = self.df.new_var("fragment_topk_rank")
            best_col = self.df.new_var("fragment_topk_column_selected")
            rank = self.df.new_var("fragment_topk_candidate")
            found = self.df.new_var("fragment_topk_found")
            self.emit(f"{last_rank} = {rank_type}(0)")
            self.emit(f"{last_col} = cutlass.Int64(-1)")
            loop = cast(
                "ast.For", statement_from_string(f"for {slot} in range({k}):\n    pass")
            )
            loop.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                self.emit(f"{best_rank} = {rank_type}(0)")
                self.emit(f"{best_col} = cutlass.Int64(0)")
                self.emit(f"{found} = cutlass.Boolean(False)")
                scan = cast(
                    "ast.For",
                    statement_from_string(f"for {col} in range({n}):\n    pass"),
                )
                scan.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", scan.body)):
                    value = source.read((*outer, col))
                    if float_rank:
                        bits = self.df.new_var("fragment_topk_bits")
                        magnitude = self.df.new_var("fragment_topk_magnitude")
                        sign = self.df.new_var("fragment_topk_sign")
                        narrow = source.dtype != torch.float32
                        bit_type = "cutlass.Int16" if narrow else "cutlass.Int32"
                        mask = 32767 if narrow else 2147483647
                        infinity = (
                            0x7C00
                            if source.dtype == torch.float16
                            else (0x7F80 if narrow else 0x7F800000)
                        )
                        self.emit(
                            f"{bits} = cutlass.Int32({value}.bitcast({bit_type}))"
                        )
                        self.emit(f"{magnitude} = {bits} & cutlass.Int32({mask})")
                        self.emit(f"{sign} = {bits} >> cutlass.Int32(31)")
                        self.emit(f"{rank} = ({magnitude} ^ {sign}) - {sign}")
                        self.emit(
                            f"if {magnitude} > cutlass.Int32({infinity}):\n    {rank} = cutlass.Int32({mask})"
                        )
                    else:
                        self.emit(f"{rank} = {rank_type}({value})")
                    eligible = f"({slot} == 0 or {rank} {worse} {last_rank} or ({rank} == {last_rank} and {col} > {last_col}))"
                    preferred = f"(not {found} or {rank} {better} {best_rank})"
                    predicate = self.predicate((eligible, preferred))
                    self.emit(
                        f"if {predicate}:\n    {found} = cutlass.Boolean(True)\n    {best_rank} = {rank}\n    {best_col} = cutlass.Int64({col})"
                    )
                self.cg.add_statement(scan)
                self.emit(
                    f"{selected.read((*outer, slot))} = {source.read((*outer, best_col))}"
                )
                self.emit(f"{indices.read((*outer, slot))} = {best_col}")
                self.emit(f"{last_rank} = {best_rank}")
                self.emit(f"{last_col} = {best_col}")
            self.cg.add_statement(loop)

        if use_network:
            self.network_topk(
                source, selected, None if values_only64 else indices, n, k, largest
            )
        else:
            self.elements(source.shape[:-1], select)
        self.held.pop()
        self.held.pop()
        self.held.pop()

        # Last-axis capacity is exact. Logical leading tails retain their own
        # shape/offset ownership; no data-derived column owns an output row.
        def domain(coords: tuple[str, ...]) -> tuple[str, ...]:
            return source.domain((*coords[:-1], "0"))

        return (
            dataclasses.replace(
                selected, logical_domain=domain, dependencies=(source,)
            ),
            dataclasses.replace(indices, logical_domain=domain, dependencies=(source,)),
        )

    def network_topk(
        self,
        source: Fragment,
        selected: Fragment,
        indices: Fragment | None,
        n: int,
        k: int,
        largest: bool,
    ) -> None:
        """A physical warp selects one complete leading coordinate.

        Shared source/output lifetimes and CTA barriers are the serial owner's.
        Only key loading and exact selection change. Every lane participates in
        each network, including logical/physical padding; cyclic output owners
        store each public slot once and gather the original typed payload.
        An Int64 value-only boundary instead carries the value itself, without
        another original-column field or a payload gather. Complementing signed
        keys reverses their complete order, including both Int64 extrema.
        """
        from .selection.selection_lowering import emit_selection_network

        assert self.threads % 32 == 0
        assert indices is not None or source.dtype == torch.int64
        if (
            indices is None
            and self.df.config.get("cute_fragment_cta_topk", False)
            and source.dtype == torch.int64
        ):
            self.cta_int64_topk(source, selected, n, k, largest)
            return
        padded_n = 1 << (n - 1).bit_length()
        padded_k = 1 << (k - 1).bit_length()
        local_n = max(1, padded_n // 32)
        local_k = max(1, padded_k // 32)
        bits = max(1, (n - 1).bit_length())
        index_mask = (1 << bits) - 1

        def line(row: tuple[str, ...]) -> None:
            lane = self.df.new_var("fragment_topk_lane")
            keys = self.df.new_var("fragment_topk_keys")
            local = self.df.new_var("fragment_topk_local")
            col = self.df.new_var("fragment_topk_col")
            value = self.df.new_var("fragment_topk_value")
            rank = self.df.new_var("fragment_topk_rank")
            self.emit(f"{lane} = {self.thread} % 32")
            self.emit(f"{keys} = cute.make_rmem_tensor({local_n}, cutlass.Int64)")
            self.emit(f"{keys}.fill(cutlass.Int64(-9223372036854775808))")
            loop = cast(
                "ast.For",
                statement_from_string(
                    f"for {local} in cutlass.range_constexpr({local_n}):\n    pass"
                ),
            )
            loop.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                self.emit(f"{col} = {local} * 32 + {lane}")
                branch = cast(
                    "ast.If", statement_from_string(f"if {col} < {n}:\n    pass")
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    self.emit(
                        f"{value} = {self.cast(source.read((*row, col)), source.dtype)}"
                    )
                    if indices is None:
                        self.emit(
                            f"{keys}[{local}] = {value if largest else f'~{value}'}"
                        )
                    elif source.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        narrow = source.dtype != torch.float32
                        mask = 0x7FFF if narrow else 0x7FFFFFFF
                        infinity = {
                            torch.float16: 0x7C00,
                            torch.bfloat16: 0x7F80,
                            torch.float32: 0x7F800000,
                        }[source.dtype]
                        word = self.df.new_var("fragment_topk_bits")
                        magnitude = self.df.new_var("fragment_topk_magnitude")
                        sign = self.df.new_var("fragment_topk_sign")
                        self.emit(
                            f"{word} = cutlass.Int32({value}.bitcast(cutlass.{'Int16' if narrow else 'Int32'}))"
                        )
                        self.emit(f"{magnitude} = {word} & cutlass.Int32({mask})")
                        self.emit(f"{sign} = {word} >> cutlass.Int32(31)")
                        self.emit(
                            f"{rank} = cutlass.Int64(({magnitude} ^ {sign}) - {sign})"
                        )
                        self.emit(
                            f"if {magnitude} > cutlass.Int32({infinity}):\n    {rank} = cutlass.Int64({mask})"
                        )
                    else:
                        self.emit(f"{rank} = cutlass.Int64({value})")
                    if indices is not None and not largest:
                        # Widen first: smallest Int32 is a valid finite key.
                        self.emit(f"{rank} = -{rank}")
                    if indices is not None:
                        self.emit(
                            f"{keys}[{local}] = ({rank} << cutlass.Int32({bits})) | cutlass.Int64({index_mask} - {col})"
                        )
                self.cg.add_statement(branch)
            self.cg.add_statement(loop)
            result = emit_selection_network(
                self.cg,
                keys=keys,
                dtype=torch.int64,
                groups=32,
                registers=local_n,
                k=padded_k,
                lane=lane,
            ).name
            output = self.df.new_var("fragment_topk_output")
            slot = self.df.new_var("fragment_topk_slot")
            column = self.df.new_var("fragment_topk_selected_column")
            loop = cast(
                "ast.For",
                statement_from_string(
                    f"for {output} in cutlass.range_constexpr({local_k}):\n    pass"
                ),
            )
            loop.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                self.emit(f"{slot} = {output} * 32 + {lane}")
                branch = cast(
                    "ast.If", statement_from_string(f"if {slot} < {k}:\n    pass")
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    if indices is None:
                        value = f"{result}[{output}]"
                        self.emit(
                            f"{selected.read((*row, slot))} = {self.cast(value if largest else f'~{value}', torch.int64)}"
                        )
                    else:
                        self.emit(
                            f"{column} = cutlass.Int32({index_mask}) - cutlass.Int32({result}[{output}] & cutlass.Int64({index_mask}))"
                        )
                        # Resident views can still emit coordinate assignments.
                        # Resolve their read only after the selected column exists,
                        # inside the public-slot mask, never before this loop.
                        self.emit(
                            f"{selected.read((*row, slot))} = {self.cast(source.read((*row, column)), source.dtype)}"
                        )
                        self.emit(
                            f"{indices.read((*row, slot))} = cutlass.Int64({column})"
                        )
                self.cg.add_statement(branch)
            self.cg.add_statement(loop)

        self.elements(source.shape[:-1], line, threads_per_element=32)

    def cta_int64_topk(
        self,
        source: Fragment,
        selected: Fragment,
        n: int,
        k: int,
        largest: bool,
    ) -> None:
        """Sort complete Int64 values with bounded private slots and CTA peers.

        Every physical thread enters each shared publication and retirement
        barrier, including padded owners. Comparisons within a lane use a
        simultaneous private snapshot; exchanges within a warp use shuffles;
        exchanges between warps use one held shared scratch vector. Original
        indices are not observed on this path.
        """
        if not (0 < k <= n <= 1024 and self.threads & (self.threads - 1) == 0):
            raise exc.InvalidConfig("CTA selection requires a bounded Int64 fragment")
        padded = max(32, 1 << (n - 1).bit_length())
        slots = (padded + self.threads - 1) // self.threads
        keys = self.df.new_var("fragment_cta_sort_keys")
        temporary = self.df.new_var("fragment_cta_sort_next")
        self.emit(f"{keys} = cute.make_rmem_tensor(({slots},), cutlass.Int64)")
        self.emit(f"{temporary} = cute.make_rmem_tensor(({slots},), cutlass.Int64)")
        scratch = (
            self.allocate(Fragment((padded,), torch.int64, lambda _: "0"))
            if self.threads > 32 and padded > 32
            else None
        )
        self.held.append(scratch)
        leading = self.df.new_var("fragment_cta_sort_row")
        slot = self.df.new_var("fragment_cta_sort_slot")
        column = self.df.new_var("fragment_cta_sort_column")
        mine = self.df.new_var("fragment_cta_sort_mine")
        peer = self.df.new_var("fragment_cta_sort_peer")
        maximum = self.df.new_var("fragment_cta_sort_maximum")
        choose = self.df.new_var("fragment_cta_sort_choose")
        outer = cast(
            "ast.For",
            statement_from_string(
                f"for {leading} in range({math.prod(source.shape[:-1])}):\n    pass"
            ),
        )
        outer.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", outer.body)):
            coords = self.coordinates(leading, source.shape[:-1])
            fill = cast(
                "ast.For",
                statement_from_string(
                    f"for {slot} in cutlass.range_constexpr({slots}):\n    pass"
                ),
            )
            fill.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", fill.body)):
                self.emit(f"{column} = {self.thread} + {slot} * {self.threads}")
                self.emit(f"{mine} = cutlass.Int64(-9223372036854775808)")
                valid = cast(
                    "ast.If", statement_from_string(f"if {column} < {n}:\n    pass")
                )
                valid.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", valid.body)):
                    value = self.cast(source.read((*coords, column)), torch.int64)
                    self.emit(f"{mine} = {value if largest else f'~({value})'}")
                self.cg.add_statement(valid)
                self.emit(f"{keys}[{slot}] = {mine}")
            self.cg.add_statement(fill)
            for level in range(padded.bit_length() - 1):
                span = 1 << (level + 1)
                for bit in range(level, -1, -1):
                    distance = 1 << bit
                    cross_warp = 32 <= distance < self.threads
                    if cross_warp:
                        assert scratch is not None
                        self.emit(
                            f"for {slot} in cutlass.range_constexpr({slots}):\n"
                            f"    {column} = {self.thread} + {slot} * {self.threads}\n"
                            f"    if {column} < {padded}:\n"
                            f"        {scratch.read((column,))} = {keys}[{slot}]"
                        )
                        self.synchronize()
                    update = cast(
                        "ast.For",
                        statement_from_string(
                            f"for {slot} in cutlass.range_constexpr({slots}):\n    pass"
                        ),
                    )
                    update.body.clear()
                    with self.cg.set_statements(cast("list[ast.AST]", update.body)):
                        self.emit(f"{column} = {self.thread} + {slot} * {self.threads}")
                        self.emit(f"{mine} = {keys}[{slot}]")
                        valid = cast(
                            "ast.If",
                            statement_from_string(f"if {column} < {padded}:\n    pass"),
                        )
                        valid.body.clear()
                        with self.cg.set_statements(cast("list[ast.AST]", valid.body)):
                            if distance >= self.threads:
                                expression = (
                                    f"{keys}[{slot} ^ {distance // self.threads}]"
                                )
                            elif distance < 32:
                                expression = (
                                    f"cute.arch.shuffle_sync_bfly({mine}, offset={distance}, "
                                    "mask=0xffffffff, mask_and_clamp=31)"
                                )
                            else:
                                assert scratch is not None
                                expression = scratch.read((f"{column} ^ {distance}",))
                            self.emit(f"{peer} = {expression}")
                            self.emit(
                                f"{maximum} = (({column} & {span}) == 0) == (({column} & {distance}) == 0)"
                            )
                            self.emit(
                                f"{choose} = ({mine} < {peer}) if {maximum} else ({mine} > {peer})"
                            )
                            self.emit(
                                f"{temporary}[{slot}] = {peer} if {choose} else {mine}"
                            )
                        valid.orelse = [
                            statement_from_string(f"{temporary}[{slot}] = {mine}")
                        ]
                        self.cg.add_statement(valid)
                    self.cg.add_statement(update)
                    if cross_warp:
                        # Every peer read completes before the next shared pass.
                        self.synchronize()
                    self.emit(
                        f"for {slot} in cutlass.range_constexpr({slots}):\n    {keys}[{slot}] = {temporary}[{slot}]"
                    )
            self.emit(
                f"for {slot} in cutlass.range_constexpr({slots}):\n"
                f"    {column} = {self.thread} + {slot} * {self.threads}\n"
                f"    if {column} < {k}:\n"
                f"        {selected.read((*coords, column))} = "
                f"{keys + '[' + slot + ']' if largest else '~' + keys + '[' + slot + ']'}"
            )
            self.synchronize()
        self.cg.add_statement(outer)
        self.held.pop()

    def topk_gather(self, node: Node, values: dict[Node, object]) -> Fragment:
        source = values[cast("Node", node.args[0])]
        indices = values[cast("Node", node.args[2])]
        assert isinstance(source, Fragment) and isinstance(indices, Fragment)
        source = self.materialize(source)
        self.held.append(source)
        result = self.materialize(
            Fragment(
                indices.shape,
                source.dtype,
                lambda coords: source.read((*coords[:-1], indices.read(coords))),
                dependencies=(source, indices),
                logical_domain=indices.logical_domain,
            )
        )
        self.held.pop()
        return result

    def bounded_gather(self, node: Node, values: dict[Node, object]) -> Fragment:
        from .bounded_gather import prove_gather

        proof = prove_gather(self.env, node, graphs=self.graphs)
        assert proof is not None
        source_shape, index_shape = proof.source_shape, proof.index_shape
        source = values[cast("Node", node.args[0])]
        indices = values[cast("Node", node.args[2])]
        assert isinstance(source, Fragment) and isinstance(indices, Fragment)
        # Both original recipes remain live while either is materialized. In
        # particular an index recipe may read storage that source allocation
        # would otherwise recycle. Snapshot indices before gathering the source.
        self.held.extend((source, indices))
        indices = self.materialize(indices)
        self.held.append(indices)
        source = self.materialize(source)
        self.held.append(source)

        def source_coordinates(coords: tuple[str, ...]) -> tuple[str, ...]:
            return (*coords[:-1], indices.read(coords))

        def output_domain(coords: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(
                f"0 <= ({coord}) and ({coord}) < ({self.sym(size)})"
                for coord, size in zip(coords, index_shape, strict=True)
            )

        def read(coords: tuple[str, ...]) -> str:
            location = source_coordinates(coords)
            valid = self.predicate(
                [
                    *output_domain(coords),
                    *source.domain(location),
                    *(
                        f"0 <= ({coord}) and ({coord}) < ({self.sym(size)})"
                        for coord, size in zip(location, source_shape, strict=True)
                    ),
                ]
            )
            # Padded output coordinates are not covered by the logical range
            # proof. Do not evaluate their source reads, even if a padded iota
            # makes an otherwise proved index exceed the source allocation.
            return f"({source.read(location)} if {valid} else {self.cast('0', source.dtype)})"

        result = self.materialize(
            Fragment(
                indices.shape,
                source.dtype,
                read,
                dependencies=(source, indices),
                # Once copied, the result domain must not retain a value read
                # from the index/source allocations that can now be recycled.
                logical_domain=output_domain,
            )
        )
        self.held.pop()
        self.held.pop()
        self.held.pop()
        self.held.pop()
        return result
