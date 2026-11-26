"""Nested control flow of a computed fragment: loops, branches, resident while.

``FragmentControlFlow`` lowers ``hl.tile`` / ``for`` loops with carried
fragments, data-dependent ``while`` loops over register-resident state and
``if`` graphs whose outputs join in shared memory; each body is walked through
``FragmentContext.graph``."""

from __future__ import annotations

import ast
from typing import Literal
from typing import cast

import sympy
import torch
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ..ast_extension import statement_from_string
from ..device_ir import ForLoopGraphInfo
from ..device_ir import IfGraphInfo
from ..device_ir import control_flow_parent_entries
from ..host_function import HostFunction
from ..inductor_lowering import ReductionLowering
from ..variable_origin import GridOrigin
from ..variable_origin import HostOrigin
from ..variable_origin import TileEndOrigin
from .fragment_context import Fragment
from .fragment_context import FragmentContext
from .fragment_context import _loop_carry_slots
from .fragment_range import device_range_expr
from .fragment_range import dynamic_fragment_step
from .integer_atomic_epochs import can_defer_integer_epoch
from .local_atomic import terminal_finalizer_inputs
from .local_atomic import terminal_loop_symbols
from .local_atomic import uniform_local_branch_inputs
from .private_scalar_loops import privatize_scalar_loop
from .register_loads import host_load_is_readonly
from .register_snapshots import SnapshotBinding
from .register_snapshots import frame_snapshot_recipes
from .register_snapshots import snapshot_capture_slots
from .register_snapshots import snapshot_logical_shape
from .scalar_fragment_assertion import checked_range_domain


class FragmentControlFlow(FragmentContext):
    """Loops, resident while loops and conditionals over fragments."""

    def _control_scalar(
        self, value: object, snapshots: dict[int, str], *, predicate: bool = False
    ) -> str:
        """Snapshot a shared-dependent header operand in each lane before the fence."""
        if not isinstance(value, Fragment):
            # An implicit integer truth test must evaluate the Python value at
            # its proved width before deciding whether to execute the branch.
            return self.value_scalar(value) if predicate else self.scalar(value)
        if not self.referenced_buffers((value,)):
            return self.scalar(value)
        if (name := snapshots.get(id(value))) is None:
            name = self.df.new_var("fragment_control")
            self.emit(f"{name} = {self.scalar(value)}")
            snapshots[id(value)] = name
        return name

    def _range_plan(
        self, begin: object, end: object, step: object
    ) -> tuple[str, int | Literal["dynamic", "counted"] | None]:
        """Choose the counter and guard only a backedge that can wrap."""
        operands = (begin, end, step)
        start, stop, stride = (
            self.static_extent(v)
            if isinstance(v, (int, torch.SymInt, sympy.Expr))
            else None
            for v in operands
        )
        i64 = torch.iinfo(torch.int64)
        if any(
            v is not None and not i64.min <= v <= i64.max for v in (start, stop, stride)
        ):
            raise exc.InvalidConfig("fragment loop exceeds signed Int64 range")
        if stride == 0:
            raise exc.InvalidConfig("fragment loops require a nonzero step")
        for value in operands:
            if (
                isinstance(value, Node)
                and isinstance(fake := value.meta["val"], torch.Tensor)
                and fake.dtype == torch.uint64
            ):
                # An unsigned cast from a narrower unsigned tensor preserves
                # every value and never sets the sign bit of the loop counter.
                if value.target in (
                    torch.ops.prims.convert_element_type.default,
                    torch.ops.aten._to_copy.default,
                ):
                    source = value.args[0]
                    if (
                        isinstance(source, Node)
                        and isinstance(original := source.meta.get("val"), torch.Tensor)
                        and original.dtype in (torch.uint8, torch.uint16, torch.uint32)
                    ):
                        continue
                raise exc.InvalidConfig(
                    "UInt64 fragment loop operands are not proved to fit signed Int64"
                )
        if start is None or stop is None or stride is None:

            def span(value: object, literal: int | None) -> tuple[int, int] | None:
                if literal is not None:
                    return literal, literal
                fake = value.meta["val"] if isinstance(value, Node) else value
                if isinstance(fake, torch.SymInt):
                    symbol = fake._sympy_()
                    entry = HostFunction.current().expr_to_origin.get(symbol)
                    grid = self.cg.current_grid_state
                    # A root tile's end is min(offset + block_size, grid_end).
                    # Its upper bound can prove the SDK increment safe even
                    # when the starting tile offset is only known at runtime.
                    if (
                        entry is not None
                        and isinstance(entry.origin, TileEndOrigin)
                        and symbol not in self.sym_indices
                        and grid is not None
                        and (info := grid.block_id_to_info.get(entry.origin.block_id))
                        is not None
                        and info.end_expr is not None
                        and self.bounds.get(entry.origin.block_id) == info.end_var_name
                        and (cap := self.static_extent(info.end_expr)) is not None
                    ):
                        return i64.min, cap
                if isinstance(fake, torch.Tensor) and fake.dtype in (
                    torch.int8,
                    torch.uint8,
                    torch.int16,
                    torch.uint16,
                    torch.int32,
                    torch.uint32,
                ):
                    info = torch.iinfo(fake.dtype)
                    return info.min, info.max
                return None

            if stride is None:
                return "Int64", "dynamic"

            begin_span, end_span = span(begin, start), span(end, stop)
            if stride in (-1, 1) and all(
                v is not None and v[0] >= -(1 << 31) and v[1] < 1 << 31
                for v in (begin_span, end_span)
            ):
                return "Int32", None
            if stride is not None and abs(stride) > 1:
                # Ascending scf.for can overshoot stop by stride-1; the
                # SDK's reflected descending loop can overshoot start instead.
                upper = begin_span if stride < 0 else end_span
                if (
                    stride == i64.min
                    or upper is None
                    or upper[1] + abs(stride) - 1 > i64.max
                ):
                    # A zero-based positive range can count whole steps.
                    # Its ordinal never reaches Int64_MAX, even when its stop does.
                    return "Int64", "counted" if start == 0 and stride > 1 else stride
            return "Int64", None
        domain = range(start, stop, stride)
        # The loop advances once after the last value used by its body.
        after = domain[-1] + stride if domain else start
        low, high = min(start, stop, stride, after), max(start, stop, stride, after)
        if stride < 0:
            # The SDK emits an ascending scf.for from stop to start. It advances
            # the same distance (start - after), but in the opposite direction.
            high = max(high, -stride, stop + (start - after))
        if low >= -(1 << 31) and high < (1 << 31):
            return "Int32", None
        if i64.min <= low and high <= i64.max:
            return "Int64", None
        # All user operands fit. Only an internal SDK update exceeds the range.
        return "Int64", stride

    def loop(self, node: Node, values: dict[Node, object]) -> list[Fragment]:
        graph_id = node.args[0]
        assert isinstance(graph_id, int)
        graph = self.graphs[graph_id]
        assert isinstance(graph, ForLoopGraphInfo)
        if self.checked_root and any(bid in self.offsets for bid in graph.block_ids):
            # A captured outer tile symbol would now name the inner range.
            raise exc.BackendUnsupported(
                "cute", "device assertion in a loop that reuses an active tile block"
            )
        if self.local_allocations and any(
            self.df.resolved_block_size(bid) != 1 for bid in graph.block_ids
        ):
            raise exc.InvalidConfig("CTA-local atomics require uniform scalar loops")
        finalizer_needs_int64 = False
        if self.local_allocations:
            parent, _slot = control_flow_parent_entries(self.graphs)[graph_id]
            owner = next(info for info in self.graphs if info.graph is parent.graph)
            if isinstance(owner, IfGraphInfo):
                self.uniform_finalizer_symbols(
                    terminal_loop_symbols(node, self.graphs), set(graph.block_ids)
                )
                # A proved scalar finalizer loop accepts runtime host integers.
                # Keep their signed 64-bit range through its induction variable;
                # narrowing to tile-index Int32 changes bounds and body indices.
                finalizer_needs_int64 = True
        args = [values[x] for x in cast("list[Node]", node.args[3])]
        placeholders = list(graph.graph.find_nodes(op="placeholder"))
        output_node = graph.graph.find_nodes(op="output")[0]
        outputs = cast("list[Node]", output_node.args[0])
        carry_slots = _loop_carry_slots(node)
        # Match each output to its captured value through the actual phi edge.
        # Read-only captures can appear before, between, or after mutable ones.
        carries: list[Fragment] = []
        copied_slots: set[int] = set()
        self.held.append(carries)
        inner_args = list(args)
        for index, output in enumerate(outputs):
            if index in carry_slots:
                slot = carry_slots[index]
                incoming = cast("Fragment", args[slot])
                carry = (
                    incoming
                    if id(incoming) in self.local_logical_sizes
                    else self.materialize(incoming, copy=True)
                )
                inner_args[slot] = carry
                if carry is not incoming:
                    copied_slots.add(slot)
            else:
                fake = cast("torch.Tensor", output.meta["val"])
                carry = self.allocate(
                    Fragment(self.shape(fake.shape), fake.dtype, lambda _: "0")
                )
            carries.append(carry)
        body: list[ast.AST] = []
        deferred_barrier: ast.AST | None = None
        outer_offsets, outer_bounds = dict(self.offsets), dict(self.bounds)
        # Range arguments are evaluated in the enclosing scope, including
        # tensor-derived scalar recipes. Rendering an end inside the body can
        # make the loop header refer to a temporary that the body defines.
        outer_operands = self.stride_operands
        body_operands = dict(outer_operands)
        starts: list[str] = []
        axes = []
        checked_domains: dict[int, range] = {}
        steps: list[str] = []
        index_types: list[str] = []
        guarded_steps: dict[str, int | Literal["dynamic", "counted"]] = {}
        checked_operands: set[str] = set()
        entry_steps: list[ast.stmt] = []

        snapshotted_ranges: set[Node] = set()
        control_snapshots: dict[int, str] = {}

        def range_scalar(value: object, wide: bool) -> str:
            result = values[value] if isinstance(value, Node) else value
            if wide and isinstance(result, (torch.SymInt, sympy.Expr)):
                expr = result._sympy_() if isinstance(result, torch.SymInt) else result
                return device_range_expr(self, expr, "endpoint")
            scalar = self._control_scalar(result, control_snapshots)
            if isinstance(value, Node) and id(result) in control_snapshots:
                snapshotted_ranges.add(value)
            return scalar

        def proof_source(value: object, rendered: str) -> str:
            # Record lexical arithmetic, not the register holding an already
            # guarded divisor or checked stride. Nested proofs need its range.
            result = values[value] if isinstance(value, Node) else value
            if isinstance(result, (torch.SymInt, sympy.Expr)):
                return self.sym(result, wide=True)
            return rendered

        for index, (bid, begin, end) in enumerate(
            zip(
                graph.block_ids,
                cast("list[object]", node.args[1]),
                cast("list[object]", node.args[2]),
                strict=True,
            )
        ):
            tile_size = self.df.resolved_block_size(bid)
            explicit = (
                cast("list[object]", node.args[4])[index]
                if node.target is _tracing_ops._for_loop_step
                else None
            )
            step = tile_size if explicit is None else explicit
            if self.checked_root:
                checked_domains[bid] = checked_range_domain(self, begin, end, step)
            index_type, guarded_step = self._range_plan(begin, end, step)
            if finalizer_needs_int64:
                index_type = "Int64"
            wide = index_type == "Int64"
            starts.append(range_scalar(begin, wide))
            stop = range_scalar(end, wide)
            if guarded_step == "dynamic":
                operand, check, step_proof = dynamic_fragment_step(self, node, step)
                # A Cartesian grid evaluates every axis before walking the
                # first one, even when that axis has zero iterations.
                delta = self.df.new_var("fragment_step")
                entry_steps.append(statement_from_string(f"{delta} = {operand}"))
                if check:
                    entry_steps.append(
                        statement_from_string(
                            f"if {delta} == 0:\n"
                            "    _cute_inline_asm_elementwise((), asm='mov.u32 $0, 0; trap;', "
                            "constraints='=r', dtype=cutlass.Int32, is_pure=False)"
                        )
                    )
                    checked_operands.add(delta)
                steps.append(delta)
            else:
                steps.append(
                    self.sym(tile_size)
                    if explicit is None
                    else range_scalar(explicit, wide)
                )
                step_proof = proof_source(step, steps[-1])
            index_types.append(index_type)
            lexical_interval = explicit is None or (
                type(explicit) is int
                and type(tile_size) is int
                and explicit == tile_size
            )
            proof = (
                proof_source(begin, starts[-1]),
                proof_source(end, stop),
                step_proof,
            )
            axes.append((bid, begin, end, stop, proof, guarded_step, lexical_interval))
        # A later axis can read an outer tile that shares an earlier axis's
        # block ID. Install the new scope only after evaluating every header.
        for bid, begin, end, stop, proof, guarded_step, lexical_interval in axes:
            self.bounds[bid] = stop
            offset = self.offsets[bid] = self.df.new_var("fragment_tile")
            body_operands[offset] = proof
            if bid in checked_domains:
                self.checked_domains[offset] = checked_domains[bid]
            if guarded_step is not None:
                guarded_steps[offset] = guarded_step
            if lexical_interval:
                self.record_tile_interval(bid, begin, end)
        if control_snapshots:
            # Even a zero-trip loop reads its headers on every warp. Complete
            # all those reads before a later store can reuse any source slot.
            self.synchronize()
        # Headers use registers and mutable captures use their carry copies.
        # Only independent body captures need the original storage across a backedge.
        captured_nodes = cast("list[Node]", node.args[3])
        # A range node can also occupy an unused body placeholder. Body masks
        # already use its register snapshot; only a real FX use needs the input.
        body_inputs = {
            placeholder: value
            for captured, placeholder, value in zip(
                captured_nodes, placeholders, inner_args, strict=True
            )
            if placeholder.users or captured not in snapshotted_ranges
        }
        # A zero-trip axis suppresses only the body; all headers in this
        # Cartesian loop still evaluate in the enclosing scope.
        self.stride_operands = body_operands
        with (
            self.cg.set_statements(body),
            self.retired_loop_initials(
                node,
                values,
                snapshotted_ranges | {captured_nodes[slot] for slot in copied_slots},
                [
                    value
                    for slot, value in enumerate(args)
                    if slot not in copied_slots and placeholders[slot] in body_inputs
                ],
            ),
        ):
            # A newly copied loop carry is private to this invocation. A
            # specialized graph may update it in place only if it proves the
            # matching input/output dependence and coordinate ownership itself.
            private_outputs = {
                i: (placeholders[slot], carries[i])
                for i, slot in carry_slots.items()
                if slot in copied_slots
            }
            results = self.graph(
                graph.graph,
                body_inputs,
                private_loop_outputs=private_outputs,
            )
            assert isinstance(results, list)
            # Snapshot every value that reads a carry before writing any carry.
            # Disjoint recipes can write directly: none of their inputs changes
            # during the parallel assignment. Keep both sets live throughout.
            snapshots = []
            self.held.extend([results, snapshots])
            destinations = self.referenced_buffers(carries)
            in_place = {
                i for i, (_, target) in private_outputs.items() if results[i] is target
            }
            for i, result in enumerate(results):
                value = cast("Fragment", result)
                if i in in_place:
                    # The certified native transition has already written this
                    # result to its private target and synchronized the CTA.
                    snapshots.append(value)
                    continue
                snapshots.append(
                    self.materialize(value, copy=True)
                    if self.referenced_buffers([value]) & destinations
                    and id(value) not in self.local_logical_sizes
                    else value
                )
            for i, (target, source) in enumerate(zip(carries, snapshots, strict=True)):
                if i in in_place:
                    continue
                if id(target) in self.local_logical_sizes:
                    if source is not target:
                        raise exc.InvalidConfig(
                            "CTA-local atomic loop changed its allocation"
                        )
                else:
                    self.copy(source, target)
            self.held.pop()
            self.held.pop()
            # The backedge can revisit the same atomic target. Finish this
            # iteration's epoch even when no loop carry was materialized.
            pending = set(self.pending_local_atomics)
            self.synchronize_local_atomics()
            if (
                pending
                and self.df.config.get("cute_fragment_integer_atomic_epochs", False)
                and node not in self.private_scalar_loops
                and can_defer_integer_epoch(
                    body[:-1],
                    {name: dtype for name, dtype, _count in self.buffers},
                    pending,
                )
            ):
                assert ast.unparse(body[-1]) == "cute.arch.sync_threads()"
                deferred_barrier = body.pop()
        for index in reversed(range(len(graph.block_ids))):
            bid = graph.block_ids[index]
            start, step = starts[index], steps[index]
            loop_index_type = index_types[index]
            loop = statement_from_string(
                f"for {self.offsets[bid]} in range(cutlass.{loop_index_type}({start}), cutlass.{loop_index_type}({self.bounds[bid]}), {step}):\n    pass"
            )
            assert isinstance(loop, ast.For)
            # A symbolic bound can already be converted to the counter type.
            # Drop only an identity cast on the entire bound; casts inside
            # products and sums must still precede the arithmetic.
            range_call = cast("ast.Call", loop.iter)
            for pos, bound in enumerate(range_call.args[:2]):
                inner = cast("ast.Call", bound).args[0]
                if (
                    isinstance(inner, ast.Call)
                    and ast.unparse(inner.func) == f"cutlass.{loop_index_type}"
                ):
                    range_call.args[pos] = inner
            loop.body = cast("list[ast.stmt]", body)
            body = [loop]
        first_axis = self.offsets[graph.block_ids[0]] if guarded_steps else None
        self.stride_operands = outer_operands
        self.offsets, self.bounds = outer_offsets, outer_bounds
        if node in self.private_scalar_loops:
            if len(body) != 1 or not isinstance(body[0], ast.For):
                raise exc.InvalidConfig("private scalar loops require one scalar axis")
            outgoing = {carry.storage for carry in carries}
            if None in outgoing or any(carry.shape for carry in carries):
                raise exc.InvalidConfig(
                    "private scalar loops require shared scalar carries"
                )
            body = list(
                privatize_scalar_loop(self, body[0], cast("set[str]", outgoing))
            )
        if guarded_steps:
            # Keep the original lexical position, including loops that were
            # moved into the scalar owner's branch during privatization.
            new_var = self.df.new_var

            class GuardSteppedRange(ast.NodeTransformer):
                def visit_For(self, node: ast.For) -> ast.For | list[ast.stmt]:
                    self.generic_visit(node)
                    if not isinstance(node.target, ast.Name):
                        return node
                    name = node.target.id
                    checks = entry_steps if name == first_axis else []
                    if name not in guarded_steps:
                        return [*checks, node] if checks else node
                    stride = guarded_steps[name]
                    first, stop, step = cast("ast.Call", node.iter).args
                    end = new_var("fragment_stop")
                    if stride == "counted":
                        # Clamp before dividing; ceiling by quotient and remainder
                        # never adds stride - 1 to a potentially full-width stop.
                        delta = ast.unparse(step)
                        ordinal = new_var("fragment_ordinal")
                        count = (
                            f"{end} // {delta} + cutlass.Int64({end} % {delta} != 0)"
                        )
                        counted = statement_from_string(
                            f"for {ordinal} in range(cutlass.Int64(0), {count}, 1):\n    pass"
                        )
                        assert isinstance(counted, ast.For)
                        counted.body = [
                            statement_from_string(f"{name} = {ordinal} * {delta}"),
                            *node.body,
                        ]
                        return [
                            *checks,
                            statement_from_string(
                                f"{end} = max(cutlass.Int64(0), {ast.unparse(stop)})"
                            ),
                            counted,
                        ]
                    last = new_var("fragment_last")
                    active = new_var("fragment_active")
                    # Compare before updating. Saturating stop - stride keeps the
                    # cutoff representable; a dead final update cannot restart the loop.
                    if stride == "dynamic":
                        delta = ast.unparse(step)
                        ascending = new_var("fragment_ascending")
                        prelude = f"{ascending} = {delta} > 0\n"
                        low = f"cutlass.Int64({-(1 << 63)})"
                        high = f"cutlass.Int64({(1 << 63) - 1})"
                        cutoff = (
                            f"({low} if {end} < {low} + {delta} else {end} - {delta}) "
                            f"if {ascending} else "
                            f"({high} if {end} > {high} + {delta} else {end} - {delta})"
                        )
                        initial = f"{name} < {end} if {ascending} else {name} > {end}"
                        advance = f"{name} < {last} if {ascending} else {name} > {last}"
                        if delta in checked_operands:
                            initial = f"({delta} != 0) and ({initial})"
                    else:
                        cmp = "<" if stride > 0 else ">"
                        limit = -(1 << 63) if stride > 0 else (1 << 63) - 1
                        delta = f"cutlass.Int64({stride})"
                        prelude = ""
                        cutoff = (
                            f"cutlass.Int64({limit}) if {end} {cmp} cutlass.Int64({limit + stride}) "
                            f"else {end} - {delta}"
                        )
                        initial = f"{name} {cmp} {end}"
                        advance = f"{name} {cmp} {last}"
                    statements = ast.parse(
                        f"{name} = {ast.unparse(first)}\n"
                        f"{end} = {ast.unparse(stop)}\n"
                        f"{prelude}{last} = {cutoff}\n"
                        f"{active} = {initial}\n"
                        f"while {active}:\n"
                        f"    {active} = {advance}\n"
                        f"    {name} = {name} + {delta}"
                    ).body
                    changed = statements[-1]
                    assert isinstance(changed, ast.While)
                    changed.body[:0] = node.body
                    return [*checks, *statements]

            container = ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
            GuardSteppedRange().visit(container)
        for stmt in body:
            self.cg.add_statement(stmt)
        if deferred_barrier is not None:
            # Unconditional publication also covers zero-trip loops. Nothing
            # can read or reuse target storage before this boundary.
            self.cg.add_statement(deferred_barrier)
        self.held.pop()
        return carries

    def derived_snapshot(
        self, value: Fragment, sources: frozenset[Node]
    ) -> Fragment | None:
        """Initialize a distinct typed slot; retain the recipe's domain lifetime."""
        bindings: dict[Node, SnapshotBinding] = {}
        seen: set[int] = set()

        def collect(fragment: Fragment) -> None:
            if id(fragment) in seen:
                return
            seen.add(id(fragment))
            if fragment.snapshot is not None:
                bindings[fragment.snapshot.source] = fragment.snapshot
            for dependency in fragment.dependencies:
                collect(dependency)

        collect(value)
        if (
            not sources
            or set(bindings) != sources
            or any(
                binding.source not in self.snapshot_loads
                or binding.threads != self.threads
                or binding.shape != value.shape
                or binding.domain_storage
                or binding.logical_shape
                != self.shape(snapshot_logical_shape(binding.source))
                for binding in bindings.values()
            )
        ):
            return None
        shape = value.shape
        if len(shape) != 1:
            return None
        size = shape[0]
        owner_sizes = {
            size,
            *(binding.logical_shape[0] for binding in bindings.values()),
        }
        slots = (size + self.threads - 1) // self.threads
        name = self.df.new_var("fragment_derived_snapshot")
        self.register_snapshot_buffers[name] = size
        self.emit(
            f"{name} = cute.make_rmem_tensor(({slots},), {self.dtype(value.dtype)})"
        )
        self.emit(f"{name}.fill({self.cast('0', value.dtype)})")

        def read(coords: tuple[str, ...]) -> str:
            owner = self.snapshot_owner
            if owner is None or len(coords) != 1 or owner.size not in owner_sizes:
                raise exc.InvalidConfig(
                    "derived snapshot lacks its active coordinate owner"
                )
            owner.prove(coords[0])
            return f"{name}[{owner.slot}]"

        def fill(coords: tuple[str, ...]) -> None:
            self.emit(f"{read(coords)} = {self.cast(value.read(coords), value.dtype)}")

        self.held.append(value)
        self.elements(shape, fill)
        self.held.pop()
        # This is not a SnapshotBinding for the original load. Preserve the
        # result dtype and all domain dependencies; no lifetime is discarded.
        return Fragment(
            shape,
            value.dtype,
            read,
            True,
            dependencies=(value,),
            logical_domain=value.logical_domain,
        )

    def resident_while(self, node: Node, values: dict[Node, object]) -> list[Fragment]:
        from .gather_domains import loop_domain_facts
        from .resident_while import resident_while_plan

        plan = resident_while_plan(node, self.graphs)
        if node not in loop_domain_facts(self.env, self.graphs).resident_whiles:
            raise exc.InvalidConfig(
                "resident while lacks complete logical/readonly proof"
            )
        if (
            not plan.local_allocations
            and not plan.readonly_control
            and self.df.config.get("cute_fragment_bounded_gather", False) is not True
        ):
            raise exc.InvalidConfig("resident while requires explicit bounded gather")
        if any(
            self.df.config.get(key, False)
            for key in (
                "cute_fragment_register_loads",
                "cute_fragment_register_producers",
                "cute_fragment_producer_cache",
                "cute_fragment_published_scalars",
                "cute_fragment_private_scalar_loops",
                "cute_fragment_warp_results",
                "cute_fragment_packet_loads",
                "cute_integer_loop_reduction",
            )
        ):
            raise exc.InvalidConfig(
                "resident while requires shared uncached carry storage"
            )
        grid = self.cg.current_grid_state
        assert grid is not None
        if any(self.df.resolved_block_size(bid) != 1 for bid in grid.block_id_to_info):
            raise exc.InvalidConfig("resident while requires one scalar row per CTA")
        if any(
            self.df.resolved_block_size(bid) != 1
            for child in plan.nested_fors
            for bid in cast("ForLoopGraphInfo", child.graph).block_ids
        ):
            raise exc.InvalidConfig(
                "resident while requires scalar static packet loops"
            )

        # Preserve only already initialized immutable snapshots. All mutable
        # carries and other recipes retain the original held shared copies.
        captures: list[Fragment] = []
        self.held.append(captures)
        recipes = (
            frame_snapshot_recipes(node, self.graphs, self.env).get(
                plan.body.graph.graph_id, {}
            )
            if self.df.config.get("cute_fragment_register_snapshots", False)
            else {}
        )
        for slot, entry in enumerate(plan.body.captures):
            value = cast("Fragment", values[entry])
            binding = value.snapshot
            keep_snapshot = (
                binding is not None
                and self.df.config.get("cute_fragment_register_snapshots", False)
                and binding.source in self.snapshot_loads
                and binding.threads == self.threads
                and binding.shape == value.shape
                and binding.dtype == value.dtype
                and binding.logical_shape
                == self.shape(snapshot_logical_shape(binding.source))
                and not binding.domain_storage
                and value.resident
                and value.storage is None
                and not value.dependencies
                and slot in snapshot_capture_slots(node, self.graphs, binding.source)
            )
            derived = (
                self.derived_snapshot(value, recipes[slot])
                if not keep_snapshot and slot in recipes
                else None
            )
            captured = (
                value
                if keep_snapshot
                else derived
                if derived is not None
                else self.materialize(value, copy=True)
            )
            if plan.composed and id(value) in self.local_logical_sizes:
                # The frame proof permits only completed readonly local captures.
                # Preserve declared extent on the held shared copy, never its
                # padded capacity; vector carries remain forbidden by the plan.
                self.local_logical_sizes[id(captured)] = self.local_logical_sizes[
                    id(value)
                ]
                self.local_storage.append(captured)
            captures.append(captured)
        carries = [captures[slot] for _index, slot in plan.body.carry_map]
        predicate = self.allocate(Fragment((), torch.bool, lambda _: "False"))
        self.held.append(predicate)
        take = self.df.new_var("fragment_while_take")

        def condition() -> None:
            result = self.graph(
                plan.condition.graph.graph,
                dict(zip(plan.condition.placeholders, captures, strict=True)),
            )
            assert isinstance(result, list) and len(result) == 1
            # One owned scalar writer, followed by unconditional publication.
            self.copy(cast("Fragment", result[0]), predicate)
            self.emit(f"{take} = {predicate.read(())}")
            # Every physical thread must finish reading before any can enter
            # the next iteration and overwrite this same predicate slot.
            self.synchronize()

        condition()
        statements: list[ast.AST] = []
        with self.cg.set_statements(statements):
            results = self.graph(
                plan.body.graph.graph,
                dict(zip(plan.body.placeholders, captures, strict=True)),
            )
            assert isinstance(results, list)
            # An unused final update still belongs to this iteration. Close it
            # before carry copies or any next-iteration reset can reuse storage.
            self.synchronize_local_atomics()
            snapshots: list[Fragment] = []
            self.held.extend([results, snapshots])
            destinations = self.referenced_buffers(carries)
            for result in results:
                value = cast("Fragment", result)
                snapshots.append(
                    self.materialize(value, copy=True)
                    if self.referenced_buffers([value]) & destinations
                    else value
                )
            for target, source in zip(carries, snapshots, strict=True):
                self.copy(source, target)
            self.held.pop()
            self.held.pop()
            condition()
        loop = statement_from_string(f"while {take}:\n    pass")
        assert isinstance(loop, ast.While)
        loop.body = cast("list[ast.stmt]", statements)
        self.cg.add_statement(loop)
        self.held.pop()
        self.held.pop()
        return carries

    def uniform_finalizer_symbols(
        self, nodes: frozenset[Node], loop_blocks: set[int] | None = None
    ) -> None:
        for node in nodes:
            value = node.meta["val"]
            expression = value._sympy_() if isinstance(value, torch.SymInt) else value
            for symbol in sympy.sympify(expression).free_symbols:
                entry = HostFunction.current().expr_to_origin.get(symbol)
                origin = entry.origin if entry is not None else None
                if not isinstance(origin, HostOrigin) and not (
                    type(origin) is GridOrigin
                    and (
                        origin.block_id in self.offsets
                        or loop_blocks is not None
                        and origin.block_id in loop_blocks
                    )
                    and self.df.resolved_block_size(origin.block_id) == 1
                ):
                    raise exc.InvalidConfig(
                        "local finalizer requires proved uniform scalar origins"
                    )

    def conditional(self, node: Node, values: dict[Node, object]) -> list[Fragment]:
        composed_tree = None
        try:
            local_inputs = (
                uniform_local_branch_inputs(node, self.graphs)
                if self.local_allocations
                else None
            )
        except exc.InvalidConfig:
            local_inputs = None
        if local_inputs is None and self.local_allocations:
            from .uniform_region_tree import uniform_local_regions

            try:
                composed_tree = uniform_local_regions(self.graphs)
            except exc.InvalidConfig:
                pass
            else:
                members = [n for f in composed_tree.frames for n in f.graph.graph.nodes]
                local_inputs = (
                    frozenset(
                        n
                        for n in members
                        if isinstance(n.meta.get("lowering"), ReductionLowering)
                    ),
                    frozenset(
                        n for n in members if n.target is _tracing_ops._get_symnode
                    ),
                    frozenset(
                        n
                        for n in members
                        if n.target is memory_ops.load
                        and n not in {read.node for read in composed_tree.indexed_reads}
                    ),
                )
        if local_inputs is not None:
            reductions, symbols, loads = local_inputs
            if any(
                not host_load_is_readonly(load, self.env, self.graphs) for load in loads
            ):
                raise exc.InvalidConfig(
                    "local-buffer predicate requires readonly bound host storage"
                )
            self.uniform_finalizer_symbols(symbols)
            if reductions & self.warp_result_nodes.keys():
                raise exc.InvalidConfig(
                    "local-buffer predicate requires CTA-shared reduction ownership"
                )
            predicate = cast("Node", node.args[0])
            value = values[predicate]
            if not isinstance(value, Fragment) or value.shape != ():
                raise exc.InvalidConfig("local-buffer predicate must be scalar")
            # Even a one-element reduction may remain lazy. Evaluate exactly once
            # and exchange before all CTA threads read the same scalar address.
            values[predicate] = self.materialize(value, copy=True)
            self.held.append(
                [
                    values[predicate],
                    *(
                        values[source]
                        for side in node.args[3:5]
                        for source in cast("list[Node]", side)
                    ),
                ]
            )
        elif self.local_allocations:
            tickets, symbols = terminal_finalizer_inputs(node, self.graphs)
            predicate_value = values[cast("Node", node.args[0])]
            predicate_buffers = self.referenced_buffers([predicate_value])
            for ticket in tickets:
                result = self.scalar_ordered_tickets.get(ticket)
                if (
                    result is None
                    or result.shape != ()
                    or not result.resident
                    or result.storage is None
                    or result.storage not in predicate_buffers
                    or ticket in self.warp_result_nodes
                ):
                    raise exc.InvalidConfig(
                        "local finalizer requires one shared scalar ticket and CTA broadcast"
                    )
            self.uniform_finalizer_symbols(symbols)
        info = self.graphs[cast("int", node.args[1])]
        assert isinstance(info, IfGraphInfo)
        assert info.branches_outputs is not None
        # An unchanged branch output can be captured only by the other branch.
        # Both lists bind values in this outer graph, not branch-local results.
        outer_captures: dict[str, Node] = {}
        for side, names in enumerate((info.if_arg_names, info.else_arg_names)):
            assert names is not None
            for name, source in zip(
                names, cast("list[Node]", node.args[3 + side]), strict=True
            ):
                if name in outer_captures and outer_captures[name] is not source:
                    raise exc.InvalidConfig(
                        "computed fragment conditional has conflicting outer captures"
                    )
                outer_captures[name] = source
        for slots in info.branches_outputs:
            if any(
                isinstance(slot, str) and slot not in outer_captures for slot in slots
            ):
                raise exc.InvalidConfig(
                    "computed fragment conditional requires a captured unchanged output"
                )
        # Both parent result halves name the same merged fragments. Only these
        # declared integer slots escape each branch; other returns are tracing
        # bookkeeping. All branch computation and side effects still execute.
        output_indices = tuple(
            frozenset(
                slot
                for slots in info.branches_outputs
                for slot in (slots[side],)
                if isinstance(slot, int)
            )
            for side in range(2)
        )
        merged: list[Fragment] = []
        self.held.append(merged)
        fake_outputs = cast("list[torch.Tensor]", node.meta["val"])
        for fake in fake_outputs[: len(info.branches_outputs)]:
            merged.append(
                self.allocate(
                    Fragment(self.shape(fake.shape), fake.dtype, lambda _: "0")
                )
            )
        branch_local = local_inputs is not None and any(
            allocation.graph is self.graphs[cast("int", graph_id)].graph
            for allocation in self.local_allocations
            for graph_id in node.args[1:3]
        )
        if composed_tree is not None:
            # Descendant fresh epochs require the same per-arm retirement even
            # when an immediate arm owns no allocation itself.
            branch_local = True
        predicate = node.args[0]
        proxy = predicate.meta["val"] if isinstance(predicate, Node) else predicate
        constant = self.df.evaluate_constexpr_condition(proxy)
        control_snapshots: dict[int, str] = {}
        test = (
            repr(constant)
            if constant is not None
            else self._control_scalar(
                values[predicate] if isinstance(predicate, Node) else predicate,
                control_snapshots,
                predicate=True,
            )
        )
        if control_snapshots:
            # A barrier in a branch cannot protect a warp that skips that arm.
            self.synchronize()
        incoming_pending = self.pending_local_atomics.copy()
        branches: list[list[ast.AST]] = []
        for side in range(2):
            if composed_tree is not None:
                self.pending_local_atomics = incoming_pending.copy()
            branch = self.graphs[cast("int", node.args[1 + side])]
            captures = [
                values[source] for source in cast("list[Node]", node.args[3 + side])
            ]
            body: list[ast.AST] = []
            with self.cg.set_statements(body):
                outputs = self.graph(
                    branch.graph,
                    dict(
                        zip(
                            branch.graph.find_nodes(op="placeholder"),
                            captures,
                            strict=True,
                        )
                    ),
                    output_indices=output_indices[side],
                )
                if branch_local:
                    # Each arm starts with no pending incoming atomic epoch.
                    # An unused final update must complete inside this arm, not
                    # borrow a barrier emitted while generating its sibling.
                    self.synchronize_local_atomics()
                assert isinstance(outputs, list)
                self.held.append(outputs)
                for target, slots in zip(merged, info.branches_outputs, strict=True):
                    slot = slots[side]
                    source = (
                        outputs[slot]
                        if isinstance(slot, int)
                        else values[outer_captures[slot]]
                    )
                    assert isinstance(source, Fragment)
                    self.copy(source, target)
                self.held.pop()
            branches.append(body or [ast.Pass()])
        statement = statement_from_string(f"if {test}:\n    pass")
        assert isinstance(statement, ast.If)
        statement.body = cast("list[ast.stmt]", branches[0])
        statement.orelse = cast("list[ast.stmt]", branches[1])
        self.cg.add_statement(statement)
        if branch_local:
            # Keep predicate/captures alive until all CTA threads have left the
            # selected arm, before any branch-local storage lifetime can end.
            self.synchronize()
        self.held.pop()
        if local_inputs is not None:
            self.held.pop()
        return [*merged, *merged]
