"""Nested control flow of a computed fragment: loops, branches, resident while.

``FragmentControlFlow`` lowers ``hl.tile`` / ``for`` loops with carried
fragments, data-dependent ``while`` loops over register-resident state and
``if`` graphs whose outputs join in shared memory; each body is walked through
``FragmentContext.graph``."""

from __future__ import annotations

import ast
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
from .fragment_context import Fragment
from .fragment_context import FragmentContext
from .fragment_context import _loop_carry_slots
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


class FragmentControlFlow(FragmentContext):
    """Loops, resident while loops and conditionals over fragments."""

    def loop(self, node: Node, values: dict[Node, object]) -> list[Fragment]:
        graph_id = node.args[0]
        assert isinstance(graph_id, int)
        graph = self.graphs[graph_id]
        assert isinstance(graph, ForLoopGraphInfo)
        if self.local_allocations and any(
            self.df.resolved_block_size(bid) != 1 for bid in graph.block_ids
        ):
            raise exc.InvalidConfig("CTA-local atomics require uniform scalar loops")
        loop_index_type = "Int32"
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
                loop_index_type = "Int64"
        args = [values[x] for x in cast("list[Node]", node.args[3])]
        placeholders = list(graph.graph.find_nodes(op="placeholder"))
        output_node = graph.graph.find_nodes(op="output")[0]
        outputs = cast("list[Node]", output_node.args[0])
        carry_slots = _loop_carry_slots(node)
        # Match each output to its captured value through the actual phi edge.
        # Read-only captures can appear before, between, or after mutable ones.
        carries: list[Fragment] = []
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
            else:
                fake = cast("torch.Tensor", output.meta["val"])
                carry = self.allocate(
                    Fragment(self.shape(fake.shape), fake.dtype, lambda _: "0")
                )
            carries.append(carry)
        body: list[ast.AST] = []
        deferred_barrier: ast.AST | None = None
        outer_offsets, outer_bounds = dict(self.offsets), dict(self.bounds)
        with self.cg.set_statements(body):
            for bid, _begin, end in zip(
                graph.block_ids,
                cast("list[object]", node.args[1]),
                cast("list[object]", node.args[2]),
                strict=True,
            ):
                self.offsets[bid] = self.df.new_var("fragment_tile")
                self.bounds[bid] = self.scalar(
                    values[end] if isinstance(end, Node) else end
                )
            results = self.graph(
                graph.graph, dict(zip(placeholders, inner_args, strict=True))
            )
            assert isinstance(results, list)
            # Snapshot every value that reads a carry before writing any carry.
            # Disjoint recipes can write directly: none of their inputs changes
            # during the parallel assignment. Keep both sets live throughout.
            snapshots = []
            self.held.extend([results, snapshots])
            destinations = self.referenced_buffers(carries)
            for result in results:
                value = cast("Fragment", result)
                snapshots.append(
                    self.materialize(value, copy=True)
                    if self.referenced_buffers([value]) & destinations
                    and id(value) not in self.local_logical_sizes
                    else value
                )
            for target, source in zip(carries, snapshots, strict=True):
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
            begin = cast("list[object]", node.args[1])[index]
            start = self.scalar(values[begin] if isinstance(begin, Node) else begin)
            step = self.df.resolved_block_size(bid)
            if node.target is _tracing_ops._for_loop_step:
                explicit = cast("list[object]", node.args[4])[index]
                if explicit is not None:
                    step = self.scalar(
                        values[explicit] if isinstance(explicit, Node) else explicit
                    )
            loop = statement_from_string(
                f"for {self.offsets[bid]} in range(cutlass.{loop_index_type}({start}), cutlass.{loop_index_type}({self.bounds[bid]}), {step}):\n    pass"
            )
            assert isinstance(loop, ast.For)
            loop.body = cast("list[ast.stmt]", body)
            body = [loop]
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
        predicate = node.args[0]
        proxy = predicate.meta["val"] if isinstance(predicate, Node) else predicate
        constant = self.df.evaluate_constexpr_condition(proxy)
        test = (
            repr(constant)
            if constant is not None
            else self.scalar(
                values[predicate] if isinstance(predicate, Node) else predicate
            )
        )
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
