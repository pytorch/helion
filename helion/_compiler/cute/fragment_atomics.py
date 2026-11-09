"""Atomic read-modify-write emission for computed fragments.

``FragmentAtomicEmitter`` lowers ``hl.atomic_add`` on a fragment root: the
per-element ``cute.arch.atomic_add`` walk, CTA-local aggregation into shared
memory (``cute_fragment_local_atomics``), register-held local updates and the
fused consumer regions that read an atomic's completed value."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from .fragment_context import Fragment
from .fragment_context import FragmentContext
from .fragment_context import HostTensor
from .fragment_context import _resolve
from .fragment_expression import FragmentExpression
from .local_atomic_registers import MAX_SLOTS

if TYPE_CHECKING:
    import ast
    from collections.abc import Callable

    from torch.fx import Node

    from .atomic_consumer_fusion import AtomicConsumerRegion


class FragmentAtomicEmitter(FragmentContext):
    """Atomic updates of host tensors from a fragment root."""

    def aggregate_local_updates(
        self,
        shape: tuple[int, ...],
        storage: str,
        variables: tuple[str, ...],
        prepare: Callable[[tuple[str, ...]], None],
        *,
        register_owned: bool = False,
    ) -> None:
        """Aggregate one private relaxed Int32 update epoch, without new barriers.

        All CTA threads enter every round. Only lanes in the ballot execute
        match/redux, using the same member mask for each equal wrapped address.
        No observer can read the private target until its existing epoch barrier.
        """
        assert self.threads % 32 == 0
        register_owned = register_owned or shape in self.snapshot_shapes
        size = math.prod(shape)
        iteration = self.df.new_var("fragment_atomic_round")
        index = self.df.new_var("fragment_atomic_element")
        address, value, active = variables
        slots = (size + self.threads - 1) // self.threads
        if register_owned:
            assert 0 < slots <= MAX_SLOTS
        loop_range = "cutlass.range_constexpr" if register_owned else "range"
        loop = cast(
            "ast.For",
            statement_from_string(
                f"for {iteration} in {loop_range}({slots}):\n    pass"
            ),
        )
        loop.body.clear()
        with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
            self.emit(f"{index} = {iteration} * {self.threads} + {self.thread}")
            # Dominating definitions are required by CuTe's staged SSA lowering.
            self.emit(f"{address} = cutlass.Int64(0)")
            self.emit(f"{value} = cutlass.Int32(0)")
            self.emit(f"{active} = False")
            valid = cast(
                "ast.If", statement_from_string(f"if {index} < {size}:\n    pass")
            )
            valid.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", valid.body)):
                prior = self.local_register_slot
                if register_owned:
                    self.local_register_slot = iteration
                try:
                    with self.snapshot_scope(index, iteration, size):
                        prepare(self.coordinates(index, shape))
                finally:
                    self.local_register_slot = prior
            self.cg.add_statement(valid)
            members = self.df.new_var("fragment_atomic_members")
            peers = self.df.new_var("fragment_atomic_peers")
            total = self.df.new_var("fragment_atomic_total")
            self.emit(f"{members} = cute.arch.vote_ballot_sync({active})")
            self.emit(
                f"if {active}:\n"
                f"    {peers} = cute.arch.match_sync({members}, {address})\n"
                f"    {total} = cute.arch.warp_redux_sync({value}, 'add', {peers})\n"
                f"    if ({peers} & cute.arch.lanemask_lt()) == 0:\n"
                f"        cute.arch.atomic_add(({storage}.iterator + {address}).llvm_ptr, "
                f"{total}, sem='relaxed', scope='cta')"
            )
        self.cg.add_statement(loop)

    def atomic_add(
        self,
        node: Node,
        values: dict[Node, object],
        deferred_updates: list[tuple[Callable[[tuple[str, ...]], None], bool]]
        | None = None,
    ) -> Fragment | None:
        """Each contribution is owned once; barriers are outside lane loops."""
        target, indices, value, sem = cast(
            "tuple[object, object, object, object]", _resolve(node.args, values)
        )
        assert isinstance(target, (Fragment, HostTensor))
        assert isinstance(indices, (tuple, list))
        fake = cast("torch.Tensor", node.meta["val"])
        shape = self.shape(fake.shape)
        target_fake = cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])
        if len(indices) != target_fake.ndim or any(
            isinstance(i, slice) or i is None for i in indices
        ):
            raise exc.InvalidConfig(
                "fragment atomics require scalar/tensor indices for every axis"
            )
        local = isinstance(target, Fragment)
        if sem != "relaxed" and (local or target_fake.dtype != torch.int32):
            raise exc.InvalidConfig(
                "ordered fragment atomics require a global int32 target"
            )
        if node.users and target_fake.dtype != torch.int32:
            raise exc.InvalidConfig(
                "fragment atomic results require a global int32 target "
                "or a relaxed local int32 target"
            )
        if isinstance(target, Fragment) and (
            target.storage is None or len(target.shape) != 1
        ):
            raise exc.InvalidConfig("CTA-local atomics require direct resident storage")
        if target_fake.dtype not in (torch.float32, torch.int32):
            raise exc.InvalidConfig("fragment atomic add supports float32/int32")

        if sem in ("release", "acq_rel"):
            # Each worker publishes its preceding global writes before any
            # elected scalar/vector owner performs the release operation.
            # A fence in only the owner would not publish other workers' writes.
            # Both operations are collective, outside contribution masks/loops.
            self.emit("cute.arch.fence_acq_rel_gpu()")
            self.synchronize()

        def logical_domain(coords: tuple[str, ...]) -> tuple[str, ...]:
            masks = [
                f"({coord}) < ({self.logical_axis_extent(fake, dim, shape[dim])})"
                for dim, coord in enumerate(coords)
            ]
            # Indexing can pad the result shape after an index tensor has
            # retained its concrete logical extent. Destination bounds do not
            # exclude those extra contributions when an index wraps or clamps.
            # Preserve each index operand's broadcast domain independently.
            index_nodes = cast("list[Node | int]", node.args[1])
            for source, index in zip(index_nodes, indices, strict=True):
                if not isinstance(index, Fragment):
                    continue
                index_fake = cast("torch.Tensor", cast("Node", source).meta["val"])
                offset = len(coords) - len(index.shape)
                for dim, capacity in enumerate(index.shape):
                    if capacity == 1:
                        continue  # A singleton broadcasts over the output axis.
                    extent = self.logical_axis_extent(index_fake, dim, capacity)
                    mask = f"({coords[offset + dim]}) < ({extent})"
                    if mask not in masks:
                        masks.append(mask)
                if index.logical_domain is not None:
                    for mask in index.broadcast_domain(coords):
                        if mask not in masks:
                            masks.append(mask)
            return tuple(masks)

        # The side effect is emitted once below, never embedded in a lazy
        # element recipe. Repeated consumers read this immutable snapshot.
        warp_groups = self.warp_result_nodes.get(node)
        warp_owned = (
            not local and warp_groups is not None and math.prod(shape) == warp_groups
        )
        register = (
            self.df.new_var("fragment_warp_atomic")
            if node.users and warp_owned
            else None
        )
        if register is not None:
            self.emit(f"{register} = {self.cast('0', target_fake.dtype)}")
        iteration_ticket = (
            self.df.new_var("fragment_iteration_ticket")
            if node in self.iteration_local_returns
            else None
        )
        if iteration_ticket is not None:
            assert deferred_updates is not None and local and node.users
            assert sem == "relaxed" and target_fake.dtype == torch.int32
            assert len(shape) == 1 and not warp_owned
        local_registers = (
            self.df.new_var("fragment_local_tickets")
            if local
            and node.users
            and node in self.local_register_nodes
            and iteration_ticket is None
            else None
        )
        if local_registers is not None:
            slots = (math.prod(shape) + self.threads - 1) // self.threads
            self.emit(
                f"{local_registers} = cute.make_rmem_tensor(({slots},), cutlass.Int32)"
            )
            self.emit(f"{local_registers}.fill(0)")

        def local_register_read(_coords: tuple[str, ...]) -> str:
            assert self.local_register_slot is not None
            return f"{local_registers}[{self.local_register_slot}]"

        def iteration_read(coords: tuple[str, ...]) -> str:
            assert self.iteration_owner is not None
            assert node in self.iteration_local_returns
            assert node in self.iteration_defined
            self.iteration_owner.prove(coords[0])
            assert iteration_ticket is not None
            return iteration_ticket

        result = (
            Fragment(
                shape,
                target_fake.dtype,
                iteration_read,
                True,
                logical_domain=logical_domain,
            )
            if iteration_ticket is not None
            else Fragment(
                shape,
                target_fake.dtype,
                local_register_read,
                True,
                logical_domain=logical_domain,
            )
            if local_registers is not None
            else Fragment(
                shape,
                target_fake.dtype,
                lambda _coords: cast("str", register),
                True,
                logical_domain=logical_domain,
            )
            if register is not None
            else self.allocate(
                Fragment(
                    shape,
                    target_fake.dtype,
                    lambda coords: "0",
                    logical_domain=logical_domain,
                )
            )
            if node.users
            else None
        )

        aggregate = (
            local
            and target_fake.dtype == torch.int32
            and sem == "relaxed"
            and not node.users
            and not warp_owned
            and self.df.config.get("cute_fragment_atomic_aggregation", False) is True
        )
        if aggregate:
            root = self.cg.current_root_graph_info
            assert root is not None
            aggregate = (
                root.graph_id
                in self.env.config_spec.cute_fragment_atomic_aggregation_root_ids
            )
        # Zero has no counter effect; returned zeros additionally require every
        # observable consumer to be disabled. Keep masks, value conversion,
        # result snapshot writes and epoch publication when updates are skipped.
        skip_zero = (
            local
            and target_fake.dtype == torch.int32
            and sem == "relaxed"
            and (not node.users or node in self.dead_zero_results)
            and self.df.config.get("cute_fragment_skip_zero_atomics", False) is True
        )
        if skip_zero:
            root = self.cg.current_root_graph_info
            assert root is not None
            skip_zero = (
                root.graph_id
                in self.env.config_spec.cute_fragment_skip_zero_atomics_root_ids
            )
        aggregate_vars = (
            tuple(
                self.df.new_var(name)
                for name in (
                    "fragment_atomic_address",
                    "fragment_atomic_value",
                    "fragment_atomic_active",
                )
            )
            if aggregate
            else None
        )

        def update_in_domain(coords: tuple[str, ...], previous: str | None) -> None:
            masks = []
            positions = []
            for dim, index in enumerate(indices):
                position = (
                    index.broadcast(coords)
                    if isinstance(index, Fragment)
                    else self.scalar(index)
                )
                position = self.cg.lift(
                    expr_from_string(position), prefix="fragment_atomic_index"
                ).id
                # The logical allocation can be smaller than its padded capacity.
                size = self.sym(target_fake.shape[dim])
                if local:
                    assert isinstance(target, Fragment) and target.storage is not None
                    size = str(self.local_logical_sizes[id(target)])
                logical_extent = (
                    self.local_logical_sizes[id(target)]
                    if isinstance(target, Fragment)
                    else target_fake.shape[dim]
                )
                if (
                    isinstance(index, int)
                    and isinstance(logical_extent, int)
                    and not -logical_extent <= index < logical_extent
                ):
                    raise exc.InvalidConfig(
                        "fragment atomic static index is outside the logical target"
                    )
                # Python/PyTorch indexing wraps one valid negative extent.
                # Widen before adding a possibly 64-bit host extent; this is
                # address arithmetic, not a conversion of the update value.
                wrapped = self.df.new_var("fragment_atomic_wrapped")
                self.emit(f"{wrapped} = cutlass.Int64({position})")
                self.emit(
                    f"if {wrapped} < 0:\n    {wrapped} = {wrapped} + cutlass.Int64({size})"
                )
                masks.append(f"0 <= ({wrapped}) and ({wrapped}) < ({size})")
                positions.append(wrapped)
            if isinstance(target, Fragment):
                pointer = f"({target.storage}.iterator + ({positions[0]})).llvm_ptr"
                scope = "cta"
            else:
                offset = " + ".join(
                    f"cutlass.Int64({position}) * cutlass.Int64({self.host_stride(target, dim)})"
                    for dim, position in enumerate(positions)
                )
                pointer = f"({target.name}.iterator + ({offset})).llvm_ptr"
                scope = "gpu"
            contribution = (
                value.broadcast(coords)
                if isinstance(value, Fragment)
                else self.scalar(value)
            )
            if aggregate_vars is not None:
                address, update_value, active = aggregate_vars
                self.emit(f"{address} = cutlass.Int64({positions[0]})")
                self.emit(
                    f"if {self.predicate(masks)}:\n"
                    f"    {update_value} = {self.cast(contribution, torch.int32)}\n"
                    f"    {active} = {update_value} != 0"
                )
                return
            atomic = (
                f"cute.arch.atomic_add({pointer}, "
                f"{self.cast(contribution, target_fake.dtype)}, "
                f"sem={sem!r}, scope={scope!r})"
            )
            if skip_zero and result is None:
                update_value = self.df.new_var("fragment_atomic_nonzero_value")
                self.emit(
                    f"if {self.predicate(masks)}:\n"
                    f"    {update_value} = {self.cast(contribution, torch.int32)}\n"
                    f"    if {update_value} != cutlass.Int32(0):\n"
                    f"        cute.arch.atomic_add({pointer}, {update_value}, "
                    f"sem={sem!r}, scope={scope!r})"
                )
            elif result is None:
                self.emit(f"if {self.predicate(masks)}:\n    {atomic}")
            else:
                assert previous is not None
                if skip_zero:
                    update_value = self.df.new_var("fragment_atomic_nonzero_value")
                    self.emit(
                        f"if {self.predicate(masks)}:\n"
                        f"    {update_value} = {self.cast(contribution, torch.int32)}\n"
                        f"    if {update_value} != cutlass.Int32(0):\n"
                        f"        {previous} = cute.arch.atomic_add({pointer}, "
                        f"{update_value}, sem={sem!r}, scope={scope!r})"
                    )
                else:
                    self.emit(f"if {self.predicate(masks)}:\n    {previous} = {atomic}")

        def update(coords: tuple[str, ...]) -> None:
            previous = None
            if result is not None:
                previous = iteration_ticket or self.df.new_var(
                    "fragment_atomic_previous"
                )
                self.emit(f"{previous} = {self.cast('0', target_fake.dtype)}")

            # Evaluating a lazy index can read a captured tensor whose declared
            # extent is smaller than the padded atomic iteration domain. Check
            # the contribution domain before evaluating that index, rather than
            # masking only the final atomic after the index has already read it.
            masks = logical_domain(coords)
            if masks:
                branch = cast(
                    "ast.If",
                    statement_from_string(f"if {self.predicate(masks)}:\n    pass"),
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    update_in_domain(coords, previous)
                self.cg.add_statement(branch)
            else:
                update_in_domain(coords, previous)

            if result is not None:
                if iteration_ticket is not None:
                    self.iteration_defined.add(node)
                else:
                    # Invalid contributions still produce the existing zero
                    # result, including padded lanes of shared snapshots.
                    self.emit(f"{result.read(coords)} = {previous}")

        if deferred_updates is not None:
            assert local and not warp_owned
            assert aggregate_vars is None and target_fake.dtype == torch.int32
            deferred_updates.append((update, result is not None))
        elif aggregate_vars is None:
            # Returned local values are immutable per-element snapshots. Finish
            # every worker's update/result write before a cross-lane consumer;
            # the target and result have distinct live allocations. Local
            # unused-result epochs retain their existing coalescing behavior.
            self.elements(
                shape,
                update,
                warp_owned=warp_owned,
                synchronize=not local or result is not None,
                register_owned=node in self.local_register_nodes,
            )
        else:
            assert isinstance(target, Fragment) and target.storage is not None
            self.aggregate_local_updates(
                shape,
                target.storage,
                aggregate_vars,
                update,
                register_owned=node in self.local_register_nodes,
            )
        if local and result is None:
            assert isinstance(target, Fragment) and target.storage is not None
            self.pending_local_atomics.add(target.storage)
        if register is not None:
            # Preserve the existing atomic ordering barrier, including unrelated
            # alias loads. Only the return-value exchange becomes warp-local.
            self.emit(
                f"if {self.thread} // 32 < {warp_groups}:\n"
                f"    {register} = cute.arch.shuffle_sync({register}, 0, mask=0xffffffff, mask_and_clamp=31)"
            )
        if result is not None and not shape and sem in ("acquire", "acq_rel"):
            self.scalar_ordered_tickets[node] = result
        return result

    def can_fuse_atomics(
        self, region: AtomicConsumerRegion, values: dict[Node, object]
    ) -> bool:
        if self.df.config.get("cute_fragment_atomic_aggregation", False):
            return False
        shapes = set()
        storages = set()
        for node in region.atomics:
            fake = cast("torch.Tensor", node.meta["val"])
            target = values.get(cast("Node", node.args[0]))
            if (
                node in self.warp_result_nodes
                or not isinstance(target, Fragment)
                or target.storage is None
                or len(target.shape) != 1
                or id(target) not in self.local_logical_sizes
                or target.storage in storages
            ):
                return False
            storages.add(target.storage)
            shapes.add(self.shape(fake.shape))
        if len(shapes) != 1:
            return False
        external = {
            arg
            for node in region.nodes
            for arg in node.all_input_nodes
            if arg not in region.nodes and arg not in region.targets
        }
        if (
            self.referenced_buffers([values[arg] for arg in external if arg in values])
            & storages
        ):
            return False
        shape = next(iter(shapes))
        register_owned = any(node in self.local_register_nodes for node in region.nodes)
        if register_owned and math.prod(shape) > self.threads * MAX_SLOTS:
            return False
        return not any(node in self.warp_result_nodes for node in region.nodes)

    def fused_atomic_updates(
        self,
        region: AtomicConsumerRegion,
        updates: list[tuple[Callable[[tuple[str, ...]], None], bool]],
    ) -> None:
        assert len(updates) == len(region.atomics) + len(region.stores)
        shape = self.shape(cast("torch.Tensor", region.atomics[0].meta["val"]).shape)

        def body(coords: tuple[str, ...]) -> None:
            previous = self.expression
            # Only this proved immutable dependency closure survives the
            # individual updates. Producer objects are current SSA recipes;
            # their values include every original logical-dtype cast.
            self.expression = FragmentExpression(self.cg)
            try:
                for update, _synchronize in updates:
                    update(coords)
            finally:
                self.expression = previous

        self.elements(
            shape,
            body,
            synchronize=False,
            register_owned=any(
                node in self.local_register_nodes for node in region.nodes
            ),
            iteration_local=bool(region.local_returns),
        )
        pending = self.pending_local_atomics.copy()
        for _update, synchronize in updates:
            if synchronize:
                self.synchronize()
        # Keep the original conservative post-region publication obligations.
        # Removing these now-redundant boundaries is a separate optimization.
        self.pending_local_atomics.update(pending)
