"""Host memory traffic of a computed fragment: loads, stores, packets and slabs.

``FragmentMemoryEmitter`` stages host tensor windows into shared memory and
writes staged tiles back: scalar element walks, 16-byte row packets
(``cute_fragment_vector_io``), batched load phases, slab views
(``cute_fragment_slab_loads``) and the completed-local-atomic load."""

from __future__ import annotations

import ast
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
from ..ast_extension import statement_from_string
from ..host_function import HostFunction
from ..indexing_strategy import SubscriptIndexing
from ..variable_origin import GridOrigin
from .emitters.memory_access import _cute_wrap_negative_index
from .fragment_context import DirectStoreSink
from .fragment_context import Fragment
from .fragment_context import FragmentContext
from .fragment_context import HostTensor
from .fragment_context import _resolve
from .fragment_indexing import memory_index_coordinates
from .fragment_slab_loads import uniform_integer_expr
from .handler_ops import scalar_value_dtype
from .local_atomic import local_indexed_load
from .register_loads import has_opaque_inline_asm
from .register_loads import host_load_is_readonly
from .register_loads import lane_private_load
from .register_snapshots import SnapshotBinding
from .register_snapshots import snapshot_logical_shape
from .scalar_fragment_assertion import check_host_subscripts

if TYPE_CHECKING:
    from collections.abc import Callable

    from .fragment_domain import OwnershipMask
    from .fragment_slab_loads import ScalarLoad
    from .fragment_slab_loads import SlabGroup


class FragmentMemoryEmitter(FragmentContext):
    """Loads and stores between host tensors and staged fragments."""

    def tile_index_and_mask(
        self,
        tensor: HostTensor,
        dim: int,
        block: int,
        coord: str,
        *,
        broadcast: bool,
    ) -> tuple[str, str]:
        index = f"({self.offsets[block]}) + ({coord})"
        mask = f"(({index}) < ({self.bounds[block]}))"
        if broadcast:
            size = tensor.value.shape[dim]
            if self.env.known_equal(size, 1):
                return "0", self.tile_lane_predicate(block)(coord)
        return index, mask

    def row_packet_lanes(
        self,
        tensor: HostTensor,
        indices: list[object],
        values: dict[Node, object],
        shape: tuple[int, ...],
        dtype: torch.dtype,
        index_coordinates: list[tuple[int | None, ...]],
    ) -> int | None:
        """``cute_fragment_vector_io``: the 16-byte packet width (in elements)
        when this host window's innermost coordinate is proved to walk whole,
        aligned, in-bounds packets of a unit-stride dimension; None otherwise.
        A staged destination must also keep its packets contiguous."""
        if not self.vector_io:
            return None
        from .fragment_vector_io import row_packet_lanes

        lanes = row_packet_lanes(
            self, tensor, indices, values, shape, dtype, index_coordinates
        )
        if lanes is not None and self.physical_shape(shape, dtype)[-1] % lanes:
            return None
        return lanes

    def vector_load(
        self,
        value: Fragment,
        shape: tuple[int, ...],
        lanes: int,
        address: Callable[[tuple[str, ...]], tuple[str, str]],
        *,
        batched: bool = False,
    ) -> Fragment:
        """Stage *value* (a host window) through 16-byte packets: one address
        and one predicate per packet, a vector global load into registers
        (zeros when the packet is out of bounds, like the scalar walk) and a
        vector store into the dense staged rows."""
        from .fragment_vector_io import copy_atom

        self.held.append(value)
        result = self.allocate(value)
        self.held.pop()
        physical = self.physical_shape(shape, value.dtype)
        scalar = self.dtype(value.dtype)
        atom = copy_atom(scalar)
        assert result.storage is not None

        def body(coords: tuple[str, ...]) -> None:
            inner = self.df.new_var("fragment_packet_inner")
            self.emit(f"{inner} = ({coords[-1]}) * {lanes}")
            pointer, mask = address((*coords[:-1], inner))
            packet = self.df.new_var("fragment_packet")
            self.emit(
                f"{packet} = cute.make_rmem_tensor(cute.make_layout(({lanes},)), {scalar})"
            )
            self.emit(f"{packet}.fill({scalar}(0))")
            self.emit(
                f"if {mask}:\n"
                f"    cute.copy({atom}, cute.make_tensor({pointer}.align({lanes * value.dtype.itemsize}), cute.make_layout(({lanes},))), {packet})"
            )
            flat = self.flatten((*coords[:-1], inner), physical)
            self.emit(
                f"cute.copy({atom}, {packet}, cute.make_tensor({result.storage}.iterator + cute.assume({flat}, divby={lanes}), cute.make_layout(({lanes},))))"
            )

        self.elements(
            (*shape[:-1], shape[-1] // lanes),
            body,
            synchronize=not batched,
            batched=batched,
        )
        return result

    def vector_store(
        self,
        tensor: HostTensor,
        stored_value: Callable[[tuple[str, ...]], str],
        shape: tuple[int, ...],
        lanes: int,
        address: Callable[[tuple[str, ...]], tuple[str, str]],
    ) -> None:
        """Store a window through 16-byte packets: the value recipe is
        evaluated per lane into registers, then one predicated vector store."""
        from .fragment_vector_io import copy_atom

        scalar = self.dtype(tensor.value.dtype)
        atom = copy_atom(scalar)

        def body(coords: tuple[str, ...]) -> None:
            inner = self.df.new_var("fragment_packet_inner")
            self.emit(f"{inner} = ({coords[-1]}) * {lanes}")
            pointer, mask = address((*coords[:-1], inner))
            packet = self.df.new_var("fragment_packet")
            self.emit(
                f"{packet} = cute.make_rmem_tensor(cute.make_layout(({lanes},)), {scalar})"
            )
            for lane in range(lanes):
                lane_coords = (*coords[:-1], f"{inner} + {lane}")
                self.emit(f"{packet}[{lane}] = {stored_value(lane_coords)}")
            self.emit(
                f"if {mask}:\n"
                f"    cute.copy({atom}, {packet}, cute.make_tensor({pointer}.align({lanes * tensor.value.dtype.itemsize}), cute.make_layout(({lanes},))))"
            )

        self.elements((*shape[:-1], shape[-1] // lanes), body)

    def stage_slabs(self) -> None:
        """``cute_fragment_slab_loads``: stage every planned slab at the start
        of the root, all loads in flight together, so each member window
        reads a view of its slab wherever (root body or a conditional tier)
        it appears."""
        groups: list[SlabGroup] = []
        for group in self.slab_plan.values():
            if all(group is not other for other in groups):
                groups.append(group)
        for group in groups:
            fake = cast("torch.Tensor", group.tensor.meta["val"])
            # Only the root's own grid blocks have CTA-uniform offsets here;
            # a window under a device loop's block keeps its own walk.
            staged = all(
                (kind != "block" or value in self.offsets)
                and (kind != "slice" or self.static_extent(fake.shape[dim]) is not None)
                and (kind != "uniform" or self.uniform_at_root(value))
                for dim, (kind, value) in enumerate(group.descriptors)
            ) and all(
                self.uniform_at_root(expr)
                for load in (group.offset, group.limit)
                if load is not None
                for expr in load.indices
            )
            if not staged:
                for member in group.members:
                    cast("dict[Node, SlabGroup]", self.slab_plan).pop(member, None)
                continue
            for load in (group.offset, group.limit):
                if load is not None:
                    self.slab_scalar(load)
            group.fragment = self.load_slab(group)
            # Alive for the whole root: no later phase may recycle it.
            self.held.append(group.fragment)
        self.flush_pending_barrier()

    def uniform_at_root(self, expr: object) -> bool:
        """Whether the uniform integer expression ``expr`` is defined at the
        start of the root: its grid symbols are the root's own blocks (a
        device loop's tile symbols are not yet bound there)."""
        if not isinstance(expr, sympy.Expr):
            return False
        origins = HostFunction.current().expr_to_origin
        for symbol in expr.free_symbols:
            info = origins.get(symbol)
            origin = info.origin if info is not None else None
            if isinstance(origin, GridOrigin) and origin.block_id not in self.offsets:
                return False
        return True

    def slab_scalar(self, load: ScalarLoad) -> str:
        """The uniform 0-d integer ``load`` (a slab's row offset or limit) as
        a register of its promoted operand type. As for the body's own load, an
        out-of-bounds subscript yields zero."""
        if load in self.slab_scalars:
            return self.slab_scalars[load]
        fake = cast("torch.Tensor", load.node.meta["val"])
        tensor = HostTensor(
            self.df.tensor_arg(fake, prefer_name=load.tensor).name, fake
        )
        indices = [self.sym(expr) for expr in load.indices]
        masks = [
            f"(({index}) >= 0 and ({index}) < ({self.sym(fake.shape[dim])}))"
            for dim, index in enumerate(indices)
        ]
        offset = (
            " + ".join(
                f"cutlass.Int64({index}) * cutlass.Int64({self.host_stride(tensor, dim)})"
                for dim, index in enumerate(indices)
            )
            or "0"
        )
        name = self.df.new_var("fragment_slab_scalar")
        dtype = self.dtype(load.dtype)
        self.emit(f"{name} = {dtype}(0)")
        self.emit(
            f"if {self.predicate(masks)}:\n"
            f"    {name} = {dtype}(({tensor.name}.iterator + ({offset})).load())"
        )
        self.slab_scalars[load] = name
        return name

    def slab_member(self, node: Node, original: Fragment) -> Fragment:
        """The window *node* with the same lane ownership as an ordinary load."""
        shape, dtype = original.shape, original.dtype
        group = self.slab_plan[node]
        slab = group.fragment
        assert slab is not None, "slabs are staged before the root body"
        start, extent = group.members[node]
        if extent == 0:
            # A zero-trip consumer still renders its body after the original
            # index may have died. Preserve its empty domain without reading.
            return dataclasses.replace(
                original, element=lambda _: self.cast("0", dtype)
            )
        offset = start - group.start
        # The indexed result can have a padded shape (e.g. four lanes for a
        # three-element iota). Only the logical coordinates belong to this
        # member, even if the adjacent member occupies the same slab.
        logical_shape = tuple(
            extent
            if dim == group.window_dim and extent is not None
            else slab.shape[dim]
            for dim, (kind, _) in enumerate(group.descriptors)
            if (
                extent is not None
                if dim == group.window_dim
                else kind not in ("const", "uniform")
            )
        )
        padded = [
            (axis, logical)
            for axis, (size, logical) in enumerate(
                zip(shape, logical_shape, strict=True)
            )
            if size > logical
        ]

        def view(coords: tuple[str, ...]) -> str:
            mapped: list[str] = []
            position = 0
            for dim, (kind, _value) in enumerate(group.descriptors):
                if dim == group.window_dim:
                    if extent is None:
                        # An integer index drops the dimension.
                        mapped.append(str(offset))
                    else:
                        mapped.append(f"({coords[position]}) + {offset}")
                        position += 1
                elif kind in ("const", "uniform"):
                    mapped.append("0")
                else:
                    mapped.append(coords[position])
                    position += 1
            assert position == len(coords)
            value = slab.read(tuple(mapped))
            if not padded:
                return value
            result = self.df.new_var("fragment_slab_member")
            self.emit(f"{result} = {self.cast('0', dtype)}")
            live = self.predicate(
                [
                    f"0 <= ({coords[axis]}) and ({coords[axis]}) < {size}"
                    for axis, size in padded
                ]
            )
            self.emit(f"if {live}:\n    {result} = {value}")
            return result

        # A padded member's read emits its guard at the consumer's coordinate.
        # Mark it nonresident so a scan materializes its input before the
        # backedge, while an elementwise sum can read both windows in place.
        return Fragment(
            shape,
            dtype,
            view,
            not padded,
            (slab,),
            slab.storage,
            original.logical_domain,
            cheap=not padded,
            physical_row=None if padded else slab.physical_row,
        )

    def load_slab(self, group: SlabGroup) -> Fragment:
        """Stage one slab: the member windows' union along the varying
        dimension, with the shared block / slice / constant indices."""
        from .fragment_vector_io import host_base_aligned
        from .fragment_vector_io import packet_lanes
        from .fragment_vector_io import static_strides

        fake = cast("torch.Tensor", group.tensor.meta["val"])
        tensor = HostTensor(
            self.df.tensor_arg(
                fake, prefer_name=cast("str", group.tensor.args[0])
            ).name,
            fake,
        )
        shape: list[int] = []
        for dim, (kind, value) in enumerate(group.descriptors):
            if dim == group.window_dim:
                shape.append(group.extent)
            elif kind == "block":
                shape.append(
                    self.extent(self.df.resolved_block_size(cast("int", value)))
                )
            elif kind == "slice":
                shape.append(self.extent(fake.shape[dim]))
            elif kind in ("const", "uniform"):
                shape.append(1)
            else:
                shape.append(cast("tuple[int, int]", value)[1])
        slab_shape = tuple(shape)
        offset_var = None if group.offset is None else self.slab_scalars[group.offset]
        limit_var = None if group.limit is None else self.slab_scalars[group.limit]

        def window_row(start: int, extent: int, lane: str) -> str:
            # Static windows are proved not to wrap. A device lane is Int32:
            # widen it *before* adding a start that can cross that range.
            if start < -(2**31) or start + extent - 1 > 2**31 - 1:
                return f"cutlass.Int64({start}) + cutlass.Int64({lane})"
            return f"({start}) + ({lane})"

        def address(coords: tuple[str, ...]) -> tuple[str, str]:
            indices: list[str] = []
            masks: list[str] = []
            for dim, (kind, value) in enumerate(group.descriptors):
                if dim == group.window_dim:
                    row = window_row(group.start, group.extent, coords[dim])
                    if offset_var is not None:
                        # A window at a runtime row offset: live below the
                        # limit (in the window's own coordinates) and
                        # inside the tensor; zero past either.
                        assert group.offset is not None
                        if limit_var is not None:
                            assert group.limit is not None
                            local = self.cast(row, group.limit.dtype)
                            masks.append(f"(({local}) < ({limit_var}))")
                        local = self.cast(row, group.offset.dtype)
                        row = self.cast(
                            f"({offset_var}) + ({local})", group.offset.dtype
                        )
                    indices.append(row)
                elif kind == "block":
                    expression, mask = self.tile_index_and_mask(
                        tensor,
                        dim,
                        cast("int", value),
                        coords[dim],
                        broadcast=True,
                    )
                    masks.append(mask)
                    indices.append(expression)
                elif kind == "slice":
                    indices.append(coords[dim])
                elif kind == "const":
                    indices.append(str(value))
                elif kind == "uniform":
                    indices.append(self.sym(value))
                else:
                    start, extent = cast("tuple[int, int]", value)
                    indices.append(window_row(start, extent, coords[dim]))
                # Staging can precede a source-level branch. Only omit checks
                # proved by the slab coordinates or an identical loop bound.
                if kind == "slice" or (dim == group.window_dim and offset_var is None):
                    continue
                if dim != group.window_dim and kind in ("const", "window"):
                    start, span = (
                        (cast("int", value), 1)
                        if kind == "const"
                        else cast("tuple[int, int]", value)
                    )
                    size = self.static_extent(fake.shape[dim])
                    if size is not None and start >= 0 and start + span <= size:
                        continue
                index = indices[-1]
                size_expr = self.sym(fake.shape[dim])
                masks.append(f"(({index}) >= 0)")
                if kind != "block" or self.bounds[cast("int", value)] != size_expr:
                    masks.append(f"(({index}) < ({size_expr}))")
            offset = (
                " + ".join(
                    f"cutlass.Int64({index}) * cutlass.Int64({self.host_stride(tensor, dim)})"
                    for dim, index in enumerate(indices)
                )
                or "0"
            )
            return f"({tensor.name}.iterator + ({offset}))", self.predicate(masks)

        def load(coords: tuple[str, ...]) -> str:
            pointer, mask = address(coords)
            name = self.df.new_var("fragment_load")
            self.emit(f"{name} = {self.cast('0', fake.dtype)}")
            loaded = self.cast(f"{pointer}.load()", fake.dtype)
            self.emit(f"if {mask}:\n    {name} = {loaded}")
            return name

        value = Fragment(slab_shape, fake.dtype, load)
        batched = self.batched_loads and self.warp_producer_recorder is None
        lanes = packet_lanes(fake.dtype)
        strides = static_strides(self.env, fake)
        # The slab's rows are packets when its last dim is the whole
        # contiguous dim, or a static window of it at a packet-aligned start
        # and extent inside the dim (``hl.arange(D)`` over a D-wide dim).
        # The varying dimension (``("window", None)``: the members' windows
        # along it) is walked element by element, never packed.
        inner_size = self.static_extent(fake.shape[-1])
        inner_kind, inner_value = group.descriptors[-1]
        if group.window_dim == len(slab_shape) - 1:
            inner_packets = False
        elif inner_kind == "slice":
            inner_packets = True
        elif inner_kind == "window":
            inner_start, inner_extent = cast("tuple[int, int]", inner_value)
            inner_packets = (
                lanes is not None
                and inner_size is not None
                and inner_start % lanes == 0
                and inner_extent % lanes == 0
                and inner_start + inner_extent <= inner_size
            )
        else:
            inner_packets = False
        if (
            self.vector_io
            and lanes is not None
            and strides is not None
            and inner_packets
            and strides[-1] == 1
            and all(stride % lanes == 0 for stride in strides[:-1])
            and slab_shape[-1] % lanes == 0
            and self.physical_shape(slab_shape, fake.dtype)[-1] % lanes == 0
            and host_base_aligned(self.env, fake)
        ):
            result = self.vector_load(
                value, slab_shape, lanes, address, batched=batched
            )
        else:
            result = self.materialize(value, batched=batched)
        self.pending_barrier = batched
        return result

    def completed_local_load(
        self, node: Node, values: dict[Node, object], source: Fragment
    ) -> Fragment:
        """Snapshot a masked indexed read after the target's final atomic epoch."""
        assert id(source) in self.local_logical_sizes and source.storage is not None
        assert local_indexed_load(node)
        index_node = cast("list[object]", node.args[1])[0]
        index = values[index_node] if isinstance(index_node, Node) else index_node
        mask_node = node.args[2]
        mask = values[mask_node] if isinstance(mask_node, Node) else None
        fake = cast("torch.Tensor", node.meta["val"])
        shape = self.shape(fake.shape)
        if len(shape) > 1 or (
            isinstance(mask, Fragment)
            and (
                len(mask.shape) > len(shape)
                or any(
                    left not in (1, right)
                    for left, right in zip(
                        mask.shape, shape[len(shape) - len(mask.shape) :], strict=True
                    )
                )
            )
        ):
            raise exc.InvalidConfig("local indexed loads require scalar/vector masks")
        extent = self.local_logical_sizes[id(source)]
        tensor_index = isinstance(index_node, Node) and isinstance(
            index_node.meta.get("val"), torch.Tensor
        )
        # Match hl.load's reference contract: tensor indices with an explicit
        # mask use nonnegative bounds, while unmasked/scalar indexing follows
        # PyTorch's one-extent negative normalization. Do not assign an output
        # value to unmasked out-of-bounds accesses: require a proved range.
        wrap_negative = mask_node is None or not tensor_index
        if wrap_negative:
            from .bounded_gather import integer_bounds

            interval = integer_bounds(index_node, captured_bounds={})
            if interval is None or not -extent <= interval[0] <= interval[1] < extent:
                raise exc.InvalidConfig(
                    "unmasked local indexed loads require a proved logical index range"
                )
        dependencies = (
            source,
            *(value for value in (index, mask) if isinstance(value, Fragment)),
        )

        def output_domain(coords: tuple[str, ...]) -> tuple[str, ...]:
            # Shape ownership survives the copy; runtime index/mask values do
            # not. Never retain a lazy shared read in the result's domain.
            return tuple(
                f"({coord}) < ({self.logical_axis_extent(fake, dim, size)})"
                for dim, (coord, size) in enumerate(zip(coords, shape, strict=True))
            )

        def read(coords: tuple[str, ...]) -> str:
            result = self.df.new_var("fragment_local_load")
            self.emit(f"{result} = {self.cast('0', source.dtype)}")
            domains: list[OwnershipMask] = list(output_domain(coords))
            if isinstance(index, Fragment):
                domains.extend(index.broadcast_domain(coords))
            if isinstance(mask, Fragment):
                domains.extend(mask.broadcast_domain(coords))
            branch = statement_from_string(f"if {self.predicate(domains)}:\n    pass")
            assert isinstance(branch, ast.If)
            branch.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                location = self.df.new_var("fragment_local_index")
                expression = (
                    index.broadcast(coords)
                    if isinstance(index, Fragment)
                    else self.scalar(index)
                )
                if wrap_negative:
                    self.emit(f"{location} = cutlass.Int64({expression})")
                    self.emit(
                        f"if {location} < 0:\n"
                        f"    {location} = {location} + cutlass.Int64({extent})"
                    )
                else:
                    self.emit(f"{location} = {expression}")
                valid = [f"0 <= ({location}) and ({location}) < ({extent})"]
                if isinstance(mask, Fragment):
                    valid.append(mask.broadcast(coords))
                predicate = self.predicate(valid)
                self.emit(f"if {predicate}:\n    {result} = {source.read((location,))}")
            self.cg.add_statement(branch)
            return result

        # The normal node boundary published pending writes before entering
        # memory(). Holding every dependency prevents result allocation from
        # recycling the source or a still-lazy index/mask operand.
        return self.materialize(
            Fragment(
                shape,
                source.dtype,
                read,
                dependencies=dependencies,
                logical_domain=output_domain,
            )
        )

    def memory(
        self,
        node: Node,
        values: dict[Node, object],
        store: bool,
        deferred_updates: list[tuple[Callable[[tuple[str, ...]], None], bool]]
        | None = None,
        direct: DirectStoreSink | None = None,
        *,
        defer_load: bool = False,
    ) -> object:
        indices = cast("list[object]", node.args[1])
        tensor = values[cast("Node", node.args[0])]
        if self.checked_root and isinstance(tensor, HostTensor):
            check_host_subscripts(
                self,
                indices,
                tensor.value.shape,
                unmasked_read=not store and node.args[2] is None,
            )
        if store and node in self.direct_stores:
            # Written by the tf32 mma lanes of the dot it stores.
            return None
        if deferred_updates is not None:
            values = dict(values)
        output = (
            cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])
            if store
            else cast("torch.Tensor", node.meta["val"])
        )
        # A scalar or broadcast value does not describe the indexed write
        # domain. Use the same destination shape rule as fake tensor loads.
        shape = self.shape(
            SubscriptIndexing.node_shape(node) if store else output.shape
        )
        extra_mask = node.args[3] if store else node.args[2]
        mask_value = values[extra_mask] if isinstance(extra_mask, Node) else extra_mask
        if isinstance(tensor, Fragment):
            if (
                not store
                and id(tensor) in self.local_logical_sizes
                and local_indexed_load(node)
            ):
                return self.completed_local_load(node, values, tensor)
            if (
                store
                or extra_mask is not None
                or node.args[3] is not None
                or any(index is not None and index != slice(None) for index in indices)
            ):
                raise exc.BackendUnsupported("cute", "computed fragment indexed memory")

            def view_coordinates(coords: tuple[str, ...]) -> tuple[str, ...]:
                selected = tuple(
                    coord
                    for coord, index in zip(coords, indices, strict=False)
                    if index is not None
                )
                return (*selected, *coords[len(indices) :])

            def view(coords: tuple[str, ...]) -> str:
                return tensor.read(self.coordinate_locals(view_coordinates(coords)))

            return Fragment(
                shape,
                output.dtype,
                view,
                tensor.resident,
                dependencies=(tensor,),
                logical_domain=lambda coords: tensor.domain(view_coordinates(coords)),
            )
        assert isinstance(tensor, HostTensor)
        index_coordinates = memory_index_coordinates(
            self.env,
            [
                index.meta["val"] if isinstance(index, Node) else index
                for index in indices
            ],
            {
                i: value.shape
                for i, index in enumerate(indices)
                if isinstance(index, Node)
                and isinstance(value := values[index], Fragment)
            },
            shape,
        )

        def selected_coordinates(
            ordinal: int, coords: tuple[str, ...]
        ) -> tuple[str, ...]:
            return tuple(
                "0" if axis is None else coords[axis]
                for axis in index_coordinates[ordinal]
            )

        def address(coords: tuple[str, ...]) -> tuple[str, str]:
            tensor_indices: list[str] = []
            masks: list[OwnershipMask] = []
            dim = 0
            for ordinal, index in enumerate(indices):
                if index is None:
                    continue
                proxy = index.meta["val"] if isinstance(index, Node) else index
                block = (
                    self.env.resolve_block_id(proxy)
                    if isinstance(proxy, torch.SymInt)
                    else None
                )
                if (
                    block is not None
                    and cast("torch.SymInt", proxy)._sympy_()
                    != self.env.block_sizes[block].var._sympy_()
                ):
                    block = None
                if isinstance(index, slice):
                    if index != slice(None):
                        # The coordinates of a bounded slice would need its
                        # start and length in the address.
                        raise exc.BackendUnsupported(
                            "cute", "a computed fragment through a bounded slice"
                        )
                    expression = selected_coordinates(ordinal, coords)[0]
                elif block is not None:
                    expression, mask = self.tile_index_and_mask(
                        tensor,
                        dim,
                        block,
                        selected_coordinates(ordinal, coords)[0],
                        broadcast=not store,
                    )
                    masks.append(mask)
                elif isinstance(index, Node):
                    value = values[index]
                    if isinstance(value, Fragment) and value.shape:
                        selected = selected_coordinates(ordinal, coords)
                        domain = value.domain(selected)
                        masks.extend(domain)
                        # Index fragments may have exact static capacity while
                        # the indexed output is padded. Read only coordinates
                        # owned by the index, before dereferencing host memory.
                        expression = self.df.new_var("fragment_tensor_index")
                        self.emit(f"{expression} = {self.cast('0', value.dtype)}")
                        branch = statement_from_string(
                            f"if {self.predicate(domain)}:\n    pass"
                        )
                        assert isinstance(branch, ast.If)
                        branch.body.clear()
                        with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                            self.emit(f"{expression} = {value.read(selected)}")
                        self.cg.add_statement(branch)
                    else:
                        expression = self.scalar(value)
                else:
                    expression = self.scalar(
                        _cute_wrap_negative_index(index, tensor.value.shape[dim])
                    )
                tensor_indices.append(expression)
                size = self.sym(tensor.value.shape[dim])
                masks.append(f"(({expression}) >= 0 and ({expression}) < ({size}))")
                dim += 1
            if isinstance(mask_value, Fragment):
                masks.append(mask_value.broadcast(coords))
            elif mask_value is not None:
                masks.append(self.scalar(mask_value))
            offset = (
                " + ".join(
                    f"cutlass.Int64({index}) * cutlass.Int64({self.host_stride(tensor, dim)})"
                    for dim, index in enumerate(tensor_indices)
                )
                or "0"
            )
            return f"({tensor.name}.iterator + ({offset}))", self.predicate(masks)

        if store:
            if direct is not None:
                # Hand the window's address recipe to the producing dot's lane
                # loop instead of walking a staged result.  The walks this
                # method would otherwise choose (deferred region updates,
                # warp- or register-owned results, an extra mask) keep the
                # staged path: the sink stays unfilled.
                if (
                    deferred_updates is not None
                    or mask_value is not None
                    or self.warp_result_nodes.get(node) == math.prod(shape)
                    or node in self.local_register_nodes
                ):
                    return None
                direct.shape = shape
                direct.dtype = tensor.value.dtype
                direct.address = address
                direct.lanes = (
                    self.row_packet_lanes(
                        tensor, indices, values, shape, output.dtype, index_coordinates
                    )
                    if self.warp_producer_recorder is None
                    else None
                )
                return None
            value = _resolve(node.args[2], values)
            source_dtype = (
                value.dtype
                if isinstance(value, Fragment)
                else scalar_value_dtype(value)
            )
            if self.checked_root and isinstance(
                value, (int, float, torch.SymInt, torch.SymFloat)
            ):
                value = self.value_scalar(value, assignment_dtype=tensor.value.dtype)

            def stored_value(coords: tuple[str, ...]) -> str:
                expression = (
                    value.broadcast(coords)
                    if isinstance(value, Fragment)
                    else self.scalar(value)
                )
                # Integer-to-FP8 stores need the same conversion as fills.
                return self.convert(expression, source_dtype, tensor.value.dtype)

            def write(coords: tuple[str, ...]) -> None:
                pointer, mask = address(coords)
                self.emit(f"if {mask}:\n    {pointer}.store({stored_value(coords)})")

            if deferred_updates is not None:
                deferred_updates.append((write, True))
                return None
            warp_owned = self.warp_result_nodes.get(node) == math.prod(shape)
            register_owned = node in self.local_register_nodes
            lanes = (
                self.row_packet_lanes(
                    tensor, indices, values, shape, output.dtype, index_coordinates
                )
                if not warp_owned
                and not register_owned
                and mask_value is None
                and self.warp_producer_recorder is None
                else None
            )
            if lanes is not None:
                self.vector_store(tensor, stored_value, shape, lanes, address)
                return None
            self.elements(
                shape,
                write,
                warp_owned=warp_owned,
                register_owned=register_owned,
            )
            return None

        def load(coords: tuple[str, ...]) -> str:
            pointer, mask = address(coords)
            name = self.df.new_var("fragment_load")
            self.emit(f"{name} = {self.cast('0', output.dtype)}")
            # Pointer arithmetic can erase unsigned signedness (Uint8 -> Int8),
            # and bool pointers use byte storage. Restore the logical type before
            # a masked branch joins its typed zero or the value enters arithmetic.
            loaded = self.cast(f"{pointer}.load()", output.dtype)
            self.emit(f"if {mask}:\n    {name} = {loaded}")
            return name

        index_domains: list[tuple[int, Fragment]] = []
        domain_bounds: list[tuple[int, str]] = []
        dim = 0
        for ordinal, index in enumerate(indices):
            if index is None:
                continue
            if isinstance(index, slice):
                domain_bounds.append((ordinal, self.sym(tensor.value.shape[dim])))
            elif isinstance(index, Node) and isinstance(values[index], Fragment):
                index_domains.append((ordinal, cast("Fragment", values[index])))
            elif isinstance(index, Node):
                proxy = index.meta["val"]
                if isinstance(proxy, torch.SymInt):
                    block = self.env.resolve_block_id(proxy)
                    if (
                        block in self.offsets
                        and proxy._sympy_() == self.env.block_sizes[block].var._sympy_()
                    ):
                        domain_bounds.append(
                            (
                                ordinal,
                                self.tile_live_extent(block, self.extent(proxy)),
                            )
                        )
            dim += 1

        def logical_domain(coords: tuple[str, ...]) -> tuple[OwnershipMask, ...]:
            return (
                *(
                    f"({selected_coordinates(ordinal, coords)[0]}) < ({bound})"
                    for ordinal, bound in domain_bounds
                ),
                *(
                    mask
                    for ordinal, index_value in index_domains
                    for mask in index_value.domain(
                        selected_coordinates(ordinal, coords)
                    )
                ),
            )

        loaded_fragment = Fragment(
            shape,
            output.dtype,
            load,
            logical_domain=logical_domain if index_domains or domain_bounds else None,
        )
        if node in self.slab_plan:
            return self.slab_member(node, loaded_fragment)
        if defer_load:
            # A complete owner may consume this same guarded address recipe
            # directly. Retain every index/mask dependency until that read.
            if not host_load_is_readonly(node, self.env, self.graphs):
                raise exc.InvalidConfig(
                    "deferred fragment load requires readonly storage"
                )
            dependencies = tuple(
                value
                for value in (
                    *(values[index] for index in indices if isinstance(index, Node)),
                    mask_value,
                )
                if isinstance(value, Fragment)
            )
            return dataclasses.replace(loaded_fragment, dependencies=dependencies)
        if (
            not shape
            and extra_mask is None
            and any(
                user.target in (_tracing_ops._for_loop, _tracing_ops._for_loop_step)
                and any(
                    node in cast("list[object]", bounds) for bounds in user.args[1:3]
                )
                for user in node.users
            )
            and all(
                uniform_integer_expr(
                    self.env, index.meta["val"] if isinstance(index, Node) else index
                )
                is not None
                for index in indices
            )
            and not has_opaque_inline_asm(self.graphs)
            and host_load_is_readonly(node, self.env, self.graphs)
        ):
            # Every lane reads the same immutable word at this point. Its
            # register can carry the bound into the loop without a CTA-wide
            # shared store, barrier, and reload.
            scalar_bound = load(())
            return Fragment((), output.dtype, lambda _coords: scalar_bound, True)
        if node in self.snapshot_loads:
            slots = (math.prod(shape) + self.threads - 1) // self.threads
            name = self.df.new_var("fragment_snapshot")
            self.register_snapshot_buffers[name] = shape[0]
            self.emit(
                f"{name} = cute.make_rmem_tensor(({slots},), {self.dtype(output.dtype)})"
            )
            self.emit(f"{name}.fill({self.cast('0', output.dtype)})")

            def read_snapshot(coords: tuple[str, ...]) -> str:
                owner = self.snapshot_owner
                if (
                    owner is None
                    or len(coords) != 1
                    or owner.size
                    not in (shape[0], self.shape(snapshot_logical_shape(node))[0])
                ):
                    raise exc.InvalidConfig(
                        "register snapshot lacks its active coordinate owner"
                    )
                owner.prove(coords[0])
                return f"{name}[{owner.slot}]"

            def fill_snapshot(coords: tuple[str, ...]) -> None:
                self.emit(
                    f"{read_snapshot(coords)} = {self.cast(load(coords), output.dtype)}"
                )

            # Preserve the old materialization barrier and every guarded load.
            self.elements(shape, fill_snapshot)
            return Fragment(
                shape,
                output.dtype,
                read_snapshot,
                True,
                logical_domain=loaded_fragment.logical_domain,
                snapshot=SnapshotBinding(
                    node,
                    name,
                    self.threads,
                    shape,
                    output.dtype,
                    frozenset(
                        self.referenced_buffers(
                            [
                                values[index]
                                for index in indices
                                if isinstance(index, Node)
                            ]
                        )
                    ),
                    self.shape(snapshot_logical_shape(node)),
                ),
            )
        batched = self.batchable_host_load(node, values)
        lanes = (
            self.row_packet_lanes(
                tensor, indices, values, shape, output.dtype, index_coordinates
            )
            if mask_value is None and self.warp_producer_recorder is None
            else None
        )
        if lanes is not None:
            result = self.vector_load(
                loaded_fragment, shape, lanes, address, batched=batched
            )
            self.pending_barrier = batched
            return result
        if self.df.config.get(
            "cute_fragment_packet_loads", False
        ) and host_load_is_readonly(node, self.env, self.graphs):
            from .packet_loads import materialize_packet_load

            index_values = (
                values[index] for index in indices if isinstance(index, Node)
            )
            statements = self.cg.statements_stack[-1]
            start = len(statements)
            packet = materialize_packet_load(
                self,
                tensor,
                loaded_fragment,
                load,
                tuple(value for value in index_values if isinstance(value, Fragment))
                + ((mask_value,) if isinstance(mask_value, Fragment) else ()),
            )
            if packet is not None:
                if self.warp_producer_recorder is not None:
                    self.warp_producer_recorder.publication(statements[start:], packet)
                return packet
        if (
            self.df.config.get("cute_fragment_register_loads", False)
            and math.prod(shape) <= self.threads
            and lane_private_load(node, self.env)
            and host_load_is_readonly(node, self.env, self.graphs)
        ):
            # One typed scalar per active lane, in the current lexical body.
            # The graph proof forbids remapping, carries and branch escapes.
            # Keep the ordinary atomic/store element loops and their barriers.
            name = self.df.new_var("fragment_register_load")
            self.emit(f"{name} = {self.cast('0', output.dtype)}")
            branch = statement_from_string(
                f"if {self.thread} < {math.prod(shape)}:\n    pass"
            )
            assert isinstance(branch, ast.If)
            branch.body.clear()
            with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                loaded = load(self.coordinates(self.thread, shape))
                self.emit(f"{name} = {self.cast(loaded, output.dtype)}")
            self.cg.add_statement(branch)
            return Fragment(
                shape,
                output.dtype,
                lambda _coords: name,
                True,
                logical_domain=loaded_fragment.logical_domain,
            )
        result = self.materialize(loaded_fragment, batched=batched)
        self.pending_barrier = batched
        return result
