from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch
from torch.fx import map_arg

from .._utils import next_power_of_2
from ..language import _MEMORY_OPS
from ..language import _tracing_ops
from ..language import atomic_add
from ..language import atomic_and
from ..language import atomic_cas
from ..language import atomic_max
from ..language import atomic_min
from ..language import atomic_or
from ..language import atomic_xchg
from ..language import atomic_xor
from ..language._tracing_ops import _for_loop
from ..language._tracing_ops import _get_symnode
from ..language._tracing_ops import _host_tensor
from ..language._tracing_ops import _if
from ..language._tracing_ops import is_for_loop_target
from ..language.matmul_ops import dot as hl_dot
from ..language.matmul_ops import dot_scaled as hl_dot_scaled
from ..language.memory_ops import load
from ..language.memory_ops import prefetch
from ..language.memory_ops import prefetch_is_bulk
from ..language.memory_ops import store
from ..language.reduce_ops import _reduce
from ..language.scan_ops import _associative_scan
from ..language.view_ops import join as hl_join
from ..language.view_ops import split as hl_split
from .aten_lowering import aten_lowering_dispatch
from .compile_environment import CompileEnvironment
from .fx_side_effects import is_impure_node
from .host_function import HostFunction
from .indexing_strategy import _padded_iota_dims_once
from .indexing_strategy import load_index_axes
from .indexing_strategy import padded_index_dims
from .inductor_lowering import APIFuncLowering
from .inductor_lowering import ReductionLowering
from .loop_dependency_checker import reaches_through_pointers
from .runtime_tensor_guards import register_tensor_alias_specializations
from .runtime_tensor_guards import tensors_have_proven_disjoint_storage

if TYPE_CHECKING:
    from .compile_environment import BlockSizeInfo
    from .device_ir import DeviceIR
    from .device_ir import RolledReductionInfo

_duplicate_ops: tuple[object, ...] = (
    _host_tensor,
    _get_symnode,
    torch.ops.aten.sym_size.int,
)

# Ops that write to memory and should be treated specially when determining
# whether a node depends on the reduction dimension used for rolling.
_ATOMIC_OPS: tuple[object, ...] = (
    atomic_add,
    atomic_and,
    atomic_cas,
    atomic_max,
    atomic_min,
    atomic_or,
    atomic_xchg,
    atomic_xor,
)


# Marks a reduction of values computed outside the loop from a free
# ``hl.arange``: on CuTe it rolls for the computed-fragment lowering, and
# ``LoopedReductionStrategy`` refuses it otherwise.
FREE_IOTA_REDUCTION_META = "helion_free_iota_reduction"


class EffectfulReductionReplay(NotImplementedError):
    """Rolling needs to replay a volatile value or cross an opaque effect."""


class ReductionRoller:
    """This does the opposite of unrolling, it takes persistent reductions and turns them into looped reductions."""

    def __init__(
        self,
        device_ir: DeviceIR,
        rdim: BlockSizeInfo,
        graph_id_to_info: dict[int, RolledReductionInfo],
        enable_fusion: bool = False,
        *,
        allow_unbound_alias_facts: bool = False,
        subscript_extent: bool = False,
    ) -> None:
        self.device_ir = device_ir
        self.rdim = rdim
        self.graph_id_to_info = graph_id_to_info
        self.enable_fusion = enable_fusion
        self.allow_unbound_alias_facts = allow_unbound_alias_facts
        # True rolls a subscript extent's dim as older versions rolled the
        # power of two they padded it to (``DeviceIR._padded_range_dims``): a
        # value with the dim on several axes uses it once, since those axes
        # had literal sizes then, which name no reduction dim.
        self.subscript_extent = subscript_extent
        # inner graph contains ops on the reduction dimension
        self.inner_args: list[torch.fx.Node] = []
        self.inner_graph: torch.fx.Graph = torch.fx.Graph()
        self.inner_nodes: dict[torch.fx.Node, torch.fx.Node] = {}
        self.inner_count: int = 0
        self.inner_available: set[torch.fx.Node] = set()
        # outer graph contains ops that are not on the reduction dimension
        self.outer_graph: torch.fx.Graph = torch.fx.Graph()
        self.outer_nodes: dict[torch.fx.Node, torch.fx.Node] = {}
        self.outer_count: int = 0
        self.seen: set[torch.fx.Node] = set()
        self.available: set[torch.fx.Node] = set()
        self.graphs_added: list[int] = []
        self._size_node: torch.fx.Node | None = None
        self._node_positions: dict[torch.fx.Node, int] = {}
        self._effects_seen: list[torch.fx.Node] = []
        self._effect_cache: dict[torch.fx.Node, bool] = {}
        self._memory_read_cache: dict[torch.fx.Node, bool] = {}
        self._inner_has_effect = False
        self._replayed_reads: set[torch.fx.Node] = set()
        self._replay_error: EffectfulReductionReplay | None = None

    def _record_replay_error(self, message: str) -> None:
        # Finish the structural analysis so saved config slots retain their
        # graph ownership. process() must still fail before returning a graph;
        # these unsafe analysis copies can never be used for emitted code.
        if self._replay_error is None:
            self._replay_error = EffectfulReductionReplay(message)

    def _nested_nodes(self, node: torch.fx.Node) -> list[torch.fx.Node]:
        # DeviceIR imports the roller, so resolve its graph-edge helper lazily.
        from .device_ir import control_flow_child_graph_ids

        return [
            child
            for graph_id in control_flow_child_graph_ids(node)
            for child in self.device_ir.graphs[graph_id].graph.nodes
        ]

    def _has_effect(self, node: torch.fx.Node) -> bool:
        if node not in self._effect_cache:
            self._effect_cache[node] = self._compute_has_effect(node)
        return self._effect_cache[node]

    def _compute_has_effect(self, node: torch.fx.Node) -> bool:
        if node.op != "call_function":
            return False
        if node.target in (_tracing_ops._phi, _tracing_ops._new_var, prefetch):
            # A prefetch only warms a cache: it orders nothing, and running
            # it again in a later sweep changes no result.
            return False
        if children := self._nested_nodes(node):
            # Control-flow markers are impure for DCE, not intrinsically
            # volatile. Pure loop-carried computations may be replayed.
            return any(self._has_effect(child) for child in children)
        return is_impure_node(node)

    def _reads_memory(self, node: torch.fx.Node) -> bool:
        if node not in self._memory_read_cache:
            self._memory_read_cache[node] = node.target is load or any(
                self._reads_memory(child) for child in self._nested_nodes(node)
            )
        return self._memory_read_cache[node]

    def _effect_may_change_read(
        self, effect: torch.fx.Node, read: torch.fx.Node
    ) -> bool:
        if children := self._nested_nodes(effect):
            return any(
                self._effect_may_change_read(child, read)
                for child in children
                if self._has_effect(child)
            )
        if children := self._nested_nodes(read):
            return any(
                self._effect_may_change_read(effect, child)
                for child in children
                if self._reads_memory(child)
            )
        if effect.target not in (store, *_ATOMIC_OPS) or read.target is not load:
            return True
        written, loaded = effect.args[0], read.args[0]
        if not isinstance(written, torch.fx.Node) or not isinstance(
            loaded, torch.fx.Node
        ):
            return True
        destination, source = written.meta.get("val"), loaded.meta.get("val")
        if not isinstance(destination, torch.Tensor) or not isinstance(
            source, torch.Tensor
        ):
            return True
        env = CompileEnvironment.current()
        if all(
            env.tensor_storage_input_source(tensor) is not None
            for tensor in (destination, source)
        ):
            # Record negative facts too: a later call with disjoint storage
            # must be able to select a rolled binding instead of this one.
            register_tensor_alias_specializations(env)
        # During binding the registered classifier supplies the initial fact;
        # codegen checks the immutable fact saved in the dispatch key.
        return not tensors_have_proven_disjoint_storage(
            env,
            destination,
            source,
            allow_unbound=self.allow_unbound_alias_facts,
        )

    def is_reduction(self, node: torch.fx.Node) -> bool:
        """Check if a node is a reduction"""
        return (
            node.op == "call_function"
            and isinstance(lowering := node.meta["lowering"], ReductionLowering)
            and lowering.block_index == self.rdim.block_id
        )

    def should_go_in_inner_graph(self, node: torch.fx.Node) -> bool:
        """Nodes go in the inner graph if they use the reduction dimension"""
        if node.op in {"placeholder", "output"}:
            return False
        assert node.op == "call_function", f"Unsupported node type {node.op}"

        if node.target is _reduce:
            # TODO(jansel): support rolling user-defined reductions
            raise NotImplementedError(
                "hl._reduce operations are not compatible with reduction rolling"
            )

        if node.target is _associative_scan:
            # It appears associative scans are not rollable: a scan needs the
            # running prefix from all prior elements, but the scan codegen assumes
            # the whole scan dimension is one tile (there is no cross-tile prefix
            # carry). Rolling re-runs the scan per chunk with no carry, silently
            # dropping the cross-chunk prefix. Refuse to roll -> stay persistent.
            raise NotImplementedError(
                "hl.associative_scan operations are not compatible with reduction rolling"
            )

        if node.target in (
            torch.ops.aten.topk.default,
            torch.ops.aten.sort.default,
            torch.ops.aten.sort.stable,
        ):
            # Ordering needs every value of its selected axis. Top-k shrinks
            # that axis, so output-shape placement would leave it outside the
            # rolled producer loop, observing only its last chunk. Sorting each
            # chunk independently likewise lacks a cross-chunk merge. Other
            # axes remain independent and may still be rolled normally.
            source = node.args[0]
            assert isinstance(source, torch.fx.Node)
            value = source.meta["val"]
            assert isinstance(value, torch.Tensor)
            position = 2 if node.target is torch.ops.aten.topk.default else 1
            dim = (
                node.args[position]
                if len(node.args) > position
                else node.kwargs.get("dim", -1)
            )
            assert isinstance(dim, int)
            if (
                CompileEnvironment.current().get_block_id(value.shape[dim])
                == self.rdim.block_id
            ):
                raise NotImplementedError(
                    "selection axes require complete input or a cross-chunk merge"
                )

        if is_for_loop_target(node.target) or node.target is _if:
            if is_for_loop_target(node.target):
                graph_id, *_ = node.args
                assert isinstance(graph_id, int)
                infos = [self.graph_id_to_info[graph_id]]
            else:
                _, if_graph_id, else_graph_id, *_ = node.args
                assert isinstance(if_graph_id, int)
                assert isinstance(else_graph_id, int)
                infos = [
                    self.graph_id_to_info[if_graph_id],
                    self.graph_id_to_info[else_graph_id],
                ]
            used_infos = [info for info in infos if info.used_rdim]
            if used_infos:
                if not all(info.can_be_rolled_by_caller for info in used_infos):
                    raise NotImplementedError("for loop with mixed reduction dim usage")
                return True
            return False

        if node.target in _duplicate_ops:
            if node.target is torch.ops.aten.sym_size.int:
                arg = node.args[0]
                assert isinstance(arg, torch.fx.Node)
                return self.should_go_in_inner_graph(arg)
            return False

        if node.target is hl_split:
            base = node.args[0]
            if isinstance(base, torch.fx.Node):
                return self.should_go_in_inner_graph(base)
            return False

        if node.target is operator.getitem:
            base = node.args[0]
            if isinstance(base, torch.fx.Node) and base.target is hl_split:
                return self.should_go_in_inner_graph(base)

        if node.target is hl_join:
            left = node.args[0]
            right = node.args[1]
            left_inner = isinstance(
                left, torch.fx.Node
            ) and self.should_go_in_inner_graph(left)
            right_inner = isinstance(
                right, torch.fx.Node
            ) and self.should_go_in_inner_graph(right)
            return left_inner or right_inner

        if self.is_reduction(node):
            return True

        if node.target is store or node.target in _ATOMIC_OPS:
            # atomic_add(target, index, value, sem)
            target, index_arg, value, *_ = node.args
            if isinstance(value, torch.fx.Node):
                val = value.meta["val"]
                num_rdims = self._count_rdim_axes_in_val(val)
                if num_rdims == 0:
                    if isinstance(val, torch.Tensor) and any(
                        s == 1 for s in val.size()
                    ):
                        num_rdims = self._count_rdim_axes_in_subscript(
                            target, index_arg
                        )
                elif num_rdims != self._count_rdim_axes_in_subscript(target, index_arg):
                    # The value's rdim axis lands on positions another index
                    # names (``loss[tile] = rows_over_rdim``): a chunk of the
                    # rdim per iteration cannot fill them.
                    raise NotImplementedError(
                        "a value over the reduction dim is stored at positions "
                        "another index names"
                    )
            else:
                # When the stored value is a Python scalar (broadcast across
                # the indexed slice), it carries no shape, so the LHS
                # subscript determines whether the op covers the rdim. Wrap
                # the op in the reduction loop so the same rdim index_var is
                # active here as for the corresponding load; otherwise on
                # backends with looped reductions (e.g. cute), the slice
                # resolves to the grid block_id and only one element per
                # iteration is written.
                num_rdims = self._count_rdim_axes_in_subscript(target, index_arg)
        elif node.target is prefetch:
            # A prefetch has no value; its subscript says what it covers, less
            # the row segment a bulk prefetch covers by its byte count.
            target, index_arg, extra_mask = node.args
            assert isinstance(target, torch.fx.Node)
            assert isinstance(index_arg, (list, tuple))
            index = [*map_arg(index_arg, lambda arg: arg.meta["val"])]
            if prefetch_is_bulk(target.meta["val"], index, extra_mask is not None):
                index_arg = [*index_arg[:-1], 0]
            num_rdims = self._count_rdim_axes_in_subscript(target, index_arg)
        else:
            val = node.meta.get("val", None)
            num_rdims = self._count_rdim_axes_in_val(val)

        if num_rdims > 1 and not self.subscript_extent:
            raise NotImplementedError(
                "multiple reduction dims of same size not supported"
            )
        return num_rdims > 0

    def _count_rdim_axes_in_val(self, val: object) -> int:
        """Count how many tensor axes reference ``self.rdim``.

        Handles single tensors and ``(tuple, list)`` of tensors (e.g. the
        ``var_mean`` op returns two tensors). Non-tensor values (scalars)
        contribute 0. The caller raises when the count exceeds 1, since
        the roller's per-axis chunking only makes sense for one rdim per
        node.
        """
        if not isinstance(val, (torch.Tensor, tuple, list)):
            return 0
        env = CompileEnvironment.current()
        rdim_block_id = self.rdim.block_id

        def count_for_tensor(t: torch.Tensor) -> int:
            return sum(env.get_block_id(size) == rdim_block_id for size in t.size())

        if isinstance(val, torch.Tensor):
            return count_for_tensor(val)
        return sum(
            count_for_tensor(item) for item in val if isinstance(item, torch.Tensor)
        )

    def _count_rdim_axes_in_subscript(self, target: object, index: object) -> int:
        """Count axes in the LHS indexing pattern that match ``self.rdim``.

        Used for store/atomic ops with scalar values: their meta val carries
        no shape, so axis counts have to come from the indexing pattern
        itself. The shape is computed the same way as for loads, so the
        store and load resolve their slices to the same block_id.
        """
        if reaches_through_pointers(target) and target:
            # A stack tensor ``(tensor_like, dev_ptrs)``: the subscript
            # indexes the tensor-like's dims.
            assert isinstance(target, (list, tuple))
            target = target[0]
        if not isinstance(target, torch.fx.Node):
            return 0
        target_val = target.meta.get("val")
        if not isinstance(target_val, torch.Tensor) or not isinstance(
            index, (list, tuple)
        ):
            return 0
        from .indexing_strategy import SubscriptIndexing

        # FX records traced SymInt/Tensor index args (and symbolic slice bounds)
        # as torch.fx.Node references; compute_shape wants the underlying values,
        # the same form it receives at register_fake time (memory_ops.load).
        normalized = [*torch.fx.node.map_arg(index, lambda node: node.meta["val"])]
        shape = SubscriptIndexing.compute_shape(target_val, normalized)
        env = CompileEnvironment.current()
        rdim_block_id = self.rdim.block_id
        return sum(env.get_block_id(s) == rdim_block_id for s in shape)

    def size_node(self, meta: dict[str, object]) -> torch.fx.Node:
        """Create a node that represents the size of the reduction dimension"""
        if self._size_node is not None:
            return self._size_node
        self._size_node = node = self.outer_graph.call_function(
            _get_symnode,
            (f"rdim{self.rdim.block_id}",),
            {},
        )
        node.meta.update(meta)
        node.meta["val"] = self.rdim.size
        node.meta["lowering"] = APIFuncLowering(_get_symnode)
        return node

    def start_new_graph(self) -> None:
        if self.inner_count == 0:
            return

        inner_nodes: dict[torch.fx.Node, torch.fx.Node] = self.inner_nodes
        outputs = {}
        inner_node_set = set(inner_nodes)
        for orig_node, inner_node in inner_nodes.items():
            # A later loop reading a value re-adds it (``readd`` below), so
            # only users outside the loops need it as an output.  Kept as one
            # for a later loop, it would stay live here with nothing here
            # reading it, loads included, and the CuTe DSL keeps a dead
            # cache-hint load (side-effecting inline asm) in the sweep.
            needs_output = orig_node not in self.outer_nodes and (
                self.is_reduction(orig_node)
                or any(
                    user not in inner_node_set
                    and not self.should_go_in_inner_graph(user)
                    for user in orig_node.users
                )
            )
            if needs_output:
                outputs[orig_node] = inner_node
            self.available.add(orig_node)
        graph = self.inner_graph
        graph.output([*outputs.values()])
        # Pass only the arguments the live nodes read: this loop's argument
        # list leaves out the values only its dead nodes read (the later loop
        # that re-adds those nodes reads them itself).
        graph.eliminate_dead_code(is_impure_node=is_impure_node)
        node_args = []
        for placeholder, arg in zip(
            graph.find_nodes(op="placeholder"), self.inner_args, strict=True
        ):
            if placeholder.users:
                node_args.append(arg)
            else:
                graph.erase_node(placeholder)
        # A loop left with nothing to compute (an impure outer node split it
        # off ahead of the loop reading its values) is not emitted.
        if any(node.op == "call_function" for node in graph.nodes):
            self._emit_loop(graph, node_args, [*outputs], next(iter(inner_nodes)))

        self.inner_args = []
        self.inner_graph = torch.fx.Graph()
        self.inner_nodes = {}
        self.inner_count = 0
        self.inner_available = set()
        self._inner_has_effect = False
        self._replayed_reads = set()

        def readd(node: torch.fx.Node) -> None:
            if (
                node not in inner_nodes
                or node in self.inner_nodes
                or self.is_reduction(node)
            ):
                return
            if inner_nodes[node].op == "placeholder":
                # The loop read the value from outside it (an outer node, a
                # loop output), and so does this one: running it again here
                # would reload a per-row scalar on every iteration, after
                # any store to its tensor in between.
                self.get_inner_arg(node)
                return
            if self._has_effect(node):
                # A value used on both sides of a reduction would be recomputed
                # in the next sweep. Opaque effects and volatile results cannot
                # be replayed; keep this reduction persistent instead.
                self._record_replay_error(
                    "reduction sweeps cannot duplicate an effectful producer"
                )
            if self._reads_memory(node) and any(
                self._node_positions[node] < self._node_positions[effect]
                and self._effect_may_change_read(effect, node)
                for effect in self._effects_seen
            ):
                # Even a discarded assembly result can overwrite an earlier
                # load's storage. Replaying that load would combine the new
                # memory value with a reduction of the original value.
                self._record_replay_error(
                    "reduction sweeps cannot reload memory across an effect"
                )
            for n in node.all_input_nodes:
                readd(n)
            new_node = self.inner_graph.create_node(
                node.op,
                node.target,
                *map_arg((node.args, node.kwargs), self.get_inner_arg),
                name=node.name,
            )
            new_node.meta.update(node.meta)
            self.inner_nodes[node] = new_node
            if self._reads_memory(node):
                self._replayed_reads.add(node)

        # re-add any nodes that still have pending users
        for node in inner_nodes:
            if {*node.users} - self.seen:
                readd(node)

    def _emit_loop(
        self,
        graph: torch.fx.Graph,
        node_args: list[torch.fx.Node],
        outputs: list[torch.fx.Node],
        first: torch.fx.Node,
    ) -> None:
        """Add ``graph`` as a reduction loop over the rolled block, called with
        ``node_args``, and give each of ``outputs`` (the original nodes whose
        values the loop returns, in order) its result in the outer graph, at
        the location of ``first``, the loop's first node."""
        graph_id = self.device_ir.add_reduction_loop_graph(
            graph,
            block_index=self.rdim.block_id,
            node_args=node_args,
        )
        self.graphs_added.append(graph_id)

        location_meta = {
            "location": first.meta["location"],
            "stack_trace": first.meta.get("stack_trace", ""),
        }
        output_node = self.outer_graph.call_function(
            _for_loop,
            (graph_id, [0], [self.size_node(location_meta)], node_args),
            {},
        )
        output_node.meta.update(location_meta)
        output_node.meta["val"] = [n.meta["val"] for n in outputs]
        output_node.meta["lowering"] = APIFuncLowering(_for_loop)
        for i, orig_node in enumerate(outputs):
            self.outer_nodes[orig_node] = n = self.outer_graph.call_function(
                operator.getitem,
                (output_node, i),
                {},
            )
            n.meta.update(location_meta)
            n.meta["val"] = orig_node.meta["val"]
            n.meta["lowering"] = aten_lowering_dispatch[n.target](n)

    def get_inner_arg(self, node: torch.fx.Node) -> torch.fx.Node:
        """Get the input node for the inner graph"""
        if node in self.inner_nodes:
            return self.inner_nodes[node]
        if node.target is _get_symnode:
            # this is a fake node we can duplicate in both graphs
            self.inner_nodes[node] = new_node = self.inner_graph.create_node(
                node.op,
                node.target,
                node.args,
                node.kwargs,
                name=node.name,
            )
            new_node.meta.update(node.meta)
            return new_node
        # need to create a new placeholder arg in the inner graph
        outer_node = self.outer_nodes[node]
        if outer_node.target in _duplicate_ops:
            # These fake nodes can be duplicated
            self.inner_nodes[node] = new_node = self.inner_graph.create_node(
                node.op,
                node.target,
                *map_arg((node.args, node.kwargs), self.get_inner_arg),
                name=node.name,
            )
            new_node.meta.update(node.meta)
            return new_node
        placeholders = self.inner_graph.find_nodes(op="placeholder")
        with self.inner_graph.inserting_after(
            placeholders[-1] if placeholders else self.inner_graph._root
        ):
            self.inner_nodes[node] = placeholder = self.inner_graph.placeholder(
                outer_node.name
            )
            placeholder.meta.update(node.meta)
        self.inner_args.append(outer_node)
        return placeholder

    def cannot_roll_graph(
        self,
        graph: torch.fx.Graph,
        *,
        free_ranges: bool = True,
        reductions: bool = True,
    ) -> bool:
        """Check if a graph uses the rdim in a way no loop over it can take:
        a matmul, fixed indexing or ``dev_ptrs`` stacking with the rdim, or
        (with ``reductions``) a use along a non-tile axis
        (``has_unrollable_rdim_use``, which ``free_ranges`` goes to)."""
        return (
            self.has_matmul_with_rdim(graph)
            or self.has_indexing_along_rdim(graph)
            or self.has_stack_tensor_with_rdim(graph)
            or (
                reductions
                and self.has_unrollable_rdim_use(graph, free_ranges=free_ranges)
            )
        )

    def has_matmul_with_rdim(self, graph: torch.fx.Graph) -> bool:
        """Check if a graph contains matmul operations with rdim inputs."""

        def is_matmul_with_rdim(node: torch.fx.Node) -> bool:
            """Check if a node is a matmul operation with rdim inputs."""
            if node.op != "call_function":
                return False

            # Check multiple matmul-family operations
            if node.target not in (
                torch.ops.aten.mm.default,
                torch.ops.aten.addmm.default,
                torch.ops.aten.bmm.default,
                torch.ops.aten.baddbmm.default,
                hl_dot,
                hl_dot_scaled,
            ):
                return False

            # Check if any inputs to matmul have rdim
            env = CompileEnvironment.current()
            for input_node in node.all_input_nodes:
                val = input_node.meta.get("val", None)
                if isinstance(val, torch.Tensor):
                    for size in val.size():
                        block_idx = env.get_block_id(size)
                        if block_idx == self.rdim.block_id:
                            return True
                        # A dimension with no block_id is not tiled,
                        # so the matmul operates on it in full and
                        # cannot be sliced by the roller.
                        if block_idx is None:
                            return True
            return False

        return any(is_matmul_with_rdim(node) for node in graph.nodes)

    def has_indexing_along_rdim(self, graph: torch.fx.Graph) -> bool:
        """Check if a graph reads fixed coordinates along the rdim.

        ``torch.gather(row, dim, index)`` reads ``row`` at any position of
        ``dim``, which one chunk of the rdim per loop iteration does not hold
        (Triton's ``tl.gather`` would pick positions within the chunk). Static
        slice/select likewise address the complete tensor, not each chunk.
        Slices whose result keeps the rdim axis (start zero, unit step and full
        end) remain rollable.
        """
        env = CompileEnvironment.current()
        for node in graph.nodes:
            if node.target not in (
                torch.ops.aten.gather.default,
                torch.ops.aten.slice.Tensor,
                torch.ops.aten.select.int,
            ):
                continue
            input_node = node.args[0]
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
            assert isinstance(input_node, torch.fx.Node)
            assert isinstance(dim, int)
            val = input_node.meta["val"]
            assert isinstance(val, torch.Tensor)
            if env.get_block_id(val.size(dim)) != self.rdim.block_id:
                continue
            if node.target is torch.ops.aten.slice.Tensor:
                start, step = (
                    node.args[i]
                    if len(node.args) > i
                    else node.kwargs.get(key, default)
                    for i, key, default in (
                        (2, "start", None),
                        (4, "step", 1),
                    )
                )
                if (
                    (start is None or type(start) is int and start == 0)
                    and type(step) is int
                    and step == 1
                    and env.get_block_id(node.meta["val"].size(dim))
                    == self.rdim.block_id
                ):
                    # A literal end equal to the logical size can still erase
                    # the rdim symbol (its hint may be padded). Only a result
                    # carrying that same axis stays in the rolled inner graph.
                    continue
            return True
        return False

    def has_pinned_rdim(self) -> bool:
        """Check if a guard pinned the rdim's symbol to a constant.

        A reshape to a literal size (``x[t, :].reshape(t, 1, 64)``) or
        hl.specialize does: the rdim's dims then carry a plain int that
        ``get_block_id`` cannot tell from other dims of that size, so the
        roller would move only the nodes whose dims still name the symbol (a
        store) into the loop and leave the loads they read outside it.
        """
        return not self.rdim.var._sympy_().free_symbols

    def tied_rdims(self) -> list[int]:
        """The block ids of the other rdims a guard tied the rdim's block to.

        A literal dim (``BlockSizeInfo.literal_size``: a padded factory's, or
        that of a subscript of a padded ``hl.arange``) does not join a
        dynamic full slice of its size, but an elementwise op between the two
        guards their blocks equal and names its result by only one of them,
        as ``torch.where(hl.arange(n)[None, :] == t, -1.0, row)`` beside
        ``row = x[tile, :]`` does under dynamic shapes.  Rolling either dim
        would chunk the values that name it while the values that name the
        other keep its full extent, and the op between them would combine a
        chunk with the full extent (a shape mismatch on Triton).
        """
        env = CompileEnvironment.current()
        return [
            info.block_id
            for info in env.block_sizes
            if info.reduction
            and info.block_id != self.rdim.block_id
            and env.known_equal(info.var, self.rdim.var)
        ]

    def range_meets_full_slice(self, graph: torch.fx.Graph) -> bool:
        """Check if one op reads both a value computed from an ``iota`` on the
        rdim's lanes and a loaded one, such as ``x[tile, :] + hit`` for
        ``hit`` computed from ``hl.arange(n)[None, :]``.

        Versions that padded a subscript of a range to a power of two typed
        such an op with the padded size beside the load's dim of ``n``, which
        guarded that dim to the power of two
        (``DeviceIR._record_legacy_reduction_loops``).  Values that cross
        into control flow are not followed.
        """
        env = CompileEnvironment.current()

        def on_rdim(node: torch.fx.Node) -> bool:
            val = node.meta.get("val")
            return isinstance(val, torch.Tensor) and any(
                env.resolve_block_id(dim) == self.rdim.block_id for dim in val.size()
            )

        def reach(seeds: list[torch.fx.Node]) -> set[torch.fx.Node]:
            # The values on the rdim's lanes computed from ``seeds``.
            reached: set[torch.fx.Node] = set()
            stack = seeds
            while stack:
                node = stack.pop()
                if node in reached or not on_rdim(node):
                    continue
                reached.add(node)
                stack.extend(node.users)
            return reached

        ranges = reach(
            [
                node
                for node in graph.nodes
                if node.target is torch.ops.prims.iota.default
            ]
        )
        loads = reach(
            [node for node in graph.nodes if node.target is load and node not in ranges]
        )
        return not ranges.isdisjoint(loads)

    def has_unrollable_rdim_use(
        self, graph: torch.fx.Graph, *, free_ranges: bool = True
    ) -> bool:
        """Check if a graph reduces over the rdim along a non-tile axis, or
        reads a free ``iota`` over the rdim inside the loop.

        ``free_ranges=False`` checks only the reductions: versions before the
        free ``iota`` half had only that half.

        A reduction can only be rolled into a loop when the dimension being
        reduced is indexed by a re-bindable block index var (e.g. a ``hl.tile``
        axis or a ``[..., :]`` slice), so the producing load can be re-indexed
        in ``_REDUCTION_BLOCK``-sized chunks inside the loop.  An
        ``hl.arange(n)`` over the rdim rolls when its value names the rdim's
        symbol, so it moves into the loop and takes the chunk's positions,
        and it does not index a load directly.  Otherwise it is a fixed
        full-extent ``iota`` outside the loop, and any value inside the loop
        computed from it is wrong: a shape mismatch on Triton (issue #2643),
        an unrelated coordinate on CuTe.  Such reductions stay persistent, and
        so does the dim when no reduction reads the ``iota`` but a value
        inside the loop is computed from it, as in
        ``torch.where(cols[None, :] == t, -1.0, row)`` for
        ``cols = hl.arange(n)`` beside ``row = x[tile, :]``.

        On CuTe a free ``iota`` that only reductions read, through values
        computed outside the loop, still rolls when the computed-fragment
        lowering takes the reduction (``free_iota_reductions``): it re-reads
        the iota per chunk, and any other lowering refuses a rolled reduction
        of a value from outside its loop
        (``LoopedReductionStrategy.codegen_reduction``).

        Detected by walking back from each reduction over ``self.rdim`` and
        checking each input ``iota`` node sized to the rdim, and forward from
        each ``iota`` sized to the rdim outside the loop. The output-shape
        of a load is not a reliable signal: when the arange size coincides
        with another reduction of the same size,
        ``allocate_reduction_dimension`` unifies them and the load axis
        surfaces as the rdim symbol rather than a concrete int.
        """
        env = CompileEnvironment.current()
        rdim_block_id = self.rdim.block_id

        def is_rdim_iota(node: object) -> bool:
            if not (
                isinstance(node, torch.fx.Node)
                and node.op == "call_function"
                and node.target is torch.ops.prims.iota.default
            ):
                return False
            val = node.meta.get("val", None)
            return isinstance(val, torch.Tensor) and any(
                env.resolve_block_id(size) == rdim_block_id for size in val.size()
            )

        def indexes_load(node: torch.fx.Node) -> bool:
            # load(tensor, index_list, ...): an iota in the index list cannot
            # be re-indexed in chunks inside the reduction loop.
            return any(
                user.target is load
                and any(
                    node in (arg if isinstance(arg, (list, tuple)) else (arg,))
                    for arg in user.args
                )
                for user in node.users
            )

        def inside_loop(node: torch.fx.Node) -> bool:
            if is_for_loop_target(node.target) or node.target is _if:
                return True
            try:
                return self.should_go_in_inner_graph(node)
            except NotImplementedError:
                return True

        def read_inside_loop(iota: torch.fx.Node, *, control_flow: bool = True) -> bool:
            # Whether a value inside the loop is computed from ``iota``; a
            # reduction of a value from outside the loop ends the walk, and so
            # does control flow unless ``control_flow`` counts it as inside.
            seen: set[torch.fx.Node] = set()
            stack = list(iota.users)
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                if self.is_reduction(node):
                    source = node.args[0]
                    if isinstance(source, torch.fx.Node) and inside_loop(source):
                        return True
                    continue
                if not control_flow and (
                    is_for_loop_target(node.target) or node.target is _if
                ):
                    continue
                if inside_loop(node):
                    return True
                stack.extend(node.users)
            return False

        def unrollable(reduction: torch.fx.Node) -> bool:
            free_iota = False
            seen: set[torch.fx.Node] = set()
            stack = list(reduction.all_input_nodes)
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                stack.extend(node.all_input_nodes)
                if not is_rdim_iota(node) or (
                    inside_loop(node) and not indexes_load(node)
                ):
                    continue
                if (
                    indexes_load(node)
                    or not env.backend.rolls_free_iota_reductions()
                    or read_inside_loop(node)
                ):
                    return True
                free_iota = True
            if free_iota:
                # Only the reductions the backend's lowering takes over roll
                # (CuTe's computed fragments); every other lowering would
                # refuse them.
                if reduction not in env.backend.free_iota_reductions(
                    self.device_ir.graphs
                ):
                    return True
                reduction.meta[FREE_IOTA_REDUCTION_META] = True
            return False

        # A free ``iota`` read inside the loop keeps the dim persistent also
        # when no reduction reads it.  The walk does not enter control flow:
        # CuTe's rank-1 recurrences pass such a range into a branch, and
        # counting the branch as a read would take the dim from them
        # (test_cute_single_token_rank1_recurrence.py).  A range read only
        # pointwise inside control flow is therefore not caught here.
        return any(
            (self.is_reduction(node) and unrollable(node))
            or (
                free_ranges
                and is_rdim_iota(node)
                and not inside_loop(node)
                and read_inside_loop(node, control_flow=False)
            )
            for node in graph.nodes
        )

    def reads_padded_range(self, graph: torch.fx.Graph) -> bool:
        """Check if a reduction over the rdim reads a padded ``iota`` on the
        rdim's lanes.

        ``hl.arange(200)`` of a static extent that is not a power of two runs
        over 256 lanes, and a load over it, ``hl.load(x, [i, hl.arange(200)])``
        in a grid kernel, has the padded size 256: a reduction over the load
        gets a reduction dim of 256 that names neither the range nor the load.
        The roller leaves both outside a loop over that dim, at their full
        extent, so every chunk would fold the same values (a shape mismatch on
        Triton).  Such a dim stays persistent, as it does when the range is
        256 long (``has_unrollable_rdim_use``), but keeps its reduction_loops
        slot, pinned (``ReductionLoopSpec.pinned``): older versions offered it.

        The walk follows the values on the rdim's lanes back from each
        reduction over it, to one computed from a padded range
        (``_padded_iota_dims``) or a load indexed by one.  A range reduced away
        before does not reach the lanes: ``hl.arange(200).sum()`` scaling a
        row, or a 3000-column load summed beside a 4096-column row.
        """
        size = self.rdim.size
        if not isinstance(size, int):
            return False
        env = CompileEnvironment.current()

        def on_rdim(node: torch.fx.Node) -> bool:
            val = node.meta.get("val")
            return isinstance(val, torch.Tensor) and any(
                env.resolve_block_id(dim) == self.rdim.block_id for dim in val.size()
            )

        # One walk over the graph, which meets a value on several paths.
        memo: dict[tuple[torch.fx.Node, bool], tuple[int | None, ...] | None] = {}

        def same_shape(node: torch.fx.Node, other: torch.fx.Node) -> bool:
            # ``known_equal``: comparing the symbolic sizes would add guards.
            val = node.meta.get("val")
            other_val = other.meta.get("val")
            return (
                isinstance(val, torch.Tensor)
                and isinstance(other_val, torch.Tensor)
                and val.ndim == other_val.ndim
                and all(map(env.known_equal, val.shape, other_val.shape))
            )

        def padded_to_rdim(node: torch.fx.Node, *, any_axis: bool) -> bool | None:
            """Whether a padded range's lanes are on an axis of ``node`` on the
            rdim's lanes, or with ``any_axis`` on any axis; None where no walk
            gives ``node`` padded dims (a reduction).  A load's dims are its
            indices' (``load_index_axes``): ``x[tile, rows, :]`` puts the
            range on its second dim, beside the 256 columns."""
            dims = _padded_iota_dims_once(node, False, memo)
            if (
                dims is None
                and node.target is load
                and (axes := load_index_axes(node)) is not None
            ):
                index = node.args[1]
                assert isinstance(index, (list, tuple))
                dims = padded_index_dims(
                    node,
                    axes,
                    lambda position: _padded_iota_dims_once(
                        index[position], False, memo
                    ),
                )
            if dims is None:
                return None
            val = node.meta.get("val")
            return any(
                length is not None
                and next_power_of_2(length) == size
                and (
                    any_axis
                    or not isinstance(val, torch.Tensor)
                    or val.ndim != len(dims)
                    or env.resolve_block_id(val.size(axis)) == self.rdim.block_id
                )
                for axis, length in enumerate(dims)
            )

        for node in graph.nodes:
            if not self.is_reduction(node):
                continue
            seen: set[torch.fx.Node] = set()
            stack = [arg for arg in node.all_input_nodes if on_rdim(arg)]
            while stack:
                value = stack.pop()
                if value in seen:
                    continue
                seen.add(value)
                # A value's padded dims follow its inputs' onto its own axes:
                # in ``hl.arange(200)[:, None] * row[None, :]`` the range pads
                # the first axis, which has a reduction dim of its own, not the
                # row's columns.  Where the walk does not follow a value, an
                # input of its shape is padded on the same axes, and one of
                # another shape (the range a load's padded size comes from)
                # wherever it is padded.
                found = padded_to_rdim(value, any_axis=False)
                if found or (
                    found is None
                    and any(
                        padded_to_rdim(arg, any_axis=not same_shape(arg, value))
                        for arg in value.all_input_nodes
                    )
                ):
                    return True
                stack.extend(arg for arg in value.all_input_nodes if on_rdim(arg))
        return False

    def has_stack_tensor_with_rdim(self, graph: torch.fx.Graph) -> bool:
        """Check if a graph contains stack tensors with rdim inputs."""

        def is_stack_with_rdim(node: torch.fx.Node) -> bool:
            """Check if a node is a stack dev_ptr with rdim inputs."""
            if node.op != "call_function":
                return False

            if node.target not in _MEMORY_OPS:
                return False

            host_tensor = node.args[0]

            if not isinstance(host_tensor, tuple):
                return False

            # Check if stack dims have rdim
            if len(host_tensor) == 2:
                assert isinstance(host_tensor[1], torch.fx.Node)
                stack = host_tensor[1].meta.get("val", None)
                if isinstance(stack, torch.Tensor):
                    for size in stack.size():
                        block_idx = CompileEnvironment.current().get_block_id(size)
                        if block_idx == self.rdim.block_id:
                            return True
            return False

        return any(is_stack_with_rdim(node) for node in graph.nodes)

    def _reorder_for_fusion(self, graph: torch.fx.Graph) -> list[torch.fx.Node]:
        """Defer each reduction's outer tail (its store plus the scalar ops
        feeding it) past sibling reductions so the roller folds independent
        reductions over ``self.rdim`` into one loop sharing a single load.
        Nodes move only where data, memory, and side-effect ordering allow."""

        def is_deferrable(node: torch.fx.Node) -> bool:
            if node.op != "call_function" or self.should_go_in_inner_graph(node):
                return False
            if node.target is store or node.target in _ATOMIC_OPS:
                target = node.args[0]
                if not isinstance(target, torch.fx.Node):
                    return False
                val = target.meta.get("val")
                if not isinstance(val, torch.Tensor):
                    return False
                origin = HostFunction.current().tensor_to_origin.get(val)
                return origin is not None and not origin.is_argument()
            return not (is_impure_node(node) or node.target is load)

        order: list[torch.fx.Node] = []
        deferred: list[torch.fx.Node] = []
        deferred_set: set[torch.fx.Node] = set()

        def flush_through(cutoff: int) -> None:
            # Preserve ordering among deferred nodes when releasing a dependency.
            order.extend(deferred[: cutoff + 1])
            del deferred[: cutoff + 1]
            deferred_set.intersection_update(deferred)

        for node in graph.nodes:
            if is_deferrable(node) and any(
                inp in deferred_set or self.is_reduction(inp)
                for inp in node.all_input_nodes
            ):
                deferred.append(node)
                deferred_set.add(node)
                continue
            if deferred:
                if node.op != "call_function" or is_impure_node(node):
                    flush_through(len(deferred) - 1)
                else:
                    is_load = node.target is load
                    inputs = set(node.all_input_nodes)
                    cutoff = -1
                    for i, deferred_node in enumerate(deferred):
                        if deferred_node in inputs or (
                            is_load
                            and (
                                deferred_node.target is store
                                or deferred_node.target in _ATOMIC_OPS
                            )
                        ):
                            cutoff = i
                    if cutoff >= 0:
                        flush_through(cutoff)
            order.append(node)
        order.extend(deferred)
        return order

    def process(self, graph: torch.fx.Graph) -> torch.fx.Graph:
        nodes = list(
            self._reorder_for_fusion(graph) if self.enable_fusion else graph.nodes
        )
        self._node_positions = {node: i for i, node in enumerate(nodes)}
        for node in nodes:
            has_effect = self._has_effect(node)
            if has_effect:
                # Include the current outer effect: flushing a producer before
                # it must not prepare a replay that will execute after it.
                self._effects_seen.append(node)
            if self.should_go_in_inner_graph(node):
                if not all(
                    (n in self.available or n in self.inner_available)
                    for n in node.all_input_nodes
                ):
                    self.start_new_graph()
                new_node = self.inner_graph.create_node(
                    node.op,
                    node.target,
                    *map_arg((node.args, node.kwargs), self.get_inner_arg),
                    name=node.name,
                )
                new_node.meta.update(node.meta)
                self.inner_nodes[node] = new_node
                self.inner_count += self.is_nontrivial(node)
                self._inner_has_effect |= has_effect
                if not self.is_reduction(node):
                    self.inner_available.add(node)
            else:
                if (
                    not all((n in self.available) for n in node.all_input_nodes)
                    or node.op == "output"
                    or (is_impure_node(node) and self.inner_count > 0)
                    or (self._reads_memory(node) and self._inner_has_effect)
                ):
                    self.start_new_graph()
                if has_effect and any(
                    self._effect_may_change_read(node, read)
                    for read in self._replayed_reads
                ):
                    # A previous flush may already have queued loads for the
                    # next sweep, without increasing inner_count. This effect
                    # will execute before those loads even when there is no
                    # new graph to flush yet.
                    self._record_replay_error(
                        "reduction sweeps cannot reload memory across an effect"
                    )
                new_node = self.outer_graph.create_node(
                    node.op,
                    node.target,
                    *map_arg((node.args, node.kwargs), self.outer_nodes.__getitem__),
                    name=node.name,
                )
                new_node.meta.update(node.meta)
                self.outer_nodes[node] = new_node
                self.outer_count += self.is_nontrivial(node)
                self.available.add(node)
            self.seen.add(node)
        if self._replay_error is not None:
            raise self._replay_error
        return self.outer_graph

    def is_nontrivial(self, node: torch.fx.Node) -> bool:
        """Check if a node should be counting in (outer|inner)_count"""
        return node.op == "call_function" and node.target not in _duplicate_ops
