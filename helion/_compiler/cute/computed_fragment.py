"""CTA-local SIMT lowering for multidimensional static fragments.

The scalar lowering gives each block ID one coordinate. Static dimensions,
including repeated dimensions of a matrix, need independent coordinates. This
path owns a complete root: its pure expressions retain logical index maps and
its loads, contractions, and loop carries are materialized in shared memory.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor.ir import Reduction
from torch._inductor.virtualized import V
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import creation_ops
from ...language import inline_asm_ops
from ...language import matmul_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_ops
from ...language import view_ops
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo
from ..device_ir import HelperFunctionGraphInfo
from ..device_ir import RootGraphInfo
from ..device_ir import control_flow_parent_entries
from ..host_function import HostFunction
from ..inductor_lowering import GenerateASTFromInductor
from ..inductor_lowering import PointwiseLowering
from ..inductor_lowering import ReductionLowering
from ..inductor_lowering import SympyExprLowering
from ..inductor_lowering import install_inductor_kernel_handlers
from . import target_facts
from .atomic_consumer_fusion import AtomicConsumerRegion
from .atomic_consumer_fusion import atomic_consumer_regions
from .direct_affine_plan import DIRECT_AFFINE_ORDINARY_SCHEDULE
from .fragment_atomics import FragmentAtomicEmitter
from .fragment_context import Fragment
from .fragment_context import HostTensor
from .fragment_context import _fragment_logical_shape
from .fragment_context import _loop_carry_slots
from .fragment_context import _resolve
from .fragment_context import _static_selection_extent
from .fragment_control_flow import FragmentControlFlow
from .fragment_dot import FragmentDotEmitter
from .fragment_expression import FragmentExpression
from .fragment_scan import FragmentScanEmitter
from .fragment_selection import FragmentSelectionEmitter
from .fragment_shared_arena import SharedArenaRequest
from .fragment_shared_arena import plan_shared_arena
from .full_reduction_owner import fragment_store_destinations
from .full_reduction_owner import requires_full_reduction_owner
from .local_atomic import local_atomic_allocations
from .local_atomic import prove_local_atomics
from .matmul_utils import cute_static_k_invariant_extent
from .native_s2f6_contract import native_s2f6_asm_matches
from .pure_producer_regions import pure_producer_plan
from .reduction_coordinates import captured_reduction_coordinates
from .reduction_coordinates import free_iota_reductions
from .reduction_coordinates import independent_reduction_coordinates
from .reduction_coordinates import owned_iota_reduction_axes
from .reduction_coordinates import physical_capture_axes
from .reduction_coordinates import reduction_coordinate_proof
from .repeated_block_ids import matmul_sum_collapse_nodes
from .repeated_block_ids import scalar_repeated_block_id
from .scalar_lowering import scalar_lowering_context
from .scan_analysis import additive_scan_helper
from .scan_analysis import completed_scan_plans
from .selection.ordered_selection import is_ordered_selection
from .selection.ordered_selection import ordered_selection_args
from .static_indexing import shape_only_indices
from .static_indexing import static_tensor_index

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..autotuner_heuristics.registry import CompilerHeuristicSpecializationFact
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .fragment_domain import OwnershipMask


class FragmentOps(GenerateASTFromInductor):
    """Use Inductor's index expressions instead of dropping them as SIMT does."""

    def __init__(
        self,
        compiler: FragmentCompiler,
        inputs: dict[str, tuple[Node, Fragment]],
        input_layouts: dict[str, tuple[list[sympy.Expr], list[sympy.Expr]]],
    ) -> None:
        super().__init__(compiler.cg, {})
        self.compiler = compiler
        self.inputs = inputs
        self.input_layouts = input_layouts

    def load(self, name: str, index: sympy.Expr) -> str:
        node, value = self.inputs[name]
        fake = node.meta["val"]
        if not isinstance(fake, torch.Tensor):
            return self._lift(expr_from_string(value.read(())))
        assert isinstance(fake, torch.Tensor)
        # The inner function indexes the layout saved when it was lowered.
        # A later _mask_to can replace that input with a contiguous fake value;
        # its logical coordinates remain the same, but its strides do not.
        # Buffers synthesized within a multi-buffer lowering have no saved
        # input layout; their current fake value describes that new buffer.
        layout = self.input_layouts.get(name)
        strides = self.compiler.shape(
            layout[1] if layout is not None else fake.stride()
        )
        indices = self.direct_coordinates(index, strides, value.shape)
        if indices is None:
            indices = ["0"] * len(value.shape)
            remaining = self.compiler.sym(index)
            for dim in sorted(
                range(len(strides)), key=lambda dim: strides[dim], reverse=True
            ):
                if value.shape[dim] == 1 or strides[dim] == 0:
                    continue
                indices[dim] = f"(({remaining}) // {strides[dim]})"
                remaining = f"(({remaining}) % {strides[dim]})"
        coordinates = self.compiler.coordinate_locals(tuple(indices))
        return self._lift(expr_from_string(value.read(coordinates)))

    def direct_coordinates(
        self, index: sympy.Expr, strides: tuple[int, ...], shape: tuple[int, ...]
    ) -> list[str] | None:
        """The input coordinates when *index* is exactly the layout's affine
        form ``sum(stride[d] * i_d)`` over the current coordinate symbols:
        each symbol names one dimension's coordinate outright, so no
        ``// stride`` / ``% stride`` round trip (which the compiler cannot
        fold without the symbols' ranges) is emitted.  None when a term is
        not a plain coordinate symbol, a constant offset remains, or two
        live dimensions share a stride."""
        symbols = self.compiler.sym_indices
        if not symbols:
            return None
        indices = ["0"] * len(shape)
        used: set[int] = set()
        # A singleton dimension's coordinate is the literal zero: its stride
        # term (which Inductor keeps when the stride is not zero) vanishes.
        index = index.xreplace(
            {
                symbol: sympy.Integer(0)
                for symbol, coord in symbols.items()
                if coord == "0"
            }
        )
        # Stride factors may still be size symbols (``Max(1, u2)``): resolve
        # them the way the staged shape resolved the strides themselves.
        resolved: dict[sympy.Basic, sympy.Integer] = {}
        for symbol in index.free_symbols:
            if symbol in symbols:
                continue
            extent = self.compiler.static_extent(symbol)
            if extent is None:
                return None
            resolved[symbol] = sympy.Integer(extent)
        index = sympy.expand(index.xreplace(resolved))
        for term, coefficient in index.as_coefficients_dict().items():
            if term == 1:
                if coefficient != 0:
                    return None
                continue
            if not isinstance(term, sympy.Symbol) or term not in symbols:
                return None
            if not coefficient.is_Integer:
                return None
            candidates = [
                dim
                for dim in range(len(shape))
                if dim not in used
                and shape[dim] != 1
                and strides[dim] != 0
                and strides[dim] == int(coefficient)
            ]
            if len(candidates) != 1:
                return None
            indices[candidates[0]] = symbols[term]
            used.add(candidates[0])
        return indices

    def to_dtype(
        self,
        x: object,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> str:
        # Fragment loads and shared buffers contain logical typed scalars.
        # The parent handler's FP8 byte decoder belongs to ordinary raw-byte
        # global loads; applying it to a typed Float8 register corrupts its bits.
        return self._lift(self._create_cast_expr(x, dtype))

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> str:
        return self._lift(
            self._cast_scalar_ast(expr_from_string(self.compiler.sym(expr)), dtype)
        )


class FragmentCompiler(
    FragmentDotEmitter,
    FragmentAtomicEmitter,
    FragmentSelectionEmitter,
    FragmentScanEmitter,
    FragmentControlFlow,
):
    """The computed-fragment root owner: the graph walk and node dispatch over
    the general-purpose fragment emitters, plus the inductor-backed pointwise
    lowering of a fragment node."""

    def pointwise_read(
        self,
        producer: object,
        coordinates: tuple[str, ...],
        evaluate: Callable[[], str],
    ) -> str:
        previous = self.expression
        if previous is None or previous.body is not self.cg.statements_stack[-1]:
            self.expression = FragmentExpression(self.cg)
        try:
            assert self.expression is not None
            return self.expression.read(producer, coordinates, evaluate)
        finally:
            self.expression = previous

    def pointwise(self, node: Node, values: dict[Node, object]) -> Fragment:
        lowering = cast("PointwiseLowering | ReductionLowering", node.meta["lowering"])
        inputs = lowering.input_nodes(node)
        lookup: dict[str, tuple[Node, Fragment]] = {}
        for name, source in zip(lowering.input_names, inputs, strict=True):
            value = values[source]
            if not isinstance(value, Fragment):
                scalar = self.scalar(value)
                value = Fragment((), torch.float32, lambda _, scalar=scalar: scalar)
            lookup[name] = (source, value)
        fake = cast("torch.Tensor", node.meta["val"])
        ranges = self.shape(lowering.buffer.data.ranges)
        result_shape = self.shape(fake.shape)
        # Inductor's iteration ranges usually differ from the fake shape only
        # by singleton dimensions: pass the live coordinates through instead
        # of a flatten/unflatten round trip the compiler cannot fold.
        range_live = [i for i, size in enumerate(ranges) if size != 1]
        result_live = [i for i, size in enumerate(result_shape) if size != 1]
        structural_ranges = [ranges[i] for i in range_live] == [
            result_shape[i] for i in result_live
        ]

        def element(
            coords: tuple[str, ...], reduction_coords: tuple[str, ...] = ()
        ) -> str:
            if ranges != result_shape:
                if structural_ranges:
                    mapped = ["0"] * len(ranges)
                    for range_dim, result_dim in zip(
                        range_live, result_live, strict=True
                    ):
                        mapped[range_dim] = coords[result_dim]
                    coords = tuple(mapped)
                else:
                    coords = self.coordinate_locals(
                        self.coordinates(self.flatten(coords, result_shape), ranges)
                    )
            prior = self.sym_indices
            self.sym_indices = {
                sympy.Symbol(f"i{i}"): coord
                for i, coord in enumerate((*coords, *reduction_coords))
            }
            try:
                with (
                    node.meta["location"],
                    V.set_current_node(node),
                    install_inductor_kernel_handlers(self.cg, {}),
                    V.set_ops_handler(
                        FragmentOps(self, lookup, lowering.input_layouts)
                    ),
                ):
                    indices = [
                        sympy.Symbol(f"i{i}")
                        for i in range(len(lowering.buffer.data.ranges))
                    ]
                    if isinstance(lowering, ReductionLowering):
                        reduction_indices = [
                            sympy.Symbol(f"i{i + len(indices)}")
                            for i in range(len(reduction_coords))
                        ]
                        return str(
                            lowering.buffer.data.inner_fn(indices, reduction_indices)
                        )
                    return str(lowering.buffer.data.inner_fn(indices))
            finally:
                self.sym_indices = prior

        dependencies = tuple(value for _, value in lookup.values())
        if isinstance(lowering, ReductionLowering):
            from .autotune.heuristics.cute_fragment_reduction import (
                fragment_warp_reduction_supported,
            )

            assert isinstance(lowering.buffer.data, Reduction)
            reduction_shape = self.shape(lowering.buffer.data.reduction_ranges)
            indexed = lowering.reduction_type in ("argmin", "argmax")
            value_dtype = lowering.buffer.data.src_dtype if indexed else fake.dtype
            compute_dtype = (
                torch.float32
                if value_dtype in (torch.float16, torch.bfloat16)
                else value_dtype
            )
            warp = self.df.config.get(
                "cute_fragment_reduction", "serial"
            ) == "warp" and fragment_warp_reduction_supported(node)

            def combine(left: str, right: str) -> str:
                expression = self.env.backend.reduction_combine_expr(
                    lowering.reduction_type, left, right, compute_dtype
                )
                if (
                    lowering.reduction_type in ("min", "max")
                    and compute_dtype.is_floating_point
                ):
                    # A parallel fold must preserve NaNs from either operand,
                    # including a NaN encountered in a preceding local tile.
                    expression = f"{left} if {left} != {left} else ({expression})"
                return expression

            if node in self.scan_reductions:
                plan = self.scan_reductions[node]
                result = self.allocate(
                    Fragment(result_shape, fake.dtype, lambda _: "0")
                )
                if not self.scan_pending:
                    self.held.append([])
                cast("list[object]", self.held[-1]).extend([result, *dependencies])
                self.scan_pending.append(
                    (result, element, compute_dtype, lowering.reduction_type)
                )
                if node is plan.reductions[-1]:
                    self.completed_scan_reductions(reduction_shape)
                    self.scan_pending.clear()
                    self.held.pop()
                return result

            if node in self.snapshot_reductions:
                if (
                    self.shape(fake.shape) != ()
                    or len(reduction_shape) != 1
                    or reduction_shape not in self.snapshot_shapes
                    or compute_dtype not in (torch.int32, torch.int64)
                    or lowering.reduction_type not in ("sum", "min", "max")
                ):
                    raise exc.InvalidConfig(
                        "register snapshot requires an exact scalar integer reduction"
                    )
                total = self.df.new_var("fragment_snapshot_reduce")
                identity = self.cast(
                    self.scalar(
                        Reduction.default_accumulator(
                            lowering.reduction_type, compute_dtype
                        )
                    ),
                    compute_dtype,
                )
                self.emit(f"{total} = {identity}")

                def partial(coords: tuple[str, ...]) -> None:
                    value = element((), coords)
                    self.emit(
                        f"{total} = {self.cast(combine(total, value), compute_dtype)}"
                    )

                self.elements(reduction_shape, partial, synchronize=False)
                peer = self.df.new_var("fragment_snapshot_peer")
                for distance in (16, 8, 4, 2, 1):
                    self.emit(
                        f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                    )
                    self.emit(
                        f"{total} = {self.cast(combine(total, peer), compute_dtype)}"
                    )
                scratch = self.allocate(
                    Fragment((self.threads // 32,), compute_dtype, lambda _: "0")
                )
                self.held.append(scratch)
                self.emit(
                    f"if {self.thread} % 32 == 0:\n    {scratch.read((f'{self.thread} // 32',))} = {total}"
                )
                self.synchronize()
                result = self.allocate(Fragment((), fake.dtype, lambda _: "0"))
                branch = cast(
                    "ast.If", statement_from_string(f"if {self.thread} < 32:\n    pass")
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    self.emit(f"{total} = {identity}")
                    self.emit(
                        f"if {self.thread} < {self.threads // 32}:\n    {total} = {scratch.read((self.thread,))}"
                    )
                    for distance in (16, 8, 4, 2, 1):
                        self.emit(
                            f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                        )
                        self.emit(
                            f"{total} = {self.cast(combine(total, peer), compute_dtype)}"
                        )
                    self.emit(
                        f"if {self.thread} == 0:\n    {result.read(())} = {self.cast(total, fake.dtype)}"
                    )
                self.cg.add_statement(branch)
                self.synchronize()
                self.held.pop()
                return result

            if self.df.config.get("cute_fragment_reduction", "serial") in (
                "cta",
                "cta_unroll",
            ) and fragment_warp_reduction_supported(node):
                return self.cta_reduction(
                    result_shape,
                    fake.dtype,
                    reduction_shape,
                    compute_dtype,
                    lowering.reduction_type,
                    element,
                    combine,
                    dependencies,
                )

            def reduce(coords: tuple[str, ...]) -> str:
                reduction_type = lowering.reduction_type
                total = self.df.new_var(f"fragment_{reduction_type}")
                index = self.df.new_var("fragment_reduce_index")
                identity = (
                    0
                    if indexed
                    else Reduction.default_accumulator(reduction_type, compute_dtype)
                )
                self.emit(
                    f"{total} = {self.cast(self.scalar(identity), compute_dtype)}"
                )
                selected = self.df.new_var("fragment_selected_index") if indexed else ""
                if indexed:
                    self.emit(f"{selected} = {self.cast('0', fake.dtype)}")
                reduction_range = str(math.prod(reduction_shape))
                if warp:
                    reduction_range = f"{self.thread} % 32, {reduction_range}, 32"
                loop = statement_from_string(
                    f"for {index} in range({reduction_range}):\n    pass"
                )
                # The fragment owner retains full producer coordinates and
                # carries this accumulator across configured reduction tiles.
                # Generic graph rolling would instead capture a full fragment
                # inside a tile-local graph, losing both its coordinate window
                # and its implicit reduction carry.
                tile = self.df.config.reduction_loops
                block = lowering.block_index
                tile = self.env.config_spec.reduction_loops.config_get(
                    tile, block, None
                )
                tiled_loop: ast.For | None = None
                if tile is not None:
                    tile = self.extent(tile)
                    tile_begin = self.df.new_var("fragment_reduce_tile")
                    tiled_loop = cast(
                        "ast.For",
                        statement_from_string(
                            f"for {tile_begin} in range(0, {math.prod(reduction_shape)}, {tile}):\n    pass"
                        ),
                    )
                    assert isinstance(tiled_loop, ast.For)
                    start = (
                        f"{tile_begin} + ({self.thread} % 32)" if warp else tile_begin
                    )
                    stop = f"min({tile_begin} + {tile}, {math.prod(reduction_shape)})"
                    reduction_range = f"{start}, {stop}" + (", 32" if warp else "")
                    loop = statement_from_string(
                        f"for {index} in range({reduction_range}):\n    pass"
                    )
                assert isinstance(loop, ast.For)
                loop.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", loop.body)):
                    value = element(coords, self.coordinates(index, reduction_shape))
                    if indexed:
                        value = self.cast(value, compute_dtype)
                        compare = ">" if reduction_type == "argmax" else "<"
                        better = f"({value}) {compare} {total}"
                        if compute_dtype.is_floating_point:
                            better = f"({better}) or (({value}) != ({value}) and {total} == {total})"
                        # The increasing local reduction coordinate and strict
                        # comparison preserve the first tie, including NaNs.
                        self.emit(
                            f"if {index} == 0 or ({better}):\n"
                            f"    {total} = {value}\n"
                            f"    {selected} = {self.cast(index, fake.dtype)}"
                        )
                    else:
                        self.emit(f"{total} = {combine(total, value)}")
                if tiled_loop is not None:
                    tiled_loop.body = [loop]
                    self.cg.add_statement(tiled_loop)
                else:
                    self.cg.add_statement(loop)
                if warp:
                    # Output coordinates are warp-uniform, including a short
                    # final output tile. Lanes with no input use the identity;
                    # the complete warp reconverges before each shuffle.
                    peer = self.df.new_var("fragment_reduce_peer")
                    for distance in (16, 8, 4, 2, 1):
                        self.emit(
                            f"{peer} = cute.arch.shuffle_sync_bfly({total}, offset={distance}, mask=0xffffffff, mask_and_clamp=31)"
                        )
                        self.emit(f"{total} = {combine(total, peer)}")
                return selected if indexed else self.cast(total, fake.dtype)

            if node in self.warp_result_nodes:
                groups = self.warp_result_nodes[node]
                name = self.df.new_var("fragment_warp_reduction")
                self.emit(f"{name} = {self.cast('0', fake.dtype)}")
                branch = cast(
                    "ast.If",
                    statement_from_string(
                        f"if {self.thread} // 32 < {groups}:\n    pass"
                    ),
                )
                branch.body.clear()
                with self.cg.set_statements(cast("list[ast.AST]", branch.body)):
                    coords = self.coordinates(
                        f"{self.thread} // 32", self.shape(fake.shape)
                    )
                    if warp:
                        self.emit(f"{name} = {reduce(coords)}")
                    else:
                        leader = cast(
                            "ast.If",
                            statement_from_string(
                                f"if {self.thread} % 32 == 0:\n    pass"
                            ),
                        )
                        leader.body.clear()
                        with self.cg.set_statements(cast("list[ast.AST]", leader.body)):
                            self.emit(f"{name} = {reduce(coords)}")
                        self.cg.add_statement(leader)
                        self.emit(
                            f"{name} = cute.arch.shuffle_sync({name}, 0, mask=0xffffffff, mask_and_clamp=31)"
                        )
                self.cg.add_statement(branch)
                # Keep the existing lifetime/mutation ordering boundary.
                self.synchronize()
                return Fragment(
                    self.shape(fake.shape), fake.dtype, lambda _coords: name, True
                )
            return self.materialize(
                Fragment(
                    self.shape(fake.shape),
                    fake.dtype,
                    reduce,
                    dependencies=dependencies,
                ),
                threads_per_element=32 if warp else 1,
            )
        producer = object()
        logical_dependencies = tuple(
            value for value in dependencies if value.logical_domain is not None
        )
        # These elementwise operations preserve broadcast coordinates. Other
        # lowerings (for example gather) define a different output domain.
        pointwise_domain = isinstance(node.target, torch._ops.OpOverload) and (
            torch.Tag.pointwise in node.target.tags
            or node.target is torch.ops.aten._to_copy.default
        )
        # A bare dtype cast of one staged tile (or of a cheap view of one)
        # reads exactly that tile: the mma lanes may read it in place.
        cheap = (
            node.target
            in (
                torch.ops.aten._to_copy.default,
                torch.ops.prims.convert_element_type.default,
            )
            and len(dependencies) == 1
            and ranges == result_shape
            and (dependencies[0].cheap or dependencies[0].physical_row is not None)
            and dependencies[0].shape == result_shape
        )
        return Fragment(
            self.shape(fake.shape),
            fake.dtype,
            lambda coords: self.pointwise_read(
                producer, coords, lambda: self.cast(element(coords), fake.dtype)
            ),
            dependencies=dependencies,
            logical_domain=(
                lambda coords: tuple(
                    mask
                    for value in logical_dependencies
                    for mask in value.broadcast_domain(coords)
                )
            )
            if pointwise_domain and logical_dependencies
            else None,
            cheap=cheap,
            physical_row=dependencies[0].physical_row if cheap else None,
        )

    def graph(
        self,
        graph: torch.fx.Graph,
        values: dict[Node, object],
        *,
        output_indices: frozenset[int] | None = None,
    ) -> object:
        from .producer_cache import producer_cache_candidates

        cache_nodes = (
            producer_cache_candidates(graph, self.shape)
            if self.df.config.get("cute_fragment_producer_cache", False)
            and not self.scopes
            else frozenset()
        )
        previous_opaque = self.lazy_opaque_producers
        pure_plan = (
            pure_producer_plan(
                graph,
                self.env,
                cache_nodes=cache_nodes,
                shape=self.shape,
                graphs=self.graphs,
            )
            if self.df.config.get("cute_fragment_pure_producer_regions", False)
            and not self.scopes
            else None
        )
        self.lazy_opaque_producers = pure_plan.lazy if pure_plan else frozenset()
        if pure_plan is not None:
            cache_nodes = (cache_nodes - pure_plan.replicated) | pure_plan.publications
        previous_scans, previous_reductions = self.scan_plans, self.scan_reductions
        plans = (
            completed_scan_plans(graph, self.shape)
            if self.df.config.get("cute_fragment_scan_exports", False)
            and not self.scopes
            else ()
        )
        self.scan_plans = {plan.scan: plan for plan in plans}
        self.scan_reductions = {
            node: plan for plan in plans for node in plan.reductions
        }
        uses = {node: len(node.users) for node in graph.nodes}
        dependencies = {node: node.all_input_nodes for node in graph.nodes}
        for node in graph.nodes:
            if node.target is _tracing_ops._phi:
                # This emitter's loop carries are initialized before the loop,
                # and its two branch extractions already name one joined slot.
                # The phi selects args[1]; args[0] is tracing bookkeeping only.
                selected = node.args[1]
                dependencies[node] = [selected] if isinstance(selected, Node) else []
                for source in node.all_input_nodes:
                    if source not in dependencies[node]:
                        uses[source] -= 1
        if output_indices is not None:
            output = graph.find_nodes(op="output")[0]
            returns = cast("list[Node]", output.args[0])
            retained = {returns[index] for index in output_indices}
            # An FX user is one edge even if a value occurs in several slots.
            # Remove only the unobserved return edge, not any executable use.
            for source in output.all_input_nodes:
                if source not in retained:
                    uses[source] -= 1
        regions = (
            atomic_consumer_regions(self.graphs, self.env)
            if self.df.config.get("cute_fragment_atomic_consumer_fusion", False)
            and not self.scopes
            else {}
        )
        region: AtomicConsumerRegion | None = None
        updates: list[tuple[Callable[[tuple[str, ...]], None], bool]] = []
        held: list[object] = []
        previous_cache_nodes, previous_region = self.cache_nodes, self.active_region
        self.cache_nodes = frozenset(cache_nodes)
        self.active_region = None
        self.scopes.append(values)
        self.scope_uses.append(uses)
        try:
            for node in graph.nodes:
                if self.warp_producer_recorder is not None:
                    self.warp_producer_recorder.enter(node)
                if node.op == "placeholder":
                    continue
                if node.op == "output":
                    assert region is None
                    self.flush_pending_barrier()
                    if output_indices is not None:
                        return [
                            _resolve(source, values)
                            if index in output_indices
                            else None
                            for index, source in enumerate(
                                cast("list[Node]", node.args[0])
                            )
                        ]
                    return _resolve(node.args[0], values)
                if (
                    node in regions
                    and not cache_nodes.intersection(regions[node].nodes)
                    and self.can_fuse_atomics(regions[node], values)
                ):
                    assert region is None
                    region = regions[node]
                    self.active_region = region
                    assert not self.iteration_local_returns
                    self.iteration_local_returns = frozenset(region.local_returns)
                    updates, held = [], []
                    self.held.append(held)
                if region is not None:
                    held.extend(
                        (
                            _resolve(node.args, values),
                            [_resolve(value, values) for value in node.kwargs.values()],
                        )
                    )
                with node.meta["location"], V.set_current_node(node):
                    values[node] = (
                        self.node(node, values, updates)
                        if region is not None
                        and node in (*region.atomics, *region.stores)
                        else self.node(node, values)
                    )
                    if region is not None:
                        held.append(values[node])
                    if node in cache_nodes:
                        value = values[node]
                        assert isinstance(value, Fragment)
                        values[node] = self.materialize(value)
                for source in dependencies[node]:
                    uses[source] -= 1
                    if uses[source] == 0:
                        values.pop(source, None)
                if uses[node] == 0:
                    values.pop(node, None)
                if region is not None and node is region.nodes[-1]:
                    self.fused_atomic_updates(region, updates)
                    self.held.pop()
                    region = None
                    self.active_region = None
                    self.iteration_local_returns = frozenset()
        finally:
            self.iteration_local_returns = frozenset()
            self.lazy_opaque_producers = previous_opaque
            self.scan_plans, self.scan_reductions = previous_scans, previous_reductions
            self.cache_nodes, self.active_region = previous_cache_nodes, previous_region
            self.scopes.pop()
            self.scope_uses.pop()
        raise AssertionError("graph has no output")

    def node(
        self,
        node: Node,
        values: dict[Node, object],
        deferred_updates: list[tuple[Callable[[tuple[str, ...]], None], bool]]
        | None = None,
    ) -> object:
        target = node.target
        # A run of batched host loads ends at the first node that is neither
        # one of them nor a lazy recipe (which emits nothing until consumed).
        if (
            self.pending_barrier
            and not self.batchable_host_load(node, values)
            and not self.transparent_node(node)
        ):
            self.synchronize()
        if target is _tracing_ops._phi:
            return _resolve(node.args[1], values)
        args = cast("tuple[object, ...]", _resolve(node.args, values))
        fake = node.meta.get("val")
        if target is _tracing_ops._host_tensor:
            assert isinstance(fake, torch.Tensor)
            return HostTensor(
                self.df.tensor_arg(fake, prefer_name=cast("str", node.args[0])).name,
                fake,
            )
        if target in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int):
            return fake
        if target is _tracing_ops._new_var:
            return args[0]
        if target is operator.getitem:
            sequence, index = args
            assert isinstance(sequence, (list, tuple)) and isinstance(index, int)
            return sequence[index]
        # CTA-local targets have a proved initialize/update/final-read lifetime.
        # Coalesce only updates to disjoint targets. Repeated updates to one
        # target retain their collective epoch boundary, including floating
        # association across threads. Finish pending writes before a read,
        # storage reuse, or conservative effect boundary.
        # Alias-only nodes above do not access memory. Check physical storage,
        # including lazy dependencies, so consumers cannot hide an alias.
        local_update = (
            target is atomic_ops.atomic_add
            and isinstance(args[0], Fragment)
            and id(args[0]) in self.local_logical_sizes
            and not node.users
        )
        read_args = args[1:] if local_update else args
        if (
            target in (_tracing_ops._if, _tracing_ops._while_loop)
            or _tracing_ops.is_for_loop_target(target)
            or (target is atomic_ops.atomic_add and not local_update)
            or (
                local_update
                and cast("Fragment", args[0]).storage in self.pending_local_atomics
            )
            or self.pending_local_atomics
            & self.referenced_buffers(
                [
                    *read_args,
                    *(_resolve(value, values) for value in node.kwargs.values()),
                ]
            )
        ):
            # A loop can execute zero times. Never let a barrier generated only
            # in its body discharge a write from before that loop.
            self.synchronize_local_atomics()
        if target is _tracing_ops._if:
            return self.conditional(node, values)
        if target is _tracing_ops._while_loop:
            return self.resident_while(node, values)
        if _tracing_ops.is_for_loop_target(target):
            return self.loop(node, values)
        if target is atomic_ops.atomic_add:
            if deferred_updates is not None:
                return self.atomic_add(node, values, deferred_updates)
            return self.atomic_add(node, values)
        if target in (memory_ops.load, memory_ops.store):
            return self.memory(
                node, values, target is memory_ops.store, deferred_updates
            )
        if is_ordered_selection(node):
            return self.topk(node, values)
        if target is torch.ops.aten.gather.default:
            from .bounded_gather import prove_gather

            if (
                self.df.config.get("cute_fragment_bounded_gather", False) is True
                and prove_gather(self.env, node, graphs=self.graphs) is not None
            ):
                return self.bounded_gather(node, values)
            return self.topk_gather(node, values)
        if target is matmul_ops.dot:
            return self.dot(node, values)
        if target is scan_ops._associative_scan:
            return self.scan(node, values)
        if target is inline_asm_ops.inline_asm_elementwise:
            assert isinstance(fake, torch.Tensor)
            asm, constraints, operands, dtype, _is_pure, _pack = args
            assert isinstance(operands, (tuple, list))
            fragments = tuple(cast("Fragment", operand) for operand in operands)

            def inline_asm(coords: tuple[str, ...]) -> str:
                inputs = ", ".join(
                    self.cast(operand.broadcast(coords), operand.dtype)
                    for operand in fragments
                )
                arguments = f"({inputs},)" if inputs else "()"
                return (
                    f"_cute_inline_asm_elementwise({arguments}, asm={asm!r}, "
                    f"constraints={constraints!r}, dtype={self.dtype(fake.dtype)}, "
                    "is_pure=True)"
                )

            # This exact compiler decoder reads only its two scalar inputs;
            # unlike arbitrary "pure" ASM it cannot observe the executing lane.
            # Keep it pointwise like the integer recipe it replaces, rather
            # than adding a shared-memory publication that may exceed capacity.
            native_decode = native_s2f6_asm_matches(
                asm, constraints, dtype, _is_pure, _pack, len(fragments)
            ) and all(operand.dtype is torch.int32 for operand in fragments)
            if native_decode or node in self.lazy_opaque_producers:
                return Fragment(
                    self.shape(fake.shape),
                    fake.dtype,
                    lambda coords: self.pointwise_read(
                        node, coords, lambda: self.cast(inline_asm(coords), fake.dtype)
                    ),
                    dependencies=fragments,
                )

            # Opaque scalar programs may be expensive and can feed several
            # consumers. Evaluate each logical element once before sharing it.
            return self.materialize(
                Fragment(
                    self.shape(fake.shape),
                    fake.dtype,
                    inline_asm,
                    dependencies=fragments,
                )
            )
        if target in (tile_ops.tile_begin, tile_ops.tile_end, tile_ops.tile_id):
            return fake
        if target is tile_ops.tile_index:
            assert isinstance(fake, torch.Tensor)
            block_id = self.env.resolve_block_id(args[0])
            assert block_id in self.offsets
            offset = self.offsets[block_id]
            bound = self.bounds[block_id]
            return Fragment(
                self.shape(fake.shape),
                fake.dtype,
                lambda coords: self.cast(f"({offset}) + ({coords[0]})", fake.dtype),
                logical_domain=lambda coords: (
                    f"({offset}) + ({coords[0]}) < ({bound})",
                ),
                pure=True,
            )
        if target in (torch.ops.aten.fmin.default, torch.ops.aten.fmax.default):
            assert isinstance(fake, torch.Tensor)
            inputs = tuple(cast("Fragment", value) for value in args)
            logical_inputs = tuple(value for value in inputs if value.logical_domain)
            lowering = node.meta["lowering"]

            def numeric_extremum(coords: tuple[str, ...]) -> str:
                operands = {
                    cast("Node", source): expr_from_string(value.broadcast(coords))
                    for source, value in zip(node.args, inputs, strict=True)
                }
                context = scalar_lowering_context(self.cg, operands)
                expression = lowering.codegen(context, node)
                assert isinstance(expression, ast.AST)
                return self.cast(ast.unparse(expression), fake.dtype)

            return Fragment(
                self.shape(fake.shape),
                fake.dtype,
                lambda coords: self.pointwise_read(
                    node, coords, lambda: numeric_extremum(coords)
                ),
                dependencies=inputs,
                logical_domain=(
                    lambda coords: tuple(
                        mask
                        for value in logical_inputs
                        for mask in value.broadcast_domain(coords)
                    )
                )
                if logical_inputs
                else None,
            )
        if isinstance(node.meta["lowering"], (PointwiseLowering, ReductionLowering)):
            return self.pointwise(node, values)
        if not isinstance(fake, torch.Tensor):
            return fake
        shape = self.shape(fake.shape)
        if target in (
            creation_ops.full,
            torch.ops.aten.full.default,
            torch.ops.aten.scalar_tensor.default,
            _tracing_ops._constant_tensor,
        ):
            value = (
                args[1]
                if target in (creation_ops.full, torch.ops.aten.full.default)
                else args[0]
            )
            declared_shape = (
                self.shape(args[0])
                if target in (creation_ops.full, torch.ops.aten.full.default)
                else shape
            )
            result = Fragment(
                shape,
                fake.dtype,
                lambda _: self.cast(self.scalar(value), fake.dtype),
                dependencies=(value,) if isinstance(value, Fragment) else (),
                logical_domain=(
                    lambda coords: tuple(
                        f"({coord}) < ({size})"
                        for coord, size in zip(coords, declared_shape, strict=True)
                    )
                )
                if declared_shape != shape
                else None,
                pure=not isinstance(value, Fragment),
            )
            if node in self.local_allocations:
                result = self.materialize(result)
                assert result.storage is not None
                self.local_storage.append(result)
                self.local_logical_sizes[id(result)] = cast("list[int]", node.args[0])[
                    0
                ]
            return result
        if target is torch.ops.prims.iota.default:
            start = _resolve(node.kwargs.get("start", 0), values)
            step = _resolve(node.kwargs.get("step", 1), values)
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: self.cast(
                    f"{self.scalar(start)} + ({coords[0]}) * {self.scalar(step)}",
                    fake.dtype,
                ),
                pure=True,
                affine=(start, step)
                if len(shape) == 1
                and isinstance(start, int)
                and isinstance(step, int)
                and not isinstance(start, bool)
                and not isinstance(step, bool)
                else None,
            )
        source = args[0]
        assert isinstance(source, Fragment)
        if target is torch.ops.aten.alias.default:
            return source
        if target is torch.ops.aten.view.dtype:
            # Equal-width reinterpretation retains element ownership. Round
            # any widened arithmetic back to its declared dtype first.
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: (
                    f"{self.cast(source.read(coords), source.dtype)}"
                    f".bitcast({self.dtype(fake.dtype)})"
                ),
                dependencies=(source,),
                logical_domain=source.logical_domain,
            )
        if target is _tracing_ops._mask_to:
            source_fake = cast("torch.Tensor", cast("Node", node.args[0]).meta["val"])

            def masked(coords: tuple[str, ...]) -> str:
                # A local allocation can have padded physical capacity while
                # retaining its declared logical domain. Its initialized pad
                # values are not necessarily neutral for this reduction.
                masks: list[OwnershipMask] = (
                    list(source.domain(coords)) if source.logical_domain else []
                )
                for coord, size in zip(coords, source_fake.shape, strict=True):
                    bid = self.env.resolve_block_id(size)
                    if bid is None:
                        continue
                    if bid in self.offsets:
                        masks.append(
                            f"(({self.offsets[bid]}) + ({coord}) < ({self.bounds[bid]}))"
                        )
                    elif self.env.block_sizes[bid].reduction:
                        masks.append(
                            f"(({coord}) < ({self.sym(self.env.block_sizes[bid].numel)}))"
                        )
                mask = self.predicate(masks)
                return f"({self.cast(source.read(coords), fake.dtype)} if {mask} else {self.cast(self.scalar(args[1]), fake.dtype)})"

            return Fragment(
                shape,
                fake.dtype,
                masked,
                dependencies=(source,),
                logical_domain=source.logical_domain,
            )
        tensor_index = static_tensor_index(node)
        if target is view_ops.subscript or tensor_index is not None:
            slices = (
                tensor_index[1]
                if tensor_index is not None
                else cast("list[object]", args[1])
            )

            def subscript_coordinates(coords: tuple[str, ...]) -> tuple[str, ...]:
                result: list[str] = []
                dim = 0
                source_dim = 0
                for item in slices:
                    if item is None:
                        dim += 1
                    elif isinstance(item, slice):
                        begin = item.start or 0
                        step = item.step or 1
                        result.append(
                            f"({self.scalar(begin)}) + ({coords[dim]}) * ({self.scalar(step)})"
                        )
                        dim += 1
                        source_dim += 1
                    else:
                        position = self.scalar(item)
                        result.append(f"({position}) % {source.shape[source_dim]}")
                        source_dim += 1
                result.extend(coords[dim:])
                return tuple(result)

            def subscript(coords: tuple[str, ...]) -> str:
                return source.read(
                    self.coordinate_locals(subscript_coordinates(coords))
                )

            # ``x[None, :, :]``: new singleton axes and whole slices only keep
            # the staged tile's row order.
            cheap_view = (
                (source.cheap or source.physical_row is not None)
                and all(item is None or item == slice(None) for item in slices)
                and len(shape) >= 2
                and len(source.shape) >= 2
                and slices[-1] == slice(None)
                and (len(slices) < 2 or slices[-2] == slice(None))
            )
            return Fragment(
                shape,
                fake.dtype,
                subscript,
                source.resident,
                (source,),
                logical_domain=lambda coords: source.domain(
                    subscript_coordinates(coords)
                ),
                cheap=cheap_view,
                physical_row=source.physical_row if cheap_view else None,
            )
        if target is torch.ops.aten.permute.default:
            order = cast("list[int]", args[1])
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: source.read(
                    tuple(coords[order.index(i)] for i in range(len(order)))
                ),
                source.resident,
                (source,),
                logical_domain=lambda coords: source.domain(
                    tuple(coords[order.index(i)] for i in range(len(order)))
                ),
            )
        if target in (torch.ops.aten.expand.default, torch.ops.aten.clone.default):
            # A broadcast of a staged tile (or of a cheap view of one) keeps
            # the tile's own row order wherever it is not broadcast.
            cheap_view = (
                (source.cheap or source.physical_row is not None)
                and len(shape) >= 2
                and source.shape[-2:] == shape[-2:]
            )
            return Fragment(
                shape,
                fake.dtype,
                source.broadcast,
                source.resident,
                (source,),
                logical_domain=source.broadcast_domain,
                cheap=cheap_view,
                physical_row=source.physical_row if cheap_view else None,
            )
        if target in (
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.dims,
        ):
            # Singleton views preserve logical element order. Read through the
            # source recipe so permuted layouts and resident aliases retain
            # their own physical addressing and shared-buffer lifetimes.
            source_live = [i for i, size in enumerate(source.shape) if size != 1]
            target_live = [i for i, size in enumerate(shape) if size != 1]
            structural = [source.shape[i] for i in source_live] == [
                shape[i] for i in target_live
            ]

            def view_coordinates(coords: tuple[str, ...]) -> tuple[str, ...]:
                if structural:
                    # The view only inserts or drops singleton dimensions:
                    # each live coordinate passes through unchanged, with no
                    # flatten/unflatten round trip for the compiler to fold.
                    mapped = ["0"] * len(source.shape)
                    for source_dim, target_dim in zip(
                        source_live, target_live, strict=True
                    ):
                        mapped[source_dim] = coords[target_dim]
                    return tuple(mapped)
                return self.coordinates(self.flatten(coords, shape), source.shape)

            def view_domain(coords: tuple[str, ...]) -> tuple[OwnershipMask, ...]:
                if structural:
                    return source.domain(view_coordinates(coords))
                # Indexing may pad a singleton view without preserving numel.
                # Bound the flat coordinate before unflattening can wrap it.
                return (
                    f"({self.flatten(coords, shape)}) < ({math.prod(source.shape)})",
                    *source.domain(view_coordinates(coords)),
                )

            # A singleton-only view of a staged tile (or of a cheap view of
            # one) whose last two live dimensions are the tile's keeps the
            # tile's row order.
            cheap_view = (
                structural
                and (source.cheap or source.physical_row is not None)
                and len(target_live) >= 2
                and len(source_live) >= 2
                and target_live[-1] == len(shape) - 1
                and source_live[-1] == len(source.shape) - 1
            )
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: source.read(
                    self.coordinate_locals(view_coordinates(coords))
                ),
                source.resident,
                (source,),
                logical_domain=view_domain,
                cheap=cheap_view,
                physical_row=source.physical_row if cheap_view else None,
            )
        if target is torch.ops.aten.where.self:
            condition, lhs, rhs = cast("tuple[Fragment, Fragment, Fragment]", args)
            logical_dependencies = tuple(
                value
                for value in (condition, lhs, rhs)
                if value.logical_domain is not None
            )
            return Fragment(
                shape,
                fake.dtype,
                lambda coords: (
                    f"({self.cast(lhs.broadcast(coords), fake.dtype)} if {condition.broadcast(coords)} else {self.cast(rhs.broadcast(coords), fake.dtype)})"
                ),
                dependencies=(condition, lhs, rhs),
                # All operands share the broadcast output domain, regardless of
                # which value the predicate selects. Preserve logical tails for
                # the reduction's neutral mask; load/value masks are separate.
                logical_domain=(
                    lambda coords: tuple(
                        mask
                        for value in logical_dependencies
                        for mask in value.broadcast_domain(coords)
                    )
                )
                if logical_dependencies
                else None,
            )
        raise exc.BackendUnsupported("cute", f"computed fragment operation {target}")


def computed_fragment_supported(
    env: CompileEnvironment,
    graphs: list[GraphInfo],
    *,
    physical_axes: frozenset[int] = frozenset(),
    owned_iota_axes: frozenset[int] = frozenset(),
    snapshot_owned: bool = False,
    bounded_gather_owned: bool = False,
    allow_unbound: bool = False,
) -> bool:
    """Structural root ownership shared by search discovery and code generation.

    Exact local sizes, strides, scalar predicates and shared capacity remain
    configuration-dependent checks in the emitter.
    """
    # Jagged loops carry per-parent bounds in their first captured tensor.
    # This owner currently tracks only scalar loop ends; preserve the ordinary
    # jagged lowering instead of treating the maximum end as every row's end.
    if any(
        isinstance(info, ForLoopGraphInfo)
        and any(env.is_jagged_tile(bid) for bid in info.block_ids)
        for info in graphs
    ):
        return False
    while_calls = {
        node
        for info in graphs
        for node in info.graph.nodes
        if node.target is _tracing_ops._while_loop
    }
    if while_calls:
        from .gather_domains import loop_domain_facts
        from .resident_while import resident_while_plan

        try:
            plans = [resident_while_plan(call, graphs) for call in while_calls]
            mutable_while = any(plan.local_allocations for plan in plans)
            needs_bounded_gather = any(
                not plan.local_allocations and not plan.readonly_control
                for plan in plans
            )
            for plan in plans:
                if plan.readonly_control:
                    from .uniform_region_tree import uniform_output_stores_are_writeonly
                    from .uniform_region_tree import uniform_readonly_regions

                    if not uniform_output_stores_are_writeonly(
                        env,
                        graphs,
                        uniform_readonly_regions(graphs),
                        allow_unbound=allow_unbound,
                    ):
                        return False
            if mutable_while:
                prove_local_atomics(graphs)
        except exc.InvalidConfig:
            return False

        if (
            (not bounded_gather_owned and needs_bounded_gather)
            or loop_domain_facts(
                env, graphs, allow_unbound=allow_unbound
            ).resident_whiles
            != while_calls
            or (
                not mutable_while
                and any(
                    node.target in atomic_ops.ATOMIC_OPS
                    for info in graphs
                    for node in info.graph.nodes
                )
            )
        ):
            return False
    # If-only frames also require the complete bound output proof: they need
    # not have a host load or a while-domain check to trigger readonly validation.
    if any(
        node.target is memory_ops.store
        for info in graphs
        if not isinstance(info, RootGraphInfo)
        for node in info.graph.nodes
    ):
        from .uniform_region_tree import uniform_local_regions
        from .uniform_region_tree import uniform_output_stores_are_writeonly

        if any(
            len(node.args) != 4 or node.kwargs
            for info in graphs
            for node in info.graph.nodes
            if node.target is memory_ops.store
        ):
            return False
        try:
            tree = uniform_local_regions(graphs)
        except exc.InvalidConfig:
            pass  # Preserve the independently proved legacy terminal path.
        else:
            if not uniform_output_stores_are_writeonly(
                env, graphs, tree, allow_unbound=allow_unbound
            ):
                return False
    graph_by_id = {info.graph_id: info for info in graphs}
    coordinates = reduction_coordinate_proof(
        env, graphs, physical_axes=physical_axes, owned_iota_axes=owned_iota_axes
    )
    local_allocations = local_atomic_allocations(graphs)
    if not local_allocations and any(
        node.target is atomic_ops.atomic_add
        for info in graphs
        for node in info.graph.nodes
    ):
        return False

    def configured_axis(size: int | torch.SymInt) -> bool:
        if isinstance(size, int):
            return size > 0
        expr = env.specialize_expr(cast("sympy.Expr", size._sympy_()))
        if expr.is_number:
            return int(expr) > 0
        return all(env.get_block_id(symbol) is not None for symbol in expr.free_symbols)

    grid_block_ids = {*env.config_spec.grid_block_ids}

    def resident_state(value: object) -> bool:
        # Per-row state such as an online-softmax max over ``hl.tile([B, L])``
        # has one element per grid coordinate, like a 1-D carry: the ordinary
        # lowering already owns each element. Only carries with a non-grid axis
        # are resident matrices that need independent coordinates.
        return (
            isinstance(value, torch.Tensor)
            and value.ndim > 1
            and not all(
                env.resolve_block_id(size) in grid_block_ids for size in value.shape
            )
        )

    carried: set[Node] = set()
    captures: dict[Node, Node] = {}
    parents = control_flow_parent_entries(graphs)
    for info in graphs:
        if info.graph_id not in parents:
            continue
        node, argument_slot = parents[info.graph_id]
        placeholders = list(info.graph.find_nodes(op="placeholder"))
        captures.update(
            zip(
                placeholders,
                cast("list[Node]", node.args[argument_slot]),
                strict=True,
            )
        )
        if _tracing_ops.is_for_loop_target(node.target):
            for slot in _loop_carry_slots(node).values():
                if resident_state(placeholders[slot].meta.get("val")):
                    carried.add(placeholders[slot])

    contractions = {
        node
        for info in graphs
        for node in info.graph.nodes
        if node.target is matmul_ops.dot
    }

    def reads_any(node: Node, sources: set[Node]) -> bool:
        pending = [node]
        seen: set[Node] = set()
        while pending:
            source = pending.pop()
            if source in seen:
                continue
            seen.add(source)
            if source in sources:
                return True
            if source.target in (memory_ops.load, _tracing_ops._host_tensor):
                continue
            if source in captures:
                pending.append(captures[source])
            else:
                pending.extend(source.all_input_nodes)
        return False

    # Only roots needing independent static coordinates or resident computed
    # contractions/scans use this fallback. Tiled contractions also need it
    # when a value repeats scalar coordinates; the ordinary matmul/stack/view
    # exemptions and native collective configuration checks still apply.
    def computed(node: Node) -> bool:
        if node.target in (
            torch.ops.aten.permute.default,
            torch.ops.aten.expand.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.dims,
            torch.ops.prims.convert_element_type.default,
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.select.int,
            view_ops.subscript,
        ):
            return isinstance(node.args[0], Node) and computed(node.args[0])
        return node.target not in (memory_ops.load, _tracing_ops._host_tensor)

    def repeated_axes(value: torch.Tensor) -> bool:
        sizes = [
            size._sympy_() if isinstance(size, torch.SymInt) else size
            for size in value.shape
        ]
        sizes = [size for size in sizes if size != 1]
        return len(sizes) != len(set(sizes))

    selections = [
        node
        for info in graphs
        for node in info.graph.nodes
        if is_ordered_selection(node)
    ]

    collapsed_nodes: set[Node] | None = None

    def needs_coordinates(node: Node) -> bool:
        nonlocal collapsed_nodes
        index = static_tensor_index(node)
        if index is not None and not shape_only_indices(*index):
            return True
        if is_ordered_selection(node):
            source_node = cast("Node", node.args[0])
            source = source_node.meta.get("val")
            if len(selections) > 1 or (
                isinstance(source, torch.Tensor) and source.ndim > 2
            ):
                return True
            if (
                not isinstance(source, torch.Tensor)
                or source.ndim != 2
                or not computed(source_node)
            ):
                return False
            from .selection.topk import match_topk_root

            # A computed row may lack the scalar sortable-load recipe while
            # still having complete static coordinates. Preserve the existing
            # optimized owner's structural match, including its later alias
            # and layout checks; only an unmatched root needs this fallback.
            return (
                match_topk_root(
                    graphs,
                    noncanonical_block_ids=HostFunction.current().device_ir.noncanonical_task_origin_block_ids,
                )
                is None
            )
        if node in local_allocations or node in coordinates:
            return True
        if node.target is _tracing_ops._host_tensor:
            return False
        if (
            node.op == "call_function"
            and scalar_repeated_block_id(env, node) is not None
        ):
            if collapsed_nodes is None:
                collapsed_nodes = matmul_sum_collapse_nodes(env, graphs)
            if node not in collapsed_nodes:
                return True
        if node.target in (
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
        ):
            source = node.args[0]
            assert isinstance(source, Node)
            before = cast("torch.Tensor", source.meta["val"])
            after = cast("torch.Tensor", node.meta["val"])
            # Factoring a full static axis requires independent coordinates for
            # every factor. A scalar recipe for its producer cannot represent
            # both the original flat consumer and the factored consumer.
            static_before = [
                size for size in before.shape if isinstance(size, int) and size != 1
            ]
            static_after = [
                size for size in after.shape if isinstance(size, int) and size != 1
            ]
            symbolic_before = [
                size._sympy_()
                for size in before.shape
                if isinstance(size, torch.SymInt)
            ]
            symbolic_after = [
                size._sympy_() for size in after.shape if isinstance(size, torch.SymInt)
            ]
            if (
                static_before != static_after
                and math.prod(static_before) == math.prod(static_after)
                and symbolic_before == symbolic_after
            ):
                return True
        if node.target is scan_ops._associative_scan:
            source = node.args[1]
            return isinstance(source, Node) and computed(source)
        if isinstance(node.meta.get("lowering"), ReductionLowering):
            if any(reads_any(source, carried) for source in node.all_input_nodes):
                return True
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and node.target in (
            torch.ops.aten.gt.Tensor,
            torch.ops.aten.ge.Tensor,
            torch.ops.aten.lt.Tensor,
            torch.ops.aten.le.Tensor,
            torch.ops.aten.eq.Tensor,
            torch.ops.aten.ne.Tensor,
        ):
            if repeated_axes(value):
                return True
        if node.target is not matmul_ops.dot:
            return False
        lhs, rhs = node.args[:2]
        assert isinstance(lhs, Node) and isinstance(rhs, Node)
        left, right = lhs.meta["val"], rhs.meta["val"]
        bid = env.resolve_block_id(left.shape[-1])
        static_k = isinstance(left.shape[-1], int) and left.shape[-1] > 1
        if bid is not None:
            static_k = env.block_sizes[bid].reduction
        # No tile or full slice walks K: a plain static extent, or the kept
        # extent of a range subscript that no full slice joined.
        free_k = bid is None or (
            bid in env.subscript_extent_block_ids
            and bid not in env.subscript_extents_joined_by_slice
        )
        return (
            left.dtype == right.dtype
            and left.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and static_k
            and (
                # A single scalar block coordinate cannot represent two axes
                # within the same operand. This also applies to captured full
                # matrix loads, not just matrices created by comparisons.
                repeated_axes(left)
                or repeated_axes(right)
                # Resident state cannot be reconstructed from global loads for
                # each contraction coordinate, including after a dtype cast.
                or reads_any(lhs, carried)
                or reads_any(rhs, carried)
                # A computed contraction operand needs the completed earlier
                # dot at every K coordinate, including through half casts.
                # Scalar load recipes cannot reconstruct that producer DAG.
                or reads_any(lhs, contractions)
                or reads_any(rhs, contractions)
                # Existing scalar/native half-precision recipes retain their
                # paths when their operands have independent coordinates.
                or (
                    left.dtype == torch.float32
                    and (
                        computed(lhs)
                        or computed(rhs)
                        or left.ndim > 2
                        or right.ndim > 2
                    )
                )
                # A computed operand along such a K has no scalar recipe
                # unless the pair is K-invariant: the fallback needs an
                # active K tile or plain load views.  The extent a range
                # subscript padded to could repeat another axis here, which
                # gave the dot this owner only at such sizes.
                or (
                    free_k
                    and (computed(lhs) or computed(rhs))
                    and cute_static_k_invariant_extent(lhs, rhs) is None
                )
            )
        )

    if not (
        snapshot_owned or bounded_gather_owned or requires_full_reduction_owner(graphs)
    ) and not any(
        needs_coordinates(node) for info in graphs for node in info.graph.nodes
    ):
        return False
    supported = {
        _tracing_ops._if,
        _tracing_ops._while_loop,
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._new_var,
        _tracing_ops._phi,
        _tracing_ops._constant_tensor,
        _tracing_ops._mask_to,
        _tracing_ops._for_loop,
        _tracing_ops._for_loop_step,
        memory_ops.load,
        memory_ops.store,
        atomic_ops.atomic_add,
        matmul_ops.dot,
        scan_ops._associative_scan,
        inline_asm_ops.inline_asm_elementwise,
        creation_ops.full,
        torch.ops.aten.full.default,
        view_ops.subscript,
        tile_ops.tile_begin,
        tile_ops.tile_end,
        tile_ops.tile_id,
        tile_ops.tile_index,
        operator.getitem,
        torch.ops.aten.sym_size.int,
        torch.ops.prims.iota.default,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.where.self,
        torch.ops.aten.view.dtype,
        torch.ops.aten.alias.default,
        torch.ops.aten.topk.default,
        torch.ops.aten.sort.default,
        torch.ops.aten.sort.stable,
        torch.ops.aten.gather.default,
        torch.ops.aten.fmin.default,
        torch.ops.aten.fmax.default,
    }
    if any(
        node.op == "call_function"
        and node.target not in supported
        and static_tensor_index(node) is None
        and not isinstance(
            node.meta.get("lowering"),
            (PointwiseLowering, SympyExprLowering, ReductionLowering),
        )
        for info in graphs
        for node in info.graph.nodes
    ):
        return False
    for info in graphs:
        for node in info.graph.nodes:
            if is_ordered_selection(node):
                arguments = ordered_selection_args(node)
                if arguments is None:
                    return False
                source_node, k, dim, *_ = arguments
                source = source_node.meta.get("val")
                width = _static_selection_extent(env, source_node)
                if k is None:
                    k = width
                if (
                    not isinstance(source, torch.Tensor)
                    or source.ndim < 1
                    or source.dtype
                    not in (
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                        torch.int32,
                        torch.int64,
                    )
                    or dim not in (-1, source.ndim - 1)
                    or width is None
                    or not isinstance(k, int)
                    or not 0 < k <= width
                    or not all(configured_axis(size) for size in source.shape)
                ):
                    return False
            if node.target is torch.ops.aten.gather.default:
                from .bounded_gather import prove_gather

                if (
                    bounded_gather_owned
                    and prove_gather(
                        env, node, graphs=graphs, allow_unbound=allow_unbound
                    )
                    is not None
                ):
                    continue
                source, dim, index = node.args[:3]
                if not isinstance(source, Node) or not isinstance(index, Node):
                    return False
                sf, ix = source.meta.get("val"), index.meta.get("val")
                if (
                    not isinstance(sf, torch.Tensor)
                    or not isinstance(ix, torch.Tensor)
                    or sf.ndim != ix.ndim
                    or dim not in (-1, sf.ndim - 1)
                    or index.target is not operator.getitem
                    or index.args[1] != 1
                    or not isinstance(index.args[0], Node)
                    or not is_ordered_selection(index.args[0])
                ):
                    return False
                original_node = cast("Node", index.args[0].args[0])
                original = original_node.meta["val"]
                original_shape = _fragment_logical_shape(env, original_node)
                gathered_shape = _fragment_logical_shape(env, source)
                if (
                    original_shape is None
                    or gathered_shape is None
                    or tuple(map(sympy.expand, original_shape))
                    != tuple(map(sympy.expand, gathered_shape))
                ):
                    return False
                if (
                    original.ndim != sf.ndim
                    or any(
                        not env.known_equal(a, b)
                        for a, b in zip(original.shape, sf.shape, strict=True)
                    )
                    or any(
                        not env.known_equal(a, b)
                        for a, b in zip(ix.shape[:-1], sf.shape[:-1], strict=True)
                    )
                ):
                    return False
            if node.target is torch.ops.aten.view.dtype:
                source = cast("Node", node.args[0]).meta["val"]
                target = node.meta["val"]
                if (
                    not isinstance(source, torch.Tensor)
                    or not isinstance(target, torch.Tensor)
                    or source.dtype == torch.bool
                    or target.dtype == torch.bool
                    or source.dtype.itemsize != target.dtype.itemsize
                ):
                    return False
            if node.target is inline_asm_ops.inline_asm_elementwise:
                if (
                    node.args[4] is not True
                    or node.args[5] != 1
                    or not isinstance(node.args[3], torch.dtype)
                    or not isinstance(node.meta.get("val"), torch.Tensor)
                ):
                    return False
            if node.target is scan_ops._associative_scan:
                if node.args[4]:
                    return False
                value = node.meta["val"]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype
                    not in (
                        torch.int8,
                        torch.uint8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                        torch.float64,
                    )
                    or value.ndim == 0
                ):
                    return False
                if not -value.ndim <= cast("int", node.args[2]) < value.ndim:
                    return False
                # Every local axis must be constant or have a configured block
                # owner. Config-dependent values and shared capacity are checked
                # again by the emitter, never replaced by a shape-specific mode.
                if not all(configured_axis(size) for size in value.shape):
                    return False
                helper = graph_by_id[cast("int", node.args[0])]
                assert isinstance(helper, HelperFunctionGraphInfo)
                if not additive_scan_helper(helper):
                    return False
            lowering = node.meta.get("lowering")
            if isinstance(
                lowering, ReductionLowering
            ) and lowering.reduction_type not in (
                "sum",
                "max",
                "min",
                "prod",
                "any",
                "argmin",
                "argmax",
            ):
                return False
            if _tracing_ops.is_for_loop_target(node.target) and any(
                not isinstance(value, torch.Tensor) for value in node.meta["val"]
            ):
                return False
    return True


def computed_fragment_specialization_facts(
    env: CompileEnvironment, ir: DeviceIR
) -> frozenset[CompilerHeuristicSpecializationFact]:
    """Guard shape hints used as capacities before bound-cache publication."""
    if env.settings.static_shapes or ir.host_function is None:
        return frozenset()
    with ir.host_function:
        if computed_fragment_supported(
            env, ir.graphs
        ) or independent_reduction_coordinates(
            env, ir.graphs, include_rolled_outputs=True
        ):
            return frozenset({"input_tensor_metadata"})
    return frozenset()


def prefer_computed_fragment_root(cg: GenerateAST, root_graph_id: int) -> bool:
    """Try fragment ownership before other schedules when this root needs it
    (consulted by the CuTe root-emitter registry only)."""
    env = CompileEnvironment.current()
    if cg.device_function.cute_state.root_plans.register_subblocks_plan is not None:
        # The register sub-block family replaces the root.
        return False
    if cg.device_function.cute_state.root_plans.mma_dag_plan is not None:
        # The warp-MMA recurrence owner replaces the root (its captured
        # reductions are the dot's K extent).
        return False
    graphs = cg.host_function.device_ir.graphs
    captured = captured_reduction_coordinates(env, graphs, root_graph_id=root_graph_id)
    owned_iota_axes = owned_iota_reduction_axes(cg)
    free = any(
        node.graph is graphs[root_graph_id].graph
        and node.meta["lowering"].block_index not in owned_iota_axes
        for node in free_iota_reductions(env, graphs)
    )
    config = cg.device_function.config
    spec = env.config_spec
    return (
        bool(captured)
        or free
        or any(
            enabled and root_graph_id in roots
            for enabled, roots in (
                (
                    config.get("cute_fragment_producer_cache", False),
                    spec.cute_fragment_producer_cache_root_ids,
                ),
                (
                    config.get("cute_fragment_warp_scan", False),
                    spec.cute_fragment_warp_scan_root_ids,
                ),
                (
                    config.get("cute_fragment_register_loads", False),
                    spec.cute_fragment_register_load_root_ids,
                ),
                (
                    config.get("cute_fragment_warp_results", False),
                    spec.cute_fragment_warp_result_root_ids,
                ),
                (
                    config.get("cute_fragment_threads", 128) != 128,
                    spec.cute_fragment_thread_root_ids,
                ),
                (
                    config.get("cute_fragment_scan", "serial") == "cooperative",
                    spec.cute_fragment_scan_root_ids,
                ),
                (
                    config.get("cute_fragment_reduction", "serial") == "warp",
                    spec.cute_fragment_reduction_root_ids,
                ),
            )
        )
    )


def codegen_computed_fragment_root(cg: GenerateAST) -> bool:
    """Own static axes and complete contractions that scalar indexing cannot express."""
    from .autotune.heuristics.cute_fragment_reduction import (
        fragment_warp_reduction_supported,
    )
    from .autotune.heuristics.cute_fragment_warp_scan import active_warp_scan_roots

    env = CompileEnvironment.current()
    if env.backend_name != "cute":
        return False
    root = cg.current_root_graph_info
    assert root is not None
    threads_required = (
        cg.device_function.config.get("cute_fragment_threads", 128) != 128
        and root.graph_id in env.config_spec.cute_fragment_thread_root_ids
    )
    published_scalars_required = (
        cg.device_function.config.get("cute_fragment_published_scalars", False)
        and root.graph_id in env.config_spec.cute_fragment_published_scalar_root_ids
    )
    pure_regions_required = (
        cg.device_function.config.get("cute_fragment_pure_producer_regions", False)
        and root.graph_id
        in env.config_spec.cute_fragment_pure_producer_regions_root_ids
    )
    producer_cache_required = (
        cg.device_function.config.get("cute_fragment_producer_cache", False)
        and root.graph_id in env.config_spec.cute_fragment_producer_cache_root_ids
    )
    warp_scan_required = cg.device_function.config.get(
        "cute_fragment_warp_scan", False
    ) and root.graph_id in active_warp_scan_roots(env, cg.device_function.config)
    snapshots_required = (
        cg.device_function.config.get("cute_fragment_register_snapshots", False) is True
        and root.graph_id in env.config_spec.cute_fragment_register_snapshots_root_ids
    )
    topk_network_required = (
        cg.device_function.config.get("cute_fragment_topk_network", False)
        and root.graph_id in env.config_spec.cute_fragment_topk_network_root_ids
    )
    bounded_gather_required = (
        cg.device_function.config.get("cute_fragment_bounded_gather", False) is True
        and root.graph_id in env.config_spec.cute_fragment_bounded_gather_root_ids
    )
    register_loads_required = (
        cg.device_function.config.get("cute_fragment_register_loads", False)
        and root.graph_id in env.config_spec.cute_fragment_register_load_root_ids
    )
    warp_results_required = (
        cg.device_function.config.get("cute_fragment_warp_results", False)
        and root.graph_id in env.config_spec.cute_fragment_warp_result_root_ids
    )
    # These options require the fragment owner's coordinate layout instead of
    # borrowing the ordinary scalar/iota layout. Warp-result publication also
    # owns captured reductions, but does not change free-iota ownership.
    coordinate_schedule_required = (
        threads_required
        or register_loads_required
        or producer_cache_required
        or pure_regions_required
        or published_scalars_required
        or topk_network_required
        or bounded_gather_required
        or snapshots_required
        or warp_scan_required
    )
    capture_schedule_required = coordinate_schedule_required or warp_results_required
    local_required = bool(local_atomic_allocations(cg.host_function.device_ir.graphs))
    owned_iota_axes = (
        frozenset()
        if coordinate_schedule_required
        or cg.device_function.config.get("cute_fragment_reduction", "serial")
        != "serial"
        else owned_iota_reduction_axes(cg)
    )
    free_required = any(
        node.graph is cg.host_function.device_ir.graphs[root.graph_id].graph
        and node.meta["lowering"].block_index not in owned_iota_axes
        for node in free_iota_reductions(env, cg.host_function.device_ir.graphs)
    )
    physical_axes = (
        frozenset()
        if capture_schedule_required
        or cg.device_function.config.get("cute_fragment_scan", "serial") != "serial"
        or cg.device_function.config.get("cute_fragment_reduction", "serial")
        != "serial"
        else physical_capture_axes(cg)
    )
    captured_required = bool(
        captured_reduction_coordinates(
            env,
            cg.host_function.device_ir.graphs,
            root_graph_id=root.graph_id,
            physical_axes=physical_axes,
        )
    )
    fragment_schedule_required = (
        (
            cg.device_function.config.get("cute_fragment_private_scalar_loops", False)
            and root.graph_id
            in env.config_spec.cute_fragment_private_scalar_loop_root_ids
        )
        or capture_schedule_required
        or (
            cg.device_function.config.get("cute_fragment_scan", "serial")
            == "cooperative"
            and root.graph_id in env.config_spec.cute_fragment_scan_root_ids
        )
        or (
            cg.device_function.config.get("cute_fragment_reduction", "serial")
            in ("warp", "cta", "cta_unroll")
            and root.graph_id in env.config_spec.cute_fragment_reduction_root_ids
        )
    )
    state = cg.device_function.cute_state
    full_reduction_required = requires_full_reduction_owner(
        cg.host_function.device_ir.graphs
    )
    if any(
        plan is not None
        for plan in (
            state.root_plans.block_scaled_plan,
            state.root_plans.topk_plan,
            state.root_plans.radix_selection_plan,
            state.root_plans.chunk_prepare_plan,
            state.root_plans.chunk_recurrence_plan,
            state.root_plans.single_token_rank1_plan,
            state.root_plans.split_single_token_rank1_plan,
            state.root_plans.fixed_token_rank1_plan,
            state.root_plans.mma_dag_plan,
            state.flash.block_ids,
            state.flash.bwd_block_ids,
        )
    ):
        # Pre-codegen selected a complete root owner. It must execute its
        # late layout/alias checks; an implicit coordinate requirement must
        # not bypass those checks by replacing its producer with fragments.
        if local_required or fragment_schedule_required:
            raise exc.InvalidConfig(
                "a planned root cannot share a computed fragment schedule"
            )
        return False
    if free_required or full_reduction_required:
        # A new complete owner changes the placement of global stores relative
        # to vector reductions. Separate destinations need a disjointness
        # proof; declining into the unowned scalar iota path would lose the
        # complete reduction. Repeated stores through one tensor retain their
        # existing ordered semantics.
        destinations = {
            cg.device_function.tensor_arg(tensor).name
            for tensor in fragment_store_destinations(cg.host_function.device_ir.graphs)
        }
        disjoint = cg.device_function.proven_disjoint_tensor_pairs()
        if any(
            frozenset((left, right)) not in disjoint
            for left in destinations
            for right in destinations
            if left < right
        ):
            if local_required or fragment_schedule_required or full_reduction_required:
                raise exc.InvalidConfig(
                    "fragment store destinations require disjoint storage"
                )
            raise exc.BackendUnsupported(
                "cute", "fragment store destinations require disjoint storage"
            )
    if (
        cg.device_function.config.get(
            "cute_affine_scan_schedule", DIRECT_AFFINE_ORDINARY_SCHEDULE
        )
        != DIRECT_AFFINE_ORDINARY_SCHEDULE
    ):
        # The direct affine schedule must first emit its ordinary producer,
        # then prove and replace that entire region. Implicit free-iota and
        # capture ownership must not erase the producer before that proof.
        if (
            local_required
            or fragment_schedule_required
            or full_reduction_required
            or cg.device_function.config.get("cute_reduction_schedule", "scalar")
            != "scalar"
        ):
            raise exc.InvalidConfig(
                "direct affine schedules cannot share a computed fragment root"
            )
        return False
    if cg.device_function.config.get("cute_reduction_schedule", "scalar") != "scalar":
        # An explicit resident schedule owns its rectangular carries and
        # performs its own complete proof during materialization. Replacing
        # its scalar producer here erases the structure that proof consumes.
        if (
            local_required
            or free_required
            or fragment_schedule_required
            or full_reduction_required
        ):
            raise exc.InvalidConfig(
                "resident reduction schedules cannot share a computed fragment root"
            )
        return False
    if (
        len(cg.host_function.device_ir.root_ids) != 1
        or cg.device_function.config.get("cute_collective_mma", False)
        or cg.device_function.config.get("cute_register_chain", False)
    ):
        if full_reduction_required:
            raise exc.InvalidConfig(
                "unmerged full reductions require an unconflicted complete fragment root"
            )
        if local_required:
            raise exc.InvalidConfig(
                "CTA-local atomics require a complete fragment root"
            )
        if free_required:
            raise exc.InvalidConfig(
                "free iota reductions require a complete fragment root"
            )
        if captured_required or capture_schedule_required:
            raise exc.InvalidConfig(
                "captured full reductions require a computed fragment root"
                if captured_required
                else "CTA threads/register loads require a computed fragment root"
            )
        return False
    graphs = (
        cg.host_function.device_ir.build_codegen_graphs(
            cg.device_function.config, roll_reductions=False
        )
        if captured_required
        or local_required
        or free_required
        or snapshots_required
        or bounded_gather_required
        else cg.codegen_graphs
    )

    scan_required = warp_scan_required or (
        cg.device_function.config.get("cute_fragment_scan", "serial") == "cooperative"
        and root.graph_id
        in CompileEnvironment.current().config_spec.cute_fragment_scan_root_ids
    )
    reduction_required = (
        cg.device_function.config.get("cute_fragment_reduction", "serial")
        in ("warp", "cta", "cta_unroll")
        and root.graph_id
        in CompileEnvironment.current().config_spec.cute_fragment_reduction_root_ids
    )

    def decline() -> bool:
        if full_reduction_required:
            raise exc.InvalidConfig(
                "unmerged full reductions require a supported complete fragment root"
            )
        if free_required:
            raise exc.InvalidConfig(
                "free iota reductions require a supported complete fragment root"
            )
        if local_required:
            raise exc.InvalidConfig(
                "CTA-local atomics require a supported complete fragment root"
            )
        if (
            captured_required
            or scan_required
            or reduction_required
            or capture_schedule_required
        ):
            raise exc.InvalidConfig(
                "captured full reductions require a computed fragment root"
                if captured_required
                else "cooperative scan requires a computed fragment root"
                if scan_required
                else "warp reduction requires a computed fragment root"
                if reduction_required
                else "CTA threads/register loads require a computed fragment root"
            )
        return False

    if scan_required and not any(
        node.target is scan_ops._associative_scan
        for info in graphs
        for node in info.graph.nodes
    ):
        return decline()
    if reduction_required and not any(
        fragment_warp_reduction_supported(node)
        for info in graphs
        for node in info.graph.nodes
    ):
        return decline()
    if not computed_fragment_supported(
        CompileEnvironment.current(),
        graphs,
        physical_axes=physical_axes,
        owned_iota_axes=owned_iota_axes,
        snapshot_owned=snapshots_required,
        bounded_gather_owned=bounded_gather_required,
    ):
        return decline()
    # This owner implements configured reduction tiling directly. Keep the
    # logical producer graph, including every other codegen transformation,
    # rather than applying scalar graph rolling before fragment ownership.
    if (
        not captured_required
        and not local_required
        and not free_required
        and not snapshots_required
        and not bounded_gather_required
    ):
        graphs = cg.host_function.device_ir.build_codegen_graphs(
            cg.device_function.config, roll_reductions=False
        )
    if not computed_fragment_supported(
        CompileEnvironment.current(),
        graphs,
        physical_axes=physical_axes,
        owned_iota_axes=owned_iota_axes,
        snapshot_owned=snapshots_required,
        bounded_gather_owned=bounded_gather_required,
    ):
        return decline()
    root = graphs[root.graph_id]
    from .conditional_attention_codegen import conditional_attention_compiler

    compiler = (
        conditional_attention_compiler(cg, graphs)
        if not fragment_schedule_required
        else None
    ) or FragmentCompiler(cg, graphs)
    if compiler.df.config.get("cute_fragment_warp_producer_regions", False):
        from .warp_producer_regions import WarpProducerRecorder
        from .warp_producer_regions import producer_prefix

        plan = producer_prefix(root.graph, env, graphs, compiler.shape)
        if plan is not None:
            compiler.warp_producer_recorder = WarpProducerRecorder(compiler, plan)
    if (
        pure_regions_required
        and not pure_producer_plan(root.graph, env, shape=compiler.shape).lazy
    ):
        raise exc.InvalidConfig(
            "pure producer regions require a current same-owner publication"
        )
    compiler.local_allocations = prove_local_atomics(graphs)
    for info in graphs:
        for node in info.graph.nodes:
            if node.target is _tracing_ops._if:
                predicate = node.args[0]
                value = (
                    predicate.meta["val"] if isinstance(predicate, Node) else predicate
                )
                if (
                    isinstance(value, torch.Tensor)
                    and math.prod(compiler.shape(value.shape)) != 1
                ):
                    return decline()
    for info in graphs:
        for node in info.graph.nodes:
            lowering = node.meta.get("lowering")
            if not isinstance(lowering, (PointwiseLowering, ReductionLowering)):
                continue
            ranges = lowering.buffer.data.ranges
            if isinstance(lowering, ReductionLowering):
                assert isinstance(lowering.buffer.data, Reduction)
                ranges = [*ranges, *lowering.buffer.data.reduction_ranges]
            if any(compiler.static_extent(dim) is None for dim in ranges):
                return decline()
            # Captured placeholders do not invoke their producer's inner function.
            if node.op != "call_function":
                continue
            inputs = lowering.input_nodes(node)
            input_layouts = {
                source: lowering.input_layouts[name]
                for name, source in zip(lowering.input_names, inputs, strict=True)
                if name in lowering.input_layouts
            }
            for source in node.all_input_nodes:
                value = source.meta.get("val")
                if not isinstance(value, torch.Tensor):
                    if isinstance(
                        value,
                        (torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool),
                    ):
                        continue
                    return decline()
                shape = tuple(compiler.static_extent(dim) for dim in value.shape)
                layout = input_layouts.get(source)
                index_strides = layout[1] if layout is not None else value.stride()
                strides = tuple(compiler.static_extent(dim) for dim in index_strides)
                if len(shape) != len(strides) or any(
                    dim is None for dim in (*shape, *strides)
                ):
                    # Saved strides decode the inner function's index, but loads
                    # address the current fragment shape. Their ranks must agree.
                    # This owner also requires static capacities and strides.
                    return decline()
                span = 1
                for stride, size in sorted(
                    zip(
                        cast("tuple[int, ...]", strides),
                        cast("tuple[int, ...]", shape),
                        strict=True,
                    )
                ):
                    if size <= 1 or stride == 0:
                        continue
                    if stride < span:
                        return decline()
                    span += (size - 1) * stride
    grid = cg.current_grid_state
    assert grid is not None
    if local_required:
        if any(
            cg.device_function.resolved_block_size(bid) != 1
            for bid in grid.block_id_to_info
        ):
            raise exc.InvalidConfig("CTA-local atomics require scalar grid owners")
    compiler.begin()
    for bid, info in grid.block_id_to_info.items():
        compiler.offsets[bid] = grid.strategy.grid_origin_var(bid)
        compiler.bounds[bid] = cast("str", info.end_var_name)
    body: list[ast.AST] = []
    with cg.set_statements(body):
        compiler.emit(f"{compiler.thread} = cutlass.Int32(cute.arch.thread_idx()[0])")
        compiler.stage_slabs()
        compiler.graph(root.graph, {})
        compiler.flush_pending_barrier()
        # Root lifetime can be nested in an outer generated grid loop.
        compiler.synchronize_local_atomics()
    if compiler.warp_producer_recorder is not None:
        compiler.warp_producer_recorder.lower(body)
    capacity = target_facts.per_cta_smem_capacity_bytes(compiler.env.device)
    required_owner = (
        local_required
        or free_required
        or fragment_schedule_required
        or full_reduction_required
        or any(
            node.target is _tracing_ops._while_loop
            for info in graphs
            for node in info.graph.nodes
        )
    )
    if (
        required_owner
        and cg.statements_stack[-1] is compiler.df.body
        and not str(compiler.df.config.get("pid_type", "flat")).startswith("persistent")
        and not compiler.df.config.get("cute_pdl", False)
        and not compiler.env.config_spec.has_explicit_pdl
        and capacity
        and compiler.smem_bytes > capacity
        and not compiler.local_allocations
        and compiler.warp_producer_recorder is None
        and not compiler.wants_mma_preamble
        and not published_scalars_required
        and not compiler.df.config.get("cute_fragment_register_producers", False)
    ):
        # Only a required owner may replace an over-capacity layout with an
        # arena. Optional owners keep their ordinary scalar fallback. Planning
        # covers this top-level root only: nested/persistent dispatch and PDL
        # are excluded before committing storage, not by late validation.
        arena = plan_shared_arena(
            compiler.buffers, compiler.buffer_interference, cast("list[ast.stmt]", body)
        )
        if arena is not None and arena.byte_size <= capacity:
            compiler.shared_arena_plan = arena
            compiler.shared_arena_name = compiler.df.new_var("fragment_arena")
            compiler.smem_bytes = arena.byte_size
    if not capacity and full_reduction_required:
        raise exc.InvalidConfig(
            "unmerged full reductions require a known shared memory capacity"
        )
    if not capacity and any(
        node.target is _tracing_ops._while_loop
        for info in graphs
        for node in info.graph.nodes
    ):
        raise exc.InvalidConfig(
            "resident while requires a known shared memory capacity"
        )
    if local_required and not capacity:
        raise exc.InvalidConfig(
            "CTA-local atomics require a known shared memory capacity"
        )
    if producer_cache_required and not capacity:
        raise exc.InvalidConfig(
            "shared producer caches require a known shared memory capacity"
        )
    if capacity and compiler.smem_bytes > capacity:
        if required_owner:
            # These roots have no other owner: the config is invalid.
            raise exc.InvalidConfig(
                f"computed fragments need {compiler.smem_bytes} shared bytes, exceeding {capacity}"
            )
        # The fragment program materializes every row of this root in shared
        # memory, so a wide static axis can exceed the CTA capacity for every
        # tile size (layer_norm backward at N=8192 needs 245 KB with one row).
        # The implicit preference for this owner (captured reductions) is a
        # performance choice; the ordinary scalar lowering owns such roots
        # with lane/thread accumulation, so decline instead of failing the
        # default and every other scalar-schedule config.  Nothing emitted
        # above escaped the scratch ``body``: the compiler only allocated
        # variable names and kernel tensor arguments.  A root only this owner
        # can lower (repeated block ids, static matrices) makes the generic
        # lowering refuse; GenerateAST reports that as this config failure.
        cg.device_function.cute_state.root_plans.fragment_capacity_decline = (
            compiler.smem_bytes,
            capacity,
        )
        return False
    if published_scalars_required:
        from .published_scalars import PublishedScalarRequest

        compiler.df.cute_state.published_scalar_requests.append(
            PublishedScalarRequest(
                frozenset(name for name, _, _ in compiler.buffers),
                compiler.thread,
                compiler.threads,
            )
        )
    if compiler.df.config.get("cute_fragment_register_producers", False):
        from .register_producers import RegisterProducerRequest

        compiler.df.cute_state.register_producer_requests.append(
            RegisterProducerRequest(
                frozenset(name for name, _, _ in compiler.buffers),
                tuple(compiler.register_snapshot_buffers.items()),
                compiler.thread,
                compiler.threads,
            )
        )
    cg.device_function.cute_state.lanes.owned_root_block_dims = (compiler.threads, 1, 1)
    compiler.emit_mma_preamble()
    allocations = compiler.allocations()
    if compiler.shared_arena_plan is not None:
        compiler.df.cute_state.shared_arena_requests.append(
            SharedArenaRequest(
                compiler.shared_arena_plan,
                compiler.shared_arena_name,
                ast.unparse(
                    ast.Module(
                        body=cast("list[ast.stmt]", allocations), type_ignores=[]
                    )
                ),
            )
        )
    for statement in (*allocations, *body):
        cg.add_statement(statement)
    return True
