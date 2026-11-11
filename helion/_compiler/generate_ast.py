from __future__ import annotations

import ast
import collections
import contextlib
import dataclasses
import logging
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import sympy
import torch
from torch.utils._device import _device_constructors
from torch.utils._ordered_set import OrderedSet

from .. import exc
from ..language._decorators import is_api_func
from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .ast_extension import NodeVisitor
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import dead_assignment_elimination
from .ast_read_writes import dead_expression_elimination
from .ast_read_writes import definitely_does_not_have_side_effects
from .compile_environment import CompileEnvironment
from .device_function import DeviceFunction
from .emitter_protocols import KernelTransforms
from .helper_function import CodegenInterface
from .inductor_lowering import CodegenState
from .inductor_lowering import codegen_call_with_graph
from .output_header import get_needed_import_lines
from .program_id import ForEachProgramID
from .tile_strategy import DeviceGridState
from .tile_strategy import DeviceLoopState
from .tile_strategy import EmitPipelineLoopState
from .tile_strategy import ForiLoopState
from .variable_origin import ArgumentOrigin

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator

    from torch.fx.node import Node

    from ..runtime.config import Config
    from .cute.bounded_cache_codegen import BoundedCacheRequest
    from .device_ir import GraphInfo
    from .host_function import HostFunction
    from .pallas.compact_worklist import ResidentPrepHoist
    from .pallas.dma import DmaResources
    from .tile_strategy import DeviceLoopOrGridState
    from .type_info import TensorType

log = logging.getLogger(__name__)


def _flatten_starred_args(args: list[ast.expr]) -> list[ast.expr]:
    """Expand ``f(*xs)`` into per-element nodes (``xs[0]``, ``xs[1]``, ...).

    Type propagation flattens starred args the same way (see
    `TypePropagation.visit_Call`), so host codegen must match it to keep the ast
    args aligned with the proxy args of an API function.
    """
    from .type_info import SequenceType

    if not any(isinstance(arg, ast.Starred) for arg in args):
        return args
    result: list[ast.expr] = []
    for arg in args:
        if not isinstance(arg, ast.Starred):
            result.append(arg)
            continue
        value = arg.value
        assert isinstance(value, ExtendedAST)
        seq_type = value._type_info
        assert isinstance(seq_type, SequenceType)
        for i, element_type in enumerate(seq_type.unpack()):
            result.append(
                create(
                    ast.Subscript,
                    value=value,
                    slice=create(ast.Constant, value=i, kind=None),
                    ctx=ast.Load(),
                    _type_info=element_type,
                )
            )
    return result


_PURE_SCALAR_CALL_MODULES = frozenset({"cutlass", "operator"})


def _is_pure_scalar_expr(expr: ast.expr) -> bool:
    """Whether ``expr`` only combines names and constants: arithmetic,
    comparisons, ``operator`` calls and ``cutlass`` scalar casts."""
    for node in ast.walk(expr):
        if isinstance(node, ast.Call):
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in _PURE_SCALAR_CALL_MODULES
                and not node.keywords
            ):
                return False
        elif not isinstance(
            node,
            (
                ast.Name,
                ast.Constant,
                ast.Attribute,
                ast.BinOp,
                ast.UnaryOp,
                ast.BoolOp,
                ast.Compare,
                ast.IfExp,
                ast.Tuple,
                ast.expr_context,
                ast.operator,
                ast.unaryop,
                ast.boolop,
                ast.cmpop,
            ),
        ):
            return False
    return True


def _drop_dead_lane_derived_assignments(
    body: list[ast.AST], preamble: list[ast.AST], lane_names: set[str]
) -> None:
    """Delete the dead assignments computed from lane-loop definitions.

    Once a collective matmul has suppressed the root lane loops, the generic
    lowering of what it took over (the masks of its operand loads and of the
    store its epilogue performs, the K loop's copies of them) still reads the
    per-element coordinates, which nothing defines any more: executed, those
    statements fail at launch.  Only pure scalar assignments that no
    statement reads are deleted (each becomes ``pass``).
    """
    from .ast_read_writes import ReadWrites

    def target(node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _is_pure_scalar_expr(node.value)
        ):
            return node.targets[0].id
        return None

    module = ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
    derived = set(lane_names)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(module):
            name = target(node)
            if (
                name is not None
                and name not in derived
                and set(ReadWrites.from_ast(cast("ast.Assign", node).value).reads)
                & derived
            ):
                derived.add(name)
                changed = True
    removable = derived - lane_names

    class DropDead(ast.NodeTransformer):
        def __init__(self, reads: set[str]) -> None:
            self.reads = reads
            self.changed = False

        def visit_Assign(self, node: ast.Assign) -> ast.AST:
            name = target(node)
            if name in removable and name not in self.reads:
                self.changed = True
                return ast.copy_location(ast.Pass(), node)
            return node

    while True:
        drop = DropDead(set(ReadWrites.from_list([*preamble, *body]).reads))
        drop.visit(module)
        if not drop.changed:
            break


@dataclasses.dataclass(frozen=True)
class ResidentPrepLowering:
    hoist: ResidentPrepHoist
    resident_window_name: str
    cache_name: str
    # Fill written to the padded tail and used to identify redundant masks.
    tail_fill_value: float


class GenerateAST(NodeVisitor, CodegenInterface):
    def __init__(
        self,
        func: HostFunction,
        config: Config,
        *,
        store_transform: Callable[..., ast.AST] | None = None,
        load_transform: Callable[..., ast.AST] | None = None,
        extra_params: list[str] | None = None,
        codegen_graphs: list[GraphInfo] | None = None,
    ) -> None:
        # Initialize NodeVisitor first
        NodeVisitor.__init__(self)

        # Must be set before DeviceFunction is created so device_function.codegen._extra_params is available immediately.
        self._extra_params: list[str] = extra_params or []

        assert not (
            collisions := {a.arg for a in func.args.args} & set(self._extra_params)
        ), f"extra_params names collide with existing function args: {collisions}"

        # Initialize our attributes
        self.host_function = func
        backend = CompileEnvironment.current().backend
        self.backend = backend
        # The backend's per-pass state (``Backend.new_codegen_state``); the
        # CuTe backend reads it through ``cute.state.cute_codegen_state``.
        self.backend_state: object | None = backend.new_codegen_state()
        config = backend.codegen_config(config)
        self.codegen_graphs = (
            func.device_ir.build_codegen_graphs(config)
            if codegen_graphs is None
            else codegen_graphs
        )
        self.host_statements: list[ast.AST] = []
        self.module_statements: list[ast.stmt] = []
        self._ensured_module_sources: set[str] = set()
        self.statements_stack: list[list[ast.AST]] = [self.host_statements]
        self.on_device = False
        self.active_device_loops: dict[int, list[DeviceLoopOrGridState]] = (
            collections.defaultdict(list)
        )
        self.current_grid_state: DeviceGridState | None = None
        # The names only the lane loops of roots emitted without them define
        # (a collective matmul suppressed them): dead reads are dropped
        # (``_drop_dead_lane_derived_assignments``) and live ones refused once
        # codegen is done (``Backend.after_device_codegen``).
        self.suppressed_lane_names: set[str] = set()
        self.divergent_control_flow_depth = 0
        self.current_root_graph_info: GraphInfo | None = None
        self.max_thread_block_dims = [1, 1, 1]
        self.root_thread_block_dims = [1, 1, 1]
        self.referenced_thread_block_dims = [1, 1, 1]
        self.store_transform = store_transform
        self.load_transform = load_transform
        # Per-pass state of ``load_transform``: the fused inputs whose prologue
        # placeholder this pass has emitted, mapped to their first dim-index
        # expressions (``HelionTemplateBuffer._codegen_prologue_fusion``).  It
        # lives on the pass rather than on the transform so the register-tile
        # retry in ``generate_ast`` emits every placeholder again.
        self.prologue_first_indexing: dict[str, str] = {}
        self._statement_owner_fx_node: Node | None = None
        self._codegen_results_by_owner_node_id: dict[int, object] = {}
        self.resident_prep_lowering_stack: list[
            dict[tuple[int, str], ResidentPrepLowering]
        ] = []
        # Grouping-2 worklists codegen the compact body twice, once per static
        # compact block size. Shape-independent ordered-loop resources are shared
        # across those mutually exclusive bodies.
        self.grouped_compact_common_statements: list[ast.AST] | None = None
        self.grouped_resident_prep_lowering_cache: dict[
            tuple[object, ...], list[ResidentPrepLowering]
        ] = {}
        self.grouped_resident_prep_refill_cache: dict[tuple[object, ...], str] = {}
        self.grouped_fori_dma_resource_cache: dict[
            tuple[object, ...], DmaResources
        ] = {}
        self._statements_by_owner_node_id: dict[
            int, list[tuple[list[ast.AST], ast.AST]]
        ] = {}
        self._track_statement_owners = backend.tracks_statement_owners()

        # Now create device function and initialize CodegenInterface
        self.device_function = DeviceFunction(
            f"_helion_{func.name}",
            config,
            self,
        )
        CodegenInterface.__init__(self, self.device_function)

        # Decide once which sibling for-loops need a tl.debug_barrier()
        # to make global writes visible to subsequent reads.
        self._compute_inter_loop_barriers()
        # Same, for a store->load read-after-write *within* one loop body.
        self._compute_intra_loop_barriers()

    def allow_dead_assignments_owned_by_nodes(self, nodes: tuple[Node, ...]) -> None:
        """Allow liveness-based DCE for assignments from proven pure FX nodes.

        Unlike removing their statements, this preserves any CSE temporary
        still read by another node or by a replacement collective lowering.
        The caller must prove that the supplied nodes have no side effects.
        """
        for node in nodes:
            for _body, statement in self._statements_by_owner_node_id.get(id(node), ()):
                for assignment in ast.walk(statement):
                    if (
                        isinstance(assignment, ast.Assign)
                        and len(assignment.targets) == 1
                        and isinstance(target := assignment.targets[0], ast.Name)
                    ):
                        self.device_function.dce_vars.append(target.id)

    def get_graph(self, graph_id: int) -> GraphInfo:
        return self.codegen_graphs[graph_id]

    @contextlib.contextmanager
    def resident_prep_lowering_scope(
        self, lowerings: list[ResidentPrepLowering]
    ) -> Iterator[None]:
        by_node = {
            (lowering.hoist.graph_id, lowering.hoist.prep_node_name): lowering
            for lowering in lowerings
        }
        self.resident_prep_lowering_stack.append(by_node)
        try:
            yield
        finally:
            self.resident_prep_lowering_stack.pop()

    def resident_prep_lowering_for_node(
        self, node: Node
    ) -> ResidentPrepLowering | None:
        for scope in reversed(self.resident_prep_lowering_stack):
            for (graph_id, prep_node_name), lowering in scope.items():
                if prep_node_name != node.name:
                    continue
                if self.get_graph(graph_id).graph is not node.graph:
                    continue
                return lowering
        return None

    def offset_var(self, block_idx: int) -> str:
        return self.active_device_loops[block_idx][-1].strategy.offset_var(block_idx)

    def tile_begin_var(self, block_idx: int) -> str:
        """Uniform first index of the current tile along ``block_idx``.

        ``tile.begin`` / ``tile.end`` / ``tile.id`` and their symbols render
        through this; the CuTe backend derives it from the owning strategy's
        thread / lane partition (``Backend.tile_begin_expr``).
        """
        expr = self.backend.tile_begin_expr(self, block_idx)
        if expr is not None:
            return expr
        return self.active_device_loops[block_idx][-1].strategy.tile_begin_var(
            block_idx
        )

    def index_var(self, block_idx: int) -> str:
        return self.active_device_loops[block_idx][-1].strategy.index_var(block_idx)

    def rolled_reduction_index_var(self, size: object) -> str | None:
        """Index of the rolled reduction loop over the dim of extent ``size``.

        A value over a reduction dim holds every position of the dim.  Where
        the roller moved it into the reduction loop, an iteration holds one
        chunk of them, at the loop's index (``roffset + local``): neither the
        full-extent range nor the chunk-local coordinate.  ``None`` unless
        ``size`` is a reduction dim whose rolled loop is active.
        """
        env = CompileEnvironment.current()
        block_id = env.resolve_block_id(size)
        if (
            block_id is None
            or not env.block_sizes[block_id].reduction
            or env.canonical_block_id(block_id) != block_id
        ):
            return None
        loops = self.active_device_loops.get(block_id)
        if not loops or not isinstance(loops[-1], DeviceLoopState):
            # A persistent reduction covers the whole dim at once.
            return None
        return loops[-1].strategy.index_var(block_id)

    def mask_var(self, block_idx: int) -> str | None:
        if loops := self.active_device_loops[block_idx]:
            return loops[-1].strategy.mask_var(block_idx)
        return None

    def _compute_inter_loop_barriers(self) -> None:
        """Walk every codegen graph; for each pair of consecutive sibling
        ``_for_loop`` / ``_for_loop_step`` nodes, set ``needs_barrier_before``
        on the second loop's ``ForLoopGraphInfo`` when their underlying tensor
        storages have a RAW dependency.

        TileIR shares Triton surface syntax but ``tl.debug_barrier()`` lowers
        to ``ttg.barrier`` which the TileIR pass pipeline does not legalize,
        so the analysis is a no-op there.
        """
        from ..language._tracing_ops import _for_loop
        from ..language._tracing_ops import _for_loop_step
        from .device_ir import ForLoopGraphInfo
        from .loop_dependency_checker import graph_memory_access_storage_ids

        env = CompileEnvironment.current()
        if not env.backend.supports_inter_loop_barriers:
            return

        for graph_info in self.codegen_graphs:
            # Pending writes accumulate across ALL prior sibling for-loops
            # since the last emitted barrier.  When a barrier is inserted
            # before a loop, it flushes all earlier writes, so the pending set
            # is reset and only writes from loops AFTER the barrier need to be
            # tracked for subsequent siblings.
            pending_writes: set[int] = set()
            for node in graph_info.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target not in (_for_loop, _for_loop_step):
                    continue
                cur_id = node.args[0]
                assert isinstance(cur_id, int)
                cur_info = self.codegen_graphs[cur_id]
                if not isinstance(cur_info, ForLoopGraphInfo):
                    continue
                reads, writes = graph_memory_access_storage_ids(
                    self.codegen_graphs, cur_id
                )
                need_barrier = bool(pending_writes & reads)
                cur_info.needs_barrier_before = need_barrier
                if need_barrier:
                    # Barrier flushes everything written before it.
                    pending_writes.clear()
                # Accumulate the current loop's writes for future siblings.
                pending_writes.update(writes)

    def _compute_intra_loop_barriers(self) -> None:
        """Mark loads that read a tensor an earlier store in the same body wrote,
        so the Triton ``load`` codegen emits a ``tl.debug_barrier()`` before them
        (the CuTe ``load`` codegen emits ``cute.arch.sync_threads()``).

        TileIR is excluded for the same reason as ``_compute_inter_loop_barriers``:
        ``tl.debug_barrier()`` lowers to ``ttg.barrier`` which the TileIR pass
        pipeline does not legalize.
        """
        from .loop_dependency_checker import mark_intra_loop_raw_barriers

        backend = CompileEnvironment.current().backend
        if not backend.supports_intra_loop_barriers(in_divergent_control_flow=False):
            return
        mark_intra_loop_raw_barriers(
            self.codegen_graphs,
            self.host_function.device_ir.root_ids,
            mark_in_divergent_control_flow=backend.supports_intra_loop_barriers(
                in_divergent_control_flow=True
            ),
        )

    def ensure_module_statements(self, source: str) -> None:
        """Append the module-level statements of ``source`` (an import or a
        helper a lowering needs once per generated module) the first time
        they are requested; later requests of the same text are no-ops."""
        if source in self._ensured_module_sources:
            return
        self._ensured_module_sources.add(source)
        for statement in ast.parse(source).body:
            self.module_statements.append(statement)

    def append_statement(
        self,
        body: list[ast.AST],
        stmt: ast.AST | str | None,
    ) -> None:
        """Append a statement while preserving its exact FX owner."""

        if stmt is None:
            return
        if isinstance(stmt, str):
            stmt = statement_from_string(stmt)
        body.append(stmt)
        owner_node = self._statement_owner_fx_node
        if owner_node is not None and self._track_statement_owners:
            self._statements_by_owner_node_id.setdefault(id(owner_node), []).append(
                (body, stmt)
            )
        self._record_statement_thread_references([stmt])

    def add_statement(self, stmt: ast.AST | str | None) -> None:
        self.append_statement(self.statements_stack[-1], stmt)

    def remove_statements_owned_by_nodes(self, nodes: tuple[Node, ...]) -> None:
        """Remove statements emitted earlier for exactly these FX nodes."""
        for node in nodes:
            entries = self._statements_by_owner_node_id.pop(id(node), ())
            for body, stmt in entries:
                with contextlib.suppress(ValueError):
                    body.remove(stmt)

    def statements_owned_by_node(
        self, node: Node
    ) -> tuple[tuple[list[ast.AST], ast.AST], ...]:
        """Return exact statement/container pairs recorded for one FX node."""

        return tuple(self._statements_by_owner_node_id.get(id(node), ()))

    def replace_owned_statement_span(
        self,
        body: list[ast.AST],
        nodes: tuple[Node, ...],
        replacement: tuple[ast.AST, ...],
    ) -> bool:
        """Atomically replace one contiguous, exactly-owned statement span.

        The method performs every ownership and contiguity check before
        changing ``body`` or the owner index.  This gives late CuTe rewrites a
        fail-closed commit point after constructing and validating detached
        replacement AST.
        """

        if not nodes or len(set(nodes)) != len(nodes):
            return False
        if any(not isinstance(statement, ast.stmt) for statement in replacement):
            return False
        source_ast_ids = {
            id(child)
            for statement in body
            for child in ast.walk(statement)
            if isinstance(child, (ast.stmt, ast.expr))
        }
        replacement_ast_ids = {
            id(child)
            for statement in replacement
            for child in ast.walk(statement)
            if isinstance(child, (ast.stmt, ast.expr))
        }
        if source_ast_ids & replacement_ast_ids:
            return False
        positions = {id(statement): index for index, statement in enumerate(body)}
        if len(positions) != len(body):
            return False
        claimed: list[ast.AST] = []
        for node in nodes:
            entries = self.statements_owned_by_node(node)
            if not entries or any(owner_body is not body for owner_body, _ in entries):
                return False
            claimed.extend(statement for _, statement in entries)
        claimed_ids = [id(statement) for statement in claimed]
        if len(claimed_ids) != len(set(claimed_ids)) or any(
            statement_id not in positions for statement_id in claimed_ids
        ):
            return False
        first = min(positions[statement_id] for statement_id in claimed_ids)
        last = max(positions[statement_id] for statement_id in claimed_ids)
        if {id(statement) for statement in body[first : last + 1]} != set(claimed_ids):
            return False

        referenced_dims = getattr(self, "referenced_thread_block_dims", None)
        previous_referenced_dims = (
            list(referenced_dims) if isinstance(referenced_dims, list) else None
        )
        try:
            self._record_statement_thread_references(list(replacement))
        except Exception:
            if previous_referenced_dims is not None and isinstance(
                referenced_dims, list
            ):
                referenced_dims[:] = previous_referenced_dims
            raise
        body[first : last + 1] = replacement
        for node in nodes:
            self._statements_by_owner_node_id.pop(id(node), None)
        return True

    def record_codegen_result(self, node: Node, result: object) -> None:
        """Record the final result returned for an FX node without mutating metadata."""

        self._codegen_results_by_owner_node_id[id(node)] = result

    def codegen_result_for_node(self, node: Node) -> tuple[bool, object]:
        """Return the final generated result for ``node``, if it was lowered."""

        key = id(node)
        if key not in self._codegen_results_by_owner_node_id:
            return False, None
        return True, self._codegen_results_by_owner_node_id[key]

    def get_rng_seed_buffer_statements(self) -> list[ast.AST]:
        from .compile_environment import CompileEnvironment

        env = CompileEnvironment.current()

        import_stmt = statement_from_string(
            "from torch._inductor import inductor_prims"
        )

        seed_buffer_stmt = statement_from_string(
            f"_rng_seed_buffer = {env.backend.rng_seed_buffer_expr(self.device_function.rng_seed_count)}"
        )

        return [import_stmt, seed_buffer_stmt]

    def lift(self, expr: ast.AST, *, dce: bool = False, prefix: str = "v") -> ast.Name:
        if isinstance(expr, ast.Name):
            return expr
        assert isinstance(expr, ExtendedAST), expr
        with expr:
            varname = self.tmpvar(dce=dce, prefix=prefix)
            self.add_statement(
                statement_from_string(f"{varname} = {{expr}}", expr=expr)
            )
            return create(ast.Name, id=varname, ctx=ast.Load())

    def lift_symnode(
        self,
        expr: ast.AST,
        sym_expr: sympy.Expr,
        *,
        dce: bool = False,
        prefix: str = "symnode",
    ) -> ast.Name:
        if isinstance(expr, ast.Name):
            return expr
        assert isinstance(expr, ExtendedAST), expr

        target_statements = self.statements_stack[-1]
        env = CompileEnvironment.current()
        from .host_function import HostFunction
        from .variable_origin import BlockSizeOrigin
        from .variable_origin import GridOrigin

        # Identify every block dimension the symbolic value depends on so we know
        # which loop nests the expression depends on.
        dep_block_ids: set[int] = set()
        active_loop_stack = self._active_loop_stack()
        for symbol in sym_expr.free_symbols:
            if not isinstance(symbol, sympy.Symbol):
                continue
            origin_info = HostFunction.current().expr_to_origin.get(symbol)
            if origin_info is None or not isinstance(
                origin_info.origin, GridOrigin | BlockSizeOrigin
            ):
                continue
            canonical_block_id = env.canonical_block_id(origin_info.origin.block_id)
            matching_loop_ids = {
                block_id
                for loop_state in active_loop_stack
                for block_id in loop_state.block_ids
                if env.canonical_block_id(block_id) == canonical_block_id
            }
            if matching_loop_ids:
                dep_block_ids.update(matching_loop_ids)
            else:
                dep_block_ids.add(origin_info.origin.block_id)

        # Walk outward through the active device loops: as soon as we see a loop
        # whose block id appears in the dependency set we must stop, otherwise we
        # can safely hoist into that loop's outer prefix (which executes before the
        # loop body).
        for loop_state in reversed(active_loop_stack):
            if dep_block_ids.intersection(loop_state.block_ids):
                break
            target_statements = loop_state.outer_prefix

        with expr:
            varname = self.tmpvar(dce=dce, prefix=prefix)
            # Emit the temporary into the chosen statement list so the symbolic
            # expression is computed exactly once at the appropriate scope.
            target_statements.append(
                statement_from_string(f"{varname} = {{expr}}", expr=expr)
            )
            # Reuse the temporary everywhere else in the kernel body.
            return create(ast.Name, id=varname, ctx=ast.Load())

    def _active_loop_stack(
        self,
    ) -> list[DeviceLoopState | EmitPipelineLoopState | ForiLoopState]:
        seen: set[int] = set()
        stack: list[DeviceLoopState | EmitPipelineLoopState | ForiLoopState] = []
        for loops in self.active_device_loops.values():
            for loop_state in loops:
                if not isinstance(
                    loop_state, (DeviceLoopState, EmitPipelineLoopState, ForiLoopState)
                ):
                    continue
                key = id(loop_state)
                if key not in seen:
                    stack.append(loop_state)
                    seen.add(key)
        return stack

    @contextlib.contextmanager
    def statement_owner_node(self, node: Node) -> Iterator[None]:
        prior = self._statement_owner_fx_node
        self._statement_owner_fx_node = node
        try:
            yield
        finally:
            self._statement_owner_fx_node = prior

    def _record_thread_axis_sizes(self, axis_sizes: dict[int, int]) -> None:
        for axis, size in axis_sizes.items():
            if 0 <= axis < 3:
                self.max_thread_block_dims[axis] = max(
                    self.max_thread_block_dims[axis], size
                )

    def _record_active_thread_axis_sizes(self) -> None:
        self._record_thread_axis_sizes(self._current_active_thread_axis_sizes())

    @contextlib.contextmanager
    def divergent_control_flow(self) -> Iterator[None]:
        """Generate statements of a branch or while loop whose condition a
        CuTe SIMT thread may evaluate differently from its neighbours; a
        block-wide barrier must not be placed inside
        (``divergent_control_flow_depth``)."""
        self.divergent_control_flow_depth += 1
        try:
            yield
        finally:
            self.divergent_control_flow_depth -= 1

    def active_thread_axis_sizes(self) -> dict[int, int]:
        """The thread count along each launch axis the body being generated may
        address: the active loops' and the free ``hl.arange`` dims'."""
        return self._current_active_thread_axis_sizes()

    def launch_thread_axis_sizes(self) -> dict[int, int]:
        """The thread count along each launch axis, as far as the kernel has claimed it.

        The maximum over the loops emitted so far (``max_thread_block_dims``,
        the launch's block), the active loops and the free ``hl.arange`` dims,
        and every strategy's reserved axes, entered or not.  A block-wide
        barrier concerns every thread of the launch: the threads a loop
        nested in a body addresses run the rest of the body too.
        """
        sizes = self._strategy_thread_axis_sizes()
        for axis, size in self._current_active_thread_axis_sizes().items():
            sizes[axis] = max(sizes.get(axis, 1), size)
        for axis, size in enumerate(self.max_thread_block_dims):
            sizes[axis] = max(sizes.get(axis, 1), size)
        return sizes

    def _current_active_thread_axis_sizes(self) -> dict[int, int]:
        seen: set[int] = set()
        axis_sizes: dict[int, int] = {}
        for loops in self.active_device_loops.values():
            for loop_state in loops:
                key = id(loop_state)
                if key in seen:
                    continue
                seen.add(key)
                for axis, size in loop_state.thread_axis_sizes.items():
                    axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        # The thread axes the backend claims outside the strategy loop states
        # (CuTe: the free ``hl.arange`` dims' synthetic axes) fold in here too,
        # so ``_record_statement_thread_references`` grows the launch block to
        # cover the lanes they address.
        for axis, size in self.backend.synthetic_thread_axis_sizes(self).items():
            axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        return axis_sizes

    def _strategy_thread_axis_sizes(self) -> dict[int, int]:
        sizes: dict[int, int] = {}
        for loops in self.active_device_loops.values():
            for loop_state in loops:
                for axis, size in loop_state.thread_axis_sizes.items():
                    sizes[axis] = max(sizes.get(axis, 1), size)
        if self.current_grid_state is not None:
            for axis, size in self.current_grid_state.thread_axis_sizes.items():
                sizes[axis] = max(sizes.get(axis, 1), size)
        # Fold in axes reserved by strategies that have not been entered yet
        # (e.g. a matmul K-reduction on axis 0) so synthetic ``hl.arange`` dims
        # both avoid those axes and count them toward the thread budget.
        for axis, size in self._all_strategy_reserved_axes().items():
            sizes[axis] = max(sizes.get(axis, 1), size)
        return sizes

    def _all_strategy_reserved_axes(self) -> dict[int, int]:
        """Thread axes reserved by every dispatcher strategy and their extent.

        Unlike ``_strategy_thread_axis_sizes`` (which only sees *active* loop
        states), this includes strategies that have not begun codegen yet — e.g.
        a matmul's K-reduction strategy that always claims thread axis 0. A free
        ``hl.arange`` allocated before that reduction's loop is entered must still
        avoid its axis, otherwise the arange's row/col lanes collide with the
        reduction's warp lanes and silently corrupt the result.
        """
        sizes: dict[int, int] = {}
        tile_strategy = getattr(self.device_function, "tile_strategy", None)
        if tile_strategy is None:
            return sizes
        for strategy in getattr(tile_strategy, "strategies", []):
            if strategy.thread_axes_used() <= 0:
                continue
            base_axis = tile_strategy.thread_axis_for_strategy(strategy)
            if base_axis is None:
                continue
            for block_id in strategy.block_ids:
                axis = tile_strategy.thread_axis_for_block_id(block_id)
                extent = tile_strategy.thread_extent_for_block_id(block_id)
                if axis is None or not isinstance(extent, int) or extent <= 1:
                    continue
                sizes[axis] = max(sizes.get(axis, 1), extent)
        return sizes

    def _record_statement_thread_references(
        self,
        statements: list[ast.AST],
        axis_sizes: dict[int, int] | None = None,
    ) -> None:
        if axis_sizes is None:
            axis_sizes = self._current_active_thread_axis_sizes()
        for axis in self.backend.referenced_thread_axes(statements):
            self.referenced_thread_block_dims[axis] = max(
                self.referenced_thread_block_dims[axis],
                axis_sizes.get(axis, 1),
            )

    @contextlib.contextmanager
    def set_statements(self, new_statements: list[ast.AST] | None) -> Iterator[None]:
        if new_statements is None:
            yield
        else:
            expr_to_var_info = self.device_function.expr_to_var_info
            # We don't want to reuse vars assigned in a nested scope, so copy it
            self.device_function.expr_to_var_info = expr_to_var_info.copy()
            self.statements_stack.append(new_statements)
            try:
                yield
            finally:
                self.statements_stack.pop()
                self.device_function.expr_to_var_info = expr_to_var_info

    @contextlib.contextmanager
    def set_on_device(self) -> Iterator[None]:
        assert self.on_device is False
        self.on_device = True
        prior = self.host_statements
        self.host_statements = self.statements_stack[-1]
        try:
            yield
        finally:
            self.on_device = False
            self.host_statements = prior

    @contextlib.contextmanager
    def add_device_loop(
        self,
        device_loop: DeviceLoopState,
        *,
        needs_barrier_before: bool = False,
    ) -> Iterator[None]:
        with self.set_statements(device_loop.inner_statements):
            for idx in device_loop.block_ids:
                active_loops = self.active_device_loops[idx]
                active_loops.append(device_loop)
                if len(active_loops) > 1:
                    raise exc.NestedDeviceLoopsConflict
            self._record_active_thread_axis_sizes()
            self._record_statement_thread_references(device_loop.inner_statements)
            try:
                yield
                # Finalizers first: a collected tile-vector store that a later
                # statement of the body observes returns to its scalar form
                # here, so the nest check and the barrier pass below judge the
                # body as it is emitted.
                for finalize in device_loop.body_finalizers:
                    finalize()
            finally:
                for idx in device_loop.block_ids:
                    self.active_device_loops[idx].pop()
        # The body is complete: the CuTe per-thread lane loops nested around
        # it must be the tile program (or pin their uniform atomics).
        device_loop.check_lane_loop_nest()
        if needs_barrier_before:
            for statement in self.device_function.cta_barrier():
                self.add_statement(statement)
        self.statements_stack[-1].extend(device_loop.outer_prefix)
        self.add_statement(device_loop.for_node)
        self.statements_stack[-1].extend(device_loop.outer_suffix)

    @contextlib.contextmanager
    def add_emit_pipeline_loop(
        self, pipeline_state: EmitPipelineLoopState
    ) -> Iterator[None]:
        """Context manager for emit_pipeline-based loops on Pallas/TPU.

        Redirects body codegen into ``pipeline_state.inner_statements``
        and registers block_ids in ``active_device_loops``.  The caller
        is responsible for emitting the function def and pipeline call
        after the context exits.
        """
        with self.set_statements(pipeline_state.inner_statements):
            for idx in pipeline_state.block_ids:
                active_loops = self.active_device_loops[idx]
                active_loops.append(pipeline_state)
                if len(active_loops) > 1:
                    raise exc.NestedDeviceLoopsConflict
            try:
                yield
            finally:
                for idx in pipeline_state.block_ids:
                    self.active_device_loops[idx].pop()
        # Flush any symnode bindings hoisted into the loop's outer_prefix
        # (via lift_symnode) into the parent scope, so they precede the
        # function def + pipeline call the caller is about to add.
        self.statements_stack[-1].extend(pipeline_state.outer_prefix)

    @contextlib.contextmanager
    def add_fori_loop(self, fori_state: ForiLoopState) -> Iterator[None]:
        """Context manager for fori_loop-based loops on Pallas/TPU.

        Redirects body codegen into ``fori_state.inner_statements``
        and registers block_ids in ``active_device_loops``.  The caller
        is responsible for emitting the function def and fori_loop call
        after the context exits.
        """
        with self.set_statements(fori_state.inner_statements):
            for idx in fori_state.block_ids:
                active_loops = self.active_device_loops[idx]
                active_loops.append(fori_state)
                if len(active_loops) > 1:
                    raise exc.NestedDeviceLoopsConflict
            try:
                yield
            finally:
                for idx in fori_state.block_ids:
                    self.active_device_loops[idx].pop()
        self.statements_stack[-1].extend(fori_state.outer_prefix)

    def set_active_loops(self, device_grid: DeviceLoopOrGridState) -> None:
        if isinstance(device_grid, DeviceGridState):
            # The root's own axes: reductions lowered in the body add theirs
            # to ``thread_axis_sizes`` later, and a reduction axis may launch
            # fewer threads than it records (MMA fallbacks, collectives).
            device_grid.root_axis_sizes = {
                axis: size
                for axis, size in device_grid.thread_axis_sizes.items()
                if 0 <= axis < 3
            }
            for axis, size in device_grid.root_axis_sizes.items():
                self.root_thread_block_dims[axis] = max(
                    self.root_thread_block_dims[axis], size
                )
        self.current_grid_state = (
            device_grid if isinstance(device_grid, DeviceGridState) else None
        )
        for idx in device_grid.block_ids:
            self.active_device_loops[idx] = [device_grid]
        self._record_active_thread_axis_sizes()
        if isinstance(device_grid, DeviceGridState):
            self._record_statement_thread_references(device_grid.lane_setup_statements)

    def push_active_loops(self, device_loop: DeviceLoopOrGridState) -> None:
        for idx in device_loop.block_ids:
            self.active_device_loops[idx].append(device_loop)
        self._record_active_thread_axis_sizes()

    def generic_visit(self, node: ast.AST) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        fields = {}
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                fields[field] = new_list = []
                with self.set_statements(
                    new_list
                    if old_value and isinstance(old_value[0], ast.stmt)
                    else None
                ):
                    for item in old_value:
                        new_list.append(self.visit(item))  # mutation in visit
            elif isinstance(old_value, ast.AST):
                fields[field] = self.visit(  # pyrefly: ignore[unsupported-operation]
                    old_value
                )
            else:
                fields[field] = old_value
        # pyrefly: ignore[bad-return, bad-argument-type]
        return node.new(fields)

    def visit_For(self, node: ast.For) -> ast.AST | None:
        assert isinstance(node, ExtendedAST)
        if node._loop_type == LoopType.GRID:
            assert not node.orelse

            assert node._root_id is not None
            if len(self.host_function.device_ir.root_ids) == 1:
                body = self.device_function.body
            else:
                assert len(self.host_function.device_ir.root_ids) > 1
                # Multiple top level for loops

                if node._root_id == 0:
                    self.device_function.set_pid(
                        ForEachProgramID(
                            self.device_function.new_var("pid_shared", dce=False),
                        )
                    )
                    self.device_function.body.extend(
                        # pyrefly: ignore [missing-attribute]
                        self.device_function.pid.codegen_pid_init()
                    )
                body = []
            with (
                self.set_on_device(),
                self.set_statements(body),
            ):
                assert node._root_id is not None
                root_graph_info = self.get_graph(
                    self.host_function.device_ir.root_ids[node._root_id],
                )
                previous_root_graph_info = self.current_root_graph_info
                self.current_root_graph_info = root_graph_info
                try:
                    iter_node = node.iter
                    assert isinstance(iter_node, ExtendedAST)
                    with iter_node:
                        assert isinstance(iter_node, ast.Call)
                        args = []
                        kwargs = {}
                        for arg_node in _flatten_starred_args(iter_node.args):
                            assert isinstance(arg_node, ExtendedAST)
                            assert arg_node._type_info is not None
                            args.append(arg_node._type_info.proxy())
                        for kwarg_node in iter_node.keywords:
                            assert kwarg_node.arg is not None
                            assert isinstance(kwarg_node.value, ExtendedAST)
                            assert kwarg_node.value._type_info is not None
                            kwargs[kwarg_node.arg] = kwarg_node.value._type_info.proxy()
                        fn_node = iter_node.func
                        assert isinstance(fn_node, ExtendedAST)
                        assert fn_node._type_info is not None
                        fn = fn_node._type_info.proxy()
                        assert is_api_func(fn)
                        env = CompileEnvironment.current()
                        codegen_fn = fn._codegen.lookup(
                            env.codegen_name,
                            backend_name=env.backend_name,
                            what=f"codegen for API function {fn.__qualname__}",
                        )
                        bound = fn._signature.bind(*args, **kwargs)
                        bound.apply_defaults()
                        from .inductor_lowering import CodegenState

                        state = CodegenState(
                            self,
                            fx_node=None,
                            proxy_args=[*bound.arguments.values()],
                            # pyrefly: ignore [bad-argument-type]
                            ast_args=None,
                        )

                        codegen_fn(state)
                    if isinstance(self.current_grid_state, DeviceGridState):
                        self.current_grid_state.hoist_parent_statements = (
                            self.statements_stack[-1]
                        )
                    root = root_graph_info.graph
                    # The backend's special-purpose root emitters, in priority
                    # order; none owning the root leaves it to the ordinary
                    # lowering below.
                    root_emitters = CompileEnvironment.current().backend.root_emitters()
                    if not any(emitter.try_emit(self) for emitter in root_emitters):
                        grid_state = self.current_grid_state
                        if isinstance(grid_state, DeviceGridState):
                            # Codegen the body first so synthetic free-``hl.arange``
                            # lane loops registered *during* body lowering (CuTe
                            # over-budget chunking) are visible to the wrap below.
                            wrapped_body: list[ast.AST] = []
                            with self.set_statements(wrapped_body):
                                codegen_call_with_graph(self, root, [])
                            if self.backend.rewrite_lowered_root(
                                self, grid_state, wrapped_body
                            ):
                                self.statements_stack[-1].extend(
                                    grid_state.outer_prefix
                                )
                                self.statements_stack[-1].extend(wrapped_body)
                                self.statements_stack[-1].extend(
                                    grid_state.outer_suffix
                                )
                            elif grid_state.has_lane_loops():
                                self.statements_stack[-1].extend(
                                    grid_state.outer_prefix
                                )
                                if self.backend.consume_root_lane_loop_suppression(
                                    self.device_function
                                ):
                                    self.suppressed_lane_names |= (
                                        grid_state.lane_defined_names()
                                    )
                                    self.statements_stack[-1].extend(wrapped_body)
                                else:
                                    self.statements_stack[-1].extend(
                                        grid_state.wrap_body(wrapped_body)
                                    )
                                self.statements_stack[-1].extend(
                                    grid_state.outer_suffix
                                )
                            else:
                                # A shared launch can require coordinate and
                                # surplus-mask setup even with no lane loops.
                                self.statements_stack[-1].extend(
                                    grid_state.outer_prefix
                                )
                                self.statements_stack[-1].extend(
                                    grid_state.wrap_body(wrapped_body)
                                )
                                self.statements_stack[-1].extend(
                                    grid_state.outer_suffix
                                )
                        else:
                            codegen_call_with_graph(self, root, [])
                finally:
                    self.current_root_graph_info = previous_root_graph_info

                # Reduction preambles can register extents that depend on a later
                # root's block size. Its cached name exists before its host
                # assignment, so wait until every root has emitted its definitions.
                if node._root_id == len(self.host_function.device_ir.root_ids) - 1:
                    self.device_function.flush_deferred_rdim_defs(self)

                pid = self.device_function.pid
                if isinstance(pid, ForEachProgramID):
                    pid.case_bodies.append(body)
                    pid.case_phases.append(
                        self.host_function.device_ir.phase_for_root(node._root_id)
                    )
                    if node._root_id < len(self.host_function.device_ir.root_ids) - 1:
                        pid.case_branches.append(
                            create(
                                ast.If,
                                test=pid.codegen_test(state),
                                body=body,
                                orelse=[],
                            )
                        )
            if node._root_id == len(self.host_function.device_ir.root_ids) - 1:
                if isinstance(self.device_function.pid, ForEachProgramID):
                    self.device_function.body.extend(
                        self.device_function.pid.codegen_dispatch()
                    )
                if self.suppressed_lane_names:
                    _drop_dead_lane_derived_assignments(
                        self.device_function.body,
                        self.device_function.preamble,
                        self.suppressed_lane_names,
                    )
                if self.device_function.pid is not None:
                    persistent_body = self.device_function.pid.setup_persistent_kernel(
                        self.device_function
                    )
                    if persistent_body is not None:
                        # pyrefly: ignore [bad-assignment]
                        self.device_function.body = persistent_body
                    else:
                        # The persistent path pulls the backend's post-loop
                        # cleanup out of the work-tile loop; the flat-grid path
                        # must order it last the same way (a tcgen05 multi-store
                        # fan-out's one-shot teardown runs after every store
                        # reads the accumulator).
                        self.device_function.body = (
                            self.backend.reorder_post_loop_statements(
                                self.device_function, list(self.device_function.body)
                            )
                        )
                # Mark extra params as placeholder args — they appear only in
                # placeholder strings, not in the AST body, so DCE would
                # otherwise remove them.
                for param in self._extra_params:
                    self.device_function.placeholder_args.add(param)
                CompileEnvironment.current().backend.lower_owned_lane_loop(self)
                self.device_function.dead_code_elimination()
                if not self.device_function.preamble and not self.device_function.body:
                    raise exc.EmptyDeviceLoopAfterDCE
                return self.device_function.codegen_function_call()
            return None
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        assert isinstance(node, ExtendedAST)
        if isinstance(node.ctx, ast.Load) and node._type_info is not None:
            origin = node._type_info.origin
            if (
                isinstance(origin, ArgumentOrigin)
                and origin.name in self.host_function.constexpr_args
            ):
                value = self.host_function.constexpr_args[origin.name]
                return expr_from_string(
                    self.host_function.literal_expr(value)
                    if type(value) in (torch.dtype, torch.device)
                    else repr(value)
                )
            if origin.needs_rename():
                # `x` => `_source_module.x`
                return expr_from_string(origin.host_str())
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        from .type_info import CallableType
        from .type_info import SequenceType
        from .type_info import TileIndexType

        func_node = node.func
        assert isinstance(func_node, ExtendedAST)

        assert isinstance(node, ExtendedAST)
        env = CompileEnvironment.current()
        if self.on_device:
            pass
        elif isinstance(type_info := node._type_info, TileIndexType):
            return expr_from_string(
                self.host_function.literal_expr(
                    self.device_function.resolved_block_size(type_info.block_id)
                )
            )
        elif isinstance(type_info, SequenceType) and all(
            isinstance(x, TileIndexType) for x in type_info.unpack()
        ):
            values = [
                value
                for value in type_info.unpack()
                if isinstance(value, TileIndexType)
            ]
            return expr_from_string(
                self.host_function.literal_expr(
                    [
                        self.device_function.resolved_block_size(x.block_id)
                        for x in values
                    ]
                )
            )
        elif isinstance(fn_type_info := func_node._type_info, CallableType) and (
            is_api_func(api := fn_type_info.value)
        ):
            codegen_fn = api._codegen.lookup(
                env.codegen_name,
                backend_name=env.backend_name,
                what=f"codegen for API function {api.__qualname__}",
            )
            ast_args = []
            ast_kwargs = {}
            proxy_args = []
            proxy_kwargs = {}
            for arg in _flatten_starred_args(node.args):
                assert isinstance(arg, ExtendedAST)
                assert arg._type_info is not None
                ast_args.append(arg)
                proxy_args.append(arg._type_info.proxy())
            for kwarg in node.keywords:
                assert kwarg.arg is not None
                assert isinstance(kwarg.value, ExtendedAST)
                assert kwarg.value._type_info is not None
                ast_kwargs[kwarg.arg] = kwarg.value
                proxy_kwargs[kwarg.arg] = kwarg.value._type_info.proxy()
            ast_params = api._signature.bind(*ast_args, **ast_kwargs)
            proxy_params = api._signature.bind(*proxy_args, **proxy_kwargs)
            ast_params.apply_defaults()
            proxy_params.apply_defaults()
            # pyrefly: ignore [bad-return]
            return codegen_fn(
                CodegenState(
                    self,
                    None,
                    proxy_args=[*proxy_params.arguments.values()],
                    ast_args=[*ast_params.arguments.values()],
                )
            )
        if not self.on_device and self._needs_device_kwarg(node):
            node = self._inject_device_kwarg(node)
        return self.generic_visit(node)

    def _needs_device_kwarg(self, node: ast.Call) -> bool:
        """Check if a host-level torch factory call is missing device=."""
        from .type_info import CallableType

        func_node = node.func
        if not isinstance(func_node, ExtendedAST):
            return False
        fn_type = func_node._type_info
        if not isinstance(fn_type, CallableType):
            return False
        if fn_type.value not in _device_constructors():
            return False
        return not any(kw.arg == "device" for kw in node.keywords)

    def _inject_device_kwarg(self, node: ast.Call) -> ast.Call:
        for name, val in self.host_function.params.arguments.items():
            if isinstance(val, torch.Tensor):
                device_expr = expr_from_string(f"{name}.device")
                new_kw = create(ast.keyword, arg="device", value=device_expr)
                node.keywords = [*node.keywords, new_kw]
                return node
        return node

    def host_dead_code_elimination(self) -> None:
        dce_vars: OrderedSet[str] = OrderedSet()
        allow_compiler_shape_helpers = (
            self.device_function.config.cross_loop_pipeline != "barrier"
        )
        for stmt in self.host_statements:
            if (
                isinstance(stmt, ast.Assign)
                and definitely_does_not_have_side_effects(
                    stmt.value,
                    allow_compiler_shape_helpers=allow_compiler_shape_helpers,
                )
                and all(isinstance(name, ast.Name) for name in stmt.targets)
            ):
                for name in stmt.targets:
                    assert isinstance(name, ast.Name)
                    dce_vars.add(name.id)

        dead_assignment_elimination(self.host_statements, list(dce_vars))
        dead_expression_elimination(self.host_statements)


class TensorReference(NamedTuple):
    node: ast.AST
    name: str
    type_info: TensorType

    @property
    def is_host(self) -> bool:
        return self.type_info.origin.is_host()


def emit_main_def() -> ast.stmt:
    return statement_from_string("""
if __name__ == "__main__":
    call()
    """)


def _maybe_emit_compact_worklist_builder(codegen: GenerateAST) -> None:
    """Emit the module-level jnp ``_build_worklist`` for a compact-worklist kernel."""
    env = CompileEnvironment.current()
    plan = env.compact_worklist_plan
    if plan is None:
        return
    from .pallas.compact_worklist import render_build_worklist

    source, offset_params = render_build_worklist(
        plan,
        block_expr=str(env.compact_worklist_block),
        upper_expr=str(env.compact_worklist_upper),
    )
    env.compact_worklist_offset_params = offset_params
    codegen.module_statements.append(statement_from_string(source))


def generate_ast(
    func: HostFunction,
    config: Config,
    emit_repro_caller: bool,
    *,
    store_transform: Callable[..., ast.AST] | None = None,
    load_transform: Callable[..., ast.AST] | None = None,
    extra_params: list[str] | None = None,
    _codegen_graphs: list[GraphInfo] | None = None,
    _memory_counters: dict[str, int] | None = None,
    _host_prefix: list[ast.AST] | None = None,
    _bounded_cache_request: BoundedCacheRequest | None = None,
) -> ast.Module:
    """Generate the kernel module for ``config``.

    A backend may ask for the kernel to be generated again after a decision
    made before the tile body existed is revisited once it does: its
    ``retry_policy`` names the regeneration signals and repairs the config for
    the next pass (``CuteRegeneration`` documents the CuTe ones).  An abandoned
    pass leaves nothing behind but its host prefix, which is dropped here;
    per-pass state of the memory transforms lives on the pass's
    ``GenerateAST`` (``prologue_first_indexing``), so a retry starts it afresh.
    """
    prefix_length = len(_host_prefix) if _host_prefix is not None else 0
    policy = CompileEnvironment.current().backend.retry_policy()
    with policy.scope():
        while True:
            try:
                return _generate_ast(
                    func,
                    config,
                    emit_repro_caller,
                    store_transform=store_transform,
                    load_transform=load_transform,
                    extra_params=extra_params,
                    _codegen_graphs=_codegen_graphs,
                    _memory_counters=_memory_counters,
                    _host_prefix=_host_prefix,
                    _bounded_cache_request=_bounded_cache_request,
                )
            except policy.signals as signal:
                # ``NO_RETRY`` has no signals: ``except ()`` catches nothing
                # and every error propagates from the single pass above.
                if _host_prefix is not None:
                    del _host_prefix[prefix_length:]
                repaired = policy.repair(
                    signal, config, caller_graphs=_codegen_graphs is not None
                )
                if repaired is None:
                    raise
                config = repaired


def _generate_ast(
    func: HostFunction,
    config: Config,
    emit_repro_caller: bool,
    *,
    store_transform: Callable[..., ast.AST] | None = None,
    load_transform: Callable[..., ast.AST] | None = None,
    extra_params: list[str] | None = None,
    _codegen_graphs: list[GraphInfo] | None = None,
    _memory_counters: dict[str, int] | None = None,
    _host_prefix: list[ast.AST] | None = None,
    _bounded_cache_request: BoundedCacheRequest | None = None,
) -> ast.Module:
    with func:
        env = CompileEnvironment.current()
        if _codegen_graphs is None:
            env.backend.validate_codegen_config(env, func.device_ir, config)
        # The backend's whole-kernel generators, first match wins; one that
        # applies but cannot honor this call's transforms declines (raising or
        # rewriting ``config``) and the next is tried.
        transforms = KernelTransforms(
            store_transform=store_transform,
            load_transform=load_transform,
            extra_params=extra_params,
            codegen_graphs=_codegen_graphs,
            memory_counters=_memory_counters,
            host_prefix=_host_prefix,
            bounded_cache_request=_bounded_cache_request,
        )
        for generator in env.backend.kernel_generators():
            if not generator.applies(func, config, transforms):
                continue
            if not generator.transform_policy.honors(config, transforms):
                config = generator.transform_policy.decline(config)
                continue
            return generator.generate(func, config, emit_repro_caller, transforms)
        if len(func.device_ir.phases) > 1:
            if not str(config.pid_type).startswith("persistent"):
                raise exc.BarrierRequiresPersistent(config.pid_type)
        codegen = GenerateAST(
            func,
            config,
            store_transform=store_transform,
            load_transform=load_transform,
            extra_params=extra_params,
            codegen_graphs=_codegen_graphs,
        )
        if _memory_counters is not None:
            for name, value in _memory_counters.items():
                setattr(codegen.device_function, name, value)
        with codegen.device_function:
            CompileEnvironment.current().backend.pre_codegen(
                graphs=codegen.codegen_graphs,
                config=config,
                tile_strategy=codegen.device_function.tile_strategy,
            )

            # Emit the worklist builder + record its offset params BEFORE the host
            # body is visited (the launcher call -- which reads the offset params
            # via build_launcher_args -- is generated during that visit).
            _maybe_emit_compact_worklist_builder(codegen)

            prefix_recorded = False
            for stmt in func.body:
                if (
                    _host_prefix is not None
                    and not prefix_recorded
                    and isinstance(stmt, ExtendedAST)
                    and stmt._loop_type is LoopType.GRID
                ):
                    _host_prefix.extend(codegen.host_statements)
                    prefix_recorded = True
                try:
                    codegen.add_statement(codegen.visit(stmt))
                except exc.BackendUnsupported as error:
                    # A lane-loop pass after the root dispatch can refuse a
                    # root the fragment owner declined for capacity.
                    failure = env.backend.explain_root_failure(codegen, error)
                    if failure is error:
                        # ``raise error from error`` would make the exception
                        # its own cause and recurse while being printed.
                        raise
                    raise failure from error
            env.backend.after_device_codegen(codegen, func, config)
            try:
                if _bounded_cache_request is None:
                    kernel_def = codegen.device_function.codegen_function_def()
                else:
                    kernel_def = codegen.device_function.codegen_function_def(
                        bounded_cache_request=_bounded_cache_request
                    )
            except exc.BackendUnsupported as error:
                # The backend may know a prior owner declined this root for a
                # config-level reason (``Backend.explain_root_failure``).
                failure = env.backend.explain_root_failure(codegen, error)
                if failure is error:
                    raise
                raise failure from error
            codegen.host_dead_code_elimination()

            # Retarget output-only tensor allocations to ``device='meta'`` so
            # the factory call produces a zero-storage metadata-only tensor
            # instead of allocating real HBM. The launcher reassigns the
            # variable to the real result tensor.
            output_only_names = CompileEnvironment.current().backend.output_only_names()
            if output_only_names:
                oo_set = set(output_only_names)
                # ``static_shapes=True``: cache the output-only meta placeholder
                # ``torch.empty(..., device='meta')`` on the inner device
                # function so repeat calls reuse it (shape/dtype/device are
                # constant).  ``static_shapes=False`` keeps the per-call alloc.
                cache_static_shapes = (
                    CompileEnvironment.current().settings.static_shapes
                )
                inner_fn_name = codegen.device_function.name
                cached_meta_index = 0
                new_host_statements: list[ast.AST] = []
                for stmt in codegen.host_statements:
                    if not (
                        isinstance(stmt, ast.Assign)
                        and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id in oo_set
                        and not getattr(stmt, "_is_kernel_call", False)
                        and isinstance(stmt.value, ast.Call)
                    ):
                        new_host_statements.append(stmt)
                        continue
                    call = stmt.value
                    call.keywords = [
                        kw for kw in call.keywords if kw.arg != "device"
                    ] + [ast.keyword(arg="device", value=ast.Constant(value="meta"))]
                    if not cache_static_shapes:
                        new_host_statements.append(stmt)
                        continue
                    # Read the cache slot (``getattr`` -> ``None`` on the first
                    # call) and populate it inline on the miss.  Kept inline (no
                    # helper/lambda) so the warm path stays a plain attr read.
                    varname = stmt.targets[0].id
                    cache_attr = f"_helion_output_meta_cache_{cached_meta_index}"
                    cached_meta_index += 1
                    get_stmt = statement_from_string(
                        f"{varname} = getattr({inner_fn_name}, '{cache_attr}', None)"
                    )
                    if_stmt = statement_from_string(
                        f"if {varname} is None:\n"
                        f"    {varname} = {inner_fn_name}.{cache_attr} = "
                        f"{{__orig_call__}}\n",
                        __orig_call__=call,
                    )
                    new_host_statements.extend([get_stmt, if_stmt])
                if cache_static_shapes:
                    codegen.host_statements = new_host_statements

            # Inject RNG seed buffer creation if needed
            rng_statements = (
                codegen.get_rng_seed_buffer_statements()
                if codegen.device_function.has_rng_ops()
                else []
            )
            final_host_statements: list[ast.AST] = [
                *rng_statements,
                *codegen.host_statements,
            ]
            post_kernel_metadata_statements: list[ast.AST] = []
            # Assert sourceless prologue params were actually removed by DCE
            if codegen.device_function.sourceless_prologue_params:
                remaining = codegen.device_function.sourceless_prologue_params & {
                    arg.name for arg in codegen.device_function.arguments
                }
                assert not remaining, (
                    f"sourceless prologue params not removed by DCE: {remaining}"
                )

            final_host_statements, fill_kernels = env.backend.finalize_host_statements(
                codegen, func, final_host_statements
            )
            post_kernel_metadata_statements.extend(fill_kernels)

            host_def = func.codegen_function_def(
                final_host_statements,
                extra_params=codegen._extra_params,
                removed_args=codegen.device_function.sourceless_prologue_params,
            )

            call_def = []
            main_def = []
            if emit_repro_caller:
                call_def = [func.codegen_call_function()]
                main_def = [emit_main_def()]

            module_body: list[ast.stmt] = []
            for stmt in (
                *func.codegen_imports(),
                *codegen.module_statements,
                *codegen.device_function.codegen_helper_functions(),
                *kernel_def,
                *post_kernel_metadata_statements,
                host_def,
                *call_def,
                *main_def,
            ):
                assert isinstance(stmt, ast.stmt)
                module_body.append(stmt)
            result = ast.Module(module_body, [])
            existing_imports = {
                ast.unparse(stmt)
                for stmt in result.body
                if isinstance(stmt, (ast.Import, ast.ImportFrom))
            }
            missing_imports = [
                line
                for line in get_needed_import_lines(result)
                if line not in existing_imports
            ]
            insert_at = 0
            while insert_at < len(result.body):
                stmt = result.body[insert_at]
                if not isinstance(stmt, ast.ImportFrom) or stmt.module != "__future__":
                    break
                insert_at += 1
            result.body[insert_at:insert_at] = [
                statement_from_string(line) for line in missing_imports
            ]
            if _memory_counters is not None:
                for name in _memory_counters:
                    _memory_counters[name] = getattr(codegen.device_function, name)
            # break circular reference for better GC
            del codegen.device_function.codegen
            return result
