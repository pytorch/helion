from __future__ import annotations

import ast
import collections
import contextlib
import re
from typing import TYPE_CHECKING
from typing import NamedTuple

import sympy
import torch
from torch.utils._device import _device_constructors
from torch.utils._ordered_set import OrderedSet

from .. import exc
from ..language._decorators import is_api_func
from ..runtime.config import Config
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
from .device_function import ConstExprArg
from .device_function import DeviceFunction
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

    from ..runtime import Config
    from .device_ir import GraphInfo
    from .host_function import HostFunction
    from .loop_dependency_checker import LoopDependencyChecker
    from .tile_strategy import DeviceLoopOrGridState
    from .type_info import TensorType


class GenerateAST(NodeVisitor, CodegenInterface):
    def __init__(
        self,
        func: HostFunction,
        config: Config,
        *,
        store_transform: Callable[..., ast.AST] | None = None,
        load_transform: Callable[..., ast.AST] | None = None,
        extra_params: list[str] | None = None,
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
        self.codegen_graphs = func.device_ir.build_codegen_graphs(config)
        self.host_statements: list[ast.AST] = []
        self.module_statements: list[ast.stmt] = []
        self.cute_wrapper_plans: list[dict[str, object]] = []
        self.cute_uses_matmul: bool = False
        self.statements_stack: list[list[ast.AST]] = [self.host_statements]
        self.on_device = False
        self.active_device_loops: dict[int, list[DeviceLoopOrGridState]] = (
            collections.defaultdict(list)
        )
        self.current_grid_state: DeviceGridState | None = None
        self.current_root_graph_info: GraphInfo | None = None
        self.max_thread_block_dims = [1, 1, 1]
        self.root_thread_block_dims = [1, 1, 1]
        self.referenced_thread_block_dims = [1, 1, 1]
        # CuTe only: synthetic per-thread axes allocated for free/unbound
        # ``hl.arange`` index dims that are not bound to any tile/reduction/grid
        # axis. Maps a stable per-arange key (length, start-repr, step-repr) to
        # the launch thread axis it occupies. ``thread_axis_sizes`` records the
        # extent of each allocated synthetic axis so the launch-dim recovery in
        # ``backend.py`` can grow the thread block to cover those lanes.
        self.cute_synthetic_arange_axes: dict[tuple[object, ...], int] = {}
        self.cute_synthetic_arange_axis_sizes: dict[int, int] = {}
        # CuTe only: stack of ``(if_node_id, branch_side)`` entries describing the
        # mutually-exclusive control-flow branch the current codegen is inside.
        # ``branch_side`` is 0 for the ``if`` body and 1 for the ``else`` body of
        # a given dynamic ``_if``. Two free ``hl.arange`` dims that live in
        # mutually-exclusive branches (their paths diverge at a common ``_if``)
        # may reuse the same synthetic thread axis since only one branch ever
        # runs per program instance.
        self._cute_branch_path: list[tuple[int, int]] = []
        # Records the branch path captured when each synthetic arange axis was
        # allocated, so a later arange in a mutually-exclusive branch can reuse it.
        self._cute_synthetic_arange_axis_branch_paths: dict[
            int, list[list[tuple[int, int]]]
        ] = {}
        # Records the branch path under which a strategy (e.g. a reduction) claimed
        # a thread axis. A free ``hl.arange`` in a branch mutually-exclusive with
        # every recorded user of a strategy axis may reuse that axis instead of
        # claiming a fresh one (which would silently widen the launch block and
        # turn the strategy's single-axis shared-memory reduction into a cross-axis
        # race). Mirrors ``_cute_synthetic_arange_axis_branch_paths``.
        self._cute_strategy_axis_branch_paths: dict[
            int, list[list[tuple[int, int]]]
        ] = {}
        # CuTe only: free ``hl.arange`` dims whose (joint) thread count would
        # exceed the 1024-thread budget are chunked onto a sequential lane loop
        # instead of claiming a fresh thread axis. Maps the per-arange ``key`` to
        # the resolved per-thread coordinate expression so a load and store over
        # the same arange share one lane loop.
        self.cute_synthetic_arange_lane_exprs: dict[tuple[object, ...], str] = {}
        self.next_else_block: list[ast.AST] | None = None
        self.store_transform = store_transform
        self.load_transform = load_transform
        self._statement_owner_fx_node: Node | None = None
        self._statements_by_owner_node_id: dict[
            int, list[tuple[list[ast.AST], ast.AST]]
        ] = {}

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

    def get_graph(self, graph_id: int) -> GraphInfo:
        return self.codegen_graphs[graph_id]

    def offset_var(self, block_idx: int) -> str:
        return self.active_device_loops[block_idx][-1].strategy.offset_var(block_idx)

    def index_var(self, block_idx: int) -> str:
        return self.active_device_loops[block_idx][-1].strategy.index_var(block_idx)

    def mask_var(self, block_idx: int) -> str | None:
        if loops := self.active_device_loops[block_idx]:
            return loops[-1].strategy.mask_var(block_idx)
        return None

    def _phase_checker(self, root_id: int) -> LoopDependencyChecker:
        phase_idx = self.host_function.device_ir.phase_for_root(root_id)
        return self.host_function.device_ir.phases[phase_idx].loop_dependency_checker

    def _compute_inter_loop_barriers(self) -> None:
        """Walk every codegen graph; for each pair of consecutive sibling
        ``_for_loop`` / ``_for_loop_step`` nodes, set ``needs_barrier_before``
        on the second loop's ``ForLoopGraphInfo`` when there is a global RAW
        dependency.

        TileIR shares Triton surface syntax but ``tl.debug_barrier()`` lowers
        to ``ttg.barrier`` which the TileIR pass pipeline does not legalize,
        so the analysis is a no-op there.
        """
        from ..language._tracing_ops import _for_loop
        from ..language._tracing_ops import _for_loop_step
        from .device_ir import ForLoopGraphInfo
        from .loop_dependency_checker import (
            needs_inter_loop_debug_barrier_for_global_raw,
        )

        env = CompileEnvironment.current()
        if env.codegen_name != "triton" or env.backend.name == "tileir":
            return

        for graph_info in self.codegen_graphs:
            # Pending writes accumulate across ALL prior sibling for-loops
            # since the last emitted barrier.  When a barrier is inserted
            # before a loop, it flushes all earlier writes, so the pending set
            # is reset and only writes from loops AFTER the barrier need to be
            # tracked for subsequent siblings.
            pending_global_writes: set[str] = set()
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
                need_barrier = needs_inter_loop_debug_barrier_for_global_raw(
                    pending_global_writes,
                    cur_info.host_loop_reads,
                    global_barrier_tensor_names=self._triton_global_barrier_tensor_names,
                )
                cur_info.needs_barrier_before = need_barrier
                if need_barrier:
                    # Barrier flushes everything written before it.
                    pending_global_writes = set()
                # Accumulate the current loop's writes for future siblings.
                pending_global_writes |= self._triton_global_barrier_tensor_names(
                    cur_info.host_loop_writes
                )

    def _triton_global_barrier_tensor_names(self, names: frozenset[str]) -> set[str]:
        """Names that may participate in cross-wavefront global (HBM) coherence.

        Triton-specific: the Pallas SMEM filter is intentionally omitted here
        because the only caller (``_compute_inter_loop_barriers``) gates on
        Triton codegen.  The ``triton_`` prefix and the assertion below encode
        that precondition so a future non-Triton caller fails loudly rather
        than silently mis-classifying SMEM-only tensors as needing a global
        barrier.
        """
        from .type_info import StackTensorType
        from .type_info import TensorType

        env = CompileEnvironment.current()
        assert env.codegen_name == "triton" and env.backend.name != "tileir", (
            "_triton_global_barrier_tensor_names called outside Triton codegen"
        )

        out: set[str] = set()
        scratch_names = {s.name for s in self.device_function._scratch_args}
        local_types = self.host_function.local_types
        for name in names:
            if name in scratch_names:
                continue
            if local_types is None:
                out.add(name)
                continue
            ti = local_types.get(name)
            if ti is None:
                out.add(name)
                continue
            if isinstance(ti, (TensorType, StackTensorType)):
                out.add(name)
        return out

    def add_statement(self, stmt: ast.AST | str | None) -> None:
        if stmt is None:
            return
        if isinstance(stmt, str):
            stmt = statement_from_string(stmt)
        self.statements_stack[-1].append(stmt)
        owner_node = self._statement_owner_fx_node
        if owner_node is not None:
            self._statements_by_owner_node_id.setdefault(id(owner_node), []).append(
                (self.statements_stack[-1], stmt)
            )
        self._record_statement_thread_references([stmt])
        self._record_tcgen05_owned_statement(stmt)

    def remove_statements_owned_by_nodes(self, nodes: tuple[Node, ...]) -> None:
        """Remove statements emitted earlier for exactly these FX nodes."""
        for node in nodes:
            entries = self._statements_by_owner_node_id.pop(id(node), ())
            for body, stmt in entries:
                with contextlib.suppress(ValueError):
                    body.remove(stmt)

    def _record_tcgen05_owned_statement(self, stmt: ast.AST) -> None:
        owner_node = self._statement_owner_fx_node
        if owner_node is None:
            return
        cute_state = self.device_function.cute_state
        # The generic add_statement hook stays inert unless CuTe tcgen05
        # lowering registered this exact FX node for ownership tracking.
        if not cute_state.is_collective_handled_load_or_dependency_node(owner_node):
            return
        current_statements = self.statements_stack[-1]
        for loop_state in reversed(self._active_loop_stack()):
            if isinstance(loop_state, DeviceLoopState):
                if current_statements is loop_state.inner_statements:
                    cute_state.register_tcgen05_kloop_owned_stmts(loop_state, [stmt])
                    return

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

    @contextlib.contextmanager
    def cute_branch_scope(self, if_node_id: int, branch_side: int) -> Iterator[None]:
        """Mark codegen as inside one branch of a dynamic ``_if`` (CuTe only).

        ``branch_side`` is 0 for the ``if`` body and 1 for the ``else`` body.
        Used so synthetic ``hl.arange`` axes allocated in mutually-exclusive
        branches can share a single thread axis.
        """
        self._cute_branch_path.append((if_node_id, branch_side))
        try:
            yield
        finally:
            self._cute_branch_path.pop()

    def _cute_branch_paths_mutually_exclusive(
        self,
        path_a: list[tuple[int, int]],
        path_b: list[tuple[int, int]],
    ) -> bool:
        """True when two branch paths can never both execute."""
        from .device_ir import DeviceIR

        return DeviceIR.branch_paths_mutually_exclusive(path_a, path_b)

    def _record_thread_axis_sizes(self, axis_sizes: dict[int, int]) -> None:
        for axis, size in axis_sizes.items():
            if 0 <= axis < 3:
                self.max_thread_block_dims[axis] = max(
                    self.max_thread_block_dims[axis], size
                )

    def _record_active_thread_axis_sizes(self) -> None:
        self._record_thread_axis_sizes(self._current_active_thread_axis_sizes())

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
        # Synthetic axes for free ``hl.arange`` index dims (CuTe only) live
        # outside the strategy loop states, so fold their extents in here too
        # — that way ``_record_statement_thread_references`` grows the launch
        # block to cover the lanes those arange dims address.
        for axis, size in self.cute_synthetic_arange_axis_sizes.items():
            axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        return axis_sizes

    def allocate_cute_synthetic_arange_coord(
        self, key: tuple[object, ...], size: int
    ) -> str | None:
        """Resolve a free ``hl.arange`` dim to a per-thread coordinate expr.

        Returns ``cute.arch.thread_idx()[axis]`` when the arange fits a fresh (or
        reusable) CUDA thread axis within the 1024-thread budget. When it would
        instead overflow the budget, the arange is chunked onto a sequential lane
        loop (``thread_idx()[axis] * 1 + lane * 1`` collapses to ``lane``, or
        ``thread_idx()[axis] + lane * nt`` when some thread lanes still fit) so
        the full extent stays addressable; the lane loop wraps the grid body.
        Returns ``None`` when no synthetic axis can be assigned (axis index >= 3).
        """
        lane_expr = self.cute_synthetic_arange_lane_exprs.get(key)
        if lane_expr is not None:
            return lane_expr
        if (
            key not in self.cute_synthetic_arange_axes
            and self._cute_arange_needs_lane_loop(key, size)
        ):
            return self._allocate_cute_synthetic_arange_lane_loop(key, size)
        axis = self.allocate_cute_synthetic_arange_axis(key, size)
        if axis >= 3:
            return None
        return f"cutlass.Int32(cute.arch.thread_idx()[{axis}])"

    def _cute_arange_proposed_total(self, axis: int, size: int) -> int:
        """Joint thread count if ``size`` were placed on a fresh ``axis``."""
        proposed_sizes = dict(self.cute_synthetic_arange_axis_sizes)
        proposed_sizes[axis] = size
        strategy_threads = 1
        for strat_axis, strat_size in self._strategy_thread_axis_sizes().items():
            if strat_axis not in proposed_sizes:
                strategy_threads *= strat_size
        total = strategy_threads
        for axis_size in proposed_sizes.values():
            total *= axis_size
        return total

    def _cute_arange_needs_lane_loop(self, key: tuple[object, ...], size: int) -> bool:
        # The lane loop is hosted by the grid body wrapper, so only chunk when a
        # grid state exists to carry it; otherwise keep the (raising) thread-axis
        # path rather than emitting an un-iterated lane variable.
        if self.current_grid_state is None:
            return False
        # A mutually-exclusive branch reuse never grows the budget, so prefer it.
        if self._mutually_exclusive_synthetic_axis() is not None:
            return False
        used_axes = set(self._strategy_thread_axes())
        used_axes.update(self.cute_synthetic_arange_axes.values())
        axis = 0
        while axis in used_axes:
            axis += 1
        from .cute.thread_budget import MAX_THREADS_PER_BLOCK

        if axis >= 3:
            return True
        return self._cute_arange_proposed_total(axis, size) > MAX_THREADS_PER_BLOCK

    def _allocate_cute_synthetic_arange_lane_loop(
        self, key: tuple[object, ...], size: int
    ) -> str:
        """Chunk a free ``hl.arange`` onto a sequential lane loop.

        Uses ``nt`` live thread lanes on a fresh axis (``nt`` is the largest
        power-of-2 that keeps the joint budget within 1024, possibly 1) and a
        ``ceil(size / nt)`` sequential lane loop covering the rest. The arange's
        per-thread coordinate is ``thread_idx()[axis] + lane * nt``.
        """
        from torch._inductor.runtime.runtime_utils import next_power_of_2

        from .cute.thread_budget import MAX_THREADS_PER_BLOCK

        used_axes = set(self._strategy_thread_axes())
        used_axes.update(self.cute_synthetic_arange_axes.values())
        axis = 0
        while axis in used_axes and axis < 3:
            axis += 1

        # Threads already committed (strategy axes + other synthetic axes).
        committed = 1
        for strat_size in self._strategy_thread_axis_sizes().values():
            committed *= strat_size
        for axis_size in self.cute_synthetic_arange_axis_sizes.values():
            committed *= axis_size
        budget = max(1, MAX_THREADS_PER_BLOCK // max(1, committed))
        nt = 1
        if axis < 3:
            nt = min(next_power_of_2(size), 1 << (budget.bit_length() - 1))
            nt = max(1, min(nt, size))
        lane_extent = (size + nt - 1) // nt

        lane_var = self.device_function.new_var(
            f"arange_lane_{len(self.cute_synthetic_arange_lane_exprs)}", dce=False
        )
        grid_state = self.current_grid_state
        if grid_state is not None:
            grid_state.add_lane_loop(-1, lane_var, lane_extent)
        if nt > 1 and axis < 3:
            self.cute_synthetic_arange_axes[key] = axis
            self.cute_synthetic_arange_axis_sizes[axis] = max(
                self.cute_synthetic_arange_axis_sizes.get(axis, 1), nt
            )
            self._record_synthetic_axis_branch_path(axis)
            self._record_active_thread_axis_sizes()
            coord = (
                f"(cutlass.Int32(cute.arch.thread_idx()[{axis}])"
                f" + cutlass.Int32({lane_var}) * {nt})"
            )
        else:
            coord = f"cutlass.Int32({lane_var})"
        self.cute_synthetic_arange_lane_exprs[key] = coord
        return coord

    def allocate_cute_synthetic_arange_axis(
        self, key: tuple[object, ...], size: int
    ) -> int:
        """Allocate (or reuse) a per-thread axis for a free ``hl.arange`` dim.

        A free ``hl.arange(n)`` used directly as a load/store index is not
        bound to any tile/reduction/grid block id, so it has no strategy
        thread axis. We map each distinct arange onto its own CUDA thread
        axis: thread ``thread_idx()[axis]`` holds element ``axis``. Arange dims
        that share the same ``key`` (same length/start/step) describe the same
        logical lane and reuse the same axis so a value loaded on a lane is
        stored back on that lane.

        The chosen axis follows any thread axes already claimed by real tile
        strategies (so an ``hl.arange`` mixed with an ``hl.tile`` index does
        not collide with the tile's axis). The size is recorded so the
        launch-dim recovery enlarges the thread block accordingly.
        """
        existing = self.cute_synthetic_arange_axes.get(key)
        if existing is not None:
            self.cute_synthetic_arange_axis_sizes[existing] = max(
                self.cute_synthetic_arange_axis_sizes.get(existing, 1), size
            )
            self._record_synthetic_axis_branch_path(existing)
            self._record_active_thread_axis_sizes()
            return existing
        # Reuse a synthetic axis from a mutually-exclusive control-flow branch:
        # if every arange already mapped onto some axis lives in a branch that
        # can never co-execute with the current one, the axes never need lanes
        # at the same time, so they may share one thread axis (size = max). This
        # keeps the joint thread budget bounded for branch-by-grid kernels whose
        # branches each use a distinct free ``hl.arange``.
        shared = self._mutually_exclusive_synthetic_axis()
        if shared is not None:
            self.cute_synthetic_arange_axes[key] = shared
            self.cute_synthetic_arange_axis_sizes[shared] = max(
                self.cute_synthetic_arange_axis_sizes.get(shared, 1), size
            )
            self._record_synthetic_axis_branch_path(shared)
            self._record_active_thread_axis_sizes()
            return shared
        used_axes = set(self._strategy_thread_axes())
        used_axes.update(self.cute_synthetic_arange_axes.values())
        axis = 0
        while axis in used_axes:
            axis += 1
        from .cute.thread_budget import check_thread_limit

        # Validate the joint thread count once the new axis is added.
        proposed_sizes = dict(self.cute_synthetic_arange_axis_sizes)
        proposed_sizes[axis] = size
        strategy_threads = 1
        for strat_axis, strat_size in self._strategy_thread_axis_sizes().items():
            if strat_axis not in proposed_sizes:
                strategy_threads *= strat_size
        total = strategy_threads
        for axis_size in proposed_sizes.values():
            total *= axis_size
        check_thread_limit(total, context=f"free hl.arange axis size={size}")
        self.cute_synthetic_arange_axes[key] = axis
        self.cute_synthetic_arange_axis_sizes[axis] = size
        self._record_synthetic_axis_branch_path(axis)
        self._record_active_thread_axis_sizes()
        return axis

    def _record_synthetic_axis_branch_path(self, axis: int) -> None:
        """Remember the current branch path for an arange mapped onto ``axis``."""
        paths = self._cute_synthetic_arange_axis_branch_paths.setdefault(axis, [])
        current = list(self._cute_branch_path)
        if current not in paths:
            paths.append(current)

    def record_cute_strategy_axis_branch_path(self, axis: int) -> None:
        """Remember the current branch path for a strategy (e.g. reduction) axis.

        Called from reduction codegen while the dynamic ``_if`` branch scope is
        live, so a free ``hl.arange`` in a mutually-exclusive sibling branch can
        reuse this axis rather than claiming a fresh one. Recording only happens
        inside a branch (an unbranched strategy axis is always co-live and must
        never be reused).
        """
        if not self._cute_branch_path:
            return
        paths = self._cute_strategy_axis_branch_paths.setdefault(axis, [])
        current = list(self._cute_branch_path)
        if current not in paths:
            paths.append(current)

    def _mutually_exclusive_synthetic_axis(self) -> int | None:
        """Find a thread axis whose every recorded use is in a branch that can
        never co-execute with the current branch path, so it can be safely reused.

        Considers both synthetic ``hl.arange`` axes and strategy (reduction) axes:
        a free arange in one grid branch may reuse the axis a reduction claimed in
        a sibling branch when the two can never run together. An axis recorded in
        BOTH maps must be mutually exclusive across all of its recorded paths.
        """
        if not self._cute_branch_path:
            return None
        current = list(self._cute_branch_path)
        candidate_axes = (
            self._cute_synthetic_arange_axis_branch_paths.keys()
            | self._cute_strategy_axis_branch_paths.keys()
        )
        for axis in sorted(candidate_axes):
            paths = [
                *self._cute_synthetic_arange_axis_branch_paths.get(axis, []),
                *self._cute_strategy_axis_branch_paths.get(axis, []),
            ]
            if paths and all(
                self._cute_branch_paths_mutually_exclusive(current, path)
                for path in paths
            ):
                return axis
        return None

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

    def _strategy_thread_axes(self) -> set[int]:
        axes = set(self._strategy_thread_axis_sizes())
        axes.update(self._all_strategy_reserved_axes())
        # A strategy can structurally occupy a thread axis even when its block
        # size is 1 (e.g. an ``hl.grid`` whose offset is
        # ``pid * BLOCK + thread_idx[0]`` with ``BLOCK == 1``). Such an axis is
        # not recorded in ``thread_axis_sizes`` (which only keeps sizes > 1) but
        # it is referenced in the already-emitted setup statements, so scan
        # those to avoid handing a synthetic arange an axis the grid already
        # uses (which would mis-filter most lanes via the grid's bounds mask).
        statement_groups: list[list[ast.AST]] = list(self.statements_stack)
        grid_state = self.current_grid_state
        if grid_state is not None:
            statement_groups.extend(
                (grid_state.outer_prefix, grid_state.lane_setup_statements)
            )
        for loops in self.active_device_loops.values():
            for loop_state in loops:
                outer_prefix = getattr(loop_state, "outer_prefix", None)
                if isinstance(outer_prefix, list):
                    statement_groups.append(outer_prefix)
        for statements in statement_groups:
            for stmt in statements:
                axes.update(
                    int(axis_text)
                    for axis_text in re.findall(
                        r"cute\.arch\.thread_idx\(\)\[(\d+)\]",
                        ast.unparse(stmt),
                    )
                )
        return axes

    def _record_statement_thread_references(
        self,
        statements: list[ast.AST],
        axis_sizes: dict[int, int] | None = None,
    ) -> None:
        if axis_sizes is None:
            axis_sizes = self._current_active_thread_axis_sizes()
        for stmt in statements:
            text = ast.unparse(stmt)
            for axis_text in re.findall(
                r"cute\.arch\.thread_idx\(\)\[(\d+)\]",
                text,
            ):
                axis = int(axis_text)
                if 0 <= axis < 3:
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
            finally:
                for idx in device_loop.block_ids:
                    self.active_device_loops[idx].pop()
        if needs_barrier_before:
            self.add_statement(statement_from_string("tl.debug_barrier()"))
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
            for axis, size in device_grid.thread_axis_sizes.items():
                if 0 <= axis < 3:
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
            # Loop dependency checks were already run during lowering; phase checker kept for symmetry/debug.
            self._phase_checker(node._root_id)

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
                if node._root_id < len(self.host_function.device_ir.root_ids) - 1:
                    body = []
                else:
                    # This is the last top level for, dont emit more if statements
                    assert self.next_else_block is not None
                    body = self.next_else_block
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
                        for arg_node in iter_node.args:
                            assert not isinstance(arg_node, ast.Starred)
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
                        try:
                            codegen_fn = fn._codegen[env.codegen_name]
                        except KeyError:
                            raise exc.BackendImplementationMissing(
                                env.backend_name,
                                f"codegen for API function {fn.__qualname__}",
                            ) from None
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
                    root = root_graph_info.graph
                    grid_state = self.current_grid_state
                    if isinstance(grid_state, DeviceGridState):
                        # Codegen the body first so synthetic free-``hl.arange``
                        # lane loops registered *during* body lowering (CuTe
                        # over-budget chunking) are visible to the wrap below.
                        wrapped_body: list[ast.AST] = []
                        with self.set_statements(wrapped_body):
                            codegen_call_with_graph(self, root, [])
                        if grid_state.has_lane_loops():
                            self.statements_stack[-1].extend(grid_state.outer_prefix)
                            if self.device_function.cute_state.consume_root_lane_loop_suppression():
                                self.statements_stack[-1].extend(wrapped_body)
                            else:
                                self.statements_stack[-1].extend(
                                    grid_state.wrap_body(wrapped_body)
                                )
                            self.statements_stack[-1].extend(grid_state.outer_suffix)
                        else:
                            self.statements_stack[-1].extend(wrapped_body)
                    else:
                        codegen_call_with_graph(self, root, [])
                finally:
                    self.current_root_graph_info = previous_root_graph_info

                # Flush deferred RDIM definitions now that block sizes are determined
                # This ensures block size and rdim vars are defined in the correct order
                self.device_function.flush_deferred_rdim_defs(self)

                if isinstance(self.device_function.pid, ForEachProgramID):
                    self.device_function.pid.case_phases.append(
                        self.host_function.device_ir.phase_for_root(node._root_id)
                    )

                # If we are in a multi top level loop, for all loops except for the last one
                # emit ifthenelse blocks
                if node._root_id < len(self.host_function.device_ir.root_ids) - 1:
                    block = (
                        self.device_function.body
                        if self.next_else_block is None
                        else self.next_else_block
                    )
                    self.next_else_block = []
                    block.append(
                        create(
                            ast.If,
                            # pyrefly: ignore [missing-attribute]
                            test=self.device_function.pid.codegen_test(state),
                            body=body,
                            orelse=self.next_else_block,
                        )
                    )
            if node._root_id == len(self.host_function.device_ir.root_ids) - 1:
                if self.device_function.pid is not None:
                    persistent_body = self.device_function.pid.setup_persistent_kernel(
                        self.device_function
                    )
                    if persistent_body is not None:
                        # pyrefly: ignore [bad-assignment]
                        self.device_function.body = persistent_body
                    else:
                        # The persistent path pulls tcgen05 post-loop cleanup to
                        # the end of the body; the non-persistent (flat-grid)
                        # path must do the same so a multi-store fan-out's
                        # one-shot teardown runs after every store reads the
                        # accumulator. No-op when there are no post-loop marks.
                        self.device_function.body = self.device_function.cute_state.move_tcgen05_post_loop_stmts_to_end(
                            list(self.device_function.body)
                        )
                # Fused tcgen05 flash-attention (HELION_CUTE_FLASH): the
                # detector set ``attention_flash_block_ids``; replace the
                # FX-derived scalar body with the dedicated tensor-core flash
                # kernel that mirrors ``.notes/spikes/fa_tcgen05_spike.py``.
                if (
                    self.device_function.cute_state.attention_flash_block_ids
                    is not None
                ):
                    from .cute.cute_flash import codegen_attention_flash

                    if not codegen_attention_flash(self):
                        # Codegen declined a config the detector could not fully
                        # vet before the device-function arguments were populated
                        # (e.g. non-contiguous operands). Clear the flash state so
                        # the launch override does not force block=(N,1,1) onto
                        # the scalar fallback body that stays in df.body.
                        self.device_function.cute_state.attention_flash_block_ids = None
                # Mark extra params as placeholder args — they appear only in
                # placeholder strings, not in the AST body, so DCE would
                # otherwise remove them.
                for param in self._extra_params:
                    self.device_function.placeholder_args.add(param)
                if CompileEnvironment.current().backend.name == "cute":
                    from .tile_strategy import hoist_lane_invariant_chunk_recurrence
                    from .tile_strategy import (
                        interchange_lane_outside_serial_reductions,
                    )
                    from .tile_strategy import restore_unprocessed_lane_reduce_markers
                    from .tile_strategy import split_lane_loop_reductions

                    # First interchange any ``for LANE: ... for MB: ...`` nest
                    # whose inner serial loop carries lane-reduce markers into a
                    # lane-outside-mb accumulator nest plus a lane-inside-mb
                    # reduction nest; then split the (now inner) lane loops into
                    # the two-pass accumulate/finalize/consume structure.
                    self.device_function.body = (
                        interchange_lane_outside_serial_reductions(
                            list(self.device_function.body)
                        )
                    )
                    self.device_function.body = split_lane_loop_reductions(
                        list(self.device_function.body)
                    )
                    # Safety net: revert any lane-reduce marker that neither pass
                    # rewrote so no ``_helion_lane_reduce`` call leaks into the
                    # emitted kernel.
                    self.device_function.body = restore_unprocessed_lane_reduce_markers(
                        list(self.device_function.body)
                    )
                    # Restructure chunked-recurrence ``for chunk: for lane:``
                    # nests whose matmul ``dot_acc`` running-sum needs the
                    # lane-invariant rescale / chunk-entry stores / final combine
                    # hoisted to run once per chunk (gdn_fwd_h).
                    self.device_function.body = hoist_lane_invariant_chunk_recurrence(
                        list(self.device_function.body)
                    )
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
                return expr_from_string(
                    repr(self.host_function.constexpr_args[origin.name])
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
            try:
                codegen_fn = api._codegen[env.codegen_name]
            except KeyError:
                raise exc.BackendImplementationMissing(
                    env.backend_name,
                    f"codegen for API function {api.__qualname__}",
                ) from None
            ast_args = []
            ast_kwargs = {}
            proxy_args = []
            proxy_kwargs = {}
            for arg in node.args:
                assert not isinstance(arg, ast.Starred)
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
        for stmt in self.host_statements:
            if (
                isinstance(stmt, ast.Assign)
                and definitely_does_not_have_side_effects(stmt.value)
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


def _maybe_emit_compact_worklist_builder(codegen: GenerateAST, config: Config) -> None:
    """Emit the module-level jnp ``_build_worklist`` for a compact-worklist kernel."""
    if config.get("pallas_loop_type") != "compact_worklist":
        return
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
) -> ast.Module:
    with func:
        env = CompileEnvironment.current()
        env.cute_resolved_wrapper_plans = []
        if len(func.device_ir.phases) > 1:
            if not str(config.pid_type).startswith("persistent"):
                raise exc.BarrierRequiresPersistent(config.pid_type)
        codegen = GenerateAST(
            func,
            config,
            store_transform=store_transform,
            load_transform=load_transform,
            extra_params=extra_params,
        )
        with codegen.device_function:
            CompileEnvironment.current().backend.pre_codegen(
                graphs=codegen.codegen_graphs,
                config=config,
                tile_strategy=codegen.device_function.tile_strategy,
            )

            # Emit the worklist builder + record its offset params BEFORE the host
            # body is visited (the launcher call -- which reads the offset params
            # via build_launcher_args -- is generated during that visit).
            _maybe_emit_compact_worklist_builder(codegen, config)

            for stmt in func.body:
                codegen.add_statement(codegen.visit(stmt))
            codegen.device_function.cute_state.finalize_tcgen05_pure_lifecycle_stores()
            kernel_def = codegen.device_function.codegen_function_def()
            codegen.host_dead_code_elimination()

            # Retarget output-only tensor allocations to ``device='meta'`` so
            # the factory call produces a zero-storage metadata-only tensor
            # instead of allocating real HBM. The launcher reassigns the
            # variable to the real result tensor.
            output_only_names = getattr(
                CompileEnvironment.current().backend, "_output_only_names", []
            )
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
            final_host_statements = rng_statements + codegen.host_statements
            small_biased_wrapper_only = codegen.cute_wrapper_plans and all(
                plan.get("kind") == "helion_small_biased_attention"
                for plan in codegen.cute_wrapper_plans
            )
            if (
                codegen.cute_uses_matmul or codegen.cute_wrapper_plans
            ) and not small_biased_wrapper_only:
                final_host_statements = [
                    statement_from_string(
                        f"{codegen.device_function.name}._helion_cute_disable_bake_tensor_shapes = True"
                    ),
                    *final_host_statements,
                ]
            launcher_arg_positions: dict[str, int] | None = None
            host_arg_positions = {
                arg.arg: idx for idx, arg in enumerate(func.args.args)
            }

            def resolve_cute_plan_arg_positions(
                plans: list[dict[str, object]],
            ) -> list[dict[str, object]]:
                nonlocal launcher_arg_positions
                if launcher_arg_positions is None:
                    launcher_arg_positions = {}
                    for idx, arg in enumerate(
                        [
                            arg
                            for arg in codegen.device_function.sorted_args()
                            if not (
                                isinstance(arg, ConstExprArg)
                                and arg.host_str() != arg.name
                            )
                        ]
                    ):
                        launcher_arg_positions[arg.name] = idx
                resolved_plans: list[dict[str, object]] = []
                for plan in plans:
                    resolved = dict(plan)
                    # The ``key[:-5] + "_idx"`` substring rewrite turns each
                    # tensor name into a positional index resolved against the
                    # device function's sorted-arg ordering so the runtime
                    # launcher can identify the tensor args.
                    for key in (
                        "lhs_name",
                        "rhs_name",
                        "c_name",
                        "d_name",
                        "q_name",
                        "k_name",
                        "v_name",
                        "o_name",
                        "lse_name",
                        "bias_name",
                        "alibi_name",
                        "document_name",
                        "layout_name",
                        "n_sizes_name",
                        "k_sizes_name",
                        "direct_pointers_name",
                        "direct_strides_name",
                    ):
                        if key in resolved:
                            arg_name = str(resolved.pop(key))
                            resolved[key[:-5] + "_idx"] = launcher_arg_positions[
                                arg_name
                            ]
                            if arg_name in host_arg_positions:
                                resolved[key[:-5] + "_bind_idx"] = host_arg_positions[
                                    arg_name
                                ]
                    resolved_plans.append(resolved)
                return resolved_plans

            post_kernel_metadata_statements: list[ast.AST] = []
            resolved_wrapper_plans: list[dict[str, object]] = []
            if codegen.cute_wrapper_plans:
                resolved_wrapper_plans = resolve_cute_plan_arg_positions(
                    codegen.cute_wrapper_plans
                )
                env.cute_resolved_wrapper_plans = resolved_wrapper_plans
                post_kernel_metadata_statements.append(
                    statement_from_string(
                        f"{codegen.device_function.name}._helion_cute_wrapper_plans = helion.runtime._freeze_cute_wrapper_plans({resolved_wrapper_plans!r})"
                    )
                )
            if codegen.device_function.cute_state.cluster_shape is not None:
                post_kernel_metadata_statements.append(
                    statement_from_string(
                        f"{codegen.device_function.name}._helion_cute_cluster_shape = {codegen.device_function.cute_state.cluster_shape!r}"
                    )
                )
            # Assert sourceless prologue params were actually removed by DCE
            if codegen.device_function.sourceless_prologue_params:
                remaining = codegen.device_function.sourceless_prologue_params & {
                    arg.name for arg in codegen.device_function.arguments
                }
                assert not remaining, (
                    f"sourceless prologue params not removed by DCE: {remaining}"
                )

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
            # break circular reference for better GC
            del codegen.device_function.codegen
            return result
