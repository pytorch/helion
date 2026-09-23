from __future__ import annotations

import ast
import collections
import contextlib
import dataclasses
import re
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

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
from .cute.direct_affine_plan import DIRECT_AFFINE_ORDINARY_SCHEDULE
from .device_function import ConstExprArg
from .device_function import DeviceFunction
from .device_function import TensorArg
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

    from .cute.bounded_cache_codegen import BoundedCacheRequest
    from .device_ir import GraphInfo
    from .host_function import HostFunction
    from .pallas.compact_worklist import ResidentPrepHoist
    from .pallas.dma import DmaResources
    from .tile_strategy import DeviceLoopOrGridState
    from .type_info import TensorType


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
        config = CompileEnvironment.current().backend.codegen_config(config)
        self.codegen_graphs = (
            func.device_ir.build_codegen_graphs(config)
            if codegen_graphs is None
            else codegen_graphs
        )
        self.host_statements: list[ast.AST] = []
        self.module_statements: list[ast.stmt] = []
        self.cute_wrapper_plans: list[dict[str, object]] = []
        self._cute_uses_matmul: bool = False
        self._cute_matmul_declaration_count = 0
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
        self._track_statement_owners = (
            CompileEnvironment.current().backend.name == "cute"
        )

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

    @property
    def cute_uses_matmul(self) -> bool:
        return self._cute_uses_matmul

    @cute_uses_matmul.setter
    def cute_uses_matmul(self, value: bool) -> None:
        self._cute_uses_matmul = value
        # Count declarations, not False-to-True transitions: an unrelated
        # matmul path must not inherit a register chain's shape-bake exemption.
        if value:
            self._cute_matmul_declaration_count += 1

    def _cute_can_bake_tensor_shapes(self) -> bool:
        if not self.cute_uses_matmul:
            return True
        if self.cute_wrapper_plans:
            return all(
                plan.get("kind")
                in {
                    "helion_small_biased_attention",
                    "helion_flash",
                    "chunk_prepare_tma",
                    "chunk_recurrence_sm100",
                    "chunk_recurrence_warp_dv4",
                    "helion_flash_bwd",
                    "gathered_mma_tma",
                    "block_scaled_mma",
                }
                for plan in self.cute_wrapper_plans
            )
        state = self.device_function.cute_state
        return (
            state.collective_register_chain_lowered
            and self._cute_matmul_declaration_count >= 2
            and self._cute_matmul_declaration_count == len(state.collective_mma_sites)
        ) or self._cute_has_complete_static_collectives()

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

    def index_var(self, block_idx: int) -> str:
        return self.active_device_loops[block_idx][-1].strategy.index_var(block_idx)

    def mask_var(self, block_idx: int) -> str | None:
        if loops := self.active_device_loops[block_idx]:
            return loops[-1].strategy.mask_var(block_idx)
        return None

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

    def _compute_intra_loop_barriers(self) -> None:
        """Mark loads that read a tensor an earlier store in the same body wrote,
        so the Triton ``load`` codegen emits a ``tl.debug_barrier()`` before them
        (the CuTe ``load`` codegen emits ``cute.arch.sync_threads()``).

        TileIR is excluded for the same reason as ``_compute_inter_loop_barriers``:
        ``tl.debug_barrier()`` lowers to ``ttg.barrier`` which the TileIR pass
        pipeline does not legalize.
        """
        from .loop_dependency_checker import mark_intra_loop_raw_barriers

        env = CompileEnvironment.current()
        if env.backend.name != "cute" and (
            env.codegen_name != "triton" or env.backend.name == "tileir"
        ):
            return
        mark_intra_loop_raw_barriers(
            self.codegen_graphs,
            self.host_function.device_ir.root_ids,
            # A CuTe SIMT branch condition can vary per thread, and the
            # convergent ``sync_threads`` barrier deadlocks in a divergent
            # branch.  Triton branches are uniform per program, so the
            # barrier is always placeable there.
            mark_in_divergent_control_flow=env.backend.name != "cute",
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

    def _clear_attention_flash_state(self) -> None:
        cute_state = self.device_function.cute_state
        cute_state.attention_flash_block_ids = None
        cute_state.attention_flash_score_plan = None
        cute_state.attention_flash_threads = 128

    def _try_codegen_attention_flash_root(self) -> bool:
        cute_state = self.device_function.cute_state
        if cute_state.attention_flash_bwd_block_ids is not None:
            from .cute.cute_flash_bwd import codegen_attention_flash_bwd

            if codegen_attention_flash_bwd(self):
                return True
            cute_state.attention_flash_bwd_block_ids = None
            cute_state.attention_flash_bwd_match = None
            raise exc.BackendUnsupported(
                "cute", "flash attention backward failed late validation"
            )
        if cute_state.attention_flash_block_ids is None:
            return False

        from .cute.cute_flash import codegen_attention_flash

        if codegen_attention_flash(self):
            return True
        self._clear_attention_flash_state()
        raise exc.BackendUnsupported("cute", "flash attention failed late validation")

    def _try_codegen_single_token_rank1_root(self) -> bool:
        plan = self.device_function.cute_state.single_token_rank1_plan
        if plan is None:
            return False
        from .cute.single_token_rank1_recurrence import (
            codegen_single_token_rank1_recurrence,
        )

        if codegen_single_token_rank1_recurrence(self):
            return True
        self.device_function.cute_state.single_token_rank1_plan = None
        raise exc.BackendUnsupported(
            "cute", "single-token rank-1 recurrence failed late validation"
        )

    def _try_codegen_split_single_token_rank1_root(self) -> bool:
        plan = self.device_function.cute_state.split_single_token_rank1_plan
        if plan is None:
            return False
        from .cute.split_single_token_rank1_recurrence import (
            codegen_split_single_token_rank1_recurrence,
        )

        if codegen_split_single_token_rank1_recurrence(self, plan):
            return True
        self.device_function.cute_state.split_single_token_rank1_plan = None
        raise exc.BackendUnsupported(
            "cute", "split single-token rank-1 recurrence failed late validation"
        )

    def _try_codegen_fixed_token_rank1_root(self) -> bool:
        plan = self.device_function.cute_state.fixed_token_rank1_plan
        if plan is None:
            return False
        from .cute.fixed_token_rank1_recurrence import (
            codegen_fixed_token_rank1_recurrence,
        )

        if codegen_fixed_token_rank1_recurrence(self):
            return True
        self.device_function.cute_state.fixed_token_rank1_plan = None
        raise exc.BackendUnsupported(
            "cute", "fixed-token rank-1 recurrence failed late validation"
        )

    def _try_codegen_chunk_prepare_root(self) -> bool:
        plan = self.device_function.cute_state.chunk_prepare_plan
        if plan is None:
            return False
        from .cute.chunk_prepare import codegen_chunk_prepare

        if codegen_chunk_prepare(self):
            return True
        self.device_function.cute_state.chunk_prepare_plan = None
        raise exc.BackendUnsupported("cute", "chunk prepare failed late validation")

    def _try_codegen_block_scaled_root(self) -> bool:
        if self.device_function.cute_state.block_scaled_plan is None:
            return False
        from .cute.block_scaled_mma import codegen_block_scaled

        if codegen_block_scaled(self):
            return True
        raise exc.BackendUnsupported("cute", "block scaling failed late validation")

    def _try_codegen_chunk_recurrence_root(self) -> bool:
        plan = self.device_function.cute_state.chunk_recurrence_plan
        if plan is None:
            return False
        from .cute.chunk_recurrence import codegen_chunk_recurrence

        if codegen_chunk_recurrence(self):
            return True
        self.device_function.cute_state.chunk_recurrence_plan = None
        raise exc.BackendUnsupported("cute", "chunk recurrence failed late validation")

    def _try_lower_direct_affine_root(
        self,
        grid: DeviceGridState,
        root_body: list[ast.AST],
    ) -> bool:
        """Replace one proven affine-scan root after ordinary CuTe lowering."""

        if (
            self.device_function.config.cute_affine_scan_schedule
            == DIRECT_AFFINE_ORDINARY_SCHEDULE
        ):
            return False
        root = self.current_root_graph_info
        if root is None:
            raise exc.BackendUnsupported(
                "cute", "direct affine scan has no active root"
            )
        candidates = tuple(
            candidate
            for candidate in self.device_function.cute_state.direct_affine_candidates
            if candidate.graph_id == root.graph_id
        )
        if len(candidates) != 1:
            raise exc.BackendUnsupported(
                "cute", "direct affine scan requires one candidate in its root"
            )

        from .cute.direct_affine_lowering import resolve_direct_affine_lowering
        from .cute.direct_affine_replay import replace_direct_affine_replay

        resolved = resolve_direct_affine_lowering(
            candidates[0],
            root,
            self,
            grid,
            root_body,
            name_prefix=self.device_function.new_var("_helion_direct_affine"),
        )
        if resolved is None:
            raise exc.BackendUnsupported(
                "cute", "direct affine scan failed late validation"
            )

        existing_module_statements = {
            ast.dump(statement, include_attributes=False)
            for statement in self.module_statements
        }
        module_additions: list[ast.stmt] = []
        for statement in resolved.emission.module_statements:
            key = ast.dump(statement, include_attributes=False)
            if key not in existing_module_statements:
                module_additions.append(statement)
                existing_module_statements.add(key)

        body_snapshot = tuple(root_body)
        owner_snapshot = {
            owner: list(entries)
            for owner, entries in self._statements_by_owner_node_id.items()
        }
        thread_dims_snapshot = tuple(self.referenced_thread_block_dims)
        module_snapshot = tuple(self.module_statements)
        previous_plan = self.device_function.cute_state.direct_affine_plan
        previous_has_barrier = self.device_function.has_barrier
        try:
            if not replace_direct_affine_replay(
                resolved.replay,
                root,
                self,
                root_body,
                resolved.emission.replacement_statements,
            ):
                raise exc.BackendUnsupported(
                    "cute", "direct affine scan failed late validation"
                )
            self.module_statements.extend(module_additions)
            self.device_function.cute_state.direct_affine_plan = resolved.plan
            self.device_function.has_barrier = True
        except Exception:
            root_body[:] = body_snapshot
            self._statements_by_owner_node_id.clear()
            self._statements_by_owner_node_id.update(owner_snapshot)
            self.referenced_thread_block_dims[:] = thread_dims_snapshot
            self.module_statements[:] = module_snapshot
            self.device_function.cute_state.direct_affine_plan = previous_plan
            self.device_function.has_barrier = previous_has_barrier
            raise
        return True

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
        if body is self.statements_stack[-1]:
            self._record_tcgen05_owned_statement(stmt)

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

    def _cute_has_complete_static_collectives(self) -> bool:
        state = self.device_function.cute_state
        return (
            cast(
                "bool",
                self.device_function.config.get(
                    "cute_collective_static_layouts", False
                ),
            )
            and not self.cute_wrapper_plans
            and state.collective_mma_static_layouts
            and self._cute_matmul_declaration_count > 0
            and self._cute_matmul_declaration_count == len(state.collective_mma_sites)
        )

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
                    if isinstance(self.current_grid_state, DeviceGridState):
                        self.current_grid_state.hoist_parent_statements = (
                            self.statements_stack[-1]
                        )
                    root = root_graph_info.graph
                    if (
                        not self._try_codegen_block_scaled_root()
                        and not self._try_codegen_chunk_prepare_root()
                        and not self._try_codegen_chunk_recurrence_root()
                        and not self._try_codegen_single_token_rank1_root()
                        and not self._try_codegen_split_single_token_rank1_root()
                        and not self._try_codegen_fixed_token_rank1_root()
                        and not self._try_codegen_attention_flash_root()
                    ):
                        grid_state = self.current_grid_state
                        if isinstance(grid_state, DeviceGridState):
                            # Codegen the body first so synthetic free-``hl.arange``
                            # lane loops registered *during* body lowering (CuTe
                            # over-budget chunking) are visible to the wrap below.
                            wrapped_body: list[ast.AST] = []
                            with self.set_statements(wrapped_body):
                                codegen_call_with_graph(self, root, [])
                            if self._try_lower_direct_affine_root(
                                grid_state, wrapped_body
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
                                if self.device_function.cute_state.consume_root_lane_loop_suppression():
                                    self.statements_stack[-1].extend(wrapped_body)
                                else:
                                    self.statements_stack[-1].extend(
                                        grid_state.wrap_body(wrapped_body)
                                    )
                                self.statements_stack[-1].extend(
                                    grid_state.outer_suffix
                                )
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
                    from .tile_strategy import validate_lane_reduce_owners

                    # First interchange any ``for LANE: ... for MB: ...`` nest
                    # whose inner serial loop carries lane-reduce markers into a
                    # lane-outside-mb accumulator nest plus a lane-inside-mb
                    # reduction nest; then split the (now inner) lane loops into
                    # the two-pass accumulate/finalize/consume structure.
                    proven_disjoint_pairs = (
                        self.device_function.proven_disjoint_tensor_pairs()
                    )
                    if self.device_function.cute_state.resident_sequence_regions:
                        from .cute.resident_sequence import (
                            materialize_resident_sequences,
                        )

                        fn = self.device_function
                        env = CompileEnvironment.current()
                        fn.body = materialize_resident_sequences(
                            list(fn.body),
                            regions=fn.cute_state.resident_sequence_regions,
                            tensor_dtypes={
                                argument.name: env.backend.dtype_str(
                                    argument.fake_value.dtype
                                )
                                for argument in fn.arguments
                                if isinstance(argument, TensorArg)
                            },
                            rename_groups={
                                name: aliases[0]
                                for name, aliases in fn._variable_renames.items()
                            },
                            disjoint_pairs=proven_disjoint_pairs,
                            boundary_names={argument.name for argument in fn.arguments}
                            | set(self._extra_params),
                            resident=fn.config.config.get("cute_reduction_sequence")
                            == "resident",
                            new_var=fn.new_var,
                        )
                    if self.device_function.cute_state.resident_reduction_layouts:
                        from .cute.resident_reductions import (
                            materialize_resident_reductions,
                        )
                        from .cute.resident_reductions import (
                            proven_resident_tensor_alignments,
                        )
                        from .cute.resident_reductions import (
                            proven_resident_tensor_strides,
                        )
                        from .reduction_strategy import _cute_shared_memory_budget_bytes

                        fn = self.device_function
                        env = CompileEnvironment.current()
                        pipelined = (
                            fn.config.config.get("cute_reduction_schedule")
                            == "pipelined"
                        )
                        constexpr_values = {
                            name: size
                            for block_ids, name in fn.block_size_var_cache.items()
                            if len(block_ids) == 1
                            and isinstance(
                                size := fn.resolved_block_size(block_ids[0]), int
                            )
                        }
                        fn.body = materialize_resident_reductions(
                            list(fn.body),
                            layouts=fn.cute_state.resident_reduction_layouts,
                            tensor_dtypes={
                                argument.name: env.backend.dtype_str(
                                    argument.fake_value.dtype
                                )
                                for argument in fn.arguments
                                if isinstance(argument, TensorArg)
                            },
                            tensor_strides=proven_resident_tensor_strides(fn),
                            tensor_alignments=proven_resident_tensor_alignments(fn),
                            group_rows=cast(
                                "int",
                                fn.config.config.get("cute_reduction_group_rows", 1),
                            ),
                            pipelined=pipelined,
                            pipeline_depth=fn.config.cute_reduction_pipeline_depth,
                            row_schedule=cast(
                                "str",
                                fn.config.get("cute_reduction_row_schedule", "batched"),
                            ),
                            pack_output=cast(
                                "bool",
                                fn.config.get("cute_reduction_pack_output", False),
                            ),
                            local_tree=cast(
                                "bool",
                                fn.config.get("cute_reduction_local_tree", False),
                            ),
                            shared_memory_budget=_cute_shared_memory_budget_bytes()
                            if pipelined
                            else 0,
                            disjoint_pairs=proven_disjoint_pairs,
                            rename_groups={
                                name: aliases[0]
                                for name, aliases in fn._variable_renames.items()
                            },
                            new_var=fn.new_var,
                            constexpr_values=constexpr_values,
                            uniform_names={argument.name for argument in fn.arguments}
                            | set(self._extra_params),
                            require_proof=True,
                        )
                    from .cute.nested_lane_reductions import (
                        normalize_nested_lane_reductions,
                    )
                    from .cute.nested_lane_reductions import resolve_pruned_lane_owners

                    resolve_pruned_lane_owners(
                        list(self.device_function.body),
                        self.device_function.cute_state.reshape_lane_fallbacks,
                    )
                    self.device_function.body = normalize_nested_lane_reductions(
                        list(self.device_function.body),
                        uniform_names={
                            *(
                                argument.name
                                for argument in self.device_function.arguments
                            ),
                            *self._extra_params,
                        },
                        proven_disjoint_tensor_pairs=proven_disjoint_pairs,
                        proven_tensor_stride_values=self.device_function.proven_tensor_stride_values(),
                        rename_groups={
                            name: aliases[0]
                            for name, aliases in self.device_function._variable_renames.items()
                        },
                    )
                    validate_lane_reduce_owners(list(self.device_function.body))
                    self.device_function.body = interchange_lane_outside_serial_reductions(
                        list(self.device_function.body),
                        proven_disjoint_tensor_pairs=proven_disjoint_pairs,
                        protected_names={
                            alias
                            for name, aliases in self.device_function._variable_renames.items()
                            if any(alias != name for alias in aliases)
                            for alias in (name, *aliases)
                        },
                    )
                    validate_lane_reduce_owners(list(self.device_function.body))
                    self.device_function.body = split_lane_loop_reductions(
                        list(self.device_function.body),
                        uniform_names={
                            *(
                                argument.name
                                for argument in self.device_function.arguments
                            ),
                            *self._extra_params,
                        },
                        proven_disjoint_tensor_pairs=proven_disjoint_pairs,
                        proven_tensor_stride_values=(
                            self.device_function.proven_tensor_stride_values()
                        ),
                        rename_groups={
                            name: aliases[0]
                            for name, aliases in self.device_function._variable_renames.items()
                        },
                        running_sums=self.device_function.cute_matmul_running_sums,
                    )
                    # Reject any owned marker without a proved lowering, so an
                    # incomplete per-lane input cannot stand in for a reduction.
                    self.device_function.body = restore_unprocessed_lane_reduce_markers(
                        list(self.device_function.body)
                    )
                    # Restructure chunked-recurrence ``for chunk: for lane:``
                    # nests whose matmul ``dot_acc`` running-sum needs the
                    # lane-invariant rescale / chunk-entry stores / final combine
                    # hoisted to run once per chunk (gdn_fwd_h).
                    self.device_function.body = hoist_lane_invariant_chunk_recurrence(
                        list(self.device_function.body),
                        rename_groups={
                            name: aliases[0]
                            for name, aliases in self.device_function._variable_renames.items()
                        },
                        running_sums=self.device_function.cute_matmul_running_sums,
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
    with func:
        env = CompileEnvironment.current()
        if (
            env.backend_name == "cute"
            and config.get("cute_materialized_operand_schedule", "off") != "off"
        ):
            from .cute.packed_operand_codegen import generate_packed_operand

            if (
                store_transform is not None
                or load_transform is not None
                or extra_params
                or _codegen_graphs is not None
                or _memory_counters is not None
                or _host_prefix is not None
                or _bounded_cache_request is not None
                or config.get("cute_materialized_schedule", "off") != "off"
            ):
                raise exc.InvalidConfig(
                    "packed-operand schedule requires complete ordinary codegen"
                )
            return generate_packed_operand(func, config, emit_repro_caller)
        if (
            env.backend_name == "cute"
            and config.get("cute_materialized_schedule", "off") != "off"
        ):
            from .cute.row_resident_codegen import generate_row_resident

            if (
                store_transform is not None
                or load_transform is not None
                or extra_params
                or _codegen_graphs is not None
                or _memory_counters is not None
                or _host_prefix is not None
                or _bounded_cache_request is not None
            ):
                raise exc.InvalidConfig(
                    "row-resident schedule does not support external transforms or partial codegen"
                )
            return generate_row_resident(func, config, emit_repro_caller)
        if (
            env.backend.name == "cute"
            and config.get("cute_mma_f32_conversion") == "warp_raw"
        ):
            from .cute.tcgen05_config import CuteTcgen05Config
            from .cute.tcgen05_flat_grouped_codegen import (
                generate_grouped_warp_tf32_candidate,
            )
            from .cute.tcgen05_flat_grouped_config import (
                CONFIG_KEYS as GROUPED_WARP_KEYS,
            )

            if (
                store_transform is None
                and load_transform is None
                and not extra_params
                and _codegen_graphs is None
                and _memory_counters is None
                and _host_prefix is None
                and _bounded_cache_request is None
            ):
                return generate_grouped_warp_tf32_candidate(
                    func,
                    config,
                    emit_repro_caller,
                    shared_capacity=CuteTcgen05Config.per_cta_smem_capacity_bytes(
                        env.device
                    ),
                )
            config = Config.from_dict(
                {
                    key: value
                    for key, value in config.config.items()
                    if key not in GROUPED_WARP_KEYS
                }
            )
        if env.backend.name == "cute" and (
            config.get("cute_grouped_rna_warps", 0)
            or config.get("cute_mma_f32_conversion") == "tma_rn"
        ):
            from .cute.tcgen05_config import CuteTcgen05Config
            from .cute.tcgen05_flat_grouped_codegen import (
                generate_flat_grouped_native_candidate,
            )
            from .cute.tcgen05_flat_grouped_config import (
                CONFIG_KEYS as GROUPED_RNA_KEYS,
            )
            from .cute.tcgen05_flat_grouped_config import K_KEY as GROUPED_RNA_K_KEY
            from .cute.tcgen05_flat_grouped_config import K_STAGES
            from .cute.tcgen05_flat_grouped_config import (
                STAGES_KEY as GROUPED_RNA_STAGES_KEY,
            )
            from .cute.tcgen05_flat_grouped_config import (
                WARPS_KEY as GROUPED_RNA_WARPS_KEY,
            )

            if (
                store_transform is None
                and load_transform is None
                and not extra_params
                and _codegen_graphs is None
                and _memory_counters is None
                and _host_prefix is None
                and _bounded_cache_request is None
            ):
                block_k = config[GROUPED_RNA_K_KEY]
                converter_warps = config.get(GROUPED_RNA_WARPS_KEY, 0)
                assert isinstance(block_k, int)
                assert isinstance(converter_warps, int)
                return generate_flat_grouped_native_candidate(
                    func,
                    config,
                    emit_repro_caller,
                    converter_warps=converter_warps,
                    block_k=block_k,
                    ab_stages=cast(
                        "int", config.get(GROUPED_RNA_STAGES_KEY, K_STAGES[block_k])
                    ),
                    shared_capacity=CuteTcgen05Config.per_cta_smem_capacity_bytes(
                        env.device
                    ),
                )
            config = Config.from_dict(
                {
                    key: value
                    for key, value in config.config.items()
                    if key not in GROUPED_RNA_KEYS
                }
            )
        if env.backend.name == "cute" and config.get("cute_reduction_sequence") in {
            "bounded",
            "bounded_layout",
        }:
            if (
                store_transform is None
                and load_transform is None
                and not extra_params
                and _codegen_graphs is None
                and _memory_counters is None
                and _host_prefix is None
            ):
                from .cute.bounded_cache_codegen import generate_bounded_cache

                return generate_bounded_cache(func, config, emit_repro_caller)
            # Unsupported external transforms keep the ordinary lowering.
            config = type(config).from_dict(
                {**config.config, "cute_reduction_sequence": "scalar"}
            )
        if env.cute_fission_plan is not None and len(func.device_ir.root_ids) > 1:
            from .cute.materialized_fission_codegen import generate_materialized_fission

            return generate_materialized_fission(
                func,
                config,
                emit_repro_caller,
                env.cute_fission_plan,
                store_transform=store_transform,
                load_transform=load_transform,
                extra_params=extra_params,
            )
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
                codegen.add_statement(codegen.visit(stmt))
            codegen.device_function.cute_state.finalize_tcgen05_pure_lifecycle_stores()
            if _bounded_cache_request is None:
                kernel_def = codegen.device_function.codegen_function_def()
            else:
                kernel_def = codegen.device_function.codegen_function_def(
                    bounded_cache_request=_bounded_cache_request
                )
            block_dims = (
                codegen.device_function.cute_state.collective_register_chain_block_dims
            )
            if block_dims is not None:
                from .cute.thread_block_projection import update_launch_block

                assert (
                    update_launch_block(
                        codegen.host_statements,
                        codegen.device_function.name,
                        block_dims,
                    )
                    > 0
                )
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
            final_host_statements: list[ast.AST] = [
                *rng_statements,
                *codegen.host_statements,
            ]
            if not codegen._cute_can_bake_tensor_shapes():
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
                        "lhs_scale_name",
                        "rhs_scale_name",
                        "workspace_name",
                        "c_name",
                        "d_name",
                        "q_name",
                        "k_name",
                        "g_name",
                        "beta_name",
                        "a_log_name",
                        "dt_name",
                        "cu_seqlens_name",
                        "cu_chunks_name",
                        "chunk_to_seq_name",
                        "kd_name",
                        "qd_name",
                        "ak_name",
                        "aq_name",
                        "gt_name",
                        "v_name",
                        "o_name",
                        "out_name",
                        "state_name",
                        "do_name",
                        "delta_name",
                        "dq_name",
                        "dk_name",
                        "dv_name",
                        "lse_name",
                        "bias_name",
                        "alibi_name",
                        "document_name",
                        "layout_name",
                        "n_sizes_name",
                        "k_sizes_name",
                        "direct_pointers_name",
                        "direct_strides_name",
                        "scale_name",
                        "m_extent_name",
                    ):
                        if key in resolved:
                            arg_name = str(resolved.pop(key))
                            resolved[key[:-5] + "_idx"] = launcher_arg_positions[
                                arg_name
                            ]
                            if key in {"layout_name", "n_sizes_name"} and (
                                arg_name in host_arg_positions
                            ):
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
                        f"{codegen.device_function.name}._helion_cute_wrapper_plans = {resolved_wrapper_plans!r}"
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

            if env.backend_name == "cute":
                from .cute.host_paired_sum import PAIRED_SUM_KEY
                from .cute.host_paired_sum import lower_host_sum_pairs
                from .cute.host_single_sum import lower_host_single_sum

                final_host_statements = lower_host_sum_pairs(
                    func,
                    final_host_statements,
                    cast(
                        "str", codegen.device_function.config.get(PAIRED_SUM_KEY, "off")
                    ),
                )
                final_host_statements = lower_host_single_sum(
                    func,
                    final_host_statements,
                    cast(
                        "str", codegen.device_function.config.get(PAIRED_SUM_KEY, "off")
                    ),
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
            if _memory_counters is not None:
                for name in _memory_counters:
                    _memory_counters[name] = getattr(codegen.device_function, name)
            # break circular reference for better GC
            del codegen.device_function.codegen
            return result
