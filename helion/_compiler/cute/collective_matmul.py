"""Collective warp MMA for a serial output-row tile with computed operands.

Scalar operand code remains the source of indexing and masking semantics. A
late pass replays those pure recipes at cooperative shared-memory coordinates,
then leaves the original scalar epilogue and row traversal in place.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import starmap
import operator
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import cast

import torch

from ... import exc
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import matmul_ops
from ...language.memory_ops import load as language_load
from ..ast_extension import expr_from_string
from ..ast_read_writes import ReadWrites
from ..ast_read_writes import ast_rename
from ..ast_read_writes import dead_assignment_elimination
from ..ast_read_writes import dead_lane_loop_elimination
from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo
from ..device_ir import control_flow_parent_entries
from ..reduction_strategy import PersistentReductionStrategy
from ..tile_strategy import DeviceLoopState
from ._ast_pass_utils import _bound_names
from ._ast_pass_utils import _fresh_prefix
from .collective_operand_packet import plan_collective_operand_packet
from .collective_tcgen05 import CollectiveOperandSmem
from .collective_tcgen05 import CollectiveTcgen05Plan
from .collective_tcgen05 import CollectiveTmemResource
from .collective_tmem_seed import NativeCollectiveLifetime
from .collective_vector_recipe import emit_vector_recipe
from .collective_warp_pipeline import pipeline_collective_loop
from .collective_warp_pipeline import signed_int32_bound
from .contiguous_copy import CopyTensorFacts
from .contiguous_copy import plan_contiguous_copy
from .contiguous_copy import recipe_has_integer_result
from .cute_mma import _is_zero_init_acc_node
from .cute_mma import _mma_loop_is_exclusive
from .cute_mma import _mma_result_can_be_deferred
from .fold_noop_stores import _is_read_only
from .indexing import CutePackedAffineLoad
from .indexing import CutePackedTerms
from .input_view_layout import input_view_copy_facts
from .matmul_utils import cute_resolve_active_block_id
from .matmul_utils import cute_synthetic_lane_k_extent
from .mma_support import cute_fp32_dot_uses_tf32
from .packed_matmul import packed_matmul_axis
from .scalar_recipe import ScalarRecipe
from .scalar_recipe import build_recipe
from .scalar_recipe_cache import plan_scalar_recipe_cache
from .scalar_recipe_rounding import preserve_fp32_multiply_rounding

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch.fx import Node

    from ..aten_lowering import LoweringContext
    from ..device_function import DeviceFunction
    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState


MARKER = "_helion_pending_collective_mma"


@dataclass(frozen=True)
class CollectiveMmaSite:
    identity: int
    m_index: str
    n_index: str
    k_index: str
    m_offset: str
    n_offset: str
    k_offset: str
    bm: int
    bn: int
    bk: int
    m_axis: int
    n_axis: int
    dtype: str
    grid_row_lane: str | None
    zero_seed: bool = True
    synthetic_k_lane: str | None = None
    static_k_extent: int | None = None
    k_factor: int = 1


@dataclass(frozen=True)
class CollectiveSiteAnalysis:
    """The existing scalar, control and alias proofs before any MMA emission."""

    site: CollectiveMmaSite
    row_loop: ast.For
    reduction_loop: ast.For
    path: list[tuple[Sequence[ast.stmt], int]]
    row_depth: int
    reduction_depth: int
    result_name: str
    reduction_iterator: ast.Call
    a_recipe: ScalarRecipe
    b_recipe: ScalarRecipe
    seed_recipe: ScalarRecipe | None
    dominating: list[ast.stmt]
    seed_dominating: list[ast.stmt]
    pure_statement_ids: frozenset[int]


def _native_site_enabled(site: CollectiveMmaSite, config: Mapping[str, object]) -> bool:
    return (
        config.get("cute_collective_compute", "warp") == "tcgen05"
        and site.bm in (64, 128)
        and (
            site.zero_seed or config.get("cute_collective_native_seeded", False) is True
        )
    )


def _is_direct_zero_seed(
    graphs: Sequence[GraphInfo], node: Node, acc_node: Node
) -> bool:
    """A zero created in this scope, excluding an enclosing loop's carry.

    Tracing a phi all the way to its original zero is insufficient: a later
    iteration of the enclosing loop may already hold a nonzero accumulator.
    """
    current = acc_node
    while current.op == "call_function" and current.target is _tracing_ops._new_var:
        assert isinstance(current.args[0], torch.fx.Node)
        current = current.args[0]
    if current.op == "placeholder":
        graph = next(info for info in graphs if info.graph is node.graph)
        parent_entry = control_flow_parent_entries(graphs).get(graph.graph_id)
        if not isinstance(graph, ForLoopGraphInfo) or parent_entry is None:
            return False
        parent, argument_slot = parent_entry
        captured = parent.args[argument_slot]
        assert isinstance(captured, (list, tuple))
        position = list(node.graph.find_nodes(op="placeholder")).index(current)
        current = captured[position]
        if not isinstance(current, torch.fx.Node):
            return False
    return current.target is creation_ops.full and _is_zero_init_acc_node(
        current, graphs=graphs
    )


def has_collective_native_seed_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """Avoid duplicate native seeds when every contraction starts at zero.

    This is only a search hint. Code generation still proves the direct
    loop carry, pure seed recipe, aliases, and uniform collective control.
    Use the same immediate-scope zero test so an outer carry originating at
    zero is not mistaken for a fresh zero on every iteration.
    """
    for graph in graphs:
        for node in graph.graph.nodes:
            if node.target in (
                torch.ops.aten.addmm.default,
                torch.ops.aten.baddbmm.default,
            ):
                accumulator = node.args[0]
            elif node.target is matmul_ops.dot and len(node.args) >= 3:
                accumulator = node.args[2]
            else:
                continue
            if not isinstance(accumulator, torch.fx.Node):
                continue
            value = accumulator.meta.get("val")
            if (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.float32
                and not _is_direct_zero_seed(graphs, node, accumulator)
            ):
                return True
    return False


def has_collective_tmem_operand_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """Seed register-produced half A operands, leaving final admission to AST.

    A direct load already has an asynchronous shared-memory copy alternative.
    An expression supplying A needs registers before either storage choice.
    This search hint does not grant any purity, layout, alias or control proof.
    """
    for graph in graphs:
        for node in graph.graph.nodes:
            if node.target in (
                torch.ops.aten.addmm.default,
                torch.ops.aten.baddbmm.default,
            ):
                lhs = node.args[1]
            elif node.target in (
                torch.ops.aten.mm.default,
                torch.ops.aten.bmm.default,
                matmul_ops.dot,
            ):
                lhs = node.args[0]
            else:
                continue
            if not isinstance(lhs, torch.fx.Node):
                continue
            while lhs.target is _tracing_ops._new_var and isinstance(
                lhs.args[0], torch.fx.Node
            ):
                lhs = lhs.args[0]
            value = lhs.meta.get("val")
            if (
                lhs.op == "call_function"
                and lhs.target is not language_load
                and isinstance(value, torch.Tensor)
                and value.dtype in (torch.float16, torch.bfloat16)
            ):
                return True
    return False


def _direct_loop_carry(
    cg: GenerateAST, node: Node, acc_node: Node, k_block_id: int
) -> bool:
    """Require one result and a direct captured accumulator in this exact loop.

    The late AST proof also checks the canonical phi binding: the marker must
    read the same scalar that it assigns. This excludes a loop that repeatedly
    overwrites an output using an invariant (possibly nonzero) input.
    """
    graph = next(info for info in cg.codegen_graphs if info.graph is node.graph)
    if not isinstance(graph, ForLoopGraphInfo) or graph.block_ids != [k_block_id]:
        return False
    outputs = tuple(node.graph.find_nodes(op="output"))
    if len(outputs) != 1 or outputs[0].args != ([node],):
        return False
    current = acc_node
    while current.op == "call_function" and current.target is _tracing_ops._new_var:
        if len(current.args) != 1 or not isinstance(current.args[0], torch.fx.Node):
            return False
        current = current.args[0]
    if current.op != "placeholder" or current.graph is not node.graph:
        return False
    parent_entry = control_flow_parent_entries(cg.codegen_graphs).get(graph.graph_id)
    if parent_entry is None:
        return False
    parent, argument_slot = parent_entry
    captured = parent.args[argument_slot]
    assert isinstance(captured, (tuple, list))
    position = list(node.graph.find_nodes(op="placeholder")).index(current)
    original_seed = captured[position]
    return bool(parent.users) and all(
        user.target is operator.getitem
        and user.args == (parent, 0)
        and bool(user.users)
        and all(
            phi.target is _tracing_ops._phi and phi.args == (original_seed, user)
            for phi in user.users
        )
        for user in parent.users
    )


def _exclusive_synthetic_k_operands(
    cg: GenerateAST, node: Node, lhs_node: Node, rhs_node: Node, k_block_id: int
) -> tuple[Node, ...] | None:
    """Prove that removing this synthetic K lane cannot change another value.

    The candidate may feed a later contraction or pointwise epilogue. Every
    value still carrying its K coordinate must belong exclusively to the pure
    operand slice; a second reduction or escaping operand keeps the old path.
    """
    dependencies: set[Node] = set()
    pending = [lhs_node, rhs_node]
    while pending:
        current = pending.pop()
        if current in dependencies:
            continue
        if current.graph is not node.graph or not _is_read_only(current):
            return None
        dependencies.add(current)
        pending.extend(current.all_input_nodes)
    for current in node.graph.nodes:
        if current.target in (_tracing_ops._host_tensor, _tracing_ops._constant_tensor):
            # A materialized tensor's static dimensions do not denote live
            # scalar coordinates, even when one equals this full-slice K.
            continue
        value = current.meta.get("val")
        if not isinstance(value, torch.Tensor) or not any(
            cute_resolve_active_block_id(cg, size) == k_block_id for size in value.shape
        ):
            continue
        if current not in dependencies or any(
            user not in dependencies and user is not node for user in current.users
        ):
            return None
    return tuple(dependencies)


def mark_collective_matmul(
    ctx: LoweringContext,
    node: Node,
    *,
    k_block_id: int | None,
    lhs: ast.AST | CutePackedAffineLoad | CutePackedTerms,
    rhs: ast.AST | CutePackedTerms,
    acc: ast.AST,
) -> ast.expr | None:
    from ..generate_ast import GenerateAST

    cg = ctx.cg
    if not isinstance(cg, GenerateAST):
        return None
    if (
        node.target
        not in (
            torch.ops.aten.addmm.default,
            torch.ops.aten.baddbmm.default,
        )
        or k_block_id is None
    ):
        return None
    acc_node, lhs_node, rhs_node = node.args
    if not all(isinstance(arg, torch.fx.Node) for arg in node.args):
        return None
    assert isinstance(acc_node, torch.fx.Node)
    assert isinstance(lhs_node, torch.fx.Node)
    assert isinstance(rhs_node, torch.fx.Node)
    return _mark_collective(
        cg,
        node,
        lhs_node,
        rhs_node,
        acc_node,
        k_block_id=k_block_id,
        lhs=lhs,
        rhs=rhs,
        acc=acc,
    )


def mark_collective_dot(
    state: CodegenState,
    *,
    k_block_id: int | None,
    lhs: ast.AST | CutePackedAffineLoad | CutePackedTerms,
    rhs: ast.AST | CutePackedTerms,
    acc: ast.AST,
) -> ast.expr | None:
    node = state.fx_node
    if node is None or k_block_id is None or len(node.args) < 3:
        return None
    lhs_node, rhs_node, acc_node = node.args[:3]
    if not all(
        isinstance(arg, torch.fx.Node) for arg in (lhs_node, rhs_node, acc_node)
    ):
        return None
    assert isinstance(lhs_node, torch.fx.Node)
    assert isinstance(rhs_node, torch.fx.Node)
    assert isinstance(acc_node, torch.fx.Node)
    return _mark_collective(
        state.codegen,
        node,
        lhs_node,
        rhs_node,
        acc_node,
        k_block_id=k_block_id,
        lhs=lhs,
        rhs=rhs,
        acc=acc,
    )


def _mark_collective(
    cg: GenerateAST,
    node: Node,
    lhs_node: Node,
    rhs_node: Node,
    acc_node: Node,
    *,
    k_block_id: int,
    lhs: ast.AST | CutePackedAffineLoad | CutePackedTerms,
    rhs: ast.AST | CutePackedTerms,
    acc: ast.AST,
) -> ast.expr | None:
    df = cg.device_function
    register_chain = df.config.get("cute_register_chain", False)
    if not register_chain and not df.config.get("cute_collective_mma", False):
        return None
    env = CompileEnvironment.current()
    k_factor = 1
    if isinstance(lhs, (CutePackedAffineLoad, CutePackedTerms)) or isinstance(
        rhs, CutePackedTerms
    ):
        if register_chain:
            return None
        if not isinstance(
            lhs, (CutePackedAffineLoad, CutePackedTerms)
        ) or not isinstance(rhs, CutePackedTerms):
            return None
        packed_axis = packed_matmul_axis(env, lhs_node, rhs_node)
        if (
            packed_axis is None
            or packed_axis.block_id != k_block_id
            or len(lhs.terms) != packed_axis.factor
            or len(rhs.terms) != packed_axis.factor
        ):
            return None
        k_factor = packed_axis.factor
        lhs = ast.Tuple(
            elts=list(cast("tuple[ast.expr, ...]", lhs.terms)), ctx=ast.Load()
        )
        rhs = ast.Tuple(
            elts=list(cast("tuple[ast.expr, ...]", rhs.terms)), ctx=ast.Load()
        )
    a_value, b_value, c_value = (
        lhs_node.meta["val"],
        rhs_node.meta["val"],
        acc_node.meta["val"],
    )
    if (
        a_value.ndim not in (2, 3)
        or b_value.ndim != a_value.ndim
        or c_value.ndim != a_value.ndim
        or a_value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or b_value.dtype != a_value.dtype
        or c_value.dtype != torch.float32
    ):
        return None
    if register_chain and a_value.dtype not in (torch.float16, torch.bfloat16):
        return None
    tf32 = a_value.dtype == torch.float32
    compute = df.config.get("cute_collective_compute", "warp")
    if tf32 and (
        not cute_fp32_dot_uses_tf32()
        or compute not in ("warp", "tcgen05")
        or env.config_spec.target_device_capability is None
        or env.config_spec.target_device_capability[0] < 8
        or compute == "tcgen05"
        and env.config_spec.target_device_capability[0] != 10
    ):
        return None
    m_block_id = cute_resolve_active_block_id(cg, a_value.shape[-2])
    n_block_id = cute_resolve_active_block_id(cg, b_value.shape[-1])
    if m_block_id is None or n_block_id is None:
        return None
    grid = cg.current_grid_state
    m_states = cg.active_device_loops[m_block_id]
    k_states = cg.active_device_loops[k_block_id]
    if grid is None or n_block_id not in grid.block_ids or not k_states:
        return None
    if a_value.ndim == 3:
        # A CTA still computes one matrix. A batch tile is admissible only
        # when all three operands carry the same singleton grid coordinate.
        batch_ids = tuple(
            cute_resolve_active_block_id(cg, value.shape[0])
            for value in (a_value, b_value, c_value)
        )
        batch = batch_ids[0]
        if (
            batch is None
            or batch_ids != (batch, batch, batch)
            or batch in (m_block_id, n_block_id, k_block_id)
            or batch not in grid.block_ids
            or df.resolved_block_size(batch) != 1
        ):
            return None
    k_loop = k_states[-1]
    zero_seed = _is_direct_zero_seed(cg.codegen_graphs, node, acc_node)
    if (
        tf32
        and compute == "tcgen05"
        and not (
            zero_seed or df.config.get("cute_collective_native_seeded", False) is True
        )
    ):
        return None
    synthetic_k_lane = None
    static_k_extent = None
    synthetic_dependencies: tuple[Node, ...] = ()
    if isinstance(k_loop.strategy, PersistentReductionStrategy):
        static_k_extent = cute_synthetic_lane_k_extent(cg, k_block_id)
        synthetic_k_lane = k_loop.strategy._synthetic_cute_lane_var
        if (
            k_factor != 1
            or not zero_seed
            or static_k_extent is None
            or not 1 < static_k_extent <= 128
            or synthetic_k_lane is None
            or (
                dependencies := _exclusive_synthetic_k_operands(
                    cg, node, lhs_node, rhs_node, k_block_id
                )
            )
            is None
        ):
            return None
        synthetic_dependencies = dependencies
    elif (
        not isinstance(k_loop, DeviceLoopState)
        or k_loop.block_ids != [k_block_id]
        or not _mma_loop_is_exclusive(node)
        or not _mma_result_can_be_deferred(node)
        or not _direct_loop_carry(cg, node, acc_node, k_block_id)
    ):
        return None
    grid_row_lane = None
    owned_synthetic_lanes = {
        site.synthetic_k_lane
        for site in df.cute_state.collective_mma_sites
        if site.synthetic_k_lane is not None
    }
    if synthetic_k_lane is not None:
        owned_synthetic_lanes.add(synthetic_k_lane)
    row_lanes = [
        name for name, _ in grid.lane_loops if name not in owned_synthetic_lanes
    ]
    if m_block_id in grid.block_ids:
        # A single synthetic row lane encloses the original scalar epilogue.
        # Hoist the collective above that lane, retaining the grid's row tile.
        if m_block_id not in grid.lane_loop_blocks or len(row_lanes) != 1:
            return None
        grid_row_lane = row_lanes[0]
        m_axis = grid.block_thread_axes.get(m_block_id)
    else:
        if (
            grid.has_lane_loops()
            or not m_states
            or not isinstance(m_loop := m_states[-1], DeviceLoopState)
            or m_loop.block_ids != [m_block_id]
        ):
            return None
        m_axis = m_loop.block_thread_axes.get(m_block_id)
    spec = env.config_spec
    sizes = tuple(
        df.resolved_block_size(b) for b in (m_block_id, n_block_id, k_block_id)
    )
    if not all(isinstance(size, int) for size in sizes):
        return None
    bm, bn, bk = cast("tuple[int, int, int]", sizes)
    bk *= k_factor
    if static_k_extent is not None:
        bk = max(16, bk)
    m_choices = (16, 32, 64, 128) if register_chain else (32, 64, 128)
    n_choices = (16, 32, 64) if register_chain else (32, 64)
    if bm not in m_choices or bn not in n_choices or bk not in (16, 32, 64, 128):
        return None
    if tf32 and compute == "tcgen05" and bm not in (64, 128):
        return None
    threads = {
        block_id: spec.num_threads.config_get(df.config.num_threads, block_id, 0)
        for block_id in (m_block_id, n_block_id, k_block_id)
    }
    collective_threads = 64 if register_chain else 128
    if threads != {m_block_id: collective_threads // bn, n_block_id: bn, k_block_id: 1}:
        return None
    if any(
        spec.cute_vector_widths.config_get(
            cast("list[int]", df.config.get("cute_vector_widths", [])), block_id, 1
        )
        != 1
        for block_id in (m_block_id, n_block_id, k_block_id)
    ):
        return None
    n_axis = grid.block_thread_axes.get(n_block_id)
    if m_axis is None or n_axis is None or m_axis == n_axis:
        return None
    axes = {
        axis: size
        for axis, size in cg._current_active_thread_axis_sizes().items()
        if size > 1
    }
    if axes != {n_axis: bn, m_axis: collective_threads // bn}:
        return None
    # Any dynamic branch enclosing the collective must be a scalar predicate.
    # A tile predicate would let only part of the CTA reach its barriers.
    parents = control_flow_parent_entries(cg.codegen_graphs)
    graph = next(graph for graph in cg.codegen_graphs if graph.graph is node.graph)
    while graph.graph_id in parents:
        parent, _ = parents[graph.graph_id]
        if parent.target is _tracing_ops._if:
            predicate = parent.args[0]
            if not isinstance(predicate, torch.fx.Node):
                return None
            value = predicate.meta.get("val")
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                return None
        graph = next(item for item in cg.codegen_graphs if item.graph is parent.graph)
    sites = df.cute_state.collective_mma_sites
    site = CollectiveMmaSite(
        len(sites),
        cg.index_var(m_block_id),
        cg.index_var(n_block_id),
        cg.index_var(k_block_id),
        cg.offset_var(m_block_id),
        cg.offset_var(n_block_id),
        df.new_var("collective_k_offset")
        if synthetic_k_lane
        else cg.offset_var(k_block_id),
        bm,
        bn,
        bk,
        m_axis,
        n_axis,
        "cutlass.TFloat32" if tf32 else env.backend.dtype_str(a_value.dtype),
        grid_row_lane,
        zero_seed=zero_seed,
        synthetic_k_lane=synthetic_k_lane,
        static_k_extent=static_k_extent,
        k_factor=k_factor,
    )
    sites.append(site)
    if synthetic_dependencies:
        cg.allow_dead_assignments_owned_by_nodes((*synthetic_dependencies, acc_node))
    cg.cute_uses_matmul = True
    return cast(
        "ast.expr",
        expr_from_string(
            f"{MARKER}({site.identity}, {{lhs}}, {{rhs}}, {{acc}})",
            lhs=lhs,
            rhs=rhs,
            acc=acc,
        ),
    )


def _is_marker(node: ast.AST, identity: int) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == MARKER
        and len(node.args) == 4
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == identity
    )


def _marker_path(
    body: Sequence[ast.stmt], identity: int
) -> list[tuple[Sequence[ast.stmt], int]] | None:
    for index, statement in enumerate(body):
        if isinstance(statement, ast.Assign) and _is_marker(statement.value, identity):
            return [(body, index)]
        if isinstance(statement, (ast.For, ast.If)):
            for branch in (statement.body, statement.orelse):
                if path := _marker_path(branch, identity):
                    return [(body, index), *path]
    return None


def _tensor_roots(
    body: Sequence[ast.AST],
    method: str,
    *,
    tensor_names: frozenset[str] = frozenset(),
) -> set[str] | None:
    result: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            if (
                method == "load"
                and isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in tensor_names
            ):
                # Packed-affine loads retain CuTe tensor indexing rather than
                # the ordinary pointer.load spelling. Both read the tensor.
                result.add(node.value.id)
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == method
            ):
                continue
            pointer = node.func.value
            if (
                method == "load"
                and isinstance(pointer, ast.Attribute)
                and pointer.attr == "arch"
                and isinstance(pointer.value, ast.Name)
                and pointer.value.id == "cute"
            ):
                if len(node.args) != 2 or node.keywords:
                    return None
                pointer = node.args[0]
            roots = {
                part.value.id
                for part in ast.walk(pointer)
                if isinstance(part, ast.Attribute)
                and part.attr == "iterator"
                and isinstance(part.value, ast.Name)
            }
            if len(roots) != 1:
                return None
            result.update(roots)
    return result


def _relaxed_atomic_add_root(call: ast.Call) -> str | None:
    """Classify an unused-result atomic without changing its statement.

    Only explicitly relaxed additions can move relative to disjoint operand
    reads. Acquire/release atomics may communicate through other allocations,
    so a disjoint pointer alone does not make those reorderings safe.
    """
    if ast.unparse(call.func) != "cute.arch.atomic_add" or len(call.args) != 1:
        return None
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if len(keywords) != len(call.keywords) or set(keywords) not in (
        {"val", "sem"},
        {"val", "sem", "scope"},
    ):
        return None
    sem = keywords["sem"]
    if not isinstance(sem, ast.Constant) or sem.value != "relaxed":
        return None
    if "scope" in keywords:
        scope = keywords["scope"]
        if not isinstance(scope, ast.Constant) or scope.value not in (
            "cta",
            "cluster",
            "gpu",
            "sys",
        ):
            return None
    pointer = call.args[0]
    if not isinstance(pointer, ast.Attribute) or pointer.attr != "llvm_ptr":
        return None
    for value in (pointer.value, keywords["val"]):
        names = set(ReadWrites.from_ast(value).reads)
        if build_recipe(value, [], names) is None:
            return None

    def integer_offset(value: ast.expr) -> bool:
        if isinstance(value, ast.Constant):
            return type(value.value) is int
        if isinstance(value, ast.Call):
            callee = ast.unparse(value.func)
            return not value.keywords and (
                callee
                in {
                    "cutlass.Int32",
                    "cutlass.Int64",
                    "cutlass.Uint32",
                    "cutlass.Uint64",
                }
                and len(value.args) == 1
                or callee == "cute.crd2idx"
                and len(value.args) == 2
            )
        if isinstance(value, ast.UnaryOp) and isinstance(
            value.op, (ast.UAdd, ast.USub)
        ):
            return integer_offset(value.operand)
        if isinstance(value, ast.BinOp) and isinstance(
            value.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
        ):
            return integer_offset(value.left) and integer_offset(value.right)
        return False

    # Counted iterator references are insufficient: a conditional can select
    # an opaque pointer alias in another branch. Require a single explicit
    # tensor base and offsets whose operations guarantee an integer result.
    base = pointer.value
    while isinstance(base, ast.BinOp) and isinstance(base.op, (ast.Add, ast.Sub)):
        if not integer_offset(base.right):
            return None
        base = base.left
    if (
        isinstance(base, ast.Attribute)
        and base.attr == "iterator"
        and isinstance(base.value, ast.Name)
    ):
        return base.value.id
    return None


def _region_write_roots(
    body: Sequence[ast.stmt],
    identity: int | set[int],
    *,
    emitted_statement_ids: set[int] | None = None,
    allow_relaxed_atomics: bool = True,
) -> set[str] | None:
    """Prove the effects of the region across which staging moves.

    Recipes prove scalar dataflow, not freedom to cross effects. Accept only
    pure scalar statements and classified pointer writes with a known tensor
    root. An unused relaxed atomic add is a read/write of its output root;
    the caller must prove that root disjoint from operands and control reads.
    Other atomics, barriers, opaque calls, and write forms refuse staging.
    Whole-region replacements that preserve only stores must disable relaxed
    atomics, since knowing their write root does not preserve their effect.
    """
    roots: set[str] = set()
    identities = {identity} if isinstance(identity, int) else identity

    def pure(value: ast.expr) -> bool:
        if any(_is_marker(value, candidate) for candidate in identities):
            assert isinstance(value, ast.Call)
            return all(pure(argument) for argument in value.args)
        names = set(ReadWrites.from_ast(value).reads)
        return build_recipe(value, [], names) is not None

    def check(statement: ast.stmt) -> bool:
        if emitted_statement_ids is not None and id(statement) in emitted_statement_ids:
            return True
        if isinstance(statement, ast.Assign):
            return all(
                isinstance(target, ast.Name) for target in statement.targets
            ) and pure(statement.value)
        if isinstance(statement, ast.Expr):
            value = statement.value
            if (
                allow_relaxed_atomics
                and isinstance(value, ast.Call)
                and (root := _relaxed_atomic_add_root(value)) is not None
            ):
                roots.add(root)
                return True
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "store"
            ):
                written = _tensor_roots([value], "store")
                if (
                    written is None
                    or len(value.args) != 1
                    or value.keywords
                    or not pure(value.func.value)
                    or not pure(value.args[0])
                ):
                    return False
                roots.update(written)
                return True
            return pure(value)
        if isinstance(statement, ast.For):
            iterator = statement.iter
            return (
                isinstance(statement.target, ast.Name)
                and isinstance(iterator, ast.Call)
                and ast.unparse(iterator.func)
                in {"range", "cutlass.range", "cutlass.range_constexpr"}
                and all(pure(argument) for argument in iterator.args)
                and all(
                    keyword.arg is not None and pure(keyword.value)
                    for keyword in iterator.keywords
                )
                and all(check(child) for child in statement.body)
                and all(check(child) for child in statement.orelse)
            )
        if isinstance(statement, ast.If):
            return (
                pure(statement.test)
                and all(check(child) for child in statement.body)
                and all(check(child) for child in statement.orelse)
            )
        return isinstance(statement, ast.Pass)

    return roots if all(check(statement) for statement in body) else None


def _statements(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


def _is_scalar_carry_alias(
    value: ast.expr, result_name: str, prefix: Sequence[ast.stmt]
) -> bool:
    if not isinstance(value, ast.Name):
        return False
    current = value.id
    for statement in reversed(prefix):
        if current == result_name:
            return True
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == current
        ):
            if not isinstance(statement.value, ast.Name):
                return False
            current = statement.value.id
    return current == result_name


def _remove_dead_staged_operands(
    body: list[ast.stmt],
    df: DeviceFunction,
    *,
    pure_statement_ids: set[int] | None = None,
) -> None:
    """Delete only unused scalar assignments proven pure by staging recipes.

    The scalar loads are lifted outside the FX statement-owner scope. Preserve
    assignment identity so an unrelated effectful rebinding of the same name
    can never become eligible for deletion.
    """
    owned = (
        df.cute_state.collective_mma_pure_stmt_ids
        if pure_statement_ids is None
        else pure_statement_ids
    )

    class DeleteUnused(ast.NodeTransformer):
        def __init__(self, reads: set[str]) -> None:
            self.reads = reads
            self.changed = False

        def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
            if (
                id(node) in owned
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id not in self.reads
            ):
                self.changed = True
                return None
            return node

    # Each successful round removes at least one registered definition.
    for _ in range(len(owned) + 1):
        transformer = DeleteUnused(set(ReadWrites.from_list(body).reads))
        module = transformer.visit(ast.Module(body=body, type_ignores=[]))
        assert isinstance(module, ast.Module)
        body[:] = module.body
        if not transformer.changed:
            break
    dead_assignment_elimination(cast("list[ast.AST]", body), df.dce_vars)
    dead_lane_loop_elimination(cast("list[ast.AST]", body))
    synthetic_lanes = {
        site.synthetic_k_lane
        for site in df.cute_state.collective_mma_sites
        if site.synthetic_k_lane is not None
    }
    if any(
        isinstance(node, ast.Name) and node.id in synthetic_lanes
        for statement in body
        for node in ast.walk(statement)
    ):
        raise exc.BackendUnsupported(
            "cute", "collective MMA: full-slice K still has scalar consumers"
        )


def _index_at(offset: str, coordinate: str) -> ast.expr:
    return ast.BinOp(
        left=ast.Name(id=offset, ctx=ast.Load()),
        op=ast.Add(),
        right=ast.Name(id=coordinate, ctx=ast.Load()),
    )


def _uses_thread_coordinates(statements: Sequence[ast.AST]) -> bool:
    return any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func)
        in {"cute.arch.thread_idx", "cute.arch.lane_idx", "cute.arch.warp_idx"}
        for statement in statements
        for node in ast.walk(statement)
    )


def _uniform_reuse_names(
    prefix: Sequence[ast.stmt],
    loop_body: Sequence[ast.stmt],
    boundary_names: set[str],
    df: DeviceFunction,
    *,
    integer_tensor_names: frozenset[str] = frozenset(),
) -> tuple[set[str], set[str]]:
    """Reuse immutable CTA-uniform definitions already dominating staging.

    The caller first proves alias safety using fully expanded operand recipes.
    Cutting recipes at these names must not hide a read from that proof.
    Also retain proven integer results for copy alignment analysis.
    """
    immutable = set(ReadWrites.from_list(list(prefix)).writes) - set(
        ReadWrites.from_list(list(loop_body)).writes
    )
    result = set()
    integer_names = set()
    for name in sorted(immutable):
        recipe = build_recipe(ast.Name(id=name, ctx=ast.Load()), prefix, boundary_names)
        if recipe is None:
            continue
        statements, value = recipe.emit({}, df.new_var)
        if not _uses_thread_coordinates([*statements, value]):
            result.add(name)
            if recipe_has_integer_result(statements, value, integer_tensor_names):
                integer_names.add(name)
    return result, integer_names


def _copy_tensor_facts(df: DeviceFunction) -> dict[str, CopyTensorFacts]:
    from ..device_function import TensorArg
    from .memory_ops import runtime_tensor_has_specialized_alignment

    env = CompileEnvironment.current()
    strides = df.proven_tensor_stride_values()
    result = {}
    for argument in df.arguments:
        if not isinstance(argument, TensorArg):
            continue
        tensor = argument.fake_value
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            continue
        if (view_facts := input_view_copy_facts(env, tensor)) is not None:
            result[argument.name] = view_facts
            continue
        values = tuple(strides.get((argument.name, dim)) for dim in range(tensor.ndim))
        if any(value is None for value in values):
            continue
        if not runtime_tensor_has_specialized_alignment(env, tensor, 16):
            continue
        result[argument.name] = CopyTensorFacts(
            env.backend.dtype_str(tensor.dtype), cast("tuple[int, ...]", values), 16
        )
    return result


def _lower_site(
    body: list[ast.stmt],
    site: CollectiveMmaSite,
    df: DeviceFunction,
    boundary_names: set[str],
    disjoint_pairs: set[frozenset[str]],
    tmem_resource: CollectiveTmemResource | None = None,
    *,
    fast_math: bool = False,
    operand_smem: CollectiveOperandSmem | None = None,
    native_lifetimes: list[NativeCollectiveLifetime] | None = None,
    register_chain_analysis: list[CollectiveSiteAnalysis] | None = None,
    analysis_fresh_name: Callable[[str], str] | None = None,
) -> tuple[ast.For, str, str]:
    from ..device_function import TensorArg

    def reject(reason: str) -> NoReturn:
        raise exc.BackendUnsupported("cute", f"collective MMA: {reason}")

    fresh = df.new_var if analysis_fresh_name is None else analysis_fresh_name
    tensor_names = frozenset(
        argument.name for argument in df.arguments if isinstance(argument, TensorArg)
    )
    collective_threads = 64 if register_chain_analysis is not None else 128
    expected_axes = {
        site.m_axis: collective_threads // site.bn,
        site.n_axis: site.bn,
    }
    expected_dims = tuple(expected_axes.get(axis, 1) for axis in range(3))
    actual_dims = tuple(
        starmap(
            max,
            zip(
                df.tile_strategy.thread_block_dims(),
                df.codegen.max_thread_block_dims,
                strict=True,
            ),
        )
    )
    if actual_dims != expected_dims:
        reject(
            f"the complete launch is not the proven {collective_threads}-thread collective"
        )
    path = _marker_path(body, site.identity)
    if path is None:
        reject("pending result was not a scalar assignment")
    assert path is not None
    statements = [part[index] for part, index in path]
    m_depths = [
        i
        for i, statement in enumerate(statements)
        if isinstance(statement, ast.For)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == (site.grid_row_lane or site.m_offset)
    ]
    k_depths = [
        i
        for i, statement in enumerate(statements)
        if isinstance(statement, ast.For)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == (site.synthetic_k_lane or site.k_offset)
    ]
    if len(m_depths) != 1 or len(k_depths) != 1:
        reject("expected one row loop and one reduction loop")
    m_depth, k_depth = m_depths[0], k_depths[0]
    if not m_depth < k_depth:
        reject("row/reduction loop order")
    if any(
        not isinstance(statement, ast.For)
        for statement in statements[m_depth : k_depth + 1]
    ):
        reject("conditional inside row/reduction region")
    m_loop = cast("ast.For", statements[m_depth])
    # Serial M tiles insert into the tile-loop body; a grid M tile inserts
    # immediately before its scalar row-lane wrapper in the enclosing scope.
    m_parent = path[m_depth][0]
    region_body = list(m_parent) if site.grid_row_lane else m_loop.body
    # Scalar tensor rank does not imply CTA uniformity: inline assembly and
    # scalar lane/warp queries can produce different values in each thread.
    # Prove every condition and loop iterator that still encloses the new
    # collective at its insertion point, including the serial M tile loop.
    control_reads: set[str] = set()
    uniform_boundaries = set(boundary_names)
    control_depth = m_depth if site.grid_row_lane else m_depth + 1
    for depth in range(control_depth):
        statement = statements[depth]
        control_values: list[ast.expr]
        if isinstance(statement, ast.If):
            control_values = [statement.test]
        elif isinstance(statement, ast.For):
            iterator = statement.iter
            if (
                not isinstance(statement.target, ast.Name)
                or not isinstance(iterator, ast.Call)
                or ast.unparse(iterator.func)
                not in {"range", "cutlass.range", "cutlass.range_constexpr"}
                or any(keyword.arg is None for keyword in iterator.keywords)
                or statement.target.id in ReadWrites.from_list(statement.body).writes
                or any(
                    isinstance(child, ast.For)
                    and isinstance(child.target, ast.Name)
                    and child.target.id == statement.target.id
                    for child in ast.walk(
                        ast.Module(body=statement.body, type_ignores=[])
                    )
                )
            ):
                reject("enclosing loop is not a uniform scalar range")
            assert isinstance(iterator, ast.Call)
            control_values = [
                *iterator.args,
                *(keyword.value for keyword in iterator.keywords),
            ]
        else:
            reject("unsupported control flow enclosing the collective")
        control_prefix = [
            value for part, index in path[: depth + 1] for value in part[:index]
        ]
        for value in control_values:
            recipe = build_recipe(value, control_prefix, uniform_boundaries)
            if recipe is None:
                reject("enclosing control flow is not provably CTA-uniform")
            assert recipe is not None
            proof_statements, proof_value = recipe.emit({}, fresh)
            proof = [*proof_statements, proof_value]
            if _uses_thread_coordinates(proof):
                reject("enclosing control flow varies within the CTA")
            roots = _tensor_roots(proof, "load", tensor_names=tensor_names)
            if roots is None:
                reject("unclassified memory read in enclosing control flow")
            assert roots is not None
            control_reads.update(roots)
        if isinstance(statement, ast.For):
            assert isinstance(statement.target, ast.Name)
            uniform_boundaries.add(statement.target.id)
    k_loop = cast("ast.For", statements[k_depth])
    assignment = cast("ast.Assign", statements[-1])
    marker = cast("ast.Call", assignment.value)
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        reject("non-scalar accumulator binding")
    result_name = cast("ast.Name", assignment.targets[0]).id
    # The only output of the K loop is its accumulator. Keep every scalar
    # epilogue operation, including guarded read/modify/write stores, in place.
    k_parent, k_index = path[k_depth]
    if site.synthetic_k_lane is None:
        carry_prefix = [
            statement
            for part, index in path[k_depth + 1 :]
            for statement in part[:index]
        ]
        if not _is_scalar_carry_alias(marker.args[3], result_name, carry_prefix):
            reject("reduction does not directly carry its accumulator")
        k_writes = set(ReadWrites.from_ast(k_loop).writes)
        live_after = set(ReadWrites.from_list(list(k_parent[k_index + 1 :])).reads)
        if (k_writes & live_after) - {result_name}:
            reject("additional reduction-loop live-outs")
    emitted_statement_ids = df.cute_state.collective_mma_emitted_stmt_ids
    shared_results = df.cute_state.collective_mma_shared_results
    dominating = [
        statement
        for part, index in path
        for statement in part[:index]
        if id(statement) not in emitted_statement_ids
    ]
    seed_dominating = [
        statement
        for part, index in path[: k_depth + 1]
        for statement in part[:index]
        if id(statement) not in emitted_statement_ids
    ]
    if not isinstance(k_loop.iter, ast.Call):
        reject("noncanonical reduction iterator")
    assert isinstance(k_loop.iter, ast.Call)
    reduction_iterator = k_loop.iter
    if site.synthetic_k_lane is not None:
        assert site.static_k_extent is not None
        reduction_iterator = cast(
            "ast.Call",
            expr_from_string(
                f"cutlass.range(0, {site.static_k_extent}, {site.bk}, unroll=1)"
            ),
        )
    bound_statements: list[ast.stmt] = []
    bound_values: list[ast.expr] = []
    for value in reduction_iterator.args:
        recipe = build_recipe(
            value,
            dominating,
            uniform_boundaries | {site.m_offset, site.n_offset},
            mutable_names={result_name},
        )
        if recipe is None:
            reject("reduction bound is not a pure scalar expression")
        assert recipe is not None
        bound_setup, expression = recipe.emit({}, fresh)
        if _uses_thread_coordinates([*bound_setup, expression]):
            reject("reduction bound varies within the output tile")
        bound_statements.extend(bound_setup)
        bound_values.append(expression)
    boundaries = (
        boundary_names
        | shared_results
        | {
            site.m_offset,
            site.n_offset,
            site.k_offset,
            site.m_index,
            site.n_index,
            site.k_index,
        }
    )
    a_recipe = build_recipe(
        cast("ast.expr", marker.args[1]),
        dominating,
        boundaries,
        mutable_names={result_name},
    )
    b_recipe = build_recipe(
        cast("ast.expr", marker.args[2]),
        dominating,
        boundaries,
        mutable_names={result_name},
    )
    if a_recipe is None or b_recipe is None:
        reject("operand is not a rematerializable scalar recipe")
    seed_recipe = None
    if not site.zero_seed:
        seed_recipe = build_recipe(
            ast.Name(id=result_name, ctx=ast.Load()),
            seed_dominating,
            boundaries,
        )
        if seed_recipe is None:
            reject("initial accumulator is not a rematerializable scalar recipe")
    pure_statement_ids = set()
    for recipe, source in (
        (a_recipe, dominating),
        (b_recipe, dominating),
        (seed_recipe, seed_dominating),
    ):
        if recipe is not None:
            pure_statement_ids.update(
                id(source[index]) for index in recipe.source_statement_indices
            )
    prefix = _fresh_prefix(
        "collective",
        _bound_names(ast.Module(body=body, type_ignores=[])) | boundary_names,
        fresh,
    )
    tid = f"{prefix}_tid"
    local_m, local_n, local_k = f"{prefix}_m", f"{prefix}_n", f"{prefix}_k"
    k_coordinates = {
        site.k_index: _index_at(site.k_offset, local_k),
    }
    a_statements, a_value = a_recipe.emit(
        {
            site.m_index: _index_at(site.m_offset, local_m),
            **k_coordinates,
        },
        fresh,
    )
    b_statements, b_value = b_recipe.emit(
        {
            site.n_index: _index_at(site.n_offset, local_n),
            **k_coordinates,
        },
        fresh,
    )
    seed_statements: list[ast.Assign] = []
    seed_value: ast.expr = ast.Constant(value=0.0)
    if seed_recipe is not None:
        seed_statements, seed_value = seed_recipe.emit(
            {
                site.m_index: _index_at(site.m_offset, local_m),
                site.n_index: _index_at(site.n_offset, local_n),
            },
            fresh,
        )
    bound_reads = _tensor_roots(
        [*bound_statements, *bound_values], "load", tensor_names=tensor_names
    )
    read_roots = _tensor_roots(
        [
            *bound_statements,
            *bound_values,
            *a_statements,
            a_value,
            *b_statements,
            b_value,
            *seed_statements,
            seed_value,
        ],
        "load",
        tensor_names=tensor_names,
    )
    marker_ids = {item.identity for item in df.cute_state.collective_mma_sites}
    write_roots = _region_write_roots(
        region_body, marker_ids, emitted_statement_ids=emitted_statement_ids
    )
    if write_roots is None:
        reject("unclassified effects in the row-loop region")
    if read_roots is None or any(
        frozenset((read, write)) not in disjoint_pairs
        for read in read_roots
        for write in write_roots
    ):
        reject("operand loads may alias row-loop writes")
    if bound_reads is None:
        reject("unclassified memory read in the reduction bound")
    assert bound_reads is not None
    control_reads.update(bound_reads)
    if control_reads:
        # All threads must observe the same control decision. A writer anywhere
        # in the device function could race a load-derived predicate/bound in
        # another lane or CTA, even when it lies outside this M-loop region.
        device_writes = _region_write_roots(
            body, marker_ids, emitted_statement_ids=emitted_statement_ids
        )
        if device_writes is None:
            reject("unclassified device effects may race collective control flow")
        assert device_writes is not None
        if any(
            frozenset((read, write)) not in disjoint_pairs
            for read in control_reads
            for write in device_writes
        ):
            reject("collective control-flow reads may alias device writes")
    if register_chain_analysis is not None:
        register_chain_analysis.append(
            CollectiveSiteAnalysis(
                site=site,
                row_loop=m_loop,
                reduction_loop=k_loop,
                path=path,
                row_depth=m_depth,
                reduction_depth=k_depth,
                result_name=result_name,
                reduction_iterator=reduction_iterator,
                a_recipe=a_recipe,
                b_recipe=b_recipe,
                seed_recipe=seed_recipe,
                dominating=dominating,
                seed_dominating=seed_dominating,
                pure_statement_ids=frozenset(pure_statement_ids),
            )
        )
        return m_loop, tid, ""

    df.cute_state.collective_mma_pure_stmt_ids.update(pure_statement_ids)
    flat = f"{prefix}_flat"
    m_parent, m_index = path[m_depth]
    insertion_body = (
        cast("list[ast.stmt]", m_parent) if site.grid_row_lane else m_loop.body
    )
    insertion_index = m_index if site.grid_row_lane else 0
    prefix_before_m = [
        statement
        for part, index in path[: m_depth + 1]
        for statement in part[:index]
        if id(statement) not in emitted_statement_ids
    ]
    reuse, integer_reuse = _uniform_reuse_names(
        prefix_before_m,
        m_loop.body,
        boundary_names,
        df,
        integer_tensor_names=frozenset(
            argument.name
            for argument in df.arguments
            if isinstance(argument, TensorArg)
            and argument.fake_value.dtype in (torch.int32, torch.int64)
        ),
    )
    # Reused scalars and generated tile coordinates retain their integer
    # type, but no divisibility beyond one. Stronger alignment is separate.
    copy_boundary_alignments = dict.fromkeys(
        integer_reuse | {local_m, local_n, local_k, site.m_offset}, 1
    )
    if reuse:
        a_recipe = build_recipe(
            cast("ast.expr", marker.args[1]),
            dominating,
            boundaries | reuse,
            mutable_names={result_name},
        )
        b_recipe = build_recipe(
            cast("ast.expr", marker.args[2]),
            dominating,
            boundaries | reuse,
            mutable_names={result_name},
        )
        assert a_recipe is not None and b_recipe is not None
        a_statements, a_value = a_recipe.emit(
            {
                site.m_index: _index_at(site.m_offset, local_m),
                **k_coordinates,
            },
            fresh,
        )
        b_statements, b_value = b_recipe.emit(
            {
                site.n_index: _index_at(site.n_offset, local_n),
                **k_coordinates,
            },
            fresh,
        )
    # Reuse can rebuild the recipes above. Protect the final emitted versions
    # so cooperative placement cannot introduce new multiply/add contraction.
    if not fast_math:
        a_statements, a_value = preserve_fp32_multiply_rounding(a_statements, a_value)
        b_statements, b_value = preserve_fp32_multiply_rounding(b_statements, b_value)
        seed_statements, seed_value = preserve_fp32_multiply_rounding(
            seed_statements, seed_value
        )
    bm, bn, bk, dtype = site.bm, site.bn, site.bk, site.dtype
    tf32 = dtype == "cutlass.TFloat32"
    copy_strategy = df.config.get("cute_collective_copy", "scalar")
    vector_copy_eligible = (
        site.k_factor == 1
        and copy_strategy in ("async", "async_cached")
        and (site.static_k_extent is None or site.static_k_extent % bk == 0)
    )
    recipe_strategy = df.config.get("cute_collective_recipe", "scalar")
    vector_recipe_eligible = (
        recipe_strategy != "scalar"
        and site.k_factor == 1
        and (site.static_k_extent is None or site.static_k_extent % bk == 0)
    )
    tensor_facts = (
        _copy_tensor_facts(df) if vector_copy_eligible or vector_recipe_eligible else {}
    )
    copy_plans = [
        plan_contiguous_copy(
            statements,
            value,
            coordinate=coordinate,
            tensors=tensor_facts,
            aligned_names={
                **copy_boundary_alignments,
                coordinate: 8,
                site.k_offset: bk,
                site.n_offset: bn,
            },
        )
        if vector_copy_eligible
        else None
        for statements, value, coordinate in (
            (a_statements, a_value, local_k),
            (b_statements, b_value, local_n),
        )
    ]
    b_k_major = False
    if vector_copy_eligible and copy_plans[1] is None:
        # A transposed RHS can be contiguous along K rather than N. Keep the
        # original preference when N copies are legal; otherwise use the same
        # alignment/mask proof to choose K-major shared staging and MMA operands.
        copy_plans[1] = plan_contiguous_copy(
            b_statements,
            b_value,
            coordinate=local_k,
            tensors=tensor_facts,
            aligned_names={
                **copy_boundary_alignments,
                local_k: 8,
                site.k_offset: bk,
                site.n_offset: bn,
            },
        )
        b_k_major = copy_plans[1] is not None
    native = (
        CollectiveTcgen05Plan(
            prefix,
            tid,
            bm,
            bn,
            bk,
            dtype,
            tmem_resource,
            b_k_major=b_k_major,
            zero_seed=site.zero_seed,
            operands=operand_smem,
            a_in_tmem=(
                df.config.get("cute_collective_tmem_a", False) is True
                and site.k_factor == 1
                and dtype in ("cutlass.Float16", "cutlass.BFloat16")
                and copy_plans[0] is None
            ),
        )
        if tmem_resource is not None and _native_site_enabled(site, df.config)
        else None
    )
    b_major_extent = bk if b_k_major else bn
    b_strides: tuple[int, ...] = (bk, 1) if b_k_major else (1, bn)
    k_start, k_stop = (
        (ast.Constant(value=0), bound_values[0])
        if len(bound_values) == 1
        else (bound_values[0], bound_values[1])
    )
    stages = (
        cast("int", df.config.get("cute_collective_stages", 1))
        if native is None
        and all(plan is not None for plan in copy_plans)
        and signed_int32_bound(k_start)
        and signed_int32_bound(k_stop)
        else 1
    )
    a_shape = (bm, bk, stages) if stages > 1 else (bm, bk)
    a_strides = (bk, 1, bm * bk) if stages > 1 else (bk, 1)
    b_shape = (bn, bk, stages) if stages > 1 else (bn, bk)
    if stages > 1:
        b_strides = (*b_strides, bn * bk)
    fast_axis, slow_axis = sorted(expected_axes)
    # MMA's lane/warp identity must follow CUDA's physical linear thread ID,
    # independently of which output dimension the grid maps onto threadIdx.x.
    thread_index = (
        f"cutlass.Int32(cute.arch.thread_idx()[{fast_axis}]) + "
        f"cutlass.Int32(cute.arch.thread_idx()[{slow_axis}]) * "
        f"{expected_axes[fast_axis]}"
    )
    mma_k = 8 if tf32 else 16
    mma_operation = (
        f"cute.nvgpu.warp.MmaTF32Op((16, 8, {mma_k}))"
        if tf32
        else f"cute.nvgpu.warp.MmaF16BF16Op({dtype}, cutlass.Float32, (16, 8, {mma_k}))"
    )
    copy_a = copy_b = (
        "cute.nvgpu.CopyUniversalOp()"
        if tf32
        else "cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)"
    )
    if not tf32:
        copy_b = f"cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose={not b_k_major}, num_matrices=4)"
    copy_options = ", num_bits_per_copy=32" if tf32 else ""
    a_layout = f"cute.make_composed_layout(cute.make_swizzle({min(3, (bk // 8).bit_length() - 1)}, 3, 3), 0, cute.make_layout({a_shape}, stride={a_strides}))"
    b_layout = f"cute.make_composed_layout(cute.make_swizzle({min(3, (b_major_extent // 8).bit_length() - 1)}, 3, 3), 0, cute.make_layout({b_shape}, stride={b_strides}))"
    # Universal 32-bit copies require a plain layout after partitioning. Carry
    # the TF32 swizzle on the pointer so dynamic K/stage slices retain it without
    # leaving an unresolved ComposedLayout in the shared-to-register copy.
    # Layout swizzles act on element indices, while pointer swizzles act on byte
    # addresses: recast the swizzle before attaching it to either typed view.
    operand_setup = (
        f"""
{prefix}_al = {a_layout}
{prefix}_bl = {b_layout}
{prefix}_as = cute.recast_layout(8, {dtype}.width, {prefix}_al).inner
{prefix}_bs = cute.recast_layout(8, {dtype}.width, {prefix}_bl).inner
{prefix}_a = cute.make_tensor(cute.recast_ptr({prefix}_ap, {prefix}_as, dtype={dtype}), {prefix}_al.outer)
{prefix}_b = cute.make_tensor(cute.recast_ptr({prefix}_bp, {prefix}_bs, dtype={dtype}), {prefix}_bl.outer)
"""
        if tf32
        else f"""
{prefix}_a = cute.make_tensor({prefix}_ap, {a_layout})
{prefix}_b = cute.make_tensor({prefix}_bp, {b_layout})
"""
    )
    operand_pointers = (
        operand_smem.pointers(prefix, dtype, bm * bk * stages, bn * bk * stages)
        if operand_smem is not None and native is None
        else f"""
{prefix}_ap = cute.arch.alloc_smem({dtype}, {bm * bk * stages}, alignment=1024)
{prefix}_bp = cute.arch.alloc_smem({dtype}, {bn * bk * stages}, alignment=1024)
"""
    )
    setup = _statements(f"""
{tid} = {thread_index}
{operand_pointers}
{prefix}_cp = cute.arch.alloc_smem(cutlass.Float32, {bm * bn}, alignment=128)
{operand_setup}
{prefix}_c = cute.make_tensor({prefix}_cp, cute.make_layout(({bm}, {bn}), stride=({bn}, 1)))
{prefix}_mma = cute.make_tiled_mma({mma_operation}, atom_layout_mnk=(2, 2, 1), permutation_mnk=(32, cute.make_ordered_layout((8, 2, 2), order=(0, 2, 1)), {mma_k}))
{prefix}_thr = {prefix}_mma.get_slice({tid})
{prefix}_acc = cute.make_rmem_tensor({prefix}_mma.partition_shape_C(({bm}, {bn})), cutlass.Float32)
{prefix}_ra = {prefix}_thr.make_fragment_A({prefix}_thr.partition_shape_A(({bm}, {mma_k})))
{prefix}_rb = {prefix}_thr.make_fragment_B({prefix}_thr.partition_shape_B(({bn}, {mma_k})))
{prefix}_ca = cute.make_tiled_copy_A(cute.make_copy_atom({copy_a}, {dtype}{copy_options}), {prefix}_mma)
{prefix}_cb = cute.make_tiled_copy_B(cute.make_copy_atom({copy_b}, {dtype}{copy_options}), {prefix}_mma)
{prefix}_tca = {prefix}_ca.get_slice({tid})
{prefix}_tcb = {prefix}_cb.get_slice({tid})
{prefix}_pa = {prefix}_tca.partition_S({prefix}_a)
{prefix}_pb = {prefix}_tcb.partition_S({prefix}_b)
{prefix}_da = {prefix}_tca.retile({prefix}_ra)
{prefix}_db = {prefix}_tcb.retile({prefix}_rb)
""")
    if native is not None:
        setup = native.setup(thread_index)
    elif seed_recipe is None:
        setup.extend(_statements(f"{prefix}_acc.fill(0.0)"))
    before_seed = tuple([*bound_statements, *setup])
    seed_begin = len(setup)

    def vectorize_recipe_loop(
        loop: ast.For,
        statements: Sequence[ast.Assign],
        value: ast.expr,
        *,
        coordinate: str,
        destination: str,
        indices: tuple[str, str],
        elements: int,
        operand_a: bool | None = None,
    ) -> None:
        # Packet size is a single 128-bit input transaction. Shared stores are
        # scalar unless this operand's layout also proves a complete packet.
        width = 4 if tf32 else 8
        shared_packet = (
            plan_collective_operand_packet(
                dtype=dtype,
                native=native is not None,
                bm=bm,
                bn=bn,
                bk=bk,
                operand_a=operand_a,
                k_major=b_k_major,
                coordinate_k=coordinate == local_k,
                stages=stages,
                width=width,
                base_alignment=1024,
            )
            if operand_a is not None
            and df.config.get("cute_collective_operand_packets", False)
            else None
        )
        slots = (elements + 128 * width - 1) // (128 * width)
        loop.iter = cast(
            "ast.expr",
            expr_from_string(
                f"cutlass.range_constexpr({slots})"
                if recipe_strategy == "vector_unrolled"
                else f"cutlass.range({slots}, unroll=1)"
            ),
        )
        flat_assignment = loop.body[0]
        assert isinstance(flat_assignment, ast.Assign)
        flat_assignment.value = ast.BinOp(
            left=flat_assignment.value, op=ast.Mult(), right=ast.Constant(width)
        )
        vector_body = emit_vector_recipe(
            statements,
            value,
            coordinate=coordinate,
            width=width,
            tensors=tensor_facts,
            aligned_names={
                **copy_boundary_alignments,
                coordinate: width,
                site.k_offset: bk,
                site.n_offset: bn,
            },
            destination=destination,
            destination_indices=tuple(
                ast.Name(id=name, ctx=ast.Load()) for name in indices
            ),
            fresh_name=fresh,
            shared_packet=shared_packet,
        )
        loop.body = [
            *loop.body[:3],
            ast.If(
                test=cast("ast.expr", expr_from_string(f"{flat} < {elements}")),
                body=vector_body,
                orelse=[],
            ),
        ]

    if seed_recipe is not None:
        seed_copy = cast(
            "ast.For",
            _statements(f"""
for {prefix}_ci in cutlass.range({bm * bn // 128}, unroll=1):
    {flat} = {tid} + {prefix}_ci * 128
    {local_m} = {flat} // {bn}
    {local_n} = {flat} % {bn}
""")[0],
        )
        seed_copy.body.extend(seed_statements)
        seed_copy.body.extend(
            _statements(
                f"{prefix}_c[{local_m}, {local_n}] = cutlass.Float32({ast.unparse(seed_value)})"
            )
        )
        if vector_recipe_eligible:
            vectorize_recipe_loop(
                seed_copy,
                seed_statements,
                cast(
                    "ast.expr",
                    expr_from_string(f"cutlass.Float32({ast.unparse(seed_value)})"),
                ),
                coordinate=local_n,
                destination=f"{prefix}_c",
                indices=(local_m, local_n),
                elements=bm * bn,
            )
        setup.extend([seed_copy, *_statements("cute.arch.sync_threads()")])
        setup.extend(
            native.seed_from_shared()
            if native is not None
            else _statements(f"""
{prefix}_seed = {prefix}_thr.partition_C({prefix}_c)
cute.autovec_copy({prefix}_seed, {prefix}_acc)
""")
        )
    physical_bk = bk // site.k_factor
    seed_nodes = tuple(setup[seed_begin:])
    a_copy = cast(
        "ast.For",
        _statements(f"""
for {prefix}_ai in cutlass.range({(bm * physical_bk + 127) // 128}, unroll=1):
    {flat} = {tid} + {prefix}_ai * 128
    {local_m} = {flat} // {physical_bk}
    {local_k} = {flat} % {physical_bk}
""")[0],
    )
    b_copy = cast(
        "ast.For",
        _statements(f"""
for {prefix}_bi in cutlass.range({(bn * physical_bk + 127) // 128}, unroll=1):
    {flat} = {tid} + {prefix}_bi * 128
    {local_n} = {flat} {"// " + str(physical_bk) if b_k_major else "% " + str(bn)}
    {local_k} = {flat} {"% " + str(physical_bk) if b_k_major else "// " + str(bn)}
""")[0],
    )
    for copy_loop, statements, value, tensor, row, count in (
        (a_copy, a_statements, a_value, f"{prefix}_a", local_m, bm * physical_bk),
        (b_copy, b_statements, b_value, f"{prefix}_b", local_n, bn * physical_bk),
    ):
        copy_loop.body.extend(statements)
        if site.k_factor == 1:
            values = [value]
        else:
            assert isinstance(value, ast.Tuple) and len(value.elts) == site.k_factor
            values = value.elts
        for index, term in enumerate(values):
            k_coord = (
                local_k
                if site.k_factor == 1
                else f"{local_k} * {site.k_factor} + {index}"
            )
            converted = (
                "cutlass.TFloat32("
                f"cutlass.Uint32(cute.arch.cvt_f32_tf32({ast.unparse(term)}))"
                ".bitcast(cutlass.Float32))"
                if dtype == "cutlass.TFloat32"
                else f"{dtype}({ast.unparse(term)})"
            )
            copy_loop.body.extend(
                _statements(f"{tensor}[{row}, {k_coord}] = {converted}")
            )
        if count % 128:
            copy_loop.body[3:] = [
                ast.If(
                    test=cast("ast.expr", expr_from_string(f"{flat} < {count}")),
                    body=copy_loop.body[3:],
                    orelse=[],
                )
            ]
    if site.static_k_extent is not None and site.static_k_extent % bk:
        for copy_loop, tensor, coordinates in (
            (a_copy, f"{prefix}_a", (local_m, local_k)),
            (b_copy, f"{prefix}_b", (local_n, local_k)),
        ):
            copy_loop.body[3:] = [
                ast.If(
                    test=cast(
                        "ast.expr",
                        expr_from_string(
                            f"{site.k_offset} + {local_k} < {site.static_k_extent}"
                        ),
                    ),
                    body=copy_loop.body[3:],
                    orelse=_statements(
                        f"{tensor}[{', '.join(coordinates)}] = {dtype}(0)"
                    ),
                )
            ]
    async_copies = False
    tf32_rounding: list[ast.stmt] = []
    cache_prefill: list[ast.stmt] = []
    if vector_recipe_eligible:
        for loop, plan, statements, value, coordinate, tensor, indices, elements in (
            (
                a_copy,
                copy_plans[0],
                a_statements,
                a_value,
                local_k,
                f"{prefix}_a",
                (local_m, local_k),
                bm * bk,
            ),
            (
                b_copy,
                copy_plans[1],
                b_statements,
                b_value,
                local_n,
                f"{prefix}_b",
                (local_n, local_k),
                bn * bk,
            ),
        ):
            if plan is not None:
                continue
            converted_value = (
                "cutlass.TFloat32("
                f"cutlass.Uint32(cute.arch.cvt_f32_tf32({ast.unparse(value)}))"
                ".bitcast(cutlass.Float32))"
                if tf32
                else f"{dtype}({ast.unparse(value)})"
            )
            vectorize_recipe_loop(
                loop,
                statements,
                cast("ast.expr", expr_from_string(converted_value)),
                coordinate=coordinate,
                destination=tensor,
                indices=indices,
                elements=elements,
                operand_a=loop is a_copy,
            )
    if vector_copy_eligible:
        tensor_dtypes = {
            argument.name: CompileEnvironment.current().backend.dtype_str(
                argument.fake_value.dtype
            )
            for argument in df.arguments
            if isinstance(argument, TensorArg)
        }
        for copy_loop, plan, indices, tensor, elements in (
            (
                a_copy,
                copy_plans[0],
                (local_m, local_k),
                f"{prefix}_a",
                bm * bk,
            ),
            (
                b_copy,
                copy_plans[1],
                (local_n, local_k),
                f"{prefix}_b",
                bn * bk,
            ),
        ):
            if plan is None:
                continue
            async_copies = True
            slots = (elements + 128 * plan.width - 1) // (128 * plan.width)
            copy_loop.iter = ast.parse(
                f"cutlass.range_constexpr({slots})"
                if stages > 1
                else f"cutlass.range({slots}, unroll=1)",
                mode="eval",
            ).body
            flat_assignment = cast("ast.Assign", copy_loop.body[0])
            flat_assignment.value = ast.BinOp(
                left=flat_assignment.value,
                op=ast.Mult(),
                right=ast.Constant(value=plan.width),
            )
            copy_destination = tensor
            if dtype == "cutlass.TFloat32":
                # cp.async is a bit copy. Load the original FP32 bytes into a
                # matching shared view, then round each value exactly once
                # before exposing that buffer to TF32 compute. Both the native
                # and warp tensors carry their swizzle on the shared pointer.
                operand = "a" if tensor == f"{prefix}_a" else "b"
                copy_destination = f"{tensor}_raw"
                byte_swizzle = (
                    f"{prefix}_{operand}l.inner"
                    if native is not None
                    else f"{prefix}_{operand}s"
                )
                raw_view = f"cute.make_tensor(cute.recast_ptr({prefix}_{operand}p, {byte_swizzle}, dtype=cutlass.Float32), {prefix}_{operand}l.outer)"
                setup.extend(_statements(f"{copy_destination} = {raw_view}"))
                rounding_slot = fresh("tf32_round_slot")
                rounding_flat = fresh("tf32_round_flat")
                first, second = indices
                contiguous_k = operand == "a" or b_k_major
                major = bk if contiguous_k else bn
                first_value = (
                    f"{rounding_flat} // {major}"
                    if contiguous_k
                    else f"{rounding_flat} % {major}"
                )
                second_value = (
                    f"{rounding_flat} % {major}"
                    if contiguous_k
                    else f"{rounding_flat} // {major}"
                )
                stage_coordinate = f", {prefix}_stage_read" if stages > 1 else ""
                tf32_rounding.extend(
                    _statements(f"""
for {rounding_slot} in cutlass.range({elements // 128}, unroll=1):
    {rounding_flat} = {tid} + {rounding_slot} * 128
    {first} = {first_value}
    {second} = {second_value}
    {tensor}[{first}, {second}{stage_coordinate}] = cutlass.TFloat32(cutlass.Uint32(cute.arch.cvt_f32_tf32({copy_destination}[{first}, {second}{stage_coordinate}])).bitcast(cutlass.Float32))
""")
                )
            vector_body = plan.emit_to_aligned_smem(
                copy_destination,
                tuple(
                    ast.Name(id=name, ctx=ast.Load())
                    for name in (
                        (*indices, f"{prefix}_stage_write") if stages > 1 else indices
                    )
                ),
                fresh,
                preserve_pointer_swizzle=native is not None or tf32,
            )
            slot_guard = ast.parse(f"{flat} < {elements}", mode="eval").body
            if copy_strategy == "async_cached":
                prefix_end = next(
                    (
                        i
                        for i, statement in enumerate(vector_body)
                        if not isinstance(statement, ast.Assign)
                    ),
                    len(vector_body),
                )
                assert isinstance(copy_loop.target, ast.Name)
                cache_plan = plan_scalar_recipe_cache(
                    cast("list[ast.Assign]", vector_body[:prefix_end]),
                    vector_body[prefix_end:],
                    slot_name=copy_loop.target.id,
                    slot_count=slots,
                    slot_prelude=cast("list[ast.Assign]", copy_loop.body[:3]),
                    slot_guard=slot_guard,
                    boundary_names=boundary_names
                    | reuse
                    | {site.m_offset, site.n_offset, tid},
                    varying_names={site.k_offset},
                    tensor_dtypes=tensor_dtypes,
                    readonly_tensors=read_roots,
                    max_slots=16,
                )
                if cache_plan is not None:
                    # Canonical hl.tile K loops have a positive tile-size step.
                    # Preserve their empty-range behavior when hoisting loads.
                    start, stop = (
                        (ast.Constant(value=0), bound_values[0])
                        if len(bound_values) == 1
                        else (bound_values[0], bound_values[1])
                    )
                    emission = cache_plan.emit(
                        fresh,
                        execution_guard=ast.Compare(
                            left=start, ops=[ast.Lt()], comparators=[stop]
                        ),
                    )
                    cache_prefill.extend(emission.prologue)
                    vector_body = [*emission.prefix, *vector_body[prefix_end:]]
                    copy_loop.iter = emission.slot_iterator
            copy_loop.body = [
                *copy_loop.body[:3],
                ast.If(
                    test=slot_guard,
                    body=vector_body,
                    orelse=[],
                ),
            ]
    a_publish: list[ast.stmt] = []
    if native is not None and (tmem_a := native.tmem_a) is not None:
        # This is the same already-proven scalar recipe. A direct asynchronous
        # A copy keeps its shared path; register-produced A can instead follow
        # the native TMEM store partition. Its range is disjoint from every C
        # tile, including a seed carried from the previous contraction.
        a_copy = tmem_a.emit_recipe(
            a_statements,
            a_value,
            local_m=local_m,
            local_k=local_k,
            k_offset=site.k_offset,
            static_k_extent=site.static_k_extent,
            vectorize=vector_recipe_eligible,
            tensors=tensor_facts,
            aligned_names={
                **copy_boundary_alignments,
                local_k: 8,
                site.k_offset: bk,
                site.n_offset: bn,
            },
            fresh_name=fresh,
        )
        a_publish = tmem_a.publish()
    mma_range = (
        f"cutlass.range_constexpr({bk // mma_k})"
        if stages > 1
        else f"cutlass.range({bk // mma_k}, unroll=1)"
    )
    stage_index = f", {prefix}_stage_read" if stages > 1 else ""
    compute = (
        native.compute(ast.Name(id=site.k_offset, ctx=ast.Load()), k_start)
        if native is not None
        else _statements(f"""
cute.arch.sync_threads()
for {prefix}_ki in {mma_range}:
    cute.copy({prefix}_ca, {prefix}_pa[None, None, {prefix}_ki{stage_index}], {prefix}_da[None, None, 0])
    cute.copy({prefix}_cb, {prefix}_pb[None, None, {prefix}_ki{stage_index}], {prefix}_db[None, None, 0])
    cute.gemm({prefix}_mma, {prefix}_acc, {prefix}_ra[None, None, 0], {prefix}_rb[None, None, 0], {prefix}_acc)
cute.arch.sync_threads()
""")
    )
    if stages > 1 and tf32_rounding:
        # Round only the waited read stage after publishing its async copies.
        # The original leading barrier then publishes the rounded values.
        compute = [
            *_statements("cute.arch.sync_threads()"),
            *tf32_rounding,
            *compute,
        ]
        tf32_rounding = []
    reduction_body = [
        a_copy,
        *a_publish,
        b_copy,
        *(
            _statements(
                "cute.arch.cp_async_commit_group()\ncute.arch.cp_async_wait_group(0)"
            )
            if async_copies
            else []
        ),
        *(_statements("cute.arch.sync_threads()") if tf32_rounding else []),
        *tf32_rounding,
        *compute,
    ]
    collective_k = ast.copy_location(
        ast.For(
            target=ast.Name(id=site.k_offset, ctx=ast.Store()),
            iter=ast.Call(
                func=reduction_iterator.func,
                args=bound_values,
                keywords=reduction_iterator.keywords,
            ),
            body=reduction_body,
            orelse=[],
        ),
        k_loop,
    )
    reduction_schedule = (
        pipeline_collective_loop(
            collective_k,
            [a_copy, b_copy],
            compute,
            prefix=prefix,
            start=k_start,
            stop=k_stop,
            block_k=bk,
            stages=stages,
        )
        if stages > 1
        else [collective_k]
    )
    finish = _statements(f"""
{prefix}_tc = {prefix}_thr.partition_C({prefix}_c)
cute.autovec_copy({prefix}_acc, {prefix}_tc)
cute.arch.sync_threads()
""")
    if native is not None:
        finish = native.finish(k_start, k_stop)
    replacement = _statements(
        f"{result_name} = {prefix}_c[{site.m_index} - {site.m_offset}, {site.n_index} - {site.n_offset}]"
    )[0]
    replacement_parent, replacement_index = (
        path[-1] if site.synthetic_k_lane is not None else (k_parent, k_index)
    )
    assert isinstance(replacement_parent, list)
    replacement_parent[replacement_index] = replacement
    emitted = [
        *bound_statements,
        *setup,
        *cache_prefill,
        *reduction_schedule,
        *finish,
    ]
    insertion_body[insertion_index:insertion_index] = emitted
    emitted_statement_ids.update(map(id, emitted))
    shared_results.add(f"{prefix}_c")
    if native is not None and native_lifetimes is not None:
        native_lifetimes.append(
            NativeCollectiveLifetime(
                plan=native,
                m_offset=site.m_offset,
                n_offset=site.n_offset,
                local_m=local_m,
                local_n=local_n,
                thread_dimensions=cast("tuple[int, int, int]", expected_dims),
                before_seed=before_seed,
                seed_nodes=seed_nodes,
                seed_statements=tuple(seed_statements),
                seed_value=seed_value,
                finish=tuple(finish),
                initialized=cast(
                    "ast.expr",
                    expr_from_string(
                        "True"
                        if not native.zero_seed
                        else f"({ast.unparse(k_start)}) < ({ast.unparse(k_stop)})"
                    ),
                ),
            )
        )
    return m_loop, tid, f"{prefix}_c"


def lower_collective_matmul(
    body: list[ast.stmt],
    df: DeviceFunction,
    *,
    boundary_names: set[str],
    disjoint_pairs: set[frozenset[str]],
    rename_groups: dict[str, str],
) -> list[ast.stmt]:
    if df.config.get("cute_register_chain", False):
        from .collective_register_chain_lowering import lower_register_chain

        return lower_register_chain(
            body,
            df,
            boundary_names=boundary_names,
            disjoint_pairs=disjoint_pairs,
            rename_groups=rename_groups,
        )
    if not df.cute_state.collective_mma_sites:
        if df.config.get("cute_collective_compute", "warp") == "tma_gather":
            raise exc.BackendUnsupported(
                "cute", "gathered MMA requires an admitted collective contraction"
            )
        return body
    # Phi names must be canonical before reduction live-out analysis. This is
    # the same final renaming already applied to the generated function.
    module = ast.Module(body=body, type_ignores=[])
    ast_rename(module, rename_groups)
    if df.config.get("cute_collective_compute", "warp") == "tma_gather":
        from .gathered_mma_lowering import lower_gathered_matmul

        return lower_gathered_matmul(
            body,
            df,
            boundary_names=boundary_names,
            disjoint_pairs=disjoint_pairs,
        )
    tmem_resource = None
    occupied = _bound_names(module) | boundary_names
    native_sites = [
        site
        for site in df.cute_state.collective_mma_sites
        if _native_site_enabled(site, df.config)
    ]
    if native_sites:
        if (
            df.cute_state.matmul_plan is not None
            or df.cute_state.matmul_fx_nodes
            or df.cute_state.has_tcgen05_fragment_epilogue_plan
        ):
            raise exc.BackendUnsupported(
                "cute", "independent collective and role-specialized TMEM lifetimes"
            )
        tmem_resource = CollectiveTmemResource(
            _fresh_prefix("collective_tmem", occupied, df.new_var),
            max(site.bn for site in native_sites),
        )
    operand_smem = (
        CollectiveOperandSmem(
            _fresh_prefix("collective_operands", occupied, df.new_var)
        )
        if len(df.cute_state.collective_mma_sites) > 1
        else None
    )
    lowered_sites = []
    native_lifetimes: list[NativeCollectiveLifetime] = []
    for site in df.cute_state.collective_mma_sites:
        loop, tid, shared_result = _lower_site(
            body,
            site,
            df,
            boundary_names,
            disjoint_pairs,
            tmem_resource,
            fast_math=CompileEnvironment.current().settings.fast_math,
            operand_smem=operand_smem,
            native_lifetimes=native_lifetimes
            if df.config.get("cute_collective_tmem_seed", False)
            else None,
        )
        lowered_sites.append((site, loop, tid, shared_result))
    _remove_dead_staged_operands(body, df)
    if df.config.get("cute_collective_epilogue", "scalar") != "scalar":
        from .collective_vector_epilogue import vectorize_collective_store_epilogues

        vectorize_collective_store_epilogues(
            body,
            df,
            lowered_sites,
            boundary_names=boundary_names,
            disjoint_pairs=disjoint_pairs,
        )
    if native_lifetimes:
        from .collective_tmem_seed import bridge_collective_tmem_seeds

        bridge_collective_tmem_seeds(body, native_lifetimes, df.new_var)
    if tmem_resource is not None:
        body[:0] = tmem_resource.prologue()
        body.extend(tmem_resource.epilogue())
    if operand_smem is not None:
        body[:0] = operand_smem.prologue()
    if any(
        isinstance(node, ast.Name) and node.id == MARKER
        for statement in body
        for node in ast.walk(statement)
    ):
        raise exc.BackendUnsupported("cute", "unlowered collective MMA marker")
    # Record completion only after every admitted marker has been replaced.
    # Universal MMA and native wrapper plans retain their layout guards.
    df.cute_state.collective_mma_static_layouts = (
        df.config.get("cute_collective_compute", "warp") == "warp"
        and not native_sites
        and df.cute_state.matmul_plan is None
        and not df.cute_state.matmul_fx_nodes
        and not df.cute_state.has_tcgen05_fragment_epilogue_plan
    )
    df.namespace._used_names.update(_bound_names(ast.Module(body, [])))
    return body
