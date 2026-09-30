"""Code generation for proven materialized regions of one BoundKernel.

Every region gets a single-root DeviceFunction, but all regions keep the owning
environment and guards. Proven pointwise regions can override the owning PID
schedule and omit the native-matmul-only compile option. Graph IDs and
memory-hint positions stay global.
Generated stage modules are loaded once through PyCodeCache; the outer source
contains their full source and allocates shared outputs once before launching.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...exc import InvalidConfig
from ...language import _tracing_ops
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..device_ir import KernelPhase
from ..device_ir import RootGraphInfo
from ..host_function import HostFunction
from ..output_header import get_needed_import_lines
from .active_blocks import active_block_ids
from .tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from .materialized_fission import MaterializedFissionPlan


_MEMORY_COUNTERS = (
    "atomic_op_index",
    "device_load_index",
    "device_load_cache_modifier_index",
    "device_store_index",
    "device_store_cache_modifier_index",
    "device_memory_op_index",
)


_WARP_ROLE_SELECTOR = "cute.arch.make_warp_uniform(cute.arch.warp_idx())"


def _is_griddepcontrol_wait(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and ast.unparse(stmt.value.func) == "cute.arch.griddepcontrol_wait"
    )


def _loads_through_atom(node: ast.AST, atom: str) -> bool:
    """Whether ``node`` issues a TMA load through ``atom``.

    ``cute.copy(atom, ...)`` is the one statement that reads operand memory
    through a TMA atom; ``tma_partition`` and ``prefetch_descriptor`` also
    name the atom but read layout and descriptor metadata only.
    """
    return any(
        isinstance(call, ast.Call)
        and ast.unparse(call.func) == "cute.copy"
        and bool(call.args)
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == atom
        for call in ast.walk(node)
    )


def _dependent_load_before_wait(body: list[ast.stmt], atom: str) -> bool:
    """Whether a load through ``atom`` in ``body`` runs before any wait.

    Statements execute in source order, so a ``griddepcontrol_wait`` at the
    top level of ``body`` covers every later statement of that body.  A wait
    nested in a branch or loop body covers only the rest of that body: a load
    after the nested block still needs a wait of its own.  Any other compound
    statement holding a load fails closed.
    """
    for stmt in body:
        if _is_griddepcontrol_wait(stmt):
            return False
        if isinstance(stmt, (ast.If, ast.While, ast.For)):
            if any(
                _dependent_load_before_wait(nested, atom)
                for nested in (stmt.body, stmt.orelse)
            ):
                return True
            continue
        if _loads_through_atom(stmt, atom):
            return True
    return False


def _tma_role_waits_before_dependent_loads(device: ast.FunctionDef, atom: str) -> bool:
    """Whether the TMA-load role waits on the producer grid ahead of ``atom``.

    Every load through the dependent operand's atom must sit in one warp-role
    block ``if <warp selector> == <n>:`` of the device function, and inside
    it, in the one-shot and the ``while ... is_valid_tile`` layouts alike, a
    ``griddepcontrol_wait`` must dominate the first such load.
    """
    roles = [stmt for stmt in device.body if _loads_through_atom(stmt, atom)]
    if len(roles) != 1:
        return False
    role = roles[0]
    if not (
        isinstance(role, ast.If)
        and not role.orelse
        and isinstance(role.test, ast.Compare)
        and ast.unparse(role.test.left) == _WARP_ROLE_SELECTOR
        and len(role.test.ops) == 1
        and isinstance(role.test.ops[0], ast.Eq)
    ):
        return False
    return not _dependent_load_before_wait(role.body, atom)


def _stage_config(
    config: Config,
    plan: MaterializedFissionPlan,
    stage_index: int,
    stage: HostFunction | None = None,
    *,
    releases_dependents_at_entry: bool = False,
) -> Config:
    pointwise = stage_index in plan.pointwise_region_indices
    project_ffi = pointwise and config.get(TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY) is True
    has_region_stages = any(
        key in config.config
        for key in ("tcgen05_region_ab_stages", "tcgen05_region_c_stages")
    )
    fanout = config.get("tcgen05_epilogue_fanout") == "shared"
    # The stage that releases its dependents at entry (the materialized-PDL
    # producer) has to stay an ordinary launch.  Launched as a programmatic
    # dependent itself it would start, and release the consumer, while the
    # kernels ahead of it still run, and the consumer's TMA role loads the
    # ordinary operand ahead of its own wait.  ``cute_pdl`` never reaches it.
    drop_pdl = releases_dependents_at_entry and config.get("cute_pdl") is True
    if (
        "cute_pointwise_pid_type" not in config.config
        and not project_ffi
        and not has_region_stages
        and not fanout
        and not drop_pdl
    ):
        return config
    result = copy.deepcopy(config)
    if drop_pdl:
        result.config.pop("cute_pdl")
    for kind in ("ab", "c"):
        key = f"tcgen05_region_{kind}_stages"
        region_stages = result.config.pop(key, None)
        if region_stages is not None:
            assert (
                isinstance(region_stages, list)
                and len(region_stages) == plan.region_count
            )
            count = region_stages[stage_index]
            assert type(count) is int and 0 <= count <= 16
            if count:
                result.config[f"tcgen05_{kind}_stages"] = count
    if fanout:
        assert stage is not None
        state = CompileEnvironment.current().config_spec._cute_tcgen05_config
        active = stage.device_ir.codegen_active_block_ids
        assert active is not None
        owns_pair = any(
            set(pair.block_ids).issubset(active) for pair in state.epilogue_fanout_plans
        )
        if not owns_pair:
            result.config.pop("tcgen05_epilogue_fanout")
            result.config.pop("tcgen05_aux_load_placement", None)
            if result.config.get("tcgen05_c_acquire_placement") == "before_store":
                result.config.pop("tcgen05_c_acquire_placement")
    # This option belongs to the bundle, not either emitted stage. In
    # particular, the native consumer keeps its original scheduling config.
    pid_type = result.config.pop("cute_pointwise_pid_type", "inherit")
    if pointwise and pid_type == "flat":
        result.config["pid_type"] = "flat"
        # Match ConfigSpec's non-persistent PID policy for these two options.
        result.config.pop("num_sm_multiplier", None)
        result.config.pop("maxnreg", None)
    if project_ffi:
        result.config[TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY] = False
    return result


def _region_graph_ids(graphs: list[GraphInfo], root_id: int) -> frozenset[int]:
    pending = [root_id]
    result: set[int] = set()
    while pending:
        graph_id = pending.pop()
        assert graph_id not in result, "fission regions cannot share nested graphs"
        result.add(graph_id)
        for node in graphs[graph_id].graph.nodes:
            assert node.target is not _tracing_ops._if
            if _tracing_ops.is_for_loop_target(node.target):
                child = node.args[0]
                assert isinstance(child, int)
                pending.append(child)
    return frozenset(result)


def _empty_graph(graph_id: int) -> RootGraphInfo:
    graph = torch.fx.Graph()
    graph.output(None)
    return RootGraphInfo(graph_id, graph)


def _stage_host(
    host: HostFunction,
    root: ast.For,
    stage_index: int,
    graphs: list[GraphInfo],
    graph_ids: frozenset[int],
    plan: MaterializedFissionPlan,
) -> HostFunction:
    assert isinstance(root, ExtendedAST)
    copied_root = root.copy()
    copied_root._root_id = 0
    definition = dataclasses.replace(
        host.definition,
        name=f"{host.name}__region_{stage_index}",
        body=[
            *host.body[: plan.root_index],
            cast("ast.For", copied_root),
            *host.body[plan.root_index + plan.region_count :],
        ],
    )
    stage = HostFunction(definition, host.location)
    stage.compiler_state = host.compiler_state
    stage.local_types = host.local_types
    ir = copy.copy(host.device_ir)
    stage.device_ir = ir
    ir.host_function = stage
    ir.root_ids = [host.device_ir.root_ids[stage_index]]
    ir.grid_block_ids = [host.device_ir.grid_block_ids[stage_index]]
    ir.task_families = [host.device_ir.task_families[stage_index]]
    ir.phases = [KernelPhase([0], [cast("ast.For", copied_root)])]
    ir.implicit_dependency_starts = frozenset()
    ir.tile_dependency_graph = None
    ir.graphs = [
        graph if graph.graph_id in graph_ids else _empty_graph(graph.graph_id)
        for graph in graphs
    ]
    root_info = ir.graphs[ir.root_ids[0]]
    assert isinstance(root_info, RootGraphInfo)
    ir.graphs[ir.root_ids[0]] = dataclasses.replace(root_info, phase_index=0)
    with stage:
        ir.codegen_active_block_ids = active_block_ids(
            ir.graphs, ir.grid_block_ids[0], CompileEnvironment.current()
        )
    return stage


def _capture_allocations(host_def: ast.FunctionDef, names: tuple[str, ...]) -> None:
    """Replace generated stage allocations with required keyword captures."""
    captures = set(names)
    host_def.body = [
        stmt
        for stmt in host_def.body
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in captures
        )
    ]
    for stmt in host_def.body:
        if isinstance(stmt, ast.Return):
            stmt.value = None
    host_def.args.kwonlyargs.extend(ast.arg(arg=name) for name in names)
    host_def.args.kw_defaults.extend(None for name in names)


def _module_source(module: ast.Module) -> str:
    # Unparse strips compiler-owned AST metadata without copying FakeTensors.
    return (
        "from __future__ import annotations\n\n"
        + ast.unparse(ast.fix_missing_locations(module))
        + "\n"
    )


def generate_materialized_fission(
    func: HostFunction,
    config: Config,
    emit_repro_caller: bool,
    plan: MaterializedFissionPlan,
    *,
    store_transform: Callable[..., ast.AST] | None = None,
    load_transform: Callable[..., ast.AST] | None = None,
    extra_params: list[str] | None = None,
) -> ast.Module:
    from ..generate_ast import VloopSinkNotApplied
    from ..generate_ast import emit_main_def
    from ..generate_ast import generate_ast
    from ..generate_ast import vloop_sink_off_config

    env = CompileEnvironment.current()
    roots = func.body[plan.root_index : plan.root_index + plan.region_count]
    assert len(func.device_ir.root_ids) == len(roots) == plan.region_count
    assert all(
        isinstance(root, ast.For)
        and isinstance(root, ExtendedAST)
        and root._loop_type is LoopType.GRID
        for root in roots
    )
    graphs = func.device_ir.build_codegen_graphs(config)
    memory_counters = dict.fromkeys(_MEMORY_COUNTERS, 0)
    prefix: list[ast.AST] = []
    resolved_plans: list[dict[str, object]] = []
    module_loads: list[ast.stmt] = []
    calls: list[ast.AST] = []
    kernels: list[ast.expr] = []
    generated_return: ast.Return | None = None
    claimed_graphs: set[int] = set()
    used_names = {
        node.id
        for stmt in func.body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
    }
    used_names.update(arg.arg for arg in func.args.args)
    used_names.update(extra_params or ())

    pdl_roots = env.config_spec._cute_tcgen05_config.materialized_operand_pdl_roots
    if not config.get("tcgen05_materialized_pdl", False):
        pdl_roots = None

    def generate_stage(
        stage: HostFunction,
        stage_config: Config,
        stage_index: int,
        *,
        releases_dependents_at_entry: bool,
    ) -> ast.Module:
        return generate_ast(
            stage,
            _stage_config(
                stage_config,
                plan,
                stage_index,
                stage,
                releases_dependents_at_entry=releases_dependents_at_entry,
            ),
            False,
            store_transform=store_transform,
            load_transform=load_transform,
            extra_params=extra_params,
            _codegen_graphs=stage.device_ir.graphs,
            _memory_counters=memory_counters,
            _host_prefix=prefix if stage_index == 0 else None,
        )

    for stage_index, root in enumerate(roots):
        assert isinstance(root, ast.For)
        graph_ids = _region_graph_ids(graphs, func.device_ir.root_ids[stage_index])
        assert not claimed_graphs.intersection(graph_ids)
        claimed_graphs.update(graph_ids)
        stage = _stage_host(func, root, stage_index, graphs, graph_ids, plan)
        root_id = func.device_ir.root_ids[stage_index]
        releases_at_entry = pdl_roots is not None and root_id == pdl_roots[0]
        try:
            module = generate_stage(
                stage,
                config,
                stage_index,
                releases_dependents_at_entry=releases_at_entry,
            )
        except VloopSinkNotApplied:
            # ``cute_vloop_sink`` shaped this stage's thread layout but sank
            # nothing.  The knob must not change the code by itself, so
            # regenerate exactly the knob-off stage, from graphs the knob-on
            # layout planning has not annotated.
            off_config = vloop_sink_off_config(config)
            stage = _stage_host(
                func,
                root,
                stage_index,
                func.device_ir.build_codegen_graphs(off_config),
                graph_ids,
                plan,
            )
            module = generate_stage(
                stage,
                off_config,
                stage_index,
                releases_dependents_at_entry=releases_at_entry,
            )
        if releases_at_entry:
            # Every launched CTA must release the dependent grid, including
            # persistent CTAs with no work. Keep this ahead of all entry guards.
            device = next(
                stmt
                for stmt in module.body
                if isinstance(stmt, ast.FunctionDef)
                and stmt.name == f"_helion_{stage.name}"
            )
            device.body.insert(
                0, statement_from_string("cute.arch.griddepcontrol_launch_dependents()")
            )
            # Releasing ahead of everything else is only sound for an
            # ordinary launch, one that started after its predecessors
            # completed; ``_stage_config`` keeps ``cute_pdl`` away from this
            # stage and no native plan of a pointwise stage launches with it.
            assert not any(
                isinstance(stmt, ast.Assign)
                and ast.unparse(stmt.targets[0])
                == f"{device.name}._helion_cute_use_pdl"
                for stmt in module.body
            )
        if pdl_roots is not None and root_id == pdl_roots[1]:
            ab_plans = [
                item
                for item in env.cute_resolved_wrapper_plans
                if item["kind"] == "tcgen05_ab_tma"
            ]
            device = next(
                stmt
                for stmt in module.body
                if isinstance(stmt, ast.FunctionDef)
                and stmt.name == f"_helion_{stage.name}"
            )
            # The consumer must launch with the programmatic dependency AND its
            # TMA role must wait on it before its first load of the
            # materialized operand (the ordinary operand's initial loads may
            # precede the wait); the static persistent role-local builder
            # emits that wait for the paired TWO family and, on request, for
            # the plain one-CTA family.
            consumer_ok = False
            if len(ab_plans) == 1:
                ab_plan = ab_plans[0]
                atom_a, _tensor_a, atom_b, _tensor_b = cast(
                    "list[str]", ab_plan["kernel_args"]
                )
                dependent_atom = atom_b if pdl_roots.dependent_side == "rhs" else atom_a
                consumer_ok = (
                    ab_plan.get("use_pdl") is True
                    and ab_plan["cluster_n"] == 1
                    and (
                        (
                            ab_plan.get("use_2cta_instrs", ab_plan["bm"] == 256)
                            and ab_plan["cluster_m"] == 2
                        )
                        or (
                            not ab_plan.get("use_2cta_instrs", False)
                            and ab_plan["cluster_m"] == 1
                        )
                    )
                    and _tma_role_waits_before_dependent_loads(device, dependent_atom)
                )
            if not consumer_ok:
                raise InvalidConfig(
                    "materialized operand PDL requires the emitted native TMA "
                    "consumer whose TMA role waits on the dependency before its "
                    "first load of the materialized operand"
                )
        resolved_plans.extend(env.cute_resolved_wrapper_plans)
        host_def = next(
            stmt
            for stmt in module.body
            if isinstance(stmt, ast.FunctionDef) and stmt.name == stage.name
        )
        # Retain the exact generated return expression before captures are
        # removed. All stage host functions originally saw the same return.
        returns = [stmt for stmt in host_def.body if isinstance(stmt, ast.Return)]
        assert len(returns) == 1
        if generated_return is None:
            assert isinstance(returns[0], ExtendedAST)
            generated_return = cast("ast.Return", returns[0].copy())
        _capture_allocations(host_def, plan.materialized_names)
        module_name = f"_helion_cute_region_{stage_index}_module"
        while module_name in used_names:
            module_name += "_"
        used_names.add(module_name)
        stage_source = _module_source(module)
        load_statement = ast.Assign(
            targets=[ast.Name(id=module_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="PyCodeCache", ctx=ast.Load()),
                    attr="load",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=stage_source)],
                keywords=[],
            ),
        )
        calls.append(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=module_name, ctx=ast.Load()),
                        attr=stage.name,
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Name(id=arg.arg, ctx=ast.Load())
                        for arg in host_def.args.args
                    ],
                    keywords=[
                        ast.keyword(arg=name, value=ast.Name(id=name, ctx=ast.Load()))
                        for name in ("_launcher", *plan.materialized_names)
                    ],
                )
            )
        )
        kernels.append(
            ast.Attribute(
                value=ast.Name(id=module_name, ctx=ast.Load()),
                attr=f"_helion_{stage.name}",
                ctx=ast.Load(),
            )
        )
        source_hash_statement = ast.Assign(
            targets=[
                ast.Attribute(
                    value=kernels[-1],
                    attr="_helion_cute_source_hash",
                    ctx=ast.Store(),
                )
            ],
            value=ast.Constant(value=hashlib.sha256(stage_source.encode()).hexdigest()),
        )
        module_loads.extend((load_statement, source_hash_statement))
    env.cute_resolved_wrapper_plans = resolved_plans
    assert generated_return is not None
    for name in plan.materialized_names:
        assert (
            sum(
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == name
                for stmt in prefix
            )
            == 1
        ), f"shared allocation {name} must be emitted exactly once"
    host_def = func.codegen_function_def(
        [*prefix, *calls, generated_return], extra_params=extra_params
    )
    metadata = ast.Assign(
        targets=[
            ast.Attribute(
                value=ast.Name(id=func.name, ctx=ast.Load()),
                attr="_helion_cute_kernels",
                ctx=ast.Store(),
            )
        ],
        value=ast.Tuple(elts=kernels, ctx=ast.Load()),
    )
    result = ast.Module(
        body=[
            *func.codegen_imports(),
            statement_from_string("from torch._inductor.codecache import PyCodeCache"),
            *module_loads,
            host_def,
            metadata,
            *(
                [func.codegen_call_function(), emit_main_def()]
                if emit_repro_caller
                else []
            ),
        ],
        type_ignores=[],
    )
    existing = {
        ast.unparse(stmt)
        for stmt in result.body
        if isinstance(stmt, (ast.Import, ast.ImportFrom))
    }
    result.body[:0] = [
        statement_from_string(line)
        for line in get_needed_import_lines(result)
        if line not in existing
    ]
    return result
