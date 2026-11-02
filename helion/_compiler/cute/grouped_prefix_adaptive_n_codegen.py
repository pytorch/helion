"""Compose two proved typed MN contractions over one live row-prefix worklist.

Each exclusive core comes from ordinary native lowering. Only its scheduler
transport and shared allocation addresses change: operand, accumulator and
output pipeline payloads stay in their original order.
"""

from __future__ import annotations

import ast
import builtins
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from textwrap import dedent
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ...exc import InvalidConfig
from ...runtime.config import Config
from .grouped_prefix_adaptive_n import GroupedPrefixAdaptiveNProfile
from .grouped_prefix_adaptive_n import GroupedPrefixNBranch
from .grouped_prefix_adaptive_n import grouped_prefix_adaptive_n_input_domain
from .tcgen05_constants import TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY
from .tcgen05_constants import TCGEN05_GROUPED_PREFIX_ADAPTIVE_N_CONFIG_KEY

if TYPE_CHECKING:
    from ..host_function import HostFunction


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise InvalidConfig(f"grouped prefix-adaptive N: {reason}")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Expr):
        node = node.value
    return ast.unparse(node.func) if isinstance(node, ast.Call) else None


def _references(node: ast.AST, prefix: str) -> bool:
    return any(
        isinstance(value, ast.Name) and value.id.startswith(prefix)
        for value in ast.walk(node)
    )


def _integer(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if (
        isinstance(node, ast.Call)
        and _call_name(node) in ("cutlass.Int32", "cutlass.Int64")
        and len(node.args) == 1
    ):
        return _integer(node.args[0])
    raise InvalidConfig("grouped prefix-adaptive N requires static allocation sizes")


def _assignment(tree: ast.AST, name: str) -> ast.expr:
    values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    _require(len(values) == 1, f"missing or repeated native binding {name}")
    return values[0]


def _one_assignment_for_call(tree: ast.AST, call: str) -> tuple[str, ast.expr]:
    values = [
        (node.targets[0].id, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and _call_name(node.value) == call
    ]
    _require(len(values) == 1, f"expected one native {call} binding")
    return values[0]


def _plans(module: ast.Module) -> tuple[ast.Assign, list[dict[str, Any]]]:
    values = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_helion_cute_wrapper_plans"
    ]
    _require(len(values) == 1, "requires one native grouped launch plan")
    node = values[0]
    value = ast.literal_eval(node.value)
    _require(
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(plan, dict) for plan in value)
        and [plan.get("kind") for plan in value]
        == ["tcgen05_grouped_static_persistent", "tcgen05_ab_tma", "tcgen05_d_tma"],
        "requires one immutable grouped contraction and one output",
    )
    return node, cast("list[dict[str, Any]]", value)


@dataclass(frozen=True)
class _ArenaView:
    name: str
    dtype: str
    offset: int
    size: int
    alignment: int
    common: bool

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass
class _NativeCore:
    module: ast.Module
    kernel: ast.FunctionDef
    active: ast.If
    prefix: list[ast.stmt]
    plan_assignment: ast.Assign
    plans: list[dict[str, Any]]
    views: list[_ArenaView]
    branch: GroupedPrefixNBranch
    d_signature: tuple[tuple[str, ...], tuple[str, ...]]


def _query_native_layouts(
    kernel: ast.FunctionDef,
    plans: list[dict[str, Any]],
    profile: GroupedPrefixAdaptiveNProfile,
) -> tuple[dict[str, int], tuple[tuple[str, ...], tuple[str, ...]]]:
    """Inspect the emitted SDK layouts without compiling or initializing CUDA."""
    import cutlass
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute

    from ...runtime.cute.launcher import _append_cute_wrapper_plan
    from ._mlir_compat import ir

    grouped, ab, output = plans
    m, n, k, groups = (
        grouped[key] for key in ("m_size", "n_size", "k_total_size", "group_count")
    )
    scope: dict[str, Any] = {"cutlass": cutlass, "cute": cute}
    names_and_values = []
    mma, value = _one_assignment_for_call(
        kernel, "cutlass.utils.blackwell_helpers.make_trivial_tiled_mma"
    )
    names_and_values.append((mma, value))
    a_layout, value = _one_assignment_for_call(
        kernel, "cutlass.utils.blackwell_helpers.make_smem_layout_a"
    )
    names_and_values.append((a_layout, value))
    b_layout, value = _one_assignment_for_call(
        kernel, "cutlass.utils.blackwell_helpers.make_smem_layout_b"
    )
    names_and_values.append((b_layout, value))
    fragment, value = _one_assignment_for_call(kernel, f"{mma}.make_fragment_C")
    names_and_values.append((fragment, value))
    columns, value = _one_assignment_for_call(
        kernel, "cutlass.utils.get_num_tmem_alloc_cols"
    )
    names_and_values.append((columns, value))
    d_layout, d_value = _one_assignment_for_call(
        kernel, "cutlass.utils.blackwell_helpers.make_smem_layout_epi"
    )
    _require(
        isinstance(d_value, ast.Call) and len(d_value.args) == 4, "unexpected D layout"
    )
    assert isinstance(d_value, ast.Call)
    _require(isinstance(d_value.args[2], ast.Name), "requires a native epilogue tile")
    assert isinstance(d_value.args[2], ast.Name)
    tile = d_value.args[2].id
    names_and_values.extend(((tile, _assignment(kernel, tile)), (d_layout, d_value)))
    payload: dict[str, int] = {}
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp("grouped_prefix_layout_contract", ([], []))
            with ir.InsertionPoint(function.add_entry_block()):
                for name, expression in names_and_values:
                    scope[name] = eval(
                        compile(
                            ast.Expression(expression),
                            "<native grouped layout>",
                            "eval",
                        ),
                        scope,
                    )
                _require(
                    int(scope[columns]) == profile.tmem_columns,
                    "native TMEM ownership disagrees with the selected profile",
                )
                _require(
                    tuple(int(cute.size(mode)) for mode in scope[tile])
                    == profile.common_d_tile,
                    "requires the selected common native output tile",
                )
                for node in ast.walk(kernel):
                    if (
                        isinstance(node, ast.Assign)
                        and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)
                        and isinstance(node.value, ast.Call)
                        and _call_name(node.value) == "cute.arch.alloc_smem"
                        and len(node.value.args) >= 2
                        and isinstance(node.value.args[1], ast.Call)
                        and _call_name(node.value.args[1]) == "cute.cosize"
                    ):
                        count = node.value.args[1]
                        _require(
                            len(count.args) == 1
                            and isinstance(count.args[0], ast.Attribute)
                            and isinstance(count.args[0].value, ast.Name)
                            and count.args[0].attr == "outer",
                            "requires an ordinary staged native layout",
                        )
                        outer = cast("ast.Attribute", count.args[0])
                        layout_name = cast("ast.Name", outer.value).id
                        _require(
                            layout_name in (a_layout, b_layout, d_layout),
                            "unexpected shared payload",
                        )
                        payload[node.targets[0].id] = (
                            int(cute.cosize(scope[layout_name].outer)) * 2
                        )
                # Build the same host D atom as the launcher, including its
                # actual N-mode permutation and shifted rank-three tensor.
                dtype = getattr(
                    cutlass, output["output_dtype"].removeprefix("cutlass.")
                )
                index = output["d_idx"]
                scope[f"arg{index}"] = cute.make_tensor(
                    cute.make_ptr(
                        dtype, 1 << 36, cute.AddressSpace.gmem, assumed_align=16
                    ),
                    cute.make_layout((m, n), stride=(n, 1)),
                )
                for dim, (extent, stride) in enumerate(((m, n), (n, 1))):
                    scope[f"arg{index}_shape{dim}"] = extent
                    scope[f"arg{index}_stride{dim}"] = stride
                body: list[str] = []
                call_args: list[str] = []
                _append_cute_wrapper_plan(body, call_args, output)
                exec(
                    compile(
                        ast.parse(dedent("\n".join(body))),
                        "<native grouped D atom>",
                        "exec",
                    ),
                    scope,
                )
                atom = scope[output["kernel_args"][0]]
                signature = (
                    tuple(str(value.type) for value in atom.__extract_mlir_values__()),
                    tuple(str(value) for value in atom.__extract_mlir_attributes__()),
                )
                func.ReturnOp([])
            _require(module.operation.verify(), "native layout IR failed verification")
    _require(
        ab["input_dtype"] in ("cutlass.BFloat16", "cutlass.Float16")
        and grouped_prefix_adaptive_n_input_domain(
            groups, m, n, k, source_m_tile=profile.row_tile
        ),
        "requires the signed native input domain",
    )
    return payload, signature


def _load_native(
    module: ast.Module,
    branch: GroupedPrefixNBranch,
    profile: GroupedPrefixAdaptiveNProfile,
) -> _NativeCore:
    # Completed ordinary lowering no longer needs ExtendedAST provenance.
    tree = ast.parse(ast.unparse(module))
    plan_assignment, plans = _plans(tree)
    grouped, ab, output = plans
    kernels = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(value) == "cute.kernel" for value in node.decorator_list)
    ]
    _require(len(kernels) == 1, "requires a single complete native kernel")
    kernel = kernels[0]
    target = cast("ast.Attribute", plan_assignment.targets[0])
    _require(
        isinstance(target.value, ast.Name) and target.value.id == kernel.name,
        "plan/kernel mismatch",
    )
    _require(
        grouped.get("scheduler_mode") == "device_group_search"
        and grouped.get("balanced_workers") is True
        and grouped.get("bm") == profile.row_tile
        and grouped.get("bn") == branch.n
        and grouped.get("bk") == 64
        and grouped.get("source_m_tile") == profile.row_tile
        and grouped.get("cluster_m") == 2
        and grouped.get("cluster_n") == 1
        and grouped.get("l2_swizzle_size") == 8
        and grouped.get("device_panel_swizzle") is True
        and grouped.get("device_layout_kind") == "offsets"
        and grouped.get("immutable_ab_tensormaps") is True
        and grouped.get("bounded_output_shift") is True
        and ab.get("full_allocation_a") is True
        and ab.get("full_allocation_b") is True
        and ab.get("ab_stage_count") == branch.ab_stages
        and output.get("bounded_output_shift") is True
        and output.get("c_stage_count") == branch.c_stages,
        "ordinary lowering did not prove the requested immutable paired recipe",
    )
    _require(
        (
            profile.row_tile == 128
            and grouped.get("tcgen05_grouped_l2_swizzle_alternate") is None
            and grouped.get("tcgen05_grouped_l2_swizzle_mn_ratio") is None
        )
        or (
            profile.row_tile == 256
            and grouped.get("tcgen05_grouped_l2_swizzle_alternate") == 4
            and grouped.get("tcgen05_grouped_l2_swizzle_mn_ratio") == 1
        ),
        "native per-group panel policy disagrees with the selected profile",
    )
    _require(
        grouped["n_size"] % branch.n == 0 or ab.get("full_allocation_b_rank3") is True,
        "a feature-tail core requires its native expert-bounded B descriptor",
    )
    _require(
        any(
            isinstance(value, ast.Call)
            and any(
                keyword.arg == "block" and ast.literal_eval(keyword.value) == (32, 8, 1)
                for keyword in value.keywords
            )
            for value in ast.walk(tree)
        ),
        "requires the native full-warp role layout",
    )
    active_scopes = [
        node
        for node in kernel.body
        if isinstance(node, ast.If)
        and "tcgen05_grouped_total_tiles" in ast.unparse(node.test)
    ]
    _require(
        len(active_scopes) == 1 and kernel.body[-1] is active_scopes[0],
        "unexpected native active scope",
    )
    active = active_scopes[0]
    balance_indices = [
        index
        for index, node in enumerate(kernel.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("tcgen05_balance_partial")
    ]
    _require(len(balance_indices) == 1, "missing native metadata boundary")
    payload, signature = _query_native_layouts(kernel, plans, profile)
    allocations = sorted(
        (
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.Assign)
            and _call_name(node.value) == "cute.arch.alloc_smem"
        ),
        key=lambda node: node.lineno,
    )
    views: list[_ArenaView] = []
    cursor = 0
    dtype_bytes = {
        "cutlass.Int32": 4,
        "cutlass.Int64": 8,
        "cutlass.BFloat16": 2,
        "cutlass.Float16": 2,
    }
    for node in allocations:
        _require(
            len(node.targets) == 1 and isinstance(node.targets[0], ast.Name),
            "unexpected native allocation target",
        )
        name = cast("ast.Name", node.targets[0]).id
        call = cast("ast.Call", node.value)
        dtype = ast.unparse(call.args[0])
        _require(dtype in dtype_bytes, "unsupported native shared element type")
        size = (
            payload[name]
            if name in payload
            else _integer(call.args[1]) * dtype_bytes[dtype]
        )
        alignment = next(
            (
                _integer(keyword.value)
                for keyword in call.keywords
                if keyword.arg == "alignment"
            ),
            dtype_bytes[dtype],
        )
        _require(alignment > 0 and size > 0, "invalid native shared extent")
        cursor = (cursor + alignment - 1) // alignment * alignment
        views.append(
            _ArenaView(
                name, dtype, cursor, size, alignment, node.lineno < active.lineno
            )
        )
        cursor += size
    _require(
        len(views) == 11
        and cursor == branch.arena_bytes
        and sorted(payload.values())
        == sorted((branch.a_bytes, branch.b_bytes, branch.d_bytes))
        and views[-1].offset == branch.d_offset
        and any(
            view.offset == branch.a_offset and view.size == branch.a_bytes
            for view in views
        )
        and any(
            view.offset == branch.b_offset and view.size == branch.b_bytes
            for view in views
        )
        and any(
            view.offset == branch.ab_barriers_offset
            and view.size == 16 * branch.ab_stages
            for view in views
        ),
        "SDK allocation layout disagrees with the complete arena budget",
    )
    return _NativeCore(
        tree,
        kernel,
        active,
        kernel.body[: balance_indices[0]],
        plan_assignment,
        plans,
        views,
        branch,
        signature,
    )


class _ArenaRewriter(ast.NodeTransformer):
    def __init__(self, views: list[_ArenaView], arena: str) -> None:
        self.views = {view.name: view for view in views}
        self.arena = arena
        self.replaced: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node
        view = self.views.get(node.targets[0].id)
        if view is None:
            return node
        _require(
            _call_name(node.value) == "cute.arch.alloc_smem",
            "shared binding was overwritten",
        )
        node.value = ast.parse(
            f"cute.recast_ptr({self.arena} + {view.offset}, dtype={view.dtype})",
            mode="eval",
        ).body
        self.replaced.add(view.name)
        return node


def _is_role(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and any(
        isinstance(value, ast.While)
        and isinstance(value.test, ast.Name)
        and value.test.id.startswith("tcgen05_role_local_")
        for value in node.body
    )


def _role_payload(loop: ast.While) -> list[ast.stmt]:
    starts = [
        index
        for index, node in enumerate(loop.body)
        if _call_name(node) == "tcgen05_sched_pipeline_consumer_state.advance"
    ]
    waits = [
        index
        for index, node in enumerate(loop.body)
        if _call_name(node) == "tcgen05_sched_pipeline.consumer_wait"
    ]
    _require(
        len(starts) == len(waits) == 1 and starts[0] < waits[0],
        "unexpected native scheduler phases",
    )
    return loop.body[starts[0] + 1 : waits[0]]


def _replace_role_schedulers(
    core: ast.FunctionDef,
    *,
    decoder: str,
    total: str,
    groups: int,
    n: int,
    width: int,
    stride: str = "tcgen05_grouped_virtual_grid[2]",
) -> None:
    updated: list[ast.stmt] = []
    role_count = 0
    for statement in core.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id.startswith("tcgen05_sched_pipeline")
            for target in statement.targets
        ):
            continue
        if isinstance(statement, ast.If) and any(
            _call_name(value)
            == "cutlass.utils.StaticPersistentGroupTileScheduler.create"
            for value in ast.walk(statement)
        ):
            continue
        if not _is_role(statement):
            updated.append(statement)
            continue
        role = cast("ast.If", statement)
        loop_indices = [
            index
            for index, value in enumerate(role.body)
            if isinstance(value, ast.While)
        ]
        _require(len(loop_indices) == 1, "requires one local loop per native role")
        index = loop_indices[0]
        loop = cast("ast.While", role.body[index])
        _require(
            all(
                _call_name(value)
                in (
                    "tcgen05_sched_pipeline.consumer_release",
                    "tcgen05_sched_pipeline_consumer_state.advance",
                    "cute.arch.sync_warp",
                )
                for value in role.body[index + 1 :]
            ),
            "unexpected role tail outside native work payload",
        )
        payload = deepcopy(_role_payload(loop))
        decoded = {
            value.targets[0].id
            for value in loop.body
            if isinstance(value, ast.Assign)
            and isinstance(value.targets[0], ast.Name)
            and _references(value.value, "tcgen05_work_tile_smem")
        }
        linear = f"{total}_linear{role_count}"
        preparation = ast.parse(
            "tcgen05_grouped_cta_tile_idx_m, tcgen05_grouped_cta_tile_idx_n, "
            "tcgen05_grouped_group_idx, tcgen05_grouped_problem_m = "
            f"{decoder}({linear}, tcgen05_grouped_problem_sizes, {(n + width - 1) // width}, {groups})"
        ).body
        if "tcgen05_grouped_problem_n" in decoded:
            preparation += ast.parse(
                f"tcgen05_grouped_problem_n = cutlass.Int32({n})"
            ).body
        if "tcgen05_grouped_global_m_start" in decoded:
            preparation += ast.parse(
                "tcgen05_grouped_global_m_start = tcgen05_grouped_starts[tcgen05_grouped_group_idx]"
            ).body
        retained = [
            value
            for value in role.body[:index]
            if not _references(value, "tcgen05_sched_pipeline")
            and not _references(value, "tcgen05_work_tile_smem")
            and not _references(value, "tcgen05_role_local_")
            and _call_name(value)
            not in ("cute.arch.fence_view_async_shared", "cute.arch.sync_warp")
        ]
        new_loop = ast.While(
            test=ast.parse(f"{linear} < {total}", mode="eval").body,
            body=[
                *preparation,
                *payload,
                ast.parse(f"{linear} = {linear} + {stride}").body[0],
            ],
            orelse=[],
        )
        _require(
            ast.dump(ast.Module(new_loop.body[len(preparation) : -1], []))
            == ast.dump(ast.Module(_role_payload(loop), [])),
            "native operand/ACC/C payload changed during scheduler replacement",
        )
        role.body = [
            *retained,
            ast.parse(f"{linear} = cutlass.Int32(cute.arch.block_idx()[2])").body[0],
            new_loop,
        ]
        updated.append(role)
        role_count += 1
    _require(role_count == 3, "requires one TMA, MMA and four-warp epilogue cohort")
    core.body = updated
    _require(
        not _references(core, "tcgen05_sched_pipeline")
        and not _references(core, "tcgen05_work_tile_smem")
        and "StaticPersistentGroupTileScheduler" not in ast.unparse(core),
        "scheduler transport was not completely removed",
    )


def _prefix_publication(
    prefix: list[ast.stmt],
    groups: int,
    namespace: str,
    *,
    row_tile: int = 128,
    preserve_mailbox: bool = False,
) -> list[ast.stmt]:
    retained: list[ast.stmt] = []
    for node in prefix:
        if not _references(node, "tcgen05_work_tile_smem"):
            retained.append(node)
        elif (
            preserve_mailbox
            and isinstance(node, ast.Assign)
            and _call_name(node.value) == "cute.arch.alloc_smem"
        ):
            _require(
                len(node.targets) == 1 and isinstance(node.targets[0], ast.Name),
                "unexpected native scheduler allocation",
            )
            node.targets = [ast.Name(f"{namespace}_reserved_mailbox", ast.Store())]
            retained.append(node)
    prefix = retained
    loops = [
        (index, node)
        for index, node in enumerate(prefix)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "tcgen05_grouped_metadata_group"
    ]
    _require(len(loops) == 1, "requires the native typed metadata initializer")
    index, loop = loops[0]
    prefix.insert(index, ast.parse(f"{namespace}_row_lane = cutlass.Int32(0)").body[0])
    fields = [
        node
        for node in loop.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].slice, ast.Tuple)
        and ast.unparse(node.targets[0].value) == "tcgen05_grouped_problem_sizes"
        and _integer(node.targets[0].slice.elts[-1]) == 3
    ]
    _require(len(fields) == 1, "requires the otherwise unused problem-size L field")
    loop.body.remove(fields[0])
    loop.body += ast.parse(
        f"{namespace}_row_lane = (cutlass.Int32(tcgen05_grouped_visible_extent) + "
        f"cutlass.Int32({row_tile - 1})) // cutlass.Int32({row_tile})"
    ).body
    scan = (
        "if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(0):\n"
        f"    {namespace}_lane = cute.arch.lane_idx()\n"
        f"    {namespace}_scan = {namespace}_row_lane\n"
    )
    for shift in (1, 2, 4, 8, 16):
        if shift < groups:
            scan += (
                f"    {namespace}_partner{shift} = cute.arch.shuffle_sync_up({namespace}_scan, {shift})\n"
                f"    if {namespace}_lane >= cutlass.Int32({shift}):\n"
                f"        {namespace}_scan = {namespace}_scan + {namespace}_partner{shift}\n"
            )
    scan += (
        f"    if {namespace}_lane < cutlass.Int32({groups}):\n"
        f"        tcgen05_grouped_problem_sizes[{namespace}_lane, cutlass.Int32(3)] = {namespace}_scan\n"
    )
    prefix.insert(index + 2, ast.parse(scan).body[0])
    _require(
        _call_name(prefix[-1]) == "cute.arch.sync_threads",
        "metadata prefix must publish before the native CTA barrier",
    )
    return prefix


def _decoder(name: str) -> ast.FunctionDef:
    return cast(
        "ast.FunctionDef",
        ast.parse(
            f"""
@cute.jit
def {name}(linear, problem_sizes, feature_tiles, groups):
    lane = cute.arch.lane_idx()
    ordinal = linear // cutlass.Int32(feature_tiles)
    lane_prefix = cutlass.Int32(0)
    if lane < cutlass.Int32(groups):
        lane_prefix = problem_sizes[lane, cutlass.Int32(3)]
    matches = (lane < cutlass.Int32(groups)) & (ordinal < lane_prefix)
    mask = cute.arch.vote_ballot_sync(matches)
    expert = cute.arch.make_warp_uniform(cutlass.Int32(groups) - cute.arch.popc(mask))
    previous = cutlass.Int32(0)
    if expert > cutlass.Int32(0):
        previous = problem_sizes[expert - cutlass.Int32(1), cutlass.Int32(3)]
    row = ordinal - previous
    column = linear % cutlass.Int32(feature_tiles)
    rows = problem_sizes[expert, cutlass.Int32(0)]
    return row, column, expert, rows
"""
        ).body[0],
    )


def _native_panel_decoder(control: _NativeCore, name: str) -> ast.FunctionDef:
    """Replace only the SDK expert lookup in the proved native panel raster."""
    choices: list[tuple[list[ast.stmt], int]] = []
    for parent in ast.walk(control.kernel):
        for field in ("body", "orelse"):
            body = getattr(parent, field, None)
            if not isinstance(body, list):
                continue
            for index, statement in enumerate(body):
                if (
                    isinstance(statement, ast.If)
                    and isinstance(statement.test, ast.Compare)
                    and isinstance(statement.test.left, ast.Name)
                    and statement.test.left.id.startswith(
                        "tcgen05_grouped_panel_choice_m_clusters"
                    )
                ):
                    choices.append((body, index))
    _require(len(choices) == 1, "requires one complete native adaptive panel raster")
    body, index = choices[0]
    _require(index > 0, "missing the native panel row-count binding")
    native = deepcopy(body[index - 1 : index + 1])
    rows = [
        node
        for statement in native
        for node in ast.walk(statement)
        if isinstance(node, ast.Attribute)
        and node.attr == "problem_shape_m"
        and _references(node, "tcgen05_grouped_selected_group_search_result")
    ]
    local = [
        node
        for statement in native
        for node in ast.walk(statement)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and isinstance(node.left, ast.Attribute)
        and node.left.attr == "_current_work_linear_idx"
        and isinstance(node.right, ast.Attribute)
        and node.right.attr == "tile_count_prev_group"
    ]
    _require(
        bool(rows)
        and bool(local)
        and len({ast.dump(value) for value in rows}) == 1
        and len({ast.dump(value) for value in local}) == 1,
        "unexpected native SDK panel inputs",
    )
    replacements = {
        f"{name}_rows": rows[0],
        f"{name}_local": local[0],
    }

    class RewritePanel(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            for target, original in replacements.items():
                if ast.dump(node) == ast.dump(original):
                    return ast.Name(target, ast.Load())
            return super().visit(node)

    class RestorePanel(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            return deepcopy(replacements.get(node.id, node))

    panel = [
        cast("ast.stmt", RewritePanel().visit(deepcopy(value))) for value in native
    ]
    _require(
        ast.dump(ast.Module(native, []))
        == ast.dump(
            ast.Module(
                [
                    cast("ast.stmt", RestorePanel().visit(deepcopy(value)))
                    for value in panel
                ],
                [],
            )
        ),
        "the native panel inverse changed during expert-lookup replacement",
    )
    decoder = _decoder(name)
    # The common lookup is independent of the per-group panel. Retain its
    # complete ballot prefix and replace only its final row/column decoding.
    decoder.body = decoder.body[:9]
    decoder.body += ast.parse(
        f"{name}_rows = problem_sizes[expert, cutlass.Int32(0)]\n"
        f"{name}_local = linear - previous * cutlass.Int32(feature_tiles)\n"
        "tcgen05_grouped_selected_cta_tile_idx_m = cutlass.Int32(0)\n"
        "tcgen05_grouped_selected_cta_tile_idx_n = cutlass.Int32(0)"
    ).body
    decoder.body += panel
    decoder.body += ast.parse(
        "return tcgen05_grouped_selected_cta_tile_idx_m, "
        f"tcgen05_grouped_selected_cta_tile_idx_n, expert, {name}_rows"
    ).body
    _require(
        not _references(decoder, "tcgen05_grouped_selected_sched")
        and not _references(decoder, "tcgen05_grouped_selected_group_search_result"),
        "native panel retained an SDK reader",
    )
    return decoder


class _PackedArenaRewriter(_ArenaRewriter):
    def __init__(
        self,
        views: list[_ArenaView],
        arena: str,
        common: dict[str, str],
    ) -> None:
        super().__init__(views, arena)
        self.common = common
        self.common_replaced: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in self.common:
                _require(
                    _call_name(node.value) == "cute.arch.alloc_smem",
                    "the native shared pipeline binding was overwritten",
                )
                node.value = ast.Name(self.common[name], ast.Load())
                self.common_replaced.add(name)
                return node
        return super().visit_Assign(node)


def _packed_operand_views(control: _NativeCore) -> list[_ArenaView]:
    branch = control.branch
    views: list[_ArenaView] = []
    for native_offset, packed_offset, size, alignment in (
        (branch.a_offset, branch.packed_a_offset, branch.a_bytes, 128),
        (branch.b_offset, branch.packed_b_offset, branch.b_bytes, 128),
        (branch.d_offset, branch.packed_d_offset, branch.d_bytes, 1024),
    ):
        _require(packed_offset is not None, "missing packed operand address")
        assert packed_offset is not None
        matching = [
            view
            for view in control.views
            if view.offset == native_offset
            and view.size == size
            and view.dtype in ("cutlass.BFloat16", "cutlass.Float16")
        ]
        _require(len(matching) == 1, "missing a proved native operand allocation")
        original = matching[0]
        _require(
            packed_offset % alignment == 0,
            "packed operand does not preserve its native alignment",
        )
        views.append(
            _ArenaView(
                original.name,
                original.dtype,
                packed_offset,
                size,
                alignment,
                False,
            )
        )
    ordered = sorted(views, key=lambda view: view.offset)
    _require(
        all(left.end <= right.offset for left, right in pairwise(ordered)),
        "the exclusive native payload views overlap",
    )
    return views


def _header_extent(body: list[ast.stmt]) -> int:
    # The native CuTe shared allocator reserves its first 16 bytes. Its
    # alignment rounding is unchanged by the explicit operand arena.
    cursor = 16
    sizes = {"cutlass.Int32": 4, "cutlass.Int64": 8}
    for statement in body:
        if (
            not isinstance(statement, ast.Assign)
            or _call_name(statement.value) != "cute.arch.alloc_smem"
        ):
            continue
        call = cast("ast.Call", statement.value)
        dtype = ast.unparse(call.args[0])
        _require(dtype in sizes, "unexpected common-header element type")
        alignment = next(
            (
                _integer(value.value)
                for value in call.keywords
                if value.arg == "alignment"
            ),
            sizes[dtype],
        )
        cursor = (cursor + alignment - 1) // alignment * alignment
        cursor += _integer(call.args[1]) * sizes[dtype]
    return cursor


def _compose_m256_native_modules(
    primary: _NativeCore,
    alternate: _NativeCore,
    profile: GroupedPrefixAdaptiveNProfile,
    *,
    balanced: bool,
) -> ast.Module:
    """Compose the M256 native cores behind one common pipeline header."""
    _require(
        profile.row_tile == 256
        and profile.tmem_columns == 512
        and profile.common_d_tile == (128, 32)
        and ast.dump(ast.Module(primary.prefix, []))
        == ast.dump(ast.Module(alternate.prefix, []))
        and ast.dump(primary.kernel.args) == ast.dump(alternate.kernel.args)
        and primary.d_signature == alternate.d_signature,
        "native M256 cores disagree on metadata, input ABI or output ownership",
    )
    module = deepcopy(primary.module)
    used = (
        {value.id for value in ast.walk(module) if isinstance(value, ast.Name)}
        | {value.name for value in module.body if isinstance(value, ast.FunctionDef)}
        | {value.arg for value in ast.walk(module) if isinstance(value, ast.arg)}
    )
    namespace = "_helion_grouped_prefix"
    while any(name.startswith(namespace) for name in used):
        namespace += "_"
    arena, total, stride = (
        f"{namespace}_{suffix}" for suffix in ("smem", "total", "stride")
    )
    grouped = primary.plans[0]
    groups, n = (grouped[key] for key in ("group_count", "n_size"))
    common = {
        "tcgen05_tmem_holding_buf": f"{namespace}_holding",
        "tcgen05_tmem_dealloc_mbar_ptr": f"{namespace}_dealloc",
        "tcgen05_acc_pipeline_barriers": f"{namespace}_acc_mbars",
        "tcgen05_ab_pipeline_mbars": f"{namespace}_ab_mbars",
    }
    original_args = [value.arg for value in primary.kernel.args.args]
    extra_args = [
        "tcgen05_grouped_problem_sizes",
        "tcgen05_grouped_starts",
        total,
        stride,
        arena,
        *common.values(),
    ]
    cores: list[ast.FunctionDef] = []
    decoders: list[ast.FunctionDef] = []
    for control in (primary, alternate):
        decoder = _native_panel_decoder(
            control, f"{namespace}_decode_n{control.branch.n}"
        )
        decoders.append(decoder)
        core = deepcopy(control.kernel)
        core.name = f"{namespace}_n{control.branch.n}"
        core.decorator_list = [ast.parse("cute.jit", mode="eval").body]
        core.args.args = [ast.arg(arg=name) for name in (*original_args, *extra_args)]
        rewrite = _PackedArenaRewriter(_packed_operand_views(control), arena, common)
        core.body = [rewrite.visit(node) for node in deepcopy(control.active.body)]
        _require(
            rewrite.replaced == set(rewrite.views)
            and rewrite.common_replaced == set(common),
            "the native pipeline or operand allocation was not fully replaced",
        )
        _replace_role_schedulers(
            core,
            decoder=decoder.name,
            total=total,
            groups=groups,
            n=n,
            width=control.branch.n,
            stride=stride,
        )
        _require(
            not _references(core, "tcgen05_grouped_virtual_grid"),
            "the typed payload retained a different persistent stride",
        )
        cores.append(core)
    entry = deepcopy(primary.kernel)
    alternate_ab = deepcopy(alternate.plans[1])
    alternate_args = {
        name: f"{namespace}_{name}_alternate" for name in alternate_ab["kernel_args"]
    }
    alternate_ab["kernel_args"] = list(alternate_args.values())
    entry.args.args += [ast.arg(arg=name) for name in alternate_args.values()]
    entry.body = _prefix_publication(
        deepcopy(primary.prefix), groups, namespace, row_tile=256, preserve_mailbox=True
    )
    for original, target in common.items():
        allocation = deepcopy(_assignment(primary.kernel, original))
        _require(
            _call_name(allocation) == "cute.arch.alloc_smem",
            "missing native pipeline allocation",
        )
        if original == "tcgen05_ab_pipeline_mbars":
            assert isinstance(allocation, ast.Call)
            allocation.args[1] = ast.parse(
                f"cutlass.Int32({2 * max(profile.primary.ab_stages, profile.alternate.ab_stages)})",
                mode="eval",
            ).body
        if original == "tcgen05_ab_pipeline_mbars":
            padding = deepcopy(
                _assignment(primary.kernel, "tcgen05_sched_pipeline_mbars")
            )
            entry.body.append(
                ast.Assign(
                    [ast.Name(f"{namespace}_scheduler_padding", ast.Store())], padding
                )
            )
        entry.body.append(ast.Assign([ast.Name(target, ast.Store())], allocation))
    header = _header_extent(entry.body)
    _require(
        header <= 1024
        and profile.compiler_smem_bytes == profile.arena_bytes + 1024
        and all(
            view.end <= profile.arena_bytes
            for control in (primary, alternate)
            for view in _packed_operand_views(control)
        ),
        "the complete native pipeline header and payload exceed the arena budget",
    )
    entry.body += ast.parse(
        f"{arena} = cute.arch.alloc_smem(cutlass.Uint8, {profile.arena_bytes}, alignment=1024)\n"
        f"{namespace}_rows = tcgen05_grouped_problem_sizes[cutlass.Int32({groups - 1}), cutlass.Int32(3)]\n"
        f"{namespace}_cap = cutlass.Int32(cute.arch.grid_dim()[2])"
    ).body
    selector: dict[str, list[ast.stmt]] = {
        field: [] for field in ("total", "waves", "workers", "cost")
    }
    for branch in (profile.primary, profile.alternate):
        width = branch.n
        features = (n + width - 1) // width
        worker_count = (
            f"{namespace}_total{width} // {namespace}_waves{width} + cutlass.Int32({namespace}_total{width} % {namespace}_waves{width} != cutlass.Int32(0))"
            if balanced
            else f"{namespace}_total{width}"
        )
        for field, expression in (
            (
                "total",
                f"cutlass.Int32(cutlass.Int64({namespace}_rows) * cutlass.Int64({features}))",
            ),
            (
                "waves",
                f"max(cutlass.Int32(1), {namespace}_total{width} // {namespace}_cap + cutlass.Int32({namespace}_total{width} % {namespace}_cap != cutlass.Int32(0)))",
            ),
            ("workers", f"max(cutlass.Int32(1), min({namespace}_cap, {worker_count}))"),
            (
                "cost",
                f"cutlass.Int64({namespace}_waves{width}) * cutlass.Int64({width})",
            ),
        ):
            selector[field] += ast.parse(
                f"{namespace}_{field}{width} = {expression}"
            ).body
    entry.body.extend(statement for field in selector.values() for statement in field)
    calls: list[ast.If] = []
    for core, branch in zip(cores, (profile.primary, profile.alternate), strict=True):
        width = branch.n
        arguments = [
            alternate_args.get(name, name) if branch is profile.alternate else name
            for name in original_args
        ]
        arguments += [
            "tcgen05_grouped_problem_sizes",
            "tcgen05_grouped_starts",
            f"{namespace}_total{width}",
            f"{namespace}_workers{width}",
            arena,
            *common.values(),
        ]
        calls.append(
            cast(
                "ast.If",
                ast.parse(
                    f"if {namespace}_total{width} > cutlass.Int32(0) and cute.arch.block_idx()[2] < {namespace}_workers{width}:\n"
                    f"    {core.name}({', '.join(arguments)})"
                ).body[0],
            )
        )
    entry.body.append(
        ast.If(
            ast.parse(
                f"{namespace}_cost{profile.primary.n} <= {namespace}_cost{profile.alternate.n}",
                mode="eval",
            ).body,
            [calls[0]],
            [calls[1]],
        )
    )
    index = module.body.index(
        next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == entry.name
        )
    )
    module.body[index : index + 1] = [*decoders, *cores, entry]
    plans = deepcopy(primary.plans)
    plans[0]["balanced_workers"] = balanced
    plans[0]["prefix_adaptive_n"] = {
        "widths": [profile.primary.n, profile.alternate.n],
        "ab_stages": [profile.primary.ab_stages, profile.alternate.ab_stages],
        "c_stages": [profile.primary.c_stages, profile.alternate.c_stages],
        "arena_bytes": profile.arena_bytes,
        "compiler_smem_bytes": profile.compiler_smem_bytes,
        "common_header_bytes": header,
        "row_tile": profile.row_tile,
        "tmem_columns": profile.tmem_columns,
        "output_tile": list(profile.common_d_tile),
    }
    plans.append(alternate_ab)
    node, _ = _plans(module)
    node.value = ast.parse(repr(plans), mode="eval").body
    global_names = set(vars(builtins))
    for statement in module.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            global_names.update(
                alias.asname or alias.name.split(".", 1)[0] for alias in statement.names
            )
        elif isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
            global_names.add(statement.name)
        elif isinstance(statement, ast.Assign):
            global_names.update(
                value.id
                for target in statement.targets
                for value in ast.walk(target)
                if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Store)
            )
    for core in (*decoders, *cores):
        locals_ = {value.arg for value in core.args.args} | {
            value.id
            for value in ast.walk(core)
            if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Store)
        }
        free = (
            {
                value.id
                for value in ast.walk(core)
                if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Load)
            }
            - locals_
            - global_names
        )
        _require(not free, f"unbound native core names: {sorted(free)}")
    ast.fix_missing_locations(module)
    source = ast.unparse(module)
    _require(
        source.count("cute.arch.alloc_smem(") == 9
        and "tcgen05_work_tile_smem" not in source
        and "tcgen05_sched_pipeline" not in source
        and "warp_reduction_sum" not in source
        and not any(isinstance(node, ast.Return) for node in ast.walk(entry)),
        "requires one common header, one payload arena and structured inactive control",
    )
    return module


def _compose_native_modules(
    primary: _NativeCore,
    alternate: _NativeCore,
    profile: GroupedPrefixAdaptiveNProfile,
    *,
    balanced: bool,
) -> ast.Module:
    if profile.layout_kind == "operand_payload":
        return _compose_m256_native_modules(
            primary, alternate, profile, balanced=balanced
        )
    _require(profile.layout_kind == "whole_arena", "unsupported arena recipe")
    _require(
        ast.dump(ast.Module(primary.prefix, []))
        == ast.dump(ast.Module(alternate.prefix, []))
        and ast.dump(primary.kernel.args) == ast.dump(alternate.kernel.args)
        and primary.d_signature == alternate.d_signature,
        "typed cores disagree on metadata, input ABI or the common D atom",
    )
    module = deepcopy(primary.module)
    used = (
        {value.id for value in ast.walk(module) if isinstance(value, ast.Name)}
        | {value.name for value in module.body if isinstance(value, ast.FunctionDef)}
        | {value.arg for value in ast.walk(module) if isinstance(value, ast.arg)}
    )
    namespace = "_helion_grouped_prefix"
    while any(name.startswith(namespace) for name in used):
        namespace += "_"
    arena, decoder, total = (
        f"{namespace}_{suffix}" for suffix in ("smem", "decode", "total")
    )
    groups, n = (primary.plans[0][key] for key in ("group_count", "n_size"))
    common = [view for view in primary.views if view.common]
    _require(
        common == [view for view in alternate.views if view.common]
        and len(common) == 3,
        "requires identical native metadata storage",
    )
    original_args = [arg.arg for arg in primary.kernel.args.args]
    extra_args = [
        total,
        "tcgen05_grouped_problem_sizes",
        "tcgen05_grouped_starts",
        "tcgen05_grouped_virtual_grid",
        arena,
    ]
    cores: list[ast.FunctionDef] = []
    for control in (primary, alternate):
        core = deepcopy(control.kernel)
        core.name = f"{namespace}_n{control.branch.n}"
        core.decorator_list = [ast.parse("cute.jit", mode="eval").body]
        core.args.args = [ast.arg(arg=name) for name in (*original_args, *extra_args)]
        rewrite = _ArenaRewriter(
            [view for view in control.views if not view.common], arena
        )
        core.body = [rewrite.visit(node) for node in deepcopy(control.active.body)]
        _require(
            rewrite.replaced == set(rewrite.views),
            "native branch allocation was not replaced",
        )
        _replace_role_schedulers(
            core,
            decoder=decoder,
            total=total,
            groups=groups,
            n=n,
            width=control.branch.n,
        )
        cores.append(core)
    entry = deepcopy(primary.kernel)
    alternate_ab = deepcopy(alternate.plans[1])
    alternate_args = {
        name: f"{namespace}_{name}_alternate" for name in alternate_ab["kernel_args"]
    }
    alternate_ab["kernel_args"] = list(alternate_args.values())
    entry.args.args += [ast.arg(arg=name) for name in alternate_args.values()]
    common_rewrite = _ArenaRewriter(
        [view for view in common if not view.name.startswith("tcgen05_work_tile_smem")],
        arena,
    )
    prefix = _prefix_publication(deepcopy(primary.prefix), groups, namespace)
    prefix = [common_rewrite.visit(node) for node in prefix]
    _require(
        common_rewrite.replaced == set(common_rewrite.views),
        "native metadata allocations were not replaced",
    )
    entry.body = [
        ast.parse(
            f"{arena} = cute.arch.alloc_smem(cutlass.Uint8, {profile.arena_bytes}, alignment=1024)"
        ).body[0],
        *prefix,
        *ast.parse(
            f"{namespace}_rows = tcgen05_grouped_problem_sizes[cutlass.Int32({groups - 1}), cutlass.Int32(3)]\n"
            f"{namespace}_cap = cutlass.Int32(cute.arch.grid_dim()[2])"
        ).body,
    ]
    selector_statements: list[ast.stmt] = []
    for branch in (profile.primary, profile.alternate):
        width = branch.n
        variables = (
            f"{namespace}_total{width} = cutlass.Int32(cutlass.Int64({namespace}_rows) * cutlass.Int64({n // width}))\n"
            f"{namespace}_waves{width} = max(cutlass.Int32(1), {namespace}_total{width} // {namespace}_cap + cutlass.Int32({namespace}_total{width} % {namespace}_cap != cutlass.Int32(0)))\n"
        )
        if balanced:
            worker_count = (
                f"{namespace}_total{width} // {namespace}_waves{width} + "
                f"cutlass.Int32({namespace}_total{width} % {namespace}_waves{width} != cutlass.Int32(0))"
            )
        else:
            worker_count = f"{namespace}_total{width}"
        variables += (
            f"{namespace}_workers{width} = max(cutlass.Int32(1), min({namespace}_cap, {worker_count}))\n"
            f"{namespace}_grid{width} = (cutlass.Int32(2), cutlass.Int32(1), {namespace}_workers{width})\n"
            f"{namespace}_cost{width} = cutlass.Int64({namespace}_waves{width}) * cutlass.Int64({width})\n"
        )
        selector_statements.extend(ast.parse(variables).body)
    # Keep the independently validated selector's binding order in both cores.
    for field in ("total", "waves", "workers", "grid", "cost"):
        entry.body.extend(
            statement
            for statement in selector_statements
            if isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id.startswith(f"{namespace}_{field}")
        )
    branch_calls: list[ast.If] = []
    for core, branch in zip(cores, (profile.primary, profile.alternate), strict=True):
        width = branch.n
        arguments = [
            alternate_args.get(name, name) if branch is profile.alternate else name
            for name in original_args
        ]
        arguments += [
            f"{namespace}_total{width}",
            "tcgen05_grouped_problem_sizes",
            "tcgen05_grouped_starts",
            f"{namespace}_grid{width}",
            arena,
        ]
        branch_calls.append(
            cast(
                "ast.If",
                ast.parse(
                    f"if {namespace}_total{width} > cutlass.Int32(0) and cute.arch.block_idx()[2] < {namespace}_workers{width}:\n"
                    f"    {core.name}({', '.join(arguments)})"
                ).body[0],
            )
        )
    entry.body.append(
        ast.If(
            test=ast.parse(
                f"{namespace}_cost192 <= {namespace}_cost256", mode="eval"
            ).body,
            body=[branch_calls[0]],
            orelse=[branch_calls[1]],
        )
    )
    kernel_index = module.body.index(
        next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == entry.name
        )
    )
    module.body[kernel_index : kernel_index + 1] = [_decoder(decoder), *cores, entry]
    plans = deepcopy(primary.plans)
    plans[0]["balanced_workers"] = balanced
    plans[0]["prefix_adaptive_n"] = {
        "widths": [profile.primary.n, profile.alternate.n],
        "ab_stages": [profile.primary.ab_stages, profile.alternate.ab_stages],
        "c_stages": [profile.primary.c_stages, profile.alternate.c_stages],
        "arena_bytes": profile.arena_bytes,
        "tmem_columns": profile.tmem_columns,
        "output_tile": list(profile.common_d_tile),
    }
    plans.append(alternate_ab)
    plan_node, _ = _plans(module)
    plan_node.value = ast.parse(repr(plans), mode="eval").body
    # Reentry can introduce config-specific globals; every core must bind from
    # the retained module, its parameters, its local statements or builtins.
    global_names = {core.name for core in cores} | {decoder} | set(vars(builtins))
    for statement in module.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            global_names.update(
                alias.asname or alias.name.split(".", 1)[0] for alias in statement.names
            )
        elif isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
            global_names.add(statement.name)
        elif isinstance(statement, ast.Assign):
            global_names.update(
                node.id
                for target in statement.targets
                for node in ast.walk(target)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            )
    for core in cores:
        local_names = {arg.arg for arg in core.args.args} | {
            value.id
            for value in ast.walk(core)
            if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Store)
        }
        free = (
            {
                value.id
                for value in ast.walk(core)
                if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Load)
            }
            - local_names
            - global_names
        )
        _require(not free, f"unbound native core names: {sorted(free)}")
    ast.fix_missing_locations(module)
    source = ast.unparse(module)
    _require(
        source.count("cute.arch.alloc_smem(") == 1
        and "tcgen05_work_tile_smem" not in source
        and "tcgen05_sched_pipeline" not in source
        and "warp_reduction_sum" not in source
        and not any(isinstance(node, ast.Return) for node in ast.walk(entry)),
        "requires one arena and structured inactive-cluster control",
    )
    return module


def generate_grouped_prefix_adaptive_n_candidate(
    func: HostFunction, config: Config, emit_repro_caller: bool
) -> ast.Module:
    from ..compile_environment import CompileEnvironment
    from ..generate_ast import generate_ast

    env = CompileEnvironment.current()
    model = env.config_spec._cute_tcgen05_config
    model._normalize_grouped_prefix_adaptive_n(config.config, fix_invalid=False)
    profile = model.get_grouped_prefix_adaptive_n_profile(config.config)
    _require(
        model.grouped_prefix_adaptive_n_supported and profile is not None,
        "native purity/resource proof is unavailable",
    )
    assert profile is not None
    native: list[_NativeCore] = []
    for branch in (profile.primary, profile.alternate):
        values = {
            key: value
            for key, value in config.config.items()
            if key != TCGEN05_GROUPED_PREFIX_ADAPTIVE_N_CONFIG_KEY
        }
        values.update(
            tcgen05_grouped_worklist_mma_n=branch.n,
            tcgen05_ab_stages=branch.ab_stages,
            tcgen05_c_stages=branch.c_stages,
        )
        if profile.row_tile == 256 and branch is profile.alternate:
            values.update(
                tcgen05_layout_strategy="explicit_epi_tile",
                tcgen05_layout_overrides_epi_tile_m=profile.common_d_tile[0],
                tcgen05_layout_overrides_epi_tile_n=profile.common_d_tile[1],
                tcgen05_layout_overrides_d_store_box_n=profile.common_d_tile[1],
            )
        # This existing structured scope provides the exact native role payload
        # even when the final recipe requests the ordinary unbalanced workers.
        values[TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY] = True
        physical = env.config_spec.normalized_config(Config.from_dict(values))
        _require(
            physical.block_sizes == config.block_sizes,
            "physical reentry changed a logical block bound",
        )
        native.append(
            _load_native(
                generate_ast(func, physical, emit_repro_caller), branch, profile
            )
        )
    module = _compose_native_modules(
        native[0],
        native[1],
        profile,
        balanced=config.get(TCGEN05_GROUPED_BALANCED_WORKERS_CONFIG_KEY, False) is True,
    )
    plan_assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_helion_cute_wrapper_plans"
    )
    env.cute_resolved_wrapper_plans = ast.literal_eval(plan_assignment.value)
    return module
