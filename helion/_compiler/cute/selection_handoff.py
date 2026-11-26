"""Place four independent warp selections and their merge in a single CTA."""

from __future__ import annotations

import ast
from math import prod
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import memory_ops
from ...runtime.config import Config
from ..compile_environment import CompileEnvironment
from ..device_function import TensorArg
from ..program_id import ForEachProgramID
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import phase_barrier_call
from .active_phase_joins import static_phase_bound
from .analysis.match_common import node_tensor
from .analysis.selection_handoff import match_selection_handoff
from .autotune.config_keys import SELECTION_HANDOFF_KEY
from .cooperative_row_normalization import _dispatch
from .loop_nesting import boundary_only_grid_blocks
from .selection.selection_lowering import emit_selection_network
from .thread_block_projection import literal_launch_block
from .thread_budget import tile_loop_thread_count

if TYPE_CHECKING:
    from ...autotuner.config_spec import ConfigSpec
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR


def permits(config: dict[str, object]) -> bool:
    return (
        config.get("pid_type") == "persistent_interleaved"
        and config.get("cross_loop_pipeline") == "barrier"
        and config.get("cute_fragment_topk_network") is True
        and not config.get("cute_phase_participation")
        and not config.get("cute_pdl")
        and not config.get("cute_fragment_cta_topk")
    )


def add_seed(
    env: CompileEnvironment, ir: DeviceIR, prior: list[Config]
) -> list[Config]:

    if not env.config_spec.cute_selection_handoff_roots or ir.host_function is None:
        return prior
    with ir.host_function:
        inactive = boundary_only_grid_blocks(env, ir, ir.graphs)
        for c in reversed(prior):
            if (
                permits(c.config)
                and tile_loop_thread_count(
                    env, ir, ir.graphs, c, inactive_block_ids=inactive
                )
                == 128
            ):
                opts = c.config | {SELECTION_HANDOFF_KEY: True}
                # Retain the invariant input of an immediately preceding
                # proved warp-row reduction when its single output is the
                # score vector that feeds this selection. Other roots keep
                # their original reload policy.
                root = env.config_spec.cute_selection_handoff_roots[0]
                ordinal = ir.root_ids.index(root)
                prior_root = ir.root_ids[ordinal - 1]
                row = env.config_spec.cute_warp_row_reduction_roots.get(prior_root)
                ids = env.config_spec.cute_reduction_reloads.valid_block_ids()
                if (
                    row is not None
                    and c.get("cute_unroll_warp_row_sweeps")
                    and row[1] in ids
                ):
                    pair = match_selection_handoff(env, ir)
                    writes = [
                        n
                        for n in ir.graphs[prior_root].graph.nodes
                        if n.target is memory_ops.store
                    ]
                    if (
                        pair is not None
                        and len(writes) == 1
                        and node_tensor(writes[0].args[0]) is pair.logits
                    ):
                        reloaded = c.get("cute_reduction_reloads")
                        modes = (
                            list(reloaded)
                            if isinstance(reloaded, list) and len(reloaded) == len(ids)
                            else ["auto"] * len(ids)
                        )
                        modes[ids.index(row[1])] = "register"
                        opts["cute_reduction_reloads"] = modes
                return [*prior, Config.from_dict(opts)]
    return prior


def normalize(
    spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    val = config.get(SELECTION_HANDOFF_KEY)
    if val is None:
        return
    if val is False:
        config.pop(SELECTION_HANDOFF_KEY)
    elif (
        val is not True or not spec.cute_selection_handoff_roots or not permits(config)
    ):
        if not fix_invalid:
            raise exc.InvalidConfig(
                "cooperative selection requires a proved adjacent selection pair and full local interleaved barriers"
            )
        config.pop(SELECTION_HANDOFF_KEY)


def _memory(call: ast.Call) -> tuple[str, str, ast.expr] | None:
    func = call.func
    if (
        not isinstance(func, ast.Attribute)
        or func.attr not in ("load", "store")
        or call.keywords
    ):
        return None
    v = func.value
    if (
        not isinstance(v, ast.BinOp)
        or not isinstance(v.op, ast.Add)
        or not isinstance(v.left, ast.Attribute)
        or v.left.attr != "iterator"
        or not isinstance(v.left.value, ast.Name)
    ):
        return None
    return v.left.value.id, func.attr, v.right


def _local_buffers(definition: ast.FunctionDef, loop: ast.For) -> set[str]:
    """Discard only private allocation contents that cannot be observed outside this loop."""
    parents = {c: n for n in ast.walk(definition) for c in ast.iter_child_nodes(n)}
    in_loop = set(ast.walk(loop))
    names: dict[str, list[ast.Name]] = {}
    for node in ast.walk(definition):
        if isinstance(node, ast.Name):
            names.setdefault(node.id, []).append(node)
    # For shared allocations, prove the allocator and tensor are fresh, local
    # and unaliased. A register tensor may also be constructed in the loop.
    constructors: set[str] = set()
    for stmt in definition.body[: definition.body.index(loop)]:
        if (
            isinstance(stmt, ast.Assign)
            and (name := assigned_name(stmt))
            and isinstance(stmt.value, ast.Call)
            and ast.unparse(stmt.value.func) == "cutlass.utils.SmemAllocator"
        ):
            if all(
                n is stmt.targets[0]
                or isinstance(n.ctx, ast.Load)
                and isinstance(use := parents[n], ast.Attribute)
                and use.value is n
                and use.attr == "allocate_tensor"
                and isinstance(call := parents[use], ast.Call)
                and call.func is use
                for n in names[name]
            ):
                constructors.add(name)
    safe = set()
    for stmt in (*definition.body[: definition.body.index(loop)], *in_loop):
        if (
            not isinstance(stmt, ast.Assign)
            or not (name := assigned_name(stmt))
            or not isinstance(stmt.value, ast.Call)
        ):
            continue
        func = stmt.value.func
        register = stmt in in_loop and ast.unparse(func) == "cute.make_rmem_tensor"
        shared = (
            stmt not in in_loop
            and isinstance(func, ast.Attribute)
            and func.attr == "allocate_tensor"
            and isinstance(func.value, ast.Name)
            and func.value.id in constructors
        )
        if not (register or shared):
            continue

        def private_use(
            n: ast.Name,
            target: ast.expr = stmt.targets[0],
            is_register: bool = register,
        ) -> bool:
            if n is target:
                return True
            if n not in in_loop or not isinstance(n.ctx, ast.Load):
                return False
            use = parents[n]
            return (isinstance(use, ast.Subscript) and use.value is n) or (
                is_register
                and isinstance(use, ast.Attribute)
                and use.value is n
                and use.attr == "fill"
                and isinstance(call := parents[use], ast.Call)
                and call.func is use
            )

        if all(private_use(n) for n in names[name]):
            safe.add(name)
    return safe


def _local_effects(
    loop: ast.For, expected: list[tuple[str, str]], buffers: set[str]
) -> bool:
    # Method names alone do not prove a call is local: a fill through an
    # unrelated tensor would be an observable write. Check the constructor
    # for every definition of the receiver in the phase we are replacing.
    definitions: dict[str, set[str]] = {}
    for stmt in ast.walk(loop):
        if isinstance(stmt, ast.Assign) and (name := assigned_name(stmt)):
            origin = (
                ast.unparse(stmt.value.func) if isinstance(stmt.value, ast.Call) else ""
            )
            definitions.setdefault(name, set()).add(origin)
    calls = []
    for n in ast.walk(loop):
        if (
            isinstance(n, (ast.Subscript, ast.Attribute))
            and isinstance(n.ctx, (ast.Store, ast.Del))
            and not (
                isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name)
                and n.value.id in buffers
                and isinstance(n.ctx, ast.Store)
            )
        ):
            return False
        if not isinstance(n, ast.Call):
            continue
        if m := _memory(n):
            calls.append(m[:2])
            continue
        if ast.unparse(n.func) in _LOCAL_OPS:
            continue
        if (
            isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and (
                n.func.attr == "fill"
                and n.func.value.id in buffers
                and definitions.get(n.func.value.id) == {"cute.make_rmem_tensor"}
                or n.func.attr == "bitcast"
                and definitions.get(n.func.value.id) == {"cutlass.Float32"}
            )
        ):
            continue
        return False
    return sorted(calls) == sorted(expected)


_LOCAL_OPS = {
    "cute.arch.block_dim",
    "cute.arch.block_idx",
    "cute.arch.shuffle_sync",
    "cute.arch.shuffle_sync_bfly",
    "cute.arch.sync_threads",
    "cute.arch.thread_idx",
    "cute.make_rmem_tensor",
    "cute.math.max",
    "cute.math.min",
    "cutlass.Float32",
    "cutlass.Int32",
    "cutlass.Int64",
    "cutlass.range_constexpr",
    "range",
}


def _private_scratch(
    fn: DeviceFunction, definition: ast.FunctionDef, start: int, end: int
) -> tuple[str, list[ast.stmt]]:
    # Reuse private, uninitialized shared storage *only* if its tensor is not
    # observed before or during our pair and never escapes via a pointer,
    # descriptor, view or call. The next original full join ends our lifetime.
    statements = definition.body
    prefix = statements[:start]
    for i, stmt in enumerate(prefix[:-1]):
        if (
            not isinstance(stmt, ast.Assign)
            or not (a := assigned_name(stmt))
            or ast.unparse(stmt.value) != "cute.arch.alloc_smem(cutlass.Float32, 64)"
        ):
            continue
        view = prefix[i + 1]
        if (
            not isinstance(view, ast.Assign)
            or not (b := assigned_name(view))
            or ast.unparse(view.value) != f"cute.make_tensor({a}, (64,))"
        ):
            continue
        parent = {c: n for n in ast.walk(definition) for c in ast.iter_child_nodes(n)}

        def all_names(seq: list[ast.stmt]) -> list[ast.Name]:
            return [n for s in seq for n in ast.walk(s) if isinstance(n, ast.Name)]

        if any(
            n.id in (a, b)
            for n in all_names([*prefix[:i], *statements[i + 2 : end + 1]])
        ):
            continue
        if any(
            n.id == a
            or n.id == b
            and not (
                isinstance(n.ctx, ast.Load)
                and isinstance((use := parent.get(n)), ast.Subscript)
                and use.value is n
            )
            for n in all_names(statements[end + 1 :])
        ):
            continue
        return b, []
    ptr = fn.new_var("selection_bridge_ptr")
    name = fn.new_var("selection_bridge")
    return name, ast.parse(
        f"{ptr}=cute.arch.alloc_smem(cutlass.Float32,64)\n{name}=cute.make_tensor({ptr},(64,))"
    ).body


def _pair_starts(
    fn: DeviceFunction, ir: DeviceIR, producer: int, consumer: int
) -> tuple[int, int] | None:
    """Use the dispatcher's actual cumulative PID intervals for these graph IDs."""
    schedule = fn.pid
    if not isinstance(schedule, ForEachProgramID) or len(schedule.cases) != len(
        ir.root_ids
    ):
        return None
    j = ir.root_ids.index(producer)
    if j + 1 >= len(ir.root_ids) or ir.root_ids[j + 1] != consumer:
        return None
    # Stops are exclusive root ordinals. A symbolic earlier phase, or a
    # schedule that combines either root, cannot certify the integer bases
    # accepted by the concrete interleaved dispatch matcher below.
    bounds = {stop: value for value, stop in schedule._phase_boundaries()}
    if not all(n in bounds for n in (j, j + 1, j + 2)):
        return None
    first, second, end = (
        static_phase_bound(ast.parse(bounds[n], mode="eval").body)
        for n in (j, j + 1, j + 2)
    )
    if first is None or second is None or end is None:
        return None
    return (first, second) if second == first + 4 and end == second + 1 else None


def install(fn: DeviceFunction, definition: ast.FunctionDef) -> None:
    env = CompileEnvironment.current()
    ir = fn.codegen.host_function.device_ir
    plan = match_selection_handoff(env, ir)
    block = literal_launch_block(fn.codegen.host_statements, fn.name)
    if (
        plan is None
        or not permits(fn.config.config)
        or block is None
        or prod(block) != 128
    ):
        raise exc.InvalidConfig(
            "cooperative selection requires four complete physical warps and exact private disjoint candidate storage"
        )
    starts = _pair_starts(fn, ir, plan.producer, plan.consumer)
    if starts is None:
        raise exc.InvalidConfig(
            "cooperative selection requires static dispatch intervals for the proved FX roots"
        )
    args = {id(v.fake_value): v.name for v in fn.arguments if isinstance(v, TensorArg)}
    tensors = (plan.logits, plan.values, plan.ids, plan.flat_values, plan.flat_ids)
    if any(id(t) not in args for t in tensors):
        raise exc.InvalidConfig("cooperative selection has no installed device operand")
    x, values, ids, flat, fi = (args[id(t)] for t in tensors)
    body = definition.body

    def full(s: ast.stmt) -> str | None:
        call = phase_barrier_call(s)
        return (
            ast.dump(call)
            if call is not None
            and isinstance(call.func, ast.Name)
            and call.func.id == "_cute_grid_barrier"
            else None
        )

    candidates = []
    for j in range(1, len(body) - 3):
        a, join, b = body[j : j + 3]
        if (
            not isinstance(a, ast.For)
            or not isinstance(b, ast.For)
            or not (key := full(join))
            or full(body[j - 1]) != key
            or full(body[j + 3]) != key
            or (left := _dispatch(a, 4)) is None
            or (right := _dispatch(b, 1)) is None
            or (left[2], right[2]) != starts
        ):
            continue
        if not _local_effects(
            a,
            [(x, "load"), (values, "store"), (ids, "store")],
            _local_buffers(definition, a),
        ):
            continue
        uses = [
            (node, m)
            for node in ast.walk(b)
            if isinstance(node, ast.Call)
            and (m := _memory(node))
            and m[0] in (flat, fi)
        ]
        if sorted((m[0], m[1]) for _, m in uses) != sorted(
            [(flat, "load"), (fi, "load")]
        ):
            continue
        if any(
            isinstance(node, ast.Name)
            and node.id in (flat, fi)
            and not isinstance(node.ctx, ast.Load)
            for node in ast.walk(b)
        ):
            continue
        candidates.append((j, b, uses))
    if len(candidates) != 1:
        raise exc.InvalidConfig(
            "cooperative selection needs precisely one complete selection/merge dispatch between unconditional local joins"
        )
    j, b, uses = candidates[0]
    scratch, alloc = _private_scratch(fn, definition, j, j + 3)

    # Build keys in registers using the same monotone FP32/NaN rank and
    # original-column tie rule as FragmentSelection.network_topk. A full warp
    # gathers the unmodified bits; the first eight lanes alone publish.
    def var(text: str) -> str:
        return fn.new_var("selection_" + text)

    lane, group, bits, mag, sign, rank, keys, col, raw, slot, eid = [
        var(n)
        for n in (
            "lane",
            "group",
            "bits",
            "magnitude",
            "sign",
            "rank",
            "keys",
            "column",
            "word",
            "slot",
            "id",
        )
    ]
    work = ast.parse(f"""
{lane}=cutlass.Int32(cute.arch.lane_idx())
{group}=cutlass.Int32(cute.arch.warp_idx())
{bits}=cutlass.Int32(({x}.iterator + cutlass.Int64({group}*32 + {lane})).load().bitcast(cutlass.Int32))
{mag}={bits} & cutlass.Int32(2147483647)
{sign}={bits} >> cutlass.Int32(31)
{rank}=cutlass.Int64(({mag} ^ {sign}) - {sign})
if {mag} > cutlass.Int32(2139095040):
    {rank}=cutlass.Int64(2147483647)
{keys}=cute.make_rmem_tensor(1,cutlass.Int64)
{keys}[0]={rank} << cutlass.Int32(5) | cutlass.Int64(31 - {lane})
""").body
    with fn.codegen.set_statements(cast("list[ast.AST]", work)):
        net = emit_selection_network(
            fn.codegen,
            keys=keys,
            dtype=torch.int64,
            groups=32,
            registers=1,
            k=8,
            lane=lane,
        )
    work.extend(
        ast.parse(f"""
{col}=cutlass.Int32(31)-cutlass.Int32({net.name}[0]&cutlass.Int64(31))
{raw}=cute.arch.shuffle_sync({bits},offset={col},mask_and_clamp=31)
if {lane}<8:
    {slot}={group}*8+{lane}
    {eid}={group}*32+{col}
    ({values}.iterator+cutlass.Int64({slot})).store({raw}.bitcast(cutlass.Float32))
    ({ids}.iterator+cutlass.Int64({slot})).store({eid})
    {scratch}[{slot}]={raw}.bitcast(cutlass.Float32)
    {scratch}[32+{slot}]={eid}.bitcast(cutlass.Float32)
cute.arch.sync_threads()
""").body
    )
    local = ast.parse(
        "if cutlass.Int32(cute.arch.block_idx()[0]) == 0:\n    pass"
    ).body[0]
    assert isinstance(local, ast.If)
    local.body = work

    class Bridge(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            for src, (name, _, offset) in uses:
                if node is src:
                    idx = ast.unparse(offset)
                    s = (
                        f"{scratch}[{idx}]"
                        if name == flat
                        else f"{scratch}[32 + ({idx})].bitcast(cutlass.Int32)"
                    )
                    return ast.copy_location(ast.parse(s, mode="eval").body, node)
            return self.generic_visit(node)

    Bridge().visit(b)
    body[j : j + 2] = [local]
    body[:0] = alloc
