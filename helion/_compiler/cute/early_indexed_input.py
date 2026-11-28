"""Use spare CTAs before a full grid join to prepare an immutable indexed row input."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from math import prod
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _tracing_ops as tr
from ...language import creation_ops
from ...language import memory_ops
from ...language import tile_ops
from ...language import view_ops
from ...runtime.config import Config
from ..ast_extension import clone_ast
from ..compile_environment import CompileEnvironment
from ..device_function import TensorArg
from ..fx_side_effects import is_impure_node
from ..program_id import NUM_SM_VAR
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_written
from ._ast_pass_utils import parse_expr
from ._ast_pass_utils import phase_barrier_call
from ._ast_pass_utils import same_expr
from .active_phase_joins import participants
from .analysis.match_common import host_tensor_name
from .analysis.match_common import node_tensor
from .autotune.config_keys import EARLY_INDEXED_INPUT_KEY
from .autotune.config_keys import PHASE_PARTICIPATION_KEY
from .autotune.config_keys import REUSE_INDEXED_INPUT_KEY
from .autotune.config_keys import STAGED_GRID_KEY
from .cooperative_row_normalization import _dispatch
from .cooperative_row_normalization import private_scratch
from .local_atomic import _reachable_graphs
from .register_loads import host_load_is_readonly
from .thread_block_projection import literal_launch_block

if TYPE_CHECKING:
    from ...autotuner.config_spec import ConfigSpec
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from .analysis.indexed_row_phases import IndexedRowPhase

_LINE_WORDS = 32
# These are control-flow / scalar / index constructors, not external memory
# effects. Unknown DSL calls, peer operations or inline asm decline the move.
_PURE = frozenset(
    (
        operator.add,
        operator.sub,
        operator.mul,
        operator.floordiv,
        operator.mod,
        operator.eq,
        operator.getitem,
        tr._host_tensor,
        tr._get_symnode,
        tr._inductor_lowering_extra,
        tr._mask_to,
        tr._new_var,
        creation_ops.full,
        view_ops.subscript,
        tile_ops.tile_begin,
        tile_ops.tile_index,
    )
)
_FLOW = frozenset((tr._for_loop, tr._if, tr._phi))


def has_known_effects(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return node.op in ("placeholder", "output")
    target = node.target
    if target in _FLOW or target is memory_ops.store:
        return True
    if is_impure_node(node):
        return False
    if target is memory_ops.load or target in _PURE:
        return True
    return (
        isinstance(target, torch._ops.OpOverload)
        and target.namespace in ("aten", "prims")
        and not target._schema.is_mutable
        and (
            torch.Tag.pointwise in target.tags
            or target
            in (
                torch.ops.aten.amax.default,
                torch.ops.aten.mean.dim,
                torch.ops.aten.sum.dim_IntList,
                torch.ops.aten.topk.default,
                torch.ops.prims.iota.default,
            )
        )
    )


def immutable_input(
    env: CompileEnvironment, ir: DeviceIR, site: IndexedRowPhase, *, allow_unbound: bool
) -> bool:
    if (
        not site.paired
        or not site.invariant_rows
        or site.input_scale is None
        or ir.root_ids[0] == site.root
    ):
        return False
    graphs = _reachable_graphs(ir.graphs)
    names = {site.vector, site.input_scale}
    found = set()
    for graph in graphs:
        for node in graph.graph.nodes:
            if not has_known_effects(node):
                return False
            if (
                node.target is memory_ops.load
                and (host := host_tensor_name(node.args[0])) in names
            ):
                tensor = node_tensor(node.args[0])
                if (
                    tensor is None
                    or env.tensor_input_source(tensor) is None
                    or not host_load_is_readonly(
                        node, env, graphs, allow_unbound=allow_unbound
                    )
                ):
                    return False
                found.add(host)
    return names == found


def eligible(spec: ConfigSpec, cfg: dict[str, object]) -> bool:
    return bool(
        spec.cute_early_indexed_axis is not None
        and cfg.get(REUSE_INDEXED_INPUT_KEY) is True
        and cfg.get(STAGED_GRID_KEY) in (2, 3)
        and not cfg.get(PHASE_PARTICIPATION_KEY)
        and cfg.get("pid_type") == "persistent_interleaved"
        and cfg.get("cross_loop_pipeline") == "barrier"
    )


def add_seed(spec: ConfigSpec, seeds: list[Config]) -> list[Config]:
    for existing in reversed(seeds):
        if eligible(spec, existing.config):
            return [
                *seeds,
                Config.from_dict(existing.config | {EARLY_INDEXED_INPUT_KEY: True}),
            ]
    return seeds


@dataclass(frozen=True)
class Prelude:
    width: int
    matrix: str
    vector: str
    scale: str
    statements: tuple[str, ...]


def record(
    fn: DeviceFunction,
    original: IndexedRowPhase,
    names: dict[str, str],
    prelude: list[ast.stmt],
) -> None:
    env = CompileEnvironment.current()
    if (
        fn.cute_state.lanes.early_indexed_input is not None
        or original.row_axis != env.config_spec.cute_early_indexed_axis
        or original.input_scale is None
        or not immutable_input(
            env, fn.codegen.host_function.device_ir, original, allow_unbound=False
        )
    ):
        raise exc.InvalidConfig(
            "early indexed input needs one proven immutable reused projection"
        )
    fn.cute_state.lanes.early_indexed_input = Prelude(
        original.k,
        names[original.matrix],
        names[original.vector],
        names[original.input_scale],
        tuple(ast.dump(s) for s in prelude),
    )


def reserve(words: int, width: int) -> tuple[int, int]:
    """Leave every original word untouched, and keep a trailing cache line."""
    start = (words + _LINE_WORDS - 1) // _LINE_WORDS * _LINE_WORDS
    return start, start + (
        (width + 2 * _LINE_WORDS - 1) // (2 * _LINE_WORDS) + 1
    ) * _LINE_WORDS


def _copy(stmts: list[ast.stmt]) -> list[ast.stmt]:
    return cast("list[ast.stmt]", clone_ast(stmts))


def launch_allocation(
    host: list[ast.AST], definition: ast.FunctionDef, arg: TensorArg, threads: int
) -> ast.Call:
    """The actual marked call must bind the same fresh buffer to the proved formal."""
    formals = [a.arg for a in definition.args.args]
    if formals.count(arg.name) != 1:
        raise exc.InvalidConfig("early indexed input requires a unique scratch formal")
    i = 2 + formals.index(arg.name)
    calls = []
    for st in host:
        for n in ast.walk(st):
            if (
                vars(n).get("_is_kernel_call") is not True
                or not isinstance(n, (ast.Expr, ast.Assign))
                or not isinstance(c := n.value, ast.Call)
            ):
                continue
            if not (
                isinstance(c.func, ast.Name)
                and c.func.id == "_launcher"
                and c.args
                and isinstance(c.args[0], ast.Name)
                and c.args[0].id == definition.name
            ):
                continue
            block = literal_launch_block([st], definition.name)
            if not (
                len(c.args) > i
                and isinstance(v := c.args[i], ast.Call)
                and same_expr(v, parse_expr(arg.host_str()))
                and same_expr(c.args[1], parse_expr(f"({NUM_SM_VAR},)"))
                and block is not None
                and min(block) > 0
                and prod(block) == threads
            ):
                raise exc.InvalidConfig(
                    "early indexed input needs the checked launch, block and private allocation"
                )
            calls.append(v)
    if len(calls) != 1:
        raise exc.InvalidConfig(
            "early indexed input needs exactly one marked local launch"
        )
    return calls[0]


def _row_phase_participants(fn: DeviceFunction, region: list[ast.stmt]) -> int | None:
    """Count row work only; every CTA retains the scheduler's scalar preheader."""
    helpers = {"cutlass", "max", "_cute_inline_asm_elementwise"}
    if any(a.name in helpers for a in fn.arguments) or any(
        helpers & names_written(s) for s in region
    ):
        return None
    inputs = {
        a.name
        for a in fn.arguments
        if type(a) is TensorArg
        and a.fake_value.ndim == 0
        and a.fake_value.dtype in (torch.int32, torch.int64)
    }

    def count_expr(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return type(node.value) is int
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return count_expr(node.left) and count_expr(node.right)
        if (
            isinstance(node, ast.Call)
            and same_expr(node.func, parse_expr("max"))
            and not node.keywords
            and len(node.args) == 2
            and same_expr(node.args[0], ast.Constant(0))
        ):
            return count_expr(node.args[1])
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) in ("cutlass.Int64", "cutlass.Uint64")
            and not node.keywords
            and len(node.args) == 1
        ):
            arg = node.args[0]
            return count_expr(arg) or (
                same_expr(node.func, parse_expr("cutlass.Int64"))
                and isinstance(arg, ast.Subscript)
                and isinstance(arg.value, ast.Name)
                and arg.value.id in inputs
                and same_expr(arg.slice, ast.Constant(0))
            )
        return False

    # The wide scheduler emits overflow traps before the first phase. Each CTA
    # still executes them before it can publish anything. Only discount the
    # original compiler preamble's exact trap on a scalar count comparison.
    preamble = {ast.dump(stmt) for stmt in fn.preamble if isinstance(stmt, ast.If)}
    trap = ast.parse(
        CompileEnvironment.current().backend.device_error_stmt(
            "root grid count exceeds signed Int64"
        )
    ).body[0]
    checked: list[ast.stmt] = []
    for stmt in region:
        # This analysis copy preserves name rebindings. The real count read and
        # every other original statement stay before the inserted publication.
        if (
            isinstance(stmt, ast.If)
            and ast.dump(stmt) in preamble
            and all(isinstance(other, ast.Pass) for other in stmt.orelse)
            and len(stmt.body) == 1
            and same_expr(stmt.body[0], trap)
            and isinstance(stmt.test, ast.Compare)
            and len(stmt.test.ops) == 1
            and isinstance(stmt.test.ops[0], ast.Gt)
            and count_expr(stmt.test.left)
            and count_expr(stmt.test.comparators[0])
        ):
            continue
        if (
            isinstance(stmt, ast.Assign)
            and assigned_name(stmt)
            and count_expr(stmt.value)
        ):
            checked.append(ast.Assign(targets=stmt.targets, value=ast.Constant(0)))
        else:
            checked.append(stmt)
        inputs.difference_update(names_written(stmt))
    return participants(checked)


def install(fn: DeviceFunction, definition: ast.FunctionDef) -> None:
    plan = fn.cute_state.lanes.early_indexed_input
    if plan is None or not eligible(
        CompileEnvironment.current().config_spec, fn.config.config
    ):
        raise exc.InvalidConfig(
            "early indexed input needs an installed complete row prelude and whole-grid join"
        )
    stmts = definition.body
    first = next(
        (i for i, s in enumerate(stmts) if phase_barrier_call(s) is not None), 0
    )
    stage_offsets = [
        i
        for i in range(first + 1, len(stmts) - len(plan.statements) + 1)
        if all(ast.dump(stmts[i + j]) == sig for j, sig in enumerate(plan.statements))
    ]
    if first < 1 or len(stage_offsets) != 1:
        raise exc.InvalidConfig(
            "early indexed input requires a unique untouched prelude after the first join"
        )
    (stage,) = stage_offsets
    call = phase_barrier_call(stmts[first])
    # validate_staged_grid has already rejected nested and remote barriers
    # and authenticated the real launch stride and cooperative admission.
    if (
        call is None
        or not isinstance(call.func, ast.Name)
        or call.func.id != "_cute_grid_barrier"
        or len(call.args) != 1
        or call.keywords
        or not isinstance(ptr := call.args[0], ast.Attribute)
        or ptr.attr != "iterator"
        or not isinstance(ptr.value, ast.Name)
    ):
        raise exc.InvalidConfig(
            "early indexed input needs an unconditional local grid semaphore"
        )
    prefix = _row_phase_participants(fn, stmts[:first])
    if (
        prefix is None
        or not isinstance(loop := stmts[first - 1], ast.For)
        or _dispatch(loop, prefix) is None
        or sum(isinstance(s, ast.For) for s in stmts[:first]) != 1
    ):
        raise exc.InvalidConfig(
            "early indexed input needs a single proved active prefix before the first join"
        )
    threads = prod(fn.tile_strategy.thread_block_dims())
    width = plan.width
    if (
        threads < 32
        or threads % 32
        or width % threads
        or not 0 < prefix < prefix + width // threads < 2**31
    ):
        raise exc.InvalidConfig("early indexed input needs a whole number of row bands")
    body = stmts[stage : stage + len(plan.statements)]
    # Reuse exactly the already-proven emitter's lane mapping, RMS arithmetic
    # and final BF16 multiply. The only new addresses are private unit-stride words.
    if (
        len(body) != 8
        or not isinstance(sweep := body[-2], ast.For)
        or not isinstance(sweep.target, ast.Name)
        or not same_expr(sweep.iter, parse_expr(f"range({width // threads})"))
        or not isinstance(last := sweep.body[-1], ast.Assign)
        or len(last.targets) != 1
        or not isinstance(dst := last.targets[0], ast.Subscript)
        or not isinstance(dst.slice, ast.Name)
        or not same_expr(body[-1], ast.parse("cute.arch.sync_threads()").body[0])
        or not isinstance(sweep.body[0], ast.Assign)
        or assigned_name(sweep.body[0]) != dst.slice.id
    ):
        raise exc.InvalidConfig(
            "early indexed input lost its proved bands or shared publication"
        )
    arg = next(
        (a for a in fn.arguments if type(a) is TensorArg and a.name == ptr.value.id),
        None,
    )
    if not isinstance(arg, TensorArg) or not private_scratch(
        arg, words := int(arg.fake_value.numel())
    ):
        raise exc.InvalidConfig("early indexed input needs fresh local grid scratch")
    alloc = launch_allocation(fn.codegen.host_statements, definition, arg, threads)
    start, total = reserve(words, width)
    if total >= 2**31:
        raise exc.InvalidConfig("early indexed input scratch exceeds int32 addressing")
    end = prefix + width // threads
    scratch = fn.new_var("early_input_scratch")
    block = "cutlass.Int32(cute.arch.block_idx()[0])"
    ready = f"cutlass.const_expr({NUM_SM_VAR} >= {end} and {plan.matrix}.iterator.alignment >= 16 and {plan.vector}.iterator.alignment >= 16 and {plan.scale}.iterator.alignment >= 16)"
    ptr_expr = f"cute.recast_ptr({arg.name}.iterator + {start}, dtype=cutlass.BFloat16)"
    first_three = _copy(body[:3])
    red = _copy(body[3:-2])
    band = sweep.target.id
    k = dst.slice.id
    write = (
        first_three
        + red
        + ast.parse(f"{band} = {block} - {prefix}\n{scratch} = {ptr_expr}").body
        + _copy(sweep.body[:-1])
        + [
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=parse_expr(f"({scratch} + {k})"),
                        attr="store",
                        ctx=ast.Load(),
                    ),
                    args=[cast("ast.expr", clone_ast(last.value))],
                    keywords=[],
                )
            )
        ]
    )
    # Give the moved program fresh locals. Its later copy and the original
    # fallback remain in the lexical scope of the existing indexed projection.
    locals_ = set.union(*(names_written(st) for st in write))
    external = set.union(*(names_read(st) for st in write)) - locals_
    formals = {a.arg for a in definition.args.args}
    globals_ = {"cutlass", "cute", "range"}
    if (
        formals & globals_
        or not external <= formals | globals_
        or any(external & names_written(st) for st in stmts[:stage])
    ):
        raise exc.InvalidConfig(
            "early indexed input operands must be unchanged incoming arguments"
        )
    rename = {
        name: fn.new_var("early_input_" + name.lstrip("_")) for name in sorted(locals_)
    }
    for st in write:
        for name in ast.walk(st):
            if isinstance(name, ast.Name) and name.id in rename:
                name.id = rename[name.id]
    publish = ast.If(
        test=parse_expr(ready),
        orelse=[],
        body=[
            ast.If(
                test=parse_expr(f"{block} >= {prefix} and {block} < {end}"),
                body=write,
                orelse=[],
            )
        ],
    )
    cp = cast("ast.For", _copy([sweep])[0])
    cp.body = [
        cp.body[0],
        cast(
            "ast.stmt",
            ast.parse(f"{ast.unparse(dst)} = ({scratch} + {k}).load()").body[0],
        ),
    ]
    consume = ast.If(
        test=parse_expr(ready),
        body=[*ast.parse(f"{scratch} = {ptr_expr}").body, cp],
        orelse=_copy(body[3:-1]),
    )
    stmts[stage : stage + len(body)] = [*body[:3], consume, body[-1]]
    stmts.insert(first, publish)
    # Device allocation shape follows the checked grid. The small-grid
    # fallback preserves both the original program and its allocation.
    alloc.args[0] = parse_expr(f"(({total} if {NUM_SM_VAR} >= {end} else {words}),)")
    arg._host_str = ast.unparse(alloc)
    arg.fake_value = arg.fake_value.new_empty((total,))
