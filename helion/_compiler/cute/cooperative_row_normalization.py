"""Publish per-warp squares of rounded BF16 pairs and normalize after a grid join."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from math import prod
import textwrap
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ...runtime.config import Config
from ..ast_extension import clone_ast
from ..compile_environment import CompileEnvironment
from ..device_function import TensorArg
from ..program_id import NUM_SM_VAR
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import parse_expr
from ._ast_pass_utils import phase_barrier_call
from ._ast_pass_utils import same_expr
from .active_phase_joins import participants
from .active_phase_joins import static_phase_bound
from .autotune.config_keys import ORDERED_COLUMN_PAIRS_KEY
from .autotune.config_keys import PHASE_PARTICIPATION_KEY
from .autotune.config_keys import STAGED_GRID_KEY
from .thread_budget import linear_thread_index

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...autotuner.config_spec import ConfigSpec
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR

KEY = "cute_cooperative_row_norm"
_WARP = 32
# The first cache line remains exclusively the original grid semaphore.
_PARTIALS = 32


@dataclass(frozen=True)
class PublishedPairs:
    root: int
    source: str
    output: str
    source_rank: int
    width: int
    tile: int
    origin: str
    tile_symbol: str
    predicate: str
    thread: str
    column: str
    low: str
    high: str


@dataclass(frozen=True)
class NormalizedRow:
    root: int
    source: str
    weight: str
    output: str
    width: int
    epsilon: str
    accumulator: str
    inverse: str
    thread: str
    sweep: tuple[str, ...] = ()
    approximate: bool = False


def consecutive_terminal_roots(ir: DeviceIR) -> tuple[int, int] | None:
    if len(ir.phases) < 2 or len(ir.root_ids) < 2:
        return None
    p = len(ir.root_ids) - 2
    if ir.phases[-2].roots != [p] or ir.phases[-1].roots != [p + 1]:
        return None
    return ir.root_ids[-2], ir.root_ids[-1]


def terminal_column(spec: ConfigSpec, ir: DeviceIR) -> tuple[int, int] | None:
    pair = consecutive_terminal_roots(ir)
    if pair is None:
        return None
    p, c = pair
    col = spec.cute_ordered_column_sum_roots.get(p)
    row = spec.cute_single_row_publication_roots.get(c)
    if col is None or row is None or col[3] != row[1]:
        return None
    # The RMS input must be the BF16 row just published. Equal widths alone
    # do not establish a dependency between otherwise independent roots.
    dsts = [n.args[0] for n in ir.graphs[p].graph.nodes if n.target is memory_ops.store]
    if len(dsts) != 1 or not isinstance(dst := dsts[0], torch.fx.Node):
        return None
    for node in ir.graphs[c].graph.nodes:
        if node.target is not torch.ops.aten.mean.dim or node.args not in (
            (None, [-1]),
            (None, (-1,)),
        ):
            continue
        extra = node.kwargs.get("_extra_args")
        if (
            not isinstance(extra, (list, tuple))
            or len(extra) != 1
            or not isinstance(op := extra[0], torch.fx.Node)
            or op.target is not _tracing_ops._inductor_lowering_extra
            or len(op.args) != 1
            or not isinstance(args := op.args[0], (list, tuple))
            or len(args) != 1
            or not isinstance(square := args[0], torch.fx.Node)
        ):
            continue
        if (
            square.target is not torch.ops.aten.mul.Tensor
            or len(square.args) != 2
            or square.args[0] is not square.args[1]
        ):
            continue
        widen = square.args[0]
        if (
            not isinstance(widen, torch.fx.Node)
            or widen.target is not torch.ops.prims.convert_element_type.default
            or widen.args[1] is not torch.float32
        ):
            continue
        load = widen.args[0]
        if (
            isinstance(load, torch.fx.Node)
            and load.target is memory_ops.load
            and isinstance(src := load.args[0], torch.fx.Node)
            and src.target is dst.target
            and src.args == dst.args
            and src.meta.get("val") is dst.meta.get("val")
        ):
            return col[0], row[1]
    return None


def can_offer(spec: ConfigSpec, cfg: dict[str, object]) -> bool:
    fact = spec.cute_normalizable_column
    blocks = cfg.get("block_sizes")
    if fact is None or not isinstance(blocks, list):
        return False
    column, width = fact
    chunk = spec.block_sizes.config_get(blocks, column, 0)
    return bool(
        type(chunk) is int
        and 0 < chunk < width
        and chunk % (2 * _WARP) == 0
        and width % chunk == 0
        and cfg.get(ORDERED_COLUMN_PAIRS_KEY) is True
        and cfg.get("cute_packed_row_publication") is True
        and cfg.get(STAGED_GRID_KEY) in (2, 3)
        and not cfg.get(PHASE_PARTICIPATION_KEY)
        and cfg.get("pid_type") == "persistent_interleaved"
        and cfg.get("cross_loop_pipeline") == "barrier"
    )


def add_seed(spec: ConfigSpec, seeds: list[Config]) -> list[Config]:
    for prior in reversed(seeds):
        config = dict(prior.config)
        # Active-prefix joins use the original consumer's narrower domain.
        # Retain the proven whole-grid handoff for the distributed consumer.
        config.pop(PHASE_PARTICIPATION_KEY, None)
        if can_offer(spec, config):
            return [*seeds, Config.from_dict(config | {KEY: True})]
    return seeds


def scratch_words(width: int) -> int:
    if width <= 0 or width % (2 * _WARP):
        raise exc.InvalidConfig("cooperative normalization needs full pairs per warp")
    partials = width // (2 * _WARP)
    return _PARTIALS + ((partials + _WARP - 1) // _WARP) * _WARP


def private_scratch(argument: TensorArg, words: int) -> bool:
    """Match the fresh zero allocation that interleaved-grid emission appends."""
    if (
        argument._host_str is None
        or argument.fake_value.dtype != torch.uint32
        or argument.fake_value.shape != (words,)
    ):
        return False
    call = parse_expr(argument._host_str)
    return (
        isinstance(call, ast.Call)
        and same_expr(call.func, parse_expr("torch.zeros"))
        and len(call.args) == 1
        and same_expr(call.args[0], parse_expr(f"({words},)"))
        and len(call.keywords) == 2
        and call.keywords[0].arg == "device"
        and call.keywords[1].arg == "dtype"
        and same_expr(call.keywords[1].value, parse_expr("torch.uint32"))
    )


def _require(ok: object, message: str) -> None:
    if not ok:
        raise exc.InvalidConfig(f"cooperative row normalization requires {message}")


def _dispatch(loop: ast.For, tasks: int) -> tuple[list[ast.stmt], str, int] | None:
    """The verified interleaved emitter assigns a contiguous interval of virtual PIDs."""
    if len(loop.body) != 2 or loop.orelse:
        return None
    pid, guard = loop.body
    if not (
        isinstance(pid, ast.Assign)
        and (name := assigned_name(pid))
        and isinstance(guard, ast.If)
        and isinstance(guard.test, ast.BoolOp)
        and isinstance(guard.test.op, ast.And)
        and len(guard.test.values) == 2
        and all(isinstance(x, ast.Pass) for x in guard.orelse)
        and isinstance(first := guard.test.values[0], ast.Compare)
        and isinstance(second := guard.test.values[1], ast.Compare)
        and len(first.comparators) == len(second.comparators) == 1
        and (base := static_phase_bound(first.comparators[0])) is not None
        and static_phase_bound(second.comparators[0]) == base + tasks
        and isinstance(loop.target, ast.Name)
    ):
        return None
    lo, hi = ast.unparse(first.comparators[0]), ast.unparse(second.comparators[0])
    if not (
        same_expr(guard.test, parse_expr(f"{name} >= ({lo}) and {name} < ({hi})"))
        and same_expr(
            pid.value,
            parse_expr(
                f"({lo}) + (cutlass.Int32(cute.arch.block_idx()[0]) + {loop.target.id} * {NUM_SM_VAR})"
            ),
        )
    ):
        return None
    if participants([loop]) != tasks:
        return None
    return guard.body, name, base


def _tile_coordinate(
    prefix: list[ast.stmt],
    origin: str,
    virtual: str,
    base: int,
    tile_symbol: str,
    tile: int,
    tasks: int,
) -> str | None:
    """Prove origin = tile * logical CTA and the partial index equals that CTA.

    Interpret only the integer expressions emitted by the interleaved grid.
    Every intermediate is bounded on the admitted virtual interval.
    """
    values = {virtual: (1, base), tile_symbol: (0, tile)}

    def affine(e: ast.expr) -> tuple[int, int] | None:
        if isinstance(e, ast.Name):
            return values.get(e.id)
        if isinstance(e, ast.Constant) and type(e.value) is int:
            return 0, e.value
        if (
            isinstance(e, ast.Call)
            and same_expr(e.func, parse_expr("cutlass.Int32"))
            and len(e.args) == 1
            and not e.keywords
        ):
            return affine(e.args[0])
        if not isinstance(e, ast.BinOp):
            return None
        a, b = affine(e.left), affine(e.right)
        if a is None or b is None:
            return None
        if isinstance(e.op, ast.Sub):
            return a[0] - b[0], a[1] - b[1]
        if isinstance(e.op, ast.Mult) and b[0] == 0:
            return a[0] * b[1], a[1] * b[1]
        if isinstance(e.op, ast.FloorDiv) and b == (0, 1):
            return a
        return None

    owner = None
    for stmt in prefix:
        if (
            not isinstance(stmt, ast.Assign)
            or (name := assigned_name(stmt)) is None
            or name == tile_symbol
        ):
            return None
        value = affine(stmt.value)
        if value is None or any(
            not -(2**31) <= value[0] * x + value[1] < 2**31 for x in (0, tasks - 1)
        ):
            return None
        if name == origin:
            if (
                owner is not None
                or value != (tile, 0)
                or not isinstance(stmt.value, ast.BinOp)
                or not isinstance(stmt.value.op, ast.Mult)
                or not isinstance(stmt.value.left, ast.Name)
                or values.get(stmt.value.left.id) != (1, 0)
            ):
                return None
            owner = stmt.value.left.id
        elif name == owner:
            return None
        values[name] = value
    return owner


def _producer_store(
    loop: ast.For, plan: PublishedPairs, tasks: int
) -> tuple[list[ast.stmt], int, str] | None:
    decoded = _dispatch(loop, tasks)
    if decoded is None:
        return None
    body, virtual, base = decoded
    sources = [s for s in body if isinstance(s, ast.If)]
    if len(sources) != 1 or sources[0] != body[-1]:
        return None
    branch = sources[0]
    # The ordered-column emitter proved origin/column/mask coverage and the
    # sole store. Tie its final word and origin to this specific phase.
    tile_pid = _tile_coordinate(
        body[:-1], plan.origin, virtual, base, plan.tile_symbol, plan.tile, tasks
    )
    if tile_pid is None or ast.dump(branch.test) != plan.predicate:
        return None
    endpoint = parse_expr(
        f"cute.recast_ptr({plan.output}.iterator + {plan.column}, "
        f"dtype=cutlass.Uint32).store(_cute_column_pack_bf16x2({plan.low}, {plan.high}))"
    )
    if (
        not branch.body
        or not isinstance(last := branch.body[-1], ast.Expr)
        or not same_expr(last.value, endpoint)
    ):
        return None
    return branch.body, len(branch.body) - 1, tile_pid


def sweep_signature(statements: Sequence[ast.AST], inverse: str) -> tuple[str, ...]:
    """Allow only the backend's empty else or eliminated loop-private scalar copy."""
    body = cast("list[ast.AST]", clone_ast(list(statements)))

    def names(
        nodes: Sequence[ast.AST], context: type[ast.expr_context]
    ) -> Counter[str]:
        return Counter(
            n.id
            for st in nodes
            for n in ast.walk(st)
            if isinstance(n, ast.Name) and isinstance(n.ctx, context)
        )

    read, written = names(body, ast.Load), names(body, ast.Store)
    for st in body:
        for node in ast.walk(st):
            if isinstance(node, ast.If) and all(
                isinstance(n, ast.Pass) for n in node.orelse
            ):
                node.orelse = []
            if not isinstance(node, ast.For):
                continue
            replaced = []
            for i, stmt in enumerate(node.body):
                if (
                    not isinstance(stmt, ast.Assign)
                    or not isinstance(stmt.value, ast.Name)
                    or (target := assigned_name(stmt)) is None
                    or not isinstance(stmt.targets[0], ast.Name)
                    or written[target] != 1
                    or read[target] == 0
                ):
                    continue
                src = stmt.value.id
                if src != inverse:
                    continue
                tail = node.body[i + 1 :]
                # All uses are dominated by this assignment and stay inside
                # the same iteration, and the source is never rebound there.
                if (
                    names(tail, ast.Load)[target] != read[target]
                    or names(node.body, ast.Store)[src]
                ):
                    continue
                for n in (n for x in tail for n in ast.walk(x)):
                    if (
                        isinstance(n, ast.Name)
                        and n.id == target
                        and isinstance(n.ctx, ast.Load)
                    ):
                        n.id = src
                replaced.append(stmt)
            node.body = [x for x in node.body if x not in replaced]
    return tuple(ast.dump(s) for s in body)


def _complete_consumer(loop: ast.For, plan: NormalizedRow) -> bool:
    """Replacing a whole phase must not discard an unrelated operation."""
    dispatch = _dispatch(loop, 1)
    if dispatch is None or not plan.sweep:
        return False
    body, virtual, base = dispatch
    split = len(body) - len(plan.sweep)
    if split < 1 or sweep_signature(body[split:], plan.inverse) != plan.sweep:
        return False
    # Only PID decomposition and the original FP32 additive identity may
    # precede the two sweeps. Reject local side effects and argument rebinding.
    local_coords = {virtual}
    initialized = False
    for stmt in body[:split]:
        if not isinstance(stmt, ast.Assign) or (name := assigned_name(stmt)) is None:
            return False
        if name == plan.accumulator:
            if initialized or not same_expr(
                stmt.value, parse_expr("cutlass.Float32(0)")
            ):
                return False
            initialized = True
        elif (
            isinstance(stmt.value, ast.Name)
            and stmt.value.id in local_coords
            or (
                isinstance(stmt.value, ast.BinOp)
                and isinstance(stmt.value.op, ast.Sub)
                and isinstance(stmt.value.left, ast.Name)
                and stmt.value.left.id in local_coords
                and same_expr(stmt.value.right, parse_expr(f"cutlass.Int32({base})"))
            )
        ):
            local_coords.add(name)
        else:
            return False
        if name in {plan.source, plan.output, plan.weight, plan.epsilon}:
            return False
    return initialized


def install(fn: DeviceFunction, definition: ast.FunctionDef) -> None:
    if not fn.config.get(KEY):
        return
    env = CompileEnvironment.current()
    ir = fn.codegen.host_function.device_ir
    state = fn.cute_state.lanes
    pair = consecutive_terminal_roots(ir)
    p, c = state.cooperative_pair_producer, state.cooperative_pair_consumer
    if p is None or c is None or pair != (p.root, c.root) or p.output != c.source:
        raise exc.InvalidConfig(
            "cooperative normalization needs consecutive producers and consumers of the same BF16 row"
        )
    _require(
        can_offer(env.config_spec, fn.config.config),
        "a checked whole-grid interleaved launch",
    )
    dims = fn.tile_strategy.thread_block_dims()
    threads = prod(dims)
    _require(
        state.packed_row_publication
        and state.ordered_column_dims == {dims}
        and p.source_rank == 3
        and p.width == c.width
        and p.tile == 2 * threads
        and threads % _WARP == 0
        and p.width % p.tile == 0,
        "one complete pair per thread and full producer tiles",
    )
    count = p.width // p.tile
    _require(count > 1, "more than one complete producer tile")
    _require(len(definition.body) >= 3, "the final two roots and their original join")
    producer, join, consumer = definition.body[-3:]
    call = phase_barrier_call(join)
    if not (
        isinstance(producer, ast.For)
        and isinstance(consumer, ast.For)
        and _complete_consumer(consumer, c)
        and call is not None
        and isinstance(call.func, ast.Name)
        and call.func.id == "_cute_grid_barrier"
        and isinstance(arg := call.args[0], ast.Attribute)
        and arg.attr == "iterator"
        and isinstance(arg.value, ast.Name)
    ):
        raise exc.InvalidConfig(
            "cooperative normalization needs one unconditional local publication join"
        )
    sem = arg.value.id
    _require(
        any(
            type(a) is TensorArg
            and a.name == sem
            and private_scratch(a, scratch_words(p.width))
            for a in fn.arguments
        ),
        "private partial storage disjoint from the grid semaphore",
    )
    target = _producer_store(producer, p, count)
    if target is None:
        raise exc.InvalidConfig(
            "cooperative normalization cannot prove the actual producer store"
        )
    suite, pos, tile_pid = target
    # Effective alignment and all layout-dependent conditions are uniform
    # across these two phases. If too few CTAs are launched to cover the row,
    # run the complete original consumer without reading the private partials.
    keys = (
        "ready",
        "rounded",
        "word",
        "lo",
        "hi",
        "lo_square",
        "hi_square",
        "square",
        "warp_square",
        "partial",
        "block",
        "tid",
        "lane",
        "subtotal",
        "total",
        "inv",
        "idx",
        "norm",
        "weight",
        "result",
    )
    v = {k: fn.new_var("normalization_" + k, dce=False) for k in keys}
    si = p.source_rank - 1
    operands = (p.source, p.output, c.weight, c.output)
    align = " and ".join(f"{n}.iterator.alignment >= 4" for n in operands)
    strides = (
        f"{p.source}.layout.stride[{si}] == 1 and {p.source}.layout.stride[{si - 1}] % 2 == 0 "
        f"and {p.output}.layout.stride[1] == 1 and {c.output}.layout.stride[1] == 1 and {c.weight}.layout.stride[0] == 1"
    )
    pre = ast.parse(
        f"{v['ready']} = cutlass.const_expr({align} and {NUM_SM_VAR} >= {count}) and ({strides})\n{v['rounded']} = cutlass.Uint32(0)"
    ).body
    coordinate = linear_thread_index(
        {i: n for i, n in enumerate(dims) if i == 0 or n > 1},
        lambda x: f"cutlass.Int32({x})",
        cast_strides=False,
    )

    def code(template: str) -> list[ast.stmt]:
        return ast.parse(
            template.format(
                **v,
                src=p.source,
                pub=p.output,
                dest=c.output,
                wt=c.weight,
                a=p.low,
                b=p.high,
                col=p.column,
                pidthread=p.thread,
                warps=threads // _WARP,
                tilepid=tile_pid,
                sem=sem,
                start=_PARTIALS,
                bound=p.width,
                eps=c.epsilon,
                statistic=(
                    f"cute.math.rsqrt(cute.math.div(cutlass.Float32({v['total']}), cutlass.Float32({p.width}), approx=True, ftz=True) + {c.epsilon}, fastmath=True)"
                    if c.approximate
                    else f"cute.math.rsqrt({v['total']} / cutlass.Float32({p.width}) + {c.epsilon})"
                ),
                count=count,
                threads=threads,
                coordinate=coordinate,
            )
        ).body

    suite[pos:] = code("""
{rounded} = cutlass.Uint32(_cute_column_pack_bf16x2({a}, {b}))
cute.recast_ptr({pub}.iterator + {col}, dtype=cutlass.Uint32).store({rounded})
if {ready}:
    {lo}, {hi} = _cute_column_unpack_bf16x2({rounded})
    {lo_square}, {hi_square} = cute.arch.mul_packed_f32x2(({lo}, {hi}), ({lo}, {hi}), rnd="rn")
    {square} = {lo_square} + {hi_square}
    {warp_square} = cutlass.Float32(cute.arch.warp_reduction_sum({square}, threads_in_group=32))
    if cutlass.Int32(cute.arch.lane_idx()) == 0:
        {partial} = cute.recast_ptr({sem}.iterator + {start}, dtype=cutlass.Float32)
        ({partial} + {warps} * {tilepid} + {pidthread} // 32).store({warp_square})
""")
    partials = p.width // (2 * _WARP)
    if partials >= _WARP:
        loads = "{subtotal} = ({partial} + {lane}).load()\n"
    else:
        loads = "{subtotal} = cutlass.Float32(0)\n"
    for start in range(_WARP, partials - partials % _WARP, _WARP):
        loads += (
            f"{{subtotal}} = {{subtotal}} + ({{partial}} + {start} + {{lane}}).load()\n"
        )
    if tail := partials % _WARP:
        start = partials - tail
        loads += f"if {{lane}} < {tail}:\n    {{subtotal}} = {{subtotal}} + ({{partial}} + {start} + {{lane}}).load()\n"
    [branch] = code(
        """if {ready}:
    {block} = cutlass.Int32(cute.arch.block_idx()[0])
    if {block} < {count}:
        {tid} = {coordinate}
        {lane} = cutlass.Int32(cute.arch.lane_idx())
        {partial} = cute.recast_ptr({sem}.iterator + {start}, dtype=cutlass.Float32)
"""
        + textwrap.indent(loads, "        ")
        + """        {total} = cutlass.Float32(cute.arch.warp_reduction_sum({subtotal}, threads_in_group=32))
        {inv} = {statistic}
        {idx} = {threads} * {block} + {tid}
        {word} = (cute.recast_ptr({pub}.iterator, dtype=cutlass.Uint32) + {idx}).load()
        {lo}, {hi} = _cute_column_unpack_bf16x2({word})
        {norm} = _cute_publication_pack({lo} * {inv}, {hi} * {inv})
        {weight} = (cute.recast_ptr({wt}.iterator, dtype=cutlass.Uint32) + {idx}).load()
        {result} = _cute_publication_mul({norm}, {weight})
        (cute.recast_ptr({dest}.iterator, dtype=cutlass.Uint32) + {idx}).store(cutlass.Uint32({result}))
"""
    )
    assert isinstance(branch, ast.If)
    branch.orelse = [consumer]
    definition.body[-3:] = [*pre, producer, join, branch]
