"""Pair adjacent BF16 columns in a proven full-CTA square sum and output sweep."""

from __future__ import annotations

import ast
import copy
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from ..compile_environment import CompileEnvironment
from ..device_function import TensorArg
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import same_expr
from .warp_row_partition import sweep_tensor_base

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from ..tile_strategy import DeviceGridState


def possible_row_rms(ir: DeviceIR) -> bool:
    """Structural flat-field membership; shape/stride and source proof follow."""
    if len(ir.root_ids) < 2:
        return False
    for root in ir.root_ids:
        targets = {n.target for n in ir.graphs[root].graph.nodes}
        if (
            torch.ops.aten.mean.dim in targets
            and torch.ops.aten.rsqrt.default in targets
        ):
            return True
    return False


def _expand(expr: ast.expr, bindings: dict[str, ast.expr]) -> ast.expr:
    """Inline only preceding scalar assignments; never chase a later rebinding."""

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            return copy.deepcopy(bindings.get(node.id, node))

    value = Substitute().visit(ast.parse(ast.unparse(expr), mode="eval").body)
    assert isinstance(value, ast.expr)
    return value


def _definitions(seq: Sequence[ast.AST]) -> dict[str, ast.expr] | None:
    bindings: dict[str, ast.expr] = {}
    if len(seq) > 12:
        return None
    for stmt in seq:
        if (
            not isinstance(stmt, ast.Assign)
            or (name := assigned_name(stmt)) is None
            or name in bindings
        ):
            return None
        bindings[name] = _expand(stmt.value, bindings)
    return bindings


def _ptr(name: str, rank: int, k: str) -> str:
    base = f"{name}.iterator"
    if rank == 2:
        base += f" + cutlass.Int32(0) * cutlass.Int32({name}.layout.stride[0])"
    return f"({base} + cutlass.Int32({k}) * cutlass.Int32({name}.layout.stride[{rank - 1}]))"


def _matches(value: ast.expr, source: str) -> bool:
    return same_expr(value, ast.parse(source, mode="eval").body)


def _loads(body: list[ast.stmt]) -> list[str]:
    return [
        name
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call) and (name := sweep_tensor_base(node, "load"))
    ]


def paired_sweeps(
    fn: DeviceFunction,
    parallel: list[ast.stmt],
    between: list[ast.AST],
    output: ast.For,
    acc: str,
    stat: str,
    dst: str,
    terms: list[ast.stmt],
    n: int,
    threads: int,
    k: str,
    tid: str,
) -> tuple[list[ast.stmt], ast.If] | None:
    """Retain the original scalar path and statistic, including both BF16 roundings."""
    if n % (2 * threads) or not 1 <= (trip := n // (2 * threads)) <= 32:
        return None
    reduction = _definitions(terms)
    middle = _definitions(between)
    values = _definitions(output.body[1:-1])
    inputs = _loads(terms)
    outputs = _loads(output.body[1:-1])
    if (
        len(inputs) != 1
        or len(outputs) != 2
        or inputs[0] not in outputs
        or len(set(outputs)) != 2
        or reduction is None
        or middle is None
        or values is None
        or acc not in reduction
    ):
        return None
    src = inputs[0]
    weight = next(x for x in outputs if x != src)
    tensors = {a.name: a.fake_value for a in fn.arguments if isinstance(a, TensorArg)}
    env = CompileEnvironment.current()
    for name, rank in ((src, 2), (weight, 1), (dst, 2)):
        if (
            (tensor := tensors.get(name)) is None
            or tensor.dtype != torch.bfloat16
            or tensor.ndim != rank
            or env.static_kernel_tensor_extent(tensor.shape[-1]) != n
            or (rank == 2 and env.static_kernel_tensor_extent(tensor.shape[0]) != 1)
        ):
            return None
    load = f"cutlass.Float32({_ptr(src, 2, k)}.load())"
    if not _matches(reduction[acc], f"{acc} + cutlass.Float32({load} * {load})"):
        return None
    # Keep the complete scalar statistic in the original emitted math. This
    # path consumes its FP32 inverse only when it is exactly a regular RMS.
    if not between or (inverse := assigned_name(between[-1])) is None:
        return None
    calc = middle[inverse]
    if not (
        isinstance(calc, ast.Call)
        and ast.unparse(calc.func) == "cute.math.rsqrt"
        and len(calc.args) == 1
        and isinstance(total := calc.args[0], ast.BinOp)
        and isinstance(total.op, ast.Add)
        and isinstance(total.right, (ast.Name, ast.Constant))
        and (
            _matches(
                calc,
                f"cute.math.rsqrt({stat} / cutlass.Float32({n}) + {ast.unparse(total.right)})",
            )
            or (
                env.settings.fast_math
                and _matches(
                    calc,
                    f"cute.math.rsqrt(cute.math.div(cutlass.Float32({stat}), cutlass.Float32(cutlass.Float32({n})), approx=True, ftz=True) + {ast.unparse(total.right)}, fastmath=True)",
                )
            )
        )
    ):
        return None
    store = output.body[-1]
    if not (
        isinstance(store, ast.Expr)
        and isinstance(call := store.value, ast.Call)
        and len(call.args) == 1
        and sweep_tensor_base(call, "store") == dst
        and _matches(
            _expand(call.args[0], values),
            f"cutlass.BFloat16(cutlass.BFloat16({load} * {inverse}) * {_ptr(weight, 1, k)}.load())",
        )
    ):
        return None
    if fn.config.get("cute_cooperative_row_norm"):
        from .cooperative_row_normalization import NormalizedRow

        root = fn.codegen.current_root_graph_info
        if root is not None and (
            isinstance(epsilon := total.right, ast.Constant)
            or (
                isinstance(epsilon, ast.Name)
                and epsilon.id in {a.name for a in fn.arguments}
            )
        ):
            fn.cute_state.lanes.cooperative_pair_consumer = NormalizedRow(
                root.graph_id,
                src,
                weight,
                dst,
                n,
                ast.unparse(total.right),
                acc,
                inverse,
                tid,
                approximate=bool(calc.keywords),
            )
    names = {
        key: fn.new_var(f"publication_{key}", dce=False)
        for key in (
            "words",
            "scalar_words",
            "word",
            "word_idx",
            "pair_step",
            "input_ptr",
            "weight_ptr",
            "output_ptr",
            "sq0",
            "sq1",
            "lo",
            "hi",
            "normalized",
            "w",
            "result",
        )
    }

    def body(src: str) -> list[ast.stmt]:
        return ast.parse(
            src.format(
                **names,
                src=src_name,
                wt=weight,
                dst=dst,
                tid=tid,
                threads=threads,
                trip=trip,
                acc=acc,
                inv=inverse,
            )
        ).body

    src_name = src
    guard = ast.parse(
        f"cutlass.const_expr({src}.iterator.alignment >= 4 and {weight}.iterator.alignment >= 4 and {dst}.iterator.alignment >= 4) "
        f"and {src}.layout.stride[1] == 1 and {weight}.layout.stride[0] == 1 and {dst}.layout.stride[1] == 1",
        mode="eval",
    ).body
    (cache,) = body("{words} = cute.make_rmem_tensor({trip}, cutlass.Uint32)")
    # Splitting the scalar loops into two layout branches makes the generic
    # load-forwarding pass lose sight of their common source. Preserve that
    # cache for the fallback as well: the public store is proved disjoint.
    (scalar_cache,) = body(
        "{scalar_words} = cute.make_rmem_tensor(2 * {trip}, cutlass.BFloat16)"
    )
    scalar_red = ast.parse(ast.unparse(parallel[1])).body[0]
    scalar_pub = ast.parse(ast.unparse(output)).body[0]
    if not (
        isinstance(scalar_red, ast.For)
        and isinstance(scalar_pub, ast.For)
        and isinstance(output.target, ast.Name)
    ):
        return None
    slot = f"{names['scalar_words']}[{output.target.id}]"

    def unique_source(loop: ast.For) -> tuple[int, ast.Assign] | None:
        loads = [
            (i, st)
            for i, st in enumerate(loop.body)
            if isinstance(st, ast.Assign)
            and isinstance(st.value, ast.Call)
            and sweep_tensor_base(st.value, "load") == src
        ]
        return loads[0] if len(loads) == 1 else None

    first, second = unique_source(scalar_red), unique_source(scalar_pub)
    if first is None or second is None or (value := assigned_name(first[1])) is None:
        return None
    scalar_red.body.insert(first[0] + 1, ast.parse(f"{slot} = {value}").body[0])
    second[1].value = ast.parse(slot, mode="eval").body
    red = body("""
{input_ptr} = cute.recast_ptr({src}.iterator, dtype=cutlass.Uint32)
{sq0} = cutlass.Float32(0)
{sq1} = cutlass.Float32(0)
for {pair_step} in cutlass.range_constexpr({trip}):
    {word_idx} = {tid} + {threads} * {pair_step}
    {word} = ({input_ptr} + {word_idx}).load()
    {words}[{pair_step}] = {word}
    {lo}, {hi} = _cute_publication_unpack({word})
    {sq0}, {sq1} = cute.arch.fma_packed_f32x2(({lo}, {hi}), ({lo}, {hi}), ({sq0}, {sq1}), rnd='rn')
{acc} = {sq0} + {sq1}
""")
    pub = body("""
{weight_ptr} = cute.recast_ptr({wt}.iterator, dtype=cutlass.Uint32)
{output_ptr} = cute.recast_ptr({dst}.iterator, dtype=cutlass.Uint32)
for {pair_step} in cutlass.range_constexpr({trip}):
    {word_idx} = {tid} + {threads} * {pair_step}
    {lo}, {hi} = _cute_publication_unpack({words}[{pair_step}])
    {normalized} = _cute_publication_pack({lo} * {inv}, {hi} * {inv})
    {w} = ({weight_ptr} + {word_idx}).load()
    {result} = _cute_publication_mul({normalized}, {w})
    ({output_ptr} + {word_idx}).store(cutlass.Uint32({result}))
""")
    return [
        parallel[0],
        cache,
        scalar_cache,
        ast.If(test=guard, body=red, orelse=[scalar_red]),
        *parallel[2:],
    ], ast.If(test=copy.deepcopy(guard), body=pub, orelse=[scalar_pub])


def install_paired_sweeps(
    fn: DeviceFunction,
    grid: DeviceGridState,
    expected: list[ast.AST],
    plans: tuple[list[ast.stmt], ast.If],
) -> None:
    from ...exc import InvalidConfig

    original = [ast.dump(s) for s in expected]
    packed, publish = plans

    def finish(wrapped: list[ast.AST]) -> None:
        # The normal lane/barrier pass must accept the scalar source. Replace
        # only its two sweeps after checking they still belong to the same
        # complete, previously verified region. Never discard a barrier it added.
        found: list[tuple[list[ast.AST], int]] = []

        def visit(body: list[ast.AST]) -> None:
            for i, stmt in enumerate(body):
                if isinstance(stmt, ast.Assign) and assigned_name(
                    stmt
                ) == assigned_name(expected[0]):
                    if [ast.dump(s) for s in body[i : i + len(original)]] == original:
                        found.append((body, i))
                for attr in ("body", "orelse"):
                    child = getattr(stmt, attr, None)
                    if isinstance(child, list):
                        visit(child)

        visit(wrapped)
        if len(found) != 1:
            raise InvalidConfig(
                "packed row publication requires unchanged original barrier placement"
            )
        suite, i = found[0]
        suite[i + len(original) - 1] = publish
        suite[i + 1 : i + 2] = packed[1:4]
        fact = fn.cute_state.lanes.cooperative_pair_consumer
        if fact is not None and assigned_name(expected[0]) == fact.thread:
            from .cooperative_row_normalization import sweep_signature

            fn.cute_state.lanes.cooperative_pair_consumer = replace(
                fact,
                sweep=sweep_signature(suite[i : i + len(original) + 2], fact.inverse),
            )
        fn.cute_state.lanes.packed_row_publication = True

    grid.body_finalizers.append(finish)
