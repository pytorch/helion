"""Complete per-thread fragments for proved online max/exp/sum chunks."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import math
from typing import TypeVar
from typing import cast

from ..ast_read_writes import ReadWrites
from ..tile_strategy import _is_proven_relocatable_assignment
from . import cluster_online_pair as pair
from . import rolled_cluster_online_pair as rolled
from .uniform_packet_mask import integer_alignment
from .uniform_packet_mask import loop_alignment
from .uniform_packet_mask import uniform_packet_mask

_Statement = TypeVar("_Statement", bound=ast.stmt)


def _clone(node: _Statement) -> _Statement:
    return cast("_Statement", ast.parse(ast.unparse(node)).body[0])


def _call(statement: ast.stmt, name: str) -> ast.Call | None:
    if not pair._is_name_assign(statement):
        return None
    value = statement.value
    return (
        value
        if isinstance(value, ast.Call) and ast.unparse(value.func) == name
        else None
    )


def _count(loop: ast.For) -> int | None:
    iterator = loop.iter
    if (
        not loop.orelse
        and isinstance(loop.target, ast.Name)
        and isinstance(iterator, ast.Call)
        and ast.unparse(iterator.func) in ("range", "cutlass.range_constexpr")
        and not iterator.keywords
        and len(iterator.args) == 1
        and isinstance(iterator.args[0], ast.Constant)
        and type(iterator.args[0].value) is int
    ):
        return iterator.args[0].value
    return None


def _definitions(statements: list[ast.stmt]) -> dict[str, ast.expr] | None:
    definitions = {}
    for statement in statements:
        if not (
            pair._is_name_assign(statement)
            and _is_proven_relocatable_assignment(statement, allow_load=True)
        ):
            return None
        name = pair._assigned_name(statement)
        if name in definitions or name in pair._loads(statement.value):
            return None
        definitions[name] = pair._reparse_expr(statement.value)
    return definitions


def _canonical(node: ast.expr, definitions: dict[str, ast.expr]) -> ast.expr:
    return pair._strip_fp32_casts(
        pair._Inliner(definitions).visit(pair._reparse_expr(node))
    )


def _reduce(statement: ast.stmt, op: str) -> ast.Call | None:
    if not pair._is_name_assign(statement) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        ast.unparse(call.func)
        not in (
            "_cute_grouped_reduce_shared_two_stage",
            "_cute_grouped_reduce_shared_tree",
            "_cute_grouped_reduce_warp",
        )
        or len(call.args) < 4
        or not isinstance(call.args[0], ast.Name)
        or not isinstance(call.args[1], ast.Constant)
        or call.args[1].value != op
        or not rolled._float32(call.args[2])
        or rolled._literal(call.args[2]) != (-math.inf if op == "max" else 0.0)
    ):
        return None
    return call


@dataclass
class _Proof:
    width: int
    packets: int
    first_loop: ast.For
    max_fold: ast.Assign
    value: ast.expr
    caches: set[str]
    maximum_call_index: int
    sum_call_index: int
    next_maximum: str
    maximum: str
    denominator: str
    scale: float
    safe_ftz: bool


def _prove(
    loop: ast.For,
    outside: list[ast.stmt],
    initial_values: dict[str, float],
    read_counts: Counter[str],
) -> _Proof | None:
    packets = _count(loop)
    if packets not in (2, 4, 8, 16):
        return None
    maximum_sites = [
        (index, call)
        for index, statement in enumerate(loop.body)
        if (call := _reduce(statement, "max")) is not None
    ]
    sum_sites = [
        (index, call)
        for index, statement in enumerate(loop.body)
        if (call := _reduce(statement, "sum")) is not None
    ]
    if len(maximum_sites) != 1 or len(sum_sites) != 1:
        return None
    maximum_index, maximum_call = maximum_sites[0]
    sum_index, sum_call = sum_sites[0]
    if not maximum_index < sum_index or ast.unparse(maximum_call.func) != ast.unparse(
        sum_call.func
    ):
        return None
    if len(maximum_call.args) != len(sum_call.args) or [
        (kw.arg, ast.unparse(kw.value)) for kw in maximum_call.keywords
    ] != [(kw.arg, ast.unparse(kw.value)) for kw in sum_call.keywords]:
        return None
    before = loop.body[:maximum_index]
    first_loops = [statement for statement in before if isinstance(statement, ast.For)]
    if len(first_loops) != 1:
        return None
    first_loop = first_loops[0]
    width = _count(first_loop)
    if (
        width not in (2, 4, 8, 16)
        or width * packets > 128
        or first_loop is not before[-1]
    ):
        return None
    assert isinstance(first_loop.target, ast.Name)
    assert isinstance(loop.target, ast.Name)
    maximum_statement = cast("ast.Assign", loop.body[maximum_index])
    sum_statement = cast("ast.Assign", loop.body[sum_index])
    max_acc = cast("ast.Name", maximum_call.args[0]).id
    max_initializers = [
        statement
        for statement in before
        if pair._is_name_assign(statement) and pair._assigned_name(statement) == max_acc
    ]
    if (
        len(max_initializers) != 1
        or not rolled._float32(max_initializers[0].value)
        or rolled._literal(max_initializers[0].value) != -math.inf
    ):
        return None
    folds = [
        (index, statement)
        for index, statement in enumerate(first_loop.body)
        if pair._is_name_assign(statement) and pair._assigned_name(statement) == max_acc
    ]
    if len(folds) != 1:
        return None
    fold_index, max_fold = folds[0]
    fold_call = max_fold.value
    if (
        not isinstance(fold_call, ast.Call)
        or ast.unparse(fold_call.func) != "cute.arch.fmax"
        or len(fold_call.args) != 2
        or [(kw.arg, ast.unparse(kw.value)) for kw in fold_call.keywords]
        != [("nan", "True")]
        or not isinstance(fold_call.args[0], ast.Name)
        or fold_call.args[0].id != max_acc
        or max_fold is not first_loop.body[-1]
    ):
        return None
    caches = {
        pair._assigned_name(cast("ast.Assign", statement))
        for statement in before
        if _call(statement, "cute.make_rmem_tensor") is not None
    }
    if len(caches) != 1 or any(
        not isinstance(statement.value, ast.Call)
        or ast.unparse(statement.value.func) != "cute.make_rmem_tensor"
        or len(statement.value.args) != 2
        or ast.unparse(statement.value.args[0]) != str(width)
        or ast.unparse(statement.value.args[1]) != "cutlass.Float32"
        or statement.value.keywords
        for statement in before
        if pair._is_name_assign(statement) and pair._assigned_name(statement) in caches
    ):
        return None
    cache_stores = [
        (index, statement)
        for index, statement in enumerate(first_loop.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Subscript)
        and isinstance(statement.targets[0].value, ast.Name)
        and statement.targets[0].value.id in caches
    ]
    if len(cache_stores) != 1:
        return None
    store_index, store = cache_stores[0]
    store_target = store.targets[0]
    assert isinstance(store_target, ast.Subscript)
    if ast.unparse(store_target.slice) != first_loop.target.id:
        return None
    first_prefix = [
        statement
        for statement in first_loop.body[:fold_index]
        if statement is not store
    ]
    first_definitions = _definitions(first_prefix)
    prefix_definitions = _definitions(
        [
            statement
            for statement in before[:-1]
            if not (
                pair._is_name_assign(statement)
                and pair._assigned_name(statement) in caches | {max_acc}
            )
        ]
    )
    if first_definitions is None or prefix_definitions is None:
        return None
    definitions = prefix_definitions | first_definitions
    value = fold_call.args[1]
    maximum_domain, maximum_value = rolled._domain(
        _canonical(value, definitions), -math.inf
    )
    if not rolled._float32(maximum_value):
        return None
    region = loop.body[maximum_index + 1 : sum_index]
    second_loops = [statement for statement in region if isinstance(statement, ast.For)]
    if len(second_loops) != 1 or _count(second_loops[0]) != width:
        return None
    second_loop = second_loops[0]
    assert isinstance(second_loop.target, ast.Name)
    if second_loop.target.id != first_loop.target.id:
        return None
    max_result = pair._assigned_name(maximum_statement)
    next_sites = [
        statement
        for statement in region
        if (call := _call(statement, "cute.math.max")) is not None
        and len(call.args) == 2
        and [(kw.arg, ast.unparse(kw.value)) for kw in call.keywords]
        == [("propagate_nan", "True")]
    ]
    if len(next_sites) != 1:
        return None
    next_statement = cast("ast.Assign", next_sites[0])
    next_call = cast("ast.Call", next_statement.value)
    next_maximum = pair._assigned_name(next_statement)
    region_prefix = region[: region.index(next_statement)]
    max_definitions = _definitions(region_prefix)
    if max_definitions is None:
        return None
    operands = [_canonical(operand, max_definitions) for operand in next_call.args]
    if not all(rolled._float32(operand) for operand in operands):
        return None
    names = [cast("ast.Call", operand).args[0] for operand in operands]
    if not all(isinstance(name, ast.Name) for name in names):
        return None
    names = [cast("ast.Name", name) for name in names]
    if names[1].id == max_result:
        maximum = names[0].id
    elif names[0].id == max_result:
        maximum = names[1].id
    else:
        return None
    scalar_definitions = _definitions(
        [statement for statement in region if statement is not second_loop]
    )
    if scalar_definitions is None:
        return None
    # The next maximum is a scalar boundary, rather than a value to substitute.
    scalar_definitions.pop(next_maximum, None)
    if [
        ast.unparse(_canonical(arg, prefix_definitions))
        for arg in maximum_call.args[3:]
    ] != [
        ast.unparse(_canonical(arg, scalar_definitions)) for arg in sum_call.args[3:]
    ]:
        return None
    sum_acc = cast("ast.Name", sum_call.args[0]).id
    sum_initializers = [
        statement
        for statement in region
        if pair._is_name_assign(statement) and pair._assigned_name(statement) == sum_acc
    ]
    if (
        len(sum_initializers) != 1
        or not rolled._float32(sum_initializers[0].value)
        or rolled._literal(sum_initializers[0].value) != 0.0
    ):
        return None
    sum_folds = [
        (index, statement)
        for index, statement in enumerate(second_loop.body)
        if pair._is_name_assign(statement) and pair._assigned_name(statement) == sum_acc
    ]
    if len(sum_folds) != 1 or sum_folds[0][1] is not second_loop.body[-1]:
        return None
    sum_fold_index, sum_fold = sum_folds[0]
    if not (
        isinstance(sum_fold.value, ast.BinOp)
        and isinstance(sum_fold.value.op, ast.Add)
        and isinstance(sum_fold.value.left, ast.Name)
        and sum_fold.value.left.id == sum_acc
    ):
        return None
    sum_definitions = _definitions(second_loop.body[:sum_fold_index])
    if sum_definitions is None:
        return None
    sum_term = _canonical(
        sum_fold.value.right,
        prefix_definitions | scalar_definitions | sum_definitions,
    )
    cache_name = cast("ast.Name", store_target.value).id
    cached_value = _canonical(store.value, definitions)

    class Restore(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == cache_name
                and ast.unparse(node.slice) == cast("ast.Name", first_loop.target).id
            ):
                return pair._reparse_expr(cached_value)
            return self.generic_visit(node)

    sum_term = Restore().visit(sum_term)
    sum_term = pair._strip_fp32_casts(sum_term)
    sum_domain, exponential = rolled._domain(sum_term, 0.0)
    if (ast.unparse(maximum_domain) if maximum_domain is not None else None) != (
        ast.unparse(sum_domain) if sum_domain is not None else None
    ):
        return None
    while rolled._float32(exponential):
        exponential = exponential.args[0]
    if not pair._is_exp2(exponential) or exponential.keywords:
        return None
    argument = exponential.args[0]
    if not (
        isinstance(argument, ast.BinOp)
        and isinstance(argument.op, ast.Mult)
        and isinstance(argument.right, ast.Constant)
        and type(argument.right.value) is float
        and math.isfinite(argument.right.value)
        and argument.right.value > 0
    ):
        return None
    shift = argument.left
    while rolled._float32(shift):
        shift = shift.args[0]
    if not (
        isinstance(shift, ast.BinOp)
        and isinstance(shift.op, ast.Sub)
        and isinstance(shift.right, ast.Name)
        and shift.right.id == next_maximum
        and rolled._under_domain(shift.left, maximum_domain)
        == rolled._under_domain(maximum_value, maximum_domain)
    ):
        return None
    # Match the two carried values and the rescaling update exactly.
    tail = loop.body[sum_index + 1 :]
    tail_definitions = _definitions(tail[:-2])
    if tail_definitions is None or len(tail) < 2:
        return None
    denominator_statement = cast("ast.Assign", tail[-2])
    maximum_statement = tail[-1]
    update = _call(denominator_statement, "cute.math.fma")
    if (
        update is None
        or len(update.args) != 3
        or update.keywords
        or not isinstance(update.args[0], ast.Name)
        or update.args[0].id != pair._assigned_name(denominator_statement)
        or not pair._is_name_assign(maximum_statement)
        or pair._assigned_name(maximum_statement) != maximum
        or ast.unparse(maximum_statement.value) != next_maximum
    ):
        return None
    denominator = pair._assigned_name(denominator_statement)
    before_reads = Counter(
        name for statement in before for name in pair._loads(statement)
    )
    loop_reads = Counter(pair._loads(loop))
    if (
        before_reads[maximum]
        or before_reads[denominator]
        or loop_reads[max_acc] != 2
        or loop_reads[sum_acc] != 2
    ):
        return None
    alpha = _canonical(update.args[1], scalar_definitions)
    total = _canonical(update.args[2], tail_definitions)
    if not (
        pair._is_exp2(alpha)
        and not alpha.keywords
        and ast.unparse(alpha.args[0])
        == f"cutlass.Float32({maximum} - {next_maximum}) * {argument.right.value!r}"
        and rolled._float32(total)
        and isinstance(total.args[0], ast.Name)
        and total.args[0].id == pair._assigned_name(sum_statement)
    ):
        return None
    writes = Counter(
        node.id
        for node in ast.walk(loop)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    if (
        writes[maximum] != 1
        or writes[denominator] != 1
        or writes[max_acc] != 2
        or writes[sum_acc] != 2
        or any(writes[cache] != 1 for cache in caches)
    ):
        return None
    local_names = set(writes) - {maximum, denominator}
    inside_reads = Counter(pair._loads(loop))
    if any(read_counts[name] > inside_reads[name] for name in local_names) or any(
        set(pair._loads(statement)) & local_names for statement in outside
    ):
        return None
    # Every load, assignment and scalar update is pure. Only the proved local
    # cache stores are writable, so no moved load crosses a memory effect.
    allowed = {
        id(max_fold),
        id(sum_fold),
        id(store),
        id(loop.body[maximum_index]),
        id(loop.body[sum_index]),
        *(
            id(statement)
            for statement in before
            if pair._is_name_assign(statement)
            and pair._assigned_name(statement) in caches
        ),
    }
    for statement in loop.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.stmt) and not isinstance(node, ast.For):
                if id(node) in allowed:
                    continue
                if not _is_proven_relocatable_assignment(node, allow_load=True):
                    return None
    # Group coordinates must be identical for every packet, not one group per
    # lane. The packet variable may occur only in the value-producing prefix.
    for coordinate in maximum_call.args[3:]:
        expanded = _canonical(coordinate, prefix_definitions)
        if loop.target.id in pair._loads(expanded):
            return None
    return _Proof(
        width,
        packets,
        first_loop,
        max_fold,
        value,
        caches,
        maximum_index,
        sum_index,
        next_maximum,
        maximum,
        denominator,
        argument.right.value,
        initial_values.get(maximum) == -math.inf
        and initial_values.get(denominator) == 0.0,
    )


def lower_online_chunk_reduction(
    body: list[ast.stmt], constants: dict[str, int] | None = None
) -> list[ast.stmt]:
    """Fold all packets of a static online chunk before its block reductions."""
    used = {
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    counter = 0
    reads = Counter(
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )

    def fresh(kind: str) -> str:
        nonlocal counter
        while (name := f"_online_chunk_{kind}_{counter}") in used:
            counter += 1
        counter += 1
        used.add(name)
        return name

    def replace(
        loop: ast.For, proof: _Proof, alignments: dict[str, int]
    ) -> list[ast.stmt]:
        row = fresh("row")
        maximum_local = fresh("max")
        values = fresh("values")
        exponents = fresh("exponents")
        maximum_call = cast("ast.Assign", loop.body[proof.maximum_call_index])
        maximum_reduce = cast("ast.Call", maximum_call.value)
        maximum_argument = cast("ast.Name", maximum_reduce.args[0])
        max_acc = maximum_argument.id
        loop_index = cast("ast.Name", loop.target).id
        first_loop_index_name = cast("ast.Name", proof.first_loop.target).id
        clone = _clone(loop)
        first_loop_index = loop.body.index(proof.first_loop)
        cloned_first = clone.body[first_loop_index]
        assert isinstance(cloned_first, ast.For)
        cloned_first.body = [
            statement
            for statement in cloned_first.body
            if not (
                pair._is_name_assign(statement)
                and pair._assigned_name(statement) == max_acc
            )
            and not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Subscript)
                and isinstance(statement.targets[0].value, ast.Name)
                and statement.targets[0].value.id in proof.caches
            )
        ]
        cloned_first.body.append(
            pair._stmt(
                f"{row}[{loop_index} * {proof.width} + "
                f"{first_loop_index_name}] = {ast.unparse(proof.value)}"
            )
        )
        clone.iter = ast.parse(
            f"cutlass.range_constexpr({proof.packets})", mode="eval"
        ).body
        clone.body = [
            statement
            for statement in clone.body[: proof.maximum_call_index]
            if not (
                pair._is_name_assign(statement)
                and pair._assigned_name(statement) in proof.caches | {max_acc}
            )
        ]
        packet_alignments = alignments | {loop_index: 1}
        masked = uniform_packet_mask(
            clone.body[:-1], cloned_first, proof.width, row, packet_alignments
        )
        if masked is not None:
            clone.body[-1:] = masked
        result = [
            pair._stmt(
                f"{row} = cute.make_rmem_tensor({proof.width * proof.packets}, "
                "cutlass.Float32)"
            ),
            clone,
            pair._stmt(f"{values} = {row}.load()"),
        ]
        # Keep NaN propagation exact and balance the local max dependency tree.
        names = [f"{row}[{index}]" for index in range(proof.width * proof.packets)]
        while len(names) > 1:
            next_names = []
            for index in range(0, len(names), 2):
                name = fresh("tree")
                result.append(
                    pair._stmt(
                        f"{name} = cute.arch.fmax({names[index]}, "
                        f"{names[index + 1]}, nan=True)"
                    )
                )
                next_names.append(name)
            names = next_names
        result.append(pair._stmt(f"{maximum_local} = {names[0]}"))
        maximum_statement = _clone(maximum_call)
        cast("ast.Call", maximum_statement.value).args[0] = ast.Name(
            id=maximum_local, ctx=ast.Load()
        )
        result.append(maximum_statement)
        for statement in loop.body[proof.maximum_call_index + 1 : proof.sum_call_index]:
            if statement is proof.first_loop or isinstance(statement, ast.For):
                continue
            if (
                pair._is_name_assign(statement)
                and pair._assigned_name(statement)
                == cast(
                    "ast.Name",
                    cast(
                        "ast.Call",
                        cast("ast.Assign", loop.body[proof.sum_call_index]).value,
                    ).args[0],
                ).id
            ):
                continue
            result.append(_clone(statement))
        result.append(
            pair._stmt(
                f"{exponents} = cute.math.exp2(({values} - "
                f"{proof.next_maximum}) * {proof.scale!r}, "
                f"fastmath={proof.safe_ftz!r})"
            )
        )
        sum_statement = _clone(cast("ast.Assign", loop.body[proof.sum_call_index]))
        cast("ast.Call", sum_statement.value).args[0] = ast.parse(
            f"cutlass.Float32({exponents}.reduce(cute.ReductionOp.ADD, "
            "init_val=cutlass.Float32(0), reduction_profile=0))",
            mode="eval",
        ).body
        result.append(sum_statement)
        result.extend(
            _clone(statement) for statement in loop.body[proof.sum_call_index + 1 :]
        )
        return result

    def visit(
        statements: list[ast.stmt],
        initial: dict[str, float],
        initial_alignments: dict[str, int],
    ) -> list[ast.stmt]:
        result = []
        known = dict(initial)
        alignments = dict(initial_alignments)
        for index, statement in enumerate(statements):
            if isinstance(statement, ast.For):
                proof = _prove(statement, statements[index + 1 :], known, reads)
                if proof is not None:
                    result.extend(replace(statement, proof, alignments))
                else:
                    loop_facts = dict(alignments)
                    alignment = loop_alignment(statement, alignments)
                    if isinstance(statement.target, ast.Name):
                        loop_facts.pop(statement.target.id, None)
                        if alignment is not None:
                            loop_facts[statement.target.id] = alignment
                    statement.body = visit(statement.body, known, loop_facts)
                    result.append(statement)
                for name in ReadWrites.from_ast(statement).writes:
                    known.pop(name, None)
                    alignments.pop(name, None)
            elif isinstance(statement, ast.If):
                statement.body = visit(statement.body, known, alignments)
                statement.orelse = visit(statement.orelse, known, alignments)
                result.append(statement)
                for name in ReadWrites.from_ast(statement).writes:
                    known.pop(name, None)
                    alignments.pop(name, None)
            else:
                result.append(statement)
                alignment = (
                    integer_alignment(statement.value, alignments)
                    if pair._is_name_assign(statement)
                    else None
                )
                for name in ReadWrites.from_ast(statement).writes:
                    known.pop(name, None)
                    alignments.pop(name, None)
                if pair._is_name_assign(statement):
                    value = rolled._literal(statement.value)
                    if value is not None:
                        known[pair._assigned_name(statement)] = value
                    if alignment is not None:
                        alignments[pair._assigned_name(statement)] = alignment
        return result

    return visit(
        body, {}, {name: abs(value) for name, value in (constants or {}).items()}
    )
