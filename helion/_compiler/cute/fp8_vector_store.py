"""Fuse scalar FP8 conversions collected by a proven packet store."""

from __future__ import annotations

import ast
from collections import Counter
from typing import TYPE_CHECKING

import torch

from ..ast_read_writes import ReadWrites

if TYPE_CHECKING:
    from collections.abc import Mapping

FP8_STORAGE_DTYPES = {
    torch.float8_e4m3fn: "cutlass.Float8E4M3FN",
    torch.float8_e5m2: "cutlass.Float8E5M2",
}
FP8_STORE_DTYPE_INFO = dict.fromkeys(
    FP8_STORAGE_DTYPES.values(), ("cutlass.Uint8", 1, "_cute_store_u8_vec")
)
FP8_CONVERSION_STORE_HELPERS = {
    "cutlass.Float8E4M3FN": "_cute_store_fp8e4m3fn_vec",
    "cutlass.Float8E5M2": "_cute_store_fp8e5m2_vec",
}
FP8_STORE_HELPERS = frozenset(
    {"_cute_store_u8_vec", *FP8_CONVERSION_STORE_HELPERS.values()}
)


def _flush_call(statement: ast.stmt) -> ast.Call | None:
    if isinstance(statement, ast.If) and len(statement.body) == 1:
        if statement.orelse:
            return None
        statement = statement.body[0]
    if not isinstance(statement, ast.Expr):
        return None
    call = statement.value
    if (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_cute_store_u8_vec"
        and len(call.args) == 2
        and not call.keywords
        and isinstance(call.args[1], ast.Name)
    ):
        return call
    return None


def _conversion_source(value: ast.expr) -> tuple[str, ast.expr] | None:
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "bitcast"
        and len(value.args) == 1
        and ast.unparse(value.args[0]) == "cutlass.Uint8"
        and not value.keywords
    ):
        return None
    cast = value.func.value
    if not (
        isinstance(cast, ast.Call)
        and ast.unparse(cast.func) in FP8_CONVERSION_STORE_HELPERS
        and len(cast.args) == 1
        and not cast.keywords
    ):
        return None
    dtype = ast.unparse(cast.func)
    source = cast.args[0]
    # Repeated casts to the same FP8 type are idempotent. Keep conversions
    # through other types, whose rounding is part of the scalar program.
    while (
        isinstance(source, ast.Call)
        and ast.unparse(source.func) == dtype
        and len(source.args) == 1
        and not source.keywords
    ):
        source = source.args[0]
    return dtype, source


def _fuse_loop(
    body: list[ast.stmt], index: int, name_counts: Mapping[str, int]
) -> None:
    loop = body[index]
    if not (
        isinstance(loop, ast.For)
        and not loop.orelse
        and isinstance(loop.iter, ast.Call)
        and ast.unparse(loop.iter.func) == "cutlass.range_constexpr"
        and len(loop.iter.args) == 1
        and isinstance(loop.iter.args[0], ast.Constant)
        and loop.iter.args[0].value in (2, 4, 8, 16)
        and index + 1 < len(body)
    ):
        return
    flush = _flush_call(body[index + 1])
    if flush is None:
        return
    values = flush.args[1]
    assert isinstance(values, ast.Name)
    list_name = values.id
    # A packet always contains exactly this loop's lanes, with no prior
    # elements or observable users of its compile-time collection.
    initializer = next(
        (
            statement
            for statement in reversed(body[:index])
            if list_name in ReadWrites.from_ast(statement).writes
        ),
        None,
    )
    if not (
        isinstance(initializer, ast.Assign)
        and len(initializer.targets) == 1
        and isinstance(initializer.targets[0], ast.Name)
        and initializer.targets[0].id == list_name
        and isinstance(initializer.value, ast.List)
        and not initializer.value.elts
    ):
        return
    append = None
    append_index = 0
    for statement_index, statement in enumerate(loop.body):
        if not isinstance(statement, ast.Expr):
            continue
        call = statement.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == list_name
            and call.func.attr == "append"
            and len(call.args) == 1
            and not call.keywords
        ):
            if append is not None:
                return
            append = call
            append_index = statement_index
    if (
        append is None
        or sum(
            isinstance(node, ast.Name) and node.id == list_name
            for node in ast.walk(loop)
        )
        != 1
    ):
        return
    if name_counts[list_name] != 3:
        return
    source = _conversion_source(append.args[0])
    if source is None:
        return
    dtype, argument = source
    if isinstance(argument, ast.Name) and name_counts[argument.id] == 2:
        # A scalar conversion often has its own SSA-like assignment. When
        # only this append consumes it, collect its input at the assignment's
        # original position. This retains assignment-time values even when an
        # input variable changes before the append.
        definition = next(
            (
                statement
                for statement in loop.body[:append_index]
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == argument.id
            ),
            None,
        )
        if (
            definition is not None
            and isinstance(definition.value, ast.Call)
            and ast.unparse(definition.value.func) == dtype
            and len(definition.value.args) == 1
            and not definition.value.keywords
        ):
            definition.value = definition.value.args[0]
    append.args[0] = argument
    assert isinstance(flush.func, ast.Name)
    flush.func.id = FP8_CONVERSION_STORE_HELPERS[dtype]


def fuse_fp8_vector_stores(body: list[ast.stmt]) -> list[ast.stmt]:
    """Convert collected FP32 lanes together, preserving scalar fallback.

    Pointer alignment, masks and memory ordering were already proved by the
    packet-store pass. This changes only its conversion operands. The tracing
    helper uses vector ``arith.truncf`` for FP32 inputs, the same operation as
    their scalar cast; other source types keep their original scalar casts.
    """
    if not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_cute_store_u8_vec"
        for statement in body
        for node in ast.walk(statement)
    ):
        return body
    name_counts = Counter(
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    )

    def rewrite(statements: list[ast.stmt]) -> None:
        for index, statement in enumerate(statements):
            _fuse_loop(statements, index, name_counts)
            for field in ("body", "orelse", "finalbody"):
                children = getattr(statement, field, None)
                if isinstance(children, list):
                    rewrite(children)

    rewrite(body)
    return body
