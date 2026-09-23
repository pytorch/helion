"""Guarded cp.async rings for a proven uniform warp-MMA reduction."""

from __future__ import annotations

import ast
from typing import cast

from ..ast_read_writes import ast_rename
from .scalar_recipe import _clone


def _statements(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


def signed_int32_bound(value: ast.expr) -> bool:
    """Make the lookahead arithmetic safe in a widened signed integer.

    A general Int64 range could overflow when adding the prefetch distance.
    Unknown or wider bounds retain the ordinary one-stage implementation.
    """
    if isinstance(value, ast.Constant):
        return type(value.value) is int and -(1 << 31) <= value.value < (1 << 31)
    return (
        isinstance(value, ast.Call)
        and ast.unparse(value.func) == "cutlass.Int32"
        and len(value.args) == 1
        and not value.keywords
    )


def pipeline_collective_loop(
    loop: ast.For,
    copies: list[ast.stmt],
    compute: list[ast.stmt],
    *,
    prefix: str,
    start: ast.expr,
    stop: ast.expr,
    block_k: int,
    stages: int,
) -> list[ast.stmt]:
    """Prefetch pure operand recipes and drain before shared storage escapes.

    The caller proves uniform loop bounds, a positive tile-size step, disjoint
    input/output storage, and exclusively asynchronous A/B staging. It supplies
    a separate ring stage in every shared copy destination and MMA source.
    Scalar vector tails remain legal: the CTA barrier publishes those writes
    together with the waited asynchronous copies. The compute body ends with
    a CTA barrier, so the next iteration cannot recycle a still-live stage.
    """
    assert stages in (2, 3, 4) and block_k > 0
    assert signed_int32_bound(start) and signed_int32_bound(stop)
    assert isinstance(loop.target, ast.Name)
    offset = loop.target.id
    stage_write = f"{prefix}_stage_write"
    stage_read = f"{prefix}_stage_read"
    prefetch = f"{prefix}_prefetch_k"
    recipe_coordinate = f"{prefix}_recipe_k"
    prefetch_index = f"{prefix}_prefetch_index"
    first, last = ast.unparse(start), ast.unparse(stop)
    staged_copies = ast_rename(
        ast.Module(body=copies, type_ignores=[]), {offset: recipe_coordinate}
    ).body

    def issue() -> list[ast.stmt]:
        # No operand recipe is evaluated outside the original reduction range.
        # Commit even an empty stage to preserve wait-group distances at tails.
        guarded = cast(
            "ast.If",
            _statements(f"if {prefetch} < cutlass.Int64({last}):\n    pass")[0],
        )
        # The range is signed Int32, so every active coordinate fits that type.
        # Widen only scheduling: operand index/mask arithmetic must retain its
        # original integer width, including overflow before a later cast.
        guarded.body = [
            *_statements(f"{recipe_coordinate} = cutlass.Int32({prefetch})"),
            *[_clone(statement) for statement in staged_copies],
        ]
        return [guarded, *_statements("cute.arch.cp_async_commit_group()")]

    prefill = cast(
        "ast.For",
        _statements(f"""
for {prefetch_index} in cutlass.range_constexpr({stages - 1}):
    {stage_write} = {prefetch_index}
    {prefetch} = cutlass.Int64({first}) + {prefetch_index} * {block_k}
""")[0],
    )
    prefill.body.extend(issue())
    index = f"((cutlass.Int64({offset}) - cutlass.Int64({first})) // {block_k})"
    loop.body = [
        *_statements(f"""
{stage_write} = cutlass.Int32(({index} + {stages - 1}) % {stages})
{prefetch} = cutlass.Int64({offset}) + {(stages - 1) * block_k}
"""),
        *issue(),
        *_statements(f"""
cute.arch.cp_async_wait_group({stages - 1})
{stage_read} = cutlass.Int32({index} % {stages})
"""),
        *compute,
    ]
    return [
        prefill,
        loop,
        *_statements("cute.arch.cp_async_wait_group(0)\ncute.arch.sync_threads()"),
    ]
