from __future__ import annotations

import ast

import pytest

from test.test_cute_full_tile_bounds import SOURCE
from test.test_cute_full_tile_bounds import _execute
from test.test_cute_full_tile_bounds import _rewrite
from test.test_cute_full_tile_bounds import _sdk
from test.test_cute_full_tile_bounds import _text


def _source(expression: str) -> str:
    return SOURCE.replace(
        "record(index)",
        f"value = {expression}\n        record(cutlass.Int64(value))",
    )


@pytest.mark.parametrize("expression", ["index", "index // 2", "index % 127"])
@pytest.mark.parametrize("length", [8, 24, 1027, 33554432, ((1 << 31) - 1) // 8 * 8])
def test_checked_index_widening_preserves_full_tiles_and_exact_fallback(
    expression: str, length: int
) -> None:
    original, rewritten = _rewrite(_source(expression))
    assert len(rewritten) == 1 and isinstance(rewritten[0], ast.If)
    branch = rewritten[0]
    assert _text(branch.orelse) == _text(original)
    assert "cutlass.Int64(cutlass.Uint32(value))" in _text(branch.body)
    assert "cutlass.Int64(cutlass.Uint32(value))" not in _text(branch.orelse)
    sdk = _sdk()["cutlass"]
    sdk.Int32 = lambda value: (int(value) + (1 << 31)) % (1 << 32) - (1 << 31)
    sdk.Uint32 = lambda value: int(value) % (1 << 32)
    sdk.Int64 = lambda value: (int(value) + (1 << 63)) % (1 << 64) - (1 << 63)
    for block in sorted({0, (length - 1) // 8}):
        for thread in range(4):
            assert _execute(
                original, length, block=block, thread=thread, cutlass=sdk
            ) == _execute(rewritten, length, block=block, thread=thread, cutlass=sdk)


@pytest.mark.parametrize(
    "expression",
    [
        "index - 1",
        "index + 2147483647",
        "index * 2147483647",
        "runtime",
        "prefix[index]",
        "cutlass.Float32(index)",
        "index >= 0",
        "cutlass.Int64(2147483647) + 1",
    ],
)
def test_unproved_signed_or_wrapping_values_keep_original_widening(
    expression: str,
) -> None:
    original, rewritten = _rewrite(_source(expression))
    assert "cutlass.Int64(cutlass.Uint32(value))" not in _text(rewritten)
    assert "cutlass.Int64(value)" in _text(rewritten)
    assert isinstance(rewritten[0], ast.If)
    assert _text(rewritten[0].orelse) == _text(original)


@pytest.mark.parametrize(
    "body",
    [
        (
            "value = cutlass.Int32(cute.arch.thread_idx()[0])\n"
            "for step in range(2):\n"
            "    record(cutlass.Int64(value))\n"
            "    value = runtime\n"
        ),
        (
            "value = cutlass.Int32(cute.arch.thread_idx()[0])\n"
            "if runtime:\n"
            "    value = runtime\n"
            "record(cutlass.Int64(value))\n"
        ),
    ],
)
def test_control_flow_invalidates_the_widening_proof(body: str) -> None:
    original, rewritten = _rewrite(body)
    assert rewritten is original
    assert "cutlass.Int64(cutlass.Uint32(value))" not in _text(rewritten)


def test_non_rng_scalar_expression_uses_the_same_proof() -> None:
    original, rewritten = _rewrite(
        "index = cutlass.Int32(cute.arch.thread_idx()[0])\n"
        "record(cutlass.Int64(index))\n"
    )
    assert isinstance(rewritten[0], ast.If)
    assert "cutlass.Int64(cutlass.Uint32(index))" in _text(rewritten[0].body)
    assert _text(rewritten[0].orelse) == _text(original)
