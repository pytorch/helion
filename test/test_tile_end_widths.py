from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import skipIfRefEagerMark
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import load_generated_module
import helion.language as hl

pytestmark = [
    pytest.mark.cpu_only,
    skipIfRefEagerMark("checks generated tile edge code"),
]


@helion.kernel(backend="triton", static_shapes=False)
def _shape_edges(out):
    for tile in hl.tile(out.size(0), block_size=4):
        out[tile] = tile.end


@helion.kernel(backend="triton", static_shapes=False)
def _jagged_edges(out, offsets):
    for row in hl.grid(1):
        for tile in hl.tile(offsets[row], offsets[row + 1], block_size=4):
            out[tile] = tile.end


@skipUnlessBackends(["triton"])
@pytest.mark.parametrize("jagged", [False, True], ids=["shape", "tensor-bounds"])
def test_live_tile_ends_keep_the_partial_last_edge(jagged, tmp_path):
    from triton.runtime.interpreter import InterpretedFunction

    kernel = _jagged_edges if jagged else _shape_edges

    def inputs(n, start, end):
        backing = torch.full((n + 2,), -73, dtype=torch.int32)
        offsets = torch.tensor([start, end], dtype=torch.int32)
        args = (backing[1:-1], offsets) if jagged else (backing[1:-1],)
        expected = backing.clone()
        for begin in range(start, end, 4):
            stop = min(begin + 4, end)
            expected[1 + begin : 1 + stop] = stop
        return backing, args, expected

    sizes = [(9, 1, 8), (11, 2, 11)] if jagged else [(9, 0, 9), (11, 0, 11)]
    with cpu_render_target():
        source = cpu_bind(kernel, inputs(*sizes[0])[1]).to_code(
            helion.Config(block_sizes=[])
        )
    path = tmp_path / "dynamic_edges.py"
    path.write_text(source)
    run = getattr(load_generated_module(path), kernel.fn.__name__)

    def launch(fn, grid, *args, **kwargs):
        InterpretedFunction(fn.fn)[grid](*args)

    for size in sizes:
        backing, args, expected = inputs(*size)
        run(*args, _launcher=launch)
        torch.testing.assert_close(backing, expected)

    if not jagged:
        # A zero-origin input size needs no widened coordinate to clip the
        # partial last tile. A final cast to the int32 output could hide one.
        (edge,) = [
            n.value
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "tile_end"
        ]
        assert "tl.int64" not in ast.unparse(edge)

        # Evaluate only the generated scalar at the largest int32 shape.
        # Also cover an empty or surplus tile without allocating that shape.
        evaluate = compile(ast.Expression(edge), "<dynamic tile end>", "eval")
        with np.errstate(over="raise"):
            for end, begin in [(2**31 - 1, 2**31 - 4), (6, 12), (0, 0)]:
                scope = {
                    "out_size_0": np.int32(end),
                    "offset_0": np.int32(begin),
                    "_BLOCK_SIZE_0": np.int32(4),
                    "tl": SimpleNamespace(where=lambda cond, a, b: a if cond else b),
                }
                assert eval(evaluate, scope) == min(end, begin + 4)
