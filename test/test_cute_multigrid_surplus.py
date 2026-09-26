from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest
import torch

from .test_cute_grid_launch_extents import _code
from .test_cute_grid_launch_extents import _launch_block
from .test_cute_grid_launch_extents import _mixed_rank_copy
import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _prefix_copy(
    x: torch.Tensor,
    y: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    for row, col in hl.tile(x.shape, block_size=[1, None]):
        first[row, col] = x[row, col] + row.index[:, None].to(x.dtype)
    for tile in hl.tile(y.numel()):
        second[tile] = y[tile.index]
    return first, second


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _prefix_copy_reverse(
    x: torch.Tensor,
    y: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    for tile in hl.tile(y.numel()):
        second[tile] = y[tile.index]
    for row, col in hl.tile(x.shape, block_size=[1, None]):
        first[row, col] = x[row, col] + row.index[:, None].to(x.dtype)
    return first, second


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _two_fixed_axes(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    other = torch.empty_like(y)
    for batch, row, col in hl.tile(x.shape, block_size=[1, 1, None]):
        out[batch, row, col] = x[batch, row, col] + row.index[None, :, None]
    for tile in hl.tile(y.numel()):
        other[tile] = y[tile.index]
    return out, other


def _config(
    *,
    second_tile: int = 512,
    first_threads: int = 128,
    second_threads: int = 128,
    vector: int = 1,
    layout: str = "blocked",
    reverse: bool = False,
    first_tile: int = 128,
) -> helion.Config:
    blocks = [first_tile, second_tile]
    threads = [first_threads, second_threads]
    widths = [1, vector, vector]
    layouts = ["blocked", layout, layout]
    if reverse:
        blocks.reverse()
        threads.reverse()
        widths = [vector, 1, vector]
        layouts = [layout, "blocked", layout]
    return helion.Config(
        block_sizes=blocks,
        num_threads=threads,
        cute_vector_widths=widths,
        cute_lane_layouts=layouts,
    )


@pytest.mark.parametrize("second_tile", [256, 512, 1024])
@pytest.mark.parametrize("first_threads", [0, 128])
@pytest.mark.parametrize("vector", [1, 8])
@pytest.mark.parametrize("layout", ["blocked", "strided"])
def test_fixed_block_ids_do_not_index_tunable_slots(
    second_tile: int, first_threads: int, vector: int, layout: str
) -> None:
    code = _code(
        _mixed_rank_copy,
        (torch.empty((2, 1024)), torch.empty(2048), torch.empty(2048)),
        _config(
            second_tile=second_tile,
            first_threads=first_threads,
            vector=vector,
            layout=layout,
        ),
    )
    assert _launch_block(code) == (1, 128, 1)
    # No surplus exists: each root needs 128 physical threads even though the
    # second root walks multiple elements per thread. Keep safe mask elision.
    assert "mask_1 =" not in code
    assert "mask_2 =" not in code


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("second_threads", [0, 128, 512])
def test_actual_surplus_and_logical_tails_keep_their_masks(
    reverse: bool, tail: bool, second_threads: int
) -> None:
    columns = 1027 if tail else 1024
    y_size = 2051 if tail else 2048
    args = (
        torch.empty((3, columns)),
        torch.empty(y_size),
        torch.empty((3, columns * 2))[:, :columns],
        torch.empty(y_size),
    )
    code = _code(
        _prefix_copy_reverse if reverse else _prefix_copy,
        args,
        _config(second_threads=second_threads, reverse=reverse),
    )
    width = 512 if second_threads in (0, 512) else 128
    assert _launch_block(code) == (1, width, 1)
    col_id = 2 if reverse else 1
    mask = f"mask_{col_id}"
    predicate = None
    if width > 128 or tail:
        assignments = [
            node
            for node in ast.walk(ast.parse(code))
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == mask for t in node.targets)
        ]
        assert len(assignments) == 1
        expression = ast.unparse(assignments[0].value)
        assert f"indices_{col_id} < {columns}" in expression
        assert f"thread_idx()[1]) < _BLOCK_SIZE_{col_id}" in expression
        assert f"if {mask}:" in code
        predicate = compile(ast.Expression(assignments[0].value), "<mask>", "eval")
    else:
        assert f"{mask} =" not in code
    # Evaluate the emitted mask for every physical thread, including a
    # nonzero final program ID. An overwide CTA must never touch the preserved
    # suffix of the destination view, even if backing allocation bounds fit.
    active = 0
    for begin in range(0, columns, 128):
        for thread in range(width):
            index = begin + thread
            actual = True
            if predicate is not None:
                actual = eval(
                    predicate,
                    {"__builtins__": {}},
                    {
                        "cutlass": SimpleNamespace(Int32=int),
                        "cute": SimpleNamespace(
                            arch=SimpleNamespace(
                                thread_idx=lambda thread=thread: (0, thread, 0)
                            )
                        ),
                        f"indices_{col_id}": index,
                        f"_BLOCK_SIZE_{col_id}": 128,
                    },
                )
            assert actual == (thread < 128 and index < columns)
            active += actual
    assert active == columns


def test_default_thread_counts_keep_surplus_mask() -> None:
    config = _config()
    config.config.pop("num_threads")
    code = _code(
        _mixed_rank_copy,
        (torch.empty((2, 1024)), torch.empty(2048), torch.empty(2048)),
        config,
    )
    assert _launch_block(code) == (1, 512, 1)
    assert "thread_idx()[1]) < _BLOCK_SIZE_1" in code
    assert "if mask_1:" in code


def test_two_fixed_axes_resolve_the_logical_block_id() -> None:
    code = _code(
        _two_fixed_axes,
        (torch.empty((2, 3, 1024)), torch.empty(2048)),
        helion.Config(
            block_sizes=[128, 512],
            num_threads=[128, 128],
            cute_vector_widths=[1, 1, 1, 1],
            cute_lane_layouts=["blocked"] * 4,
        ),
    )
    assert _launch_block(code) == (1, 128, 1)


@pytest.mark.parametrize("threads", [(512, 128), (512, 512)])
def test_oversized_explicit_threads_keep_existing_rejection(
    threads: tuple[int, int],
) -> None:
    with pytest.raises(exc.BackendUnsupported, match="not divisible"):
        _code(
            _mixed_rank_copy,
            (torch.empty((2, 1024)), torch.empty(2048), torch.empty(2048)),
            _config(first_threads=threads[0], second_threads=threads[1]),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("vector", [1, 8])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("wide", [False, True])
def test_multigrid_prefix_runtime_canaries(
    reverse: bool, vector: int, tail: bool, wide: bool
) -> None:
    columns = 1027 if tail else 1024
    y_size = 2051 if tail else 2048
    x = torch.arange(3 * columns, device=DEVICE).reshape(3, columns).remainder(17)
    x = x.to(torch.bfloat16)
    y = torch.arange(y_size, device=DEVICE).remainder(13).to(torch.bfloat16)
    first_storage = torch.full(
        (3, columns * 2 + 32), -8192, dtype=x.dtype, device=DEVICE
    )
    first = first_storage[:, 16 : columns + 16]
    second_storage = torch.full((y_size + 32,), -8192, dtype=y.dtype, device=DEVICE)
    second = second_storage[16:-16]
    args = (x, y, first, second)
    saved = (x.clone(), y.clone())
    expected = x + torch.arange(3, device=DEVICE)[:, None].to(x.dtype)
    kernel = _prefix_copy_reverse if reverse else _prefix_copy
    run = kernel._bind_isolated(args).compile_config(
        _config(
            reverse=reverse,
            vector=vector,
            layout="strided",
            first_threads=0,
            second_threads=512 if wide else 128,
        )
    )

    def check() -> None:
        torch.testing.assert_close(first, expected, atol=0, rtol=0)
        torch.testing.assert_close(second, y, atol=0, rtol=0)
        torch.testing.assert_close((x, y), saved, atol=0, rtol=0)
        assert torch.all(first_storage[:, :16] == -8192)
        assert torch.all(first_storage[:, columns + 16 :] == -8192)
        assert torch.all(second_storage[:16] == -8192)
        assert torch.all(second_storage[-16:] == -8192)

    run(*args)
    check()
    run(*args)
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(*args)
    first.fill_(float("nan"))
    second.fill_(float("nan"))
    graph.replay()
    check()
