from __future__ import annotations

import ast
import operator
import re
from types import SimpleNamespace

import pytest
import torch

from .test_cute_grid_launch_extents import _code
from .test_cute_grid_launch_extents import _mixed_rank_copy
from .test_cute_grid_launch_extents import _single_thread_axis
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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_sums_and_row_maxes(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.empty([x.size(0)], device=x.device)
    maxes = torch.empty([y.size(0)], device=y.device)
    for tile in hl.tile(x.size(0)):
        sums[tile] = x[tile, :].sum(-1)
    for tile in hl.tile(y.size(0)):
        maxes[tile] = y[tile, :].amax(-1)
    return sums, maxes


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
    _single_thread_axis(code, 128)
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
    axis = _single_thread_axis(code, width)
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
        # Launch repair spells the physical bound literally; a tail-only mask
        # keeps the tile strategy's constexpr spelling. Both evaluate to 128.
        assert re.search(
            rf"thread_idx\(\)\[{axis}\]\) < (128|_BLOCK_SIZE_{col_id})\b", expression
        )
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
                                thread_idx=lambda thread=thread: tuple(
                                    thread if dim == axis else 0 for dim in range(3)
                                )
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
    axis = _single_thread_axis(code, 512)
    assert f"thread_idx()[{axis}]) < 128" in code
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
    _single_thread_axis(code, 128)


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


def _execute_grid_copy_program(source, inputs):
    """Execute the generated scalar program with its real launch and addresses."""
    import itertools

    writes = {}
    current = {"block": (0, 0, 0), "thread": (0, 0, 0)}

    class Pointer:
        def __init__(self, tensor, offset=0):
            self.tensor = tensor
            self.offset = int(offset)

        def __add__(self, offset):
            return Pointer(self.tensor, self.offset + int(offset))

        def storage(self):
            count = self.tensor.untyped_storage().nbytes() // self.tensor.element_size()
            index = self.tensor.storage_offset() + self.offset
            assert 0 <= index < count, "generated address escapes allocation"
            return self.tensor.as_strided((count,), (1,), storage_offset=0), index

        def load(self):
            storage, index = self.storage()
            return storage[index].item()

        def store(self, value):
            storage, index = self.storage()
            key = (storage.data_ptr(), index)
            writes[key] = writes.get(key, 0) + 1
            storage[index] = value

    def launch(function, grid, *arguments, block):
        wrapped = [
            SimpleNamespace(
                iterator=Pointer(arg), layout=SimpleNamespace(stride=arg.stride())
            )
            if isinstance(arg, torch.Tensor)
            else arg
            for arg in arguments
        ]
        for cta in itertools.product(*(range(size) for size in grid)):
            current["block"] = (*cta, *((0,) * (3 - len(cta))))
            for thread in itertools.product(*(range(size) for size in block)):
                current["thread"] = thread
                function(*wrapped)

    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
        body.append(node)
    namespace = {
        "torch": torch,
        "_cute_python_mod": operator.mod,
        "cutlass": SimpleNamespace(Int32=int, Int64=int, Float32=float, Boolean=bool),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                block_idx=lambda: current["block"], thread_idx=lambda: current["thread"]
            )
        ),
        "_default_cute_launcher": launch,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            "<multigrid-copy-model>",
            "exec",
        ),
        namespace,
    )
    wrapper = next(
        node.name for node in reversed(body) if isinstance(node, ast.FunctionDef)
    )
    result = namespace[wrapper](*inputs)
    assert len(writes) == sum(output.numel() for output in result)
    assert set(writes.values()) == {1}, "every output must have exactly one writer"
    return result


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("wide_first", [False, True])
def test_multigrid_no_lane_root_keeps_coordinate_setup(reverse, tail, wide_first):
    rows, columns, count = 3, 67 if tail else 64, 35 if tail else 32
    x = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    y = torch.arange(count, dtype=torch.float32)
    first_storage = torch.full((rows, columns + 9), -12345.0)
    second_storage = torch.full((count + 11,), -54321.0)
    first, second = first_storage[:, 3 : 3 + columns], second_storage[4 : 4 + count]
    inputs = (x, y, first, second)
    before = (x.clone(), y.clone())
    config = _config(
        first_tile=64 if wide_first else 16,
        second_tile=16 if wide_first else 64,
        first_threads=32 if wide_first else 16,
        second_threads=16 if wide_first else 32,
        reverse=reverse,
    )
    source = _code(_prefix_copy_reverse if reverse else _prefix_copy, inputs, config)
    actual = _execute_grid_copy_program(source, inputs)
    expected = x + torch.arange(rows)[:, None]
    torch.testing.assert_close(actual, (expected, y), rtol=0, atol=0)
    expected_first = torch.full_like(first_storage, -12345.0)
    expected_first[:, 3 : 3 + columns] = expected
    expected_second = torch.full_like(second_storage, -54321.0)
    expected_second[4 : 4 + count] = y
    torch.testing.assert_close(first_storage, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(second_storage, expected_second, rtol=0, atol=0)
    torch.testing.assert_close((x, y), before, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("x_shape", "y_shape"), [((16, 8), (8, 40)), ((64, 32), (32, 16))]
)
def test_reduction_narrower_than_the_shared_launch_is_rejected(
    x_shape: tuple[int, int], y_shape: tuple[int, int]
) -> None:
    # Each root reduces its rows over its own lane count, but the roots share
    # the widest launch: the narrower reduction's surplus lanes read past its
    # row, reduce among themselves and race the row's result.
    with pytest.raises(exc.BackendUnsupported, match="surplus lanes"):
        _code(
            _row_sums_and_row_maxes,
            (torch.empty(x_shape), torch.empty(y_shape)),
            helion.Config(block_sizes=[1, 1]),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reductions_spanning_the_shared_launch_run() -> None:
    x = torch.randn(64, 32, device=DEVICE)
    y = torch.randn(48, 32, device=DEVICE)
    run = _row_sums_and_row_maxes._bind_isolated((x, y)).compile_config(
        helion.Config(block_sizes=[1, 1])
    )
    sums, maxes = run(x, y)
    torch.testing.assert_close(sums, x.sum(-1), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(maxes, y.amax(-1))
