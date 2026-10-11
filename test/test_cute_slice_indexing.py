from __future__ import annotations

from examples.concatenate import concat2d_dim1_simple
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
import helion.language as hl
from helion.runtime.settings import _get_backend

pytestmark = skipUnlessBackends(["triton", "cute"])


@helion.kernel(static_shapes=True)
def _atomic_slice(
    x: torch.Tensor, out: torch.Tensor, reduce: hl.constexpr, partial: hl.constexpr
) -> torch.Tensor:
    n = x.size(1)
    for row in hl.tile(x.size(0)):
        if reduce:
            if partial:
                hl.atomic_add(out, [slice(3, n + 3)], x[row, :].sum(0))
            else:
                hl.atomic_add(out, [slice(None)], x[row, :].sum(0))
        else:
            if partial:
                hl.atomic_add(out, [row, slice(3, n + 3)], x[row, :])
            else:
                hl.atomic_add(out, [row, slice(None)], x[row, :])
    return out


@pytest.mark.parametrize("shape", [(33, 64), (65, 37), (256, 128)])
@pytest.mark.parametrize("reduce", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@skipIfNotCUDA()
@skipIfRefEager("compiles a pinned config; ref mode runs the kernel eagerly")
def test_atomic_slice_ownership_and_reduction(
    shape: tuple[int, int], reduce: bool, partial: bool
) -> None:
    x = torch.randn(shape, device=DEVICE)
    n = shape[1] + (6 if partial else 0)
    out_shape = (n,) if reduce else (shape[0], n)
    # A strided target checks that pointer lowering retains the tensor layout.
    storage = torch.randn((*out_shape[:-1], 2 * n), device=DEVICE)
    out = storage[..., ::2]
    untouched = storage[..., 1::2].clone()
    bound = _atomic_slice.bind((x, out, reduce, partial))
    compiled = bound.compile_config(helion.Config(block_sizes=[16]))
    for _ in range(3):
        x.normal_()
        expected = out.clone()
        update = x.sum(0) if reduce else x
        if partial:
            expected[..., 3:-3] += update
        else:
            expected += update
        torch.testing.assert_close(
            compiled(x, out, reduce, partial), expected, atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(storage[..., 1::2], untouched, atol=0, rtol=0)


def _concatenate_parallel_config(proven_bounds: bool) -> helion.Config:
    return helion.Config(
        block_sizes=[128],
        num_threads=[0, 1, 8],
        reduction_loops=[8],
        cute_vector_widths=[8, 1, 1],
        cute_proven_bounds=proven_bounds,
    )


@pytest.mark.parametrize(
    ("shape", "proven_bounds"),
    [
        ((256, 128, 256), None),
        ((33, 37, 64), None),
        ((65, 1, 128), None),
        ((65, 128, 1), None),
        ((256, 128, 256), False),
        ((256, 128, 256), True),
    ],
)
@skipUnlessBackends(["cute"])
@skipIfNotCUDA()
def test_concatenate_serial_slice_coordinates(
    shape: tuple[int, int, int], proven_bounds: bool | None
) -> None:
    m, n1, n2 = shape
    x = torch.randn((m, n1), device=DEVICE)
    y = torch.randn((m, n2), device=DEVICE)
    kernel = helion.kernel(concat2d_dim1_simple.fn, backend="cute", static_shapes=True)
    bound = kernel.bind((x, y))
    config = (
        helion.Config(block_sizes=[32])
        if proven_bounds is None
        else _concatenate_parallel_config(proven_bounds)
    )
    compiled = bound.compile_config(config)
    for _ in range(3):
        x.normal_()
        y.normal_()
        torch.testing.assert_close(
            compiled(x, y), torch.cat((x, y), dim=1), atol=0, rtol=0
        )


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("proven_bounds", [False, True])
def test_concatenate_separates_grid_and_full_slice_thread_axes(
    proven_bounds: bool,
) -> None:
    from test._cute_binding import _cpu_bind

    kernel = helion.kernel(concat2d_dim1_simple.fn, backend="cute", static_shapes=True)
    bound = _cpu_bind(kernel, (torch.empty(256, 128), torch.empty(256, 256)))
    # This candidate used to put the row and the second input's full slice
    # on the same axis. The serial first slice must not reserve a thread axis.
    code = bound.to_code(_concatenate_parallel_config(proven_bounds))
    # The single-thread slice claims no axis (``_claims_thread_axis``), so the
    # eight-thread full slice owns axis 0 and the 128-row tile axis 1.
    assert "block=(8, 128, 1)" in code
    assert (
        "offsets_0 = pid_flat * _BLOCK_SIZE_0 + cutlass.Int32(cute.arch.thread_idx()[1])"
        in code
    )
    assert "indices_2 = cutlass.Int32(cute.arch.thread_idx()[0])" in code
    # The eight slice threads take axis 0 and the rows axis 1.
    assert "cutlass.Int32(synthetic_lane_2) * 8" in code
    assert "block=(8, 128, 1)" in code


@skipUnlessBackends(["cute"])
def test_atomic_slice_rejects_distinct_update_coordinate_axis() -> None:
    from test._cute_binding import _cpu_bind

    @helion.kernel(backend="cute", static_shapes=True)
    def atomic_slice(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        for row, col in hl.tile((8, 8), block_size=(8, 8)):
            hl.atomic_add(out, [row.index + 1, slice(None)], x[row, col])
        return out

    bound = _cpu_bind(atomic_slice, (torch.empty((8, 8)), torch.empty((9, 8))))
    with pytest.raises(BackendUnsupported, match="distinct tile axes"):
        bound.to_code(helion.Config())


@skipUnlessBackends(["cute"])
@skipIfNotCUDA()
def test_atomic_cas_slice_masks_tail_and_returns_previous_values() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def cas(out: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        previous = torch.empty_like(out)
        for row in hl.tile(out.size(0)):
            previous[row, :] = hl.atomic_cas(out, [row, slice(None)], 0, values[row, :])
        return previous

    out = torch.randint(0, 2, (33, 37), device=DEVICE, dtype=torch.int32)
    values = torch.randint(2, 100, out.shape, device=DEVICE, dtype=out.dtype)
    before = out.clone()
    bound = cas.bind((out, values))
    previous = bound.compile_config(helion.Config(block_sizes=[16]))(out, values)
    torch.testing.assert_close(previous, before, atol=0, rtol=0)
    torch.testing.assert_close(
        out, torch.where(before == 0, values, before), atol=0, rtol=0
    )


@pytest.mark.parametrize("partial", [False, True])
@skipUnlessBackends(["cute"])
def test_atomic_slice_rejects_flattened_distinct_update_axes(partial: bool) -> None:
    from test._cute_binding import _cpu_bind

    @helion.kernel(backend="cute", static_shapes=True)
    def atomic_slice(
        x: torch.Tensor, out: torch.Tensor, partial: hl.constexpr
    ) -> torch.Tensor:
        for row, col in hl.tile((8, 8), block_size=(8, 8)):
            if partial:
                hl.atomic_add(out, [slice(1, 65)], x[row, col].reshape(-1))
            else:
                hl.atomic_add(out, [slice(None)], x[row, col].reshape(-1))
        return out

    bound = _cpu_bind(
        atomic_slice, (torch.empty((8, 8)), torch.empty(66 if partial else 64), partial)
    )
    # The flattening reshape is refused before the atomic sees its value: an
    # atomic reads each thread's own element, which the merge has moved.
    with pytest.raises(BackendUnsupported, match="moves elements between threads"):
        bound.to_code(helion.Config())


@helion.kernel(static_shapes=True)
def _register_block_slice_store(x: torch.Tensor, start: int) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m, 64], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc += x[tile_m, tile_n]
        out[tile_m, start : start + bn] = acc
    return out


@pytest.mark.parametrize("start", [0, 5])
@skipIfRefEager("checks the compiled store coordinates")
def test_register_block_accumulator_slice_store(start: int) -> None:
    """``out[tile_m, start:start + bn] = acc`` writes each column of the
    accumulator of the ``bn`` loop, which outlives the loop on its thread
    axis: CuTe indexes the range by that loop's axis, not by a new one."""
    x = torch.randint(-4, 5, (64, 512), device=DEVICE).float()
    extra = {"num_threads": [16, 2]} if _get_backend() == "cute" else {}
    config = helion.Config(block_sizes=[16, 2], **extra)
    out = _register_block_slice_store.bind((x, start)).compile_config(config)(x, start)
    expected = torch.zeros_like(out)
    expected[:, start : start + 16] = x.view(64, -1, 16).sum(1)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@skipUnlessBackends(["cute"])
@skipIfRefEager("checks a compiled refusal")
def test_register_block_accumulator_slice_store_lane_loop_refused() -> None:
    """With a lane loop over ``bn`` each thread keeps one accumulator for
    its several columns, so the carry stored after the loop is refused."""
    x = torch.randint(-4, 5, (64, 512), device=DEVICE).float()
    config = helion.Config(block_sizes=[16, 1], num_threads=[4, 1])
    bound = _register_block_slice_store.bind((x, 0))
    with pytest.raises(BackendUnsupported, match="own loop's block"):
        bound.compile_config(config)


@helion.kernel(static_shapes=True)
def _tile_bounds_slices(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n.begin : tile_n.end] = (
                x[tile_m, tile_n.begin : tile_n.end] * 2
            )
    return out


@skipIfRefEager("checks the compiled tile indexing")
def test_tile_begin_end_slice_is_the_tile() -> None:
    """``x[tile.begin:tile.end]`` indexes what ``x[tile]`` does, including the
    partial last tile, on both backends."""
    x = torch.randn(64, 250, device=DEVICE)
    extra = {"num_threads": [2, 32]} if _get_backend() == "cute" else {}
    config = helion.Config(block_sizes=[2, 32], **extra)
    out = _tile_bounds_slices.bind((x,)).compile_config(config)(x)
    torch.testing.assert_close(out, x * 2, atol=0, rtol=0)
