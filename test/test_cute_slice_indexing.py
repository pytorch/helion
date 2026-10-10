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
    with pytest.raises(BackendUnsupported, match="distinct tile axes"):
        bound.to_code(helion.Config())
