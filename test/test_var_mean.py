from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _sums_masked_deviations(code: str) -> bool:
    # var_mean's second sum reads its squared deviations through a mask, in
    # Triton's tl.sum or CuTe's persistent or warp reduction.
    return any(
        f"{reduce}(_mask_to_" in code
        for reduce in ("tl.sum", "cutlass.Float32", "warp_reduction_sum")
    )


def _row_variance(x: torch.Tensor) -> torch.Tensor:
    var = torch.empty([x.size(0), 1], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        var[tile_b, :], _ = torch.var_mean(
            x[tile_b, :], dim=1, keepdim=True, correction=0
        )
    return var


@helion._testing.onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("renders generated code")
def test_var_mean_masks_the_squared_deviations_past_a_row() -> None:
    # A row of 200 runs over 256 lanes.  Fused into var_mean's second sum,
    # the squared deviations escaped the sum's input mask, so each lane past
    # the row added ``(0 - mean) ** 2``; now they are the sum's masked input.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_row_variance, static_shapes=True), (torch.empty(4, 200),)
        )
        code = bound.to_code(bound.config_spec.default_config())
    assert _sums_masked_deviations(code), code


def _jagged_row_variances(
    x_data: torch.Tensor, x_offsets: torch.Tensor
) -> torch.Tensor:
    b = x_offsets.size(0) - 1
    out = torch.zeros([b], dtype=x_data.dtype, device=x_data.device)
    for tile_b in hl.tile(b):
        starts = x_offsets[tile_b]
        nnz = x_offsets[tile_b.index + 1] - starts
        acc = hl.zeros([tile_b], dtype=x_data.dtype)
        for tile_k in hl.jagged_tile(nnz):
            idx = starts[:, None] + tile_k.index[None, :]
            v, _ = torch.var_mean(x_data[idx], dim=1, keepdim=True, correction=0)
            acc += v.sum(-1)
        out[tile_b] = acc
    return out


@helion._testing.onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("lowers the kernel")
def test_var_mean_over_the_jagged_dim_is_rejected() -> None:
    # As for the mean: each row of the parent tile holds its own number of
    # elements, so there is no one count to divide by.
    offsets = torch.tensor([0, 3, 4, 8, 10], dtype=torch.long)
    with (
        cpu_render_target(),
        pytest.raises(
            helion.exc.InductorLoweringError,
            match="variance over the jagged tile dim",
        ),
    ):
        cpu_bind(
            helion.kernel(_jagged_row_variances, autotune_effort="none"),
            (torch.empty(10), offsets),
        )


def _tile_variances(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    means = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        acc = hl.zeros([tile_b], dtype=x.dtype)
        mean_acc = hl.zeros([tile_b], dtype=x.dtype)
        for tile_m in hl.tile(x.size(1)):
            var, mean = torch.var_mean(
                x[tile_b, tile_m], dim=1, keepdim=True, correction=0
            )
            acc = acc + var.sum(-1)
            mean_acc = mean_acc + mean.sum(-1)
        out[tile_b] = acc
        means[tile_b] = mean_acc
    return out, means


@helion._testing.onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("renders generated code")
def test_var_mean_masks_the_squared_deviations_past_a_partial_tile() -> None:
    # The last 32-wide tile of 200 columns holds 8, and its 24 masked lanes
    # added ``(0 - mean) ** 2`` each to the tile's variance the same way.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_tile_variances, static_shapes=True), (torch.empty(4, 200),)
        )
        code = bound.to_code(helion.Config(block_sizes=[1, 32]))
    assert _sums_masked_deviations(code), code


def _row_var_mean(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    var = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    mean = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        var[tile_b], mean[tile_b] = torch.var_mean(x[tile_b, :], dim=1)
    return var, mean


@helion._testing.onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("renders generated code")
def test_var_mean_without_keepdim_returns_a_mean_of_its_own() -> None:
    # Without keepdim, the mean var_mean returns squeezed the realized mean:
    # a view, which the lowering refused as a float32 result (no cast back to
    # the input dtype makes it a buffer of its own).
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_row_var_mean, static_shapes=True), (torch.empty(4, 200),)
        )
        bound.to_code(bound.config_spec.default_config())


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@helion._testing.onlyBackends(["triton", "cute"])
def test_var_mean_counts_only_the_elements_of_a_row() -> None:
    kernel = helion.kernel(_row_variance, static_shapes=True)
    var_mean = helion.kernel(_row_var_mean, static_shapes=True)
    for width in (200, 1000):
        x = torch.randn(16, width, device=helion._testing.DEVICE) + 3.0
        _, var = helion._testing.code_and_output(kernel, (x,))
        torch.testing.assert_close(
            var, x.var(-1, correction=0, keepdim=True), rtol=1e-4, atol=1e-4
        )
        _, (unbiased, mean) = helion._testing.code_and_output(var_mean, (x,))
        expected_var, expected_mean = torch.var_mean(x, dim=1)
        torch.testing.assert_close(unbiased, expected_var, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(mean, expected_mean, rtol=1e-4, atol=1e-4)


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@helion._testing.onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("reference mode ignores block_sizes")
def test_var_mean_counts_only_the_elements_of_a_partial_tile() -> None:
    # The last 32-wide tile of 200 columns holds 8, which the variance and
    # the mean var_mean returns divide by.
    x = torch.randn(16, 200, device=helion._testing.DEVICE) + 3.0
    _, (tile_var, tile_mean) = helion._testing.code_and_output(
        helion.kernel(_tile_variances, static_shapes=True), (x,), block_sizes=[1, 32]
    )
    chunks = x.split(32, dim=1)
    torch.testing.assert_close(
        tile_var,
        sum(chunk.var(-1, correction=0) for chunk in chunks),
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        tile_mean, sum(chunk.mean(-1) for chunk in chunks), rtol=1e-4, atol=1e-4
    )


def _row_var_and_std(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    var = torch.empty([x.size(0), 1], dtype=x.dtype, device=x.device)
    std = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        var[tile_b, :] = torch.var(x[tile_b, :], dim=1, keepdim=True)
        std[tile_b] = torch.std(x[tile_b, :], dim=1)
    return var, std


def _tile_var_and_std(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    var = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    std = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(x.size(0)):
        var_acc = hl.zeros([tile_b], dtype=x.dtype)
        std_acc = hl.zeros([tile_b], dtype=x.dtype)
        for tile_m in hl.tile(x.size(1)):
            var_acc = var_acc + torch.var(x[tile_b, tile_m], dim=1)
            std_acc = std_acc + torch.std(x[tile_b, tile_m], dim=1)
        var[tile_b] = var_acc
        std[tile_b] = std_acc
    return var, std


@helion._testing.onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("renders generated code")
def test_var_and_std_mask_the_squared_deviations() -> None:
    # Their sum of squared deviations had two inputs, the value and its mean
    # ("reductions with >1 input"), until the deviations got a buffer of
    # their own; now it reads them masked, past a 200-wide row and past the
    # last 32-wide tile of 200 columns.
    with cpu_render_target():
        for fn, block_sizes in ((_row_var_and_std, None), (_tile_var_and_std, [1, 32])):
            bound = cpu_bind(
                helion.kernel(fn, static_shapes=True), (torch.empty(4, 200),)
            )
            config = (
                bound.config_spec.default_config()
                if block_sizes is None
                else helion.Config(block_sizes=block_sizes)
            )
            code = bound.to_code(config)
            assert _sums_masked_deviations(code), code


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@helion._testing.onlyBackends(["triton", "cute"])
def test_var_and_std_count_only_the_elements_of_a_row() -> None:
    # Divided by the elements less one (the default correction), not by the
    # padded row.
    kernel = helion.kernel(_row_var_and_std, static_shapes=True)
    for width in (200, 1000):
        x = torch.randn(16, width, device=helion._testing.DEVICE) + 3.0
        _, (var, std) = helion._testing.code_and_output(kernel, (x,))
        torch.testing.assert_close(var, x.var(-1, keepdim=True), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(std, x.std(-1), rtol=1e-4, atol=1e-4)


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@helion._testing.onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("reference mode ignores block_sizes")
def test_var_and_std_count_only_the_elements_of_a_partial_tile() -> None:
    # The last 32-wide tile of 200 columns holds 8: divided by them less one,
    # not by the block.
    x = torch.randn(16, 200, device=helion._testing.DEVICE) + 3.0
    _, (var, std) = helion._testing.code_and_output(
        helion.kernel(_tile_var_and_std, static_shapes=True), (x,), block_sizes=[1, 32]
    )
    chunks = x.split(32, dim=1)
    torch.testing.assert_close(
        var, sum(chunk.var(-1) for chunk in chunks), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        std, sum(chunk.std(-1) for chunk in chunks), rtol=1e-4, atol=1e-4
    )
