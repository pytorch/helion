"""Batch statistics of the split BatchNorm ports of the better-benchmark study.

Inductor's kernels for these repros run Welford recurrences in fp32. The
ports take each block's sum and its second moment about a pivot in fp64
instead, pass them on in fp64, and round only what they return or store in
fp32 (``benchmarks/cute/better_benchmark/PORTING.md``, "Counts and
statistics"). The tests run each port's statistics kernels and compare every
kernel's outputs with a float64 evaluation of that kernel's inputs, and the
statistics a port returns with the float64 statistics of the activation
(the graph's math, its scalars the fp32 constants they are in the graph's
fp32 ops): on bf16 activations that are not centered (ReLU outputs, a coarse
grid, sparse channels, a large offset, small values whose variance is 0.38
of the eps 1e-5 of three ports and 0.004 of the other two's 1e-3), where fp32
means carry the bias of a rounded reciprocal, fp32 sums of squared
deviations round one way and fp32 partials lose what few-row blocks need;
with an infinity, and with finite values whose squares overflow fp32; and on
partials and fp32 activations whose common offset is a million times their
spread, where moments that are not taken about a pivot cancel.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

from benchmarks.cute.better_benchmark import common
import pytest
import torch

from helion._testing import DEVICE
from helion._testing import onlyOn
from helion._testing import output_only
from helion._testing import skipUnlessBackends
from helion.runtime.settings import _get_backend

if TYPE_CHECKING:
    from collections.abc import Callable
    import types

pytestmark = skipUnlessBackends(["triton", "cute"])

# Enough channels that a quotient by fp32 `/`, biased by a tenth of an ulp and
# more for the counts below, leaves some of them off the correctly rounded
# value.
CHANNELS = 1024
# Channels with special values; the others cycle through five families.
INFINITY = 5  # one +inf that is not a block's first row
HUGE = 13  # every element 1e25: finite, its square overflows fp32
HUGE_ONE = 17  # one element 1e25
CONSTANT = 21  # every element 1000
FLT_MAX = torch.finfo(torch.float32).max

# The graphs' momentum weights of the batch and of the running statistics.
MOMENTUM = 0.1
KEEP = 0.9

# A level's partials: each block's sum, mean, m2 and count.
_Partials = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
_Bounds = list[tuple[int, int]]


@functools.cache
def _port(repro: str) -> types.ModuleType:
    """The port of ``repro`` on the backend under test, loaded once per test
    process: its kernels keep what they compiled from one test to the next."""
    return common.load_port(repro, _get_backend())


def _f32(value: float) -> float:
    """A scalar of the graph as the fp32 constant it is in the graph's fp32
    ops (in eager on CUDA and in Inductor's kernels), as a Python float."""
    return torch.tensor(value, dtype=torch.float32).item()


def _bounds(count: int, block: int) -> _Bounds:
    return [(lo, min(lo + block, count)) for lo in range(0, count, block)]


@dataclasses.dataclass(frozen=True)
class _Port:
    """How to run the statistics kernels of one port on a test activation.

    ``levels`` holds, per split level, the ``[lo, hi)`` ranges of its inputs
    (activation rows for the first level, the first level's partials for a
    second one) that each of its blocks covers. ``last_level`` names the
    port's last statistics kernel; ``eps`` and ``unbias`` are the graph's
    literals for it. ``image`` is an ``(N, H, W)`` that the port's
    ``forward`` splits into blocks, and ``forward_statistics`` runs
    ``forward`` on such an activation (NCHW shape, channels-last memory) and
    running statistics, which it updates in place, and returns the batch
    mean and rstd.
    """

    rows: int
    levels: list[_Bounds]
    first_level: Callable[[types.ModuleType, torch.Tensor], _Partials]
    last_level: str
    eps: float
    unbias: float
    image: tuple[int, int, int]
    forward_statistics: Callable[
        [types.ModuleType, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]
    second_level: Callable[[types.ModuleType, _Partials], _Partials] | None = None


def _affine(channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A BatchNorm weight and bias (the statistics do not read them)."""
    weight = torch.ones([channels], dtype=torch.float32, device=DEVICE)
    return weight, torch.zeros([channels], dtype=torch.float32, device=DEVICE)


def _forward_one(
    mean: int, rstd: int
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """``forward`` of a port with one BatchNorm, whose outputs ``mean`` and
    ``rstd`` are the batch statistics."""

    def run(
        port: types.ModuleType,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = port.forward([x, running_mean, running_var, *_affine(x.size(1))])
        return outputs[mean].reshape(-1), outputs[rstd].reshape(-1)

    return run


def _forward_two(
    mean: int, rstd: int, *, contiguous: bool, pooled: bool
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """``forward`` of a port that adds two BatchNorms: the first one's
    statistics; the second one reads the same activation and its own running
    statistics."""

    def run(
        port: types.ModuleType,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, c = x.shape[:2]
        if contiguous:
            x = x.contiguous()
        affine = _affine(c)
        other = [torch.zeros_like(running_mean), torch.ones_like(running_var)]
        inputs = [x, running_mean, running_var, *affine, x.clone(), *other, *affine]
        if pooled:
            # The pooled output's as_strided size and strides, and its view.
            inputs += [[n, c, 1, 1], [c, 1, 1, 1], [n, c]]
        outputs = port.forward(inputs)
        return outputs[mean].reshape(-1), outputs[rstd].reshape(-1)

    return run


# The fp32 reciprocals of the row counts below are off by 0.7 to 1.0 times
# 2**-24, so a mean or a variance divided with fp32 `/` is biased. Where a
# block is a dim of a view, `reduction_loops=[8]` adds its rows in eight
# chains per channel; the looped kernels add chunks of 16 rows and of 2
# partials, and the last block of each of their levels is partial. The
# unbiased-variance factors are the graphs' literals.
PORTS: dict[str, _Port] = {
    # 64 blocks of 120 rows.
    "var_mean_42fad1ece813": _Port(
        rows=64 * 120,
        levels=[_bounds(64 * 120, 120)],
        first_level=lambda port, x: output_only(
            port.bn_welford_partials,
            (x.view(64, 120, x.size(1)),),
            block_sizes=[4, 16],
            reduction_loops=[8],
        ),
        last_level="bn_welford_combine_update",
        eps=1e-3,
        unbias=1.0001220852154804,
        image=(16, 8, 8),  # forward: 64 blocks of 16 rows
        forward_statistics=_forward_one(mean=2, rstd=0),
    ),
    # 49 blocks of 83 rows.
    "var_mean_var_mean_mean_0efb69a14422": _Port(
        rows=49 * 83,
        levels=[_bounds(49 * 83, 83)],
        first_level=lambda port, x: output_only(
            port.bn_welford_partials,
            (x.view(49, 83, x.size(1)),),
            block_sizes=[4, 16],
            reduction_loops=[8],
        ),
        last_level="bn_welford_combine_update",
        eps=1e-5,
        unbias=1.0001594642002871,
        image=(8, 7, 7),  # forward: 49 blocks of 8 rows
        forward_statistics=_forward_two(mean=0, rstd=1, contiguous=False, pooled=True),
    ),
    # 64 blocks of 120 rows, then 8 blocks of 8 partials.
    "var_mean_mean_5a1b8c788e4b": _Port(
        rows=64 * 120,
        levels=[_bounds(64 * 120, 120), _bounds(64, 8)],
        first_level=lambda port, x: output_only(
            port.bn_welford_row_partials,
            (x.view(64, 120, x.size(1)),),
            block_sizes=[4, 16],
            reduction_loops=[8],
        ),
        second_level=lambda port, partials: port.bn_welford_combine_partials(
            *(p.view(8, 8, p.size(1)) for p in partials)
        ),
        last_level="bn_stats_running_update",
        eps=1e-5,
        unbias=1.0000024912370735,
        image=(8, 16, 32),  # forward: 1024 blocks of 4 rows, 8 of 128 partials
        forward_statistics=_forward_one(mean=0, rstd=1),
    ),
    # 96 blocks of 40 rows and one of 13, then 3 blocks of 32 partials and
    # one of a single partial.
    "var_mean_mean_e790938418f4": _Port(
        rows=96 * 40 + 13,
        levels=[_bounds(96 * 40 + 13, 40), _bounds(97, 32)],
        first_level=lambda port, x: output_only(
            port.bn_welford_row_partials, (x, 40), block_sizes=[16, 16, 16]
        ),
        second_level=lambda port, partials: output_only(
            port.bn_welford_combine_partials, (*partials, 32), block_sizes=[4, 16, 2]
        ),
        last_level="bn_stats_running_update",
        eps=1e-3,
        unbias=1.0001594642002871,
        image=(24, 16, 16),  # forward: 9 blocks of 680 rows and one of 24
        forward_statistics=_forward_one(mean=0, rstd=1),
    ),
    # 5 images of 193 elements per channel in 4 blocks: three of 242
    # elements and one of 239, which straddle the images.
    "var_mean_var_mean_5cc92f5d49c1": _Port(
        rows=5 * 193,
        levels=[_bounds(5 * 193, 242)],
        first_level=lambda port, x: output_only(
            port.bn_welford_partials,
            (x.view(5, 193, x.size(1)), 4),
            block_sizes=[4, 16, 16],
        ),
        last_level="bn_welford_combine_update",
        eps=1e-5,
        unbias=1.0000398612827361,
        image=(4, 32, 32),  # forward: 4 blocks of 1024 elements
        forward_statistics=_forward_two(mean=4, rstd=0, contiguous=True, pooled=False),
    ),
}


def _activation(rows: int) -> torch.Tensor:
    """A bf16 ``[rows, CHANNELS]`` activation whose channels are not centered."""
    generator = torch.Generator(device=DEVICE).manual_seed(rows)
    z = torch.randn([rows, CHANNELS], generator=generator, device=DEVICE)
    family = torch.arange(CHANNELS, device=DEVICE) % 5
    x = torch.where(
        family == 0,
        torch.relu(z),  # a ReLU output
        torch.where(
            family == 1,
            3.0 + 0.1 * z,  # a coarse grid: 6 bf16 steps per standard deviation
            # a sparse ReLU output, and an offset of 30 standard deviations
            torch.where(family == 2, torch.relu(z - 2.0), z + 30.0),
        ),
    )
    # Small values, whose variance is about 2**-18 (3.8e-6): 0.004 of the eps
    # 1e-3 of var_mean_42fad1ece813 and var_mean_mean_e790938418f4, where rstd
    # follows eps, and 0.38 of the other three ports' 1e-5, where eps is the
    # larger term of `var + eps`.
    x = torch.where(family == 4, 2.0**-9 * (z + 1.0), x)
    x[rows // 2 + 1, INFINITY] = float("inf")
    x[:, HUGE] = 1e25
    x[rows // 3, HUGE_ONE] = 1e25
    x[:, CONSTANT] = 1000.0
    return x.to(torch.bfloat16)


def _block_moments(values: torch.Tensor, bounds: _Bounds) -> _Partials:
    """float64 ``(sum, mean, m2, count)`` of each block of the rows of
    ``values``."""
    sums, means, m2s, counts = [], [], [], []
    for lo, hi in bounds:
        # About the block's first row: exact differences.
        shifted = values[lo:hi] - values[lo]
        offset = shifted.mean(dim=0)
        sums.append(values[lo:hi].sum(dim=0))
        means.append(values[lo] + offset)
        m2s.append(((shifted - offset) ** 2).sum(dim=0))
        counts.append(torch.full_like(offset, hi - lo))
    return torch.stack(sums), torch.stack(means), torch.stack(m2s), torch.stack(counts)


def _combined_moments(partials: _Partials, bounds: _Bounds) -> _Partials:
    """float64 n-way (Chan) combine of each block of the ``partials``
    ``(sum, mean, m2, count)``; a partial without elements counts for
    nothing."""
    sum_p, mean_p, m2_p, weight_p = (p.to(torch.float64) for p in partials)
    sums, means, m2s, weights = [], [], [], []
    for lo, hi in bounds:
        weight = weight_p[lo:hi]
        total = weight.sum(dim=0)
        total_sum = sum_p[lo:hi].sum(dim=0)
        delta = mean_p[lo:hi] - total_sum / total
        sums.append(total_sum)
        means.append(total_sum / total)
        m2s.append((m2_p[lo:hi] + weight * delta**2).sum(dim=0))
        weights.append(total)
    return torch.stack(sums), torch.stack(means), torch.stack(m2s), torch.stack(weights)


def _ulp_errors(actual: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """``actual - exact`` in fp32 ulps of the float64 ``exact``."""
    ulp = torch.exp2(torch.floor(torch.log2(exact.abs())) - 23)
    return (actual.to(torch.float64) - exact) / ulp


def _assert_correctly_rounded(
    what: str, actual: torch.Tensor, exact: torch.Tensor
) -> None:
    """``actual`` is the float64 ``exact`` rounded to the nearest fp32."""
    assert actual.dtype == torch.float32, what
    zero = exact == 0
    overflow = exact > FLT_MAX
    assert (actual[zero] == 0).all(), what
    assert (actual[overflow] == float("inf")).all(), what
    errors = _ulp_errors(actual, exact)[~(zero | overflow)]
    # Half an ulp, and the float64 arithmetic of both sides.
    assert errors.abs().max().item() <= 0.5 + 1e-3, what


def _assert_partials(
    what: str, actual: _Partials, exact: _Partials, columns: torch.Tensor
) -> None:
    """The fp64 sums, means and second moments and the fp32 counts a kernel
    stores: the sums of these inputs are exact; the mean and m2 lose a few
    of fp64's last bits to a quotient and a subtraction."""
    sums, means, m2, weight = (t[:, columns] for t in actual)
    exact_sums, exact_means, exact_m2, exact_weight = (t[:, columns] for t in exact)
    assert sums.dtype == means.dtype == m2.dtype == torch.float64, what
    assert weight.dtype == torch.float32, what
    assert (sums == exact_sums).all(), f"{what} sum"
    spread = (exact_m2 / exact_weight.clamp_min(1.0)).sqrt()
    slack = 1e-13 * exact_means.abs() + 1e-12 * spread
    assert ((means - exact_means).abs() <= slack).all(), f"{what} mean"
    assert ((m2 - exact_m2).abs() <= 1e-9 * exact_m2).all(), f"{what} m2"
    assert (weight == exact_weight).all(), f"{what} weight"


def _running_statistics(channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A running mean and a running variance, positive like the test
    activations' means (a momentum update that cancels would magnify the
    batch statistic's own rounding) and zero in every fourth channel, where
    the update is the batch statistic times its weight alone."""
    generator = torch.Generator(device=DEVICE).manual_seed(channels)
    running_mean = torch.rand([channels], generator=generator, device=DEVICE) + 0.5
    running_var = torch.rand([channels], generator=generator, device=DEVICE) + 0.5
    running_mean[::4] = 0.0
    running_var[::4] = 0.0
    return running_mean, running_var


def _last_level(
    what: str,
    spec: _Port,
    port: types.ModuleType,
    partials: _Partials,
    columns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the port's last statistics kernel as its ``forward`` does and
    check the mean, rstd and the momentum updates it stores; returns the
    mean and rstd."""
    running_mean, running_var = _running_statistics(partials[0].size(1))
    old_mean = running_mean.to(torch.float64)[columns]
    old_var = running_var.to(torch.float64)[columns]
    mean, rstd = getattr(port, spec.last_level)(
        *partials,
        running_mean,
        running_var,
        port.EPS,
        port.MOMENTUM,
        port.KEEP,
        port.UNBIAS,
    )
    exact = _combined_moments(partials, [(0, partials[0].size(0))])
    _, exact_mean, exact_m2, total = (t[0, columns] for t in exact)
    exact_var = exact_m2 / total
    # The graph's math in float64 with its scalars as fp32 constants, each
    # result rounded once; the running mean is updated from the mean as the
    # kernel stores it. fp32 rsqrt, the GPU's approximation, is off by more
    # than half an ulp; an fp32 momentum update rounds two or three times;
    # and in float64 the literals themselves are other numbers than the
    # graph's constants (a quarter of an ulp and more, in one direction).
    eps, unbias = _f32(spec.eps), _f32(spec.unbias)
    momentum, keep = _f32(MOMENTUM), _f32(KEEP)
    _assert_correctly_rounded(f"{what} mean", mean[columns], exact_mean)
    _assert_correctly_rounded(f"{what} rstd", rstd[columns], (exact_var + eps).rsqrt())
    _assert_correctly_rounded(
        f"{what} running_mean",
        running_mean[columns],
        mean[columns].to(torch.float64) * momentum + old_mean * keep,
    )
    _assert_correctly_rounded(
        f"{what} running_var",
        running_var[columns],
        exact_var * unbias * momentum + old_var * keep,
    )
    return mean, rstd


def _statistics_of(
    x64: torch.Tensor, spec: _Port
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """float64 mean, variance and rstd of each channel of the rows ``x64``."""
    exact_mean = x64.mean(dim=0)
    exact_var = x64.var(dim=0, correction=0)
    return exact_mean, exact_var, torch.rsqrt(exact_var + _f32(spec.eps))


@onlyOn(["cuda", "xpu"])
@pytest.mark.parametrize("repro", list(PORTS))
def test_every_kernel_stores_the_statistics_of_what_it_read(repro: str) -> None:
    spec = PORTS[repro]
    port = _port(repro)
    x = _activation(spec.rows)
    x64 = x.to(torch.float64)
    columns = torch.ones([CHANNELS], dtype=torch.bool, device=DEVICE)
    columns[INFINITY] = False

    # Every kernel against a float64 evaluation of what it read.
    first = spec.first_level(port, x)
    _assert_partials("level 0", first, _block_moments(x64, spec.levels[0]), columns)
    partials = first
    if spec.second_level is not None:
        partials = spec.second_level(port, first)
        exact = _combined_moments(first, spec.levels[1])
        _assert_partials("level 1", partials, exact, columns)
    mean, rstd = _last_level("last level", spec, port, partials, columns)

    # End to end: the mean and rstd are the activation's float64 statistics
    # rounded once (partials that were rounded to fp32 on the way would add
    # their own roundings, a few ulps where blocks have few rows).
    exact_mean, _, exact_rstd = _statistics_of(x64[:, columns], spec)
    _assert_correctly_rounded("mean", mean[columns], exact_mean)
    _assert_correctly_rounded("rstd", rstd[columns], exact_rstd)

    # A block that holds an infinity stores NaN statistics, as Inductor's
    # recurrence does (inf - inf); the other blocks do not.
    row = spec.rows // 2 + 1
    block = next(i for i, (lo, hi) in enumerate(spec.levels[0]) if lo < row < hi)
    others = [i for i in range(first[0].size(0)) if i != block]
    assert torch.isnan(first[0][block, INFINITY])
    assert not torch.isfinite(first[1][block, INFINITY])
    assert torch.isnan(first[2][block, INFINITY])
    for stored in first[:3]:
        assert torch.isfinite(stored[others, INFINITY]).all()
    assert torch.isnan(mean[INFINITY]) and torch.isnan(rstd[INFINITY])
    # Finite values never do: a constant channel has its value as the mean
    # and no variance, however large the value, and one huge element leaves
    # a finite mean and a variance that fp64 holds where fp32 overflows.
    huge = x[0, HUGE].to(torch.float32)
    assert mean[HUGE] == huge and mean[CONSTANT] == 1000.0
    assert rstd[HUGE] == rstd[CONSTANT] and torch.isfinite(rstd[CONSTANT])
    assert torch.isfinite(mean[HUGE_ONE]) and rstd[HUGE_ONE] > 0

    # The combines take their second moments about a pivot too: partial
    # means whose common offset is a million times their spread, where the
    # weighted sum of squares minus the square of the weighted sum cancels
    # to nothing.
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    noise = torch.randn(first[0].shape, generator=generator, device=DEVICE)
    means = (2.0**20 + noise).to(torch.float32).to(torch.float64)
    m2 = torch.rand(first[0].shape, generator=generator, device=DEVICE) + 0.5
    partials = (means * first[3], means, m2.to(torch.float64), first[3])
    everything = torch.ones([CHANNELS], dtype=torch.bool, device=DEVICE)
    if spec.second_level is not None:
        offset = spec.second_level(port, partials)
        exact = _combined_moments(partials, spec.levels[1])
        _assert_partials("offset level 1", offset, exact, everything)
        partials = offset
    _last_level("offset last level", spec, port, partials, everything)


@onlyOn(["cuda", "xpu"])
@pytest.mark.parametrize("repro", list(PORTS))
def test_the_block_moments_are_taken_about_a_pivot(repro: str) -> None:
    # An fp32 activation whose offset is a million times its spread: sums of
    # the values and of their squares would lose the variance, where sums of
    # the deviations from one of the block's rows keep it. And an infinity
    # in a block's first row, the pivot, reaches the statistics as NaN like
    # any other.
    spec = PORTS[repro]
    port = _port(repro)
    channels = 64
    generator = torch.Generator(device=DEVICE).manual_seed(spec.rows)
    z = torch.randn([spec.rows, channels], generator=generator, device=DEVICE)
    x = (2.0**20 + z).to(torch.float32)
    x[spec.levels[0][2][0], INFINITY] = float("-inf")
    partials = spec.first_level(port, x)
    columns = torch.ones([channels], dtype=torch.bool, device=DEVICE)
    columns[INFINITY] = False
    exact = _block_moments(x.to(torch.float64), spec.levels[0])
    _assert_partials("level 0", partials, exact, columns)
    assert torch.isnan(partials[0][2, INFINITY])
    assert torch.isnan(partials[2][2, INFINITY])
    assert torch.isfinite(partials[0][:2, INFINITY]).all()


@onlyOn(["cuda", "xpu"])
@pytest.mark.parametrize("repro", list(PORTS))
def test_forward_returns_the_graphs_statistics_rounded_once(repro: str) -> None:
    # forward() end to end, on images that give its blocks few rows: every
    # statistic it returns or stores is the graph's math on the activation in
    # float64, with the graph's scalars as fp32 constants, rounded once. The
    # running mean is the update of the mean that forward returns. Partials
    # in fp32, an fp32 momentum update, fp32 rsqrt, or a scalar handed on as
    # the literal's fp64 value each leave elements that are not the rounded
    # value.
    spec = PORTS[repro]
    port = _port(repro)
    n, h, w = spec.image
    rows = _activation(n * h * w)
    x = rows.view(n, h, w, CHANNELS).permute(0, 3, 1, 2)
    running_mean, running_var = _running_statistics(CHANNELS)
    old_mean = running_mean.to(torch.float64)
    old_var = running_var.to(torch.float64)
    mean, rstd = spec.forward_statistics(port, x, running_mean, running_var)
    finite = torch.ones([CHANNELS], dtype=torch.bool, device=DEVICE)
    finite[INFINITY] = False
    exact_mean, exact_var, exact_rstd = _statistics_of(
        rows.to(torch.float64)[:, finite], spec
    )
    unbias, momentum, keep = _f32(spec.unbias), _f32(MOMENTUM), _f32(KEEP)
    _assert_correctly_rounded("mean", mean[finite], exact_mean)
    _assert_correctly_rounded("rstd", rstd[finite], exact_rstd)
    _assert_correctly_rounded(
        "running_var",
        running_var[finite],
        exact_var * unbias * momentum + old_var[finite] * keep,
    )
    _assert_correctly_rounded(
        "running_mean",
        running_mean[finite],
        mean[finite].to(torch.float64) * momentum + old_mean[finite] * keep,
    )
    for statistic in (mean, rstd, running_mean, running_var):
        assert torch.isnan(statistic[INFINITY])


@onlyOn(["cuda", "xpu"])
# The counts for which var_mean_var_mean_5cc92f5d49c1's four blocks of
# ceil(count / 4) elements leave a block (three of them for one element)
# past the end.
@pytest.mark.parametrize("count", [1, 2, 3, 5, 6, 9])
def test_a_block_past_the_end_has_no_weight(count: int) -> None:
    # Five elements in four blocks are two, two, one and none. A block past
    # the end stores zeros of zero weight, which the combine does not see.
    repro = "var_mean_var_mean_5cc92f5d49c1"
    port = _port(repro)
    channels = 64
    generator = torch.Generator(device=DEVICE).manual_seed(count)
    z = torch.randn([count, channels], generator=generator, device=DEVICE)
    x = (z + 3.0).to(torch.bfloat16)
    partials = port.bn_welford_partials(x.view(1, count, channels), 4)
    block = (count + 3) // 4
    bounds = _bounds(count, block)
    exact = _block_moments(x.to(torch.float64), bounds)
    everything = torch.ones([channels], dtype=torch.bool, device=DEVICE)
    _assert_partials(
        "blocks", tuple(t[: len(bounds)] for t in partials), exact, everything
    )
    for name, stored in zip(("sum", "mean", "m2", "weight"), partials, strict=True):
        assert (stored[len(bounds) :] == 0).all(), name
    mean, rstd = _last_level("last level", PORTS[repro], port, partials, everything)
    exact_mean, _, exact_rstd = _statistics_of(x.to(torch.float64), PORTS[repro])
    _assert_correctly_rounded("mean", mean, exact_mean)
    _assert_correctly_rounded("rstd", rstd, exact_rstd)


@onlyOn(["cuda", "xpu"])
@skipUnlessBackends(["cute"])
def test_the_looped_kernels_sum_in_fp64_at_a_folding_config() -> None:
    # At these CuTe configs the sums inside the looped kernels' tile loops are
    # folded into per-lane accumulators and no tile needs a mask (each block
    # size divides its extent), so the kernels' fp64 sums run through the
    # fold. An fp32 accumulator would lose 1e-7 of the squares' and the second
    # level's moments' m2, and the low bits of the second level's sums.
    fold = {
        "num_threads": [1, 32, 32],
        "cute_vector_widths": [2, 8, 4],
        "cute_defer_tile_reduce": True,
    }
    channels = 64
    everything = torch.ones([channels], dtype=torch.bool, device=DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(7)

    def activation(rows: int) -> torch.Tensor:
        z = torch.randn([rows, channels], generator=generator, device=DEVICE)
        return (z + 3.0).to(torch.bfloat16)

    # var_mean_var_mean_5cc92f5d49c1: four blocks of 1024 elements.
    port = _port("var_mean_var_mean_5cc92f5d49c1")
    x = activation(4096)
    partials = output_only(
        port.bn_welford_partials,
        (x.view(8, 512, channels), 4),
        block_sizes=[4, 32, 256],
        **fold,
    )
    exact = _block_moments(x.to(torch.float64), _bounds(4096, 1024))
    _assert_partials("5cc9 level 0", partials, exact, everything)

    # var_mean_mean_e790938418f4: three blocks of 512 rows and one of 208,
    # then 397 partials in blocks of 128.
    port = _port("var_mean_mean_e790938418f4")
    x = activation(3 * 512 + 208)
    partials = output_only(
        port.bn_welford_row_partials, (x, 512), block_sizes=[4, 32, 256], **fold
    )
    exact = _block_moments(x.to(torch.float64), _bounds(3 * 512 + 208, 512))
    _assert_partials("e790 level 0", partials, exact, everything)
    noise = torch.randn([397, channels], generator=generator, device=DEVICE)
    weights = torch.full([397, channels], 680.0, device=DEVICE)
    # Multiples of 2**-20 of about 31 bits: fp64 adds them exactly in any
    # order, fp32 does not hold one of them.
    sums = torch.round((2040.0 + 1000.0 * noise.to(torch.float64)) * 2.0**20)
    sums = sums / 2.0**20
    m2 = torch.rand([397, channels], generator=generator, device=DEVICE) + 0.5
    partials = (sums, sums / 680.0, m2.to(torch.float64), weights)
    combined = output_only(
        port.bn_welford_combine_partials,
        (*partials, 128),
        block_sizes=[4, 32, 128],
        **fold,
    )
    exact = _combined_moments(partials, _bounds(397, 128))
    _assert_partials("e790 level 1", combined, exact, everything)
