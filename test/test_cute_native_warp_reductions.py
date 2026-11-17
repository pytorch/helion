"""Native warp reductions for a row's direct max, min and any.

Under ``cute_warp_fp32_redux`` the grouped reduce helpers take the native
fp32 extrema (``redux.sync``); the direct warp reductions a row emits now
take them too, and a 32-bit integer OR over a whole warp (``any``) takes the
native integer reduction.  The rewrite runs last, so the passes before it see
the shuffle reductions they recognize: the exact division of a softmax keeps
its hoisted form.  The kernels render on CPU; the ``cuda`` test checks the
native forms give the same bits as the shuffles.  The knob is searchable
where a kernel has a reduction it takes natively (``cute_native_redux_sites``),
at every width.
"""

from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING

import pytest
import torch

from test.test_cute_persistent_chunked_rows import _row_flags
from test.test_cute_persistent_chunked_rows import _row_floor
from test.test_cute_persistent_chunked_rows import _row_softmax

import helion
from helion._compiler.cute.autotune.heuristics.cute import (
    CuteMaskedRolledClusterHeuristic,
)
from helion._compiler.cute.native_warp_reduction import enable_fp32_warp_redux
from helion._compiler.cute.native_warp_reduction import select_native_warp_reductions
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import Kernel

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _guarded_softmax(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.size(0)
    out = torch.empty_like(x)
    low = torch.empty([rows], dtype=torch.float32, device=x.device)
    for row in hl.tile(rows):
        data = x[row, :].to(torch.float32)
        infinite = torch.abs(data) == float("inf")
        finite = torch.logical_not(torch.any(infinite, dim=-1, keepdim=True))
        shifted = torch.where(
            finite, data - torch.amax(data, dim=-1, keepdim=True), 0.0
        )
        e = torch.exp(shifted)
        out[row, :] = (e / e.sum(-1, keepdim=True)).to(x.dtype)
        low[row] = torch.amin(data, dim=-1)
    return out, low


def _config(redux: bool) -> helion.Config:
    return helion.Config(
        block_sizes=[1],
        num_threads=[0, 32],
        reduction_loops=[None],
        cute_vector_widths=[8, 1],
        cute_reduction_reloads=["register_prefetch"],
        cute_warp_fp32_redux=redux,
    )


@functools.cache
def _code(redux: bool) -> str:
    x = torch.empty((8, 512), dtype=torch.bfloat16)
    with mock_cuda_unavailable(), forbid_native_compile():
        return cpu_bind(_guarded_softmax, (x,)).to_code(_config(redux))


def test_direct_reductions_take_the_native_helpers() -> None:
    code = _code(True)
    for helper, count in (
        ("_cute_warp_reduce_max_redux(", 1),
        ("_cute_warp_reduce_min_redux(", 1),
        ("_cute_warp_reduce_or_redux(", 1),
    ):
        assert code.count(helper) == count, code
    assert "cute.arch.warp_reduction_max(" not in code, code
    # The sum keeps its shuffle reduction.
    assert code.count("cute.arch.warp_reduction_sum(") == 1, code


def test_grouped_reduces_keep_their_own_fp32_redux_dispatch() -> None:
    """The direct rewrite leaves the grouped helpers to ``fp32_redux``, whose
    warp form picks between ``redux.sync`` per sibling and the partial
    butterfly (``_cute_grouped_reduce_warp``)."""
    body = ast.parse(
        "a = _cute_grouped_reduce_warp(v, 'max', cutlass.Float32(float('-inf')), "
        "lane, pre=4, group_span=32)\n"
        "b = cute.arch.warp_reduction_max(v, op=_cute_nan_max, threads_in_group=32)"
    ).body
    grouped, direct = select_native_warp_reductions(enable_fp32_warp_redux(body))
    assert ast.unparse(grouped) == (
        "a = _cute_grouped_reduce_warp(v, 'max', cutlass.Float32(float('-inf')), "
        "lane, pre=4, group_span=32, fp32_redux=True)"
    )
    assert ast.unparse(direct) == (
        "b = _cute_warp_reduce_max_redux(v, threads_in_group=32)"
    )


def test_knob_off_keeps_the_shuffles() -> None:
    code = _code(False)
    assert "_redux(" not in code, code
    assert code.count("cute.arch.warp_reduction_max(") == 1, code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_rms(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        data = x[row, :].to(torch.float32)
        scale = torch.rsqrt(torch.mean(data * data, dim=-1, keepdim=True) + 1e-6)
        out[row, :] = (data * scale).to(x.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_floor_unwidened(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        data = x[row, :]
        out[row, :] = data - torch.amax(data, dim=-1, keepdim=True)
    return out


def _redux_choices(
    kernel: Kernel, x: torch.Tensor, capability: tuple[int, int] = (10, 0)
) -> tuple[ConfigSpec, object]:
    """``kernel``'s config spec and the values the search samples for
    ``cute_warp_fp32_redux``."""
    with mock_cuda_unavailable(), forbid_native_compile():
        spec = cpu_bind(kernel, (x,), capability=capability).config_spec
    fragment = spec._flat_fields()["cute_warp_fp32_redux"]
    return spec, fragment.search_choices  # pyrefly: ignore[missing-attribute]


@pytest.mark.parametrize(
    ("kernel", "capability", "native"),
    [
        (_row_softmax, (10, 0), True),
        (_row_floor, (10, 0), True),
        # fp32 ``redux.sync`` extrema are the SM100 family's alone; an
        # integer OR is native on every target.
        (_row_softmax, (9, 0), False),
        (_row_floor, (9, 0), False),
        (_row_flags, (9, 0), True),
        (_row_rms, (10, 0), False),
    ],
)
def test_native_redux_opens_on_the_kernels_reductions(
    kernel: Kernel, capability: tuple[int, int], native: bool
) -> None:
    """A max, a min or an ``any`` the target reduces natively opens
    ``cute_warp_fp32_redux`` (``cute_native_redux_sites``); sums do not."""
    spec, choices = _redux_choices(
        kernel, torch.empty((64, 1000), dtype=torch.float32), capability
    )
    assert spec.cute_native_redux_sites is native
    assert choices == ((False, True) if native else (False,))


@pytest.mark.parametrize(
    ("dtype", "native"),
    [
        # bf16 and f16 extrema reduce as fp32 values; integer and fp64 ones
        # keep the shuffle reduction, which the knob does not change.
        (torch.bfloat16, True),
        (torch.float16, True),
        (torch.int32, False),
        (torch.float64, False),
    ],
)
def test_native_redux_opens_on_fp32_reduced_extrema_only(
    dtype: torch.dtype, native: bool
) -> None:
    spec, choices = _redux_choices(
        _row_floor_unwidened, torch.empty((64, 512), dtype=dtype)
    )
    assert spec.cute_native_redux_sites is native
    assert choices == ((False, True) if native else (False,))


@pytest.mark.parametrize(("kernel", "redux"), [(_row_softmax, True), (_row_rms, False)])
def test_native_redux_searchability_does_not_depend_on_the_width(
    kernel: Kernel, redux: bool
) -> None:
    """``autotune_multi`` shares configs across the shapes of one structural
    fingerprint: a row's max opens ``cute_warp_fp32_redux`` at every width
    (its seed is the one-warp persistent row of 512 bf16 halves alone, not
    of 1024 or of an fp32 row of 1000), and a row with sums only never, not
    even at 16384 elements, where the masked rolled cluster seeds only a
    kernel the knob changes."""
    fingerprints = set()
    for dtype, width in (
        (torch.bfloat16, 512),
        (torch.bfloat16, 1024),
        (torch.float32, 1000),
        (torch.bfloat16, 16384),
        (torch.float32, 16384),
    ):
        spec, choices = _redux_choices(kernel, torch.empty((64, width), dtype=dtype))
        fingerprints.add(spec.structural_fingerprint())
        assert choices == ((False, True) if redux else (False,))
    assert len(fingerprints) == 1


@pytest.mark.parametrize(
    ("kernel", "copies"), [(_row_flags, False), (_row_floor, True)]
)
def test_cluster_seeds_take_the_knob_for_native_extrema_only(
    kernel: Kernel, copies: bool
) -> None:
    """The masked rolled cluster's grouped reduce helpers take a max or a min
    natively, not an ``any``: the cluster seeds of a row of 16384 bf16
    elements get copies with ``cute_warp_fp32_redux`` for its ``amin`` on
    SM100, none for its ``any``, where they would render their parents."""
    x = torch.empty((64, 16384), dtype=torch.bfloat16)
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(kernel, (x,), capability=(10, 0))
        with bound.env:
            seeds = CuteMaskedRolledClusterHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )
    assert seeds
    assert any(seed.config.get("cute_warp_fp32_redux") for seed in seeds) is copies


@pytest.mark.cuda
@skipIfNotCUDA()
def test_cuda_native_reductions_match_the_shuffles() -> None:
    x = torch.randn((64, 512), device=DEVICE).to(torch.bfloat16)
    x[3, 17] = float("inf")
    x[5, 100] = float("-inf")
    x[7] = -0.0
    # The rows where a combine order could show in ``low`` (the row's amin):
    # a minimum of both +0.0 and -0.0 beside positive values, and a NaN,
    # which the native extrema propagate like the shuffle's combine.
    x[9] = x[9].abs() + 1
    x[9, ::2] = 0.0
    x[9, 1::4] = -0.0
    x[11, 200] = float("nan")
    bound = _guarded_softmax._bind_isolated((x,))
    shuffle = bound.compile_config(_config(False))(x)
    native = bound.compile_config(_config(True))(x)
    assert torch.equal(native[0].view(torch.int16), shuffle[0].view(torch.int16))
    assert torch.equal(native[1].view(torch.int32), shuffle[1].view(torch.int32))
    assert native[1][11].isnan()
