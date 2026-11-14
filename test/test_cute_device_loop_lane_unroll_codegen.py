"""The loads-first unroll of a device loop's lane loop, checked on CPU.

A lane loop of an ``hl.tile`` axis inside the kernel, here the K groups of
the nvfp4 GEMV, takes ``cute_lane_unroll`` as a grid axis's lane loop does.
These checks only normalize configs and inspect the generated source;
``test_cute_device_loop_lane_unroll.py`` runs the unrolled kernels on a GPU.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest

import helion
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.nvfp4_gemv_args import nvfp4_gemv_args

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

_examples_nvfp4_gemv = import_example("nvfp4_gemv")
_nvfp4_gemv_fp4in_body = _examples_nvfp4_gemv._nvfp4_gemv_fp4in_body

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

# 64 rows of 2048 packed bytes: 256 groups of eight.
_ROWS, _GROUPS = 64, 256


def _config(threads: int, unroll: int) -> helion.Config:
    return helion.Config(
        block_sizes=[1, _GROUPS],
        num_threads=[1, threads],
        cute_vector_widths=[1, 1],
        cute_lane_layouts=["blocked", "strided"],
        cute_lane_unroll=unroll,
    )


@functools.cache
def _bound() -> BoundKernel:
    kernel = helion.kernel(
        _nvfp4_gemv_fp4in_body,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    args, _ = nvfp4_gemv_args(_ROWS, _GROUPS * 8, random=False)
    with mock_cuda_unavailable():
        return cpu_bind(kernel, args)


def test_device_loop_lane_loop_keeps_the_unroll_knob() -> None:
    """A lane loop of an ``hl.tile`` axis inside the kernel (not a grid axis)
    forms a lane loop too, so ``cute_lane_unroll`` survives normalization."""
    spec = _bound().config_spec
    assert spec.normalized_config(_config(32, 8)).config["cute_lane_unroll"] == 8
    # No lane loop at all (one element per thread): the knob is inert.
    assert spec.normalized_config(_config(256, 8)).config["cute_lane_unroll"] == 1


def test_unrolled_lane_loop_is_off_without_the_knob() -> None:
    with mock_cuda_unavailable():
        source = _bound().to_code(_config(32, 1))
    assert "for lane_1 in range(8):" in source
    assert "_lane_unroll_loads_" not in source
