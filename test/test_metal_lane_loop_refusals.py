"""Refusals from the lane-loop passes CuTe and Metal share
(``cute/lane_loops``) name the backend compiling the kernel, on CPU without a
Metal device."""

from __future__ import annotations

from pathlib import Path
import re

import pytest
import torch

import helion
from helion._compiler.cute.lane_loops.unsupported import lane_loop_unsupported
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import forbid_cuda_work
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


@helion.kernel(backend="metal", static_shapes=True)
def _copy(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile]
    return out


@forbid_cuda_work()
def test_lane_loop_refusal_names_the_compiling_backend():
    bound = _copy.bind((torch.zeros(16),))
    with bound.env:
        error = lane_loop_unsupported("a lane-loop detail")
    assert str(error) == "Backend 'metal' does not support: a lane-loop detail"
    # Outside a compile only CuTe's unit tests run the passes.
    error = lane_loop_unsupported("a lane-loop detail")
    assert str(error) == "Backend 'cute' does not support: a lane-loop detail"


def test_lane_loops_raise_no_hardcoded_backend_refusal():
    """Every refusal in the lane-loop passes goes through
    ``lane_loop_unsupported``, so none names CuTe for a Metal kernel."""
    package = Path(lane_loop_unsupported.__code__.co_filename).parent
    hardcoded = {
        path.name
        for path in package.glob("*.py")
        if re.search(r"BackendUnsupported\(\s*\"cute\"", path.read_text())
        and path.name != "unsupported.py"
    }
    assert not hardcoded, f"hardcoded 'cute' refusals in {sorted(hardcoded)}"
