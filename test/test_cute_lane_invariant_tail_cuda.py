"""GPU numerics for the lane-invariant tail of a split lane loop.

The GPU-free companion (``test_cute_lane_invariant_tail.py``) checks where the
tail statements are emitted; these kernels run them.  A row's zeroing store
after the copy of the row used to be overwritten by the consume pass, so
``out[:, 0]`` held ``x[:, 0]`` instead of zero.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from test.test_cute_lane_invariant_tail import _CONFIG
from test.test_cute_lane_invariant_tail import _CONFIG_FEW_THREADS
from test.test_cute_lane_invariant_tail import _copy_then_flag
from test.test_cute_lane_invariant_tail import _copy_then_sum_into_the_row
from test.test_cute_lane_invariant_tail import _copy_then_zero
from test.test_cute_lane_invariant_tail import _copy_zero_copy_again
from test.test_cute_lane_invariant_tail import _softmax_then_zero
from test.test_cute_lane_invariant_tail import _softmax_zero_softmax_again

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def _run(kernel: object, x: torch.Tensor, **config: object) -> Any:
    bound = kernel.bind((x,))  # pyrefly: ignore [missing-attribute]
    return bound.compile_config(helion.Config.from_dict(config))(x)


def _rows() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn((8, 256), device=DEVICE)


@pytest.mark.parametrize(
    "config", [_CONFIG, _CONFIG_FEW_THREADS], ids=["threads32", "threads8"]
)
def test_zero_after_the_copy_matches_reference(config: dict[str, object]) -> None:
    x = _rows()
    out, sums = _run(_copy_then_zero, x, **config)
    expected = x.clone()
    expected[:, 0] = 0.0
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(sums, x.sum(dim=1))


@pytest.mark.parametrize(
    "config", [_CONFIG, _CONFIG_FEW_THREADS], ids=["threads32", "threads8"]
)
def test_zero_after_the_softmax_matches_reference(config: dict[str, object]) -> None:
    x = _rows()
    out = _run(_softmax_then_zero, x, **config)
    expected = torch.softmax(x, dim=1)
    expected[:, 0] = 0.0
    torch.testing.assert_close(out, expected)


def test_reduced_value_stored_into_the_copied_row_matches_reference() -> None:
    x = _rows()
    out = _run(_copy_then_sum_into_the_row, x, **_CONFIG)
    expected = x.clone()
    expected[:, 0] = x.sum(dim=1)
    torch.testing.assert_close(out, expected)


def test_store_into_another_tensor_matches_reference() -> None:
    x = _rows()
    out, out2, flags = _run(_copy_then_flag, x, **_CONFIG)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    torch.testing.assert_close(out2, x * 2, rtol=0, atol=0)
    torch.testing.assert_close(flags, x.sum(dim=1))


@pytest.mark.parametrize(
    "kernel",
    [_copy_zero_copy_again, _softmax_zero_softmax_again],
    ids=["one_pass", "dependent"],
)
def test_store_between_two_per_lane_stores_of_its_tensor_rejects_the_config(
    kernel: object,
) -> None:
    x = _rows()
    with pytest.raises(exc.BackendUnsupported, match="a body a later pass rewrites"):
        _run(kernel, x, **_CONFIG)
