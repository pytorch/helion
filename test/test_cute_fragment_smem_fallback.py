"""The computed-fragment owner declines a root its shared memory cannot hold."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.layer_norm import layer_norm_bwd
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._testing import skipUnlessBackends

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel

pytestmark = skipUnlessBackends("cute")


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Iterator[None]:
    # The CPU model has no device to query: pin the B200 CTA capacity.
    with (
        _mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_capacity_bytes",
            return_value=232448,
        ),
    ):
        yield


def _bind(rows: int, columns: int) -> BoundKernel:
    x = torch.empty((rows, columns), dtype=torch.bfloat16)
    args = (
        torch.empty((rows, columns), dtype=torch.bfloat16),
        x,
        torch.empty(rows, dtype=torch.float32),
        torch.empty(rows, dtype=torch.float32),
        torch.empty(columns, dtype=torch.bfloat16),
        True,
    )
    kernel = helion.kernel(
        layer_norm_bwd.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    return _cpu_bind(kernel, args)


def test_wide_rows_fall_back_to_the_scalar_lowering() -> None:
    # One fp32 row of 8192 features is 32 KB; the fragment program keeps
    # about eight of them live, which exceeds the 227 KB CTA capacity for
    # every row tile. The default config used to raise InvalidConfig here,
    # failing the autotuning reference baseline and the whole search.
    bound = _bind(8192, 8192)
    code = bound.to_code(bound.config_spec.default_config())
    assert "fragment_smem" not in code
    assert "_cute_grouped_reduce_shared_two_stage" in code
    assert "_launcher(_helion_layer_norm_bwd" in code


def test_narrow_rows_keep_the_fragment_owner() -> None:
    bound = _bind(64, 1024)
    code = bound.to_code(bound.config_spec.default_config())
    assert "fragment_smem" in code


def test_hard_fragment_requirements_still_reject_the_config() -> None:
    # A knob that only the fragment owner implements cannot fall back.
    bound = _bind(8192, 8192)
    spec = bound.config_spec
    if not spec.cute_fragment_thread_root_ids:
        pytest.skip("fragment thread knob unavailable for this root")
    config = spec.default_config()
    config.config["cute_fragment_threads"] = 256
    with pytest.raises(helion.exc.InvalidConfig, match="shared bytes"):
        bound.to_code(config)
