"""Generated source of the CuTe attention kernels with two head-dim reductions.

The CPU twin of ``test_cute_single_thread_reduction.py``, which runs the same
kernels (``helion._testing.cute_attention_kernels``) on the GPU: the launch
of their two head-dim reductions and the per-thread K fold of a second
matmul, rendered without a GPU.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import helion
from helion._testing import skipUnlessBackends
from helion._testing.cute_attention_kernels import attention_inputs
from helion._testing.cute_attention_kernels import attention_mla
from helion._testing.cute_attention_kernels import attention_rope
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

_attention_mla = helion.kernel(attention_mla, backend="cute", static_shapes=True)
_attention_rope = helion.kernel(attention_rope, backend="cute", static_shapes=True)


def _cpu_render(
    kernel: helion.Kernel, args: tuple[object, ...], **overrides: object
) -> str:
    """Render a kernel without a GPU, from the default config plus ``overrides``."""
    with (
        mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU test")),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        bound = cpu_bind(kernel, args)
        settings = dict(bound.config_spec.default_config().config)
        settings.update(overrides)
        return bound.to_code(helion.Config.from_dict(settings))


def test_attention_head_dim_reductions_own_two_axes() -> None:
    """The q/k head dim (192, 256 threads) and the value head dim (128, 4
    threads once the first reduction took the budget) own axes 0 and 1, the
    causal mask allocates no reduction dim, so the launch is ``(256, 4, 1)``
    and no coordinate reads a fourth axis.  ``torch.bmm(..., torch.float32)``
    folds its transposed operand like ``bmm.default`` (no shared-memory
    shuffle)."""
    q, k, v = attention_inputs("cpu", 192, 192, 128)
    code = _cpu_render(_attention_mla, (q, k, v), block_sizes=[1, 128, 128])
    assert "block=(256, 4, 1)" in code
    assert "thread_idx()[3]" not in code
    assert "permute_smem" not in code


def test_second_matmul_contraction_folds_per_thread_in_the_key_lane_loop() -> None:
    """A RoPE side matmul adds a reduction dim (64) beside the head dim's;
    codegen reaches the lane-split scheduler and the side matmul's K axis
    sits on the key lane loop: the per-thread K fold (``fold_dot``)
    accumulates it in registers instead of declining the staged lane
    schedule."""
    q, k, v, qr, kr = attention_inputs("cpu", 128, 128, 128, 64, 64)
    code = _cpu_render(_attention_rope, (q, k, v, qr, kr), block_sizes=[1, 128, 128])
    assert "_cute_fold_dot(" in code
    assert "staged lane schedule" not in code
