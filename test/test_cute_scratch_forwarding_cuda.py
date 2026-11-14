"""GPU tests: a forwarded scratch store reads back bitwise what the reload
read, and a kernel without its dead scratch stores still computes its result.

The opaque move keeps the optimizer from merging the second sweep's
products with the first sweep's (which would change FMA contraction), so
with ``cute_scratch_forwarding`` the kernel matches, bit for bit, the same
kernel with the knob off, which stores and reloads its workspace.  The knob
also drops the stores no load reads any more (``drop_scratch_stores``).  The
codegen sides are ``test_cute_scratch_forwarding.py`` and
``test_cute_scratch_stores.py``.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing.scratch_kernels import RELU_SINK_CONFIG
from helion._testing.scratch_kernels import ROW_SWEEPS_CONFIG
from helion._testing.scratch_kernels import relu_spatial_mean
from helion._testing.scratch_kernels import row_softmax_bwd
from helion._testing.scratch_kernels import unread_workspace


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float64], ids=str
)
def test_a_forwarded_workspace_is_bitwise_the_reloaded_one(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    logits = torch.randn(64, 1024, device=DEVICE, dtype=torch.bfloat16) * 4
    grad = torch.randn(64, 1024, device=DEVICE, dtype=torch.bfloat16)
    row_max = logits.float().amax(-1)
    row_sum = torch.exp(logits.float() - row_max[:, None]).sum(-1)
    args = (grad, logits, row_max, row_sum, torch.empty(1, device=DEVICE, dtype=dtype))
    kernels = [
        helion.kernel(
            row_softmax_bwd,
            backend="cute",
            static_shapes=True,
            config=helion.Config(
                **ROW_SWEEPS_CONFIG,
                cute_reduction_reloads=["register"],
                cute_scratch_forwarding=forwarding,
            ),
        )
        for forwarding in (True, False)
    ]
    forwarded, reloaded = (kernel(*args) for kernel in kernels)
    # On the GPU's target as on the CPU's, the second sweep reads the
    # workspace back through the move instead of reloading it.
    code = kernels[0].bind(args).to_code()
    assert "mov.b" in code
    assert not [
        line
        for line in code.splitlines()
        if "probs_ws.iterator" in line and ".load()" in line
    ]
    assert torch.equal(forwarded.view(torch.int16), reloaded.view(torch.int16))
    probs = torch.exp(logits.float() - row_max[:, None]) / row_sum[:, None]
    total = (grad.float() * probs).sum(-1, keepdim=True)
    reference = probs.to(dtype).float() * (grad.float() - total)
    torch.testing.assert_close(forwarded.float(), reference, rtol=2e-2, atol=1e-2)


def test_a_kernel_without_its_dead_scratch_stores_matches_eager() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 1024, device=DEVICE, dtype=torch.bfloat16)
    kernel = helion.kernel(
        unread_workspace,
        backend="cute",
        static_shapes=True,
        config=helion.Config(block_sizes=[1], cute_scratch_forwarding=True),
    )
    out = kernel(x)
    torch.testing.assert_close(out, (x.float() * 2 + 1).to(x.dtype), rtol=0, atol=0)


def test_a_dropped_packet_flush_leaves_the_pool_bitwise() -> None:
    # Without ``act``'s flush, packet list and appends, the pool is bitwise
    # the one the kernel that stores ``act`` computes.
    torch.manual_seed(0)
    x = torch.randn(8, 784, 120, device=DEVICE, dtype=torch.bfloat16)
    shift = torch.randn(120, device=DEVICE)
    dropped, stored = (
        helion.kernel(
            relu_spatial_mean,
            backend="cute",
            static_shapes=True,
            config=helion.Config(
                **RELU_SINK_CONFIG, cute_scratch_forwarding=forwarding
            ),
        )(x, shift)
        for forwarding in (True, False)
    )
    assert torch.equal(dropped, stored)
    reference = torch.relu(x.float() - shift).mean(dim=1).to(x.dtype)
    torch.testing.assert_close(dropped, reference, rtol=1e-2, atol=1e-2)
