"""Native checks of loads ordered around opaque writes."""

from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfTileIR
from helion._testing.inline_asm_kernels import overwrite_after_reduction
from helion._testing.inline_asm_kernels import overwritten_reduction_input
from helion._testing.inline_asm_kernels import read_after_reduction_effect
from helion._testing.inline_asm_kernels import store_between_reductions


@onlyBackends(["triton", "cute"])
@skipIfRefEager("checks explicit reduction rolling and guarded bound dispatch")
@pytest.mark.parametrize("width", [128, 2**21])
def test_input_store_dispatches_when_destination_aliasing_changes(width):
    # The large extent exceeds Triton's persistent tensor limit. Constant
    # inputs keep the exact numerical check independent of summation order.
    x = (
        torch.arange(3 * width, dtype=torch.float32, device=DEVICE).reshape(3, width)
        if width == 128
        else torch.ones(1, width, dtype=torch.float32, device=DEVICE)
    )
    original = x.clone()
    target_storage = torch.full_like(x, -17)
    target = target_storage[:, 0]
    bound = store_between_reductions.bind((x, target, False))
    config = bound.config_spec.default_config()
    chunk = 32 if width == 128 else 4096
    config.config.update(reduction_loops=[chunk], reduction_fusion=True)
    # The storage guard keeps the rolled version for disjoint inputs and
    # dispatches aliasing calls to a persistent binding.
    normalized = bound._normalized_config_copy(config)
    assert normalized.config["reduction_loops"] == [chunk]
    compiled = bound.compile_config(normalized)
    bound.set_config(normalized)
    total = original.sum(-1)
    expected = total * 2 + original.amax(-1)
    torch.testing.assert_close(compiled(x, target, False), expected, rtol=0, atol=0)
    for alias in (False, True, False):
        if alias and width > 2**20:
            # An overlapping store cannot safely replay the original input.
            # Its persistent path is covered at the smaller extent.
            continue
        x.copy_(original)
        destination = x[:, 0] if alias else target
        actual = bound(x, destination, False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_x = original.clone()
        if alias:
            expected_x[:, 0] = total
        torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
        torch.testing.assert_close(destination, total, rtol=0, atol=0)
        torch.testing.assert_close(
            target_storage[:, 1:], torch.full_like(target_storage[:, 1:], -17)
        )
    assert bound.compile_config(normalized) is compiled


@onlyBackends(["triton", "cute"])
@pytest.mark.skipif(DEVICE.type != "cuda", reason="PTX needs CUDA")
@skipIfRocm("PTX inline assembly requires CUDA")
@skipIfTileIR("TileIR does not support inline_asm_elementwise")
@pytest.mark.parametrize("replay_load", [False, True])
def test_opaque_write_preserves_original_and_later_reads(replay_load):
    x = torch.arange(3 * 128, dtype=torch.int32, device=DEVICE).reshape(3, 128)
    original = x.clone()
    pointers = (
        x.data_ptr()
        + torch.arange(x.numel(), dtype=torch.int64, device=DEVICE).reshape(x.shape)
        * x.element_size()
    )
    kernel = overwritten_reduction_input if replay_load else read_after_reduction_effect
    # One warp owns each 32-element chunk. Replicated Triton layouts may run
    # opaque assembly in multiple physical lanes for one logical element.
    _, actual = code_and_output(
        kernel, (x, pointers), reduction_loops=[32], num_warps=1
    )
    expected = (
        original - original.sum(-1, keepdim=True)
        if replay_load
        else original[:, 0] + 1 + original.sum(-1)
    ).to(torch.int32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, original + 1, rtol=0, atol=0)


@onlyBackends(["triton", "cute"])
@pytest.mark.skipif(DEVICE.type != "cuda", reason="PTX needs CUDA")
@skipIfRocm("PTX inline assembly requires CUDA")
@skipIfTileIR("TileIR does not support inline_asm_elementwise")
def test_queued_input_reload_keeps_values_from_before_the_write():
    x = torch.arange(3 * 128, dtype=torch.int32, device=DEVICE).reshape(3, 128)
    original = x.clone()
    pointers = (
        x.data_ptr()
        + torch.arange(x.shape[0], dtype=torch.int64, device=DEVICE)
        * x.stride(0)
        * x.element_size()
    )
    _, actual = code_and_output(
        overwrite_after_reduction, (x, pointers), reduction_loops=[32], num_warps=1
    )
    total = original.sum(-1, keepdim=True)
    torch.testing.assert_close(
        actual, (original - total).to(torch.int32), rtol=0, atol=0
    )
    expected_x = original.clone()
    expected_x[:, 0] = (total[:, 0] + 1).to(torch.int32)
    torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
