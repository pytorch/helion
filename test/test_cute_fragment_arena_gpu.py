from __future__ import annotations

from unittest.mock import patch

import torch

from helion._testing import DEVICE
from helion._testing import skipUnlessCuteAvailable
from helion._testing.cute_fragment_arena import resident_while_recurrence
from helion._testing.cute_fragment_arena import sequential_votes


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_boolean_arena_preserves_masks_votes_and_changed_inputs():
    x = torch.zeros((3, 8, 8), dtype=torch.bfloat16, device=DEVICE)
    y = torch.zeros((3, 8, 8), dtype=torch.float32, device=DEVICE)
    bound = sequential_votes.bind((x, y))
    config = bound.config_spec.default_config()
    with patch(
        "helion._compiler.cute.target_facts.per_cta_smem_capacity_bytes",
        return_value=350,
    ):
        source = bound.to_code(config)
        assert "fragment_arena = fragment_smem.allocate_tensor(" in source
        assert "cute.recast_tensor(" in source
        compiled = bound.compile_config(config)
    for change in (False, True, False):
        x.fill_(-1)
        y.zero_()
        x[0, 3, 7] = 1
        y[1, 6, 2] = 1
        if change:
            x.neg_()
            y.neg_()
        # Three inversions force Boolean loop-carried storage, then invert once.
        expected_masks = torch.stack((x <= 0, y <= 0), dim=1)
        expected_votes = expected_masks.flatten(2).any(dim=2)
        before = x.clone(), y.clone()
        votes, masks = compiled(x, y)
        torch.testing.assert_close(votes, expected_votes, rtol=0, atol=0)
        torch.testing.assert_close(masks, expected_masks, rtol=0, atol=0)
        torch.testing.assert_close(x, before[0], rtol=0, atol=0)
        torch.testing.assert_close(y, before[1], rtol=0, atol=0)


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_boolean_arena_predicate_controls_zero_and_multiple_iterations():
    x_cpu = (torch.arange(51).reshape(3, 17) % 7 - 3).float()
    x = x_cpu.to(DEVICE)
    limits = torch.tensor([0, 1, 3], dtype=torch.int32, device=DEVICE)
    args = (x, limits, 0, 32)
    bound = resident_while_recurrence.bind(args)
    config = bound.config_spec.default_config()
    config.config.update(cute_fragment_bounded_gather=True, cute_fragment_threads=128)
    with patch(
        "helion._compiler.cute.target_facts.per_cta_smem_capacity_bytes",
        return_value=1220,
    ):
        source = bound.to_code(config)
        assert "fragment_arena = fragment_smem.allocate_tensor(" in source
        assert "while fragment_while_take" in source
        compiled = bound.compile_config(config)
    for counts in ([0, 1, 3], [3, 0, 1], [0, 1, 3]):
        limits.copy_(torch.tensor(counts, dtype=torch.int32, device=DEVICE))
        expected = []
        for row, count in enumerate(counts):
            left, right = x_cpu[row], x_cpu[row] + 1
            for _iteration in range(count):
                left, right = left + right, left - right
            expected.append(left + right)
        out, actual_counts = compiled(*args)
        torch.testing.assert_close(out.cpu(), torch.stack(expected), rtol=0, atol=0)
        torch.testing.assert_close(actual_counts, limits, rtol=0, atol=0)
        torch.testing.assert_close(x.cpu(), x_cpu, rtol=0, atol=0)
