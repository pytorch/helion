"""A bit-for-bit comparison must fail for an output element no store reached."""

from __future__ import annotations

import pytest
import torch

from helion._testing.poisoned_outputs import poison_
from helion._testing.poisoned_outputs import poisoned_empty
from helion._testing.poisoned_outputs import poisoned_launch
from helion._testing.poisoned_outputs import written_bits

pytestmark = pytest.mark.cpu_only


def _kernel(
    stored: int, output: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stores the first ``stored`` of four items of a new or a given output."""
    if output is None:
        output = torch.empty(4, 3, dtype=torch.bfloat16)
    lse = torch.empty(4, dtype=torch.float32)
    output[:stored] = 1.5
    lse[:stored] = float("-inf")
    return output, lse


def test_a_launch_that_stores_everything_returns_its_bits():
    empty = torch.empty
    output, lse = poisoned_launch(lambda: _kernel(4))
    assert torch.empty is empty
    assert (output.dtype, lse.dtype) == (torch.int16, torch.int32)
    assert torch.equal(output.view(torch.bfloat16), torch.full((4, 3), 1.5))
    assert torch.isneginf(lse.view(torch.float32)).all()


def test_an_item_no_store_reached_fails_whatever_the_memory_held():
    with pytest.raises(AssertionError, match="no store reached 1 of 4 .* first .3."):
        poisoned_launch(lambda: _kernel(3))
    # A preallocated output that already holds the reference's values, as the
    # block of a freed result does.
    kept = torch.full((4, 3), 1.5, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="no store reached 1 of 4"):
        poisoned_launch(lambda: _kernel(3, kept), [kept])
    assert len(poisoned_launch(lambda: _kernel(4, kept), [kept])) == 2


def test_an_output_that_did_not_start_poisoned_is_refused():
    kept = torch.full((4, 3), 1.5, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="did not start poisoned"):
        poisoned_launch(lambda: _kernel(4, kept))


@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
def test_an_output_of_a_dtype_without_a_payload_is_named(dtype):
    """Only BF16 and FP32 outputs have a payload. ``torch.empty`` leaves a
    tensor of another dtype alone, so a launch that returns one is told the
    dtype, not to allocate with ``torch.empty`` as it did; a preallocated
    one is refused the same way before the launch."""
    named = rf"dtype {dtype}: poisoned outputs are torch\.bfloat16 and torch\.float32"
    with pytest.raises(TypeError, match=named):
        poisoned_launch(lambda: (torch.empty(4, dtype=dtype),))
    kept = torch.zeros(4, dtype=dtype)
    launched = []
    with pytest.raises(TypeError, match=named):
        poisoned_launch(lambda: launched.append(kept), [kept])
    assert not launched
    for helper in (poison_, written_bits):
        with pytest.raises(TypeError, match=named):
            helper(kept)


def test_replayed_outputs_are_poisoned_by_hand():
    """A captured call's outputs are allocated once; each replay's stores are
    checked against a fill made ahead of it."""
    with poisoned_empty() as fresh:
        output, lse = _kernel(4)
        # A dtype without a payload is left alone.
        torch.empty(4, dtype=torch.int32)
    assert len(fresh) == 2 and fresh[0] is output and fresh[1] is lse
    poison_(output, lse)
    lse[:] = 0.0
    assert not written_bits(lse).any()
    with pytest.raises(AssertionError, match="no store reached 4 of 4"):
        written_bits(output)
