"""CPU checks for the authored recurrence bounds."""

from __future__ import annotations

import inspect

from benchmarks.cute.kda_prefill_kernels import BT
import pytest

from helion._compiler.cute.chunk_recurrence_dv4_sm100 import emit_bt16_recurrence
from helion._testing.cute_codegen import forbid_cuda_work

pytestmark = pytest.mark.cpu_only


@forbid_cuda_work()
def test_dv4_uses_semantic_length_when_chunk_metadata_disagrees() -> None:
    # Adversarial metadata: the chunk offsets claim one chunk for 17 tokens.
    # The Helion carrier and DV2 execute ceil(17 / 16) == 2 chunks, so DV4 must
    # derive the same bound from sequence lengths instead of trusting the delta.
    token_start, token_end = 0, 17
    chunk_start, chunk_end = 0, 1
    assert (token_end - token_start + BT - 1) // BT == 2
    assert chunk_end - chunk_start == 1

    # CuTe preprocessing replaces the function code and removes the decorator
    # from inspect.getsource after an earlier compile. Check the recurrence body
    # in either state rather than the decorator's presence.
    source = inspect.getsource(emit_bt16_recurrence)
    assert "num_chunks = cute.ceil_div(token_end - token_start, BT)" in source
    assert "gcu_chunks[seq + 1]" not in source
