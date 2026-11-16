"""Bounded full authored-source numerical gate; no performance measurements."""

from __future__ import annotations

import pytest
import torch

from helion._testing.cute_conditional_native import validate_conditional_source
from helion._testing.cute_conditional_phase_binding import phase_fixture
from helion._testing.cute_fragment_shared_replay import begin_record
from helion._testing.cute_fragment_shared_replay import require_sm100_native
from helion._testing.cute_fragment_shared_replay import run_authored_storage_replay
from helion._testing.cute_fragment_storage_oracle import assert_attention_result
from helion._testing.cute_fragment_storage_oracle import attention_cpu
from helion._testing.cute_fragment_storage_oracle import move_backings
from helion._testing.cute_paged_conditional_semantics import (
    written_conditional_attention,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.slow,
]


def test_conditional_attention_full_source_and_replay(tmp_path, record_property):
    output, record = begin_record("authored", tmp_path)
    record_property("raw_bits", str(output / "raw.pt"))
    run_authored_storage_replay(output, record, validate_conditional_source)


def test_conditional_attention_multiwave_flat_replay():
    device = require_sm100_native()
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    batch = sm_count // 2 + 1
    assert batch * 2 > sm_count
    values = list(phase_fixture(batch, buckets=2))
    lengths = values[6]
    assert isinstance(lengths, torch.Tensor)
    lengths.clamp_(max=256)
    values[10:] = [3, 252]
    original = tuple(values)
    changed = move_backings(original, torch.device("cpu"))
    query = changed[0]
    assert isinstance(query, torch.Tensor)
    query.mul_(-1)
    expected = [
        attention_cpu(move_backings(state, torch.device("cpu")))
        for state in (original, changed)
    ]
    args = move_backings(original, torch.device("cuda"))
    bound = written_conditional_attention.bind(args)
    config = bound.config_spec.default_config()
    config.config["pid_type"] = "flat"
    source = bound.to_code(config)
    assert "_conditional_attention_begin(" in source
    assert "for virtual_pid in range(" not in source
    compiled = bound.compile_config(config)
    actual = compiled(*args)
    torch.cuda.synchronize()
    assert_attention_result(tuple(value.cpu() for value in actual), expected[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*args)
    target_query = args[0]
    assert isinstance(target_query, torch.Tensor)
    for state, reference in ((changed, expected[1]), (original, expected[0])):
        source_query = state[0]
        assert isinstance(source_query, torch.Tensor)
        target_query.copy_(source_query)
        graph.replay()
        torch.cuda.synchronize()
        assert_attention_result(tuple(value.cpu() for value in captured), reference)
    graph.reset()
