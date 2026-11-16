"""A captured recurrence gets a deterministic executable ordered reference."""

from __future__ import annotations

import pytest
import torch

from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_ordered_reference import local_histogram_phases
from helion._testing.cute_ordered_reference import partitioned_dot_sum

pytestmark = pytest.mark.cpu_only


def test_captured_recurrence_reference_uses_ordered_launches():
    with cpu_target():
        bound = cpu_bind(
            partitioned_dot_sum,
            (
                torch.ones(7, 8, dtype=torch.bfloat16),
                torch.ones(8, 8, 11, dtype=torch.bfloat16),
            ),
        )
        spec = bound.config_spec
        assert spec.allowed_pid_types[0] == "flat"
        pid_field = spec._flat_fields()["pid_type"]
        assert pid_field.choices == spec.allowed_pid_types
        assert any(choice.startswith("persistent") for choice in pid_field.choices)
        assert spec.autotune_reference_config().pid_type == "flat"
        assert spec.default_config().pid_type == "flat"
        # The conservative default stays executable without a pinned config.
        reference = bound.to_code(spec.autotune_reference_config())
        assert reference.count("PyCodeCache.load(") == 2
        assert bound.to_code(spec.default_config()).count("PyCodeCache.load(") == 2
        # Later admission passes can leave only flat. Do not expose an
        # invariant coordinate as if persistent scheduling were searchable.
        spec.allowed_pid_types = ("flat",)
        assert "pid_type" not in spec._flat_fields()


def test_local_atomic_reference_uses_ordered_launches():
    with cpu_target():
        bound = cpu_bind(
            local_histogram_phases, (torch.ones(3, 11, dtype=torch.int32),)
        )
        spec = bound.config_spec
        assert spec.allowed_pid_types[0] == "flat"
        assert any(pid.startswith("persistent") for pid in spec.allowed_pid_types)
        source = bound.to_code(spec.autotune_reference_config())
    assert source.count("PyCodeCache.load(") == 2
