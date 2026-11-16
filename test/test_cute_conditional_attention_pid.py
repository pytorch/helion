"""A persistent CTA must never reacquire relinquished tensor-memory permits."""

from __future__ import annotations

import pytest

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_conditional_attention import conditional_attention_args
from helion._testing.cute_conditional_attention import (
    conditional_online_attention_prefix,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "pid", ["flat", "persistent_blocked", "persistent_interleaved"]
)
def test_conditional_attention_schedule_requires_flat_pid(pid):
    # Six tiles on two modelled SMs makes persistent CTAs execute multiple tiles.
    with cpu_render_target(fake_device_sm_count=2):
        bound = cpu_bind(
            conditional_online_attention_prefix, conditional_attention_args()
        )
        config = bound.config_spec.default_config()
        config.config["pid_type"] = pid
        if pid != "flat":
            with pytest.raises(helion.exc.InvalidConfig, match="shared bytes"):
                bound.to_code(config)
            return
        source = bound.to_code(config)
    assert "_conditional_attention_begin(" in source


def test_conditional_attention_search_and_seeds_use_flat_pid():
    with cpu_render_target(fake_device_sm_count=2):
        bound = cpu_bind(
            conditional_online_attention_prefix, conditional_attention_args()
        )
        generation = bound.config_spec.create_config_generation()
        # Fragment sampling omits the PID field, so implicit configs use flat
        # scheduling. Explicit persistent requests are checked above.
        assert "pid_type" not in generation._key_to_flat_indices
        configs = [
            bound.config_spec.default_config(),
            *bound.config_spec.autotune_seed_configs(),
        ]
        configs.extend(generation.random_config() for _ in range(8))
        assert all(config.pid_type == "flat" for config in configs)
