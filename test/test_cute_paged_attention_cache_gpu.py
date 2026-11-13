from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_optional_lse import check_optional_lse_replay
from helion._testing.cute_optional_lse import initialize_paged_attention_inputs
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_optional_lse import optional_lse_fixture


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "form,query_len,mode,policies",
    [
        ("reduced", 4, "attention", ("", "", "", "")),
        ("mapped", 1, "attention", ("", "l2_last", "streaming", "")),
        ("mapped", 4, "attention_scoped", ("l2_last", "streaming", "", "l2_last")),
        ("mapped", 5, "attention_paired", ("streaming", "", "l2_last", "streaming")),
        ("mapped", 5, "attention_paired", ("streaming",) * 4),
        ("mapped", 5, "attention_paired", ("l2_last",) * 4),
    ],
)
def test_paged_tma_cache_policies_and_changed_replay(form, query_len, mode, policies):
    """Six compiles cover all bodies, both priorities and mixed raw/scale hints."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paged attention requires SM100 tensor memory")
    kernel, args = optional_lse_fixture(form, True, DEVICE, query_len=query_len)
    initialize_paged_attention_inputs(args)
    bound = helion.kernel(
        kernel.fn, backend="cute", static_shapes=True, fast_math=True
    ).bind(args)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    slots = facts.attention.tma_load_slots
    assert len(slots) == 4
    selected = [""] * bound.config_spec.load_eviction_policies.length
    for slot, policy in zip(slots, policies, strict=True):
        assert slot is not None
        selected[slot] = policy
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag_pass_warps=mode,
        cute_min_blocks_per_mp=2 if mode == "attention_scoped" else 1,
        load_eviction_policies=selected,
    )
    if mode == "attention_paired":
        config.config["cute_mma_dag_v_lag"] = 2
    source = bound.to_code(config)
    assert ("tma_load_hints=" in source) == any(policies)
    check_optional_lse_replay(bound, config, args, form, True)
