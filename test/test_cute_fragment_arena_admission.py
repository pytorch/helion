"""Arena ownership and final scalar-pass compatibility."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_fragment_arena import explicit_pdl_votes
from helion._testing.cute_fragment_arena import resident_while_recurrence
from helion._testing.cute_fragment_arena import sequential_votes

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("mode", ["flat", "pdl", "explicit_pdl", "persistent"])
def test_required_arena_checks_dispatch_before_commitment(mode):
    with cpu_target():
        bound = cpu_bind(
            explicit_pdl_votes if mode == "explicit_pdl" else sequential_votes,
            (
                torch.ones(1, 8, 8, dtype=torch.bfloat16),
                torch.ones(1, 8, 8),
            ),
        )
        config = bound.config_spec.default_config()
        if mode == "pdl":
            config.config["cute_pdl"] = True
        elif mode == "persistent":
            config.config["pid_type"] = "persistent_blocked"
        with patch(
            "helion._compiler.cute.target_facts.per_cta_smem_capacity_bytes",
            return_value=350,
        ):
            if mode != "flat":
                with (
                    patch(
                        "helion._compiler.cute.computed_fragment.plan_shared_arena",
                        side_effect=AssertionError("ineligible arena was planned"),
                    ),
                    pytest.raises(
                        helion.exc.InvalidConfig,
                        match="computed fragments need .*shared bytes",
                    ),
                ):
                    bound.to_code(config)
                return
            code = bound.to_code(config)
    assert "fragment_arena = fragment_smem.allocate_tensor(cutlass.Uint8," in code
    assert "cute.recast_ptr(fragment_arena.iterator" in code
    assert "cute.recast_tensor(" in code


def test_resident_while_predicate_uses_boolean_arena_view():
    with cpu_target():
        bound = cpu_bind(
            resident_while_recurrence,
            (torch.ones(3, 17), torch.tensor([0, 1, 3], dtype=torch.int32), 0, 32),
        )
        config = bound.config_spec.default_config()
        config.config.update(
            cute_fragment_bounded_gather=True, cute_fragment_threads=128
        )
        with patch(
            "helion._compiler.cute.target_facts.per_cta_smem_capacity_bytes",
            return_value=1220,
        ):
            code = bound.to_code(config)
    assert "fragment_arena = fragment_smem.allocate_tensor(" in code
    assert "while fragment_while_take" in code
    assert "cute.recast_tensor(" in code
