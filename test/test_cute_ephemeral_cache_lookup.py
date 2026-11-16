from __future__ import annotations

import importlib
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing import default_cute_mma_support
from helion._testing.cute_codegen import cpu_bind

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("other_entry", [False, True])
def test_uncached_grouped_aot_config_does_not_inherit_conflicting_defaults(other_entry):
    module = importlib.import_module(
        "pretuned_kernels.grouped_gemm_deepgemm.grouped_gemm_deepgemm"
    )
    with cpu_render_target(mma_support=default_cute_mma_support()):
        bound = cpu_bind(
            module.grouped_gemm_deepgemm,
            (
                torch.empty(672, 512, dtype=torch.bfloat16),
                torch.empty(2, 128, 512, dtype=torch.bfloat16),
                torch.tensor([[0, 0, 193, 224], [1, 224, 257, 448]], dtype=torch.int32),
            ),
        )
        bound.env.config_spec.tcgen05.search_enabled = True
        selected = module._worklist_config(7)
        original = selected.to_json()
        normalized = bound._normalized_config_copy(selected)
        assert normalized.config["tcgen05_persistence_model"] == "static_persistent"
        # The generic default belongs to another schedule. Its inferred fields
        # cannot be copied into this valid saved AOT config during cache lookup.
        defaults = bound.config_spec.default_config()
        assert defaults.config["tcgen05_persistence_model"] == "non_persistent"
        with pytest.raises(helion.exc.InvalidConfig, match="contradicts pid_type"):
            bound._normalized_config_copy(defaults.config | selected.config)
        if other_entry:
            bound._compile_cache[defaults] = Mock()
        before = dict(bound._compile_cache)
        with patch.object(bound.env.backend, "_compiled_kernels") as kernels:
            bound.env.backend.finalize_ephemeral_cache(bound, selected)
        kernels.assert_not_called()
        assert bound._compile_cache == before
        assert selected.to_json() == original
        # An explicit contradiction remains an error in the actual config.
        invalid = helion.Config.from_dict(
            selected.config | {"tcgen05_persistence_model": "non_persistent"}
        )
        with pytest.raises(helion.exc.InvalidConfig, match="contradicts pid_type"):
            bound.env.backend.finalize_ephemeral_cache(bound, invalid)
