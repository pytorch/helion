"""CPU regression for paired-pipeline search config canonicalization."""

from __future__ import annotations

import pytest
import torch

from test.cute_population_contracts import with_flat_min_blocks_default
from test.test_cute_pipeline_smem import _cpu_b200 as _cpu_b200

import helion
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _direct_rhs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile((m, n)):
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[row, inner], b[inner, column])
        out[row, column] = acc.to(a.dtype)
    return out


@pytest.mark.parametrize(
    "shape,controls_active",
    [((256, 64, 192), False), ((512, 1024, 512), True)],
)
def test_paired_seed_roundtrip_and_strict_transfer(
    shape: tuple[int, int, int], controls_active: bool
) -> None:
    initialized = torch.cuda.is_initialized()
    m, k, n = shape
    args = (
        torch.empty((m, k), dtype=torch.float16),
        torch.empty((k, n), dtype=torch.float16),
    )
    bound = helion.kernel(backend="cute")(_direct_rhs.fn)._bind_isolated(args)
    spec = bound.config_spec
    controls = (
        TCGEN05_TVM_FFI_LAUNCH_CONFIG_KEY,
        TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY,
    )
    with bound.env:
        assert all((key in spec._flat_fields()) is controls_active for key in controls)
        seeds = [
            seed
            for seed in spec.compiler_seed_configs
            if seed.get("tcgen05_cta_group") == "two"
        ]
        assert seeds
        generation = spec.create_config_generation()
        for seed in seeds:
            normalized = helion.Config.from_dict(
                spec.default_config().config | seed.config
            )
            spec.normalize(normalized, _fix_invalid=True)
            assert all(
                (key in normalized.config) is controls_active for key in controls
            )
            assert all(normalized.config.get(key, False) is False for key in controls)
            expected = with_flat_min_blocks_default(spec, normalized)
            assert generation.unflatten(generation.flatten(normalized)) == expected
            flat, effective = generation.strict_config_pair(normalized)
            assert effective == expected
            assert generation.unflatten(flat) == effective
            for key in controls:
                supplied = helion.Config.from_dict(normalized.config | {key: True})
                with pytest.raises(InvalidConfig):
                    generation.strict_config_pair(supplied)
                if not controls_active:
                    supplied = helion.Config.from_dict(normalized.config | {key: False})
                    with pytest.raises(
                        InvalidConfig,
                        match=f"Coverage transfer changed supplied field '{key}'",
                    ):
                        generation.strict_config_pair(supplied)
                    # Repair may remove inactive neutral payloads; strict
                    # transfer above must still reject their silent loss.
                    spec.normalize(supplied, _fix_invalid=True)
                    assert supplied == normalized
    assert torch.cuda.is_initialized() is initialized
