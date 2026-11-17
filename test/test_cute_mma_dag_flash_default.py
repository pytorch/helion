"""Default ownership when fused flash and MMA-DAG both admit attention."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_flash import memoized_flash_fragments
from helion._testing.cute_flash_overlap import overlapping_attention

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("dtype", "head_dim", "k_major"),
    [
        (torch.bfloat16, 64, False),
        (torch.float16, 64, False),
        (torch.bfloat16, 64, True),
        (torch.bfloat16, 128, True),
    ],
)
def test_fused_flash_retains_its_default_with_mma_dag_admission(
    dtype: torch.dtype, head_dim: int, k_major: bool
) -> None:
    q = torch.empty((2, 256, head_dim), dtype=dtype)
    k = torch.empty_like(q)
    v = torch.empty((2, head_dim, 256) if k_major else q.shape, dtype=dtype)
    with cpu_target(), memoized_flash_fragments():
        bound = cpu_bind(
            helion.kernel(overlapping_attention, backend="cute", static_shapes=True),
            (q, k, v, k_major),
        )
        spec = bound.config_spec
        assert spec.cute_flash_search_enabled
        assert spec.cute_mma_dag is not None
        assert spec.cute_mma_dag_facts is not None
        assert "warp" in spec.cute_mma_dag.choices
        config = spec.default_config()
        assert config.config["cute_mma_dag"] == "off"
        source = bound.to_code(config)
    assert "_helion_flash_rt" in source
