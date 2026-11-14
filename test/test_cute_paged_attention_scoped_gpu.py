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
    "form,query_len,fast_math,return_lse,pdl",
    [
        ("mapped", 1, False, False, False),
        ("mapped", 4, True, True, True),
        ("mapped", 5, False, False, False),
        ("per_key", 1, True, True, False),
        ("per_key", 4, False, False, False),
        ("per_key", 5, True, True, True),
    ],
)
def test_scoped_attention_roles_and_changed_replay(
    form, query_len, fast_math, return_lse, pdl
):
    # Each physical panel runs with both layouts at the proved entry budget;
    # distribute math/LSE/PDL coverage without a Cartesian native grid.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the attention transport uses SM100 tensor memory")
    kernel, args = optional_lse_fixture(form, return_lse, DEVICE, query_len=query_len)
    initialize_paged_attention_inputs(args)
    bound = helion.kernel(
        kernel.fn, backend="cute", static_shapes=True, fast_math=fast_math
    ).bind(args)
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag_pass_warps="attention_scoped",
        cute_min_blocks_per_mp=2,
        cute_pdl=pdl,
    )
    source = bound.to_code(config)
    assert "scoped_registers=True" in source
    assert "block=(384, 1, 1)" in source
    check_optional_lse_replay(bound, config, args, form, return_lse)
