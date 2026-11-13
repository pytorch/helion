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
@pytest.mark.parametrize("form", ["reduced", "per_key", "mapped"])
@pytest.mark.parametrize("return_lse", [False, True])
def test_optional_lse_outputs_and_changed_replay(form: str, return_lse: bool) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the attention transport uses SM100 tensor memory")
    kernel, args = optional_lse_fixture(form, return_lse, DEVICE)
    initialize_paged_attention_inputs(args)
    # Cover both math policies without doubling the native configuration grid.
    bound = helion.kernel(
        kernel.fn, backend="cute", static_shapes=True, fast_math=form != "reduced"
    ).bind(args)
    config = optional_lse_config()
    source = bound.to_code(config)
    expected_body = (
        "paged_attention_body("
        if form == "reduced"
        else "paged_attention_partial_body("
    )
    assert expected_body in source
    check_optional_lse_replay(bound, config, args, form, return_lse)
