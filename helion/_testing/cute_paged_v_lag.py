"""Shared paired-loader fixtures and replay checks for bounded V lookbehind."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import cast

import torch

from . import cpu_render_target
from .cuda_graph import cuda_graph_for_replay
from .cute_codegen import cpu_bind
from .cute_optional_lse import bind_paged_attention_length_replay
from .cute_optional_lse import check_optional_lse_replay
from .cute_optional_lse import check_paged_attention_length_replay
from .cute_optional_lse import initialize_paged_attention_inputs
from .cute_optional_lse import optional_lse_config
from .cute_optional_lse import optional_lse_fixture
from .cute_paged_attention import paged_online_attention
from .cute_paged_attention_schedule import paged_attention_args
from .cute_paged_base2 import base2_attention_fixture
from .cute_paged_reference import reference_attention_fixture
import helion
from helion._compiler.cute.cute_mma_dag import MMA_DAG_V_LAG_CHOICES

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


@functools.cache
def paired_v_lag_bound(*, heads: int = 2, per_key: bool = True) -> BoundKernel[object]:
    args = (
        *paged_attention_args(rows=40 if per_key else 32, heads=heads),
        None,
        per_key,
    )
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(cast("Kernel[object]", paged_online_attention), args)


def paired_v_lag_config(lag: object) -> helion.Config:
    config = optional_lse_config()
    config.config.update(
        block_sizes=[64],
        cute_mma_dag_pass_warps="attention_paired",
        cute_mma_dag_v_lag=lag,
    )
    return config


def without_retirement_seeds(seeds: list[helion.Config]) -> list[helion.Config]:
    """Check the independent retirement suffix and return the prior seed prefix."""
    count = len(MMA_DAG_V_LAG_CHOICES)
    assert len(seeds) > count
    prefix, suffix = seeds[:-count], seeds[-count:]
    key = "cute_mma_dag_early_tmem_retirement"
    assert not any(seed.get(key, False) for seed in prefix)
    assert all(seed.get(key) is True for seed in suffix)
    assert (
        tuple(seed.get("cute_mma_dag_v_lag", 0) for seed in suffix)
        == MMA_DAG_V_LAG_CHOICES
    )
    assert all(
        seed.get("cute_mma_dag_probability_schedule", "serial") == "serial"
        for seed in suffix
    )
    return prefix


def check_paired_v_lag_replay(
    lag: int,
    form: str,
    query_len: int,
    base2: bool,
    reverse: bool,
    fast_math: bool,
    device: torch.device | str,
    *,
    heads: int = 2,
    early_tmem_retirement: bool = False,
) -> None:
    if base2:
        kernel, args = base2_attention_fixture(form, query_len, True, device)
        args = (*args, reverse)
    else:
        kernel, args = optional_lse_fixture(form, True, device, query_len=query_len)
        args = (*args, False, reverse)
    if form == "per_key" and heads != 2:
        tensors = paged_attention_args(rows=query_len * 8, heads=heads, device=device)
        args = (*tensors[:7], *args[7:])
    bound, args = bind_paged_attention_length_replay(
        kernel,
        args,
        form,
        fast_math=fast_math,
        reverse_pages=reverse,
    )
    config = paired_v_lag_config(lag)
    config.config["block_sizes"] = [helion.next_power_of_2(query_len * 8)]
    if early_tmem_retirement:
        config.config["cute_mma_dag_early_tmem_retirement"] = True
    source = bound.to_code(config)
    assert "paged_attention_paired_body(" in source
    assert ("early_tmem_retirement=True" in source) == early_tmem_retirement
    assert "probability_schedule=" not in source
    assert (f"v_lag={lag}" in source) == (lag > 0)
    assert ("base2_scores=True" in source) == base2
    assert ("reflected_dtype=" in source) == reverse
    check_optional_lse_replay(
        bound,
        config,
        args,
        form,
        True,
        reference_scores=base2,
        reverse_pages=reverse,
        base2_scores=base2,
    )
    check_paged_attention_length_replay(
        bound,
        config,
        args,
        form,
        True,
        reference_scores=base2,
        reverse_pages=reverse,
        base2_scores=base2,
    )


def check_paired_v_lag_extreme_bounds(lag: int, device: torch.device | str) -> None:
    """Guarded drains must skip extreme empty bounds and replay small reverse ranges."""
    kernel, initial = reference_attention_fixture("per_key", 4, True, device)
    args = list(initial)
    initialize_paged_attention_inputs(tuple(args))
    q, k, v, ks, vs, _pages, lengths = (cast("torch.Tensor", x) for x in args[:7])
    q.zero_()
    k.zero_()
    v.fill_(1)
    ks.fill_(127)
    vs.fill_(127)
    lengths.copy_(lengths.new_tensor([129, 257]))
    bounds = torch.tensor([[1, -(2**63)], [2**63 - 1, -1]], device=device)
    args[9] = -1
    args[10] = bounds
    inputs = (*args, True)
    bound = kernel.bind(inputs)
    config = paired_v_lag_config(lag)
    config.config["block_sizes"] = [32]
    source = bound.to_code(config)
    assert "reflected_dtype=cutlass.Int64" in source
    compiled = bound.compile_config(config)
    compiled(*inputs)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            captured = compiled(*inputs)
        for changed in (False, True):
            if changed:
                bounds.copy_(bounds.new_tensor([[0, 1], [0, 2]]))
            graph.replay()
            output, lse = captured
            assert lse is not None
            if not changed:
                assert torch.count_nonzero(output).item() == 0
                assert torch.isneginf(lse).all().item()
            else:
                torch.testing.assert_close(
                    output, torch.ones_like(output), rtol=0, atol=0
                )
                rows = torch.arange(32, device=device) // 8
                visible = torch.stack(
                    ((126 + rows).clamp_max(128), (254 + rows).clamp_max(256))
                )
                expected = visible.float().log().repeat_interleave(2, dim=0)
                torch.testing.assert_close(lse, expected, rtol=0, atol=1e-6)
