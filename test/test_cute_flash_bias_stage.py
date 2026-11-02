"""``cute_flash_bias_stage``: the fa4 body's additive tensor bias read from a
TMA-staged shared-memory ring instead of global memory.

The direct path (``add_score_bias_t2r_contiguous``) issues one 8-byte global
load per 4 columns from every softmax thread, each lane of a warp on a
different row: 32 sectors per request, 4.2 M L1 wavefronts for a 32 MiB bias
at (2, 8, 1024, 64), +41 us over the plain fa4 kernel against Triton's
pipelined +10 us. The staged form (``tma``) has the load warp TMA-copy both
query tiles' (128, 64) boxes of each KV tile into a ring one step ahead of
the softmax, which reads its rows with the same per-element arithmetic
(bitwise the same result). The resolver admits it on the plain one-CTA fa4
chunked body alone and only where the ring leaves a two-deep K/V ring; it is
then the default. These tests render on the CPU except the GPU equality test.
"""

from __future__ import annotations

from examples.attention import attention
from examples.attention import biased_attention
from examples.attention import biased_attention_output
import pytest
import torch

from test._cute_binding import _cpu_bind

import helion
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

STAGED_MARKERS = (
    "_flash_tma_bias",
    "sBias = storage.sBias.get_tensor(_flash_bsl.outer, swizzle=_flash_bsl.inner)",
    "cute.arch.mbarrier_init(flash_bias_empty_ptr + flash_st, 256)",
    "_helion_flash_rt.add_score_bias_smem_t2r(",
    "cute.arch.mbarrier_arrive_and_expect_tx(flash_bias_full_ptr + flash_bias_slot",
    "_helion_flash_rt.mbarrier_arrive(flash_bias_empty_ptr + flash_bias_slot)",
)
DIRECT_MARKER = "_helion_flash_rt.add_score_bias_t2r_contiguous("


def _kernel(fn, **settings):
    return helion.kernel(
        fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        autotune_effort="none",
        autotune_accuracy_check=False,
        **settings,
    )


def _cpu_args(shape, dtype=torch.bfloat16, bias=True):
    b, h, s, d = shape
    tensors = [torch.empty(shape, dtype=dtype) for _ in range(3)]
    if bias:
        tensors.append(torch.empty(b, h, s, s, dtype=dtype))
    return tuple(tensors)


def _bias_stage_fragment(spec):
    return spec._cute_flash_autotune_fragments()["cute_flash_bias_stage"]


@skipUnlessBackends(["cute"])
def test_biased_plan_stages_the_bias_by_default() -> None:
    bound = _cpu_bind(_kernel(biased_attention.fn), _cpu_args((2, 8, 1024, 64)))
    spec = bound.config_spec
    assert spec.cute_flash_search_enabled
    cfg = spec.default_config()
    assert cfg.config["cute_flash_bias_stage"] == "tma"
    fragment = _bias_stage_fragment(spec)
    assert set(fragment.choices) == {"tma", "direct"}
    assert set(fragment.search_choices) == {"tma", "direct"}
    staged = bound.to_code(cfg)
    for marker in STAGED_MARKERS:
        assert marker in staged, marker
    assert DIRECT_MARKER not in staged
    # Two ring stages at d64 (both query tiles' boxes: 64 KB each) beside the
    # three-deep K/V ring; the launcher reads the depth from the plan.
    assert "'bias_stage': 2" in staged
    assert "bias_elems=65536, bias_stages=2" in staged
    assert "'kv_stage': 3" in staged
    direct_cfg = helion.Config(**{**cfg.config, "cute_flash_bias_stage": "direct"})
    assert spec.normalized_config(direct_cfg).config["cute_flash_bias_stage"] == (
        "direct"
    )
    direct = bound.to_code(direct_cfg)
    assert DIRECT_MARKER in direct
    for marker in STAGED_MARKERS:
        assert marker not in direct, marker
    assert "bias_stage" not in direct


@skipUnlessBackends(["cute"])
def test_head_dim_128_keeps_a_two_deep_kv_ring() -> None:
    """At d128 the ring competes with the staged output tiles and the K/V
    ring for shared memory: the d128 default stages the output
    (``cute_flash_epi_tma``), which leaves no room for a bias ring beside
    two K/V stages, so the default reads ``direct`` (fail closed) while the
    knob stays searchable; without the staged output the resolver takes the
    one-stage ring and keeps a two-deep K/V ring."""
    bound = _cpu_bind(
        _kernel(biased_attention_output.fn), _cpu_args((1, 2, 256, 128), torch.float16)
    )
    spec = bound.config_spec
    cfg = spec.default_config()
    assert cfg.config["cute_flash_epi_tma"] is True
    assert cfg.config["cute_flash_bias_stage"] == "direct"
    assert set(_bias_stage_fragment(spec).search_choices) == {"tma", "direct"}
    requested = helion.Config(
        **{**cfg.config, "cute_flash_epi_tma": False, "cute_flash_bias_stage": "tma"}
    )
    normalized = spec.normalized_config(requested)
    assert normalized.config["cute_flash_bias_stage"] == "tma"
    assert normalized.config["cute_flash_kv_stage"] >= 2
    code = bound.to_code(requested)
    assert "'bias_stage': 1" in code
    assert "bias_elems=32768, bias_stages=1" in code
    # One stage: slot 0, parity from the step counter's low bit.
    assert "flash_bias_slot = cutlass.Int32(0)" in code
    assert "flash_bias_par = flash_bias_pidx & 1" in code
    # With the staged output the ring does not fit: the request normalizes
    # back onto the direct path (one identity with the default).
    staged_output = helion.Config(**{**cfg.config, "cute_flash_bias_stage": "tma"})
    assert spec.normalized_config(staged_output).config["cute_flash_bias_stage"] == (
        "direct"
    )
    # The search starts from the staged bias beside the plain epilogue too
    # (133 us against 239 us for the default at (2, 16, 2048, 128)).
    assert any(
        seed.config.get("cute_flash_bias_stage") == "tma"
        and seed.config.get("cute_flash_epi_tma") is False
        for seed in spec.autotune_seed_configs()
    )


@skipUnlessBackends(["cute"])
def test_plans_without_a_tensor_bias_pin_direct() -> None:
    bound = _cpu_bind(_kernel(attention.fn), _cpu_args((2, 8, 1024, 64), bias=False))
    spec = bound.config_spec
    cfg = spec.default_config()
    assert cfg.config["cute_flash_bias_stage"] == "direct"
    fragment = _bias_stage_fragment(spec)
    assert fragment.choices == ("direct",)
    # Off the plan's surface the staged form is not a legal value (the
    # fragment validation, like every pinned flash knob); the resolver's own
    # check rejects unknown values.
    for value in ("tma", "coop"):
        with pytest.raises(InvalidConfig, match="cute_flash_bias_stage"):
            spec.normalized_config(
                helion.Config(**{**cfg.config, "cute_flash_bias_stage": value})
            )
    assert "bias" not in bound.to_code(cfg)


@skipUnlessBackends(["cute"])
@skipIfRefEager("compares the two bias lowerings' kernels")
@skipIfCudaCapabilityLessThan((10, 0), reason="the fused fa4 flash kernel")
@pytest.mark.parametrize(
    ("shape", "dtype", "with_lse"),
    [
        ((1, 2, 256, 64), torch.bfloat16, True),
        ((1, 2, 256, 128), torch.float16, False),
    ],
)
def test_staged_bias_matches_the_direct_path_bitwise(
    shape: tuple[int, int, int, int], dtype: torch.dtype, with_lse: bool
) -> None:
    torch.manual_seed(0)
    b, h, s, d = shape
    q, k, v = (torch.randn(shape, dtype=dtype, device=DEVICE) for _ in range(3))
    bias = torch.randn(b, h, s, s, dtype=dtype, device=DEVICE) * 0.5
    kernel = _kernel((biased_attention if with_lse else biased_attention_output).fn)
    args = (q, k, v, bias)
    bound = kernel.bind(args)
    spec = bound.config_spec
    base = dict(spec.default_config().config)
    if base["cute_flash_bias_stage"] != "tma":
        # d128: the ring fits beside the plain epilogue, not the staged
        # output tile (see ``test_head_dim_128_keeps_a_two_deep_kv_ring``).
        base["cute_flash_epi_tma"] = False
        base["cute_flash_epi_stg"] = False
    results = {}
    for stage in ("tma", "direct"):
        values = {**base, "cute_flash_bias_stage": stage}
        assert (
            spec.normalized_config(helion.Config(**values)).config[
                "cute_flash_bias_stage"
            ]
            == stage
        )
        code, out = code_and_output(kernel, args, **values)
        assert ("add_score_bias_smem_t2r" in code) == (stage == "tma")
        results[stage] = out if isinstance(out, tuple) else (out,)
    for staged, direct in zip(results["tma"], results["direct"], strict=True):
        assert torch.equal(staged, direct)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), attn_mask=bias.float()
    )
    torch.testing.assert_close(
        results["tma"][0].float(), expected, atol=2e-2, rtol=2e-2
    )
