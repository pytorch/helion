"""The resident value-graph softmax lowering on the dense LSE surface.

``examples/attention.py``'s ``attention`` and the dense-BlockMask flex kernel
return ``(out, lse)``.  The resident lowering used to be offered only on the
output-only dense surface; the LSE row needs nothing it lacks (the final row
statistics stay in the softmax warps' registers and the store follows the
row-sum publish, as on the standard lowering), so the surface fact
``dense_lse_output`` admits it there too.  The CPU tests bind the
example as if on sm_100 and read the surface; the GPU tests run the two-tile
fa4 body with the resident lowering against the standard lowering and the
fp32 reference, output and LSE.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention.flex_attention import _create_empty_block_mask
from torch.nn.attention.flex_attention import flex_attention as torch_flex

import helion
from helion._compiler.autotuner_heuristics.cute import CuteFlashAttentionHeuristic
from helion._compiler.cute import cute_flash
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import EnumFragment

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

pytestmark = skipUnlessBackends(["cute"])


def _meta_args(shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(*shape, dtype=torch.bfloat16, device="meta")  # @ignore-device-lint
        for _ in range(3)
    )


def _example_kernel(name: str) -> helion.Kernel:
    import examples.attention as attention_module

    kernel = getattr(attention_module, name)
    return helion.Kernel(kernel.fn, settings=kernel.settings.copy(backend="cute"))


def _bind_as_if_sm100(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...]
) -> BoundKernel:
    with (
        patch("helion._compiler.backend._attention_flash_supported", return_value=True),
        patch(
            "helion._compiler.backend._attention_flash_gate_enabled", return_value=True
        ),
    ):
        return kernel.bind(args)


def _resident_seeds(bound: BoundKernel) -> list[helion.Config]:
    with bound.env:
        seeds = CuteFlashAttentionHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    return [
        seed
        for seed in seeds or ()
        if seed.config.get(cute_flash.FLASH_SOFTMAX_LOWERING_KEY)
        == "resident_value_graph"
    ]


def test_attention_with_lse_is_the_dense_lse_surface() -> None:
    """The example that returns ``(out, lse)`` reports the LSE fact, not the
    output-only one; the output-only example reports the reverse."""
    with_lse = _bind_as_if_sm100(
        _example_kernel("attention"), _meta_args((2, 16, 2048, 128))
    )
    spec = with_lse.config_spec
    assert spec.cute_flash_search_enabled
    assert not spec._cute_flash_standard_dense_output
    assert spec._cute_flash_dense_lse_output
    assert not spec._cute_flash_has_row_epilogue

    output_only = _bind_as_if_sm100(
        _example_kernel("attention_output"), _meta_args((2, 16, 2048, 128))
    )
    spec = output_only.config_spec
    assert spec.cute_flash_search_enabled
    assert spec._cute_flash_standard_dense_output
    assert not spec._cute_flash_dense_lse_output


def test_lse_surface_offers_and_seeds_the_resident_lowering() -> None:
    """The fragment offers the resident lowering on the LSE surface and the
    compiler seeds carry the two-tile fa4 body with it; the seed survives the
    spec's normalization with the lowering intact (a seed whose structural
    prerequisites are not pinned is canonicalized away silently)."""
    bound = _bind_as_if_sm100(
        _example_kernel("attention"), _meta_args((2, 16, 2048, 128))
    )
    spec = bound.config_spec
    with bound.env:
        generation = spec.create_config_generation()
        fragment = generation._flat_fields()[cute_flash.FLASH_SOFTMAX_LOWERING_KEY]
        assert isinstance(fragment, EnumFragment)
        assert "resident_value_graph" in fragment._active_choices()
    seeds = _resident_seeds(bound)
    assert seeds, "the LSE surface seeds no resident body"
    with bound.env:
        for seed in seeds:
            values = dict(seed.config)
            spec.normalize(values, _fix_invalid=False)
            assert (
                values[cute_flash.FLASH_SOFTMAX_LOWERING_KEY] == "resident_value_graph"
            )
            assert values[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == "fa4"
            assert values[cute_flash.FLASH_Q_TILE_COUNT_KEY] == 2
            resolved = spec._resolve_cute_flash_config(values)
            assert resolved.softmax_lowering == "resident_value_graph"


def test_relu_row_program_surface_keeps_the_resident_lowering_off() -> None:
    """The fused row program on ``O`` is not ``out + lse``: no resident seed,
    no resident choice (``has_row_epilogue`` fails closed)."""
    import examples.attention as attention_module

    if not hasattr(attention_module, "attention_relu_output"):
        pytest.skip("no relu row-program example")
    bound = _bind_as_if_sm100(
        _example_kernel("attention_relu_output"), _meta_args((2, 16, 2048, 128))
    )
    spec = bound.config_spec
    if not spec._cute_flash_has_row_epilogue:
        pytest.skip("the relu output is not a row program on this surface")
    assert not spec._cute_flash_dense_lse_output
    assert not _resident_seeds(bound)


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(
        q.size(-1)
    )
    lse = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out, lse


@onlyBackends(["cute"])
@skipIfNotCUDA()
@pytest.mark.parametrize(
    ("family", "scale"), (("fa4", 1.0), ("fa4", 4.0), ("fa4_2cta", 1.0))
)
def test_resident_two_tile_body_stores_the_lse(family: str, scale: float) -> None:
    """The two-tile fa4 body (and its 2-CTA sibling) with the resident
    lowering on the LSE example: output and LSE against the standard lowering
    of the same body and the fp32 reference."""
    import examples.attention as attention_module

    kernel = attention_module.attention
    torch.manual_seed(0)
    q, k, v = (
        (torch.randn(2, 4, 1024, 128, dtype=torch.bfloat16, device=DEVICE) * scale)
        for _ in range(3)
    )
    bound = _bind_as_if_sm100(
        helion.Kernel(kernel.fn, settings=kernel.settings.copy(backend="cute")),
        (q, k, v),
    )
    seeds = _resident_seeds(bound)
    assert seeds
    # The surface seeds the one-CTA body; the 2-CTA sibling is the same seed
    # requested on its family (the fragment offers the lowering there too).
    resident = {**seeds[0].config, cute_flash.FLASH_PIPELINE_FAMILY_KEY: family}
    with bound.env:
        bound.config_spec.normalize(resident, _fix_invalid=False)
    assert resident[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == family
    assert resident[cute_flash.FLASH_SOFTMAX_LOWERING_KEY] == "resident_value_graph"
    standard = {**resident, cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "standard"}
    code, (out, lse) = code_and_output(
        helion.Kernel(kernel.fn, settings=kernel.settings.copy(backend="cute")),
        (q, k, v),
        **resident,
    )
    assert "resident_softmax_value_graph(" in code
    assert code.count("_flash_mLSE[") == 2
    _, (std_out, std_lse) = code_and_output(
        helion.Kernel(kernel.fn, settings=kernel.settings.copy(backend="cute")),
        (q, k, v),
        **standard,
    )
    ref_out, ref_lse = _reference(q, k, v)
    assert not torch.isnan(lse).any().item()
    torch.testing.assert_close(out, ref_out, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=5e-3, rtol=1e-3)
    # The two lowerings run the same exp2 (MUFU) on the same scores; the row
    # sums differ only in reduction order.
    torch.testing.assert_close(lse, std_lse, atol=5e-3, rtol=1e-3)
    torch.testing.assert_close(out, std_out, atol=3e-2, rtol=2e-2)


def _identity(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return score


def _flex_args(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask: Any
) -> tuple[Any, ...]:
    """The dense-BlockMask arguments the flex example's wrapper builds."""
    import examples.flex_attention as flex_module

    full_num = block_mask.full_kv_num_blocks
    full_idx = block_mask.full_kv_indices
    if full_num is None or full_idx is None:
        full_num = flex_module._zero_block_table_like(block_mask.kv_num_blocks)
        full_idx = flex_module._zero_block_table_like(block_mask.kv_indices)
    return (
        q,
        k,
        v,
        _identity,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        full_num,
        full_idx,
        block_mask.mask_mod,
        block_mask.BLOCK_SIZE[0],
        block_mask.BLOCK_SIZE[1],
        1.0 / math.sqrt(q.size(-1)),
        k.size(1) != q.size(1),
    )


@onlyBackends(["cute"])
@skipIfNotCUDA()
@pytest.mark.parametrize(
    ("family", "kv_heads"), (("fa4", 4), ("fa4_2cta", 4), ("fa4", 2))
)
def test_resident_two_tile_body_stores_the_flex_lse(family: str, kv_heads: int) -> None:
    """The dense-BlockMask flex kernel (the example wrapper's ``block_mask=None``
    surface: one KV block covering the sequence, natural-log LSE, the scale a
    scalar argument) with the resident lowering on the two-tile fa4 body, its
    2-CTA sibling and a 4:2 GQA head layout: output and LSE against the
    standard lowering and ``flex_attention``."""
    import examples.flex_attention as flex_module

    kernel = flex_module.helion_flex_attention_kernel
    torch.manual_seed(0)
    q = torch.randn(1, 4, 1024, 128, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(1, kv_heads, 1024, 128, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(1, kv_heads, 1024, 128, dtype=torch.bfloat16, device=DEVICE)
    block_mask = _create_empty_block_mask(q, k)
    args = _flex_args(q, k, v, block_mask)
    bound = _bind_as_if_sm100(
        helion.Kernel(kernel.fn, settings=kernel.settings.copy(backend="cute")), args
    )
    spec = bound.config_spec
    assert spec._cute_flash_dense_lse_output
    assert not spec._cute_flash_standard_dense_output
    seeds = _resident_seeds(bound)
    assert seeds, "the flex dense-BlockMask surface seeds no resident body"
    resident = {**seeds[0].config, cute_flash.FLASH_PIPELINE_FAMILY_KEY: family}
    with bound.env:
        spec.normalize(resident, _fix_invalid=False)
    assert resident[cute_flash.FLASH_PIPELINE_FAMILY_KEY] == family
    assert resident[cute_flash.FLASH_SOFTMAX_LOWERING_KEY] == "resident_value_graph"
    standard = {**resident, cute_flash.FLASH_SOFTMAX_LOWERING_KEY: "standard"}
    code, (out, lse) = code_and_output(
        helion.Kernel(kernel.fn, settings=kernel.settings.copy(backend="cute")),
        args,
        **resident,
    )
    assert "resident_softmax_value_graph(" in code
    _, (std_out, std_lse) = code_and_output(
        helion.Kernel(kernel.fn, settings=kernel.settings.copy(backend="cute")),
        args,
        **standard,
    )
    ref_out, ref_lse = torch_flex(
        q.float(),
        k.float(),
        v.float(),
        block_mask=block_mask,
        scale=1.0 / math.sqrt(q.size(-1)),
        enable_gqa=kv_heads != q.size(1),
        return_lse=True,
    )
    assert not torch.isnan(lse).any().item()
    torch.testing.assert_close(out.float(), ref_out, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(lse.float(), ref_lse, atol=5e-3, rtol=1e-3)
    torch.testing.assert_close(lse, std_lse, atol=5e-3, rtol=1e-3)
    torch.testing.assert_close(out, std_out, atol=3e-2, rtol=2e-2)
