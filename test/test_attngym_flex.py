"""CPU coverage of the benchmark's output-only FlexAttention envelope."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from benchmarks.cute import attngym_figure
from benchmarks.cute import attngym_flex
from benchmarks.cute import attngym_variants as agv
import pytest
import torch
from torch.nn.attention.flex_attention import _identity
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import create_mask
from torch.nn.attention.flex_attention import noop_mask
import torch.nn.functional as F

from test._cute_aux import _cpu_codegen

import helion
from helion._compiler.cute import flex_match

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


def _causal(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return q_idx >= kv_idx


def _batch_head_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return (q_idx >= 8 + b * 3 + h) & (q_idx >= kv_idx) & ((kv_idx // 128) % 2 == b)


def _dependent_mods(device: torch.device) -> tuple[None, agv.MaskMod]:
    return None, _batch_head_mask


def _qkv(
    m: int, n: int, d: int, dv: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return tuple(
        torch.randn(shape, dtype=torch.bfloat16, generator=gen)
        for shape in ((2, 4, m, d), (2, 2, n, d), (2, 2, n, dv))
    )


def _arguments(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    batch_head_mask: bool = False,
    q_block: int = 128,
) -> tuple[object, ...]:
    block_mask = create_block_mask(
        _batch_head_mask if batch_head_mask else _causal,
        2 if batch_head_mask else None,
        4 if batch_head_mask else None,
        q.size(-2),
        k.size(-2),
        device="cpu",
        BLOCK_SIZE=(q_block, 128),
    )
    full_num = block_mask.full_kv_num_blocks
    full_idx = block_mask.full_kv_indices
    if full_num is None or full_idx is None:
        full_num = torch.zeros_like(block_mask.kv_num_blocks)
        full_idx = torch.zeros_like(block_mask.kv_indices)
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
        True,
    )


@pytest.mark.parametrize("q_block", (128, 256))
def test_asymmetric_output_only_kernel_matches_flash(q_block: int) -> None:
    q, k, v = _qkv(512, 512, 192, 128)
    args = _arguments(q, k, v, q_block=q_block)
    kernel = helion.kernel(
        attngym_flex.attention_gym_flex_kernel,
        backend="cute",
        static_shapes=True,
        fast_math=True,
    )
    with (
        _cpu_codegen(),
        patch("helion._compiler.backend._attention_flash_supported", return_value=True),
        patch(
            "helion._compiler.backend._attention_flash_gate_enabled", return_value=True
        ),
    ):
        bound = kernel._bind_isolated(args)
        host = bound.host_function
        assert host is not None
        reasons: list[str] = []
        with bound.env:
            match = flex_match.match_flex_envelope(
                host.device_ir.graphs,
                host.device_ir.grid_block_ids[0],
                env=bound.env,
                host_function=host,
                reasons=reasons,
            )
    assert match is not None, reasons
    assert (match.head_dim, match.head_dim_v, match.kv_heads) == (192, 128, 2)
    assert match.lse_name is None
    assert bound.config_spec.cute_flash_search_enabled
    assert bound.config_spec._cute_flash_head_dim_v == 128


def test_batch_and_head_dependent_tables_match_flash() -> None:
    q, k, v = _qkv(384, 512, 64, 64)
    args = _arguments(q, k, v, batch_head_mask=True)
    kernel = helion.kernel(
        attngym_flex.attention_gym_flex_kernel,
        backend="cute",
        static_shapes=True,
        fast_math=True,
    )
    with (
        _cpu_codegen(),
        patch("helion._compiler.backend._attention_flash_supported", return_value=True),
        patch(
            "helion._compiler.backend._attention_flash_gate_enabled", return_value=True
        ),
    ):
        bound = kernel._bind_isolated(args)
    assert bound.config_spec.cute_flash_search_enabled
    assert args[4].shape[:2] == (2, 4)


@pytest.mark.parametrize(
    ("name", "m", "n", "params"),
    (
        ("vsa", 256, 512, {"tile_numel": 128, "top_k": 2}),
        (
            "graphormer-spatial",
            512,
            512,
            {"max_distance": 4, "path_len": 3, "edge_types": 8},
        ),
        (
            "graphormer-edge",
            512,
            512,
            {"max_distance": 4, "path_len": 3, "edge_types": 8},
        ),
        ("document", 512, 512, {"document_lengths": [128] * 4}),
        (
            "jetspec-tree",
            64,
            576,
            {"prefix": 512, "parents": [-1, *[(i - 1) // 2 for i in range(1, 64)]]},
        ),
    ),
)
def test_figure_gather_hooks_match_the_fused_envelope(
    name: str, m: int, n: int, params: dict[str, Any]
) -> None:
    device = torch.device("cpu")
    case = attngym_figure._case(name, "cpu-test", m, n=n, params=params)
    q, k, v = agv.make_qkv(case, device)
    _, torch_mask = case.make_mods(device)
    block_mask = create_block_mask(
        noop_mask if torch_mask is None else torch_mask,
        case.mask_batch,
        case.mask_heads,
        m,
        n,
        device="cpu",
        BLOCK_SIZE=128,
    )
    with (
        patch.dict("os.environ", {"HELION_BACKEND": "cute"}),
        patch.object(agv, "build_block_mask", return_value=block_mask),
    ):
        kernel, args = attngym_flex.kernel_call_args(case, q, k, v, device)
    with (
        _cpu_codegen(),
        patch("helion._compiler.backend._attention_flash_supported", return_value=True),
        patch(
            "helion._compiler.backend._attention_flash_gate_enabled", return_value=True
        ),
    ):
        bound = kernel._bind_isolated(args)
        host = bound.host_function
        assert host is not None
        reasons: list[str] = []
        with bound.env:
            match = flex_match.match_flex_envelope(
                host.device_ir.graphs,
                host.device_ir.grid_block_ids[0],
                env=bound.env,
                host_function=host,
                reasons=reasons,
            )
    assert match is not None, reasons
    assert match.lse_name is None
    assert match.aux_names
    assert bound.config_spec.cute_flash_search_enabled
    if name.startswith("graphormer"):
        assert match.plan.score_region is not None
    else:
        assert match.plan.mask_region is not None
    if name == "vsa":
        assert match.plan.kv_walk.table_batch == case.b
        assert match.plan.kv_walk.table_heads == case.hq


def test_call_args_and_baseline_cover_empty_rows_asymmetry_and_gqa() -> None:
    q, k, v = _qkv(129, 257, 192, 128)
    tables = _arguments(q, k, v, batch_head_mask=True)
    block_mask = SimpleNamespace(
        kv_num_blocks=tables[4],
        kv_indices=tables[5],
        full_kv_num_blocks=None,
        full_kv_indices=None,
        BLOCK_SIZE=(128, 128),
    )
    case = agv.Variant(
        "dependent-test",
        2,
        4,
        2,
        129,
        257,
        192,
        torch.bfloat16,
        _dependent_mods,
        dv=128,
        mask_batch=2,
        mask_heads=4,
    )
    with (
        patch.dict("os.environ", {"HELION_BACKEND": "cute"}),
        patch.object(agv, "build_block_mask", return_value=block_mask),
        patch.object(
            attngym_figure, "helion_mods", return_value=(None, _batch_head_mask)
        ),
    ):
        compiled_kernel, args = attngym_flex.kernel_call_args(
            case, q, k, v, torch.device("cpu")
        )
        cached_kernel, _ = attngym_flex.kernel_call_args(
            case, q, k, v, torch.device("cpu")
        )
        assert cached_kernel is compiled_kernel
        assert compiled_kernel.settings.autotune_baseline_fn is not None
        baseline = compiled_kernel.settings.autotune_baseline_fn(*args)
    assert baseline.shape == (2, 4, 129, 128)
    assert torch.isfinite(baseline).all()
    assert torch.count_nonzero(baseline[:, :, :8]) == 0
    assert args[4].shape[:2] == (2, 4)
    assert torch.count_nonzero(args[6]) == 0
    assert torch.count_nonzero(args[7]) == 0
    mask = create_mask(_batch_head_mask, 2, 4, 129, 257, device="cpu")
    expected = F.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(2, dim=1),
        v.float().repeat_interleave(2, dim=1),
        attn_mask=mask,
    ).to(q.dtype)
    torch.testing.assert_close(baseline, expected, atol=5e-2, rtol=2e-2)


def test_mismatched_value_sequence_is_rejected() -> None:
    q, k, v = _qkv(128, 256, 64, 64)
    args = list(_arguments(q, k, v))
    args[2] = v[:, :, :128].contiguous()
    kernel = helion.kernel(
        attngym_flex.attention_gym_flex_kernel,
        backend="triton",
        ref_mode=helion.RefMode.EAGER,
    )
    with pytest.raises(AssertionError):
        kernel(*args)
