"""Codegen coverage for fixed-config TPU kernels under ``pretuned_kernels``."""

from __future__ import annotations

from pretuned_kernels.causal_conv1d.causal_conv1d import CAUSAL_CONV1D_CONFIG
from pretuned_kernels.causal_conv1d.causal_conv1d import _causal_conv1d_decode_kernel
from pretuned_kernels.gdn_decode.gdn_decode import GDN_DECODE_CONFIG
from pretuned_kernels.gdn_decode.gdn_decode import gdn_decode
import torch


def test_causal_conv1d_decode_uses_indirect_dma() -> None:
    tokens, heads, head_dim = 512, 4, 128
    args = (
        torch.empty(tokens, heads, head_dim, dtype=torch.bfloat16),
        torch.empty(4, heads, head_dim, dtype=torch.bfloat16),
        torch.empty(tokens, 3, heads, head_dim, dtype=torch.bfloat16),
        torch.empty(heads, head_dim, dtype=torch.bfloat16),
        torch.arange(tokens, dtype=torch.int32),
        torch.tensor([tokens, tokens, tokens], dtype=torch.int32),
        torch.ones(tokens, dtype=torch.int32),
    )
    code = _causal_conv1d_decode_kernel.bind(args).to_code()
    assert CAUSAL_CONV1D_CONFIG.block_sizes == [128]
    assert "_BLOCK_SIZE_1 = int(128)" in code
    assert "conv_state_gather_buf" in code
    assert "_scatter_copy" in code
    assert "one_hot" not in code


def test_gdn_decode_uses_indirect_dma() -> None:
    tokens, heads, dim = 512, 2, 128
    args = (
        *(torch.empty(tokens, heads, dim, dtype=torch.bfloat16) for _ in range(4)),
        torch.empty(tokens, heads, dim, dim, dtype=torch.float32),
        torch.arange(tokens, dtype=torch.int32),
        torch.tensor([tokens, tokens, tokens], dtype=torch.int32),
        torch.empty(tokens, heads, dim, dtype=torch.bfloat16),
        torch.empty(heads, dim, dtype=torch.float32),
        torch.empty(heads, dim, dtype=torch.float32),
        torch.ones(tokens, dtype=torch.int32),
        dim**-0.5,
    )
    code = gdn_decode.bind(args).to_code()
    assert GDN_DECODE_CONFIG.block_sizes == [32]
    # Indirect DMA expands the logical token tile to its aligned physical extent.
    assert "_BLOCK_SIZE_1 = int(128)" in code
    assert "recurrent_state_gather_buf" in code
    assert "_scatter_copy" in code
    assert "one_hot" not in code
