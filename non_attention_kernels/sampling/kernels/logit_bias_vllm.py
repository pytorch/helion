# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Source: vllm/v1/worker/gpu/sample/logit_bias.py
"""
Logit bias kernel from vLLM.
Applies allowed token IDs, logit biases, and min tokens constraints.
"""
import torch

import triton
import triton.language as tl


@triton.jit
def _bias_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    idx_mapping_ptr,
    # Allowed token IDs.
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    # Logit bias.
    num_logit_bias_ptr,
    bias_token_ids_ptr,
    bias_token_ids_stride,
    bias_ptr,
    bias_stride,
    # Min tokens.
    pos_ptr,
    min_lens_ptr,
    num_stop_token_ids_ptr,
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    block = tl.arange(0, BLOCK_SIZE)

    # Allowed token IDs.
    num_allowed_token_ids = tl.load(num_allowed_token_ids_ptr + req_state_idx)
    if num_allowed_token_ids > 0:
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < num_allowed_token_ids

        # Save logits for allowed token IDs.
        allowed_token_ids = tl.load(
            allowed_token_ids_ptr + req_state_idx * allowed_token_ids_stride + block,
            mask=mask,
        )
        logits = tl.load(
            logits_ptr + batch_idx * logits_stride + allowed_token_ids, mask=mask
        )

        # Set logits to -inf for all tokens.
        for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
            offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
            tl.store(
                logits_ptr + batch_idx * logits_stride + offset,
                -float("inf"),
                mask=offset < vocab_size,
            )

        # Restore logits for allowed token IDs.
        tl.store(
            logits_ptr + batch_idx * logits_stride + allowed_token_ids,
            logits,
            mask=mask,
        )

    # Logit bias.
    num_logit_bias = tl.load(num_logit_bias_ptr + req_state_idx)
    if num_logit_bias > 0:
        mask = block < num_logit_bias
        token_ids = tl.load(
            bias_token_ids_ptr + req_state_idx * bias_token_ids_stride + block,
            mask=mask,
        )
        bias = tl.load(bias_ptr + req_state_idx * bias_stride + block, mask=mask)
        logits = tl.load(logits_ptr + batch_idx * logits_stride + token_ids, mask=mask)
        logits += bias
        tl.store(logits_ptr + batch_idx * logits_stride + token_ids, logits, mask=mask)

    # Apply min tokens.
    num_stop_token_ids = tl.load(num_stop_token_ids_ptr + req_state_idx)
    pos = tl.load(pos_ptr + batch_idx)
    min_len = tl.load(min_lens_ptr + req_state_idx)
    if num_stop_token_ids > 0 and pos < min_len:
        mask = block < num_stop_token_ids
        stop_token_ids = tl.load(
            stop_token_ids_ptr + req_state_idx * stop_token_ids_stride + block,
            mask=mask,
        )
        tl.store(
            logits_ptr + batch_idx * logits_stride + stop_token_ids,
            -float("inf"),
            mask=mask,
        )


def apply_logit_bias(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    pos: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    logit_bias_token_ids: torch.Tensor,
    logit_bias: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
) -> None:
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(
        max(
            allowed_token_ids.shape[-1],
            logit_bias_token_ids.shape[-1],
            stop_token_ids.shape[-1],
        )
    )
    LOGITS_BLOCK_SIZE = 8192
    _bias_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        vocab_size,
        idx_mapping,
        num_allowed_token_ids,
        allowed_token_ids,
        allowed_token_ids.stride(0),
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias_token_ids.stride(0),
        logit_bias,
        logit_bias.stride(0),
        pos,
        min_lens,
        num_stop_token_ids,
        stop_token_ids,
        stop_token_ids.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        LOGITS_BLOCK_SIZE=LOGITS_BLOCK_SIZE,
    )
