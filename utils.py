# Data Generation Utils
# Extracted from mkl/ops/utils/jfa_utils.py

import math
from typing import Any, Optional

import torch
from torch._prims_common import DeviceLikeType


def generate_sparse_seq_len(
    size: int,
    max_seq_len: int,
    sparsity: float,
    device: DeviceLikeType,
) -> torch.Tensor:
    """Generate random sequence lengths with given sparsity"""
    if sparsity == 0.0:
        return torch.zeros(size=(size,), device=device, dtype=torch.int)
    elif sparsity == 1.0:
        return torch.ones(size=(size,), device=device, dtype=torch.int) * max_seq_len
    elif sparsity >= 0.5:
        min_seq_len: int = int((2 * sparsity - 1.0) * max_seq_len)
        return torch.randint(
            low=max(min_seq_len, 1),
            high=max_seq_len,
            size=(size,),
            device=device,
            dtype=torch.int,
        )
    else:
        min_seq_len: int = 0
        max_seq_len: int = int(2 * sparsity * max_seq_len)
        return torch.randint(
            low=max(min_seq_len, 1),
            high=max(max_seq_len, 2),
            size=(size,),
            device=device,
            dtype=torch.int,
        )


def generate_jagged_data(
    B: int,
    max_M: int,
    D: int,
    H: int = 1,
    sparsity: float = 0.5,
    dense_q: bool = False,
    num_grouped_q: int = 1,
    bias: bool = True,
    num_objects: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[DeviceLikeType] = None,
    dense_q_len: int | None = None,
    broadcast_q: bool = False,
    dff: int | None = None,
    use_normal_distribution: bool = False,
    requires_grad: bool = True,
) -> dict[str, Any]:
    """Generate jagged tensor test data"""
    if device is None:
        device = torch.device("cuda:0")

    rand_fn = torch.randn if use_normal_distribution else torch.rand

    if num_objects is None:
        num_objects = generate_sparse_seq_len(
            size=B,
            max_seq_len=max_M,
            sparsity=sparsity,
            device=device,
        )
    num_objects_q = num_objects
    x_offsets = torch.cat(
        [torch.IntTensor([0]).to(device), num_objects.cumsum(dim=0)], dim=0
    )
    q_offsets = x_offsets

    D = D // H

    q_weights = rand_fn(
        int(num_objects.sum().item()),
        H,
        D,
        device=device,
        requires_grad=requires_grad,
        dtype=dtype,
    )

    k_weights = rand_fn(
        int(num_objects.sum().item()),
        H // num_grouped_q,
        D,
        device=device,
        requires_grad=requires_grad,
        dtype=dtype,
    )

    v_weights = rand_fn(
        int(num_objects.sum().item()),
        H // num_grouped_q,
        D,
        device=device,
        requires_grad=requires_grad,
        dtype=dtype,
    )

    output_offsets = None
    grad_o = None
    if dense_q:
        if dense_q_len is None:
            dense_q_len = max_M

        grad_o = torch.rand(B * dense_q_len, H, D, device=device, dtype=dtype) * 0.01
        if not broadcast_q:
            q_weights = torch.rand(B * dense_q_len, H, D, device=device, dtype=dtype)
            num_objects_q = torch.tensor(
                [dense_q_len] * B, device=device, dtype=torch.int32
            )
            q_offsets = torch.cat(
                [torch.IntTensor([0]).to(device), num_objects_q.cumsum(dim=0)], dim=0
            )
        else:
            q_weights = torch.rand(dense_q_len, H, D, device=device, dtype=dtype)
            num_objects_q = torch.tensor(
                [dense_q_len] * B, device=device, dtype=torch.int32
            )
            q_offsets = torch.tensor([0, dense_q_len], dtype=torch.int, device=device)
            output_offsets = (
                torch.arange(
                    B + 1,
                    dtype=torch.int,
                    device=device,
                )
                * dense_q_len
            )
    if dff:
        k_weights = rand_fn(
            B * dff,
            H,
            D,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        ).contiguous()
        v_weights = rand_fn(
            B * dff,
            H,
            D,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        ).contiguous()
        x_offsets = (
            torch.arange(
                B + 1,
                dtype=torch.int,
                device=device,
            )
            * dff
        )

    q_weights = q_weights.contiguous().detach()
    k_weights = k_weights.contiguous().detach()
    v_weights = v_weights.contiguous().detach()

    q_weights.requires_grad = requires_grad
    k_weights.requires_grad = requires_grad
    v_weights.requires_grad = requires_grad

    # Create attention lengths and offsets (simplified - no fbgemm dependency)
    attn_lengths = num_objects_q * num_objects
    # Simple cumsum instead of fbgemm.asynchronous_complete_cumsum
    attn_offsets = torch.cat([torch.IntTensor([0]).to(device), attn_lengths.cumsum(dim=0)], dim=0)

    invalid_attn_mask = (
        torch.tril(
            torch.ones(
                (max_M, max_M),
                dtype=torch.bool,
            ),
        )
        .fill_diagonal_(False)
        .to(device)
    ) * (-math.inf)

    invalid_attn_mask = invalid_attn_mask.to(dtype)
    bias_tensor = None
    if bias:
        bias_list = []
        for q_length, k_length in zip(num_objects_q, num_objects):
            bias_list.append(
                torch.randn(
                    q_length, k_length, device=device, dtype=torch.float32
                ).flatten()
            )
        bias_tensor = torch.cat(bias_list)

    if grad_o is None:
        if use_normal_distribution:
            grad_o = torch.randn_like(q_weights) * 0.01
        else:
            grad_o = torch.rand_like(q_weights) * 0.01

    return {
        "q_weights": q_weights,
        "k_weights": k_weights,
        "v_weights": v_weights,
        "num_objects": num_objects,
        "num_objects_k": num_objects,
        "num_objects_q": num_objects_q,
        "x_offsets": x_offsets,
        "q_offsets": q_offsets,
        "k_offsets": x_offsets,
        "output_offsets": output_offsets,
        "attn_lengths": attn_lengths,
        "attn_offsets": attn_offsets,
        "max_seq_len": max(
            max_M, dense_q_len if dense_q and dense_q_len else 0, dff if dff else 0
        ),
        "max_seq_len_q": dense_q_len if dense_q and dense_q_len else max_M,
        "max_seq_len_k": dff if dff else max_M,
        "dense_q_len": dense_q_len if dense_q else None,
        "bias": bias_tensor,
        "invalid_attn_mask": invalid_attn_mask.contiguous(),
        "do": grad_o,
        "mask_lower_triangle": False,
        "broadcast_q": broadcast_q,
        "dff": dff,
        "batch_size": B,
        "H": H,
        "D": D,
        "max_M": max_M,
        "sparsity": sparsity,
    }
