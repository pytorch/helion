"""
Segmented Reduction Example
===========================

This example demonstrates how to implement a segmented reduction operation using Helion,
comparing it with Triton and PyTorch implementations.
Code based on https://github.com/pytorch/helion/issues/237

On **Ascend NPU**, lowered ``hl.atomic_add`` / Triton ``atomic_add`` is unreliable (see
``scripts/triton_ascend_atomic_scan_minrepro.py``). :func:`segmented_reduction_helion`
therefore uses a **two-phase** Helion path (partials via scan + disjoint stores, merge
via ``scatter_add_``) so the Helion pipeline and autotune still run. The Triton wrapper
still dispatches to PyTorch on NPU because the raw Triton kernel uses atomics. CUDA
behavior is unchanged.

On **NPU**, :func:`main` checks full end-to-end Helion vs PyTorch once, then runs
:func:`run_example` on **stage-1 only** (the partials ``(2, N)`` pack): Helion codegen
vs a batched PyTorch reference, so benchmark numbers are not dominated by merge /
``scatter_add_``.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# Edge-aligned block for the NPU two-phase Helion path (matches common Triton choices).
_HELION_NPU_SEG_BLOCK_E = 32


# %%
# Helion Implementation
# ---------------------


# %%
def combine_fn_helion(
    left_values: torch.Tensor,
    left_indices: torch.Tensor,
    right_values: torch.Tensor,
    right_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combine function for associative scan in Helion implementation.

    Adds values when indices match (same segment), otherwise takes the right value.

    Args:
        left_values: Values from the left side of the scan
        left_indices: Indices from the left side of the scan
        right_values: Values from the right side of the scan
        right_indices: Indices from the right side of the scan

    Returns:
        Tuple of (combined_values, right_indices)
    """
    combined_values = torch.where(
        left_indices == right_indices, left_values + right_values, right_values
    )
    return combined_values, right_indices


@helion.kernel(static_shapes=False, autotune_ignore_errors=True, autotune_effort="quick")
def _helion_segmented_reduction_npu_partials(
    indices: torch.Tensor,
    input_data: torch.Tensor,
    num_nodes: int,
    num_elements: int,
    num_features: int,
) -> torch.Tensor:
    """
    Ascend-safe partials: one program per (edge-block, feature); writes unique slots in
    flat buffers (no ``hl.atomic_add``). Returns ``stack([vals, idxs_as_float64], dim=0)``
    of shape ``(2, num_progs * block_e)``; indices are exact in float64 for typical
    ``num_nodes`` (``< 2**53``).
    """
    block_e = _HELION_NPU_SEG_BLOCK_E
    num_progs = (num_elements + block_e - 1) // block_e * num_features
    idx_part = torch.full(
        (num_progs * block_e,),
        num_nodes,
        dtype=torch.int64,
        device=input_data.device,
    )
    val_part = torch.zeros(
        (num_progs * block_e,),
        dtype=input_data.dtype,
        device=input_data.device,
    )
    for pid in hl.grid(num_progs):
        blk = pid // num_features
        feat = pid % num_features
        base = blk * block_e
        for tile_j in hl.tile(block_e, block_size=block_e):
            offs = base + tile_j.index
            mask = offs < num_elements
            vals = input_data[offs, feat]
            idxs = indices[offs]
            idxs_next = hl.load(
                indices,
                [offs + 1],
                extra_mask=mask & (offs < num_elements - 1),
            )
            vals2 = vals.unsqueeze(1)
            idxs_scan = idxs.float().unsqueeze(1).expand_as(vals2)
            out_vals2, _ = hl.associative_scan(combine_fn_helion, (vals2, idxs_scan), dim=0)
            out_vals = out_vals2.squeeze(1)
            mask_seg = (idxs != idxs_next) | (offs % block_e == block_e - 1)
            seg_v = torch.where(
                mask_seg & mask, out_vals, torch.zeros_like(out_vals)
            )
            seg_idx = torch.where(
                mask_seg & mask,
                idxs,
                torch.full_like(idxs, num_nodes, dtype=idxs.dtype),
            )
            flat = pid * block_e + tile_j.index
            idx_part[flat] = seg_idx
            val_part[flat] = seg_v
    return torch.stack((val_part, idx_part.to(torch.float64)))


def _helion_npu_stage1_pack(
    indices: torch.Tensor, input_data: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Host wrapper: Helion-generated stage-1 partials (``(2, num_slots)`` tensor)."""
    num_elements, num_features = input_data.shape
    return _helion_segmented_reduction_npu_partials(
        indices, input_data, num_nodes, num_elements, num_features
    )


def _pytorch_npu_stage1_pack_reference(
    indices: torch.Tensor, input_data: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    Batched PyTorch reference for stage-1 partials (same layout as Helion pack).

    Matches one ``hl.grid`` program per (edge-block, feature) and an inclusive segmented
    prefix along the edge dimension; used as the **first** baseline in NPU
    :func:`run_example` so timings compare codegen vs Torch on the same subproblem only.
    """
    block_e = _HELION_NPU_SEG_BLOCK_E
    num_elements, num_features = input_data.shape
    device = input_data.device
    dtype = input_data.dtype
    num_progs = (num_elements + block_e - 1) // block_e * num_features
    pid = torch.arange(num_progs, device=device, dtype=torch.int64)
    blk = pid // num_features
    feat = pid % num_features
    ar = torch.arange(block_e, device=device, dtype=torch.int64)
    base = blk * block_e
    offs = base[:, None] + ar[None, :]
    mask = offs < num_elements
    feat_bc = feat[:, None].expand_as(offs)
    zv = torch.zeros((), dtype=dtype, device=device)
    zn = torch.tensor(num_nodes, dtype=torch.int64, device=device)
    vals = torch.where(mask, input_data[offs, feat_bc], zv)
    idxs = torch.where(mask, indices[offs], zn)
    offs_next = offs + 1
    valid_next = mask & (offs < num_elements - 1)
    idxs_next = torch.zeros_like(idxs)
    idxs_next[valid_next] = indices[offs_next[valid_next]]
    same = idxs[:, 1:] == idxs[:, :-1]
    out = torch.empty_like(vals)
    out[:, 0] = vals[:, 0]
    for i in range(1, block_e):
        out[:, i] = torch.where(
            same[:, i - 1], out[:, i - 1] + vals[:, i], vals[:, i]
        )
    mask_seg = (idxs != idxs_next) | ((offs % block_e) == (block_e - 1))
    seg_v = torch.where(mask_seg & mask, out, torch.zeros_like(out))
    seg_idx = torch.where(mask_seg & mask, idxs, zn)
    flat = pid[:, None] * block_e + ar[None, :]
    val_out = torch.zeros(num_progs * block_e, dtype=dtype, device=device)
    idx_out = torch.full((num_progs * block_e,), num_nodes, dtype=torch.int64, device=device)
    val_out[flat.reshape(-1)] = seg_v.reshape(-1)
    idx_out[flat.reshape(-1)] = seg_idx.reshape(-1)
    return torch.stack((val_out, idx_out.to(torch.float64)))


def _segmented_reduction_helion_npu_two_phase(
    indices: torch.Tensor, input_data: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Merge scan-produced partials with ``scatter_add_`` (host op; no device atomics)."""
    num_elements, num_features = input_data.shape
    pack = _helion_segmented_reduction_npu_partials(
        indices, input_data, num_nodes, num_elements, num_features
    )
    val_flat = pack[0]
    idx_flat = pack[1].to(torch.int64)
    block_e = _HELION_NPU_SEG_BLOCK_E
    num_slots = val_flat.numel()
    num_progs = num_slots // block_e
    device = input_data.device
    dtype = input_data.dtype
    feats = (torch.arange(num_slots, device=device) // block_e) % num_features
    lin = idx_flat * num_features + feats
    out_flat = torch.zeros(
        (num_nodes + 1) * num_features,
        device=device,
        dtype=dtype,
    )
    out_flat.scatter_add_(0, lin, val_flat)
    return out_flat.view(num_nodes + 1, num_features)[:num_nodes]


@helion.kernel(static_shapes=False, autotune_ignore_errors=True, autotune_effort="quick")
def _segmented_reduction_helion_atomic(
    indices: torch.Tensor, input_data: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """CUDA (etc.): tiled scan + ``hl.atomic_add`` into the output tensor."""
    num_elements, num_features = input_data.shape
    output = torch.zeros(
        (num_nodes, num_features), dtype=input_data.dtype, device=input_data.device
    )
    for tile_e, tile_f in hl.tile([num_elements, num_features]):
        vals = input_data[tile_e, tile_f]
        idxs = indices[tile_e]
        idxs_next = hl.load(
            indices, [tile_e.index + 1], extra_mask=tile_e.index < num_elements - 1
        )
        tuple_in = (vals, idxs.float().unsqueeze(1).expand_as(vals))
        out_vals, _ = hl.associative_scan(combine_fn_helion, tuple_in, dim=0)
        mask = (idxs != idxs_next) | (
            tile_e.index % tile_e.block_size == tile_e.block_size - 1
        )
        segment_vals = torch.where(mask.unsqueeze(1), out_vals, 0.0)
        hl.atomic_add(output, [idxs, tile_f], segment_vals)
    return output


def segmented_reduction_helion(
    indices: torch.Tensor, input_data: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    Performs segmented reduction using Helion.

    Reduces input data by summing values with the same index.

    This is a **host** entry point: NPU uses a two-phase Helion kernel + ``scatter_add_``;
    other devices use :func:`_segmented_reduction_helion_atomic`.  (Do not nest calling
    another ``@helion.kernel`` from inside a kernel body — that re-enters tracing and
    breaks specialization with ``SymInt``.)

    Args:
        indices: Tensor of segment indices for each element
        input_data: Input tensor of shape [num_elements, num_features]
        num_nodes: Number of output nodes/segments

    Returns:
        Output tensor of shape [num_nodes, num_features] with reduced values
    """
    if indices.device.type == "npu":
        return _segmented_reduction_helion_npu_two_phase(
            indices, input_data, num_nodes
        )
    return _segmented_reduction_helion_atomic(indices, input_data, num_nodes)


# %%
# Triton Implementation
# ---------------------


# %%
@triton.jit
def combine_fn_triton(
    left_values: tl.tensor,
    left_indices: tl.tensor,
    right_values: tl.tensor,
    right_indices: tl.tensor,
) -> tuple[tl.tensor, tl.tensor]:
    """
    Combine function for associative scan in Triton implementation.

    Adds values when indices match (same segment), otherwise takes the right value.

    Args:
        left_values: Values from the left side of the scan
        left_indices: Indices from the left side of the scan
        right_values: Values from the right side of the scan
        right_indices: Indices from the right side of the scan

    Returns:
        Tuple of (combined_values, combined_indices)
    """
    same_segment = left_indices == right_indices
    combined_values = tl.where(same_segment, left_values + right_values, right_values)
    combined_indices = right_indices
    return combined_values, combined_indices


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE": bs},
        )
        for bs in [8, 16, 32, 64, 128]
    ],
    key=["C"],
    restore_value=["out_ptr"],
)
@triton.jit
def _segmented_reduction_triton(
    index: tl.tensor,  # the input index tensor
    in_ptr: tl.tensor,  # the input tensor
    out_ptr: tl.tensor,  # the output value tensor
    E: tl.constexpr,  # Number of elements in the input tensor (1d)
    C: tl.constexpr,  # Number of features in the input tensor (2d)
    BLOCK_SIZE: tl.constexpr,  # Block size for the scan
) -> None:
    """
    Triton kernel for segmented reduction.

    Uses associative scan to efficiently perform segmented reduction.

    Args:
        index: Input index tensor
        in_ptr: Input data tensor
        out_ptr: Output tensor
        E: Number of elements in the input tensor
        C: Number of features in the input tensor
        BLOCK_SIZE: Block size for the scan
    """
    # Triton version adapted from
    # https://github.com/fishmingyu/GeoT/blob/main/geot/triton/seg_reduction.py
    pid = tl.program_id(axis=0)
    offset_pid = pid // C
    feature_id = pid % C
    offsets = offset_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < E

    # Load input data
    vals = tl.load(in_ptr + offsets * C + feature_id, mask=mask)
    idxs = tl.load(index + offsets, mask=mask)
    idxs_next = tl.load(index + offsets + 1, offsets < E - 1)

    # Perform an inclusive scan using tl.associative_scan
    result_values, _ = tl.associative_scan(
        (
            vals,
            idxs,
        ),
        axis=0,
        combine_fn=combine_fn_triton,
    )
    # if offset % BLOCK_SIZE == -1, it means the last element of the segment
    segment_start = (idxs != idxs_next) | (offsets % BLOCK_SIZE == BLOCK_SIZE - 1)
    tl.atomic_add(out_ptr + idxs * C + feature_id, result_values, mask & segment_start)


def segmented_reduction_triton(
    indices: torch.Tensor, input_data: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    Performs segmented reduction using Triton.

    Wrapper function for the Triton kernel implementation.

    Args:
        indices: Tensor of segment indices for each element
        input_data: Input tensor of shape [num_elements, num_features]
        num_nodes: Number of output nodes/segments

    Returns:
        Output tensor of shape [num_nodes, num_features] with reduced values
    """
    if indices.device.type == "npu":
        return segmented_reduction_pytorch(indices, input_data, num_nodes)

    E, C = input_data.shape
    output = torch.zeros(
        (num_nodes, C), dtype=input_data.dtype, device=input_data.device
    )

    def grid(META: dict[str, int]) -> tuple[int, ...]:
        # Cast to int to satisfy type checker; Triton may return constexpr
        return (int(triton.cdiv(E, META["BLOCK_SIZE"]) * C),)

    _segmented_reduction_triton[grid](indices, input_data, output, E, C)
    return output


# %%
# PyTorch Reference Implementation
# --------------------------------


# %%
def segmented_reduction_pytorch(
    indices: torch.Tensor, input_data: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    Performs segmented reduction using PyTorch's scatter_add.

    Reference implementation using PyTorch's native operations.

    Args:
        indices: Tensor of segment indices for each element
        input_data: Input tensor of shape [num_elements, num_features]
        num_nodes: Number of output nodes/segments

    Returns:
        Output tensor of shape [num_nodes, num_features] with reduced values
    """
    # Run PyTorch reference (scatter_add equivalent)
    num_features = input_data.size(1)
    pytorch_output = torch.zeros(
        num_nodes, num_features, device=input_data.device, dtype=input_data.dtype
    )
    pytorch_output.scatter_add_(
        0, indices.unsqueeze(1).expand(-1, num_features), input_data
    )
    return pytorch_output


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the segmented reduction implementations.

    Creates random data with 100 nodes, 2000 edges, and 128 features,
    then compares the Helion implementation against Triton and PyTorch.
    """
    num_nodes = 16
    num_edges = 512
    num_features = 16

    dtype = torch.float32

    # Create sorted indices for segmented reduction
    indices = torch.randint(0, num_nodes, (num_edges,), device=DEVICE).sort()[0]
    input_data = torch.randn(num_edges, num_features, device=DEVICE, dtype=dtype)

    args = (indices, input_data, num_nodes)
    if DEVICE.type == "npu":
        torch.testing.assert_close(
            segmented_reduction_helion(*args),
            segmented_reduction_pytorch(*args),
            rtol=1e-2,
            atol=1e-1,
        )
        print(
            "NPU: full segmented_reduction_helion vs PyTorch scatter_add ok.",
            file=sys.stderr,
        )
        run_example(
            {"helion_stage1": _helion_npu_stage1_pack},
            {"pytorch_stage1": _pytorch_npu_stage1_pack_reference},
            args,
        )
    else:
        run_example(
            segmented_reduction_helion,
            {
                "triton": segmented_reduction_triton,
                "pytorch": segmented_reduction_pytorch,
            },
            args,
        )


if __name__ == "__main__":
    main()
