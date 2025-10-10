"""
Segmented Reduction Example
===========================

This example demonstrates how to implement a segmented reduction operation using Helion,
comparing it with Triton and PyTorch implementations.
Code based on https://github.com/pytorch/helion/issues/237
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch
import triton
import triton.language as tl

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

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


@helion.kernel()
def segmented_reduction_helion(
    indices: torch.Tensor, input_data: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """
    Performs segmented reduction using Helion.

    Reduces input data by summing values with the same index.

    Args:
        indices: Tensor of segment indices for each element
        input_data: Input tensor of shape [num_elements, num_features]
        num_nodes: Number of output nodes/segments

    Returns:
        Output tensor of shape [num_nodes, num_features] with reduced values
    """
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
    num_nodes = 100
    num_edges = 2000
    num_features = 128

    dtype = torch.float32

    # Create sorted indices for segmented reduction
    indices = torch.randint(0, num_nodes, (num_edges,), device=DEVICE).sort()[0]
    input_data = torch.randn(num_edges, num_features, device=DEVICE, dtype=dtype)

    run_example(
        segmented_reduction_helion,
        {
            "triton": segmented_reduction_triton,
            "pytorch": segmented_reduction_pytorch,
        },
        (indices, input_data, num_nodes),
    )


if __name__ == "__main__":
    main()
