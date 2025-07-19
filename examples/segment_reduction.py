# Code based on https://github.com/pytorch-labs/helion/issues/237
from __future__ import annotations

import torch
import triton
import triton.language as tl

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


def combine_fn_helion(
    left_values: torch.Tensor,
    left_indices: torch.Tensor,
    right_values: torch.Tensor,
    right_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_values = torch.where(
        left_indices == right_indices, left_values + right_values, right_values
    )
    return combined_values, right_indices


@helion.kernel()
def segmented_reduction_helion(
    indices: torch.Tensor, input_data: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    num_elements, num_features = input_data.shape
    output = torch.zeros(
        (num_nodes, num_features), dtype=input_data.dtype, device=input_data.device
    )
    for tile_e, tile_f in hl.tile([num_elements, num_features]):
        vals = input_data[tile_e, tile_f]
        idxs = indices[tile_e]
        tile_e_indices = hl.tile_index(tile_e)
        idxs_next = hl.load(
            indices, [tile_e_indices + 1], extra_mask=tile_e_indices < num_elements - 1
        )
        tuple_in = (vals, idxs.float().unsqueeze(1).expand_as(vals))
        out_vals, _ = hl.associative_scan(combine_fn_helion, tuple_in, dim=0)
        block_size = hl.tile_block_size(tile_e)
        mask = (idxs != idxs_next) | (tile_e_indices % block_size == block_size - 1)
        segment_vals = torch.where(mask.unsqueeze(1), out_vals, 0.0)
        hl.atomic_add(output, [idxs, tile_f], segment_vals)
    return output


@triton.jit
def combine_fn_triton(
    left_values: tl.tensor,
    left_indices: tl.tensor,
    right_values: tl.tensor,
    right_indices: tl.tensor,
) -> tuple[tl.tensor, tl.tensor]:
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
    E, C = input_data.shape
    output = torch.zeros(
        (num_nodes, C), dtype=input_data.dtype, device=input_data.device
    )

    def grid(META: dict[str, int]) -> tuple[int, ...]:
        return (triton.cdiv(E, META["BLOCK_SIZE"]) * C,)

    _segmented_reduction_triton[grid](indices, input_data, output, E, C)
    return output


def segmented_reduction_pytorch(
    indices: torch.Tensor, input_data: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    # Run PyTorch reference (scatter_add equivalent)
    num_features = input_data.size(1)
    pytorch_output = torch.zeros(
        num_nodes, num_features, device=input_data.device, dtype=input_data.dtype
    )
    pytorch_output.scatter_add_(
        0, indices.unsqueeze(1).expand(-1, num_features), input_data
    )
    return pytorch_output


def main() -> None:
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
