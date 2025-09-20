"""Minimal Triton repro for the Hopper TMA misaligned-address failure."""

import torch
import triton
import triton.language as tl


def _alloc(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.uint8)


triton.set_allocator(_alloc)


@triton.jit
def _tma_misaligned(x_ptr, out_ptr, num_sm: tl.constexpr, block_cols: tl.constexpr):
    desc = tl.make_tensor_descriptor(x_ptr, [8192, 4096], [4096, 1], [1, block_cols])
    out_desc = tl.make_tensor_descriptor(out_ptr, [8192, 4096], [4096, 1], [1, block_cols])
    total_rows = 8192
    for row in tl.range(tl.program_id(0), total_rows, num_sm, loop_unroll_factor=1, num_stages=1):
        for col in tl.range(0, 4096, block_cols, loop_unroll_factor=1, num_stages=4, flatten=False):
            tile = desc.load([row, col])
            out_desc.store([row, col], tile)


def main() -> None:
    x = torch.randn((8192, 4096), device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    num_sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    block_cols = 4  # 16-byte tile width triggers Hopper TMA misalignment
    _tma_misaligned[(num_sm,)](x, out, num_sm, block_cols, num_warps=16, num_stages=8)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
