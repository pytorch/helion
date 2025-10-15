from __future__ import annotations

import torch

import matmul_subtile
from helion.runtime import config as config_mod


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This repro expects CUDA to be available for tensor descriptor lowering.")

    device = torch.device("cuda")
    x = torch.randn((64, 64), device=device, dtype=torch.float16)
    y = torch.randn((64, 64), device=device, dtype=torch.float16)

    config = config_mod.Config(
        indexing="tensor_descriptor",
        block_sizes=[64, 128, 32],
        num_warps=4,
        num_stages=2,
        l2_groupings=[1],
        load_eviction_policies=["", ""],
        loop_orders=[[0, 1]],
        range_flattens=[None, None],
        range_multi_buffers=[None, None],
        range_num_stages=[0, 0],
        range_unroll_factors=[0, 0],
        range_warp_specializes=[None, None],
        pid_type="flat",
    )

    kernel = matmul_subtile.matmul.bind((x, y))
    triton_ir = kernel.to_triton_code(config)
    print(triton_ir)


if __name__ == "__main__":
    main()
