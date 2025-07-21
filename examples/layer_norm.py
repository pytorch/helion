from __future__ import annotations

from typing import Any

import torch

import helion
from helion._testing import run_example
import helion.language as hl


# TODO(PaulZhang12): Support autotuning, setting reduction_loops currently errors
@helion.kernel(
    static_shapes=True,
    config=helion.Config(
        block_sizes=[32],
        reduction_loops=[None],
        range_unroll_factors=[0],
        range_warp_specializes=[],
        range_num_stages=[0],
        range_multi_buffers=[None],
        range_flattens=[None],
        num_warps=4,
        num_stages=3,
        indexing="pointer",
        pid_type="flat",
    ),
)
def layer_norm_fwd(
    x: torch.Tensor,
    dims: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    m, n = x.size()
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {m}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {m}"
    assert len(dims) == 1 and dims[0] == n, f"dim mismatch {dims} != {n}"
    out = torch.empty([m, n], dtype=torch.float16, device=x.device)

    for tile_m in hl.tile(m):
        acc = x[tile_m, :].to(
            torch.float32
        )  # TODO (PaulZhang12): Eliminate this cast, currently necessary

        var, mean = torch.var_mean(acc, dim=-1, keepdim=True, correction=0)

        normalized = (acc - mean) * torch.rsqrt(var + eps)
        acc = normalized * (weight[:].to(torch.float32)) + (bias[:].to(torch.float32))

        out[tile_m, :] = acc
    return out


def layer_norm_torch_callable(
    dims: list[int],
) -> Any:  # noqa: ANN401
    return lambda x, weight, bias, eps: torch.nn.functional.layer_norm(
        x, dims, weight, bias, eps
    )


def main() -> None:
    batch_size = 32
    dim = 64
    device = "cuda"

    x = torch.randn([batch_size, dim], device=device, dtype=torch.float16)
    weight = torch.randn([dim], device=device, dtype=torch.float16)
    bias = torch.randn([dim], device=device, dtype=torch.float16)
    eps = 1e-4

    run_example(
        layer_norm_fwd,
        torch.nn.functional.layer_norm,
        (x, [dim], weight, bias, eps),
        kernel_name="helion",
        baseline_name="torch",
        rtol=1e-4,
        atol=1e-4,
    )


if __name__ == "__main__":
    main()
