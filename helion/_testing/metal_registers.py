"""Kernels and configs the tests of MPP's register operands share
(``mpp_registers``): the GPU module (``test/test_metal_mpp_registers_mps.py``)
runs the kernels and the CPU-only one (``test/test_metal_mpp_registers.py``)
reads their code and MSL; the affine weights' kernel is
``metal_fragment.affine_mm``."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

F32 = torch.float32


def register_config(
    block_sizes: list[int], num_threads: list[int] | None = None, **knobs: object
) -> helion.Config:
    """An MPP config (``metal_matmul`` left at its default) whose launch is
    one 32-thread SIMD group, by default 4 x 8 threads over M and N."""
    return helion.Config.from_dict(
        {"block_sizes": block_sizes, "num_threads": num_threads or [4, 8, 1], **knobs}
    )


@helion.kernel(backend="metal", static_shapes=True)
def inner_tiled_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(2 x) @ y with M and N tiled in a loop of a one-program grid, so the
    launch's thread axes codegen sees differ inside the K loop."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=F32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk] * 2, y[tk, tn])
            out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def dequant_forms_mm(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    form: hl.constexpr,
) -> torch.Tensor:
    """x @ W.T for 4-bit codes in int16 words (``metal_fragment.affine_mm``'s
    format, groups of 32) dequantized four ways (``form``): ``promoted``,
    its fp16 scale and bias promoted by the float32 codes; ``half``, all of
    it in fp16, the weight already half; ``bf16_product``, the product
    rounded to bf16; ``negative_group``, the group index ``k // -32 + 9``,
    which isn't constant over an aligned tile."""
    m, k = x.size()
    n, _ = w.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            bit = tile_k.index * 4
            q = (w[tile_n, bit // 16].to(torch.int32) >> (bit % 16)[None, :]) & 15
            if form == "negative_group":
                grp = tile_k.index // -32 + 9
            else:
                grp = tile_k.index // 32
            if form == "half":
                deq = q.to(torch.float16) * scales[tile_n, grp] + biases[tile_n, grp]
                acc = torch.addmm(acc, x[tile_m, tile_k], deq.T)
            else:
                if form == "bf16_product":
                    deq = (
                        q.to(torch.bfloat16) * scales[tile_n, grp].to(torch.bfloat16)
                    ).to(F32) + biases[tile_n, grp].to(F32)
                else:
                    deq = q.to(F32) * scales[tile_n, grp] + biases[tile_n, grp]
                acc = torch.addmm(acc, x[tile_m, tile_k].to(F32), deq.T)
        out[tile_m, tile_n] = acc
    return out


def dequant_reference(
    x: torch.Tensor, w: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor
) -> torch.Tensor:
    """:func:`dequant_forms_mm`'s ``promoted`` form in float64 on the CPU."""
    x, w, scales, biases = (t.cpu() for t in (x, w, scales, biases))
    bit = torch.arange(x.size(1)) * 4
    codes = ((w.to(torch.int64) & 0xFFFF)[:, bit // 16] >> (bit % 16)) & 15
    grp = torch.arange(x.size(1)) // 32
    deq = codes.double() * scales.double()[:, grp] + biases.double()[:, grp]
    return x.double() @ deq.T
