"""Kernels and configs the tests of MPP's register operands share
(``mpp_registers``): the GPU module (``test/test_metal_mpp_registers_mps.py``)
runs the kernels and the CPU-only one (``test/test_metal_mpp_registers.py``)
reads their code and MSL; the affine weights' kernel is
``metal_fragment.affine_mm``."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from unittest import mock

import torch

import helion
from helion._compiler.metal import mpp_registers
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .._compiler.metal.mpp_matmul import Graphs
    from .._compiler.metal.mpp_matmul import MppGroup
    from .._compiler.metal.mpp_matmul import MppPlan

F32 = torch.float32


@contextlib.contextmanager
def unread_x() -> Iterator[None]:
    """Leave every group's left operand unmarked for registers
    (``check_register_operands``), as for a widened x that loads a tensor the
    kernel writes."""
    check = mpp_registers.RegisterLowering.check_group

    def check_group(group: MppGroup, plan: MppPlan, graphs: Graphs) -> None:
        check(group, plan, graphs)
        group.lhs.registers = False

    with mock.patch.object(
        mpp_registers.RegisterLowering, "check_group", staticmethod(check_group)
    ):
        yield


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
def inner_tiled_weight_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x @ (y / 2) with M and N tiled in a loop of a one-program grid, x a
    half view and the weight computed: the thread axes codegen sees at the
    accumulator aren't the launch's."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for _g in hl.grid(1):
        for tm, tn in hl.tile([m, n]):
            acc = hl.zeros([tm, tn], dtype=F32)
            for tk in hl.tile(k):
                acc = torch.addmm(acc, x[tm, tk], (y[tk, tn] * 0.5).to(x.dtype))
            out[tm, tn] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def shifted_x_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x[8:] @ (y / 2) for x of 8 rows more than the output: x's view starts
    8 rows past its tile's offset."""
    m = x.size(0) - 8
    k = x.size(1)
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            xs = hl.load(x, [tile_m.index + 8, tile_k.index])
            acc = torch.addmm(acc, xs, (y[tile_k, tile_n] * 0.5).to(x.dtype))
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def offset_k_mm(x: torch.Tensor, y: torch.Tensor, k0: int) -> torch.Tensor:
    """x[:, k0 : k0 + 128] @ (y[k0 : k0 + 128] / 2): a K loop whose begin is
    a run-time value and whose extent is 128."""
    m = x.size(0)
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k0, k0 + 128):
            acc = torch.addmm(
                acc, x[tile_m, tile_k], (y[tile_k, tile_n] * 0.5).to(x.dtype)
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def long_k_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x @ (y / 2) over y's K, x's K dim shorter than the loop: x's view
    reads past its tensor's K, where the staged op's view zero-fills."""
    m = x.size(0)
    k, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(
                acc, x[tile_m, tile_k], (y[tile_k, tile_n] * 0.5).to(x.dtype)
            )
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def staged_apart_mm(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(x @ (y / 2)) * s and x @ (z / 4) stored apart from one K loop, s a
    row vector summed over v's rows: the first's staged vector keeps it on
    the staged op."""
    m, k = x.size()
    _, n = y.size()
    out_a = torch.empty([m, n], dtype=F32, device=x.device)
    out_b = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        s = hl.zeros([tile_n], dtype=F32)
        for r in hl.grid(v.size(0)):
            s = s + v[r, tile_n]
        a = hl.zeros([tile_m, tile_n], dtype=F32)
        b = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            xs = x[tile_m, tile_k]
            a = torch.addmm(a, xs, (y[tile_k, tile_n] * 0.5).to(x.dtype))
            b = torch.addmm(b, xs, (z[tile_k, tile_n] * 0.25).to(x.dtype))
        out_a[tile_m, tile_n] = a * s[None, :]
        out_b[tile_m, tile_n] = b
    return out_a, out_b


@helion.kernel(backend="metal", static_shapes=True)
def counted_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x @ (y / 2) whose K loop also counts its tiles, a value it carries
    beside the accumulator, which SIMD groups splitting the loop would each
    count only their share of."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        count = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k].to(F32), y[tile_k, tile_n] * 0.5)
            count = count + 1
        out[tile_m, tile_n] = acc + count
    return out


@helion.kernel(backend="metal", static_shapes=True)
def copying_mm(x: torch.Tensor, y: torch.Tensor, copy: torch.Tensor) -> torch.Tensor:
    """x @ (y / 2) whose K loop also copies x's tiles, a write that SIMD
    groups splitting the loop would each make only for their share."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            a = x[tile_m, tile_k]
            copy[tile_m, tile_k] = a
            acc = torch.addmm(acc, a.to(F32), y[tile_k, tile_n] * 0.5)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def transposed_out_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(x @ (y / 2)).T: the store reads the accumulator in the Scalar
    layout (materialized through threadgroup memory), which SIMD groups
    splitting the K loop would each write."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([n, m], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k].to(F32), y[tile_k, tile_n] * 0.5)
        out[tile_n, tile_m] = acc.T
    return out


@helion.kernel(backend="metal", static_shapes=True)
def row_scaled_mm(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(x @ (y / 2)) * s for a row vector s summed over v's rows in a loop
    before the K loop: the accumulator's element loop reads s staged
    through threadgroup memory, which SIMD groups splitting the K loop
    would each write."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        s = hl.zeros([tile_n], dtype=F32)
        for r in hl.grid(v.size(0)):
            s = s + v[r, tile_n]
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k].to(F32), y[tile_k, tile_n] * 0.5)
        out[tile_m, tile_n] = acc * s[None, :]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def filled_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """0.5 + x @ (y / 2): an accumulator filled with a constant other than
    0, which under a K split only the first SIMD group starts from."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.full([tile_m, tile_n], 0.5, dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k].to(F32), y[tile_k, tile_n] * 0.5)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def two_acc_mm(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """x @ (y / 2) and x @ (z / 4): two accumulators of one K loop, each
    stored as it is, which a K split's combine adds up together."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    out2 = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        a = hl.zeros([tile_m, tile_n], dtype=F32)
        b = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            xs = x[tile_m, tile_k].to(F32)
            a = torch.addmm(a, xs, y[tile_k, tile_n] * 0.5)
            b = torch.addmm(b, xs, z[tile_k, tile_n] * 0.25)
        out[tile_m, tile_n] = a
        out2[tile_m, tile_n] = b
    return out, out2


@helion.kernel(backend="metal", static_shapes=True)
def rowsum_mm(x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(x @ (y / 2)) * s for s the sums of v's rows: a reduction, whose
    threads come from the num_warps budget, beside the matmul."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k].to(F32), y[tile_k, tile_n] * 0.5)
        s = v[tile_m, :].sum(-1)
        out[tile_m, tile_n] = acc * s[:, None]
    return out


@helion.kernel(backend="metal", static_shapes=True)
def free_cols_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x @ (y / 2) over a free hl.arange of the columns, whose threads come
    from the num_warps budget."""
    m, k = x.size()
    _, n = y.size()
    bn = hl.register_block_size(n)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m in hl.tile(m):
        cols = hl.arange(bn)
        acc = hl.zeros([tile_m, bn], dtype=F32)
        for tile_k in hl.tile(k):
            w = hl.load(y, [tile_k.index, cols]) * 0.5
            acc = torch.addmm(acc, x[tile_m, tile_k].to(F32), w)
        hl.store(out, [tile_m.index, cols], acc)
    return out


@helion.kernel(backend="metal", static_shapes=True)
def mm_and_double(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """x @ (y / 2) beside an element-wise root, 2 z: a kernel of two roots,
    whose launch normalization doesn't model."""
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    out2 = torch.empty_like(z)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k].to(F32), y[tile_k, tile_n] * 0.5)
        out[tile_m, tile_n] = acc
    for tile in hl.tile(z.size(0)):
        out2[tile] = z[tile] * 2
    return out, out2


@helion.kernel(backend="metal", static_shapes=True)
def dequant_forms_mm(
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    form: hl.constexpr,
) -> torch.Tensor:
    """x @ W.T for 4-bit codes in int16 words (``metal_fragment.affine_mm``'s
    format, groups of 32) dequantized five ways (``form``): ``promoted``,
    its fp16 scale and bias promoted by the float32 codes; ``half``, all of
    it in fp16, the weight already half; ``bfloat``, all of it in bf16
    beside a bf16 x; ``bf16_product``, the product
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
            elif form == "bfloat":
                bf = torch.bfloat16
                deq = q.to(bf) * scales[tile_n, grp].to(bf) + biases[tile_n, grp].to(bf)
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
    x: torch.Tensor,
    w: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    weight: torch.dtype = torch.float64,
) -> torch.Tensor:
    """:func:`dequant_forms_mm` in float64 on the CPU, the weight dequantized
    in ``weight``: its ``promoted`` form's float64, its ``half`` form's
    ``torch.float16``."""
    x, w, scales, biases = (t.cpu() for t in (x, w, scales, biases))
    bit = torch.arange(x.size(1)) * 4
    codes = ((w.to(torch.int64) & 0xFFFF)[:, bit // 16] >> (bit % 16)) & 15
    grp = torch.arange(x.size(1)) // 32
    deq = codes.to(weight) * scales.to(weight)[:, grp] + biases.to(weight)[:, grp]
    return x.double() @ deq.double().T


@helion.kernel(backend="metal", static_shapes=True)
def bounded_k_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x @ (y / 2) over each column tile's own K positions: a K loop the N
    block bounds, whose block normalization clamps to the N block's."""
    m, _ = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(tile_n.begin, tile_n.end):
            acc = torch.addmm(acc, x[tile_m, tile_k].to(F32), y[tile_k, tile_n] * 0.5)
        out[tile_m, tile_n] = acc
    return out
