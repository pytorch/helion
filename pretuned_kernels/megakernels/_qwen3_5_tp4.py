"""Constants and host helpers shared by the Qwen3.5 TP4 megakernels
(``qwen3_5_moe_tp4/``, ``qwen3_5_gdn_tp4/``, ``qwen3_5_attn_tp4/``): the FP8
range and the RMSNorm epsilon, synthetic weights, SGLang's static FP8 input
quant and the FP8 linear references, the symmetric partials, the all-reduce +
residual + RMSNorm reference, per-stage error rows, the multi-layer CUDA-graph
timing and the modules' four-rank ``main()``."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import Any

from pretuned_kernels.megakernels._distributed import initialize
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.distributed_c10d import GroupName

if TYPE_CHECKING:
    from collections.abc import Callable

WORLD_SIZE = 4
SIGNAL_PAD_BYTES = 32 * 1024
FP8_MAX = 448.0
NORM_EPS = 1e-6


def randn(
    g: torch.Generator,
    *shape: int,
    scale: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """``scale`` times a standard normal draw from ``g``, as ``dtype``."""
    return (torch.randn(*shape, generator=g, device=device) * scale).to(dtype)


def fp8_weight(
    g: torch.Generator, rows: int, cols: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """A random [rows, cols] weight as ModelOpt's per-tensor FP8 E4M3 and its
    scale (amax / 448)."""
    w = torch.randn(rows, cols, generator=g, device=device) * 0.02
    scale = (w.abs().amax() / FP8_MAX).reshape(())
    return (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), scale


def sglang_quant(y: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """SGLang's ``static_quant_fp8`` (apply_fp8_linear's static input quant)."""
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    q, _ = static_quant_fp8(y.contiguous(), scale.reshape(1), repeat_scale=False)
    return q


def fp8_linear(
    x8: torch.Tensor,
    stored: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """A ModelOpt per-tensor FP8 linear on its FP8 input ``x8``: FP32 accumulate,
    * input_scale * weight_scale, BF16. ``stored`` is SGLang's ``layer.weight``
    ([K, N], the transposed FP8 weight)."""
    acc = x8.float() @ stored.float()
    return (acc * (input_scale * weight_scale)).to(torch.bfloat16)


_SYMMETRIC_BUFFERS: list[torch.Tensor] = []


def symmetric_buffer(
    shape: tuple[int, ...], dtype: torch.dtype, group_name: str, device: torch.device
) -> torch.Tensor:
    """A symmetric-memory buffer rendezvoused on ``group_name``. It lives as
    long as the process: the rendezvous handle is cached by address, so a larger
    buffer allocated where a freed one lived would get the freed one's size."""
    buffer = symm_mem.empty(shape, dtype=dtype, device=device)
    symm_mem.rendezvous(buffer, GroupName(group_name))
    _SYMMETRIC_BUFFERS.append(buffer)
    return buffer


def symmetric_partials(
    rows: int, hidden: int, group_name: str, device: torch.device
) -> list[torch.Tensor]:
    """Two [rows, hidden] BF16 symmetric-memory partials for consecutive launches
    to alternate between: with keyed peer reads, a launch's peers may still read
    the previous one's. (The pinned configs push the partials in band into
    compiler-owned peer mailboxes, so no peer reads them there; every caller
    still alternates, so any config is safe.)"""
    shape = (rows, hidden)
    return [
        symmetric_buffer(shape, torch.bfloat16, group_name, device) for _ in range(2)
    ]


def all_reduce(
    partial: torch.Tensor, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """The ranks' ``partial`` summed in rank order in FP32."""
    parts = [torch.empty_like(partial) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, partial, group=group)
    total = torch.zeros_like(partial, dtype=torch.float32)
    for part in parts:
        total = total + part.float()
    return total


def all_reduce_norm(
    partial: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ranks' ``partial`` summed in rank order in FP32 plus ``residual``,
    rounded to BF16, and its RMSNorm with the pre-folded ``norm_weight``
    (GemmaRMSNorm's ``1 + w``): (normed output, residual output)."""
    pre = (all_reduce(partial) + residual.float()).to(torch.bfloat16)
    pf = pre.float()
    out = pf * torch.rsqrt(pf.square().mean(-1, keepdim=True) + eps)
    return (out * norm_weight.float()).to(torch.bfloat16), pre


def say(text: str) -> None:
    """Print ``text`` in one write, so that the ranks' blocks do not interleave."""
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def verdict(ok: bool, group: dist.ProcessGroup, name: str) -> None:
    """Raise on every rank if the check failed on any: the ranks all-reduce
    their results first, so no rank leaves early and strands the others in a
    later collective."""
    failed = torch.tensor([0 if ok else 1], dtype=torch.int32, device="cuda")
    dist.all_reduce(failed, group=group)
    if failed.item():
        raise AssertionError(
            f"rank {group.rank()}: {name} differs from its reference "
            f"({failed.item()} rank(s) failed)"
        )


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality: floats are compared as same-width integers, so ``-0.0``
    differs from ``+0.0`` and equal NaN bit patterns match."""
    if a.dtype != b.dtype:
        return False
    if a.is_floating_point():
        bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
        dtype = bits[a.element_size()]
        a, b = a.view(dtype), b.view(dtype)
    return torch.equal(a, b)


def errors(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """The maximum absolute and the relative RMS error."""
    got, expected = got.float(), expected.float()
    diff = (got - expected).abs()
    norm = expected.norm().item()
    return (
        diff.max().item(),
        diff.norm().item() / norm if norm > 0 else diff.norm().item(),
    )


def report(
    rows: list[str],
    label: str,
    got: torch.Tensor,
    expected: torch.Tensor,
    tol: float,
    bitwise: bool = False,
) -> bool:
    """Append one stage's errors to ``rows``; False when the relative RMS error
    exceeds ``tol`` (NaN included) or, with ``bitwise``, the bits differ."""
    max_abs, rel = errors(got, expected)
    exact = same(got, expected)
    rows.append(
        f"  {label:<22} max {max_abs:.3g} rel-rms {rel:.3g}"
        + (" (bitwise)" if exact else "")
    )
    if (bitwise and not exact) or not rel <= tol:
        rows[-1] += "  <-- FAIL"
        return False
    return True


def bytes_equal(
    rows: list[str],
    label: str,
    got: torch.Tensor,
    expected: torch.Tensor,
    fraction: float = 0.999,
) -> bool:
    """Append the share of equal bytes to ``rows``; False below ``fraction``
    (an FP8 rounding tie may resolve differently)."""
    share = (got.view(torch.uint8) == expected.view(torch.uint8)).float().mean().item()
    rows.append(f"  {label:<22} bytes equal {share * 100:.4f}%")
    if share < fraction:
        rows[-1] += "  <-- FAIL"
        return False
    return True


def graph_time(
    step: Callable[[], object], calls: int, reps: int, collective: bool = True
) -> tuple[float, float]:
    """Median and minimum microseconds per launch over 10 replays of a CUDA graph
    of ``reps`` steps (``step`` makes ``calls`` launches).

    Two eager steps run first, so that each kernel compiles and sets up its
    persistent launch state (a rendezvous of the ranks) outside the capture.
    With ``collective``, the ranks meet at a barrier before each replay."""
    for _ in range(2):
        step()
        torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            step()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if collective:
            dist.barrier()
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / (reps * calls))
    times.sort()
    return times[len(times) // 2], times[0]


def run_main(
    kernel_name: str,
    correctness_check: Callable[..., None],
    benchmark: Callable[..., dict[str, Any]],
    verbose: bool,
    check_only: bool,
) -> dict[str, Any]:
    """A module's ``main()``: the four-rank check, then the timing
    (``check_only``: the check alone)."""
    _rank, _local_rank, group = initialize(
        kernel_name=kernel_name,
        world_size=WORLD_SIZE,
        signal_pad_bytes=SIGNAL_PAD_BYTES,
    )
    try:
        correctness_check(group, verbose=verbose)
        return {} if check_only else benchmark(group, verbose=verbose)
    finally:
        dist.destroy_process_group()
