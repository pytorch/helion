# pyrefly: ignore-errors
"""DeepSeek-V3 MLA decode megakernel (one request), pretuned for NVIDIA B200.

TP8 shapes: 16 heads, one query token (16 query rows), kv_lora 512 + rope 64,
BF16 paged KV (64-token pages), FP32 softmax / accumulation / LSE.  The kernel
takes any query-token count (multi-token prediction is causal over the query
tokens); QUERY_TOKENS sets the benchmarked one.  One Helion function, two roots
in one dynamic cross-loop launch:

* split-KV partial attention: one task per (KV split, 128-column slice of the
  512-wide output).  Every task holds all query rows and recomputes its split's
  QK^T over the full 576-wide key, so a short context still spreads over
  split x slice tasks (the layout FlashInfer's trtllm-gen MLA kernel uses on
  B200).  The split count and pages per split derive on device from the runtime
  length inside a fixed envelope of 64 splits (one page per split up to 64
  pages, then whole pages per split), so every sequence length runs the same
  compiled kernel (dynamic shapes, no host-side bucketing);
* log-sum-exp combine of the live splits into the BF16 output.

The default Triton preset targets short decode contexts. Its column slices
repeat QK^T at long contexts. The optional ``flash_mla_cute`` entry targets
one query on SM100 with 16-byte aligned query and KV pointers and matching
NVRTC and CUDA cooperative-groups headers for SM100. Each preset
runs in one launch, without programmatic dependent launch. The benchmark
compares with FlashInfer TRT-LLM-gen as vLLM calls it in its decode graph
(see MAX_MODEL_LEN).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import is_fake

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

QUERY_TOKENS = 1
NUM_HEADS = 16
KV_LORA_RANK = 512
ROPE_DIM = 64
QK_DIM = KV_LORA_RANK + ROPE_DIM
CACHE_BLOCK_SIZE = 64
MLA_SCALE = QK_DIM**-0.5
# The benchmarked decode context; the correctness check also covers longer
# contexts (one compiled kernel serves every length up to MAX_CONTEXT).
CONTEXTS = (1024,)
CHECK_CONTEXTS = (1, 63, 1024, 4096, 16384, 65536)
MAX_CONTEXT = 65536
BLOCK_TABLE_CAPACITY = MAX_CONTEXT // CACHE_BLOCK_SIZE
# DeepSeek-V3's max_model_len.  vLLM captures its decode CUDA graph with
# max_seq_len = max_model_len (and a block table that wide), and FlashInfer's
# MLA time depends on max_seq_len, so the baseline is called the same way.
MAX_MODEL_LEN = 163840
# DeepSeek-V3's non-rotary query/key head dim (before absorption into kv_lora).
QK_NOPE_HEAD_DIM = 128


@helion.aot_kernel(
    static_shapes=False,
    backend="triton",
    triton_do_not_specialize=False,
)
def flash_mla(
    query: torch.Tensor,  # [1, Q, H, 576]
    kv_cache: torch.Tensor,  # [blocks, 1, 64, 576]
    block_tables: torch.Tensor,  # [1, capacity]
    seq_lens: torch.Tensor,  # [1]
    scale: float,
) -> torch.Tensor:
    """Split-KV partial attention and the LSE combine as two roots of one megakernel."""
    _, query_len, num_heads, query_dim = query.shape
    query_len = hl.specialize(query_len)
    num_heads = hl.specialize(num_heads)
    query_dim = hl.specialize(query_dim)
    page = hl.specialize(kv_cache.size(2))
    value_dim = hl.specialize(query_dim - 64)
    # A literal: a module-level int reaches the kernel as a runtime scalar.
    split_capacity = hl.specialize(64)
    rows = query_len * num_heads
    q2d = query.view(rows, query_dim)
    kv3 = kv_cache.view(kv_cache.size(0), page, query_dim)
    k_lat3 = kv3[:, :, :value_dim]
    partial = torch.empty(
        (split_capacity, rows, value_dim), dtype=torch.float32, device=query.device
    )
    partial_lse = torch.empty(
        (split_capacity, rows), dtype=torch.float32, device=query.device
    )
    output = torch.empty((rows, value_dim), dtype=torch.bfloat16, device=query.device)
    qk_scale = scale * 1.44269504
    block_k = hl.register_block_size(query_dim)
    block_c = hl.register_block_size(split_capacity)

    # Partial attention over this split's pages for one slice of the output
    # columns.  QK^T contracts the whole 576-wide row (latent and rope are
    # contiguous) in one pipelined chunk loop.
    for tile_s, tile_d in hl.tile([split_capacity, value_dim], block_size=[1, None]):
        seq_len = seq_lens[0]
        num_pages = (seq_len + page - 1) // page
        per_split = (num_pages + split_capacity - 1) // split_capacity
        begin = tile_s.begin * per_split
        end = torch.minimum(begin + per_split, num_pages)
        if begin < end:
            # Row q * H + h attends to positions < seq_len - (Q - 1) + q (causal
            # over the query tokens).
            limit = seq_len - (query_len - 1) + hl.arange(rows) // num_heads
            m_i = hl.full([rows], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([rows], dtype=torch.float32)
            acc = hl.zeros([rows, tile_d], dtype=torch.float32)
            for tile_j in hl.tile(begin, end, block_size=1):
                phys = block_tables[0, tile_j.begin]
                scores = hl.zeros([rows, page], dtype=torch.float32)
                for tile_k in hl.tile(query_dim, block_size=block_k):
                    scores = hl.dot(q2d[:, tile_k], kv3[phys, :, tile_k].T, acc=scores)
                pos = tile_j.begin * page + hl.arange(page)
                scores = torch.where(
                    pos[None, :] < limit[:, None], scores * qk_scale, float("-inf")
                )
                m_new = torch.maximum(m_i, torch.amax(scores, dim=-1))
                m_safe = torch.where(m_new == float("-inf"), 0.0, m_new)
                alpha = torch.exp2(m_i - m_safe)
                p = torch.exp2(scores - m_safe[:, None])
                l_i = l_i * alpha + torch.sum(p, dim=-1)
                v = k_lat3[phys, :, tile_d]
                acc = hl.dot(p.to(torch.bfloat16), v, acc=acc * alpha[:, None])
                m_i = m_new
            empty = l_i == 0
            safe_l = torch.where(empty, 1.0, l_i)
            partial[tile_s.begin, :, tile_d] = torch.where(
                empty[:, None], 0.0, acc / safe_l[:, None]
            )
            if tile_d.begin == 0:
                partial_lse[tile_s.begin, :] = torch.where(
                    empty, float("-inf"), m_i + torch.log2(safe_l)
                )

    # Combine over the live splits (masked loads skip the envelope's dead
    # splits): one pass for the maximum log-sum-exp, one for the weighted sum.
    for tile_r, tile_e in hl.tile([rows, value_dim]):
        seq_len = seq_lens[0]
        num_pages = (seq_len + page - 1) // page
        per_split = (num_pages + split_capacity - 1) // split_capacity
        live = (num_pages + per_split - 1) // per_split
        best = hl.full([tile_r], float("-inf"), dtype=torch.float32)
        for tile_c in hl.tile(split_capacity, block_size=block_c):
            if tile_c.begin < live:
                ok = tile_c.index < live
                lse = hl.load(partial_lse, [tile_c, tile_r], extra_mask=ok[:, None])
                lse = torch.where(ok[:, None], lse, float("-inf"))
                best = torch.maximum(best, torch.amax(lse, dim=0))
        best = torch.where(best == float("-inf"), 0.0, best)
        denom = hl.zeros([tile_r], dtype=torch.float32)
        out = hl.zeros([tile_r, tile_e], dtype=torch.float32)
        for tile_c in hl.tile(split_capacity, block_size=block_c):
            if tile_c.begin < live:
                ok = tile_c.index < live
                lse = hl.load(partial_lse, [tile_c, tile_r], extra_mask=ok[:, None])
                w = torch.where(ok[:, None], torch.exp2(lse - best[None, :]), 0.0)
                vals = hl.load(
                    partial, [tile_c, tile_r, tile_e], extra_mask=ok[:, None, None]
                )
                denom = denom + torch.sum(w, dim=0)
                out = out + torch.sum(w[:, :, None] * vals, dim=0)
        output[tile_r, tile_e] = (out / denom[:, None]).to(torch.bfloat16)

    return output.view(1, query_len, num_heads, value_dim)


def _cute_grid_available() -> bool:
    # Load this CuTe-only prerequisite when the opt-in preset is requested.
    from helion._compiler.cute.cooperative_grid_ptx import GridCompilerUnavailable
    from helion._compiler.cute.cooperative_grid_ptx import grid_sync_asm

    try:
        return grid_sync_asm((10, 0)) is not None
    except GridCompilerUnavailable:
        return False


def _cute_key(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> int:
    """The one-query B200 preset keeps sequence lengths and page IDs on device."""
    if (
        query.device.type != "cuda"
        or torch.cuda.get_device_capability(query.device) != (10, 0)
        or query.shape != (1, 1, NUM_HEADS, QK_DIM)
        or query.dtype != torch.bfloat16
        or kv_cache.ndim != 4
        or kv_cache.shape[1:] != (1, CACHE_BLOCK_SIZE, QK_DIM)
        or kv_cache.dtype != torch.bfloat16
        or block_tables.shape != (1, BLOCK_TABLE_CAPACITY)
        or block_tables.dtype != torch.int32
        or seq_lens.shape != (1,)
        or seq_lens.dtype != torch.int32
        or scale != MLA_SCALE
        or any(
            t.device != query.device or not t.is_contiguous()
            for t in (query, kv_cache, block_tables, seq_lens)
        )
    ):
        raise ValueError(
            "flash_mla_cute requires SM100 and the single-query BF16 decode layout"
        )
    # The compiled kernel reads the physical storage of all four operands.
    # A lazy negative view shares that storage but has different logical values.
    # The gathered-page config matches an int32 page-lane range. Auto
    # indexing widens that range when any input exceeds the int32 limit.
    if kv_cache.numel() > torch.iinfo(
        torch.int32
    ).max or flash_mla_cute.settings.index_dtype not in (None, torch.int32):
        raise ValueError(
            "flash_mla_cute requires int32 indexing and a KV pool with at most "
            "2**31 - 1 elements; use flash_mla for wider indexing"
        )
    if any(t.is_neg() for t in (query, kv_cache, block_tables, seq_lens)):
        raise ValueError("flash_mla_cute requires resolved negative views")
    # Fake operands have no storage addresses. Keep metadata binding separate
    # from eager dispatch, where both actual inputs must be aligned. Inductor
    # may realign its graph inputs before calling the generated launch.
    has_fake_operand = False
    for operand in (query, kv_cache):
        if is_fake(operand):
            has_fake_operand = True
        elif operand.data_ptr() % 16:
            raise ValueError(
                "flash_mla_cute requires 16-byte aligned query and KV pointers"
            )
    if not _cute_grid_available():
        raise RuntimeError(
            "flash_mla_cute requires matching NVRTC and CUDA cooperative-groups headers "
            "for SM100; use flash_mla for the Triton preset"
        )
    return int(has_fake_operand)


# B200, one query / 16 heads: stage eight aligned Q/K panels, then combine
# the split weights in the same persistent launch. The default Triton entry
# above and the CuTe entry share the exact same Helion function.
flash_mla_cute = helion.kernel(
    flash_mla.fn,
    backend="cute",
    static_shapes=False,
    config=helion.Config(
        block_sizes=[32, 1, 128, 1, 128],
        num_threads=[1, 1, 1, 1, 128, 1, 1],
        pid_type="persistent_interleaved",
        cross_loop_pipeline="barrier",
        cute_fragment_half_mma=True,
        cute_gathered_page_pipeline=True,
        cute_gathered_page_stages=8,
        cute_in_kernel_grid_init=True,
    ),
    key=_cute_key,
)


def _make_inputs(context: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """One request at ``context`` tokens; its QUERY_TOKENS query tokens are its last
    positions."""
    torch.manual_seed(seed)
    blocks = math.ceil(context / CACHE_BLOCK_SIZE)
    query = torch.randn(
        (1, QUERY_TOKENS, NUM_HEADS, QK_DIM), device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        (BLOCK_TABLE_CAPACITY, 1, CACHE_BLOCK_SIZE, QK_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    block_tables = torch.zeros(
        (1, BLOCK_TABLE_CAPACITY), device="cuda", dtype=torch.int32
    )
    block_tables[0, :blocks] = torch.randperm(BLOCK_TABLE_CAPACITY, device="cuda")[
        :blocks
    ].to(torch.int32)
    seq_lens = torch.tensor([context], device="cuda", dtype=torch.int32)
    return {
        "query": query,
        "kv_cache": kv_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
    }


def _kernel_args(tensors: dict[str, torch.Tensor]) -> tuple[object, ...]:
    return (
        tensors["query"],
        tensors["kv_cache"],
        tensors["block_tables"],
        tensors["seq_lens"],
        MLA_SCALE,
    )


def reference(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """FP32 PyTorch reference, causal over the query tokens."""
    n = int(tensors["seq_lens"][0])
    blocks = math.ceil(n / CACHE_BLOCK_SIZE)
    kv = (
        tensors["kv_cache"][tensors["block_tables"][0, :blocks].long(), 0]
        .reshape(-1, QK_DIM)[:n]
        .float()
    )
    q = tensors["query"][0].float()
    s = torch.einsum("qhd,ld->qhl", q, kv) * MLA_SCALE
    query_len = q.size(0)
    pos = torch.arange(n, device=q.device)
    limit = n - query_len + 1 + torch.arange(query_len, device=q.device)
    s = torch.where(pos[None, None, :] < limit[:, None, None], s, float("-inf"))
    return torch.einsum("qhl,ld->qhd", s.softmax(-1), kv[:, :KV_LORA_RANK]).to(
        torch.bfloat16
    )[None]


# vLLM's FlashInfer MLA backend owns a zeroed 128 MiB workspace and a persistent
# zeroed multi-CTA KV counter (trtllm-gen resets it after every launch); both
# are allocated the same way here, without importing vLLM.
_FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024


def has_flashinfer() -> bool:
    """Whether FlashInfer's trtllm-gen MLA decode (the vLLM/SGLang B200 backend) is
    importable."""
    try:
        from flashinfer.decode import (  # noqa: F401
            trtllm_batch_decode_with_kv_cache_mla,
        )
    except ImportError:
        return False
    return True


def use_cudagraph() -> bool:
    """The timed closures replay pre-captured CUDA graphs."""
    return True


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("flash_mla is pretuned only for NVIDIA SM100")


def _make_flashinfer_call(
    tensors: dict[str, torch.Tensor], enable_pdl: bool | None = None
) -> Callable[[], torch.Tensor]:
    """The production B200 MLA decode of vLLM and SGLang: FlashInfer trtllm-gen,
    called as vLLM's decode CUDA graph calls it (max_seq_len and block-table width
    from MAX_MODEL_LEN).  ``enable_pdl=None`` keeps FlashInfer's default
    (programmatic dependent launch on sm90+, as both frameworks ship)."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.utils import get_device_sm_count
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    query = tensors["query"]
    block_tables = torch.zeros(
        (1, MAX_MODEL_LEN // CACHE_BLOCK_SIZE), device=query.device, dtype=torch.int32
    )
    block_tables[:, :BLOCK_TABLE_CAPACITY] = tensors["block_tables"]
    counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
        1, NUM_HEADS, get_device_sm_count(query.device)
    )
    workspace = torch.zeros(
        _FLASHINFER_WORKSPACE_BYTES, dtype=torch.int8, device=query.device
    )
    counter = torch.zeros(counter_bytes, dtype=torch.uint8, device=query.device)

    def launch() -> torch.Tensor:
        return trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=tensors["kv_cache"],
            workspace_buffer=workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=ROPE_DIM,
            block_tables=block_tables,
            seq_lens=tensors["seq_lens"],
            max_seq_len=MAX_MODEL_LEN,
            bmm1_scale=MLA_SCALE,
            bmm2_scale=1.0,
            return_lse=False,
            multi_ctas_kv_counter_buffer=counter,
            enable_pdl=enable_pdl,
        )

    return launch


def correctness_check() -> None:
    """Every CHECK_CONTEXTS length against the FP32 reference and FlashInfer, with
    one compiled kernel."""
    _require_sm100()
    for context in CHECK_CONTEXTS:
        tensors = _make_inputs(context, seed=context)
        out = flash_mla(*_kernel_args(tensors))
        torch.testing.assert_close(
            out.float(), reference(tensors).float(), atol=0.02, rtol=0.05
        )
        if has_flashinfer():
            torch.testing.assert_close(
                out.float(),
                _make_flashinfer_call(tensors)().float(),
                atol=0.02,
                rtol=0.05,
            )


@torch.inference_mode()
def main(verbose: bool = True, *, include_cute: bool = False) -> dict:
    """Compare captured graphs; request CuTe to also time the shipped Triton preset."""
    _require_sm100()
    if not has_flashinfer():
        raise RuntimeError("FlashInfer is required for this comparison")
    if include_cute and not _cute_grid_available():
        raise RuntimeError(
            "The CuTe benchmark requires NVRTC and matching cooperative-groups headers"
        )

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _bench import capture_cuda_graph
    from _bench import run_sweep

    # CUDA graphs store raw addresses, not ownership of captured Python
    # tensors. Keep the callable (including FlashInfer's private table,
    # counter and workspace), its output and its reference through timing.
    captures: list[
        tuple[Callable, torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]
    ] = []

    def capture(call: Callable[[], torch.Tensor], expected: torch.Tensor) -> Callable:
        graph, output = capture_cuda_graph(call)
        graph.replay()
        torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.05)
        captures.append((call, graph, output, expected))
        return graph.replay

    def check_timed_graphs() -> None:
        for _owner, graph, output, expected in captures:
            graph.replay()
            torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.05)
        captures.clear()

    def make_calls(context: int) -> tuple:
        # Drop the previous shape's storage only after its timed replays.
        check_timed_graphs()
        tensors = _make_inputs(context, seed=context)
        args = _kernel_args(tensors)
        expected = reference(tensors).float()
        triton = capture(lambda: flash_mla(*args), expected)
        baseline = capture(_make_flashinfer_call(tensors), expected)
        baselines = [("flashinfer_trtllm_gen", baseline)]
        selected = triton
        if include_cute:
            selected = capture(lambda: flash_mla_cute(*args), expected)
            baselines.insert(0, ("helion_triton", triton))
        return selected, baselines, f"{context:>8d}"

    result = run_sweep(
        CONTEXTS,
        make_calls,
        use_cudagraph=False,
        pre_captured_cudagraph=True,
        thermal_warmup_ms=5_000,
        verbose=verbose,
        shape_header=f"{'context':>8s}",
    )
    check_timed_graphs()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cute",
        action="store_true",
        help="compare CuTe, shipped Triton and FlashInfer",
    )
    main(include_cute=parser.parse_args().cute)
