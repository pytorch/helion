# pyrefly: ignore-errors
# ruff: noqa: ANN202
"""Cross-check for ``sglang_layer.py --workload dsv3_mla``: the same FlashInfer trtllm-gen MLA decode call
without building the SGLang model (a cold-L2 port of hvm_bench/steady_mla_b1_sglang.py, self-contained).

The arguments mirror what SGLang's TRTLLMMLABackend passes on its CUDA-graph decode path for DeepSeek-V3 at
one TP8 rank (sglang/srt/layers/attention/trtllm_mla_backend.py: _run_decode_kernel, init_cuda_graph_state,
_init_cuda_graph_metadata): 16 heads, one query token, 576-wide keys / 512-wide values, bf16 KV in 64-token
pages, max_seq_len = the model context length (163840) with the block table padded to it (-1 filled), a
150 MB zeroed workspace, a persistent multi-CTA counter sized for 8192 requests, softmax scale
qk_head_dim**-0.5 * yarn_mscale**2, enable_pdl = sglang.kernels.jit.utils.is_arch_support_pdl().  ``--eager-path``
uses the arguments of SGLang's eager (non-CUDA-graph) path instead (max_seq_len = 1024).

usage: sglang_mla_standalone.py [--context 1024] [--out JSON]   (SGLang env; CUDA_VISIBLE_DEVICES selects the GPU)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
NUM_HEADS = 16
KV_LORA_RANK = 512
ROPE_DIM = 64
QK_NOPE_HEAD_DIM = 128
QK_DIM = KV_LORA_RANK + ROPE_DIM
PAGE = 64
MAX_CONTEXT = 163840  # DeepSeek-V3 max_position_embeddings = SGLang's max_context_len
WORKSPACE_BYTES = 150 * 1024 * 1024  # trtllm_mla_backend.DEFAULT_WORKSPACE_SIZE_MB
NUM_PAGES = 4096  # page pool the request's pages are drawn from


def yarn_scale() -> float:
    # DeepSeek-V3: rope_scaling yarn factor 40, mscale_all_dim 1.0 -> scaling *= mscale**2
    mscale = 0.1 * 1.0 * math.log(40) + 1.0
    return (QK_NOPE_HEAD_DIM + ROPE_DIM) ** -0.5 * mscale * mscale


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--eager-path", action="store_true")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument(
        "--coldl2",
        default=os.environ.get("COLDL2_PATH", str(HERE.parent / "coldl2.py")),
    )
    ap.add_argument(
        "--out",
        default=os.environ.get("SGLANG_BENCH_SCRATCH", "sglang_scratch")
        + "/sglang_dsv3_mla_standalone.json",
    )
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("coldl2", args.coldl2)
    coldl2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coldl2)
    import flashinfer
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLM_MLA_MAX_BATCH_SIZE
    from sglang.srt.layers.attention.trtllm_mla_backend import (
        make_persistent_multi_ctas_kv_counter_buffer,
    )

    n = args.context
    gen = torch.Generator(device="cuda")
    gen.manual_seed(400)
    blocks = math.ceil(n / PAGE)
    query = torch.randn((1, 1, NUM_HEADS, QK_DIM), generator=gen, device="cuda").to(
        torch.bfloat16
    )
    kv_cache = torch.randn(
        (NUM_PAGES, 1, PAGE, QK_DIM), generator=gen, device="cuda"
    ).to(torch.bfloat16)
    pages = torch.randperm(NUM_PAGES, generator=gen, device="cuda")[:blocks].to(
        torch.int32
    )
    if args.eager_path:
        max_seq_len = n
        width = 64  # SGLang's eager plan pads cdiv(1024, 64) = 16 pages to 64 (its LCM page constraint)
    else:
        max_seq_len = MAX_CONTEXT
        width = MAX_CONTEXT // PAGE  # 2560, already a multiple of the constraint
    block_tables = torch.full((1, width), -1, dtype=torch.int32, device="cuda")
    block_tables[0, :blocks] = pages
    seq_lens = torch.tensor([n], dtype=torch.int32, device="cuda")
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.int8, device="cuda")
    counter = make_persistent_multi_ctas_kv_counter_buffer(
        torch.device("cuda"), NUM_HEADS, TRTLLM_MLA_MAX_BATCH_SIZE
    )
    scale = yarn_scale()
    pdl = is_arch_support_pdl()

    def call():
        return trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            enable_pdl=pdl,
            workspace_buffer=workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=ROPE_DIM,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=scale,
            return_lse=False,
            skip_softmax_threshold_scale_factor=None,
            multi_ctas_kv_counter_buffer=counter,
        )

    out = call()
    kv_rows = kv_cache[pages.long(), 0].reshape(-1, QK_DIM)[:n].float()
    s = torch.einsum("hd,ld->hl", query[0, 0].float(), kv_rows) * scale
    ref = s.softmax(-1) @ kv_rows[:, :KV_LORA_RANK]
    err = (out.float().reshape(ref.shape) - ref).norm() / ref.norm()
    assert err < 1e-2, err
    timing = coldl2.time_cold(call, reps=args.reps)  # one coldl2 session per process
    res = {
        "system": "SGLang-style FlashInfer call (standalone cross-check)",
        "flashinfer": flashinfer.__version__,
        "path": "eager" if args.eager_path else "cuda_graph",
        "max_seq_len": max_seq_len,
        "block_table_width": width,
        "enable_pdl": pdl,
        "bmm1_scale": scale,
        "median_us": timing["median_us"],
        "loop_us": timing["loop_us"],
        "p10_us": timing["p10_us"],
        "p90_us": timing["p90_us"],
        "reps": timing["reps"],
        "rel_l2_vs_fp32_reference": err.item(),
        "gpu": coldl2.gpu_info(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "gpu"}))


if __name__ == "__main__":
    main()
