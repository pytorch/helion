# pyrefly: ignore-errors
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202
"""SGLang cold-L2 timing of ONE DeepSeek-V3 NVFP4 MoE block at TP4, batch-1 decode, exactly as SGLang's
CUDA-graph decode runs it on four B200s.

The block matches Helion's ``deepseek_v3_moe_nvfp4_tp`` and vLLM's baseline in ``../deepseek_tp4.py``: one
decode token's MoE input (the post-attention RMSNorm output, replicated on every rank) through the router
(256 x 7168 GEMM), DeepSeek-V3 grouped top-8 of 256 experts, the routed experts and the shared expert on each
rank's 512-wide intermediate shard, the all-reduce of the four ranks' partial outputs, the residual add and
the next RMSNorm.  In SGLang that is the end of one ``DeepseekV2DecoderLayer.forward`` plus the head of the
next one:

    with layer.ffn_boundary.exit(fb) as ffn_exit, moe_output_buffer_ctx(buf):
        h = layer.mlp(h_in, fb, None)        # router, routing + routed experts (FlashInfer TRT-LLM NVFP4 MoE),
    h = ffn_exit.finish(h)                   # shared expert on SGLang's alt stream, finalize
    out = next_layer.attn_boundary.prepare(h, fb, ...)   # fused all-reduce + residual add + RMSNorm

The model is built by SGLang itself (``sglang.benchmark.one_batch.main`` + ``one_batch.load_model``, TP4, the
DeepSeek-V3 NVFP4 checkpoint config with 5 layers, all MoE, dummy weights, SGLang's default backends on
B200).  SGLang's dummy loader draws every float uniform in +-1e-3 and leaves the packed FP4 weights
uninitialized; it is replaced in each rank by an initializer at realistic scales (``realistic_init``):
packed FP4 weights random bytes, FP8 block scales uniform in [0.5, 1.5], FP32 global weight scales such
that every projection has unit output variance, activation input scales 0.01, bf16 matrices unit output
variance, norm gammas in [0.8, 1.2].  Only values change: shapes, layouts, kernels and graphs are SGLang's.

Routing cases (Helion's ``ROUTING_CASES``): spread, clustered and edge choose experts
(3,17,35,50,77,95,113,127), (0..7), (128,159,160,191,192,223,224,255).  Case k lives on layer k+1 (layers
1..3; layers 2..4 provide the next RMSNorm): its router weight is set, as Helion's ``_router_weight``
does, so the token's router logits are 8, 7, ..., 1 on the chosen experts and -12 elsewhere, with Helion's
correction bias ((7 e mod 17) - 8) / 1000.  The routing is checked in PyTorch.

Inputs: the MoE input and residual each block sees inside SGLang's own eager forward of the decode token.

Timing (as ``../deepseek_tp4.py``): each case is captured as a CUDA graph inside SGLang's
``graph_capture()`` / ``model_capture_mode()`` (as its decode graph runner captures); the three cases are
timed interleaved in ONE ``coldl2.time_cold_many`` session per process (rank); before each call every rank
restores the residual, flushes L2 and the ranks align with a one-shot symmetric-memory all-reduce of a few
bytes, all outside the timed span.  Reported: the largest of the four ranks' medians, plus every rank's.

Validation, every rank, same processes:
  (a) numerics vs SGLang's forward: each case's graph replay equals the next layer's attention input and
      residual inside SGLang's eager forward of the decode token (identical inputs).
  (b) numerics vs an independent FP32 reference: DeepSeek-V3 routing, NVFP4 dequantized experts (from the
      weights the initializer wrote) with emulated NVFP4 activation quantization, the four ranks' partial
      sums, residual add, RMSNorm; and each rank's pre-all-reduce partial / shared-expert output of SGLang's
      eager MoE vs the reference's.  The reference without the case's lowest-weight expert must disagree.
  (c) kernel trace: the real decode step (``ModelRunner.forward`` -> SGLang's decode CUDA graph) is
      profiled after timing; each stream's kernel sequence of the timed call (name, grid, block, smem) must
      appear in it for every MoE layer that is followed by another layer.

usage: sglang_dsv3_moe_tp4.py --out JSON [--gpus 0,1,2,4] [--reps 200]
Run in the SGLang env (``source ../env.sh; sglang_python sglang_dsv3_moe_tp4.py ...``); see README.md and
run_tp4.sh.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.util
import json
import math
import operator
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import zlib

HERE = Path(__file__).resolve().parent
DEFAULT_COLDL2 = HERE.parent / "coldl2.py"
MODELS = Path(os.environ.get("SGLANG_MODELS_DIR", HERE / "models"))
SGLANG_DIR = Path(os.environ.get("SGLANG_DIR", Path.home() / "local/sglang"))
LLAMA_TOKENIZER = os.environ.get(
    "LLAMA_TOKENIZER", str(Path.home() / "local/hvm_envs/ae_e2e/llama-3.2-1b")
)
SCRATCH = os.environ.get("SGLANG_BENCH_SCRATCH", "sglang_scratch")
OPTS_ENV = "SGLANG_DSV3_TP4_BENCH_OPTS"
MAX_RETRIES = 3
WORLD_SIZE = 4
NUM_LAYERS = 5
WORKLOAD = "deepseek_v3_moe_tp4"

# Helion's routing cases (pretuned_kernels/megakernels/deepseek_v3_moe_nvfp4_tp/_setup.py), one per layer.
CASES = (
    ("spread", 1, (3, 17, 35, 50, 77, 95, 113, 127)),
    ("clustered", 2, (0, 1, 2, 3, 4, 5, 6, 7)),
    ("edge", 3, (128, 159, 160, 191, 192, 223, 224, 255)),
)
ROUTING_LOGITS = (8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
NUM_EXPERTS, TOP_K, NUM_GROUPS, TOPK_GROUPS, ROUTED_SCALE = 256, 8, 8, 4, 2.5
RMS_EPS = 1e-6

# realistic_init constants
FP4_RMS = math.sqrt(
    (0 + 0.25 + 1 + 2.25 + 4 + 9 + 16 + 36) / 8
)  # rms of a uniformly random E2M1 code
BLOCK_SCALE_LO, BLOCK_SCALE_HI = 0.5, 1.5
BLOCK_SCALE_MS = (BLOCK_SCALE_HI**3 - BLOCK_SCALE_LO**3) / (
    3 * (BLOCK_SCALE_HI - BLOCK_SCALE_LO)
)
INPUT_SCALE = 0.01  # = amax / (6 * 448) for an activation amax of 27: unit-variance inputs never saturate
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def load_coldl2(path: Path):
    spec = importlib.util.spec_from_file_location("coldl2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------
# GPU hygiene (by UUID, so CUDA_VISIBLE_DEVICES ordering does not matter)
# ----------------------------------------------------------------------------------------------------
def gpu_uuids() -> dict[int, str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        int(i): u.strip()
        for i, u in (line.split(",") for line in out.strip().splitlines())
    }


def compute_apps() -> list[tuple[str, int, str]]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    apps = []
    for line in out.strip().splitlines():
        if line.strip():
            uuid, pid, mem = (s.strip() for s in line.split(","))
            apps.append((uuid, int(pid), mem))
    return apps


def foreign_apps(uuids: set[str], allowed_pids: set[int]) -> list[str]:
    return [
        f"{u},{p},{m}"
        for u, p, m in compute_apps()
        if u in uuids and p not in allowed_pids
    ]


def wait_for_idle_gpus(indices: list[int], poll_s: float = 30.0) -> None:
    uuids = {gpu_uuids()[i] for i in indices}
    while users := foreign_apps(uuids, set()):
        print(
            f"[{time.strftime('%H:%M:%S')}] GPUs {indices} busy ({users}); waiting {poll_s:.0f} s",
            flush=True,
        )
        time.sleep(poll_s)


def sglang_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(SGLANG_DIR), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # not a git checkout: report the installed version instead
        import importlib.metadata

        return "sglang " + importlib.metadata.version("sglang")


# ----------------------------------------------------------------------------------------------------
# Kernel traces (names and launch shapes only)
# ----------------------------------------------------------------------------------------------------
def profile_kernels(fn, sync) -> list[dict]:
    """Every device kernel ``fn()`` runs, in start order: name, start/duration (us), stream, launch shape."""
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile

    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        sync()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        events = json.loads(Path(path).read_text())["traceEvents"]
    finally:
        os.unlink(path)
    kernels = sorted(
        (e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"),
        key=operator.itemgetter("ts"),
    )
    return [
        {
            "name": e["name"],
            "ts": float(e["ts"]),
            "dur": float(e["dur"]),
            "stream": e["args"].get("stream"),
            "grid": e["args"].get("grid"),
            "block": e["args"].get("block"),
            "smem": e["args"].get("shared memory"),
        }
        for e in kernels
    ]


def kernel_id(k: dict) -> str:
    return f"{k['name']} grid={k['grid']} block={k['block']} smem={k['smem']}"


def by_stream(kernels: list[dict]) -> dict:
    streams: dict = {}
    for k in kernels:
        streams.setdefault(k["stream"], []).append(k)
    return streams


def count_occurrences(seq: list[str], pattern: list[str]) -> int:
    n = len(pattern)
    return sum(seq[i : i + n] == pattern for i in range(len(seq) - n + 1))


def short(name: str, width: int = 110) -> str:
    return name if len(name) <= width else name[: width - 3] + "..."


# ----------------------------------------------------------------------------------------------------
# Realistic dummy weights (replaces SGLang's initialize_dummy_weights in each rank)
# ----------------------------------------------------------------------------------------------------
def make_realistic_init(
    orig,
    torch,
    tp_rank: int,
    tp_size: int,
    keep: dict[int, tuple[int, ...]],
    raw: dict,
    report: dict,
):
    """SGLang's dummy init, then every parameter re-drawn at realistic scales (see the module docstring).

    Seeds are per parameter name, so replicated parameters (norms, router, attention) are identical on every
    rank; the TP-sharded expert / shared-expert weights add the rank to the seed, so the four ranks' partial
    outputs differ.  For the layers in ``keep`` the raw (checkpoint-layout) weights of the case's experts and
    of the shared expert are copied to ``raw`` for the FP32 reference, before SGLang re-lays them out.
    """

    def init(model, *args, **kwargs):
        orig(model, *args, **kwargs)
        params = dict(model.named_parameters())
        counts: dict[str, int] = {}

        def note(rule):
            counts[rule] = counts.get(rule, 0) + 1

        with torch.no_grad():
            for name, p in params.items():
                if p.numel() == 0:
                    continue
                ffn_shard = ".mlp.experts." in name or ".mlp.shared_experts." in name
                seed = zlib.crc32(name.encode()) + (
                    1000003 * (tp_rank + 1) if ffn_shard else 0
                )
                gen = torch.Generator(device=p.device)
                gen.manual_seed(seed)
                if p.dtype == torch.uint8 and name.endswith(
                    "weight"
                ):  # packed NVFP4: two E2M1 codes/byte
                    p.random_(0, 256, generator=gen)
                    note("fp4 weight: random codes")
                elif name.endswith("weight_scale") and p.dtype == torch.float8_e4m3fn:
                    tmp = torch.empty(p.shape, dtype=torch.float32, device=p.device)
                    p.copy_(
                        tmp.uniform_(BLOCK_SCALE_LO, BLOCK_SCALE_HI, generator=gen).to(
                            p.dtype
                        )
                    )
                    note("fp8 block scale: U[0.5, 1.5]")
                elif name.endswith("weight_scale_2"):
                    sibling = params[name[: -len("_scale_2")]]
                    k = 2 * sibling.shape[-1]
                    if (
                        "w2_weight" in name or "down_proj" in name
                    ):  # row-parallel: full contraction length
                        k *= tp_size
                    p.fill_(1.0 / (FP4_RMS * math.sqrt(BLOCK_SCALE_MS * k)))
                    note("fp32 global weight scale: unit output variance")
                elif name.endswith("input_scale"):
                    p.fill_(INPUT_SCALE)
                    note(f"fp32 activation input scale: {INPUT_SCALE}")
                elif not torch.is_floating_point(p):
                    note(f"left as SGLang initialized it ({p.dtype})")
                elif "scale" in name.split(".")[-1]:
                    p.fill_(1.0)
                    note("other scale: 1.0")
                elif p.dim() == 1:
                    p.uniform_(0.8, 1.2, generator=gen)
                    note("1-D float (norm gamma, correction bias): U[0.8, 1.2]")
                else:
                    bound = (3.0 / p.shape[-1]) ** 0.5
                    p.uniform_(-bound, bound, generator=gen)
                    note("float matrix: unit output variance")
            for layer, experts in keep.items():
                sel = torch.tensor(experts, device="cuda")
                prefix = f"model.layers.{layer}.mlp."
                for name, p in params.items():
                    if not name.startswith(prefix) or ".gate." in name:
                        continue
                    key = name[len(prefix) :]
                    if (
                        ".experts." in name
                        and p.dim() >= 1
                        and p.shape[0] == NUM_EXPERTS
                    ):
                        raw.setdefault(layer, {})[key] = p.detach()[sel].clone()
                    else:
                        raw.setdefault(layer, {})[key] = p.detach().clone()
        report["rules"] = counts
        report["kept_raw"] = {
            layer: {k: [list(v.shape), str(v.dtype)] for k, v in d.items()}
            for layer, d in raw.items()
        }

    return init


# ----------------------------------------------------------------------------------------------------
# FP32 reference of the block (independent of SGLang's kernels)
# ----------------------------------------------------------------------------------------------------
def dequant_nvfp4(torch, packed, block_scale, global_scale):
    """[.., K/2] packed E2M1 codes (low nibble first) x [.., K/16] FP8 block scales x global scale."""
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    return lut[codes] * block_scale.float().repeat_interleave(16, dim=-1) * global_scale


def quant_dequant_nvfp4(torch, x, global_scale: float):
    """NVFP4 activation quantization (as TRT-LLM's cvt_warp_fp16_to_fp4), returned dequantized, FP32."""
    blocks = x.float().reshape(-1, 16)
    amax = blocks.abs().amax(-1, keepdim=True)
    sf = (amax * (global_scale / 6.0)).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    out_scale = torch.where(sf > 0, global_scale / sf, torch.zeros_like(sf))
    scaled = (blocks * out_scale).clamp(-6.0, 6.0)
    grid = torch.tensor(E2M1, device=x.device)
    q = grid[(scaled.abs()[..., None] - grid).abs().argmin(-1)] * scaled.sign()
    return (q * sf / global_scale).reshape(x.shape)


def deepseek_routing(torch, logits, bias):
    """DeepSeek-V3 noaux_tc grouped top-8: (expert ids, weights incl. routed_scaling_factor)."""
    scores = torch.sigmoid(logits.float())
    choice = scores + bias.float()
    groups = choice.view(NUM_GROUPS, -1)
    group_scores = groups.topk(2, dim=-1).values.sum(-1)
    keep = torch.zeros(NUM_GROUPS, dtype=torch.bool, device=logits.device)
    keep[group_scores.topk(TOPK_GROUPS).indices] = True
    masked = torch.where(
        keep[:, None], groups, torch.full_like(groups, float("-inf"))
    ).flatten()
    ids = masked.topk(TOP_K).indices
    w = scores[ids]
    return ids, w / w.sum() * ROUTED_SCALE


def gated_projection(torch, raw: dict, prefix: str, x, rows_axis_halves: bool = True):
    """FP32 x NVFP4 projection ``prefix`` (weight/weight_scale/weight_scale_2 in raw) of quantized x."""
    w, s, s2 = (
        raw[prefix + "weight"],
        raw[prefix + "weight_scale"],
        raw[prefix + "weight_scale_2"].float(),
    )
    deq = dequant_nvfp4(torch, w, s, 1.0)
    if (
        s2.numel() >= 2 and rows_axis_halves
    ):  # gate / up halves with their own global scale
        half = deq.shape[0] // 2
        deq = torch.cat((deq[:half] * s2.flatten()[0], deq[half:] * s2.flatten()[1]))
    else:
        deq = deq * s2.flatten()[0]
    return deq @ x


def moe_partial_reference(
    torch,
    raw: dict,
    experts: tuple[int, ...],
    h_in,
    gate_w,
    bias,
    up_first: bool,
    exclude: tuple[int, ...] = (),
) -> dict:
    """This rank's partial MoE output (routed + shared) of the bf16 token ``h_in``, FP32.  ``up_first``:
    the expert w13 rows (and block scales) are stored [up; gate] (SGLang's ``load_up_proj_weight_first``
    for the TRT-LLM layout); the per-expert global scales stay [gate, up].  Experts in ``exclude`` are left
    out (the sensitivity control)."""
    logits = h_in.float().flatten() @ gate_w.float().T
    ids, weights = deepseek_routing(torch, logits, bias)
    x = h_in.float().flatten()
    routed = torch.zeros_like(x)
    order = list(experts)

    # global activation scales are FP32 reciprocals of the input scales, as SGLang computes them (a double
    # reciprocal of 0.01f is 100.0000022 instead of 100.0 and moves near-tie E2M1 roundings)
    def global_scale(t):
        return float(torch.reciprocal(t.float().max()))

    gs13 = global_scale(raw["experts.w13_input_scale"])
    gs2 = global_scale(raw["experts.w2_input_scale"])
    xq = quant_dequant_nvfp4(torch, x, gs13)
    for e, wt in zip(ids.tolist(), weights.tolist(), strict=True):
        if e not in order or e in exclude:
            continue  # an expert the reference has no weights for is reported as missing
        j = order.index(e)
        w13 = dequant_nvfp4(
            torch, raw["experts.w13_weight"][j], raw["experts.w13_weight_scale"][j], 1.0
        )
        s2 = raw["experts.w13_weight_scale_2"][j].float().flatten()
        half = w13.shape[0] // 2
        gate_rows, up_rows = (
            (w13[half:], w13[:half]) if up_first else (w13[:half], w13[half:])
        )
        gate = (gate_rows * s2[0]) @ xq
        up = (up_rows * s2[-1]) @ xq
        act = quant_dequant_nvfp4(torch, torch.nn.functional.silu(gate) * up, gs2)
        w2 = dequant_nvfp4(
            torch,
            raw["experts.w2_weight"][j],
            raw["experts.w2_weight_scale"][j],
            float(raw["experts.w2_weight_scale_2"][j].float().flatten()[0]),
        )
        routed += wt * (w2 @ act)
    sh_in = global_scale(raw["shared_experts.gate_up_proj.input_scale"])
    sh_dn_in = global_scale(raw["shared_experts.down_proj.input_scale"])
    gu = gated_projection(
        torch, raw, "shared_experts.gate_up_proj.", quant_dequant_nvfp4(torch, x, sh_in)
    )
    half = gu.shape[0] // 2
    act = quant_dequant_nvfp4(
        torch, torch.nn.functional.silu(gu[:half]) * gu[half:], sh_dn_in
    )
    shared = gated_projection(
        torch, raw, "shared_experts.down_proj.", act, rows_axis_halves=False
    )
    return {
        "partial": routed + shared,
        "shared": shared,
        "routed_ids": sorted(ids.tolist()),
        "missing": sorted(set(ids.tolist()) - set(order)),
    }


def compare(torch, a, b) -> dict:
    a, b = a.float().flatten(), b.float().flatten()
    return {
        "max_abs": (a - b).abs().max().item(),
        "ref_max_abs": b.abs().max().item(),
        "rel_l2": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "bitwise_equal": bool(torch.equal(a, b)),
    }


# ----------------------------------------------------------------------------------------------------
# Rank worker (spawned by one_batch.main; must be a top-level function: multiprocessing uses spawn)
# ----------------------------------------------------------------------------------------------------
def tp_worker(server_args, port_args, bench_args, gpu_id, tp_rank):
    opts = json.loads(os.environ[OPTS_ENV])
    try:
        from sglang.benchmark import one_batch as ob
        from sglang.srt.model_loader import loader as sgl_loader
        from sglang.srt.runtime_context import SpawnRanks
        from sglang.srt.runtime_context import publish
        from sglang.srt.runtime_context import spawn_world_rank
        from sglang.srt.utils import configure_logger
        import torch
        import torch.distributed as dist

        # == one_batch.latency_test's prologue ==
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(
                world_rank=spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0),
                gpu_id=gpu_id,
            ),
        )
        ob._init_process_global_configs()
        configure_logger(server_args, prefix=f" TP{tp_rank}")
        coldl2 = load_coldl2(Path(opts["coldl2"]))
        raw: dict = {}
        init_report: dict = {}
        orig_init = sgl_loader.initialize_dummy_weights
        sgl_loader.initialize_dummy_weights = make_realistic_init(
            orig_init,
            torch,
            tp_rank,
            server_args.tp_size,
            {layer: experts for _, layer, experts in CASES},
            raw,
            init_report,
        )
        try:
            runner, _ = ob.load_model(server_args, port_args, gpu_id, tp_rank)
        finally:
            sgl_loader.initialize_dummy_weights = orig_init
        assert init_report.get("rules"), "realistic_init did not run"
        result = run_rank(
            opts, ob, runner, coldl2, torch, dist, tp_rank, raw, init_report
        )
        if tp_rank == 0:
            Path(opts["child_out"]).write_text(json.dumps(result, indent=1))
        from sglang.srt.distributed.parallel_state import get_tp_group

        dist.barrier(group=get_tp_group().cpu_group)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)  # skip NCCL / IPC teardown
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # The other ranks would wait in a collective forever: end the benchmark's own process group
        # (the launcher starts it in a new session).
        if opts.get("pgid") == os.getpgid(0):
            os.killpg(os.getpgid(0), signal.SIGKILL)
        os._exit(1)


def run_rank(opts, ob, runner, coldl2, torch, dist, tp_rank, raw, init_report) -> dict:
    import flashinfer
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.distributed.parallel_state import graph_capture
    from sglang.srt.eplb.expert_distribution import (
        get_global_expert_distribution_recorder,
    )
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
    from sglang.srt.layers.moe.moe_runner import base as moe_base
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_context import ForwardContext
    from sglang.srt.model_executor.forward_context import forward_context
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
    from sglang.srt.runtime_context import get_model
    import torch.distributed._symmetric_memory as symm_mem
    import triton

    mr = runner.torch_runner
    model = mr.model
    sync = torch.cuda.synchronize
    tp = get_tp_group()
    world = tp.world_size
    assert world == WORLD_SIZE, world
    dev = torch.device("cuda", torch.cuda.current_device())

    def gather(obj):
        out = [None] * world
        dist.all_gather_object(out, obj, group=tp.cpu_group)
        return out

    input_len = opts["input_len"]
    # ---- warm-up exactly like one_batch.latency_test (JIT / autotune everything once) ----
    reqs = ob.prepare_synthetic_inputs_for_latency_test(1, input_len)
    ob.latency_test_run_once(
        "warmup",
        runner,
        lambda *a, **k: None,
        reqs,
        1,
        input_len,
        4,
        log_decode_step=0,
        profile=False,
        profile_record_shapes=False,
        profile_activities=("CPU", "GPU"),
        profile_prefix="",
        profile_stage="all",
        tp_rank=tp_rank,
    )
    attn = mr.attn_backend
    view = resolving_view(mr.server_args)
    layers = model.model.layers
    assert len(layers) == NUM_LAYERS
    experts0 = layers[1].mlp.experts
    info = {
        "attention_backend": type(attn).__name__,
        "server_args": {
            k: str(getattr(view, k, None))
            for k in (
                "tp_size",
                "quantization",
                "attention_backend",
                "decode_attention_backend",
                "moe_runner_backend",
                "moe_a2a_backend",
                "enable_flashinfer_allreduce_fusion",
                "disable_shared_experts_fusion",
                "disable_custom_all_reduce",
                "enable_symm_mem",
                "kv_cache_dtype",
                "cuda_graph_config",
            )
        },
        "decode_cuda_graph_runner": type(mr.decode_cuda_graph_runner).__name__,
        "model_class": type(model).__name__,
        "num_hidden_layers": len(layers),
        "moe": {
            "class": type(layers[1].mlp).__name__,
            "experts_class": type(experts0).__name__,
            "quant_method": type(experts0.quant_method).__name__,
            "runner_backend": str(getattr(experts0, "runner_backend", None)),
            "shared_experts_on_alt_stream": layers[1].mlp.alt_stream is not None,
            "num_fused_shared_experts": layers[1].mlp.num_fused_shared_experts,
            "shared_expert_tp1": layers[1].mlp._shared_expert_tp1,
        },
        "weights": "SGLang dummy weights re-drawn at realistic scales (realistic_init)",
        "weight_init": init_report,
    }
    assert not layers[1].mlp._shared_expert_tp1, (
        "the reference assumes a TP-sharded shared expert"
    )
    # SGLang loads w13 rows as [up; gate] for the TRT-LLM MoE layout (load_up_proj_weight_first)
    up_first = bool(experts0.quant_method.load_up_proj_weight_first)
    info["moe"]["w13_rows_up_first"] = up_first
    versions = {
        "sglang_commit": sglang_commit(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "flashinfer": flashinfer.__version__,
    }

    def model_ctx(fbx):
        ctx = contextlib.ExitStack()
        ctx.enter_context(
            get_model().override(shared_experts=model.shared_experts_config)
        )
        ctx.enter_context(
            forward_context(ForwardContext(attn_backend=attn), forward_batch=fbx)
        )
        return ctx

    # ---- prefill, then the decode step (one_batch.decode without the forward) ----
    runner.clear()
    reqs = ob.prepare_synthetic_inputs_for_latency_test(1, input_len)
    next_token_ids, _, batch = runner.extend(reqs)
    batch.input_ids = next_token_ids.to(torch.int64)
    batch.prepare_for_decode()
    ob._maybe_prepare_mlp_sync_batch(batch, mr)

    def new_fb():
        return ForwardBatch.init_new(batch, mr, return_hidden_states_before_norm=False)

    for _ in range(3):
        out = mr.forward(new_fb())
        sync()
    assert out.can_run_graph, "SGLang did not run its decode CUDA graph"

    # ---- routing cases: Helion's correction bias everywhere, each case's router weight on its layer ----
    idx = torch.arange(NUM_EXPERTS, device=dev, dtype=torch.int32)
    bias = (((idx * 7) % 17) - 8).float() * 0.001
    for layer in layers:
        layer.mlp.gate.e_score_correction_bias.data.copy_(bias)

    def eager_pass():
        """SGLang's eager forward of the decode token, recording each case's MoE input / residual (entry)
        and the next layer's attention input / residual (the block's outputs)."""
        rec: dict = {L: {} for _, L, _ in CASES}
        hooks = []
        for _, L, _ in CASES:
            r = rec[L]

            def mlp_pre(mod, a, kw, r=r):
                s = a[1].residual_stream
                r["h_in"] = a[0].clone()
                r["residual"] = s.residual.clone()
                r["entry_pending"] = (
                    None if s.pending is None else type(s.pending).__name__
                )
                r["entry_transport"] = repr(s.transport)
                r["zero_allocator"] = repr(
                    a[2] if len(a) > 2 else kw.get("gemm_output_zero_allocator")
                )

            def mlp_post(mod, a, kw, output, r=r):
                r["mlp_output_type"] = type(output).__name__

            def attn_pre(mod, a, kw, r=r):
                h = kw["hidden_states"]
                r["out"] = h.clone()
                r["res_out"] = kw["forward_batch"].residual_stream.residual.clone()

            hooks += [
                layers[L].mlp.register_forward_pre_hook(mlp_pre, with_kwargs=True),
                layers[L].mlp.register_forward_hook(mlp_post, with_kwargs=True),
                layers[L + 1].self_attn.register_forward_pre_hook(
                    attn_pre, with_kwargs=True
                ),
            ]
        buffers = []
        orig_ctx = moe_base.moe_output_buffer_ctx

        def recording_ctx(buf):
            buffers.append([list(buf.shape), str(buf.dtype)])
            return orig_ctx(buf)

        moe_base.moe_output_buffer_ctx = recording_ctx
        try:
            fbx = new_fb()
            with model_ctx(fbx):
                mr.eager_runner.execute(fbx)
            sync()
        finally:
            moe_base.moe_output_buffer_ctx = orig_ctx
            for h in hooks:
                h.remove()
        return rec, buffers

    def router_weight(hidden, target):  # Helion's _router_weight
        hidden_f32 = hidden.float()
        return (target.float().T * hidden_f32 / torch.sum(hidden_f32 * hidden_f32)).to(
            torch.bfloat16
        )

    for (
        _,
        L,
        experts,
    ) in CASES:  # in layer order: a layer's input depends only on the layers before it
        rec, _ = eager_pass()
        target = torch.full((1, NUM_EXPERTS), -12.0, device=dev, dtype=torch.bfloat16)
        for e, v in zip(experts, ROUTING_LOGITS, strict=True):
            target[0, e] = v
        layers[L].mlp.gate.weight.data.copy_(router_weight(rec[L]["h_in"], target))
    rec, moe_buffers = eager_pass()  # the reference pass
    use_moe_buffer = len(moe_buffers) > 0

    cases_out: dict = {}
    for label, L, experts in CASES:
        r = rec[L]
        logits = r["h_in"].float().flatten() @ layers[L].mlp.gate.weight.float().T
        ids, weights = deepseek_routing(torch, logits, bias)
        cases_out[label] = {
            "layer": L,
            "next_layer": L + 1,
            "experts": list(experts),
            "routing_check": {
                "expected_experts": sorted(experts),
                "pytorch_grouped_topk_experts": sorted(ids.tolist()),
                "pass": sorted(ids.tolist()) == sorted(experts),
                "router_logits_on_case_experts": [
                    round(float(logits[e]), 3) for e in experts
                ],
                "max_router_logit_elsewhere": round(
                    float(
                        logits[
                            [e for e in range(NUM_EXPERTS) if e not in experts]
                        ].max()
                    ),
                    3,
                ),
                "routing_weights": [round(float(w), 4) for w in weights.tolist()],
            },
            "entry_state": {
                k: r[k]
                for k in (
                    "entry_pending",
                    "entry_transport",
                    "zero_allocator",
                    "mlp_output_type",
                )
            },
            "h_in_checksum": float(r["h_in"].double().sum()),
        }
        assert r["entry_pending"] is None, r["entry_pending"]
    info["moe_output_buffer_ctx_calls_per_forward"] = moe_buffers

    # ---- the timed calls ----
    fb = new_fb()
    recorder = get_global_expert_distribution_recorder()
    calls, resets, static, inputs = {}, {}, {}, {}
    for label, L, _ in CASES:
        r = rec[L]
        layer, nxt = layers[L], layers[L + 1]
        h_in, residual = r["h_in"].clone(), r["residual"].clone()
        h_in0, residual0 = h_in.clone(), residual.clone()
        moe_buf = torch.empty_like(h_in)
        inputs[label] = (h_in0, residual0)

        def call(
            label=label,
            L=L,
            layer=layer,
            nxt=nxt,
            h_in=h_in,
            residual=residual,
            moe_buf=moe_buf,
        ):
            fb.residual_stream = ResidualStream(
                residual
            )  # the stream as ffn_boundary.prepare leaves it
            buf_ctx = (
                moe_base.moe_output_buffer_ctx(moe_buf)
                if use_moe_buffer
                else contextlib.nullcontext()
            )
            with recorder.with_current_layer(L):
                with layer.ffn_boundary.exit(fb) as ffn_exit, buf_ctx:
                    h = layer.mlp(h_in, fb, None)
                h = ffn_exit.finish(h)
            with recorder.with_current_layer(L + 1):
                out = nxt.attn_boundary.prepare(
                    h,
                    fb,
                    capture_gathered=None,
                    quant_format=nxt._resolve_gfx95_quant_format(),
                )
            static[label] = (out, fb.residual_stream.residual)

        def reset(h_in=h_in, residual=residual, h_in0=h_in0, residual0=residual0):
            h_in.copy_(h_in0)
            residual.copy_(residual0)

        calls[label], resets[label] = call, reset

    # Captured as SGLang's decode graph runner captures: model_capture_mode + graph_capture() (custom
    # all-reduce registration, capture stream), warm-up calls on the capture stream first.
    graphs = {}
    with model_ctx(fb), model_capture_mode():
        with graph_capture() as gctx:
            for label, call in calls.items():
                for _ in range(3):
                    resets[label]()
                    call()
                sync()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=gctx.stream):
                    call()
                graphs[label] = g
        sync()

    # ---- numerics (a): graph replay vs SGLang's eager forward; (b): vs FP32 reference ----
    correctness = {}
    for label, L, experts in CASES:
        resets[label]()
        graphs[label].replay()
        sync()
        out, res_out = static[label]
        r = rec[L]
        c = {
            "output_vs_sglang_forward": compare(torch, out, r["out"]),
            "residual_vs_sglang_forward": compare(torch, res_out, r["res_out"]),
        }
        h_in0, residual0 = inputs[label]
        gamma = layers[L + 1].input_layernorm.weight

        def block_reference(
            exclude=(),
            L=L,
            experts=experts,
            h_in0=h_in0,
            residual0=residual0,
            gamma=gamma,
        ):
            part = moe_partial_reference(
                torch,
                raw[L],
                experts,
                h_in0,
                layers[L].mlp.gate.weight,
                bias,
                up_first,
                exclude,
            )
            total = part["partial"].clone()
            dist.all_reduce(
                total, group=tp.device_group
            )  # the four ranks' partial outputs
            total = total + residual0.float().flatten()
            return (
                total * torch.rsqrt(total.square().mean() + RMS_EPS) * gamma.float(),
                total,
                part,
            )

        y, total, part = block_reference()
        # this rank's partial output and shared-expert output from SGLang's eager MoE (before the all-reduce)
        with model_ctx(fb):
            fb.residual_stream = ResidualStream(residual0.clone())
            mo = layers[L].mlp(h_in0.clone(), fb, None, return_moe_output=True)
            sg_shared, sg_partial = mo.shared.clone(), mo.get_merged().clone()
        sync()
        c["sglang_rank_partial_vs_fp32_reference"] = compare(
            torch, sg_partial, part["partial"]
        )
        c["sglang_rank_shared_expert_vs_fp32_reference"] = compare(
            torch, sg_shared, part["shared"]
        )
        if opts.get("dump_dir"):
            torch.save(
                {
                    "raw": raw[L],
                    "h_in": h_in0,
                    "residual": residual0,
                    "gamma": gamma.detach().clone(),
                    "gate_w": layers[L].mlp.gate.weight.detach().clone(),
                    "bias": bias,
                    "out": out.clone(),
                    "res_out": res_out.clone(),
                    "sg_shared": sg_shared,
                    "sg_partial": sg_partial,
                    "ref_partial": part["partial"],
                    "ref_shared": part["shared"],
                    "experts": experts,
                },
                Path(opts["dump_dir"]) / f"{label}_rank{tp_rank}.pt",
            )
        c["output_vs_fp32_reference"] = compare(torch, out, y)
        c["residual_vs_fp32_reference"] = compare(torch, res_out, total)
        c["reference_routed_experts"] = part["routed_ids"]
        c["reference_missing_experts"] = part["missing"]
        yc, _, _ = block_reference(exclude=(experts[-1],))
        c["control_output_vs_reference_without_lowest_weight_expert"] = compare(
            torch, out, yc
        )
        c["pass"] = (
            c["output_vs_sglang_forward"]["rel_l2"] < 1e-2
            and c["residual_vs_sglang_forward"]["rel_l2"] < 1e-2
            and c["output_vs_fp32_reference"]["rel_l2"] < 1e-2
            and c["residual_vs_fp32_reference"]["rel_l2"] < 1e-2
            and c["sglang_rank_partial_vs_fp32_reference"]["rel_l2"] < 2e-2
            and not part["missing"]
            and c["control_output_vs_reference_without_lowest_weight_expert"]["rel_l2"]
            > max(0.02, 5 * c["output_vs_fp32_reference"]["rel_l2"])
            and cases_out[label]["routing_check"]["pass"]
        )
        c["reference"] = (
            "(a) the next layer's attention input and residual inside SGLang's eager forward of the decode "
            "token (identical inputs); (b) FP32: DeepSeek-V3 grouped top-8 routing of the bf16 router logits, "
            "NVFP4-dequantized experts (raw weights from realistic_init) with emulated NVFP4 activation "
            "quantization (global scale 1/input_scale, FP8 block scales, E2M1 round-to-nearest), routed "
            "weights x 2.5 + shared expert, summed over the 4 ranks, + residual, RMSNorm(eps 1e-6); control: "
            "the same reference without the case's lowest-weight expert must disagree (>5x the error). "
            "Also this rank's partial (pre-all-reduce) output and shared-expert output of SGLang's eager MoE vs "
            "the reference's"
        )
        correctness[label] = c
        if tp_rank == 0:
            print(
                f"{label}: correctness {json.dumps({k: v for k, v in c.items() if k != 'reference'})}",
                flush=True,
            )

    # ---- timing: this process's one coldl2 / profiler session ----
    align = symm_mem.empty(16, dtype=torch.bfloat16, device=dev).zero_()
    symm_mem.rendezvous(align, tp.device_group.group_name)
    flusher = coldl2.L2Flusher(dev)
    group_name = tp.device_group.group_name

    def flush_and_align():
        flusher()
        torch.ops.symm_mem.one_shot_all_reduce(align, "sum", group_name)

    flush_and_align()
    sync()
    uuid = "GPU-" + str(torch.cuda.get_device_properties(dev).uuid)
    pids = gather(os.getpid())
    uuids = set(gather(uuid))
    allowed = set(pids) | {os.getppid()}
    deadline = time.time() + float(os.environ.get("MAX_WAIT", 3600))
    while True:  # every rank waits until none of the four GPUs runs a foreign process
        busy = any(gather(foreign_apps({uuid}, allowed)))
        if not busy:
            break
        if time.time() > deadline:
            raise RuntimeError("GPUs stayed shared; not timing")
        if tp_rank == 0:
            print(
                f"[{time.strftime('%H:%M:%S')}] a foreign process is on the GPUs; waiting 30 s",
                flush=True,
            )
        time.sleep(30)
    dist.barrier(group=tp.cpu_group)
    timings = coldl2.time_cold_many(
        {label: g.replay for label, g in graphs.items()},
        reps=opts["reps"],
        warmup=opts["warmup"],
        graph=False,
        flusher=flush_and_align,
        resets=resets,
    )
    others = foreign_apps(uuids, allowed)
    all_others = sorted({o for lst in gather(others) for o in lst})
    if tp_rank == 0:
        for label, t in timings.items():
            print(
                f"{label}: rank-0 median {t['median_us']:.2f} us (p10 {t['p10_us']:.2f}, p90 "
                f"{t['p90_us']:.2f})"
                + (
                    f"; OTHER PROCESSES ON THE GPUS: {all_others}" if all_others else ""
                ),
                flush=True,
            )

    # ---- kernel trace check (later profiler sessions: names and shapes only) ----
    singles = {}
    for label, g in graphs.items():
        resets[label]()
        singles[label] = profile_kernels(g.replay, sync)
    real = profile_kernels(lambda: mr.forward(new_fb()), sync)
    real_streams = {s: [kernel_id(k) for k in ks] for s, ks in by_stream(real).items()}
    trace = {}
    for label, single in singles.items():
        streams = by_stream(single)
        main_stream = single[-1]["stream"]
        per_stream = []
        for s, ks in streams.items():
            ids_ = [kernel_id(k) for k in ks]
            occ = sum(count_occurrences(seq, ids_) for seq in real_streams.values())
            # every MoE block followed by another layer (the last block ends in the final norm, which may or
            # may not reuse the same fused kernel)
            expected = NUM_LAYERS - 1
            per_stream.append(
                {
                    "role": "main (ends with the all-reduce + RMSNorm)"
                    if s == main_stream
                    else "side (alt stream)",
                    "kernels": [
                        f"grid={k['grid']} block={k['block']}  {short(k['name'], 140)}"
                        for k in ks
                    ],
                    "occurrences_in_real_decode": occ,
                    "expected_at_least": expected,
                    "pass": occ >= expected,
                }
            )
        trace[label] = {
            "pass": all(p["pass"] for p in per_stream),
            "match_key": "per CUDA stream: contiguous kernel sequence, name + grid + block + shared memory",
            "streams": per_stream,
            "num_kernels": len(single),
        }
    real_desc = [
        f"stream={k['stream']} grid={k['grid']} {short(k['name'], 120)}" for k in real
    ]
    if tp_rank == 0:
        for label, t in trace.items():
            print(
                f"{label}: trace check {t['pass']}: "
                + "; ".join(
                    f"{p['role'].split()[0]} {len(p['kernels'])} kernels x{p['occurrences_in_real_decode']}"
                    for p in t["streams"]
                ),
                flush=True,
            )

    rank_result = {
        "rank": tp_rank,
        "pid": os.getpid(),
        "gpu": coldl2.gpu_info(),
        "gpu_uuid": uuid,
        "timings": timings,
        "correctness": correctness,
        "trace_pass": {label: t["pass"] for label, t in trace.items()},
        "h_in_checksums": {label: c["h_in_checksum"] for label, c in cases_out.items()},
    }
    ranks = gather(rank_result)
    if tp_rank != 0:
        return {}
    return {
        "info": info,
        "versions": versions,
        "cases": cases_out,
        "trace": trace,
        "real_decode_kernels": real_desc,
        "ranks": ranks,
        "other_processes_during_timing": all_others,
        "align": "torch symm_mem one_shot_all_reduce of 16 bf16 on SGLang's TP group",
    }


# ----------------------------------------------------------------------------------------------------
# Child (one_batch.main launcher) and parent (retries, aggregation)
# ----------------------------------------------------------------------------------------------------
def server_cli(args) -> list[str]:
    return [
        "--model-path",
        str(MODELS / "deepseek_v3_fp4"),
        "--load-format",
        "dummy",
        "--skip-tokenizer-init",
        "--tokenizer-path",
        LLAMA_TOKENIZER,  # any tokenizer: the prompt is synthetic token ids
        "--tp-size",
        str(WORLD_SIZE),
        "--json-model-override-args",
        json.dumps({"num_hidden_layers": NUM_LAYERS, "first_k_dense_replace": 0}),
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--batch-size",
        "1",
        "--input-len",
        str(args.input_len),
        "--output-len",
        "2",
        *args.extra_server_args,
    ]


def child_main(args) -> None:
    from sglang.benchmark import one_batch as ob
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ob.BenchArgs.add_cli_args(parser)
    ns = parser.parse_args(server_cli(args))
    server_args = ServerArgs.from_cli_args(ns)
    bench_args = ob.BenchArgs.from_cli_args(ns)
    os.environ[OPTS_ENV] = json.dumps(
        {
            "coldl2": args.coldl2,
            "child_out": args.child_out,
            "reps": args.reps,
            "warmup": args.warmup,
            "input_len": args.input_len,
            "dump_dir": args.dump_dir,
            # killed as a group on a rank failure only when this process leads its own group (the launcher's)
            "pgid": os.getpgid(0) if os.getpgid(0) == os.getpid() else None,
        }
    )
    # one_batch.main resolves the args as `python -m sglang.benchmark.one_batch` does (decode CUDA graph for
    # batch 1) and spawns one latency_test per TP rank; run tp_worker in its place.
    ob.latency_test = tp_worker
    ob.main(server_args, bench_args)


def run_child(args, child_out: Path, log: Path) -> int:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--child-out",
        str(child_out),
        *sys.argv[1:],
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    print(f"CUDA_VISIBLE_DEVICES={args.gpus} {' '.join(cmd)}  (log {log})", flush=True)
    with open(log, "w") as f:
        proc = subprocess.Popen(
            cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(
                proc.pid, signal.SIGKILL
            )  # the child's own session: its TP workers
            proc.wait()
            return -1


NOTES = (
    "Timed: one SGLang DeepSeek-V3 NVFP4 MoE block at TP4 for one decode token, as SGLang's decode CUDA graph "
    "runs it: DeepseekV2MoE (router GEMM, FlashInfer TRT-LLM NVFP4 MoE with in-kernel DeepSeek-V3 routing, "
    "the NVFP4 shared expert on SGLang's alt stream, finalize) inside ffn_boundary.exit / ffn_exit.finish, "
    "then the next layer's attn_boundary.prepare = SGLang's fused all-reduce + residual add + RMSNorm. "
    "Model built by SGLang (one_batch.main + load_model, --tp-size 4, DeepSeek-V3 NVFP4 checkpoint config with "
    "5 layers, all MoE, dummy weights re-drawn at realistic scales, default backends). Reported time: the "
    "largest of the four ranks' medians (deepseek_tp4.py's convention)."
)


def parent_main(args) -> None:
    out_path = Path(args.out)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    child_out = scratch / f"{stem}_child.json"
    log = scratch / f"{stem}_child.log"
    gpus = [int(g) for g in args.gpus.split(",")]
    assert len(gpus) == WORLD_SIZE, args.gpus
    retries = []
    if args.aggregate_only:
        r = json.loads(child_out.read_text())
    else:
        for attempt in range(MAX_RETRIES + 1):
            wait_for_idle_gpus(gpus)
            if child_out.exists():
                child_out.unlink()
            rc = run_child(args, child_out, log)
            if rc != 0 or not child_out.exists():
                sys.exit(f"benchmark process failed (exit {rc}); see {log}")
            r = json.loads(child_out.read_text())
            if not r["other_processes_during_timing"]:
                break
            retries.append(
                {
                    "attempt": attempt,
                    "other_processes": r["other_processes_during_timing"],
                }
            )
            print(
                "another process was on the GPUs during timing; redoing the run",
                flush=True,
            )
        else:
            sys.exit(f"the GPUs stayed shared for {MAX_RETRIES + 1} attempts")
    ranks = sorted(r["ranks"], key=operator.itemgetter("rank"))
    info = r["info"]
    ar_kernels = {
        k
        for t in r["trace"].values()
        for p in t["streams"]
        if p["role"].startswith("main")
        for k in p["kernels"][-1:]
    }
    ar = (
        "FlashInfer MNNVL one-shot"
        if any("mnnvl" in k for k in ar_kernels)
        else "FlashInfer"
    )
    impl = f"SGLang (FlashInfer TRT-LLM NVFP4 MoE + {ar} all-reduce/residual/RMSNorm, decode CUDA graph)"
    rows = []
    for label, layer, experts in CASES:
        medians = [x["timings"][label]["median_us"] for x in ranks]
        loops = [x["timings"][label]["loop_us"] for x in ranks]
        slow = ranks[max(range(len(ranks)), key=lambda i: medians[i])]["timings"][label]
        c = r["cases"][label]
        rows.append(
            {
                "workload": WORKLOAD,
                "case": label,
                "impl": impl,
                "median_us": max(medians),
                "rank_medians_us": medians,
                "loop_us": max(loops),
                "rank_loop_us": loops,
                "p10_us": slow["p10_us"],
                "p90_us": slow["p90_us"],
                "reps": slow["reps"],
                "rank0": ranks[0]["timings"][label],
                "ranks": [
                    {
                        k: x["timings"][label][k]
                        for k in ("median_us", "p10_us", "p90_us", "reps")
                    }
                    for x in ranks
                ],
                "layer": layer,
                "experts": list(experts),
                "routing_check": c["routing_check"],
                "correctness": {
                    "pass": all(x["correctness"][label]["pass"] for x in ranks),
                    "per_rank": [x["correctness"][label] for x in ranks],
                },
                "kernel_trace_check": {
                    "pass": all(x["trace_pass"][label] for x in ranks),
                    **r["trace"][label],
                },
                "entry_state": c["entry_state"],
            }
        )
    result = {
        "system": "sglang",
        "gpu": ranks[0]["gpu"],
        "gpus": [x["gpu"] for x in ranks],
        "versions": r["versions"],
        "rows": rows,
        "workload": WORKLOAD,
        "headline_mean_over_cases_us": statistics.fmean(
            row["median_us"] for row in rows
        ),
        "backends": info,
        "h_in_identical_across_ranks": {
            label: len({x["h_in_checksums"][label] for x in ranks}) == 1
            for label, _, _ in CASES
        },
        "timing": {
            "method": "coldl2.time_cold_many, one session per rank process: each case a CUDA graph captured "
            "in SGLang's graph_capture()/model_capture_mode(), the three cases interleaved; before "
            "every call: residual restored, coldl2's L2 flush, cross-rank align ("
            + r["align"]
            + "), all outside the CUPTI span (first kernel start to last kernel end, both streams)",
            "reported": "median_us = max over the 4 ranks of each rank's median; rank_medians_us = every "
            "rank's; p10/p90 of the slowest rank",
            "gpus": args.gpus,
            "gpu_check": "nvidia-smi compute apps on the 4 GPUs checked before the run and before/after the "
            "timing session; a run another process overlapped is redone",
            "redone_runs": retries,
            "reps": args.reps,
            "warmup": args.warmup,
        },
        "real_decode_kernels_rank0": r["real_decode_kernels"],
        "sglang_server_args": server_cli(args),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "notes": NOTES,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    for row in rows:
        print(
            f"{WORKLOAD} {row['case']:>9s}: {row['median_us']:.2f} us (rank medians "
            f"{', '.join(f'{m:.2f}' for m in row['rank_medians_us'])}); correctness {row['correctness']['pass']}, "
            f"trace {row['kernel_trace_check']['pass']}, routing {row['routing_check']['pass']}",
            flush=True,
        )
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="result JSON")
    ap.add_argument(
        "--gpus", default=os.environ.get("GPUS4", "0,1,2,4"), help="four GPU indices"
    )
    ap.add_argument("--reps", type=int, default=200, help="timed calls per case")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--input-len",
        type=int,
        default=63,
        help="prompt tokens before the decode token",
    )
    ap.add_argument("--mem-fraction", type=float, default=0.3)
    ap.add_argument(
        "--timeout", type=float, default=2400, help="seconds before the run is killed"
    )
    ap.add_argument(
        "--coldl2", default=os.environ.get("COLDL2_PATH", str(DEFAULT_COLDL2))
    )
    ap.add_argument(
        "--scratch", default=str(Path(SCRATCH) / "tp4"), help="child JSON and log"
    )
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="re-aggregate the child JSON in --scratch",
    )
    ap.add_argument(
        "--dump-dir", default=None, help="debug: save each case's tensors per rank here"
    )
    ap.add_argument(
        "--extra-server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="appended to the SGLang server args (must be last)",
    )
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        child_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
