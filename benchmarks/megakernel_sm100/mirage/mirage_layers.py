# pyrefly: ignore-errors
# (it imports Mirage and its demo models from $MIRAGE_HOME at run time)
"""Mirage MPK per-layer decode cost on B200, cold L2, by layer differencing.

One process builds the same workload with L = 1, 2, 4, 8 layers, --instances times each (every
layer with its own weights and its own KV cache), runs one launch per model that computes
exactly one decode step at a context of 1024 KV rows (1023 cached rows plus the new token's
row), checks the outputs against PyTorch, and times every model in one session of the shared
cold-L2 timer (benchmarks/megakernel_sm100/coldl2.py time_cold_many: models interleaved, an L2
flush before every launch, CUPTI span of the launch's kernels, CUDA-graph replay).  The
per-layer cost is the least-squares slope of T(L) over L through every (L, instance median):
the fixed per-launch work (launch, MPK scheduler start-up, embedding lookup, MPK's termination,
which alone varies by ~+-20 us between identical builds) cancels.

Workloads (batch 1, bf16):
  qwen3_8b       Qwen3-8B decoder layer (demo/qwen3/demo.py's graph; split-K o_proj/down on sm100)
  llama32_1b     Llama-3.2-1B decoder layer (variant builder: the same graph without q/k norm;
                 variant demo: demo/llama3/demo.py's graph with residual linears)
  qwen3_30b_moe  Qwen3-30B-A3B MoE block (demo/qwen3/demo_30B_A3B.py's layer calls)
  dsv3_mla       DeepSeek-V3 absorbed MLA decode, 16 heads (TP8 shard), one query token:
                 mla_mtp_decode_tp8 + its reduce (models/deepseek_v3/builder.py's TP8 path)

Needs a Mirage checkout with mirage_mpk_bench.patch applied and built (see README.md), and
MPK_EXTRA_NVCC_FLAGS=-DMPK_BENCH_DECODE=1 (set here by default).
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import TYPE_CHECKING

HERE = Path(__file__).resolve().parent
if not os.environ.get("MIRAGE_HOME"):
    raise SystemExit(
        "set MIRAGE_HOME to the Mirage mpk checkout with mirage_mpk_bench.patch (README.md)"
    )
MIRAGE_HOME = Path(os.environ["MIRAGE_HOME"]).resolve()
# Mirage's compile() takes its headers from MIRAGE_HOME: it must be the imported checkout.
os.environ["MIRAGE_HOME"] = str(MIRAGE_HOME)
os.environ.setdefault("MPK_EXTRA_NVCC_FLAGS", "-DMPK_BENCH_DECODE=1")
sys.path.insert(0, str(MIRAGE_HOME / "python"))
sys.path.insert(1, str(MIRAGE_HOME / "demo" / "qwen3"))

import pynvml  # noqa: E402
import torch  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable
    import types

COLDL2 = Path(os.environ.get("COLDL2_PY", HERE.parent / "coldl2.py"))
CTX = 1024  # KV rows the decode step attends over (the new token is row CTX - 1)
RMS_EPS = (
    1e-6  # Mirage's rmsnorm task hard-codes 1e-6 (task_register.cc rmsnorm_hopper)
)


def load_coldl2() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("coldl2", COLDL2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_commit(path: Path) -> dict:
    def run(*cmd: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *cmd], capture_output=True, text=True
        ).stdout.strip()

    return {
        "path": str(path),
        "commit": run("rev-parse", "HEAD"),
        "describe": run("log", "-1", "--format=%h %ci %s"),
        "dirty_files": run("status", "--short", "--untracked-files=no").splitlines(),
    }


class GpuWatch:
    """Other processes on this GPU (matched by UUID, via NVML). wait_quiet() waits until none has
    been seen for quiet_s; between start() and stop() a thread polls every period_s and records
    every other process seen, so that a timing session that overlapped one can be discarded."""

    def __init__(self, log: list, quiet_s: float = 30, period_s: float = 0.5) -> None:
        pynvml.nvmlInit()
        uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
        self.handle = next(h for h in handles if pynvml.nvmlDeviceGetUUID(h) == uuid)
        self.log, self.quiet_s, self.period_s = log, quiet_s, period_s
        self.seen: dict[int, int] = {}
        self.polls = 0
        self._stop = threading.Event()

    def others(self) -> dict[int, int]:
        return {
            p.pid: p.usedGpuMemory
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            if p.pid != os.getpid()
        }

    def wait_quiet(self, max_wait_s: float = 4 * 3600) -> None:
        t0 = last_busy = time.time()
        last_log = 0.0
        while True:
            busy = self.others()
            now = time.time()
            if busy:
                last_busy = now
                if now - last_log > 300:
                    self.log.append(
                        f"{time.strftime('%H:%M:%S')} waiting, GPU busy: {busy}"
                    )
                    print(self.log[-1], flush=True)
                    last_log = now
            elif now - last_busy >= self.quiet_s:
                return
            if now - t0 > max_wait_s:
                raise SystemExit("GPU stayed busy")
            time.sleep(1)

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            self.polls += 1
            self.seen.update(self.others())

    def start(self) -> None:
        self.seen.update(self.others())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[int, int]:
        self._stop.set()
        self._thread.join()
        self.seen.update(self.others())
        return self.seen


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(
        x.dtype
    )


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


def errors(got: torch.Tensor, ref: torch.Tensor) -> dict:
    d = got.float() - ref.float()
    return {
        "max_abs": d.abs().max().item(),
        "rel_l2": (d.norm() / ref.float().norm()).item(),
        "max_rel": (d.abs().max() / ref.float().abs().max()).item(),
    }


class Model:
    """One compiled MPK launcher: launch() runs one decode step after reset()."""

    def __init__(
        self,
        mpk: object,
        meta: dict,
        check: Callable[[], dict],
        layers: int,
        info: dict,
    ) -> None:
        self.mpk, self.meta, self.check, self.layers, self.info = (
            mpk,
            meta,
            check,
            layers,
            info,
        )
        mod = (
            mpk.launch_func.__self__
        )  # the launcher extension module compile() imported
        # RTLD_NOLOAD: the already-loaded launcher; RTLD_LOCAL keeps each launcher's symbols private.
        self.lib = ctypes.CDLL(mod.__file__, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
        self.init_request = mod.init_request_func
        self.lib.mpk_bench_set_steps(ctypes.c_int(1))  # one decode step per launch

    def reset(self) -> None:
        """Mirage's init_request_resources: request 0 not started, page queue full (host-synchronous)."""
        self.init_request()

    def launch(self) -> None:
        self.mpk()

    def set_steps(self, n: int) -> None:
        self.lib.mpk_bench_set_steps(ctypes.c_int(n))


def compile_mpk(mpk: object, out_dir: Path) -> None:
    """mpk.compile(), but the launcher .so stays at out_dir/so/ instead of a temporary directory.

    compile() builds test.cpython-*.so in a TemporaryDirectory that is deleted on return.  The
    next model's .so can then get the deleted file's inode, and dlopen (which matches loaded
    libraries by device/inode) hands back the FIRST launcher: several models in one process
    would all run the first one.  Persistent, distinct files avoid that."""

    class KeptDir:
        def __init__(self, *a: object, **k: object) -> None:
            self.name = str(out_dir / f"so_pid{os.getpid()}")
            Path(self.name).mkdir(parents=True, exist_ok=True)

    saved = tempfile.TemporaryDirectory
    tempfile.TemporaryDirectory = KeptDir
    try:
        mpk.compile(output_dir=str(out_dir))
    finally:
        tempfile.TemporaryDirectory = saved


def new_mpk(
    mi: types.ModuleType,
    *,
    max_seq: int,
    kv_plan: object,
    max_num_pages: int,
    meta: dict,
) -> object:
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    return mi.PersistentKernel(
        mode="offline",
        world_size=1,
        mpi_rank=0,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=max_seq,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=max_num_pages,
        kv_groups=kv_plan.group_specs(),
        eos_token_id=-1,
        meta_tensors=meta,
        profiler_tensor=None,
        trace_name="",
        spec_decode_config=None,
        use_cutlass_kernel=True,
        test_mode=False,
    )


def base_meta(max_seq: int, kv_plan: object) -> dict:
    dev = "cuda"
    return {
        "step": torch.zeros(1, dtype=torch.int32, device=dev),
        "tokens": torch.zeros((1, max_seq), dtype=torch.long, device=dev),
        "input_tokens": torch.zeros((1, 1), dtype=torch.long, device=dev),
        "output_tokens": torch.zeros((1, 1), dtype=torch.long, device=dev),
        "num_new_tokens": torch.ones(1, dtype=torch.int32, device=dev),
        "prompt_lengths": torch.full((1,), CTX, dtype=torch.int32, device=dev),
        "qo_indptr_buffer": torch.empty(2, dtype=torch.int32, device=dev),
        **kv_plan.build_meta_tensors(
            max_num_batched_requests=1, max_seq_length=max_seq
        ),
    }


# --------------------------------------------------------------------------- dense layers
DENSE = {
    "qwen3_8b": {
        "H": 4096,
        "I": 12288,
        "QH": 32,
        "KVH": 8,
        "HD": 128,
        "vocab": 151936,
        "theta": 1_000_000.0,
        "qk_norm": True,
    },
    "llama32_1b": {
        "H": 2048,
        "I": 8192,
        "QH": 32,
        "KVH": 8,
        "HD": 64,
        "vocab": 128256,
        "theta": 500_000.0,
        "qk_norm": False,
    },
}


def grid_rmsnorm_linear_builder(size: int) -> int:
    """mirage.mpk.models.utils.grid_for_rmsnorm_linear_layer (the maintained builder, = demo/qwen3)."""
    from mirage.mpk.models.utils import grid_for_rmsnorm_linear_layer

    return grid_for_rmsnorm_linear_layer(size)


def grid_rmsnorm_linear_llama_demo(size: int) -> int:
    """demo/llama3/demo.py grid_for_rmsnorm_linear_layer."""
    if size / 96 > 400:
        return size // 256
    for g in (96, 80, 64, 48):
        if size % g == 0 and (size // g >= 128 or size // g == 64):
            return g
    raise ValueError(size)


def build_dense(
    mi: types.ModuleType,
    workload: str,
    variant: str,
    L: int,
    args: argparse.Namespace,
    inst: int = 0,
) -> Model:
    from mirage.mpk.kv_planner import resolve_pool_size
    from mirage.mpk.models.utils import grid_for_splitk_linear_layer
    from models.configuration_qwen3 import Qwen3Config
    from models.modeling_qwen3 import Qwen3ForCausalLM
    from models.modeling_qwen3 import plan_qwen3_kv_cache

    d = DENSE[workload]
    H, I, QH, KVH, HD = d["H"], d["I"], d["QH"], d["KVH"], d["HD"]  # noqa: E741
    # variant: <linear>[-splitkv]; linear = builder (split-K o_proj/down on sm100, models/utils grids)
    # or demo (llama3 demo: residual linears, its own grids).  demo/qwen3 == builder for Qwen3.
    linear = variant.split("-")[0]
    split_kv = variant.endswith("-splitkv")
    if split_kv and not d["qk_norm"]:
        # paged_attention_split_kv_layer has no enable_qk_norm switch: it applies q/k RMSNorm whenever
        # norm tensors are given, and it needs them (MPK rejects None inputs).
        raise SystemExit(
            "Mirage mpk's split-KV attention always applies q/k norm: not usable for Llama"
        )
    grid_fn = (
        grid_rmsnorm_linear_llama_demo
        if linear == "demo"
        else grid_rmsnorm_linear_builder
    )
    max_seq, page_size = args.max_seq, args.page_size
    torch.manual_seed(1000 + 100 * inst + L)
    config = Qwen3Config(
        vocab_size=d["vocab"],
        hidden_size=H,
        intermediate_size=I,
        num_hidden_layers=L,
        num_attention_heads=QH,
        num_key_value_heads=KVH,
        head_dim=HD,
        rms_norm_eps=RMS_EPS,
        rope_theta=d["theta"],
        max_position_embeddings=max(max_seq, 4096),
        torch_dtype="bfloat16",
    )
    kv_plan = plan_qwen3_kv_cache(config, 1, page_size)
    max_num_pages = resolve_pool_size(
        kv_plan,
        kv_budget=None,
        max_num_pages=math.ceil(max_seq / page_size) + 1,
        max_seq_length=max_seq,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        device=0,
        verbose=False,
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = Qwen3ForCausalLM(config, 1, max_num_pages, page_size, kv_plan=kv_plan)
    torch.set_default_dtype(prev_dtype)
    model.lm_head = (
        None  # unused (the graph ends at the last layer); frees ~1 GB per model
    )
    torch.cuda.empty_cache()
    kv_plan = model.model.kv_plan
    kv_plan.max_num_pages = max_num_pages
    layers = model.model.layers
    with torch.no_grad():
        for prm in model.parameters():
            prm.normal_(0.0, 0.02)
        for lyr in layers:
            for norm in (
                lyr.input_layernorm,
                lyr.post_attention_layernorm,
                lyr.self_attn.q_norm,
                lyr.self_attn.k_norm,
            ):
                norm.weight.uniform_(0.8, 1.2)
        for view in model.model.kv_cache:
            view.normal_(0.0, 1.0)
    page_size = kv_plan.groups[0].block_size
    positions = torch.arange(max_seq, device="cuda").unsqueeze(0)
    cos, sin = model.model.rotary_emb(positions)
    cos, sin = cos[0].contiguous(), sin[0].contiguous()

    meta = base_meta(max_seq, kv_plan)
    meta["tokens"].random_(0, d["vocab"])
    mpk = new_mpk(
        mi, max_seq=max_seq, kv_plan=kv_plan, max_num_pages=max_num_pages, meta=meta
    )

    def nt(dims: tuple[int, ...], name: str, dt: object = mi.bfloat16) -> object:
        return mpk.new_tensor(dims=dims, dtype=dt, name=name, io_category="cuda_tensor")

    def a(t: torch.Tensor, name: str) -> object:
        return mpk.attach_input(torch_tensor=t, name=name)

    qkv_dim = (QH + 2 * KVH) * HD
    chunks = max(1, max_seq // 256)  # demo.py num_kv_cache_chunks
    x_t = torch.zeros((1, H), dtype=torch.bfloat16, device="cuda")
    x = a(x_t, "embed_out")
    tok = a(meta["input_tokens"], "input_token")
    cos_dt, sin_dt = a(cos, "cos_position_embedding"), a(sin, "sin_position_embedding")
    rms_out = nt((1, H), "rmsnorm_out")
    attn_in = nt((1, qkv_dim), "attn_in")
    attn_out = nt((1, QH * HD), "attn_out")
    mlp_mid = nt((1, 2 * I), "mlp_mid")
    silu_out = nt((1, I), "silu_mul_out")
    if split_kv:
        lse = mpk.new_tensor(
            dims=(1, chunks * QH // KVH, KVH),
            strides=(chunks * QH, 1, chunks * QH // KVH),
            dtype=mi.float32,
            name="lse",
            io_category="cuda_tensor",
        )
        attn_tmp = mpk.new_tensor(
            dims=(1, chunks * QH // KVH * HD, KVH),
            strides=(chunks * QH, 1, chunks * QH // KVH * HD),
            dtype=mi.bfloat16,
            name="attn_out_tmp",
            io_category="cuda_tensor",
        )
    if linear == "demo":
        attn_proj_out = nt((1, H), "attn_proj_out")
        mlp_out_t = torch.zeros((1, H), dtype=torch.bfloat16, device="cuda")
        mlp_out = a(mlp_out_t, "mlp_out")
    mpk.embed_layer(
        input=tok,
        weight=a(model.model.embed_tokens.weight, "embed_tokens"),
        output=x,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        input_source=1,
    )
    for li, lyr in enumerate(layers):
        sa, mlp = lyr.self_attn, lyr.mlp
        w_qkv = mpk.shuffle_tensors(
            inputs=[
                a(sa.q_proj.weight, f"layer_{li}_q_proj"),
                a(sa.k_proj.weight, f"layer_{li}_k_proj"),
                a(sa.v_proj.weight, f"layer_{li}_v_proj"),
            ],
            shuffled_dim=0,
            num_groups=KVH,
            name=f"layer_{li}_qkv_proj",
        )
        mpk.rmsnorm_layer(
            input=x,
            weight=a(lyr.input_layernorm.weight, f"layer_{li}_input_layernorm"),
            output=rms_out,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        mpk.linear_layer(
            input=rms_out,
            weight=w_qkv,
            output=attn_in,
            grid_dim=(grid_fn(qkv_dim), 1, 1),
            block_dim=(128, 1, 1),
        )
        group_id, slot_id = kv_plan.layer_info(li)
        k_cache = a(model.model.kv_cache[0][slot_id], f"layer_{li}_k_cache")
        v_cache = a(model.model.kv_cache[1][slot_id], f"layer_{li}_v_cache")
        # Llama has no q/k norm: Mirage takes dummy norm tensors plus enable_qk_norm=False.
        qn, kn = (
            a(sa.q_norm.weight, f"layer_{li}_q_norm"),
            a(sa.k_norm.weight, f"layer_{li}_k_norm"),
        )
        if split_kv:
            mpk.paged_attention_split_kv_layer(
                input=attn_in,
                k_cache=k_cache,
                v_cache=v_cache,
                q_norm=qn,
                k_norm=kn,
                cos_pos_embed=cos_dt,
                sin_pos_embed=sin_dt,
                lse=lse,
                output=attn_tmp,
                attention_params=(QH, chunks),
                grid_dim=(1, KVH, chunks),
                block_dim=(128, 1, 1),
                group_id=group_id,
            )
            mpk.paged_attention_split_kv_merge_layer(
                lse=lse,
                output_tmp=attn_tmp,
                output=attn_out,
                attention_params=(QH, HD),
                grid_dim=(1, KVH, 1),
                block_dim=(128, 1, 1),
                group_id=group_id,
            )
        else:
            mpk.paged_attention_layer(
                input=attn_in,
                k_cache=k_cache,
                v_cache=v_cache,
                q_norm=qn,
                k_norm=kn,
                cos_pos_embed=cos_dt,
                sin_pos_embed=sin_dt,
                output=attn_out,
                enable_qk_norm=d["qk_norm"],
                grid_dim=(1, KVH, 1),
                block_dim=(128, 1, 1),
                group_id=group_id,
            )
        w_o = a(sa.o_proj.weight, f"layer_{li}_o_proj")
        if linear == "builder":
            mpk.splitk_linear_layer(
                input=attn_out,
                weight=w_o,
                output=x,
                grid_dim=grid_for_splitk_linear_layer(H, w_o.dim(1)),
                block_dim=(256, 1, 1),
            )
        else:
            mpk.linear_with_residual_layer(
                input=attn_out,
                weight=w_o,
                residual=x,
                output=attn_proj_out,
                grid_dim=(H // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            x = attn_proj_out
        ffn_tasks = grid_fn(2 * I)
        w_gu = mpk.shuffle_tensors(
            inputs=[
                a(mlp.gate_proj.weight, f"layer_{li}_gate_proj"),
                a(mlp.up_proj.weight, f"layer_{li}_up_proj"),
            ],
            shuffled_dim=0,
            num_groups=ffn_tasks // 2,
            name=f"layer_{li}_gatedup_proj",
        )
        mpk.rmsnorm_layer(
            input=x,
            weight=a(
                lyr.post_attention_layernorm.weight, f"layer_{li}_post_attn_layernorm"
            ),
            output=rms_out,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        mpk.linear_layer(
            input=rms_out,
            weight=w_gu,
            output=mlp_mid,
            grid_dim=(ffn_tasks, 1, 1),
            block_dim=(128, 1, 1),
        )
        mpk.silu_mul_layer(
            input=mlp_mid,
            output=silu_out,
            grid_dim=(ffn_tasks // 2, 1, 1),
            block_dim=(128, 1, 1),
        )
        w_d = a(mlp.down_proj.weight, f"layer_{li}_down_proj")
        if linear == "builder":
            mpk.splitk_linear_layer(
                input=silu_out,
                weight=w_d,
                output=x,
                grid_dim=grid_for_splitk_linear_layer(H, w_d.dim(1)),
                block_dim=(256, 1, 1),
            )
        else:
            mpk.linear_with_residual_layer(
                input=silu_out,
                weight=w_d,
                residual=x,
                output=mlp_out,
                grid_dim=(H // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            x = mlp_out
    compile_mpk(mpk, Path(args.build_dir) / f"{workload}_{variant}_L{L}_i{inst}")
    out_t = x_t if linear == "builder" else mlp_out_t
    tokens = meta["tokens"]
    pages = meta["paged_kv_indices_buffer_0"]

    def check(rows_delta: int = 0) -> dict:
        """Output vs PyTorch for one decode step at position CTX-1 attending over rows 0..CTX-1.
        rows_delta=-1: a wrong reference that drops row 0 (1023 rows), a negative control."""
        pos = CTX - 1
        first = -rows_delta if rows_delta < 0 else 0
        res = {}
        with torch.no_grad():
            h_x = model.model.embed_tokens.weight[tokens[0, pos]].view(1, H)
            k_err = 0.0
            for li, lyr in enumerate(layers):
                sa, mlp = lyr.self_attn, lyr.mlp
                _, slot = kv_plan.layer_info(li)
                pool_k, pool_v = (
                    model.model.kv_cache[0][slot],
                    model.model.kv_cache[1][slot],
                )
                h = rmsnorm(h_x, lyr.input_layernorm.weight)
                qkv = (
                    h
                    @ torch.cat(
                        [sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight]
                    ).T
                )
                q, k, v = qkv.split([QH * HD, KVH * HD, KVH * HD], -1)
                q, k, v = q.view(QH, HD), k.view(KVH, HD), v.view(KVH, HD)
                if d["qk_norm"]:
                    q, k = rmsnorm(q, sa.q_norm.weight), rmsnorm(k, sa.k_norm.weight)
                q, k = rope(q, cos[pos], sin[pos]), rope(k, cos[pos], sin[pos])
                rows = torch.arange(first, CTX, device="cuda")
                phys = pages[rows // page_size].long()
                K = pool_k[phys, rows % page_size].clone()  # [ctx, KVH, HD]
                V = pool_v[phys, rows % page_size].clone()
                k_err = max(
                    k_err, (K[-1].float() - k.float()).abs().max().item()
                )  # the row Mirage wrote
                K[-1], V[-1] = k, v
                qg = q.view(KVH, QH // KVH, HD).float()
                s = torch.einsum("hgd,thd->hgt", qg, K.float()) / math.sqrt(HD)
                o = (
                    torch.einsum("hgt,thd->hgd", torch.softmax(s, -1), V.float())
                    .reshape(1, QH * HD)
                    .to(torch.bfloat16)
                )
                h_x = h_x + o @ sa.o_proj.weight.T
                h2 = rmsnorm(h_x, lyr.post_attention_layernorm.weight)
                g, u = h2 @ mlp.gate_proj.weight.T, h2 @ mlp.up_proj.weight.T
                h_x = (
                    h_x
                    + (torch.nn.functional.silu(g.float()) * u.float()).to(
                        torch.bfloat16
                    )
                    @ mlp.down_proj.weight.T
                )
        res.update(errors(out_t, h_x))
        res["new_k_row_max_abs_err"] = k_err
        return res

    return Model(
        mpk,
        meta,
        check,
        L,
        {
            "page_size": page_size,
            "max_seq": max_seq,
            "max_num_pages": max_num_pages,
            "linear": linear,
            "split_kv": split_kv,
            "qkv_tasks": grid_fn(qkv_dim),
            "gate_up_tasks": grid_fn(2 * I),
        },
    )


# --------------------------------------------------------------------------- MoE block
def build_moe(
    mi: types.ModuleType,
    workload: str,
    variant: str,
    L: int,
    args: argparse.Namespace,
    inst: int = 0,
) -> Model:
    from mirage.mpk.kv_planner import KVSpec
    from mirage.mpk.kv_planner import plan_kv_groups
    from mirage.mpk.kv_planner import resolve_pool_size

    H, E, TOPK, I = 2048, 128, 8, 768  # noqa: E741
    max_seq = args.max_seq
    torch.manual_seed(2000 + 100 * inst + L)
    kv_plan = plan_kv_groups(
        [
            KVSpec(
                "attention",
                per_entry_bytes=4096,
                layer_ids=(0,),
                preferred_block_size=4096,
            )
        ]
    )
    max_num_pages = resolve_pool_size(
        kv_plan,
        kv_budget=None,
        max_num_pages=math.ceil(max_seq / 4096) + 1,
        max_seq_length=max_seq,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        device=0,
        verbose=False,
    )
    kv_plan.max_num_pages = max_num_pages
    meta = base_meta(max_seq, kv_plan)
    mpk = new_mpk(
        mi, max_seq=max_seq, kv_plan=kv_plan, max_num_pages=max_num_pages, meta=meta
    )
    bf = torch.bfloat16
    x_ts = [torch.randn(1, H, dtype=bf, device="cuda")] + [
        torch.zeros(1, H, dtype=bf, device="cuda") for _ in range(L)
    ]
    W = []
    for _ in range(L):
        W.append(
            {
                "norm": torch.rand(H, dtype=bf, device="cuda") * 0.4 + 0.8,
                "gate": torch.randn(E, H, dtype=bf, device="cuda") * 0.05,
                "w13": torch.randn(E, 2 * I, H, dtype=bf, device="cuda") * (H**-0.5),
                "w2": torch.randn(E, H, I, dtype=bf, device="cuda") * (I**-0.5),
            }
        )

    def a(t: torch.Tensor, name: str) -> object:
        return mpk.attach_input(torch_tensor=t, name=name)

    def nt(dims: tuple[int, ...], name: str, dt: object = mi.bfloat16) -> object:
        return mpk.new_tensor(dims=dims, dtype=dt, name=name, io_category="cuda_tensor")

    xs = [a(t, f"moe_x{i}") for i, t in enumerate(x_ts)]
    # intermediates shared by all layers, as demo_30B_A3B.py does
    rms_out = nt((1, H), "rmsnorm_out")
    gate_out = nt((1, E), "moe_gate_out")
    routing = nt((E, 1), "moe_routing_indices", mi.int32)
    mask = nt((E + 1,), "moe_mask", mi.int32)
    topk_w = nt((1, TOPK), "moe_topk_weight", mi.float32)
    mid = nt((1, TOPK, 2 * I), "mlp_mid")
    act = nt((1, TOPK, I), "silu_mul_out")
    eout = nt((1, TOPK, H), "mlp_out")
    for li, w in enumerate(W):
        mpk.rmsnorm_layer(
            input=xs[li],
            weight=a(w["norm"], f"layer_{li}_post_attn_layernorm"),
            output=rms_out,
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),
        )
        mpk.linear_layer(
            input=rms_out,
            weight=a(w["gate"], f"layer_{li}_moe_gate"),
            output=gate_out,
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),
        )
        mpk.moe_topk_softmax_routing_layer(
            input=gate_out,
            output=(topk_w, routing, mask),
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),
        )
        mpk.moe_w13_linear_layer(
            input=rms_out,
            weight=a(w["w13"], f"layer_{li}_gateup_proj"),
            moe_routing_indices=routing,
            moe_mask=mask,
            output=mid,
            grid_dim=(10, 2 * I // 128, 1),
            block_dim=(256, 1, 1),
        )
        mpk.moe_silu_mul_layer(
            input=mid, output=act, grid_dim=(1, TOPK, 1), block_dim=(256, 1, 1)
        )
        mpk.moe_w2_linear_layer(
            input=act,
            weight=a(w["w2"], f"layer_{li}_down_proj"),
            moe_routing_indices=routing,
            moe_mask=mask,
            output=eout,
            grid_dim=(8, 16, 1),
            block_dim=(256, 1, 1),
        )
        mpk.moe_mul_sum_add_layer(
            input=eout,
            weight=topk_w,
            residual=xs[li],
            output=xs[li + 1],
            grid_dim=(1, H // 256, 1),
            block_dim=(256, 1, 1),
        )
    compile_mpk(mpk, Path(args.build_dir) / f"{workload}_{variant}_L{L}_i{inst}")

    def check(rows_delta: int = 0) -> dict:
        worst, experts = None, []
        with torch.no_grad():
            for li, w in enumerate(W):
                x = x_ts[
                    li
                ]  # Mirage's own input to this layer (its previous layer's output)
                h = rmsnorm(x, w["norm"])
                probs = torch.softmax(h.float() @ w["gate"].float().T, -1)
                tw, ti = torch.topk(probs, TOPK, -1)
                tw = tw / tw.sum(-1, keepdim=True)
                ref = x.float().clone()
                for j in range(TOPK):
                    e = int(ti[0, j])
                    gu = (h.float() @ w["w13"][e].float().T).to(bf).float()
                    y = (torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]).to(bf).float()
                    ref[0] += tw[0, j] * (y @ w["w2"][e].float().T).to(bf).float()[0]
                err = errors(x_ts[li + 1], ref)
                experts.append(sorted(ti[0].tolist()))
                if worst is None or err["rel_l2"] > worst["rel_l2"]:
                    worst = dict(err, layer=li)
        worst["experts_per_layer"] = experts
        return worst

    return Model(
        mpk,
        meta,
        check,
        L,
        {
            "max_seq": max_seq,
            "w13_grid": [10, 2 * I // 128, 1],
            "w2_grid": [8, 16, 1],
            "router": "linear_layer grid (1,1,1) (demo default, no --splitk-gate)",
        },
    )


# --------------------------------------------------------------------------- MLA decode
def build_mla(
    mi: types.ModuleType,
    workload: str,
    variant: str,
    L: int,
    args: argparse.Namespace,
    inst: int = 0,
) -> Model:
    from mirage.mpk.kv_planner import KVSpec
    from mirage.mpk.kv_planner import plan_kv_groups
    from mirage.mpk.kv_planner import resolve_pool_size

    NH, DK, DV, PS, Q = 16, 576, 512, 64, 1
    mscale = 0.1 * math.log(40.0) + 1.0
    scale = (
        (1.0 / math.sqrt(192.0)) * mscale * mscale
    )  # the scale task_register.cc bakes in
    max_seq = args.max_seq
    torch.manual_seed(3000 + 100 * inst + L)
    kv_plan = plan_kv_groups(
        [
            KVSpec(
                "attention",
                per_entry_bytes=DK * 2,
                layer_ids=(0,),
                preferred_block_size=PS,
            )
        ]
    )
    max_num_pages = resolve_pool_size(
        kv_plan,
        kv_budget=None,
        max_num_pages=math.ceil(max_seq / PS) + 1,
        max_seq_length=max_seq,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        device=0,
        verbose=False,
    )
    kv_plan.max_num_pages = max_num_pages
    meta = base_meta(max_seq, kv_plan)
    mpk = new_mpk(
        mi, max_seq=max_seq, kv_plan=kv_plan, max_num_pages=max_num_pages, meta=meta
    )
    bf = torch.bfloat16

    def a(t: torch.Tensor, name: str) -> object:
        return mpk.attach_input(torch_tensor=t, name=name)

    # The TP8 kernels pad Q_LEN to even: the reduce writes 2 query rows of NH*DV, so every output
    # buffer holds QP = 2 rows (a [1, NH*DV] output would be overrun by 16 KB).  Layer li's query
    # is row 0 of qbuf[li] [QP, NH*DK]; layer li-1's reduce writes the first QP*NH*DV elements of
    # qbuf[li] (query 0's output = qbuf[li][0, :NH*DV]).
    QP = 2
    qbuf_ts = [torch.randn(QP, NH * DK, dtype=bf, device="cuda") for _ in range(L)]
    # 64 spare rows past max_seq so the negative control can read a 1025th row (the kernel reads kv_len rows)
    kv_ts = [torch.randn(max_seq + 64, DK, dtype=bf, device="cuda") for _ in range(L)]
    out_t = torch.zeros(QP, NH * DV, dtype=bf, device="cuda")
    qbufs = [a(t, f"layer_{i}_qbuf") for i, t in enumerate(qbuf_ts)]
    qs = [mpk.narrow(qb, 0, 0, Q) for qb in qbufs]  # [1, NH*DK] views
    kvs = [a(t, f"layer_{i}_contiguous_kv") for i, t in enumerate(kv_ts)]
    out = a(out_t, "attn_out")
    # partials shared by all layers, as builder.py does (TP8: qpg 2, one group for Q <= 2)
    num_groups, max_splits = 1, (max_seq + 127) // 128
    part_o = mpk.new_tensor(
        dims=(num_groups * max_splits, DV * 128),
        dtype=mi.bfloat16,
        name="mla_partial_o",
        io_category="cuda_tensor",
    )
    part_l = mpk.new_tensor(
        dims=(num_groups * max_splits, 128),
        dtype=mi.float32,
        name="mla_partial_lse",
        io_category="cuda_tensor",
    )
    for li in range(L):
        mpk.mla_mtp_decode_tp8_layer(qs[li], kvs[li], part_o, part_l, Q, max_seq)
        # Layer li's output is written into the first NH*DV columns of layer li+1's query, so layer
        # li+1 starts after layer li's reduce (as in a model, where o_proj..q_proj sit in between)
        # with no extra work in the chain.
        dst = (
            out if li == L - 1 else mpk.narrow(qbufs[li + 1], 1, 0, NH * DV)
        )  # [QP, NH*DV] window
        mpk.mla_mtp_decode_tp8_reduce_layer(part_o, part_l, dst, Q, max_seq)
    compile_mpk(mpk, Path(args.build_dir) / f"{workload}_{variant}_L{L}_i{inst}")

    def check(rows_delta: int = 0) -> dict:
        """Output vs softmax(q K^T) V over rows 0..CTX-1; rows_delta=+-1: wrong references (negative controls)."""
        worst = None
        with torch.no_grad():
            for li in range(L):
                kv = kv_ts[li][: CTX + rows_delta].float()
                qr = (
                    qbuf_ts[li][:Q].view(Q, NH, DK).float()
                )  # after the launch: layer li-1's output
                s = torch.einsum("qhd,td->qht", qr, kv) * scale
                ref = torch.einsum(
                    "qht,td->qhd", torch.softmax(s, -1), kv[:, :DV]
                ).reshape(Q, NH * DV)
                got = (
                    (out_t if li == L - 1 else qbuf_ts[li + 1])
                    .reshape(-1)[: Q * NH * DV]
                    .view(Q, NH * DV)
                )
                err = errors(got, ref)
                if worst is None or err["rel_l2"] > worst["rel_l2"]:
                    worst = dict(err, layer=li)
        return worst

    return Model(
        mpk,
        meta,
        check,
        L,
        {
            "max_seq": max_seq,
            "num_splits": max_splits,
            "page_size": PS,
            "kv": "contiguous [max_seq, 576] per layer (decode + reduce only, no gather)",
            "softmax_scale": scale,
        },
    )


BUILDERS = {
    "qwen3_8b": build_dense,
    "llama32_1b": build_dense,
    "qwen3_30b_moe": build_moe,
    "dsv3_mla": build_mla,
}
DEFAULT_VARIANT = {
    "qwen3_8b": "builder",
    "llama32_1b": "builder",
    "qwen3_30b_moe": "demo",
    "dsv3_mla": "tp8",
}


def lstsq_slope(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of [(x, y), ...]."""
    xs, ys = [x for x, _ in points], [y for _, y in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)


def stats(spans: list[float]) -> dict:
    s = sorted(spans)
    return {
        "median_us": statistics.median(s),
        "p10_us": s[len(s) // 10],
        "p90_us": s[(len(s) * 9) // 10],
        "mean_us": statistics.fmean(s),
        "n": len(s),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--workload", required=True, choices=sorted(BUILDERS))
    p.add_argument(
        "--variant",
        default="",
        help="dense: builder|demo[-splitkv]; default per workload",
    )
    p.add_argument("--layers", default="1,2,4,8")
    p.add_argument(
        "--instances",
        type=int,
        default=3,
        help="independently built models per layer count (MPK's per-launch termination cost varies "
        "by ~+-20 us between otherwise identical builds; the slope uses every instance)",
    )
    p.add_argument(
        "--max-seq",
        type=int,
        default=CTX,
        help="MPK max_seq_length (smallest that holds the 1024 rows)",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=4096,
        help="dense KV page size in tokens (Mirage demo default)",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=100,
        help="timed rounds (every model launched once per round)",
    )
    p.add_argument(
        "--graph", type=int, default=1, help="1: replay each launch from a CUDA graph"
    )
    p.add_argument(
        "--placement-seed",
        type=int,
        default=0,
        help="N>0: before each build allocate a random-sized (seeded by N) GPU pad, so this process's "
        "builds land at other addresses (MPK's termination cost depends on where its scheduler "
        "queues land; identical runs reproduce the same per-build values)",
    )
    p.add_argument(
        "--build-dir", required=True, help="where the megakernels are compiled"
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()
    variant = args.variant or DEFAULT_VARIANT[args.workload]
    Ls = [int(v) for v in args.layers.split(",")]
    if (
        "MPK_BENCH_DECODE"
        not in (
            MIRAGE_HOME / "include/mirage/persistent_kernel/persistent_kernel.cuh"
        ).read_text()
    ):
        raise SystemExit(f"{MIRAGE_HOME} lacks mirage_mpk_bench.patch")
    if shutil.which("nvcc") is None:
        raise SystemExit(
            "nvcc not on PATH (put CUDA 13.0's bin first on PATH, as ../env.sh does)"
        )
    coldl2 = load_coldl2()
    torch.cuda.set_device(0)
    import mirage as mi

    assert Path(mi.__file__).resolve().is_relative_to(MIRAGE_HOME), (
        mi.__file__,
        MIRAGE_HOME,
    )
    log: list = []
    keys = [(L, i) for i in range(args.instances) for L in Ls]
    models = {}
    pads = []
    rng = random.Random(args.placement_seed)
    for L, i in keys:
        if args.placement_seed:
            big = rng.randint(1, 64) * (1 << 20) + rng.randint(1, 4096) * 512
            small = rng.randint(1, 4096) * 512
            pads.extend(
                torch.empty(n, dtype=torch.uint8, device="cuda") for n in (big, small)
            )
        t0 = time.time()
        models[(L, i)] = BUILDERS[args.workload](
            mi, args.workload, variant, L, args, inst=i
        )
        print(f"built L={L} instance {i} in {time.time() - t0:.0f} s", flush=True)

    def meta_ok() -> None:
        """No launch wrote into another model's meta tensors."""
        bad = {
            f"L{L}i{i}": (
                int(m.meta["prompt_lengths"][0]),
                int(m.meta["num_new_tokens"][0]),
            )
            for (L, i), m in models.items()
            if int(m.meta["prompt_lengths"][0]) != CTX
            or int(m.meta["num_new_tokens"][0]) != 1
        }
        if bad:
            raise RuntimeError(
                f"meta tensors clobbered (prompt_lengths, num_new_tokens): {bad}"
            )

    # (a)+(b): one launch = one decode step at CTX rows, output vs PyTorch; the step counter
    # ends at CTX (the request starts at position CTX-1 with one token).
    correctness, verify = {}, {}
    for (L, i), m in models.items():
        m.reset()
        m.launch()
        torch.cuda.synchronize()
        correctness[f"L{L}i{i}"] = m.check()
        verify[f"L{L}i{i}"] = {
            "launcher": m.lib._name,
            "step_after_launch": int(m.meta["step"][0]),
            "last_page_len": int(m.meta["paged_kv_last_page_len_buffer_0"][0]),
        }
        print(
            f"L={L} i={i} correctness {correctness[f'L{L}i{i}']} verify {verify[f'L{L}i{i}']}",
            flush=True,
        )
    meta_ok()
    # cross-model interference check: relaunch all in reverse order, each still right
    for k in keys[::-1]:
        models[k].reset()
        models[k].launch()
    torch.cuda.synchronize()
    meta_ok()
    verify["correctness_reverse_order"] = {
        f"L{L}i{i}": models[(L, i)].check() for L, i in keys
    }
    m1 = models[keys[0]]
    # References over 1023 / 1025 rows must NOT match (the row count is exactly CTX).
    if args.workload != "qwen3_30b_moe":
        verify["wrong_ref_1023_rows"] = m1.check(-1)
        if args.workload == "dsv3_mla":
            verify["wrong_ref_1025_rows"] = m1.check(+1)

    # One coldl2.time_cold_many session times every model (and the steps check) interleaved: each
    # round runs every call once, each after its own reset (Mirage's request reset, host
    # synchronous) and L2 flush.  "steps1"/"steps2": the L-smallest model running 1 / 2 decode
    # steps ending at CTX rows (the request starts at CTX - n): a 2-step launch should cost about
    # one more step.
    def reset_for(m: Model, steps: int) -> Callable[[], None]:
        def reset() -> None:
            m.set_steps(steps)
            m.meta["prompt_lengths"].fill_(CTX + 1 - steps)
            m.reset()

        return reset

    calls, resets = {}, {}
    for (L, i), m in models.items():
        calls[f"L{L}i{i}"] = m.launch
        resets[f"L{L}i{i}"] = reset_for(m, 1)
    for n in (1, 2):
        calls[f"steps{n}"] = m1.launch
        resets[f"steps{n}"] = reset_for(m1, n)
    watch = GpuWatch(log)
    watch.wait_quiet()
    t0 = time.time()
    watch.start()
    timing = coldl2.time_cold_many(
        calls, reps=args.reps, warmup=5, graph=bool(args.graph), resets=resets
    )
    overlap = watch.stop()
    print(
        f"timed {len(calls)} calls x {args.reps} rounds in {time.time() - t0:.0f} s "
        f"({watch.polls} GPU polls)",
        flush=True,
    )
    if overlap:  # another process used the GPU during the timing session: discard (run_all retries)
        print(
            f"other processes on the GPU while timing: {overlap}; discarded", flush=True
        )
        os._exit(3)
    reset_for(m1, 1)()
    meta_ok()
    verify[f"launch_steps_check_L{keys[0][0]}i0"] = {
        **{f"T_steps{n}_us": timing[f"steps{n}"]["median_us"] for n in (1, 2)},
        **{f"loop_steps{n}_us": timing[f"steps{n}"]["loop_us"] for n in (1, 2)},
    }
    print(f"steps check {verify[f'launch_steps_check_L{keys[0][0]}i0']}", flush=True)
    for m in models.values():  # still right after all the replays
        m.reset()
        m.launch()
    torch.cuda.synchronize()
    meta_ok()
    after = {f"L{L}i{i}": models[(L, i)].check() for L, i in keys}

    inst = {}
    for L, i in keys:
        spans = timing[f"L{L}i{i}"]["spans_us"]
        half = len(spans) // 2
        inst[f"L{L}i{i}"] = dict(
            stats(spans),
            half_medians_us=[
                statistics.median(spans[:half]),
                statistics.median(spans[half:]),
            ],
            loop_us=timing[f"L{L}i{i}"]["loop_us"],
            spans_us=spans,
        )
        print(
            f"L={L} i={i}: median {inst[f'L{L}i{i}']['median_us']:.2f} us (p10 {inst[f'L{L}i{i}']['p10_us']:.2f}, "
            f"p90 {inst[f'L{L}i{i}']['p90_us']:.2f}), loop_us {inst[f'L{L}i{i}']['loop_us']:.2f}",
            flush=True,
        )
    T = {}
    for L in Ls:
        ins = [inst[f"L{L}i{i}"] for i in range(args.instances)]
        T[L] = dict(
            stats([x for d in ins for x in d["spans_us"]]),
            instance_medians_us=[d["median_us"] for d in ins],
            instance_means_us=[d["mean_us"] for d in ins],
            instance_loop_us=[d["loop_us"] for d in ins],
            loop_us=statistics.median([d["loop_us"] for d in ins]),
        )
    per_layer = lstsq_slope([(L, inst[f"L{L}i{i}"]["median_us"]) for L, i in keys])
    res = {
        "system": "mirage_mpk",
        "workload": args.workload,
        "variant": variant,
        "commit": git_commit(MIRAGE_HOME),
        "ctx_kv_rows": CTX,
        "decode_position": CTX - 1,
        "batch": 1,
        "per_layer_us": per_layer,
        "per_layer_method": "least-squares slope over every (L, instance median) point",
        "per_layer_us_from_means": lstsq_slope(
            [(L, inst[f"L{L}i{i}"]["mean_us"]) for L, i in keys]
        ),
        "per_layer_us_from_pooled_medians": lstsq_slope(
            [(L, T[L]["median_us"]) for L in Ls]
        ),
        "per_layer_loop_us": lstsq_slope(
            [(L, inst[f"L{L}i{i}"]["loop_us"]) for L, i in keys]
        ),
        "half_slopes_us": [
            lstsq_slope([(L, inst[f"L{L}i{i}"]["half_medians_us"][h]) for L, i in keys])
            for h in (0, 1)
        ],
        "pairwise_us": (T[Ls[-1]]["median_us"] - T[Ls[0]]["median_us"])
        / (Ls[-1] - Ls[0]),
        "T_by_layers": T,
        "instances": {
            k: {kk: vv for kk, vv in v.items() if kk != "spans_us"}
            for k, v in inst.items()
        },
        "correctness": correctness,
        "correctness_after_timing": after,
        "verify": verify,
        "build": {f"L{L}i{i}": models[(L, i)].info for L, i in keys},
        "graph": bool(args.graph),
        "reps": args.reps,
        "timer": "coldl2.time_cold_many (one session, all models interleaved)",
        "n_instances": args.instances,
        "placement_seed": args.placement_seed,
        "gpu": coldl2.gpu_info(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_wait_log": log,
        "gpu_watch": {
            "polls_during_timing": watch.polls,
            "period_s": watch.period_s,
            "quiet_s_before_timing": watch.quiet_s,
            "other_processes_during_timing": watch.seen,
        },
        "env": {
            k: os.environ.get(k)
            for k in ("MIRAGE_HOME", "MPK_EXTRA_NVCC_FLAGS", "CUDA_HOME")
        },
        "argv": sys.argv,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spans_us": {k: v["spans_us"] for k, v in inst.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(
        f"{args.workload}/{variant}: per-layer slope {per_layer:.2f} us (from means {res['per_layer_us_from_means']:.2f}, "
        f"loop_us {res['per_layer_loop_us']:.2f}); "
        + ", ".join(
            f"T({L})={T[L]['median_us']:.2f} [{', '.join(f'{v:.1f}' for v in T[L]['instance_medians_us'])}]"
            for L in Ls
        ),
        flush=True,
    )
    os._exit(0)  # skip Mirage's finalizers (several launchers in one process)


if __name__ == "__main__":
    main()
