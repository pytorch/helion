# pyrefly: ignore-errors
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202
"""SGLang cold-L2 timing of ONE decoder layer (Qwen3-8B, Llama-3.2-1B), ONE Qwen3-30B-A3B MoE block or ONE
DeepSeek-V3 MLA decode attention call, batch 1 decode over 1024 KV rows, exactly as SGLang's CUDA-graph
decode runs it.

The model is built by SGLang itself: ``sglang.benchmark.one_batch.main`` resolves the server args (dummy
weights, a config with a few layers, SGLang's default attention / MoE backends for this GPU) and
``one_batch.load_model`` builds the ModelRunner (memory pools, attention backend, CUDA graphs).  A request
of CONTEXT-1 prompt tokens is prefilled through ``one_batch.extend``, then the decode batch is prepared as
``one_batch.decode`` does (``ScheduleBatch.prepare_for_decode`` + ``ForwardBatch.init_new``), so the decode
token attends over CONTEXT rows (CONTEXT-1 cached + its own).

Per-step work that is not part of a layer is done once, before timing, the way SGLang's decode graph
runner does it: the attention metadata for the CUDA-graph path (``init_forward_metadata_out_graph`` +
``init_forward_metadata_in_graph``: SGLang's ``update_trtllm_mha_graph_metadata`` kernel at the head of the
decode graph, or the trtllm-MLA block-table build), the graph runner's input copies, the embedding, the
other layers, the final norm, the LM head and sampling.

Timed call (CUDA graph, captured inside SGLang's ``model_capture_mode`` and ``forward_context`` as its
decode graph runner does), with ONE ``coldl2.time_cold_many`` session per process (coldl2 allows one):
  * layer: ``model.model.layers[i](positions, hidden_states, forward_batch)`` for a middle layer i, with
    the residual stream (``forward_batch.residual_stream``) in the state the previous layer leaves it in,
    so the layer starts with the fused add+RMSNorm of the previous layer's MLP output into the residual.
  * moe: the rest of ``Qwen3MoeDecoderLayer.forward`` after ``self_attn``: ``attn_boundary.finish`` ->
    ``ffn_boundary.prepare`` (post-attention fused add+RMSNorm) -> ``mlp`` (router GEMM, top-k, experts,
    weighted sum) -> ``ffn_boundary.finish``.
  * mla: the ``flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`` call of layer i, with the exact
    arguments SGLang's TRTLLMMLABackend passes on its CUDA-graph decode path (recorded from an eager run of
    the decode graph's body: SGLang's query/KV/block-table/seq-len buffers, max_seq_len, scale, workspace,
    multi-CTA counter, enable_pdl).  Query and the 1024 KV rows are refilled with N(0, 1) values.  The same
    call with the arguments of SGLang's eager (non-graph) path is timed in the same session as a diagnostic.
Inputs the call modifies in place (the hidden/residual pair of the fused add+RMSNorm) are restored before
every timed call (coldl2 ``resets``, ahead of the L2 flush), so every call sees identical inputs (and
identical MoE routing).

Validation in the same process, after the timing session (later profiler sessions are only used for
kernel names and launch shapes, never for times):
  (a) kernel trace: the real decode step (``ModelRunner.forward``, which replays SGLang's decode CUDA
      graph) is profiled; the timed call's kernel sequence (name, grid, block, shared memory) must appear
      contiguously in it once per layer (layers 1..L-1 for a full layer: layer 0 starts with a plain RMSNorm).
  (b) numerics (before timing): layer/moe: the timed call's outputs are compared with the same layer's
      outputs inside an eager full forward of the model (forward hooks), from identical inputs.  mla: FP32
      PyTorch attention over the same paged KV rows.

usage: sglang_layer.py --workload {qwen3_8b,llama32_1b,qwen3_30b_moe,dsv3_mla} [--processes 1] [--gpu 5]
       [--out JSON] [--aggregate-only]
Run in the SGLang env (``source ../env.sh; sglang_python sglang_layer.py ...``); see README.md.
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
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
DEFAULT_COLDL2 = HERE.parent / "coldl2.py"
MODELS = Path(os.environ.get("SGLANG_MODELS_DIR", HERE / "models"))
SGLANG_DIR = Path(os.environ.get("SGLANG_DIR", Path.home() / "local/sglang"))
LLAMA_TOKENIZER = os.environ.get(
    "LLAMA_TOKENIZER", str(Path.home() / "local/hvm_envs/ae_e2e/llama-3.2-1b")
)
SCRATCH = os.environ.get("SGLANG_BENCH_SCRATCH", "sglang_scratch")
# Retries of a process during whose timing session another process appeared on the GPU.
MAX_RETRIES = 3

# DeepSeek-V3 (public HF config), bf16, as one TP8 rank sees its attention: 128 / 8 = 16 heads.  Every
# layer dense (first_k_dense_replace) so no MoE is built; the MLA decode call does not depend on the MLP.
DSV3_TP8_RANK_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "aux_loss_alpha": 0.001,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "ep_size": 1,
    "first_k_dense_replace": 1000,
    "hidden_act": "silu",
    "hidden_size": 7168,
    "initializer_range": 0.02,
    "intermediate_size": 18432,
    "kv_lora_rank": 512,
    "max_position_embeddings": 163840,
    "model_type": "deepseek_v3",
    "moe_intermediate_size": 2048,
    "moe_layer_freq": 1,
    "n_group": 8,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 16,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 4,
    "num_key_value_heads": 16,
    "num_nextn_predict_layers": 1,
    "pretraining_tp": 1,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "seq_aux": True,
    "tie_word_embeddings": False,
    "topk_group": 4,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "v_head_dim": 128,
    "vocab_size": 129280,
}

WORKLOADS = {
    "qwen3_8b": {
        "name": "qwen3_8b_layer",
        "model": "qwen3_8b_bf16",
        "kind": "layer",
        "desc": "Qwen3-8B decoder layer, batch 1, bf16, 1024 KV rows",
    },
    "llama32_1b": {
        "name": "llama32_1b_layer",
        "model": "llama3_2_1b",
        "kind": "layer",
        "tokenizer": LLAMA_TOKENIZER,
        "desc": "Llama-3.2-1B decoder layer, batch 1, bf16, 1024 KV rows",
    },
    "qwen3_30b_moe": {
        "name": "qwen3_30b_a3b_moe",
        "model": "qwen3_30b_a3b",
        "kind": "moe",
        "desc": "Qwen3-30B-A3B MoE block (post-attention fused add+RMSNorm, router, top-8 of 128 experts, "
        "experts, weighted sum), batch 1, bf16",
    },
    "dsv3_mla": {
        "name": "deepseek_v3_mla",
        "model": None,  # written from DSV3_TP8_RANK_CONFIG
        "kind": "mla",
        "tokenizer": LLAMA_TOKENIZER,  # any tokenizer: the prompt is synthetic token ids (< 10000)
        "desc": "DeepSeek-V3 absorbed MLA decode attention call, 16 heads (TP8 rank), 1 query token, "
        "576-wide keys / 512-wide values, bf16 KV in 64-token pages, 1024 KV rows",
    },
}


def load_coldl2(path: Path):
    spec = importlib.util.spec_from_file_location("coldl2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------
# GPU hygiene: wait until no other process uses the GPU; flag a timing session another process overlapped.
# ----------------------------------------------------------------------------------------------------
def gpu_uuid(index: int) -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in out.strip().splitlines():
        i, uuid = (s.strip() for s in line.split(","))
        if int(i) == index:
            return uuid
    raise RuntimeError(f"no GPU with index {index}")


def gpu_users(index: int, ignore_pid: int | None = None) -> list[str]:
    """nvidia-smi compute processes on GPU ``index`` (other than ``ignore_pid``)."""
    uuid = gpu_uuid(index)
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
    lines = [
        line.strip()
        for line in out.strip().splitlines()
        if line.strip().startswith(uuid)
    ]
    return [line for line in lines if int(line.split(",")[1]) != ignore_pid]


def wait_for_idle_gpu(
    index: int, poll_s: float = 30.0, ignore_pid: int | None = None
) -> None:
    while users := gpu_users(index, ignore_pid):
        print(
            f"[{time.strftime('%H:%M:%S')}] GPU {index} busy ({users}); waiting {poll_s:.0f} s",
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
    """Every device kernel ``fn()`` runs, in start order: name, start/duration (us), grid, block, smem."""
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
            "grid": e["args"].get("grid"),
            "block": e["args"].get("block"),
            "smem": e["args"].get("shared memory"),
        }
        for e in kernels
    ]


def kernel_id(k: dict) -> str:
    return f"{k['name']} grid={k['grid']} block={k['block']} smem={k['smem']}"


def find_occurrences(seq: list[str], pattern: list[str]) -> list[int]:
    n = len(pattern)
    return [i for i in range(len(seq) - n + 1) if seq[i : i + n] == pattern]


def short(name: str, width: int = 110) -> str:
    return name if len(name) <= width else name[: width - 3] + "..."


# ----------------------------------------------------------------------------------------------------
# Child process: build the model with SGLang, validate, time.
# ----------------------------------------------------------------------------------------------------
def child_main(args) -> None:
    import torch

    coldl2 = load_coldl2(Path(args.coldl2))
    wl = WORKLOADS[args.workload]
    if wl["model"] is None:
        model_dir = Path(args.scratch) / "dsv3_bf16_tp8rank"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            json.dumps(DSV3_TP8_RANK_CONFIG, indent=1)
        )
    else:
        model_dir = MODELS / wl["model"]
    server_cli = [
        "--model-path",
        str(model_dir),
        "--load-format",
        "dummy",
        "--dtype",
        "bfloat16",
        "--skip-tokenizer-init",
        "--json-model-override-args",
        json.dumps({"num_hidden_layers": args.layers}),
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--batch-size",
        "1",
        "--input-len",
        str(args.context - 1),
        "--output-len",
        "2",
    ]
    if wl.get("tokenizer"):
        server_cli += ["--tokenizer-path", wl["tokenizer"]]
    server_cli += args.extra_server_args

    from sglang.benchmark import one_batch as ob
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ob.BenchArgs.add_cli_args(parser)
    ns = parser.parse_args(server_cli)
    server_args = ServerArgs.from_cli_args(ns)
    bench_args = ob.BenchArgs.from_cli_args(ns)

    result: dict = {"server_cli": server_cli}

    def work(server_args, port_args, bench_args, gpu_id, tp_rank):
        # == one_batch.latency_test's prologue ==
        from sglang.srt.runtime_context import SpawnRanks
        from sglang.srt.runtime_context import publish
        from sglang.srt.runtime_context import spawn_world_rank
        from sglang.srt.utils import configure_logger

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
        runner, _ = ob.load_model(server_args, port_args, gpu_id, tp_rank)
        result.update(run_benchmark(args, wl, ob, runner, coldl2, torch))

    # one_batch.main resolves the args exactly as `python -m sglang.benchmark.one_batch` does (decode CUDA
    # graph for batch 1) and then calls latency_test; run our work function in its place.
    ob.latency_test = work
    ob.main(server_args, bench_args)
    Path(args.child_out).write_text(json.dumps(result, indent=1))


def rescale_weights(model, torch) -> None:
    """Re-draw SGLang's dummy weights (uniform +-1e-3, norm weights included, which makes every activation
    ~1e-11) at realistic scales, in place: matrices uniform with unit output variance (bound
    sqrt(3 / fan_in), fan_in = last dim), 1-D weights (norm gammas) uniform in [0.8, 1.2].  Only values
    change: shapes, layouts, dtypes, kernels and CUDA graphs are SGLang's."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if not torch.is_floating_point(p) or p.numel() == 0:
                continue
            if p.dim() == 1:
                p.uniform_(0.8, 1.2, generator=gen)
            else:
                bound = (3.0 / p.shape[-1]) ** 0.5
                p.uniform_(-bound, bound, generator=gen)


def compare(torch, a, b) -> dict:
    a, b = a.float(), b.float()
    return {
        "max_abs": (a - b).abs().max().item(),
        "ref_max_abs": b.abs().max().item(),
        "rel_l2": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "bitwise_equal": bool(torch.equal(a, b)),
    }


def run_benchmark(args, wl, ob, runner, coldl2, torch) -> dict:
    import flashinfer
    import numpy as np
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_context import ForwardContext
    from sglang.srt.model_executor.forward_context import forward_context
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
    from sglang.srt.runtime_context import get_model
    from sglang.srt.runtime_context import get_schedule
    import triton

    kind = wl["kind"]
    mr = runner.torch_runner
    model = mr.model
    sync = torch.cuda.synchronize
    np.random.seed(0)  # noqa: NPY002 (any NumPy sampling inside SGLang)
    if kind != "mla" and not args.sglang_dummy_weights:
        rescale_weights(model, torch)
    context = args.context

    # ---- warm-up exactly like one_batch.latency_test (JIT / autotune everything once) ----
    reqs = ob.prepare_synthetic_inputs_for_latency_test(1, context - 1)
    ob.latency_test_run_once(
        "warmup",
        runner,
        lambda *a, **k: None,
        reqs,
        1,
        context - 1,
        4,
        log_decode_step=0,
        profile=False,
        profile_record_shapes=False,
        profile_activities=("CPU", "GPU"),
        profile_prefix="",
        profile_stage="all",
        tp_rank=0,
    )

    attn = mr.attn_backend
    view = resolving_view(mr.server_args)
    info = {
        "attention_backend": type(attn).__name__,
        "server_args": {
            k: str(getattr(view, k, None))
            for k in (
                "attention_backend",
                "decode_attention_backend",
                "prefill_attention_backend",
                "moe_runner_backend",
                "moe_a2a_backend",
                "page_size",
                "kv_cache_dtype",
                "cuda_graph_config",
            )
        },
        "page_size": get_schedule().page_size,
        "decode_cuda_graph_runner": type(mr.decode_cuda_graph_runner).__name__,
        "decode_graph_backend": type(
            getattr(mr.decode_cuda_graph_runner, "backend", None)
        ).__name__,
        "model_class": type(model).__name__,
        "num_hidden_layers": args.layers,
        "timed_layer_index": args.layer_index,
        "weights": (
            "SGLang dummy (uniform +-1e-3); the timed call's query/KV refilled with N(0,1)"
            if kind == "mla"
            else "SGLang dummy (uniform +-1e-3)"
            if args.sglang_dummy_weights
            else "SGLang dummy re-drawn at realistic scales (rescale_weights)"
        ),
    }
    versions = {
        "sglang_commit": sglang_commit(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "flashinfer": flashinfer.__version__,
    }
    li = args.layer_index
    layer = model.model.layers[li]
    if kind == "moe":
        experts = layer.mlp.experts
        info["moe"] = {
            "experts_class": type(experts).__name__,
            "runner_backend": str(getattr(experts, "runner_backend", None)),
            "quant_method": type(getattr(experts, "quant_method", None)).__name__,
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

    def graph_metadata(fbx):
        """The decode graph runner's per-step attention metadata, for the graph's static buffers."""
        attn.init_forward_metadata_out_graph(fbx)
        attn.init_forward_metadata_in_graph(fbx)
        if hasattr(attn, "decode_cuda_graph_metadata") and hasattr(
            attn, "forward_decode_metadata"
        ):
            # trtllm-MLA reads the metadata object its graph was captured with
            attn.forward_decode_metadata = attn.decode_cuda_graph_metadata[
                fbx.batch_size
            ]
        sync()

    case = {"context_rows": context}
    # ---- prefill CONTEXT-1 tokens, then the decode step (one_batch.decode without the forward) ----
    runner.clear()
    reqs = ob.prepare_synthetic_inputs_for_latency_test(1, context - 1)
    next_token_ids, _, batch = runner.extend(reqs)
    batch.input_ids = next_token_ids.to(torch.int64)
    batch.prepare_for_decode()
    ob._maybe_prepare_mlp_sync_batch(batch, mr)

    def new_fb():
        return ForwardBatch.init_new(batch, mr, return_hidden_states_before_norm=False)

    assert int(new_fb().seq_lens[0]) == context

    # The real decode step (ModelRunner.forward replays SGLang's decode CUDA graph), unprofiled for now.
    for _ in range(3):
        out = mr.forward(new_fb())
        sync()
    assert out.can_run_graph, "SGLang did not run its decode CUDA graph"

    saved = []
    diag_call = None
    if kind in ("layer", "moe"):
        # ---- eager full forward with hooks: the reference inputs/outputs of layer i ----
        rec: dict = {}

        def stream_state(fbx):
            s = fbx.residual_stream
            return {
                "residual": None if s.residual is None else s.residual.clone(),
                "pending": s.pending,
                "transport": s.transport,
                "update": None if s.pending is None else s.pending.update,
            }

        def post(mod, a, kw, output):
            rec["out"] = output.clone()
            rec["out_state"] = stream_state(a[2])

        hooks = [layer.register_forward_hook(post, with_kwargs=True)]
        if kind == "layer":

            def pre(mod, a, kw):
                rec["in"] = stream_state(a[2])
                rec["in_hidden"] = a[1].clone()

            hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        else:

            def attn_post(mod, a, kw, output):
                rec["attn_out"] = output.clone()
                rec["in"] = stream_state(
                    kw.get("forward_batch", a[2] if len(a) > 2 else None)
                )

            hooks.append(
                layer.self_attn.register_forward_hook(attn_post, with_kwargs=True)
            )
        fb_eager = new_fb()
        with model_ctx(fb_eager):
            mr.eager_runner.execute(fb_eager)
        sync()
        for h in hooks:
            h.remove()
        st = rec["in"]
        case["residual_stream_at_entry"] = {
            "pending": None if st["pending"] is None else type(st["pending"]).__name__,
            "owed": None if st["pending"] is None else repr(st["pending"].owed),
            "update": None if st["update"] is None else type(st["update"]).__name__,
            "transport": repr(st["transport"]),
        }
        print(
            "residual stream at the timed call's entry:",
            case["residual_stream_at_entry"],
            flush=True,
        )

        fb = new_fb()
        positions = fb.positions
        if kind == "layer":
            assert (
                st["pending"] is not None
                and st["pending"].owed is None
                and st["transport"] is None
            )
            hidden_in = rec["in_hidden"].clone()
            residual = st["residual"].clone()
            update = st["update"]
            saved = [(hidden_in, hidden_in.clone()), (residual, residual.clone())]

            def call():
                h, fb.residual_stream = ResidualStream.from_handoff(
                    hidden_in, residual, update
                )
                return layer(positions, h, fb)
        else:
            assert st["pending"] is None and st["transport"] is None
            attn_out = rec["attn_out"].clone()
            residual = st["residual"].clone()
            saved = [(attn_out, attn_out.clone()), (residual, residual.clone())]

            def call():
                fb.residual_stream = ResidualStream(residual)
                h = layer.attn_boundary.finish(attn_out, fb)
                h = layer.ffn_boundary.prepare(h, fb)
                h = layer.mlp(h, fb)
                return layer.ffn_boundary.finish(h, fb)

        def check(y):
            c = {"layer_output_vs_in_forward": compare(torch, y, rec["out"])}
            if (
                fb.residual_stream is not None
                and fb.residual_stream.residual is not None
            ):
                c["residual_vs_in_forward"] = compare(
                    torch, fb.residual_stream.residual, rec["out_state"]["residual"]
                )
            c["pass"] = all(v["rel_l2"] < 1e-2 for v in c.values())
            c["reference"] = (
                "same layer inside an eager full forward of the SGLang model (forward "
                "hooks), identical inputs"
            )
            return c
    else:
        # ---- mla: record the decode kernel's arguments from an eager run of the decode graph's body ----
        import flashinfer.decode as fi_decode

        orig = fi_decode.trtllm_batch_decode_with_kv_cache_mla
        calls = []

        def recorder(*a, **kw):
            o = orig(*a, **kw)
            calls.append((a, kw, o))
            return o

        def record(run):
            calls.clear()
            fi_decode.trtllm_batch_decode_with_kv_cache_mla = recorder
            try:
                run()
            finally:
                fi_decode.trtllm_batch_decode_with_kv_cache_mla = orig
            sync()
            assert len(calls) == args.layers, len(calls)
            a, kw, _ = calls[li]
            assert not a, "expected keyword arguments"
            calls.clear()
            return kw

        fb = new_fb()

        def graph_body():
            with model_ctx(fb), model_capture_mode():
                graph_metadata(fb)
                model.forward(fb.input_ids, fb.positions, fb)

        kw = record(graph_body)
        # Diagnostic only: the arguments of SGLang's eager (no CUDA graph) decode path, which plans
        # max_seq_len = the batch's longest sequence instead of the model context length.
        fb_eager = new_fb()

        def eager_forward():
            with model_ctx(fb_eager):
                mr.eager_runner.execute(fb_eager)

        kw_eager = record(eager_forward)
        q, kv, bt, sl = kw["query"], kw["kv_cache"], kw["block_tables"], kw["seq_lens"]
        n = int(sl[0])
        assert n == context, n
        blocks = math.ceil(n / kv.shape[2])
        page_ids = bt[0, :blocks].long()
        assert torch.equal(kw_eager["block_tables"][0, :blocks].long(), page_ids)
        assert kw_eager["kv_cache"].data_ptr() == kv.data_ptr()
        gen = torch.Generator(device="cuda")
        gen.manual_seed(400)
        q.copy_(torch.randn(q.shape, generator=gen, device="cuda", dtype=torch.float32))
        kw_eager["query"].copy_(q)
        kv[page_ids] = torch.randn(
            (blocks, *tuple(kv.shape[1:])),
            generator=gen,
            device="cuda",
            dtype=torch.float32,
        ).to(kv.dtype)

        def describe(kws):
            return {
                k: (
                    f"tensor{tuple(v.shape)} {v.dtype}"
                    if isinstance(v, torch.Tensor)
                    else repr(v)
                )
                for k, v in kws.items()
            }

        case["mla_call"] = describe(kw)
        case["mla_call"]["block_tables_valid_pages"] = blocks
        case["mla_call_eager_path_diagnostic"] = describe(kw_eager)
        print("SGLang MLA decode call (CUDA-graph path):", case["mla_call"], flush=True)
        print(
            "SGLang MLA decode call (eager path, diagnostic):",
            case["mla_call_eager_path_diagnostic"],
            flush=True,
        )

        def call():
            return orig(**kw)

        def diag_call():
            return orig(**kw_eager)

        def check(y):
            kv_rows = kv[page_ids, 0].reshape(-1, kv.shape[-1])[:n].float()
            qf = q.reshape(-1, q.shape[-2], q.shape[-1]).float()  # [tokens, heads, 576]
            s = torch.einsum("thd,ld->thl", qf, kv_rows) * kw["bmm1_scale"]
            ref = torch.einsum(
                "thl,ld->thd", s.softmax(-1), kv_rows[:, : kw["kv_lora_rank"]]
            )
            res = compare(torch, y.reshape(ref.shape), ref)
            return {
                "output_vs_fp32_reference": res,
                "pass": res["rel_l2"] < 1e-2,
                "reference": f"FP32 PyTorch softmax attention over the same {n} paged KV rows",
            }

    def restore():
        for dst, src in saved:
            dst.copy_(src)

    outs = {}

    def call_store():
        outs["y"] = call()

    diagnostics = {}
    with model_ctx(fb), model_capture_mode():
        if kind != "mla":
            graph_metadata(
                fb
            )  # per-step metadata for the graph path (excluded from the timed call)
        md = getattr(attn, "forward_metadata", None)
        if md is not None and getattr(md, "cache_seqlens_int32", None) is not None:
            case["metadata_cache_seqlens"] = md.cache_seqlens_int32.tolist()
        # numerics: one replay of the captured call from the reference inputs
        restore()
        g = coldl2.graph_of(call_store)
        restore()
        g.replay()
        sync()
        correctness = check(outs["y"])
        print("correctness:", correctness, flush=True)
        timed = {"call": call}
        if diag_call is not None:
            diagnostics["mla_eager_path_args"] = {
                "output_vs_graph_path_call": compare(torch, diag_call(), outs["y"])
            }
            timed["eager_path_args"] = diag_call
        # ---- timing: this process's one coldl2 / profiler session ----
        wait_for_idle_gpu(args.gpu, ignore_pid=os.getpid())
        timings = coldl2.time_cold_many(
            timed, reps=args.reps, warmup=args.warmup, resets={"call": restore}
        )
        others = gpu_users(args.gpu, ignore_pid=os.getpid())
        t = timings["call"]
        print(
            f"ctx {context}: median {t['median_us']:.2f} us (p10 {t['p10_us']:.2f}, p90 {t['p90_us']:.2f})"
            + (f"; OTHER PROCESSES ON THE GPU: {others}" if others else ""),
            flush=True,
        )
        if diag_call is not None:
            d = timings["eager_path_args"]
            diagnostics["mla_eager_path_args"].update(
                {
                    "what": "same kernel call with the arguments of SGLang's eager (non-CUDA-graph) decode path "
                    "(max_seq_len = longest sequence, block table sized to it), timed interleaved in "
                    "the same session; not the headline",
                    "median_us": d["median_us"],
                    "p10_us": d["p10_us"],
                    "p90_us": d["p90_us"],
                }
            )
        # ---- kernel lists for the trace check (later profiler sessions: names and shapes only) ----
        restore()
        single = profile_kernels(g.replay, sync)
        del g
        if diag_call is not None:
            diagnostics["mla_eager_path_args"]["kernels"] = [
                f"grid={k['grid']} {short(k['name'], 120)}"
                for k in profile_kernels(diag_call, sync)
            ]
    real = profile_kernels(lambda: mr.forward(new_fb()), sync)

    # ---- trace check ----
    real_ids = [kernel_id(k) for k in real]
    single_ids = [kernel_id(k) for k in single]
    occ = find_occurrences(real_ids, single_ids)
    expected = args.layers - 1 if kind == "layer" else args.layers
    covered = {j for i in occ for j in range(i, i + len(single_ids))}
    case["kernel_trace_check"] = {
        "pass": len(occ) == expected,
        "match_key": "kernel name + grid + block + shared memory",
        "timed_call_kernels": [
            f"grid={k['grid']} block={k['block']}  {short(k['name'], 140)}"
            for k in single
        ],
        "real_decode_num_kernels": len(real_ids),
        "occurrences_in_real_decode": len(occ),
        "expected_occurrences": expected,
        "real_decode_kernels": [
            ("  [timed-call match]  " if j in covered else "  [outside]          ")
            + f"grid={k['grid']} {short(k['name'], 120)}"
            for j, k in enumerate(real)
        ],
        "summary": (
            f"the timed call's {len(single_ids)}-kernel sequence appears contiguously {len(occ)}x "
            f"(expected {expected}) in the real decode step (ModelRunner.forward -> decode CUDA "
            f"graph replay, {len(real_ids)} kernels); matched on name+grid+block+smem"
        ),
    }
    print(case["kernel_trace_check"]["summary"], flush=True)
    case.update(
        {
            "median_us": t["median_us"],
            "loop_us": t["loop_us"],
            "p10_us": t["p10_us"],
            "p90_us": t["p90_us"],
            "reps": t["reps"],
            "spans_us": t["spans_us"],
            "correctness": correctness,
            "diagnostics": diagnostics,
        }
    )
    return {
        "info": info,
        "versions": versions,
        "case": case,
        "other_processes_during_timing": others,
        "gpu": coldl2.gpu_info(),
    }


# ----------------------------------------------------------------------------------------------------
# Parent: run N child processes (one coldl2 session each), aggregate, write the result JSON.
# ----------------------------------------------------------------------------------------------------
NOTES = {
    "layer": (
        "Timed: one SGLang decoder layer call (layers[i](positions, hidden, forward_batch)) = fused "
        "add+RMSNorm of the previous layer's MLP output into the residual, QKV GEMM, q/k norm (Qwen3), "
        "RoPE, KV-cache write, trtllm-gen paged decode attention over 1024 rows, o_proj, post-attention "
        "fused add+RMSNorm, gate/up GEMM with fused SiLU*up, down GEMM (+ split-K reduce). Excluded (per "
        "decode step, not per layer): the decode graph's attention-metadata kernel "
        "(update_trtllm_mha_graph_metadata), graph input copies, embedding, final norm, LM head, sampling."
    ),
    "moe": (
        "Timed: the part of one SGLang Qwen3MoeDecoderLayer.forward after self_attn: attn_boundary.finish, "
        "post-attention fused add+RMSNorm (ffn_boundary.prepare), router GEMM, routing (softmax top-8, "
        "renormalized), experts with SGLang's default MoE runner on this GPU, weighted sum (finalize), "
        "ffn_boundary.finish. The residual add of the MoE output happens in the next layer's fused "
        "add+RMSNorm (not timed)."
    ),
    "mla": (
        "Timed: one flashinfer trtllm_batch_decode_with_kv_cache_mla call with the arguments SGLang's "
        "TRTLLMMLABackend passes in its CUDA-graph decode (max_seq_len = the model context length, "
        "block table padded to it, as the graph path does; enable_pdl = is_arch_support_pdl()). Excluded: "
        "the fused KV-write + q-concat kernel (set_mla_kv_concat_q) SGLang runs before it, the block-table "
        "build (per step). The model is DeepSeek-V3's config with 16 attention heads (one TP8 rank) and "
        "all-dense layers, built by SGLang on one GPU."
    ),
}


def parent_main(args) -> None:
    wl = WORKLOADS[args.workload]
    out_path = Path(args.out)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    per_proc = []
    retries = []
    for p in range(args.processes):
        child_out = scratch / f"{stem}_proc{p}.json"
        log = scratch / f"{stem}_proc{p}.log"
        if args.aggregate_only:  # re-aggregate the per-process JSONs of an earlier run
            per_proc.append(json.loads(child_out.read_text()))
            continue
        for attempt in range(MAX_RETRIES + 1):
            wait_for_idle_gpu(args.gpu)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--child-out",
                str(child_out),
            ]
            cmd += sys.argv[1:]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
            print(
                f"process {p}: CUDA_VISIBLE_DEVICES={args.gpu} {' '.join(cmd)}  (log {log})",
                flush=True,
            )
            with open(log, "w") as f:
                rc = subprocess.run(
                    cmd, env=env, stdout=f, stderr=subprocess.STDOUT
                ).returncode
            if rc != 0:
                sys.exit(f"process {p} failed (exit {rc}); see {log}")
            r = json.loads(child_out.read_text())
            if not r["other_processes_during_timing"]:
                break
            retries.append(
                {
                    "process": p,
                    "attempt": attempt,
                    "median_us": r["case"]["median_us"],
                    "other_processes": r["other_processes_during_timing"],
                }
            )
            print(
                f"process {p}: another process was on the GPU during timing; redoing it",
                flush=True,
            )
        else:
            sys.exit(
                f"process {p}: the GPU stayed shared for {MAX_RETRIES + 1} attempts"
            )
        c = r["case"]
        print(
            f"process {p}: median {c['median_us']:.2f} us, p10 {c['p10_us']:.2f}, p90 {c['p90_us']:.2f}; "
            f"trace check {c['kernel_trace_check']['pass']}, correctness {c['correctness']['pass']}",
            flush=True,
        )
        per_proc.append(r)
    first = per_proc[0]
    info = first["info"]
    sa = info["server_args"]
    if wl["kind"] == "layer":
        impl = f"SGLang ({sa['attention_backend']} attention, decode CUDA graph)"
    elif wl["kind"] == "moe":
        impl = f"SGLang ({info['moe']['runner_backend'].split('.')[-1].lower()} MoE)"
    else:
        impl = f"SGLang ({sa['attention_backend']}: FlashInfer trtllm-gen MLA decode, PDL on)"
    cs = [r["case"] for r in per_proc]
    c0 = cs[0]
    spans = sorted(s for c in cs for s in c["spans_us"])
    row = {
        "workload": wl["name"],
        "case": "b1" if wl["kind"] == "moe" else f"ctx{c0['context_rows']}",
        "impl": impl,
        # headline: median over processes of each process's median (one coldl2 session per process)
        "median_us": statistics.median(c["median_us"] for c in cs),
        # profiler-free cross-check (coldl2), same combination over processes
        "loop_us": statistics.median(c["loop_us"] for c in cs),
        "p10_us": spans[len(spans) // 10],
        "p90_us": spans[(len(spans) * 9) // 10],
        "reps": len(spans),
        "spans_us": [round(x, 3) for c in cs for x in c["spans_us"]],
        "processes": [
            {
                "median_us": c["median_us"],
                "loop_us": c["loop_us"],
                "p10_us": c["p10_us"],
                "p90_us": c["p90_us"],
                "reps": c["reps"],
            }
            for c in cs
        ],
        "context_rows": c0["context_rows"],
        "kernel_trace_check": {
            "pass": all(c["kernel_trace_check"]["pass"] for c in cs),
            **{k: v for k, v in c0["kernel_trace_check"].items() if k != "pass"},
        },
        "correctness": {
            "pass": all(c["correctness"]["pass"] for c in cs),
            "per_process": [c["correctness"] for c in cs],
        },
        "diagnostics": [c["diagnostics"] for c in cs],
    }
    row.update(
        {
            k: c0[k]
            for k in (
                "residual_stream_at_entry",
                "metadata_cache_seqlens",
                "mla_call",
                "mla_call_eager_path_diagnostic",
            )
            if k in c0
        }
    )
    result = {
        "system": "sglang",
        "gpu": first["gpu"],
        "versions": first["versions"],
        "rows": [row],
        # headline repeated at the top level
        "workload": wl["name"],
        "description": wl["desc"],
        "commit": first["versions"]["sglang_commit"],
        "backends": info,
        "median_us": row["median_us"],
        "p10_us": row["p10_us"],
        "p90_us": row["p90_us"],
        "processes": row["processes"],
        "kernel_trace_check": {
            "pass": row["kernel_trace_check"]["pass"],
            "summary": row["kernel_trace_check"]["summary"],
        },
        "correctness": {
            "pass": row["correctness"]["pass"],
            "summary": row["correctness"]["per_process"][0],
        },
        "timing": {
            "method": "coldl2.time_cold_many, one session per process: CUDA graph of the call, "
            "coldl2's L2 flush before every call (512 MiB int32 copied to an int64 buffer, "
            "then summed), CUPTI span from the call's first kernel start to its last kernel "
            "end; loop_us: coldl2's profiler-free cross-check",
            "gpu_index": args.gpu,
            "gpu_check": "nvidia-smi compute apps checked before each process and before/after its "
            "timing session; a process another process overlapped is redone",
            "redone_processes": retries,
            "reps_per_process": args.reps,
            "warmup": args.warmup,
            "processes": args.processes,
            "headline": "median over processes of the per-process medians; p10/p90 over all spans",
        },
        "gpu_per_process": [r["gpu"] for r in per_proc],
        "sglang_server_args": first["server_cli"],
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([sys.executable, *sys.argv]),
        "notes": NOTES[wl["kind"]]
        + " Model built by SGLang (one_batch.main + one_batch.load_model, dummy "
        "weights, default backends); see README.md.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    print(
        f"wrote {out_path} [{row['workload']}/{row['case']}]: median {row['median_us']:.2f} us "
        f"(p10 {row['p10_us']:.2f}, p90 {row['p90_us']:.2f}); per-process medians "
        f"{[round(c['median_us'], 2) for c in cs]}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--out", required=True, help="result JSON")
    ap.add_argument(
        "--processes",
        type=int,
        default=1,
        help="separate processes (the repeat dimension: one coldl2 session per process)",
    )
    ap.add_argument("--gpu", type=int, default=int(os.environ.get("GPU", 5)))
    ap.add_argument(
        "--context",
        type=int,
        default=1024,
        help="KV rows the decode token attends over",
    )
    ap.add_argument(
        "--layers", type=int, default=4, help="num_hidden_layers of the SGLang model"
    )
    ap.add_argument(
        "--layer-index", type=int, default=2, help="layer whose call is timed"
    )
    ap.add_argument("--reps", type=int, default=200, help="timed calls per process")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--mem-fraction", type=float, default=0.3)
    ap.add_argument(
        "--sglang-dummy-weights",
        action="store_true",
        help="keep SGLang's dummy weights (+-1e-3) instead of re-drawing them at realistic scales",
    )
    ap.add_argument(
        "--coldl2", default=os.environ.get("COLDL2_PATH", str(DEFAULT_COLDL2))
    )
    ap.add_argument("--scratch", default=SCRATCH, help="per-process JSONs and logs")
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="only re-aggregate the per-process JSONs already in --scratch",
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
