# pyrefly: ignore-errors
# (it imports the megakernels package from $TK_DIR at run time)
"""ThunderKittens Megakernels (HazyResearch, low-latency Llama-3.2-1B demo) decode cost of one
L-layer megakernel on B200, cold L2; the per-layer cost is the slope over L (summarize.py).

The megakernel bakes the layer count in (LLAMA_1B_NUM_LAYERS), so build_tk.sh builds one
extension per L, and a process loads exactly one build (two builds in one process give wrong
results or hang).  The process builds --instances models with that layer count (each with its
own random bf16 weights and KV cache, the real Llama-3.2-1B shapes), checks one decode step at
position 1023 (attending over 1024 KV rows: 1023 random cached rows plus the new token's) against
the megakernels package's own PyTorch path, and times one megakernel launch per model with the
shared cold-L2 timer (coldl2.time_cold_many: an L2 flush before every launch; the barrier reset
every launch needs runs before the flush).  run_all.sh runs one process per L per round and
../mirage/summarize.py fits the slope of T(L) over L.  The launch also runs the LM head (vocab
128256, a fixed per-launch cost that cancels in the slope); the embedding lookup and the argmax
are separate PyTorch ops in TK's generate loop and are not timed.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import types
from typing import TYPE_CHECKING

import pynvml
import torch

HERE = Path(__file__).resolve().parent
if not os.environ.get("TK_DIR"):
    raise SystemExit("set TK_DIR to the Megakernels checkout (README.md)")
MK_ROOT = Path(os.environ["TK_DIR"])
COLDL2 = Path(os.environ.get("COLDL2_PY", HERE.parent / "coldl2.py"))
sys.path.insert(0, str(MK_ROOT))
from transformers import LlamaConfig  # noqa: E402  (before the accelerate stub below)

# megakernels.llama imports accelerate.init_empty_weights for from_pretrained only (unused here)
sys.modules.setdefault(
    "accelerate",
    types.SimpleNamespace(init_empty_weights=lambda **_: contextlib.nullcontext()),
)
from megakernels.demos.latency.mk import LatencyMK_Interpreter  # noqa: E402
from megakernels.dispatch import make_schedule_builder  # noqa: E402
from megakernels.llama import LlamaForCausalLM  # noqa: E402
from megakernels.model_types import BatchState  # noqa: E402
from megakernels.model_types import ExtraModelConfig  # noqa: E402
from megakernels.scheduler import assign_to_sms  # noqa: E402
from megakernels.scheduler import tensorize_instructions  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

CTX = 1024
POS = CTX - 1
MAX_LEN = 2048


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


def load_mk(so_path: Path) -> Callable[..., None]:
    """The mk_llama function of one build (several builds, one module object each)."""
    spec = importlib.util.spec_from_file_location("mk_llama", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.mk_llama


def build_model(layers: int, seed: int) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=layers,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        max_position_embeddings=131072,
        tie_word_embeddings=True,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        torch_dtype="bfloat16",
    )
    extra = ExtraModelConfig(
        interleave_rope=True, max_len_override=MAX_LEN, max_batch_size=1
    )
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    model = LlamaForCausalLM(config, extra)
    torch.set_default_dtype(torch.float32)
    model.dtype, model.device = torch.bfloat16, torch.device("cuda")
    model.requires_grad_(False)
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.ndim == 1:
                p.copy_(torch.empty_like(p).uniform_(0.8, 1.2, generator=g))
            else:
                p.copy_(torch.empty_like(p).normal_(0.0, 0.02, generator=g))
        model.lm_head.lm_head.weight.copy_(
            model.model.embed_tokens.embed_tokens.weight
        )  # tied
        model.model.interleave_rope()
    model.stack_params()
    model.setup_caches()
    with torch.no_grad():  # 1023 cached rows of random K/V (rows >= CTX stay zero)
        for c in model.stacked_kv_cache:
            c[:, :, :POS].normal_(0.0, 1.0, generator=g)
    return model


@torch.inference_mode()
def torch_logits(model: LlamaForCausalLM, token: torch.Tensor) -> torch.Tensor:
    """One decode step at position POS through megakernels' PyTorch path (writes the new KV row)."""
    st = BatchState(
        input_ids=token.view(1, 1),
        position_ids=torch.tensor([[POS]], device="cuda"),
        seq_len=POS + 1,
    )
    out = model.model(st)
    h = model.lm_head.input_norm(out.hidden_states)
    return model.lm_head.lm_head(h).float().view(-1)


class TK:
    def __init__(self, L: int, so: Path, seed: int) -> None:
        self.L = L
        self.model = build_model(L, seed)
        self.schedule = make_schedule_builder("latency").build(self.model)
        tensorize_instructions(
            self.schedule.globs, assign_to_sms("rr", schedule=self.schedule)
        )
        self.interp = LatencyMK_Interpreter.__new__(LatencyMK_Interpreter)
        self.interp.mk_func = load_mk(so)
        self.globs = self.schedule.globs
        self.globs.pos_id = POS
        self.token = torch.randint(
            0,
            128256,
            (1, 1),
            device="cuda",
            generator=torch.Generator("cuda").manual_seed(seed + 1),
        )

    def reset(self) -> None:
        self.globs.barriers.zero_()

    def launch(self) -> None:
        self.interp.interpret(self.globs)

    def mk_logits(self) -> torch.Tensor:
        """Embed (as MK_Generator.run), reset, launch; the megakernel's logits."""
        with torch.inference_mode():
            hid = self.model.model.embed_tokens(
                BatchState(input_ids=self.token)
            ).hidden_states
            self.globs.hidden_states[:] = hid.view(-1)
        self.reset()
        self.launch()
        torch.cuda.synchronize()
        return self.globs.logits.float().view(-1)[:128256].clone()

    def check(self) -> dict:
        ref = torch_logits(self.model, self.token)
        got = self.mk_logits()
        d = got - ref
        res = {
            "max_abs": d.abs().max().item(),
            "rel_l2": (d.norm() / ref.norm()).item(),
            "max_rel": (d.abs().max() / ref.abs().max()).item(),
            "cos": torch.nn.functional.cosine_similarity(got, ref, dim=0).item(),
            "argmax_match": int(got.argmax()) == int(ref.argmax()),
        }
        # run-to-run change of the megakernel's own output (its reductions are not deterministic)
        res["max_abs_change_rerun"] = (self.mk_logits() - got).abs().max().item()
        k, v = self.model.stacked_kv_cache
        with (
            torch.no_grad()
        ):  # a row past the context must not change the output (beyond rerun noise) ...
            k[:, :, CTX] += 5.0
            v[:, :, CTX] += 5.0
            res["max_abs_change_row1024_perturbed"] = (
                (self.mk_logits() - got).abs().max().item()
            )
            k[:, :, CTX] -= 5.0
            v[:, :, CTX] -= 5.0
            saved = (k[:, :, 0].clone(), v[:, :, 0].clone())  # ... row 0 must
            k[:, :, 0] += 5.0
            v[:, :, 0] += 5.0
            res["max_abs_change_row0_perturbed"] = (
                (self.mk_logits() - got).abs().max().item()
            )
            k[:, :, 0], v[:, :, 0] = saved
        return res


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
    p.add_argument(
        "--layers",
        type=int,
        required=True,
        help="the layer count of the build to load (one per process)",
    )
    p.add_argument(
        "--round",
        type=int,
        default=1,
        help="run_all.sh round (one process per L per round)",
    )
    p.add_argument(
        "--instances",
        type=int,
        default=2,
        help="models (own weights/caches) of this layer count",
    )
    p.add_argument(
        "--build-dir",
        required=True,
        help="build_tk.sh output: <dir>/L<N>/mk_llama*.so",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=100,
        help="timed rounds (every model launched once per round)",
    )
    p.add_argument("--graph", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    L = args.layers
    coldl2 = load_coldl2()
    torch.cuda.set_device(0)
    log: list = []
    so = next((Path(args.build_dir) / f"L{L}").glob("mk_llama*.so"))
    models = {i: TK(L, so, seed=100 * i + L) for i in range(args.instances)}
    correctness = {}
    for i, m in models.items():
        correctness[f"L{L}i{i}"] = m.check()
        print(f"L={L} i={i} correctness {correctness[f'L{L}i{i}']}", flush=True)
    # One coldl2.time_cold_many session: the models interleaved, each launch after its own barrier
    # reset (the kernel needs zeroed barriers) and L2 flush.
    calls = {f"L{L}i{i}": m.launch for i, m in models.items()}
    resets = {f"L{L}i{i}": m.reset for i, m in models.items()}
    watch = GpuWatch(log)
    watch.wait_quiet()
    watch.start()
    timing = coldl2.time_cold_many(
        calls, reps=args.reps, warmup=5, graph=bool(args.graph), resets=resets
    )
    overlap = watch.stop()
    if overlap:  # another process used the GPU during the timing session: discard (run_all retries)
        print(
            f"other processes on the GPU while timing: {overlap}; discarded", flush=True
        )
        sys.exit(3)
    after = {}
    for i, m in models.items():  # still right after the timed launches
        ref = torch_logits(m.model, m.token)
        got = m.mk_logits()
        after[f"L{L}i{i}"] = {"rel_l2": ((got - ref).norm() / ref.norm()).item()}
    inst = {}
    for i in models:
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
            f"L={L} i={i}: median {inst[f'L{L}i{i}']['median_us']:.2f} us, "
            f"loop_us {inst[f'L{L}i{i}']['loop_us']:.2f}",
            flush=True,
        )
    ins = list(inst.values())
    T = {
        L: dict(
            stats([x for d in ins for x in d["spans_us"]]),
            instance_medians_us=[d["median_us"] for d in ins],
            instance_means_us=[d["mean_us"] for d in ins],
            instance_loop_us=[d["loop_us"] for d in ins],
            loop_us=statistics.median([d["loop_us"] for d in ins]),
        )
    }
    res = {
        "system": "thunderkittens",
        "workload": "llama32_1b",
        "variant": "latency",
        "layers": L,
        "round": args.round,
        "commit": git_commit(MK_ROOT),
        "thunderkittens_commit": git_commit(MK_ROOT / "ThunderKittens"),
        "ctx_kv_rows": CTX,
        "decode_position": POS,
        "batch": 1,
        "T_by_layers": T,
        "instances": {
            k: {kk: vv for kk, vv in v.items() if kk != "spans_us"}
            for k, v in inst.items()
        },
        "correctness": correctness,
        "correctness_after_timing": after,
        "graph": bool(args.graph),
        "reps": args.reps,
        "timer": "coldl2.time_cold_many (one session, this L's models interleaved)",
        "n_instances": args.instances,
        "build": {L: str(so)},
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
        "argv": sys.argv,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spans_us": {k: v["spans_us"] for k, v in inst.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(
        f"tk llama32_1b round {args.round}: T({L}) = {T[L]['median_us']:.2f} us "
        f"(instances {', '.join(f'{d:.2f}' for d in T[L]['instance_medians_us'])}), "
        f"loop_us {T[L]['loop_us']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
