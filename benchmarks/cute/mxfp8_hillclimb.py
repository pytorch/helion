"""Cold autotune and paired verification for the 24 K3 MXFP8 shapes (M=8 replaces the original M=7 cases).

One shape and one CuTe configuration per process. All providers share live
E4M3FN data and unswizzled E8M0 bytes, including graph replay mutations.
DeepGEMM's required scale packing is inside every timed invocation.

Example (the kernel-library checkout must also be on PYTHONPATH)::

    CUDA_VISIBLE_DEVICES=1 python benchmarks/cute/mxfp8_hillclimb.py \
        --shape-index 21 --mode autotune \
        --result-file artifacts/<run>/cold-shape21.json
    CUDA_VISIBLE_DEVICES=1 python benchmarks/cute/mxfp8_hillclimb.py \
        --shape-index 21 --mode verify --config-file artifacts/<run>/winner.json \
        --result-file artifacts/<run>/verify-shape21.json
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import sys
import time
import traceback
from typing import TYPE_CHECKING
from typing import Any
import warnings

import torch

import helion
import helion.language as hl
from helion.runtime import cute_cuda_graph
from helion.runtime.cute_structural_config import CuteStructuralConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator

SHAPES = tuple(
    (m, n, k)
    for m in (1, 8, 16, 32, 128, 1024)
    for n, k in ((1792, 5120), (8192, 1280), (2048, 4096), (5120, 2048))
)
ROOT = Path(__file__).resolve().parents[2]


def decode(data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    factors = torch.exp2((scale.view(torch.uint8).int() - 127).to(dtype))
    return data.to(dtype) * factors.repeat_interleave(32, dim=1)


def reference(
    a: torch.Tensor, w: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
) -> torch.Tensor:
    """Independent unrounded FP64 oracle; not a measured provider."""
    return decode(a, sa, torch.float64) @ decode(w, sb, torch.float64).T


def torch_dequant(
    a: torch.Tensor, w: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
) -> torch.Tensor:
    return (decode(a, sa, torch.float32) @ decode(w, sb, torch.float32).T).bfloat16()


@helion.kernel(backend="cute", autotune_effort="full", autotune_budget_seconds=None)
def native_mxfp8(
    a: torch.Tensor, w: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n, k2 = w.shape
    assert k == k2 and k % 32 == 0
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = hl.dot_scaled(
            a[tile_m, :],
            sa[tile_m, :],
            "e4m3",
            w[tile_n, :].T,
            sb[tile_n, :],
            "e4m3",
        ).to(out.dtype)
    return out


def source(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n")


def initialize_tool_path() -> list[str]:
    """Find executable scripts for the active Python and campaign dependencies."""
    from repro.baselines.gb300 import gemm  # pyrefly: ignore[missing-import]

    campaign_scripts = (
        Path(inspect.getfile(gemm)).resolve().parents[2]
        / ".venv-gb300-main-20261012"
        / "bin"
    )
    directories = [str(Path(sys.executable).parent)]
    if campaign_scripts.is_dir() and str(campaign_scripts) not in directories:
        directories.append(str(campaign_scripts))
    os.environ["PATH"] = os.pathsep.join([*directories, os.environ.get("PATH", "")])
    return directories


def provenance() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda": torch.version.cuda,
        "helion_source": source(Path(helion.__file__).resolve()),
        "helion_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "helion_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(ROOT), "diff", "HEAD"])
        ).hexdigest(),
        "harness_source": source(Path(__file__).resolve()),
        "gpu": props.name,
        "gpu_uuid": str(props.uuid),
        "architecture": f"sm_{props.major}{props.minor}",
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_start": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,temperature.gpu,power.limit,power.draw,clocks.sm,clocks.mem",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines(),
    }


def validate(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    from repro.numerical_policy import tolerances  # pyrefly: ignore[missing-import]
    from repro.numerics import check_accuracy  # pyrefly: ignore[missing-import]

    assert actual.dtype == torch.bfloat16 and actual.shape == expected.shape
    gate = tolerances("mxfp8_dense_gemm", torch.bfloat16)
    return check_accuracy(
        actual,
        expected,
        atol=gate["atol"],
        rtol=gate["rtol"],
        l2_limit=gate["l2_limit"],
    )


def capture(
    call: Callable[[], torch.Tensor], *, cute: bool
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    if cute:
        with cute_cuda_graph() as graph:
            output = call()
    else:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def audit(
    call: Callable[[], torch.Tensor],
    tensors: tuple[torch.Tensor, ...],
    expected: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    mutate: bool,
) -> dict[str, Any]:
    snapshots = [t.view(torch.uint8).clone() for t in tensors]
    checks: dict[str, Any] = {}

    def check(ref: torch.Tensor) -> dict[str, Any]:
        if output is not None:
            output.fill_(float("nan"))
        actual = call()
        torch.cuda.synchronize()
        metric = validate(actual, ref)
        saved = actual.clone()
        if output is not None:
            output.fill_(float("nan"))
        again = call()
        torch.cuda.synchronize()
        assert torch.equal(again, saved), "Output changed on identical invocation"
        validate(again, ref)
        return metric

    checks["original"] = check(expected)
    if mutate:
        for name, tensor in zip(("a", "w", "sa", "sb"), tensors, strict=True):
            original = tensor.clone()
            try:
                if name in {"sa", "sb"}:
                    tensor.add_(1)
                else:
                    tensor.copy_((-0.5 * tensor.float()).to(tensor.dtype))
                checks[f"live_{name}"] = check(reference(*tensors))
            finally:
                tensor.copy_(original)
        checks["restored"] = check(expected)
    assert all(
        torch.equal(t.view(torch.uint8), snapshot)
        for t, snapshot in zip(tensors, snapshots, strict=True)
    ), "Provider changed input data"
    checks["input_preserved"] = True
    checks["repeatability"] = "PASS"
    checks["poisoned_output"] = "PASS" if output is not None else "NOT_APPLICABLE"
    return checks


@contextmanager
def public_library_backend(backend: str) -> Generator[None, None, None]:
    """Keep public AOT dispatch independent between the two library providers."""
    settings = {"HELION_BACKEND": backend, "HELION_AOT_MODE": "evaluate"}
    previous = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def public_library_call(
    operation: Callable[..., torch.Tensor],
    tensors: tuple[torch.Tensor, ...],
    backend: str,
) -> Callable[[], torch.Tensor]:
    def call() -> torch.Tensor:
        with public_library_backend(backend):
            return operation(*tensors)

    return call


def load_public_library_operation() -> Callable[..., torch.Tensor]:
    """Load the optional library provider with its original Triton backend."""
    with public_library_backend("triton"):
        from helion_kernel_library.ops import (  # pyrefly: ignore[missing-import]
            mxfp8_dense_gemm,
        )

    return mxfp8_dense_gemm


def build_baselines(
    tensors: tuple[torch.Tensor, ...],
    providers: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Callable[[], torch.Tensor]], dict[str, Any]]:
    a, w, sa, sb = tensors
    calls: dict[str, Callable[[], torch.Tensor]] = {}
    metadata: dict[str, Any] = {}
    if "deepgemm" in providers:
        import deep_gemm
        from repro.baselines.gb300.gemm import (  # pyrefly: ignore[missing-import]
            _pack_deepgemm_scales,
        )

        out = torch.empty(
            (a.shape[0], w.shape[0]), device=a.device, dtype=torch.bfloat16
        )

        def deepgemm() -> torch.Tensor:
            packed_a = _pack_deepgemm_scales(sa)
            packed_b = _pack_deepgemm_scales(sb)
            deep_gemm.fp8_gemm_nt((a, packed_a), (w, packed_b), out, recipe=(1, 1, 32))
            return out

        calls["deepgemm"] = deepgemm
        metadata["deepgemm"] = {
            "public_entrypoint": "deep_gemm.fp8_gemm_nt",
            "module": source(Path(deep_gemm.__file__).resolve()),
            "config": {"recipe": [1, 1, 32], "selection": "public dispatch"},
            "scale_packing": "both live inputs, every invocation; included in timing",
            "packing_source": source(Path(inspect.getfile(_pack_deepgemm_scales))),
        }
    if "triton" in providers:
        mxfp8_dense_gemm = load_public_library_operation()
        calls["triton"] = public_library_call(mxfp8_dense_gemm, tensors, "triton")
        calls["triton"]()
        library_kernel = mxfp8_dense_gemm.__globals__["_mxfp8_dense_gemm"]
        with public_library_backend("triton"):
            selected = library_kernel.bind(tensors)._config
        assert selected is not None, "Triton public op did not select saved config"
        metadata["triton"] = {
            "public_entrypoint": "helion_kernel_library.ops.mxfp8_dense_gemm",
            "config": json.loads(selected.to_json()),
            "selection": "normal public JSON-backed AOT",
            "runtime_autotuning": False,
            "kernel_source": source(Path(inspect.getfile(library_kernel.fn))),
        }
    if "published-cute" in providers:
        mxfp8_dense_gemm = load_public_library_operation()
        calls["published-cute"] = public_library_call(mxfp8_dense_gemm, tensors, "cute")
        calls["published-cute"]()
        from helion_kernel_library.configs import (  # pyrefly: ignore[missing-import]
            get_platform,
        )

        with public_library_backend("cute"):
            platform = get_platform(a.device)
        padded_m = max(a.shape[0], 64)
        flag = torch.empty((1,), device=a.device, dtype=torch.int32)
        left = torch.empty(
            (padded_m, a.shape[1]), device=a.device, dtype=torch.bfloat16
        )
        right = torch.empty(w.shape, device=a.device, dtype=torch.bfloat16)
        result = torch.empty(
            (a.shape[0], w.shape[0]), device=a.device, dtype=torch.bfloat16
        )
        stages = {}
        for role, name, stage_args in (
            ("decode_a", "mxfp8_decode_cute", (a, sa, flag, padded_m)),
            ("decode_w", "mxfp8_decode_cute", (w, sb, flag, 0)),
            ("bf16_gemm", "mxfp8_bf16_gemm_cute", (left, right)),
            ("unsafe_patch", "mxfp8_patch_cute", (result, a, w, sa, sb, flag)),
        ):
            module = importlib.import_module(
                "helion_kernel_library._kernels.mxfp8_dense_gemm." + name
            )
            assert module.__file__ is not None
            module_path = Path(module.__file__).resolve()
            chosen = module.config_picker(stage_args, platform=platform)
            stages[role] = {
                "config": json.loads(chosen.to_json()),
                "shape_key": module.config_key(stage_args),
                "module_source": source(module_path),
                "config_source": source(
                    module_path.parents[2] / "configs" / name / (platform + ".json")
                ),
            }
        metadata["published-cute"] = {
            "public_entrypoint": "helion_kernel_library.ops.mxfp8_dense_gemm",
            "public_source": source(Path(inspect.getfile(mxfp8_dense_gemm))),
            "selection": "published JSON-backed AOT; not cold retuned",
            "runtime_autotuning": False,
            "stages": stages,
            "timed_work": "live FP8-to-BF16 decoding of both inputs, padded BF16 GEMM, unsafe-scale patch and allocations",
        }
    if "published-cute" in providers and a.shape[0] == 8:
        metadata["published-cute"]["shape_extension"] = {
            "reason": "User replaced the original M=7 panel with M=8",
            "dispatch": "isolated library guard extended to M=8; same published decode/BF16/patch algorithm",
            "configs": "normal saved JSON AOT selection; nearest keys where M=8 is absent",
            "runtime_autotuning": False,
        }
    if "torch" in providers:
        calls["torch"] = lambda: torch_dequant(*tensors)
        metadata["torch"] = {
            "config": {"allow_tf32": False, "dtype": "FP32 decode/matmul, BF16 output"},
            "decode_and_matmul": "both live inputs every invocation; included in timing",
        }
    if "flashinfer-cute" in providers:
        import flashinfer
        from flashinfer.autotuner import AutoTuner
        from flashinfer.autotuner import autotune
        from repro.baselines.gb300.gemm import (  # pyrefly: ignore[missing-import]
            pack_scales,
        )

        cute_out = torch.empty(
            (a.shape[0], w.shape[0]), device=a.device, dtype=torch.bfloat16
        )

        def flashinfer_invoke() -> torch.Tensor:
            return flashinfer.mm_mxfp8(
                a,
                w.T,
                pack_scales(sa),
                pack_scales(sb),
                out=cute_out,
                out_dtype=torch.bfloat16,
                use_8x4_sf_layout=False,
                backend="cute-dsl",
            )

        cache = args.flashinfer_cache
        skip_receipt = cache.with_suffix(".skip.json")
        contract = {
            "shape": {"m": a.shape[0], "n": w.shape[0], "k": a.shape[1]},
            "operand_dtypes": [str(a.dtype), str(w.dtype)],
            "scale_dtypes": [str(sa.dtype), str(sb.dtype)],
            "output_dtype": "torch.bfloat16",
        }
        AutoTuner.get().clear_cache()
        if args.mode == "autotune":
            if cache.exists() or skip_receipt.exists():
                raise ValueError("Cold FlashInfer tuning requires a fresh tactic cache")
            cache.parent.mkdir(parents=True, exist_ok=True)
            with autotune(True, cache=str(cache), tuning_buckets=(a.shape[0],)):
                flashinfer_invoke()
            if not cache.is_file():
                skipped: dict[str, Any] = {
                    "status": "SKIP",
                    "reason": (
                        "FlashInfer AutoTuner found no supported CuTe MXFP8 tactic "
                        f"for M={a.shape[0]}, N={w.shape[0]}, K={a.shape[1]}; "
                        "all profiled tactics were rejected and no tactic cache "
                        "was saved"
                    ),
                    "public_entrypoint": "flashinfer.mm_mxfp8",
                    "backend": "cute-dsl",
                    "requested_tactic_cache": str(cache),
                    "contract": contract,
                }
                write_json(skip_receipt, skipped)
                skipped["skip_receipt_source"] = source(skip_receipt)
                metadata["flashinfer-cute"] = skipped
                return calls, metadata
        elif not cache.exists() and skip_receipt.is_file():
            skipped = json.loads(skip_receipt.read_text())
            if (
                skipped.get("status") != "SKIP"
                or skipped.get("backend") != "cute-dsl"
                or skipped.get("public_entrypoint") != "flashinfer.mm_mxfp8"
                or skipped.get("contract") != contract
            ):
                raise RuntimeError(
                    "Invalid saved FlashInfer unsupported-tactic receipt"
                )
            metadata["flashinfer-cute"] = skipped
            metadata["flashinfer-cute"]["skip_receipt_source"] = source(skip_receipt)
            return calls, metadata
        elif not AutoTuner.get().load_configs(str(cache)):
            raise RuntimeError("FlashInfer did not load the pinned tactic cache")

        def flashinfer_call() -> torch.Tensor:
            with autotune(False, tuning_buckets=(a.shape[0],)):
                return flashinfer_invoke()

        calls["flashinfer-cute"] = flashinfer_call
        metadata["flashinfer-cute"] = {
            "public_entrypoint": "flashinfer.mm_mxfp8",
            "backend": "cute-dsl",
            "module": source(Path(flashinfer.__file__).resolve()),
            "scale_packing": "both live inputs, every invocation; included in timing",
            "packing_source": source(Path(inspect.getfile(pack_scales))),
            "config": json.loads(cache.read_text()),
            "tactic_cache_source": source(cache),
            "selection": "fresh public AutoTuner search then pinned exact-M tactic",
            "tuning_buckets": [a.shape[0]],
            "runtime_autotuning": False,
        }
    return calls, metadata


def cuda_event_timing(
    graph: torch.cuda.CUDAGraph, args: argparse.Namespace
) -> dict[str, Any]:
    # Flush outside each event span, using four times reported L2 capacity.
    flush = torch.empty(
        max(64 * 1024 * 1024, 4 * torch.cuda.get_device_properties(0).L2_cache_size),
        device="cuda",
        dtype=torch.uint8,
    )
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.event_samples):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "benchmark_timer": "cuda_event",
        "cold_l2_cache": True,
        "pre_captured_graph": True,
        "l2_flush_bytes": flush.numel(),
    }


def measure(graph: torch.cuda.CUDAGraph, args: argparse.Namespace) -> dict[str, Any]:
    if args.timer == "cuda-event":
        return cuda_event_timing(graph, args)
    from repro.breadth.common import bench_steady  # pyrefly: ignore[missing-import]

    return bench_steady(
        graph.replay,
        num_runs=1,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        thermal_warmup_ms=0,
        pre_captured_graph=True,
    )


def graph_autotune_benchmark(
    fns: list[Callable[[], object]],
    *,
    repeat: int,
    desc: str | None = None,
    args: argparse.Namespace,
) -> list[float]:
    """Use captured calls with short trial windows and mirrored finalists.

    `repeat` is recorded as requested by the search. Sampling uses a 20 ms
    initial trial or a 100 ms finalist window for each callable, independent
    of the search's number of candidates and generations.
    """
    if repeat < 1 or not fns:
        raise ValueError("A graph benchmark needs callables and positive repeat")
    window_ms = 20 if len(fns) == 1 else 100
    sweeps = 1 if len(fns) == 1 else 4
    traces: list[dict[str, Any]] = []
    values: list[list[float]] = [[] for _ in fns]
    graphs = []
    held_outputs = []
    for fn in fns:
        held = [fn() for _ in range(3)]
        torch.cuda.synchronize()
        with cute_cuda_graph() as graph:
            held.append(fn())
        # Keep the graph, captured output and warmup outputs together.
        held_outputs.append(held)
        graphs.append(graph)
    for sweep in range(sweeps):
        order = list(range(len(fns)))
        if sweep % 2:
            order.reverse()
        for index in order:
            started = time.monotonic()
            if args.timer == "cuda-event":
                timing = cuda_event_timing(graphs[index], args)
                measured = timing["samples_ms"]
            else:
                from flashinfer.testing.utils import bench_gpu_time

                with warnings.catch_warnings():
                    warnings.filterwarnings("error", message=".*Falling back to CUDA.*")
                    measured = bench_gpu_time(
                        graphs[index].replay,
                        enable_cupti=True,
                        use_cuda_graph=False,
                        cold_l2_cache=True,
                        dry_run_time_ms=1,
                        repeat_time_ms=window_ms // sweeps,
                    )
                timing = {
                    "benchmark_timer": "cupti",
                    "benchmark_timer_source": source(
                        Path(inspect.getfile(bench_gpu_time))
                    ),
                    "cold_l2_cache": True,
                    "pre_captured_graph": True,
                }
            if not measured or any(
                not isinstance(value, float) or not math.isfinite(value) or value <= 0
                for value in measured
            ):
                raise RuntimeError("Autotune graph timer returned invalid samples")
            values[index].extend(measured)
            traces.append(
                {
                    "sweep": sweep,
                    "callable_index": index,
                    "samples_ms": measured,
                    "samples_count": len(measured),
                    "elapsed_seconds": time.monotonic() - started,
                    "timer": timing,
                }
            )
    medians = [statistics.median(samples) for samples in values]
    with args.result_file.with_suffix(".autotune_timer.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "desc": desc,
                    "requested_repeat": repeat,
                    "window_ms_per_callable": window_ms,
                    "callable_count": len(fns),
                    "medians_ms": medians,
                    "traces": traces,
                },
                allow_nan=False,
            )
            + "\n"
        )
    return medians


def load_pinned_config(path: Path) -> helion.Config | CuteStructuralConfig:
    payload = json.loads(path.read_text())
    if payload.get("schema") == "helion.cute.structural_config":
        return CuteStructuralConfig.from_dict(payload)
    payload = payload.get("config_envelope", payload.get("config", payload))
    if payload.get("schema") == "helion.cute.structural_config":
        return CuteStructuralConfig.from_dict(payload)
    return helion.Config.from_dict(payload)


def run(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    result["tool_script_directories"] = initialize_tool_path()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    from repro.breadth.common import (  # pyrefly: ignore[missing-import]
        _gpu_temperature_c,
    )

    startup_temperature = _gpu_temperature_c()
    if startup_temperature is None:
        raise RuntimeError("Cannot record selected GPU startup temperature")
    result["provenance"] = provenance()
    result["startup_temperature_c"] = startup_temperature
    m, n, k = SHAPES[args.shape_index]
    result["shape"] = {"m": m, "n": n, "k": k}
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    a = (torch.randn((m, k), device="cuda", generator=generator) / 3).to(
        torch.float8_e4m3fn
    )
    w = (torch.randn((n, k), device="cuda", generator=generator) / 3).to(
        torch.float8_e4m3fn
    )
    sa = torch.randint(
        125, 129, (m, k // 32), device="cuda", generator=generator, dtype=torch.uint8
    )
    sb = torch.randint(
        125, 129, (n, k // 32), device="cuda", generator=generator, dtype=torch.uint8
    )
    tensors = (a, w, sa, sb)
    result["inputs"] = {
        "recipe": "independent normal(0,1)/3 E4M3FN; independent E8M0 bytes [125,129)",
        "sha256": [
            hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for t in tensors
        ],
        "same_live_tensor_objects_all_providers": True,
    }
    expected = reference(*tensors)
    calls, result["baseline_configs"] = build_baselines(
        tensors, args.baselines.split(","), args
    )
    pinned = load_pinned_config(args.config_file) if args.mode == "verify" else None
    target = native_mxfp8
    if isinstance(pinned, CuteStructuralConfig):
        target = helion.kernel(
            native_mxfp8.fn,
            backend="cute",
            config=pinned,
            autotune_effort="full",
            autotune_budget_seconds=None,
        )
    bound = target.bind(tensors)
    bound.settings.autotune_baseline_fn = torch_dequant

    def check_autotune_accuracy(actual: object, ref: object) -> None:
        assert isinstance(actual, torch.Tensor), "MXFP8 kernel must return a tensor"
        validate(actual, expected)

    bound.settings.autotune_baseline_accuracy_check_fn = check_autotune_accuracy
    bound.settings.autotune_budget_seconds = None
    bound.settings.autotune_max_generations = None
    bound.settings.autotune_effort = "full"
    bound.settings.autotune_random_seed = args.autotune_seed
    bound.settings.autotune_benchmark_fn = functools.partial(
        graph_autotune_benchmark, args=args
    )
    bound.settings.autotune_log = str(args.result_file.with_suffix(".autotune.log"))
    started = time.monotonic()
    if args.mode == "autotune":
        os.environ["HELION_SKIP_CACHE"] = "1"
        config = bound.autotune(tensors, force=True)
        result["cold_autotune"] = {
            "effort": "full",
            "skip_cache": True,
            "budget_seconds": None,
            "random_seed": args.autotune_seed,
            "elapsed_seconds": time.monotonic() - started,
        }
    else:
        assert pinned is not None
        bound.set_config(pinned)
        config = pinned.config if isinstance(pinned, CuteStructuralConfig) else pinned
        result["pinned_config_source"] = source(args.config_file)
    result["config"] = json.loads(config.to_json())
    result["config_envelope"] = bound.config_envelope(config).to_dict()
    write_json(args.result_file.with_suffix(".winner.json"), result["config_envelope"])
    calls = {"native_cute": lambda: bound(*tensors), **calls}
    result["correctness_eager"] = {
        name: audit(call, tensors, expected, mutate=True)
        for name, call in calls.items()
    }
    graphs, outputs = {}, {}
    result["correctness_graph"] = {}
    for name, call in calls.items():
        graph, output = capture(
            call, cute=name in {"native_cute", "flashinfer-cute", "published-cute"}
        )
        graphs[name], outputs[name] = graph, output

        def replay(
            graph: torch.cuda.CUDAGraph = graph,
            output: torch.Tensor = output,
        ) -> torch.Tensor:
            graph.replay()
            return output

        result["correctness_graph"][name] = audit(
            replay, tensors, expected, output=output, mutate=True
        )
    write_json(args.result_file, result)
    from repro.breadth.common import _gpu_warmup  # pyrefly: ignore[missing-import]
    from repro.breadth.common import (  # pyrefly: ignore[missing-import]
        _wait_for_gpu_cooldown,
    )

    result["thermal_cooldown"] = _wait_for_gpu_cooldown(startup_temperature + 1)
    _gpu_warmup(args.thermal_warmup_ms)
    timings: dict[str, list[dict[str, Any]]] = {name: [] for name in calls}
    result["rounds"] = []
    # Interleave all providers each round and reverse the order every second
    # round, so both nearby rounds and long-term drift are paired.
    names = list(calls)
    for round_index in range(args.rounds):
        order = names if round_index % 2 == 0 else list(reversed(names))
        round_result = {"round": round_index, "order": order, "timings": {}}
        for name in order:
            timing = measure(graphs[name], args)
            timings[name].append(timing)
            round_result["timings"][name] = timing
        result["rounds"].append(round_result)
        write_json(args.result_file, result)
    result["timings"] = {
        name: {
            "median_ms": statistics.median(t["median_ms"] for t in values),
            "runs_ms": [t["median_ms"] for t in values],
            "tflops": 2
            * m
            * n
            * k
            / (statistics.median(t["median_ms"] for t in values) * 1e9),
        }
        for name, values in timings.items()
    }
    best = min(
        (name for name in result["timings"] if name != "native_cute"),
        key=lambda name: result["timings"][name]["median_ms"],
    )
    result["best_baseline"] = best
    result["ratio"] = (
        result["timings"][best]["median_ms"]
        / result["timings"]["native_cute"]["median_ms"]
    )
    result["paired_ratios"] = [
        b["median_ms"] / a["median_ms"]
        for a, b in zip(timings["native_cute"], timings[best], strict=True)
    ]
    result["correctness_after_timing"] = {}
    for name, graph in graphs.items():
        graph.replay()
        torch.cuda.synchronize()
        result["correctness_after_timing"][name] = validate(outputs[name], expected)
    result["status"] = "PASS"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-index", type=int, required=True, choices=range(24))
    parser.add_argument("--mode", choices=("autotune", "verify"), default="autotune")
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--flashinfer-cache", type=Path)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--autotune-seed", type=int, default=secrets.randbits(31))
    parser.add_argument("--timer", choices=("cupti", "cuda-event"), default="cupti")
    parser.add_argument(
        "--baselines", default="deepgemm,triton,flashinfer-cute,published-cute,torch"
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=500)
    parser.add_argument("--thermal-warmup-ms", type=int, default=10000)
    parser.add_argument("--event-samples", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "verify" and args.config_file is None:
        parser.error("--mode verify requires --config-file")
    if (
        args.result_file.exists()
        or args.result_file.with_suffix(".winner.json").exists()
    ):
        parser.error("Use a fresh result path; historical raw results are immutable")
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    if not args.baselines or any(
        name not in {"deepgemm", "triton", "flashinfer-cute", "published-cute", "torch"}
        for name in args.baselines.split(",")
    ):
        parser.error(
            "--baselines must select deepgemm,triton,flashinfer-cute,published-cute,torch"
        )
    if args.flashinfer_cache is None:
        if args.mode == "verify":
            stem = args.config_file.stem.removesuffix(".winner")
            args.flashinfer_cache = args.config_file.with_name(
                stem + ".flashinfer.json"
            )
        else:
            args.flashinfer_cache = args.result_file.with_suffix(".flashinfer.json")
    result: dict[str, Any] = {
        "status": "RUNNING",
        "shape_index": args.shape_index,
        "options": vars(args),
        "reference": "independent unrounded FP64 decode/matmul",
        "timing_scope": "all call-time GPU work including transformations; graph replay cold L2",
    }
    try:
        run(args, result)
    except Exception as exc:
        result.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        write_json(args.result_file, result)
    print(
        json.dumps(
            {
                "shape_index": args.shape_index,
                "status": result["status"],
                "ratio": result["ratio"],
                "best_baseline": result["best_baseline"],
                "timings": result["timings"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
