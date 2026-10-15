"""K18 MXFP8 comparison with independently pinned configurations.

Run with helion-kernel-library on PYTHONPATH. Every case/provider runs in its
own process. CUPTI times captured kernels with L2 cleared before each sample.
Raw results and generated code are preserved.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
from typing import Any
from typing import Callable
from typing import NoReturn

import torch

import helion
from helion.autotuner.accuracy import assert_close
from helion.autotuner.benchmarking import do_bench

SHAPES = ((1, 128), (137, 64), (128, 1024), (4096, 4096), (8192, 8192), (8192, 16384))
IMPLEMENTATIONS = (
    "helion-cute",
    "helion-triton",
    "flashinfer-cute",
    "transformer-engine-cuda",
    "torch-reference-compile",
    "torch-reference-eager",
)
ROOT = Path(__file__).resolve().parents[2]


def exact_output(actual: object, expected: object) -> None:
    if isinstance(actual, (tuple, list)):
        assert isinstance(expected, (tuple, list))
        assert len(actual) == len(expected)
        for value, target in zip(actual, expected, strict=True):
            exact_output(value, target)
    elif isinstance(actual, torch.Tensor) and actual.dtype == torch.float8_e4m3fn:
        assert isinstance(expected, torch.Tensor)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    else:
        assert_close(actual, expected, atol=0, rtol=0)


def reference(
    x: torch.Tensor, recipe: str = "torchao-rceil"
) -> tuple[torch.Tensor, torch.Tensor]:
    assert recipe == "torchao-rceil"
    data = x.float().view(x.shape[0], -1, 32)
    ratio = data.abs().amax(-1) * 0.0022321429569274187
    bits = ratio.view(torch.int32)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    increment = (mantissa != 0) & ((exponent != 0) | (mantissa > 0x400000))
    code = torch.where(ratio > 0, (exponent + increment).clamp_max(254), 0)
    inverse_bits = torch.where(code == 254, 0x400000, (254 - code) << 23)
    inverse = inverse_bits.view(torch.float32)
    encoded = (data * inverse[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return encoded.view(x.shape), code.to(torch.uint8).contiguous()


def gpu_state() -> dict[str, Any]:
    fields = (
        "name",
        "uuid",
        "temperature.gpu",
        "power.limit",
        "clocks.sm",
        "utilization.gpu",
    )
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            os.environ["CUDA_VISIBLE_DEVICES"],
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return dict(
        zip(fields, [value.strip() for value in output.split(",")], strict=True)
    )


def cooldown(target_c: float) -> dict[str, Any]:
    torch.cuda.synchronize()
    began = time.monotonic()
    state = gpu_state()
    while float(state["temperature.gpu"]) > target_c:
        if time.monotonic() - began > 240:
            raise RuntimeError(f"GPU did not cool to {target_c} C: {state}")
        time.sleep(5)
        state = gpu_state()
    return {"elapsed_seconds": time.monotonic() - began, "state": state}


def make_call(
    args: argparse.Namespace, x: torch.Tensor, directory: Path
) -> tuple[Callable[[], Any], dict[str, Any]]:
    if args.implementation.startswith("helion-"):
        # Import optional kernels only in the worker, after its backend is set.
        library_kernel = importlib.import_module(
            "helion_kernel_library._kernels.mxfp8_quantize.mxfp8_quantize"
        )
        mxfp8_quantize = library_kernel.mxfp8_quantize

        if args.kernel == "library":
            ops = importlib.import_module("helion_kernel_library.ops")

            from helion.autotuner.base_search import BaseSearch

            def forbidden_search(
                *search_args: object, **search_kwargs: object
            ) -> NoReturn:
                raise AssertionError("The public K18 call invoked autotuning")

            BaseSearch.autotune = forbidden_search
            inputs = (x, "torchao-rceil")
            config = library_kernel.config_picker(inputs)
            bound = mxfp8_quantize.bind(inputs)
            generated = bound.to_code(config)
            (directory / "config.json").write_text(config.to_json() + "\n")
            (directory / "generated.py").write_text(generated)
            source_path = Path(inspect.getfile(mxfp8_quantize.fn))
            return functools.partial(ops.mxfp8_quantize, x, recipe="torchao-rceil"), {
                "tuning": "public-aot-json-pinned",
                "config": json.loads(config.to_json()),
                "source_path": str(source_path),
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
            }

        function = mxfp8_quantize.fn
        target = helion.kernel(
            function,
            backend=args.implementation.removeprefix("helion-"),
            static_shapes=True,
            autotune_effort="full" if args.autotune == "full" else "none",
            autotune_random_seed=args.tune_seed,
            autotune_baseline_fn=reference,
            autotune_baseline_accuracy_check_fn=exact_output,
            autotune_log=str(directory / "autotune.csv"),
            autotune_log_details=True,
            autotune_precompile_jobs=args.precompile_jobs,
        )
        inputs = (x, "torchao-rceil")
        bound = target.bind(inputs)
        start = time.monotonic()
        if args.config is not None:
            saved = json.loads(args.config.read_text())
            config = helion.Config(**saved.get("config", saved))
            bound.set_config(config)
            tuning = "pinned"
        elif args.autotune == "full":
            config = bound.autotune(inputs, force=True)
            tuning = "cold-full"
        else:
            config = bound.config_spec.default_config()
            bound.set_config(config)
            tuning = "default-probe"
        config = bound.config_spec.normalized_config(config)
        (directory / "config.json").write_text(config.to_json() + "\n")
        generated = bound.to_code(config)
        (directory / "generated.py").write_text(generated)
        source_path = Path(inspect.getfile(function))
        return functools.partial(target, *inputs), {
            "tuning": tuning,
            "tune_seed": args.tune_seed,
            "tuning_seconds": time.monotonic() - start,
            "config": json.loads(config.to_json()),
            "source_path": str(source_path),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "generated_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        }
    if args.implementation == "flashinfer-cute":
        from flashinfer import mxfp8_quantize

        options = {
            "is_sf_swizzled_layout": False,
            "alignment": 32,
            "backend": "cute-dsl",
            "enable_pdl": False,
        }

        def flashinfer_call() -> tuple[torch.Tensor, torch.Tensor]:
            values, scales = mxfp8_quantize(x, **options)
            return values[:, : x.shape[1]].contiguous(), scales.view(
                torch.uint8
            ).reshape(x.shape[0], -1)[:, : x.shape[1] // 32].contiguous()

        return flashinfer_call, {"config": options, "tuning": "provider-policy"}
    if args.implementation == "transformer-engine-cuda":
        te = importlib.import_module("transformer_engine.pytorch.tensor.mxfp8_tensor")
        tex = importlib.import_module("transformer_engine_torch")

        quantizer = te.MXFP8Quantizer(
            tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            with_2d_quantization=False,
        )
        quantizer.optimize_for_gemm = False
        rows, width = x.shape
        padded_rows = (rows + 31) // 32 * 32

        def te_call() -> tuple[torch.Tensor, torch.Tensor]:
            padded = (
                torch.nn.functional.pad(x, (0, 0, 0, padded_rows - rows))
                if padded_rows != rows
                else x
            )
            result = quantizer(padded)
            return result._rowwise_data[:rows].view(
                torch.float8_e4m3fn
            ).contiguous(), result._rowwise_scale_inv[:rows, : width // 32].view(
                torch.uint8
            ).contiguous()

        return te_call, {
            "config": {
                "rowwise": True,
                "columnwise": False,
                "with_2d_quantization": False,
                "optimize_for_gemm": False,
            },
            "tuning": "provider-policy",
        }
    if args.implementation == "torch-reference-compile":
        validation = importlib.import_module(
            "benchmarks._support.mxfp8_quantize.validation"
        )
        compiled = torch.compile(
            validation.reference_values, fullgraph=True, dynamic=False
        )
        return functools.partial(compiled, x), {
            "config": {"fullgraph": True, "dynamic": False},
            "tuning": "inductor-default",
        }
    return functools.partial(reference, x), {"config": {}, "tuning": "eager"}


def worker(args: argparse.Namespace) -> None:
    directory = args.output_dir / f"case{args.case:02d}" / args.implementation
    directory.mkdir(parents=True, exist_ok=True)
    rows, width = SHAPES[args.case]
    generator = torch.Generator(device="cuda").manual_seed(args.input_seed)
    x = torch.randn(
        (rows, width), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    snapshot = x.view(torch.uint8).clone()
    input_sha256 = hashlib.sha256(
        x.cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    expected = reference(x)
    call, metadata = make_call(args, x, directory)
    print("VALIDATE eager and graph", flush=True)
    exact_output(call(), expected)
    assert torch.equal(x.view(torch.uint8), snapshot)
    if args.kernel == "library" and args.implementation.startswith("helion-"):
        library_kernel = importlib.import_module(
            "helion_kernel_library._kernels.mxfp8_quantize.mxfp8_quantize"
        )
        bound = library_kernel.mxfp8_quantize.bind((x, "torchao-rceil"))
        assert bound._config is not None
        actual_config = bound.config_spec.normalized_config(bound._config)
        selected_config = bound.config_spec.normalized_config(
            helion.Config(**metadata["config"])
        )
        assert actual_config == selected_config
        actual_source = Path(inspect.getfile(bound._run)).read_bytes()
        (directory / "actual_run.py").write_bytes(actual_source)
        metadata["actual_config_verified"] = True
        metadata["actual_run_sha256"] = hashlib.sha256(actual_source).hexdigest()
    from helion.runtime import cute_cuda_graph

    with cute_cuda_graph() as graph:
        captured_output = call()
    graph.replay()
    torch.cuda.synchronize()
    exact_output(captured_output, expected)
    x.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    exact_output(captured_output, reference(x))
    x.view(torch.uint8).copy_(snapshot)
    graph.replay()
    torch.cuda.synchronize()
    exact_output(captured_output, expected)
    print(f"COOLDOWN {gpu_state()}", flush=True)
    thermal = cooldown(args.cooldown_temp)
    print(f"MEASURE with {args.timer} and cold L2", flush=True)
    samples_ms = []
    if args.timer == "cupti":
        from flashinfer.testing.utils import bench_gpu_time_with_cupti

        for _ in range(args.num_runs):
            samples_ms.append(
                bench_gpu_time_with_cupti(
                    call,
                    dry_run_time_ms=args.warmup_ms,
                    repeat_time_ms=args.rep_ms,
                    use_cuda_graph=True,
                    cold_l2_cache=True,
                )
            )
        runs_ms = [statistics.median(samples) for samples in samples_ms]
        timer_name = (
            "flashinfer.testing.utils.bench_gpu_time_with_cupti (CUDA graph, cold L2)"
        )
    else:
        runs_ms = []
        for _ in range(args.num_runs):
            timing = do_bench(
                call,
                warmup=args.warmup_ms,
                rep=args.rep_ms,
                return_mode="median",
                default_cudagraph=True,
            )
            assert isinstance(timing, float)
            runs_ms.append(timing)
        timer_name = "helion.autotuner.benchmarking.do_bench (CUDA events, batched short graphs, flush subtraction)"
    median_ms = statistics.median(runs_ms)
    bytes_per_call = x.numel() * 3 + x.numel() // 32
    versions = {
        package: importlib.metadata.version(package)
        for package in (
            "torch",
            "triton",
            "nvidia-cutlass-dsl",
            "flashinfer-python",
            "transformer-engine",
        )
    }
    result = {
        "case": args.case,
        "shape": [rows, width],
        "dtype": "bfloat16",
        "recipe": "torchao-rceil",
        "implementation": args.implementation,
        "kernel": args.kernel,
        "status": "PASS",
        "correctness": {
            "byte_exact": True,
            "input_preserved": True,
            "graph": True,
            "changed_input_graph": True,
        },
        "input_seed": args.input_seed,
        "input_sha256": input_sha256,
        "runs_ms": runs_ms,
        "median_ms": median_ms,
        "bytes_per_call": bytes_per_call,
        "gb_s": bytes_per_call / median_ms / 1e6,
        "timer": timer_name,
        "samples_ms": samples_ms,
        "cold_l2": True,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "num_runs": args.num_runs,
        "host": socket.gethostname(),
        "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu": gpu_state(),
        "cooldown": thermal,
        "versions": versions,
        "helion_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "helion_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
        ).hexdigest(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"PASS case{args.case} {rows}x{width} {args.implementation}: {median_ms * 1000:.3f} us, {result['gb_s']:.1f} GB/s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--case", type=int)
    parser.add_argument("--implementations", default=",".join(IMPLEMENTATIONS[:-1]))
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS)
    parser.add_argument("--gpu", default="2")
    parser.add_argument(
        "--kernel",
        choices=("library", "original"),
        default="original",
    )
    parser.add_argument("--timer", choices=("cupti", "events"), default="cupti")
    parser.add_argument("--autotune", choices=("full", "default"), default="default")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input-seed", type=int, default=0)
    parser.add_argument("--tune-seed", type=int, default=20261014)
    parser.add_argument("--precompile-jobs", type=int, default=4)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--cooldown-temp", type=float, default=47)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.worker:
        worker(args)
        return
    cases = (
        list(range(len(SHAPES)))
        if args.cases == "all"
        else [int(value) for value in args.cases.split(",")]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        for implementation in args.implementations.split(","):
            directory = args.output_dir / f"case{case:02d}" / implementation
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "result.json").exists():
                raise FileExistsError(f"Preserved result already exists: {directory}")
            env = os.environ.copy()
            env.update(
                CUDA_VISIBLE_DEVICES=args.gpu,
                HELION_BACKEND=implementation.removeprefix("helion-")
                if implementation.startswith("helion-")
                else "triton",
                HELION_BENCHMARK_CUDAGRAPH="1",
                HELION_AUTOTUNE_PROGRESS_BAR="0",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                TORCHINDUCTOR_COMPILE_THREADS="1",
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--case",
                str(case),
                "--implementation",
                implementation,
                "--output-dir",
                str(args.output_dir),
                "--kernel",
                args.kernel,
                "--autotune",
                args.autotune,
                "--timer",
                args.timer,
                "--input-seed",
                str(args.input_seed),
                "--tune-seed",
                str(args.tune_seed),
                "--precompile-jobs",
                str(args.precompile_jobs),
                "--num-runs",
                str(args.num_runs),
                "--warmup-ms",
                str(args.warmup_ms),
                "--rep-ms",
                str(args.rep_ms),
                "--cooldown-temp",
                str(args.cooldown_temp),
            ]
            if args.config is not None:
                command.extend(("--config", str(args.config.resolve())))
            print(f"START case{case} {implementation}", flush=True)
            with (directory / "worker.log").open("w") as log:
                status = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                ).returncode
            if status:
                print((directory / "worker.log").read_text()[-8000:], flush=True)
                raise RuntimeError(f"case{case} {implementation} failed: {status}")
            result = json.loads((directory / "result.json").read_text())
            print(
                f"DONE case{case} {implementation}: {result['median_ms'] * 1000:.3f} us ({result['tuning']})",
                flush=True,
            )


if __name__ == "__main__":
    main()
