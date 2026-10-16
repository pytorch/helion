"""One-process K2 CuTe cold-full search or fixed-config graph comparison.

Run with the common GB300 baseline Python and CUDA_VISIBLE_DEVICES set to one
idle GPU. Output directories must be fresh children of artifacts/<run-name>.
The frozen HKL inputs, reference, and numerical gate are reused. M256 frontend
runs call the public Helion preset wrapper; legacy MN fixed runs retain the HKL
wrapper. Archived compiler-transposed configs are copied into physical frontend
coordinates and recorded with their original source. This worker never writes
HKL configs. A fixed run does not establish that a cold search found its config.
"""

from __future__ import annotations

import argparse
import copy
import csv
import functools
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import secrets
import socket
import statistics
import subprocess
import sys
import time
import traceback
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import TypeVar
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from types import MethodType

    from helion.autotuner.base_search import BaseAutotuner
    from helion.autotuner.local_cache import LocalAutotuneCache
    from helion.runtime.kernel import BoundKernel

_CaptureOutput = TypeVar("_CaptureOutput")

ROOT = Path(__file__).resolve().parents[2]
HKL = (
    Path(
        os.environ.get(
            "K2_HKL_ROOT",
            str(ROOT.parent / "helion-kernel-library"),
        )
    )
    .expanduser()
    .resolve()
)
SHAPES = ((64, 4096, 4096), (256, 4096, 2048), (128, 2880, 5120), (4096, 4096, 2880))
MODES = ("tensor-tensor", "tensor-channel", "token-tensor", "token-channel")
_TRANSPOSED_SHAPE = (256, 4096, 2048)
_KernelImplementation = Literal["hkl_mn", "frontend_transposed"]


def _kernel_config_for_run(
    shape: tuple[int, int, int],
    phase: str,
    payload: dict[str, Any] | None = None,
) -> tuple[_KernelImplementation, dict[str, Any] | None, dict[str, Any] | None]:
    """Select the body and copy an archived config into that body's coordinates.

    A new result.json identifies its kernel explicitly. Bare configs from the
    current transposed table are recognized by their physical 128x64 two-CTA
    tile; legacy M256 MN configs remain bound to the vendored MN body.
    """
    if phase == "cold":
        return (
            "frontend_transposed" if shape == _TRANSPOSED_SHAPE else "hkl_mn",
            None,
            None,
        )
    assert payload is not None
    original = copy.deepcopy(payload.get("config", payload))
    values = copy.deepcopy(original)
    orientation = values.pop("tcgen05_orientation", None)
    changes: dict[str, Any] = {}
    if orientation is not None:
        changes["tcgen05_orientation"] = {"original": orientation, "effective": None}
    if orientation == "nm":
        if shape != _TRANSPOSED_SHAPE:
            raise ValueError(
                "Archived NM configs require (M, K, N) = (256, 4096, 2048)"
            )
        blocks = values["block_sizes"]
        physical_blocks = [blocks[1], blocks[0], *blocks[2:]]
        changes["block_sizes"] = {"original": blocks, "effective": physical_blocks}
        changes["tcgen05_cta_group"] = {
            "original": values.get("tcgen05_cta_group", "auto"),
            "effective": "two",
        }
        values["block_sizes"] = physical_blocks
        values["tcgen05_cta_group"] = "two"
        if "indexing" in values:
            changes["indexing"] = {
                "original": values.pop("indexing"),
                "effective": None,
            }
        implementation: _KernelImplementation = "frontend_transposed"
    elif orientation == "mn":
        implementation = "hkl_mn"
    elif orientation is not None:
        raise ValueError(f"Unsupported archived tcgen05_orientation={orientation!r}")
    elif payload.get("kernel_implementation") == "frontend_transposed":
        if shape != _TRANSPOSED_SHAPE:
            raise ValueError("Frontend-transposed configs require the M256 tuned shape")
        implementation = "frontend_transposed"
    elif payload.get("kernel_implementation") == "hkl_mn":
        implementation = "hkl_mn"
    elif (
        shape == _TRANSPOSED_SHAPE
        and values.get("block_sizes", [])[:2] == [128, 64]
        and values.get("tcgen05_cta_group") == "two"
    ):
        implementation = "frontend_transposed"
    else:
        implementation = "hkl_mn"
    return implementation, values, {"original_config": original, "changes": changes}


def _source(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def _selected_gpu_uuid() -> str:
    """Resolve monitoring through CUDA, preserving scheduler-owned visibility."""
    import torch

    uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    return uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}"


def _execution_provenance() -> dict[str, object]:
    return {
        "hostname": socket.gethostname(),
        "architecture": platform.machine(),
        "process_id": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "helion_root": str(ROOT),
        "hkl_root": str(HKL),
        "path_overrides": {
            name: os.environ.get(name) for name in ("K2_HKL_ROOT", "K2_QUACK_ROOT")
        },
        "slurm": {
            name: os.environ[name]
            for name in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_STEP_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_CLUSTER_NAME",
                "SLURM_JOB_NODELIST",
                "SLURMD_NODENAME",
                "SLURM_PROCID",
                "SLURM_LOCALID",
                "SLURM_JOB_GPUS",
                "SLURM_STEP_GPUS",
            )
            if name in os.environ
        },
    }


def _gpu_snapshot() -> dict[str, object]:
    fields = (
        "index",
        "uuid",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.current.sm",
        "clocks.current.memory",
        "pstate",
        "utilization.gpu",
    )
    device = _selected_gpu_uuid()
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    values = next(csv.reader([raw.strip()], skipinitialspace=True))
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **dict(zip(fields, values, strict=True)),
    }


def _manifest() -> dict[str, dict[str, str]]:
    paths = [ROOT / p for p in _git(ROOT, "ls-files", "helion").splitlines()]
    paths.extend((ROOT / "helion").rglob("*.py"))
    paths += [
        Path(__file__).resolve(),
        ROOT / "pretuned_kernels/_bench.py",
        ROOT / "benchmarks/cute/k2_graph_timer.py",
        ROOT / "pretuned_kernels/scaled_mm_cute/scaled_mm_cute.py",
        ROOT
        / "pretuned_kernels/scaled_mm_cute/_helion_aot_scaled_mm_cute_cuda_sm100.py",
        ROOT
        / "pretuned_kernels/scaled_mm_cute/_helion_aot_scaled_mm_cute_cuda_sm103.py",
        HKL / "helion_kernel_library/ops.py",
        HKL / "helion_kernel_library/configs/__init__.py",
        HKL / "helion_kernel_library/configs/scaled_mm_cute/nvidia_sm103.json",
        HKL / "helion_kernel_library/_kernels/scaled_mm/scaled_mm_cute.py",
        HKL / "repro/gb300/cases.py",
        HKL / "repro/gb300/providers.py",
        HKL / "repro/gb300/provenance/reviewed_runtime_restore.json",
        HKL / "repro/breadth/common.py",
        HKL / "repro/breadth/compare_scaled_mm.py",
        HKL / "repro/baselines/gemm_validation.py",
        HKL / "repro/numerics.py",
        HKL / "repro/numerical_policy.py",
        HKL / "repro/numerical_policy.json",
        HKL / "benchmarks/_support/gemm_validation.py",
        HKL / "benchmarks/scaled_mm.py",
    ]
    adapter = ROOT / "benchmarks/cute/k2_scaled_mm_baselines.py"
    if adapter.is_file():
        paths.append(adapter)
    return {str(path): _source(path) for path in paths if path.is_file()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-index", type=int, choices=range(4), required=True)
    parser.add_argument("--scale-mode", choices=MODES, required=True)
    parser.add_argument("--phase", choices=("cold", "fixed"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Required for fixed verification")
    parser.add_argument("--quack-config", type=Path)
    parser.add_argument("--input-seed", type=int, default=0)
    parser.add_argument("--search-seed", type=int, default=None)
    parser.add_argument(
        "--candidate-batch-size", "--candidate-repetitions", type=int, default=256
    )
    parser.add_argument("--batch-size", "--repetitions", type=int, default=256)
    parser.add_argument("--timer-samples", type=int, default=10)
    parser.add_argument("--timer-warmup-ms", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cooldown-margin-c", type=float, default=2.0)
    parser.add_argument("--thermal-warmup-ms", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT / "artifacts"):
        parser.error("--output-dir must be under artifacts/<run-name>/<variant>")
    relative = args.output_dir.relative_to(ROOT / "artifacts")
    if len(relative.parts) < 2:
        parser.error("--output-dir must be under artifacts/<run-name>/<variant>")
    if args.phase == "fixed" and args.config is None:
        parser.error("fixed verification requires --config")
    if args.phase == "cold" and args.config is not None:
        parser.error("cold search uses the saved HKL config as a seed; omit --config")
    if (
        min(args.candidate_batch_size, args.batch_size, args.timer_samples, args.rounds)
        < 1
    ):
        parser.error("all repetition counts must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty; existing raw artifacts are immutable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def _run(args: argparse.Namespace, result: dict[str, Any]) -> None:
    # Import the requested compiler tree and frozen HKL, independent of editable
    # installs in the baseline environment. Configure before importing Helion.
    sys.path[:0] = [str(ROOT), str(HKL)]
    os.environ.update(HELION_BACKEND="cute", HELION_SKIP_CACHE="1")
    os.environ.pop("HELION_AOT_MODE", None)
    os.environ.pop("HELION_AUTOTUNER", None)
    os.environ.pop("HELION_CAP_AUTOTUNE_NUM_NEIGHBORS", None)
    os.environ.pop("HELION_CAP_REBENCHMARK_REPEAT", None)

    # Direct script execution resolves these siblings through the script directory.
    # pyrefly: ignore [missing-import]
    from k2_graph_timer import InvalidGraphTiming

    # pyrefly: ignore [missing-import]
    from k2_graph_timer import benchmark_graphs

    # pyrefly: ignore [missing-import]
    from k2_graph_timer import prepared_graph_batches
    from pretuned_kernels import _bench

    # The frozen external HKL tree was added to sys.path above.
    # pyrefly: ignore [missing-import]
    from repro.breadth import common

    # pyrefly: ignore [missing-import]
    from repro.breadth.compare_scaled_mm import audit

    # pyrefly: ignore [missing-import]
    from repro.breadth.compare_scaled_mm import bytes_of

    # pyrefly: ignore [missing-import]
    from repro.breadth.compare_scaled_mm import changed_input_audit

    # pyrefly: ignore [missing-import]
    from repro.gb300 import cases

    # pyrefly: ignore [missing-import]
    from repro.gb300 import providers
    import torch

    import helion
    from helion.autotuner.logger import classify_triton_exception
    from helion.autotuner.logger import match_unrecoverable_runtime_error
    import helion.runtime as helion_runtime
    from helion.runtime.settings import default_autotuner_fn

    module = importlib.import_module(
        "helion_kernel_library._kernels.scaled_mm.scaled_mm_cute"
    )
    if Path(helion.__file__).resolve().parent != ROOT / "helion":
        raise RuntimeError("Imported Helion is outside the requested worktree")
    module_file = module.__file__
    assert module_file is not None
    if Path(module_file).resolve() != (
        HKL / "helion_kernel_library/_kernels/scaled_mm/scaled_mm_cute.py"
    ):
        raise RuntimeError("Imported HKL kernel is outside the frozen worktree")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    result["device_identity"] = {
        "gpu_uuid": _selected_gpu_uuid(),
        "cuda_device": torch.cuda.current_device(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    # Slurm/container CUDA ordinals need not match nvidia-smi's physical indices.
    # Keep the frozen thermal algorithm, but query the GPU CUDA actually uses.
    common._selected_physical_gpu = _selected_gpu_uuid
    common.capture_startup_gpu_temperature()
    startup_temperature = common._gpu_temperature_c()
    result["startup_temperature_c"] = startup_temperature
    result["startup_gpu"] = _gpu_snapshot()
    result["helion_commit"] = _git(ROOT, "rev-parse", "HEAD")
    result["hkl_commit"] = _git(HKL, "rev-parse", "HEAD")
    result["helion_worktree_diff_sha256"] = hashlib.sha256(
        _git(ROOT, "diff", "HEAD", "--", "helion").encode()
    ).hexdigest()
    result["environment"] = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_commit": torch.version.git_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_uuid": _selected_gpu_uuid(),
        "cuda_device": torch.cuda.current_device(),
        "monitoring_device_selection": "current CUDA device UUID",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("triton", "nvidia-cutlass-dsl", "vllm")
        },
        "helion_environment": {
            key: value for key, value in os.environ.items() if key.startswith("HELION_")
        },
    }
    m, k, n = SHAPES[args.shape_index]
    shape = {"m": m, "k": k, "n": n, "scale_mode": args.scale_mode}
    if shape not in cases.panels(2):
        raise RuntimeError("Requested original case is absent from the frozen panel")
    result["shape"] = shape
    work = cases.make_workload(2, shape, seed=args.input_seed)
    a, weights, scale_a, scale_b = work.case
    normalized = (
        a,
        weights.T,
        scale_a.reshape(()) if scale_a.numel() == 1 else scale_a.reshape(-1),
        scale_b.reshape(()) if scale_b.numel() == 1 else scale_b.reshape(-1),
    )
    fingerprints = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(
                bytes_of(tensor).cpu().numpy().tobytes()
            ).hexdigest(),
        }
        for name, tensor in work.tensors.items()
    }
    result["inputs"] = fingerprints
    result["input_sha256"] = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True).encode()
    ).hexdigest()
    expected = work.reference()
    payload = json.loads(args.config.read_text()) if args.config is not None else None
    implementation, fixed_config, migration = _kernel_config_for_run(
        (m, k, n), args.phase, payload
    )
    kernel_args = normalized
    helion_call = work.call
    if implementation == "frontend_transposed":
        preset = importlib.import_module(
            "pretuned_kernels.scaled_mm_cute.scaled_mm_cute"
        )
        heuristic = importlib.import_module(
            "pretuned_kernels.scaled_mm_cute._helion_aot_scaled_mm_cute_cuda_sm103"
        )
        kernel = preset.scaled_mm_transposed_cute
        kernel_args = (normalized[1].T, normalized[0].T, *normalized[2:])
        seed_config = helion.Config(
            **heuristic.autotune_scaled_mm_transposed_cute(*kernel_args)
        )

        def frontend_call() -> torch.Tensor:
            return preset.scaled_mm(a, weights.T, scale_a, scale_b)

        helion_call = frontend_call
        preset_file = preset.__file__
        heuristic_file = heuristic.__file__
        assert preset_file is not None and heuristic_file is not None
        kernel_source = Path(preset_file)
        seed_source = Path(heuristic_file)
        public_entry = "pretuned_kernels.scaled_mm_cute.scaled_mm_cute.scaled_mm"
        argument_order = ["b.T", "a.T", "scale_a", "scale_b"]
    else:
        kernel = module.scaled_mm_cute
        seed_config = module.config_picker(normalized, platform="nvidia_sm103")
        kernel_source = Path(module_file)
        seed_source = (
            HKL / "helion_kernel_library/configs/scaled_mm_cute/nvidia_sm103.json"
        )
        public_entry = "helion_kernel_library.ops.scaled_mm"
        argument_order = ["a", "b", "scale_a", "scale_b"]
    result["kernel_implementation"] = implementation
    result["kernel_provenance"] = {
        "kernel_name": kernel.fn.__name__,
        "kernel_source": _source(kernel_source),
        "seed_config_source": _source(seed_source),
        "public_entry": public_entry,
        "physical_argument_order": argument_order,
        "argument_shapes": [list(value.shape) for value in kernel_args],
        "argument_strides": [list(value.stride()) for value in kernel_args],
    }
    if args.config is not None:
        result["fixed_config_source"] = _source(args.config.resolve())
        result["fixed_config_migration"] = migration
    kernel._key_fn = None
    settings = kernel.settings
    settings.autotune_effort = "full"
    settings.autotune_budget_seconds = None
    settings.autotune_max_generations = None
    settings.autotune_best_of_k = 1
    settings.autotune_random_seed = result["search_seed"]
    settings.autotune_initial_population_strategy = "from_random"
    settings.autotune_seed_configs = [seed_config]
    settings.autotune_config_overrides = {}
    settings.autotune_search_acf = []
    settings.autotune_cache = "LocalAutotuneCache"
    settings.autotune_ignore_errors = True
    settings.autotune_accuracy_check = True
    settings.disable_autotuner_heuristics = False
    settings.autotune_benchmark_subprocess = False
    settings.autotune_precompile = None
    settings.autotune_log = str(args.output_dir / "autotune.jsonl")
    settings.autotune_log_details = True
    settings.autotune_log_search_space = True
    settings.autotune_log_search_space_path = str(args.output_dir / "search_space.json")
    settings.autotune_baseline_fn = lambda *unused: expected
    settings.autotune_baseline_accuracy_check_fn = work.validate

    def measure_graphs(
        graphs: Sequence[torch.cuda.CUDAGraph],
        *,
        stage: str,
        batch_size: int,
        samples: int,
        context: dict[str, Any] | None = None,
        calibrate_candidate: bool = False,
    ) -> dict[str, Any]:
        try:
            calibration = None
            if calibrate_candidate:
                assert len(graphs) == 1
                requested_samples = samples
                samples = max(10, samples)
                estimate_ms = _bench.bench_pre_captured_cudagraph(
                    graphs[0].replay, rep=3
                )
                calibration = {
                    "estimate_ms": estimate_ms,
                    "estimate_timer": "pre-captured outer CUDA events; duration estimate only",
                    "estimate_repetitions": 3,
                    "target_batch_ms": 10.0,
                    "requested_batch_size": batch_size,
                    "requested_samples": requested_samples,
                    "samples": samples,
                }
                context = {**(context or {}), "calibration": calibration}
                if not math.isfinite(estimate_ms) or estimate_ms <= 0:
                    raise InvalidGraphTiming(
                        "Invalid preliminary candidate duration estimate", calibration
                    )
                batch_size = min(batch_size, max(1, math.ceil(10.0 / estimate_ms)))
                calibration["used_batch_size"] = batch_size
            measured = benchmark_graphs(
                graphs,
                batch_size=batch_size,
                samples=samples,
                warmup_ms=args.timer_warmup_ms,
            )
            if calibration is not None:
                measured["calibration"] = calibration
            return measured
        except InvalidGraphTiming as error:
            path = (
                args.output_dir
                / f"measurement_failure_{stage}_{secrets.token_hex(8)}.json"
            )
            failure = {
                "status": "MEASUREMENT_ERROR",
                "stage": stage,
                "error": str(error),
                "measurement": error.measurement,
                "context": context,
                "batch_size": batch_size,
                "requested_samples": samples,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with path.open("x") as stream:
                json.dump(failure, stream, indent=2, sort_keys=True, default=str)
                stream.write("\n")
            result.setdefault("measurement_failures", []).append(str(path))
            raise

    def capture(
        call: Callable[[], _CaptureOutput],
    ) -> tuple[torch.cuda.CUDAGraph, _CaptureOutput]:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with helion_runtime.cute_cuda_graph(stream=stream) as graph:
            graph.enable_debug_mode()
            output = call()
        graph.instantiate()
        torch.cuda.synchronize()
        return graph, output

    class ColdGraphProvider(_bench.CapturedCudagraphBenchmarkProvider):
        def _capture_validated_replay(
            self, config: helion.Config, fn: Callable[..., torch.Tensor]
        ) -> Callable[[], object] | None:
            output = fn(*self.args)
            torch.cuda.synchronize()
            if not self._validate_against_baseline(config, output, self.args):
                self._record_accuracy_failure(config)
                return None
            graph, captured_output = capture(functools.partial(fn, *self.args))
            captured_output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            if not self._validate_against_baseline(config, captured_output, self.args):
                self._record_accuracy_failure(config)
                return None
            return graph.replay

        def _benchmark_function(
            self,
            config: helion.Config,
            fn: Callable[..., torch.Tensor],
            *,
            effective_source_hash: str | None = None,
        ) -> float:
            # Keep the shared provider's validation/error handling. The timer
            # batches launches to amortize CUDA event processing on this GPU.
            self._autotune_metrics.num_configs_tested += 1
            try:
                replay = self._capture_validated_replay(config, fn)
                if replay is None:
                    return float("inf")
                measured = measure_graphs(
                    [cast("torch.cuda.CUDAGraph", cast("MethodType", replay).__self__)],
                    stage="candidate",
                    batch_size=args.candidate_batch_size,
                    samples=args.timer_samples,
                    context={
                        "config": json.loads(config.to_json()),
                        "effective_source_hash": effective_source_hash,
                    },
                    calibrate_candidate=True,
                )
                with (args.output_dir / "candidate_timings.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "config": json.loads(config.to_json()),
                                "effective_source_hash": effective_source_hash,
                                "measurement": measured,
                            }
                        )
                        + "\n"
                    )
                return measured["median_ms"][0]
            except InvalidGraphTiming:
                # Measurement failures are not rejected kernel configurations.
                raise
            except Exception as error:
                error.__traceback__ = None
                if match_unrecoverable_runtime_error(error):
                    self.kernel.maybe_log_repro(self.log.error, self.args, config)
                    raise helion.exc.TritonUnrecoverableRuntimeError(
                        reason=str(error),
                        decorator=self.kernel.format_kernel_decorator(
                            config, self.settings
                        ),
                        error=f"{type(error).__qualname__}: {error}",
                    ) from error
                action = self.config_spec.backend.classify_autotune_exception(
                    error
                ) or classify_triton_exception(error)
                if action == "raise" and not self.settings.autotune_ignore_errors:
                    raise
                self.log.debug(
                    f"Skipping captured candidate after {type(error).__name__}: {error}"
                )
                self._record_compile_failure(config)
                return float("inf")
            finally:
                self._clear_jit_fast_path_caches(fn)

    rebenchmark_rounds: list[dict[str, Any]] = []

    def rebenchmark(
        calls: list[Callable[[], object]], *, repeat: int, desc: str | None = None
    ) -> list[float]:
        graphs = []
        outputs = []
        for call in calls:
            graph, output = capture(call)
            graphs.append(graph)
            outputs.append(output)
        sample_count = max(
            args.timer_samples, math.ceil(repeat / args.candidate_batch_size)
        )
        measured = measure_graphs(
            graphs,
            stage="rebenchmark",
            batch_size=args.candidate_batch_size,
            samples=sample_count,
            context={"requested_repeat": repeat, "desc": desc},
        )
        rebenchmark_rounds.append(
            {
                "requested_repeat": repeat,
                "actual_samples": sample_count,
                "desc": desc,
                "measurement": measured,
            }
        )
        _write(args.output_dir / "rebenchmark_rounds.json", rebenchmark_rounds)
        return measured["median_ms"]

    settings.autotune_benchmark_fn = rebenchmark
    searches = []

    def autotuner(
        bound: BoundKernel, values: Sequence[object], **kwargs: object
    ) -> BaseAutotuner:
        wrapped = default_autotuner_fn(bound, values, **kwargs)
        # At this compiler revision LFBO does not expose provider_cls as a public
        # constructor argument. Set its provider before _prepare creates it.
        search = cast("LocalAutotuneCache", wrapped).autotuner
        search._benchmark_provider_cls = functools.partial(
            ColdGraphProvider, repetitions=args.candidate_batch_size
        )
        result["search_profile"] = search._algorithm_cache_policy()
        _write(args.output_dir / "search_profile.json", result["search_profile"])
        searches.append(search)
        return wrapped

    settings.autotuner_fn = autotuner
    bound = kernel.bind(kernel_args)
    result["seed_config"] = json.loads(seed_config.to_json())
    result["effective_settings"] = settings.to_dict()
    result["autotune_protocol"] = {
        "effort": "full",
        "wall_budget_seconds": None,
        "generation_override": None,
        "force": True,
        "cache_reads": False,
        "cache_writes": False,
        "user_seed": f"current {implementation} mode-selected config; not a search restriction",
        "candidate_batch_size": args.candidate_batch_size,
        "finalist_batch_size": args.candidate_batch_size,
        "timer_samples": args.timer_samples,
        "timer_warmup_ms": args.timer_warmup_ms,
        "candidate_batch_calibration": {
            "target_batch_ms": 10.0,
            "estimate_repetitions": 3,
            "estimate_timer": "pre-captured outer CUDA events; never the search objective",
            "maximum_batch_size": args.candidate_batch_size,
            "minimum_batch_size": 1,
            "samples": max(10, args.timer_samples),
            "minimum_samples": 10,
            "finalist_and_final_calibration": False,
        },
        "candidate_execution": "in-process captured graph; no per-candidate kill timeout",
        "search": "full LFBOTreeSearch profile, with a cold graph provider",
        # A custom autotune_benchmark_fn turns off Helion's final-selection
        # shootout, so record that the full profile ran without it.
        "final_selection": "disabled by the custom autotune_benchmark_fn",
    }
    _write(args.output_dir / "started.json", result)
    started = time.monotonic()
    if args.phase == "cold":
        result["search_thermal_warmup"] = {
            "duration_ms": args.thermal_warmup_ms,
            "before": _gpu_snapshot(),
        }
        _bench.thermal_warmup(args.thermal_warmup_ms)
        result["search_thermal_warmup"]["after"] = _gpu_snapshot()
        winner = bound.autotune(kernel_args, force=True)
        result["autotune_metrics"] = searches[-1]._autotune_metrics.to_dict()
    else:
        assert fixed_config is not None
        winner = helion.Config(**fixed_config)
        bound.set_config(winner)
    result["autotune_elapsed_s"] = time.monotonic() - started
    result["config"] = json.loads(winner.to_json())
    # Future physical tiles can also be legal MN tiles. Preserve the selected
    # body so --config winner.json rebinds them without guessing coordinates.
    _write(
        args.output_dir / "winner.json",
        {"config": result["config"], "kernel_implementation": implementation},
    )
    (args.output_dir / "generated.py").write_text(bound.to_code(winner))
    result["generated_source"] = _source(args.output_dir / "generated.py")

    calls = {"helion-cute": helion_call}
    cutlass, cutlass_metadata = providers.prepare(2, work, shape)
    calls["vllm-cutlass"] = cutlass
    result["providers"] = {
        "helion-cute": {"config": result["config"]},
        "vllm-cutlass": cutlass_metadata,
    }

    def torch_call() -> torch.Tensor:
        # Mixed scalar/row modes require rowwise scaling on both operands in
        # torch._scaled_mm. All necessary live-scale copies stay inside capture.
        sa, sb = scale_a, scale_b
        if sa.numel() != 1 or sb.numel() != 1:
            sa = sa.reshape(-1, 1).expand(m, 1).contiguous()
            sb = sb.reshape(1, -1).expand(1, n).contiguous()
        return torch._scaled_mm(a, weights.T, sa, sb, out_dtype=torch.bfloat16)

    try:
        torch_output = torch_call()
        torch.cuda.synchronize()
    except (RuntimeError, NotImplementedError) as error:
        result["providers"]["torch"] = {
            "status": "UNSUPPORTED",
            "error": f"{type(error).__name__}: {error}",
        }
    else:
        work.validate(torch_output, expected)
        calls["torch"] = torch_call
        result["providers"]["torch"] = {
            "entry": "torch._scaled_mm",
            "scale_adapter": "live scalar expansion inside timed call for mixed modes",
            "config": "library-selected; no exposed user configuration",
        }
    if args.quack_config is not None:
        # The provider is a sibling loaded only when this optional baseline is used.
        # pyrefly: ignore [missing-import]
        from k2_scaled_mm_baselines import prepare_quack_scaled_mm

        payload = json.loads(args.quack_config.read_text())
        prepared = prepare_quack_scaled_mm(
            a, weights.T, scale_a, scale_b, payload.get("config", payload)
        )
        calls["quack-cute"] = prepared.call
        result["providers"]["quack-cute"] = {
            "config": prepared.config,
            "provenance": prepared.provenance,
            "config_source": _source(args.quack_config.resolve()),
        }
    else:
        result["providers"]["quack-cute"] = {"status": "NOT_REQUESTED"}

    audit_args = argparse.Namespace(correctness_launches=3, atol=None, rtol=None)
    graphs = {}
    outputs = {}
    for name, call in calls.items():
        snapshots = [(t, bytes_of(t).clone()) for t in work.tensors.values()]
        eager = audit(call, expected, snapshots, audit_args)
        graph, output = capture(call)

        def replay(
            g: torch.cuda.CUDAGraph = graph, out: torch.Tensor = output
        ) -> torch.Tensor:
            g.replay()
            return out

        graph_check = audit(replay, expected, snapshots, audit_args, poison=output)
        live = changed_input_audit(
            replay, a, weights, scale_a, scale_b, audit_args, poison=output
        )
        restored = audit(replay, expected, snapshots, audit_args, poison=output)
        graphs[name] = graph
        outputs[name] = output
        result["providers"][name].update(
            status="PASS",
            correctness=eager,
            graph_correctness=graph_check,
            live_input_checks=live,
            restored_graph_correctness=restored,
        )
    with prepared_graph_batches(list(graphs.values()), batch_size=2) as batches:
        for index, name in enumerate(graphs):
            output = outputs[name]

            def batch_replay(
                i: int = index, out: torch.Tensor = output
            ) -> torch.Tensor:
                batches.replay(i)
                return out

            snapshots = [(t, bytes_of(t).clone()) for t in work.tensors.values()]
            raw_check = audit(
                batch_replay, expected, snapshots, audit_args, poison=output
            )
            raw_live = changed_input_audit(
                batch_replay, a, weights, scale_a, scale_b, audit_args, poison=output
            )
            raw_restored = audit(
                batch_replay, expected, snapshots, audit_args, poison=output
            )
            result["providers"][name].update(
                raw_batch_correctness=raw_check,
                raw_batch_live_input_checks=raw_live,
                raw_batch_restored_correctness=raw_restored,
            )
    # Public calls must resolve to the exact bound winner, with no second tune.
    if kernel.bind(kernel_args) is not bound or bound._config != winner:
        raise RuntimeError("Public wrapper did not retain the selected bound config")
    if startup_temperature is not None:
        result["cooldown"] = common._wait_for_gpu_cooldown(
            startup_temperature + args.cooldown_margin_c
        )
    result["measurement_thermal_warmup"] = {
        "duration_ms": args.thermal_warmup_ms,
        "before": _gpu_snapshot(),
    }
    _bench.thermal_warmup(args.thermal_warmup_ms)
    result["measurement_thermal_warmup"]["after"] = _gpu_snapshot()
    names = list(graphs)
    rounds = []
    components = []
    for round_index in range(args.rounds):
        order = names[round_index % len(names) :] + names[: round_index % len(names)]
        if round_index % 2:
            order.reverse()
        measured = measure_graphs(
            [graphs[name] for name in order],
            stage="final",
            batch_size=args.batch_size,
            samples=args.timer_samples,
            context={"round_index": round_index, "order": order},
        )
        components.append({"order": order, "measurement": measured})
        rounds.append(
            dict(zip(order, [t * 1000 for t in measured["median_ms"]], strict=True))
        )
    result["post_measurement_gpu"] = _gpu_snapshot()
    medians = {name: statistics.median(row[name] for row in rounds) for name in names}
    baseline = min(
        (name for name in names if name != "helion-cute"), key=medians.__getitem__
    )
    result["timing"] = {
        "timer": "CUDA events around batched child graphs; paired cache-clear-only subtraction",
        "l2": "256 MiB clear before every preserved child graph invocation",
        "ordering": "balanced providers and total/clear pair ordering",
        "batch_size": args.batch_size,
        "samples_per_round": args.timer_samples,
        "timer_warmup_ms": args.timer_warmup_ms,
        "rounds": args.rounds,
        "raw_components": components,
        "per_round_median_us": rounds,
        "median_us": medians,
        "best_baseline": baseline,
        "speedup_vs_best_baseline": medians[baseline] / medians["helion-cute"],
        "speedup_vs_cutlass": medians["vllm-cutlass"] / medians["helion-cute"],
        "tflops": {
            name: 2 * m * k * n / (latency * 1e6) for name, latency in medians.items()
        },
    }
    result["status"] = "PASS"


def main() -> int:
    args = _parse_args()
    result: dict[str, Any] = {
        "status": "ERROR",
        "phase": args.phase,
        "shape_index": args.shape_index,
        "scale_mode": args.scale_mode,
        "input_seed": args.input_seed,
        "search_seed": args.search_seed
        if args.search_seed is not None
        else secrets.randbits(32),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "execution": _execution_provenance(),
    }
    sources = _manifest()
    _write(args.output_dir / "sources.json", sources)
    (args.output_dir / "worker_source.py").write_bytes(Path(__file__).read_bytes())
    (args.output_dir / "timer_source.py").write_bytes(
        (ROOT / "benchmarks/cute/k2_graph_timer.py").read_bytes()
    )
    (args.output_dir / "compiler.patch").write_text(
        _git(ROOT, "diff", "HEAD", "--", "helion") + "\n"
    )
    try:
        _run(args, result)
        if _manifest() != sources:
            raise RuntimeError("Source manifest changed during this worker")
        result["source_manifest"] = _source(args.output_dir / "sources.json")
        result["source_unchanged"] = True
    except Exception as error:
        result.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        (args.output_dir / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(args.output_dir / "result.json", result)
    print(
        json.dumps({"status": result["status"], "output": str(args.output_dir)}),
        flush=True,
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
