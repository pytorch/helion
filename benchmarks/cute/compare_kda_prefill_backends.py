"""Reproduce the two reportable KDA prefill comparisons.

Each case uses its frozen exhaustive-search winner. Helion is compared with
the public FlashInfer CAKE and CuTe-DSL routes using private one-invocation
CUDA graphs and balanced ABBA/BAAB timing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import functools
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.cute import kda_benchmark_utils as benchmark_utils  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import (  # noqa: E402
    assert_stable as _assert_stable,
)
from benchmarks.cute.kda_benchmark_utils import (  # noqa: E402, F401
    file_sha256 as _sha256,
)
from benchmarks.cute.kda_benchmark_utils import gpu_info as _gpu_info  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import (  # noqa: E402
    source_provenance as _source_provenance,
)
from benchmarks.cute.kda_prefill_kernels import KDA_PREPARE_CONFIG  # noqa: E402
from benchmarks.cute.kda_prefill_kernels import KDA_RECURRENCE_CONFIG  # noqa: E402
from benchmarks.cute.kda_prefill_kernels import kda_chunk_prepare  # noqa: E402
from benchmarks.cute.kda_prefill_kernels import kda_chunk_recurrence  # noqa: E402
from benchmarks.cute.kda_prefill_staged import create_staged_kda_resources  # noqa: E402
from benchmarks.cute.kda_prefill_staged import launch_staged_kda_prefill  # noqa: E402
import torch  # noqa: E402

import helion  # noqa: E402
from helion.runtime.config import Config  # noqa: E402

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel

DEFAULT_FLASHINFER_SHA = "f67bc2ed555c1ad6a764ad68f7aa9622178e9eae"
HEAD_DIM = 128
SAFE_LOWER_BOUND = -5.0
LOG2_E = math.log2(math.e)

_CUTE_CHUNK_PREPARE_SCHEDULE_KEY = "cute_chunk_prepare_schedule"
_CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY = "cute_chunk_recurrence_dv_partitions"
_CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY = "cute_chunk_recurrence_register_cap"

_BT16_PREPARE_VARIANTS = frozenset(("bt16_prepare", "bt16_prepare_beta_tma"))
_BT16_CHAIN_VARIANTS = frozenset(
    ("bt16_chain_m64_s7", "bt16_chain_m64_s8", "bt16_chain_m64_s9")
)
_BT16_COMBINED_VARIANTS = frozenset(("bt16_prepare_chain_m64_s8",))


@dataclass(frozen=True)
class Case:
    name: str
    num_heads: int
    seq_lens: tuple[int, ...]
    packed: bool
    seed: int
    lower_bound: float = SAFE_LOWER_BOUND
    expected_cake_variant: str = "bt16_prepare_chain_m64"

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def batch_size(self) -> int:
        return 1 if self.packed else len(self.seq_lens)

    @property
    def num_sequences(self) -> int:
        return len(self.seq_lens)


CASES = {
    case.name: case
    for case in (
        Case("h12_fixed_512", 12, (512,), False, 12002),
        Case("h12_packed_mixed", 12, (1300, 547, 2048, 963, 271, 3063), True, 12004),
    )
}
OFFICIAL_CASES = tuple(CASES)
IMPLEMENTATIONS = ("helion-cute", "flashinfer-cake", "flashinfer-cute")


@dataclass(frozen=True)
class HelionConfig:
    topology: Literal["origin_aux", "serial"]
    critical_prepare_schedule: str
    auxiliary_prepare_schedule: str | None
    recurrence_dv_partitions: int
    recurrence_register_cap: int | None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "staged_topology": self.topology,
            "critical_prepare_schedule": self.critical_prepare_schedule,
            "recurrence_dv_partitions": self.recurrence_dv_partitions,
            "recurrence_register_cap": self.recurrence_register_cap,
        }
        if self.auxiliary_prepare_schedule is not None:
            result["auxiliary_prepare_schedule"] = self.auxiliary_prepare_schedule
        return result


# Winners from the complete 25/250-candidate cold search, kept immutable here.
PINNED_HELION_CONFIGS = {
    "h12_fixed_512": HelionConfig("serial", "split_alias_cpc1", None, 4, None),
    "h12_packed_mixed": HelionConfig(
        "origin_aux", "split_alias_cpc2", "split_alias_cpc2", 2, None
    ),
}


def _semantic_metadata(case: Case) -> dict[str, Any]:
    return {
        "case": case.name,
        "seed": case.seed,
        "shape": {
            "batch_size": case.batch_size,
            "num_sequences": case.num_sequences,
            "total_tokens": case.total_tokens,
            "num_heads": case.num_heads,
            "head_dim": HEAD_DIM,
            "seq_lens": list(case.seq_lens),
        },
        "layout": "packed" if case.packed else "fixed",
        "activation_dtype": "bfloat16",
        "canonical_state_dtype": "bfloat16",
        "canonical_state_layout": "[sequence,head,value,key]",
        "qk_l2norm": True,
        "qk_l2norm_epsilon": 1.0e-24,
        "scale": HEAD_DIM**-0.5,
        "gate_input": "raw",
        "gate_mode": "lower-bound-sigmoid",
        "lower_bound": case.lower_bound,
        "beta_input": "logit",
        "indexed_state": False,
        "checkpoint_every_n_tokens": 0,
        "input_distribution": {
            "q_k_v_raw_gate": "normal-float32-rounded-to-bfloat16",
            "raw_beta": "normal-float32-rounded-to-bfloat16",
            "a_log": "uniform-[0,1)-float32",
            "dt_bias": "uniform-[0,1)-float32",
            "initial_state": "normal-std-0.25-float32-rounded-to-bfloat16",
        },
        "timing": "one-call-captured-graph-cuda-events-with-cold-l2",
        "mutation_isolation": "private-state-and-output-per-captured-graph",
    }


@dataclass
class KDAInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    raw_gate: torch.Tensor
    raw_beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    initial_state: torch.Tensor
    cu_seqlens: torch.Tensor | None

    def clone_mutable(self) -> KDAInputs:
        return KDAInputs(
            q=self.q,
            k=self.k,
            v=self.v,
            raw_gate=self.raw_gate,
            raw_beta=self.raw_beta,
            a_log=self.a_log,
            dt_bias=self.dt_bias,
            initial_state=self.initial_state.clone(),
            cu_seqlens=self.cu_seqlens,
        )


Invoke = Callable[[KDAInputs, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _physical_gpu_index() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = visible.split(",", 1)[0].strip()
    return first or "0"


def _gpu_field(field: str) -> float | None:
    try:
        output = subprocess.run(
            (
                "nvidia-smi",
                "-i",
                _physical_gpu_index(),
                f"--query-gpu={field}",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        return float(output.splitlines()[0])
    except (IndexError, OSError, subprocess.SubprocessError, ValueError):
        return None


def _wait_for_cooldown(target_c: float, timeout_s: float) -> dict[str, Any]:
    torch.cuda.synchronize()
    start = _gpu_field("temperature.gpu")
    current = start
    started = time.monotonic()
    while current is not None and current > target_c:
        if time.monotonic() - started >= timeout_s:
            break
        time.sleep(5)
        current = _gpu_field("temperature.gpu")
    if start is None or current is None:
        status = "unavailable"
    elif current > target_c:
        status = "timeout"
    else:
        status = "cooled"
    return {
        "status": status,
        "valid": status == "cooled",
        "start_temp_c": start,
        "end_temp_c": current,
        "waited_s": round(time.monotonic() - started, 1),
        "target_c": target_c,
        "timeout_s": timeout_s,
    }


def _prepare_build_environment() -> None:
    executable_bin = str(Path(sys.executable).resolve().parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if executable_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join((executable_bin, *path_parts))
    cuda_root = Path("/usr/local/cuda")
    if cuda_root.is_dir():
        os.environ.setdefault("CUDA_HOME", str(cuda_root))


def _git_files(root: Path, pathspecs: tuple[str, ...]) -> dict[str, Path]:
    output = subprocess.run(
        ("git", "ls-files", "--cached", "-z", "--", *pathspecs),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        relative: root / relative
        for relative in sorted(
            item.decode("utf-8") for item in output.split(b"\0") if item
        )
    }


def _flashinfer_sources(root: Path) -> dict[str, Path]:
    relative_paths = (
        "flashinfer/api_logging.py",
        "flashinfer/kda.py",
        "flashinfer/kda_prefill.py",
        "flashinfer/kda_prefill_cute.py",
        "flashinfer/utils.py",
        "flashinfer/trace/template.py",
        "flashinfer/trace/templates/kda.py",
        "flashinfer/jit/_kda_jit_common.py",
        "flashinfer/jit/cake_kda.py",
        "flashinfer/jit/core.py",
        "flashinfer/jit/cpp_ext.py",
        "flashinfer/jit/flash_kda.py",
    )
    sources = {relative: root / relative for relative in relative_paths}
    sources.update(
        _git_files(
            root,
            ("csrc/kda", "flashinfer/cute_dsl", "flashinfer/kda_kernels"),
        )
    )
    return sources


def _install_namespace(name: str, path: Path) -> ModuleType:
    expected_path = path.resolve()
    existing = sys.modules.get(name)
    if isinstance(existing, ModuleType):
        module_path = existing.__dict__.get("__path__")
        actual_paths = (
            ()
            if module_path is None
            else tuple(Path(item).resolve() for item in module_path)
        )
        if actual_paths != (expected_path,):
            raise ImportError(
                f"preloaded namespace {name} is outside the requested checkout: "
                f"expected={(expected_path,)}, actual={actual_paths}"
            )
        return existing
    module = ModuleType(name)
    module.__path__ = [str(expected_path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _load_module(name: str, path: Path) -> ModuleType:
    expected_path = path.resolve()
    existing = sys.modules.get(name)
    if isinstance(existing, ModuleType):
        actual_path = existing.__dict__.get("__file__")
        if actual_path is None or Path(actual_path).resolve() != expected_path:
            raise ImportError(
                f"preloaded module {name} is outside the requested checkout: "
                f"expected={expected_path}, actual={actual_path}"
            )
        return existing
    spec = importlib.util.spec_from_file_location(name, expected_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load {name} from {expected_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _unavailable_decode(*args: object, **kwargs: object) -> object:
    raise RuntimeError("decode-only FlashInfer function is unavailable in this loader")


def _load_flashinfer(root: Path) -> ModuleType:
    """Load only the source modules required by public recurrent-KDA prefill."""
    root = root.resolve()
    package_root = root / "flashinfer"
    source = package_root / "kda.py"
    if not source.is_file():
        raise FileNotFoundError(f"not a FlashInfer source checkout: {root}")

    if "pynvml" not in sys.modules and importlib.util.find_spec("pynvml") is None:
        sys.modules["pynvml"] = ModuleType("pynvml")
    _install_namespace("flashinfer", package_root)
    _install_namespace("flashinfer.jit", package_root / "jit")
    _install_namespace("flashinfer.cute_dsl", package_root / "cute_dsl")
    _install_namespace("flashinfer.trace", package_root / "trace")
    _install_namespace(
        "flashinfer.trace.templates", package_root / "trace" / "templates"
    )

    # The facade imports decode/training symbols even though this benchmark only
    # exercises prefill. Keep those optional extensions out of the process.
    kernels = _install_namespace("flashinfer.kda_kernels", package_root / "kda_kernels")
    kernels.run_packed_kda_decode = _unavailable_decode  # type: ignore[attr-defined]
    kernels.run_recurrent_kda = _unavailable_decode  # type: ignore[attr-defined]
    fused = ModuleType("flashinfer.kda_kernels.fused_kda_decode")
    fused.run_fused_kda_decode = _unavailable_decode  # type: ignore[attr-defined]
    sys.modules[fused.__name__] = fused
    output_only = ModuleType("flashinfer.kda_kernels.kda_decode_wy_output_only")
    output_only.kda_wy_output_only = _unavailable_decode  # type: ignore[attr-defined]
    sys.modules[output_only.__name__] = output_only

    backward = ModuleType("flashinfer.kda_backward")
    backward.RecurrentKDABackwardWorkspace = type(  # type: ignore[attr-defined]
        "RecurrentKDABackwardWorkspace", (), {}
    )
    backward.recurrent_kda_backward = _unavailable_decode  # type: ignore[attr-defined]
    sys.modules[backward.__name__] = backward
    training = ModuleType("flashinfer.kda_training")
    training.RecurrentKDATrainingContext = type(  # type: ignore[attr-defined]
        "RecurrentKDATrainingContext", (), {}
    )
    training.recurrent_kda_training_backward = _unavailable_decode  # type: ignore[attr-defined]
    training.recurrent_kda_training_forward = _unavailable_decode  # type: ignore[attr-defined]
    sys.modules[training.__name__] = training

    importlib.import_module("flashinfer.trace.templates.kda")
    return _load_module("flashinfer.kda", source)


def _identity_tensor_cache(
    fn: Callable[[torch.Tensor, int], torch.Tensor],
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    cache: list[tuple[torch.Tensor, int, int, torch.Tensor]] = []

    @functools.wraps(fn)
    def wrapper(tensor: torch.Tensor, chunk_size: int) -> torch.Tensor:
        version = tensor._version
        for cached_tensor, cached_version, cached_chunk, result in cache:
            if (
                tensor is cached_tensor
                and version == cached_version
                and chunk_size == cached_chunk
            ):
                return result
        result = fn(tensor, chunk_size)
        cache.append((tensor, version, chunk_size, result))
        del cache[:-4]
        return result

    return wrapper


@_identity_tensor_cache
def _prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_counts = torch.div(
        lengths + chunk_size - 1, chunk_size, rounding_mode="floor"
    )
    local_chunks = torch.cat(
        [
            torch.arange(int(count), device=cu_seqlens.device)
            for count in chunk_counts.tolist()
        ]
    )
    sequence = torch.repeat_interleave(
        torch.arange(chunk_counts.numel(), device=cu_seqlens.device), chunk_counts
    )
    return torch.stack((sequence, local_chunks), dim=1).to(cu_seqlens)


@_identity_tensor_cache
def _prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_counts = torch.div(
        lengths + chunk_size - 1, chunk_size, rounding_mode="floor"
    )
    return torch.cat((cu_seqlens.new_zeros(1), chunk_counts)).cumsum(0)


def _make_inputs(case: Case) -> KDAInputs:
    generator = torch.Generator(device="cuda").manual_seed(case.seed)
    shape = (
        case.batch_size,
        case.total_tokens // case.batch_size,
        case.num_heads,
        HEAD_DIM,
    )

    def random_bf16(tensor_shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(
            tensor_shape,
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        ).to(torch.bfloat16)

    q = random_bf16(shape)
    k = random_bf16(shape)
    v = random_bf16(shape)
    raw_gate = random_bf16(shape)
    raw_beta = random_bf16(shape[:-1])
    a_log = torch.rand(
        case.num_heads,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    dt_bias = torch.rand(
        case.num_heads,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    initial_state = (
        torch.randn(
            case.num_sequences,
            case.num_heads,
            HEAD_DIM,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.25
    ).to(torch.bfloat16)
    if case.packed:
        offsets = [0]
        for length in case.seq_lens:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    else:
        cu_seqlens = None
    return KDAInputs(
        q=q,
        k=k,
        v=v,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )


def _invoke_flashinfer(
    module: ModuleType,
    case: Case,
    backend: str,
    *,
    prefill_workspace: object | None = None,
) -> Invoke:
    if backend not in ("cake", "cute-dsl"):
        raise ValueError(f"unsupported FlashInfer backend: {backend}")

    def invoke(
        inputs: KDAInputs, output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs: dict[str, Any] = {
            "q": inputs.q,
            "k": inputs.k,
            "v": inputs.v,
            "g": inputs.raw_gate,
            "beta": inputs.raw_beta,
            "A_log": inputs.a_log,
            "dt_bias": inputs.dt_bias,
            "scale": HEAD_DIM**-0.5,
            "initial_state": inputs.initial_state,
            "output_final_state": False,
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "lower_bound": case.lower_bound,
            "cu_seqlens": inputs.cu_seqlens,
            "ssm_state_indices": None,
            "output": output,
            "beta_is_logit": True,
            "state_checkpoints": None,
            "checkpoint_cu_starts": None,
            "checkpoint_every_n_tokens": 0,
            "backend": backend,
        }
        if prefill_workspace is not None:
            kwargs["prefill_workspace"] = prefill_workspace
        result = module.recurrent_kda(**kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise AssertionError(
                f"unexpected recurrent_kda result: {type(result).__name__}"
            )
        return result[0], inputs.initial_state

    return invoke


def _resolve_safe_gate_modules(
    modules: list[dict[str, str]],
) -> tuple[str, str, list[str]]:
    """Validate one physical safe-gate CAKE dispatch observation."""
    routes = [(item["variant"], item["target"]) for item in modules]
    if len(routes) == 1:
        variant, target = routes[0]
        if variant in _BT16_COMBINED_VARIANTS:
            return "bt16_prepare_chain_m64", target, [variant]
    elif len(routes) == 2:
        (prepare_variant, prepare_target), (chain_variant, chain_target) = routes
        if (
            prepare_variant in _BT16_PREPARE_VARIANTS
            and chain_variant in _BT16_CHAIN_VARIANTS
            and prepare_target == chain_target
        ):
            return (
                "bt16_prepare_chain_m64",
                prepare_target,
                [prepare_variant, chain_variant],
            )
    raise AssertionError(
        "expected one combined safe-gate CAKE module or one ordered BT16 "
        f"prepare/chain pair, got {routes}"
    )


def _record_flashinfer_route(
    module: ModuleType,
    invoke: Invoke,
    metadata: dict[str, Any],
    *,
    case: Case,
    expected_route: str,
) -> Invoke:
    """Record and validate the public CAKE or CuTe route on first invocation."""
    if expected_route not in ("cake-safe-gate", "cute-dsl"):
        raise ValueError(f"unsupported prefill route: {expected_route}")
    recorded = False

    def recording_invoke(
        inputs: KDAInputs, output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal recorded
        if recorded:
            return invoke(inputs, output)

        prefill_module = module._kda_prefill
        cute_module = module._kda_prefill_cute
        original_get_flash_module = prefill_module._get_flash_kda_prefill_module
        original_cute_run = cute_module._run_cute_dsl_kda_prefill
        cake_modules: list[dict[str, str]] = []
        cute_dispatches = 0

        def recording_get_flash_module(variant: str, target: str) -> object:
            cake_modules.append({"variant": variant, "target": target})
            return original_get_flash_module(variant, target)

        def recording_cute_run(**kwargs: object) -> object:
            nonlocal cute_dispatches
            cute_dispatches += 1
            return original_cute_run(**kwargs)

        prefill_module._get_flash_kda_prefill_module = recording_get_flash_module
        cute_module._run_cute_dsl_kda_prefill = recording_cute_run
        try:
            result = invoke(inputs, output)
        finally:
            prefill_module._get_flash_kda_prefill_module = original_get_flash_module
            cute_module._run_cute_dsl_kda_prefill = original_cute_run

        if cute_dispatches:
            if cute_dispatches != 1 or cake_modules:
                raise AssertionError(
                    "CuTe route must dispatch exactly once without loading CAKE"
                )
            sm_count = torch.cuda.get_device_properties(
                inputs.q.device
            ).multi_processor_count
            variant = (
                "decomp"
                if case.num_sequences * case.num_heads * 2 <= sm_count
                else "engine"
            )
            resolved_route = f"cute-dsl-{variant}"
            metadata["cute_variant"] = variant
        elif cake_modules:
            logical_variant, target, physical_variants = _resolve_safe_gate_modules(
                cake_modules
            )
            if logical_variant != case.expected_cake_variant:
                raise AssertionError(
                    f"expected CAKE variant {case.expected_cake_variant}, "
                    f"got {logical_variant}"
                )
            expected_target = prefill_module._select_flash_kda_prefill_target(
                inputs.q.device
            )
            if target != expected_target:
                raise AssertionError(
                    f"expected CAKE target {expected_target}, got {target}"
                )
            resolved_route = "cake-safe-gate"
            metadata.update(
                {
                    "cake_logical_variant": logical_variant,
                    "cake_target": target,
                    "cake_physical_variants": physical_variants,
                }
            )
        else:
            raise AssertionError("no FlashInfer prefill route was observed")

        if not (
            resolved_route.startswith("cute-dsl-")
            if expected_route == "cute-dsl"
            else resolved_route == expected_route
        ):
            raise AssertionError(
                f"expected FlashInfer route {expected_route}, got {resolved_route}"
            )
        metadata.update(
            {
                "requested_route": expected_route,
                "resolved_route": resolved_route,
                "physical_modules": cake_modules,
                "cute_dispatches": cute_dispatches,
                "indexed_state": False,
            }
        )
        recorded = True
        return result

    return recording_invoke


def _check_result(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    actual_output, actual_state = actual
    expected_output, expected_state = expected
    torch.cuda.synchronize()
    output_difference = (actual_output.float() - expected_output.float()).abs()
    state_difference = (actual_state.float() - expected_state.float()).abs()
    output_max_abs = float(output_difference.max())
    state_max_abs = float(state_difference.max())
    output_rms = float(output_difference.square().mean().sqrt())
    state_rms = float(state_difference.square().mean().sqrt())
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=2e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=2e-2, rtol=1e-2
    )
    return {
        "output_max_abs": output_max_abs,
        "state_max_abs": state_max_abs,
        "output_rms": output_rms,
        "state_rms": state_rms,
    }


@dataclass(frozen=True)
class _GeneratedRoute:
    role: str
    phase: str
    bound: BoundKernel[object]
    config: Config
    compiled: object
    plan_kind: str
    source_marker: str
    source_pattern: str

    def verify(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "phase": self.phase,
            "config": dict(self.config.config),
            **benchmark_utils.verify_helion_generated_wrapper(
                self.bound,
                self.config,
                expected_plan_kind=self.plan_kind,
                expected_source_markers=(self.source_marker,),
                expected_source_patterns=(self.source_pattern,),
                compiled_fn=self.compiled,
            ),
        }


@dataclass
class _BenchmarkArm:
    name: str
    inputs: KDAInputs
    output: torch.Tensor
    pristine_state: torch.Tensor
    launch: Callable[[], object]
    owners: tuple[object, ...]
    route: dict[str, Any]
    generated_routes: tuple[_GeneratedRoute, ...] = ()

    def reset(self) -> None:
        self.inputs.initial_state.copy_(self.pristine_state)
        self.output.zero_()

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.output, self.inputs.initial_state

    def paired(self) -> benchmark_utils.PairedGraphArm:
        return benchmark_utils.PairedGraphArm(self.name, self.launch, self.reset)

    def verify_generated_routes(self) -> list[dict[str, Any]]:
        return [route.verify() for route in self.generated_routes]


def _case_offsets(case: Case) -> tuple[int, ...]:
    offsets = [0]
    for length in case.seq_lens:
        offsets.append(offsets[-1] + length)
    return tuple(offsets)


def _component_config(base: Config, overrides: dict[str, object]) -> Config:
    values = dict(base.config)
    values.update(overrides)
    return Config.from_dict(values)


def _require_matched_choice(
    choices: tuple[object, ...] | None,
    selected: object,
    *,
    phase: str,
    role: str,
) -> None:
    if choices is None or selected not in choices:
        raise RuntimeError(
            f"pinned {phase} choice {selected!r} did not match {role}: {choices}"
        )


def _build_helion_arm(case: Case, inputs: KDAInputs) -> _BenchmarkArm:
    config = PINNED_HELION_CONFIGS[case.name]
    if (
        config.recurrence_dv_partitions == 2
        and config.recurrence_register_cap is not None
    ):
        raise ValueError("the pinned DV2 recurrence must be uncapped")
    resources = create_staged_kda_resources(
        _case_offsets(case), heads=case.num_heads, device=inputs.q.device
    )
    groups = (resources.critical, *resources.auxiliary)
    if (config.auxiliary_prepare_schedule is None) != (len(groups) == 1):
        raise ValueError("pinned auxiliary schedule does not match staged groups")

    call_inputs = inputs.clone_mutable()
    output = torch.empty_like(inputs.v)
    prepare_kernels: list[Callable[..., object]] = []
    recurrence_kernels: list[Callable[..., object]] = []
    generated_routes: list[_GeneratedRoute] = []
    for index, group in enumerate(groups):
        role = "critical" if index == 0 else f"auxiliary-{index - 1}"
        metadata = group.metadata
        workspace = group.workspace
        sequence_begin, sequence_end = metadata.host.sequence_range
        prepare_bound = cast(
            "BoundKernel[object]",
            kda_chunk_prepare.bind(
                (
                    call_inputs.q,
                    call_inputs.k,
                    call_inputs.raw_gate,
                    call_inputs.raw_beta,
                    call_inputs.a_log,
                    call_inputs.dt_bias,
                    metadata.cu_seqlens,
                    metadata.cu_chunks,
                    metadata.chunk_to_seq,
                    workspace.kd,
                    workspace.qd,
                    workspace.ak,
                    workspace.aq,
                    workspace.g_total,
                    None,
                    case.lower_bound * LOG2_E,
                )
            ),
        )
        recurrence_bound = cast(
            "BoundKernel[object]",
            kda_chunk_recurrence.bind(
                (
                    workspace.kd,
                    workspace.qd,
                    workspace.ak,
                    workspace.aq,
                    workspace.g_total,
                    call_inputs.v,
                    output,
                    call_inputs.initial_state[sequence_begin:sequence_end],
                    metadata.cu_seqlens,
                    metadata.cu_chunks,
                    HEAD_DIM**-0.5,
                )
            ),
        )
        schedule = (
            config.critical_prepare_schedule
            if index == 0
            else cast("str", config.auxiliary_prepare_schedule)
        )
        prepare_fragment = cast(
            "Any", prepare_bound.config_spec
        ).cute_chunk_prepare_schedule
        recurrence_dv_fragment = cast(
            "Any", recurrence_bound.config_spec
        ).cute_chunk_recurrence_dv_partitions
        recurrence_cap_fragment = cast(
            "Any", recurrence_bound.config_spec
        ).cute_chunk_recurrence_register_cap
        _require_matched_choice(
            None if prepare_fragment is None else prepare_fragment.choices,
            schedule,
            phase="prepare",
            role=role,
        )
        _require_matched_choice(
            None if recurrence_dv_fragment is None else recurrence_dv_fragment.choices,
            config.recurrence_dv_partitions,
            phase="recurrence DV partition",
            role=role,
        )
        _require_matched_choice(
            None
            if recurrence_cap_fragment is None
            else recurrence_cap_fragment.choices,
            config.recurrence_register_cap,
            phase="recurrence register cap",
            role=role,
        )

        prepare_config = _component_config(
            KDA_PREPARE_CONFIG, {_CUTE_CHUNK_PREPARE_SCHEDULE_KEY: schedule}
        )
        recurrence_config = _component_config(
            KDA_RECURRENCE_CONFIG,
            {
                _CUTE_CHUNK_RECURRENCE_DV_PARTITIONS_KEY: (
                    config.recurrence_dv_partitions
                ),
                _CUTE_CHUNK_RECURRENCE_REGISTER_CAP_KEY: (
                    config.recurrence_register_cap
                ),
            },
        )
        prepare_kernel = cast(
            "Callable[..., object]",
            prepare_bound.compile_config(prepare_config, allow_print=False),
        )
        recurrence_kernel = cast(
            "Callable[..., object]",
            recurrence_bound.compile_config(recurrence_config, allow_print=False),
        )
        prepare_kernels.append(prepare_kernel)
        recurrence_kernels.append(recurrence_kernel)
        generated_routes.append(
            _GeneratedRoute(
                role,
                "prepare",
                prepare_bound,
                prepare_config,
                prepare_kernel,
                "chunk_prepare_tma",
                "from helion._compiler.cute.chunk_prepare_split_alias_device "
                "import emit_bt16_prepare",
                rf".*'kind': 'chunk_prepare_tma'.*'schedule': '{schedule}'.*",
            )
        )
        if config.recurrence_dv_partitions == 2:
            recurrence_kind = "chunk_recurrence_sm100"
            recurrence_marker = (
                "from helion._compiler.cute.chunk_recurrence_sm100 import "
                "host_chain_dv2 as _helion_sm100_chain_host"
            )
        else:
            recurrence_kind = "chunk_recurrence_warp_dv4"
            recurrence_marker = (
                "from helion._compiler.cute.chunk_recurrence_dv4_sm100 import "
                "_recurrence_entry as _helion_sm100_warp_dv4_host"
            )
        generated_routes.append(
            _GeneratedRoute(
                role,
                "recurrence",
                recurrence_bound,
                recurrence_config,
                recurrence_kernel,
                recurrence_kind,
                recurrence_marker,
                rf".*'kind': '{recurrence_kind}'.*",
            )
        )

    prepare_tuple = tuple(prepare_kernels)
    recurrence_tuple = tuple(recurrence_kernels)

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return launch_staged_kda_prefill(
            resources,
            q=call_inputs.q,
            k=call_inputs.k,
            gate=call_inputs.raw_gate,
            beta_logits=call_inputs.raw_beta,
            a_log=call_inputs.a_log,
            dt_bias=call_inputs.dt_bias,
            values=call_inputs.v,
            output=output,
            state=call_inputs.initial_state,
            scale=HEAD_DIM**-0.5,
            gate_scale_log2=case.lower_bound * LOG2_E,
            topology=config.topology,
            prepare_kernels=prepare_tuple,
            recurrence_kernels=recurrence_tuple,
        )

    return _BenchmarkArm(
        "helion-cute",
        call_inputs,
        output,
        inputs.initial_state,
        launch,
        (resources, prepare_tuple, recurrence_tuple),
        {"backend": "helion-cute", "config": config.as_dict()},
        tuple(generated_routes),
    )


def _build_flashinfer_arm(
    module: ModuleType,
    case: Case,
    inputs: KDAInputs,
    backend: Literal["cake", "cute-dsl"],
) -> _BenchmarkArm:
    name = "flashinfer-cake" if backend == "cake" else "flashinfer-cute"
    call_inputs = inputs.clone_mutable()
    output = torch.empty_like(inputs.v)
    workspace = module._kda_prefill.RecurrentKDAPrefillWorkspace(inputs.q.device)
    route: dict[str, Any] = {
        "backend": backend,
        "public_entrypoint": "flashinfer.recurrent_kda",
        "prefill_workspace": "private",
    }
    invoke = _record_flashinfer_route(
        module,
        _invoke_flashinfer(module, case, backend, prefill_workspace=workspace),
        route,
        case=case,
        expected_route="cake-safe-gate" if backend == "cake" else "cute-dsl",
    )

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return invoke(call_inputs, output)

    return _BenchmarkArm(
        name,
        call_inputs,
        output,
        inputs.initial_state,
        launch,
        (workspace,),
        route,
    )


def _run_eager(arm: _BenchmarkArm) -> tuple[torch.Tensor, torch.Tensor]:
    arm.reset()
    arm.launch()
    torch.cuda.synchronize()
    output, state = arm.result()
    return output.clone(), state.clone()


def _run_pair(
    module: ModuleType,
    case: Case,
    inputs: KDAInputs,
    reference: tuple[torch.Tensor, torch.Tensor],
    backend: Literal["cake", "cute-dsl"],
    args: argparse.Namespace,
) -> dict[str, Any]:
    helion_arm = _build_helion_arm(case, inputs)
    baseline_arm = _build_flashinfer_arm(module, case, inputs, backend)
    generated_before = helion_arm.verify_generated_routes()
    stream = torch.cuda.Stream()
    arms = (baseline_arm.paired(), helion_arm.paired())
    captured = benchmark_utils.capture_paired_cuda_graphs(arms, stream)

    graph_correctness: dict[str, Any] = {}
    for arm in (baseline_arm, helion_arm):
        with torch.cuda.stream(stream):
            arm.reset()
            captured[arm.name].graph.replay()
        stream.synchronize()
        graph_correctness[arm.name] = {
            "passes": True,
            **_check_result(arm.result(), reference),
        }

    timing = benchmark_utils.measure_paired_cuda_graphs(
        captured,
        baseline=baseline_arm.name,
        candidate=helion_arm.name,
        measurement_cycles=args.graph_measurement_cycles,
        primer_cycles=args.graph_primer_cycles,
        cooldown_temp_c=args.cooldown_temp_c,
        cooldown_timeout_s=args.cooldown_timeout_s,
        stream=stream,
    )
    generated_after = helion_arm.verify_generated_routes()
    _assert_stable(
        f"{case.name} Helion generated routes during {baseline_arm.name} timing",
        generated_before,
        generated_after,
    )
    return {
        "baseline": baseline_arm.name,
        "candidate": helion_arm.name,
        "correctness": graph_correctness,
        "providers": {
            baseline_arm.name: baseline_arm.route,
            helion_arm.name: {
                **helion_arm.route,
                "generated_routes": generated_after,
            },
        },
        "fairness_audit": {
            "same_process": True,
            "same_cuda_stream": True,
            "same_cuda_event_timer": True,
            "one_provider_invocation_per_graph": True,
            "one_graph_replay_per_timed_interval": True,
            "reset_outside_timing": True,
            "cold_l2_before_every_replay": True,
            "private_mutable_state_per_arm": True,
            "private_output_per_arm": True,
            "shared_immutable_inputs": True,
            "balanced_position_schedule": True,
            "outlier_policy": "none discarded",
        },
        **timing,
    }


def _source_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    if helion.__file__ is None:
        raise RuntimeError("loaded Helion package has no source path")
    local_sources = {
        "benchmark": Path(__file__).resolve(),
        "paired_graph_helper": Path(benchmark_utils.__file__).resolve(),
        "prefill_kernels": Path(kda_chunk_prepare.fn.__code__.co_filename).resolve(),
        "staged_pipeline": Path(
            launch_staged_kda_prefill.__code__.co_filename
        ).resolve(),
        "helion_package": Path(helion.__file__).resolve(),
    }
    flashinfer_root = args.flashinfer_root.resolve()
    return {
        "local": _source_provenance(
            REPO_ROOT,
            local_sources,
            pathspecs=(
                "helion",
                "benchmarks/cute/compare_kda_prefill_backends.py",
                "benchmarks/cute/kda_benchmark_utils.py",
                "benchmarks/cute/kda_prefill_kernels.py",
                "benchmarks/cute/kda_prefill_staged.py",
            ),
        ),
        "flashinfer": _source_provenance(
            flashinfer_root,
            _flashinfer_sources(flashinfer_root),
            expected_commit=args.flashinfer_sha,
        ),
    }


def _run_case(
    module: ModuleType, case: Case, args: argparse.Namespace
) -> dict[str, Any]:
    inputs = _make_inputs(case)
    cake_oracle = _build_flashinfer_arm(module, case, inputs, "cake")
    reference = _run_eager(cake_oracle)
    eager_helion = _build_helion_arm(case, inputs)
    eager_cute = _build_flashinfer_arm(module, case, inputs, "cute-dsl")
    eager_correctness = {
        "helion-cute": {
            "passes": True,
            **_check_result(_run_eager(eager_helion), reference),
        },
        "flashinfer-cute": {
            "passes": True,
            **_check_result(_run_eager(eager_cute), reference),
        },
    }
    eager_routes = {
        "flashinfer-cake": cake_oracle.route,
        "flashinfer-cute": eager_cute.route,
        "helion-cute": {
            **eager_helion.route,
            "generated_routes": eager_helion.verify_generated_routes(),
        },
    }
    comparisons = {
        "flashinfer-cake": _run_pair(module, case, inputs, reference, "cake", args),
        "flashinfer-cute": _run_pair(module, case, inputs, reference, "cute-dsl", args),
    }
    return {
        "case": case.name,
        "semantic_metadata": _semantic_metadata(case),
        "pinned_helion_config": PINNED_HELION_CONFIGS[case.name].as_dict(),
        "eager_correctness": eager_correctness,
        "eager_routes": eager_routes,
        "comparisons": comparisons,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-cuda-graph", action="store_true")
    parser.add_argument("--impls", default=",".join(IMPLEMENTATIONS))
    parser.add_argument("--cases", default=",".join(OFFICIAL_CASES))
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--flashinfer-sha", default=DEFAULT_FLASHINFER_SHA)
    parser.add_argument("--graph-primer-cycles", type=int, default=2)
    parser.add_argument("--graph-measurement-cycles", type=int, default=12)
    parser.add_argument(
        "--cooldown-temp-c", type=float, default=benchmark_utils.DEFAULT_COOLDOWN_C
    )
    parser.add_argument("--cooldown-timeout-s", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    implementations = tuple(value.strip() for value in args.impls.split(","))
    if implementations != IMPLEMENTATIONS:
        parser.error(f"--impls must be exactly {','.join(IMPLEMENTATIONS)}")
    cases = tuple(value.strip() for value in args.cases.split(","))
    if cases != OFFICIAL_CASES:
        parser.error(f"--cases must be exactly {','.join(OFFICIAL_CASES)}")
    if not args.paired_cuda_graph:
        parser.error("--paired-cuda-graph is required")
    if args.graph_primer_cycles < 1:
        parser.error("--graph-primer-cycles must be positive")
    if args.graph_measurement_cycles < 2 or args.graph_measurement_cycles % 2:
        parser.error("--graph-measurement-cycles must be an even integer >= 2")
    if args.cooldown_timeout_s <= 0:
        parser.error("--cooldown-timeout-s must be positive")
    if not math.isfinite(args.cooldown_temp_c):
        parser.error("--cooldown-temp-c must be finite")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    if os.environ.get("HELION_BACKEND") != "cute":
        parser.error("HELION_BACKEND=cute is required")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    _prepare_build_environment()
    source_before = _source_fingerprint(args)
    benchmark_utils.validate_pinned_source("FlashInfer", source_before["flashinfer"])
    module = _load_flashinfer(args.flashinfer_root)
    cases = {name: _run_case(module, CASES[name], args) for name in OFFICIAL_CASES}
    source_after = _source_fingerprint(args)
    _assert_stable(
        "source fingerprint during prefill benchmark", source_before, source_after
    )
    result = {
        "valid": True,
        "cases": cases,
        "implementations_order": list(IMPLEMENTATIONS),
        "source_fingerprint_before": source_before,
        "source_fingerprint_after": source_after,
        "environment": {
            "helion_backend": os.environ.get("HELION_BACKEND"),
            "helion_fast_math": os.environ.get("HELION_FAST_MATH"),
            "helion_skip_cache": os.environ.get("HELION_SKIP_CACHE"),
            "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "gpu": _gpu_info(),
        "argv": sys.argv,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("RESULT_JSON: " + json.dumps(result))


if __name__ == "__main__":
    main()
