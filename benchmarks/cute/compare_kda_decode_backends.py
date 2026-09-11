"""Reproduce the reportable B1/B256 Helion-versus-CAKE decode comparisons.

The harness has one protocol: each provider gets a private one-invocation CUDA
graph, state/output restoration and an L2 flush happen outside the timed
region, and samples alternate in balanced ABBA/BAAB order. The tensor contract
matches FlashInfer's public packed KDA decode API.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
from typing import Callable
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.cute.kda_benchmark_utils import DEFAULT_COOLDOWN_C  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import PairedGraphArm  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import assert_stable  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import canonical_config_json  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import capture_paired_cuda_graphs  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import file_sha256  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import gpu_info  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import measure_paired_cuda_graphs  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import parse_json_object  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import source_provenance  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import text_sha256  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import validate_pinned_source  # noqa: E402
from benchmarks.cute.kda_benchmark_utils import (  # noqa: E402
    verify_helion_generated_wrapper,
)
import torch  # noqa: E402

import helion  # noqa: E402

DEFAULT_BATCH_SIZES = (1, 256)
IMPLEMENTATIONS = ("flashinfer-cake", "helion-cute")
PINNED_FLASHINFER_SHA = "f67bc2ed555c1ad6a764ad68f7aa9622178e9eae"
PINNED_SGLANG_SHA = "55509b3f421a25c631f89de8a139772f625038d8"
EXPECTED_CAKE_VARIANTS = {
    1: "register_tile16",
    256: "cpasync_tile128_register_pipeline",
}
HEADS = 12
HEAD_DIM = 128
LOWER_BOUND = -5.0
L2_EPSILON = 1e-6
STATE_SLOT_PADDING = 256
MIXED_QKV_ROW_STRIDE = 4 * HEADS * HEAD_DIM


@dataclass(frozen=True)
class HelionConfigSpec:
    json_text: str
    source: str
    source_sha256: str
    config_sha256: str


@dataclass
class KDAInputs:
    mixed_qkv: torch.Tensor
    raw_gate: torch.Tensor
    raw_beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    state_indices: torch.Tensor
    output: torch.Tensor

    def clone_mutable(self) -> KDAInputs:
        cloned_state = torch.empty_strided(
            self.state.shape,
            self.state.stride(),
            dtype=self.state.dtype,
            device=self.state.device,
        )
        cloned_state.copy_(self.state)
        return KDAInputs(
            mixed_qkv=self.mixed_qkv,
            raw_gate=self.raw_gate,
            raw_beta=self.raw_beta,
            a_log=self.a_log,
            dt_bias=self.dt_bias,
            state=cloned_state,
            state_indices=self.state_indices,
            output=torch.empty_like(self.output),
        )

    def helion_args(self) -> tuple[object, ...]:
        return (
            self.mixed_qkv,
            self.raw_gate,
            self.raw_beta,
            self.a_log,
            self.dt_bias,
            HEAD_DIM**-0.5,
            LOWER_BOUND,
            self.state,
            self.output,
            self.state_indices,
            True,
            True,
            True,
        )


def _load_helion_config(args: argparse.Namespace) -> HelionConfigSpec:
    if args.helion_config is not None:
        canonical = canonical_config_json(args.helion_config, "--helion-config")
        return HelionConfigSpec(
            json_text=canonical,
            source="inline-cli",
            source_sha256=text_sha256(args.helion_config),
            config_sha256=text_sha256(canonical),
        )
    if args.helion_config_file is None:
        raise ValueError("a pinned Helion config is required")
    path = args.helion_config_file.resolve()
    contents = path.read_bytes()
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"--helion-config-file must be UTF-8: {path}") from error
    canonical = canonical_config_json(text, "--helion-config-file")
    return HelionConfigSpec(
        json_text=canonical,
        source=str(path),
        source_sha256=file_sha256(path),
        config_sha256=text_sha256(canonical),
    )


def _source_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    harness_path = Path(__file__).resolve()
    helper_path = Path(measure_paired_cuda_graphs.__code__.co_filename).resolve()
    if helion.__file__ is None:
        raise RuntimeError("loaded Helion package has no source path")

    flashinfer_root = args.flashinfer_root.resolve()
    flashinfer_package = flashinfer_root / "flashinfer"
    sglang_root = args.sglang_root.resolve()
    sglang_helion = (
        sglang_root / "python" / "sglang" / "kernels" / "ops" / "attention" / "helion"
    )
    return {
        "benchmark": {"path": str(harness_path), "sha256": file_sha256(harness_path)},
        "paired_graph_helper": {
            "path": str(helper_path),
            "sha256": file_sha256(helper_path),
        },
        "helion": source_provenance(
            REPO_ROOT,
            {"package_init": Path(helion.__file__).resolve()},
            pathspecs=("helion",),
        ),
        "flashinfer": source_provenance(
            flashinfer_root,
            {
                "cake_packed_decode": (
                    flashinfer_package / "kda_kernels" / "cake_packed_kda_decode.py"
                ),
                "cake_packed_t1_jit": (
                    flashinfer_package / "jit" / "cake_kda_packed_t1.py"
                ),
                "cake_b1_body": (
                    flashinfer_root
                    / "csrc"
                    / "kda"
                    / "cake_kda_packed_t1_register_tile16.cu"
                ),
                "cake_b256_body": (
                    flashinfer_root
                    / "csrc"
                    / "kda"
                    / "cake_kda_packed_t1_cpasync_tile128_register_pipeline.cu"
                ),
            },
            expected_commit=args.flashinfer_sha,
        ),
        "sglang": source_provenance(
            sglang_root,
            {
                "helion_package": sglang_helion / "__init__.py",
                "kda_decode": sglang_helion / "kda_decode.py",
            },
            expected_commit=args.sglang_sha,
        ),
    }


def _install_namespace(name: str, path: Path) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if module.__file__ is None or Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError(
            f"expected module {path.resolve()}, loaded {module.__file__}"
        )
    return module


def _read_source_int_constant(path: Path, name: str) -> int:
    """Read one literal integer assignment without executing package code."""

    tree = ast.parse(path.read_text(), filename=str(path))
    values = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, int)
        and not isinstance(node.value.value, bool)
    ]
    if len(values) != 1:
        raise ValueError(f"expected one literal integer {name} assignment in {path}")
    return values[0]


def _load_flashinfer_cake(root: Path) -> ModuleType:
    package_root = (root / "flashinfer").resolve()
    source = package_root / "kda_kernels" / "cake_packed_kda_decode.py"
    if not source.is_file():
        raise FileNotFoundError(f"not a FlashInfer source checkout: {root}")
    if importlib.util.find_spec("pynvml") is None:
        sys.modules["pynvml"] = ModuleType("pynvml")
    for name, relative in (
        ("flashinfer", ""),
        ("flashinfer.jit", "jit"),
        ("flashinfer.kda_kernels", "kda_kernels"),
    ):
        _install_namespace(name, package_root / relative)
    return _load_module("flashinfer.kda_kernels.cake_packed_kda_decode", source)


def _load_sglang_decode(root: Path) -> ModuleType:
    package_root = (root / "python" / "sglang").resolve()
    helion_root = package_root / "kernels" / "ops" / "attention" / "helion"
    source = helion_root / "kda_decode.py"
    if not source.is_file():
        raise FileNotFoundError(f"not an SGLang source checkout: {root}")
    small_value_head_threshold = _read_source_int_constant(
        helion_root / "__init__.py", "KDA_SMALL_VALUE_HEAD_THRESHOLD"
    )
    for name, path in {
        "sglang": package_root,
        "sglang.kernels": package_root / "kernels",
        "sglang.kernels.ops": package_root / "kernels" / "ops",
        "sglang.kernels.ops.attention": package_root / "kernels" / "ops" / "attention",
        "sglang.kernels.ops.attention.helion": helion_root,
    }.items():
        module = _install_namespace(name, path)
        if name.endswith(".helion"):
            module.KDA_SMALL_VALUE_HEAD_THRESHOLD = small_value_head_threshold  # type: ignore[attr-defined]
    return _load_module("sglang.kernels.ops.attention.helion.kda_decode", source)


def _make_inputs(batch: int, seed: int) -> KDAInputs:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    mixed_qkv = torch.empty_strided(
        (batch, 3 * HEADS * HEAD_DIM),
        (MIXED_QKV_ROW_STRIDE, 1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    mixed_qkv.normal_(mean=0.0, std=0.25, generator=generator)
    raw_gate = torch.randn(
        batch,
        HEADS * HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.25)
    raw_beta = torch.randn(
        batch,
        HEADS,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    a_log = torch.empty(HEADS, device="cuda", dtype=torch.float32)
    a_log.uniform_(-2.0, -0.1, generator=generator)
    dt_bias = torch.randn(
        HEADS * HEAD_DIM,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).mul_(0.1)
    compact_slot = HEADS * HEAD_DIM * HEAD_DIM
    state = torch.empty_strided(
        (batch + 8, HEADS, HEAD_DIM, HEAD_DIM),
        (compact_slot + STATE_SLOT_PADDING, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state.normal_(mean=0.0, std=0.02, generator=generator)
    state_indices = torch.arange(batch, 0, -1, device="cuda", dtype=torch.int32)
    output = torch.empty(batch, 1, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    return KDAInputs(
        mixed_qkv=mixed_qkv,
        raw_gate=raw_gate,
        raw_beta=raw_beta,
        a_log=a_log,
        dt_bias=dt_bias,
        state=state,
        state_indices=state_indices,
        output=output,
    )


def _semantic_metadata(batch_size: int, seed: int) -> dict[str, Any]:
    return {
        "variant": f"packed_decode_t1_b{batch_size}_h{HEADS}_d{HEAD_DIM}",
        "seed": seed,
        "shape": {
            "batch_size": batch_size,
            "num_tokens": 1,
            "num_heads": HEADS,
            "head_dim": HEAD_DIM,
        },
        "dtype": "bfloat16",
        "state_dtype": "bfloat16",
        "mixed_qkv_row_stride": MIXED_QKV_ROW_STRIDE,
        "state_layout": "[pool,head,value,key]",
        "state_pool_size": batch_size + 8,
        "state_slot_padding_elements": STATE_SLOT_PADDING,
        "state_index_pattern": "descending-from-batch",
        "qk_l2norm": True,
        "l2_epsilon": L2_EPSILON,
        "scale": HEAD_DIM**-0.5,
        "gate_mode": "raw-lower-bound-sigmoid",
        "lower_bound": LOWER_BOUND,
        "beta_is_logit": True,
        "input_distribution": {
            "mixed_qkv": "normal-std-0.25-bfloat16",
            "raw_gate": "normal-std-0.25-bfloat16",
            "raw_beta": "normal-std-1.0-bfloat16",
            "a_log": "uniform-minus-2-to-minus-0.1-float32",
            "dt_bias": "normal-std-0.1-float32",
            "state": "normal-std-0.02-bfloat16",
        },
        "timed_state_semantics": "pristine-reset-before-each-cuda-graph-replay",
        "correctness_state_semantics": "pristine-clone-single-launch",
    }


def _torch_reference(inputs: KDAInputs) -> tuple[torch.Tensor, torch.Tensor]:
    batch = inputs.mixed_qkv.shape[0]
    q_end = HEADS * HEAD_DIM
    k_end = 2 * q_end
    q = inputs.mixed_qkv[:, :q_end].reshape(batch, HEADS, HEAD_DIM).float()
    k = inputs.mixed_qkv[:, q_end:k_end].reshape(batch, HEADS, HEAD_DIM).float()
    v = inputs.mixed_qkv[:, k_end:].reshape(batch, HEADS, HEAD_DIM).float()
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + L2_EPSILON)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + L2_EPSILON)
    q = q * HEAD_DIM**-0.5

    gate_input = inputs.raw_gate.reshape(batch, HEADS, HEAD_DIM).float()
    gate_input = gate_input + inputs.dt_bias.reshape(1, HEADS, HEAD_DIM)
    decay_parameter = torch.exp(inputs.a_log).reshape(1, HEADS, 1)
    log_decay = LOWER_BOUND * torch.sigmoid(decay_parameter * gate_input)
    beta = torch.sigmoid(inputs.raw_beta.float())
    state_indices = inputs.state_indices.long()
    state = inputs.state.index_select(0, state_indices).float()
    state = state * torch.exp(log_decay)[:, :, None, :]
    residual = v - (state * k[:, :, None, :]).sum(-1)
    state = state + (residual * beta[:, :, None])[:, :, :, None] * k[:, :, None, :]
    output = (state * q[:, :, None, :]).sum(-1)
    inputs.output[:, 0] = output.to(inputs.output.dtype)
    inputs.state[state_indices] = state.to(inputs.state.dtype)
    return inputs.output, inputs.state


def _effective_bytes(batch: int) -> int:
    element_bytes = 2
    state_elements = batch * HEADS * HEAD_DIM * HEAD_DIM
    activation_elements = batch * (
        3 * HEADS * HEAD_DIM + HEADS * HEAD_DIM + HEADS + HEADS * HEAD_DIM
    )
    parameter_bytes = (HEADS + HEADS * HEAD_DIM) * 4
    index_bytes = batch * 4
    return (
        2 * state_elements * element_bytes
        + activation_elements * element_bytes
        + parameter_bytes
        + index_bytes
    )


def _check_result(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    output, state = actual
    expected_output, expected_state = expected
    output_max_abs = float((output.float() - expected_output.float()).abs().max())
    state_max_abs = float((state.float() - expected_state.float()).abs().max())
    torch.testing.assert_close(output, expected_output, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(state, expected_state, atol=2e-2, rtol=1e-2)
    return {"output_max_abs": output_max_abs, "state_max_abs": state_max_abs}


def _make_helion_kernel(module: ModuleType, args: argparse.Namespace) -> helion.Kernel:
    return helion.kernel(
        module._helion_fused_recurrent_kda_packed_decode_body,
        static_shapes=False,
        autotune_effort="full",
        autotune_random_seed=args.autotune_seed,
        ignore_warnings=[helion.exc.ProcessGroupNameNotFound],
    )


def _helion_codegen_expectation(
    batch_size: int, config: dict[str, Any]
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if batch_size == 1:
        return "single-token-rank1", ("rank1_helper_abi_version = 4",), ()
    uses_canonical_async_bf16x2_plan = (
        config.get("block_sizes") == [128]
        and config.get("num_threads") == [8, 16]
        and config.get("cute_bf16x2_recurrence") is True
        and config.get("cute_async_load_stages") == 5
        and config.get("cute_async_load_group_rows") == 2
        and config.get("cute_async_store_policy") == "l2_evict_last"
        and config.get("cute_vector_widths") == [8, 1, 1, 1]
    )
    if uses_canonical_async_bf16x2_plan:
        return (
            "generic-async-state-ring-bf16x2",
            (
                (
                    "from helion._compiler.cute.l2_policy import "
                    "store_u32x4_l2_evict_last as "
                    "_cute_store_u32x4_l2_evict_last"
                ),
                "cute.arch.cp_async_wait_group(0)",
            ),
            (
                (
                    r"_async_state_[0-9]+_smem = "
                    r"cutlass\.Array\(cutlass\.Uint16, 10240, "
                    r"space=cutlass\.AddressSpace\.smem, alignment=128\)"
                ),
                (
                    r"_async_state_[0-9]+_scalar_pairs = "
                    r"cute\.make_rmem_tensor\(8, cutlass\.Uint32\)"
                ),
                r"_bf16x2_[0-9]+_abi_version = 1",
                r"_cute_store_u32x4_l2_evict_last\(.*\)",
            ),
        )
    return "generic-decode", ("@cute.kernel",), ()


def _paired_input_ownership(
    pristine: KDAInputs, arms: dict[str, KDAInputs]
) -> dict[str, Any]:
    immutable_fields = (
        "mixed_qkv",
        "raw_gate",
        "raw_beta",
        "a_log",
        "dt_bias",
        "state_indices",
    )
    shared = {
        field: all(
            getattr(inputs, field) is getattr(pristine, field)
            for inputs in arms.values()
        )
        for field in immutable_fields
    }
    if not all(shared.values()):
        raise RuntimeError(f"paired immutable inputs are not shared: {shared}")
    state_pointers = {
        "pristine": pristine.state.data_ptr(),
        **{name: inputs.state.data_ptr() for name, inputs in arms.items()},
    }
    output_pointers = {
        "pristine": pristine.output.data_ptr(),
        **{name: inputs.output.data_ptr() for name, inputs in arms.items()},
    }
    if len(set(state_pointers.values())) != len(state_pointers):
        raise RuntimeError(f"paired state buffers alias: {state_pointers}")
    if len(set(output_pointers.values())) != len(output_pointers):
        raise RuntimeError(f"paired output buffers alias: {output_pointers}")
    return {
        "shared_immutable_fields": shared,
        "private_state_buffers": True,
        "private_output_buffers": True,
        "state_pointers": state_pointers,
        "output_pointers": output_pointers,
    }


def _cross_provider_comparison(
    baseline: KDAInputs, candidate: KDAInputs
) -> dict[str, float | bool]:
    return {
        "output_max_abs": float(
            (baseline.output.float() - candidate.output.float()).abs().max()
        ),
        "state_max_abs": float(
            (baseline.state.float() - candidate.state.float()).abs().max()
        ),
        "output_exact": bool(torch.equal(baseline.output, candidate.output)),
        "state_exact": bool(torch.equal(baseline.state, candidate.state)),
    }


def _make_cake_launch(module: ModuleType, inputs: KDAInputs) -> Callable[[], object]:
    def launch() -> object:
        return module.run_packed_kda_decode(  # pyrefly: ignore[missing-attribute]
            inputs.mixed_qkv,
            inputs.raw_gate,
            inputs.raw_beta,
            inputs.a_log,
            inputs.dt_bias,
            inputs.state,
            inputs.state_indices,
            inputs.output,
        )

    return launch


def _tracked_cake_call(
    module: ModuleType,
    batch_size: int,
    inputs: KDAInputs,
    launch: Callable[[], object],
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
    loaded: list[tuple[str, str, str]] = []
    original_cake = module.get_cake_kda_packed_t1_module  # pyrefly: ignore[missing-attribute]
    original_legacy = module.get_flash_kda_packed_t1_module  # pyrefly: ignore[missing-attribute]

    def track_cake(variant: str, target: str) -> object:
        loaded.append(("cake-kda-packed-t1", variant, target))
        return original_cake(variant, target)

    def track_legacy(variant: str, target: str) -> object:
        loaded.append(("legacy-flash-kda-packed-t1", variant, target))
        return original_legacy(variant, target)

    module.get_cake_kda_packed_t1_module = track_cake  # pyrefly: ignore[missing-attribute]
    module.get_flash_kda_packed_t1_module = track_legacy  # pyrefly: ignore[missing-attribute]
    try:
        launch()
    finally:
        module.get_cake_kda_packed_t1_module = original_cake  # pyrefly: ignore[missing-attribute]
        module.get_flash_kda_packed_t1_module = original_legacy  # pyrefly: ignore[missing-attribute]

    expected = EXPECTED_CAKE_VARIANTS[batch_size]
    if (
        len(loaded) != 1
        or loaded[0][0] != "cake-kda-packed-t1"
        or loaded[0][1] != expected
    ):
        raise AssertionError(
            f"B{batch_size} expected one CAKE {expected} module, got {loaded}"
        )
    family, variant, target = loaded[0]
    jit_module = importlib.import_module("flashinfer.jit.cake_kda_packed_t1")
    jit_uri = jit_module.get_cake_kda_packed_t1_uri(variant, target)  # pyrefly: ignore[missing-attribute]
    return (inputs.output, inputs.state), {
        "variant": variant,
        "expected_variant": expected,
        "module_family": family,
        "jit_target": target,
        "jit_uri": jit_uri,
    }


def _run_paired_decode_graph(
    args: argparse.Namespace, config_spec: HelionConfigSpec
) -> dict[str, Any]:
    if os.environ.get("HELION_BACKEND") != "cute":
        raise RuntimeError("set HELION_BACKEND=cute for Helion graph timing")

    source_before = _source_fingerprint(args)
    validate_pinned_source("FlashInfer", source_before["flashinfer"])
    validate_pinned_source("SGLang", source_before["sglang"])
    if _load_helion_config(args) != config_spec:
        raise RuntimeError("Helion config input changed before paired graph timing")

    batch_size = args.batch_size
    pristine = _make_inputs(batch_size, args.seed)
    expected = _torch_reference(pristine.clone_mutable())
    arm_inputs = {name: pristine.clone_mutable() for name in IMPLEMENTATIONS}
    ownership = _paired_input_ownership(pristine, arm_inputs)

    cake_module = _load_flashinfer_cake(args.flashinfer_root.resolve())
    cake_inputs = arm_inputs["flashinfer-cake"]
    cake_launch = _make_cake_launch(cake_module, cake_inputs)
    state_aligned, aux_vec4_aligned = cake_module._optimized_alignment_flags(  # pyrefly: ignore[missing-attribute]
        cake_inputs.mixed_qkv,
        cake_inputs.raw_gate,
        cake_inputs.dt_bias,
        cake_inputs.state,
    )
    selected_cake_variant = cake_module.select_cake_kda_packed_t1_variant(  # pyrefly: ignore[missing-attribute]
        batch_size,
        state_aligned=state_aligned,
        aux_vec4_aligned=aux_vec4_aligned,
    )
    expected_cake_variant = EXPECTED_CAKE_VARIANTS[batch_size]
    if selected_cake_variant != expected_cake_variant:
        raise RuntimeError(
            f"B{batch_size} expected CAKE variant {expected_cake_variant}, "
            f"got {selected_cake_variant}"
        )

    sglang_module = _load_sglang_decode(args.sglang_root.resolve())
    helion_inputs = arm_inputs["helion-cute"]
    kernel_args = helion_inputs.helion_args()
    helion_bound = _make_helion_kernel(sglang_module, args).bind(kernel_args)
    requested_config = helion.Config(
        **parse_json_object(config_spec.json_text, "pinned Helion config")
    )
    config = helion_bound.config_spec.normalized_config(requested_config)
    helion_bound.set_config(config)
    resolved_config = canonical_config_json(config.to_json(), "resolved Helion config")

    def launch_helion(
        bound_kernel: Callable[..., object] = cast(
            "Callable[..., object]", helion_bound
        ),
        bound_args: tuple[object, ...] = kernel_args,
    ) -> object:
        return bound_kernel(*bound_args)

    def reset_cake() -> None:
        cake_inputs.state.copy_(pristine.state)
        cake_inputs.output.zero_()

    def reset_helion() -> None:
        helion_inputs.state.copy_(pristine.state)
        helion_inputs.output.zero_()

    arms = (
        PairedGraphArm("flashinfer-cake", cake_launch, reset_cake),
        PairedGraphArm("helion-cute", launch_helion, reset_helion),
    )

    eager_correctness: dict[str, Any] = {}
    reset_cake()
    cake_actual, cake_route = _tracked_cake_call(
        cake_module, batch_size, cake_inputs, cake_launch
    )
    torch.cuda.synchronize()
    eager_correctness["flashinfer-cake"] = _check_result(cake_actual, expected)
    reset_helion()
    launch_helion()
    torch.cuda.synchronize()
    eager_correctness["helion-cute"] = _check_result(
        (helion_inputs.output, helion_inputs.state), expected
    )

    config_values = parse_json_object(resolved_config, "resolved Helion config")
    plan_kind, source_markers, source_patterns = _helion_codegen_expectation(
        batch_size, config_values
    )
    generated_wrapper = verify_helion_generated_wrapper(
        helion_bound,
        config,
        expected_plan_kind=plan_kind,
        expected_source_markers=source_markers,
        expected_source_patterns=source_patterns,
    )
    provider_metadata = {
        "flashinfer-cake": {
            "backend": "cake",
            "route": cake_route,
            "state_aligned": state_aligned,
            "aux_vec4_aligned": aux_vec4_aligned,
            "provenance": source_before["flashinfer"],
        },
        "helion-cute": {
            "backend": "helion-cute",
            "requested_config": json.loads(config_spec.json_text),
            "config": json.loads(resolved_config),
            "config_metadata": {
                "mode": "pinned",
                "source": config_spec.source,
                "source_sha256": config_spec.source_sha256,
                "input_config_sha256": config_spec.config_sha256,
                "resolved_config_sha256": text_sha256(resolved_config),
            },
            "generated_wrapper": generated_wrapper,
            "sglang_provenance": source_before["sglang"],
        },
    }

    stream = torch.cuda.Stream()
    captured = capture_paired_cuda_graphs(arms, stream)
    graph_correctness: dict[str, Any] = {}
    for arm in arms:
        with torch.cuda.stream(stream):
            arm.reset()
            captured[arm.name].graph.replay()
        stream.synchronize()
        inputs = arm_inputs[arm.name]
        graph_correctness[arm.name] = _check_result(
            (inputs.output, inputs.state), expected
        )
    cross_provider_correctness = _cross_provider_comparison(cake_inputs, helion_inputs)

    timing = measure_paired_cuda_graphs(
        captured,
        baseline="flashinfer-cake",
        candidate="helion-cute",
        measurement_cycles=args.graph_measurement_cycles,
        primer_cycles=args.graph_primer_cycles,
        cooldown_temp_c=args.cooldown_temp_c,
        cooldown_timeout_s=args.cooldown_timeout_s,
        stream=stream,
    )
    effective_bytes = _effective_bytes(batch_size)
    for values in timing["implementations"].values():
        values["effective_gbps"] = effective_bytes / (values["median_ms"] * 1e6)

    source_after = _source_fingerprint(args)
    assert_stable(
        "source fingerprint during paired graph run", source_before, source_after
    )
    if _load_helion_config(args) != config_spec:
        raise RuntimeError("Helion config input changed during paired graph timing")
    final_config = getattr(helion_bound, "_config", None)
    if (
        final_config is None
        or canonical_config_json(final_config.to_json(), "final Helion config")
        != resolved_config
    ):
        raise RuntimeError("Helion config changed during paired graph timing")
    final_wrapper = verify_helion_generated_wrapper(
        helion_bound,
        final_config,
        expected_plan_kind=plan_kind,
        expected_source_markers=source_markers,
        expected_source_patterns=source_patterns,
    )
    assert_stable(
        "Helion generated wrapper during paired graph run",
        generated_wrapper,
        final_wrapper,
    )

    return {
        "valid": True,
        "variant": f"packed_decode_t1_b{batch_size}_h{HEADS}_d{HEAD_DIM}",
        "batch_size": batch_size,
        "implementations_order": list(IMPLEMENTATIONS),
        "semantic_metadata": _semantic_metadata(batch_size, args.seed),
        "effective_bytes": effective_bytes,
        "timed_state_semantics": "pristine-reset-before-each-cuda-graph-replay",
        "correctness_state_semantics": "identical-pristine-clone-one-replay",
        "correctness": {
            "eager_vs_oracle": eager_correctness,
            "graph_replay_vs_oracle": graph_correctness,
            "cross_provider_after_one_graph": cross_provider_correctness,
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
        "input_ownership": ownership,
        "graph_capture": {
            "graphs": len(captured),
            "provider_invocations_per_graph": 1,
            "graph_objects_kept_alive": all(
                isinstance(item.graph, torch.cuda.CUDAGraph)
                for item in captured.values()
            ),
        },
        "providers": provider_metadata,
        "source_fingerprint_before": source_before,
        "source_fingerprint_after": source_after,
        "environment": {
            "helion_backend": os.environ.get("HELION_BACKEND"),
            "helion_fast_math": os.environ.get("HELION_FAST_MATH"),
            "helion_skip_cache": os.environ.get("HELION_SKIP_CACHE"),
            "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "seed": args.seed,
        "argv": sys.argv,
        "gpu": gpu_info(),
        **timing,
    }


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-cuda-graph", action="store_true")
    parser.add_argument("--impls", default=",".join(IMPLEMENTATIONS))
    parser.add_argument(
        "--batch-size", type=int, choices=DEFAULT_BATCH_SIZES, required=True
    )
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--flashinfer-sha", default=PINNED_FLASHINFER_SHA)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--sglang-sha", default=PINNED_SGLANG_SHA)
    parser.add_argument("--autotune-seed", type=int, default=1337)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--helion-config", help="Helion config JSON object")
    config_group.add_argument("--helion-config-file", type=Path)
    parser.add_argument("--graph-primer-cycles", type=int, default=2)
    parser.add_argument("--graph-measurement-cycles", type=int, default=12)
    parser.add_argument("--cooldown-temp-c", type=float, default=DEFAULT_COOLDOWN_C)
    parser.add_argument("--cooldown-timeout-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path)
    return parser


def _validate_cli_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> HelionConfigSpec:
    implementations = tuple(value.strip() for value in args.impls.split(","))
    if implementations != IMPLEMENTATIONS:
        parser.error(f"--impls must be exactly {','.join(IMPLEMENTATIONS)}")
    if not args.paired_cuda_graph:
        parser.error("--paired-cuda-graph is required; legacy timing was removed")
    if args.graph_primer_cycles <= 0:
        parser.error("--graph-primer-cycles must be positive")
    if args.graph_measurement_cycles < 2 or args.graph_measurement_cycles % 2:
        parser.error("--graph-measurement-cycles must be an even integer >= 2")
    if args.cooldown_timeout_s <= 0:
        parser.error("--cooldown-timeout-s must be positive")
    if not math.isfinite(args.cooldown_temp_c):
        parser.error("--cooldown-temp-c must be finite")
    if (
        args.output is not None
        and args.helion_config_file is not None
        and args.output.resolve() == args.helion_config_file.resolve()
    ):
        parser.error("--output and --helion-config-file must be different files")
    try:
        return _load_helion_config(args)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))


def main() -> None:
    parser = _create_parser()
    args = parser.parse_args()
    config_spec = _validate_cli_args(parser, args)
    result = _run_paired_decode_graph(args, config_spec)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output_file:
            output_file.write(json.dumps(result) + "\n")
    print("RESULT_JSON: " + json.dumps(result))


if __name__ == "__main__":
    main()
